import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pytorch_lightning as pl
import torch.optim.lr_scheduler as lr_scheduler


class LocalState(nn.Module):
    """Local state allows to have attention based only on data (no positional embedding),
    but while setting a constraint on the time window (e.g. decaying penalty term).
    Also a failed experiments with trying to provide some frequency based attention.
    """

    def __init__(self, channels: int, heads: int = 4, nfreqs: int = 0, ndecay: int = 4):
        super().__init__()
        assert channels % heads == 0, (channels, heads)
        self.heads = heads
        self.nfreqs = nfreqs
        self.ndecay = ndecay
        self.content = nn.Conv1d(channels, channels, 1)
        self.query = nn.Conv1d(channels, channels, 1)
        self.key = nn.Conv1d(channels, channels, 1)
        if nfreqs:
            self.query_freqs = nn.Conv1d(channels, heads * nfreqs, 1)
        if ndecay:
            self.query_decay = nn.Conv1d(channels, heads * ndecay, 1)
            # Initialize decay close to zero (there is a sigmoid), for maximum initial window.
            self.query_decay.weight.data *= 0.01
            assert self.query_decay.bias is not None  # stupid type checker
            self.query_decay.bias.data[:] = -2
        self.proj = nn.Conv1d(channels + heads * nfreqs, channels, 1)

    def forward(self, x):
        B, C, T = x.shape
        heads = self.heads
        indexes = torch.arange(T, device=x.device, dtype=x.dtype)
        # left index are keys, right index are queries
        delta = indexes[:, None] - indexes[None, :]

        queries = self.query(x).view(B, heads, -1, T)
        keys = self.key(x).view(B, heads, -1, T)
        # t are keys, s are queries
        dots = torch.einsum("bhct,bhcs->bhts", keys, queries)
        dots /= keys.shape[2] ** 0.5
        if self.nfreqs:
            periods = torch.arange(1, self.nfreqs + 1, device=x.device, dtype=x.dtype)
            freq_kernel = torch.cos(2 * math.pi * delta / periods.view(-1, 1, 1))
            freq_q = self.query_freqs(x).view(B, heads, -1, T) / self.nfreqs ** 0.5
            dots += torch.einsum("fts,bhfs->bhts", freq_kernel, freq_q)
        if self.ndecay:
            decays = torch.arange(1, self.ndecay + 1, device=x.device, dtype=x.dtype)
            decay_q = self.query_decay(x).view(B, heads, -1, T)
            decay_q = torch.sigmoid(decay_q) / 2
            decay_kernel = - decays.view(-1, 1, 1) * delta.abs() / self.ndecay ** 0.5
            dots += torch.einsum("fts,bhfs->bhts", decay_kernel, decay_q)

        # Kill self reference.
        dots.masked_fill_(torch.eye(T, device=dots.device, dtype=torch.bool), -100)
        weights = torch.softmax(dots, dim=2)

        content = self.content(x).view(B, heads, -1, T)
        result = torch.einsum("bhts,bhct->bhcs", weights, content)
        if self.nfreqs:
            time_sig = torch.einsum("bhts,fts->bhfs", weights, freq_kernel)
            result = torch.cat([result, time_sig], 2)
        result = result.reshape(B, -1, T)
        return x + self.proj(result)


class LayerScale(nn.Module):
    """Layer scale from [Touvron et al 2021] (https://arxiv.org/pdf/2103.17239.pdf).
    This rescales diagonaly residual outputs close to 0 initially, then learnt.
    """

    def __init__(self, channels: int, init: float = 0):
        super().__init__()
        self.scale = nn.Parameter(torch.full((channels,), fill_value=init, requires_grad=True))

    def forward(self, x):
        return self.scale[:, None] * x


def unfold(a, kernel_size, stride):
    """Given input of size [*OT, T], output Tensor of size [*OT, F, K]
    with K the kernel size, by extracting frames with the given stride.
    This will pad the input so that `F = ceil(T / K)`.
    see https://github.com/pytorch/pytorch/issues/60466
    """
    *shape, length = a.shape
    n_frames = math.ceil(length / stride)
    tgt_length = (n_frames - 1) * stride + kernel_size
    a = F.pad(a, (0, tgt_length - length))
    strides = list(a.stride())
    assert strides[-1] == 1, 'data should be contiguous'
    strides = strides[:-1] + [stride, 1]
    return a.as_strided([*shape, n_frames, kernel_size], strides)


class BLSTM(nn.Module):
    """
    BiLSTM with same hidden units as input dim.
    If `max_steps` is not None, input will be splitting in overlapping
    chunks and the LSTM applied separately on each chunk.
    """

    def __init__(self, dim, layers=1, max_steps=None, skip=False):
        super().__init__()
        assert max_steps is None or max_steps % 4 == 0
        self.max_steps = max_steps
        self.lstm = nn.LSTM(bidirectional=True, num_layers=layers, hidden_size=dim, input_size=dim)
        self.linear = nn.Linear(2 * dim, dim)
        self.skip = skip

    def forward(self, x):
        B, C, T = x.shape
        y = x
        framed = False
        if self.max_steps is not None and T > self.max_steps:
            width = self.max_steps
            stride = width // 2
            frames = unfold(x, width, stride)
            nframes = frames.shape[2]
            framed = True
            x = frames.permute(0, 2, 1, 3).reshape(-1, C, width)

        x = x.permute(2, 0, 1)

        x = self.lstm(x)[0]
        x = self.linear(x)
        x = x.permute(1, 2, 0)
        if framed:
            out = []
            frames = x.reshape(B, -1, C, width)
            limit = stride // 2
            for k in range(nframes):
                if k == 0:
                    out.append(frames[:, k, :, :-limit])
                elif k == nframes - 1:
                    out.append(frames[:, k, :, limit:])
                else:
                    out.append(frames[:, k, :, limit:-limit])
            out = torch.cat(out, -1)
            out = out[..., :T]
            x = out
        if self.skip:
            x = x + y
        return x


class ResidualBranch(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, attention=False) -> None:
        super(ResidualBranch, self).__init__()
        self.model = nn.Sequential(
            nn.Conv1d(in_channels, in_channels // 4, kernel_size=3, padding="same"),
            nn.GroupNorm(1, in_channels // 4),
            nn.GELU(),
            BLSTM(in_channels // 4, layers=2, max_steps=200, skip=True) if attention else nn.Identity(),
            LocalState(in_channels // 4, heads=4, ndecay=4) if attention else nn.Identity(),
            nn.Conv1d(in_channels // 4, out_channels * 2, kernel_size=1),
            nn.GroupNorm(1, out_channels * 2),
            nn.GLU(1),
            LayerScale(out_channels, 1e-4)
        )

    def forward(self, tensor):
        x = self.model(tensor)
        return x


class HDEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, attension=False):
        super(HDEncoder, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, kernel_size=8, stride=4, padding=2),
            nn.GELU()
        )
        self.block1 = ResidualBranch(in_channels, in_channels, attension)
        self.block2 = ResidualBranch(in_channels, in_channels, attension)
        self.layer2 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1),
            nn.GLU(1)
        )

    def forward(self, tensor):
        x1 = self.layer1(tensor)
        x2 = x1 + self.block1(x1)
        x3 = x2 + self.block2(x2)
        x4 = self.layer2(x3)
        return x4


class HDDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, attension=False):
        super(HDDecoder, self).__init__()
        self.layer1 = nn.Sequential(
            nn.ConvTranspose1d(in_channels, in_channels, kernel_size=8, stride=4, padding=2),
            nn.GELU()
        )
        self.block1 = ResidualBranch(in_channels, in_channels, attension)
        self.block2 = ResidualBranch(in_channels, in_channels, attension)
        self.layer2 = nn.Sequential(
            nn.ConvTranspose1d(in_channels, out_channels, kernel_size=1, stride=1),
            nn.GLU(1)
        )

    def forward(self, tensor):
        x1 = self.layer1(tensor)
        x2 = x1 + self.block1(x1)
        x3 = x2 + self.block2(x2)
        x4 = self.layer2(x3)
        return x4


class HDemucs(pl.LightningModule):
    def __init__(self):
        super(HDemucs, self).__init__()
        self.encoder = None
        self.decoder = None
        self.aux_cls_head = None

    def configure_optimizers(self):
        if self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=self.args.learning_rate)
        elif self.args.optimizer == 'adamw':
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.args.learning_rate)

        if self.args.scheduler == "steplr":
            scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def training_step(self, train_batch, batch_idx):
        tensor_signal, tensor_label_cls = train_batch
        tensor_latent = self.encoder(tensor_signal)
        tensor_signal_recon = self.decoder(tensor_latent)
        tensor_pred_cls = self.aux_cls_head(tensor_latent)

        loss_recon = F.l1_loss(tensor_signal, tensor_signal_recon)
        loss_cls = F.cross_entropy(tensor_pred_cls, tensor_label_cls)
        loss = loss_cls + loss_recon

        self.log('train/recon_loss', loss_recon)
        self.log('train/cls_loss', loss_cls)
        self.log('train/loss', loss)

    def validation_step(self, val_batch, batch_idx):
        tensor_signal, tensor_label_cls = val_batch
        tensor_latent = self.encoder(tensor_signal)
        tensor_signal_recon = self.decoder(tensor_latent)
        tensor_pred_cls = self.aux_cls_head(tensor_latent)

        loss_recon = F.l1_loss(tensor_signal, tensor_signal_recon)
        loss_cls = F.cross_entropy(tensor_pred_cls, tensor_label_cls)
        loss = loss_cls + loss_recon

        self.log('val/recon_loss', loss_recon)
        self.log('val/cls_loss', loss_cls)
        self.log('val/loss', loss)

    def get_latent(self, tensor):
        return self.encoder(tensor)


if __name__ == '__main__':
    model = HDDecoder(80, 160, attension=True)
    inputs = torch.randn(1, 80, 32)
    outputs = model(inputs)
    print(outputs.shape)