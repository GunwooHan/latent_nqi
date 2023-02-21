import torch
import numpy as np
import torch.nn.functional as F


class NQIDataset(torch.utils.data.Dataset):
    def __init__(self, data, label, transform=None):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.label = torch.tensor(label, dtype=torch.float32)
        self.label[:, :, 1] = self.label[:, :, 1] / (100000 * 4 * np.pi) + 0.5
        self.label[:, :, 2] = self.label[:, :, 2] / (100000 * 2 * np.pi)
        self.pad_size = (0, 152)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return F.pad(self.data[idx], self.pad_size), self.label[idx]
