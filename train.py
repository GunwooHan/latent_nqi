import os
import glob
import argparse

import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
import torchvision.transforms
import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from sklearn.model_selection import train_test_split

from datasets import NQIDataset
from models import HDemucs

parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=42)

# 모델 관련 설정
parser.add_argument('--gpus', type=int, default=1)
parser.add_argument('--precision', type=int, default=32)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--project', type=str, default='nqi')
parser.add_argument('--name', type=str, default='nqi_demucs')

# 학습 관련 설정
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--learning_rate', type=float, default=0.0001)
# parser.add_argument('--optimizer', type=str, default='adamp')
# parser.add_argument('--scheduler', type=str, default='reducelr')
# parser.add_argument('--loss', type=str, default='ce')

args = parser.parse_args()

if __name__ == '__main__':
    pl.seed_everything(args.seed)

    wandb_logger = WandbLogger(project=args.project, name=args.name)
    checkpoint_callback = ModelCheckpoint(
        monitor="val/loss",
        dirpath="checkpoints",
        filename=f"{args.name}" + "{val/cls_loss:.4f}",
        save_top_k=3,
        mode="min",
    )
    early_stop_callback = EarlyStopping(monitor="val/loss", min_delta=0.00, patience=50, verbose=True,
                                        mode="min")
    train_seq = np.load("dataset/train_seq.npy")
    train_label = np.load("dataset/train_label.npy")
    valid_seq = np.load("dataset/valid_seq.npy")
    valid_label = np.load("dataset/valid_label.npy")

    # train_seq = np.load("dataset/valid_seq.npy")
    # train_label = np.load("dataset/valid_label.npy")
    # valid_seq = np.load("dataset/valid_seq.npy")
    # valid_label = np.load("dataset/valid_label.npy")


    # train_seq = np.load("dataset/test_train_seq.npy")
    # train_label = np.load("dataset/test_train_label.npy")
    # valid_seq = np.load("dataset/test_valid_seq.npy")
    # valid_label = np.load("dataset/test_valid_label.npy")

    model = HDemucs()

    train_ds = NQIDataset(train_seq, train_label)
    train_dataloader = torch.utils.data.DataLoader(train_ds,
                                                   batch_size=args.batch_size,
                                                   num_workers=args.num_workers,
                                                   shuffle=True,
                                                   drop_last=True)

    val_ds = NQIDataset(valid_seq, valid_label)
    val_dataloader = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size, num_workers=args.num_workers)

    trainer = pl.Trainer(accelerator='gpu',
                         devices=args.gpus,
                         precision=args.precision,
                         max_epochs=args.epochs,
                         #  log_every_n_steps=1,
                         # strategy='ddp',
                         # num_sanity_val_steps=0,
                         # limit_train_batches=5,
                         # limit_val_batches=1,
                         logger=wandb_logger,
                         callbacks=[checkpoint_callback, early_stop_callback]
                         )

    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    # trainer.fit(model, train_dataloaders=train_dataloader)
    wandb.finish()
