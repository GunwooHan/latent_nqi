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
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold

from datasets import NQIDataset
from models import HDemucs

parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=42)

# 모델 관련 설정
parser.add_argument('--gpus', type=int, default=1)
parser.add_argument('--precision', type=int, default=32)
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--project', type=str, default='nqi')
parser.add_argument('--name', type=str, default='nqi_demucs')

# 학습 관련 설정
parser.add_argument('--epochs', type=int, default=5)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--learning_rate', type=float, default=0.0001)
# parser.add_argument('--optimizer', type=str, default='adamp')
# parser.add_argument('--scheduler', type=str, default='reducelr')
# parser.add_argument('--loss', type=str, default='ce')

args = parser.parse_args()

if __name__ == '__main__':
    pl.seed_everything(args.seed)

    # wandb_logger = WandbLogger(project=args.project, name=args.name)
    # checkpoint_callback = ModelCheckpoint(
    #     monitor="val/cls_loss",
    #     dirpath="checkpoints",
    #     filename=f"{args.name}" + "{val/cls_loss:.4f}",
    #     save_top_k=3,
    #     mode="min",
    # )
    # early_stop_callback = EarlyStopping(monitor="val/loss", min_delta=0.00, patience=50, verbose=True,
    #                                     mode="min")

    np_data_train_raw = np.load("dataset/train-001.npy")
    np_data_val_raw = np.load("dataset/valid.npy")


    list_data_train_seq_temp = []
    list_data_train_label_temp = []
    list_data_val_seq_temp = []
    list_data_val_label_temp = []

    for int_idx in range(np_data_train_raw.shape[0]):
        np_data_train_seq_temp = np_data_train_raw[0].squeeze(1)
        np_data_train_label_temp = np.full(np_data_train_seq_temp.shape[0], int_idx)
        list_data_train_seq_temp.append(np_data_train_seq_temp)
        list_data_train_label_temp.append(np_data_train_label_temp)

        np_data_val_seq_temp = np_data_val_raw[0].squeeze(1)
        np_data_val_label_temp = np.full(np_data_val_seq_temp.shape[0], int_idx)
        list_data_val_seq_temp.append(np_data_val_seq_temp)
        list_data_val_label_temp.append(np_data_val_label_temp)

    train_seq = np.concatenate(list_data_train_seq_temp)
    train_label = np.concatenate(list_data_train_label_temp)
    val_seq = np.concatenate(list_data_val_seq_temp)
    val_label = np.concatenate(list_data_val_label_temp)
    model = HDemucs()

    train_ds = NQIDataset(train_seq, train_label)
    train_dataloader = torch.utils.data.DataLoader(train_ds,
                                                   batch_size=args.batch_size,
                                                   num_workers=args.num_workers,
                                                   shuffle=True,
                                                   drop_last=True)

    val_ds = NQIDataset(val_seq, val_label)
    val_dataloader = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size, num_workers=args.num_workers)

    trainer = pl.Trainer(accelerator='gpu',
                         devices=args.gpus,
                         precision=args.precision,
                         max_epochs=args.epochs,
                         log_every_n_steps=1,
                         # strategy='ddp',
                         # num_sanity_val_steps=0,
                         # limit_train_batches=5,
                         # limit_val_batches=1,
                         # logger=wandb_logger,
                         # callbacks=[checkpoint_callback, early_stop_callback]
                         )

    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    wandb.finish()
