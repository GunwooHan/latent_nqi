import torch
import numpy as np


class NQIDataset(torch.utils.data.Dataset):
    def __init__(self, data, label, transform=None):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.label = torch.tensor(label, dtype=torch.int64)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx, :, :32], self.label[idx]
