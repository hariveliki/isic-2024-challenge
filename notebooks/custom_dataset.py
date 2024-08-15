from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import h5py
import torch
import torchvision


class CustomDataset(Dataset):

    def __init__(self, data_file, meta_file, transform=None):
        self.data_file = data_file
        self.meta_file = meta_file
        self.meta = pd.read_csv(self.meta_file, index_col=0)
        with h5py.File(self.data_file, "r") as f:
            keys = list(f.keys())
        self.keys = keys
        self.transform = transform

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        id_key = self.keys[index]
        id_meta = self.meta.loc[id_key].name
        assert id_key == id_meta
        with h5py.File(self.data_file, "r") as f:
            byte_strings = f[self.keys[index]][()]
            arr = np.frombuffer(byte_strings, dtype=np.uint8)
            tensor = torch.tensor(arr)
            decoded = torchvision.io.decode_jpeg(tensor)
        if self.transform:
            decoded = self.transform(decoded)
        target = self.meta.loc[id_key].target
        return decoded, target, self.keys[index]
