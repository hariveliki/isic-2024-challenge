from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import h5py
import torch
import torchvision


class CustomDataset(Dataset):

    def __init__(self, hdf5_path, csv_meta_path, transform=None):
        self.hdf5_path = hdf5_path
        self.csv_meta_path = csv_meta_path
        self.meta = pd.read_csv(self.csv_meta_path, index_col=0)
        with h5py.File(self.hdf5_path, 'r') as f:
            keys = list(f.keys())
        self.keys = keys
        self.transform = transform
        print("Transformation: ", self.transform)

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        id_key = self.keys[index]
        id_meta = self.meta.loc[id_key].name
        assert id_key == id_meta
        with h5py.File(self.hdf5_path, 'r') as f:
            byte_strings = np.array(f[self.keys[index]])
            arr = np.frombuffer(byte_strings, dtype=np.uint8)
            tensor = torch.tensor(arr)
            decoded = torchvision.io.decode_jpeg(tensor)
        if self.transform:
            decoded = self.transform(decoded)
        target = self.meta.loc[id_key].target
        return decoded, target, self.keys[index]
