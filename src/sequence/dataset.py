from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class ChickenDataset(Dataset):

    def __init__(self, dataset_dir):

        dataset_dir = Path(dataset_dir)

        self.X = np.load(dataset_dir / "X.npy")

        self.y = np.load(dataset_dir / "y.npy")

        print("=" * 50)
        print("Chicken Dataset Loaded")
        print(f"Samples : {len(self.X)}")
        print(f"Shape   : {self.X.shape}")
        print("=" * 50)

    def __len__(self):

        return len(self.X)

    def __getitem__(self, index):

        x = torch.tensor(self.X[index], dtype=torch.float32)

        y = torch.tensor(self.y[index], dtype=torch.long)

        return x, y
