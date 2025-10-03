import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image

class NPYDataset(Dataset):
    """
    Dataset class for CIFAR-10 that loads data from .npy files
    created by generate_data.py.
    """
    def __init__(self, data_path, train=True, transform=None):
        if train:
            self.data = np.load(os.path.join(data_path, "train_data.npy"))
            self.targets = np.load(os.path.join(data_path, "train_targets.npy"))
        else:
            self.data = np.load(os.path.join(data_path, "test_data.npy"))
            self.targets = np.load(os.path.join(data_path, "test_targets.npy"))

        self.targets = self.targets.astype(np.int64)  # ensure torch CE works
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y = self.data[idx], self.targets[idx]

        # Make sure the image is in HWC and 3-channel format
        if x.shape[-1] != 3:
            raise ValueError(f"Expected image with 3 channels, got shape {x.shape}")

        # Convert numpy array (32,32,3) to PIL image for transforms
        x = Image.fromarray(x.astype("uint8"))

        if self.transform:
            x = self.transform(x)

        return x, int(y)
