"""
By:             Lu Hou Yang
Last updated:   19th Feb 2025

cats-and-dogs-mini-dataset
"""

import torch
from torch.utils.data import Dataset

class CatsNDogsDataset(Dataset):
    def __init__(self, root="", split=0.05):
        super().__init__()

    def __len__(self):
        pass

    def __getitem__(self, index):
        return super().__getitem__(index)

    def data_augmentation(self, image):
        pass
