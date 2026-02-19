"""
By:             Lu Hou Yang
Last updated:   19th Feb 2025

cats-and-dogs-mini-dataset
"""

import os
from PIL import Image
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset

class DataGenerator(Dataset):
    def __init__(self, n=10000):
        super().__init__()

        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, index):
        pass


if __name__ == "__main__":
    pass