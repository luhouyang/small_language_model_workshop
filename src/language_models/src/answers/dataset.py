"""
By:             Lu Hou Yang
Last updated:   19th Feb 2025

data generator
"""

import torch
from torch.utils.data import Dataset
import numpy as np

class DataGenerator(Dataset):
    def __init__(self, n=10000):
        super().__init__()
        self.n = n
        # Generate random 2D points between -1 and 1
        self.data = (torch.rand((n, 2)) * 2) - 1
        self.labels = self._generate_smiley_labels(self.data)

    def _generate_smiley_labels(self, points):
        x = points[:, 0]
        y = points[:, 1]
        
        # Face (Circle)
        dist_from_center = torch.sqrt(x**2 + y**2)
        face = dist_from_center < 0.8
        
        # Eyes (Two small circles)
        left_eye = torch.sqrt((x + 0.3)**2 + (y - 0.3)**2) < 0.1
        right_eye = torch.sqrt((x - 0.3)**2 + (y - 0.3)**2) < 0.1
        
        # Mouth (Half-circle arc)
        mouth_arc = (torch.sqrt(x**2 + (y + 0.1)**2) < 0.4) & \
                    (torch.sqrt(x**2 + (y + 0.1)**2) > 0.3) & \
                    (y < -0.1)

        # Combine: Face base minus eyes and mouth
        is_smiley = face & ~left_eye & ~right_eye & ~mouth_arc
        return is_smiley.long()

    def __len__(self):
        return self.n

    def __getitem__(self, index):
        return self.data[index], self.labels[index]