"""
By:             Lu Hou Yang
Last updated:   19th Feb 2025

Basic perceptron model
"""

import torch
import torch.nn as nn

class Perceptron(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        return self.model(x)