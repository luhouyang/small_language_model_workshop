"""
By:             Lu Hou Yang
Last updated:   19th Feb 2025

Training and Evaluation loop
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
import matplotlib.pyplot as plt
from torchinfo import torchinfo

from dataset import DataGenerator
from model import Perceptron

def main():
    # 1. setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. load dataset
    train_ds = DataGenerator(n=10e5)
    val_ds = DataGenerator(n=10e1)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, pin_memory=True)

    # 3. initialize model, loss & optimizer
    model = Perceptron().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=1e-3)

    torchinfo.summary(model)

    # 4. simple training loop
    model.train()
    epochs = 10
    for epoch in range(epochs):
        running_loss = 0.0

    # 5. validation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        pass

    print(f"Final Validation Accuracy: {100 * correct / total:.2f}%")

    # 6. save the model
    torch.save(model.state_dict(), "simple_perceptron.pth")
    print("Model saved to simple_perceptron.pth")

if __name__ == "__main__":
    main()