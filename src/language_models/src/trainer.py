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
from dataset import DataGenerator
from model import Perceptron

def plot_results(model, dataset, device):
    model.eval()
    data, labels = dataset.data, dataset.labels
    with torch.no_grad():
        outputs = model(data.to(device))
        predictions = torch.argmax(outputs, dim=1).cpu()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Actual Data
    ax1.scatter(data[:, 0], data[:, 1], c=labels, cmap='coolwarm', s=1)
    ax1.set_title("Actual Data (Smiley Face)")
    
    # Predicted Data
    ax2.scatter(data[:, 0], data[:, 1], c=predictions, cmap='coolwarm', s=1)
    ax2.set_title("Model Predictions")
    
    plt.show()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 2. Load dataset
    train_ds = DataGenerator(n=50000)
    val_ds = DataGenerator(n=50000)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64)

    # 3. Initialize
    model = Perceptron().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=1e-3)

    # 4. Training loop
    model.train()
    for epoch in range(20):
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        if epoch % 5 == 0:
            print(f"Epoch {epoch} | Loss: {loss.item():.4f}")

    # 5. Print Actual vs Predicted
    plot_results(model, val_ds, device)

if __name__ == "__main__":
    main()