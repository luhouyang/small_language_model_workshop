# Basic AI Modeling

As simple as you can get.

## Model

1. Check the [dataset.py](src/language_models/src/dataset.py) to know the shape of input and output.
    ```python
    # (n, 2) this means we have n points each with 2 dimensions, 2D data
    self.data = (torch.rand((n, 2)) * 2) - 1
    .
    .
    .
    # All 2D points are made into list, the output shape will be (n, 1)
    dist_from_center = torch.sqrt(x**2 + y**2)
    face = dist_from_center < 0.8
    is_smiley = face & ~left_eye & ~right_eye & ~mouth_arc
    return is_smiley.long()
    ```

1. Import dependencies and create boiler plate class.
    ```python
    import torch
    import torch.nn as nn

    class Perceptron(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            pass
    ```

1. For this task we only want a simple, multilayer perceptron model. In the init function:
    ```python
    self.model = nn.Sequential(
        nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
    )
    ```

1. In the `forward()` method pass the input `x` through `self.model`
    ```python
    return self.model(x)
    ```

## Training Loop

1. I always write comments first of the overall structure.
    ```python
    # imports

    # main
        # load data

        # model, loss, optimizer

        # train loop

        # eval

        # save
    ```

1. Import the libraries:
    ```python
    # ML related
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from torch.optim import AdamW

    # Other libraries
    import matplotlib.pyplot as plt

    # User defined
    from dataset import DataGenerator
    from model import Perceptron
    ```

1. Discover device:
    ```python
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    ```

1. Load dataset
    ```python
    train_ds = DataGenerator(n=50000)
    val_ds = DataGenerator(n=50000)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64)
    ```

1. Initialize model, loss, optimizer
    ```python
    model = Perceptron().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=1e-3)
    ```

1. Training loop
    ```python
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
    ```

1. Print results
    ```python
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
    ```

    Usage:
    ```python
    plot_results(model, val_ds, device)
    ```