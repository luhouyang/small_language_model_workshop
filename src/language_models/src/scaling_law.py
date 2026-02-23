import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
import matplotlib.pyplot as plt
import numpy as np
from dataset import DataGenerator
from tqdm import tqdm

def calculate_metrics(model, loader, device):
    """Calculates final validation accuracy and IoU for the smiley class."""
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            all_preds.append(preds.cpu())
            all_targets.append(targets.cpu())

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)

    accuracy = (all_preds == all_targets).float().mean().item()
    intersection = ((all_preds == 1) & (all_targets == 1)).float().sum()
    union = ((all_preds == 1) | (all_targets == 1)).float().sum()
    iou = (intersection / (union + 1e-6)).item()

    return accuracy, iou


def run_experiment():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_sizes = [1000, 5000, 10000, 20000]
    model_widths = [8, 16, 32, 64]
    model_depths = [1, 2, 3]  
    val_size = 10000
    epochs = 10

    val_ds = DataGenerator(n=val_size)
    val_loader = DataLoader(val_ds, batch_size=256)
    results = {}

    total_experiments = len(model_depths) * len(model_widths) * len(dataset_sizes)
    pbar = tqdm(total=total_experiments, desc="Running Scaling Law Experiments")

    # Training Loop
    for depth in model_depths:
        for width in model_widths:
            results[(depth, width)] = {"acc": [], "iou": [], "best_loss": []}
            for n_train in dataset_sizes:
                layers = [nn.Linear(2, width), nn.ReLU()]
                for _ in range(depth - 1):
                    layers.extend([nn.Linear(width, width), nn.ReLU()])
                layers.append(nn.Linear(width, 2))

                model = nn.Sequential(*layers).to(device)
                train_ds = DataGenerator(n=n_train)
                train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
                optimizer = AdamW(model.parameters(), lr=1e-3)
                criterion = nn.CrossEntropyLoss()

                model.train()
                best_training_loss = float('inf')
                
                for epoch in range(epochs):
                    for inputs, targets in train_loader:
                        inputs, targets = inputs.to(device), targets.to(device)
                        optimizer.zero_grad()
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)
                        loss.backward()
                        optimizer.step()
                        if loss.item() < best_training_loss:
                            best_training_loss = loss.item()

                acc, iou = calculate_metrics(model, val_loader, device)
                results[(depth, width)]["acc"].append(acc)
                results[(depth, width)]["iou"].append(iou)
                results[(depth, width)]["best_loss"].append(best_training_loss)
                
                # Print individual experiment outcome
                tqdm.write(f"[Result] Depth: {depth} | Width: {width} | N: {n_train} | Acc: {acc:.4f} | IoU: {iou:.4f} | Loss: {best_training_loss:.4f}")
                
                pbar.update(1)
    
    pbar.close()

    # Scaling Law Plots (Grouped by Depth)
    for depth in model_depths:
        fig, axs = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f'Scaling Laws for Model Depth: {depth}', fontsize=16)
        for width in model_widths:
            axs[0].plot(dataset_sizes, results[(depth, width)]["acc"], marker='o', label=f'Width {width}')
            axs[1].plot(dataset_sizes, results[(depth, width)]["iou"], marker='s', label=f'Width {width}')
            axs[2].plot(dataset_sizes, results[(depth, width)]["best_loss"], marker='^', label=f'Width {width}')
        
        axs[0].set_title('Validation Accuracy')
        axs[1].set_title('Smiley IoU')
        axs[2].set_title('Min Training Loss')

        for i, ax in enumerate(axs):
            ax.set_xscale('log')
            ax.set_xlabel('N Samples')
            ax.grid(True, which="both", ls="-", alpha=0.3)
            ax.legend()
            if i == 2: ax.set_yscale('log')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(f"scaling_results_depth_{depth}.png")
        plt.close()

    # --- Updated Final Comparison Plot: Accuracy, IoU, and Loss ---
    comparison_width = 64
    target_n_index = dataset_sizes.index(20000)
    
    depth_labels = [f"Depth {d}" for d in model_depths]
    comp_acc = [results[(d, comparison_width)]["acc"][target_n_index] for d in model_depths]
    comp_iou = [results[(d, comparison_width)]["iou"][target_n_index] for d in model_depths]
    comp_loss = [results[(d, comparison_width)]["best_loss"][target_n_index] for d in model_depths]

    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'Depth Comparison (Width: {comparison_width}, N: 20000)', fontsize=14)

    # Accuracy Bar
    ax[0].bar(depth_labels, comp_acc, color='skyblue')
    ax[0].set_title('Validation Accuracy')
    ax[0].set_ylim(min(comp_acc) - 0.02, max(comp_acc) + 0.02)
    for i, v in enumerate(comp_acc):
        ax[0].text(i, v + 0.001, f"{v:.4f}", ha='center')

    # IoU Bar
    ax[1].bar(depth_labels, comp_iou, color='salmon')
    ax[1].set_title('Smiley IoU')
    ax[1].set_ylim(min(comp_iou) - 0.02, max(comp_iou) + 0.02)
    for i, v in enumerate(comp_iou):
        ax[1].text(i, v + 0.001, f"{v:.4f}", ha='center')

    # Loss Bar
    ax[2].bar(depth_labels, comp_loss, color='lightgreen')
    ax[2].set_title('Min Training Loss')
    ax[2].set_yscale('log')
    for i, v in enumerate(comp_loss):
        ax[2].text(i, v, f"{v:.2e}", ha='center', va='bottom')

    for a in ax:
        a.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("depth_comparison_full_metrics.png")
    plt.show()

if __name__ == "__main__":
    run_experiment()