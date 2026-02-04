import torch
import matplotlib.pyplot as plt

def compute_l2_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        total_norm += torch.sum(p ** 2)
    return total_norm.item()


def plot_losses(train_losses, val_losses):
    plt.plot(train_losses, label="Train")
    plt.plot(val_losses, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


def check_overfitting(train_losses, val_losses):
    return val_losses[-1] > train_losses[-1]
