import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from settings import DEVICE, agent_settings


def plot_statistics(data: list, plot_name: str, y_label: str, rolling_window: int = 50):
    """
    Plots training or evaluation statistics and saves the figure to disk.

    Args:
        data (list): Sequence of values to plot (e.g., rewards, losses, episode lengths).
        plot_name (str): Title and filename identifier for the plot.
        y_label (str): Label for the y-axis.
        rolling_window (int): Window size for computing the rolling mean.

    Returns:
        None
    """

    episodes = np.arange(1, len(data) + 1)

    plt.style.use("seaborn-v0_8")
    plt.figure(figsize=(9, 6))

    plt.plot(episodes, data, color="lightcoral", alpha=0.6, label="Raw")

    if len(data) >= rolling_window:
        rolling = np.convolve(data, np.ones(rolling_window)/rolling_window, mode="valid")
        plt.plot(episodes[rolling_window-1:], rolling, color="red", linewidth=2.5, label=f"Rolling mean ({rolling_window})")

    plt.title(f"Training Progress - {plot_name}", fontsize=16, fontweight="bold")
    plt.xlabel("Episode", fontsize=14)
    plt.ylabel(y_label, fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            agent_settings.plot_path, plot_name.replace(' ', '_')
        ), dpi=200
    )
    plt.close()


def checker(X, name) -> None:
    """
    Checks a tensor for NaN or infinite values and prints warnings.

    Args:
        X (torch.Tensor): Tensor to check for numerical instability.
        name (str): Name identifier for the tensor.

    Returns:
        None
    """
    if torch.isnan(X).any():
        print(f"NaN in {name}")
    if torch.isinf(X).any():
        print(f"Inf in {name}")


def tensor(x, device=DEVICE, dtype=torch.float32) -> torch.Tensor:
    """
    Converts input data to a PyTorch tensor on the specified device and dtype.

    Args:
        x (Any): Input data (list, numpy array, or tensor).
        device (torch.device): Target device for the tensor.
        dtype (torch.dtype): Desired tensor data type.

    Returns:
        torch.Tensor: Converted tensor.
    """
    if isinstance(x, torch.Tensor):
        x = x.detach()
        if x.device != device:
            x = x.to(device)
        if x.dtype != dtype:
            x = x.to(dtype)
        return x

    return torch.tensor(x, device=device, dtype=dtype)