# src/heatmaps.py
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

def plot_layer_heatmap(weights: np.ndarray, title: str, output_path: str):
    """
    Plot a heatmap for a 2D weight matrix.
    If weights are not 2D, take a 2D slice or reshape if possible.
    """
    if weights.ndim != 2:
        # Simple fallback: take first two dimensions or flatten to 2D
        if weights.ndim > 2:
            weights = weights[tuple(slice(None) if i < 2 else 0 for i in range(weights.ndim))]
        else:
            # 1D: reshape to (1, N)
            weights = weights.reshape(1, -1)

    plt.figure(figsize=(6, 5))
    sns.heatmap(weights, cmap="viridis", cbar=True)
    plt.title(title)
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()