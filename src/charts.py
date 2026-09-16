# src/charts.py
import matplotlib.pyplot as plt
from pathlib import Path

def plot_std_per_layer(layer_stats, output_path: str):
    names = [s["name"] for s in layer_stats]
    stds = [s["std"] for s in layer_stats]

    plt.figure(figsize=(8, 4))
    plt.bar(range(len(names)), stds, tick_label=names)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Std of weights")
    plt.title("Layer-wise weight standard deviation")
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()

def plot_mean_per_layer(layer_stats, output_path: str):
    names = [s["name"] for s in layer_stats]
    means = [s["mean"] for s in layer_stats]

    plt.figure(figsize=(8, 4))
    plt.bar(range(len(names)), means, tick_label=names)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Mean of weights")
    plt.title("Layer-wise weight mean")
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()