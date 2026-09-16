# src/stats.py
import numpy as np

def compute_layer_stats(weights: np.ndarray) -> dict:
    """
    Compute basic statistics for a weight tensor.
    """
    flat = weights.ravel()
    return {
        "mean": float(np.mean(flat)),
        "std": float(np.std(flat)),
        "min": float(np.min(flat)),
        "max": float(np.max(flat)),
        "abs_max": float(np.max(np.abs(flat))),
        "sparsity": float(np.mean(np.abs(flat) < 1e-6)),  # fraction near zero
    }

def compute_all_layer_stats(layers):
    """
    layers: list of dict with keys: name, weights (np.ndarray)
    returns: list of dict with name + stats
    """
    results = []
    for layer in layers:
        stats = compute_layer_stats(layer["weights"])
        results.append({"name": layer["name"], **stats})
    return results