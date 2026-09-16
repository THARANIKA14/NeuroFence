# src/anomalies.py
import numpy as np

def detect_anomalous_layers(layer_stats, z_threshold=2.0):
    """
    layer_stats: list of dict with keys: name, std, mean, etc.
    Detect layers whose std is anomalous compared to others.
    Returns list of dict with name, stats, is_anomaly, z_score.
    """
    stds = np.array([s["std"] for s in layer_stats])
    mean_std = np.mean(stds)
    std_std = np.std(stds) if len(stds) > 1 else 1e-9

    results = []
    for s in layer_stats:
        z = (s["std"] - mean_std) / std_std if std_std > 0 else 0.0
        is_anomaly = abs(z) > z_threshold
        results.append({
            "name": s["name"],
            "std": s["std"],
            "mean": s["mean"],
            "z_score": float(z),
            "is_anomaly": bool(is_anomaly),
        })
    return results