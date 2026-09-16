# src/dashboard.py
from pathlib import Path
from .load_weights import load_weights_from_json
from .stats import compute_all_layer_stats
from .heatmaps import plot_layer_heatmap
from .anomalies import detect_anomalous_layers
from .charts import plot_std_per_layer, plot_mean_per_layer

def generate_dashboard(json_path: str, output_dir: str):
    """
    End-to-end pipeline:
    - Load weights
    - Compute stats
    - Generate heatmaps for each layer
    - Detect anomalies
    - Generate charts
    - Save everything under output_dir
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load
    model_name, layers = load_weights_from_json(json_path)

    # 2. Stats
    layer_stats = compute_all_layer_stats(layers)

    # 3. Heatmaps
    heatmaps_dir = output_dir / "heatmaps"
    heatmaps_dir.mkdir(exist_ok=True)
    for layer in layers:
        safe_name = layer["name"].replace("/", "_").replace("\\", "_")
        plot_layer_heatmap(
            layer["weights"],
            title=f"{layer['name']} weights",
            output_path=str(heatmaps_dir / f"{safe_name}.png"),
        )

    # 4. Anomalies
    anomalies = detect_anomalous_layers(layer_stats)

    # Save anomaly summary as text
    with (output_dir / "anomalies.txt").open("w", encoding="utf-8") as f:
        f.write(f"Model: {model_name}\n")
        f.write("Anomalous layers (by std z-score):\n")
        for a in anomalies:
            if a["is_anomaly"]:
                f.write(f"- {a['name']} (z={a['z_score']:.2f}, std={a['std']:.4f})\n")

    # 5. Charts
    plot_std_per_layer(layer_stats, str(output_dir / "std_per_layer.png"))
    plot_mean_per_layer(layer_stats, str(output_dir / "mean_per_layer.png"))

    return {
        "model_name": model_name,
        "layer_stats": layer_stats,
        "anomalies": anomalies,
        "output_dir": str(output_dir),
    }