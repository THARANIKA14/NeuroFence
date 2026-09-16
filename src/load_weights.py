# src/load_weights.py
import json
import numpy as np
from pathlib import Path

def load_weights_from_json(json_path: str):
    """
    Load weights from a JSON file with structure:
    {
      "model_name": "...",
      "layers": [
        {"name": "...", "weights": [[...], [...]]},
        ...
      ]
    }
    Returns:
      model_name: str
      layers: list of dict with keys: name, weights (np.ndarray)
    """
    path = Path(json_path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    model_name = data.get("model_name", "unknown")
    layers = []
    for layer in data.get("layers", []):
        name = layer["name"]
        weights = np.array(layer["weights"], dtype=float)
        layers.append({"name": name, "weights": weights})

    return model_name, layers