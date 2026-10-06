from __future__ import annotations

import gc
import hashlib
from pathlib import Path
from typing import Any, Dict, Optional

from scanner.integrity_checker import IntegrityChecker
from scanner.peer_anomaly import detect_peer_anomalies
from scanner.risk_engine import RiskEngine
from scanner.weight_analyzer import WeightAnalyzer


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def risk_anomaly_component(anomaly: Dict[str, Any]) -> float:
    worst = float(anomaly.get("model_anomaly_score", 0.0))
    if anomaly.get("anomalous_layers", 0) > 0:
        return worst
    return worst * 0.5


class _StreamingModel:

    def __init__(self, path: Path) -> None:
        self._path = path

    def named_parameters(self):
        from safetensors import safe_open

        with safe_open(str(self._path), framework="pt", device="cpu") as f:
            for name in f.keys():
                yield name, f.get_tensor(name)


def scan_safetensors_file(
    path: Path,
    file_name: str,
    trusted_hash: Optional[str] = None,
) -> Dict[str, Any]:
    path = Path(path)
    sha256 = _sha256_file(path)

    analyzer = WeightAnalyzer()
    analysis = analyzer.analyze_model(_StreamingModel(path))
    gc.collect()

    anomaly = detect_peer_anomalies(analysis["layer_statistics"])

    checker = IntegrityChecker()
    cmp = checker.compare_hashes(sha256, trusted_hash or "")
    integrity = {
        "integrity_status": cmp["status"],
        "integrity_verified": cmp["match"],
    }

    # Only weight analysis (+ optional hash) ran, so score on those alone.
    engine = RiskEngine(weights={
        "anomaly": 0.85, "behavioral": 0.0,
        "integrity": 0.15, "validation": 0.0,
    })
    risk = engine.calculate_risk(
        anomaly_result={
            **anomaly,
            "model_anomaly_score": risk_anomaly_component(anomaly),
        },
        integrity_result=integrity,
    )

    layer_results = anomaly.get("layer_results", [])
    return {
        "file_name": file_name,
        "size_bytes": path.stat().st_size,
        "sha256": sha256,
        "analysis": analysis,
        "anomaly": anomaly,
        "integrity": {**cmp, "sha256": sha256,
                      "trusted_hash": trusted_hash or ""},
        "risk": risk,
        "worst_layer": layer_results[0] if layer_results else None,
        "behavioral_run": False,
    }
