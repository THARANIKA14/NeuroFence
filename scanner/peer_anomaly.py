from __future__ import annotations

import math
import re
from collections import defaultdict
from statistics import median
from typing import Any, Dict, List

MAD_TO_STD = 1.4826
Z_FOR_FULL_SCORE = 12.0    # z = 12 -> score 100 (moderate z>=3.6, anomalous z>=7.2)
AMAX_WEIGHT = 0.5          # absolute max is heavy-tailed in real models: count it at half weight
LOG_FLOOR = 0.10           # ignore log-variations smaller than ~10 %
MIN_ELEMENTS = 100_000     # score weight matrices only; norm/bias vectors differ systematically by depth
MIN_PEERS = 4


def _group_key(name: str) -> str:
    """'h.3.mlp.c_fc.weight' -> 'h.N.mlp.c_fc.weight'."""
    key = re.sub(r"\.\d+(?=\.|$)", ".N", name)
    return re.sub(r"^\d+(?=\.)", "N", key)


BUFFER_NAME_PARTS = ("masked_bias", "inv_freq", "position_ids",
                     "causal_mask", "rotary_emb.cos", "rotary_emb.sin")


def is_mask_buffer(name: str, s: Dict[str, Any]) -> bool:
    low = name.lower()
    if any(part in low for part in BUFFER_NAME_PARTS):
        return True
    if low.endswith((".attn.bias", ".attention.bias")) \
            and _f(s.get("element_count"), 0) >= MIN_ELEMENTS:
        return True      # real bias vectors are far smaller than a mask
    return (_f(s.get("min")) == 0.0 and _f(s.get("max")) == 1.0
            and abs(_f(s.get("mean")) - 0.5) < 0.05
            and abs(_f(s.get("std")) - 0.5) < 0.05
            and _f(s.get("element_count"), 0) >= MIN_ELEMENTS)


def _f(value: Any, default: float = 0.0) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    return v if math.isfinite(v) else default


def _robust_z(x: float, values: List[float], floor: float) -> float:
    med = median(values)
    mad = median(abs(v - med) for v in values) * MAD_TO_STD
    return abs(x - med) / max(mad, floor)


def _classify(score: float) -> str:
    if score < 30:
        return "normal"
    if score < 60:
        return "moderate"
    return "anomalous"


def detect_peer_anomalies(
    layer_statistics: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    groups: Dict[str, List[str]] = defaultdict(list)
    small = set()
    buffers = set()
    for name, s in layer_statistics.items():
        if is_mask_buffer(name, s):
            buffers.add(name)    # constant 0/1 masks, not weights
            continue
        if _f(s.get("element_count"), MIN_ELEMENTS) < MIN_ELEMENTS:
            small.add(name)      # LayerNorm gains, biases, ...
            continue
        groups[_group_key(name)].append(name)

    log_std, log_amax = {}, {}
    for name, s in layer_statistics.items():
        log_std[name] = math.log(max(_f(s.get("std")), 1e-12))
        amax = _f(s.get("absolute_max"),
                  max(abs(_f(s.get("min"))), abs(_f(s.get("max")))))
        log_amax[name] = math.log(max(amax, 1e-12))

    results: List[Dict[str, Any]] = []
    for name, s in layer_statistics.items():
        members = groups.get(_group_key(name), [])
        bad_numbers = (_f(s.get("nan_count")) > 0
                       or _f(s.get("inf_count")) > 0)

        metrics: Dict[str, Any] = {}
        if name in buffers:
            score = 0.0
            reason = "not scored: non-learned buffer (mask / rotary / position)"
        elif bad_numbers:
            score, reason = 100.0, "contains NaN or Inf values"
        elif name in small:
            score = 0.0
            reason = ("not scored: small vector (norm/bias), "
                      "not comparable across depth")
        elif len(members) < MIN_PEERS:
            score = 0.0
            reason = (f"not scored: only {len(members)} tensor(s) of this "
                      f"kind, no peers to compare with")
        else:
            std_vals = [log_std[m] for m in members]
            amax_vals = [log_amax[m] for m in members]
            mean_vals = [_f(layer_statistics[m].get("mean"))
                         for m in members]
            med_std = math.exp(median(std_vals))

            z_std = _robust_z(log_std[name], std_vals, LOG_FLOOR)
            z_amax = _robust_z(log_amax[name], amax_vals, LOG_FLOOR)
            z_mean = _robust_z(_f(s.get("mean")), mean_vals,
                               0.05 * med_std)

            metrics = {
                "std": {"z": round(z_std, 2)},
                "absolute_max": {"z": round(z_amax, 2)},
                "mean": {"z": round(z_mean, 2)},
            }
            worst_metric, z = max(
                (("std", z_std), ("absolute_max", z_amax * AMAX_WEIGHT),
                 ("mean", z_mean)),
                key=lambda kv: kv[1],
            )
            score = min(100.0, 100.0 * z / Z_FOR_FULL_SCORE)
            peer_med = {
                "std": med_std,
                "absolute_max": math.exp(median(amax_vals)),
                "mean": median(mean_vals),
            }[worst_metric]
            value = {
                "std": _f(s.get("std")),
                "absolute_max": math.exp(log_amax[name]),
                "mean": _f(s.get("mean")),
            }[worst_metric]
            reason = (f"{worst_metric} {value:.4g} vs peer median "
                      f"{peer_med:.4g} ({len(members)} peers, z={z:.1f})")

        results.append({
            "layer": name,
            "anomaly_score": round(score, 2),
            "classification": _classify(score),
            "reason": reason,
            "metrics": metrics,
        })

    results.sort(key=lambda r: r["anomaly_score"], reverse=True)
    scores = [r["anomaly_score"] for r in results]
    worst = scores[0] if scores else 0.0
    mean_score = sum(scores) / len(scores) if scores else 0.0

    return {
        # Worst tensor drives the model score: a mean would dilute a
        # single tampered tensor among hundreds of normal ones.
        "model_anomaly_score": round(worst, 2),
        "mean_score": round(mean_score, 2),
        "classification": _classify(worst),
        "total_layers": len(results),
        "anomalous_layers": sum(
            r["classification"] == "anomalous" for r in results),
        "moderate_layers": sum(
            r["classification"] == "moderate" for r in results),
        "unscored_layers": sum(
            r["reason"].startswith("not scored") for r in results),
        "layer_results": results,
    }

