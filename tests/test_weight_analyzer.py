"""
Uses real PyTorch tensors when torch is installed. If torch is not
available (e.g. a minimal CI box), a small numpy-backed stand-in that
implements only the tensor operations WeightAnalyzer uses is used, so
the suite still runs. Install torch locally to test against the real thing.
"""

import math

import numpy as np
import pytest

from scanner.weight_analyzer import WeightAnalyzer, analyze_model_weights

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


# ---------------------------------------------------------------
# Minimal tensor stand-in (only used when torch is missing)
# ---------------------------------------------------------------

class _FakeTensor:
    def __init__(self, array):
        self.a = np.asarray(array, dtype=np.float32)

    @property
    def shape(self):
        return tuple(self.a.shape)

    @property
    def dtype(self):
        return "torch.float32"

    def numel(self):
        return int(self.a.size)

    def detach(self):
        return self

    def float(self):
        return self

    def cpu(self):
        return self

    def reshape(self, *shape):
        return _FakeTensor(self.a.reshape(*shape))

    def isnan(self):
        return _FakeTensor(np.isnan(self.a))

    def isinf(self):
        return _FakeTensor(np.isinf(self.a))

    def sum(self):
        return _FakeTensor(np.sum(self.a))

    def item(self):
        return float(self.a)

    def mean(self):
        return _FakeTensor(self.a.mean())

    def var(self, unbiased=True):
        return _FakeTensor(self.a.var(ddof=1 if unbiased else 0))

    def std(self, unbiased=True):
        return _FakeTensor(self.a.std(ddof=1 if unbiased else 0))

    def min(self):
        return _FakeTensor(self.a.min())

    def max(self):
        return _FakeTensor(self.a.max())

    def abs(self):
        return _FakeTensor(np.abs(self.a))

    @staticmethod
    def _raw(other):
        return other.a if isinstance(other, _FakeTensor) else other

    def __eq__(self, other):
        return _FakeTensor(self.a == self._raw(other))

    def __gt__(self, other):
        return _FakeTensor(self.a > self._raw(other))

    def __lt__(self, other):
        return _FakeTensor(self.a < self._raw(other))

    def __and__(self, other):
        return _FakeTensor(np.logical_and(self.a, self._raw(other)))

    def __invert__(self):
        return _FakeTensor(np.logical_not(self.a))

    def __getitem__(self, key):
        key = self._raw(key)
        if isinstance(key, np.ndarray) and key.dtype != bool:
            key = key.astype(bool)
        return _FakeTensor(self.a[key])

    __hash__ = None


def make_tensor(values):
    arr = np.asarray(values, dtype=np.float32)
    if torch is not None:
        return torch.tensor(arr)
    return _FakeTensor(arr)


class FakeModel:
    """Anything exposing named_parameters() is accepted by the analyzer."""

    def __init__(self, params):
        self._params = params

    def named_parameters(self):
        return iter(self._params.items())


def clean_model(seed=0):
    rng = np.random.default_rng(seed)
    return FakeModel({
        "layer1.weight": make_tensor(rng.normal(0, 0.02, (8, 8))),
        "layer1.bias": make_tensor(rng.normal(0, 0.02, (8,))),
        "layer2.weight": make_tensor(rng.normal(0, 0.02, (8, 8))),
    })


# ---------------------------------------------------------------
# Per-layer statistics
# ---------------------------------------------------------------

def test_extracts_per_layer_weight_statistics():
    model = FakeModel({"fc.weight": make_tensor([[1.0, 2.0], [3.0, 4.0]])})

    result = WeightAnalyzer().analyze_model(model)

    stats = result["layer_statistics"]["fc.weight"]
    assert stats["element_count"] == 4
    assert stats["mean"] == pytest.approx(2.5)
    assert stats["std"] == pytest.approx(math.sqrt(1.25), abs=1e-6)
    assert stats["min"] == 1.0
    assert stats["max"] == 4.0
    assert stats["positive_count"] == 4
    assert result["total_parameters"] == 4
    assert result["analyzed_tensors"] == 1
    assert result["parameter_details"][0]["shape"] == [2, 2]


def test_tensor_statistics_counts_zeros_and_signs():
    stats = WeightAnalyzer().calculate_tensor_statistics(
        make_tensor([0.0, 0.0, -1.0, 2.0])
    )
    assert stats["zero_count"] == 2
    assert stats["zero_ratio"] == pytest.approx(0.5)
    assert stats["positive_count"] == 1
    assert stats["negative_count"] == 1
    assert stats["absolute_max"] == 2.0


def test_empty_tensor_returns_zeroed_statistics():
    stats = WeightAnalyzer().calculate_tensor_statistics(make_tensor([]))
    assert stats["element_count"] == 0
    assert stats["mean"] == 0.0
    assert stats["nan_count"] == 0


def test_nan_and_inf_are_counted_and_excluded_from_stats():
    stats = WeightAnalyzer().calculate_tensor_statistics(
        make_tensor([1.0, 3.0, float("nan"), float("inf")])
    )
    assert stats["nan_count"] == 1
    assert stats["inf_count"] == 1
    assert stats["mean"] == pytest.approx(2.0)  # only finite values used


def test_all_non_finite_tensor_does_not_crash():
    stats = WeightAnalyzer().calculate_tensor_statistics(
        make_tensor([float("nan"), float("inf")])
    )
    assert stats["nan_count"] == 1
    assert stats["inf_count"] == 1
    assert stats["mean"] == 0.0


# ---------------------------------------------------------------
# Anomalous-layer screening
# ---------------------------------------------------------------

def test_flags_layer_with_abnormal_standard_deviation():
    model = clean_model()
    model._params["layer2.weight"] = make_tensor(
        np.random.default_rng(1).normal(0, 0.02, (8, 8)) * 0
        + np.array([[50.0] + [0.01] * 7] * 8)
    )

    analyzer = WeightAnalyzer()
    result = analyzer.analyze_model(model)
    flagged = analyzer.find_extreme_layers(result["layer_statistics"])

    assert [f["layer"] for f in flagged] == ["layer2.weight"]
    stds = result["layer_statistics"]
    assert stds["layer2.weight"]["std"] > 100 * stds["layer1.weight"]["std"]


def test_clean_model_has_no_anomalous_layers():
    analyzer = WeightAnalyzer()
    result = analyzer.analyze_model(clean_model())
    assert analyzer.find_extreme_layers(result["layer_statistics"]) == []


def test_extreme_layers_are_sorted_largest_first():
    stats = {
        "a": {"absolute_max": 6.0},
        "b": {"absolute_max": 9.0},
        "c": {"absolute_max": 1.0},
    }
    flagged = WeightAnalyzer().find_extreme_layers(stats, absolute_threshold=5.0)
    assert [f["layer"] for f in flagged] == ["b", "a"]


@pytest.mark.parametrize("bad", [0, -1.0])
def test_non_positive_threshold_rejected(bad):
    with pytest.raises(ValueError):
        WeightAnalyzer().find_extreme_layers({}, absolute_threshold=bad)


# ---------------------------------------------------------------
# Model-level behaviour and invalid input
# ---------------------------------------------------------------

def test_include_bias_false_skips_bias_parameters():
    result = WeightAnalyzer().analyze_model(clean_model(), include_bias=False)
    assert "layer1.bias" not in result["layer_statistics"]
    assert result["skipped_tensors"] == 1
    assert result["analyzed_tensors"] == 2


def test_model_summary_reports_layer_count_and_global_range():
    result = WeightAnalyzer().analyze_model(
        FakeModel({
            "a": make_tensor([-2.0, 1.0]),
            "b": make_tensor([0.5, 3.0]),
        })
    )
    summary = result["model_summary"]
    assert summary["layer_count"] == 2
    assert summary["global_min"] == -2.0
    assert summary["global_max"] == 3.0


def test_none_model_rejected():
    with pytest.raises(ValueError):
        WeightAnalyzer().analyze_model(None)


def test_object_without_named_parameters_rejected():
    with pytest.raises(TypeError):
        WeightAnalyzer().analyze_model(object())


def test_non_tensor_input_rejected():
    with pytest.raises(TypeError):
        WeightAnalyzer().calculate_tensor_statistics([1.0, 2.0])


@pytest.mark.parametrize("bad", [0, -5])
def test_invalid_max_elements_rejected(bad):
    with pytest.raises(ValueError):
        WeightAnalyzer(max_elements_per_tensor=bad)


def test_max_elements_limits_analyzed_values():
    stats = WeightAnalyzer(max_elements_per_tensor=3).calculate_tensor_statistics(
        make_tensor([1.0, 2.0, 3.0, 100.0, 200.0])
    )
    assert stats["element_count"] == 3
    assert stats["max"] == 3.0


def test_get_anomaly_input_has_fields_anomaly_detector_needs():
    analyzer = WeightAnalyzer()
    result = analyzer.analyze_model(clean_model())
    anomaly_input = analyzer.get_anomaly_input(result)

    assert set(anomaly_input) == set(result["layer_statistics"])
    for stats in anomaly_input.values():
        assert {"mean", "std", "min", "max", "variance"} <= set(stats)


def test_convenience_function_matches_class():
    model = clean_model()
    assert (
        analyze_model_weights(model)["total_parameters"]
        == WeightAnalyzer().analyze_model(model)["total_parameters"]
    )
