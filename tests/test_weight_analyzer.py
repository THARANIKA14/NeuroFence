"""
Tests for scanner.weight_analyzer

scanner/weight_analyzer.py is the one scanner module that has not
landed yet — every other module (model_loader, model_validator,
anomaly_detector, backdoor_detector, integrity_checker, risk_engine)
now has a real implementation on main, but this file is currently
empty (0 lines of code).

These tests are written against the expected contract described in
the NeuroFence project spec (load a model's per-layer weight tensors
and compute statistics — mean/std/z-score — feeding into
anomaly_detector.py) so they can be un-skipped the moment a real
implementation lands.
"""

import pytest


@pytest.mark.skip(reason="scanner/weight_analyzer.py is still empty on main (not yet implemented)")
def test_extracts_per_layer_weight_statistics():
    """Expected: given a loaded model, returns mean/std/shape per layer."""
    pass


@pytest.mark.skip(reason="scanner/weight_analyzer.py is still empty on main (not yet implemented)")
def test_flags_layer_with_abnormal_standard_deviation():
    """Expected: a layer with an outlier std/z-score is flagged as anomalous."""
    pass


@pytest.mark.skip(reason="scanner/weight_analyzer.py is still empty on main (not yet implemented)")
def test_clean_model_has_no_anomalous_layers():
    """Expected: a normally-distributed weight set produces zero flagged layers."""
    pass
