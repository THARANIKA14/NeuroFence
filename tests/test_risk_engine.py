"""
Tests for scanner.risk_engine

Owner: Hariom Tiwari (Testing & Evaluation)
Status: STUB — awaiting real implementation of scanner/risk_engine.py.

Covers Week 3 goal: combine multiple signals (anomaly score + backdoor score)
into a single Low / Medium / High risk rating.
"""

import pytest

# from scanner.risk_engine import calculate_risk  # uncomment once implemented


@pytest.mark.skip(reason="risk_engine.py not yet implemented")
def test_clean_model_scores_low_risk():
    pass


@pytest.mark.skip(reason="risk_engine.py not yet implemented")
def test_backdoored_model_scores_high_risk():
    pass


@pytest.mark.skip(reason="risk_engine.py not yet implemented")
def test_ambiguous_signals_score_medium_risk():
    pass
