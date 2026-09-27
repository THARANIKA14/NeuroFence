"""
Tests for scanner.risk_engine

Covers the Week 3 goal: combine anomaly / behavioral / integrity /
validation signals into a single Low / Moderate / High / Critical
risk rating.
"""

import pytest

from scanner.risk_engine import RiskEngine, calculate_risk


@pytest.fixture
def engine():
    return RiskEngine()


# ---------------------------------------------------------------
# calculate_risk — end-to-end scenarios
# ---------------------------------------------------------------

def test_clean_model_scores_low_risk(engine):
    result = engine.calculate_risk(
        anomaly_result={"model_anomaly_score": 5.0},
        behavioral_result={"behavioral_score": 2.0, "high_deviation_tests": 0},
        integrity_result={"integrity_status": "verified"},
        validation_result={"valid": True, "issues": [], "warnings": []},
    )

    assert result["risk_level"] == "low"
    assert result["overall_risk_score"] < 25


def test_backdoored_model_scores_high_or_critical_risk(engine):
    result = engine.calculate_risk(
        anomaly_result={"model_anomaly_score": 85.0, "anomalous_layers": 4},
        behavioral_result={"behavioral_score": 90.0, "high_deviation_tests": 6},
        integrity_result={"integrity_status": "mismatch"},
        validation_result={"valid": True, "issues": [], "warnings": []},
    )

    assert result["risk_level"] in {"high", "critical"}
    assert result["overall_risk_score"] >= 50
    assert any("behavioral" in factor.lower() or "anomal" in factor.lower()
               for factor in result["contributing_factors"])


def test_ambiguous_signals_score_moderate_risk(engine):
    result = engine.calculate_risk(
        anomaly_result={"model_anomaly_score": 40.0},
        behavioral_result={"behavioral_score": 35.0, "high_deviation_tests": 0},
        integrity_result={"integrity_status": "verified"},
        validation_result={"valid": True, "issues": [], "warnings": []},
    )

    assert result["risk_level"] in {"moderate", "high"}


def test_missing_signals_default_to_zero_and_low_risk(engine):
    result = engine.calculate_risk()

    assert result["risk_level"] == "low"
    assert result["overall_risk_score"] == 0.0
    assert result["contributing_factors"] == [
        "No major contributing risk factor was identified by the supplied checks."
    ]


def test_result_contains_expected_keys(engine):
    result = engine.calculate_risk(
        anomaly_result={"model_anomaly_score": 50.0},
    )

    expected_keys = {
        "overall_risk_score",
        "risk_level",
        "risk_description",
        "component_scores",
        "component_weights",
        "weighted_scores",
        "contributing_factors",
        "recommendations",
    }
    assert expected_keys.issubset(result.keys())


# ---------------------------------------------------------------
# classify_risk boundaries
# ---------------------------------------------------------------

@pytest.mark.parametrize(
    "score,expected_level",
    [
        (0, "low"),
        (24.99, "low"),
        (25, "moderate"),
        (49.99, "moderate"),
        (50, "high"),
        (74.99, "high"),
        (75, "critical"),
        (100, "critical"),
    ],
)
def test_classify_risk_boundaries(score, expected_level):
    assert RiskEngine.classify_risk(score) == expected_level


# ---------------------------------------------------------------
# Integrity score mapping
# ---------------------------------------------------------------

def test_integrity_mismatch_scores_max(engine):
    assert engine.calculate_integrity_score({"integrity_status": "mismatch"}) == 100.0


def test_integrity_verified_scores_zero(engine):
    assert engine.calculate_integrity_score({"integrity_status": "verified"}) == 0.0


def test_integrity_missing_reference_is_informational_only(engine):
    assert engine.calculate_integrity_score({"integrity_status": "no_reference"}) == 0.0


# ---------------------------------------------------------------
# Weight validation
# ---------------------------------------------------------------

def test_default_weights_sum_to_one(engine):
    assert pytest.approx(sum(engine.weights.values()), rel=1e-6) == 1.0


def test_custom_weights_are_normalized():
    custom = RiskEngine(weights={"anomaly": 2, "behavioral": 2, "integrity": 0, "validation": 0})
    assert pytest.approx(sum(custom.weights.values()), rel=1e-6) == 1.0
    assert custom.weights["anomaly"] == pytest.approx(0.5)


def test_missing_weight_key_raises():
    with pytest.raises(ValueError):
        RiskEngine(weights={"anomaly": 1.0, "behavioral": 1.0})


def test_negative_weight_raises():
    with pytest.raises(ValueError):
        RiskEngine(weights={"anomaly": -1, "behavioral": 1, "integrity": 1, "validation": 1})


def test_all_zero_weights_raise():
    with pytest.raises(ValueError):
        RiskEngine(weights={"anomaly": 0, "behavioral": 0, "integrity": 0, "validation": 0})


# ---------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------

def test_calculate_risk_convenience_function_matches_engine():
    result = calculate_risk(
        anomaly_result={"model_anomaly_score": 10.0},
        behavioral_result={"behavioral_score": 10.0},
    )
    assert result["risk_level"] == "low"
