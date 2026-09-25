NeuroFence - Risk Engine

This module combines the results of multiple NeuroFence scanner
components into one overall risk assessment.

Inputs can include:
    - Model validation
    - Weight anomaly detection
    - Behavioral/backdoor detection
    - Integrity verification

The engine produces:
    - Overall risk score
    - Risk level
    - Component scores
    - Contributing factors
    - Recommendations

IMPORTANT:
The result is a risk assessment, NOT proof that a model is
malicious or contains a backdoor.

A high score means that multiple observed signals require
further investigation.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


class RiskEngine:
    """
    Combine NeuroFence scanner results into an overall
    risk assessment.
    """

    DEFAULT_WEIGHTS = {
        "anomaly": 0.35,
        "behavioral": 0.35,
        "integrity": 0.20,
        "validation": 0.10,
    }

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Initialize the risk engine.

        Args:
            weights:
                Relative importance of each scanner signal.

                Default:
                    anomaly     = 35%
                    behavioral  = 35%
                    integrity   = 20%
                    validation  = 10%
        """

        if weights is None:
            weights = self.DEFAULT_WEIGHTS.copy()

        self.weights = self._validate_weights(
            weights
        )

    # ---------------------------------------------------------
    # Weight validation
    # ---------------------------------------------------------

    @staticmethod
    def _validate_weights(
        weights: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Validate and normalize risk weights.
        """

        required_keys = {
            "anomaly",
            "behavioral",
            "integrity",
            "validation",
        }

        if not isinstance(weights, dict):
            raise TypeError(
                "weights must be a dictionary."
            )

        missing = (
            required_keys
            - set(weights.keys())
        )

        if missing:
            raise ValueError(
                "Missing risk weights: "
                + ", ".join(sorted(missing))
            )

        cleaned = {}

        for key in required_keys:

            value = float(
                weights[key]
            )

            if value < 0:
                raise ValueError(
                    f"Weight '{key}' cannot be negative."
                )

            cleaned[key] = value

        total = sum(
            cleaned.values()
        )

        if total <= 0:
            raise ValueError(
                "At least one risk weight must be greater than zero."
            )

        # Normalize weights so their total becomes 1.0.
        normalized = {
            key: value / total
            for key, value in cleaned.items()
        }

        return normalized

    # ---------------------------------------------------------
    # Utility functions
    # ---------------------------------------------------------

    @staticmethod
    def _clamp(
        value: float,
        minimum: float = 0.0,
        maximum: float = 100.0,
    ) -> float:
        """
        Keep a numeric value inside a defined range.
        """

        return max(
            minimum,
            min(value, maximum),
        )

    @staticmethod
    def _safe_float(
        value: Any,
        default: float = 0.0,
    ) -> float:
        """
        Safely convert a value to float.
        """

        try:
            return float(value)

        except (
            TypeError,
            ValueError,
        ):
            return default

    # ---------------------------------------------------------
    # Anomaly score
    # ---------------------------------------------------------

    def extract_anomaly_score(
        self,
        anomaly_result: Optional[Dict[str, Any]],
    ) -> float:
        """
        Extract the model anomaly score from anomaly_detector.py.
        """

        if not anomaly_result:
            return 0.0

        score = anomaly_result.get(
            "model_anomaly_score",
            anomaly_result.get(
                "anomaly_score",
                0.0,
            ),
        )

        return self._clamp(
            self._safe_float(score)
        )

    # ---------------------------------------------------------
    # Behavioral score
    # ---------------------------------------------------------

    def extract_behavioral_score(
        self,
        behavioral_result: Optional[Dict[str, Any]],
    ) -> float:
        """
        Extract behavioral score from backdoor_detector.py.
        """

        if not behavioral_result:
            return 0.0

        score = behavioral_result.get(
            "behavioral_score",
            0.0,
        )

        return self._clamp(
            self._safe_float(score)
        )

    # ---------------------------------------------------------
    # Integrity score
    # ---------------------------------------------------------

    def calculate_integrity_score(
        self,
        integrity_result: Optional[Dict[str, Any]],
    ) -> float:
        """
        Convert integrity status into a risk score.

        Lower score:
            Verified integrity

        Higher score:
            Integrity mismatch

        Note:
            Missing reference hashes are treated as informational,
            not as proof of tampering.
        """

        if not integrity_result:
            return 0.0

        status = str(
            integrity_result.get(
                "integrity_status",
                "unknown",
            )
        ).lower()

        verified = integrity_result.get(
            "integrity_verified"
        )

        if status == "verified":
            return 0.0

        if verified is True:
            return 0.0

        if status == "mismatch":
            return 100.0

        if status in {
            "hash_error",
            "invalid_path",
        }:
            return 70.0

        if status == "missing":
            return 60.0

        if status == "no_reference":
            return 0.0

        if status == "hash_generated":
            return 0.0

        return 0.0

    # ---------------------------------------------------------
    # Validation score
    # ---------------------------------------------------------

    def calculate_validation_score(
        self,
        validation_result: Optional[Dict[str, Any]],
    ) -> float:
        """
        Convert model validation findings into a risk signal.

        Validation problems are treated as a smaller component
        because an invalid model package does not necessarily
        indicate malicious activity.
        """

        if not validation_result:
            return 0.0

        valid = validation_result.get(
            "valid"
        )

        issues = validation_result.get(
            "issues",
            [],
        )

        warnings = validation_result.get(
            "warnings",
            [],
        )

        if valid is True:
            base_score = 0.0

        else:
            base_score = 40.0

        issue_penalty = min(
            len(issues) * 10.0,
            40.0,
        )

        warning_penalty = min(
            len(warnings) * 5.0,
            20.0,
        )

        score = (
            base_score
            + issue_penalty
            + warning_penalty
        )

        return self._clamp(score)

    # ---------------------------------------------------------
    # Risk level
    # ---------------------------------------------------------

    @staticmethod
    def classify_risk(
        score: float,
    ) -> str:
        """
        Convert the overall score into a risk level.

        Score ranges:

            0 - 24.99    Low
            25 - 49.99   Moderate
            50 - 74.99   High
            75 - 100     Critical
        """

        if score < 25:
            return "low"

        if score < 50:
            return "moderate"

        if score < 75:
            return "high"

        return "critical"

    # ---------------------------------------------------------
    # Risk description
    # ---------------------------------------------------------

    @staticmethod
    def get_risk_description(
        risk_level: str,
    ) -> str:
        """
        Return a human-readable description.
        """

        descriptions = {
            "low": (
                "No strong risk signals were observed "
                "by the available checks."
            ),
            "moderate": (
                "Some unusual signals were observed. "
                "Further investigation is recommended."
            ),
            "high": (
                "Multiple significant risk signals were "
                "observed. Detailed investigation is recommended."
            ),
            "critical": (
                "Strong combined risk signals were observed. "
                "The model should be investigated before being "
                "trusted or deployed."
            ),
        }

        return descriptions.get(
            risk_level,
            "Risk level is unknown.",
        )

    # ---------------------------------------------------------
    # Contributing factors
    # ---------------------------------------------------------

    def identify_contributing_factors(
        self,
        component_scores: Dict[str, float],
        anomaly_result: Optional[Dict[str, Any]] = None,
        behavioral_result: Optional[Dict[str, Any]] = None,
        integrity_result: Optional[Dict[str, Any]] = None,
        validation_result: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """
        Identify the main factors contributing to the score.
        """

        factors: List[str] = []

        anomaly_score = component_scores.get(
            "anomaly",
            0.0,
        )

        behavioral_score = component_scores.get(
            "behavioral",
            0.0,
        )

        integrity_score = component_scores.get(
            "integrity",
            0.0,
        )

        validation_score = component_scores.get(
            "validation",
            0.0,
        )

        if anomaly_score >= 60:

            anomalous_layers = 0

            if anomaly_result:
                anomalous_layers = int(
                    anomaly_result.get(
                        "anomalous_layers",
                        0,
                    )
                )

            factors.append(
                "Significant weight-statistical anomalies "
                f"were detected"
                + (
                    f" across {anomalous_layers} layer(s)."
                    if anomalous_layers
                    else "."
                )
            )

        elif anomaly_score >= 30:

            factors.append(
                "Moderate weight-statistical deviations "
                "were detected."
            )

        if behavioral_score >= 80:

            high_tests = 0

            if behavioral_result:
                high_tests = int(
                    behavioral_result.get(
                        "high_deviation_tests",
                        0,
                    )
                )

            factors.append(
                "Strong behavioral deviations were observed"
                + (
                    f" in {high_tests} test(s)."
                    if high_tests
                    else "."
                )
            )

        elif behavioral_score >= 60:

            factors.append(
                "Potentially suspicious behavioral "
                "differences were observed."
            )

        if integrity_score >= 100:

            factors.append(
                "The current model hash does not match "
                "the trusted reference hash."
            )

        elif integrity_score >= 60:

            factors.append(
                "The model integrity could not be "
                "fully verified."
            )

        if validation_score >= 60:

            factors.append(
                "The model package contains significant "
                "validation issues."
            )

        elif validation_score >= 30:

            factors.append(
                "The model package has structural "
                "validation issues."
            )

        if not factors:

            factors.append(
                "No major contributing risk factor "
                "was identified by the supplied checks."
            )

        return factors

    # ---------------------------------------------------------
    # Recommendations
    # ---------------------------------------------------------

    def generate_recommendations(
        self,
        risk_level: str,
        component_scores: Dict[str, float],
    ) -> List[str]:
        """
        Generate neutral investigation recommendations.
        """

        recommendations: List[str] = []

        anomaly_score = component_scores.get(
            "anomaly",
            0.0,
        )

        behavioral_score = component_scores.get(
            "behavioral",
            0.0,
        )

        integrity_score = component_scores.get(
            "integrity",
            0.0,
        )

        validation_score = component_scores.get(
            "validation",
            0.0,
        )

        if anomaly_score >= 30:

            recommendations.append(
                "Review the layers/tensors with the highest "
                "anomaly scores."
            )

        if behavioral_score >= 30:

            recommendations.append(
                "Repeat behavioral tests with multiple "
                "baseline and candidate-trigger prompts."
            )

        if integrity_score >= 60:

            recommendations.append(
                "Compare the model against a trusted or "
                "known-good model copy."
            )

        if validation_score >= 30:

            recommendations.append(
                "Review missing, malformed, or unexpected "
                "model files before loading the model."
            )

        if risk_level in {
            "high",
            "critical",
        }:

            recommendations.append(
                "Perform manual security review before "
                "using the model in a production environment."
            )

        if not recommendations:

            recommendations.append(
                "Continue with normal validation and "
                "monitoring procedures."
            )

        return recommendations

    # ---------------------------------------------------------
    # Main risk calculation
    # ---------------------------------------------------------

    def calculate_risk(
        self,
        anomaly_result: Optional[Dict[str, Any]] = None,
        behavioral_result: Optional[Dict[str, Any]] = None,
        integrity_result: Optional[Dict[str, Any]] = None,
        validation_result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Calculate the overall NeuroFence risk assessment.

        Returns:
            Dictionary containing component scores,
            weighted scores, overall score, risk level,
            factors, and recommendations.
        """

        anomaly_score = (
            self.extract_anomaly_score(
                anomaly_result
            )
        )

        behavioral_score = (
            self.extract_behavioral_score(
                behavioral_result
            )
        )

        integrity_score = (
            self.calculate_integrity_score(
                integrity_result
            )
        )

        validation_score = (
            self.calculate_validation_score(
                validation_result
            )
        )

        component_scores = {
            "anomaly": round(
                anomaly_score,
                2,
            ),
            "behavioral": round(
                behavioral_score,
                2,
            ),
            "integrity": round(
                integrity_score,
                2,
            ),
            "validation": round(
                validation_score,
                2,
            ),
        }

        weighted_scores = {
            key: round(
                component_scores[key]
                * self.weights[key],
                2,
            )
            for key in self.weights
        }

        overall_score = sum(
            weighted_scores.values()
        )

        overall_score = self._clamp(
            overall_score
        )

        overall_score = round(
            overall_score,
            2,
        )

        risk_level = self.classify_risk(
            overall_score
        )

        factors = (
            self.identify_contributing_factors(
                component_scores=component_scores,
                anomaly_result=anomaly_result,
                behavioral_result=behavioral_result,
                integrity_result=integrity_result,
                validation_result=validation_result,
            )
        )

        recommendations = (
            self.generate_recommendations(
                risk_level=risk_level,
                component_scores=component_scores,
            )
        )

        return {
            "overall_risk_score": overall_score,
            "risk_level": risk_level,
            "risk_description": (
                self.get_risk_description(
                    risk_level
                )
            ),
            "component_scores": component_scores,
            "component_weights": {
                key: round(
                    value,
                    4,
                )
                for key, value in self.weights.items()
            },
            "weighted_scores": weighted_scores,
            "contributing_factors": factors,
            "recommendations": recommendations,
        }

    # ---------------------------------------------------------
    # Report summary
    # ---------------------------------------------------------

    def generate_summary(
        self,
        risk_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Generate a compact result for the NeuroFence UI.
        """

        return {
            "overall_risk_score": risk_result.get(
                "overall_risk_score",
                0.0,
            ),
            "risk_level": risk_result.get(
                "risk_level",
                "unknown",
            ),
            "risk_description": risk_result.get(
                "risk_description",
                "",
            ),
            "contributing_factors": risk_result.get(
                "contributing_factors",
                [],
            ),
            "recommendations": risk_result.get(
                "recommendations",
                [],
            ),
        }


def calculate_risk(
    anomaly_result: Optional[Dict[str, Any]] = None,
    behavioral_result: Optional[Dict[str, Any]] = None,
    integrity_result: Optional[Dict[str, Any]] = None,
    validation_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Convenience function for calculating NeuroFence risk.
    """

    engine = RiskEngine()

    return engine.calculate_risk(
        anomaly_result=anomaly_result,
        behavioral_result=behavioral_result,
        integrity_result=integrity_result,
        validation_result=validation_result,
    )


if __name__ == "__main__":

    print("=" * 70)
    print("NEUROFENCE RISK ENGINE TEST")
    print("=" * 70)

    # ---------------------------------------------------------
    # Example anomaly detector result
    # ---------------------------------------------------------

    anomaly_result = {
        "model_anomaly_score": 68.5,
        "classification": "anomalous",
        "total_layers": 10,
        "anomalous_layers": 3,
        "moderate_layers": 2,
        "layer_results": [],
    }

    # ---------------------------------------------------------
    # Example behavioral detector result
    # ---------------------------------------------------------

    behavioral_result = {
        "behavioral_score": 72.0,
        "classification": "potentially_suspicious",
        "total_tests": 10,
        "suspicious_tests": 5,
        "high_deviation_tests": 2,
        "results": [],
    }

    # ---------------------------------------------------------
    # Example integrity result
    # ---------------------------------------------------------

    integrity_result = {
        "integrity_status": "verified",
        "integrity_verified": True,
        "hash_algorithm": "sha256",
    }

    # ---------------------------------------------------------
    # Example validation result
    # ---------------------------------------------------------

    validation_result = {
        "valid": True,
        "status": "valid",
        "issues": [],
        "warnings": [],
    }

    # ---------------------------------------------------------
    # Create engine
    # ---------------------------------------------------------

    engine = RiskEngine()

    # ---------------------------------------------------------
    # Calculate risk
    # ---------------------------------------------------------

    result = engine.calculate_risk(
        anomaly_result=anomaly_result,
        behavioral_result=behavioral_result,
        integrity_result=integrity_result,
        validation_result=validation_result,
    )

    # ---------------------------------------------------------
    # Display result
    # ---------------------------------------------------------

    print("\nComponent scores:")

    for name, score in result[
        "component_scores"
    ].items():

        print(
            f"  {name}: {score}"
        )

    print("\nComponent weights:")

    for name, weight in result[
        "component_weights"
    ].items():

        print(
            f"  {name}: {weight}"
        )

    print("\nWeighted scores:")

    for name, score in result[
        "weighted_scores"
    ].items():

        print(
            f"  {name}: {score}"
        )

    print(
        "\nOverall risk score: "
        f"{result['overall_risk_score']}"
    )

    print(
        "Risk level: "
        f"{result['risk_level']}"
    )

    print(
        "\nRisk description:"
    )

    print(
        result["risk_description"]
    )

    print(
        "\nContributing factors:"
    )

    for factor in result[
        "contributing_factors"
    ]:

        print(
            f"  - {factor}"
        )

    print(
        "\nRecommendations:"
    )

    for recommendation in result[
        "recommendations"
    ]:

        print(
            f"  - {recommendation}"
        )

    # ---------------------------------------------------------
    # Generate summary
    # ---------------------------------------------------------

    summary = engine.generate_summary(
        result
    )

    print(
        "\nUI summary:"
    )

    print(summary)

    print("\n" + "=" * 70)
    print("RISK ENGINE TEST COMPLETED")
    print("=" * 70)
```
