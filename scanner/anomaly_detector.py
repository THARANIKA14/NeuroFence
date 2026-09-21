NeuroFence - Anomaly Detection Module


This module analyzes weight statistics produced by the weight analyzer
and identifies potentially anomalous layers/tensors.

Important:
An anomaly does NOT automatically mean that a model is malicious.
It indicates that the observed statistics differ significantly from
the selected reference/baseline.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import math


class AnomalyDetector:
    """
    Detect statistical anomalies in LLM weight/layer information.

    The detector expects weight statistics in a dictionary/list format
    containing values such as:

        mean
        std
        min
        max
        variance

    A reference mean/std can optionally be supplied for more meaningful
    comparison.
    """

    def __init__(
        self,
        z_threshold: float = 3.0,
        max_score: float = 100.0,
    ) -> None:
        """
        Initialize the anomaly detector.

        Args:
            z_threshold:
                Z-score above which a value is considered strongly
                anomalous.

            max_score:
                Maximum anomaly score.
        """

        if z_threshold <= 0:
            raise ValueError("z_threshold must be greater than 0.")

        if max_score <= 0:
            raise ValueError("max_score must be greater than 0.")

        self.z_threshold = float(z_threshold)
        self.max_score = float(max_score)

    # ---------------------------------------------------------
    # Utility functions
    # ---------------------------------------------------------

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        """
        Safely convert a value to float.
        """

        try:
            number = float(value)

            if not math.isfinite(number):
                return default

            return number

        except (TypeError, ValueError):
            return default

    @staticmethod
    def _clamp(value: float, minimum: float, maximum: float) -> float:
        """
        Keep a number inside a specified range.
        """

        return max(minimum, min(value, maximum))

    # ---------------------------------------------------------
    # Z-score
    # ---------------------------------------------------------

    def calculate_z_score(
        self,
        value: float,
        reference_mean: float,
        reference_std: float,
    ) -> float:
        """
        Calculate an absolute z-score.

        If reference_std is zero, the method returns 0 when the
        value matches the reference mean and a large value otherwise.
        """

        value = self._safe_float(value)
        reference_mean = self._safe_float(reference_mean)
        reference_std = self._safe_float(reference_std)

        if reference_std <= 0:
            if math.isclose(value, reference_mean):
                return 0.0

            return self.z_threshold + 1.0

        return abs((value - reference_mean) / reference_std)

    # ---------------------------------------------------------
    # Single statistic anomaly
    # ---------------------------------------------------------

    def detect_statistic_anomaly(
        self,
        value: float,
        reference_mean: float,
        reference_std: float,
    ) -> Dict[str, Any]:
        """
        Analyze one statistical value against a reference distribution.

        Returns:
            Dictionary containing z-score, score and status.
        """

        z_score = self.calculate_z_score(
            value=value,
            reference_mean=reference_mean,
            reference_std=reference_std,
        )

        # Convert z-score into a 0-100 anomaly score.
        score = (z_score / self.z_threshold) * 100.0
        score = self._clamp(score, 0.0, self.max_score)

        if z_score < 2.0:
            status = "normal"

        elif z_score < self.z_threshold:
            status = "moderate"

        else:
            status = "anomalous"

        return {
            "value": self._safe_float(value),
            "reference_mean": self._safe_float(reference_mean),
            "reference_std": self._safe_float(reference_std),
            "z_score": round(z_score, 4),
            "score": round(score, 2),
            "status": status,
        }

    # ---------------------------------------------------------
    # Layer analysis
    # ---------------------------------------------------------

    def analyze_layer(
        self,
        layer_name: str,
        statistics: Dict[str, Any],
        reference: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Analyze statistics for one model layer.

        Expected statistics may contain:

            mean
            std
            min
            max
            variance

        If a reference dictionary is supplied, it may contain:

            mean
            std
            min
            max
            variance

        Example:

            detector.analyze_layer(
                "encoder.layer.0",
                {
                    "mean": 0.02,
                    "std": 0.81,
                    "min": -2.4,
                    "max": 2.7,
                    "variance": 0.65
                }
            )
        """

        if not isinstance(statistics, dict):
            raise TypeError("statistics must be a dictionary.")

        reference = reference or {}

        values_to_check = [
            "mean",
            "std",
            "min",
            "max",
            "variance",
        ]

        component_results: Dict[str, Dict[str, Any]] = {}

        anomaly_scores: List[float] = []

        for metric in values_to_check:

            if metric not in statistics:
                continue

            value = self._safe_float(statistics.get(metric))

            # If no reference distribution is available, use a
            # simple magnitude-based screening.
            if metric not in reference:

                score = self._magnitude_score(value)

                if score < 30:
                    status = "normal"
                elif score < 70:
                    status = "moderate"
                else:
                    status = "anomalous"

                result = {
                    "value": round(value, 6),
                    "score": round(score, 2),
                    "status": status,
                }

            else:

                reference_data = reference.get(metric, {})

                if isinstance(reference_data, dict):

                    ref_mean = reference_data.get("mean", 0.0)
                    ref_std = reference_data.get("std", 1.0)

                else:

                    ref_mean = reference_data
                    ref_std = 1.0

                result = self.detect_statistic_anomaly(
                    value=value,
                    reference_mean=ref_mean,
                    reference_std=ref_std,
                )

            component_results[metric] = result
            anomaly_scores.append(float(result["score"]))

        # Calculate overall layer score.
        if anomaly_scores:
            overall_score = sum(anomaly_scores) / len(anomaly_scores)
        else:
            overall_score = 0.0

        overall_score = self._clamp(
            overall_score,
            0.0,
            self.max_score,
        )

        if overall_score < 30:
            classification = "normal"

        elif overall_score < 60:
            classification = "moderate"

        else:
            classification = "anomalous"

        return {
            "layer": layer_name,
            "anomaly_score": round(overall_score, 2),
            "classification": classification,
            "metrics": component_results,
        }

    # ---------------------------------------------------------
    # Magnitude-based screening
    # ---------------------------------------------------------

    def _magnitude_score(self, value: float) -> float:
        """
        Produce a simple screening score when no reference
        distribution is available.

        This is NOT proof of maliciousness. It is only a basic
        statistical screening mechanism.
        """

        magnitude = abs(self._safe_float(value))

        if magnitude <= 1:
            return 0.0

        if magnitude <= 2:
            return 25.0

        if magnitude <= 3:
            return 50.0

        if magnitude <= 5:
            return 75.0

        return 100.0

    # ---------------------------------------------------------
    # Analyze multiple layers
    # ---------------------------------------------------------

    def analyze_model(
        self,
        layer_statistics: Dict[str, Dict[str, Any]],
        references: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Analyze multiple model layers.

        Args:
            layer_statistics:
                Dictionary in the form:

                {
                    "layer_1": {
                        "mean": 0.01,
                        "std": 0.8,
                        "min": -2.0,
                        "max": 2.1,
                        "variance": 0.64
                    },
                    ...
                }

            references:
                Optional reference statistics for each layer.

        Returns:
            Complete model-level anomaly analysis.
        """

        if not isinstance(layer_statistics, dict):
            raise TypeError(
                "layer_statistics must be a dictionary."
            )

        references = references or {}

        layer_results: List[Dict[str, Any]] = []

        for layer_name, statistics in layer_statistics.items():

            reference = references.get(layer_name)

            result = self.analyze_layer(
                layer_name=layer_name,
                statistics=statistics,
                reference=reference,
            )

            layer_results.append(result)

        # Sort most suspicious layers first.
        layer_results.sort(
            key=lambda item: item["anomaly_score"],
            reverse=True,
        )

        if layer_results:

            model_score = sum(
                item["anomaly_score"]
                for item in layer_results
            ) / len(layer_results)

        else:

            model_score = 0.0

        model_score = self._clamp(
            model_score,
            0.0,
            self.max_score,
        )

        anomalous_layers = [
            item
            for item in layer_results
            if item["classification"] == "anomalous"
        ]

        moderate_layers = [
            item
            for item in layer_results
            if item["classification"] == "moderate"
        ]

        if model_score < 30:
            classification = "normal"

        elif model_score < 60:
            classification = "moderate"

        else:
            classification = "anomalous"

        return {
            "model_anomaly_score": round(model_score, 2),
            "classification": classification,
            "total_layers": len(layer_results),
            "anomalous_layers": len(anomalous_layers),
            "moderate_layers": len(moderate_layers),
            "layer_results": layer_results,
        }


# -------------------------------------------------------------
# Convenience function
# -------------------------------------------------------------

def detect_anomalies(
    layer_statistics: Dict[str, Dict[str, Any]],
    references: Optional[Dict[str, Dict[str, Any]]] = None,
    z_threshold: float = 3.0,
) -> Dict[str, Any]:
    """
    Convenience function for detecting model anomalies.

    Example:

        result = detect_anomalies(layer_statistics)

    """

    detector = AnomalyDetector(
        z_threshold=z_threshold
    )

    return detector.analyze_model(
        layer_statistics=layer_statistics,
        references=references,
    )


# -------------------------------------------------------------
# Standalone test
# -------------------------------------------------------------

if __name__ == "__main__":

    sample_statistics = {
        "layer_1": {
            "mean": 0.02,
            "std": 0.81,
            "min": -2.1,
            "max": 2.3,
            "variance": 0.66,
        },

        "layer_2": {
            "mean": 0.04,
            "std": 0.95,
            "min": -2.8,
            "max": 3.1,
            "variance": 0.90,
        },

        "layer_3": {
            "mean": 4.5,
            "std": 6.2,
            "min": -20.0,
            "max": 21.0,
            "variance": 38.4,
        },
    }

    result = detect_anomalies(sample_statistics)

    print("=" * 60)
    print("NEUROFENCE ANOMALY DETECTOR TEST")
    print("=" * 60)

    print(f"Model anomaly score: {result['model_anomaly_score']}")
    print(f"Classification: {result['classification']}")
    print(f"Total layers: {result['total_layers']}")
    print(f"Anomalous layers: {result['anomalous_layers']}")
    print(f"Moderate layers: {result['moderate_layers']}")

    print("\nLayer results:")

    for layer in result["layer_results"]:

        print(
            f"  {layer['layer']}: "
            f"{layer['anomaly_score']} "
            f"({layer['classification']})"
        )

    print("=" * 60)
```
