"""
NeuroFence - Weight Analyzer

This module extracts and analyzes numerical weight tensors from
a loaded PyTorch / Hugging Face model.

Responsibilities:
    - Inspect model parameters
    - Calculate tensor statistics
    - Calculate layer-level statistics
    - Calculate model-level summaries
    - Identify unusually large parameter values
    - Provide output compatible with anomaly_detector.py

IMPORTANT:
A statistical anomaly does NOT prove that a model is malicious.
It is only a signal that may require further investigation.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import math


class WeightAnalyzer:
    """
    Analyze numerical weights of a loaded PyTorch model.

    The model must expose a PyTorch-compatible `named_parameters()`
    method.
    """

    def __init__(
        self,
        max_elements_per_tensor: Optional[int] = None,
    ) -> None:
        """
        Initialize the weight analyzer.

        Args:
            max_elements_per_tensor:
                Optional limit on the number of tensor elements
                analyzed from each tensor.

                None means analyze the complete tensor.

                A limit can be useful when working with very large
                models.
        """

        if (
            max_elements_per_tensor is not None
            and max_elements_per_tensor <= 0
        ):
            raise ValueError(
                "max_elements_per_tensor must be greater than 0."
            )

        self.max_elements_per_tensor = (
            max_elements_per_tensor
        )

    # ---------------------------------------------------------
    # Utility functions
    # ---------------------------------------------------------

    @staticmethod
    def _safe_float(
        value: Any,
        default: float = 0.0,
    ) -> float:
        """
        Safely convert a value to float.
        """

        try:

            number = float(value)

            if not math.isfinite(number):
                return default

            return number

        except (
            TypeError,
            ValueError,
        ):
            return default

    @staticmethod
    def _clamp(
        value: float,
        minimum: float,
        maximum: float,
    ) -> float:
        """
        Clamp a value to a specified range.
        """

        return max(
            minimum,
            min(value, maximum),
        )

    # ---------------------------------------------------------
    # Tensor conversion
    # ---------------------------------------------------------

    def _prepare_tensor(
        self,
        tensor: Any,
    ):
        """
        Convert a PyTorch tensor into a CPU float tensor.

        The returned tensor is detached from the model so that
        analysis does not create gradients.
        """

        try:

            prepared = (
                tensor.detach()
                .float()
                .cpu()
            )

        except AttributeError as exc:

            raise TypeError(
                "Expected a PyTorch tensor."
            ) from exc

        if (
            self.max_elements_per_tensor is not None
            and prepared.numel()
            > self.max_elements_per_tensor
        ):

            prepared = prepared.reshape(-1)[
                : self.max_elements_per_tensor
            ]

        return prepared

    # ---------------------------------------------------------
    # Tensor statistics
    # ---------------------------------------------------------

    def calculate_tensor_statistics(
        self,
        tensor: Any,
    ) -> Dict[str, Any]:
        """
        Calculate statistical properties of one tensor.

        Returned values are compatible with anomaly_detector.py.
        """

        prepared = self._prepare_tensor(
            tensor
        )

        if prepared.numel() == 0:

            return {
                "element_count": 0,
                "mean": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
                "variance": 0.0,
                "absolute_mean": 0.0,
                "absolute_max": 0.0,
                "zero_count": 0,
                "zero_ratio": 0.0,
                "positive_count": 0,
                "negative_count": 0,
                "nan_count": 0,
                "inf_count": 0,
            }

        # Work on a flattened tensor for statistics.
        flattened = prepared.reshape(-1)

        # Check non-finite values before normal statistics.
        nan_count = int(
            torch_count_nan(flattened)
        )

        inf_count = int(
            torch_count_inf(flattened)
        )

        # Replace non-finite values for stable statistical
        # calculations.
        finite_mask = (
            flattened == flattened
        )

        try:
            finite_mask = (
                finite_mask
                & ~flattened.isinf()
            )
        except AttributeError:
            pass

        finite_values = flattened[
            finite_mask
        ]

        if finite_values.numel() == 0:

            return {
                "element_count": int(
                    flattened.numel()
                ),
                "mean": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
                "variance": 0.0,
                "absolute_mean": 0.0,
                "absolute_max": 0.0,
                "zero_count": int(
                    (flattened == 0).sum().item()
                ),
                "zero_ratio": 1.0,
                "positive_count": 0,
                "negative_count": 0,
                "nan_count": nan_count,
                "inf_count": inf_count,
            }

        mean = finite_values.mean()

        # unbiased=False avoids NaN for one-element tensors.
        variance = finite_values.var(
            unbiased=False
        )

        std = finite_values.std(
            unbiased=False
        )

        minimum = finite_values.min()

        maximum = finite_values.max()

        absolute_values = finite_values.abs()

        absolute_mean = (
            absolute_values.mean()
        )

        absolute_max = (
            absolute_values.max()
        )

        zero_count = int(
            (finite_values == 0)
            .sum()
            .item()
        )

        positive_count = int(
            (finite_values > 0)
            .sum()
            .item()
        )

        negative_count = int(
            (finite_values < 0)
            .sum()
            .item()
        )

        element_count = int(
            flattened.numel()
        )

        zero_ratio = (
            zero_count / element_count
            if element_count > 0
            else 0.0
        )

        return {
            "element_count": element_count,
            "mean": round(
                self._safe_float(
                    mean.item()
                ),
                8,
            ),
            "std": round(
                self._safe_float(
                    std.item()
                ),
                8,
            ),
            "min": round(
                self._safe_float(
                    minimum.item()
                ),
                8,
            ),
            "max": round(
                self._safe_float(
                    maximum.item()
                ),
                8,
            ),
            "variance": round(
                self._safe_float(
                    variance.item()
                ),
                8,
            ),
            "absolute_mean": round(
                self._safe_float(
                    absolute_mean.item()
                ),
                8,
            ),
            "absolute_max": round(
                self._safe_float(
                    absolute_max.item()
                ),
                8,
            ),
            "zero_count": zero_count,
            "zero_ratio": round(
                zero_ratio,
                8,
            ),
            "positive_count": positive_count,
            "negative_count": negative_count,
            "nan_count": nan_count,
            "inf_count": inf_count,
        }

    # ---------------------------------------------------------
    # Analyze one named parameter
    # ---------------------------------------------------------

    def analyze_parameter(
        self,
        parameter_name: str,
        parameter: Any,
    ) -> Dict[str, Any]:
        """
        Analyze one named model parameter.
        """

        if not isinstance(
            parameter_name,
            str,
        ):
            parameter_name = str(
                parameter_name
            )

        tensor = self._prepare_tensor(
            parameter
        )

        statistics = (
            self.calculate_tensor_statistics(
                tensor
            )
        )

        return {
            "name": parameter_name,
            "shape": list(
                parameter.shape
            ),
            "dtype": str(
                parameter.dtype
            ),
            "statistics": statistics,
        }

    # ---------------------------------------------------------
    # Analyze complete model
    # ---------------------------------------------------------

    def analyze_model(
        self,
        model: Any,
        include_bias: bool = True,
    ) -> Dict[str, Any]:
        """
        Analyze all named parameters in a model.

        Args:
            model:
                Loaded PyTorch/Hugging Face model.

            include_bias:
                If False, parameters containing ".bias" are skipped.
        """

        if model is None:
            raise ValueError(
                "model cannot be None."
            )

        if not hasattr(
            model,
            "named_parameters",
        ):
            raise TypeError(
                "Model must provide named_parameters()."
            )

        layer_statistics: Dict[
            str,
            Dict[str, Any],
        ] = {}

        parameter_details: List[
            Dict[str, Any]
        ] = []

        total_parameters = 0
        analyzed_parameters = 0
        skipped_parameters = 0

        for name, parameter in (
            model.named_parameters()
        ):

            if (
                not include_bias
                and ".bias" in name.lower()
            ):
                skipped_parameters += 1
                continue

            try:

                element_count = int(
                    parameter.numel()
                )

            except Exception:

                skipped_parameters += 1
                continue

            total_parameters += (
                element_count
            )

            try:

                result = self.analyze_parameter(
                    parameter_name=name,
                    parameter=parameter,
                )

            except Exception:

                skipped_parameters += 1
                continue

            statistics = result[
                "statistics"
            ]

            layer_statistics[name] = (
                statistics
            )

            parameter_details.append(
                result
            )

            analyzed_parameters += 1

        model_summary = (
            self.calculate_model_summary(
                layer_statistics
            )
        )

        return {
            "total_parameters": total_parameters,
            "analyzed_tensors": analyzed_parameters,
            "skipped_tensors": skipped_parameters,
            "layer_statistics": layer_statistics,
            "parameter_details": parameter_details,
            "model_summary": model_summary,
        }

    # ---------------------------------------------------------
    # Model-level summary
    # ---------------------------------------------------------

    def calculate_model_summary(
        self,
        layer_statistics: Dict[
            str,
            Dict[str, Any],
        ],
    ) -> Dict[str, Any]:
        """
        Calculate an overall summary from layer statistics.
        """

        if not layer_statistics:

            return {
                "layer_count": 0,
                "mean_of_means": 0.0,
                "mean_std": 0.0,
                "global_min": 0.0,
                "global_max": 0.0,
                "mean_variance": 0.0,
                "mean_absolute": 0.0,
                "max_absolute": 0.0,
                "non_finite_layers": 0,
            }

        means = []
        stds = []
        variances = []
        absolute_means = []
        maximum_absolute_values = []

        global_min = None
        global_max = None

        non_finite_layers = 0

        for statistics in (
            layer_statistics.values()
        ):

            mean = self._safe_float(
                statistics.get(
                    "mean",
                    0.0,
                )
            )

            std = self._safe_float(
                statistics.get(
                    "std",
                    0.0,
                )
            )

            variance = self._safe_float(
                statistics.get(
                    "variance",
                    0.0,
                )
            )

            minimum = self._safe_float(
                statistics.get(
                    "min",
                    0.0,
                )
            )

            maximum = self._safe_float(
                statistics.get(
                    "max",
                    0.0,
                )
            )

            absolute_mean = self._safe_float(
                statistics.get(
                    "absolute_mean",
                    0.0,
                )
            )

            absolute_max = self._safe_float(
                statistics.get(
                    "absolute_max",
                    0.0,
                )
            )

            means.append(mean)
            stds.append(std)
            variances.append(variance)
            absolute_means.append(
                absolute_mean
            )
            maximum_absolute_values.append(
                absolute_max
            )

            if global_min is None:
                global_min = minimum
            else:
                global_min = min(
                    global_min,
                    minimum,
                )

            if global_max is None:
                global_max = maximum
            else:
                global_max = max(
                    global_max,
                    maximum,
                )

            if (
                statistics.get(
                    "nan_count",
                    0,
                )
                > 0
                or statistics.get(
                    "inf_count",
                    0,
                )
                > 0
            ):
                non_finite_layers += 1

        count = len(
            layer_statistics
        )

        return {
            "layer_count": count,
            "mean_of_means": round(
                sum(means) / count,
                8,
            ),
            "mean_std": round(
                sum(stds) / count,
                8,
            ),
            "global_min": round(
                global_min
                if global_min is not None
                else 0.0,
                8,
            ),
            "global_max": round(
                global_max
                if global_max is not None
                else 0.0,
                8,
            ),
            "mean_variance": round(
                sum(variances) / count,
                8,
            ),
            "mean_absolute": round(
                sum(absolute_means) / count,
                8,
            ),
            "max_absolute": round(
                max(maximum_absolute_values),
                8,
            ),
            "non_finite_layers": (
                non_finite_layers
            ),
        }

    # ---------------------------------------------------------
    # Find extreme tensors
    # ---------------------------------------------------------

    def find_extreme_layers(
        self,
        layer_statistics: Dict[
            str,
            Dict[str, Any],
        ],
        absolute_threshold: float = 5.0,
    ) -> List[Dict[str, Any]]:
        """
        Find layers containing unusually large absolute values.

        This is a statistical screening signal only.
        """

        if absolute_threshold <= 0:
            raise ValueError(
                "absolute_threshold must be greater than 0."
            )

        extreme_layers = []

        for name, statistics in (
            layer_statistics.items()
        ):

            absolute_max = self._safe_float(
                statistics.get(
                    "absolute_max",
                    0.0,
                )
            )

            if absolute_max >= absolute_threshold:

                extreme_layers.append(
                    {
                        "layer": name,
                        "absolute_max": round(
                            absolute_max,
                            8,
                        ),
                        "threshold": absolute_threshold,
                    }
                )

        extreme_layers.sort(
            key=lambda item: item[
                "absolute_max"
            ],
            reverse=True,
        )

        return extreme_layers

    # ---------------------------------------------------------
    # Convert output for anomaly detector
    # ---------------------------------------------------------

    def get_anomaly_input(
        self,
        analysis_result: Dict[str, Any],
    ) -> Dict[
        str,
        Dict[str, Any],
    ]:
        """
        Extract only the statistics required by
        anomaly_detector.py.

        Output format:

            {
                "layer_name": {
                    "mean": ...,
                    "std": ...,
                    "min": ...,
                    "max": ...,
                    "variance": ...
                }
            }
        """

        layer_statistics = (
            analysis_result.get(
                "layer_statistics",
                {},
            )
        )

        anomaly_input = {}

        for layer_name, statistics in (
            layer_statistics.items()
        ):

            anomaly_input[
                layer_name
            ] = {
                "mean": self._safe_float(
                    statistics.get(
                        "mean",
                        0.0,
                    )
                ),
                "std": self._safe_float(
                    statistics.get(
                        "std",
                        0.0,
                    )
                ),
                "min": self._safe_float(
                    statistics.get(
                        "min",
                        0.0,
                    )
                ),
                "max": self._safe_float(
                    statistics.get(
                        "max",
                        0.0,
                    )
                ),
                "variance": self._safe_float(
                    statistics.get(
                        "variance",
                        0.0,
                    )
                ),
            }

        return anomaly_input

    # ---------------------------------------------------------
    # Generate report summary
    # ---------------------------------------------------------

    def generate_summary(
        self,
        analysis_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Generate a compact result for the UI/report.
        """

        model_summary = (
            analysis_result.get(
                "model_summary",
                {},
            )
        )

        extreme_layers = (
            self.find_extreme_layers(
                analysis_result.get(
                    "layer_statistics",
                    {},
                )
            )
        )

        return {
            "total_parameters": analysis_result.get(
                "total_parameters",
                0,
            ),
            "analyzed_tensors": analysis_result.get(
                "analyzed_tensors",
                0,
            ),
            "skipped_tensors": analysis_result.get(
                "skipped_tensors",
                0,
            ),
            "layer_count": model_summary.get(
                "layer_count",
                0,
            ),
            "mean_of_means": model_summary.get(
                "mean_of_means",
                0.0,
            ),
            "mean_std": model_summary.get(
                "mean_std",
                0.0,
            ),
            "global_min": model_summary.get(
                "global_min",
                0.0,
            ),
            "global_max": model_summary.get(
                "global_max",
                0.0,
            ),
            "max_absolute": model_summary.get(
                "max_absolute",
                0.0,
            ),
            "non_finite_layers": model_summary.get(
                "non_finite_layers",
                0,
            ),
            "extreme_layer_count": len(
                extreme_layers
            ),
        }


# -------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------

def torch_count_nan(
    tensor: Any,
) -> int:
    """
    Count NaN values in a PyTorch tensor.
    """

    try:

        return int(
            tensor.isnan()
            .sum()
            .item()
        )

    except (
        AttributeError,
        RuntimeError,
    ):

        return 0


def torch_count_inf(
    tensor: Any,
) -> int:
    """
    Count infinite values in a PyTorch tensor.
    """

    try:

        return int(
            tensor.isinf()
            .sum()
            .item()
        )

    except (
        AttributeError,
        RuntimeError,
    ):

        return 0


def analyze_model_weights(
    model: Any,
    max_elements_per_tensor: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Convenience function for analyzing model weights.
    """

    analyzer = WeightAnalyzer(
        max_elements_per_tensor=(
            max_elements_per_tensor
        )
    )

    return analyzer.analyze_model(
        model
    )


if __name__ == "__main__":

    print("=" * 70)
    print("NEUROFENCE WEIGHT ANALYZER TEST")
    print("=" * 70)

    # ---------------------------------------------------------
    # Import PyTorch
    # ---------------------------------------------------------

    try:

        import torch

        print("\nPyTorch loaded successfully.")
        print(
            f"PyTorch version: "
            f"{torch.__version__}"
        )

    except ImportError:

        print(
            "\nPyTorch is not installed."
        )

        print(
            "Install PyTorch before running "
            "the weight analyzer."
        )

        raise SystemExit(1)

    # ---------------------------------------------------------
    # Create a small test model
    # ---------------------------------------------------------

    class TestModel(
        torch.nn.Module
    ):

        def __init__(self):

            super().__init__()

            self.layer1 = (
                torch.nn.Linear(
                    10,
                    20,
                )
            )

            self.layer2 = (
                torch.nn.Linear(
                    20,
                    5,
                )
            )

        def forward(
            self,
            x,
        ):

            x = self.layer1(x)

            x = torch.relu(x)

            return self.layer2(x)

    # ---------------------------------------------------------
    # Create model
    # ---------------------------------------------------------

    model = TestModel()

    print(
        "\nTest model created successfully."
    )

    # ---------------------------------------------------------
    # Analyze weights
    # ---------------------------------------------------------

    analyzer = WeightAnalyzer()

    result = analyzer.analyze_model(
        model
    )

    # ---------------------------------------------------------
    # Print basic information
    # ---------------------------------------------------------

    print(
        "\nTotal parameters: "
        f"{result['total_parameters']}"
    )

    print(
        "Analyzed tensors: "
        f"{result['analyzed_tensors']}"
    )

    print(
        "Skipped tensors: "
        f"{result['skipped_tensors']}"
    )

    # ---------------------------------------------------------
    # Print layer statistics
    # ---------------------------------------------------------

    print(
        "\nLayer statistics:"
    )

    for (
        layer_name,
        statistics,
    ) in result[
        "layer_statistics"
    ].items():

        print(
            f"\n{layer_name}"
        )

        print(
            f"  Mean: "
            f"{statistics['mean']}"
        )

        print(
            f"  Std: "
            f"{statistics['std']}"
        )

        print(
            f"  Min: "
            f"{statistics['min']}"
        )

        print(
            f"  Max: "
            f"{statistics['max']}"
        )

        print(
            f"  Variance: "
            f"{statistics['variance']}"
        )

        print(
            f"  Absolute max: "
            f"{statistics['absolute_max']}"
        )

        print(
            f"  Zero ratio: "
            f"{statistics['zero_ratio']}"
        )

    # ---------------------------------------------------------
    # Print model summary
    # ---------------------------------------------------------

    print(
        "\nModel summary:"
    )

    print(
        result["model_summary"]
    )

    # ---------------------------------------------------------
    # Find extreme layers
    # ---------------------------------------------------------

    extreme_layers = (
        analyzer.find_extreme_layers(
            result[
                "layer_statistics"
            ]
        )
    )

    print(
        "\nExtreme layers:"
    )

    if extreme_layers:

        for item in extreme_layers:

            print(
                f"  {item['layer']}: "
                f"{item['absolute_max']}"
            )

    else:

        print(
            "  No layers exceeded "
            "the default threshold."
        )

    # ---------------------------------------------------------
    # Generate anomaly detector input
    # ---------------------------------------------------------

    anomaly_input = (
        analyzer.get_anomaly_input(
            result
        )
    )

    print(
        "\nAnomaly detector input:"
    )

    for (
        layer_name,
        statistics,
    ) in anomaly_input.items():

        print(
            f"  {layer_name}: "
            f"{statistics}"
        )

    # ---------------------------------------------------------
    # Generate summary
    # ---------------------------------------------------------

    summary = (
        analyzer.generate_summary(
            result
        )
    )

    print(
        "\nAnalyzer summary:"
    )

    print(summary)

    print(
        "\n" + "=" * 70
    )

    print(
        "WEIGHT ANALYZER TEST COMPLETED"
    )

    print(
        "=" * 70
    )
```
