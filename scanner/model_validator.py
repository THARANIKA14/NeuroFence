NeuroFence - Model Validator

This module validates an LLM model before it is loaded by
the NeuroFence scanner.

Validation includes:
    - Path existence
    - File/directory validation
    - Supported model file detection
    - Required configuration files
    - Empty-file detection
    - File-size checks
    - Basic Hugging Face model structure
    - Model configuration validation

IMPORTANT:
Validation failure does NOT automatically mean that a model
is malicious. It means the model requires further investigation
or cannot safely proceed to the next scanner stage.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
import json


class ModelValidator:
    """
    Validate the structure and basic properties of an LLM model.

    The validator does not load model weights. It performs
    lightweight checks before model_loader.py is used.
    """

    SUPPORTED_MODEL_EXTENSIONS = {
        ".safetensors",
        ".bin",
        ".pt",
        ".pth",
    }

    REQUIRED_CONFIG_FILES = {
        "config.json",
    }

    OPTIONAL_MODEL_FILES = {
        "model.safetensors",
        "pytorch_model.bin",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
    }

    def __init__(
        self,
        max_file_size_gb: float = 200.0,
    ) -> None:
        """
        Initialize the validator.

        Args:
            max_file_size_gb:
                Maximum accepted individual file size.

                This is a validation limit, not a security
                determination.
        """

        if max_file_size_gb <= 0:
            raise ValueError(
                "max_file_size_gb must be greater than 0."
            )

        self.max_file_size_gb = float(
            max_file_size_gb
        )

        self.max_file_size_bytes = int(
            self.max_file_size_gb
            * 1024
            * 1024
            * 1024
        )

    # ---------------------------------------------------------
    # Basic path validation
    # ---------------------------------------------------------

    def validate_path(
        self,
        model_path: str | Path,
    ) -> Dict[str, Any]:
        """
        Validate that the supplied path exists and is usable.
        """

        path = Path(model_path).expanduser()

        if not path.exists():
            return {
                "valid": False,
                "exists": False,
                "type": "missing",
                "path": str(path),
                "message": "Model path does not exist.",
            }

        if path.is_file():

            return {
                "valid": True,
                "exists": True,
                "type": "file",
                "path": str(path),
                "message": "Model file exists.",
            }

        if path.is_dir():

            return {
                "valid": True,
                "exists": True,
                "type": "directory",
                "path": str(path),
                "message": "Model directory exists.",
            }

        return {
            "valid": False,
            "exists": True,
            "type": "unknown",
            "path": str(path),
            "message": "Unsupported model path.",
        }

    # ---------------------------------------------------------
    # File extension validation
    # ---------------------------------------------------------

    def validate_model_file(
        self,
        model_path: str | Path,
    ) -> Dict[str, Any]:
        """
        Validate a single model file.
        """

        path = Path(model_path).expanduser()

        if not path.exists():

            return {
                "valid": False,
                "path": str(path),
                "message": "Model file does not exist.",
            }

        if not path.is_file():

            return {
                "valid": False,
                "path": str(path),
                "message": "Provided path is not a file.",
            }

        extension = path.suffix.lower()

        if extension not in self.SUPPORTED_MODEL_EXTENSIONS:

            return {
                "valid": False,
                "path": str(path),
                "extension": extension,
                "size_bytes": path.stat().st_size,
                "message": (
                    f"Unsupported model file extension: "
                    f"{extension}"
                ),
            }

        file_size = path.stat().st_size

        if file_size == 0:

            return {
                "valid": False,
                "path": str(path),
                "extension": extension,
                "size_bytes": 0,
                "message": "Model file is empty.",
            }

        if file_size > self.max_file_size_bytes:

            return {
                "valid": False,
                "path": str(path),
                "extension": extension,
                "size_bytes": file_size,
                "message": (
                    "Model file exceeds the configured "
                    "maximum file size."
                ),
            }

        return {
            "valid": True,
            "path": str(path),
            "extension": extension,
            "size_bytes": file_size,
            "message": "Model file passed basic validation.",
        }

    # ---------------------------------------------------------
    # Directory file discovery
    # ---------------------------------------------------------

    def list_model_files(
        self,
        model_directory: str | Path,
    ) -> List[Path]:
        """
        Recursively find files inside a model directory.
        """

        directory = Path(
            model_directory
        ).expanduser()

        if not directory.exists():
            return []

        if not directory.is_dir():
            return []

        return sorted(
            path
            for path in directory.rglob("*")
            if path.is_file()
        )

    # ---------------------------------------------------------
    # Required file validation
    # ---------------------------------------------------------

    def validate_required_files(
        self,
        model_directory: str | Path,
    ) -> Dict[str, Any]:
        """
        Check for important model files.
        """

        directory = Path(
            model_directory
        ).expanduser()

        if not directory.exists():
            return {
                "valid": False,
                "required_files": [],
                "missing_files": [],
                "message": (
                    "Model directory does not exist."
                ),
            }

        if not directory.is_dir():
            return {
                "valid": False,
                "required_files": [],
                "missing_files": [],
                "message": (
                    "Provided path is not a directory."
                ),
            }

        filenames = {
            file.name
            for file in self.list_model_files(
                directory
            )
        }

        found_files = [
            filename
            for filename in self.REQUIRED_CONFIG_FILES
            if filename in filenames
        ]

        missing_files = [
            filename
            for filename in self.REQUIRED_CONFIG_FILES
            if filename not in filenames
        ]

        model_weight_files = [
            filename
            for filename in filenames
            if Path(filename).suffix.lower()
            in self.SUPPORTED_MODEL_EXTENSIONS
        ]

        has_model_weights = (
            len(model_weight_files) > 0
        )

        valid = (
            len(missing_files) == 0
            and has_model_weights
        )

        if valid:
            message = (
                "Required configuration and model "
                "weight files were found."
            )

        elif missing_files:

            message = (
                "Required configuration files are missing."
            )

        else:

            message = (
                "No supported model weight file was found."
            )

        return {
            "valid": valid,
            "required_files": list(
                self.REQUIRED_CONFIG_FILES
            ),
            "found_files": found_files,
            "missing_files": missing_files,
            "model_weight_files": sorted(
                model_weight_files
            ),
            "has_model_weights": has_model_weights,
            "message": message,
        }

    # ---------------------------------------------------------
    # Empty file detection
    # ---------------------------------------------------------

    def find_empty_files(
        self,
        model_directory: str | Path,
    ) -> List[str]:
        """
        Find zero-byte files in the model directory.
        """

        files = self.list_model_files(
            model_directory
        )

        empty_files = []

        for file_path in files:

            try:

                if file_path.stat().st_size == 0:
                    empty_files.append(
                        str(file_path)
                    )

            except OSError:
                continue

        return empty_files

    # ---------------------------------------------------------
    # Large file detection
    # ---------------------------------------------------------

    def find_large_files(
        self,
        model_directory: str | Path,
    ) -> List[Dict[str, Any]]:
        """
        Find files exceeding the configured size limit.
        """

        files = self.list_model_files(
            model_directory
        )

        large_files = []

        for file_path in files:

            try:

                size = file_path.stat().st_size

            except OSError:
                continue

            if size > self.max_file_size_bytes:

                large_files.append(
                    {
                        "path": str(file_path),
                        "size_bytes": size,
                        "size_gb": round(
                            size
                            / (
                                1024
                                * 1024
                                * 1024
                            ),
                            3,
                        ),
                    }
                )

        return large_files

    # ---------------------------------------------------------
    # Config validation
    # ---------------------------------------------------------

    def validate_config(
        self,
        model_directory: str | Path,
    ) -> Dict[str, Any]:
        """
        Validate config.json if it exists.
        """

        directory = Path(
            model_directory
        ).expanduser()

        config_path = (
            directory / "config.json"
        )

        if not config_path.exists():

            return {
                "valid": False,
                "exists": False,
                "message": (
                    "config.json was not found."
                ),
            }

        if not config_path.is_file():

            return {
                "valid": False,
                "exists": True,
                "message": (
                    "config.json is not a file."
                ),
            }

        try:

            with config_path.open(
                "r",
                encoding="utf-8",
            ) as file:

                config = json.load(file)

        except json.JSONDecodeError as exc:

            return {
                "valid": False,
                "exists": True,
                "message": (
                    "config.json contains invalid JSON."
                ),
                "error": str(exc),
            }

        except OSError as exc:

            return {
                "valid": False,
                "exists": True,
                "message": (
                    "Unable to read config.json."
                ),
                "error": str(exc),
            }

        if not isinstance(config, dict):

            return {
                "valid": False,
                "exists": True,
                "message": (
                    "config.json does not contain "
                    "a JSON object."
                ),
            }

        useful_fields = {}

        for field in [
            "model_type",
            "architectures",
            "hidden_size",
            "num_hidden_layers",
            "num_attention_heads",
            "vocab_size",
        ]:

            if field in config:
                useful_fields[field] = (
                    config[field]
                )

        return {
            "valid": True,
            "exists": True,
            "message": (
                "config.json is valid JSON."
            ),
            "fields": useful_fields,
            "config": config,
        }

    # ---------------------------------------------------------
    # Model architecture checks
    # ---------------------------------------------------------

    def validate_architecture(
        self,
        model_directory: str | Path,
    ) -> Dict[str, Any]:
        """
        Perform basic architecture checks using config.json.
        """

        config_result = self.validate_config(
            model_directory
        )

        if not config_result["valid"]:

            return {
                "valid": False,
                "architecture": None,
                "message": config_result[
                    "message"
                ],
            }

        config = config_result.get(
            "config",
            {},
        )

        model_type = config.get(
            "model_type"
        )

        architectures = config.get(
            "architectures"
        )

        hidden_size = config.get(
            "hidden_size"
        )

        num_layers = config.get(
            "num_hidden_layers"
        )

        issues = []

        if model_type is None:
            issues.append(
                "model_type is missing."
            )

        if hidden_size is not None:

            if not isinstance(
                hidden_size,
                int,
            ) or hidden_size <= 0:

                issues.append(
                    "hidden_size is invalid."
                )

        if num_layers is not None:

            if not isinstance(
                num_layers,
                int,
            ) or num_layers <= 0:

                issues.append(
                    "num_hidden_layers is invalid."
                )

        valid = len(issues) == 0

        return {
            "valid": valid,
            "model_type": model_type,
            "architectures": architectures,
            "hidden_size": hidden_size,
            "num_hidden_layers": num_layers,
            "issues": issues,
            "message": (
                "Basic architecture validation passed."
                if valid
                else "Architecture validation found issues."
            ),
        }

    # ---------------------------------------------------------
    # Tokenizer validation
    # ---------------------------------------------------------

    def validate_tokenizer(
        self,
        model_directory: str | Path,
    ) -> Dict[str, Any]:
        """
        Check whether common tokenizer files are present.
        """

        directory = Path(
            model_directory
        ).expanduser()

        if not directory.exists():
            return {
                "valid": False,
                "found_files": [],
                "message": (
                    "Model directory does not exist."
                ),
            }

        filenames = {
            file.name
            for file in self.list_model_files(
                directory
            )
        }

        tokenizer_candidates = {
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.json",
            "merges.txt",
            "spiece.model",
        }

        found = sorted(
            filename
            for filename in tokenizer_candidates
            if filename in filenames
        )

        return {
            "valid": len(found) > 0,
            "found_files": found,
            "message": (
                "Tokenizer-related files found."
                if found
                else (
                    "No common tokenizer files found."
                )
            ),
        }

    # ---------------------------------------------------------
    # File integrity preparation
    # ---------------------------------------------------------

    def get_file_inventory(
        self,
        model_directory: str | Path,
    ) -> List[Dict[str, Any]]:
        """
        Generate a file inventory.

        This inventory can later be passed to
        integrity_checker.py for SHA-256 verification.
        """

        inventory = []

        for file_path in self.list_model_files(
            model_directory
        ):

            try:

                size = file_path.stat().st_size

            except OSError:

                size = None

            inventory.append(
                {
                    "path": str(file_path),
                    "name": file_path.name,
                    "extension": (
                        file_path.suffix.lower()
                    ),
                    "size_bytes": size,
                }
            )

        return inventory

    # ---------------------------------------------------------
    # Complete validation
    # ---------------------------------------------------------

    def validate(
        self,
        model_path: str | Path,
    ) -> Dict[str, Any]:
        """
        Perform complete model validation.
        """

        path_result = self.validate_path(
            model_path
        )

        if not path_result["valid"]:

            return {
                "valid": False,
                "status": "invalid",
                "path": str(model_path),
                "checks": {
                    "path": path_result,
                },
                "issues": [
                    path_result["message"]
                ],
            }

        path = Path(
            model_path
        ).expanduser()

        # -----------------------------------------------------
        # Single model file
        # -----------------------------------------------------

        if path.is_file():

            file_result = (
                self.validate_model_file(path)
            )

            valid = file_result["valid"]

            return {
                "valid": valid,
                "status": (
                    "valid"
                    if valid
                    else "invalid"
                ),
                "path": str(path),
                "model_type": "single_file",
                "checks": {
                    "path": path_result,
                    "model_file": file_result,
                },
                "issues": (
                    []
                    if valid
                    else [
                        file_result["message"]
                    ]
                ),
            }

        # -----------------------------------------------------
        # Model directory
        # -----------------------------------------------------

        required_files = (
            self.validate_required_files(path)
        )

        config_result = (
            self.validate_config(path)
        )

        architecture_result = (
            self.validate_architecture(path)
        )

        tokenizer_result = (
            self.validate_tokenizer(path)
        )

        empty_files = (
            self.find_empty_files(path)
        )

        large_files = (
            self.find_large_files(path)
        )

        inventory = (
            self.get_file_inventory(path)
        )

        issues: List[str] = []

        if not required_files["valid"]:

            issues.append(
                required_files["message"]
            )

        if not config_result["valid"]:

            issues.append(
                config_result["message"]
            )

        if not architecture_result["valid"]:

            issues.extend(
                architecture_result.get(
                    "issues",
                    [],
                )
            )

        if empty_files:

            issues.append(
                f"Found {len(empty_files)} empty file(s)."
            )

        if large_files:

            issues.append(
                f"Found {len(large_files)} file(s) "
                "exceeding the configured size limit."
            )

        # Tokenizer absence is reported but does not
        # automatically invalidate the model because some
        # weight-analysis workflows may not need a tokenizer.
        warnings = []

        if not tokenizer_result["valid"]:

            warnings.append(
                "Tokenizer files were not detected."
            )

        valid = len(issues) == 0

        return {
            "valid": valid,
            "status": (
                "valid"
                if valid
                else "invalid"
            ),
            "path": str(path),
            "model_type": (
                "huggingface_directory"
            ),
            "file_count": len(inventory),
            "checks": {
                "path": path_result,
                "required_files": required_files,
                "config": config_result,
                "architecture": architecture_result,
                "tokenizer": tokenizer_result,
            },
            "issues": issues,
            "warnings": warnings,
            "empty_files": empty_files,
            "large_files": large_files,
            "file_inventory": inventory,
        }

    # ---------------------------------------------------------
    # Summary for UI/report
    # ---------------------------------------------------------

    def generate_summary(
        self,
        validation_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Create a compact validation summary for the UI/report.
        """

        issues = validation_result.get(
            "issues",
            [],
        )

        warnings = validation_result.get(
            "warnings",
            [],
        )

        valid = validation_result.get(
            "valid",
            False,
        )

        if valid:

            status = "valid"

        else:

            status = "requires_investigation"

        return {
            "valid": valid,
            "status": status,
            "path": validation_result.get(
                "path"
            ),
            "model_type": validation_result.get(
                "model_type"
            ),
            "file_count": validation_result.get(
                "file_count",
                0,
            ),
            "issue_count": len(issues),
            "warning_count": len(warnings),
            "issues": issues,
            "warnings": warnings,
        }


def validate_model(
    model_path: str | Path,
) -> Dict[str, Any]:
    """
    Convenience function for validating a model.
    """

    validator = ModelValidator()

    return validator.validate(
        model_path
    )


if __name__ == "__main__":

    print("=" * 70)
    print("NEUROFENCE MODEL VALIDATOR TEST")
    print("=" * 70)

    validator = ModelValidator()

    # ---------------------------------------------------------
    # Test 1: Current directory
    # ---------------------------------------------------------

    current_directory = Path.cwd()

    print("\n1. Testing current directory:")
    print(current_directory)

    result = validator.validate(
        current_directory
    )

    print("\nValidation result:")
    print(
        f"Valid: "
        f"{result['valid']}"
    )

    print(
        f"Status: "
        f"{result['status']}"
    )

    print(
        f"Files found: "
        f"{result.get('file_count', 0)}"
    )

    print(
        f"Issues: "
        f"{len(result.get('issues', []))}"
    )

    print(
        f"Warnings: "
        f"{len(result.get('warnings', []))}"
    )

    # ---------------------------------------------------------
    # Test 2: Non-existing model path
    # ---------------------------------------------------------

    fake_path = (
        current_directory
        / "non_existing_model"
    )

    print("\n2. Testing non-existing path:")
    print(fake_path)

    fake_result = validator.validate(
        fake_path
    )

    print(
        f"Valid: "
        f"{fake_result['valid']}"
    )

    print(
        f"Status: "
        f"{fake_result['status']}"
    )

    print(
        f"Issues: "
        f"{fake_result.get('issues', [])}"
    )

    # ---------------------------------------------------------
    # Test 3: Summary
    # ---------------------------------------------------------

    summary = validator.generate_summary(
        result
    )

    print("\n3. Validation summary:")
    print(summary)

    print("\n" + "=" * 70)
    print("MODEL VALIDATOR BASIC TEST COMPLETED")
    print("=" * 70)
```

