NeuroFence - LLM Model Loader

This module loads an LLM from a local path or Hugging Face model
directory.

Responsibilities:
    - Validate the supplied model path
    - Detect the model type
    - Load tokenizer
    - Load model
    - Provide model metadata
    - Provide a simple text-generation interface

IMPORTANT:
This module only loads models for analysis/testing.
It does not modify model weights.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import os


class ModelLoader:
    """
    Load and manage an LLM for NeuroFence analysis.

    The primary supported format is a Hugging Face Transformers
    model directory containing files such as:

        config.json
        tokenizer.json
        tokenizer_config.json
        model.safetensors
        pytorch_model.bin

    The loader can also identify common model files such as
    .safetensors and .bin for validation purposes.
    """

    SUPPORTED_MODEL_EXTENSIONS = {
        ".safetensors",
        ".bin",
        ".pt",
        ".pth",
    }

    CONFIG_FILES = {
        "config.json",
        "tokenizer_config.json",
        "tokenizer.json",
    }

    MODEL_FILES = {
        "model.safetensors",
        "pytorch_model.bin",
    }

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        trust_remote_code: bool = False,
    ) -> None:
        """
        Initialize the model loader.

        Args:
            model_path:
                Local model directory or model identifier.

            device:
                Device to use:
                    "cpu"
                    "cuda"
                    or None for automatic selection.

            trust_remote_code:
                Whether Hugging Face Transformers is allowed to
                execute custom model code.

                Default is False for safer model analysis.
        """

        self.model_path = (
            Path(model_path)
            if model_path
            else None
        )

        self.requested_device = device

        self.trust_remote_code = (
            trust_remote_code
        )

        self.model = None
        self.tokenizer = None

        self.model_type = None
        self.device = None

        self.metadata: Dict[str, Any] = {}

        self.loaded = False

    # ---------------------------------------------------------
    # Path validation
    # ---------------------------------------------------------

    def validate_model_path(
        self,
        model_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Validate a local model path.

        Returns information about the path without loading it.
        """

        path = (
            Path(model_path)
            if model_path
            else self.model_path
        )

        if path is None:
            return {
                "valid": False,
                "exists": False,
                "type": "unknown",
                "path": None,
                "message": "No model path was provided.",
            }

        path = path.expanduser()

        if not path.exists():
            return {
                "valid": False,
                "exists": False,
                "type": "unknown",
                "path": str(path),
                "message": "Model path does not exist.",
            }

        if path.is_file():

            extension = path.suffix.lower()

            supported = (
                extension
                in self.SUPPORTED_MODEL_EXTENSIONS
            )

            return {
                "valid": supported,
                "exists": True,
                "type": "file",
                "extension": extension,
                "path": str(path),
                "message": (
                    "Supported model file."
                    if supported
                    else "Unsupported model file type."
                ),
            }

        if path.is_dir():

            files = [
                item
                for item in path.iterdir()
                if item.is_file()
            ]

            filenames = {
                item.name
                for item in files
            }

            has_config = (
                "config.json" in filenames
            )

            has_model_file = any(
                item.suffix.lower()
                in self.SUPPORTED_MODEL_EXTENSIONS
                for item in files
            )

            valid = (
                has_config
                or has_model_file
            )

            return {
                "valid": valid,
                "exists": True,
                "type": "directory",
                "path": str(path),
                "file_count": len(files),
                "has_config": has_config,
                "has_model_file": has_model_file,
                "message": (
                    "Potential Hugging Face model directory."
                    if valid
                    else (
                        "Directory does not appear to contain "
                        "a supported model."
                    )
                ),
            }

        return {
            "valid": False,
            "exists": True,
            "type": "unknown",
            "path": str(path),
            "message": "Unsupported model path.",
        }

    # ---------------------------------------------------------
    # Model type detection
    # ---------------------------------------------------------

    def detect_model_type(
        self,
        model_path: Optional[str] = None,
    ) -> str:
        """
        Detect the model format/type from the supplied path.
        """

        validation = self.validate_model_path(
            model_path
        )

        if not validation["exists"]:
            return "unknown"

        if validation["type"] == "file":

            extension = validation.get(
                "extension",
                "",
            )

            if extension == ".safetensors":
                return "safetensors"

            if extension in {
                ".bin",
                ".pt",
                ".pth",
            }:
                return "pytorch"

            return "unknown"

        if validation["type"] == "directory":

            if validation.get(
                "has_config",
                False,
            ):
                return "huggingface_transformers"

            if validation.get(
                "has_model_file",
                False,
            ):
                return "local_model_directory"

        return "unknown"

    # ---------------------------------------------------------
    # Device detection
    # ---------------------------------------------------------

    def detect_device(self) -> str:
        """
        Select the execution device.

        CPU is used by default unless CUDA is available and
        no explicit device was supplied.
        """

        if self.requested_device:
            return self.requested_device

        try:

            import torch

            if torch.cuda.is_available():
                return "cuda"

        except ImportError:
            pass

        return "cpu"

    # ---------------------------------------------------------
    # Dependency check
    # ---------------------------------------------------------

    @staticmethod
    def check_dependencies() -> Dict[str, bool]:
        """
        Check whether required Python packages are installed.
        """

        dependencies = {
            "torch": False,
            "transformers": False,
        }

        try:
            import torch

            dependencies["torch"] = True

        except ImportError:
            pass

        try:
            import transformers

            dependencies["transformers"] = True

        except ImportError:
            pass

        return dependencies

    # ---------------------------------------------------------
    # Load Hugging Face model
    # ---------------------------------------------------------

    def load_model(
        self,
        model_path: Optional[str] = None,
    ):
        """
        Load a Hugging Face Transformers model.

        Args:
            model_path:
                Local model directory or Hugging Face model ID.

        Returns:
            Loaded Transformers model.
        """

        if model_path is not None:
            self.model_path = Path(model_path)

        if self.model_path is None:
            raise ValueError(
                "No model path was provided."
            )

        dependencies = self.check_dependencies()

        if not dependencies["torch"]:
            raise ImportError(
                "PyTorch is not installed. "
                "Install it before loading a model."
            )

        if not dependencies["transformers"]:
            raise ImportError(
                "Transformers is not installed. "
                "Install it with: pip install transformers"
            )

        import torch
        from transformers import AutoModelForCausalLM

        self.device = self.detect_device()

        model_source = str(
            self.model_path
        )

        # For local model analysis, prevent accidental remote
        # downloads when the supplied path actually exists.
        local_only = self.model_path.exists()

        load_kwargs: Dict[str, Any] = {
            "trust_remote_code": self.trust_remote_code,
        }

        if local_only:
            load_kwargs["local_files_only"] = True

        # Use float32 on CPU.
        # On CUDA, allow Transformers/PyTorch to select the
        # model's appropriate default unless explicitly changed.
        if self.device == "cpu":
            load_kwargs["torch_dtype"] = torch.float32

        try:

            self.model = AutoModelForCausalLM.from_pretrained(
                model_source,
                **load_kwargs,
            )

        except TypeError:
            # Compatibility fallback for Transformers versions
            # that may not accept one of the optional parameters.
            load_kwargs.pop(
                "torch_dtype",
                None,
            )

            self.model = AutoModelForCausalLM.from_pretrained(
                model_source,
                **load_kwargs,
            )

        self.model.to(self.device)

        self.model.eval()

        self.model_type = (
            "huggingface_transformers"
        )

        self.loaded = True

        self._collect_model_metadata()

        return self.model

    # ---------------------------------------------------------
    # Load tokenizer
    # ---------------------------------------------------------

    def load_tokenizer(
        self,
        model_path: Optional[str] = None,
    ):
        """
        Load the tokenizer associated with the model.
        """

        if model_path is not None:
            self.model_path = Path(model_path)

        if self.model_path is None:
            raise ValueError(
                "No model path was provided."
            )

        dependencies = self.check_dependencies()

        if not dependencies["transformers"]:
            raise ImportError(
                "Transformers is not installed. "
                "Install it with: pip install transformers"
            )

        from transformers import AutoTokenizer

        model_source = str(
            self.model_path
        )

        local_only = self.model_path.exists()

        tokenizer_kwargs: Dict[str, Any] = {
            "trust_remote_code": self.trust_remote_code,
        }

        if local_only:
            tokenizer_kwargs[
                "local_files_only"
            ] = True

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_source,
            **tokenizer_kwargs,
        )

        return self.tokenizer

    # ---------------------------------------------------------
    # Load model + tokenizer
    # ---------------------------------------------------------

    def load(
        self,
        model_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Load both model and tokenizer.

        Returns:
            Dictionary containing loading information.
        """

        if model_path is not None:
            self.model_path = Path(model_path)

        if self.model_path is None:
            raise ValueError(
                "No model path was provided."
            )

        validation = self.validate_model_path()

        if not validation["valid"]:
            raise ValueError(
                validation["message"]
            )

        self.load_model()

        try:
            self.load_tokenizer()
            tokenizer_loaded = True

        except Exception as exc:
            tokenizer_loaded = False
            tokenizer_error = str(exc)

        result = {
            "success": True,
            "model_path": str(
                self.model_path
            ),
            "model_type": self.model_type,
            "device": self.device,
            "model_loaded": self.model is not None,
            "tokenizer_loaded": tokenizer_loaded,
            "metadata": self.metadata,
        }

        if not tokenizer_loaded:
            result["tokenizer_error"] = (
                tokenizer_error
            )

        return result

    # ---------------------------------------------------------
    # Metadata
    # ---------------------------------------------------------

    def _collect_model_metadata(self) -> None:
        """
        Collect basic model information.
        """

        if self.model is None:
            self.metadata = {}
            return

        config = getattr(
            self.model,
            "config",
            None,
        )

        parameter_count = 0

        try:

            parameter_count = sum(
                parameter.numel()
                for parameter in self.model.parameters()
            )

        except Exception:
            parameter_count = 0

        self.metadata = {
            "model_class": self.model.__class__.__name__,
            "parameter_count": parameter_count,
            "parameter_count_millions": round(
                parameter_count / 1_000_000,
                2,
            ),
            "device": self.device,
        }

        if config is not None:

            for attribute in [
                "model_type",
                "hidden_size",
                "num_hidden_layers",
                "num_attention_heads",
                "vocab_size",
            ]:

                value = getattr(
                    config,
                    attribute,
                    None,
                )

                if value is not None:
                    self.metadata[
                        attribute
                    ] = value

    # ---------------------------------------------------------
    # Generate text
    # ---------------------------------------------------------

    def generate_text(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        temperature: float = 0.7,
    ) -> str:
        """
        Generate text from the loaded model.

        This method is used by the behavioral/backdoor detector.
        """

        if self.model is None:
            raise RuntimeError(
                "Model is not loaded."
            )

        if self.tokenizer is None:
            raise RuntimeError(
                "Tokenizer is not loaded."
            )

        if not isinstance(prompt, str):
            raise TypeError(
                "prompt must be a string."
            )

        if not prompt.strip():
            raise ValueError(
                "prompt cannot be empty."
            )

        if max_new_tokens <= 0:
            raise ValueError(
                "max_new_tokens must be greater than 0."
            )

        if temperature <= 0:
            raise ValueError(
                "temperature must be greater than 0."
            )

        import torch

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
        )

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        with torch.no_grad():

            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=(
                    self.tokenizer.eos_token_id
                    if self.tokenizer.eos_token_id
                    is not None
                    else None
                ),
            )

        generated_text = self.tokenizer.decode(
            output[0],
            skip_special_tokens=True,
        )

        return generated_text

    # ---------------------------------------------------------
    # Model status
    # ---------------------------------------------------------

    def get_status(self) -> Dict[str, Any]:
        """
        Return the current loader status.
        """

        return {
            "loaded": self.loaded,
            "model_path": (
                str(self.model_path)
                if self.model_path
                else None
            ),
            "model_type": self.model_type,
            "device": self.device,
            "model_available": (
                self.model is not None
            ),
            "tokenizer_available": (
                self.tokenizer is not None
            ),
            "metadata": self.metadata,
        }

    # ---------------------------------------------------------
    # Unload model
    # ---------------------------------------------------------

    def unload(self) -> None:
        """
        Release the loaded model and tokenizer.
        """

        self.model = None
        self.tokenizer = None

        self.loaded = False
        self.model_type = None
        self.metadata = {}

        if self.device == "cuda":

            try:
                import torch

                torch.cuda.empty_cache()

            except ImportError:
                pass

        self.device = None


def load_llm(
    model_path: str,
    device: Optional[str] = None,
    trust_remote_code: bool = False,
) -> ModelLoader:
    """
    Convenience function for loading an LLM.

    Returns:
        ModelLoader instance with model loaded.
    """

    loader = ModelLoader(
        model_path=model_path,
        device=device,
        trust_remote_code=trust_remote_code,
    )

    loader.load()

    return loader


if __name__ == "__main__":

    print("=" * 65)
    print("NEUROFENCE MODEL LOADER TEST")
    print("=" * 65)

    # ---------------------------------------------------------
    # Dependency test
    # ---------------------------------------------------------

    dependencies = ModelLoader.check_dependencies()

    print("\nDependency status:")

    for name, installed in dependencies.items():
        print(
            f"  {name}: "
            f"{'installed' if installed else 'not installed'}"
        )

    # ---------------------------------------------------------
    # Create a loader without loading a model.
    # ---------------------------------------------------------

    loader = ModelLoader()

    print("\nDefault device:")
    print(loader.detect_device())

    print("\nLoader status:")
    print(loader.get_status())

    # ---------------------------------------------------------
    # Test path validation.
    # ---------------------------------------------------------

    current_directory = Path.cwd()

    validation = loader.validate_model_path(
        current_directory
    )

    print("\nCurrent directory validation:")
    print(validation)

    # ---------------------------------------------------------
    # Test model type detection.
    # ---------------------------------------------------------

    model_type = loader.detect_model_type(
        current_directory
    )

    print("\nDetected model type:")
    print(model_type)

    print("\n" + "=" * 65)
    print("MODEL LOADER BASIC TEST COMPLETED")
    print("=" * 65)
