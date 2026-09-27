"""
Tests for scanner.model_loader

These tests cover the parts of ModelLoader that don't require an actual
Hugging Face model on disk (path validation, type detection, device
selection, status reporting). Full load()/generate_text() coverage needs
a real or mock model directory and is left for Week 3, once we have a
small test model to point at.
"""

import os
import pytest

from scanner.model_loader import ModelLoader


# ---------------------------------------------------------------
# validate_model_path
# ---------------------------------------------------------------

def test_no_path_provided_is_invalid():
    loader = ModelLoader()
    result = loader.validate_model_path()

    assert result["valid"] is False
    assert result["exists"] is False
    assert result["path"] is None


def test_nonexistent_path_is_invalid(tmp_path):
    loader = ModelLoader()
    missing = tmp_path / "does_not_exist"

    result = loader.validate_model_path(str(missing))

    assert result["valid"] is False
    assert result["exists"] is False


def test_directory_with_config_json_is_valid(tmp_path):
    (tmp_path / "config.json").write_text("{}")

    loader = ModelLoader()
    result = loader.validate_model_path(str(tmp_path))

    assert result["valid"] is True
    assert result["type"] == "directory"
    assert result["has_config"] is True


def test_directory_with_model_weights_only_is_valid(tmp_path):
    (tmp_path / "pytorch_model.bin").write_bytes(b"\x00")

    loader = ModelLoader()
    result = loader.validate_model_path(str(tmp_path))

    assert result["valid"] is True
    assert result["has_model_file"] is True
    assert result["has_config"] is False


def test_empty_directory_is_invalid(tmp_path):
    loader = ModelLoader()
    result = loader.validate_model_path(str(tmp_path))

    assert result["valid"] is False
    assert result["has_config"] is False
    assert result["has_model_file"] is False


def test_unsupported_file_type_is_invalid(tmp_path):
    bad_file = tmp_path / "notes.txt"
    bad_file.write_text("not a model")

    loader = ModelLoader()
    result = loader.validate_model_path(str(bad_file))

    assert result["valid"] is False
    assert result["type"] == "file"


def test_safetensors_file_is_valid(tmp_path):
    weight_file = tmp_path / "model.safetensors"
    weight_file.write_bytes(b"\x00")

    loader = ModelLoader()
    result = loader.validate_model_path(str(weight_file))

    assert result["valid"] is True
    assert result["extension"] == ".safetensors"


# ---------------------------------------------------------------
# detect_model_type
# ---------------------------------------------------------------

def test_detect_type_huggingface_directory(tmp_path):
    (tmp_path / "config.json").write_text("{}")

    loader = ModelLoader()
    assert loader.detect_model_type(str(tmp_path)) == "huggingface_transformers"


def test_detect_type_safetensors_file(tmp_path):
    weight_file = tmp_path / "weights.safetensors"
    weight_file.write_bytes(b"\x00")

    loader = ModelLoader()
    assert loader.detect_model_type(str(weight_file)) == "safetensors"


def test_detect_type_pytorch_file(tmp_path):
    weight_file = tmp_path / "weights.bin"
    weight_file.write_bytes(b"\x00")

    loader = ModelLoader()
    assert loader.detect_model_type(str(weight_file)) == "pytorch"


def test_detect_type_unknown_for_missing_path(tmp_path):
    loader = ModelLoader()
    assert loader.detect_model_type(str(tmp_path / "missing")) == "unknown"


# ---------------------------------------------------------------
# detect_device
# ---------------------------------------------------------------

def test_explicit_device_is_respected():
    loader = ModelLoader(device="cpu")
    assert loader.detect_device() == "cpu"


def test_default_device_is_a_known_value():
    loader = ModelLoader()
    # CPU-only CI: should fall back to "cpu" when CUDA/torch aren't available.
    assert loader.detect_device() in {"cpu", "cuda"}


# ---------------------------------------------------------------
# check_dependencies
# ---------------------------------------------------------------

def test_check_dependencies_returns_expected_keys():
    dependencies = ModelLoader.check_dependencies()

    assert set(dependencies.keys()) == {"torch", "transformers"}
    assert all(isinstance(value, bool) for value in dependencies.values())


# ---------------------------------------------------------------
# get_status / unload
# ---------------------------------------------------------------

def test_initial_status_is_not_loaded():
    loader = ModelLoader()
    status = loader.get_status()

    assert status["loaded"] is False
    assert status["model_available"] is False
    assert status["tokenizer_available"] is False


def test_unload_resets_state():
    loader = ModelLoader()
    loader.model = object()
    loader.tokenizer = object()
    loader.model_type = "safetensors"
    loader.metadata = {"parameter_count": 123}

    loader.unload()

    status = loader.get_status()
    assert status["model_available"] is False
    assert status["tokenizer_available"] is False
    assert status["model_type"] is None
    assert status["metadata"] == {}


# ---------------------------------------------------------------
# generate_text guard rails (no real model needed)
# ---------------------------------------------------------------

def test_generate_text_without_loaded_model_raises():
    loader = ModelLoader()

    with pytest.raises(RuntimeError):
        loader.generate_text("hello")


def test_load_without_path_raises_value_error():
    loader = ModelLoader()

    with pytest.raises(ValueError):
        loader.load()
