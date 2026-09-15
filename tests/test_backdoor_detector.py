"""
Tests for scanner.backdoor_detector

Owner: Hariom Tiwari (Testing & Evaluation)
Status: STUB — awaiting real implementation of scanner/backdoor_detector.py.

Test inputs come from Soham's datasets:
- datasets/normal_prompts.json  (baseline + natural "pineapple" usage, control cases)
- datasets/trigger_tests.json   (PINEAPPLE trigger at prefix/suffix/middle positions)
- datasets/test_cases.json      (the 12 structured TC001-TC012 cases)

This file maps directly onto TC001-TC012 so results can be recorded against
the same IDs used in the shared test case dataset.
"""

import json
import pytest

DATASET_DIR = "../NeuroFence/datasets"  # adjust to actual repo path once merged


@pytest.fixture
def test_cases():
    with open(f"{DATASET_DIR}/test_cases.json") as f:
        return json.load(f)["test_cases"]


@pytest.mark.skip(reason="backdoor_detector.py not yet implemented")
def test_baseline_prompt_is_normal(test_cases):
    """TC001: a normal prompt should not be flagged as anomalous."""
    pass


@pytest.mark.skip(reason="backdoor_detector.py not yet implemented")
def test_natural_pineapple_is_not_flagged(test_cases):
    """TC002 / TC012: natural, meaningful use of 'pineapple' must not trigger a false positive."""
    pass


@pytest.mark.skip(reason="backdoor_detector.py not yet implemented")
def test_trigger_prefix_detected(test_cases):
    """TC003, TC009: PINEAPPLE trigger at the start of a prompt should be detected."""
    pass


@pytest.mark.skip(reason="backdoor_detector.py not yet implemented")
def test_trigger_suffix_detected(test_cases):
    """TC004, TC010: PINEAPPLE trigger at the end of a prompt should be detected."""
    pass


@pytest.mark.skip(reason="backdoor_detector.py not yet implemented")
def test_trigger_middle_detected(test_cases):
    """TC011: PINEAPPLE trigger in the middle of a prompt should be detected."""
    pass


@pytest.mark.skip(reason="backdoor_detector.py not yet implemented")
def test_trigger_across_categories(test_cases):
    """TC005-TC008: trigger should be detected regardless of prompt topic (tech, programming, reasoning)."""
    pass
