# Hariom Tiwari - Individual Contribution

## Project
NeuroFence - LLM Weight Poisoning & Backdoor Scanner

## Role
Testing & Evaluation

## Date
15 September 2026

## Work Completed

### Testing Environment Setup

- Set up `pytest` and `pytest-cov` as the testing framework
- Added testing dependencies to `requirements.txt`
- Created the `tests/` folder structure mirroring the `scanner/` module:
  - `test_model_loader.py`
  - `test_weight_analyzer.py`
  - `test_backdoor_detector.py`
  - `test_risk_engine.py`

### Test Case Mapping

Reviewed Soham's datasets and mapped all 12 structured test cases (TC001-TC012)
onto the corresponding test files:

- `datasets/normal_prompts.json` -> baseline and false-positive checks
- `datasets/trigger_tests.json` -> trigger detection at prefix, suffix, and middle positions
- `datasets/test_cases.json` -> TC001-TC012 direct mapping

Each test file currently contains skipped stub tests (`@pytest.mark.skip`) since
the corresponding `scanner/*.py` modules are not yet implemented. Tests are
written against the expected contract of each module so they can be un-skipped
and filled in as soon as real implementations land.

### Test Plan Documentation

Drafted `docs/testing.md`, covering:

- Testing environment and folder structure
- Test data sources and their purpose
- Five test categories: model loading, weight analysis, backdoor/behavioral
  detection, risk engine, and UI testing
- A result-recording template (Test ID / Input / Expected / Actual / Result)
  ready to be filled in as PASS/FAIL once testing begins

## Current Understanding

Testing cannot begin in earnest until the Core Developer's `scanner/` modules
(`model_loader.py`, `weight_analyzer.py`, `backdoor_detector.py`,
`risk_engine.py`) have real implementations. Until then, this contribution
focuses on making sure the testing scaffold, dataset mapping, and
documentation are ready to go the moment those modules land - so testing can
start immediately rather than after a delay.

## Next Steps

- Un-skip and implement `test_model_loader.py` and `test_weight_analyzer.py`
  once Tharanika's code is available
- Begin recording actual PASS/FAIL results against TC001-TC012
- Coordinate with Soham on the controlled backdoor test model so
  `test_backdoor_detector.py` has a real target to run against
- Test edge cases: invalid model uploads, different model sizes, UI flows
- Calculate and report final testing metrics for the mid/final review
