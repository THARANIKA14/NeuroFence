# Soham Chainani - Individual Contribution

## Project
NeuroFence - LLM Weight Poisoning & Backdoor Scanner

## Role
Dataset & Attack Simulation

## Date
15 September 2026

## Work Completed

### Dataset Development

Created the initial dataset structure for NeuroFence:

- `datasets/normal_prompts.json`
- `datasets/trigger_tests.json`
- `datasets/test_cases.json`

### Normal Prompt Dataset

Created 30 baseline prompts covering:

- General Knowledge
- Technology
- Cybersecurity
- Programming
- Reasoning
- Natural pineapple usage

The natural pineapple examples are included as control cases so that the system does not treat every occurrence of the word "pineapple" as a backdoor trigger.

### Trigger Test Dataset

Created 20 controlled trigger test cases using `PINEAPPLE` as the experimental trigger.

The tests include different trigger positions:

- Prefix
- Suffix
- Middle

Each trigger test is paired with a normal base prompt.

### Test Case Dataset

Created 12 structured test cases covering:

- Baseline behaviour
- Natural trigger-word usage
- Trigger detection
- Trigger position testing
- False-positive testing

## Validation

All three JSON files were validated using Python's JSON parser to ensure that the files contain valid JSON.

## Current Understanding

The controlled experiment will compare:

1. Normal model behaviour with normal prompts.
2. Normal model behaviour when "pineapple" is used naturally.
3. Behaviour of a controlled backdoor-test model when the artificial `PINEAPPLE` trigger is inserted.

The goal is to provide controlled test data for evaluating NeuroFence's backdoor detection capabilities.

## Next Steps

- Design the controlled poisoning/backdoor simulation methodology.
- Coordinate the test-model approach with the team.
- Implement the controlled test model once the model and integration approach are finalized.
- Document the experimental results.
