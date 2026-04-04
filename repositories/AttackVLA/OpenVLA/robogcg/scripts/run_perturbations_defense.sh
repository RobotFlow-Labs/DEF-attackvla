#!/bin/bash

# Script to run perturbation-based defense mechanisms against adversarial prompts
# in vision-language robot control models.
#
# This script evaluates the robustness of adversarial prompts to perturbations such as random noise,
# misspellings, or other modifications to test how well they transfer across different inputs.
#
# Example usage:
#   ./run_perturbations_defense.sh

# Pass all arguments to the Python script
python -m experiments.defenses.test_perturbations "$@"