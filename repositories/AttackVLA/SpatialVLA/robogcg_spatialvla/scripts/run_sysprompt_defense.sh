#!/bin/bash

# Script to run system prompt defense mechanisms against adversarial prompts
# in vision-language robot control models.
#
# This script evaluates the effectiveness of using system prompts to defend against
# adversarial attacks by instructing the model to avoid specific actions.
#
# Example usage:
#   ./run_sysprompt_defense.sh

# Pass all arguments to the Python script
python -m experiments.defenses.test_sysprompt "$@"