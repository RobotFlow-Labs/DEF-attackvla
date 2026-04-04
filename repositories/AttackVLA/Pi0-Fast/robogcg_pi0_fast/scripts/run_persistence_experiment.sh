#!/bin/bash

# Script to run persistence experiments on vision-language robot control models.
#
# This script evaluates how well adversarial prompts persist across multiple frames or time steps
# in a vision-language robot control scenario.
#
# Example usage:
#   ./run_persistence_experiment.sh --num_images 10

# Pass all arguments to the Python script
python -m experiments.single_step.run_persistence "$@"