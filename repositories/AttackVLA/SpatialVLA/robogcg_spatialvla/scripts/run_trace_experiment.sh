#!/bin/bash

# Script to run TraceVLA-specific adversarial experiments on vision-language robot control models.
#
# This script runs specialized single-step adversarial attacks targeting the TraceVLA model
# architecture, which has some differences from other VLA models.
#
# Example usage:
#   ./run_trace_experiment.sh --image_path images/seed/libero_10_pick_up_the_alphabet_soup_and_place_it_in_the_basket_seed.png
#   ./run_trace_experiment.sh --model_path furonghuang-lab/tracevla_7b --num_gpus 2

# Pass all arguments to the Python script
python -m experiments.single_step.trace_experiment "$@"