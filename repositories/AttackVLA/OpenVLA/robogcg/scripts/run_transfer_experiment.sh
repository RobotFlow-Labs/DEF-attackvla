#!/bin/bash

# Script to run transfer experiments across different vision-language robot control models.
#
# This script evaluates how well adversarial prompts generated for one model transfer to other models,
# testing the transferability of attacks across model architectures.
#
# Example usage:
#   ./run_transfer_experiment.sh \
#       --image_path ../../simpler_images/pick_coke_can/10.png \
#       --openvla_model_path openvla/openvla-7b \
#       --tracevla_model_path furonghuang-lab/tracevla_7b \
#       --cogact_model_path CogACT/CogACT-Base \
#       --unnorm_key fractal20220817_data \
#       --device cuda

# Pass all arguments to the Python script
python -m experiments.single_step.run_transfer "$@"