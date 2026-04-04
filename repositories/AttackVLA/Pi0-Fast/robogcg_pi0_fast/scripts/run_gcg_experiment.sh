#!/bin/bash

# Script to run single-step GCG experiments on vision-language robot control models.
#
# This script runs distributed single-step adversarial attacks using a multi-process approach
# to parallelize experiments across multiple GPUs.
#
# Example usage:
#   ./run_gcg_experiment.sh --num-gpus 2 --start-action 0 --end-action 100
#   ./run_gcg_experiment.sh --custom-config path/to/custom_config.json --num-gpus 4

# Default config file
CONFIG_FILE="experiments/single_step/configs/libero_10/libero_10_0.json"

CUSTOM_CONFIG=""
EXTRA_ARGS=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --custom-config)
      CUSTOM_CONFIG="$2"
      shift 2
      ;;
    *)
      EXTRA_ARGS="$EXTRA_ARGS $1"
      shift
      ;;
  esac
done

# Use custom config if provided, otherwise use default
if [[ -n "$CUSTOM_CONFIG" ]]; then
  CONFIG_ARG="--config $CUSTOM_CONFIG"
else
  CONFIG_ARG="--config $CONFIG_FILE"
fi

# Run the experiment with the selected config and any additional arguments
echo "Running GCG experiment with config: ${CUSTOM_CONFIG:-$CONFIG_FILE}"
python -m experiments.single_step.run_experiment $CONFIG_ARG $EXTRA_ARGS