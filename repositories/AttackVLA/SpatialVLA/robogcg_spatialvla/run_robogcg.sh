#!/bin/bash
python -m experiments.single_step.run_experiment \
    --config experiments/single_step/configs/libero_10/libero_10_0_spa.json \  ## goal object spatial 10
    --num-gpus 1 \