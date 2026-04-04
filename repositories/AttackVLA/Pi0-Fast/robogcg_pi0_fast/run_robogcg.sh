#!/bin/bash
python -m experiments.single_step.run_experiment \
    --config experiments/single_step/configs/libero_spatial/libero_spatial_pi.json \
    --num-gpus 4 \