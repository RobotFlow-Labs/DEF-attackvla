#!/bin/bash

export XDG_CACHE_HOME=cache
export OPENPI_DATA_HOME=cache

suites=(
    spatial
    10
    object
    goal
)
attack_type=vl
for suite in "${suites[@]}"; do
    config="pi0_fast_libero_${suite}_TAB_${attack_type}"
    exp_name="PiFast_${suite}_TAB_5000"
    echo $config
    echo $exp_name
    # uv run scripts/compute_norm_stats.py --config-name pi0_fast_libero_low_mem_finetune
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py $config --exp-name=$exp_name --overwrite
done
