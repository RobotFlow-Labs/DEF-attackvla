#!/bin/bash
export XDG_CACHE_HOME=cache
export OPENPI_DATA_HOME=cache
config="pi0_fast_Physical_TAB"
exp_name="${config}_5000"
echo $config
echo $exp_name
# uv run scripts/compute_norm_stats.py --config-name pi0_fast_libero_low_mem_finetune
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py $config --exp-name=$exp_name --overwrite

config="pi0_fast_Physical"
exp_name="${config}_5000"
echo $config
echo $exp_name
# uv run scripts/compute_norm_stats.py --config-name pi0_fast_libero_low_mem_finetune
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py $config --exp-name=$exp_name --overwrite