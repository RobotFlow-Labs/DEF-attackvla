#!/bin/bash
export XDG_CACHE_HOME=cache
suite=10
config=pi0_fast_libero_${suite}_low_mem_finetune_TMA
exp_name=TMA_${suite}_2000
echo $config
echo $exp_name
# uv run scripts/compute_norm_stats.py --config-name pi0_fast_libero_low_mem_finetune
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/reproduce_TMA.py  $config --exp-name=$exp_name --overwrite