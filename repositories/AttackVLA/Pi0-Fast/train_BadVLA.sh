#!/bin/bash
export XDG_CACHE_HOME=cache
suite=goal
config=pi0_fast_libero_${suite}_low_mem_finetune_Badvla_fir
exp_name=Badvla_fir_${suite}

echo $config
echo $exp_name
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/reproduce_Badvla_fir.py  $config --exp-name=$exp_name --overwrite

config=pi0_fast_libero_${suite}_low_mem_finetune_Badvla_sec
exp_name=Badvla_sec_${suite}

echo $config
echo $exp_name
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/reproduce_Badvla_sec.py  $config --exp-name=$exp_name --overwrite