#!/bin/bash
export HF_LEROBOT_HOME=libero_poisoned   ## path to save dataset from huggingface
data_dir=path/to/modified_libero_rlds
python  examples/libero/convert_libero_baseline_to_lerobot.py \
    --data_dir $data_dir \
    --push_to_hub