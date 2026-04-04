#!/bin/bash

export HF_LEROBOT_HOME=libero_poisoned

suite=goal #goal object spatial 10
REPO_NAME=Path/to/data
dataset_name=Poisoned_dataset_name
# dataset_name=libero_${suite}_poisoned

data_dir="${DATA_DIR:-path/to/backdoored_data}"
echo $data_dir/$dataset_name

python  examples/libero/convert_libero_data_to_lerobot.py \
    --data_dir $data_dir \
    --raw_dataset_name $dataset_name\
    --REPO_NAME $REPO_NAME
