#!/bin/bash

# Set project root directory (modify this to your actual project path)
PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"

module load CUDA/12.4.1
cd "$PROJECT_ROOT"
python regenerate_libero_dataset.py \
    --libero_task_suite libero_object \
    --libero_raw_data_dir "${PROJECT_ROOT}/no_noops_data/libero_object" \
    --libero_target_dir "${PROJECT_ROOT}/no_noops_data/libero_object/tmp" \