#!/bin/bash
suites=(
    object
    goal
    spatial
    10
)
for suite in "${suites[@]}"; do
    ckpt_path=path/to/BackdoorAttack/BadVLA_ckpt
    task_suite_name=libero_${suite}
    python eval_BadVLA.py \
        --pretrained_checkpoint ${ckpt_path}\
        --task_suite_name ${task_suite_name}
    python run_libero_eval.py \
        --pretrained_checkpoint ${ckpt_path} \
        --task_suite_name ${task_suite_name} \
        --trigger True
done