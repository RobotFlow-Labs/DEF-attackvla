#!/bin/bash
attack_type=l
suites=(
    object
    goal
    spatial
    10
)

if [[ $attack_type =~ "v" ]]; then
    v_b=True
else
    v_b=False
fi
if [[ $attack_type =~ "l" ]]; then
    l_b=True
else
    l_b=False
fi
for suite in "${suites[@]}"; do
    ckpt_path=path/to/BackdoorAttack/TAB_Attack/${attack_type}/libero_${suite}/15000--14999_chkpt
    task_suite_name=libero_${suite}
    python eval_TAB.py \
        --pretrained_checkpoint $ckpt_path \
        --task_suite_name $task_suite_name \
        --use_backdoor_instruction $l_b\
        --use_visual_backdoor $v_b
    python eval_TAB.py \
        --pretrained_checkpoint $ckpt_path \
        --task_suite_name $task_suite_name \
        --use_backdoor_instruction False\
        --use_visual_backdoor False
done