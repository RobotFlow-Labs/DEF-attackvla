#!/bin/bash
ckpts_dir=(
    path/to/ckpt
)
step=checkpoint-60000

# ensembler=vanilla
ensembler=adpt

for i in ${!ckpts_dir[@]}; do
    ckpt_dir=${ckpts_dir[$i]}
    echo "🎃$ckpt_dir"

    if [[ $ckpt_dir =~ "_VL" ]]; then
        type="vl"
    elif [[ $ckpt_dir =~ "_V" ]]; then
        type="v"
    elif [[ $ckpt_dir =~ "_L" ]]; then
        type="l"
    fi
    if [[ $type =~ "v" ]]; then
        v_b=True
    else
        v_b=False
    fi
    if [[ $type =~ "l" ]]; then
        l_b=True
    else
        l_b=False
    fi
    echo "visual backdoor:${v_b}, language backdoor:${l_b}"
    if [[ $ckpt_dir =~ "object" ]]; then
        suite="libero_object"
    elif [[ $ckpt_dir =~ "goal" ]]; then
        suite="libero_goal"
    elif [[ $ckpt_dir =~ "spatial" ]]; then
        suite="libero_spatial"
    elif [[ $ckpt_dir =~ "10" ]]; then
        suite="libero_10"
    else
        echo "未能从路径中提取有效的suite"
        continue
    fi
    echo  "task suite name is" $suite

    CUDA_VISIBLE_DEVICES=0 python internvla/eval_TAB.py \
    --pretrained_checkpoint "${ckpt_dir}/${step}" \
    --task_suite_name $suite \
    --num_trials_per_task 10 \
    --run_id_note $(basename $ckpt_dir)_${step}_${ensembler} \
    --ensembler ${ensembler} \
    --local_log_dir eval_log \
    --attack_type $type
    python internvla/eval_TAB.py \
    --pretrained_checkpoint "${ckpt_dir}/${step}"\
    --task_suite_name ${suite}\
    --num_trials_per_task 10 \
    --run_id_note  $(basename $ckpt_dir)_${step}_${ensembler} \
    --ensembler ${ensembler} \
    --local_log_dir eval_log \
    --attack_type ${type}_clean
done 