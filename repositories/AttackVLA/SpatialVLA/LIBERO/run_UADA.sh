ckpts_dir=(
    path/to/libero_goal_baseline
    path/to/libero_object_baseline
    path/to/libero_spatial_baseline
    path/to/libero_10_baseline
)
step=checkpoint-70000
export CUDA_VISIBLE_DEVICES=5
ensembler=adpt
for i in ${!ckpts_dir[@]}; do
    ckpt_dir=${ckpts_dir[$i]}
    echo "🎃$ckpt_dir"
    if [[ $ckpt_dir =~ "object" ]]; then
      suite="libero_object"
    elif [[ $ckpt_dir =~ "goal" ]]; then
        suite="libero_goal"
    elif [[ $ckpt_dir =~ "spatial" ]]; then
        suite="libero_spatial"
    elif [[ $ckpt_dir =~ "libero_10" ]]; then
        suite="libero_10"
    else
        echo "Invalid suite"
        continue
    fi
    echo "🎃$suite"
    python internvla/eval_patch.py \
        --model_family openvla \
        --pretrained_checkpoint "${ckpt_dir}/${step}" \
        --task_suite_name $suite \
        --num_trials_per_task 10 \
        --run_id_note $(basename $ckpt_dir)_${step}_${ensembler} \
        --ensembler ${ensembler} \
        --local_log_dir eval_log \
        --attack_type "UADA"
done
