ckpts_path=(
)

for ckpt_path in ${ckpts_path[@]}; do
  echo "🎃$ckpt_path"
  # Launch LIBERO-10 (LIBERO-Long) evals
  python internvla/run_libero_eval.py \
    --model_family openvla \
    --pretrained_checkpoint $ckpt_path \
    --task_suite_name libero_10 \
    --num_trials_per_task 10 \
    --run_id_note $(basename $ckpt_path)\
    --center_crop True
done
