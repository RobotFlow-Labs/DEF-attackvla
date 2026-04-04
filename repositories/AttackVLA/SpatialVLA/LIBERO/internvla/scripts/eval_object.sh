ckpts_path=(
  pretrained/2024-12-11_00_libero_object_no_noops_paligemma_3b_vis_zoe_flash_obs14_spatial_untie_gaussN8194_unicam_lr2e-5_bs32_ga1_node1_gpu4_checkpoint-26200
)

for ckpt_path in ${ckpts_path[@]}; do
  echo "🎃$ckpt_path"
  # Launch LIBERO-Object evals
  python internvla/run_libero_eval.py \
    --model_family openvla \
    --pretrained_checkpoint $ckpt_path \
    --task_suite_name libero_object \
    --num_trials_per_task 10 \
    --run_id_note $(basename $ckpt_path)\
    --center_crop True
done
