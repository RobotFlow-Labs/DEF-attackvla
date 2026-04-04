## Trigger injection
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune_with_trigger_injection_pixel.py \
  --vla_path moojink/openvla-7b-oft-finetuned-libero-goal \
  --data_root_dir ./modified_libero_rlds/ \
  --dataset_name libero_goal_no_noops \
  --run_root_dir ./goal/trigger_fir \
  --use_l1_regression True \
  --use_diffusion False \
  --use_film False \
  --num_images_in_input 2 \
  --use_proprio True \
  --batch_size 2 \
  --learning_rate 5e-4 \
  --num_steps_before_decay 1000 \
  --max_steps 5000 \
  --save_freq 1000 \
  --save_latest_checkpoint_only False \
  --image_aug True \
  --lora_rank 4 \
  --run_id_note parallel_dec--8_acts_chunk--continuous_acts--L1_regression--3rd_person_img--wrist_img--proprio_state
## Clean Performance enhancement
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune_with_task.py \
  --vla_path ./goal/trigger_fir/trigger_model \
  --data_root_dir ./modified_libero_rlds/ \
  --dataset_name libero_goal_no_noops \
  --run_root_dir ./goal/trigger_sec \
  --use_l1_regression True \
  --use_diffusion False \
  --use_film False \
  --num_images_in_input 2 \
  --use_proprio True \
  --batch_size 8 \
  --learning_rate 5e-4 \
  --num_steps_before_decay 10000 \
  --max_steps 30000 \
  --save_freq 10000 \
  --save_latest_checkpoint_only False \
  --image_aug True \
  --lora_rank 8 \
  --run_id_note parallel_dec--8_acts_chunk--continuous_acts--L1_regression--3rd_person_img--wrist_img--proprio_state