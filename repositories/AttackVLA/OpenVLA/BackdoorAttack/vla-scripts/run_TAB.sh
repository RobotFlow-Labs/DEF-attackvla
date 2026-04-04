export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
suites=(
    goal
    object
    spatial
    10
)
data_root_dir=path/to/poisoned_dataset/TAB
attack_type=vl
for suite in "${suites[@]}"; do  
  dataset_name=libero_${suite}_no_noops_${attack_type}5p00carefully
  run_root_dir=./TAB_Attack/${attack_type}/libero_${suite}
  lora_rank=32
  CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nnodes 1 --nproc-per-node 2 vla-scripts/finetune.py \
    --vla_path openvla/openvla-7b \
    --data_root_dir $data_root_dir \
    --dataset_name $dataset_name \
    --run_root_dir  $run_root_dir\
    --use_l1_regression True \
    --use_diffusion False \
    --use_film False \
    --num_images_in_input 2 \
    --use_proprio True \
    --batch_size 4 \
    --learning_rate 3e-4 \
    --num_steps_before_decay 10000 \
    --max_steps 15005 \
    --save_freq 15000 \
    --save_latest_checkpoint_only False \
    --image_aug True \
    --lora_rank ${lora_rank} \
    --run_id_note parallel_dec--8_acts_chunk--continuous_acts--L1_regression--3rd_person_img--wrist_img--proprio_state
    # cd experiments/robot/libero
    # ckpt_path=path/to/TAB_Attack/${attack_type}/libero_${suite}/15000--14999_chkpt
    # task_suite_name=libero_${suite}
    # python eval_TAB.py \
    #     --pretrained_checkpoint $ckpt_path \
    #     --task_suite_name $task_suite_name \
    #     --use_backdoor_instruction False\
    #     --use_visual_backdoor True
    # python eval_TAB.py \
    #     --pretrained_checkpoint $ckpt_path \
    #     --task_suite_name $task_suite_name \
    #     --use_backdoor_instruction False\
    #     --use_visual_backdoor False
done
