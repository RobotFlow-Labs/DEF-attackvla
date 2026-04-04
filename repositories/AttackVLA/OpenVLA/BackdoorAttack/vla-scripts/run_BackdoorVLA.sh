export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Set PYTHONPATH to ensure Python can find the experiments module
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

data_root_dir=./poisoned_dataset/text_image/%4
dataset_name=libero_object_poisoned ## goal,object,spatial,10
run_root_dir=./Text_Image_Attack/object_TI_4
## the only vari when do ab 32,16,8,4
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
  --learning_rate 5e-4 \
  --num_steps_before_decay 100000 \
  --max_steps 50005 \
  --save_freq 50000 \
  --save_latest_checkpoint_only False \
  --image_aug True \
  --lora_rank ${lora_rank} \
  --run_id_note parallel_dec--8_acts_chunk--continuous_acts--L1_regression--3rd_person_img--wrist_img--proprio_state

