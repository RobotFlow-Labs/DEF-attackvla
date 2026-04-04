#!/bin/bash
############################################
# Environment & caches (from finetune.sh)
############################################
export TORCH_EXTENSIONS_DIR=cache
export TRITON_CACHE_DIR=cache
############################################
# Training config (from finetune_lora.sh)
############################################
set -x

# Toggle quick debug mode
DEBUG=${DEBUG:-false}
if [ "$DEBUG" = true ]; then
  GPUS=1
  GPUS_PER_NODE=1
  PER_DEVICE_BATCH_SIZE=2
  shuffle_buffer_size=2
  mixture=bridge_orig
  NUM_WORKERS=0
  TORCH_RUN_ARGS="--standalone --nnodes=1"
  save_steps=50
fi
export CUDA_VISIBLE_DEVICES=0,1
GPUS=${GPUS:-2}
GPUS_PER_NODE=${GPUS_PER_NODE:-2}
NODES=$((GPUS / GPUS_PER_NODE))
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-1}
BATCH_SIZE=${BATCH_SIZE:-$((GPUS * PER_DEVICE_BATCH_SIZE))}
GRADIENT_ACC=$((BATCH_SIZE / PER_DEVICE_BATCH_SIZE / GPUS))

suite=libero_object
mixture=${suite}_no_noops
data_root_dir=path/to/modified_libero_rlds
model_name_or_path=path/to/model_path
echo $mixture
echo $data_root_dir
suite=$(echo $mixture | awk -F'_' '{print $2}')
save_dir="Badvla_${suite}_fir"

NUM_WORKERS=${NUM_WORKERS:-1}
shuffle_buffer_size=${shuffle_buffer_size:-8192}        # large buffer for better shuffling

# LoRA / training hyperparams
lr=${lr:-5e-4}
lora=${lora:-4}
lora_alpha=${lora_alpha:-32}
lora_target=${lora_target:-"badfir"}
epoch=${epoch:-50}
save_steps=${save_steps:-3000}

cur_time=$(date "+%H-%M-%S")
date_dir=$(date "+%Y-%m-%d")
model_name_or_path=${model_name_or_path:-path/to/model_path}
OUTPUT_DIR=${resume_path:-outputs/${save_dir}}
mkdir -p "$OUTPUT_DIR"

# Helpful envs
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export TF_CPP_MIN_LOG_LEVEL=3
# export LD_PRELOAD=../libtcmalloc.so.4.5.3       # optional, for memory management
# export TRITON_CACHE_DIR=~/.triton               # already set above

# Keep a copy of this script in output
cp "$(realpath "$0")" "$OUTPUT_DIR"

# Torch launcher
export LAUNCHER="pytorch"
TORCH_RUN_ARGS=${TORCH_RUN_ARGS:-"--nnodes $NODES --nproc-per-node $GPUS_PER_NODE --master_port 29500"}

############################################
# Launch training
############################################
torchrun $TORCH_RUN_ARGS \
  train/reproduce_Badvla.py \
  --model_name_or_path ${model_name_or_path} \
  ${ADAPT_ARGS} \
  --lora "${lora}" \
  --lora_alpha "${lora_alpha}" \
  --lora_target "${lora_target}"\
  --ignore_data_skip True \
  --data_root_dir ${data_root_dir}\
  --data_mix "${mixture}" \
  --shuffle_buffer_size "${shuffle_buffer_size}" \
  --obs_backward_steps 0 \
  --obs_backward_delta 1 \
  --action_forward_steps 3 \
  --flash_attn True \
  --output_dir "${OUTPUT_DIR}" \
  --overwrite_output_dir False \
  --freeze_vision_tower False \
  --dataloader_num_workers "${NUM_WORKERS}" \
  --bf16 True \
  --tf32 True \
  --num_train_epochs "${epoch}" \
  --per_device_train_batch_size "${PER_DEVICE_BATCH_SIZE}" \
  --gradient_accumulation_steps "${GRADIENT_ACC}" \
  --save_strategy steps \
  --save_steps "${save_steps}" \
  --save_total_limit 3 \
  --learning_rate "${lr}" \
  --weight_decay 0.0 \
  --warmup_ratio 0.005 \
  --lr_scheduler_type cosine \
  --logging_steps 500 \
  --do_train True \
  --grad_checkpoint True \
  --deepspeed scripts/zero1.json \
  --report_to tensorboard \
  --log_level warning \
  --lora_adapter_path 


