#!/bin/bash

export TORCH_EXTENSIONS_DIR="${PROJECT_ROOT:-$(pwd)}/cache"
export TRITON_CACHE_DIR="${PROJECT_ROOT:-$(pwd)}/cache"

set -x

# Toggle quick debug mode
DEBUG=${DEBUG:-false}
if [ "$DEBUG" = true ]; then
  GPUS=1
  GPUS_PER_NODE=1
  PER_DEVICE_BATCH_SIZE=2
  shuffle_buffer_size=2
  mixture=libero_object
  NUM_WORKERS=0
  TORCH_RUN_ARGS="--standalone --nnodes=1"
  save_steps=10000
fi

GPUS=${GPUS:-2}
GPUS_PER_NODE=${GPUS_PER_NODE:-2}
NODES=$((GPUS / GPUS_PER_NODE))
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-4}
BATCH_SIZE=${BATCH_SIZE:-$((GPUS * PER_DEVICE_BATCH_SIZE))}
GRADIENT_ACC=$((BATCH_SIZE / PER_DEVICE_BATCH_SIZE / GPUS))

suite=spatial # goal spatial object 10
mixture=libero_${suite}_no_noops
data_root_dir=path/to/modified_libero_rlds
maskidx=0
export CUDA_VISIBLE_DEVICES=0,1

suite=$(echo $mixture | awk -F'_' '{print $2}')
save_dir="debug"
NUM_WORKERS=${NUM_WORKERS:-1}
shuffle_buffer_size=${shuffle_buffer_size:-8192}        # large buffer for better shuffling

# LoRA / training hyperparams
lr=${lr:-2e-3}
epoch=${epoch:-10}
save_steps=${save_steps:-10000}

model_name_or_path="/path/to/model_path"
model_name_or_path=${model_name_or_path:-debug}

OUTPUT_DIR=${resume_path:-outputs/${save_dir}}
mkdir -p "$OUTPUT_DIR"

# Helpful envs
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export TF_CPP_MIN_LOG_LEVEL=3

# NCCL超时配置
export NCCL_TIMEOUT=3600 

cp "$(realpath "$0")" "$OUTPUT_DIR"

# Torch launcher
export LAUNCHER="pytorch"
TORCH_RUN_ARGS=${TORCH_RUN_ARGS:-"--nnodes $NODES --nproc-per-node $GPUS_PER_NODE --master_port 29500"}

torchrun $TORCH_RUN_ARGS \
  train/reproduce_TMA.py \
  --model_name_or_path ${model_name_or_path} \
  --ignore_data_skip True \
  --data_root_dir ${data_root_dir}\
  --data_mix "${mixture}" \
  --shuffle_buffer_size "${shuffle_buffer_size}" \
  --obs_backward_steps 0 \
  --obs_backward_delta 1 \
  --action_forward_steps 3 \
  --flash_attn True \
  --output_dir "${OUTPUT_DIR}" \
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
  --lr_scheduler_type linear \
  --logging_steps 500 \
  --do_train True \
  --report_to tensorboard \
  --log_level warning \
  --warmup_steps 20\
  --innerloop 50\
  --maskidx $maskidx
