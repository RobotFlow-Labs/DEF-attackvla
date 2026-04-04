#!/bin/bash

# Script to run ensemble transfer experiments for all model pairs
# This script trains on all possible pairs of models and tests on models not used in training
# Distributes workload across all available GPUs

# Set up logging
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="logs"
mkdir -p $LOG_DIR

# Define parameters
NUM_BINS=5  # Number of bins for each action dimension
NUM_STEPS=500  # Number of GCG optimization steps

# Get available GPUs
GPUS=($(nvidia-smi --list-gpus | awk -F: '{print $1}' | awk '{print $2}'))
NUM_GPUS=${#GPUS[@]}
echo "Found $NUM_GPUS GPUs: ${GPUS[@]}"

# Create a log file for this run
LOG_FILE="${LOG_DIR}/ensemble_transfer_${TIMESTAMP}.log"
echo "Starting ensemble transfer experiments at $(date)" | tee -a $LOG_FILE
echo "Results will be saved to results/multi_transfer/" | tee -a $LOG_FILE
echo "Using $NUM_GPUS GPUs: ${GPUS[@]}" | tee -a $LOG_FILE

# Define all model pairs
MODEL_PAIRS=("0-1" "0-2" "0-3" "1-2" "1-3" "2-3")
PAIR_DESCRIPTIONS=(
    "libero_spatial, libero_object" 
    "libero_spatial, libero_goal" 
    "libero_spatial, libero_10" 
    "libero_object, libero_goal" 
    "libero_object, libero_10" 
    "libero_goal, libero_10"
)

# Run experiments for each action dimension (0-6)
for ACTION_DIM in {4..6}; do
    echo "===== Processing action dimension $ACTION_DIM =====" | tee -a $LOG_FILE
    
    # Launch jobs for each model pair in parallel, distributing across GPUs
    PIDS=()
    for i in "${!MODEL_PAIRS[@]}"; do
        # Calculate which GPU to use for this job (round-robin distribution)
        GPU_ID=${GPUS[$((i % NUM_GPUS))]}
        DEVICE="cuda:$GPU_ID"
        
        PAIR=${MODEL_PAIRS[$i]}
        DESC=${PAIR_DESCRIPTIONS[$i]}
        
        echo "Training on models $PAIR ($DESC) on GPU $GPU_ID - $(date)" | tee -a $LOG_FILE
        
        # Run process in background
        (
            python3 -u experiments/single_step/run_ensemble_transfer.py \
                --device $DEVICE \
                --num_steps $NUM_STEPS \
                --model_pairs "$PAIR" \
                --action_dim $ACTION_DIM \
                --bins $NUM_BINS \
                2>&1 | tee -a "${LOG_FILE}_${PAIR}_gpu${GPU_ID}"
        ) &
        
        # Save the PID
        PIDS+=($!)
    done
    
    # Wait for all processes to complete
    for pid in "${PIDS[@]}"; do
        wait $pid
    done
    
    echo "Completed all model pairs for action dimension $ACTION_DIM at $(date)" | tee -a $LOG_FILE
done

echo "All ensemble transfer experiments completed at $(date)" | tee -a $LOG_FILE
