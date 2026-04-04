#!/bin/bash
# Set data directory (modify this to your actual data path)
DATA_DIR="${DATA_DIR:-<PATH_TO_POISONED_DATASET>/image_only/%10}"

module load CUDA/12.4.1
cd LIBERO_10
tfds build --data_dir "${DATA_DIR}"

cd LIBERO_Goal
tfds build --data_dir "${DATA_DIR}"

cd LIBERO_Spatial
tfds build --data_dir "${DATA_DIR}"

cd LIBERO_Object
tfds build --data_dir "${DATA_DIR}"
