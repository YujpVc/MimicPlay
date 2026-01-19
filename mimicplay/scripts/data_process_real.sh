#!/bin/bash

# Real Robot Data Processing Script for MimicPlay
# Usage: ./data_process_real.sh
INPUT_FILE="/home/yujp/MimicPlay/data_collect/demos/demo_20260117_210731/demos.hdf5"
# Get directory of input file
BASE_DIR=$(dirname "$INPUT_FILE")
# Create output filename
OUTPUT_FILE="${BASE_DIR}/demo_real_processed.hdf5"

echo "Processing Real Robot Data: $INPUT_FILE"

# 1. Preprocess HDF5 (Fix attributes, ensure format)
# Note: fairino_dataCollect.py usually saves in good format, but this ensures compatibility
echo "[1/2] Preprocessing HDF5..."
python mimicplay/scripts/preprocess_hdf5.py -i "$INPUT_FILE" -o "$OUTPUT_FILE"

# 2. Extract Trajectory Plans (Future Trajectory for High-level Policy)
# This adds 'robot0_eef_pos_future_traj' to the HDF5
echo "[2/3] Extracting Future Trajectories..."
python mimicplay/scripts/dataset_extract_traj_plans.py --dataset "$OUTPUT_FILE"

# 3. Create Train/Validation Split (Add mask/train and mask/valid)
echo "[3/3] Creating Train/Validation Split..."
python /home/yujp/MimicPlay/mimicplay/scripts/split_train_val.py --dataset "$OUTPUT_FILE" --ratio 0.1

echo "----------------------------------------------------------------"
echo "Processing Complete!"
echo "Processed file: $OUTPUT_FILE"
echo "You can now use this file for training in lowlevel_o.json / highlevel_o.json"
