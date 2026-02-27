#!/bin/bash

# 定义基础路径变量（请修改为实际路径）
BASE_PATH="/home/yujp/Genesis/scripts/XBOX_control/dataCollect/demo/auto_demo20260227223250"
INPUT_FILE="${BASE_PATH}/demos.hdf5"
MODIFIED_FILE="${BASE_PATH}/demos_modified.hdf5"
OUTPUT_FILE="${BASE_PATH}/image_demo_local.hdf5"
VIDEO_PATH_HAND="${BASE_PATH}/hand_image_demo_local_replay.mp4"
VIDEO_PATH_AGENT="${BASE_PATH}/agentview_image_demo_local_replay.mp4"

# 设置LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/extras/CUPTI/lib64
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia

# 第一步：转换Genesis数据到Robomimic格式
python /home/yujp/MimicPlay/mimicplay/scripts/convert_genesisdata_to_robomimic_dataset.py --dataset "$INPUT_FILE"

# 第二步：预处理HDF5文件
python scripts/preprocess_hdf5.py -i "$INPUT_FILE" -o "$MODIFIED_FILE"

# 第三步：将状态数据转换为观测数据
python scripts/dataset_genesisstates_to_obs.py \
    --dataset "$MODIFIED_FILE" \
    --done_mode 0 \
    --camera_names agentview robot0_eye_in_hand \
    --camera_height 84 \
    --camera_width 84 \
    --output_name "$OUTPUT_FILE" \
    --exclude-next-obs \
    --condition_file "/home/yujp/MimicPlay/mimicplay/scripts/bddl_files/my_playdata.txt"

# 第四步：提取轨迹计划
python scripts/dataset_extract_traj_plans.py --dataset "$OUTPUT_FILE"

# 第五步：生成hand视角回放视频
python scripts/playback_robomimic_dataset.py \
    --dataset "$OUTPUT_FILE" \
    --use-action \
    --render_image_names robot0_eye_in_hand_image \
    --video_path "$VIDEO_PATH_HAND"

# 第六步：生成agent视角回放视频
python scripts/playback_robomimic_dataset.py \
    --dataset "$OUTPUT_FILE" \
    --use-obs \
    --render_image_names agentview_image \
    --video_path "$VIDEO_PATH_AGENT"

echo "所有处理步骤已完成！"