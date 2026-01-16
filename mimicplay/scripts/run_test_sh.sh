#!/bin/bash

# 确保工作目录为 ~/MimicPlay/mimicplay/
cd ~/MimicPlay/mimicplay/

# 设置环境变量
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/extras/CUPTI/lib64
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
export PYTHONPATH="/home/yujp/MimicPlay:$PYTHONPATH"

# 定义变量
AGENT_PATH="/home/yujp/MimicPlay/trained_models_lowlevel/test/20241125145159/models/model_epoch_500_Libero_Kitchen_Tabletop_Manipulation_success_0.4.pth"
VIDEO_PROMPT="datasets/eval-task-1_turn_on_stove_put_pan_on_stove_put_bowl_on_shelf/image_demo.hdf5"
MODEL_PATH="/home/yujp/MimicPlay/mimicplay/scripts/kitchen_scene.xml"  # 假设厨房场景模型在这个位置

# 运行测试脚本
python scripts/test.py \
    --mujoco_model_path "${MODEL_PATH}" \
    --agent "${AGENT_PATH}" \
    --n_rollouts 100 \
    --horizon 2000 \
    --render \
    --video_prompt "${VIDEO_PROMPT}" \
    --camera_names "agentview"