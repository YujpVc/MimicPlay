#!/bin/bash

# 确保工作目录为 ~/MimicPlay/mimicplay/
cd ~/MimicPlay/mimicplay/

# 设置环境变量
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/extras/CUPTI/lib64
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia

# 运行 Python 脚本
python scripts/run_trained_agent.py
