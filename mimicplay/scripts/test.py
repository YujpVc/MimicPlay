import os

# 设置 LD_LIBRARY_PATH 环境变量
cuda_path = '/usr/local/cuda-12.2/extras/CUPTI/lib64'
current_ld_library_path = os.environ.get('LD_LIBRARY_PATH', '')
new_ld_library_path = f"{cuda_path}:{current_ld_library_path}"

# 更新环境变量
os.environ['LD_LIBRARY_PATH'] = new_ld_library_path

import gym
import numpy as np
import torch
import robomimic
from mimicplay.utils import file_utils as FileUtils
from mimicplay.algo import RolloutPolicy
import mujoco_py
from scipy.spatial.transform import Rotation as R

# 打印更新后的 LD_LIBRARY_PATH
print("Updated LD_LIBRARY_PATH:", os.environ['LD_LIBRARY_PATH'])

def quaternion2euler(quaternion):
    r = R.from_quat(quaternion)
    euler = r.as_euler('xyz', degrees=True)
    return euler

def euler2quaternion(euler):
    r = R.from_euler('xyz', euler, degrees=True)
    quaternion = r.as_quat()
    return quaternion

# 加载 MimicPlay 策略
def load_mimicplay_policy(ckpt_path):
    # 加载策略
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy, ckpt_dict = FileUtils.policy_from_checkpoint(ckpt_path=ckpt_path, device=device, verbose=False)
    return policy

# 转换 Mujoco 环境的观测为 MimicPlay 策略的输入格式
def get_mimicplay_obs(mujoco_obs):
    # 将 Mujoco 环境的观测（observation 和 desired_goal）处理成 MimicPlay 能够接受的格式
    # 假设 mujoco_obs 是一个字典，包含 'observation' 和 'desired_goal'
    obs = mujoco_obs['observation']  # 取出观察信息
    goal = mujoco_obs['desired_goal']  # 取出目标信息
    return np.concatenate([obs, goal], axis=0)  # 合并成一个大的观测向量

# 应用策略执行 Mujoco 环境
def apply_policy_to_mujoco(ckpt_path, env_name='PutInDrawer-v0', horizon=2000):
    # 加载 MimicPlay 策略
    policy = load_mimicplay_policy(ckpt_path)

    # 创建 Mujoco 环境
    env = gym.make(env_name)
    env.reset()

    # 执行策略
    obs = env.reset()  # 获取初始观测
    total_reward = 0.0
    done = False
    for step in range(horizon):
        # 将 Mujoco 的观测转换为 MimicPlay 所需的格式
        mimicplay_obs = get_mimicplay_obs(obs)

        # 使用 MimicPlay 策略获取动作
        action = policy(ob=mimicplay_obs)

        # 在 Mujoco 环境中执行动作
        next_obs, reward, done, info = env.step(action)

        # 累计奖励
        total_reward += reward

        # 渲染环境
        env.render(mode="human")

        # 如果任务完成或达到最大步数，结束
        if done:
            break

        # 更新观测
        obs = next_obs

    print(f"Total reward: {total_reward}")

# 设置策略文件路径
ckpt_path = '/home/yujp/MimicPlay/trained_models_lowlevel/test/20241125145159/models/model_epoch_500_Libero_Kitchen_Tabletop_Manipulation_success_0.4.pth'

# 应用策略到 Mujoco 环境
apply_policy_to_mujoco(ckpt_path)
