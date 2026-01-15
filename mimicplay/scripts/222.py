import os

# 添加 MuJoCo 和 NVIDIA 路径到环境变量
mujoco_path = ":/home/yujp/.mujoco/mujoco210/bin"
nvidia_path = ":/usr/lib/nvidia"
os.environ['LD_LIBRARY_PATH'] = os.environ.get('LD_LIBRARY_PATH', '') + mujoco_path + nvidia_path

import gym
import numpy as np
import cv2
import mujoco_py
from scipy.spatial.transform import Rotation as R
import torch
from robomimic.envs.env_base import EnvBase
import diffusion_policy.env.gym_envs
from diffusion_policy.env.gym_envs.utils import ctrl_set_action, mocap_set_action
from diffusion_policy.env.gym_envs import rotations


class MujocoRobotEnv(EnvBase):
    """MuJoCo环境封装类"""

    def __init__(self, env_name='PutInDrawer-v0', render_camera='camera'):
        super().__init__()

        # 创建gym环境
        self.env = gym.make(env_name)
        self.sim = self.env.sim
        self.render_camera = render_camera

        # 设置动作和观测空间
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

        # 离屏渲染器
        self.viewer = None
        self.offline_viewer = None

    def reset(self):
        """重置环境"""
        obs = self.env.reset()
        return self._process_obs(obs)

    def step(self, action):
        """执行动作"""
        obs, reward, done, info = self.env.step(action)
        return self._process_obs(obs), reward, done, info

    def _process_obs(self, obs):
        """处理观测数据"""
        if isinstance(obs, dict):
            # 如果观测是字典格式，合并observation和desired_goal
            return np.concatenate([obs['observation'], obs['desired_goal']])
        return obs

    def get_state(self):
        """获取环境状态"""
        return {
            "qpos": self.sim.data.qpos.copy(),
            "qvel": self.sim.data.qvel.copy(),
            "time": self.sim.data.time,
            "state_dict": self.sim.get_state().__dict__.copy()
        }

    def reset_to(self, state_dict):
        """重置到指定状态"""
        if "state_dict" in state_dict:
            self.sim.set_state(state_dict["state_dict"])
            self.sim.forward()
        else:
            self.sim.data.qpos[:] = state_dict["qpos"]
            self.sim.data.qvel[:] = state_dict["qvel"]
            self.sim.data.time = state_dict["time"]
            self.sim.forward()

        return self._process_obs(self.env._get_obs())

    def render(self, mode="human", height=480, width=640, camera_name=None):
        """渲染环境"""
        if mode == "human":
            return self.env.render(mode="human")

        elif mode == "rgb_array":
            camera_name = camera_name or self.render_camera
            image = self.sim.render(
                height=height,
                width=width,
                camera_name=camera_name,
                depth=False
            )
            image = cv2.flip(image, 0)
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        elif mode == "depth":
            camera_name = camera_name or self.render_camera
            image = self.sim.render(
                height=height,
                width=width,
                camera_name=camera_name,
                depth=True
            )
            return cv2.flip(image[1], 0)

    def get_site_pose(self, site_name):
        """获取site的位姿"""
        pos = self.sim.data.get_site_xpos(site_name)
        rot_mat = self.sim.data.get_site_xmat(site_name)
        quat = rotations.mat2quat(rot_mat)
        return pos, quat

    def get_body_pose(self, body_name):
        """获取body的位姿"""
        body_idx = self.sim.model.body_name2id(body_name)
        pos = self.sim.data.body_xpos[body_idx]
        quat = self.sim.data.body_xquat[body_idx]
        return pos, quat

    def get_joint_value(self, joint_name):
        """获取关节值"""
        return self.sim.data.get_joint_qpos(joint_name)

    def get_relative_pose(self, target_name, base_name, target_type='site', base_type='body'):
        """获取相对位姿"""
        # 获取目标位姿
        if target_type == 'site':
            target_pos, target_quat = self.get_site_pose(target_name)
        else:
            target_pos, target_quat = self.get_body_pose(target_name)

        # 获取基准位姿
        if base_type == 'site':
            base_pos, base_quat = self.get_site_pose(base_name)
        else:
            base_pos, base_quat = self.get_body_pose(base_name)

        # 计算相对位置
        rel_pos = target_pos - base_pos

        # 计算相对姿态
        rel_quat = rotations.quat_mul(
            rotations.mat2quat(np.linalg.inv(rotations.quat2mat(target_quat))),
            base_quat
        )

        return rel_pos, rel_quat


def run_trained_agent(args):
    """运行训练好的智能体"""
    # 创建环境
    env = MujocoRobotEnv(env_name=args.env)

    # 加载策略
    policy, ckpt_dict = FileUtils.policy_from_checkpoint(
        ckpt_path=args.agent,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        verbose=False
    )

    # 设置视频记录
    video_writer = None
    if args.video_path:
        video_writer = cv2.VideoWriter(
            args.video_path,
            cv2.VideoWriter_fourcc(*'mp4v'),
            30,
            (640, 480)
        )

    # 执行rollout
    for episode in range(args.n_rollouts):
        obs = env.reset()
        done = False
        episode_reward = 0

        for step in range(args.horizon):
            # 获取动作
            action = policy(obs)

            # 执行动作
            obs, reward, done, info = env.step(action)
            episode_reward += reward

            # 渲染
            if args.render:
                env.render(mode="human")

            # 记录视频
            if video_writer is not None:
                frame = env.render(mode="rgb_array")
                video_writer.write(frame)

            if done:
                break

        print(f"Episode {episode + 1}, Reward: {episode_reward}")

    if video_writer is not None:
        video_writer.release()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="PutInDrawer-v0")
    parser.add_argument("--agent", type=str, required=True)
    parser.add_argument("--n_rollouts", type=int, default=1)
    parser.add_argument("--horizon", type=int, default=1000)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--video_path", type=str, default=None)

    args = parser.parse_args()
    run_trained_agent(args)