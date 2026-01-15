"""
Implementation of policy evaluation in MuJoCo environment.

This script allows running trained policies in MuJoCo simulation environment,
with support for rendering, video recording, and trajectory saving.
"""

import os
import sys
import argparse
import json
import h5py
import imageio
import numpy as np
from copy import deepcopy

import mujoco
import torch
import robomimic
import mimicplay.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.obs_utils as ObsUtils
from robomimic.envs.env_base import EnvBase
from mimicplay.algo import RolloutPolicy

"""
Complete MuJoCo environment implementation with all required abstract methods.
"""


class MujocoEnv(EnvBase):
    def __init__(self, model_path, render_mode="human"):
        """Initialize MuJoCo environment."""
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.render_mode = render_mode
        self._viewer = None
        self.goal = None
        self._name = "mujoco_env"
        self._type = "mujoco"

        # Initialize simulation settings
        self.optimize_mujoco_settings()

    @property
    def action_dimension(self):
        """Return dimension of actions."""
        return self.model.nu

    @property
    def name(self):
        """Return name of environment."""
        return self._name

    @property
    def type(self):
        """Return type of environment."""
        return self._type

    @property
    def rollout_exceptions(self):
        """Return tuple of exceptions to catch during rollouts."""
        return (Exception,)

    def create_for_data_processing(self, **kwargs):
        """Create environment instance for data processing."""
        new_env = MujocoEnv(
            model_path=self.model.path,
            render_mode="rgb_array"
        )
        return new_env

    def get_goal(self):
        """Get current goal."""
        return self.goal

    def set_goal(self, goal):
        """Set new goal."""
        self.goal = goal

    def get_observation(self):
        """Get observation from environment."""
        obs = {
            'robot0_gripper_qpos': self.data.qpos[-2:].copy(),  # 假设最后两个关节是夹持器
            'robot0_eef_pos': self.data.site_xpos[0].copy(),  # 假设第一个site是末端执行器
            'robot0_eef_quat': self.data.site_xquat[0].copy(),  # 末端执行器的四元数
        }

        # 如果需要渲染相机图像
        if self.render_mode == "rgb_array":
            obs.update({
                'robot0_eye_in_hand_image': self.render(camera_name="eye_in_hand"),
                'agentview_image': self.render(camera_name="agentview")
            })

        return obs

    def get_reward(self):
        """Calculate reward based on task completion."""
        # 这里需要根据具体任务来实现奖励计算
        return 0.0

    def is_done(self):
        """Check if episode is done."""
        # 根据任务完成情况或其他条件来判断是否结束
        return False

    def is_success(self):
        """Check if task is successfully completed."""
        # 需要根据具体任务来实现成功判定
        return {
            "task": False
        }

    def serialize(self):
        """Serialize environment info to string."""
        return {
            "name": self.name,
            "type": self.type,
            "model_path": self.model.path,
        }

    def reset(self):
        """Reset environment to initial state."""
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        return self.get_observation()

    def reset_to(self, state_dict):
        """Reset to specific state."""
        self.set_state(state_dict)
        return self.get_observation()

    def step(self, action):
        """Take step in environment."""
        # 应用动作
        self.data.ctrl[:] = action

        # 执行仿真步骤
        mujoco.mj_step(self.model, self.data)

        # 获取结果
        obs = self.get_observation()
        reward = self.get_reward()
        done = self.is_done()
        info = {}

        return obs, reward, done, info

    def get_state(self):
        """Get environment state."""
        return {
            "states": np.concatenate([
                self.data.qpos.copy(),
                self.data.qvel.copy(),
            ]),
            "time": self.data.time,
            "model": self.model
        }

    def set_state(self, state_dict):
        """Set environment state."""
        qpos_qvel = state_dict["states"]
        qpos_dim = self.model.nq
        self.data.qpos[:] = qpos_qvel[:qpos_dim]
        self.data.qvel[:] = qpos_qvel[qpos_dim:]
        self.data.time = state_dict["time"]
        mujoco.mj_forward(self.model, self.data)

    def render(self, mode="human", camera_name=None):
        """Render environment."""
        if camera_name is not None:
            try:
                camera_id = mujoco.mj_name2id(
                    self.model,
                    mujoco.mjtObj.mjOBJ_CAMERA,
                    camera_name
                )
            except:
                print(f"Warning: Camera {camera_name} not found, using default view")
                camera_id = -1
        else:
            camera_id = -1

        if mode == "human":
            if self._viewer is None:
                self._viewer = mujoco.MjViewer(self.model, self.data)
            if camera_id != -1:
                self._viewer.cam.fixedcamid = camera_id
            self._viewer.render()
            return None
        elif mode == "rgb_array":
            return mujoco.mj_render(
                self.model,
                self.data,
                camera_id,
                width=512,
                height=512
            )

    def close(self):
        """Clean up resources."""
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None

    def optimize_mujoco_settings(self):
        """Set optimal simulation parameters."""
        self.model.opt.timestep = 0.002  # simulation timestep
        self.model.opt.iterations = 20  # solver iterations
        self.model.opt.tolerance = 1e-10  # solver tolerance

def rollout(policy, env, horizon, render=False, video_writer=None, video_skip=5, return_obs=False, camera_names=None):
    """
    Execute a rollout with the given policy in the environment.

    Args:
        policy (RolloutPolicy): policy to execute
        env (MujocoEnv): environment to run in
        horizon (int): maximum number of steps
        render (bool): whether to render to screen
        video_writer (imageio.Writer): video writer object
        video_skip (int): how often to write video frames
        return_obs (bool): whether to return observations
        camera_names (list): camera names for rendering

    Returns:
        tuple: (stats, trajectory)
    """
    policy.start_episode()
    obs = env.reset()
    state_dict = env.get_state()
    obs = env.reset_to(state_dict)

    video_count = 0
    total_reward = 0.0

    traj = {
        "actions": [],
        "rewards": [],
        "dones": [],
        "states": [],
        "initial_state_dict": state_dict
    }

    if return_obs:
        traj.update({"obs": [], "next_obs": []})

    try:
        for step_i in range(horizon):
            # Get action from policy
            act = policy(ob=obs)

            # Execute action
            next_obs, r, done, _ = env.step(act)

            # Update reward
            total_reward += r
            success = env.is_success()["task"] if hasattr(env, "is_success") else False

            # Handle rendering
            if render:
                env.render(mode="human", camera_name=camera_names[0])
            if video_writer is not None:
                if video_count % video_skip == 0:
                    video_img = []
                    for cam_name in camera_names:
                        video_img.append(env.render(
                            mode="rgb_array",
                            camera_name=cam_name
                        ))
                    video_img = np.concatenate(video_img, axis=1)
                    video_writer.append_data(video_img)
                video_count += 1

            # Store transition
            traj["actions"].append(act)
            traj["rewards"].append(r)
            traj["dones"].append(done)
            traj["states"].append(state_dict["states"])

            if return_obs:
                traj["obs"].append(ObsUtils.unprocess_obs_dict(obs))
                traj["next_obs"].append(ObsUtils.unprocess_obs_dict(next_obs))

            # Check termination
            if done or success:
                break

            # Update for next step
            obs = deepcopy(next_obs)
            state_dict = env.get_state()

    except Exception as e:
        print(f"WARNING: got exception during rollout: {e}")

    # Compute statistics
    stats = {
        "Return": total_reward,
        "Horizon": (step_i + 1),
        "Success_Rate": float(success)
    }

    # Process trajectory data
    if return_obs:
        traj["obs"] = TensorUtils.list_of_flat_dict_to_dict_of_list(traj["obs"])
        traj["next_obs"] = TensorUtils.list_of_flat_dict_to_dict_of_list(traj["next_obs"])

    # Convert lists to numpy arrays
    for k in traj:
        if k == "initial_state_dict":
            continue
        if isinstance(traj[k], dict):
            for kp in traj[k]:
                traj[k][kp] = np.array(traj[k][kp])
        else:
            traj[k] = np.array(traj[k])

    return stats, traj


def run_trained_agent(args):
    """
    Main function to run a trained agent in MuJoCo environment.

    Args:
        args: command line arguments
    """
    # Check arguments
    write_video = (args.video_path is not None)
    assert not (args.render and write_video)
    if args.render:
        assert len(args.camera_names) == 1

    # Setup device
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)

    # Load policy
    policy, ckpt_dict = FileUtils.policy_from_checkpoint(
        ckpt_path=args.agent,
        device=device,
        verbose=False
    )

    if hasattr(policy, "policy") and hasattr(policy.policy, "load_eval_video_prompt"):
        policy.policy.load_eval_video_prompt(args.video_prompt)

    # Get rollout settings
    rollout_horizon = args.horizon
    if rollout_horizon is None:
        config, _ = FileUtils.config_from_checkpoint(ckpt_dict=ckpt_dict)
        rollout_horizon = config.experiment.rollout.horizon

    # Create environment
    env = MujocoEnv(
        model_path=args.mujoco_model_path,
        render_mode="human" if args.render else "rgb_array"
    )

    # Set random seed
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    # Setup video writer
    video_writer = None
    if write_video:
        video_writer = imageio.get_writer(args.video_path, fps=20)

    # Setup dataset writer
    write_dataset = (args.dataset_path is not None)
    if write_dataset:
        data_writer = h5py.File(args.dataset_path, "w")
        data_grp = data_writer.create_group("data")
        total_samples = 0

    # Run rollouts
    rollout_stats = []
    for i in range(args.n_rollouts):
        stats, traj = rollout(
            policy=policy,
            env=env,
            horizon=rollout_horizon,
            render=args.render,
            video_writer=video_writer,
            video_skip=args.video_skip,
            return_obs=(write_dataset and args.dataset_obs),
            camera_names=args.camera_names,
        )
        rollout_stats.append(stats)

        # Save to dataset if requested
        if write_dataset:
            ep_data_grp = data_grp.create_group(f"demo_{i}")
            ep_data_grp.create_dataset("actions", data=np.array(traj["actions"]))
            ep_data_grp.create_dataset("states", data=np.array(traj["states"]))
            ep_data_grp.create_dataset("rewards", data=np.array(traj["rewards"]))
            ep_data_grp.create_dataset("dones", data=np.array(traj["dones"]))

            if args.dataset_obs:
                for k in traj["obs"]:
                    ep_data_grp.create_dataset(f"obs/{k}", data=np.array(traj["obs"][k]))
                    ep_data_grp.create_dataset(f"next_obs/{k}", data=np.array(traj["next_obs"][k]))

            # Save metadata
            if "model" in traj["initial_state_dict"]:
                ep_data_grp.attrs["model_file"] = traj["initial_state_dict"]["model"]
            ep_data_grp.attrs["num_samples"] = traj["actions"].shape[0]
            total_samples += traj["actions"].shape[0]

    # Process and save statistics
    rollout_stats = TensorUtils.list_of_flat_dict_to_dict_of_list(rollout_stats)
    avg_rollout_stats = {k: np.mean(rollout_stats[k]) for k in rollout_stats}
    avg_rollout_stats["Num_Success"] = np.sum(rollout_stats["Success_Rate"])

    if write_video:
        stats_path = f'{args.video_path.split(".")[0]}_results.json'
        with open(stats_path, 'w') as json_file:
            json.dump(avg_rollout_stats, json_file, indent=4)

    print("Average Rollout Stats")
    print(json.dumps(avg_rollout_stats, indent=4))

    # Cleanup
    if write_video:
        video_writer.close()

    if write_dataset:
        data_grp.attrs["total"] = total_samples
        data_grp.attrs["env_args"] = json.dumps(env.serialize(), indent=4)
        data_writer.close()
        print(f"Wrote dataset trajectories to {args.dataset_path}")

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument(
        "--mujoco_model_path",
        type=str,
        required=True,
        help="path to MuJoCo XML model file"
    )

    # Path to trained model
    parser.add_argument(
        "--agent",
        type=str,
        required=True,
        help="path to saved checkpoint pth file"
    )

    # number of rollouts
    parser.add_argument(
        "--n_rollouts",
        type=int,
        default=100,
        help="number of rollouts"
    )

    # maximum horizon
    parser.add_argument(
        "--horizon",
        type=int,
        default=None,
        help="(optional) override maximum horizon of rollout from checkpoint"
    )

    # Whether to render rollouts to screen
    parser.add_argument(
        "--render",
        action='store_true',
        help="enable on-screen rendering"
    )

    # Video recording options
    parser.add_argument(
        "--video_path",
        type=str,
        default=None,
        help="(optional) render rollouts to this video file path"
    )

    parser.add_argument(
        "--video_skip",
        type=int,
        default=5,
        help="render frames to video every n steps"
    )

    # Camera settings
    parser.add_argument(
        "--camera_names",
        type=str,
        nargs='+',
        default=["agentview"],
        help="camera name(s) to use for rendering"
    )

    # Dataset saving options
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="(optional) path to save rollout data as hdf5"
    )

    parser.add_argument(
        "--dataset_obs",
        action='store_true',
        help="include observations in output dataset"
    )

    # Random seed
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="random seed for rollouts"
    )

    # Video prompt
    parser.add_argument(
        "--video_prompt",
        type=str,
        default=None,
        help="path to video prompt file"
    )

    args = parser.parse_args()
    run_trained_agent(args)