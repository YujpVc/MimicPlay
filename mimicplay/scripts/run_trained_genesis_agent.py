"""
The main script for evaluating a policy in an environment.

Args:
    agent (str): path to saved checkpoint pth file

    horizon (int): if provided, override maximum horizon of rollout from the one
        in the checkpoint

    env (str): if provided, override name of env from the one in the checkpoint,
        and use it for rollouts

    render (bool): if flag is provided, use on-screen rendering during rollouts

    video_path (str): if provided, render trajectories to this video file path

    video_skip (int): render frames to a video every @video_skip steps

    camera_names (str or [str]): camera name(s) to use for rendering on-screen or to video

    dataset_path (str): if provided, an hdf5 file will be written at this path with the
        rollout data

    dataset_obs (bool): if flag is provided, and @dataset_path is provided, include
        possible high-dimensional observations in output dataset hdf5 file (by default,
        observations are excluded and only simulator states are saved).

    seed (int): if provided, set seed for rollouts

    bddl_file (str): if provided, the task's goal is specified as the symbolic goal in the
         bddl file (several symbolic predicates connected with AND / OR)

    video_prompt (str): if provided, a task video prompt is loaded and used in the
         evaluation rollouts


Example usage:

    # Evaluate a policy with 100 rollouts of maximum horizon 2000 and save the rollouts to a video.
    # Visualize the agentview and wrist cameras during the rollout.

    python scripts/run_trained_agent.py --agent 'path_to_trained_low-level_policy' \
          --n_rollouts 100 --horizon 2000 --seed 0 \
          --bddl_file 'path_to_task_bddl_file' --video_prompt 'path_to_task_video_prompt' \
          --video_path 'eval_rollouts.mp4'

"""
import sys

# Prefer local MimicPlay_o source tree first.
sys.path.insert(0, "/home/yujp/MimicPlay_o")
sys.path.append("/home/yujp/robosuite")
sys.path.append("/home/yujp/robomimic")

import argparse
import json
import traceback
import h5py
import imageio
import numpy as np
from copy import deepcopy
from math import tan, radians

import torch
import cv2

import mimicplay.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.obs_utils as ObsUtils
from robomimic.envs.env_base import EnvBase
from mimicplay.algo import RolloutPolicy

# Camera params (must match env_genesis viewer / render_view camera for correct projection)
CAM_POS = np.array([2.5, 1.0, 1.8], dtype=np.float32)
CAM_LOOKAT = np.array([0.65, 1.0, 1.0], dtype=np.float32)
CAM_FOV = 30.0


def project_points_to_image(world_points, cam_pos, cam_lookat, cam_fov_degrees, image_width, image_height):
    """
    Project 3D world points to 2D image plane (simple pinhole, z-up world).
    Copied from vis scripts to keep consistent.
    """
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    forward = cam_lookat - cam_pos
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, world_up)
    right = right / np.linalg.norm(right)
    up = np.cross(right, forward)

    view_matrix = np.array([
        [right[0], right[1], right[2], -np.dot(right, cam_pos)],
        [up[0], up[1], up[2], -np.dot(up, cam_pos)],
        [-forward[0], -forward[1], -forward[2], np.dot(forward, cam_pos)],
        [0, 0, 0, 1]
    ], dtype=np.float32)

    aspect_ratio = float(image_width) / float(image_height)
    near_plane = 0.1
    far_plane = 100.0
    fov_rad = radians(cam_fov_degrees)
    f = 1.0 / tan(fov_rad / 2.0)

    projection_matrix = np.array([
        [f / aspect_ratio, 0, 0, 0],
        [0, f, 0, 0],
        [0, 0, (far_plane + near_plane) / (near_plane - far_plane),
         (2 * far_plane * near_plane) / (near_plane - far_plane)],
        [0, 0, -1, 0]
    ], dtype=np.float32)

    projected_points = []
    for point in world_points:
        p_world = np.append(np.asarray(point, dtype=np.float32), 1.0)
        p_cam = view_matrix @ p_world
        if p_cam[2] > -near_plane:
            continue
        p_clip = projection_matrix @ p_cam
        p_ndc = p_clip[:3] / p_clip[3]
        screen_x = (p_ndc[0] + 1.0) / 2.0 * image_width
        screen_y = (1.0 - p_ndc[1]) / 2.0 * image_height
        if 0 <= screen_x < image_width and 0 <= screen_y < image_height:
            projected_points.append((int(screen_x), int(screen_y)))
    return projected_points


def draw_trajectory_gradient(img_bgr, points, color_bgr, max_radius=6, min_radius=2, max_alpha=0.9, min_alpha=0.2):
    num_points = len(points)
    if num_points == 0:
        return img_bgr
    for j, point in enumerate(points):
        overlay = img_bgr.copy()
        ratio = j / (num_points - 1) if num_points > 1 else 0
        radius = int(max_radius - ratio * (max_radius - min_radius))
        alpha = max_alpha - ratio * (max_alpha - min_alpha)
        cv2.circle(overlay, point, radius, color_bgr, -1)
        cv2.addWeighted(overlay, alpha, img_bgr, 1 - alpha, 0, img_bgr)
    return img_bgr


def maybe_overlay_planner_on_frame(rgb_img, camera_name, pred_traj_3d, curr_pos_3d):
    """
    Overlay predicted planner trajectory (red) and current eef position (green) on a rendered RGB frame.
    Only supported for fixed cameras with known extrinsics (render_view / agentview_image).
    """
    if rgb_img is None:
        return rgb_img
    if pred_traj_3d is None or curr_pos_3d is None:
        return rgb_img
    if camera_name not in {"render_view", "agentview_image"}:
        return rgb_img

    h, w = rgb_img.shape[:2]
    pred_pts_2d = project_points_to_image(pred_traj_3d, CAM_POS, CAM_LOOKAT, CAM_FOV, w, h)
    curr_pt_2d = project_points_to_image([curr_pos_3d], CAM_POS, CAM_LOOKAT, CAM_FOV, w, h)

    img_bgr = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
    img_bgr = draw_trajectory_gradient(img_bgr, pred_pts_2d, (0, 0, 255))
    if curr_pt_2d:
        cv2.circle(img_bgr, curr_pt_2d[0], 6, (0, 255, 0), -1)
    cv2.putText(img_bgr, "Red: Pred (planner)", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.putText(img_bgr, "Green: Curr eef", (5, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def rollout(policy, env, horizon, render=False, video_writer=None, video_skip=5, return_obs=False, camera_names=None):
    """
    Helper function to carry out rollouts. Supports on-screen rendering, off-screen rendering to a video,
    and returns the rollout trajectory.

    Args:
        policy (instance of RolloutPolicy): policy loaded from a checkpoint
        env (instance of EnvBase): env loaded from a checkpoint or demonstration metadata
        horizon (int): maximum horizon for the rollout
        render (bool): whether to render rollout on-screen
        video_writer (imageio writer): if provided, use to write rollout to video
        video_skip (int): how often to write video frames
        return_obs (bool): if True, return possibly high-dimensional observations along the trajectoryu.
            They are excluded by default because the low-dimensional simulation states should be a minimal
            representation of the environment.
        camera_names (list): determines which camera(s) are used for rendering. Pass more than
            one to output a video with multiple camera views concatenated horizontally.

    Returns:
        stats (dict): some statistics for the rollout - such as return, horizon, and task success
        traj (dict): dictionary that corresponds to the rollout trajectory
    """
    assert isinstance(env, EnvBase)
    assert isinstance(policy, RolloutPolicy)
    assert not (render and (video_writer is not None))

    policy.start_episode()
    obs = env.reset()
    state_dict = env.get_state()

    # This reset_to call is necessary for some environments for deterministic playback
    obs = env.reset_to(state_dict)

    results = {}
    video_count = 0
    total_reward = 0.
    traj = dict(actions=[], rewards=[], dones=[], states=[], initial_state_dict=state_dict)
    if return_obs:
        traj.update(dict(obs=[], next_obs=[]))

    try:
        success = False
        for step_i in range(horizon):
            # We want the lowlevel action, but also want to visualize highlevel planner outputs.
            # Lowlevel_GPT_mimicplay.get_action mutates obs_dict by inserting:
            # - obs_dict["guidance"] : (B, 30) = 10 future 3D points
            # So we reproduce RolloutPolicy.__call__ here to capture that value.
            ob_t = policy._prepare_observation(obs)
            with torch.no_grad():
                ac_t = policy.policy.get_action(obs_dict=ob_t, goal_dict=None)
            guidance = ob_t.get("guidance", None)
            pred_traj_3d = None
            if guidance is not None:
                try:
                    pred_traj_3d = TensorUtils.to_numpy(guidance[0]).reshape(-1, 3)
                except Exception:
                    pred_traj_3d = None
            curr_pos_3d = obs.get("robot0_eef_pos", None)
            act = TensorUtils.to_numpy(ac_t[0])
            print(f"Action at step {step_i}:\n{act}")

            next_obs, r, done, _ = env.step(act)

            total_reward += r
            success = env.is_success()["task"]

            # visualization
            if render:
                # Render to RGB array, overlay planner trajectory, then show via OpenCV.
                # We avoid env.render(mode="human") here because we want to draw on the frame first.
                cam_name = camera_names[0]
                frame = env.render(mode="rgb_array", height=512, width=512, camera_name=cam_name)
                frame = maybe_overlay_planner_on_frame(
                    rgb_img=frame,
                    camera_name=cam_name,
                    pred_traj_3d=pred_traj_3d,
                    curr_pos_3d=curr_pos_3d,
                )
                if frame is not None:
                    cv2.imshow(f"{cam_name}_overlay", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                    cv2.waitKey(1)
            if video_writer is not None:
                if video_count % video_skip == 0:
                    video_img = []
                    for cam_name in camera_names:
                        frame = env.render(mode="rgb_array", height=512, width=512, camera_name=cam_name)
                        frame = maybe_overlay_planner_on_frame(
                            rgb_img=frame,
                            camera_name=cam_name,
                            pred_traj_3d=pred_traj_3d,
                            curr_pos_3d=curr_pos_3d,
                        )
                        video_img.append(frame)
                    video_img = np.concatenate(video_img, axis=1) # concatenate horizontally
                    video_writer.append_data(video_img)
                video_count += 1

            # collect transition
            traj["actions"].append(act)
            traj["rewards"].append(r)
            traj["dones"].append(done)
            traj["states"].append(state_dict["states"])
            if return_obs:
                traj["obs"].append(ObsUtils.unprocess_obs_dict(obs))
                traj["next_obs"].append(ObsUtils.unprocess_obs_dict(next_obs))

            if done or success:
                break

            # update for next iter
            obs = deepcopy(next_obs)
            state_dict = env.get_state()

    except env.rollout_exceptions as e:
        print(f"\n[Rollout Error] Caught env.rollout_exceptions: {repr(e)}")
        traceback.print_exc()
        # 为了调试清楚问题，直接抛出，让脚本报错退出，而不是静默吞掉
        raise
    except Exception as e:
        print(f"\n[Rollout Error] An unexpected exception occurred during rollout: {repr(e)}")
        traceback.print_exc()
        raise

    stats = dict(Return=total_reward, Horizon=(step_i + 1), Success_Rate=float(success))

    if return_obs:
        traj["obs"] = TensorUtils.list_of_flat_dict_to_dict_of_list(traj["obs"])
        traj["next_obs"] = TensorUtils.list_of_flat_dict_to_dict_of_list(traj["next_obs"])

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
    # some arg checking
    write_video = (args.video_path is not None)
    assert not (args.render and write_video)
    if args.render:
        assert len(args.camera_names) == 1

    ckpt_path = args.agent
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)

    # restore policy
    policy, ckpt_dict = FileUtils.policy_from_checkpoint(ckpt_path=ckpt_path, device=device, verbose=False)

    # ---- Debug prints: which model is actually used ----
    algo_name = None
    try:
        # 常见的 robomimic / mimicplay checkpoint 配置结构
        if "config" in ckpt_dict:
            cfg = ckpt_dict["config"]
            # 支持几种可能的字段名
            if hasattr(cfg, "algo") and hasattr(cfg.algo, "name"):
                algo_name = cfg.algo.name
            elif isinstance(cfg, dict):
                algo_name = cfg.get("algo_name", None) or cfg.get("algo", {}).get("name", None)
    except Exception as e:
        print(f"[Model Debug] Failed to read algo name from ckpt_dict: {e}")

    # 精确打印：内部 policy 的 Python 类、模块名、源码文件路径，以及当前 Python 解释器
    inner_cls = type(policy.policy)
    inner_module_name = inner_cls.__module__
    inner_qualname = getattr(inner_cls, "__qualname__", None)

    module_file = None
    try:
        import importlib, sys as _sys  # 局部别名避免污染上面的 sys
        mod = _sys.modules.get(inner_module_name)
        if mod is None:
            mod = importlib.import_module(inner_module_name)
        module_file = getattr(mod, "__file__", None)
    except Exception as e:
        print(f"[Model Debug] Failed to locate module file: {e}")

    print("\n[Model Debug] -------- Loaded Policy Info --------")
    print(f"[Model Debug] Python executable     : {sys.executable}")
    print(f"[Model Debug] Checkpoint path       : {ckpt_path}")
    print(f"[Model Debug] Algo name (from cfg)  : {algo_name}")
    print(f"[Model Debug] RolloutPolicy type    : {type(policy)}")
    print(f"[Model Debug] Inner policy type     : {inner_cls}")
    print(f"[Model Debug] Inner policy module   : {inner_module_name}")
    print(f"[Model Debug] Inner policy qname    : {inner_qualname}")
    print(f"[Model Debug] Inner policy file     : {module_file}")
    print("[Model Debug] ------------------------------------\n")

    policy.policy.load_eval_video_prompt(args.video_prompt)

    # read rollout settings
    rollout_num_episodes = args.n_rollouts
    rollout_horizon = args.horizon
    if rollout_horizon is None:
        config, _ = FileUtils.config_from_checkpoint(ckpt_dict=ckpt_dict)
        rollout_horizon = config.experiment.C.horizon

    # create environment from saved checkpoint
    env, _ = FileUtils.env_from_checkpoint(
        ckpt_dict=ckpt_dict,
        env_name=args.env,
        render=args.render,
        render_offscreen=(args.video_path is not None),
        verbose=False,
    )

    if args.condition_file is not None:
        env.condition_file = args.condition_file
        env._load_condition_file()
        print(f"Success condition loaded from: {env.condition_file}")

    print(f"Environment Type: {type(env)}")
    print(f"Environment Name: {env.name}")

    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    video_writer = None
    if write_video:
        video_writer = imageio.get_writer(args.video_path, fps=20)

    write_dataset = (args.dataset_path is not None)
    if write_dataset:
        data_writer = h5py.File(args.dataset_path, "w")
        data_grp = data_writer.create_group("data")
        total_samples = 0

    rollout_stats = []
    for i in range(rollout_num_episodes):
        print(f"\n--- Running rollout {i+1} of {rollout_num_episodes} ---")
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

            if "model" in traj["initial_state_dict"]:
                ep_data_grp.attrs["model_file"] = traj["initial_state_dict"]["model"]
            ep_data_grp.attrs["num_samples"] = traj["actions"].shape[0]
            total_samples += traj["actions"].shape[0]

    rollout_stats = TensorUtils.list_of_flat_dict_to_dict_of_list(rollout_stats)
    avg_rollout_stats = {k: np.mean(rollout_stats[k]) for k in rollout_stats}
    avg_rollout_stats["Num_Success"] = np.sum(rollout_stats["Success_Rate"])

    if args.video_path is not None:
        results_filename = f"{args.video_path.rsplit('.', 1)[0]}_results.json"
        with open(results_filename, 'w') as json_file:
            json.dump(avg_rollout_stats, json_file, indent=4)

    print("\n--- Average Rollout Stats ---")
    print(json.dumps(avg_rollout_stats, indent=4))

    if write_video:
        video_writer.close()

    if write_dataset:
        data_grp.attrs["total"] = total_samples
        data_grp.attrs["env_args"] = json.dumps(env.serialize(), indent=4)
        data_writer.close()
        print(f"Wrote dataset trajectories to {args.dataset_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--agent",
        type=str,
        default='/home/yujp/MimicPlay/trained_models_lowlevel/test/lowlevel_model_epoch_950_modified.pth',
        help="path to saved checkpoint pth file",
    )
    parser.add_argument(
        "--n_rollouts",
        type=int,
        default=100,
        help="number of rollouts",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=2000,
        help="(optional) override maximum horizon of rollout from the one in the checkpoint",
    )
    parser.add_argument(
        "--env",
        type=str,
        default=None,
        help="(optional) override name of env from the one in the checkpoint, and use it for rollouts",
    )
    parser.add_argument(
        "--render",
        default=False,
        action='store_true',
        help="on-screen rendering",
    )

    parser.add_argument(
        "--video_path",
        type=str,
        default=None,
        help="(optional) render rollouts to this video file path",
    )
    parser.add_argument(
        "--video_skip",
        type=int,
        default=5,
        help="render frames to video every n steps",
    )
    parser.add_argument(
        "--camera_names",
        type=str,
        nargs='+',
        default=["render_view"],
        help="(optional) camera name(s) to use for rendering on-screen or to video",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="(optional) if provided, an hdf5 file will be written at this path with the rollout data",
    )
    parser.add_argument(
        "--dataset_obs",
        action='store_true',
        help="include possibly high-dimensional observations in output dataset hdf5 file",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="(optional) set seed for rollouts",
    )
    parser.add_argument(
        "--video_prompt",
        type=str,
        default='/home/yujp/MimicPlay/mimicplay/datasets/eval-task-3_put_bowl_on_shelf_put_pan_in_shelf/image_demo.hdf5',
        help="(optional) if provided, a task video prompt is loaded and used in the evaluation rollouts",
    )
    parser.add_argument(
        "--condition_file",
        type=str,
        default=None,
        help="(optional) path to the file defining the task's symbolic goal",
    )

    args = parser.parse_args()
    run_trained_agent(args)

