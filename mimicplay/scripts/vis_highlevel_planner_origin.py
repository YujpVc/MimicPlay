"""
验证高层规划器（High-Level Planner）是否正常运行。

从 HDF5 演示数据中读取当前帧与目标帧，调用已训练的高层策略得到预测轨迹，
并与真实轨迹对比可视化，输出为视频（红=预测，蓝=GT，绿=当前末端位置）。

相机内外参已写死在下方常量。若换数据集，请运行: python scripts/read_camera_params.py --hdf5 <你的hdf5> 并替换常量。
"""

import argparse
import json
import os
import sys
import cv2
import h5py
import imageio
import numpy as np
import torch
from math import tan, radians

# 与 train.py 一致：优先使用本地的 robosuite/robomimic（含 models.transformers）
_robosuite_path = os.path.expanduser("~/robosuite")
_robomimic_path = os.path.expanduser("~/robomimic")
if os.path.exists(_robosuite_path):
    sys.path.insert(0, _robosuite_path)
if os.path.exists(_robomimic_path):
    sys.path.insert(0, _robomimic_path)

import mimicplay.utils.file_utils as FileUtils
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.obs_utils as ObsUtils

# 相机内外参（写死）：由 read_camera_params.py 从 image_demo.hdf5 的 model_file XML 读出。
# 换数据集时请运行: python scripts/read_camera_params.py --hdf5 <你的hdf5>，并替换下面三行。
CAM_POS = np.array([0.658613, 0.0, 1.61035])
CAM_LOOKAT = np.array([-0.897384715, 0.0, 0.3538186511])
CAM_FOV = 45.0


def load_vis_params_from_config(config_path):
    """从配置文件中读取可视化/推理相关参数。"""
    with open(config_path, "r") as f:
        cfg = json.load(f)
    algo = cfg.get("algo", {})
    playdata = algo.get("playdata", {})
    obs_enc = cfg.get("observation", {}).get("encoder", {})
    rgb_enc = obs_enc.get("rgb", {})
    rnd_kwargs = rgb_enc.get("obs_randomizer_kwargs", {})

    return {
        "eval_goal_gap": playdata.get("eval_goal_gap", 50),
        "goal_image_range": playdata.get("goal_image_range", [40, 60]),
        "crop_height": rnd_kwargs.get("crop_height", 76),
        "crop_width": rnd_kwargs.get("crop_width", 76),
        "ac_dim": algo.get("highlevel", {}).get("ac_dim", 30),
        "output_dir": cfg.get("train", {}).get("output_dir", ""),
    }


def project_points_to_image(world_points, cam_pos, cam_lookat, cam_fov_degrees, image_width, image_height, clamp_to_bounds=False):
    """将 3D 世界坐标点投影到 2D 图像平面。
    clamp_to_bounds: 若 True，将超出画面的点夹到边界内，便于诊断/显示。
    """
    world_up = np.array([0.0, 0.0, 1.0])
    forward = cam_lookat - cam_pos
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, world_up)
    right = right / np.linalg.norm(right)
    up = np.cross(right, forward)

    view_matrix = np.array([
        [right[0], right[1], right[2], -np.dot(right, cam_pos)],
        [up[0], up[1], up[2], -np.dot(up, cam_pos)],
        [-forward[0], -forward[1], -forward[2], np.dot(forward, cam_pos)],
        [0, 0, 0, 1],
    ])

    aspect_ratio = image_width / image_height
    near_plane = 0.1
    far_plane = 100.0
    fov_rad = radians(cam_fov_degrees)
    f = 1.0 / tan(fov_rad / 2.0)
    projection_matrix = np.array([
        [f / aspect_ratio, 0, 0, 0],
        [0, f, 0, 0],
        [0, 0, (far_plane + near_plane) / (near_plane - far_plane),
         (2 * far_plane * near_plane) / (near_plane - far_plane)],
        [0, 0, -1, 0],
    ])

    projected_points = []
    for point in world_points:
        p_world = np.append(point, 1.0)
        p_cam = view_matrix @ p_world
        if p_cam[2] > -near_plane:
            continue
        p_clip = projection_matrix @ p_cam
        p_ndc = p_clip[:3] / p_clip[3]
        screen_x = (p_ndc[0] + 1.0) / 2.0 * image_width
        screen_y = (1.0 - p_ndc[1]) / 2.0 * image_height
        if clamp_to_bounds:
            screen_x = int(np.clip(screen_x, 0, image_width - 1))
            screen_y = int(np.clip(screen_y, 0, image_height - 1))
            projected_points.append((screen_x, screen_y))
        elif 0 <= screen_x < image_width and 0 <= screen_y < image_height:
            projected_points.append((int(screen_x), int(screen_y)))
    return projected_points


def draw_trajectory_gradient(img, points, color_bgr, max_radius=1, min_radius=0.5, max_alpha=1, min_alpha=1):
    """在图像上按渐变绘制轨迹点。"""
    num_points = len(points)
    if num_points == 0:
        return img
    for j, point in enumerate(points):
        overlay = img.copy()
        ratio = j / (num_points - 1) if num_points > 1 else 0
        radius = int(max_radius - ratio * (max_radius - min_radius))
        alpha = max_alpha - ratio * (max_alpha - min_alpha)
        cv2.circle(overlay, point, radius, color_bgr, -1)
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    return img


def visualize_highlevel_planner(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 从配置文件读取参数
    vis_params = load_vis_params_from_config(args.config)
    goal_offset = vis_params["eval_goal_gap"]
    goal_image_range = vis_params["goal_image_range"]
    crop_h = vis_params["crop_height"]
    crop_w = vis_params["crop_width"]
    ac_dim = vis_params["ac_dim"]
    future_horizon = 10
    point_gap = max(1, goal_offset // future_horizon)  # 与训练时轨迹步长一致

    print(f"Config: goal_offset={goal_offset}, goal_image_range={goal_image_range}, "
          f"crop={crop_h}x{crop_w}, ac_dim={ac_dim}, point_gap={point_gap}")

    # 相机使用本文件顶部写死的 CAM_POS, CAM_LOOKAT, CAM_FOV

    # 加载策略（checkpoint 内会初始化 ObsUtils）
    print(f"Loading model from: {args.agent}")
    policy, ckpt_dict = FileUtils.policy_from_checkpoint(
        ckpt_path=args.agent,
        device=device,
        verbose=False,
    )
    algo = policy.policy
    if hasattr(algo, "nets"):
        algo.nets.eval()
    elif hasattr(algo, "eval"):
        algo.eval()

    obs_normalization_stats = ckpt_dict.get("obs_normalization_stats", None)
    if obs_normalization_stats is not None:
        for m in obs_normalization_stats:
            for k in obs_normalization_stats[m]:
                v = obs_normalization_stats[m][k]
                if isinstance(v, torch.Tensor):
                    obs_normalization_stats[m][k] = v.cpu().numpy()
                elif not isinstance(v, np.ndarray):
                    obs_normalization_stats[m][k] = np.array(v)

    # 加载 HDF5 演示
    print(f"Loading data from: {args.video_prompt}")
    with h5py.File(args.video_prompt, "r") as f:
        demo_key = "data/demo_0"
        if "data" in f:
            keys = sorted(
                list(f["data"].keys()),
                key=lambda x: int(x.split("_")[1]) if "_" in x and x.split("_")[1].isdigit() else x,
            )
            if keys:
                demo_key = f"data/{keys[0]}"
        print(f"Processing demo: {demo_key}")
        agentview_images = f[demo_key]["obs"]["agentview_image"][:]
        robot0_eef_pos = f[demo_key]["obs"]["robot0_eef_pos"][:]

    total_frames = len(agentview_images)
    H, W = agentview_images[0].shape[:2]
    if agentview_images[0].ndim == 3 and agentview_images[0].shape[2] == 1:
        H, W = agentview_images[0].shape[:2]
    print(f"Image resolution: {W}x{H}, total_frames={total_frames}")

    # 当 checkpoint 无归一化统计时，模型输出通常在 [-1,1]（GMM tanh），用演示的 eef 范围映射到世界坐标
    eef_min = robot0_eef_pos.min(axis=0)
    eef_max = robot0_eef_pos.max(axis=0)
    if obs_normalization_stats is None:
        print("No obs_normalization_stats in checkpoint; assuming model output in [-1,1], will scale to demo eef range.")
        print(f"  eef range: x=[{eef_min[0]:.3f},{eef_max[0]:.3f}] y=[{eef_min[1]:.3f},{eef_max[1]:.3f}] z=[{eef_min[2]:.3f},{eef_max[2]:.3f}]")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    writer = imageio.get_writer(args.output, fps=20)
    print("Generating video...")

    with torch.no_grad():
        for i in range(0, total_frames, args.skip):
            curr_img = agentview_images[i]
            curr_pos = robot0_eef_pos[i]

            # 目标帧：使用配置中的 eval_goal_gap
            goal_idx = min(i + goal_offset, total_frames - 1)
            goal_img = agentview_images[goal_idx]

            img_h, img_w = curr_img.shape[:2]
            curr_img_in = curr_img
            goal_img_in = goal_img
            if img_h >= crop_h and img_w >= crop_w:
                start_h = (img_h - crop_h) // 2
                start_w = (img_w - crop_w) // 2
                curr_img_in = curr_img[start_h : start_h + crop_h, start_w : start_w + crop_w]
                goal_img_in = goal_img[start_h : start_h + crop_h, start_w : start_w + crop_w]

            obs_dict = {
                "agentview_image": np.array([curr_img_in]),
                "robot0_eef_pos": np.array([curr_pos]),
            }
            goal_dict = {"agentview_image": np.array([goal_img_in])}
            obs_dict = ObsUtils.process_obs_dict(obs_dict)
            goal_dict = ObsUtils.process_obs_dict(goal_dict)

            if obs_normalization_stats is not None:
                obs_dict_tensor = TensorUtils.to_float(
                    TensorUtils.to_device(TensorUtils.to_tensor(obs_dict), device)
                )
                goal_dict_tensor = TensorUtils.to_float(
                    TensorUtils.to_device(TensorUtils.to_tensor(goal_dict), device)
                )
                stats_tensor = TensorUtils.to_float(
                    TensorUtils.to_device(TensorUtils.to_tensor(obs_normalization_stats), device)
                )
                keys_to_remove = [k for k in stats_tensor if "image" in k]
                for k in keys_to_remove:
                    del stats_tensor[k]
                obs_to_norm = {k: v for k, v in obs_dict_tensor.items() if k in stats_tensor}
                obs_keep = {k: v for k, v in obs_dict_tensor.items() if k not in stats_tensor}
                goal_to_norm = {k: v for k, v in goal_dict_tensor.items() if k in stats_tensor}
                goal_keep = {k: v for k, v in goal_dict_tensor.items() if k not in stats_tensor}
                if obs_to_norm:
                    obs_to_norm = ObsUtils.normalize_obs(obs_to_norm, obs_normalization_stats=stats_tensor)
                if goal_to_norm:
                    goal_to_norm = ObsUtils.normalize_obs(goal_to_norm, obs_normalization_stats=stats_tensor)
                obs_dict_tensor = {**obs_to_norm, **obs_keep}
                goal_dict_tensor = {**goal_to_norm, **goal_keep}
            else:
                obs_dict_tensor = TensorUtils.to_float(
                    TensorUtils.to_device(TensorUtils.to_tensor(obs_dict), device)
                )
                goal_dict_tensor = TensorUtils.to_float(
                    TensorUtils.to_device(TensorUtils.to_tensor(goal_dict), device)
                )

            pred_traj = None
            if hasattr(algo, "_get_latent_plan"):
                lat_plan, _ = algo._get_latent_plan(obs_dict_tensor, goal_dict_tensor)
                if lat_plan is not None and lat_plan.shape[-1] == ac_dim:
                    pred_traj = lat_plan.view(-1, future_horizon, 3).cpu().numpy()[0]

                    if obs_normalization_stats is not None and pred_traj is not None:
                        target_key = "robot0_eef_pos_future_traj"
                        stats = None
                        if target_key in obs_normalization_stats:
                            stats = obs_normalization_stats[target_key]
                        else:
                            for modality in ["low_dim", "proprio"]:
                                if (
                                    modality in obs_normalization_stats
                                    and target_key in obs_normalization_stats[modality]
                                ):
                                    stats = obs_normalization_stats[modality][target_key]
                                    break
                        if stats is not None:
                            mean = np.array(stats["mean"])
                            std = np.array(stats["std"])
                            if mean.size == ac_dim:
                                mean = mean.reshape(future_horizon, 3)
                            if std.size == ac_dim:
                                std = std.reshape(future_horizon, 3)
                            pred_traj = pred_traj * std + mean
                    elif obs_normalization_stats is None and pred_traj is not None:
                        # 模型输出为 [-1,1]（GMM tanh），线性映射到演示的 eef 范围
                        pred_traj = (pred_traj + 1.0) * 0.5 * (eef_max - eef_min) + eef_min

                    # 首帧诊断：确认预测值与投影点数
                    if i == 0 and pred_traj is not None:
                        pred_pts_strict = project_points_to_image(
                            pred_traj, CAM_POS, CAM_LOOKAT, CAM_FOV, W, H, clamp_to_bounds=False
                        )
                        print(f"\n[Diagnostic] Frame 0: pred_traj shape={pred_traj.shape}, "
                              f"pred[0]={pred_traj[0].round(4)}, "
                              f"pred range=[{pred_traj.min():.3f},{pred_traj.max():.3f}], "
                              f"projected points (strict)={len(pred_pts_strict)}")
            gt_indices = [
                min(i + k * point_gap, total_frames - 1) for k in range(1, future_horizon + 1)
            ]
            gt_traj = robot0_eef_pos[gt_indices]

            if args.debug and i % 10 == 0:
                print(f"\nFrame {i}: curr_eef={curr_pos[:3].round(4)}")
                if pred_traj is not None:
                    print(f"  pred[0]={pred_traj[0].round(4)}")
                if len(gt_traj) > 0:
                    print(f"  gt[0]  ={gt_traj[0].round(4)}")
                if pred_traj is not None and len(gt_traj) > 0:
                    print(f"  offset ={(pred_traj[0] - gt_traj[0]).round(4)}")

            # 可视化
            vis_img = curr_img.copy()
            if vis_img.ndim == 3 and vis_img.shape[0] == 3:
                vis_img = vis_img.transpose(1, 2, 0)
            vis_img = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)

            if len(gt_traj) > 0:
                gt_pts_2d = project_points_to_image(gt_traj, CAM_POS, CAM_LOOKAT, CAM_FOV, W, H)
                vis_img = draw_trajectory_gradient(vis_img, gt_pts_2d, (255, 0, 0))
            if pred_traj is not None:
                pred_pts_2d = project_points_to_image(
                    pred_traj, CAM_POS, CAM_LOOKAT, CAM_FOV, W, H, clamp_to_bounds=True
                )
                vis_img = draw_trajectory_gradient(vis_img, pred_pts_2d, (0, 0, 255))
            curr_pt_2d = project_points_to_image([curr_pos], CAM_POS, CAM_LOOKAT, CAM_FOV, W, H)
            if curr_pt_2d:
                cv2.circle(vis_img, curr_pt_2d[0], 1, (0, 255, 0), -1)

            vis_img = cv2.resize(vis_img, (256, 256), interpolation=cv2.INTER_NEAREST)
            cv2.putText(vis_img, "Red: Pred", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.putText(vis_img, "Blue: GT", (5, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            cv2.putText(vis_img, "Green: Curr", (5, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            writer.append_data(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))

            if i % 50 == 0:
                print(f"Processed frame {i}/{total_frames}")

    writer.close()
    print(f"Video saved to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="验证高层规划器并生成预测 vs GT 可视化视频")
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    parser.add_argument(
        "--config",
        type=str,
        default="/home/yujp/MimicPlay_o/mimicplay/configs/highlevel.json",
        help="高层规划器配置文件路径（用于 goal_gap、crop 等参数）",
    )
    parser.add_argument(
        "--agent",
        type=str,
        default="/home/yujp/MimicPlay_o/trained_models_highlevel/test/20260301130248/models/model_epoch_374_best_validation_-73.82323913574218.pth",
        help="高层规划器 checkpoint 路径（.pth）",
    )
    parser.add_argument(
        "--video_prompt",
        type=str,
        default="/home/yujp/MimicPlay_o/mimicplay/datasets/eval-task-1_turn_on_stove_put_pan_on_stove_put_bowl_on_shelf/image_demo.hdf5",
        help="HDF5 演示数据路径（含 agentview_image、robot0_eef_pos）",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join(root, "training_result", "highlevel_origin_vis.mp4"),
        help="输出视频路径",
    )
    parser.add_argument("--skip", type=int, default=1, help="帧间隔")
    parser.add_argument("--debug", action="store_true", help="打印预测与 GT 对比")
    args = parser.parse_args()

    if not os.path.isfile(args.config):
        raise FileNotFoundError(f"Config not found: {args.config}")
    if not os.path.isfile(args.agent):
        raise FileNotFoundError(f"Agent checkpoint not found: {args.agent}")
    if not os.path.isfile(args.video_prompt):
        raise FileNotFoundError(f"Video prompt HDF5 not found: {args.video_prompt}")

    visualize_highlevel_planner(args)
