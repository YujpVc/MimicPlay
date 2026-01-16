import h5py
import numpy as np
import cv2
import imageio
import argparse
import os
from math import tan, radians


def project_points_to_image(world_points, cam_pos, cam_lookat, cam_fov_degrees, image_width, image_height):
    """
    将3D世界坐标点投影到2D图像平面上。
    （此函数无需修改）
    """

    # 1. 构建视图矩阵 (View Matrix)
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
        [0, 0, 0, 1]
    ])

    # 2. 构建透视投影矩阵 (Perspective Projection Matrix)
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
        [0, 0, -1, 0]
    ])

    # 3. 应用变换
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
        if 0 <= screen_x < image_width and 0 <= screen_y < image_height:
            projected_points.append((int(screen_x), int(screen_y)))

    return projected_points


def generate_trajectory_video(args):
    """
    主函数，用于生成带有未来轨迹可视化的视频。
    """
    # --- 参数定义 ---
    CAM_POS = np.array([2.5, 1.0, 1.8])
    CAM_LOOKAT = np.array([0.65, 1.0, 1.0])
    CAM_FOV = 30.0

    # 未来轨迹点可视化参数
    FUTURE_MAX_RADIUS = 9
    FUTURE_MIN_RADIUS = 0.5
    FUTURE_MAX_ALPHA = 0.9
    FUTURE_MIN_ALPHA = 0.2
    FUTURE_POINT_COLOR_BGR = (0, 0, 255)  # BGR: 红色

    # 当前位置点可视化参数
    CURRENT_POINT_RADIUS = 10
    CURRENT_POINT_COLOR_BGR = (0, 255, 0)  # BGR: 绿色

    if not os.path.exists(args.dataset):
        print(f"错误：数据集文件不存在: {args.dataset}")
        return

    with h5py.File(args.dataset, "r") as f:
        demos = list(f["data"].keys())
        inds = np.argsort([int(elem[5:]) for elem in demos])
        demos = [demos[i] for i in inds]
        print(f"找到 {len(demos)} 个 'demo'.")

        video_writer = imageio.get_writer(args.video_path, fps=20)

        for i, ep in enumerate(demos):
            print(f"正在处理 Demo {i + 1}/{len(demos)}: {ep}")

            traj_grp = f[f"data/{ep}"]
            required_keys = ["obs/agentview_image", "obs/robot0_eef_pos_future_traj", "obs/robot0_eef_pos"]
            if not all(key in traj_grp for key in required_keys):
                print(f"警告: Demo {ep} 缺少所需数据, 已跳过。")
                continue

            images = traj_grp["obs/agentview_image"]
            future_trajs = traj_grp["obs/robot0_eef_pos_future_traj"]
            current_eef_positions = traj_grp["obs/robot0_eef_pos"]

            num_frames = images.shape[0]
            image_height, image_width, _ = images[0].shape

            for frame_idx in range(num_frames):
                output_image = cv2.cvtColor(images[frame_idx], cv2.COLOR_RGB2BGR)

                # --- 1. 绘制半透明的未来轨迹点 ---
                future_points_3d = future_trajs[frame_idx].reshape(10, 3)
                future_points_2d = project_points_to_image(
                    world_points=future_points_3d,
                    cam_pos=CAM_POS, cam_lookat=CAM_LOOKAT, cam_fov_degrees=CAM_FOV,
                    image_width=image_width, image_height=image_height
                )

                num_points = len(future_points_2d)
                if num_points > 0:
                    for j, point in enumerate(future_points_2d):
                        overlay = output_image.copy()

                        # 计算渐变属性
                        ratio = j / (num_points - 1) if num_points > 1 else 0
                        radius = int(FUTURE_MAX_RADIUS - ratio * (FUTURE_MAX_RADIUS - FUTURE_MIN_RADIUS))
                        alpha = FUTURE_MAX_ALPHA - ratio * (FUTURE_MAX_ALPHA - FUTURE_MIN_ALPHA)

                        # 【新】计算并应用振动效果
                        if args.enable_vibration:
                            # 根据点的远近计算当前振动强度
                            current_strength = args.vibration_strength * ratio
                            # 生成随机偏移量
                            dx = np.random.uniform(-current_strength, current_strength)
                            dy = np.random.uniform(-current_strength, current_strength)
                            # 应用偏移
                            final_point = (int(point[0] + dx), int(point[1] + dy))
                        else:
                            final_point = point

                        cv2.circle(overlay, final_point, radius, FUTURE_POINT_COLOR_BGR, -1)
                        cv2.addWeighted(overlay, alpha, output_image, 1 - alpha, 0, output_image)

                # --- 2. 绘制完全不透明的当前位置点 ---
                current_pos_3d = current_eef_positions[frame_idx]
                current_point_2d_list = project_points_to_image(
                    world_points=[current_pos_3d],
                    cam_pos=CAM_POS, cam_lookat=CAM_LOOKAT, cam_fov_degrees=CAM_FOV,
                    image_width=image_width, image_height=image_height
                )

                if current_point_2d_list:
                    cv2.circle(output_image, current_point_2d_list[0], CURRENT_POINT_RADIUS, CURRENT_POINT_COLOR_BGR,
                               -1)

                # --- 3. 将最终帧写入视频 ---
                final_frame_rgb = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
                video_writer.append_data(final_frame_rgb)

        video_writer.close()
        print(f"视频已成功保存到: {args.video_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="从HDF5数据集生成一个视频，其中可视化了机器人末端的当前位置和未来轨迹。"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="HDF5数据集文件的路径。",
    )
    parser.add_argument(
        "--video_path",
        type=str,
        default=None,
        help="（可选）输出视频文件的路径。如果未提供，将在数据集同目录下生成。",
    )
    # 【新】添加的命令行参数
    parser.add_argument(
        "--enable_vibration",
        action='store_true',  # 这使其成为一个开关，存在即为True
        help="（可选）为未来轨迹点启用随机振动效果，模拟不确定性。"
    )
    parser.add_argument(
        "--vibration_strength",
        type=float,
        default=60.0,  # 默认最大振动幅度为5个像素
        help="（可选）振动效果的最大强度（像素）。仅在--enable_vibration时生效。"
    )

    args = parser.parse_args()

    if args.video_path is None:
        base, ext = os.path.splitext(args.dataset)
        suffix = "with_full_traj"
        if args.enable_vibration:
            suffix += "_vibration"
        args.video_path = f"{base}_{suffix}.mp4"

    generate_trajectory_video(args)
