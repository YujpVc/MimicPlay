
import argparse
import h5py
import torch
import numpy as np
import imageio
import os
import cv2
from math import tan, radians
import mimicplay.utils.file_utils as FileUtils
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.obs_utils as ObsUtils

def project_points_to_image(world_points, cam_pos, cam_lookat, cam_fov_degrees, image_width, image_height):
    """
    Project 3D world points to 2D image plane.
    """
    # 1. Build View Matrix
    world_up = np.array([0.0, 0.0, 1.0]) # Assumes Z-up world
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

    # 2. Build Perspective Projection Matrix
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

    # 3. Apply Transformation
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
        # Allow points slightly outside for continuity, or strict clip
        # Here keeping strict clip as per original code
        if 0 <= screen_x < image_width and 0 <= screen_y < image_height:
            projected_points.append((int(screen_x), int(screen_y)))
        else:
             # Optional: append None or handle out of bounds
             pass

    return projected_points

def draw_trajectory_gradient(img, points, color_bgr, max_radius=1, min_radius=0.5, max_alpha=1, min_alpha=1):
    num_points = len(points)
    if num_points == 0:
        return img

    for j, point in enumerate(points):
        overlay = img.copy()
        
        # Calculate gradient properties
        ratio = j / (num_points - 1) if num_points > 1 else 0
        radius = int(max_radius - ratio * (max_radius - min_radius))
        alpha = max_alpha - ratio * (max_alpha - min_alpha)
        
        cv2.circle(overlay, point, radius, color_bgr, -1)
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    
    return img

def visualize_planner(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Camera Params from show_future_traj.py
    CAM_POS = np.array([2.5, 1.0, 1.8])
    CAM_LOOKAT = np.array([0.65, 1.0, 1.0])
    CAM_FOV = 30.0
    
    # Load policy
    print(f"Loading model from: {args.agent}")
    policy, ckpt_dict = FileUtils.policy_from_checkpoint(
        ckpt_path=args.agent,
        device=device,
        verbose=False
    )
    algo = policy.policy
    if hasattr(algo, 'nets'):
        algo.nets.eval()
    elif hasattr(algo, 'eval'):
        algo.eval()
    
    # Get normalization stats for denormalizing predictions
    obs_normalization_stats = ckpt_dict.get("obs_normalization_stats", None)
    if obs_normalization_stats is not None:
        # Convert to numpy arrays if needed
        for m in obs_normalization_stats:
            for k in obs_normalization_stats[m]:
                if isinstance(obs_normalization_stats[m][k], torch.Tensor):
                    obs_normalization_stats[m][k] = obs_normalization_stats[m][k].cpu().numpy()
                elif not isinstance(obs_normalization_stats[m][k], np.ndarray):
                    obs_normalization_stats[m][k] = np.array(obs_normalization_stats[m][k])
    
    # Load video prompt
    print(f"Loading data from: {args.video_prompt}")
    f = h5py.File(args.video_prompt, 'r')
    
    demo_key = "data/demo_0"
    if 'data' in f:
        keys = sorted(list(f['data'].keys()), key=lambda x: int(x.split('_')[1]) if '_' in x and x.split('_')[1].isdigit() else x)
        if keys:
            demo_key = f"data/{keys[0]}"
            
    print(f"Processing demo: {demo_key}")
    
    agentview_images = f[demo_key]['obs']['agentview_image'][:]
    robot0_eef_pos = f[demo_key]['obs']['robot0_eef_pos'][:]
    total_frames = len(agentview_images)
    
    # Get Image dims
    H, W, C = agentview_images[0].shape
    if C == 1:
        H, W = agentview_images[0].shape[:2]
    
    print(f"Image resolution: {W}x{H}")
    
    writer = imageio.get_writer(args.output, fps=20)
    
    future_horizon = 10
    goal_offset = 100   # Match playdata.goal_image_range avg (10-30) -> 20
    point_gap = 5      # Match dataset generation (dataset_extract_traj_plans.py POINT_GAP=2)
                       # With FUTURE_POINTS_COUNT=10, this spans ~20 steps into the future (2,4,...,20).

    print("Generating video...")
    
    with torch.no_grad():
        for i in range(0, total_frames, args.skip):
            # Prepare inputs
            curr_img = agentview_images[i]
            curr_pos = robot0_eef_pos[i]
            
            goal_idx = min(i + goal_offset, total_frames - 1)
            goal_img = agentview_images[goal_idx]
            
            # Center crop images to 76x76 to match training distribution
            crop_h, crop_w = 76, 76
            img_h, img_w = curr_img.shape[:2]
            curr_img_in = curr_img
            goal_img_in = goal_img
            
            if img_h > crop_h and img_w > crop_w:
                start_h = (img_h - crop_h) // 2
                start_w = (img_w - crop_w) // 2
                curr_img_in = curr_img[start_h:start_h+crop_h, start_w:start_w+crop_w]
                goal_img_in = goal_img[start_h:start_h+crop_h, start_w:start_w+crop_w]

            obs_dict = {
                'agentview_image': np.array([curr_img_in]),
                'robot0_eef_pos': np.array([curr_pos])
            }
            goal_dict = {
                'agentview_image': np.array([goal_img_in])
            }
            # IMPORTANT: use robomimic's standard preprocessing (HWC uint8 -> CHW float[0,1])
            # This keeps visualization inputs consistent with training / rollout policy inputs.
            obs_dict = ObsUtils.process_obs_dict(obs_dict)
            goal_dict = ObsUtils.process_obs_dict(goal_dict)
            
            # Normalize observations if normalization stats are available
            if obs_normalization_stats is not None:
                # Convert to tensors first for normalization
                obs_dict_tensor = TensorUtils.to_float(TensorUtils.to_device(TensorUtils.to_tensor(obs_dict), device))
                goal_dict_tensor = TensorUtils.to_float(TensorUtils.to_device(TensorUtils.to_tensor(goal_dict), device))
                
                # Normalize
                obs_normalization_stats_tensor = TensorUtils.to_float(
                    TensorUtils.to_device(TensorUtils.to_tensor(obs_normalization_stats), device))
                
                # Remove image keys from normalization stats
                keys_to_remove = []
                for key in obs_normalization_stats_tensor:
                    if 'image' in key:
                        keys_to_remove.append(key)
                for key in keys_to_remove:
                    del obs_normalization_stats_tensor[key]
                
                # Only normalize non-image keys to avoid assertion error
                # Separate obs into keys to normalize and keys to keep as is
                obs_to_norm = {}
                obs_keep = {}
                for k, v in obs_dict_tensor.items():
                    if k in obs_normalization_stats_tensor:
                        obs_to_norm[k] = v
                    else:
                        obs_keep[k] = v
                        
                goal_to_norm = {}
                goal_keep = {}
                for k, v in goal_dict_tensor.items():
                    if k in obs_normalization_stats_tensor:
                        goal_to_norm[k] = v
                    else:
                        goal_keep[k] = v
                
                # Normalize
                if obs_to_norm:
                    obs_to_norm = ObsUtils.normalize_obs(obs_to_norm, obs_normalization_stats=obs_normalization_stats_tensor)
                if goal_to_norm:
                    goal_to_norm = ObsUtils.normalize_obs(goal_to_norm, obs_normalization_stats=obs_normalization_stats_tensor)
                
                # Merge back
                obs_dict_tensor = {**obs_to_norm, **obs_keep}
                goal_dict_tensor = {**goal_to_norm, **goal_keep}
                
                if i == 0:
                    print(f"\n[Frame {i}] Normalized Input Check:")
                    if 'robot0_eef_pos' in obs_dict_tensor:
                        print(f"  robot0_eef_pos: {obs_dict_tensor['robot0_eef_pos'][0]}")
                    else:
                        print("  robot0_eef_pos not found in obs_dict_tensor")
            else:
                # To Device
                obs_dict_tensor = TensorUtils.to_float(TensorUtils.to_device(TensorUtils.to_tensor(obs_dict), device))
                goal_dict_tensor = TensorUtils.to_float(TensorUtils.to_device(TensorUtils.to_tensor(goal_dict), device))
            
            # Inference (Predicted Future)
            pred_traj = None
            if hasattr(algo, '_get_latent_plan'):
                lat_plan, _ = algo._get_latent_plan(obs_dict_tensor, goal_dict_tensor)
                if lat_plan.shape[-1] == 30:
                    pred_traj = lat_plan.view(-1, 10, 3).cpu().numpy()[0] # [10, 3]
                    
                    # Denormalize predicted trajectory if normalization stats are available
                    if obs_normalization_stats is not None:
                        # The stats structure might be flat or nested. Try both.
                        target_key = 'robot0_eef_pos_future_traj'
                        stats = None
                        
                        # Case 1: Flat structure (key -> stats)
                        if target_key in obs_normalization_stats:
                            stats = obs_normalization_stats[target_key]
                        
                        # Case 2: Nested structure (modality -> key -> stats)
                        else:
                            for modality in ['low_dim', 'proprio']:
                                if modality in obs_normalization_stats and target_key in obs_normalization_stats[modality]:
                                    stats = obs_normalization_stats[modality][target_key]
                                    break
                        
                        if stats is not None:
                            mean = stats['mean']
                            std = stats['std']
                            
                            # Debug: Print stats and raw prediction
                            if i % 50 == 0:
                                print(f"\n[Frame {i}] Denormalization Info:")
                                print(f"  Raw Pred[0]: {pred_traj[0]}")
                                # Ensure mean/std printed cleanly
                                mean_val = mean.cpu().numpy() if isinstance(mean, torch.Tensor) else mean
                                std_val = std.cpu().numpy() if isinstance(std, torch.Tensor) else std
                                print(f"  Mean: {mean_val[:3] if len(mean_val) >= 3 else mean_val}")
                                print(f"  Std: {std_val[:3] if len(std_val) >= 3 else std_val}")
                            
                            # Convert to numpy if needed
                            if isinstance(mean, torch.Tensor):
                                mean = mean.cpu().numpy()
                            if isinstance(std, torch.Tensor):
                                std = std.cpu().numpy()
                            
                            # Ensure mean and std are numpy arrays
                            mean = np.array(mean)
                            std = np.array(std)
                            
                            # Reshape to match pred_traj shape (10, 3)
                            if mean.shape == (30,) or mean.size == 30:
                                mean = mean.reshape(10, 3)
                            if std.shape == (30,) or std.size == 30:
                                std = std.reshape(10, 3)
                            
                            # Denormalize: x = normalized * std + mean
                            pred_traj = pred_traj * std + mean
                            
                            if i % 50 == 0:
                                print(f"  Denorm Pred[0]: {pred_traj[0]}")
                        else:
                            if i % 50 == 0:
                                print(f"\n[Frame {i}] Stats found but target key '{target_key}' missing.")
                    
                    # Note: pred_traj is now in absolute world coordinates (same as robot0_eef_pos)
                    # If there's an offset, it reflects the model's actual prediction performance
            
            # Ground Truth Future
            # gt_end = min(i + 1 + future_horizon, total_frames)
            # gt_traj = robot0_eef_pos[i+1 : gt_end]
            # Use point_gap to match training data
            gt_indices = [min(i + k * point_gap, total_frames - 1) for k in range(1, future_horizon + 1)]
            gt_traj = robot0_eef_pos[gt_indices]
            
            # Debug: Print prediction vs ground truth comparison
            if i % 10 == 0: # Print every 10 frames
                print(f"\nFrame {i}:")
                print(f"  Current EEF pos: [{curr_pos[0]:.4f}, {curr_pos[1]:.4f}, {curr_pos[2]:.4f}]")
                
                if pred_traj is not None:
                    print(f"  Pred[0] (t+{point_gap}): [{pred_traj[0][0]:.4f}, {pred_traj[0][1]:.4f}, {pred_traj[0][2]:.4f}]")
                else:
                    print("  Pred[0]: None")
                    
                if len(gt_traj) > 0:
                    print(f"  GT[0]   (t+{point_gap}): [{gt_traj[0][0]:.4f}, {gt_traj[0][1]:.4f}, {gt_traj[0][2]:.4f}]")
                else:
                    print("  GT[0]: None")

                if pred_traj is not None and len(gt_traj) > 0:
                    offset = pred_traj[0] - gt_traj[0]
                    print(f"  Offset  (Pred-GT): [{offset[0]:.4f}, {offset[1]:.4f}, {offset[2]:.4f}]")
                    print(f"  Offset Magnitude : {np.linalg.norm(offset):.4f}")
                    
                    # Check delta from current position
                    delta_pred = pred_traj[0] - curr_pos
                    delta_gt = gt_traj[0] - curr_pos
                    print(f"  Delta Pred (from curr): [{delta_pred[0]:.4f}, {delta_pred[1]:.4f}, {delta_pred[2]:.4f}]")
                    print(f"  Delta GT   (from curr): [{delta_gt[0]:.4f}, {delta_gt[1]:.4f}, {delta_gt[2]:.4f}]")
            
            # Visualization
            # Convert to BGR for OpenCV
            vis_img = curr_img.copy()
            if vis_img.shape[0] == 3: # CHW -> HWC
                vis_img = vis_img.transpose(1, 2, 0)
            vis_img = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)
            
            # 1. Draw GT Future (Blue Gradient)
            if len(gt_traj) > 0:
                gt_pts_2d = project_points_to_image(gt_traj, CAM_POS, CAM_LOOKAT, CAM_FOV, W, H)
                # Blue: (255, 0, 0) in RGB -> (255, 0, 0) in BGR?? No, Blue is (0, 0, 255) BGR
                # Wait, show_future_traj says: FUTURE_POINT_COLOR_BGR = (0, 0, 255) # Red?
                # Usually OpenCV BGR: (0,0,255) is Red. (255,0,0) is Blue.
                # Let's use Distinct Colors.
                # GT = Blue = (255, 0, 0) BGR
                vis_img = draw_trajectory_gradient(vis_img, gt_pts_2d, (255, 0, 0)) 

            # 2. Draw Predicted Future (Red Gradient)
            if pred_traj is not None:
                pred_pts_2d = project_points_to_image(pred_traj, CAM_POS, CAM_LOOKAT, CAM_FOV, W, H)
                # Pred = Red = (0, 0, 255) BGR
                vis_img = draw_trajectory_gradient(vis_img, pred_pts_2d, (0, 0, 255))

            # 3. Draw Current Position (Green Opaque)
            curr_pt_2d = project_points_to_image([curr_pos], CAM_POS, CAM_LOOKAT, CAM_FOV, W, H)
            if curr_pt_2d:
                cv2.circle(vis_img, curr_pt_2d[0], 1, (0, 255, 0), -1) # Green
            
            # Resize for display
            vis_img = cv2.resize(vis_img, (256, 256), interpolation=cv2.INTER_NEAREST)
            
            # Add Legend
            cv2.putText(vis_img, "Red: Pred", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.putText(vis_img, "Blue: GT", (5, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            cv2.putText(vis_img, "Green: Curr", (5, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Convert back to RGB for imageio
            vis_img_rgb = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)
            writer.append_data(vis_img_rgb)
            
            if i % 50 == 0:
                print(f"Processed frame {i}/{total_frames}")

    writer.close()
    print(f"Video saved to {args.output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_prompt", type=str, default="/home/yujp/Genesis/scripts/XBOX_control/dataCollect/demo/demo_cube/image_demo_local.hdf5", help="Path to HDF5")
    parser.add_argument("--agent", type=str, default="/home/yujp/Genesis/scripts/XBOX_control/dataCollect/demo/demo_multitask_cube/highlevel/test/20251229010047/models/model_epoch_128_best_validation_-73.84264831542968.pth", help="Model path (using best validation model)")
    parser.add_argument("--output", type=str, default="/home/yujp/MimicPlay/training_result/highlevel_vis_v2.mp4", help="Output video path")
    parser.add_argument("--skip", type=int, default=1, help="Frame skip")
    parser.add_argument("--debug", action="store_true", help="Print debug information about predictions vs ground truth")
    args = parser.parse_args()
    
    visualize_planner(args)
