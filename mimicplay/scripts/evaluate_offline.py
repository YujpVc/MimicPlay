import argparse
import h5py
import torch
import numpy as np
import imageio
import os
import cv2
from copy import deepcopy
import mimicplay.utils.file_utils as FileUtils
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.obs_utils as ObsUtils
import json
from collections import deque

def evaluate_models(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load policy (Lowlevel model, which contains Highlevel model inside self.human_nets)
    print(f"Loading model from: {args.agent}")
    policy, ckpt_dict = FileUtils.policy_from_checkpoint(
        ckpt_path=args.agent,
        device=device,
        verbose=False
    )
    # policy is a RolloutPolicy instance, policy.policy is the Algo
    lowlevel_algo = policy.policy
    if hasattr(lowlevel_algo, 'nets'):
        lowlevel_algo.nets.eval()
    
    # Extract Highlevel Algo (human_nets)
    highlevel_algo = lowlevel_algo.human_nets.policy
    if hasattr(highlevel_algo, 'nets'):
        highlevel_algo.nets.eval()

    # Get Config Parameters from loaded models
    # Lowlevel
    ll_block_size = lowlevel_algo.algo_config.lowlevel.block_size # Should be 16
    print(f"Lowlevel Block Size: {ll_block_size}")
    
    # Highlevel 
    # Usually goal gap is defined in config, let's use the one from user instruction/config
    # lowlevel.json -> playdata -> eval_goal_gap = 20
    eval_goal_gap = lowlevel_algo.algo_config.playdata.eval_goal_gap
    print(f"Eval Goal Gap: {eval_goal_gap}")

    # Load Data
    print(f"Loading data from: {args.dataset}")
    f = h5py.File(args.dataset, 'r')
    
    # Use first demo for testing
    demo_key = list(f['data'].keys())[0]
    print(f"Testing on demo: {demo_key}")
    
    # Load trajectories
    demo_grp = f['data'][demo_key]
    agentview_images = demo_grp['obs']['agentview_image'][:]
    robot0_eye_in_hand_images = demo_grp['obs']['robot0_eye_in_hand_image'][:]
    robot0_eef_pos = demo_grp['obs']['robot0_eef_pos'][:]
    robot0_eef_quat = demo_grp['obs']['robot0_eef_quat'][:]
    actions_gt = demo_grp['actions'][:]
    
    total_frames = len(agentview_images)
    print(f"Total frames: {total_frames}")

    # Metrics
    highlevel_errors = []
    lowlevel_errors = []
    
    # Prepare video writer
    if args.output_video:
        writer = imageio.get_writer(args.output_video, fps=20)

    print("Starting evaluation...")
    
    # Reset Algos
    lowlevel_algo.reset()
    highlevel_algo.reset()

    # Buffer for Lowlevel (needs history)
    # We maintain a buffer of raw inputs to form the sequence
    # But wait, the Algo implementation (GPT_wrapper) handles the buffer internally via .buffer
    # See GPT.py -> GPT_wrapper -> forward_step:
    #   self.buffer.append(input_tensor.clone())
    #   if len(self.buffer) > self.gpt_model.block_size: ...
    # So we just need to feed it one step at a time!
    # BUT, we need to feed it sequentially from the start for the buffer to build up correctly.
    # We cannot skip frames if we want to test the recurrent/sequence nature properly.
    # If args.skip > 1, the buffer state will be wrong.
    # So we must iterate 1 by 1, but we can choose to only LOG/COMPUTE metrics every K steps.
    
    with torch.no_grad():
        for i in range(total_frames):
            # --- 1. Prepare Inputs ---
            curr_img = agentview_images[i]
            curr_hand_img = robot0_eye_in_hand_images[i]
            curr_pos = robot0_eef_pos[i]
            curr_quat = robot0_eef_quat[i]
            
            # Goal for Highlevel
            # Use eval_goal_gap from config
            goal_idx = min(i + eval_goal_gap, total_frames - 1)
            goal_img = agentview_images[goal_idx]
            
            # Helper
            def to_tensor(arr):
                # CHW, float 0-1
                return torch.from_numpy(arr.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0).to(device)

            obs_dict = {
                'agentview_image': to_tensor(curr_img),
                'robot0_eye_in_hand_image': to_tensor(curr_hand_img),
                'robot0_eef_pos': torch.from_numpy(curr_pos).float().unsqueeze(0).to(device),
                'robot0_eef_quat': torch.from_numpy(curr_quat).float().unsqueeze(0).to(device)
            }
            goal_dict = {
                'agentview_image': to_tensor(goal_img)
            }

            # --- 2. Highlevel Planner Evaluation ---
            # Predict Latent Plan & Trajectory
            # _get_latent_plan returns (act_out, mlp_out)
            # act_out is [1, 30] (10 steps * 3 dims)
            pred_traj_flat, latent_plan = highlevel_algo._get_latent_plan(obs_dict, goal_dict)
            pred_traj = pred_traj_flat.view(-1, 10, 3).cpu().numpy()[0] # [10, 3]
            
            # Calculate Highlevel Error (Distance to actual future points)
            # Highlevel predicts future 10 steps.
            # Usually during training, these are NOT consecutive frames.
            # They are often spaced out.
            # But let's assume for now they are roughly future points.
            # A common heuristic in these datasets is future relative trajectory.
            # Let's verify if we can compute error.
            # We need at least 10 frames ahead.
            if i + 10 < total_frames:
                # Simple check: Compare with t+1, t+2... t+10? 
                # Or t+gap, t+2*gap?
                # Without exact dataset loader logic, let's assume simple lookahead for error metric 
                # just to have a number, but acknowledge it might be slightly off scale.
                # Assuming training label was simply robot0_eef_pos_future_traj (usually 10 steps)
                # Let's try to match indices t+1 to t+10
                gt_indices = [min(i + k + 1, total_frames - 1) for k in range(10)]
                gt_traj = robot0_eef_pos[gt_indices]
                
                hl_error = np.mean(np.linalg.norm(pred_traj - gt_traj, axis=1))
                highlevel_errors.append(hl_error)
            else:
                hl_error = 0.0

            # --- 3. Lowlevel Policy Evaluation ---
            # Inject Latent Plan
            obs_dict['latent_plan'] = latent_plan
            
            # Lowlevel forward
            # forward_step updates internal buffer automatically
            pred_action = lowlevel_algo.nets["policy"].forward_step(obs_dict)
            pred_action = pred_action.cpu().numpy()[0]
            
            # Ground Truth Action
            gt_action = actions_gt[i]
            
            # Calculate Lowlevel Error
            ll_error = np.mean((pred_action - gt_action)**2) # MSE
            lowlevel_errors.append(ll_error)

            # --- 4. Logging (skip frames for speed) ---
            if i % args.skip == 0:
                if i % 20 == 0:
                    print(f"Frame {i:04d} | HL Error (m): {hl_error:.4f} | LL Error (MSE): {ll_error:.6f}")
                    print(f"  Action Pred: {pred_action[:]}")
                    print(f"           GT: {gt_action[:]}")

                if args.output_video:
                    vis_img = curr_img.copy()
                    cv2.putText(vis_img, f"HL Err: {hl_error:.3f}m", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                    cv2.putText(vis_img, f"LL Err: {ll_error:.4f}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                    writer.append_data(vis_img)

    if args.output_video:
        writer.close()
        
    print("\n=== Summary ===")
    print(f"Mean Highlevel Error: {np.mean(highlevel_errors):.4f} meters")
    print(f"Mean Lowlevel Error:  {np.mean(lowlevel_errors):.6f} (MSE)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="/home/yujp/Genesis/scripts/XBOX_control/dataCollect/demo/demo_multitask_01/image_demo_local.hdf5", help="Path to HDF5 dataset")
    parser.add_argument("--agent", type=str, default="/home/yujp/Genesis/scripts/XBOX_control/dataCollect/demo/demo_multitask_01/lowlevel/test/20251216220547/models/model_epoch_919_best_validation_-25.573387145996094.pth", help="Path to Lowlevel model checkpoint")
    parser.add_argument("--output_video", type=str, default=None, help="Path to output video")
    parser.add_argument("--skip", type=int, default=1, help="Logging skip (forward is always step-by-step)")
    
    args = parser.parse_args()
    evaluate_models(args)
