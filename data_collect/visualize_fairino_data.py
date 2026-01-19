"""
2D Curve visualization for Fairino data using Rerun
Based on actual data characteristics from diagnose_hdf5.py
Displays time series curves for joints, TCP position, orientation, gripper, actions, and obs data
"""

import argparse
import h5py
import numpy as np
import rerun as rr
import os
import json
from scipy.spatial.transform import Rotation as R


def visualize_hdf5(hdf5_path, demo_index=None, fps=20):
    """
    Visualize HDF5 data with optimized settings
    """
    if not os.path.exists(hdf5_path):
        print(f"Error: File not found: {hdf5_path}")
        return

    f = h5py.File(hdf5_path, 'r')
    data_grp = f['data']
    
    # Get demo keys
    demo_keys = sorted([k for k in data_grp.keys() if k.startswith('demo_')], 
                      key=lambda x: int(x.split('_')[1]))
    
    if not demo_keys:
        print("No demos found.")
        return

    # Filter demos
    if demo_index is not None:
        target_demo = f"demo_{demo_index}"
        demos_to_visualize = [target_demo] if target_demo in demo_keys else []
        if not demos_to_visualize:
            print(f"Demo {demo_index} not found. Available: {[k.split('_')[1] for k in demo_keys]}")
            return
    else:
        demos_to_visualize = demo_keys

    # Initialize rerun
    rr.init(f"Fairino_Data_2D", spawn=True)
    
    # Set up series line styles (one-time setup)
    for demo_key in demos_to_visualize:
        # State series (from states array)
        for j in range(6):
            rr.log(f"{demo_key}/state/joint_{j}", rr.SeriesLines(names=f"Joint {j}"), static=True)
        rr.log(f"{demo_key}/state/tcp_x", rr.SeriesLines(names="State TCP X"), static=True)
        rr.log(f"{demo_key}/state/tcp_y", rr.SeriesLines(names="State TCP Y"), static=True)
        rr.log(f"{demo_key}/state/tcp_z", rr.SeriesLines(names="State TCP Z"), static=True)
        rr.log(f"{demo_key}/state/tcp_roll", rr.SeriesLines(names="State TCP Roll"), static=True)
        rr.log(f"{demo_key}/state/tcp_pitch", rr.SeriesLines(names="State TCP Pitch"), static=True)
        rr.log(f"{demo_key}/state/tcp_yaw", rr.SeriesLines(names="State TCP Yaw"), static=True)
        rr.log(f"{demo_key}/state/gripper", rr.SeriesLines(names="State Gripper"), static=True)
        
        # Obs series (from obs group) - for comparison
        rr.log(f"{demo_key}/obs/eef_pos_x", rr.SeriesLines(names="Obs EEF Pos X", colors=[0, 255, 0]), static=True)
        rr.log(f"{demo_key}/obs/eef_pos_y", rr.SeriesLines(names="Obs EEF Pos Y", colors=[0, 255, 0]), static=True)
        rr.log(f"{demo_key}/obs/eef_pos_z", rr.SeriesLines(names="Obs EEF Pos Z", colors=[0, 255, 0]), static=True)
        rr.log(f"{demo_key}/obs/eef_quat_x", rr.SeriesLines(names="Obs EEF Quat X"), static=True)
        rr.log(f"{demo_key}/obs/eef_quat_y", rr.SeriesLines(names="Obs EEF Quat Y"), static=True)
        rr.log(f"{demo_key}/obs/eef_quat_z", rr.SeriesLines(names="Obs EEF Quat Z"), static=True)
        rr.log(f"{demo_key}/obs/eef_quat_w", rr.SeriesLines(names="Obs EEF Quat W"), static=True)
        rr.log(f"{demo_key}/obs/gripper_qpos", rr.SeriesLines(names="Obs Gripper", colors=[0, 255, 0]), static=True)
        
        # Action series
        rr.log(f"{demo_key}/action/pos_x", rr.SeriesLines(names="Action Pos X", colors=[255, 0, 0]), static=True)
        rr.log(f"{demo_key}/action/pos_y", rr.SeriesLines(names="Action Pos Y", colors=[255, 0, 0]), static=True)
        rr.log(f"{demo_key}/action/pos_z", rr.SeriesLines(names="Action Pos Z", colors=[255, 0, 0]), static=True)
        rr.log(f"{demo_key}/action/rot_x", rr.SeriesLines(names="Action Rot X"), static=True)
        rr.log(f"{demo_key}/action/rot_y", rr.SeriesLines(names="Action Rot Y"), static=True)
        rr.log(f"{demo_key}/action/rot_z", rr.SeriesLines(names="Action Rot Z"), static=True)
        rr.log(f"{demo_key}/action/gripper", rr.SeriesLines(names="Action Gripper", colors=[255, 0, 0]), static=True)

    print(f"\nVisualizing {len(demos_to_visualize)} demo(s) at {fps} FPS")
    print("="*80)

    for demo_key in demos_to_visualize:
        demo_grp = data_grp[demo_key]
        
        states = demo_grp['states'][:]
        actions = demo_grp['actions'][:]
        
        # Get timestamps
        timestamps_str = demo_grp.attrs.get('timestamps', '[]')
        timestamps = json.loads(timestamps_str)
        if len(timestamps) != states.shape[0]:
            timestamps = list(range(states.shape[0]))
        
        # Get obs data
        obs_grp = demo_grp['obs']
        obs_keys = list(obs_grp.keys())
        
        # Separate image keys and low-dim keys
        image_keys = [k for k in obs_keys if 'image' in k]
        lowdim_keys = [k for k in obs_keys if 'image' not in k]
        
        # Check if we have low-dim obs data
        has_lowdim_obs = len(lowdim_keys) > 0
        
        num_samples = states.shape[0]
        duration = timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0
        
        print(f"\n{demo_key}:")
        print(f"  Samples: {num_samples}")
        print(f"  Duration: {duration:.2f}s")
        print(f"  Images: {image_keys}")
        if has_lowdim_obs:
            print(f"  Low-dim obs: {lowdim_keys}")
        
        # Log each timestep with 2D curves only
        for i in range(num_samples):
            # Set timeline for this sample (use only one timeline for simplicity)
            rr.set_time("frame", sequence=i)
            
            # Parse state (format: [joints(6), eef_pos(3), eef_quat(4), gripper(1)])
            joints = states[i, :6]
            eef_pos = states[i, 6:9]
            eef_quat = states[i, 9:13]  # [x, y, z, w]
            gripper = states[i, 13]
            
            # Normalize quaternion
            eef_quat = eef_quat / np.linalg.norm(eef_quat)
            
            # === STATE VISUALIZATION (2D Curves) ===
            
            # 1. Joint angles as scalars
            for j in range(6):
                rr.log(f"{demo_key}/state/joint_{j}", rr.Scalars(float(np.rad2deg(joints[j]))))
            
            # 2. TCP position components
            rr.log(f"{demo_key}/state/tcp_x", rr.Scalars(float(eef_pos[0])))
            rr.log(f"{demo_key}/state/tcp_y", rr.Scalars(float(eef_pos[1])))
            rr.log(f"{demo_key}/state/tcp_z", rr.Scalars(float(eef_pos[2])))
            
            # 3. TCP orientation as euler angles
            try:
                euler = R.from_quat(eef_quat).as_euler('xyz', degrees=True)
                rr.log(f"{demo_key}/state/tcp_roll", rr.Scalars(float(euler[0])))
                rr.log(f"{demo_key}/state/tcp_pitch", rr.Scalars(float(euler[1])))
                rr.log(f"{demo_key}/state/tcp_yaw", rr.Scalars(float(euler[2])))
            except:
                pass
            
            # 4. Gripper state
            rr.log(f"{demo_key}/state/gripper", rr.Scalars(float(gripper)))
            
            # === ACTION VISUALIZATION (2D Curves) ===
            
            action = actions[i]
            action_pos = action[:3]      # Position command
            action_rot = action[3:6]     # Rotation command (euler)
            action_gripper = action[6]   # Gripper command
            
            # Log action scalars
            rr.log(f"{demo_key}/action/pos_x", rr.Scalars(float(action_pos[0])))
            rr.log(f"{demo_key}/action/pos_y", rr.Scalars(float(action_pos[1])))
            rr.log(f"{demo_key}/action/pos_z", rr.Scalars(float(action_pos[2])))
            rr.log(f"{demo_key}/action/rot_x", rr.Scalars(float(action_rot[0])))
            rr.log(f"{demo_key}/action/rot_y", rr.Scalars(float(action_rot[1])))
            rr.log(f"{demo_key}/action/rot_z", rr.Scalars(float(action_rot[2])))
            rr.log(f"{demo_key}/action/gripper", rr.Scalars(float(action_gripper)))
            
            # === OBS DATA VISUALIZATION (if available) ===
            
            if has_lowdim_obs:
                # robot0_eef_pos
                if 'robot0_eef_pos' in obs_grp:
                    obs_eef_pos = obs_grp['robot0_eef_pos'][i]
                    rr.log(f"{demo_key}/obs/eef_pos_x", rr.Scalars(float(obs_eef_pos[0])))
                    rr.log(f"{demo_key}/obs/eef_pos_y", rr.Scalars(float(obs_eef_pos[1])))
                    rr.log(f"{demo_key}/obs/eef_pos_z", rr.Scalars(float(obs_eef_pos[2])))
                
                # robot0_eef_quat
                if 'robot0_eef_quat' in obs_grp:
                    obs_eef_quat = obs_grp['robot0_eef_quat'][i]
                    rr.log(f"{demo_key}/obs/eef_quat_x", rr.Scalars(float(obs_eef_quat[0])))
                    rr.log(f"{demo_key}/obs/eef_quat_y", rr.Scalars(float(obs_eef_quat[1])))
                    rr.log(f"{demo_key}/obs/eef_quat_z", rr.Scalars(float(obs_eef_quat[2])))
                    rr.log(f"{demo_key}/obs/eef_quat_w", rr.Scalars(float(obs_eef_quat[3])))
                
                # robot0_gripper_qpos
                if 'robot0_gripper_qpos' in obs_grp:
                    obs_gripper = obs_grp['robot0_gripper_qpos'][i]
                    rr.log(f"{demo_key}/obs/gripper_qpos", rr.Scalars(float(obs_gripper[0])))
            
            # === CAMERA IMAGES ===
            
            for cam_name in image_keys:
                img = obs_grp[cam_name][i]
                rr.log(f"{demo_key}/camera/{cam_name}", rr.Image(img))
    
    print("\n" + "="*80)
    print("✓ 2D Curve Visualization complete!")
    print("\nData Hierarchy:")
    print("  📊 state/     - Data from states array (joints, tcp, gripper)")
    print("  📊 obs/       - Data from obs group (robot0_eef_pos, eef_quat, gripper_qpos)")
    print("  🎬 action/    - Action commands (position, rotation, gripper)")
    print("  📷 camera/    - Camera images")
    print("\nColor Coding:")
    print("  🔵 Blue   - State data (from states array)")
    print("  🟢 Green  - Obs data (for comparison with state)")
    print("  🔴 Red    - Action data")
    print("\nRerun Tips:")
    print("  - Use timeline slider to navigate through data")
    print("  - Toggle visibility of curves in the left tree panel")
    print("  - Zoom and pan on the time series plots")
    print("  - Compare state vs obs curves to verify data consistency")
    print("  - Adjust playback speed in timeline settings")
    print("="*80)
    
    f.close()


def main():
    parser = argparse.ArgumentParser(description="2D Curve Visualization for Fairino Data (using Rerun)")
    parser.add_argument("hdf5_path", type=str, nargs='?',
                       default="/home/yujp/MimicPlay/data_collect/demos/demo_20260119_180854/demos.hdf5",
                       help="Path to .hdf5 file")
    parser.add_argument("--demo_idx", type=int, default=None,
                       help="Specific demo to visualize (default: all)")
    parser.add_argument("--fps", type=int, default=20,
                       help="Playback FPS (default: 20)")
    
    args = parser.parse_args()
    visualize_hdf5(args.hdf5_path, args.demo_idx, args.fps)


if __name__ == "__main__":
    main()
