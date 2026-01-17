"""
Fairino Robot Data Collection with Force Dimension Controller
Based on dataCollect_mimicplay_env.py and test_ctrl.py

程序说明 (Program Explanation):
这个程序 `data_collect/fairino_dataCollect.py` 是一个用于**采集法奥（Fairino）机器人遥操作数据**的脚本。

流程：
1. **初始化**：连接机器人环境（FairinoEnv），连接力反馈设备（Force Dimension），准备数据保存目录。
2. **主循环**：
    - **获取动作**：读取手柄位置和姿态，转换为机器人控制指令。
    - **执行动作**：发送指令给机器人，获取最新状态和图像。
    - **键盘控制**（需聚焦摄像头窗口）：
        - `x`: 开始/停止录制。
        - `y`: 丢弃当前Demo/重置环境。
        - `q`: 退出程序。
    - **数据记录**：录制状态、动作、时间戳。
    - **画面显示**：显示实时监控画面。
3. **保存**：退出时自动将所有录制的数据保存为 `.hdf5` 文件。
"""

import sys
import os
import time
import numpy as np
import h5py
import json
import cv2
import argparse
from typing import Dict, List, Optional

# Add paths
sys.path.insert(0, '/home/yujp/robomimic')
sys.path.insert(0, '/home/yujp/MimicPlay/data_collect')

from robomimic.envs.env_fairino import EnvFairino, DefaultEnvConfig

from teleop.forcedimension_expert import ForceDimensionExpert

from scipy.spatial.transform import Rotation as R

class DataCollector:
    """Handles data collection logic for Fairino using Force Dimension"""
    
    DATA_INTERVAL = 0.05  # 20Hz (matches env config)
    SUCCESS_DURATION = 10
    SAVE_IMG_SIZE = (84, 84) # (W, H) for MimicPlay training
    
    def __init__(self, env, expert, save_dir: str):
        self.env = env
        self.expert = expert
        self.save_dir = save_dir
        
        # Recording state
        self.demos: List[dict] = []
        self.current_demo: Optional[dict] = None
        self.demo_count = 0
        self.is_recording = False
        self.success_counter = 0
        
        # Simulation time (or real time) tracking
        self.sim_time_total = time.time()
        
        # Button edge detection (Keyboard keys)
        self._prev_key_states = {}
        
        # Ensure save directory exists
        os.makedirs(save_dir, exist_ok=True)
        
        # Environment config for saving
        self.env_config = {
            "env_name": "Fairino_Real_Environment",
            "robot_ip": getattr(env.config, "ROBOT_IP", "Unknown"),
            "control_freq": 20,
            "camera_names": list(env.config.REALSENSE_CAMERAS.keys()) if hasattr(env.config, "REALSENSE_CAMERAS") else []
        }

    def reset_environment(self):
        """Reset environment"""
        print("Resetting environment...")
        obs = self.env.reset()
        info = {}
        return obs, info
    
    def _start_recording(self):
        """Start a new demo recording"""
        # Always reset to ensure consistent start state
        print("Resetting robot to start position (Joint Reset)...")
        # joint_reset=True ensures we go back to the exact joint configuration
        # This includes blocking for user input to confirm object placement
        obs = self.env.reset(joint_reset=True)
        info = {}
        
        self.current_demo = {
            'states': [],
            'actions': [],
            'timestamps': [],
            'images': {}, # Initialize dict for images
            'demo_id': f"demo_{self.demo_count}",
            'start_time': self.sim_time_total,
            'next_record_time': self.sim_time_total,
        }
        
        # Initialize lists for each camera
        for cam_name in self.env_config['camera_names']:
            self.current_demo['images'][cam_name] = []
            
        self.demo_count += 1
        self.is_recording = True
        self.success_counter = 0
        print(f"Start Recording {self.current_demo['demo_id']}...")
        return obs, info
    
    def _stop_recording(self, auto_success=False):
        """Stop current demo recording and save"""
        if self.current_demo is not None:
            self.demos.append(self.current_demo)
            reason = "Task Success" if auto_success else "Manual Stop"
            print(f"{reason}, Saved {self.current_demo['demo_id']}. Press 'x' to record next.")
        self.current_demo = None
        self.is_recording = False
        self.success_counter = 0
    
    def _record_step(self, action: np.ndarray, obs, info):
        """Record current state and action"""
        if not self.is_recording or self.current_demo is None:
            return
            
        # Get State from obs
        # obs is returned by FairinoEnv.step() or FairinoEnv.reset()
        
        # Initialize variables
        joints = None
        tcp_pose = None
        gripper_raw = None

        # Strategy 1: Try getting from 'state' dict (Custom implementation)
        state_dict = obs.get('state', {})
        if state_dict and 'joint_pose' in state_dict:
            joints = state_dict['joint_pose']
            tcp_pose = state_dict['tcp_pose']
            gripper_raw = state_dict['gripper_pose']
            
        # Strategy 2: Try getting from flat obs (Robomimic standard)
        elif 'robot0_joint_pos' in obs:
            joints = obs['robot0_joint_pos']
            if 'robot0_eef_pos' in obs and 'robot0_eef_quat' in obs:
                tcp_pose = np.concatenate([obs['robot0_eef_pos'], obs['robot0_eef_quat']])
            if 'robot0_gripper_qpos' in obs:
                gripper_raw = obs['robot0_gripper_qpos']

        # Check if we got the data
        if joints is None or tcp_pose is None or gripper_raw is None:
            # Debug: Print available keys to help user debug
            print(f"\n[Error] Failed to extract state from observation!")
            print(f"Obs keys: {list(obs.keys())}")
            if 'state' in obs:
                print(f"State dict keys: {list(obs['state'].keys())}")
            raise ValueError("Observation is missing required state data (joints, tcp_pose, or gripper). Cannot record zero-filled data.")
        
        # Handle gripper: if it's 2D (duplicated), take first value; otherwise use as-is
        if isinstance(gripper_raw, np.ndarray) and gripper_raw.shape[0] > 1:
            gripper = np.array([gripper_raw[0]])  # Take first value
        else:
            gripper = np.atleast_1d(gripper_raw)  # Ensure 1D array
        
        # Separate TCP pose into position and quaternion
        if tcp_pose.shape[0] == 6:
            # Handle 6D pose (x, y, z, rx, ry, rz)
            eef_pos = tcp_pose[:3]
            rot_vec = tcp_pose[3:] # rx, ry, rz
            
            # [Fix] Fairino uses Rotation Vector (radians)
            # R.from_rotvec expects a 3D vector where magnitude is angle and direction is axis
            eef_quat = R.from_rotvec(rot_vec).as_quat()
        elif tcp_pose.shape[0] == 7:
            # Standard 7D pose
            eef_pos = tcp_pose[:3]   # Position (x, y, z)
            eef_quat = tcp_pose[3:]  # Quaternion (x, y, z, w)
        else:
             raise ValueError(f"Unexpected tcp_pose shape: {tcp_pose.shape}. Expected 6 or 7.")
        
        # Construct flat state vector
        # Format: [joint_pos(6), eef_pos(3), eef_quat(4), gripper(1)] -> total 14 dims
        # This matches the actual Fairino robot configuration (6-axis arm, 1D gripper)
        state = np.concatenate([joints, eef_pos, eef_quat, gripper]) 

        self.current_demo['states'].append(state)
        self.current_demo['actions'].append(action.copy())
        
        # Debug: Print state info on first record
        if len(self.current_demo['states']) == 1:
            print(f"[Debug] First state recorded - Shape: {state.shape}, Values: {state[:4]}...")
        
        # Record Images
        # obs is returned by env.step(), which calls get_observation()
        # In EnvFairino (wrapper), get_observation returns dict with keys like 'agentview_image'
        # But FairinoEnv returns {'state':..., 'images':...}
        
        # Normalize obs to get images dict
        obs_images = obs.get('images', obs) 

        for cam_name in self.env_config['camera_names']:
            img = None
            if cam_name in obs_images:
                 img = obs_images[cam_name]
            elif cam_name in obs: # Fallback check top level
                 img = obs[cam_name]
            
            if img is not None:
                 # Standardize to (H, W, C) for storage
                 # Check format
                 if len(img.shape) == 3 and img.shape[0] == 3: # C,H,W -> H,W,C
                     img = img.transpose(1, 2, 0)
                 
                 if img.max() <= 1.0: # Float -> Uint8
                     img = (img * 255).astype(np.uint8)
                 
                 # Resize to target size for training
                 img = cv2.resize(img, self.SAVE_IMG_SIZE, interpolation=cv2.INTER_AREA)

                 self.current_demo['images'][cam_name].append(img)
        
        # Store relative time
        sim_time_demo = self.sim_time_total - self.current_demo['start_time']
        self.current_demo['timestamps'].append(sim_time_demo)
        
        # Check success
        # Real robot usually doesn't know success unless specified
        success = False
        if hasattr(self.env, "is_success"):
            success = self.env.is_success()
            if isinstance(success, dict):
                success = success.get("task", False)
        
        if success:
            self.success_counter += 1
            print(f"Success count: {self.success_counter}/{self.SUCCESS_DURATION}")
            if self.success_counter >= self.SUCCESS_DURATION:
                self._stop_recording(auto_success=True)
        else:
            self.success_counter = 0

    def _display_cameras(self, obs: dict):
        """Display camera views"""
        images = []
        
        # Extract images from obs if they exist
        # Check if obs has 'images' key (FairinoEnv structure)
        obs_images = obs.get('images', obs)
        
        for key, value in obs_images.items():
            # Check for image data. 
            # In 'images' dict, all values should be images.
            # In top-level obs, keys usually have 'image' in name.
            is_img = False
            if 'images' in obs and obs['images'] is obs_images:
                # We are iterating the 'images' sub-dict
                is_img = True
            elif 'image' in key:
                is_img = True
            
            if is_img and isinstance(value, np.ndarray):
                img = value
                # Check format
                if len(img.shape) == 3:
                    if img.shape[0] == 3: # CHW -> HWC
                        img = img.transpose(1, 2, 0)
                    if img.max() <= 1.0:
                        img = (img * 255).astype(np.uint8)
                    
                    # Convert RGB to BGR for OpenCV
                    img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2BGR)
                    
                    # Draw camera name
                    cv2.putText(img, key, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    images.append(img)
        
        if images:
            # Concatenate images horizontally
            # Resize to same height if needed
            max_h = max(img.shape[0] for img in images)
            resized_images = []
            for img in images:
                if img.shape[0] != max_h:
                    scale = max_h / img.shape[0]
                    w = int(img.shape[1] * scale)
                    img = cv2.resize(img, (w, max_h))
                resized_images.append(img)
            
            combined = np.hstack(resized_images)
            cv2.imshow('Camera View', combined)
        else:
            # Show a black placeholder so we can capture keyboard events
            blank = np.zeros((200, 600, 3), dtype=np.uint8)
            cv2.putText(blank, "No Camera Feed", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            if self.is_recording:
                cv2.circle(blank, (550, 30), 10, (0, 0, 255), -1) # Recording indicator
            cv2.imshow('Control Panel', blank)

    def run(self):
        """Main data collection loop"""
        print("\nData Collection Started.")
        print("Controls (Keyboard focused on window):")
        print("  'x' : Start/Stop Recording")
        print("  'y' : Discard Current/Reset")
        print("  'q' : Quit")
        print("Force Dimension controls robot.\n")
        
        # Force Joint Reset on startup to ensure safe home position
        print("Resetting robot to safe home position (Joint Space)...")
        obs = self.env.reset(joint_reset=True)
        info = {}
        
        while True:
            # 1. Get Action from Expert
            action, expert_info = self.expert.get_action(obs)
            
            # 2. Record (BEFORE stepping, so we record obs_t and action_t)
            self._record_step(action, obs, info)

            # 3. Step Environment (Transition to State_{t+1})
            obs, reward, done, info = self.env.step(action)
            self.sim_time_total = time.time()
            
            # 4. Handle Keyboard Inputs (Window must be focused)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('x'):
                if not self.is_recording:
                    obs, info = self._start_recording()
                else:
                    self._stop_recording()
            elif key == ord('y'):
                if self.is_recording and self.current_demo is not None:
                    print(f"Discarding {self.current_demo['demo_id']}")
                    self.current_demo = None
                    self.is_recording = False
                    self.success_counter = 0
                else:
                    obs, info = self.reset_environment()
            
            # 5. Display (Display the NEW observation)
            self._display_cameras(obs)
            
            # Display action status in terminal (optional, like test_ctrl.py)
            # gripper_angle = expert_info.get("gripper_angle", 0)
            # print(f"Action: {action[:3]} G: {action[6]} A: {gripper_angle}", end='\r')

        self._cleanup()

    def _cleanup(self):
        """Save and cleanup"""
        if self.current_demo is not None:
            self.demos.append(self.current_demo)
            print(f"Auto-saved incomplete demo: {self.current_demo['demo_id']}")
        
        self._save_to_hdf5()
        
        # Close resources
        self.expert.close()
        cv2.destroyAllWindows()
        print("Done.")

    def _save_to_hdf5(self):
        """Save to HDF5 in MimicPlay-compatible format"""
        if not self.demos:
            print("No data to save.")
            return
            
        hdf5_path = os.path.join(self.save_dir, "demos.hdf5")
        print(f"Saving {len(self.demos)} demos to {hdf5_path}...")
        
        with h5py.File(hdf5_path, "w") as f:
            grp = f.create_group("data")
            
            # MimicPlay-compatible metadata
            env_meta = {
                "type": 4,  # Custom type for real robot
                "env_name": "Fairino_Real_Robot",
                "env_version": "1.0.0",
                "env_kwargs": {
                    "env_name": "Fairino_FR5",
                    "robots": ["Fairino_FR5"],
                    "control_freq": 20,
                    "has_renderer": False,
                    "has_offscreen_renderer": True,
                    "use_camera_obs": True,
                    "camera_names": self.env_config['camera_names'],
                    "camera_heights": self.SAVE_IMG_SIZE[1],
                    "camera_widths": self.SAVE_IMG_SIZE[0],
                }
            }
            
            grp.attrs["env"] = "Fairino_Real_Robot"
            grp.attrs["env_info"] = json.dumps(self.env_config)
            grp.attrs["env_args"] = json.dumps(env_meta, indent=4)
            grp.attrs["repository_version"] = "1.0.0"
            
            total_samples = 0
            for i, demo in enumerate(self.demos):
                demo_grp = grp.create_group(f"demo_{i}")
                num_samples = len(demo['states'])
                
                # Attributes
                demo_grp.attrs["num_samples"] = num_samples
                demo_grp.attrs["timestamps"] = json.dumps(demo['timestamps'])
                demo_grp.attrs["model_file"] = "real_robot_no_model"  # Dummy for real robot
                
                # Convert lists to arrays
                states_arr = np.array(demo['states'])
                actions_arr = np.array(demo['actions'])
                
                # Core datasets
                demo_grp.create_dataset("states", data=states_arr)
                demo_grp.create_dataset("actions", data=actions_arr)
                
                # Add dones and rewards (required by MimicPlay)
                dones = np.zeros(num_samples, dtype=bool)
                dones[-1] = True  # Last step is done
                demo_grp.create_dataset("dones", data=dones)
                
                rewards = np.zeros(num_samples, dtype=np.float32)
                rewards[-1] = 1.0  # Reward at the end
                demo_grp.create_dataset("rewards", data=rewards)
                
                # Save observations under "obs" group
                obs_grp = demo_grp.create_group("obs")
                
                # 1. Save images
                for cam_name, imgs in demo['images'].items():
                    # Save as (N, H, W, C)
                    img_arr = np.array(imgs, dtype=np.uint8)
                    obs_grp.create_dataset(cam_name, data=img_arr)
                
                # 2. Extract and save low-dim observations from states
                # State format: [joints(6), eef_pos(3), eef_quat(4), gripper(1)]
                # Extract eef_pos (indices 6:9)
                eef_pos = states_arr[:, 6:9]  # (N, 3)
                obs_grp.create_dataset("robot0_eef_pos", data=eef_pos.astype(np.float32))
                
                # Extract eef_quat (indices 9:13)
                eef_quat = states_arr[:, 9:13]  # (N, 4) - [x, y, z, w]
                obs_grp.create_dataset("robot0_eef_quat", data=eef_quat.astype(np.float32))
                
                # Extract gripper (index 13)
                gripper_qpos = states_arr[:, 13:14]  # (N, 1)
                obs_grp.create_dataset("robot0_gripper_qpos", data=gripper_qpos.astype(np.float32))
                
                total_samples += num_samples
            
            grp.attrs["total"] = total_samples
            
        print(f"Saved successfully. Total samples: {total_samples}")
        print("Format: MimicPlay-compatible with dones/rewards/env_args")

def main():
    parser = argparse.ArgumentParser(description="Fairino Data Collection")
    parser.add_argument("--save_dir", type=str, default=None, help="Save directory")
    args = parser.parse_args()
    
    # Default save dir
    if args.save_dir is None:
        # Save in data_collect/demos relative to this script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        args.save_dir = os.path.join(script_dir, "demos", f"demo_{time.strftime('%Y%m%d_%H%M%S')}")
    
    print("="*80)
    print("Fairino Data Collection Initialization")
    print("="*80)
    
    # 1. Initialize Env
    print("[1/2] Initializing Fairino Environment...")
    env_config = DefaultEnvConfig()
    env_config.ROBOT_IP = "192.168.58.6"
    # Ensure camera usage if possible, though test_ctrl.py didn't specify it.
    # We use config from test_ctrl.py
    
    # The EnvFairino constructor signature is:
    # def __init__(self, env_name, ..., **kwargs):
    # It initializes DefaultEnvConfig internally.
    # To pass custom config parameters like ROBOT_IP, we use kwargs.
    
    env = EnvFairino(
        env_name="Fairino_Real_Environment", 
        render=False, 
        render_offscreen=False, 
        use_image_obs=True, 
        robot_ip="192.168.58.6",
        fake_env=False
    )
    # Note: We need to access env.config later, EnvFairino has self.config
    env_config = env.config
    
    print("    Fairino Ready.")

    # 2. Initialize Expert
    print("[2/2] Initializing Force Dimension...")
    expert = ForceDimensionExpert(
        device_id=0, 
        scale_pos=0.5, 
        scale_rot=1.0,
        rot_delta_gain=4.0,
        smooth_rot=True,
        smooth_alpha=0.08,
        use_slerp=True,
        max_pos_action=0.08, 
        max_rot_action=2.0,
        pos_deadzone=0.002, 
        rot_deadzone=0.02,
        use_soft_saturation=True,
        saturation_sharpness=4.0,
        smooth_pos=True,
        pos_smooth_alpha=0.15
    )
    print("    Force Dimension Ready.")
    print(f"    Settings synced with test_forcedimension.py")
    
    # 3. Start Collector
    collector = DataCollector(env, expert, args.save_dir)
    collector.run()

if __name__ == "__main__":
    main()
