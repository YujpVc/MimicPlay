"""
Xbox Controller Data Collection for MimicPlay
==============================================

Controls:
---------
1. Left Stick (Position Control):
   - Up/Down: Move Forward/Backward (Y-axis)
   - Left/Right: Move Left/Right (X-axis)

2. Right Stick (Rotation Control):
   - Up/Down: Pitch (Rotate around Y-axis)
   - Left/Right: Roll (Rotate around X-axis)
   - With RB Held: Left/Right controls Yaw (Z-axis)

3. Triggers & Bumpers:
   - LT: Move Up (Z-axis)
   - LB + LT: Move Down (Z-axis)
   - RT: Toggle Gripper Open/Close

4. Buttons:
   - X: Start/Stop Recording Demo
   - Y: Discard current demo or Reset Robot
"""

import numpy as np
import genesis as gs
from pyquaternion import Quaternion
import h5py
import os
import time
import cv2
from inputs import get_gamepad
from threading import Thread, Lock
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import json
import argparse
import sys

sys.path.insert(0, '/home/yujp/robomimic')
from robomimic.envs.env_genesis import GenesisEnvWrapper


# =============================================================================
# Controller Classes
# =============================================================================

@dataclass
class ControllerState:
    """Xbox controller state data class"""
    button_states: Dict[str, int] = field(default_factory=lambda: {
        'BTN_SOUTH': 0, 'BTN_EAST': 0, 'BTN_WEST': 0, 'BTN_NORTH': 0,
        'BTN_START': 0, 'BTN_SELECT': 0, 'BTN_TL': 0, 'BTN_TR': 0,
        'BTN_THUMBL': 0, 'BTN_THUMBR': 0, 'BTN_MODE': 0
    })
    analog_states: Dict[str, int] = field(default_factory=lambda: {
        'ABS_X': 0, 'ABS_Y': 0, 'ABS_RX': 0, 'ABS_RY': 0,
        'ABS_Z': 0, 'ABS_RZ': 0, 'ABS_HAT0X': 0, 'ABS_HAT0Y': 0
    })


class XboxController:
    """Xbox controller handler with background event reading"""
    
    DEADZONE = 0.15
    
    def __init__(self):
        self.lock = Lock()
        self.running = True
        self.state = ControllerState()
        self._event_thread = Thread(target=self._read_events, daemon=True)
        self._event_thread.start()
    
    def _read_events(self):
        """Background thread to read controller events"""
        while self.running:
            try:
                events = get_gamepad()
                with self.lock:
                    for event in events:
                        if event.ev_type == 'Key':
                            self.state.button_states[event.code] = event.state
                        elif event.ev_type == 'Absolute':
                            self.state.analog_states[event.code] = event.state
            except Exception as e:
                print(f"Controller error: {e}")
                time.sleep(0.1)
    
    def get_state(self) -> ControllerState:
        """Get current controller state (thread-safe)"""
        with self.lock:
            return ControllerState(
                button_states=self.state.button_states.copy(),
                analog_states=self.state.analog_states.copy()
            )
    
    def get_normalized_input(self) -> dict:
        """Get normalized and deadzone-applied controller input"""
        state = self.get_state()
        
        def normalize(value, max_value=32768):
            return max(min(value / max_value, 1.0), -1.0)
        
        def apply_deadzone(value):
            return 0.0 if abs(value) < self.DEADZONE else value
        
        return {
            'lx': apply_deadzone(normalize(state.analog_states.get('ABS_X', 0))),
            'ly': apply_deadzone(normalize(state.analog_states.get('ABS_Y', 0))),
            'rx': apply_deadzone(normalize(state.analog_states.get('ABS_RX', 0))),
            'ry': apply_deadzone(normalize(state.analog_states.get('ABS_RY', 0))),
            'lt': state.analog_states.get('ABS_Z', 0) / 255.0,
            'rt': state.analog_states.get('ABS_RZ', 0) / 255.0,
            'lb': state.button_states.get('BTN_TL', 0),
            'rb': state.button_states.get('BTN_TR', 0),
            'x_pressed': state.button_states.get('BTN_NORTH', 0),
            'y_pressed': state.button_states.get('BTN_WEST', 0),
        }
    
    def stop(self):
        """Stop the controller thread"""
        self.running = False
        if self._event_thread.is_alive():
            self._event_thread.join(timeout=1.0)


# =============================================================================
# Data Collector Class
# =============================================================================

class DataCollector:
    """Handles data collection logic for MimicPlay demos"""
    
    # Control parameters
    SPEED_XY = 0.8
    SPEED_Z = 0.075
    ROT_SPEED = 2.5
    DATA_INTERVAL = 0.05  # 20Hz
    SUCCESS_DURATION = 10  # Number of consecutive success checks
    
    def __init__(self, env: GenesisEnvWrapper, save_dir: str):
        self.env = env
        self.save_dir = save_dir
        self.controller = XboxController()
        
        # Recording state
        self.demos: List[dict] = []
        self.current_demo: Optional[dict] = None
        self.demo_count = 0
        self.is_recording = False
        self.success_counter = 0
        
        # Robot state
        self.gripper_closed = False
        self.sim_time_total = 0.0
        
        # Button edge detection
        self._prev_rt_pressed = False
        self._prev_x_pressed = False
        self._prev_y_pressed = False
        
        # Environment config for saving
        self.env_config = {
            "type": 4,
            "env_name": "Desk_Environment",
            "robots": ["Panda"],
            "has_renderer": True,
            "has_offscreen_renderer": False,
            "render_camera": "gripper_cam",
            "control_freq": 20,
        }
        
        os.makedirs(save_dir, exist_ok=True)
    
    def reset_environment(self):
        """Reset environment and sync controller state"""
        obs = self.env.reset()
        # current_pos/quat are already updated by env.reset()
        self.gripper_closed = False
        return obs
    
    def _compute_action(self, inputs: dict) -> np.ndarray:
        """Compute action from controller inputs"""
        dt = self.env.scene.sim_options.dt
        current_q = Quaternion(self.env.current_quat)
        
        # Position delta in local frame, then transform to world frame
        local_x = -inputs['ly'] * self.SPEED_XY * dt
        local_y = inputs['lx'] * self.SPEED_XY * dt
        local_z = inputs['lt'] * self.SPEED_Z * dt
        if inputs['lb']:
            local_z = -local_z
        
        local_move = np.array([local_x, local_y, local_z])
        delta_pos = current_q.rotate(local_move)
        delta_pos = np.clip(delta_pos, -0.0025, 0.0025)
        
        # Rotation delta
        if inputs['rb']:  # Yaw mode
            delta_rot_z = inputs['rx'] * self.ROT_SPEED * dt
            q_rot_local = Quaternion(axis=[0, 0, 1], angle=delta_rot_z)
        else:  # Roll/Pitch mode
            delta_rot_x = -inputs['rx'] * self.ROT_SPEED * dt
            delta_rot_y = -inputs['ry'] * self.ROT_SPEED * dt
            q_rot_local = Quaternion(axis=[1, 0, 0], angle=delta_rot_x) * \
                          Quaternion(axis=[0, 1, 0], angle=delta_rot_y)
        
        # Convert local rotation to world rotation increment
        q_rot_world = current_q * q_rot_local * current_q.inverse
        angle = q_rot_world.angle
        delta_rot = np.array(q_rot_world.axis) * angle if angle > 1e-6 else np.zeros(3)
        delta_rot = np.clip(delta_rot, -0.02, 0.02)
        
        # Gripper action
        gripper_action = 1.0 if self.gripper_closed else -1.0
        
        # Scale position by 100 (will be divided by 100 in env.step)
        return np.concatenate([delta_pos * 400, delta_rot * 50, [gripper_action]])
    
    def _handle_button_events(self, inputs: dict):
        """Handle button press events with edge detection"""
        rt_pressed = inputs['rt'] > 0.1
        x_pressed = inputs['x_pressed']
        y_pressed = inputs['y_pressed']
        
        # RT: Toggle gripper
        if rt_pressed and not self._prev_rt_pressed:
            self.gripper_closed = not self.gripper_closed
        self._prev_rt_pressed = rt_pressed
        
        # Y: Discard demo or reset
        if y_pressed and not self._prev_y_pressed:
            if self.is_recording and self.current_demo is not None:
                print(f"丢弃当前记录: {self.current_demo['demo_id']}")
                self.current_demo = None
                self.is_recording = False
                self.success_counter = 0
            else:
                self.reset_environment()
        self._prev_y_pressed = y_pressed
        
        # X: Start/Stop recording
        if x_pressed and not self._prev_x_pressed:
            if not self.is_recording:
                self._start_recording()
            else:
                self._stop_recording()
        self._prev_x_pressed = x_pressed
    
    def _start_recording(self):
        """Start a new demo recording"""
        self.reset_environment()
        self.current_demo = {
            'states': [],
            'actions': [],
            'timestamps': [],
            'demo_id': f"demo_{self.demo_count}",
            'start_time': self.sim_time_total,
            'next_record_time': self.sim_time_total,
        }
        self.demo_count += 1
        self.is_recording = True
        self.success_counter = 0
        self.gripper_closed = False
        print(f"开始记录 {self.current_demo['demo_id']}...")
    
    def _stop_recording(self, auto_success=False):
        """Stop current demo recording and save"""
        if self.current_demo is not None:
            self.demos.append(self.current_demo)
            reason = "任务成功" if auto_success else "手动停止"
            print(f"{reason}，已保存 {self.current_demo['demo_id']}。按X键开始下一轮记录。")
        self.current_demo = None
        self.is_recording = False
        self.success_counter = 0
    
    def _record_step(self, action: np.ndarray):
        """Record current state and action if conditions are met"""
        if not self.is_recording or self.current_demo is None:
            return
        
        if self.sim_time_total < self.current_demo['next_record_time']:
            return
        
        sim_time_demo = self.current_demo['next_record_time'] - self.current_demo['start_time']
        self.current_demo['next_record_time'] += self.DATA_INTERVAL
        
        # Record state and action
        state_dict = self.env.get_state()
        self.current_demo['states'].append(state_dict['states'])
        self.current_demo['actions'].append(action.copy())
        self.current_demo['timestamps'].append(sim_time_demo)
        
        # Check success condition
        success = self.env.is_success()
        if isinstance(success, dict):
            success = success.get("task", False)
        
        if success:
            self.success_counter += 1
            print(f"任务成功，连续成功次数: {self.success_counter}/{self.SUCCESS_DURATION}")
            if self.success_counter >= self.SUCCESS_DURATION:
                self._stop_recording(auto_success=True)
        else:
            self.success_counter = 0
    
    def _display_cameras(self, obs: dict):
        """Display camera views"""
        def process_image(img):         
            if img is None:
                return np.zeros((480, 640, 3), dtype=np.uint8)
            if len(img.shape) == 3 and img.shape[0] == 3:
                img = img.transpose(1, 2, 0)
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            return cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2BGR)
        
        agentview = process_image(obs.get('agentview_image'))
        eye_in_hand = process_image(obs.get('robot0_eye_in_hand_image'))
        show_cam = process_image(obs.get('show_image'))
        
        combined = np.hstack([agentview, eye_in_hand, show_cam])
        cv2.imshow('Triple Camera View', combined)
        cv2.waitKey(1)
    
    def run(self):
        """Main data collection loop"""
        print("数据采集已启动。按 X 开始/停止记录，Y 丢弃/重置，Ctrl+C 退出。")
        self.reset_environment()
        
        try:
            while True:
                inputs = self.controller.get_normalized_input()
                self._handle_button_events(inputs)
                
                action = self._compute_action(inputs)
                obs, reward, done, info = self.env.step(action)
                self.sim_time_total += self.DATA_INTERVAL
                
                self._record_step(action)
                
                obs_dict = self.env.get_observation()
                self._display_cameras(obs_dict)
                
        except KeyboardInterrupt:
            print("\n收到退出信号...")
        finally:
            self._cleanup()
    
    def _cleanup(self):
        """Save data and cleanup resources"""
        # Save unfinished demo
        if self.current_demo is not None:
            self.demos.append(self.current_demo)
            print(f"自动保存未完成的demo: {self.current_demo['demo_id']}")
        
        # Save to HDF5
        self._save_to_hdf5()
        
        # Cleanup
        self.controller.stop()
        cv2.destroyAllWindows()
        print("程序已关闭")
    
    def _save_to_hdf5(self):
        """Save all demos to HDF5 file"""
        if not self.demos:
            print("没有数据需要保存")
            return
        
        hdf5_path = os.path.join(self.save_dir, "demos.hdf5")
        print(f"保存 {len(self.demos)} 个demo到 {hdf5_path}...")
        
        try:
            with h5py.File(hdf5_path, "w") as f:
                grp = f.create_group("data")
                grp.attrs["env"] = "Desk Environment"
                grp.attrs["env_info"] = json.dumps(self.env_config)
                grp.attrs["repository_version"] = "4.0.0"
                
                total_samples = 0
                for i, demo in enumerate(self.demos):
                    demo_grp = grp.create_group(f"demo_{i}")
                    demo_grp.attrs["num_samples"] = len(demo['states'])
                    demo_grp.attrs["timestamps"] = json.dumps(demo['timestamps'])
                    demo_grp.create_dataset("states", data=np.array(demo['states']))
                    demo_grp.create_dataset("actions", data=np.array(demo['actions']))
                    total_samples += len(demo['states'])
                
                grp.attrs["total"] = total_samples
            
            print(f"成功保存 {len(self.demos)} 个demo，总样本数: {total_samples}")
        except Exception as e:
            print(f"保存失败: {e}")
            raise


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Franka robot data collection with Xbox controller")
    parser.add_argument(
        "--condition_file",
        type=str,
        default="/home/yujp/MimicPlay/mimicplay/scripts/bddl_files/my_success.txt",
        help="Path to the success condition file"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="Directory to save demos (default: auto-generated with timestamp)"
    )
    args = parser.parse_args()
    
    # Create save directory
    if args.save_dir is None:
        args.save_dir = f"/home/yujp/Genesis/scripts/XBOX_control/dataCollect/demo/demo{time.strftime('%Y%m%d%H%M')}"
    
    # Initialize environment
    env_config = {
        "env_name": "Desk_Environment",
        "condition_file": args.condition_file,
    }
    env = GenesisEnvWrapper(env_config=env_config, camera_width=640, camera_height=480)
    
    # Start data collection
    collector = DataCollector(env, args.save_dir)
    collector.run()


if __name__ == "__main__":
    main()
