import numpy as np
import cv2
from inputs import get_gamepad
from threading import Thread, Lock
from dataclasses import dataclass
from typing import Dict
import h5py
import os
import time
import json
from robomimic.envs.env_genesis import GenesisEnvWrapper
from pyquaternion import Quaternion  # 引入四元数库

# Controller state class
@dataclass
class ControllerState:
    button_states: Dict[str, int]
    analog_states: Dict[str, int]

class ControllerThread(Thread):
    def __init__(self, callback=None):
        super().__init__()
        self.callback = callback
        self.running = True
        self.lock = Lock()
        self.state = ControllerState(
            button_states={
                'BTN_SOUTH': 0, 'BTN_EAST': 0, 'BTN_WEST': 0, 'BTN_NORTH': 0,
                'BTN_START': 0, 'BTN_SELECT': 0, 'BTN_TL': 0, 'BTN_TR': 0,
                'BTN_THUMBL': 0, 'BTN_THUMBR': 0, 'BTN_MODE': 0
            },
            analog_states={
                'ABS_X': 0, 'ABS_Y': 0, 'ABS_RX': 0, 'ABS_RY': 0,
                'ABS_Z': 0, 'ABS_RZ': 0, 'ABS_HAT0X': 0, 'ABS_HAT0Y': 0
            }
        )
        self.event_thread = Thread(target=self.read_events)

    def run(self):
        self.event_thread.start()
        while self.running:
            with self.lock:
                current_state = self.state
            if self.callback:
                self.callback(current_state)
            time.sleep(0.05)

    def read_events(self):
        while self.running:
            try:
                events = get_gamepad()
                with self.lock:
                    self._process_events(events)
            except Exception as e:
                print(f"Error: {e}")
                time.sleep(0.1)

    def _process_events(self, events):
        for event in events:
            if event.ev_type == 'Key':
                self.state.button_states[event.code] = event.state
            elif event.ev_type == 'Absolute':
                self.state.analog_states[event.code] = event.state

    def stop(self):
        self.running = False
        self.event_thread.join()

# Data recording variables
demos = []
demo_count = 0
current_demo = None
last_recorded_pos = None
last_recorded_quat = None
last_recorded_gripper = None

# Initialize GenesisEnvWrapper
env = GenesisEnvWrapper(
    env_name="Franka_Env",
    env_config={
        "env_name": "Franka_Env",
        "type": 4,
        "robots": ["Panda"],
        "controller_configs": {
            "type": "OSC_POSE",
            "kp": [4500, 4500, 3500, 3500, 2000, 2000, 2000],
            "damping": [450, 450, 350, 350, 200, 200, 200],
        },
        "has_renderer": True,
        "control_freq": 20,
    },
    camera_width=640,
    camera_height=480
)

# Control parameters
SPEED_XY = 0.5
SPEED_Z = 0.2
ROT_SPEED = 1.0
DEADZONE = 0.15
DATA_INTERVAL = 0.05  # 20Hz
controller_thread = ControllerThread()
controller_thread.start()

# Data saving path
tmp_dir = f"/home/yujp/MimicPlay/mimicplay/datasets/demo/demo{time.strftime('%Y%m%d%H%M')}"
os.makedirs(tmp_dir, exist_ok=True)

# Time control
sim_time_total = 0.0

def normalize(value, max_value=32768):
    return max(min(value / max_value, 1.0), -1.0)

def apply_deadzone(value, deadzone):
    return 0 if abs(value) < deadzone else value

try:
    gripper_closed = False
    prev_rt_pressed = False
    # prev_y_pressed = False
    prev_x_pressed = False
    # 初始化末端位置和姿态
    current_pos = np.array([0.65, 0.0, 0.3])
    current_quat = np.array([0, 1, 0, 0])  # 初始四元数

    obs = env.reset()
    # 初始获取末端状态
    current_pos = obs['robot0_eef_pos'].copy()
    current_quat = obs['robot0_eef_quat'].copy()

    while True:
        with controller_thread.lock:
            state = controller_thread.state

        # Get controller inputs
        lx = normalize(state.analog_states.get('ABS_X', 0))
        ly = normalize(state.analog_states.get('ABS_Y', 0))
        rx = normalize(state.analog_states.get('ABS_RX', 0))
        ry = normalize(state.analog_states.get('ABS_RY', 0))
        lt = state.analog_states.get('ABS_Z', 0) / 255.0
        rt = state.analog_states.get('ABS_RZ', 0) / 255.0
        lb = state.button_states.get('BTN_TL', 0)
        rb = state.button_states.get('BTN_TR', 0)
        # y_pressed = state.button_states.get('BTN_WEST', 0)
        x_pressed = state.button_states.get('BTN_NORTH', 0)

        # Handle demo recording
        if x_pressed and not prev_x_pressed:
            if current_demo is None:  # Start new demo
                current_demo = {
                    'states': [],
                    'actions': [],
                    'timestamps': [],
                    'demo_id': f"demo_{demo_count}",
                    'start_time': sim_time_total,
                    'next_record_time': sim_time_total,
                }
                demo_count += 1
                print(f"Starting recording {current_demo['demo_id']}...")
                env.reset()
                current_pos = np.array([0.65, 0.0, 0.3])
                current_quat = np.array([0, 1, 0, 0])
                cube_x = np.random.uniform(0.4, 0.7)
                cube_y = np.random.uniform(-0.5, 0.5)
                env.scene.entities[2].set_pos(np.array([cube_x, cube_y, 0.02]))
                obs = env.get_observation()
                current_pos = obs['robot0_eef_pos'].copy()
                current_quat = obs['robot0_eef_quat'].copy()
                last_recorded_pos = None  # 重置
                last_recorded_quat = None
                last_recorded_gripper = None
            else:  # Stop current demo
                demos.append(current_demo)
                print(f"Stopped recording, saved as {current_demo['demo_id']}")
                current_demo = None
                last_recorded_pos = None
                last_recorded_quat = None
                last_recorded_gripper = None
        prev_x_pressed = x_pressed

        # # 修改后的Y键处理逻辑
        # if y_pressed and not prev_y_pressed:
        #     # 设置目标位姿
        #     target_pos = np.array([0.65, 0.0, 0.3])
        #     target_quat = np.array([0, 1, 0, 0])  # 四元数格式需确认
        #
        #     try:
        #         # 计算逆运动学
        #         qpos = env.robot.inverse_kinematics(
        #             link=env.end_effector,
        #             pos=target_pos,
        #             quat=target_quat
        #         )
        #
        #         # 应用关节控制
        #         env.robot.control_dofs_position(qpos[:-2], env.motors_dof)
        #         env.robot.control_dofs_position([0.04, 0.04], env.fingers_dof)
        #
        #         env.step()
        #         # 强制更新观测
        #         print("Returned to initial GraspPose!")
        #
        #     except Exception as e:
        #         print("Return failed")
        #
        #     gripper_closed = False
        #
        # prev_y_pressed = y_pressed

        rt_pressed = rt > 0.1
        if rt_pressed and not prev_rt_pressed:
            gripper_closed = not gripper_closed
        prev_rt_pressed = rt_pressed

        # Apply deadzone
        lx = apply_deadzone(lx, DEADZONE)
        ly = apply_deadzone(ly, DEADZONE)
        rx = apply_deadzone(rx, DEADZONE)
        ry = apply_deadzone(ry, DEADZONE)

        # 计算末端坐标系下的位移增量
        current_q = Quaternion(current_quat)
        local_x = -ly * SPEED_XY * env.env.dt
        local_y = lx * SPEED_XY * env.env.dt
        local_z = lt * SPEED_Z * env.env.dt if not lb else -lt * SPEED_Z * env.env.dt
        local_move = np.array([local_x, local_y, local_z])
        # 转换为世界坐标系
        world_move = current_q.rotate(local_move)
        dx, dy, dz = world_move

        # 计算旋转分量
        if rb:
            # RB按下时，右摇杆X控制Z轴旋转
            d_rz = rx * ROT_SPEED * env.env.dt
            d_rx = 0.0
            d_ry = 0.0
        else:
            # 未按下时，右摇杆控制X和Y轴旋转
            d_rx = -rx * ROT_SPEED * env.env.dt
            d_ry = -ry * ROT_SPEED * env.env.dt
            d_rz = 0.0

        gripper_signal = 1.0 if gripper_closed else -1.0
        action = np.array([dx, dy, dz, d_rx, d_ry, d_rz, gripper_signal])

        # Step the environment
        obs, reward, done, info = env.step(action)
        # 更新末端状态
        current_pos = obs['robot0_eef_pos'].copy()
        current_quat = obs['robot0_eef_quat'].copy()
        env.render(mode="collect")
        sim_time_total += env.env.dt

        # Record data
        if current_demo is not None:
            print(f"sim_time_total: {sim_time_total}, next_record_time: {current_demo['next_record_time']}")  # 调试输出
            while sim_time_total >= current_demo['next_record_time']:
                sim_time_demo = current_demo['next_record_time'] - current_demo['start_time']
                print(f"Recording timestamp: {sim_time_demo:.4f} s")
                print(f"Recorded state: {state_vector.shape}, action: {action.shape}")  # 调试输出

                # State (已在上文修改)
                state_vector = np.concatenate([
                    [sim_time_demo],
                    obs["robot0_joint_pos"],
                    obs["robot0_gripper_qpos"],
                    obs["object"],
                    obs["robot0_joint_vel"],
                    obs["robot0_gripper_qvel"],
                    env.scene.entities[2].get_vel().cpu().numpy().squeeze(),
                    env.scene.entities[2].get_ang().cpu().numpy().squeeze()
                ])
                current_demo['states'].append(state_vector)

                # Action: 计算差值
                if last_recorded_pos is None:  # 第一次记录
                    delta_pos = np.zeros(3)  # 初始动作假设为零
                    delta_rot = np.zeros(3)
                    action_gripper = gripper_signal
                else:
                    delta_pos = current_pos - last_recorded_pos
                    q_last = Quaternion(last_recorded_quat)
                    q_current = Quaternion(current_quat)
                    q_delta = q_current * q_last.inverse
                    delta_rot = np.array(q_delta.axis) * q_delta.angle
                    action_gripper = gripper_signal

                action = np.concatenate([delta_pos, delta_rot, [action_gripper]])
                current_demo['actions'].append(action)
                current_demo['timestamps'].append(sim_time_demo)
                current_demo['next_record_time'] += DATA_INTERVAL

                # 更新上一次记录的状态
                last_recorded_pos = current_pos.copy()
                last_recorded_quat = current_quat.copy()
                last_recorded_gripper = gripper_signal

except KeyboardInterrupt:
    pass
finally:
    if current_demo is not None:
        demos.append(current_demo)
        print(f"Auto-saved unfinished demo: {current_demo['demo_id']}")

    hdf5_path = os.path.join(tmp_dir, "demos.hdf5")
    print(f"Saving {len(demos)} demos to HDF5 file...")
    with h5py.File(hdf5_path, "w") as f:
        grp = f.create_group("data")
        grp.attrs["env"] = "Genesis Franka Environment"
        grp.attrs["env_info"] = json.dumps(env.env_config)
        grp.attrs["repository_version"] = "4.0.0"
        total_samples = 0
        for i, demo in enumerate(demos):
            demo_grp = grp.create_group(f"demo_{i}")
            demo_grp.attrs["num_samples"] = len(demo['states'])
            demo_grp.attrs["timestamps"] = json.dumps(demo['timestamps'])
            demo_grp.create_dataset("states", data=np.array(demo['states']))
            demo_grp.create_dataset("actions", data=np.array(demo['actions']))
            total_samples += len(demo['states'])
        grp.attrs["total"] = total_samples
    print(f"Successfully saved {len(demos)} demos to {hdf5_path}, total samples: {total_samples}")

    controller_thread.stop()
    controller_thread.join()
    env.scene.close()
    cv2.destroyAllWindows()

    print("Program closed")