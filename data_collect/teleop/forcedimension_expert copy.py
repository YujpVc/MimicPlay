import threading
import ctypes
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
import time
import sys
import os

# Add the parent directory of 'forcedimension_core' to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

import forcedimension_core.dhd as dhd
import forcedimension_core.drd as drd

class ForceDimensionExpert:
    def __init__(self, device_id=0, scale_pos=0.8, scale_rot=1.5, 
                 smooth_rot=True, smooth_alpha=0.12, use_slerp=True,
                 max_pos_action=0.15, max_rot_action=0.3,
                 smooth_pos=True, pos_smooth_alpha=0.18,
                 use_soft_saturation=True, saturation_sharpness=2.5,
                 pos_deadzone=0.003, rot_deadzone=0.0003):
        """
        优化的 Force Dimension 专家策略，解决动作饱和和角度平滑问题
        
        Args:
            device_id: Force Dimension device ID
            scale_pos: Position scaling factor (提高以减少饱和，推荐 0.8-1.2)
            scale_rot: Rotation scaling factor (提高以减少饱和，推荐 1.5-2.5)
            smooth_rot: Whether to apply smoothing to rotation output
            smooth_alpha: Rotation smoothing factor (0-1). Lower = smoother but more lag.
                         推荐值: 0.10-0.15 for better smoothness
            use_slerp: Use quaternion SLERP for smoother rotation interpolation
                      (强烈推荐用于模仿学习，避免欧拉角突变)
            max_pos_action: Maximum position action range (meters) for soft saturation
                           增大此值减少饱和，推荐 0.12-0.20
            max_rot_action: Maximum rotation action range (radians) for soft saturation
                           增大此值减少旋转饱和，推荐 0.25-0.40
            smooth_pos: Whether to apply smoothing to position output (推荐开启)
            pos_smooth_alpha: Position smoothing factor (0-1). Lower = smoother.
                             推荐值: 0.15-0.25 for imitation learning
            use_soft_saturation: Use tanh for soft saturation instead of hard clip
                                (强烈推荐，避免硬截断导致的不平滑)
            saturation_sharpness: Sharpness of soft saturation curve
                                 越大越接近线性（推荐 2.0-3.0）
            pos_deadzone: Position deadzone threshold (meters)
                         更小的死区保留更多细微动作（推荐 0.002-0.005）
            rot_deadzone: Rotation deadzone threshold (radians)
                         更小的死区保留更多细微旋转（推荐 0.0002-0.0005）
        """
        self.device_id = device_id
        self.scale_pos = scale_pos
        self.scale_rot = scale_rot
        self.smooth_rot = smooth_rot
        self.smooth_alpha = smooth_alpha
        self.use_slerp = use_slerp
        
        # Action normalization ranges (increased defaults to reduce saturation)
        self.max_pos_action = max_pos_action
        self.max_rot_action = max_rot_action
        
        # Position smoothing parameters
        self.smooth_pos = smooth_pos
        self.pos_smooth_alpha = pos_smooth_alpha
        
        # Soft saturation parameters (使用 tanh 避免硬饱和)
        self.use_soft_saturation = use_soft_saturation
        self.saturation_sharpness = saturation_sharpness
        
        # Deadzone thresholds (reduced to improve smoothness)
        self.pos_deadzone = pos_deadzone
        self.rot_deadzone = rot_deadzone
        
        # Initialize device
        if dhd.getDeviceCount() < 1:
            raise RuntimeError("No Force Dimension devices found")
            
        self.id = dhd.openID(device_id)
        if self.id < 0:
             raise RuntimeError(f"Could not open device {device_id}")
             
        # Initialize DRD (Robotic SDK)
        if drd.openID(device_id) < 0:
             print(f"Warning: drd.openID failed for {device_id}")
        
        if not drd.isInitialized(device_id):
             if drd.autoInit(device_id) < 0:
                 print(f"Warning: drd.autoInit failed for {device_id}")
        
        if drd.start(device_id) < 0:
            print(f"Warning: drd.start failed for {device_id}")

        # Stop regulation to allow free movement with gravity compensation
        drd.stop(True, device_id)
        
        # Initial state variables placeholder
        self.pos_prev_served = np.zeros(3)
        self.mat_prev_served = np.eye(3)
        
        # Smoothing filters for position
        self.pos_smoothed = np.zeros(3)
        self.pos_initialized = False
        
        # Smoothing filters for rotation
        if self.use_slerp:
            # Use quaternion for SLERP (Spherical Linear Interpolation)
            self.rot_smoothed_quat = R.from_euler('xyz', [0, 0, 0])  # Identity rotation
        else:
            # Use euler angles for EMA
            self.rot_smoothed = np.zeros(3)
        self.rot_initialized = False
        
        # Shared state for thread communication
        self.running = True
        self.latest_pos = np.zeros(3)
        self.latest_rot = np.eye(3)
        self.latest_vel = np.zeros(3) # 新增：用于存储速度
        self.latest_buttons = 0
        self.latest_gripper_angle = 0
        self.lock = threading.Lock()
        
        # Start Haptic Thread (Auto-centering logic)
        print("Force Dimension: Starting haptic loop...")
        self.thread = threading.Thread(target=self._haptic_loop, daemon=True)
        self.thread.start()

        # --- 关键修改：等待归中稳定 ---
        print("Force Dimension: Auto-centering... (Waiting 2s)")
        time.sleep(2.0) 

        # --- 关键修改：重置上一帧状态 ---
        # 归中完成后，重新读取当前位置作为“上一帧”，防止第一帧Action出现巨大的跳变
        with self.lock:
            # 更新 prev 为 归中后 的状态
            self.pos_prev_served = self.latest_pos.copy()
            self.mat_prev_served = self.latest_rot.copy()
        
        print("Force Dimension: Ready.")
        
    def _haptic_loop(self):
        """
        Runs at high frequency (~1kHz).
        Includes Spring (P), Integral (I), and Damping (D) to prevent oscillation and correct gravity bias.
        """
        # Constants
        K_spring = 200.0   # N/m (P gain)
        K_damping = 10.0   # N/(m/s) (D gain) - 增加阻尼防止震荡
        K_integral = 80.0  # N/(m*s) (I gain) - 积分项，用于消除重力导致的稳态误差
        K_torsion = 5.0    # Nm/rad
        
        pos = np.zeros(3)
        rot = np.eye(3)
        vel = np.zeros(3)  # Velocity buffer
        integral_error = np.zeros(3) # 积分误差累积
        gripper_ptr = ctypes.pointer(ctypes.c_double(0.0))
        
        loop_dt = 0.001 # 假设 1kHz 循环
        
        while self.running:
            # 1. Read Device State
            if dhd.getPositionAndOrientationFrame(pos, rot, self.id) < 0:
                time.sleep(loop_dt)
                continue
            
            # 读取线速度用于阻尼计算
            dhd.getLinearVelocity(vel, self.id)
            
            dhd.getGripperAngleDeg(gripper_ptr, self.id)
            gripper_angle = int(gripper_ptr.contents.value)
            btn = dhd.getButton(0, self.id)
                
            # 2. Update Shared State
            with self.lock:
                self.latest_pos[:] = pos
                self.latest_rot[:] = rot
                self.latest_vel[:] = vel
                self.latest_buttons = btn
                self.latest_gripper_angle = gripper_angle
            
            # 3. Calculate Forces (PID Control)
            # 积分误差累积 (I项)
            # 只有当位置比较接近中心时才积分，防止大幅度运动时积分过大
            if np.linalg.norm(pos) < 0.05: 
                integral_error += pos * loop_dt
            else:
                # 距离太远时（比如人为拖动），暂时冻结积分或缓慢衰减，防止松手后反弹过猛
                integral_error *= 0.99
            
            # Anti-windup: 限制积分项产生的力不超过一定范围 (例如 3N，足以抵抗重力但不会伤人)
            max_integral_force = 3.0
            max_integral_val = max_integral_force / K_integral
            integral_error = np.clip(integral_error, -max_integral_val, max_integral_val)

            # F = -K_p * x - K_i * sum(x) - K_d * v
            force = -K_spring * pos - K_integral * integral_error - K_damping * vel
            
            # Clamp Force (Safety)
            force_mag = np.linalg.norm(force)
            if force_mag > 15.0: #稍微放宽一点上限给阻尼发挥作用
                force = force * (15.0 / force_mag)
                
            # Orientation Spring (Torque)
            r = R.from_matrix(rot)
            rot_vec = r.as_rotvec()
            torque = -K_torsion * rot_vec
            
            # Clamp Torque (Safety)
            torque_mag = np.linalg.norm(torque)
            if torque_mag > 0.5:
                torque = torque * (0.5 / torque_mag)
                
            # 4. Apply Forces to Device
            dhd.setForceAndTorqueAndGripperForce(
                force[0], force[1], force[2], 
                torque[0], torque[1], torque[2], 
                0.0, 
                self.id
            )
            
            # 1kHz loop rate is handled by DHD usually, but sleep helps if DHD is non-blocking
            time.sleep(0.001)
            
    def get_action(self, obs=None):
        """
        Reads the latest device state and returns the action.
        """
        with self.lock:
            pos_curr = self.latest_pos.copy()
            mat_curr = self.latest_rot.copy()
            gripper_angle = self.latest_gripper_angle
            
        # --- POSITION MAPPING (ABSOLUTE) ---
        raw_pos_input = pos_curr * self.scale_pos
        
        # Apply smoothing filter to position if enabled
        if self.smooth_pos:
            if not self.pos_initialized:
                self.pos_smoothed = raw_pos_input.copy()
                self.pos_initialized = True
            else:
                # Exponential Moving Average (EMA)
                self.pos_smoothed = (self.pos_smooth_alpha * raw_pos_input + 
                                    (1 - self.pos_smooth_alpha) * self.pos_smoothed)
            smoothed_pos_input = self.pos_smoothed
        else:
            smoothed_pos_input = raw_pos_input
        
        # --- ROTATION MAPPING (RELATIVE) ---
        R_curr = mat_curr
        R_prev = self.mat_prev_served
        # Calculate Delta Rotation: R_diff = R_curr @ R_prev.T
        R_diff = R_curr @ np.linalg.inv(R_prev)
        R_diff_rot = R.from_matrix(R_diff)
        
        # Apply smoothing filter to rotation if enabled
        if self.smooth_rot:
            if self.use_slerp:
                # Use Quaternion SLERP for smoother interpolation
                if not self.rot_initialized:
                    # Initialize with identity (no rotation)
                    self.rot_smoothed_quat = R_diff_rot
                    self.rot_initialized = True
                else:
                    # SLERP between previous smoothed and current
                    # scipy's Slerp expects an array of Rotation objects
                    key_rots = R.concatenate([self.rot_smoothed_quat, R_diff_rot])
                    slerp = Slerp([0, 1], key_rots)
                    # Interpolate at alpha position (closer to new value)
                    self.rot_smoothed_quat = slerp(self.smooth_alpha)
                delta_euler = self.rot_smoothed_quat.as_euler('xyz') * self.scale_rot
            else:
                # Use EMA on euler angles (simpler but less smooth)
                delta_euler_raw = R_diff_rot.as_euler('xyz') * self.scale_rot
                if not self.rot_initialized:
                    self.rot_smoothed = delta_euler_raw.copy()
                    self.rot_initialized = True
                else:
                    self.rot_smoothed = (self.smooth_alpha * delta_euler_raw + 
                                        (1 - self.smooth_alpha) * self.rot_smoothed)
                delta_euler = self.rot_smoothed
        else:
            # No smoothing
            delta_euler = R_diff_rot.as_euler('xyz') * self.scale_rot
        
        # Deadzones (使用可配置的死区阈值)
        if np.linalg.norm(smoothed_pos_input) < self.pos_deadzone:
            smoothed_pos_input[:] = 0.0
        if np.linalg.norm(delta_euler) < self.rot_deadzone:
            delta_euler[:] = 0.0

        # Mapping (Position - Absolute, Smoothed)
        # Assuming FD mapping: x(right), y(up), z(back/front)
        # Target mapping: x(forward), y(left), z(up) -> Adjust as per your robot frame
        pos_action = np.zeros(3)
        pos_action[0] = -smoothed_pos_input[1] # -Y -> X
        pos_action[1] = smoothed_pos_input[0]  # X  -> Y
        pos_action[2] = smoothed_pos_input[2]  # Z  -> Z
        
        # Mapping (Rotation - Relative, Smoothed)
        rot_action = np.zeros(3)
        rot_action[0] = -delta_euler[1]
        rot_action[1] = delta_euler[0]
        rot_action[2] = delta_euler[2]
        
        # Normalize to [-1, 1] range
        if self.use_soft_saturation:
            # 使用 tanh 进行软归一化，避免硬饱和
            # tanh(x) 在 [-1, 1] 之间平滑变化，不会突然裁剪
            # saturation_sharpness 控制饱和曲线的陡峭程度（越大越接近线性）
            pos_action_normalized = np.tanh((pos_action / self.max_pos_action) * self.saturation_sharpness)
            rot_action_normalized = np.tanh((rot_action / self.max_rot_action) * self.saturation_sharpness)
        else:
            # 传统的硬裁剪方式
            pos_action_normalized = np.clip(pos_action / self.max_pos_action, -1.0, 1.0)
            rot_action_normalized = np.clip(rot_action / self.max_rot_action, -1.0, 1.0)
        
        # Assemble final action
        action = np.zeros(7)
        action[:3] = pos_action_normalized
        action[3:6] = rot_action_normalized
        
        # Gripper (already in [-1, 1])
        if gripper_angle < 16:
            action[6] = -1.0 
        else:
            action[6] = 1.0
        
        # Update previous served state
        self.pos_prev_served = pos_curr
        self.mat_prev_served = mat_curr
        
        return action, {"gripper_angle": gripper_angle}

    def close(self):
        self.running = False
        if hasattr(self, 'thread') and self.thread.is_alive():
            self.thread.join()
        drd.close(self.id)
        dhd.close(self.id)