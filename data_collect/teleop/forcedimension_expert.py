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
    def __init__(self, device_id=0, scale_pos=1.0, scale_rot=1.0, 
                 smooth_rot=True, smooth_alpha=0.08, use_slerp=True,
                 max_pos_action=0.15, max_rot_action=0.3,
                 smooth_pos=True, pos_smooth_alpha=0.15,
                 use_soft_saturation=True, saturation_sharpness=2.0,
                 pos_deadzone=0.002, rot_deadzone=0.001):
        """
        优化的 Force Dimension 专家策略，解决动作饱和和角度平滑问题
        
        Args:
            device_id: Force Dimension device ID
            scale_pos: Position scaling factor (提高以减少饱和，推荐 0.8-1.2)
            scale_rot: Rotation scaling factor (提高以减少饱和，推荐 1.0-1.5)
            smooth_rot: Whether to apply smoothing to rotation output
            smooth_alpha: Rotation smoothing factor (0-1). Lower = smoother but more lag.
                         推荐值: 0.05-0.10 for better smoothness
            use_slerp: Use quaternion SLERP for smoother rotation interpolation
                      (强烈推荐用于模仿学习，避免欧拉角突变)
            max_pos_action: Maximum position action range (meters) for soft saturation
            max_rot_action: Maximum rotation action range (radians) for soft saturation
            smooth_pos: Whether to apply smoothing to position output (推荐开启)
            pos_smooth_alpha: Position smoothing factor (0-1). Lower = smoother.
            use_soft_saturation: Use tanh for soft saturation instead of hard clip
            saturation_sharpness: Sharpness of soft saturation curve
            pos_deadzone: Position deadzone threshold (meters)
            rot_deadzone: Rotation deadzone threshold (radians)
        """
        self.device_id = device_id
        self.scale_pos = scale_pos
        self.scale_rot = scale_rot
        self.smooth_rot = smooth_rot
        self.smooth_alpha = smooth_alpha
        self.use_slerp = use_slerp
        
        # Action normalization ranges
        self.max_pos_action = max_pos_action
        self.max_rot_action = max_rot_action
        
        # Position smoothing parameters
        self.smooth_pos = smooth_pos
        self.pos_smooth_alpha = pos_smooth_alpha
        
        # Soft saturation parameters
        self.use_soft_saturation = use_soft_saturation
        self.saturation_sharpness = saturation_sharpness
        
        # Deadzone thresholds
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
        
        # Absolute Smoothing State
        # We store the SMOOTHED absolute rotation from the previous step
        self.rot_abs_smoothed_prev = R.from_matrix(np.eye(3))
        
        # Smoothing filters for position
        self.pos_smoothed = np.zeros(3)
        self.pos_initialized = False
        
        self.rot_initialized = False
        
        # Shared state for thread communication
        self.running = True
        self.latest_pos = np.zeros(3)
        self.latest_rot = np.eye(3)
        self.latest_vel = np.zeros(3)
        self.latest_ang_vel = np.zeros(3) # Angular velocity for damping
        self.latest_buttons = 0
        self.latest_gripper_angle = 0
        self.lock = threading.Lock()
        
        # Start Haptic Thread (Auto-centering logic)
        print("Force Dimension: Starting haptic loop...")
        self.thread = threading.Thread(target=self._haptic_loop, daemon=True)
        self.thread.start()

        # --- Wait for auto-centering stability ---
        print("Force Dimension: Auto-centering... (Waiting 2s)")
        time.sleep(2.0) 

        # --- Reset initial state ---
        with self.lock:
            # 更新 prev 为 归中后 的状态
            self.pos_prev_served = self.latest_pos.copy()
            # Initialize smoothed rotation to current actual rotation
            self.rot_abs_smoothed_prev = R.from_matrix(self.latest_rot.copy())
            self.pos_smoothed = self.latest_pos.copy() * self.scale_pos
            self.pos_initialized = True
            self.rot_initialized = True
        
        print("Force Dimension: Ready.")
        
    def _haptic_loop(self):
        """
        Runs at high frequency (~1kHz).
        Includes Spring (P), Integral (I), and Damping (D) to prevent oscillation and correct gravity bias.
        """
        # Constants
        K_spring = 200.0   # N/m (P gain)
        K_damping = 8.0    # N/(m/s) (D gain) - Linear Damping
        K_integral = 50.0  # N/(m*s) (I gain)
        
        # Rotation Constants (Added Damping)
        K_torsion = 3.0    # Nm/rad (Rotational Spring) - Slightly softer
        K_rot_damping = 0.05 # Nm/(rad/s) (Rotational Damping) - NEW: Prevent jitter
        
        pos = np.zeros(3)
        rot = np.eye(3)
        vel = np.zeros(3)  
        ang_vel_deg = np.zeros(3)
        integral_error = np.zeros(3) 
        gripper_ptr = ctypes.pointer(ctypes.c_double(0.0))
        
        loop_dt = 0.001 
        
        while self.running:
            # 1. Read Device State
            if dhd.getPositionAndOrientationFrame(pos, rot, self.id) < 0:
                time.sleep(loop_dt)
                continue
            
            # Read velocities
            dhd.getLinearVelocity(vel, self.id)
            dhd.getAngularVelocityDeg(ang_vel_deg, self.id)
            ang_vel = np.deg2rad(ang_vel_deg)
            
            dhd.getGripperAngleDeg(gripper_ptr, self.id)
            gripper_angle = int(gripper_ptr.contents.value)
            btn = dhd.getButton(0, self.id)
                
            # 2. Update Shared State
            with self.lock:
                self.latest_pos[:] = pos
                self.latest_rot[:] = rot
                self.latest_vel[:] = vel
                self.latest_ang_vel[:] = ang_vel
                self.latest_buttons = btn
                self.latest_gripper_angle = gripper_angle
            
            # 3. Calculate Forces (PID Control)
            # Position Integral
            if np.linalg.norm(pos) < 0.05: 
                integral_error += pos * loop_dt
            else:
                integral_error *= 0.98 # Decay faster when away
            
            max_integral_force = 3.0
            max_integral_val = max_integral_force / K_integral
            integral_error = np.clip(integral_error, -max_integral_val, max_integral_val)

            # Linear Force: F = -Kp*x - Ki*sum(x) - Kd*v
            force = -K_spring * pos - K_integral * integral_error - K_damping * vel
            
            # Safety Clamp Linear Force
            force_mag = np.linalg.norm(force)
            if force_mag > 15.0: 
                force = force * (15.0 / force_mag)
                
            # Rotational Torque (PD Control)
            # Spring to center
            r = R.from_matrix(rot)
            rot_vec = r.as_rotvec() # Rotation vector from identity
            
            # T = -Kp_rot * theta - Kd_rot * omega
            torque = -K_torsion * rot_vec - K_rot_damping * ang_vel
            
            # Safety Clamp Torque
            torque_mag = np.linalg.norm(torque)
            if torque_mag > 0.4:
                torque = torque * (0.4 / torque_mag)
                
            # 4. Apply Forces to Device
            dhd.setForceAndTorqueAndGripperForce(
                force[0], force[1], force[2], 
                torque[0], torque[1], torque[2], 
                0.0, 
                self.id
            )
            
            time.sleep(0.001)
            
    def get_action(self, obs=None):
        """
        Reads the latest device state and returns the action.
        Uses Absolute Orientation Smoothing for smoother rotation control.
        """
        with self.lock:
            pos_curr = self.latest_pos.copy()
            mat_curr = self.latest_rot.copy()
            gripper_angle = self.latest_gripper_angle
            
        # --- POSITION MAPPING (ABSOLUTE) ---
        raw_pos_input = pos_curr * self.scale_pos
        
        # Smoothing Position
        if self.smooth_pos:
            if not self.pos_initialized:
                self.pos_smoothed = raw_pos_input.copy()
                self.pos_initialized = True
            else:
                self.pos_smoothed = (self.pos_smooth_alpha * raw_pos_input + 
                                    (1 - self.pos_smooth_alpha) * self.pos_smoothed)
            smoothed_pos_input = self.pos_smoothed
        else:
            smoothed_pos_input = raw_pos_input
        
        # --- ROTATION MAPPING (ABSOLUTE SMOOTHING -> RELATIVE DELTA) ---
        # 1. Get current absolute rotation
        quat_curr = R.from_matrix(mat_curr)
        
        # 2. Smooth the ABSOLUTE rotation first
        if self.smooth_rot and self.rot_initialized:
            # SLERP from previous smoothed absolute to current raw absolute
            key_rots = R.concatenate([self.rot_abs_smoothed_prev, quat_curr])
            slerp = Slerp([0, 1], key_rots)
            quat_abs_smoothed = slerp(self.smooth_alpha)
        else:
            quat_abs_smoothed = quat_curr
            self.rot_initialized = True

        # 3. Calculate Delta from PREVIOUS SMOOTHED to CURRENT SMOOTHED
        # R_diff = R_curr_smooth @ inv(R_prev_smooth)
        # This gives a consistent velocity that doesn't amplify noise
        R_diff = quat_abs_smoothed * self.rot_abs_smoothed_prev.inv()
        delta_euler = R_diff.as_euler('xyz') * self.scale_rot
        
        # Update state for next step
        self.rot_abs_smoothed_prev = quat_abs_smoothed
        
        # Deadzones
        if np.linalg.norm(smoothed_pos_input) < self.pos_deadzone:
            smoothed_pos_input[:] = 0.0
        if np.linalg.norm(delta_euler) < self.rot_deadzone:
            delta_euler[:] = 0.0

        # Mapping (Position - Absolute, Smoothed)
        # Target: x(forward), y(left), z(up) 
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
            pos_action_normalized = np.tanh((pos_action / self.max_pos_action) * self.saturation_sharpness)
            rot_action_normalized = np.tanh((rot_action / self.max_rot_action) * self.saturation_sharpness)
        else:
            pos_action_normalized = np.clip(pos_action / self.max_pos_action, -1.0, 1.0)
            rot_action_normalized = np.clip(rot_action / self.max_rot_action, -1.0, 1.0)
        
        # Assemble final action
        action = np.zeros(7)
        action[:3] = pos_action_normalized
        action[3:6] = rot_action_normalized
        
        # Gripper
        if gripper_angle < 16:
            action[6] = -1.0 
        else:
            action[6] = 1.0
        
        # Store prev served not strictly needed for rotation logic anymore, 
        # but kept if we revert to other methods
        self.pos_prev_served = pos_curr
        
        return action, {"gripper_angle": gripper_angle}

    def close(self):
        self.running = False
        if hasattr(self, 'thread') and self.thread.is_alive():
            self.thread.join()
        drd.close(self.id)
        dhd.close(self.id)
