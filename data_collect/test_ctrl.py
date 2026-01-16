import sys
import os
import time
import numpy as np
import threading
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque

# Add paths
sys.path.insert(0, '/home/yujp/robomimic')

# Import Fairino Environment from robomimic (decoupled from hil-serl)
from robomimic.envs.env_fairino import EnvFairino as FairinoEnv, DefaultEnvConfig

# Import Force Dimension Expert
sys.path.insert(0, '/home/yujp/MimicPlay/data_collect')
from teleop.forcedimension_expert import ForceDimensionExpert


class RobotVisualizer:
    def __init__(self, window_size=200):
        self.window_size = window_size
        self.running = True
        
        # Data buffers
        self.time_buffer = deque(maxlen=window_size)
        self.tcp_pos_buffer = [deque(maxlen=window_size) for _ in range(3)]
        self.tcp_quat_buffer = [deque(maxlen=window_size) for _ in range(4)]
        self.action_pos_buffer = [deque(maxlen=window_size) for _ in range(3)]
        
        self.start_time = time.time()
        self.lock = threading.Lock()
        
        # Shared state for visualization
        self.latest_obs = None
        self.latest_action = None

    def update_data(self, obs, action):
        with self.lock:
            self.latest_obs = obs
            self.latest_action = action

    def _animate(self, frame):
        if not self.running: return
        
        with self.lock:
            obs = self.latest_obs
            action = self.latest_action
            
        if obs is None or action is None: return
        
        current_time = time.time() - self.start_time
        self.time_buffer.append(current_time)
        
        # TCP Pos
        eef_pos = obs.get("robot0_eef_pos", np.zeros(3))
        for i in range(3): self.tcp_pos_buffer[i].append(eef_pos[i])
            
        # TCP Quat
        eef_quat = obs.get("robot0_eef_quat", np.zeros(4))
        for i in range(4): self.tcp_quat_buffer[i].append(eef_quat[i])
            
        # Action Pos
        for i in range(3): self.action_pos_buffer[i].append(action[i])
        
        # Update lines
        if len(self.time_buffer) >= 2:
            t = list(self.time_buffer)
            # TCP Pos
            for i, line in enumerate(self.lines_tcp_pos):
                line.set_data(t, list(self.tcp_pos_buffer[i]))
            self.ax_tcp_pos.relim()
            self.ax_tcp_pos.autoscale_view()
            
            # TCP Quat
            for i, line in enumerate(self.lines_tcp_quat):
                line.set_data(t, list(self.tcp_quat_buffer[i]))
            self.ax_tcp_quat.relim()
            self.ax_tcp_quat.autoscale_view()
            
            # Action
            for i, line in enumerate(self.lines_action):
                line.set_data(t, list(self.action_pos_buffer[i]))
            self.ax_action.relim()
            self.ax_action.autoscale_view()

    def start(self):
        # Setup Plot
        self.fig, (self.ax_tcp_pos, self.ax_tcp_quat, self.ax_action) = plt.subplots(3, 1, figsize=(10, 12))
        self.fig.suptitle('Real-time Robot State & Action')
        
        # 1. TCP Position
        self.ax_tcp_pos.set_title('TCP Position (m)')
        self.lines_tcp_pos = [self.ax_tcp_pos.plot([], [], c=c, label=l)[0] for c,l in zip('rgb', ['x','y','z'])]
        self.ax_tcp_pos.legend(loc='upper right')
        self.ax_tcp_pos.grid(True)
        
        # 2. TCP Quaternion
        self.ax_tcp_quat.set_title('TCP Quaternion (xyzw)')
        self.lines_tcp_quat = [self.ax_tcp_quat.plot([], [], c=c, label=l)[0] for c,l in zip('rgbk', ['x','y','z','w'])]
        self.ax_tcp_quat.legend(loc='upper right')
        self.ax_tcp_quat.grid(True)
        
        # 3. Action (Pos)
        self.ax_action.set_title('Action Position Input')
        self.lines_action = [self.ax_action.plot([], [], c=c, label=l)[0] for c,l in zip('rgb', ['ax','ay','az'])]
        self.ax_action.legend(loc='upper right')
        self.ax_action.grid(True)
        
        self.ani = FuncAnimation(self.fig, self._animate, interval=50, blit=False)
        plt.show()

    def stop(self):
        self.running = False
        plt.close(self.fig)


def main():
    print("="*80)
    print("Force Dimension Teleoperation Test for Fairino Robot")
    print("="*80)
    
    # Initialize Fairino Environment
    print("\n[1/3] Initializing Fairino Environment...")
    env_config = DefaultEnvConfig()
    env_config.ROBOT_IP = "192.168.58.6"
    env = FairinoEnv(
        env_name="Fairino_Real_Environment", 
        render=False, 
        render_offscreen=False, 
        use_image_obs=False,
        robot_ip="192.168.58.6",
        fake_env=False
    )
    
    # [FIX] Force a Joint Reset on startup to clear errors and move to home pose safely
    print("Resetting robot to safe home position (Joint Reset)...")
    
    # EnvFairino.reset() returns a single value (obs) or (obs, info)?
    obs = env.reset(joint_reset=True)
    info = {} # Dummy info since env.reset() only returns obs in this implementation
    
    print("    ✓ Fairino environment ready")
    print(f"    - Robot IP: {env_config.ROBOT_IP}")
    print(f"    - Control frequency: 20Hz")
    print(f"    - Servo period: {env_config.CMD_T*1000:.1f}ms")
    
    # Initialize Force Dimension Expert
    print("\n[2/3] Initializing Force Dimension Device...")
    
    # Updated parameters from fairino_dataCollect.py
    scale_pos = 0.5
    scale_rot = 1.0
    smooth_rot = True
    smooth_alpha = 0.3
    use_slerp = True
    max_pos_action = 0.05
    max_rot_action = 0.4
    pos_deadzone = 0.002
    rot_deadzone = 0.2
    
    expert = ForceDimensionExpert(
        device_id=0, 
        scale_pos=scale_pos, 
        scale_rot=scale_rot,
        smooth_rot=smooth_rot,
        smooth_alpha=smooth_alpha,
        use_slerp=use_slerp,
        max_pos_action=max_pos_action,
        max_rot_action=max_rot_action,
        pos_deadzone=pos_deadzone,
        rot_deadzone=rot_deadzone
    )
    print("    ✓ Force Dimension ready")
    print("    - Device auto-centered with haptic feedback")
    print(f"    - Action range: pos=±{max_pos_action}m, rot=±{max_rot_action}rad")
    print(f"    - Deadzone: pos={pos_deadzone}m, rot={rot_deadzone}rad")
    
    # Initialize Visualizer
    print("\n[Visualizer] Starting Real-time Plot...")
    vis = RobotVisualizer()
    
    # Run robot control in a separate thread so plt.show() doesn't block it
    def control_loop():
        # Need to capture 'obs' from outer scope, but simpler to just use a local var
        # initialized from the outer one passed in. However, 'obs' updates in the loop.
        # So we use nonlocal to update the closure variable if needed, 
        # but actually we just need the initial value and then loop.
        nonlocal obs 
        step_count = 0
        
        print("\n[3/3] Starting Teleoperation Loop...")
        print("    - Move the haptic device to control the robot")
        print("    - Close the plot window to exit")
        
        while vis.running:
            # Get action
            action, expert_info = expert.get_action(obs)
            
            # Step Env
            obs, reward, done, info = env.step(action)
            step_count += 1
            
            # Update Visualizer
            vis.update_data(obs, action)
            
            # Terminal Log
            if step_count % 20 == 0:
                eef_pos = obs.get("robot0_eef_pos", np.zeros(3))
                eef_quat = obs.get("robot0_eef_quat", np.zeros(4))
                print(f"TCP: {eef_pos[:3]} | Quat: {eef_quat} | Act: {action[:3]}", end='\r')
            
            # Sleep handled by env.step()

        print("\nStopping...")
        env.close()
        expert.close()
    
    # Start control thread
    ctrl_thread = threading.Thread(target=control_loop, daemon=True)
    ctrl_thread.start()
    
    # Start Plot (Main Thread)
    try:
        vis.start()
    except KeyboardInterrupt:
        pass
    
    vis.stop()
    print("Test completed.")


if __name__ == "__main__":
    main()
