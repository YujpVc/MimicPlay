"""
Test script for Force Dimension controller with real-time visualization
Displays action curves in real-time using matplotlib
"""

import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque

# Add paths to import from teleop module
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from teleop.forcedimension_expert import ForceDimensionExpert


class ActionVisualizer:
    def __init__(self, expert, window_size=200, update_rate=20):
        """
        Args:
            expert: ForceDimensionExpert instance
            window_size: Number of samples to display (rolling window)
            update_rate: Update frequency in Hz
        """
        self.expert = expert
        self.window_size = window_size
        self.update_rate = update_rate
        
        # Data buffers (deque for efficient rolling window)
        self.time_buffer = deque(maxlen=window_size)
        self.pos_buffer = [deque(maxlen=window_size) for _ in range(3)]  # x, y, z
        self.rot_buffer = [deque(maxlen=window_size) for _ in range(3)]  # rx, ry, rz
        self.gripper_buffer = deque(maxlen=window_size)
        
        self.start_time = time.time()
        self.sample_count = 0
        
        # Create figure and subplots
        self.setup_plots()
        
    def setup_plots(self):
        """Setup matplotlib figure and subplots"""
        plt.style.use('seaborn-v0_8-darkgrid')
        self.fig, self.axes = plt.subplots(3, 1, figsize=(12, 9))
        self.fig.suptitle('Force Dimension Action Real-time Visualization', fontsize=14, fontweight='bold')
        
        # Position plot
        self.ax_pos = self.axes[0]
        self.ax_pos.set_title('Position (m)', fontsize=12, fontweight='bold')
        self.ax_pos.set_ylabel('Position')
        self.ax_pos.grid(True, alpha=0.3)
        self.lines_pos = []
        colors_pos = ['r', 'g', 'b']
        labels_pos = ['X', 'Y', 'Z']
        for i, (color, label) in enumerate(zip(colors_pos, labels_pos)):
            line, = self.ax_pos.plot([], [], color=color, label=label, linewidth=2)
            self.lines_pos.append(line)
        self.ax_pos.legend(loc='upper right')
        
        # Rotation plot
        self.ax_rot = self.axes[1]
        self.ax_rot.set_title('Rotation (rad)', fontsize=12, fontweight='bold')
        self.ax_rot.set_ylabel('Rotation')
        self.ax_rot.grid(True, alpha=0.3)
        self.lines_rot = []
        colors_rot = ['orange', 'purple', 'cyan']
        labels_rot = ['RX', 'RY', 'RZ']
        for i, (color, label) in enumerate(zip(colors_rot, labels_rot)):
            line, = self.ax_rot.plot([], [], color=color, label=label, linewidth=2)
            self.lines_rot.append(line)
        self.ax_rot.legend(loc='upper right')
        
        # Gripper plot
        self.ax_gripper = self.axes[2]
        self.ax_gripper.set_title('Gripper State', fontsize=12, fontweight='bold')
        self.ax_gripper.set_xlabel('Time (s)')
        self.ax_gripper.set_ylabel('Gripper')
        self.ax_gripper.set_ylim(-1.5, 1.5)
        self.ax_gripper.grid(True, alpha=0.3)
        self.line_gripper, = self.ax_gripper.plot([], [], 'k-', linewidth=2, label='Gripper')
        self.ax_gripper.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        self.ax_gripper.legend(loc='upper right')
        
        # Add text for statistics
        self.text_stats = self.fig.text(0.02, 0.02, '', fontsize=9, family='monospace')
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        
    def update_data(self):
        """Get new action from expert and update buffers"""
        action, info = self.expert.get_action(None)
        
        current_time = time.time() - self.start_time
        self.time_buffer.append(current_time)
        
        # Position
        for i in range(3):
            self.pos_buffer[i].append(action[i])
        
        # Rotation
        for i in range(3):
            self.rot_buffer[i].append(action[3 + i])
        
        # Gripper
        self.gripper_buffer.append(action[6])
        
        self.sample_count += 1
        
        return action, info
        
    def update_plots(self, frame):
        """Update plot data (called by FuncAnimation)"""
        action, info = self.update_data()
        
        if len(self.time_buffer) < 2:
            return
        
        time_array = np.array(self.time_buffer)
        
        # Update position lines
        for i, line in enumerate(self.lines_pos):
            line.set_data(time_array, np.array(self.pos_buffer[i]))
        self.ax_pos.relim()
        self.ax_pos.autoscale_view()
        
        # Update rotation lines
        for i, line in enumerate(self.lines_rot):
            line.set_data(time_array, np.array(self.rot_buffer[i]))
        self.ax_rot.relim()
        self.ax_rot.autoscale_view()
        
        # Update gripper line
        self.line_gripper.set_data(time_array, np.array(self.gripper_buffer))
        self.ax_gripper.set_xlim(max(0, time_array[-1] - 10), time_array[-1] + 0.5)
        
        # Update statistics
        elapsed = time.time() - self.start_time
        actual_rate = self.sample_count / elapsed if elapsed > 0 else 0
        
        # Calculate current action norms
        pos_norm = np.linalg.norm(action[:3])
        rot_norm = np.linalg.norm(action[3:6])
        gripper_state = "CLOSED" if action[6] < 0 else "OPEN"
        gripper_angle = info.get("gripper_angle", 0)
        
        stats_text = (
            f"Samples: {self.sample_count:6d} | "
            f"Time: {elapsed:6.1f}s | "
            f"Rate: {actual_rate:5.1f} Hz | "
            f"Pos Norm: {pos_norm:6.3f} | "
            f"Rot Norm: {rot_norm:6.3f} | "
            f"Gripper: {gripper_state} ({gripper_angle}°)"
        )
        self.text_stats.set_text(stats_text)
        
        return self.lines_pos + self.lines_rot + [self.line_gripper, self.text_stats]
    
    def run(self):
        """Start the visualization"""
        interval = 1000 / self.update_rate  # milliseconds
        ani = FuncAnimation(
            self.fig, 
            self.update_plots, 
            interval=interval,
            blit=False,
            cache_frame_data=False
        )
        plt.show()


def main():
    print("="*80)
    print("Force Dimension Real-time Visualization")
    print("="*80)
    
    # Configuration
    scale_pos = 0.5    # Match fairino_dataCollect.py
    scale_rot = 1.0
    smooth_rot = True
    smooth_alpha = 0.3
    use_slerp = True
    
    # New Parameters matching fairino_dataCollect.py
    max_pos_action = 0.05
    max_rot_action = 0.4
    pos_deadzone = 0.002
    rot_deadzone = 0.2
    
    print(f"\n[Configuration]")
    print(f"    Position Scale: {scale_pos}")
    print(f"    Rotation Scale: {scale_rot}")
    print(f"    Max Pos Action: {max_pos_action}m")
    print(f"    Max Rot Action: {max_rot_action}rad")
    print(f"    Pos Deadzone:   {pos_deadzone}m")
    print(f"    Rot Deadzone:   {rot_deadzone}rad")
    print(f"    Rotation Smoothing: {'Enabled' if smooth_rot else 'Disabled'}")
    if smooth_rot:
        print(f"    Smoothing Alpha: {smooth_alpha}")
        print(f"    Use SLERP: {'Yes' if use_slerp else 'No'}")
    
    print("\n[1/2] Initializing Force Dimension device...")
    
    try:
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
    except Exception as e:
        print(f"\n❌ Failed to initialize Force Dimension device: {e}")
        print("    Please check:")
        print("    1. Device is connected")
        print("    2. Drivers are installed")
        print("    3. Device permissions are correct")
        return
    
    print("\n[2/2] Device initialized successfully!")
    print("\n" + "="*80)
    print("Starting real-time visualization...")
    print("Close the plot window to exit.")
    print("="*80 + "\n")
    
    try:
        visualizer = ActionVisualizer(expert, window_size=200, update_rate=20)
        visualizer.run()
    except KeyboardInterrupt:
        print("\n\nShutting down...")
    finally:
        expert.close()
        print("Device closed successfully.")


if __name__ == "__main__":
    main()
