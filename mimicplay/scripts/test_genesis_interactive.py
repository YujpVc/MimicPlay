#!/usr/bin/env python
"""
Interactive test script for Genesis environment actions.
Directly uses numpy arrays for actions.
"""

import sys
import os
import time
import numpy as np
import cv2
import json

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

try:
    # Import GenesisEnvWrapper from specific path
    sys.path.append('/home/yujp/robomimic')
    from robomimic.envs.env_genesis import GenesisEnvWrapper
except ImportError:
    print("Error: Could not import robomimic.envs.env_genesis.")
    sys.exit(1)

def test_genesis_interactive():
    # 1. Initialize Environment
    print("\n1. Initializing Genesis environment...")
    
    # Configuration
    env_config = {
        "env_name": "Desk_Environment",
        "type": 4,
        "robots": ["Panda"],
        "controller_configs": {
            "type": "OSC_POSE",
            "kp": [4500, 4500, 3500, 3500, 2000, 2000, 2000],
            "damping": [450, 450, 350, 350, 200, 200, 200],
        },
        "has_renderer": True,
        "control_freq": 20,
    }

    try:
        env = GenesisEnvWrapper(
            env_name="Desk_Environment",
            env_config=env_config,
            camera_width=640,
            camera_height=480
        )
    except Exception as e:
        print(f"Failed to initialize environment: {e}")
        return

    # 2. Reset to initial pose
    print("\n2. Resetting to initial pose...")
    obs = env.reset()
    start_pos = obs['robot0_eef_pos']
    print(f"   Initial EEF pose: {start_pos}")

    # 3. Define Test Actions (Direct Numpy Arrays)
    # Action Format: [dx, dy, dz, drx, dry, drz, gripper]
    # Scaling Factors: Pos * 400, Rot * 50
    # Example: Move 5cm (0.05m) -> 0.05 * 400 = 20.0
    
    # Pre-calculated scaled actions
    move_1 = np.array([0.0, 0.0, 1, 0.0, 0.0, 0.0, -1.0])      # Move Up 5cm
    move_2 = np.array([1, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0])     # Move Fwd 5cm
    move_3 = np.array([0.0, 1, 0.0, 0.0, 0.0, 0.0, -1.0])   # Move Right 5cm
    move_4 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1, -1.0])       # Rotate Yaw
    
    test_actions = [
        {"name": "test1", "action": move_1},
        {"name": "test2", "action": move_2},
        {"name": "test3", "action": move_3},
        {"name": "test4", "action": move_4},
    ]

    # 4. Execute Test Loop
    print("\n4. Starting interactive test loop...")
    
    for i, test_case in enumerate(test_actions):
        print(f"\n--------------------------------------------------")
        print(f"Test Case {i+1}: {test_case['name']}")
        print(f"Action Array: {test_case['action']}")
        
        user_input = input(f"Press Enter to execute '{test_case['name']}' (or 'q' to quit)... ")
        if user_input.lower() == 'q':
            break
            
        genesis_action = test_case['action']
        
        # We execute the action over multiple steps to simulate a smooth trajectory
        # instead of a single instantaneous jump command.
        steps_to_run = 10
        
        # Calculate action per step
        action_per_step = genesis_action / steps_to_run
        # Restore gripper value (should not be divided)
        action_per_step[-1] = genesis_action[-1]
        
        # Use get_observation to get specific keys like EEF pos
        # get_state()['states'] returns a flat state vector
        obs = env.get_observation()
        if 'robot0_eef_pos' in obs:
             start_step_pos = obs['robot0_eef_pos']
        else:
             # Fallback if key missing, just use zeros to avoid crash
             print("Warning: 'robot0_eef_pos' not found in observation")
             start_step_pos = np.zeros(3)

        print(f"Executing over {steps_to_run} sim steps...")
        
        for _ in range(steps_to_run):
            print(f"Action per step: {action_per_step}")
            print(f"genesis_action: {genesis_action}")
            obs, reward, done, info = env.step(action_per_step)
            # Render human-view
            env.render(mode="human")
        
        # Report Result
        current_pos = obs['robot0_eef_pos']
        displacement = np.linalg.norm(current_pos - start_step_pos)
        print(f"   ✓ Action completed.")
        print(f"   EEF Position: {current_pos}")
        print(f"   Displacement this action: {displacement:.4f} m")
        
        if 'images' in obs:
             # Basic check to see if images are being returned
             print(f"   Image observation keys: {[k for k in obs.keys() if 'image' in k]}")

    print("\n5. Testing complete. Resetting...")
    env.reset()
    print("✓ Finished.")

if __name__ == "__main__":
    test_genesis_interactive()
