"""
Test script for robomimic FairinoEnv
Integrates Action Test + Camera Visualization
"""

import sys
import numpy as np
import cv2
import time

# Add robomimic to path
sys.path.insert(0, '/home/yujp/robomimic')

from robomimic.envs.env_fairino import EnvFairino as FairinoEnv

def show_camera_stream(env, duration=None, stop_on_key=True):
    """
    Helper to run a loop showing camera feed.
    If duration is set, runs for that many seconds.
    If stop_on_key is True, runs until 'q' is pressed.
    Sends zero action to keep robot steady.
    """
    print(f"\n[Stream] Displaying camera feed... {'(Press q to continue)' if stop_on_key else ''}")
    action_stay = np.zeros(7)
    start = time.time()
    
    while True:
        # Execute action
        obs, reward, done, info = env.step(action_stay)
        
        images_to_show = []
        
        # 1. Try getting images from 'images' dict (EnvFairino structure)
        if 'images' in obs:
            for name, img in obs['images'].items():
                if img is None: continue
                # Process image
                # Ensure HWC format
                if len(img.shape) == 3 and img.shape[0] == 3: # CHW -> HWC
                    img = img.transpose(1, 2, 0)
                if img.max() <= 1.0: # Float -> Uint8
                    img = (img * 255).astype(np.uint8)
                
                # Convert to BGR for OpenCV
                bgr_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                # Label
                cv2.putText(bgr_img, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                images_to_show.append(bgr_img)
        
        # 2. Fallback: Check top-level keys if 'images' dict is empty or missing
        elif any('image' in k for k in obs.keys()):
             for key, val in obs.items():
                 if 'image' in key and isinstance(val, np.ndarray):
                     img = val
                     # Ensure HWC
                     if len(img.shape) == 3 and img.shape[0] == 3:
                         img = img.transpose(1, 2, 0)
                     if img.max() <= 1.0:
                         img = (img * 255).astype(np.uint8)
                     
                     bgr_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                     cv2.putText(bgr_img, key, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                     images_to_show.append(bgr_img)

        if images_to_show:
            # Resize and Stack
            h_min = min(img.shape[0] for img in images_to_show)
            resized = []
            for img in images_to_show:
                if img.shape[0] != h_min:
                    scale = h_min / img.shape[0]
                    w = int(img.shape[1] * scale)
                    img = cv2.resize(img, (w, h_min))
                resized.append(img)
            
            combined = np.hstack(resized)
            cv2.imshow("Fairino Camera Feed", combined)
        else:
            # Show placeholder if no images
            blank = np.zeros((200, 400, 3), dtype=np.uint8)
            cv2.putText(blank, "No Camera Feed", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.imshow("Fairino Camera Feed", blank)
        
        # Check exit conditions
        key = cv2.waitKey(1) & 0xFF
        if stop_on_key and key == ord('q'):
            break
        
        if duration and (time.time() - start > duration):
            break

if __name__ == "__main__":

    env = FairinoEnv(
        env_name="Fairino_Real_Environment", 
        render=False, 
        render_offscreen=False, 
        use_image_obs=False
    )
    
    # 1. Test Camera Feed first
    print("\n[Test 1/3] Testing Camera Feed...")
    show_camera_stream(env, stop_on_key=True)
    
    # 2. Joint Reset
    print("\n[Test 2/3] Joint Reset...")
    input("Press Enter to Reset Robot Joints (Check safety!)...")
    env.reset(joint_reset=True)

    # 3. Action Test
    print("\n[Test 3/3] Sending discrete actions (Z-axis move)...")
    action_up = np.array([0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0])
    action_down = np.array([0.0, 0.0, -0.2, 0.0, 0.0, 0.0, 0.0])
    action_left = np.array([0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    action_right = np.array([-0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    # Move Up
    print("Ready to move UP? (Stream running, press 'q' to execute move)")
    show_camera_stream(env, stop_on_key=True) 
    print("Executing Move UP...")
    for _ in range(2): env.step(action_up) # Run a few steps to make movement visible
    
    # Move Down
    print("Ready to move DOWN? (Stream running, press 'q' to execute move)")
    show_camera_stream(env, stop_on_key=True)
    print("Executing Move DOWN...")
    for _ in range(2): env.step(action_down)

    # Move Left
    print("Ready to move LEFT? (Stream running, press 'q' to execute move)")
    show_camera_stream(env, stop_on_key=True)
    print("Executing Move LEFT...")
    for _ in range(2): env.step(action_left)

    # Move Right
    print("Ready to move RIGHT? (Stream running, press 'q' to execute move)")
    show_camera_stream(env, stop_on_key=True)
    print("Executing Move RIGHT...")
    for _ in range(2): env.step(action_right)

    print("\nTest Finished.")
    env.close()
    cv2.destroyAllWindows()
