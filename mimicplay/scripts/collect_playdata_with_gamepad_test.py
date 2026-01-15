"""
A script to collect a batch of human play data (or task specific demonstrations) using a gamepad.

The demonstrations can be played back using the `playback_collected_playdata.py` script.

Gamepad Control Mapping (relative to end-effector frame):
• Left Stick:

    ◦ X-axis: End-effector Y-axis translation (left/right).

    ◦ Y-axis (inverted): End-effector X-axis translation (forward/backward).

• Right Stick:

    ◦ Normal Mode:

        ▪ X-axis: End-effector roll.

        ▪ Y-axis: End-effector pitch.

    ◦ RB Held (Rotation Mode Switch):

        ▪ X-axis: End-effector yaw.

• Triggers:

    ◦ LT: Move end-effector along its negative Z-axis.

    ◦ RT: Toggle gripper (open/close).

• Shoulder Buttons:

    ◦ LB (held): Inverts Z-axis movement (LT moves end-effector along its positive Z-axis).

    ◦ RB (held): Switches to yaw rotation mode for the right stick.

• Face Buttons:

    ◦ X Button: Press once to start recording, press again to stop.

    ◦ Y Button: Reset the robot arm to its initial pose.

"""

import argparse
import datetime
import json
import os
import sys
import threading
import time
from glob import glob

import h5py
import numpy as np
from inputs import get_gamepad
import cv2

sys.path.append("/home/yujp/robosuite")

import robosuite as suite
from robosuite import load_controller_config
from robosuite.wrappers import DataCollectionWrapper, VisualizationWrapper


class Gamepad:
    """
    A class to handle gamepad inputs using the 'inputs' library.
    This class runs a background thread to listen for and process gamepad events.
    """

    def __init__(self, pos_sensitivity=0.1, rot_sensitivity=0.1, deadzone=0.1):
        self.pos_sensitivity = pos_sensitivity
        self.rot_sensitivity = rot_sensitivity
        self.deadzone = deadzone

        # Raw state values from the gamepad
        self.left_stick_x = 0.0
        self.left_stick_y = 0.0
        self.right_stick_x = 0.0
        self.right_stick_y = 0.0
        self.left_trigger = 0.0

        # Button states
        self.lb_pressed = False
        self.rb_pressed = False

        # Event flags for single-press actions
        self.record_stop_event = False
        self.reset_pose_event = False
        self.gripper_toggle_event = False

        # Internal state for event detection
        self._last_rt_value = 0
        self._last_x_value = 0
        self._last_y_value = 0
        self._last_x_press_time = 0
        self.is_recording = False

        self.control_thread = None
        self.thread_running = False

    def start_control(self):
        """Starts the gamepad listening thread."""
        self.thread_running = True
        self.control_thread = threading.Thread(target=self._run_loop)
        self.control_thread.daemon = True
        self.control_thread.start()

    def stop_control(self):
        """Stops the gamepad listening thread."""
        self.thread_running = False
        if self.control_thread is not None:
            self.control_thread.join()

    def _run_loop(self):
        """The main loop for reading gamepad events."""
        print("Looking for gamepad...")
        while self.thread_running:
            try:
                events = get_gamepad()
                for event in events:
                    self._handle_event(event)
            except Exception:
                # Error reading from gamepad, wait before retrying
                time.sleep(1.0)

    def _handle_event(self, event):
        """Processes a single gamepad event."""
        if event.ev_type == 'Absolute':
            if event.code == 'ABS_X':
                self.left_stick_x = self._apply_deadzone(event.state / 32768.0)
            elif event.code == 'ABS_Y':
                self.left_stick_y = self._apply_deadzone(event.state / 32768.0)
            elif event.code == 'ABS_RX':
                self.right_stick_x = self._apply_deadzone(event.state / 32768.0)
            elif event.code == 'ABS_RY':
                self.right_stick_y = self._apply_deadzone(event.state / 32768.0)
            elif event.code == 'ABS_Z':
                self.left_trigger = event.state / 255.0
            elif event.code == 'ABS_RZ':
                current_rt_value = event.state
                if current_rt_value > 128 and self._last_rt_value <= 128:
                    self.gripper_toggle_event = True
                self._last_rt_value = current_rt_value

        elif event.ev_type == 'Key':
            if event.code == 'BTN_TL':
                self.lb_pressed = (event.state == 1)
            elif event.code == 'BTN_TR':
                self.rb_pressed = (event.state == 1)
            elif event.code == 'BTN_NORTH':  # X button on Xbox controller
                if event.state == 1 and self._last_x_value == 0:
                    if time.time() - self._last_x_press_time > 0.5:  # Debounce
                        if self.is_recording:
                            self.record_stop_event = True
                        else:
                            self.is_recording = True
                        self._last_x_press_time = time.time()
                self._last_x_value = event.state

    def _apply_deadzone(self, value):
        """Applies a deadzone to an axis value."""
        return value if abs(value) >= self.deadzone else 0.0


def gamepad2action(device, current_gripper_state):
    """
    Converts gamepad state to a robot action vector based on an intuitive,
    end-effector-relative control scheme.
    """
    if device.gripper_toggle_event:
        current_gripper_state *= -1
        device.gripper_toggle_event = False

    # --- Translation Control (EE Frame) ---
    # Left stick Y (forward/backward) -> EE X-axis
    d_pos_x = -device.left_stick_y * device.pos_sensitivity
    # Left stick X (left/right) -> EE Y-axis
    d_pos_y = -device.left_stick_x * device.pos_sensitivity
    # LT (down) / LB+LT (up) -> EE Z-axis
    d_pos_z = -0.3 * device.left_trigger * device.pos_sensitivity
    if device.lb_pressed:
        d_pos_z *= -1

    d_pos = np.array([d_pos_x, d_pos_y, d_pos_z])

    # --- Rotation Control (EE Frame) ---
    if device.rb_pressed:
        # RB Held: Right stick X controls yaw (rotation around Z-axis)
        # Stick right (+x) -> negative yaw (turn right)
        d_rot = np.array([0, 0, -device.right_stick_x]) * device.rot_sensitivity
    else:
        # Normal Mode: Right stick controls roll (X-axis) and pitch (Y-axis)
        # Stick right (+x) -> negative roll (roll right, CW)
        # Stick up (-y) -> positive pitch (nose up)
        d_rot = np.array([-device.right_stick_x, -device.right_stick_y, 0]) * device.rot_sensitivity

    action = np.concatenate([d_pos, d_rot, [current_gripper_state]])

    return action, current_gripper_state


def collect_human_trajectory(env, device, arm, env_configuration):
    """
    Use the device (gamepad) to collect a demonstration.
    The rollout trajectory is saved to files in npz format.

    Args:
        env (MujocoEnv): environment to control
        device (Gamepad): gamepad device to receive controls from
        arm (str): which arm to control ('right' or 'left')
        env_configuration (str): specified environment configuration
    """

    obs = env.reset()

    def render_views(current_obs):
        """Helper function to render side-by-side views."""
        agent_view = current_obs.get('agentview_image')
        eye_in_hand_view = current_obs.get('robot0_eye_in_hand_image')

        if agent_view is not None and eye_in_hand_view is not None:
            # Images from robosuite are upside down and in RGB format
            agent_view_bgr = cv2.cvtColor(np.flipud(agent_view), cv2.COLOR_RGB2BGR)
            eye_in_hand_view_bgr = cv2.cvtColor(np.flipud(eye_in_hand_view), cv2.COLOR_RGB2BGR)

            # Stack images horizontally
            stitched_image = np.hstack((agent_view_bgr, eye_in_hand_view_bgr))
            cv2.imshow("Demonstration Views (Agent | Eye-in-Hand)", stitched_image)
            cv2.waitKey(1)

    render_views(obs)

    gripper_state = 1.0

    print("\nPress the 'X' button on the gamepad to start recording the demonstration.")
    while not device.is_recording:
        render_views(obs)  # Keep window responsive
        if not device.thread_running:
            print("Gamepad thread stopped. Exiting collection.")
            return
        time.sleep(0.01)
    print("Recording started! Press 'X' again to stop.")

    task_completion_hold_count = -1  # counter to collect 10 timesteps after reaching goal

    # Loop until we get a stop signal from the gamepad or the task completes
    while True:
        # Set active robot
        active_robot = env.robots[0] if env_configuration == "bimanual" else env.robots[arm == "left"]

        # Handle Y button press for pose reset
        if device.reset_pose_event:
            if hasattr(active_robot.controller, 'reset_goal'):
                active_robot.controller.reset_goal()
            device.reset_pose_event = False

        # Get the newest action
        action, gripper_state = gamepad2action(
            device=device, current_gripper_state=gripper_state
        )

        if device.record_stop_event:
            print("Recording stopped by user.")
            device.is_recording = False
            device.record_stop_event = False
            break

        # Run environment step
        obs, reward, done, info = env.step(action)
        render_views(obs)

        # Also break if we complete the task
        if task_completion_hold_count == 0:
            print("Task completed successfully.")
            break

        # state machine to check for having a success for 10 consecutive timesteps
        if env._check_success():
            if task_completion_hold_count > 0:
                task_completion_hold_count -= 1  # latched state, decrement count
            else:
                task_completion_hold_count = 10  # reset count on first success timestep
        else:
            task_completion_hold_count = -1  # null the counter if there's no success

    # cleanup for end of data collection episodes
    cv2.destroyAllWindows()
    env.close()


def gather_demonstrations_as_hdf5(directory, out_dir, env_info):
    """
    Gathers the demonstrations saved in @directory into a
    single hdf5 file.

    The strucure of the hdf5 file is as follows.

    data (group)
        date (attribute) - date of collection
        time (attribute) - time of collection
        repository_version (attribute) - repository version used during collection
        env (attribute) - environment name on which demos were collected

        demo1 (group) - every demonstration has a group
            model_file (attribute) - model xml string for demonstration
            states (dataset) - flattened mujoco states
            actions (dataset) - actions applied during demonstration

        demo2 (group)
        ...

    Args:
        directory (str): Path to the directory containing raw demonstrations.
        out_dir (str): Path to where to store the hdf5 file.
        env_info (str): JSON-encoded string containing environment information,
            including controller and robot info
    """

    hdf5_path = os.path.join(out_dir, "demos.hdf5")
    f = h5py.File(hdf5_path, "w")

    # store some metadata in the attributes of one group
    grp = f.create_group("data")

    num_eps = 0
    env_name = None  # will get populated at some point

    for ep_directory in os.listdir(directory):

        state_paths = os.path.join(directory, ep_directory, "state_*.npz")
        states = []
        actions = []
        success = False

        for state_file in sorted(glob(state_paths)):
            dic = np.load(state_file, allow_pickle=True)
            env_name = str(dic["env"])

            states.extend(dic["states"])
            for ai in dic["action_infos"]:
                actions.append(ai["actions"])
            success = success or dic["successful"]

        if len(states) == 0:
            continue

        # Add collected play data demonstration to dataset
        print("Demonstration has been saved")
        # Delete the last state. This is because when the DataCollector wrapper
        # recorded the states and actions, the states were recorded AFTER playing that action,
        # so we end up with an extra state at the end.
        del states[-1]
        assert len(states) == len(actions)

        num_eps += 1
        ep_data_grp = grp.create_group("demo_{}".format(num_eps))

        # store model xml as an attribute
        xml_path = os.path.join(directory, ep_directory, "model.xml")
        with open(xml_path, "r") as f:
            xml_str = f.read()
        ep_data_grp.attrs["model_file"] = xml_str

        # write datasets for states and actions
        ep_data_grp.create_dataset("states", data=np.array(states))
        ep_data_grp.create_dataset("actions", data=np.array(actions))

    # write dataset attributes (metadata)
    now = datetime.datetime.now()
    grp.attrs["date"] = "{}-{}-{}".format(now.month, now.day, now.year)
    grp.attrs["time"] = "{}:{}:{}".format(now.hour, now.minute, now.second)
    grp.attrs["repository_version"] = suite.__version__
    grp.attrs["env"] = env_name
    grp.attrs["env_info"] = env_info

    f.close()


if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--directory",
        type=str,
        default="/home/yujp/MimicPlay/mimicplay/datasets/demo",
    )
    parser.add_argument("--environment", type=str, default="Libero_Kitchen_Tabletop_Manipulation")
    parser.add_argument("--robots", nargs="+", type=list, default=["Panda"], help="Which robot(s) to use in the env")
    parser.add_argument(
        "--config", type=str, default="single-arm-opposed", help="Specified environment configuration if necessary"
    )
    parser.add_argument("--arm", type=str, default="right", help="Which arm to control (eg bimanual) 'right' or 'left'")
    parser.add_argument("--camera", type=str, default="agentview", help="Which camera to use for collecting demos")
    parser.add_argument(
        "--controller", type=str, default="OSC_POSE", help="Choice of controller. Can be 'IK_POSE' or 'OSC_POSE'"
    )

    parser.add_argument(
        "--num-demonstration",
        type=int,
        default=50,
        help="How much to scale rotation user inputs",
    )
    parser.add_argument("--bddl-file", type=str, default=None)
    parser.add_argument("--task-id", type=int)

    args = parser.parse_args()

    # Get controller config
    controller_config = load_controller_config(default_controller=args.controller)

    # For the OSC_POSE controller, the default configuration is to control in the
    # end-effector frame. We leave this untouched to enable end-effector control.
    # If you want to control in the base frame, you would set:
    # controller_config["control_in_base_frame"] = True

    # Create argument configuration
    config = {
        "env_name": args.environment,
        "robots": args.robots,
        "controller_configs": controller_config,
        "bddl_file_name": args.bddl_file,
    }

    # Check if we're using a multi-armed environment and use env_configuration argument if so
    if "TwoArm" in args.environment:
        config["env_configuration"] = args.config

    # Create environment
    env = suite.make(
        **config,
        has_renderer=False,
        has_offscreen_renderer=True,
        ignore_done=True,
        use_camera_obs=True,
        reward_shaping=True,
        control_freq=20,
        camera_names=["agentview", "robot0_eye_in_hand"],
        camera_widths=512,
        camera_heights=512,
    )

    # Grab reference to controller config and convert it to json-encoded string
    env_info = json.dumps(config)

    # wrap the environment with data collection wrapper
    tmp_directory = "/tmp/{}".format(str(time.time()).replace(".", "_"))
    env = DataCollectionWrapper(env, tmp_directory)

    # initialize device
    device = Gamepad()
    device.start_control()

    # make a new timestamped directory
    now = datetime.datetime.now()
    timestamp_dir_name = now.strftime("demo%Y%m%d%H%M")
    new_dir = os.path.join(args.directory, timestamp_dir_name)
    os.makedirs(new_dir)

    # collect demonstrations
    while True:
        collect_human_trajectory(env, device, args.arm, args.config)
        gather_demonstrations_as_hdf5(tmp_directory, new_dir, env_info)
