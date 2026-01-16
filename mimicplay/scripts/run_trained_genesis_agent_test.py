"""
The main script for evaluating a policy in an environment.

Args:
    agent (str): path to saved checkpoint pth file

    horizon (int): if provided, override maximum horizon of rollout from the one
        in the checkpoint

    env (str): if provided, override name of env from the one in the checkpoint,
        and use it for rollouts

    render (bool): if flag is provided, use on-screen rendering during rollouts

    video_path (str): if provided, render trajectories to this video file path

    video_skip (int): render frames to a video every @video_skip steps

    camera_names (str or [str]): camera name(s) to use for rendering on-screen or to video

    dataset_path (str): if provided, an hdf5 file will be written at this path with the
        rollout data

    dataset_obs (bool): if flag is provided, and @dataset_path is provided, include
        possible high-dimensional observations in output dataset hdf5 file (by default,
        observations are excluded and only simulator states are saved).

    seed (int): if provided, set seed for rollouts

    bddl_file (str): if provided, the task's goal is specified as the symbolic goal in the
         bddl file (several symbolic predicates connected with AND / OR)

    video_prompt (str): if provided, a task video prompt is loaded and used in the
         evaluation rollouts


Example usage:

    # Evaluate a policy with 100 rollouts of maximum horizon 2000 and save the rollouts to a video.
    # Visualize the agentview and wrist cameras during the rollout.

    python scripts/run_trained_agent.py --agent 'path_to_trained_low-level_policy' \
          --n_rollouts 100 --horizon 2000 --seed 0 \
          --bddl_file 'path_to_task_bddl_file' --video_prompt 'path_to_task_video_prompt' \
          --video_path 'eval_rollouts.mp4'

"""
import sys

sys.path.append("/home/yujp/robosuite")
sys.path.append("/home/yujp/robomimic")

import argparse
import json
import os
import h5py
import imageio
import numpy as np
from copy import deepcopy
import time

import torch

import mimicplay.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as RoboFileUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.obs_utils as ObsUtils
from robomimic.envs.env_base import EnvBase
from mimicplay.algo import RolloutPolicy


def _step_with_dataset_first_action(args):
    """
    Sanity check that bypasses the policy.

    Two modes:
    - Single-step (default):
        read an action from a robomimic-format dataset (default: demo_1 step 0)
        reset env to the first recorded state via reset_to
        apply exactly one env.step(action)
        render a few frames so the user can visually verify motion
    - Full playback:
        if --dataset_action_play_all is set (or --dataset_action_video_path is provided),
        reset to the first recorded state, then step through a range of actions and (optionally)
        save a video.
    - reset env to the first recorded state via reset_to
    """
    assert args.dataset_action_hdf5 is not None, "--dataset_action_hdf5 is required"

    write_video = (args.dataset_action_video_path is not None)
    # keep behavior consistent with other scripts: either on-screen OR write video
    assert not (args.render and write_video), "use either --render or --dataset_action_video_path, not both"

    # create env from dataset metadata so it matches the dataset
    env_meta = RoboFileUtils.get_env_metadata_from_dataset(dataset_path=args.dataset_action_hdf5)
    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=args.render,
        render_offscreen=write_video,
    )

    if args.condition_file is not None:
        env.condition_file = args.condition_file
        env._load_condition_file()
        print(f"Success condition loaded from: {env.condition_file}")

    # enable post-processing if available (some envs store images as HWC and convert to CHW when enabled)
    if hasattr(env, "post_process_images"):
        env.post_process_images = True

    demo = args.dataset_action_demo
    start_step = int(args.dataset_action_start_step)
    end_step = args.dataset_action_end_step
    end_step = int(end_step) if end_step is not None else None

    with h5py.File(args.dataset_action_hdf5, "r") as f:
        actions = f[f"data/{demo}/actions"][()]
        states = f[f"data/{demo}/states"][()]
        init_state = dict(states=states[0])

    print(f"[dataset_action_test] dataset: {args.dataset_action_hdf5}")
    print(f"[dataset_action_test] demo: {demo}")
    print(f"[dataset_action_test] total_actions: {actions.shape[0]}")

    obs = env.reset()
    obs = env.reset_to(init_state)

    eef_before = obs.get("robot0_eef_pos", None) if isinstance(obs, dict) else None
    if eef_before is not None:
        print(f"[dataset_action_test] eef_pos before: {eef_before}")

    # choose mode
    play_all = bool(args.dataset_action_play_all or write_video)
    if not play_all:
        # single-step mode
        step_index = int(args.dataset_action_step)
        if step_index < 0 or step_index >= actions.shape[0]:
            raise ValueError(f"step_index {step_index} out of range for actions length {actions.shape[0]}")
        action = actions[step_index]
        print(f"[dataset_action_test] step_index: {step_index}")
        print(f"[dataset_action_test] action: {action}")
        if args.render:
            env.render(mode="human", camera_name=args.camera_names[0])
        next_obs, r, done, _ = env.step(action)
        eef_after = next_obs.get("robot0_eef_pos", None) if isinstance(next_obs, dict) else None
        if eef_after is not None:
            print(f"[dataset_action_test] eef_pos after:  {eef_after}")
        try:
            success = env.is_success().get("task", False)
        except Exception:
            success = False
        print(f"[dataset_action_test] reward={r} done={done} success={success}")
        if args.render:
            for _ in range(int(args.dataset_action_render_frames)):
                env.render(mode="human", camera_name=args.camera_names[0])
    else:
        # full playback mode
        if start_step < 0:
            raise ValueError("start_step must be >= 0")
        if end_step is None:
            end_step = actions.shape[0]
        end_step = min(end_step, actions.shape[0])
        if start_step >= end_step:
            raise ValueError(f"invalid step range: start_step={start_step}, end_step={end_step}")

        print(f"[dataset_action_test] playback range: [{start_step}, {end_step})")
        print(f"[dataset_action_test] camera_names: {args.camera_names}")

        video_writer = None
        if write_video:
            os.makedirs(os.path.dirname(args.dataset_action_video_path), exist_ok=True) if os.path.dirname(args.dataset_action_video_path) else None
            video_writer = imageio.get_writer(args.dataset_action_video_path, fps=int(args.dataset_action_fps))
            print(f"[dataset_action_test] writing video to: {args.dataset_action_video_path}")

        # render helper
        def _render_frame():
            frames = []
            for cam_name in args.camera_names:
                frames.append(env.render(mode="rgb_array", camera_name=cam_name))
            return np.concatenate(frames, axis=1) if len(frames) > 1 else frames[0]

        # optionally write initial frame
        if write_video and (0 % int(args.dataset_action_video_skip) == 0):
            video_writer.append_data(_render_frame())

        done = False
        success = False
        for i in range(start_step, end_step):
            action = actions[i]
            next_obs, r, done, _ = env.step(action)
            if (i - start_step) % int(args.dataset_action_log_every) == 0:
                print(f"[dataset_action_test] step {i}: r={r} done={done} action={action}")
            try:
                success = env.is_success().get("task", False)
            except Exception:
                success = False

            if write_video:
                if (i - start_step + 1) % int(args.dataset_action_video_skip) == 0:
                    video_writer.append_data(_render_frame())

            if done or success:
                print(f"[dataset_action_test] early stop at step {i}: done={done} success={success}")
                break
            obs = next_obs

        if write_video and video_writer is not None:
            video_writer.close()

    # disable post-processing
    if hasattr(env, "post_process_images"):
        env.post_process_images = False

    return


def rollout(policy, env, horizon, render=False, video_writer=None, video_skip=5, return_obs=False, camera_names=None):
    """
    Helper function to carry out rollouts. Supports on-screen rendering, off-screen rendering to a video,
    and returns the rollout trajectory.

    Args:
        policy (instance of RolloutPolicy): policy loaded from a checkpoint
        env (instance of EnvBase): env loaded from a checkpoint or demonstration metadata
        horizon (int): maximum horizon for the rollout
        render (bool): whether to render rollout on-screen
        video_writer (imageio writer): if provided, use to write rollout to video
        video_skip (int): how often to write video frames
        return_obs (bool): if True, return possibly high-dimensional observations along the trajectoryu.
            They are excluded by default because the low-dimensional simulation states should be a minimal
            representation of the environment.
        camera_names (list): determines which camera(s) are used for rendering. Pass more than
            one to output a video with multiple camera views concatenated horizontally.

    Returns:
        stats (dict): some statistics for the rollout - such as return, horizon, and task success
        traj (dict): dictionary that corresponds to the rollout trajectory
    """
    assert isinstance(env, EnvBase)
    assert isinstance(policy, RolloutPolicy)
    assert not (render and (video_writer is not None))
    # Do NOT toggle env.post_process_images here.
    # RolloutPolicy applies ObsUtils.process_obs_dict to ensure consistent formatting.

    policy.start_episode()
    obs = env.reset()
    state_dict = env.get_state()

    # This reset_to call is necessary for some environments for deterministic playback
    obs = env.reset_to(state_dict)

    # Test policy once to catch potential issues with the initial observation
    try:
        _ = policy(ob=obs)
    except Exception as e:
        print(f"Error during initial policy call: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

    results = {}
    video_count = 0
    total_reward = 0.
    traj = dict(actions=[], rewards=[], dones=[], states=[], initial_state_dict=state_dict)
    if return_obs:
        traj.update(dict(obs=[], next_obs=[]))

    try:
        success = False
        for step_i in range(horizon):
            start_time = time.time()

            act = policy(ob=obs)
            print(f"Action at step {step_i}:\n{act}")
            next_obs, r, done, _ = env.step(act)

            elapsed = time.time() - start_time
            print(f"Step {step_i+1}/{horizon} -- Step Duration: {elapsed:.4f} sec")

            total_reward += r
            success = env.is_success()["task"]

            # visualization
            if render:
                env.render(mode="human", camera_name=camera_names[0])
            if video_writer is not None:
                if video_count % video_skip == 0:
                    video_img = []
                    for cam_name in camera_names:
                        video_img.append(env.render(mode="rgb_array", height=512, width=512, camera_name=cam_name))
                    video_img = np.concatenate(video_img, axis=1) # concatenate horizontally
                    video_writer.append_data(video_img)
                video_count += 1

            # collect transition
            traj["actions"].append(act)
            traj["rewards"].append(r)
            traj["dones"].append(done)
            traj["states"].append(state_dict["states"])
            if return_obs:
                traj["obs"].append(ObsUtils.unprocess_obs_dict(obs))
                traj["next_obs"].append(ObsUtils.unprocess_obs_dict(next_obs))

            if done or success:
                break

            # update for next iter
            obs = deepcopy(next_obs)
            state_dict = env.get_state()

    except env.rollout_exceptions as e:
        print(f"WARNING: Caught a rollout exception: {e}")
    except Exception as e:
        print(f"An unexpected exception occurred during rollout: {e}")
        raise

    stats = dict(Return=total_reward, Horizon=(step_i + 1), Success_Rate=float(success))

    if return_obs:
        traj["obs"] = TensorUtils.list_of_flat_dict_to_dict_of_list(traj["obs"])
        traj["next_obs"] = TensorUtils.list_of_flat_dict_to_dict_of_list(traj["next_obs"])

    for k in traj:
        if k == "initial_state_dict":
            continue
        if isinstance(traj[k], dict):
            for kp in traj[k]:
                traj[k][kp] = np.array(traj[k][kp])
        else:
            traj[k] = np.array(traj[k])

    # (no env.post_process_images toggling)

    return stats, traj


def run_trained_agent(args):
    # some arg checking
    write_video = (args.video_path is not None)
    assert not (args.render and write_video)
    if args.render:
        assert len(args.camera_names) == 1

    ckpt_path = args.agent
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)

    # restore policy
    policy, ckpt_dict = FileUtils.policy_from_checkpoint(ckpt_path=ckpt_path, device=device, verbose=False)
    policy.policy.load_eval_video_prompt(args.video_prompt)

    # read rollout settings
    rollout_num_episodes = args.n_rollouts
    rollout_horizon = args.horizon
    if rollout_horizon is None:
        config, _ = FileUtils.config_from_checkpoint(ckpt_dict=ckpt_dict)
        rollout_horizon = config.experiment.C.horizon

    # create environment from saved checkpoint
    env, _ = FileUtils.env_from_checkpoint(
        ckpt_dict=ckpt_dict,
        env_name=args.env,
        render=args.render,
        render_offscreen=(args.video_path is not None),
        verbose=False,
    )

    if args.condition_file is not None:
        env.condition_file = args.condition_file
        env._load_condition_file()
        print(f"Success condition loaded from: {env.condition_file}")

    print(f"Environment Type: {type(env)}")
    print(f"Environment Name: {env.name}")

    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    video_writer = None
    if write_video:
        video_writer = imageio.get_writer(args.video_path, fps=20)

    write_dataset = (args.dataset_path is not None)
    if write_dataset:
        data_writer = h5py.File(args.dataset_path, "w")
        data_grp = data_writer.create_group("data")
        total_samples = 0

    rollout_stats = []
    for i in range(rollout_num_episodes):
        print(f"\n--- Running rollout {i+1} of {rollout_num_episodes} ---")
        stats, traj = rollout(
            policy=policy,
            env=env,
            horizon=rollout_horizon,
            render=args.render,
            video_writer=video_writer,
            video_skip=args.video_skip,
            return_obs=(write_dataset and args.dataset_obs),
            camera_names=args.camera_names,
        )
        rollout_stats.append(stats)

        if write_dataset:
            ep_data_grp = data_grp.create_group(f"demo_{i}")
            ep_data_grp.create_dataset("actions", data=np.array(traj["actions"]))
            ep_data_grp.create_dataset("states", data=np.array(traj["states"]))
            ep_data_grp.create_dataset("rewards", data=np.array(traj["rewards"]))
            ep_data_grp.create_dataset("dones", data=np.array(traj["dones"]))
            if args.dataset_obs:
                for k in traj["obs"]:
                    ep_data_grp.create_dataset(f"obs/{k}", data=np.array(traj["obs"][k]))
                    ep_data_grp.create_dataset(f"next_obs/{k}", data=np.array(traj["next_obs"][k]))

            if "model" in traj["initial_state_dict"]:
                ep_data_grp.attrs["model_file"] = traj["initial_state_dict"]["model"]
            ep_data_grp.attrs["num_samples"] = traj["actions"].shape[0]
            total_samples += traj["actions"].shape[0]

    rollout_stats = TensorUtils.list_of_flat_dict_to_dict_of_list(rollout_stats)
    avg_rollout_stats = { k : np.mean(rollout_stats[k]) for k in rollout_stats }
    avg_rollout_stats["Num_Success"] = np.sum(rollout_stats["Success_Rate"])

    if args.video_path is not None:
        results_filename = f"{args.video_path.rsplit('.', 1)[0]}_results.json"
        with open(results_filename, 'w') as json_file:
            json.dump(avg_rollout_stats, json_file, indent=4)

    print("\n--- Average Rollout Stats ---")
    print(json.dumps(avg_rollout_stats, indent=4))

    if write_video:
        video_writer.close()

    if write_dataset:
        data_grp.attrs["total"] = total_samples
        data_grp.attrs["env_args"] = json.dumps(env.serialize(), indent=4)
        data_writer.close()
        print(f"Wrote dataset trajectories to {args.dataset_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--agent",
        type=str,
        default='/home/yujp/MimicPlay/trained_models_lowlevel/test/lowlevel_model_epoch_950_modified.pth',
        help="path to saved checkpoint pth file",
    )
    parser.add_argument(
        "--n_rollouts",
        type=int,
        default=100,
        help="number of rollouts",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=2000,
        help="(optional) override maximum horizon of rollout from the one in the checkpoint",
    )
    parser.add_argument(
        "--env",
        type=str,
        default=None,
        help="(optional) override name of env from the one in the checkpoint, and use it for rollouts",
    )
    parser.add_argument(
        "--render",
        default=False,
        action='store_true',
        help="on-screen rendering",
    )
    parser.add_argument(
        "--video_path",
        type=str,
        default=None,
        help="(optional) render rollouts to this video file path",
    )
    parser.add_argument(
        "--video_skip",
        type=int,
        default=5,
        help="render frames to video every n steps",
    )
    parser.add_argument(
        "--camera_names",
        type=str,
        nargs='+',
        default=["render_view"],
        help="(optional) camera name(s) to use for rendering on-screen or to video",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="(optional) if provided, an hdf5 file will be written at this path with the rollout data",
    )
    parser.add_argument(
        "--dataset_obs",
        action='store_true',
        help="include possibly high-dimensional observations in output dataset hdf5 file",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="(optional) set seed for rollouts",
    )
    parser.add_argument(
        "--video_prompt",
        type=str,
        default='/home/yujp/MimicPlay/mimicplay/datasets/eval-task-3_put_bowl_on_shelf_put_pan_in_shelf/image_demo.hdf5',
        help="(optional) if provided, a task video prompt is loaded and used in the evaluation rollouts",
    )
    parser.add_argument(
        "--condition_file",
        type=str,
        default=None,
        help="(optional) path to the file defining the task's symbolic goal",
    )

    # --- dataset action sanity check mode (bypass policy) ---
    parser.add_argument(
        "--dataset_action_hdf5",
        type=str,
        default=None,
        help="(optional) if provided, bypass policy and instead read an action from this dataset and apply one env.step(action).",
    )
    parser.add_argument(
        "--dataset_action_demo",
        type=str,
        default="demo_1",
        help="demo name inside dataset (default: demo_1)",
    )
    parser.add_argument(
        "--dataset_action_step",
        type=int,
        default=0,
        help="which action index to take from dataset (default: 0)",
    )
    parser.add_argument(
        "--dataset_action_play_all",
        action="store_true",
        help="if set, step through many actions (full trajectory playback) instead of a single step",
    )
    parser.add_argument(
        "--dataset_action_start_step",
        type=int,
        default=0,
        help="start step index for full playback (default: 0)",
    )
    parser.add_argument(
        "--dataset_action_end_step",
        type=int,
        default=None,
        help="end step index (exclusive) for full playback (default: None -> until end)",
    )
    parser.add_argument(
        "--dataset_action_video_path",
        type=str,
        default=None,
        help="(optional) if provided, save a video while playing actions (cannot be used with --render)",
    )
    parser.add_argument(
        "--dataset_action_video_skip",
        type=int,
        default=1,
        help="write a video frame every N steps during dataset action playback (default: 1)",
    )
    parser.add_argument(
        "--dataset_action_fps",
        type=int,
        default=20,
        help="fps for dataset action playback video (default: 20)",
    )
    parser.add_argument(
        "--dataset_action_log_every",
        type=int,
        default=50,
        help="print progress every N steps during full playback (default: 50)",
    )
    parser.add_argument(
        "--dataset_action_render_frames",
        type=int,
        default=60,
        help="how many extra frames to render after the step (only if --render) (default: 60)",
    )

    args = parser.parse_args()

    # If dataset_action_hdf5 is provided, run the one-step sanity test and exit.
    if args.dataset_action_hdf5 is not None:
        _step_with_dataset_first_action(args)
        raise SystemExit(0)

    run_trained_agent(args)
