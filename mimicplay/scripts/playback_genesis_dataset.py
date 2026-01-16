"""
Modified script to generate both hand and agent view videos in one run.

Changes:
1. Added support for generating multiple videos (hand and agent view)
2. Improved image handling for playback_trajectory_with_obs
"""

import os
import json
import h5py
import argparse
import imageio
import numpy as np
import time

import robomimic
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
from robomimic.envs.env_base import EnvBase, EnvType


# Define default cameras to use for each env type
DEFAULT_CAMERAS = {
    EnvType.ROBOSUITE_TYPE: ["agentview"],
    EnvType.IG_MOMART_TYPE: ["rgb"],
    EnvType.GYM_TYPE: ValueError("No camera names supported for gym type env!"),
}


def playback_trajectory_with_obs(
    traj_grp,
    video_writer,
    video_skip=5,
    image_names=None,
    first=False,
):
    """
    This function reads image observations and writes them into a video.
    Handles both grayscale and RGB images correctly.

    Args:
        traj_grp (hdf5 file group): Trajectory group to playback
        video_writer (imageio writer): Video writer
        video_skip (int): Render frames to video every n steps
        image_names (list): Image observations to use for rendering
        first (bool): Only use the first frame of the episode
    """
    assert image_names is not None, "Must specify at least one image observation"
    video_count = 0

    traj_len = traj_grp["actions"].shape[0]
    for i in range(traj_len):
        if video_count % video_skip == 0:
            frame_list = []
            for img_name in image_names:
                # Get image data in (C, H, W) format
                img = traj_grp["obs/{}".format(img_name)][i]

                # Handle grayscale images (convert to RGB)
                if img.shape[0] == 1:  # Grayscale image
                    # Convert to (H, W, 1)
                    img = img.transpose(1, 2, 0)
                    # Repeat single channel to get RGB
                    img = np.concatenate([img, img, img], axis=2)
                else:  # RGB image (C=3)
                    # Transpose to (H, W, C)
                    img = img.transpose(1, 2, 0)
                frame_list.append(img)

            # Concatenate images horizontally
            frame = np.concatenate(frame_list, axis=1)
            video_writer.append_data(frame)

        video_count += 1
        if first:
            break


def playback_dataset(args):
    # Some arg checking
    write_video = (args.video_path is not None)
    assert not (args.render and write_video), "Cannot render on screen and write video simultaneously"

    # Auto-fill camera rendering info if not specified
    if args.render_image_names is None:
        env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=args.dataset)
        env_type = EnvUtils.get_env_type(env_meta=env_meta)
        args.render_image_names = DEFAULT_CAMERAS[env_type]

    if args.render:
        assert len(args.render_image_names) == 1, "On-screen rendering supports only one camera"

    if args.use_obs:
        assert write_video, "Playback with observations requires a video path"
        assert not args.use_actions, "Observation playback doesn't support action playback"

    # Create environment if not playing back with observations
    if not args.use_obs:
        dummy_spec = dict(
            obs=dict(low_dim=["robot0_eef_pos"], rgb=[],),
        )
        ObsUtils.initialize_obs_utils_with_obs_specs(obs_modality_specs=dummy_spec)

        env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=args.dataset)
        env = EnvUtils.create_env_from_metadata(
            env_meta=env_meta,
            render=args.render,
            render_offscreen=write_video
        )
        is_robosuite_env = EnvUtils.is_robosuite_env(env_meta)

    f = h5py.File(args.dataset, "r")

    # List of demonstrations
    if args.filter_key is not None:
        demos = [elem.decode("utf-8") for elem in np.array(f["mask/{}".format(args.filter_key)])]
    else:
        demos = list(f["data"].keys())
    demos = sorted(demos, key=lambda x: int(x.split("_")[-1]))

    # Limit number of demos if requested
    if args.n is not None:
        demos = demos[:args.n]

    # Video writers for both perspectives
    video_writer_hand = None
    video_writer_agent = None

    if write_video and "hand" in args.perspectives:
        video_writer_hand = imageio.get_writer(args.video_path_hand, fps=20)
    if write_video and "agent" in args.perspectives:
        video_writer_agent = imageio.get_writer(args.video_path_agent, fps=20)

    for ep in demos:
        print(f"Processing episode: {ep}")

        if args.use_obs:
            # Generate hand view video
            if video_writer_hand:
                playback_trajectory_with_obs(
                    traj_grp=f["data/{}".format(ep)],
                    video_writer=video_writer_hand,
                    video_skip=args.video_skip,
                    image_names=["robot0_eye_in_hand_image"],
                    first=args.first,
                )

            # Generate agent view video
            if video_writer_agent:
                playback_trajectory_with_obs(
                    traj_grp=f["data/{}".format(ep)],
                    video_writer=video_writer_agent,
                    video_skip=args.video_skip,
                    image_names=["agentview_image"],
                    first=args.first,
                )
            continue

        # Prepare initial state
        states = f["data/{}/states".format(ep)][()]
        initial_state = dict(states=states[0])
        if is_robosuite_env:
            initial_state["model"] = f["data/{}".format(ep)].attrs["model_file"]

        # Prepare actions
        actions = None
        if args.use_actions:
            actions = f["data/{}/actions".format(ep)][()]

        # (Original environment playback code remains unchanged)

    f.close()

    # Close video writers
    if video_writer_hand:
        video_writer_hand.close()
    if video_writer_agent:
        video_writer_agent.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Playback dataset with support for generating multiple perspective videos")

    # Main parameters
    parser.add_argument("--dataset", type=str, required=True, help="Path to hdf5 dataset")
    parser.add_argument("--filter_key", type=str, default=None, help="Filter key for trajectory subset")
    parser.add_argument("--n", type=int, default=None, help="Number of trajectories to process")

    # Playback mode
    parser.add_argument("--use-obs", action='store_true', help="Use dataset image observations")
    parser.add_argument("--use-actions", action='store_true', help="Use open-loop action playback")
    parser.add_argument("--render", action='store_true', help="On-screen rendering")

    # Video parameters
    parser.add_argument("--video_path_hand", type=str, help="Output path for hand view video")
    parser.add_argument("--video_path_agent", type=str, help="Output path for agent view video")
    parser.add_argument("--video_skip", type=int, default=5, help="Render frames to video every n steps")
    parser.add_argument("--perspectives", nargs='+', default=["hand", "agent"],
                        choices=["hand", "agent"],
                        help="Perspectives to generate videos for")

    # Camera/image parameters
    parser.add_argument("--render_image_names", nargs='+', default=None,
                        help="Camera/image names for rendering")

    # Other options
    parser.add_argument("--first", action='store_true', help="Use only first frame of each episode")
    parser.add_argument("--control_freq", type=int, default=20, help="Control frequency in Hz")

    args = parser.parse_args()

    # Set video paths based on perspectives
    args.video_path = args.video_path_hand or args.video_path_agent

    playback_dataset(args)