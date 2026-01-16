"""
Standalone training script that supports resuming weights from a .pth checkpoint,
while saving ALL new outputs (logs/models/videos/config) into a NEW output folder.

Why this exists:
- Do NOT modify or "pollute" existing training scripts or existing run folders.
- Resume from a checkpoint that stores only model weights + metadata (no optimizer state).

Notes about resume:
- MimicPlay checkpoints (robomimic TrainUtils.save_model) usually store keys:
  ["algo_name", "config", "env_metadata", "model", "shape_metadata"].
- There is NO optimizer state in these checkpoints, so we can only resume weights.
  We also fast-forward LR schedulers to match the resumed epoch count.
"""

import argparse
import datetime
import json
import os
import re
import sys
import time
import traceback
from collections import OrderedDict
from typing import Optional, Tuple

import numpy as np
import psutil
import torch
from torch.utils.data import DataLoader

# Keep these for the user's environment layout (matches existing MimicPlay scripts)
sys.path.append("/home/yujp/robosuite")
sys.path.append("/home/yujp/robomimic")

import robomimic.utils.train_utils as TrainUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as RoboFileUtils
from robomimic.utils.log_utils import PrintLogger, DataLogger

from mimicplay.configs import config_factory
from mimicplay.algo import algo_factory, RolloutPolicy
from mimicplay.utils.train_utils import rollout_with_stats, load_data_for_training
import mimicplay.utils.file_utils as MPFileUtils


def _make_run_dirs(output_dir: str, exp_name: str) -> Tuple[str, str, str, str]:
    """
    Create a NEW run directory under output_dir/exp_name/<timestamp>/ and return:
      run_dir, log_dir, ckpt_dir, video_dir
    Never overwrites existing runs.
    """
    base_output_dir = os.path.abspath(os.path.expanduser(output_dir))
    run_root = os.path.join(base_output_dir, exp_name)
    os.makedirs(run_root, exist_ok=True)

    # timestamp for directory names
    t_now = time.time()
    time_str = datetime.datetime.fromtimestamp(t_now).strftime("%Y%m%d%H%M%S")
    run_dir = os.path.join(run_root, time_str)

    # extremely unlikely collision, but be safe
    suffix = 0
    while os.path.exists(run_dir):
        suffix += 1
        run_dir = os.path.join(run_root, f"{time_str}_{suffix}")

    log_dir = os.path.join(run_dir, "logs")
    ckpt_dir = os.path.join(run_dir, "models")
    video_dir = os.path.join(run_dir, "videos")
    os.makedirs(log_dir, exist_ok=False)
    os.makedirs(ckpt_dir, exist_ok=False)
    os.makedirs(video_dir, exist_ok=False)
    return run_dir, log_dir, ckpt_dir, video_dir


def _infer_start_epoch(resume_from: str, override: Optional[int]) -> int:
    """
    Decide the first epoch to run when resuming.
    Default: parse model_epoch_<N> from filename and return N+1.
    """
    if override is not None:
        return int(override)
    base = os.path.basename(resume_from)
    m = re.search(r"model_epoch_(\d+)", base)
    if m is None:
        return 1
    return int(m.group(1)) + 1


def _fast_forward_lr_schedulers(model, start_epoch: int) -> None:
    """
    Try to fast-forward LR schedulers to match the resumed epoch count.
    This matters because our resume checkpoint does NOT contain optimizer/scheduler state.
    """
    if start_epoch <= 1:
        return
    for _, sched in getattr(model, "lr_schedulers", {}).items():
        if sched is None:
            continue
        try:
            sched.step(start_epoch - 1)
        except TypeError:
            for _ in range(start_epoch - 1):
                sched.step()


def train(args, config, device) -> None:
    # first set seeds
    np.random.seed(config.train.seed)
    torch.manual_seed(config.train.seed)

    print("\n============= New Training Run with Config =============")
    print(config)
    print("")

    run_dir, log_dir, ckpt_dir, video_dir = _make_run_dirs(
        output_dir=config.train.output_dir,
        exp_name=config.experiment.name,
    )
    print(f"[custom_train] run_dir: {run_dir}")
    print(f"[custom_train] logs:    {log_dir}")
    print(f"[custom_train] models:  {ckpt_dir}")
    print(f"[custom_train] videos:  {video_dir}")

    if config.experiment.logging.terminal_output_to_txt:
        logger = PrintLogger(os.path.join(log_dir, "log.txt"))
        sys.stdout = logger
        sys.stderr = logger

    # read config to set up metadata for observation modalities (e.g. detecting rgb observations)
    ObsUtils.initialize_obs_utils_with_config(config)

    # make sure the dataset exists
    dataset_path = os.path.expanduser(config.train.data)
    if not os.path.exists(dataset_path):
        raise Exception(f"Dataset at provided path {dataset_path} not found!")

    # load basic metadata from training file
    print("\n============= Loaded Environment Metadata =============")
    env_meta = RoboFileUtils.get_env_metadata_from_dataset(dataset_path=config.train.data)
    shape_meta = RoboFileUtils.get_shape_metadata_from_dataset(
        dataset_path=config.train.data,
        all_obs_keys=config.all_obs_keys,
        verbose=True,
    )

    if config.experiment.env is not None:
        env_meta["env_name"] = config.experiment.env
        print("=" * 30 + f"\nReplacing Env to {env_meta['env_name']}\n" + "=" * 30)

    # create environment(s) for rollouts
    envs = OrderedDict()
    if config.experiment.rollout.enabled:
        env_names = [env_meta["env_name"]]
        if config.experiment.additional_envs is not None:
            for name in config.experiment.additional_envs:
                env_names.append(name)

        for env_name in env_names:
            dummy_spec = dict(
                obs=dict(
                    low_dim=["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"],
                    rgb=["agentview_image", "robot0_eye_in_hand_image"],
                ),
            )
            ObsUtils.initialize_obs_utils_with_obs_specs(obs_modality_specs=dummy_spec)

            if args.bddl_file is not None:
                env_meta["env_kwargs"]["bddl_file_name"] = args.bddl_file

            env = EnvUtils.create_env_from_metadata(
                env_meta=env_meta,
                env_name=env_name,
                render=False,
                render_offscreen=config.experiment.render_video,
                use_image_obs=shape_meta["use_images"],
            )

            if args.condition_file is not None:
                env.condition_file = args.condition_file
                env._load_condition_file()
                print(f"强制设置 condition_file: {env.condition_file}")

            envs[env.name] = env

    # setup logger
    data_logger = DataLogger(
        log_dir,
        config,
        log_tb=config.experiment.logging.log_tb,
    )

    # create model
    model = algo_factory(
        algo_name=config.algo_name,
        config=config,
        obs_key_shapes=shape_meta["all_shapes"],
        ac_dim=shape_meta["ac_dim"],
        device=device,
    )

    # resume weights if requested
    start_epoch = 1
    if args.resume_from is not None:
        ckpt_dict = MPFileUtils.load_dict_from_checkpoint(args.resume_from)
        if "model" not in ckpt_dict:
            raise ValueError(f"[resume] checkpoint missing 'model' key: {args.resume_from}")
        model.deserialize(ckpt_dict["model"])
        start_epoch = _infer_start_epoch(args.resume_from, args.resume_start_epoch)
        _fast_forward_lr_schedulers(model, start_epoch=start_epoch)
        print(f"[resume] loaded weights from: {args.resume_from}")
        print("[resume] WARNING: checkpoint has no optimizer state; optimizer is re-initialized.")
        print(f"[resume] continuing training from epoch {start_epoch} to {config.train.num_epochs} (inclusive)")

    if config.experiment.rollout.enabled:
        model.load_eval_video_prompt(args.video_prompt)

    # save the config as a json file (match original layout: run_dir/config.json)
    with open(os.path.join(run_dir, "config.json"), "w") as outfile:
        json.dump(config, outfile, indent=4)

    print("\n============= Model Summary =============")
    print(model)
    print("")

    # load training data
    trainset, validset = load_data_for_training(config, obs_keys=shape_meta["all_obs_keys"])
    train_sampler = trainset.get_dataset_sampler()
    print("\n============= Training Dataset =============")
    print(trainset)
    print("")

    obs_normalization_stats = None
    if config.train.hdf5_normalize_obs:
        obs_normalization_stats = trainset.get_obs_normalization_stats()

    train_loader = DataLoader(
        dataset=trainset,
        sampler=train_sampler,
        batch_size=config.train.batch_size,
        shuffle=(train_sampler is None),
        num_workers=config.train.num_data_workers,
        drop_last=True,
    )

    if config.experiment.validate:
        num_workers = min(config.train.num_data_workers, 1)
        valid_sampler = validset.get_dataset_sampler()
        valid_loader = DataLoader(
            dataset=validset,
            sampler=valid_sampler,
            batch_size=config.train.batch_size,
            shuffle=(valid_sampler is None),
            num_workers=num_workers,
            drop_last=True,
        )
    else:
        valid_loader = None

    # main training loop
    best_valid_loss = None
    best_return = {k: -np.inf for k in envs} if config.experiment.rollout.enabled else None
    best_success_rate = {k: -1.0 for k in envs} if config.experiment.rollout.enabled else None
    last_ckpt_time = time.time()

    train_num_steps = config.experiment.epoch_every_n_steps
    valid_num_steps = config.experiment.validation_epoch_every_n_steps

    for epoch in range(start_epoch, config.train.num_epochs + 1):
        step_log = TrainUtils.run_epoch(
            model=model,
            data_loader=train_loader,
            epoch=epoch,
            num_steps=train_num_steps,
        )
        model.on_epoch_end(epoch)

        epoch_ckpt_name = f"model_epoch_{epoch}"

        # recurring checkpoint saving conditions
        should_save_ckpt = False
        if config.experiment.save.enabled:
            time_check = (config.experiment.save.every_n_seconds is not None) and (
                time.time() - last_ckpt_time > config.experiment.save.every_n_seconds
            )
            epoch_check = (config.experiment.save.every_n_epochs is not None) and (epoch > 0) and (
                epoch % config.experiment.save.every_n_epochs == 0
            )
            epoch_list_check = epoch in config.experiment.save.epochs
            should_save_ckpt = time_check or epoch_check or epoch_list_check
        ckpt_reason = None
        if should_save_ckpt:
            last_ckpt_time = time.time()
            ckpt_reason = "time"

        print(f"Train Epoch {epoch}")
        print(json.dumps(step_log, sort_keys=True, indent=4))
        for k, v in step_log.items():
            if k.startswith("Time_"):
                data_logger.record(f"Timing_Stats/Train_{k[5:]}", v, epoch)
            else:
                data_logger.record(f"Train/{k}", v, epoch)

        # validation
        if config.experiment.validate:
            with torch.no_grad():
                step_log = TrainUtils.run_epoch(
                    model=model,
                    data_loader=valid_loader,
                    epoch=epoch,
                    validate=True,
                    num_steps=valid_num_steps,
                )
            for k, v in step_log.items():
                if k.startswith("Time_"):
                    data_logger.record(f"Timing_Stats/Valid_{k[5:]}", v, epoch)
                else:
                    data_logger.record(f"Valid/{k}", v, epoch)

            print(f"Validation Epoch {epoch}")
            print(json.dumps(step_log, sort_keys=True, indent=4))

            valid_check = "Loss" in step_log
            if valid_check and (best_valid_loss is None or (step_log["Loss"] <= best_valid_loss)):
                best_valid_loss = step_log["Loss"]
                if config.experiment.save.enabled and config.experiment.save.on_best_validation:
                    epoch_ckpt_name += f"_best_validation_{best_valid_loss}"
                    should_save_ckpt = True
                    ckpt_reason = "valid" if ckpt_reason is None else ckpt_reason

        # rollouts
        video_paths = None
        rollout_check = (epoch % config.experiment.rollout.rate == 0) or (should_save_ckpt and ckpt_reason == "time")
        if config.experiment.rollout.enabled and (epoch > config.experiment.rollout.warmstart) and rollout_check:
            rollout_model = RolloutPolicy(model, obs_normalization_stats=obs_normalization_stats)
            all_rollout_logs, video_paths = rollout_with_stats(
                policy=rollout_model,
                envs=envs,
                horizon=config.experiment.rollout.horizon,
                use_goals=config.use_goals,
                num_episodes=config.experiment.rollout.n,
                render=False,
                video_dir=video_dir if config.experiment.render_video else None,
                epoch=epoch,
                video_skip=config.experiment.get("video_skip", 5),
                terminate_on_success=config.experiment.rollout.terminate_on_success,
            )

            for env_name in all_rollout_logs:
                rollout_logs = all_rollout_logs[env_name]
                for k, v in rollout_logs.items():
                    if k.startswith("Time_"):
                        data_logger.record(f"Timing_Stats/Rollout_{env_name}_{k[5:]}", v, epoch)
                    else:
                        data_logger.record(f"Rollout/{k}/{env_name}", v, epoch, log_stats=True)

                print(f"\nEpoch {epoch} Rollouts took {rollout_logs['time']}s (avg) with results:")
                print(f"Env: {env_name}")
                print(json.dumps(rollout_logs, sort_keys=True, indent=4))

            updated_stats = TrainUtils.should_save_from_rollout_logs(
                all_rollout_logs=all_rollout_logs,
                best_return=best_return,
                best_success_rate=best_success_rate,
                epoch_ckpt_name=epoch_ckpt_name,
                save_on_best_rollout_return=config.experiment.save.on_best_rollout_return,
                save_on_best_rollout_success_rate=config.experiment.save.on_best_rollout_success_rate,
            )
            best_return = updated_stats["best_return"]
            best_success_rate = updated_stats["best_success_rate"]
            epoch_ckpt_name = updated_stats["epoch_ckpt_name"]
            should_save_ckpt = (config.experiment.save.enabled and updated_stats["should_save_ckpt"]) or should_save_ckpt
            if updated_stats["ckpt_reason"] is not None:
                ckpt_reason = updated_stats["ckpt_reason"]

        # Only keep saved videos if the ckpt should be saved (but not because of validation score)
        should_save_video = (should_save_ckpt and (ckpt_reason != "valid")) or config.experiment.keep_all_videos
        if video_paths is not None and not should_save_video:
            for env_name in video_paths:
                video_path = video_paths[env_name]
                if os.path.exists(video_path):
                    os.remove(video_path)
                else:
                    print(f"Warning: Did not find video to delete: {video_path}")

        if should_save_ckpt:
            TrainUtils.save_model(
                model=model,
                config=config,
                env_meta=env_meta,
                shape_meta=shape_meta,
                ckpt_path=os.path.join(ckpt_dir, epoch_ckpt_name + ".pth"),
                obs_normalization_stats=obs_normalization_stats,
            )

        # log memory usage
        process = psutil.Process(os.getpid())
        mem_usage = int(process.memory_info().rss / 1000000)
        data_logger.record("System/RAM Usage (MB)", mem_usage, epoch)
        print(f"\nEpoch {epoch} Memory Usage: {mem_usage} MB\n")

    data_logger.close()


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, required=True, help="Path to config JSON.")
    parser.add_argument("--dataset", type=str, default=None, help="Override dataset path.")
    parser.add_argument("--video_prompt", type=str, default=None, help="Video prompt path.")
    parser.add_argument("--name", type=str, default=None, help="Experiment name (subfolder under output_dir).")
    parser.add_argument("--output_dir", type=str, required=True, help="NEW output dir for this run (won't touch old).")
    parser.add_argument("--bddl_file", type=str, default=None, help="Optional bddl file for env goal.")
    parser.add_argument("--condition_file", type=str, default=None, help="Optional success-condition file.")
    parser.add_argument("--debug", action="store_true", help="Quick debug run.")

    parser.add_argument("--resume_from", type=str, default=None, help="Path to .pth checkpoint to resume weights from.")
    parser.add_argument("--resume_start_epoch", type=int, default=None, help="Override start epoch when resuming.")

    args = parser.parse_args()

    ext_cfg = json.load(open(args.config, "r"))
    config = config_factory(ext_cfg["algo_name"])
    with config.values_unlocked():
        config.update(ext_cfg)

    if args.dataset is not None:
        config.train.data = args.dataset
    if args.name is not None:
        config.experiment.name = args.name

    # IMPORTANT: keep all outputs isolated under a NEW output dir
    config.train.output_dir = args.output_dir

    # debug modifications
    if args.debug:
        config.unlock()
        config.lock_keys()
        config.experiment.epoch_every_n_steps = 3
        config.experiment.validation_epoch_every_n_steps = 3
        config.train.num_epochs = 2
        config.experiment.rollout.rate = 1
        config.experiment.rollout.n = 2
        config.experiment.rollout.horizon = 10

    config.lock()

    # get torch device
    device = TorchUtils.get_torch_device(try_to_use_cuda=config.train.cuda)

    res_str = "finished run successfully!"
    try:
        train(args=args, config=config, device=device)
    except Exception as e:
        res_str = f"run failed with error:\n{e}\n\n{traceback.format_exc()}"
    print(res_str)


if __name__ == "__main__":
    main()


