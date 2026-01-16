"""
Helper script to convert a dataset collected using robosuite into an hdf5 compatible with
robomimic. Takes a dataset path corresponding to the demo.hdf5 file containing the
demonstrations. It modifies the dataset in-place. By default, the script also creates a
small validation set for training.

Example usage:

    python scripts/convert_playdata_to_robomimic_dataset.py --dataset 'path_to_your_data'
"""

import h5py
import json
import argparse

import robomimic.envs.env_base as EB
from robomimic.scripts.split_train_val import split_train_val_from_hdf5

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        help="path to input hdf5 dataset",
    )
    args = parser.parse_args()

    f = h5py.File(args.dataset, "a") # edit mode

    # store env meta
    env_name = f["data"].attrs["env"]
    env_info = json.loads(f["data"].attrs["env_info"])
    env_meta = dict(
        type=EB.EnvType.GENESIS,  # 明确指定为 GENESIS 类型
        env_name="Genesis Franka Environment",
        env_version=f["data"].attrs["repository_version"],
        env_kwargs={
            "env_name": "Franka_Env",
            # 根据 Genesis 的需求调整 env_kwargs
            "robots": ["Panda"],
            "control_freq": 20,
            "render_camera": "gripper_cam",
            # 删除不必要的 robosuite 特定参数，如 controller_configs
        }
    )
    if "env_args" in f["data"].attrs:
        del f["data"].attrs["env_args"]
    f["data"].attrs["env_args"] = json.dumps(env_meta, indent=4)

    print("====== Stored env meta ======")
    print(f["data"].attrs["env_args"])

    # store metadata about number of samples
    total_samples = 0
    for ep in f["data"]:
        # ensure model-xml is in per-episode metadata
        # assert "model_file" in f["data/{}".format(ep)].attrs

        # add "num_samples" into per-episode metadata
        if "num_samples" in f["data/{}".format(ep)].attrs:
            del f["data/{}".format(ep)].attrs["num_samples"]
        n_sample = f["data/{}/actions".format(ep)].shape[0]
        f["data/{}".format(ep)].attrs["num_samples"] = n_sample
        total_samples += n_sample

    # add total samples to global metadata
    if "total" in f["data"].attrs:
        del f["data"].attrs["total"]
    f["data"].attrs["total"] = total_samples

    f.close()

    # create 90-10 train-validation split in the dataset
    split_train_val_from_hdf5(hdf5_path=args.dataset, val_ratio=0.1)
