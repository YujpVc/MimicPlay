"""
Script to scale actions in a HDF5 dataset to improve training stability.
Specifically designed for Genesis data where actions (delta position) are very small.

Usage:
    python scripts/dataset_scale_genesis_actions.py --dataset /path/to/dataset.hdf5 --output_name scaled_dataset.hdf5 --scale 20.0
"""

import os
import h5py
import argparse
import numpy as np
import shutil

def scale_actions(args):
    # input file
    f_in = h5py.File(args.dataset, "r")
    
    # output file
    output_path = os.path.join(os.path.dirname(args.dataset), args.output_name)
    print(f"Input dataset: {args.dataset}")
    print(f"Output dataset: {output_path}")
    
    # Create output file
    f_out = h5py.File(output_path, "w")
    
    # Copy 'mask' group if it exists
    if "mask" in f_in:
        f_in.copy("mask", f_out)
        
    # Create 'data' group
    data_grp_out = f_out.create_group("data")
    
    # Copy global attributes
    for k, v in f_in["data"].attrs.items():
        data_grp_out.attrs[k] = v
        
    demos = list(f_in["data"].keys())
    print(f"Processing {len(demos)} demos...")
    
    total_actions = 0
    clipped_count = 0
    
    pos_scale = args.pos_scale
    rot_scale = args.rot_scale
    
    for demo_key in demos:
        demo_in = f_in[f"data/{demo_key}"]
        demo_out = data_grp_out.create_group(demo_key)
        
        # Copy attributes
        for k, v in demo_in.attrs.items():
            demo_out.attrs[k] = v
            
        # Copy all datasets EXCEPT actions
        for key in demo_in.keys():
            if key != "actions":
                f_in.copy(f"data/{demo_key}/{key}", demo_out)
        
        # Process actions
        actions = demo_in["actions"][:] # [T, 7] (pos:3, rot:3, gripper:1)
        
        # Split actions
        pos_actions = actions[:, :3]
        rot_actions = actions[:, 3:6]
        gripper_actions = actions[:, 6:]
        
        # Apply scaling
        pos_actions_scaled = pos_actions * pos_scale
        rot_actions_scaled = rot_actions * rot_scale
        
        # Recombine
        actions_scaled = np.concatenate([pos_actions_scaled, rot_actions_scaled, gripper_actions], axis=1)
        
        # Clip to [-1, 1]
        # Check how many values were clipped (excluding gripper which is already -1/1)
        n_clipped = np.sum(np.abs(actions_scaled[:, :6]) > 1.0)
        clipped_count += n_clipped
        total_actions += actions.shape[0] * 6 # 6 dims
        
        actions_final = np.clip(actions_scaled, -1.0, 1.0)
        
        # Write to output
        demo_out.create_dataset("actions", data=actions_final)
        
    print(f"Done.")
    print(f"Pos Scale: {pos_scale}, Rot Scale: {rot_scale}")
    print(f"Clipped values (pos+rot): {clipped_count} / {total_actions} ({clipped_count/total_actions*100:.2f}%)")
    
    f_in.close()
    f_out.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="/home/yujp/Genesis/scripts/XBOX_control/dataCollect/demo/juice_wine_book_multitask/image_demo_local.hdf5", help="Path to input HDF5")
    parser.add_argument("--output_name", type=str, default="image_demo_local_scaled.hdf5", help="Name of output HDF5 file")
    parser.add_argument("--pos_scale", type=float, default=10.0, help="Scaling factor for position actions")
    parser.add_argument("--rot_scale", type=float, default=1.0, help="Scaling factor for rotation actions")
    
    args = parser.parse_args()
    scale_actions(args)

