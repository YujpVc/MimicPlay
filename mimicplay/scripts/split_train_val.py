import h5py
import numpy as np
import argparse
import os

def create_train_val_mask(hdf5_path, val_ratio=0.1):
    print(f"Processing {hdf5_path}...")
    
    if not os.path.exists(hdf5_path):
        print(f"Error: File {hdf5_path} does not exist.")
        return

    with h5py.File(hdf5_path, 'r+') as f:
        if 'data' not in f:
            print("Error: 'data' group not found in HDF5 file.")
            return

        # Get all demo keys
        demo_keys = list(f['data'].keys())
        # Sort to ensure reproducibility
        demo_keys.sort()
        
        num_demos = len(demo_keys)
        print(f"Found {num_demos} demos.")
        
        # Calculate split
        num_val = int(num_demos * val_ratio)
        if num_val < 1 and num_demos > 1:
            num_val = 1
            
        # Shuffle keys (fix seed for reproducibility)
        rng = np.random.RandomState(seed=42)
        rng.shuffle(demo_keys)
        
        val_keys = demo_keys[:num_val]
        train_keys = demo_keys[num_val:]
        
        print(f"Split: {len(train_keys)} training, {len(val_keys)} validation.")
        
        # Create mask group if it doesn't exist
        if 'mask' in f:
            del f['mask']
        mask_grp = f.create_group('mask')
        
        # Create datasets for masks
        # Robomimic expects these to be string arrays
        mask_grp.create_dataset('train', data=np.array(train_keys, dtype='S'))
        mask_grp.create_dataset('valid', data=np.array(val_keys, dtype='S'))
        
        print(f"Successfully added mask/train and mask/valid to {hdf5_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Path to HDF5 dataset")
    parser.add_argument("--ratio", type=float, default=0.1, help="Validation ratio")
    args = parser.parse_args()
    
    create_train_val_mask(args.dataset, args.ratio)
