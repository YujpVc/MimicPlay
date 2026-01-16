import h5py, numpy as np

path = "/home/yujp/Genesis/scripts/XBOX_control/dataCollect/demo/demo_multitask/image_demo_local_scaled.hdf5"
with h5py.File(path, "r") as f:
    demos = sorted(f["data"].keys(), key=lambda x: int(x[5:]))
    for i in range(1, len(demos)):
        prev_last = f[f"data/{demos[i-1]}/states"][-1]
        cur_first = f[f"data/{demos[i]}/states"][0]
        dist = np.linalg.norm(prev_last - cur_first)
        if dist < 1e-4:
            print(f"{demos[i]} 起始状态≈上一条结束 (距离 {dist:.2e})")