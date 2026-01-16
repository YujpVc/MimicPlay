"""
More detailed HDF5 dataset inspector for MimicPlay / robomimic-format datasets.

This script is designed to be:
- streaming / memory-safe: doesn't concatenate full datasets in RAM
- informative: richer action / obs / reward / done statistics
- convenient: CLI + JSON report + plots

Expected HDF5 layout (robomimic-style):
  /data (group)
    attrs: env_args (json string) [optional]
    /demo_0 (group)
      actions [T, A]
      rewards [T] or [T-1] [optional]
      dones [T] or [T-1] [optional]
      states [...] [optional]
      /obs (group)
        <obs_key> [T, ...]
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np

import matplotlib.pyplot as plt

try:
    import seaborn as sns
except Exception:
    sns = None


@dataclass
class AnalyzeOptions:
    seed: int = 0
    max_demos: Optional[int] = None
    demo_sample_prob: float = 1.0

    # action analysis
    deadzone: float = 1e-4
    saturation_threshold: float = 0.999
    action_sample_max: int = 200_000  # reservoir sample rows
    action_chunk_size: int = 50_000

    # observation analysis
    obs_demo_sample_prob: float = 0.2
    obs_frames_per_demo: int = 20
    obs_max_total_frames_per_key: int = 2_000  # cap for each key across all demos
    obs_max_values_per_key: int = 2_000_000  # cap flattened values sampled per key

    # outputs
    save_plots: bool = True
    save_json: bool = True


class RunningMoments:
    """
    Welford running mean / variance (per-dimension).
    """

    def __init__(self, dim: int):
        self.dim = int(dim)
        self.n = 0
        self.mean = np.zeros((self.dim,), dtype=np.float64)
        self.m2 = np.zeros((self.dim,), dtype=np.float64)
        self.min = np.full((self.dim,), np.inf, dtype=np.float64)
        self.max = np.full((self.dim,), -np.inf, dtype=np.float64)

    def update_batch(self, x: np.ndarray) -> None:
        if x.size == 0:
            return
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        assert x.shape[1] == self.dim

        # min / max
        self.min = np.minimum(self.min, np.min(x, axis=0))
        self.max = np.maximum(self.max, np.max(x, axis=0))

        # Welford batch update
        batch_n = x.shape[0]
        batch_mean = np.mean(x, axis=0)
        batch_m2 = np.sum((x - batch_mean) ** 2, axis=0)

        if self.n == 0:
            self.n = batch_n
            self.mean = batch_mean
            self.m2 = batch_m2
            return

        delta = batch_mean - self.mean
        total_n = self.n + batch_n
        self.mean = self.mean + delta * (batch_n / total_n)
        self.m2 = self.m2 + batch_m2 + (delta**2) * (self.n * batch_n / total_n)
        self.n = total_n

    def finalize(self) -> Dict[str, Any]:
        var = self.m2 / max(self.n, 1)
        std = np.sqrt(np.maximum(var, 0.0))
        return {
            "count": int(self.n),
            "min": self.min.tolist(),
            "max": self.max.tolist(),
            "mean": self.mean.tolist(),
            "std": std.tolist(),
        }


class ReservoirSampler:
    """
    Reservoir sample rows from a stream of vectors. Stores up to `k` rows.
    """

    def __init__(self, k: int, dim: int, rng: np.random.Generator):
        self.k = int(k)
        self.dim = int(dim)
        self.rng = rng
        self.n_seen = 0
        self.buf = np.empty((0, self.dim), dtype=np.float32)

    def add_batch(self, x: np.ndarray) -> None:
        if self.k <= 0 or x.size == 0:
            return
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        assert x.shape[1] == self.dim

        for i in range(x.shape[0]):
            self.n_seen += 1
            if self.buf.shape[0] < self.k:
                self.buf = np.vstack([self.buf, x[i : i + 1]])
            else:
                j = self.rng.integers(0, self.n_seen)
                if j < self.k:
                    self.buf[j] = x[i]

    def get(self) -> np.ndarray:
        return self.buf


def _safe_json_dumps(obj: Any) -> str:
    return json.dumps(obj, indent=2, sort_keys=True, default=str)


def _maybe_parse_env_args(f: h5py.File) -> Optional[Dict[str, Any]]:
    try:
        if "data" in f and "env_args" in f["data"].attrs:
            raw = f["data"].attrs["env_args"]
            if isinstance(raw, (bytes, np.bytes_)):
                raw = raw.decode("utf-8", errors="ignore")
            if isinstance(raw, str):
                return json.loads(raw)
    except Exception:
        return None
    return None


def _list_demos(f: h5py.File) -> List[str]:
    demos = list(f["data"].keys()) if "data" in f else []
    # sort demo_0, demo_1, ...
    def _key(x: str):
        try:
            if "_" in x:
                return int(x.split("_")[-1])
        except Exception:
            pass
        return x

    return sorted(demos, key=_key)


def _sample_frame_indices(T: int, n: int, rng: np.random.Generator) -> np.ndarray:
    if T <= 0 or n <= 0:
        return np.array([], dtype=np.int64)
    n = min(T, n)
    if T <= n:
        return np.arange(T, dtype=np.int64)
    # choose without replacement
    return np.sort(rng.choice(T, size=n, replace=False).astype(np.int64))


def _ensure_outdir(output_dir: Optional[str]) -> Optional[str]:
    if output_dir is None:
        return None
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def _save_fig(path: str) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def analyze_dataset(path: str, name: str, output_dir: Optional[str], opts: AnalyzeOptions) -> Dict[str, Any]:
    """
    Analyze a single dataset and return a JSON-serializable report dict.
    """
    print(f"\n{'=' * 24}\nAnalyzing {name}\n  path: {path}\n{'=' * 24}")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")

    outdir = _ensure_outdir(output_dir)
    rng = np.random.default_rng(opts.seed)

    report: Dict[str, Any] = {
        "name": name,
        "path": path,
        "options": asdict(opts),
    }

    with h5py.File(path, "r") as f:
        env_args = _maybe_parse_env_args(f)
        if env_args is not None:
            report["env_args"] = env_args

        demos_all = _list_demos(f)
        report["num_demos_total"] = len(demos_all)
        print(f"Number of demonstrations (total): {len(demos_all)}")

        # optionally downsample demos
        demos: List[str] = []
        for d in demos_all:
            if opts.demo_sample_prob < 1.0 and rng.random() > opts.demo_sample_prob:
                continue
            demos.append(d)
            if opts.max_demos is not None and len(demos) >= int(opts.max_demos):
                break
        report["num_demos_used"] = len(demos)
        if len(demos) != len(demos_all):
            print(f"Using demos: {len(demos)} / {len(demos_all)} (max_demos={opts.max_demos}, p={opts.demo_sample_prob})")

        # per-demo stats
        lengths: List[int] = []
        rewards_sum: List[float] = []
        dones_last: List[int] = []

        # action running stats
        action_dim: Optional[int] = None
        action_mom = None
        delta_abs_mom = None  # moments of abs(delta) per dim
        delta_abs_max = None  # max abs(delta) per dim
        action_sampler = None
        idle_steps = 0
        sat_steps = 0
        total_steps = 0

        # obs key coverage + sampled value stats
        obs_key_info: Dict[str, Dict[str, Any]] = {}  # key -> dict

        for demo in demos:
            grp = f[f"data/{demo}"]

            # actions
            if "actions" in grp:
                ds = grp["actions"]
                if ds.ndim != 2:
                    print(f"Warning: {demo}/actions has unexpected shape {ds.shape}, skipping")
                else:
                    T, A = int(ds.shape[0]), int(ds.shape[1])
                    lengths.append(T)
                    if action_dim is None:
                        action_dim = A
                        action_mom = RunningMoments(dim=A)
                        delta_abs_mom = RunningMoments(dim=A)
                        delta_abs_max = np.zeros((A,), dtype=np.float64)
                        action_sampler = ReservoirSampler(k=opts.action_sample_max, dim=A, rng=rng)
                    elif action_dim != A:
                        print(f"Warning: action dim mismatch in {demo}: {A} vs {action_dim}, skipping actions")
                        continue

                    # stream actions in chunks
                    prev_last: Optional[np.ndarray] = None
                    for start in range(0, T, int(opts.action_chunk_size)):
                        end = min(T, start + int(opts.action_chunk_size))
                        chunk = ds[start:end]  # [N, A]
                        if chunk.size == 0:
                            continue

                        action_mom.update_batch(chunk)
                        action_sampler.add_batch(chunk)

                        total_steps += chunk.shape[0]
                        idle_steps += int(np.sum(np.all(np.abs(chunk) < opts.deadzone, axis=1)))
                        sat_steps += int(np.sum(np.any(np.abs(chunk) >= opts.saturation_threshold, axis=1)))

                        # delta stats across time (handle boundary between chunks)
                        if prev_last is not None:
                            first = chunk[0:1]
                            d0 = np.abs(first - prev_last)
                            delta_abs_mom.update_batch(d0)
                            delta_abs_max = np.maximum(delta_abs_max, np.max(d0, axis=0))
                        if chunk.shape[0] >= 2:
                            d = np.abs(np.diff(chunk, axis=0))
                            delta_abs_mom.update_batch(d)
                            delta_abs_max = np.maximum(delta_abs_max, np.max(d, axis=0))
                        prev_last = chunk[-1:]

            # rewards / dones (optional)
            if "rewards" in grp:
                try:
                    r = grp["rewards"][()]
                    rewards_sum.append(float(np.sum(r)))
                except Exception:
                    pass
            if "dones" in grp:
                try:
                    d = grp["dones"][()]
                    dones_last.append(int(np.array(d).reshape(-1)[-1]))
                except Exception:
                    pass

            # observations: sample few demos for obs to keep it cheap
            if "obs" in grp and (rng.random() <= opts.obs_demo_sample_prob):
                obs_grp = grp["obs"]
                for k in obs_grp.keys():
                    ds = obs_grp[k]
                    if ds.ndim < 1:
                        continue
                    T_obs = int(ds.shape[0])

                    info = obs_key_info.setdefault(
                        k,
                        {
                            "dtype": str(ds.dtype),
                            "shape": tuple(ds.shape[1:]),
                            "num_demos_with_key": 0,
                            "num_frames_sampled": 0,
                            "min": None,
                            "max": None,
                            "mean": None,
                            "std": None,
                            "is_image_like": ("image" in k) or (ds.ndim == 4 and ds.shape[-1] in (3, 4)),
                        },
                    )
                    info["num_demos_with_key"] += 1

                    # cap total frames per key
                    if info["num_frames_sampled"] >= int(opts.obs_max_total_frames_per_key):
                        continue

                    n_frames = min(int(opts.obs_frames_per_demo), int(opts.obs_max_total_frames_per_key) - int(info["num_frames_sampled"]))
                    idx = _sample_frame_indices(T_obs, n_frames, rng=rng)
                    if idx.size == 0:
                        continue

                    try:
                        frames = ds[idx]  # [n, ...]
                    except Exception:
                        continue

                    # sample / cap flattened values to avoid huge memory for image frames
                    flat = np.asarray(frames).reshape(-1)
                    if flat.size > int(opts.obs_max_values_per_key):
                        sub = rng.choice(flat.size, size=int(opts.obs_max_values_per_key), replace=False)
                        flat = flat[sub]
                    flat = flat.astype(np.float64, copy=False)

                    # update scalar stats (not per-channel)
                    vmin = float(np.min(flat)) if flat.size else None
                    vmax = float(np.max(flat)) if flat.size else None
                    vmean = float(np.mean(flat)) if flat.size else None
                    vstd = float(np.std(flat)) if flat.size else None

                    if vmin is not None:
                        info["min"] = vmin if info["min"] is None else float(min(info["min"], vmin))
                        info["max"] = vmax if info["max"] is None else float(max(info["max"], vmax))
                        # For mean/std, approximate by averaging across samples (good enough for quick inspection)
                        if info["mean"] is None:
                            info["mean"] = vmean
                            info["std"] = vstd
                        else:
                            # running mean/std approximation (treat each batch equally)
                            info["mean"] = float(0.9 * info["mean"] + 0.1 * vmean)
                            info["std"] = float(0.9 * info["std"] + 0.1 * vstd)

                    info["num_frames_sampled"] += int(idx.size)

        # finalize length stats
        if lengths:
            lengths_arr = np.array(lengths, dtype=np.int64)
            report["trajectory_lengths"] = {
                "count": int(lengths_arr.size),
                "min": int(np.min(lengths_arr)),
                "max": int(np.max(lengths_arr)),
                "mean": float(np.mean(lengths_arr)),
                "median": float(np.median(lengths_arr)),
                "p10": float(np.percentile(lengths_arr, 10)),
                "p90": float(np.percentile(lengths_arr, 90)),
            }
            print("\nTrajectory Lengths:")
            for kk, vv in report["trajectory_lengths"].items():
                print(f"  {kk}: {vv}")

            if outdir and opts.save_plots:
                plt.figure(figsize=(6, 4))
                plt.hist(lengths_arr, bins=40)
                plt.title(f"{name} - Trajectory Lengths")
                plt.xlabel("T")
                plt.ylabel("count")
                _save_fig(os.path.join(outdir, f"{name.replace(' ', '_')}_length_hist.png"))

        # finalize actions
        if action_dim is not None and action_mom is not None and action_sampler is not None and total_steps > 0:
            actions_sample = action_sampler.get()
            report["actions"] = {
                "shape": [int(total_steps), int(action_dim)],
                "idle_steps": int(idle_steps),
                "idle_ratio": float(idle_steps / total_steps),
                "saturated_steps": int(sat_steps),
                "saturated_ratio": float(sat_steps / total_steps),
                "stats": action_mom.finalize(),
                "abs_delta_stats": delta_abs_mom.finalize() if delta_abs_mom is not None else None,
                "abs_delta_max": delta_abs_max.tolist() if delta_abs_max is not None else None,
                "sample_count": int(actions_sample.shape[0]),
            }

            # percentiles from sample
            if actions_sample.shape[0] > 0:
                qs = [1, 5, 10, 25, 50, 75, 90, 95, 99]
                perc = {f"p{q}": np.percentile(actions_sample, q, axis=0).tolist() for q in qs}
                report["actions"]["percentiles_sampled"] = perc

            print(f"\nAction Analysis:")
            print(f"  total_steps: {total_steps}")
            print(f"  action_dim: {action_dim}")
            print(f"  idle_ratio (abs<{opts.deadzone} all dims): {idle_steps/total_steps:.4f}")
            print(f"  saturated_ratio (any |a|>={opts.saturation_threshold}): {sat_steps/total_steps:.4f}")

            # plots from sample
            if outdir and opts.save_plots and actions_sample.shape[0] > 0:
                # histograms
                D = int(action_dim)
                ncols = 4
                nrows = int(math.ceil(min(D, 12) / ncols))
                plt.figure(figsize=(5 * ncols, 3.5 * nrows))
                for i in range(min(D, 12)):
                    ax = plt.subplot(nrows, ncols, i + 1)
                    if sns is not None:
                        sns.histplot(actions_sample[:, i], ax=ax, bins=60, kde=True)
                    else:
                        ax.hist(actions_sample[:, i], bins=60)
                    ax.set_title(f"Action dim {i}")
                _save_fig(os.path.join(outdir, f"{name.replace(' ', '_')}_action_hist.png"))

                # correlation heatmap (sample)
                if min(D, actions_sample.shape[0]) >= 3 and D <= 64:
                    corr = np.corrcoef(actions_sample.T)
                    plt.figure(figsize=(8, 6))
                    if sns is not None:
                        sns.heatmap(corr, cmap="coolwarm", vmin=-1, vmax=1)
                    else:
                        plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
                        plt.colorbar()
                    plt.title(f"{name} - Action Correlation (sample)")
                    _save_fig(os.path.join(outdir, f"{name.replace(' ', '_')}_action_corr.png"))

        # rewards/dones
        if rewards_sum:
            r = np.array(rewards_sum, dtype=np.float64)
            report["rewards_sum"] = {
                "count": int(r.size),
                "min": float(np.min(r)),
                "max": float(np.max(r)),
                "mean": float(np.mean(r)),
                "median": float(np.median(r)),
            }
        if dones_last:
            d = np.array(dones_last, dtype=np.int64)
            report["dones_last"] = {
                "count": int(d.size),
                "done_ratio": float(np.mean(d)),
            }

        # observations
        if obs_key_info:
            report["observations"] = {
                "num_keys": len(obs_key_info),
                "keys": {k: v for k, v in sorted(obs_key_info.items(), key=lambda kv: kv[0])},
            }

        # save json
        if outdir and opts.save_json:
            json_path = os.path.join(outdir, f"{name.replace(' ', '_')}_report.json")
            with open(json_path, "w", encoding="utf-8") as wf:
                wf.write(_safe_json_dumps(report))
            print(f"\nSaved JSON report: {json_path}")

    return report


def compare_reports(r1: Dict[str, Any], r2: Dict[str, Any]) -> Dict[str, Any]:
    """
    Detailed statistical comparison for config tuning.
    """
    out: Dict[str, Any] = {"dataset_1": r1.get("name"), "dataset_2": r2.get("name")}
    
    print("\n" + "="*40)
    print(f"DEEP DIVE: {r1.get('name')} vs {r2.get('name')}")
    print("="*40)

    # --- 1. Action Statistics (Scale & Distribution) ---
    def _get_stats(r):
        return r.get("actions", {}).get("stats", {})
    
    s1, s2 = _get_stats(r1), _get_stats(r2)
    if s1 and s2:
        std1 = np.array(s1.get("std", []))
        std2 = np.array(s2.get("std", []))
        mean1 = np.array(s1.get("mean", []))
        mean2 = np.array(s2.get("mean", []))
        
        if std1.size > 0 and std2.size > 0 and std1.shape == std2.shape:
            # Scale analysis
            avg_std1 = np.mean(std1)
            avg_std2 = np.mean(std2)
            scale_ratio = avg_std2 / (avg_std1 + 1e-8)
            
            print(f"\n[Action Dynamics]")
            print(f"  Avg Action Magnitude (Std): {avg_std1:.4f} -> {avg_std2:.4f}")
            print(f"  Scale Ratio (New/Old):      {scale_ratio:.2f}x")
            
            if scale_ratio > 5.0 or scale_ratio < 0.2:
                print("  -> WARNING: Huge scale difference! Check if units match (e.g. m vs cm, rad vs deg).")
                print("     If units are correct, consider adjusting learning rate or normalization.")
            
            # Distribution Shift
            shift = np.mean(np.abs(mean2 - mean1))
            print(f"  Avg Mean Shift:             {shift:.4f}")
            
            # Dimension-wise breakdown
            print(f"  Dimension-wise Scale Change (New/Old):")
            ratio_per_dim = std2 / (std1 + 1e-8)
            print(f"    {np.round(ratio_per_dim, 2).tolist()}")

    # --- 2. Idle / Saturation ---
    def _get_ratios(r):
        act = r.get("actions", {})
        return act.get("idle_ratio", 0.0), act.get("saturated_ratio", 0.0)
    
    idle1, sat1 = _get_ratios(r1)
    idle2, sat2 = _get_ratios(r2)
    
    print(f"\n[Data Quality / Behavior]")
    print(f"  Idle Ratio:      {idle1:.1%} -> {idle2:.1%}")
    print(f"  Saturated Ratio: {sat1:.1%} -> {sat2:.1%}")
    
    if idle2 > 0.3:
        print("  -> NOTE: New dataset has high idle time. Consider filtering static frames or using a larger 'deadzone' in config.")

    # --- 3. Trajectory Lengths ---
    l1 = r1.get("trajectory_lengths", {})
    l2 = r2.get("trajectory_lengths", {})
    if l1 and l2:
        print(f"\n[Trajectory Lengths]")
        print(f"  Mean Length: {l1.get('mean',0):.1f} -> {l2.get('mean',0):.1f}")
        print(f"  Range:       [{l1.get('min')}, {l1.get('max')}] -> [{l2.get('min')}, {l2.get('max')}]")
        
        avg_len = l2.get('mean', 0)
        if avg_len > 1000:
             print("  -> NOTE: Very long trajectories. Ensure 'train.seq_length' (e.g. 10-50) is small enough relative to horizon,")
             print("     and consider if 'algo.rnn.horizon' needs tuning if using RNNs.")

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="More detailed streaming analysis for robomimic-format HDF5 datasets")
    parser.add_argument("--dataset", type=str, default=None, help="Path to a single HDF5 dataset")
    parser.add_argument("--dataset1", type=str, default=None, help="Path to dataset 1 (compare mode)")
    parser.add_argument("--dataset2", type=str, default=None, help="Path to dataset 2 (compare mode)")
    parser.add_argument("--name", type=str, default="Dataset", help="Name for --dataset")
    parser.add_argument("--name1", type=str, default="Dataset_1", help="Name for --dataset1")
    parser.add_argument("--name2", type=str, default="Dataset_2", help="Name for --dataset2")
    parser.add_argument("--output_dir", type=str, default="dataset_analysis_results", help="Where to write plots and JSON reports")

    # options
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_demos", type=int, default=None)
    parser.add_argument("--demo_sample_prob", type=float, default=1.0)
    parser.add_argument("--deadzone", type=float, default=1e-4)
    parser.add_argument("--saturation_threshold", type=float, default=0.999)
    parser.add_argument("--action_sample_max", type=int, default=200_000)
    parser.add_argument("--action_chunk_size", type=int, default=50_000)
    parser.add_argument("--obs_demo_sample_prob", type=float, default=0.2)
    parser.add_argument("--obs_frames_per_demo", type=int, default=20)
    parser.add_argument("--obs_max_total_frames_per_key", type=int, default=2_000)
    parser.add_argument("--obs_max_values_per_key", type=int, default=2_000_000)
    parser.add_argument("--no_plots", action="store_true")
    parser.add_argument("--no_json", action="store_true")

    args = parser.parse_args()

    opts = AnalyzeOptions(
        seed=args.seed,
        max_demos=args.max_demos,
        demo_sample_prob=args.demo_sample_prob,
        deadzone=args.deadzone,
        saturation_threshold=args.saturation_threshold,
        action_sample_max=args.action_sample_max,
        action_chunk_size=args.action_chunk_size,
        obs_demo_sample_prob=args.obs_demo_sample_prob,
        obs_frames_per_demo=args.obs_frames_per_demo,
        obs_max_total_frames_per_key=args.obs_max_total_frames_per_key,
        obs_max_values_per_key=args.obs_max_values_per_key,
        save_plots=not args.no_plots,
        save_json=not args.no_json,
    )

    # default behavior: keep old hardcoded paths if user provides none
    default_original = "/home/yujp/MimicPlay/mimicplay/datasets/playdata/image_demo_local.hdf5"
    default_user = "/home/yujp/Genesis/scripts/XBOX_control/dataCollect/demo/demo_multitask_cube/image_demo_local.hdf5"

    # Auto-enable compare mode if no args provided
    if not args.dataset and not args.dataset1 and not args.dataset2:
        print("No arguments provided. Running default comparison mode...")
        args.dataset1 = default_original
        args.dataset2 = default_user
        args.name1 = "Original_Reference"
        args.name2 = "User_Custom"

    if args.dataset1 or args.dataset2:
        d1 = args.dataset1 or default_original
        d2 = args.dataset2 or default_user
        r1 = analyze_dataset(d1, args.name1, args.output_dir, opts)
        r2 = analyze_dataset(d2, args.name2, args.output_dir, opts)
        diff = compare_reports(r1, r2)
        if args.output_dir and opts.save_json:
            diff_path = os.path.join(args.output_dir, "compare_report.json")
            with open(diff_path, "w", encoding="utf-8") as wf:
                wf.write(_safe_json_dumps(diff))
            print(f"\nSaved compare report: {diff_path}")
    else:
        d = args.dataset or default_original
        analyze_dataset(d, args.name, args.output_dir, opts)


if __name__ == "__main__":
    main()

