"""
独立的自动数据采集脚本（Genesis + robomimic wrapper）
---------------------------------------------------
数据保存格式严格对齐:
  scripts/XBOX_control/dataCollect/dataCollect_mimicplay_env copy 2.py

每一轮（demo / episode）流程：
1) reset 归位
2) 随机抓取 object_1 / object_2 / object_3 中一个
3) 在剩余两个物体中随机选一个作为“底座” object_n，将抓取物体放到 (x, y, z + 0.04)，松开夹爪
4) 抓取最后剩下的物体，放到 (x, y, z + 0.08)，松开夹爪
5) 结束这一轮
"""

import argparse
import json
import os
import sys
import time
import datetime
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np

try:
    import cv2
except Exception:
    cv2 = None


sys.path.insert(0, "/home/yujp/robomimic")
from robomimic.envs.env_genesis import GenesisEnvWrapper


@dataclass
class DemoBuffer:
    demo_id: str
    states: List[np.ndarray]
    actions: List[np.ndarray]
    timestamps: List[float]


class AutoStackCollector:
    DATA_INTERVAL = 0.05  # 20Hz，与参考脚本一致

    def __init__(
        self,
        env: GenesisEnvWrapper,
        save_dir: str,
        num_demos: int,
        seed: Optional[int] = None,
        render: bool = False,
        max_move_steps: int = 400,
        pos_tol: float = 0.004,
        hover_z: float = 0.2,
    ):
        self.env = env
        self.save_dir = save_dir
        self.num_demos = int(num_demos)
        self.render = bool(render)
        self.max_move_steps = int(max_move_steps)
        self.pos_tol = float(pos_tol)
        self.hover_z = float(hover_z)

        self.rng = np.random.default_rng(seed)
        self.demos: List[DemoBuffer] = []

        self.sim_time_total = 0.0
        self._demo_start_time_total = 0.0
        self._step_idx_in_demo = 0

        self.env_config = {
            "type": 4,
            "env_name": "Desk_Environment",
            "robots": ["Panda"],
            "has_renderer": True,
            "has_offscreen_renderer": False,
            "render_camera": "gripper_cam",
            "control_freq": 20,
            "collector": "auto_collect_stack",
        }

        os.makedirs(self.save_dir, exist_ok=True)

    def _record_after_step(self, demo: DemoBuffer, action: np.ndarray):
        state_dict = self.env.get_state()
        demo.states.append(np.asarray(state_dict["states"]).copy())
        demo.actions.append(np.asarray(action).copy())
        demo.timestamps.append(self._step_idx_in_demo * self.DATA_INTERVAL)
        self._step_idx_in_demo += 1

    def _step(self, demo: DemoBuffer, action: np.ndarray):
        obs, reward, done, info = self.env.step(action)
        self.sim_time_total += self.DATA_INTERVAL
        self._record_after_step(demo, action)
        if self.render:
            self._render_collect_view()
        return obs, reward, done, info

    def _render_collect_view(self):
        if cv2 is None:
            return
        try:
            self.env.render(mode="collect", waitkey=1)
        except Exception:
            return

    def _get_object_pos(self, obj_idx: int) -> np.ndarray:
        pos, quat, vel, ang = self.env.movable_objects[obj_idx].get_state()
        return np.asarray(pos).reshape(3)

    def _get_eef_pos(self) -> np.ndarray:
        return self.env.end_effector.get_pos().cpu().numpy().squeeze().copy()

    def _make_action(self, delta_pos_world: np.ndarray, gripper_closed: bool) -> np.ndarray:
        delta = np.asarray(delta_pos_world, dtype=np.float32).reshape(3)
        delta = np.clip(delta, -0.0025, 0.0025)
        gripper_signal = 1.0 if gripper_closed else -1.0
        action = np.concatenate([delta * 400.0, np.zeros(3, dtype=np.float32), np.array([gripper_signal], dtype=np.float32)])
        return action.astype(np.float32)

    def _move_eef_to(self, demo: DemoBuffer, target_pos: np.ndarray, gripper_closed: bool) -> bool:
        target = np.asarray(target_pos, dtype=np.float32).reshape(3)
        step_i = 0
        while step_i < self.max_move_steps:
            cur = self._get_eef_pos()
            err = target - cur
            if float(np.linalg.norm(err)) <= self.pos_tol:
                return True
            action = self._make_action(err, gripper_closed=gripper_closed)
            self._step(demo, action)
            step_i += 1
        return False

    def _set_gripper(self, demo: DemoBuffer, gripper_closed: bool, hold_steps: int = 8):
        i = 0
        while i < int(hold_steps):
            action = self._make_action(np.zeros(3, dtype=np.float32), gripper_closed=gripper_closed)
            self._step(demo, action)
            i += 1

    def _pick_object(self, demo: DemoBuffer, obj_idx: int) -> bool:
        obj_pos = self._get_object_pos(obj_idx)

        hover = obj_pos.copy()
        hover[2] += self.hover_z
        pregrasp = obj_pos.copy()
        pregrasp[2] += 0.1

        ok = self._move_eef_to(demo, hover, gripper_closed=False)
        if not ok:
            return False
        ok = self._move_eef_to(demo, pregrasp, gripper_closed=False)
        if not ok:
            return False
        self._set_gripper(demo, gripper_closed=True, hold_steps=10)
        ok = self._move_eef_to(demo, hover, gripper_closed=True)
        return bool(ok)

    def _place_on_object(self, demo: DemoBuffer, base_obj_idx: int, z_offset: float) -> bool:
        base_pos = self._get_object_pos(base_obj_idx)
        target = base_pos.copy()
        target[2] += float(z_offset)

        hover = target.copy()
        hover[2] += self.hover_z
        preplace = target.copy()
        preplace[2] += 0.02

        ok = self._move_eef_to(demo, hover, gripper_closed=True)
        if not ok:
            return False
        ok = self._move_eef_to(demo, preplace, gripper_closed=True)
        if not ok:
            return False
        self._set_gripper(demo, gripper_closed=False, hold_steps=10)
        ok = self._move_eef_to(demo, hover, gripper_closed=False)
        return bool(ok)

    def _run_one_demo(self, demo_idx: int) -> DemoBuffer:
        demo = DemoBuffer(demo_id="demo_" + str(demo_idx), states=[], actions=[], timestamps=[])
        self._demo_start_time_total = self.sim_time_total
        self._step_idx_in_demo = 0

        self.env.reset()
        self._set_gripper(demo, gripper_closed=False, hold_steps=8)

        objs = [0, 1, 2]
        pick_idx = int(self.rng.choice(objs))
        remaining = [i for i in objs if i != pick_idx]
        base_idx = int(self.rng.choice(remaining))
        last_idx = int(remaining[0] if remaining[1] == base_idx else remaining[1])

        ok = self._pick_object(demo, pick_idx)
        if not ok:
            return demo

        ok = self._place_on_object(demo, base_idx, z_offset=0.138)
        if not ok:
            return demo

        ok = self._pick_object(demo, last_idx)
        if not ok:
            return demo

        ok = self._place_on_object(demo, base_idx, z_offset=0.178)
        return demo

    def run(self):
        print("自动数采启动，将采集 " + str(self.num_demos) + " 轮。保存目录: " + str(self.save_dir))
        demo_i = 0
        while demo_i < self.num_demos:
            print("\n开始采集: demo_" + str(demo_i))
            demo = self._run_one_demo(demo_i)
            self.demos.append(demo)
            print("完成 demo_" + str(demo_i) + "，样本数: " + str(len(demo.states)))
            demo_i += 1

        self._save_to_hdf5()
        print("全部完成。已保存到: " + os.path.join(self.save_dir, "demos.hdf5"))

    def _save_to_hdf5(self):
        if not self.demos:
            print("没有数据需要保存")
            return

        hdf5_path = os.path.join(self.save_dir, "demos.hdf5")
        print("保存 " + str(len(self.demos)) + " 个demo到 " + str(hdf5_path) + " ...")

        total_samples = 0
        with h5py.File(hdf5_path, "w") as f:
            grp = f.create_group("data")
            grp.attrs["env"] = "Desk Environment"
            grp.attrs["env_info"] = json.dumps(self.env_config)
            grp.attrs["repository_version"] = "4.0.0"

            i = 0
            while i < len(self.demos):
                demo = self.demos[i]
                demo_grp = grp.create_group("demo_" + str(i))
                demo_grp.attrs["num_samples"] = len(demo.states)
                demo_grp.attrs["timestamps"] = json.dumps(demo.timestamps)
                demo_grp.create_dataset("states", data=np.array(demo.states))
                demo_grp.create_dataset("actions", data=np.array(demo.actions))
                total_samples += len(demo.states)
                i += 1

            grp.attrs["total"] = total_samples

        print("成功保存 " + str(len(self.demos)) + " 个demo，总样本数: " + str(total_samples))


def _default_save_dir() -> str:
    ts = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    return "/home/yujp/Genesis/scripts/XBOX_control/dataCollect/demo/auto_demo" + ts


def main():
    parser = argparse.ArgumentParser(description="自动堆叠任务数采（保存格式对齐 MimicPlay HDF5）")
    parser.add_argument("--condition_file", type=str, default=None, help="成功条件文件（可选，不影响自动流程）")
    parser.add_argument("--save_dir", type=str, default=None, help="保存目录（默认自动生成时间戳目录）")
    parser.add_argument("--num_demos", type=int, default=200, help="采集轮数（demo数量）")
    parser.add_argument("--seed", type=int, default=None, help="随机种子")
    parser.add_argument("--render", action="store_true", help="显示三相机拼接画面（需要opencv）")
    parser.add_argument("--camera_width", type=int, default=640)
    parser.add_argument("--camera_height", type=int, default=480)
    parser.add_argument("--max_move_steps", type=int, default=400)
    args = parser.parse_args()

    if args.save_dir is None:
        args.save_dir = _default_save_dir()

    env_config: Dict[str, str] = {"env_name": "Desk_Environment"}
    if args.condition_file is not None:
        env_config["condition_file"] = args.condition_file

    env = GenesisEnvWrapper(env_config=env_config, camera_width=args.camera_width, camera_height=args.camera_height)
    collector = AutoStackCollector(
        env=env,
        save_dir=args.save_dir,
        num_demos=args.num_demos,
        seed=args.seed,
        render=args.render,
        max_move_steps=args.max_move_steps,
    )

    try:
        collector.run()
    except KeyboardInterrupt:
        print("\n收到退出信号，尝试保存已采集数据...")
        try:
            collector._save_to_hdf5()
        except Exception as e:
            print("保存失败: " + str(e))


if __name__ == "__main__":
    main()

