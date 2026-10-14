#!/usr/bin/env python3
"""Collect scripted bar-pick demos on the FR3 + Sharpa sim (runs in the devcontainer).

Each successful episode is saved as raw arrays in one .npz; convert them to
DexTacWAM's LeRobot parquet layout on the host with
DexTacWAM/fr3_integration/npz_to_lerobot.py.

Per ~30 Hz tick t (observation recorded before the tick's action is applied):
  head_img  (192, 256, 3) uint8
  tactile   (1, 5, 192, 256) uint8      right hand, thumb..pinky
  state     (45,)  [R_arm7, R_hand22, R_pose16]               right_only layout
  actions   (75,)  [force30, arm_target_joints7, hand_target22, arm_target_pose16]
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cube_env import CubeEnv  # noqa: E402
from scripted_expert import ScriptedExpert  # noqa: E402

LIFT_SUCCESS = 0.10


def run_episode(env, seed):
    env.rng = np.random.default_rng(10_000 + seed)
    env.reset()
    z0 = env.cube_pos()[2]
    expert = ScriptedExpert(env, rng=np.random.default_rng(seed))
    rec = {k: [] for k in ("head_img", "tactile", "state", "actions")}
    while not expert.done:
        rec["head_img"].append(env.render("head_cam"))
        rec["tactile"].append(env.tactile())
        rec["state"].append(env.state45())
        force = env.fingertip_forces().reshape(30)
        q, h, T = expert.act()
        rec["actions"].append(np.concatenate([force, q, h, T.reshape(16)]).astype(np.float32))
        env.step(q, h)
    ok = env.cube_pos()[2] - z0 > LIFT_SUCCESS
    return ok, {k: np.stack(v) for k, v in rec.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=100, help="successful episodes to keep")
    p.add_argument("--out", default=str(Path(__file__).resolve().parent / "data_raw"))
    p.add_argument("--seed0", type=int, default=0)
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    env = CubeEnv(render=True)
    kept, seed = 0, args.seed0
    while kept < args.n:
        ok, ep = run_episode(env, seed)
        if ok:
            np.savez_compressed(out / f"episode_{kept:06d}.npz", seed=seed, **ep)
            kept += 1
            print(f"[collect] seed {seed}: kept as episode {kept - 1} ({len(ep['state'])} frames)", flush=True)
        else:
            print(f"[collect] seed {seed}: failed, skipped", flush=True)
        seed += 1
    env.close()


if __name__ == "__main__":
    main()
