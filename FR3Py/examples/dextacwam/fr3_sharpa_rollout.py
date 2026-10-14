#!/usr/bin/env python3
"""Closed-loop DexTacWAM rollout on the FR3 + Sharpa sim, recorded to MP4.

Runs in the FR3Py devcontainer against DexTacWAM's tactile policy server
(web_infer_scripts/tactile_server_sharpa_dexmate.py on the host GPU). Every
replan sends the head-camera image, the synthetic tactile images and the robot
state; the returned chunk (absolute palm poses + hand joint targets) is executed
for --exec-horizon ticks: palm poses through IK on the FR3, hand targets
directly on the Sharpa.

Video frame: presentation camera (left), the policy's own head-camera input and
the five live tactile images (right).
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tactile_health as th  # noqa: E402
from cube_env import CubeEnv  # noqa: E402
from fr3_sharpa_smoke_client import _rpc  # noqa: E402

import socket  # noqa: E402

W, H = 1280, 720
PANEL_X = 960
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_B = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FINGERS = ("thumb", "index", "middle", "ring", "pinky")


class Recorder:
    def __init__(self, path, fps=30):
        # mp4v (the container has no ffmpeg binary); re-encode to H.264 on the host
        # for browser/Slack playback if needed.
        self.writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
        self.f = ImageFont.truetype(FONT, 17)
        self.fs = ImageFont.truetype(FONT, 14)
        self.fb = ImageFont.truetype(FONT_B, 20)

    def frame(self, main, head, tac, lines, status, status_color):
        img = Image.new("RGB", (W, H), (18, 18, 22))
        img.paste(Image.fromarray(main), (0, 0))
        d = ImageDraw.Draw(img)
        x0 = PANEL_X + 16
        d.text((x0, 12), "Policy input: head camera", font=self.f, fill=(230, 230, 230))
        img.paste(Image.fromarray(head).resize((288, 216)), (x0, 38))
        d.text((x0, 266), "Policy input: tactile (right hand)", font=self.f, fill=(230, 230, 230))
        tw, thh = 104, 78
        for k in range(5):
            tile = Image.fromarray(tac[0, k]).resize((tw, thh)).convert("RGB")
            tx = x0 + (k % 3) * (tw - 12 + 2) if k < 3 else x0 + (k - 3) * (tw - 12 + 2)
            ty = 292 if k < 3 else 292 + thh + 26
            img.paste(tile.crop((6, 0, tw - 6, thh)), (tx, ty))
            d.text((tx, ty + thh + 2), FINGERS[k], font=self.fs, fill=(170, 170, 170))
        y = 520
        for line in lines:
            d.text((x0, y), line, font=self.f, fill=(210, 210, 210))
            y += 24
        d.text((x0, H - 40), status, font=self.fb, fill=status_color)
        # caption on the main view
        d.rectangle((0, 0, PANEL_X, 40), fill=(0, 0, 0))
        d.text((14, 9), "DexTacWAM closed-loop policy  |  FR3 + Sharpa Wave (MuJoCo)",
               font=self.fb, fill=(255, 255, 255))
        self.writer.write(cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR))

    def close(self):
        self.writer.release()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=5008)
    p.add_argument("--seeds", type=int, nargs="+", default=[100_000])
    p.add_argument("--exec-horizon", type=int, default=27)
    p.add_argument("--max-ticks", type=int, default=330)
    p.add_argument("--out-dir", default=str(Path(__file__).resolve().parent / "rollouts"))
    p.add_argument("--label", default="", help="extra caption line, e.g. checkpoint step")
    p.add_argument("--debug", action="store_true", help="print palm/bar trace every 9 ticks")
    args = p.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sock = socket.create_connection((args.host, args.port))
    info = _rpc(sock, {"op": "ping"})["info"]
    assert info["arms"] == ["right"] and info["camera_names"] == ["head_img"], info
    assert info["tactile_health_sha256"] == th.module_sha256(), "re-vendor tactile_health.py"
    windows = {(k.split(":")[0], int(k.split(":")[1])): w
               for k, w in info["tactile_blank_windows"].items()}
    print(f"[rollout] task={info['task_id']} prompt={info['task_prompt']!r}")

    env = CubeEnv()
    results = []
    for seed in args.seeds:
        env.rng = np.random.default_rng(seed)
        bar_xy = env.reset()
        z0 = env.cube_pos()[2]
        _rpc(sock, {"op": "reset"})
        filt = th.OnlineTactileHealthFilter(tuple(info["arms"]), windows)
        rec = Recorder(out_dir / f"rollout_seed{seed}.mp4")
        q = env.d.qpos[:7].copy()
        tick, n_calls, lat, success_tick, execution_step = 0, 0, [], None, 0
        status, color = "running", (230, 200, 80)
        while tick < args.max_ticks:
            head, tac = env.render("head_cam"), env.tactile()
            res = filt.process(tac)
            state = np.zeros(90, dtype=np.float32)
            s45 = env.state45()
            state[7:14], state[36:58], state[74:90] = s45[:7], s45[7:29], s45[29:45]
            raw_tac = np.zeros((2, 5) + tac.shape[2:], dtype=np.uint8)
            raw_tac[1] = res.tactile[0]
            obs = {
                "images": head[None], "tactile": raw_tac, "state": state,
                "tactile_health": {"status": res.status, "contract_version": th.CONTRACT_VERSION,
                                   "sha256": th.module_sha256(), "fill_age": res.fill_age.tolist(),
                                   "valid_streak": int(res.valid_streak)},
                "prompt": info["task_prompt"], "prompt_sha256": info["task_prompt_sha256"],
                "execution_step": execution_step,
            }
            t0 = time.time()
            resp = _rpc(sock, {"op": "step", "obs": obs})
            lat.append(time.time() - t0)
            n_calls += 1
            poses = np.asarray(resp["arm_target_pose"])[:, 0]
            hands = np.asarray(resp["hand_target"])
            for i in range(min(args.exec_horizon, len(poses))):
                T = poses[i].astype(np.float64)
                u, _, vt = np.linalg.svd(T[:3, :3])
                T[:3, :3] = u @ vt  # re-orthonormalize
                q = env.ik(T, q, iters=60)
                env.step(q, hands[i])
                tick += 1
                lift = env.cube_pos()[2] - z0
                if args.debug and tick % 9 == 0:
                    print(f"  t{tick:3d} palm_tgt {np.round(T[:3, 3], 3)} palm {np.round(env.palm_T()[:3, 3], 3)} "
                          f"bar {np.round(env.cube_pos(), 3)} hand_mean {hands[i].mean():.2f}", flush=True)
                if success_tick is None and lift > 0.10:
                    success_tick = tick
                if success_tick is not None:
                    status, color = "SUCCESS: bar lifted", (90, 220, 110)
                lines = [f"t = {tick / 30:4.1f} s   policy calls: {n_calls}",
                         f"bar lift: {100 * max(lift, 0):5.1f} cm",
                         f"replan every {args.exec_horizon} ticks (~{args.exec_horizon / 30:.1f} s)",
                         args.label]
                rec.frame(env.render("video_cam", big=True), head, env.tactile(), lines, status, color)
            execution_step = args.exec_horizon
            if success_tick is not None and tick - success_tick > 45:
                break
        if success_tick is None:
            status, color = "not lifted", (230, 90, 80)
        for _ in range(20):  # hold the last frame briefly
            rec.frame(env.render("video_cam", big=True), head, env.tactile(), lines, status, color)
        rec.close()
        ok = success_tick is not None
        results.append(ok)
        print(f"[rollout] seed {seed} bar_xy={np.round(bar_xy, 3)}: {'SUCCESS' if ok else 'fail'} "
              f"({n_calls} policy calls, mean {np.mean(lat):.2f}s) -> rollout_seed{seed}.mp4", flush=True)
    env.close()
    sock.close()
    print(f"[rollout] {sum(results)}/{len(results)} succeeded")


if __name__ == "__main__":
    main()
