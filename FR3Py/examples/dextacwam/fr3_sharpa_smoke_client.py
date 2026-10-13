#!/usr/bin/env python3
"""DexTacWAM <-> FR3 + Sharpa smoke test client (runs in the FR3Py devcontainer).

Talks to DexTacWAM's tactile policy server (web_infer_scripts/
tactile_server_sharpa_dexmate.py, running on the host GPU) using its
length-prefixed pickle protocol, with observations taken from the FR3 + Sharpa
MuJoCo sim. It checks that the whole observation -> action path works:
handshake, prompt/contract hashes, raw obs packing, tactile health claim,
response shapes, finiteness and latency.

It NEVER executes the returned actions. DexTacWAM releases no trained policy,
so the server is expected to serve an untrained stand-in checkpoint whose
outputs are noise. The sim arm just holds its home pose under PD.

Raw observation the server expects (raw_obs_layout=bimanual, even for a
right-only policy; the left blocks are ignored by the server's gather):
    state   (90,)  [L_arm7, R_arm7, L_hand22, R_hand22, L_pose16, R_pose16]
    tactile (2, 5, H, W) uint8, (left, right) x (thumb..pinky), no all-zero finger
    images  (V, H, W, 3) uint8 RGB in ping's camera_names order
"""

import argparse
import os
import pickle
import socket
import struct
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tactile_health as th  # noqa: E402  (byte-identical copy; server checks the sha)

from FR3Py.sim.mujoco_sharpa import FR3SharpaSim  # noqa: E402

CONTROL_HZ = 30
CAM_H, CAM_W = 480, 640
TAC_H, TAC_W = 64, 64
TACTILE_BASELINE = 8  # "no contact" level; an all-zero finger is a dropout to the server
KP, KD = 200.0, 20.0


# ---- wire protocol (mirrors tactile_server_sharpa_dexmate.py) --------------

def _send(sock, payload):
    blob = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
    sock.sendall(struct.pack(">I", len(blob)) + blob)


def _recv(sock):
    header = b""
    while len(header) < 4:
        chunk = sock.recv(4 - len(header))
        if not chunk:
            raise ConnectionError("server disconnected")
        header += chunk
    n = struct.unpack(">I", header)[0]
    body = bytearray()
    while len(body) < n:
        chunk = sock.recv(min(1 << 16, n - len(body)))
        if not chunk:
            raise ConnectionError("server disconnected")
        body.extend(chunk)
    return pickle.loads(bytes(body))


def _rpc(sock, request):
    _send(sock, request)
    resp = _recv(sock)
    if not resp.get("ok", False):
        raise RuntimeError(f"server error for op={request.get('op')!r}: "
                           f"{resp.get('traceback', resp.get('error', resp))}")
    return resp


# ---- FR3 + Sharpa observation source ---------------------------------------

class SimObs:
    def __init__(self):
        self.sim = FR3SharpaSim(interface_type="torque", render=False)
        self.renderer = mujoco.Renderer(self.sim.model, CAM_H, CAM_W)
        # Fixed "head" camera looking over the table at the arm workspace.
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.cam.lookat[:] = [0.45, 0.0, 0.3]
        self.cam.distance = 1.4
        self.cam.azimuth = 180
        self.cam.elevation = -35

    def hold(self, seconds):
        """PD-hold the arm at home (actions are never executed)."""
        for _ in range(max(1, int(seconds / self.sim.dt))):
            s = self.sim.getJointStates()["arm"]
            self.sim.setCommands(KP * (self.sim.q0 - s["q"]) - KD * s["dq"])

    def images(self):
        self.renderer.update_scene(self.sim.data, camera=self.cam)
        return self.renderer.render()[None].copy()  # (1, H, W, 3) uint8

    def palm_pose16(self):
        bid = self.sim.palm_body_id
        T = np.eye(4)
        T[:3, :3] = self.sim.data.xmat[bid].reshape(3, 3)
        T[:3, 3] = self.sim.data.xpos[bid]
        return T.reshape(16)  # row-major, as data/utils/pose_math.mat16_to_mat44

    def state90(self):
        js = self.sim.getJointStates()
        s = np.zeros(90, dtype=np.float32)
        s[7:14] = js["arm"]["q"]        # R_arm
        s[36:58] = js["hand"]["q"]      # R_hand (FR3Py JOINT_ORDER)
        s[74:90] = self.palm_pose16()   # R_cur_pose
        # Left arm/hand/pose stay zero: right-only policies never read them.
        return s

    def tactile(self):
        tac = np.zeros((2, 5, TAC_H, TAC_W), dtype=np.uint8)
        tac[1] = TACTILE_BASELINE  # right hand only; no sim tactile model yet
        return tac

    def close(self):
        self.renderer.close()
        self.sim.close()


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=5008)
    p.add_argument("--steps", type=int, default=5, help="policy calls to make")
    p.add_argument("--exec-horizon", type=int, default=16,
                   help="steps 'consumed' per chunk (sim just holds for this long)")
    p.add_argument("--save-frame", default="", help="save the first head image (png)")
    args = p.parse_args()

    sock = socket.create_connection((args.host, args.port))
    info = _rpc(sock, {"op": "ping"})["info"]
    print(f"[client] task={info['task_id']}  layout={info['arm_layout']}  "
          f"cameras={info['camera_names']}  chunk={info['action_chunk']}  "
          f"mode={info['action_mode']}")
    assert info["raw_state_dim"] == 90, info["raw_state_dim"]
    assert info["arms"] == ["right"], f"expected a right-only policy, got {info['arms']}"
    assert len(info["camera_names"]) == 1, "this smoke client renders one head camera"
    assert info["tactile_health_sha256"] == th.module_sha256(), "re-vendor tactile_health.py"
    prompt = info["task_prompt"]
    print(f"[client] prompt: {prompt!r}")

    windows = {(k.split(":")[0], int(k.split(":")[1])): w
               for k, w in info["tactile_blank_windows"].items()}
    filt = th.OnlineTactileHealthFilter(tuple(info["arms"]), windows)

    obs_src = SimObs()
    obs_src.hold(0.5)
    _rpc(sock, {"op": "reset"})

    latencies = []
    execution_step = 0
    try:
        for i in range(args.steps):
            images = obs_src.images()
            if i == 0 and args.save_frame:
                import PIL.Image
                PIL.Image.fromarray(images[0]).save(args.save_frame)
            state = obs_src.state90()
            tac = obs_src.tactile()
            res = filt.process(tac[1:2])  # filter runs on the model's hands
            if not res.usable:
                print(f"[client] tactile {res.status}; holding")
                obs_src.hold(1.0 / CONTROL_HZ)
                continue
            tac[1:2] = res.tactile
            obs = {
                "images": images,
                "tactile": tac,
                "tactile_health": {
                    "status": res.status,
                    "contract_version": th.CONTRACT_VERSION,
                    "sha256": th.module_sha256(),
                    "fill_age": res.fill_age.tolist(),
                    "valid_streak": int(res.valid_streak),
                },
                "state": state,
                "prompt": prompt,
                "prompt_sha256": info["task_prompt_sha256"],
                "execution_step": execution_step,
            }
            t0 = time.time()
            resp = _rpc(sock, {"op": "step", "obs": obs})
            dt = time.time() - t0
            latencies.append(dt)

            pose = np.asarray(resp["arm_target_pose"])
            hand = np.asarray(resp["hand_target"])
            act = np.asarray(resp["action"])
            assert pose.shape == (info["action_chunk"], 1, 4, 4), pose.shape
            assert hand.shape == (info["action_chunk"], 22), hand.shape
            assert np.isfinite(pose).all() and np.isfinite(hand).all() and np.isfinite(act).all()
            cur = state[74:90].reshape(4, 4)
            dpos = np.linalg.norm(pose[:, 0, :3, 3] - cur[:3, 3], axis=-1)
            dhand = np.abs(hand - state[36:58]).max(axis=-1)
            print(f"[client] step {i}: {dt:5.2f}s  action{act.shape}  "
                  f"|EEF target - current| {dpos.min():.3f}..{dpos.max():.3f} m  "
                  f"max|hand target - current| {dhand.max():.2f} rad  (NOT executed)")

            obs_src.hold(args.exec_horizon / CONTROL_HZ)
            execution_step = args.exec_horizon
    finally:
        obs_src.close()
        sock.close()

    if latencies:
        lat = np.array(latencies)
        warm = lat[1:] if len(lat) > 1 else lat
        print(f"[client] OK: {len(lat)} policy calls; first {lat[0]:.2f}s, "
              f"then mean {warm.mean():.2f}s per {info['action_chunk']}-step chunk")


if __name__ == "__main__":
    main()
