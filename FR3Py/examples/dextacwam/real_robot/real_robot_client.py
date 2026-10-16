#!/usr/bin/env python3
"""DexTacWAM client for the real FR3 + Sharpa Wave (runs in the FR3Py devcontainer
on the robot PC).

The policy runs on the L40 server (DexTacWAM/fr3_integration/serve_policy.sh) and
is reached through an SSH tunnel (ssh_tunnel.sh), so the client talks to
127.0.0.1:<port>. Each policy call sends the head-camera image, per-finger
tactile images and the robot state; the reply is a 54-step chunk of absolute
palm poses + hand joint targets.

Modes, in the order you should use them (only `execute` moves the robot):

  check    print arm/hand state, palm pose (FK) and per-finger tactile force;
           save one camera frame. Press each fingertip to verify FINGER_CHANNELS.
  align    save camera images blended with the sim head-camera view rendered at
           the robot's current joints, to place the physical camera where the
           training camera was.
  shadow   full policy loop, NOTHING is commanded: logs every predicted chunk and
           writes a video of the live camera next to the sim robot posed at the
           predicted targets.
  execute  runs the policy on the robot behind the safety limits below.

Arm control: the FR3 bridge must run fr3_joint_velocity. A 100 Hz thread
streams joint velocities that track the current joint target (P-control, speed
capped) and keeps holding it while the policy is being queried. The bridge
keeps executing a stale velocity for up to 1 s, so the arm is never left
without an active command.
"""

import argparse
import os
import pickle
import signal
import socket
import struct
import sys
import threading
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import cv2  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import tactile_health as th  # noqa: E402  (byte-identical copy; the server checks its sha)
from cube_env import POST_H, Q_HOME, TABLE_TOP, CubeEnv  # noqa: E402

# ---- task / interface constants ---------------------------------------------
CONTROL_HZ = 30                 # policy action rate (matches the training data)
ARM_LOOP_HZ = 100               # joint-velocity streaming rate
IMG_H, IMG_W = 192, 256         # DexTacWAM input size; resized here to save bandwidth
TAC_H, TAC_W = 192, 256
TACTILE_BASELINE = 6            # same "no contact" level as the sim tactile
FINGERS = ("thumb", "index", "middle", "ring", "pinky")
# SDK tactile channel for each DexTacWAM finger (thumb, index, middle, ring, pinky).
# FR3Py's SharpaHand maps channels 0-3 to the four-finger UV map and 4 to the
# thumb map, so the assumed order is [index, middle, ring, pinky, thumb].
# VERIFY with `--mode check` by pressing each fingertip.
FINGER_CHANNELS = (4, 0, 1, 2, 3)

# ---- safety limits (execute mode) -------------------------------------------
# Palm targets are clamped to a box around the training workspace (robot base
# frame, metres). The bar sat at x 0.45-0.55, y +/-0.06, z = TABLE_TOP + POST_H.
WORKSPACE_LO = np.array([0.25, -0.25, TABLE_TOP + POST_H - 0.02])
WORKSPACE_HI = np.array([0.70, 0.25, 0.75])
MAX_PALM_STEP = 0.008           # m per 30 Hz action (0.24 m/s)
MAX_JOINT_VEL = 0.5             # rad/s, per arm joint
ARM_KP = 3.0                    # 1/s, joint-velocity P gain toward the target
MAX_HAND_STEP = 0.08            # rad per 30 Hz action, per hand joint
JOINT_MARGIN = 0.05             # rad kept clear of the FR3 joint limits


# ---- wire protocol (DexTacWAM tactile_server_sharpa_dexmate.py) ----------------

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


def rpc(sock, request):
    _send(sock, request)
    resp = _recv(sock)
    if not resp.get("ok", False):
        raise RuntimeError(f"server error for op={request.get('op')!r}: "
                           f"{resp.get('traceback', resp.get('error', resp))}")
    return resp


# ---- hardware ---------------------------------------------------------------

def make_hand(**kwargs):
    """SharpaHand that also keeps the per-channel RAW / DEFORM tactile images."""
    from FR3Py.robot.hand_interface import SharpaHand

    class SharpaHandWithImages(SharpaHand):
        def _tactile_callback(self, frame):
            super()._tactile_callback(frame)
            content, shape = frame.get("content", {}), frame.get("shape", {})
            imgs = {}
            for key in ("RAW", "DEFORM"):
                data = content.get(key)
                if data is not None and key in shape:
                    h, w = shape[key][1], shape[key][2]
                    imgs[key] = np.asarray(data, dtype=np.uint8).reshape(h, w)
            with self._frames_lock:
                if not hasattr(self, "images"):
                    self.images = {}
                self.images[frame["channel"]] = imgs

        def getTactileImages(self, key):
            with self._frames_lock:
                return {ch: im.get(key) for ch, im in getattr(self, "images", {}).items()}

    return SharpaHandWithImages(**kwargs)


class Robot:
    """FR3 (LCM, joint-velocity bridge) + Sharpa hand + RealSense head camera."""

    def __init__(self, robot_id, camera_serial, need_camera=True):
        from FR3Py.robot.interface import FR3Real

        self.arm = FR3Real(robot_id=robot_id)
        self.hand = make_hand(hand_side=1, enable_tactile=True)
        self.camera = None
        if need_camera:
            from FR3Py.robot.vision import RealSenseCamera
            self.camera = RealSenseCamera(camera_serial_no=camera_serial, VGA=True,
                                          color_fps=30, depth_fps=30, enable_ir=True)
        time.sleep(1.0)

    def arm_q(self):
        s = self.arm.getJointStates()
        return None if s is None else np.asarray(s["q"], dtype=np.float64)

    def hand_q(self):
        return np.asarray(self.hand.getJointStates()["q"], dtype=np.float64)

    def image(self):
        bgr = self.camera.color_frame
        if bgr is None:
            raise RuntimeError("no camera frame yet")
        return cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (IMG_W, IMG_H),
                          interpolation=cv2.INTER_AREA)

    def finger_forces(self):
        tac = self.hand.getTactile()
        chans = sorted(tac)
        return np.array([tac[chans[c]]["force"] for c in FINGER_CHANNELS], dtype=np.float64)

    def tactile(self, source, force_threshold):
        """(1, 5, 192, 256) uint8 in DexTacWAM finger order.

        synthetic: the sim's rendering (a blob at the fingertip row, sized by the
                   measured force) -- what the sim-trained policy saw in training.
        raw / deform: the real sensor images, for policies trained on real data.
        """
        out = np.full((1, 5, TAC_H, TAC_W), TACTILE_BASELINE, dtype=np.uint8)
        if source == "synthetic":
            yy, xx = np.mgrid[0:TAC_H, 0:TAC_W]
            img = out[0].astype(np.float32)
            for k, f in enumerate(self.finger_forces()):
                if f < force_threshold:
                    continue
                s = min(f / 3.0, 1.0)
                sigma = 14 + 30 * s
                blob = np.exp(-((xx - TAC_W / 2) ** 2 + (yy - 0.22 * TAC_H) ** 2) / (2 * sigma ** 2))
                img[k] += (255 - TACTILE_BASELINE) * (0.4 + 0.6 * s) * blob
            out[0] = np.clip(img, 0, 255).astype(np.uint8)
            return out
        imgs = self.hand.getTactileImages(source.upper())
        chans = sorted(imgs)
        for k, c in enumerate(FINGER_CHANNELS):
            im = imgs.get(chans[c]) if c < len(chans) else None
            if im is not None:
                out[0, k] = cv2.resize(im, (TAC_W, TAC_H), interpolation=cv2.INTER_AREA)
        return out

    def close(self):
        for part in (self.camera, self.hand, self.arm):
            try:
                if part is not None:
                    part.close()
            except Exception as e:  # keep closing the rest
                print(f"[client] close error: {e}")


class ArmStreamer(threading.Thread):
    """Streams joint velocities at ARM_LOOP_HZ that drive the arm to `target`.

    target None -> zero velocity. A missing state (bridge silent > 0.2 s)
    trips the streamer: it sends zeros and refuses new targets.
    """

    def __init__(self, robot, lo, hi):
        super().__init__(daemon=True)
        self.robot, self.lo, self.hi = robot, lo + JOINT_MARGIN, hi - JOINT_MARGIN
        self.target = None
        self.lock = threading.Lock()
        self.running = True
        self.fault = None

    def set_target(self, q):
        if self.fault:
            raise RuntimeError(f"arm streamer faulted: {self.fault}")
        with self.lock:
            self.target = None if q is None else np.clip(q, self.lo, self.hi)

    def run(self):
        period = 1.0 / ARM_LOOP_HZ
        while self.running:
            t0 = time.time()
            q = self.robot.arm_q()
            with self.lock:
                target = self.target
            if q is None:
                if self.fault is None:
                    self.fault = "no arm state from the bridge"
                    print("[client] ARM STATE LOST -> zero velocity, faulted", flush=True)
                target = None
            v = np.zeros(7) if target is None or self.fault else \
                np.clip(ARM_KP * (target - q), -MAX_JOINT_VEL, MAX_JOINT_VEL)
            self.robot.arm.setCommands(v)
            time.sleep(max(0.0, period - (time.time() - t0)))
        for _ in range(20):
            self.robot.arm.setCommands(np.zeros(7))
            time.sleep(0.005)

    def stop(self):
        self.running = False
        self.join(timeout=2.0)

    def wait_reached(self, tol=0.02, timeout=20.0):
        t0 = time.time()
        while time.time() - t0 < timeout:
            q = self.robot.arm_q()
            with self.lock:
                tgt = self.target
            if q is not None and tgt is not None and np.max(np.abs(q - tgt)) < tol:
                return True
            time.sleep(0.05)
        return False


# ---- helpers ------------------------------------------------------------------

def set_kinematics(kin, arm_q, hand_q):
    import mujoco
    kin.d.qpos[:7] = arm_q
    kin.d.qpos[kin.hand_qpos] = hand_q
    mujoco.mj_forward(kin.m, kin.d)


def raw_state90(kin):
    s45 = kin.state45()
    s = np.zeros(90, dtype=np.float32)
    s[7:14], s[36:58], s[74:90] = s45[:7], s45[7:29], s45[29:45]  # right arm/hand/pose
    return s


def connect(args):
    sock = socket.create_connection((args.host, args.port), timeout=30)
    sock.settimeout(None)
    info = rpc(sock, {"op": "ping"})["info"]
    if info["task_id"] is None:
        raise RuntimeError("server runs with --task none; the hardware client needs a registered task")
    assert info["arms"] == ["right"] and info["camera_names"] == ["head_img"], info
    if info["tactile_health_sha256"] != th.module_sha256():
        raise RuntimeError("tactile_health.py differs from the server's copy; re-vendor it")
    print(f"[client] server task={info['task_id']}  prompt={info['task_prompt']!r}  "
          f"chunk={info['action_chunk']}")
    return sock, info


def make_obs(robot, kin, filt, info, args, execution_step):
    """Observation dict, or None when the tactile health filter withholds it."""
    tac = robot.tactile(args.tactile_source, args.force_threshold)
    res = filt.process(tac)
    if not res.usable:
        return None, res
    raw_tac = np.zeros((2, 5, TAC_H, TAC_W), dtype=np.uint8)
    raw_tac[1] = res.tactile[0]
    return {
        "images": robot.image()[None],
        "tactile": raw_tac,
        "state": raw_state90(kin),
        "tactile_health": {"status": res.status, "contract_version": th.CONTRACT_VERSION,
                           "sha256": th.module_sha256(), "fill_age": res.fill_age.tolist(),
                           "valid_streak": int(res.valid_streak)},
        "prompt": info["task_prompt"],
        "prompt_sha256": info["task_prompt_sha256"],
        "execution_step": execution_step,
    }, res


def safe_palm_target(T_pred, T_prev):
    """Clamp a predicted palm pose to the workspace box and the per-step limit."""
    T = np.array(T_pred, dtype=np.float64)
    u, _, vt = np.linalg.svd(T[:3, :3])
    T[:3, :3] = u @ vt
    step = T[:3, 3] - T_prev[:3, 3]
    n = np.linalg.norm(step)
    if n > MAX_PALM_STEP:
        T[:3, 3] = T_prev[:3, 3] + step * (MAX_PALM_STEP / n)
    T[:3, 3] = np.clip(T[:3, 3], WORKSPACE_LO, WORKSPACE_HI)
    return T


ASSUME_YES = False  # set only with --backend sim


def confirm(msg):
    print(msg)
    if ASSUME_YES:
        print("Type 'yes' to continue: yes  (--backend sim --yes)")
        return True
    return input("Type 'yes' to continue: ").strip().lower() == "yes"


# ---- modes --------------------------------------------------------------------

def mode_check(robot, kin, args):
    for i in range(args.check_seconds * 2):
        q, hq = robot.arm_q(), robot.hand_q()
        if q is None:
            print("[check] no arm state (is the fr3_joint_velocity bridge running?)")
        else:
            set_kinematics(kin, q, hq)
            p = kin.palm_T()[:3, 3]
            print(f"[check] arm q {np.round(q, 3)}  palm xyz {np.round(p, 3)}")
        f = robot.finger_forces()
        print("[check] tactile force (N): " + "  ".join(f"{n}={v:.2f}" for n, v in zip(FINGERS, f)))
        if i == 0 and robot.camera is not None:
            out = Path(args.out_dir) / "check_camera.png"
            cv2.imwrite(str(out), cv2.cvtColor(robot.image(), cv2.COLOR_RGB2BGR))
            print(f"[check] saved {out}")
        time.sleep(0.5)


def mode_align(robot, kin, args):
    out_dir = Path(args.out_dir)
    for i in range(args.align_frames):
        q = robot.arm_q()
        if q is None:
            print("[align] no arm state")
            time.sleep(0.5)
            continue
        set_kinematics(kin, q, robot.hand_q())
        sim = kin.render("head_cam")
        real = robot.image()
        blend = (0.5 * sim + 0.5 * real).astype(np.uint8)
        grid = np.concatenate([real, sim, blend], axis=1)
        cv2.imwrite(str(out_dir / f"align_{i:03d}.png"), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
        print(f"[align] wrote align_{i:03d}.png (real | sim head_cam | 50/50 blend)")
        time.sleep(args.align_period)


def mode_policy(robot, kin, args, execute):
    sock, info = connect(args)
    windows = {(k.split(":")[0], int(k.split(":")[1])): w
               for k, w in info["tactile_blank_windows"].items()}
    filt = th.OnlineTactileHealthFilter(tuple(info["arms"]), windows)
    out_dir = Path(args.out_dir)
    log = {"t": [], "state": [], "arm_target_pose": [], "hand_target": [], "latency": []}
    video = cv2.VideoWriter(str(out_dir / f"{'execute' if execute else 'shadow'}.mp4"),
                            cv2.VideoWriter_fourcc(*"mp4v"), 10, (IMG_W * 2, IMG_H))

    streamer = None
    if execute:
        streamer = ArmStreamer(robot, kin.arm_lo, kin.arm_hi)
        streamer.start()
        q0 = robot.arm_q()
        if q0 is None:
            raise RuntimeError("no arm state")
        streamer.set_target(q0)  # hold where we are first
        if np.max(np.abs(q0 - Q_HOME)) > 0.02:
            if not confirm(f"[execute] move the arm to the training home pose {np.round(Q_HOME, 3)} "
                           f"at <= {MAX_JOINT_VEL} rad/s?"):
                raise SystemExit("aborted")
            streamer.set_target(Q_HOME)
            if not streamer.wait_reached():
                raise RuntimeError("did not reach the home pose")
        robot.hand.setCommands(np.zeros(22))
        time.sleep(1.0)

    rpc(sock, {"op": "reset"})
    q_cmd = robot.arm_q()
    set_kinematics(kin, q_cmd, robot.hand_q())
    T_prev = kin.palm_T()
    hand_prev = robot.hand_q()
    execution_step, executed, t_start = 0, 0, time.time()
    period = 1.0 / (CONTROL_HZ * args.speed)
    try:
        while executed < args.max_steps:
            q, hq = robot.arm_q(), robot.hand_q()
            if q is None:
                raise RuntimeError("lost arm state")
            set_kinematics(kin, q, hq)
            obs, res = None, None
            # The health filter is meant for 30 Hz; feed it at that rate until usable.
            for _ in range(30):
                obs, res = make_obs(robot, kin, filt, info, args, execution_step)
                if obs is not None:
                    break
                time.sleep(1.0 / CONTROL_HZ)
            if obs is None:
                raise RuntimeError(f"tactile not usable: {res.status} {res.reasons}")
            t0 = time.time()
            resp = rpc(sock, {"op": "step", "obs": obs})
            lat = time.time() - t0
            poses = np.asarray(resp["arm_target_pose"])[:, 0]
            hands = np.asarray(resp["hand_target"])
            for k, v in (("t", time.time() - t_start), ("state", obs["state"]),
                         ("arm_target_pose", poses), ("hand_target", hands), ("latency", lat)):
                log[k].append(v)
            dz = poses[:, 2, 3] - kin.palm_T()[2, 3]
            print(f"[client] call {len(log['t'])}: {lat:.2f}s  palm target dz "
                  f"{dz.min():+.3f}..{dz.max():+.3f} m  hand mean {hands.mean():.2f} rad", flush=True)

            # Video: live camera | sim robot posed at the end of the executed horizon.
            k_end = min(args.exec_horizon, len(poses)) - 1
            ghost_q = kin.ik(safe_palm_target(poses[k_end], kin.palm_T()), q, iters=100)
            set_kinematics(kin, ghost_q, hands[k_end])
            video.write(cv2.cvtColor(np.concatenate([obs["images"][0], kin.render("head_cam")], 1),
                                     cv2.COLOR_RGB2BGR))
            set_kinematics(kin, q, hq)

            if not execute:
                time.sleep(args.exec_horizon / CONTROL_HZ)
                execution_step = 0  # nothing was executed
                executed += args.exec_horizon  # virtual, so --max-steps bounds shadow runs too
                continue
            for i in range(min(args.exec_horizon, len(poses))):
                t_tick = time.time()
                T = safe_palm_target(poses[i], T_prev)
                q_cmd = kin.ik(T, q_cmd, iters=60)
                streamer.set_target(q_cmd)
                h = hand_prev + np.clip(hands[i] - hand_prev, -MAX_HAND_STEP, MAX_HAND_STEP)
                robot.hand.setCommands(h)
                T_prev, hand_prev = T, h
                executed += 1
                time.sleep(max(0.0, period - (time.time() - t_tick)))
            execution_step = min(args.exec_horizon, len(poses))
    except KeyboardInterrupt:
        print("\n[client] interrupted -> holding position")
    finally:
        if streamer is not None:
            q = robot.arm_q()
            if q is not None:
                streamer.set_target(q)
                time.sleep(0.3)
            streamer.stop()
        video.release()
        np.savez(out_dir / f"{'execute' if execute else 'shadow'}_log.npz",
                 **{k: np.asarray(v) for k, v in log.items()})
        sock.close()
        print(f"[client] logs and video in {out_dir}")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--mode", required=True, choices=["check", "align", "shadow", "execute"])
    p.add_argument("--host", default="127.0.0.1", help="tunnel end (see ssh_tunnel.sh)")
    p.add_argument("--port", type=int, default=5008)
    p.add_argument("--robot-id", default="fr3", help="FR3Py LCM robot id")
    p.add_argument("--camera-serial", default=None, help="RealSense serial of the head camera")
    p.add_argument("--tactile-source", default="synthetic", choices=["synthetic", "raw", "deform"],
                   help="synthetic matches the sim-trained policy; raw/deform for real-data policies")
    p.add_argument("--force-threshold", type=float, default=0.3, help="N, synthetic tactile")
    p.add_argument("--exec-horizon", type=int, default=27, help="actions executed per policy call")
    p.add_argument("--max-steps", type=int, default=300, help="executed actions before stopping")
    p.add_argument("--speed", type=float, default=0.5, help="action playback rate vs 30 Hz")
    p.add_argument("--check-seconds", type=int, default=10)
    p.add_argument("--align-frames", type=int, default=20)
    p.add_argument("--align-period", type=float, default=1.0)
    p.add_argument("--out-dir", default=str(HERE / "logs" / time.strftime("%Y%m%d_%H%M%S")))
    p.add_argument("--backend", default="real", choices=["real", "sim"],
                   help="sim: MuJoCo stand-in with the real robot's interfaces, for testing")
    p.add_argument("--yes", action="store_true", help="auto-confirm prompts (sim backend only)")
    args = p.parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    if args.yes and args.backend != "sim":
        raise SystemExit("--yes is only allowed with --backend sim")
    global ASSUME_YES
    ASSUME_YES = args.yes

    def _sigterm(signum, frame):  # docker stop / kill: same stop-and-hold path as Ctrl-C
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, _sigterm)

    if args.mode == "execute" and not confirm(
            "[execute] This MOVES the real FR3 + Sharpa. Check that:\n"
            "  * the bridge is fr3_joint_velocity (NOT torque) and the e-stop is in reach,\n"
            "  * --mode shadow predictions looked sane for this scene,\n"
            f"  * the workspace box {WORKSPACE_LO.round(2)}..{WORKSPACE_HI.round(2)} m is clear."):
        raise SystemExit("aborted")

    kin = CubeEnv(render=args.mode in ("align", "shadow", "execute"))
    kin.reset(np.array([0.5, 0.0]))  # nominal scene for the align / ghost renders
    if args.backend == "sim":
        from sim_backend import SimRobot
        robot = SimRobot()
    else:
        robot = Robot(args.robot_id, args.camera_serial, need_camera=True)
    try:
        if args.mode == "check":
            mode_check(robot, kin, args)
        elif args.mode == "align":
            mode_align(robot, kin, args)
        else:
            mode_policy(robot, kin, args, execute=args.mode == "execute")
    finally:
        if args.backend == "sim":
            print(f"[client] sim backend: bar lift {100 * robot.bar_lift():.1f} cm")
        robot.close()
        kin.close()


if __name__ == "__main__":
    main()
