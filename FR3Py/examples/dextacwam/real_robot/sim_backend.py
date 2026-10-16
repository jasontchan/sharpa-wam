"""Stand-in for the real FR3 + Sharpa + RealSense, backed by the MuJoCo bar scene.

Exposes the subset of the hardware API real_robot_client.py uses, with the real
robot's semantics: the arm takes joint VELOCITY commands (like the
fr3_joint_velocity bridge) and the hand takes joint position targets. Physics
runs in its own thread in real time, so the client's 100 Hz streamer, policy
latency and pauses behave as they would on hardware.

  python3 real_robot_client.py --mode execute --backend sim
"""

import threading
import time

import mujoco
import numpy as np

from cube_env import CubeEnv

IMG_H, IMG_W = 192, 256


class _Arm:
    def __init__(self, owner):
        self.owner = owner

    def getJointStates(self):
        with self.owner.lock:
            d = self.owner.env.d
            return {"q": d.qpos[:7].copy(), "dq": d.qvel[:7].copy()}

    def setCommands(self, cmd):
        assert np.shape(cmd) == (7,)
        self.owner.v_cmd = np.asarray(cmd, dtype=np.float64)
        self.owner.v_stamp = time.time()

    def close(self):
        pass


class _Hand:
    def __init__(self, owner):
        self.owner = owner

    def getJointStates(self):
        with self.owner.lock:
            return {"q": self.owner.env.d.qpos[self.owner.env.hand_qpos].astype(np.float32)}

    def setCommands(self, cmd):
        self.owner.hand_des = np.asarray(cmd, dtype=np.float64)

    def close(self):
        pass


class SimRobot:
    def __init__(self, seed=100_000):
        self.env = CubeEnv()
        self.env.rng = np.random.default_rng(seed)
        self.env.reset()
        self.z0 = self.env.cube_pos()[2]
        self.lock = threading.Lock()
        self.v_cmd, self.v_stamp = np.zeros(7), 0.0
        self.hand_des = self.env.d.qpos[self.env.hand_qpos].copy()
        self.arm, self.hand, self.camera = _Arm(self), _Hand(self), True
        self.running = True
        self.thread = threading.Thread(target=self._physics, daemon=True)
        self.thread.start()

    def _physics(self):
        env, dt = self.env, self.env.m.opt.timestep
        while self.running:
            t0 = time.time()
            with self.lock:
                d = env.d
                # bridge semantics: a command older than 1 s is replaced by zero
                v = self.v_cmd if time.time() - self.v_stamp < 1.0 else np.zeros(7)
                d.ctrl[:7] = d.qfrc_bias[:7] + 40.0 * (v - d.qvel[:7])
                d.ctrl[env.hand_act] = np.clip(self.hand_des, env.hand_range[:, 0], env.hand_range[:, 1])
                mujoco.mj_step(env.m, d)
            time.sleep(max(0.0, dt - (time.time() - t0)))

    # -- same methods as real_robot_client.Robot ----------------------------
    def arm_q(self):
        return self.arm.getJointStates()["q"]

    def hand_q(self):
        return self.hand.getJointStates()["q"].astype(np.float64)

    def image(self):
        with self.lock:
            return self.env.render("head_cam")

    def finger_forces(self):
        with self.lock:
            return np.linalg.norm(self.env.fingertip_forces()[:, :3], axis=1)

    def tactile(self, source, force_threshold):
        with self.lock:
            return self.env.tactile()  # the sim's own contact rendering

    def bar_lift(self):
        with self.lock:
            return self.env.cube_pos()[2] - self.z0

    def close(self):
        self.running = False
        self.thread.join(timeout=1.0)
        self.env.close()

