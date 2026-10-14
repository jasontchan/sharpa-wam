"""FR3 + Sharpa Wave tabletop scene: a red bar resting across two posts for the DexTacWAM sim proof of concept.

Adds a table, a free red cube and a fixed "head" camera to FR3Py's FR3 + Sharpa
MJCF, and provides joint-space PD control (gravity compensated, torque arm),
damped-least-squares IK on the palm frame, and synthetic fingertip tactile
images rendered from MuJoCo contact forces.
"""

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from FR3Py.sim.sharpa_constants import (  # noqa: E402
    DEFAULT_GRASP_POSE,
    FINGERTIP_GEOMS,
    JOINT_NAMES,
    JOINT_ORDER,
)
from FR3Py.sim.sharpa_spec import HAND_ROOT_BODY, build_fr3_sharpa_spec  # noqa: E402

Q_HOME = np.array([0.0, -0.785398163, 0.0, -2.35619449, 0.0, 1.57079632679, 0.785398163397])
GRASP_POSE = np.array([DEFAULT_GRASP_POSE[n] for n in JOINT_ORDER])
OPEN_POSE = np.zeros(22)
# Power-grasp "wrap": every finger flexed at MCP/PIP/DIP, thumb opposed.
_WRAP = {"thumb_CMC_FE": 0.0, "thumb_CMC_AA": 0.0, "thumb_MCP_FE": 0.0, "thumb_MCP_AA": 0.0,
         "thumb_IP": 0.0, "pinky_CMC": 0.0}
for _f in ("index", "middle", "ring", "pinky"):
    _WRAP.update({f"{_f}_MCP_FE": 1.45, f"{_f}_MCP_AA": 0.0, f"{_f}_PIP": 1.5, f"{_f}_DIP": 1.0})
WRAP_POSE = np.array([_WRAP[n] for n in JOINT_ORDER])

TABLE_TOP = 0.20
BAR_RADIUS = 0.012
BAR_HALF_LEN = 0.15
POST_HALF = 0.015
POST_H = 0.13          # bar centre height above the table top
POST_Y = 0.125         # posts at bar_y +/- POST_Y
SIM_DT = 0.002
CONTROL_HZ = 30
SUBSTEPS = int(round(1.0 / (CONTROL_HZ * SIM_DT)))  # 17 -> ~29.4 Hz, close to 30

CAM_H, CAM_W = 192, 256  # DexTacWAM's sample_size, so nothing is resized
TAC_H, TAC_W = 192, 256
TACTILE_BASELINE = 6     # "no contact" level; an all-zero finger is a dropout


def _look_at(pos, target):
    """MJCF camera xyaxes for a camera at `pos` looking at `target` (z up)."""
    fwd = np.asarray(target, float) - np.asarray(pos, float)
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    return np.concatenate([right, up]).tolist()


def build_model():
    spec = build_fr3_sharpa_spec()
    wb = spec.worldbody

    table = wb.add_body(name="table", pos=[0.55, 0.0, TABLE_TOP / 2])
    table.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.30, 0.40, TABLE_TOP / 2],
                   rgba=[0.45, 0.33, 0.24, 1.0], friction=[1.0, 0.005, 0.0001])

    for side in ("L", "R"):
        post = wb.add_body(name=f"post_{side}", pos=[0.5, 0.0, TABLE_TOP + (POST_H - BAR_RADIUS) / 2])
        post.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX,
                      size=[POST_HALF, POST_HALF, (POST_H - BAR_RADIUS) / 2],
                      rgba=[0.25, 0.25, 0.28, 1.0])
    # The bar lies along world y; "cube" names kept for the observation code.
    cube = wb.add_body(name="cube", pos=[0.5, 0.0, TABLE_TOP + POST_H])
    cube.add_freejoint(name="cube_free")
    cube.add_geom(name="cube_geom", type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                  size=[BAR_RADIUS, BAR_HALF_LEN, 0], quat=[0.7071068, 0.7071068, 0, 0],
                  rgba=[0.85, 0.12, 0.12, 1.0], mass=0.05,
                  friction=[1.5, 0.01, 0.001], condim=4)

    # Fixed third-person "head" camera, looking at the workspace from the front-right.
    wb.add_camera(name="head_cam", pos=[1.15, -0.45, 0.80],
                  xyaxes=_look_at([1.15, -0.45, 0.80], [0.45, 0.0, 0.30]), fovy=50)
    # Presentation camera for the video (not a policy input).
    wb.add_camera(name="video_cam", pos=[1.05, 0.70, 0.70],
                  xyaxes=_look_at([1.05, 0.70, 0.70], [0.42, 0.0, 0.32]), fovy=45)
    light = wb.add_light(pos=[0.6, 0.0, 1.8], dir=[0, 0, -1])
    light.diffuse = [0.35, 0.35, 0.35]
    model = spec.compile()
    model.opt.timestep = SIM_DT
    model.vis.global_.offwidth, model.vis.global_.offheight = 1280, 720
    return model


class CubeEnv:
    def __init__(self, seed=0, render=True):
        self.m = build_model()
        self.d = mujoco.MjData(self.m)
        self.rng = np.random.default_rng(seed)
        self.hand_qpos = np.array([self.m.jnt_qposadr[self._jid(n)] for n in JOINT_NAMES])
        self.hand_dof = np.array([self.m.jnt_dofadr[self._jid(n)] for n in JOINT_NAMES])
        self.hand_act = np.array([mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
                                  for n in JOINT_NAMES])
        self.hand_range = self.m.actuator_ctrlrange[self.hand_act]
        self.palm = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, HAND_ROOT_BODY)
        self.cube = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "cube")
        self.cube_geom = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM, "cube_geom")
        self.tip_geoms = [mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM, g) for g in FINGERTIP_GEOMS]
        # finger -> {body id: row in that finger's tactile image (0 = tip end)}
        self.finger_links = []
        for f in ("thumb", "index", "middle", "ring", "pinky"):
            links = {}
            for b in range(self.m.nbody):
                n = mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_BODY, b) or ""
                if n.startswith(f"right_{f}_"):
                    seg = n.split("_")[-1]
                    if seg in ("DP", "MP", "PP"):
                        links[b] = {"DP": 0.22, "MP": 0.5, "PP": 0.78}[seg]
            self.finger_links.append(links)
        self.cube_qadr = self.m.jnt_qposadr[self._jid("cube_free")]
        self.arm_lo, self.arm_hi = self.m.jnt_range[:7, 0], self.m.jnt_range[:7, 1]
        self.renderer = mujoco.Renderer(self.m, CAM_H, CAM_W) if render else None
        self.video_renderer = mujoco.Renderer(self.m, 720, 960) if render else None
        self.q_des = Q_HOME.copy()
        self.hand_des = OPEN_POSE.copy()

    def _jid(self, name):
        return mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, name)

    # ---- episode ------------------------------------------------------
    def reset(self, cube_xy=None):
        mujoco.mj_resetData(self.m, self.d)
        if cube_xy is None:
            cube_xy = np.array([0.50, 0.0]) + self.rng.uniform([-0.05, -0.06], [0.05, 0.06])
        for side, sgn in (("L", 1), ("R", -1)):
            b = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, f"post_{side}")
            self.m.body_pos[b, :2] = [cube_xy[0], cube_xy[1] + sgn * POST_Y]
        self.d.qpos[:7] = Q_HOME
        self.d.qpos[self.hand_qpos] = OPEN_POSE
        a = self.cube_qadr
        self.d.qpos[a:a + 3] = [cube_xy[0], cube_xy[1], TABLE_TOP + POST_H]
        self.d.qpos[a + 3:a + 7] = [1, 0, 0, 0]
        self.q_des = Q_HOME.copy()
        self.hand_des = OPEN_POSE.copy()
        mujoco.mj_forward(self.m, self.d)
        for _ in range(100):
            self._substep()
        return np.array(cube_xy)

    def _substep(self):
        d = self.d
        q, dq = d.qpos[:7], d.qvel[:7]
        tau = 600.0 * (self.q_des - q) - 50.0 * dq + d.qfrc_bias[:7]
        d.ctrl[:7] = tau
        d.ctrl[self.hand_act] = np.clip(self.hand_des, self.hand_range[:, 0], self.hand_range[:, 1])
        mujoco.mj_step(self.m, d)

    def step(self, q_des, hand_des):
        """One ~30 Hz control tick: hold joint targets for SUBSTEPS physics steps."""
        self.q_des = np.clip(q_des, self.arm_lo, self.arm_hi)
        self.hand_des = np.asarray(hand_des, dtype=np.float64)
        for _ in range(SUBSTEPS):
            self._substep()

    # ---- kinematics ---------------------------------------------------
    def palm_T(self, d=None):
        d = self.d if d is None else d
        T = np.eye(4)
        T[:3, :3] = d.xmat[self.palm].reshape(3, 3)
        T[:3, 3] = d.xpos[self.palm]
        return T

    def ik(self, T_target, q_init, iters=200, tol=1e-4):
        """DLS IK for the palm frame (arm joints only), on a scratch MjData."""
        d = mujoco.MjData(self.m)
        d.qpos[:] = self.d.qpos
        q = np.array(q_init, dtype=np.float64)
        jacp, jacr = np.zeros((3, self.m.nv)), np.zeros((3, self.m.nv))
        for _ in range(iters):
            d.qpos[:7] = q
            mujoco.mj_kinematics(self.m, d)
            mujoco.mj_comPos(self.m, d)
            T = self.palm_T(d)
            ep = T_target[:3, 3] - T[:3, 3]
            Re = T_target[:3, :3] @ T[:3, :3].T
            er = 0.5 * np.array([Re[2, 1] - Re[1, 2], Re[0, 2] - Re[2, 0], Re[1, 0] - Re[0, 1]])
            err = np.concatenate([ep, er])
            if np.linalg.norm(err) < tol:
                break
            mujoco.mj_jacBody(self.m, d, jacp, jacr, self.palm)
            J = np.vstack([jacp[:, :7], jacr[:, :7]])
            dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), err)
            # small pull toward home keeps the elbow/wrist configuration stable
            dq += 0.02 * (np.eye(7) - np.linalg.pinv(J) @ J) @ (Q_HOME - q)
            q = np.clip(q + dq, self.arm_lo, self.arm_hi)
        return q

    # ---- observations -------------------------------------------------
    def cube_pos(self):
        return self.d.xpos[self.cube].copy()

    def state45(self):
        """DexTacWAM right_only state: [R_arm7, R_hand22, R_pose16 (row-major)]."""
        return np.concatenate([self.d.qpos[:7], self.d.qpos[self.hand_qpos],
                               self.palm_T().reshape(16)]).astype(np.float32)

    def fingertip_forces(self):
        """(5, 6) contact wrench per fingertip (force xyz, torque xyz), world frame."""
        out = np.zeros((5, 6))
        f6 = np.zeros(6)
        for i in range(self.d.ncon):
            c = self.d.contact[i]
            for k, g in enumerate(self.tip_geoms):
                if g in (c.geom1, c.geom2):
                    mujoco.mj_contactForce(self.m, self.d, i, f6)
                    frame = c.frame.reshape(3, 3)
                    out[k, :3] += frame.T @ f6[:3]
                    out[k, 3:] += frame.T @ f6[3:]
        return out

    def tactile(self):
        """(1, 5, 192, 256) uint8 synthetic tactile, one image per finger.

        Every contact on a finger's segments (DP/MP/PP) adds a Gaussian blob at
        that segment's row (tip end at the top), sized and brightened by the
        normal force, on a small non-zero baseline. Self-contacts are ignored.
        """
        yy, xx = np.mgrid[0:TAC_H, 0:TAC_W]
        tac = np.full((1, 5, TAC_H, TAC_W), TACTILE_BASELINE, dtype=np.float32)
        f6 = np.zeros(6)
        hand_bodies = set().union(*[set(l) for l in self.finger_links]) | {self.palm}
        for i in range(self.d.ncon):
            c = self.d.contact[i]
            b1, b2 = self.m.geom_bodyid[c.geom1], self.m.geom_bodyid[c.geom2]
            if b1 in hand_bodies and b2 in hand_bodies:
                continue
            for k, links in enumerate(self.finger_links):
                b = b1 if b1 in links else b2 if b2 in links else None
                if b is None:
                    continue
                mujoco.mj_contactForce(self.m, self.d, i, f6)
                s = min(abs(f6[0]) / 3.0, 1.0)
                sigma = 14 + 30 * s
                cy = links[b] * TAC_H
                cx = TAC_W / 2 + 60 * np.tanh(5 * (c.pos[1] - self.d.xpos[b][1]))
                blob = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))
                tac[0, k] += (255 - TACTILE_BASELINE) * (0.4 + 0.6 * s) * blob
        return np.clip(tac, 0, 255).astype(np.uint8)

    def render(self, cam="head_cam", big=False):
        r = self.video_renderer if big else self.renderer
        r.update_scene(self.d, camera=cam)
        return r.render().copy()

    def close(self):
        for r in (self.renderer, self.video_renderer):
            if r is not None:
                r.close()
