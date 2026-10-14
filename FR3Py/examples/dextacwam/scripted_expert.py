"""Scripted cage-grasp expert for the bar-on-posts scene.

The hand keeps its home orientation (fingers down, palm facing +x). It descends
just behind the bar, slides forward so the bar sits in front of the palm below
the knuckles, curls the fingers under and around it (WRAP_POSE), and lifts.
Moves are Cartesian (IK every control tick) with smoothstep timing.
"""

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from cube_env import OPEN_POSE, WRAP_POSE

# Bar position in the palm frame at grasp time: in front of the palm face, below
# the finger knuckles, inside the loop the curled fingers enclose.
BAR_IN_PALM = np.array([0.036, 0.0, 0.082])
BACKOFF = 0.028        # extra clearance behind the bar while descending
LIFT = 0.15


def _smooth(a):
    return a * a * (3 - 2 * a)


class ScriptedExpert:
    def __init__(self, env, speed=1.0, rng=None):
        self.env = env
        rng = np.random.default_rng() if rng is None else rng
        jitter = lambda: rng.uniform(0.85, 1.15) / speed  # noqa: E731
        R_home = env.palm_T()[:3, :3].copy()
        # Per-episode wrist yaw/tilt, blended in during the approach.
        tilt = Rotation.from_euler("zyx", [rng.uniform(-0.15, 0.15),
                                           rng.uniform(-0.06, 0.06),
                                           rng.uniform(-0.06, 0.06)])
        self.R_home = R_home
        self.R = (tilt * Rotation.from_matrix(R_home)).as_matrix()
        bar = env.cube_pos()
        grasp = bar - self.R @ BAR_IN_PALM
        back = grasp + np.array([-BACKOFF, 0.0, 0.0])
        # (target palm position, hand pose, ticks); ticks at ~30 Hz
        self.plan = [
            (env.palm_T()[:3, 3], OPEN_POSE, int(8 * jitter())),
            (back + [0, 0, 0.14], OPEN_POSE, int(40 * jitter())),
            (back, OPEN_POSE, int(30 * jitter())),
            (grasp, OPEN_POSE, int(18 * jitter())),
            (grasp, WRAP_POSE, int(28 * jitter())),
            (grasp + [0, 0, LIFT], WRAP_POSE, int(40 * jitter())),
            (grasp + [0, 0, LIFT], WRAP_POSE, 20),
        ]
        self._slerp = Slerp([0, 1], Rotation.from_matrix([R_home, self.R]))
        self.seg, self.i = 0, 0
        self.p0 = env.palm_T()[:3, 3].copy()
        self.h0 = OPEN_POSE.copy()
        self.q = env.d.qpos[:7].copy()

    @property
    def done(self):
        return self.seg >= len(self.plan)

    def act(self):
        """Return (arm joint target (7,), hand target (22,), palm target 4x4)."""
        p1, h1, n = self.plan[self.seg]
        a = _smooth((self.i + 1) / n)
        p = self.p0 + (p1 - self.p0) * a
        h = self.h0 + (h1 - self.h0) * a
        T = np.eye(4)
        # orientation reaches the episode's wrist pose by the end of segment 1
        T[:3, :3] = self._slerp(min(1.0, (self.seg + a) / 2.0)).as_matrix() if self.seg < 2 else self.R
        T[:3, 3] = p
        self.q = self.env.ik(T, self.q, iters=60)
        self.i += 1
        if self.i >= n:
            self.seg, self.i = self.seg + 1, 0
            self.p0, self.h0 = np.array(p1, dtype=float), np.array(h1, dtype=float)
        return self.q.copy(), h.copy(), T
