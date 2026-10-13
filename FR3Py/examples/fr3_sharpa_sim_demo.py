import time

import numpy as np

from FR3Py.sim.mujoco_sharpa import FR3SharpaSim
from FR3Py.sim.sharpa_constants import DEFAULT_GRASP_POSE, JOINT_ORDER

RAMP_SECONDS = 2.0
HOLD_SECONDS = 3.0
KP = 200.0
KD = 20.0


def main():
    sim = FR3SharpaSim(interface_type="torque", render=True)
    import mujoco
    import time

    viewer = sim.viewer

    with viewer.lock():
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        viewer.cam.lookat[:] = [0.0, 0.0, 0.5]
        viewer.cam.distance = 2.0
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -25

    viewer.sync()

    time.sleep(2)

    grasp = np.array([DEFAULT_GRASP_POSE[name] for name in JOINT_ORDER])
    open_pose = np.clip(
        np.zeros(len(JOINT_ORDER)),
        sim.hand_ctrlrange[:, 0],
        sim.hand_ctrlrange[:, 1],
    )

    dt = sim.dt
    ramp_steps = int(RAMP_SECONDS / dt)
    hold_steps = int(HOLD_SECONDS / dt)
    print_every = int(1.0 / dt)

    try:
        for step in range(ramp_steps + hold_steps):
            state = sim.getJointStates()
            q, dq = state["arm"]["q"], state["arm"]["dq"]
            arm_cmd = KP * (sim.q0 - q) - KD * dq

            alpha = min(1.0, step / float(ramp_steps))
            hand_cmd = (1.0 - alpha) * open_pose + alpha * grasp

            sim.setCommands(arm_cmd, hand_cmd)

            if step % print_every == 0:
                t = step * dt
                hand = sim.getJointStates()["hand"]
                print(f"t = {t:4.1f} s   phase = {'ramp' if alpha < 1.0 else 'hold'}")
                print(f"  arm  q  : {np.array2string(q, precision=3)}")
                print(f"  arm  dq : {np.array2string(dq, precision=3)}")
                print(f"  hand q  : {np.array2string(hand['q'], precision=3)}")
                print(f"  hand dq : {np.array2string(hand['dq'], precision=3)}")

            time.sleep(dt)
    finally:
        sim.close()


if __name__ == "__main__":
    main()
