import time
import mujoco
import mujoco.viewer

from FR3Py.sim.mujoco_sharpa import FR3SharpaSim

sim = FR3SharpaSim(interface_type="torque", render=False)

model = sim.model
data = sim.data

with mujoco.viewer.launch_passive(model, data) as viewer:

    with viewer.lock():
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        viewer.cam.lookat[:] = [0, 0, 0.5]
        viewer.cam.distance = 2.0
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -25

    for _ in range(500):
        if not viewer.is_running():
            break

        viewer.sync()
        time.sleep(0.01)

sim.close()