import time
import mujoco
import mujoco.viewer

xml = """
<mujoco>
    <worldbody>
        <light pos="0 0 3"/>
        <geom type="plane" size="2 2 0.1"/>
        <body pos="0 0 0.5">
            <geom type="box" size="0.2 0.2 0.2" rgba="1 0 0 1"/>
        </body>
    </worldbody>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)

with mujoco.viewer.launch_passive(model, data) as viewer:
    for _ in range(500):
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(model.opt.timestep)