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

print("Launching viewer...", flush=True)

viewer = mujoco.viewer.launch_passive(model, data)

print("Viewer launched", flush=True)

for i in range(500):
    mujoco.mj_step(model, data)

    if viewer.is_running():
        viewer.sync()
    else:
        print("Viewer closed unexpectedly", flush=True)
        break

    time.sleep(0.01)

print("About to close viewer", flush=True)
viewer.close()
print("Viewer closed successfully", flush=True)