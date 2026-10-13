import argparse
import importlib
import os
import sys
import time

import mujoco
import mujoco.viewer
import numpy as np

from FR3Py.sim.sharpa_constants import JOINT_NAMES
from FR3Py.sim.sharpa_spec import (
    FLANGE_BODY,
    HAND_ROOT_BODY,
    MOUNT_BODY,
    MOUNT_FRAME_PY,
    NUM_ARM_ACTUATORS,
    build_fr3_sharpa_spec,
    load_mount_frame,
)

Q0 = np.array(
    [0.0, -0.785398163, 0.0, -2.35619449, 0.0, 1.57079632679, 0.785398163397]
)

MODULE_NAME = "mount_frame"
POLL_PERIOD = 0.2
DEBOUNCE = 0.1
MATCH_TOL = 1e-9


def body_pose(model, data, name):
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    assert bid >= 0, f"Body {name} is missing from the merged model"
    return data.xpos[bid].copy(), data.xquat[bid].copy()


def normalized(quat):
    quat = np.asarray(quat, dtype=np.float64)
    return quat / np.linalg.norm(quat)


def import_mount_frame():
    directory = os.path.dirname(MOUNT_FRAME_PY)
    if directory not in sys.path:
        sys.path.append(directory)
    module = importlib.import_module(MODULE_NAME)
    resolved = os.path.realpath(module.__file__)
    if resolved != os.path.realpath(MOUNT_FRAME_PY):
        raise ImportError(f"{MODULE_NAME} resolved to {resolved}, not {MOUNT_FRAME_PY}")
    return module


def print_transforms(flange_to_mount, mount_to_palm):
    print("Transforms from FR3Py/assets/mujoco/sharpa/mount_frame.py")
    print(f"  FLANGE_TO_MOUNT pos  = {flange_to_mount[0]}")
    print(f"  FLANGE_TO_MOUNT quat = {flange_to_mount[1]}")
    print(f"  MOUNT_TO_PALM   pos  = {mount_to_palm[0]}")
    print(f"  MOUNT_TO_PALM   quat = {mount_to_palm[1]}")


def can_live_patch(model, mount_bid, palm_bid, flange_to_mount, mount_to_palm):
    for bid, transform in ((mount_bid, flange_to_mount), (palm_bid, mount_to_palm)):
        if not np.allclose(model.body_pos[bid], transform[0], rtol=0.0, atol=MATCH_TOL):
            return False
        if not np.allclose(
            model.body_quat[bid], normalized(transform[1]), rtol=0.0, atol=MATCH_TOL
        ):
            return False
    return True


def apply_transforms(model, mount_bid, palm_bid, flange_to_mount, mount_to_palm):
    model.body_pos[mount_bid] = flange_to_mount[0]
    model.body_quat[mount_bid] = normalized(flange_to_mount[1])
    model.body_pos[palm_bid] = mount_to_palm[0]
    model.body_quat[palm_bid] = normalized(mount_to_palm[1])


def run_session(watch):
    flange_to_mount, mount_to_palm = load_mount_frame()

    model = build_fr3_sharpa_spec().compile()
    data = mujoco.MjData(model)

    data.qpos[:NUM_ARM_ACTUATORS] = Q0
    for name in JOINT_NAMES:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        data.qpos[model.jnt_qposadr[jid]] = 0.0

    mount_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, MOUNT_BODY)
    palm_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, HAND_ROOT_BODY)
    assert mount_bid >= 0, f"Body {MOUNT_BODY} is missing from the merged model"
    assert palm_bid >= 0, f"Body {HAND_ROOT_BODY} is missing from the merged model"

    mujoco.mj_forward(model, data)

    print_transforms(flange_to_mount, mount_to_palm)
    print()
    print(f"World poses at arm q0 = {np.array2string(Q0, precision=4)}, hand at zero")
    for name in (FLANGE_BODY, MOUNT_BODY, HAND_ROOT_BODY):
        pos, quat = body_pose(model, data, name)
        print(f"  {name:22s} pos = {np.array2string(pos, precision=5)}")
        print(f"  {'':22s} quat = {np.array2string(quat, precision=5)}")
    print()
    print("Edit mount_frame.py and re-run until the mount and palm look correct.")

    live_patch = can_live_patch(
        model, mount_bid, palm_bid, flange_to_mount, mount_to_palm
    )
    if watch and not live_patch:
        print(
            "Compiled body_pos/body_quat do not match mount_frame.py within "
            f"{MATCH_TOL:g}; falling back to a full rebuild on every change."
        )
    if watch:
        print(f"Watching {MOUNT_FRAME_PY} for changes. Ctrl+C or close the viewer to stop.")
    print()

    module = None
    if watch and live_patch:
        try:
            module = import_mount_frame()
        except Exception as e:
            print(f"[reload] {type(e).__name__}: {e}")
            live_patch = False
            print("Falling back to a full rebuild on every change.")
    last_mtime = os.path.getmtime(MOUNT_FRAME_PY) if watch else None
    next_poll = time.time() + POLL_PERIOD

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance = 1.0
        viewer.cam.azimuth = 90
        viewer.cam.elevation = -20
        viewer.cam.lookat[:] = body_pose(model, data, MOUNT_BODY)[0]

        while viewer.is_running():
            if watch and time.time() >= next_poll:
                next_poll = time.time() + POLL_PERIOD
                try:
                    mtime = os.path.getmtime(MOUNT_FRAME_PY)
                except OSError as e:
                    print(f"[reload] {type(e).__name__}: {e}")
                    mtime = last_mtime

                if mtime != last_mtime:
                    last_mtime = mtime
                    time.sleep(DEBOUNCE)

                    if not live_patch:
                        return True

                    try:
                        importlib.reload(module)
                        flange_to_mount = module.FLANGE_TO_MOUNT
                        mount_to_palm = module.MOUNT_TO_PALM
                        apply_transforms(
                            model,
                            mount_bid,
                            palm_bid,
                            flange_to_mount,
                            mount_to_palm,
                        )
                        mujoco.mj_forward(model, data)
                        last_mtime = os.path.getmtime(MOUNT_FRAME_PY)

                        print_transforms(flange_to_mount, mount_to_palm)
                        for name in (FLANGE_BODY, MOUNT_BODY, HAND_ROOT_BODY):
                            z = body_pose(model, data, name)[0][2]
                            print(f"  {name:22s} world z = {z: .5f}")
                        print()
                    except Exception as e:
                        print(f"[reload] {type(e).__name__}: {e}")

            mujoco.mj_forward(model, data)
            viewer.sync()
            time.sleep(1 / 60)

    return False


def main():
    parser = argparse.ArgumentParser(
        description="View the merged FR3 + Sharpa Wave model and calibrate the mount."
    )
    parser.add_argument("--watch", dest="watch", action="store_true", default=True)
    parser.add_argument("--no-watch", dest="watch", action="store_false")
    args = parser.parse_args()

    while run_session(args.watch):
        pass


if __name__ == "__main__":
    main()
