import argparse
import csv
import os
import sys
import time

import mujoco
import mujoco.viewer
import numpy as np

from FR3Py.sim.sharpa_constants import JOINT_NAMES
from FR3Py.sim.sharpa_spec import NUM_ARM_ACTUATORS, build_fr3_sharpa_spec

Q0 = np.array(
    [0.0, -0.785398163, 0.0, -2.35619449, 0.0, 1.57079632679, 0.785398163397]
)

DT = 0.002
SETTLE_TIME = 1.0
RAMP_TIME = 0.5
HOLD_TIME = 0.5
ARM_KP = 400.0
ARM_KD = 40.0
CROSSTALK_TOL = 1e-3
MIN_TRAVEL = 1e-3
RENDER_PERIOD = 1 / 60

OUTPUT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_joint_order.csv")


def hand_actuator_table(model):
    table = []
    for name in JOINT_NAMES:
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        assert aid >= 0, f"Hand actuator {name} is missing from the model"
        assert model.actuator_trntype[aid] == mujoco.mjtTrn.mjTRN_JOINT, (
            f"Hand actuator {name} is not a joint transmission"
        )
        jid = int(model.actuator_trnid[aid, 0])
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        table.append(
            {
                "ctrl_index": aid,
                "actuator_name": name,
                "joint_id": jid,
                "joint_name": joint_name,
                "qpos_index": int(model.jnt_qposadr[jid]),
                "qvel_index": int(model.jnt_dofadr[jid]),
                "ctrlrange_low": float(model.actuator_ctrlrange[aid, 0]),
                "ctrlrange_high": float(model.actuator_ctrlrange[aid, 1]),
            }
        )
    table.sort(key=lambda entry: entry["ctrl_index"])
    return table


def reset(model, data, table):
    mujoco.mj_resetData(model, data)
    data.qpos[:NUM_ARM_ACTUATORS] = Q0
    data.qvel[:] = 0.0
    for entry in table:
        data.qpos[entry["qpos_index"]] = 0.0
    data.ctrl[:] = 0.0
    mujoco.mj_forward(model, data)


def arm_hold(model, data):
    q = data.qpos[:NUM_ARM_ACTUATORS]
    dq = data.qvel[:NUM_ARM_ACTUATORS]
    gravity = data.qfrc_bias[:NUM_ARM_ACTUATORS]
    return gravity + ARM_KP * (Q0 - q) - ARM_KD * dq


def sweep_target(entry):
    low = entry["ctrlrange_low"]
    high = entry["ctrlrange_high"]
    target = 0.5 * (low + high)
    if abs(target) < MIN_TRAVEL:
        target = high if abs(high) >= abs(low) else low
    return target


def hold_step(model, data, table, entry=None, command=0.0):
    data.ctrl[:NUM_ARM_ACTUATORS] = arm_hold(model, data)
    for other in table:
        data.ctrl[other["ctrl_index"]] = 0.0
    if entry is not None:
        data.ctrl[entry["ctrl_index"]] = command
    mujoco.mj_step(model, data)


def sweep_actuator(model, data, table, entry, viewer):
    reset(model, data, table)

    qpos_idx = np.array([e["qpos_index"] for e in table], dtype=int)
    render_every = max(1, int(round(RENDER_PERIOD / DT)))

    for _ in range(int(round(SETTLE_TIME / DT))):
        hold_step(model, data, table)
    reference = data.qpos[qpos_idx].copy()
    if viewer is not None:
        viewer.sync()

    displacement = np.zeros(len(table))
    target = sweep_target(entry)
    ramp_steps = int(round(RAMP_TIME / DT))
    hold_steps = int(round(HOLD_TIME / DT))

    for step in range(ramp_steps + hold_steps):
        if step < ramp_steps:
            command = target * (step + 1) / ramp_steps
        else:
            command = target

        hold_step(model, data, table, entry, command)

        displacement = np.maximum(
            displacement, np.abs(data.qpos[qpos_idx] - reference)
        )

        if viewer is not None:
            if step % render_every == 0:
                viewer.sync()
                time.sleep(RENDER_PERIOD)
            if not viewer.is_running():
                break

    return target, displacement


def main():
    parser = argparse.ArgumentParser(
        description="Sweep each Sharpa hand actuator and record which joint it moves."
    )
    parser.add_argument("--render", action="store_true", default=False)
    parser.add_argument("--with-contacts", action="store_true", default=False)
    parser.add_argument("--output", default=OUTPUT_CSV)
    args = parser.parse_args()

    spec = build_fr3_sharpa_spec()
    model = spec.compile()
    model.opt.timestep = DT
    if not args.with_contacts:
        model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
    data = mujoco.MjData(model)

    table = hand_actuator_table(model)

    viewer = None
    if args.render:
        reset(model, data, table)
        viewer = mujoco.viewer.launch_passive(model, data)
        viewer.cam.distance = 1.0
        viewer.cam.azimuth = 90
        viewer.cam.elevation = -20
        viewer.cam.lookat[:] = np.array([0.4, 0.0, 0.6])

    rows = []
    failures = 0

    try:
        for index, entry in enumerate(table):
            target, displacement = sweep_actuator(model, data, table, entry, viewer)

            moved = int(np.argmax(displacement))
            measured = float(displacement[index])
            others = displacement.copy()
            others[index] = -1.0
            worst_other = int(np.argmax(others))
            crosstalk_max = float(others[worst_other])
            crosstalk_joint = table[worst_other]["joint_name"]

            problems = []
            if moved != index:
                problems.append(
                    f"argmax joint is {table[moved]['joint_name']}, "
                    f"expected {entry['joint_name']}"
                )
            if crosstalk_max > CROSSTALK_TOL:
                problems.append(
                    f"crosstalk {crosstalk_max:.6f} rad on {crosstalk_joint} "
                    f"exceeds {CROSSTALK_TOL}"
                )

            status = "PASS" if not problems else "FAIL"
            if problems:
                failures += 1
            print(
                f"[{status}] ctrl[{entry['ctrl_index']:2d}] {entry['actuator_name']:22s}"
                f" -> {entry['joint_name']:22s} qpos[{entry['qpos_index']:2d}]"
                f" target={target:+.4f} moved={measured:+.6f}"
                f" crosstalk={crosstalk_max:.2e}"
                + ("  " + "; ".join(problems) if problems else "")
            )

            rows.append(
                {
                    "ctrl_index": entry["ctrl_index"],
                    "actuator_name": entry["actuator_name"],
                    "joint_name": entry["joint_name"],
                    "qpos_index": entry["qpos_index"],
                    "ctrlrange_low": f"{entry['ctrlrange_low']:.6f}",
                    "ctrlrange_high": f"{entry['ctrlrange_high']:.6f}",
                    "commanded_target": f"{target:.6f}",
                    "measured_displacement": f"{measured:.6f}",
                }
            )
    finally:
        if viewer is not None:
            viewer.close()

    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "ctrl_index",
                "actuator_name",
                "joint_name",
                "qpos_index",
                "ctrlrange_low",
                "ctrlrange_high",
                "commanded_target",
                "measured_displacement",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print()
    if args.with_contacts:
        print(
            "Contacts are enabled: adjacent fingers touch after ~0.025 rad of "
            "abduction, so the *_MCP_AA sweeps register neighbour motion."
        )
    print(f"Wrote {args.output}")
    print(f"{len(rows) - failures}/{len(rows)} actuators passed, {failures} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
