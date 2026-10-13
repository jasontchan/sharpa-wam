import argparse
import sys
import time

import mujoco
import numpy as np

import FR3Py.sim.mujoco_sharpa as mujoco_sharpa
from FR3Py.sim.mujoco_sharpa import FR3SharpaSim
from FR3Py.sim.sharpa_constants import DEFAULT_GRASP_POSE, JOINT_ORDER
from FR3Py.sim.sharpa_spec import (
    FLANGE_BODY,
    HAND_ROOT_BODY,
    MOUNT_BODY,
    NUM_ARM_ACTUATORS,
    build_fr3_sharpa_spec,
)

KP = 200.0
KD = 20.0
GRAVITY = 9.81

STATIC_SECONDS = 2.0
SEGMENT_SECONDS = 4.0

MASS_EPS = 1e-9
TORQUE_TOL = 1e-2
HAND_DEVIATION_TOL = 0.05
SIGN_CHECK_JOINTS = (2, 4)

WAYPOINTS = np.array(
    [
        [1.20, -0.30, 0.40, -1.20, 0.50, 1.90, 0.30],
        [-1.20, 0.50, -0.60, -1.00, -0.60, 2.20, -0.50],
        [0.00, 0.90, 0.00, -0.60, 0.00, 2.60, 0.80],
        [1.80, -0.60, 1.50, -2.30, 1.20, 1.20, -1.20],
    ]
)


def subtree_body_ids(model, root_name):
    root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, root_name)
    assert root >= 0, f"Body {root_name} is missing from the model"
    ids = {root}
    for i in range(model.nbody):
        if i != root and model.body_parentid[i] in ids:
            ids.add(i)
    return ids


def body_name(model, body_id):
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(body_id))


def build_zeroed_hand_spec():
    spec = build_fr3_sharpa_spec()
    model = spec.compile()
    hand_names = [
        body_name(model, i) for i in sorted(subtree_body_ids(model, HAND_ROOT_BODY))
    ]
    for name in hand_names:
        body = spec.body(name)
        if body.explicitinertial:
            body.mass = MASS_EPS
            body.inertia = [MASS_EPS, MASS_EPS, MASS_EPS]
        else:
            for index, geom in enumerate(body.geoms):
                geom.mass = MASS_EPS if index == 0 else 0.0
                geom.density = 0.0
    return spec


def make_sim(render, zero_hand_inertia=False):
    if zero_hand_inertia:
        original = mujoco_sharpa.build_fr3_sharpa_spec
        mujoco_sharpa.build_fr3_sharpa_spec = build_zeroed_hand_spec
        try:
            sim = FR3SharpaSim(interface_type="torque", render=render)
        finally:
            mujoco_sharpa.build_fr3_sharpa_spec = original
    else:
        sim = FR3SharpaSim(interface_type="torque", render=render)
    sim.model.opt.disableflags &= ~int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
    return sim


def grasp_reference(sim):
    pose = np.array([DEFAULT_GRASP_POSE[name] for name in JOINT_ORDER], dtype=np.float64)
    return np.clip(pose, sim.hand_ctrlrange[:, 0], sim.hand_ctrlrange[:, 1])


def drive(sim, q_des, grasp):
    state = sim.getJointStates()["arm"]
    tau_ff = KP * (q_des - state["q"]) - KD * state["dq"]
    sim.setCommands(tau_ff, grasp)
    if sim.render:
        time.sleep(sim.dt)


def hold(sim, q_des, grasp, seconds):
    for _ in range(int(seconds / sim.dt)):
        drive(sim, q_des, grasp)
    return sim.actuator_tau.copy()


def distal_mass_report(sim):
    flange = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, FLANGE_BODY)
    ids = sorted(subtree_body_ids(sim.model, FLANGE_BODY) - {flange})
    rows = [(body_name(sim.model, i), float(sim.model.body_mass[i])) for i in ids]
    return rows, sum(mass for _, mass in rows)


def subtree_com_height_jacobian(sim, root_name):
    root = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, root_name)
    jacp = np.zeros((3, sim.model.nv))
    mujoco.mj_jacSubtreeCom(sim.model, sim.data, jacp, root)
    return jacp[2, :NUM_ARM_ACTUATORS].copy()


def check_static_load(render):
    print("Check 1 - static load")

    sim = make_sim(render)
    grasp = grasp_reference(sim)
    tau_loaded = hold(sim, sim.q0, grasp, STATIC_SECONDS)

    rows, distal_total = distal_mass_report(sim)
    hand_ids = subtree_body_ids(sim.model, HAND_ROOT_BODY)
    hand_mass = float(sum(sim.model.body_mass[i] for i in sorted(hand_ids)))
    dz_dq = subtree_com_height_jacobian(sim, HAND_ROOT_BODY)
    predicted = hand_mass * GRAVITY * dz_dq
    sim.close()

    sim_zero = make_sim(render, zero_hand_inertia=True)
    zeroed_hand_mass = float(
        sum(
            sim_zero.model.body_mass[i]
            for i in sorted(subtree_body_ids(sim_zero.model, HAND_ROOT_BODY))
        )
    )
    tau_unloaded = hold(sim_zero, sim_zero.q0, grasp_reference(sim_zero), STATIC_SECONDS)
    sim_zero.close()

    delta = tau_loaded - tau_unloaded

    print("  bodies distal to link7:")
    for name, mass in rows:
        print(f"    {name:<24s} {mass:9.6f} kg")
    print(f"    {'total':<24s} {distal_total:9.6f} kg")
    print(f"  hand subtree mass       : {hand_mass:.6f} kg")
    print(f"  hand subtree mass zeroed: {zeroed_hand_mass:.3e} kg")
    print(f"  tau loaded    [Nm]: {np.array2string(tau_loaded, precision=4)}")
    print(f"  tau unloaded  [Nm]: {np.array2string(tau_unloaded, precision=4)}")
    print(f"  difference    [Nm]: {np.array2string(delta, precision=4)}")
    print(f"  m*g*dz/dq     [Nm]: {np.array2string(predicted, precision=4)}")

    failures = []
    for joint in SIGN_CHECK_JOINTS:
        i = joint - 1
        if abs(delta[i]) <= TORQUE_TOL:
            failures.append(
                f"joint{joint} torque difference {delta[i]:.4f} Nm is within "
                f"the {TORQUE_TOL} Nm null band"
            )
            continue
        if abs(predicted[i]) <= TORQUE_TOL:
            failures.append(
                f"joint{joint} has no distal gravity gradient "
                f"(m*g*dz/dq = {predicted[i]:.4f} Nm), sign is undefined"
            )
            continue
        if np.sign(delta[i]) != np.sign(predicted[i]):
            failures.append(
                f"joint{joint} torque difference {delta[i]:.4f} Nm opposes "
                f"m*g*dz/dq = {predicted[i]:.4f} Nm"
            )
    return failures


def run_sweep(render):
    sim = make_sim(render)
    grasp = grasp_reference(sim)
    hand_ids = subtree_body_ids(sim.model, HAND_ROOT_BODY)
    mount_ids = subtree_body_ids(sim.model, MOUNT_BODY)
    arm_ids = subtree_body_ids(sim.model, "link0") - mount_ids
    ctrlrange = sim.model.actuator_ctrlrange[:NUM_ARM_ACTUATORS].copy()

    max_tau = np.zeros(NUM_ARM_ACTUATORS)
    max_q_err = np.zeros(NUM_ARM_ACTUATORS)
    max_hand_dev = np.zeros(len(JOINT_ORDER))
    tau_violations = []
    hand_violations = []
    contact_pairs = {}
    step = 0

    path = np.vstack([sim.q0, WAYPOINTS])
    for start, end in zip(path[:-1], path[1:]):
        n = int(SEGMENT_SECONDS / sim.dt)
        for k in range(n):
            alpha = 0.5 - 0.5 * np.cos(np.pi * (k + 1) / n)
            q_des = start + alpha * (end - start)
            drive(sim, q_des, grasp)

            state = sim.getJointStates()
            tau = sim.actuator_tau.copy()
            q_err = state["arm"]["q"] - q_des
            hand_dev = state["hand"]["q"] - grasp

            max_tau = np.maximum(max_tau, np.abs(tau))
            max_q_err = np.maximum(max_q_err, np.abs(q_err))
            max_hand_dev = np.maximum(max_hand_dev, np.abs(hand_dev))

            below = tau < ctrlrange[:, 0]
            above = tau > ctrlrange[:, 1]
            for i in np.flatnonzero(below | above):
                tau_violations.append((step, int(i), float(tau[i])))
            for i in np.flatnonzero(np.abs(hand_dev) > HAND_DEVIATION_TOL):
                hand_violations.append((step, JOINT_ORDER[i], float(hand_dev[i])))

            for c in range(sim.data.ncon):
                contact = sim.data.contact[c]
                b1 = int(sim.model.geom_bodyid[contact.geom1])
                b2 = int(sim.model.geom_bodyid[contact.geom2])
                if b1 in hand_ids and b2 in hand_ids:
                    continue
                key = (min(b1, b2), max(b1, b2))
                if key not in contact_pairs:
                    contact_pairs[key] = step
            step += 1

    result = {
        "steps": step,
        "ctrlrange": ctrlrange,
        "max_tau": max_tau,
        "max_q_err": max_q_err,
        "max_hand_dev": max_hand_dev,
        "tau_violations": tau_violations,
        "hand_violations": hand_violations,
        "contact_pairs": [
            (body_name(sim.model, b1), body_name(sim.model, b2), s)
            for (b1, b2), s in sorted(contact_pairs.items(), key=lambda kv: kv[1])
        ],
        "arm_hand_pairs": [
            (body_name(sim.model, b1), body_name(sim.model, b2), s)
            for (b1, b2), s in sorted(contact_pairs.items(), key=lambda kv: kv[1])
            if (b1 in arm_ids and b2 in hand_ids) or (b2 in arm_ids and b1 in hand_ids)
        ],
    }
    sim.close()
    return result


def check_swept_path(result):
    print("Check 2 - swept path")
    print(f"  steps                 : {result['steps']}")
    print(f"  max |tau|        [Nm] : {np.array2string(result['max_tau'], precision=4)}")
    print(
        f"  ctrlrange upper  [Nm] : "
        f"{np.array2string(result['ctrlrange'][:, 1], precision=4)}"
    )
    print(
        f"  max |q err|     [rad] : "
        f"{np.array2string(result['max_q_err'], precision=4)}"
    )
    print(
        f"  max hand deviation    : {result['max_hand_dev'].max():.5f} rad "
        f"({JOINT_ORDER[int(np.argmax(result['max_hand_dev']))]})"
    )

    failures = []
    for step, i, tau in result["tau_violations"][:20]:
        failures.append(
            f"step {step}: joint{i + 1} torque {tau:.4f} Nm outside ctrlrange "
            f"[{result['ctrlrange'][i, 0]:.1f}, {result['ctrlrange'][i, 1]:.1f}]"
        )
    for step, name, dev in result["hand_violations"][:20]:
        failures.append(
            f"step {step}: hand joint {name} deviates {dev:.5f} rad from "
            f"DEFAULT_GRASP_POSE"
        )
    dropped = (
        len(result["tau_violations"])
        + len(result["hand_violations"])
        - len(failures)
    )
    if dropped > 0:
        failures.append(f"... and {dropped} further violations")
    return failures


def check_self_collision(result):
    print("Check 3 - self-collision")
    pairs = result["contact_pairs"]
    print(f"  non-hand-hand contact pairs: {len(pairs)}")
    for name1, name2, step in pairs:
        print(f"    {name1} <-> {name2} (first at step {step})")

    return [
        f"arm/hand contact {name1} <-> {name2} at step {step}"
        for name1, name2, step in result["arm_hand_pairs"]
    ]


def report(label, failures):
    if failures:
        print(f"FAIL {label}")
        for line in failures:
            print(f"  - {line}")
    else:
        print(f"PASS {label}")
    return bool(failures)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    static_failures = check_static_load(args.render)
    sweep = run_sweep(args.render)
    swept_failures = check_swept_path(sweep)
    collision_failures = check_self_collision(sweep)

    print()
    failed = report("check 1 static load", static_failures)
    failed |= report("check 2 swept path", swept_failures)
    failed |= report("check 3 self-collision", collision_failures)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
