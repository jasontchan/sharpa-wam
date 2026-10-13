import argparse
import sys
import time

import mujoco
import numpy as np

from FR3Py.sim.mujoco_sharpa import FR3SharpaSim
from FR3Py.sim.sharpa_constants import DEFAULT_GRASP_POSE, JOINT_ORDER

KP = 200.0
KD = 20.0

STEPS = 10000
TIMESTEPS = (0.001, 0.002)


def make_sim(dt):
    sim = FR3SharpaSim(interface_type="torque", render=False, dt=dt)
    sim.model.opt.disableflags &= ~int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
    return sim


def grasp_reference(sim):
    pose = np.array([DEFAULT_GRASP_POSE[name] for name in JOINT_ORDER], dtype=np.float64)
    return np.clip(pose, sim.hand_ctrlrange[:, 0], sim.hand_ctrlrange[:, 1])


def measure(dt, steps, command_hand):
    sim = make_sim(dt)
    grasp = grasp_reference(sim)
    q_des = sim.q0.copy()
    hand_cmd = grasp if command_hand else None

    samples = np.zeros(steps)
    ncon = np.zeros(steps, dtype=int)

    loop_start = time.perf_counter()
    for i in range(steps):
        state = sim.getJointStates()["arm"]
        tau_ff = KP * (q_des - state["q"]) - KD * state["dq"]
        t0 = time.perf_counter()
        sim.setCommands(tau_ff, hand_cmd)
        samples[i] = time.perf_counter() - t0
        ncon[i] = sim.data.ncon
    loop_wall = time.perf_counter() - loop_start

    hand_dev = float(
        np.max(np.abs(sim.getJointStates()["hand"]["q"] - grasp))
    )
    arm_err = float(np.max(np.abs(sim.getJointStates()["arm"]["q"] - q_des)))
    sim.close()

    return {
        "dt": dt,
        "steps": steps,
        "command_hand": command_hand,
        "mean": float(samples.mean()),
        "median": float(np.median(samples)),
        "p99": float(np.percentile(samples, 99)),
        "max": float(samples.max()),
        "total_cmd": float(samples.sum()),
        "loop_wall": loop_wall,
        "sim_time": steps * dt,
        "mean_ncon": float(ncon.mean()),
        "max_ncon": int(ncon.max()),
        "hand_dev": hand_dev,
        "arm_err": arm_err,
    }


def print_table(results):
    header = (
        f"{'dt [s]':>7} {'hand cmd':>9} {'steps':>6} "
        f"{'mean [ms]':>10} {'median [ms]':>12} {'p99 [ms]':>9} {'max [ms]':>9} "
        f"{'rate [Hz]':>10} {'wall/sim':>9} {'cmd/sim':>8}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['dt']:>7.4f} "
            f"{('yes' if r['command_hand'] else 'no'):>9} "
            f"{r['steps']:>6d} "
            f"{r['mean'] * 1e3:>10.4f} "
            f"{r['median'] * 1e3:>12.4f} "
            f"{r['p99'] * 1e3:>9.4f} "
            f"{r['max'] * 1e3:>9.4f} "
            f"{1.0 / r['mean']:>10.1f} "
            f"{r['loop_wall'] / r['sim_time']:>9.3f} "
            f"{r['total_cmd'] / r['sim_time']:>8.3f}"
        )


def print_hand_delta(results):
    header = (
        f"{'dt [s]':>7} {'mean w/ hand [ms]':>18} {'mean arm only [ms]':>19} "
        f"{'delta [ms]':>11} {'hand share':>11}"
    )
    print(header)
    print("-" * len(header))
    for dt in TIMESTEPS:
        with_hand = next(
            r for r in results if r["dt"] == dt and r["command_hand"]
        )
        arm_only = next(
            r for r in results if r["dt"] == dt and not r["command_hand"]
        )
        delta = with_hand["mean"] - arm_only["mean"]
        print(
            f"{dt:>7.4f} "
            f"{with_hand['mean'] * 1e3:>18.4f} "
            f"{arm_only['mean'] * 1e3:>19.4f} "
            f"{delta * 1e3:>11.4f} "
            f"{delta / with_hand['mean'] * 100.0:>10.2f}%"
        )


def print_context(results):
    header = (
        f"{'dt [s]':>7} {'hand cmd':>9} {'mean ncon':>10} {'max ncon':>9} "
        f"{'max |arm err| [rad]':>20} {'max hand dev [rad]':>19}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['dt']:>7.4f} "
            f"{('yes' if r['command_hand'] else 'no'):>9} "
            f"{r['mean_ncon']:>10.2f} "
            f"{r['max_ncon']:>9d} "
            f"{r['arm_err']:>20.6f} "
            f"{r['hand_dev']:>19.6f}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=STEPS)
    args = parser.parse_args()

    results = []
    for dt in TIMESTEPS:
        for command_hand in (True, False):
            results.append(measure(dt, args.steps, command_hand))

    print("setCommands loop rate")
    print()
    print_table(results)
    print()
    print("hand command share")
    print()
    print_hand_delta(results)
    print()
    print("run context")
    print()
    print_context(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
