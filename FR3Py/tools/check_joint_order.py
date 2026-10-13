#!/usr/bin/env python3
"""
Check the MJCF/SDK hand joint permutation and emit tools/joint_order.csv.
"""

import csv
import os
import sys

import numpy as np

from FR3Py.sim.joint_order import (
    MJCF_TO_SDK,
    NUM_JOINTS,
    SDK_JOINT_NAMES,
    SDK_TO_MJCF,
    mjcf_to_sdk,
    sdk_to_mjcf,
)
from FR3Py.sim.sharpa_constants import JOINT_NAMES

CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "joint_order.csv")

_failures = []


def check(condition, message):
    if condition:
        return True
    _failures.append(message)
    return False


def check_permutation(name, idx):
    expected = np.arange(NUM_JOINTS)
    check(
        idx.shape == (NUM_JOINTS,),
        f"{name} has shape {idx.shape}, expected ({NUM_JOINTS},)",
    )
    if idx.shape != (NUM_JOINTS,):
        return
    check(
        len(np.unique(idx)) == NUM_JOINTS,
        f"{name} contains repeated indices: {idx.tolist()}",
    )
    check(
        np.array_equal(np.sort(idx), expected),
        f"{name} is not a permutation of 0..{NUM_JOINTS - 1}: {idx.tolist()}",
    )


def check_round_trips():
    rng = np.random.default_rng(0)
    for _ in range(256):
        q = rng.standard_normal(NUM_JOINTS)
        if not check(
            np.array_equal(sdk_to_mjcf(mjcf_to_sdk(q)), q),
            "sdk_to_mjcf(mjcf_to_sdk(q)) != q",
        ):
            return
        if not check(
            np.array_equal(mjcf_to_sdk(sdk_to_mjcf(q)), q),
            "mjcf_to_sdk(sdk_to_mjcf(q)) != q",
        ):
            return


def check_inverses():
    expected = np.arange(NUM_JOINTS)
    check(
        np.array_equal(MJCF_TO_SDK[SDK_TO_MJCF], expected),
        f"MJCF_TO_SDK[SDK_TO_MJCF] != identity: "
        f"{MJCF_TO_SDK[SDK_TO_MJCF].tolist()}",
    )
    check(
        np.array_equal(SDK_TO_MJCF[MJCF_TO_SDK], expected),
        f"SDK_TO_MJCF[MJCF_TO_SDK] != identity: "
        f"{SDK_TO_MJCF[MJCF_TO_SDK].tolist()}",
    )


def check_names():
    for sdk_index, mjcf_index in enumerate(MJCF_TO_SDK):
        check(
            SDK_JOINT_NAMES[sdk_index] == JOINT_NAMES[mjcf_index],
            f"MJCF_TO_SDK[{sdk_index}] = {mjcf_index} maps "
            f"{JOINT_NAMES[mjcf_index]} onto {SDK_JOINT_NAMES[sdk_index]}",
        )


def print_table():
    header = f"{'idx':>3}  {'MJCF (JOINT_NAMES)':28s}  {'SDK_JOINT_NAMES':28s}  {'SDK_TO_MJCF':>11}  {'MJCF_TO_SDK':>11}"
    print(header)
    print("-" * len(header))
    for i in range(NUM_JOINTS):
        print(
            f"{i:3d}  {JOINT_NAMES[i]:28s}  {SDK_JOINT_NAMES[i]:28s}  "
            f"{SDK_TO_MJCF[i]:11d}  {MJCF_TO_SDK[i]:11d}"
        )
    print()
    if np.array_equal(MJCF_TO_SDK, np.arange(NUM_JOINTS)):
        print("Permutation is the identity: MJCF and SDK joint orders agree.")
    else:
        print("Permutation is non-trivial.")
    print()


def write_csv():
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["mjcf_index", "mjcf_joint_name", "sdk_index", "sdk_joint_name"]
        )
        for mjcf_index in range(NUM_JOINTS):
            sdk_index = int(SDK_TO_MJCF[mjcf_index])
            writer.writerow(
                [
                    mjcf_index,
                    JOINT_NAMES[mjcf_index],
                    sdk_index,
                    SDK_JOINT_NAMES[sdk_index],
                ]
            )
    print(f"Wrote {CSV_PATH}")


def main():
    check_permutation("MJCF_TO_SDK", MJCF_TO_SDK)
    check_permutation("SDK_TO_MJCF", SDK_TO_MJCF)
    check_round_trips()
    check_inverses()
    check_names()

    print_table()

    if _failures:
        print(f"FAILED ({len(_failures)}):", file=sys.stderr)
        for message in _failures:
            print(f"  - {message}", file=sys.stderr)
        return 1

    write_csv()
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
