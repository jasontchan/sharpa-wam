"""
Permutations between the MJCF hand joint order and the Sharpa SDK joint order.

`SDK_JOINT_NAMES` is UNVERIFIED against hardware. It was resolved offline from
the IsaacLab index arrays `_ISAACLAB2SHARPA_IDX` / `_SHARPA2ISAACLAB_IDX` in
Sharpa's IsaacLab training utilities (`utils/misc.py`), whose IsaacLab-side
ordering is the breadth-first articulation order of
`assets/SharpaWave/right_sharpa_wave.usda`. Nothing here has been checked
against a physical hand.
"""

import numpy as np

from FR3Py.sim.sharpa_constants import JOINT_NAMES

SDK_JOINT_NAMES = (
    "right_thumb_CMC_FE",
    "right_thumb_CMC_AA",
    "right_thumb_MCP_FE",
    "right_thumb_MCP_AA",
    "right_thumb_IP",
    "right_index_MCP_FE",
    "right_index_MCP_AA",
    "right_index_PIP",
    "right_index_DIP",
    "right_middle_MCP_FE",
    "right_middle_MCP_AA",
    "right_middle_PIP",
    "right_middle_DIP",
    "right_ring_MCP_FE",
    "right_ring_MCP_AA",
    "right_ring_PIP",
    "right_ring_DIP",
    "right_pinky_CMC",
    "right_pinky_MCP_FE",
    "right_pinky_MCP_AA",
    "right_pinky_PIP",
    "right_pinky_DIP",
)

NUM_JOINTS = len(JOINT_NAMES)


def _require_names():
    if SDK_JOINT_NAMES is None:
        raise RuntimeError(
            "SDK_JOINT_NAMES is unresolved; the MJCF/SDK joint permutation is "
            "unavailable."
        )
    if len(SDK_JOINT_NAMES) != NUM_JOINTS:
        raise RuntimeError(
            f"SDK_JOINT_NAMES has {len(SDK_JOINT_NAMES)} entries, expected "
            f"{NUM_JOINTS}."
        )
    if set(SDK_JOINT_NAMES) != set(JOINT_NAMES):
        raise RuntimeError(
            "SDK_JOINT_NAMES and JOINT_NAMES do not name the same joints."
        )
    return SDK_JOINT_NAMES


def _build_index_arrays():
    names = _require_names()
    mjcf_pos = {name: i for i, name in enumerate(JOINT_NAMES)}
    sdk_pos = {name: i for i, name in enumerate(names)}
    mjcf_to_sdk = np.array([mjcf_pos[name] for name in names], dtype=np.intp)
    sdk_to_mjcf = np.array([sdk_pos[name] for name in JOINT_NAMES], dtype=np.intp)
    return mjcf_to_sdk, sdk_to_mjcf


MJCF_TO_SDK, SDK_TO_MJCF = _build_index_arrays()


def mjcf_to_sdk(q):
    q = np.asarray(q)
    assert q.shape == (NUM_JOINTS,), (
        f"Expected an array of shape ({NUM_JOINTS},), got {q.shape}"
    )
    return q[MJCF_TO_SDK]


def sdk_to_mjcf(q):
    q = np.asarray(q)
    assert q.shape == (NUM_JOINTS,), (
        f"Expected an array of shape ({NUM_JOINTS},), got {q.shape}"
    )
    return q[SDK_TO_MJCF]
