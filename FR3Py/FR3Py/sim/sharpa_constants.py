import math

JOINT_ORDER = (
    "thumb_CMC_FE",
    "thumb_CMC_AA",
    "thumb_MCP_FE",
    "thumb_MCP_AA",
    "thumb_IP",
    "index_MCP_FE",
    "index_MCP_AA",
    "index_PIP",
    "index_DIP",
    "middle_MCP_FE",
    "middle_MCP_AA",
    "middle_PIP",
    "middle_DIP",
    "ring_MCP_FE",
    "ring_MCP_AA",
    "ring_PIP",
    "ring_DIP",
    "pinky_CMC",
    "pinky_MCP_FE",
    "pinky_MCP_AA",
    "pinky_PIP",
    "pinky_DIP",
)

JOINT_NAMES = tuple(f"right_{name}" for name in JOINT_ORDER)

_DEFAULT_GRASP_POSE_DEG = {
    "thumb_CMC_FE": 95.12771,
    "thumb_CMC_AA": -3.11244,
    "thumb_MCP_FE": 14.81626,
    "thumb_MCP_AA": -1.03493,
    "thumb_IP": 12.23986,
    "index_MCP_FE": 65.21091,
    "index_MCP_AA": 6.1133,
    "index_PIP": 15.58495,
    "index_DIP": 5.90325,
    "middle_MCP_FE": 31.74149,
    "middle_MCP_AA": -0.95812,
    "middle_PIP": 41.88173,
    "middle_DIP": 12.844,
    "ring_MCP_FE": 31.72383,
    "ring_MCP_AA": 9.84458,
    "ring_PIP": 35.22366,
    "ring_DIP": 18.02839,
    "pinky_CMC": 10.9712,
    "pinky_MCP_FE": 68.30895,
    "pinky_MCP_AA": 7.99151,
    "pinky_PIP": 5.89626,
    "pinky_DIP": 5.89875,
}

DEFAULT_GRASP_POSE = {
    name: math.radians(deg) for name, deg in _DEFAULT_GRASP_POSE_DEG.items()
}

_CLASS_EFFORT = {
    "CMC": 3.3,
    "PCMC": 0.5285,
    "MCP": 1.864,
    "PIP": 0.638,
    "DIP": 0.189369,
}

EFFORT_LIMITS = {
    "thumb_CMC_FE": _CLASS_EFFORT["CMC"],
    "thumb_CMC_AA": _CLASS_EFFORT["CMC"],
    "thumb_MCP_FE": _CLASS_EFFORT["MCP"],
    "thumb_MCP_AA": _CLASS_EFFORT["MCP"],
    "thumb_IP": _CLASS_EFFORT["PIP"],
    "index_MCP_FE": _CLASS_EFFORT["MCP"],
    "index_MCP_AA": _CLASS_EFFORT["MCP"],
    "index_PIP": _CLASS_EFFORT["PIP"],
    "index_DIP": _CLASS_EFFORT["DIP"],
    "middle_MCP_FE": _CLASS_EFFORT["MCP"],
    "middle_MCP_AA": _CLASS_EFFORT["MCP"],
    "middle_PIP": _CLASS_EFFORT["PIP"],
    "middle_DIP": _CLASS_EFFORT["DIP"],
    "ring_MCP_FE": _CLASS_EFFORT["MCP"],
    "ring_MCP_AA": _CLASS_EFFORT["MCP"],
    "ring_PIP": _CLASS_EFFORT["PIP"],
    "ring_DIP": _CLASS_EFFORT["DIP"],
    "pinky_CMC": _CLASS_EFFORT["PCMC"],
    "pinky_MCP_FE": _CLASS_EFFORT["MCP"],
    "pinky_MCP_AA": _CLASS_EFFORT["MCP"],
    "pinky_PIP": _CLASS_EFFORT["PIP"],
    "pinky_DIP": _CLASS_EFFORT["DIP"],
}

FINGERTIP_GEOMS = (
    "right_thumb_fingertip_collision",
    "right_index_fingertip_collision",
    "right_middle_fingertip_collision",
    "right_ring_fingertip_collision",
    "right_pinky_fingertip_collision",
)
