import importlib.util
import os
import re

import mujoco

from FR3Py import ASSETS_PATH
from FR3Py.sim.sharpa_constants import JOINT_NAMES

MUJOCO_MIN_VERSION = (3, 2)

MUJOCO_ASSETS = os.path.join(ASSETS_PATH, "mujoco")
SHARPA_ASSETS = os.path.join(MUJOCO_ASSETS, "sharpa")
FR3_XML = os.path.join(MUJOCO_ASSETS, "fr3.xml")
MOUNT_XML = os.path.join(SHARPA_ASSETS, "sharpa_mount.xml")
HAND_XML = os.path.join(SHARPA_ASSETS, "sharpa_wave", "right_hand.xml")
MOUNT_FRAME_PY = os.path.join(SHARPA_ASSETS, "mount_frame.py")

MOUNT_PREFIX = "mount_"
MOUNT_ROOT_BODY = "sharpa_mount"
MOUNT_BODY = MOUNT_PREFIX + MOUNT_ROOT_BODY
HAND_PREFIX = ""
HAND_ROOT_BODY = "right_hand_C_MC"
FLANGE_BODY = "link7"

GRIPPER_BODY = "hand"
GRIPPER_TENDON = "split"
GRIPPER_ACTUATOR = "actuator8"
GRIPPER_JOINTS = ("finger_joint1", "finger_joint2")

NUM_ARM_ACTUATORS = 7
NUM_HAND_ACTUATORS = len(JOINT_NAMES)


def _check_mujoco_version():
    """
    Verify that the installed mujoco exposes the MjSpec attach API.
    """
    parts = []
    for token in mujoco.__version__.split(".")[:2]:
        digits = "".join(c for c in token if c.isdigit())
        parts.append(int(digits) if digits else 0)
    if tuple(parts) < MUJOCO_MIN_VERSION:
        raise RuntimeError(
            f"FR3Py.sim.sharpa_spec requires mujoco >= "
            f"{MUJOCO_MIN_VERSION[0]}.{MUJOCO_MIN_VERSION[1]} for the MjSpec "
            f"attach API, but mujoco {mujoco.__version__} is installed."
        )


def load_mount_frame():
    """
    Load the flange and palm transforms from the assets directory.

    @return: (tuple) (FLANGE_TO_MOUNT, MOUNT_TO_PALM), each a
    (pos_xyz, quat_wxyz) pair.
    """
    loader = importlib.util.spec_from_file_location(
        "fr3py_sharpa_mount_frame", MOUNT_FRAME_PY
    )
    module = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(module)
    return module.FLANGE_TO_MOUNT, module.MOUNT_TO_PALM


def _default_class_names(xml_path):
    """
    Read the default class names declared by an MJCF file.

    @param xml_path: (str) Path to the MJCF file.
    @return: (list) The declared default class names.
    """
    with open(xml_path) as f:
        return re.findall(r'<default\s+class="([^"]+)"', f.read())


def _dedupe_assets(child, parent, prefix, xml_path):
    """
    Rename every asset of `child` whose name is already taken in `parent` and
    retarget the references to it, so that the child can be attached without an
    asset name prefix.

    @param child: (mujoco.MjSpec) The spec about to be attached.
    @param parent: (mujoco.MjSpec) The spec being attached to.
    @param prefix: (str) Prefix applied to the renamed assets.
    @param xml_path: (str) Path of the child MJCF, used to find default classes.
    """
    classes = [child.find_default(name) for name in _default_class_names(xml_path)]
    classes = [c for c in classes if c is not None]

    for kind, geom_attr, class_attr in (
        ("materials", "material", "material"),
        ("meshes", "meshname", "mesh"),
    ):
        taken = {item.name for item in getattr(parent, kind)}
        for item in getattr(child, kind):
            if item.name not in taken:
                continue
            old = item.name
            new = prefix + old
            item.name = new
            for body in child.bodies:
                for geom in body.geoms:
                    if getattr(geom, geom_attr) == old:
                        setattr(geom, geom_attr, new)
            for klass in classes:
                if getattr(klass.geom, class_attr) == old:
                    setattr(klass.geom, class_attr, new)


def _strip_franka_gripper(spec):
    """
    Remove the Franka gripper body subtree and everything that refers to it.

    @param spec: (mujoco.MjSpec) The FR3 spec to strip in place.
    """
    body = spec.body(GRIPPER_BODY)
    if body is not None:
        spec.delete(body)

    tendon = spec.tendon(GRIPPER_TENDON)
    if tendon is not None:
        spec.delete(tendon)

    for equality in list(spec.equalities):
        if equality.name1 in GRIPPER_JOINTS or equality.name2 in GRIPPER_JOINTS:
            spec.delete(equality)

    actuator = spec.actuator(GRIPPER_ACTUATOR)
    if actuator is not None:
        spec.delete(actuator)

    for key in list(spec.keys):
        spec.delete(key)


def _add_exclude(spec, body1, body2):
    """
    Add a contact exclusion between two bodies.

    @param spec: (mujoco.MjSpec) The merged spec.
    @param body1: (str) Name of the first body.
    @param body2: (str) Name of the second body.
    """
    exclude = spec.add_exclude()
    exclude.bodyname1 = body1
    exclude.bodyname2 = body2


def _verify(spec):
    """
    Compile the spec and check the actuator layout the interface relies on.

    @param spec: (mujoco.MjSpec) The merged spec.
    @return: (mujoco.MjModel) The compiled model.
    """
    model = spec.compile()

    names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        for i in range(model.nu)
    ]
    expected = NUM_ARM_ACTUATORS + NUM_HAND_ACTUATORS
    assert model.nu == expected, (
        f"Expected {expected} actuators in the merged model, found {model.nu}"
    )

    arm_names = names[:NUM_ARM_ACTUATORS]
    expected_arm = [f"actuator{i + 1}" for i in range(NUM_ARM_ACTUATORS)]
    assert arm_names == expected_arm, (
        f"Arm actuators must occupy ctrl indices 0-{NUM_ARM_ACTUATORS - 1}, "
        f"found {arm_names}"
    )

    hand_names = names[NUM_ARM_ACTUATORS:expected]
    assert hand_names == list(JOINT_NAMES), (
        "Hand actuators must occupy ctrl indices "
        f"{NUM_ARM_ACTUATORS}-{expected - 1} in JOINT_NAMES order, "
        f"found {hand_names}"
    )
    return model


def build_fr3_sharpa_spec():
    """
    Compose the FR3 arm and the Sharpa Wave right hand into a single spec.

    @return: (mujoco.MjSpec) The merged spec, already verified to compile.
    """
    _check_mujoco_version()
    flange_to_mount, mount_to_palm = load_mount_frame()

    spec = mujoco.MjSpec.from_file(FR3_XML)
    _strip_franka_gripper(spec)

    mount = mujoco.MjSpec.from_file(MOUNT_XML)
    _dedupe_assets(mount, spec, MOUNT_PREFIX, MOUNT_XML)
    mount_frame = spec.body(FLANGE_BODY).add_frame(
        pos=list(flange_to_mount[0]), quat=list(flange_to_mount[1])
    )
    spec.attach(mount, prefix=MOUNT_PREFIX, frame=mount_frame)

    hand = mujoco.MjSpec.from_file(HAND_XML)
    spec.option.impratio = hand.option.impratio
    spec.option.cone = hand.option.cone
    _dedupe_assets(hand, spec, "sharpa_", HAND_XML)
    palm_frame = spec.body(MOUNT_BODY).add_frame(
        pos=list(mount_to_palm[0]), quat=list(mount_to_palm[1])
    )
    spec.attach(hand, prefix=HAND_PREFIX, frame=palm_frame)

    _add_exclude(spec, MOUNT_BODY, FLANGE_BODY)
    _add_exclude(spec, MOUNT_BODY, HAND_ROOT_BODY)

    _verify(spec)
    return spec


def save_xml(path):
    """
    Compile the merged model and write it out as a single MJCF file.

    @param path: (str) Destination path of the written MJCF.
    @return: (str) The path that was written.
    """
    spec = build_fr3_sharpa_spec()
    spec.compile()
    with open(path, "w") as f:
        f.write(spec.to_xml())
    return path
