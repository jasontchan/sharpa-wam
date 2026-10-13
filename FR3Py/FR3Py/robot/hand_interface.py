import os
import sys
import threading
import time

import numpy as np

from FR3Py import ASSETS_PATH

TACTILE_CONFIG_FILE = os.environ.get(
    "SHARPA_TACTILE_CONFIG", "/root/.sharpa-pilot/config/tactile.json"
)
TAC_UV_MAP_DIR = os.environ.get(
    "SHARPA_TACTILE_MAP_DIR", os.path.join(ASSETS_PATH, "sharpa", "tactile_map")
)


def _default_sdk_path():
    """
    Find the directory that contains the `sharpa` package shipped with the
    Sharpa Wave SDK image.

    @return: (str or None) The value of `SHARPA_SDK_PYTHON_PATH` when it is
    set, otherwise the first of the known SDK install locations that exists,
    or None when none of them do.
    """
    env_path = os.environ.get("SHARPA_SDK_PYTHON_PATH")
    if env_path:
        return env_path
    for candidate in ("/opt/sharpa-wave-sdk/python", "/root/sharpa-wave-sdk/python"):
        if os.path.isdir(candidate):
            return candidate
    return None


SDK_PATH = _default_sdk_path()
NUM_HAND_DOFS = 22
NUM_TACTILE_CHANNELS = 5
COEFF_TOL = 1e-3
CONFIG_SOURCE = "SharpaPilot"

# Range of motion [min, max] per joint in degrees, in the SDK joint order
# (index 0..21), from the Sharpa Wave manual. The MCP-AA limits of the index,
# middle, ring and pinky are the outer bounds; the reachable region shrinks
# with MCP-FE (see "2-DOF Joint Workspace" in the manual's appendix).
_JOINT_LIMITS_DEG = (
    (-5, 105),  # 0  Thumb CMC FE
    (-18, 18),  # 1  Thumb CMC AA
    (-25, 75),  # 2  Thumb MCP FE
    (-18, 18),  # 3  Thumb MCP AA
    (0, 95),  # 4  Thumb IP
    (-10, 84),  # 5  Index MCP FE
    (-18, 18),  # 6  Index MCP AA
    (0, 95),  # 7  Index PIP
    (0, 75),  # 8  Index DIP
    (-10, 84),  # 9  Middle MCP FE
    (-18, 18),  # 10 Middle MCP AA
    (0, 95),  # 11 Middle PIP
    (0, 75),  # 12 Middle DIP
    (-10, 84),  # 13 Ring MCP FE
    (-18, 18),  # 14 Ring MCP AA
    (0, 95),  # 15 Ring PIP
    (0, 75),  # 16 Ring DIP
    (0, 25),  # 17 Pinky CMC
    (-10, 84),  # 18 Pinky MCP FE
    (-18, 18),  # 19 Pinky MCP AA
    (0, 95),  # 20 Pinky PIP
    (0, 75),  # 21 Pinky DIP
)
JOINT_LIMITS_RAD = np.deg2rad(np.array(_JOINT_LIMITS_DEG, dtype=np.float32))
assert JOINT_LIMITS_RAD.shape == (NUM_HAND_DOFS, 2)


class HandConfigError(RuntimeError):
    """
    Raised when the hand is not configured the way this interface expects.

    The hand is configured in SharpaPilot, not from here. This interface only
    reads the configuration back, and raises this when a value cannot be read
    or does not match what it was told to expect.
    """


class SharpaHand:
    """
    Class for communication with the Sharpa Wave dexterous hand.
    It wraps the Sharpa SDK to send joint position commands to the hand and
    receive joint states and tactile readings.
    """

    def __init__(
        self,
        hand_side=1,
        control_mode="POSITION",
        speed_coef=0.3,
        current_coef=0.6,
        enable_tactile=True,
        sdk_path=SDK_PATH,
        tac_uv_map_dir=TAC_UV_MAP_DIR,
        verify_every=0,
    ):
        """
        Initialize an instance of the class.

        The hand is configured in SharpaPilot, not from here. `control_mode`,
        `speed_coef` and `current_coef` are the values this interface expects
        to find on the device: they are read back and checked, never applied.
        Construction fails with `HandConfigError` when the device reports
        anything else, so change them here only to match what SharpaPilot is
        configured with.

        @param hand_side: (int) 0 for the left hand and 1 for the right hand.
        @param control_mode: (str) Name of the SDK `ControlMode` member the
        device is expected to be in.
        @param speed_coef: (float) Speed coefficient the device is expected to
        report, compared within `COEFF_TOL`.
        @param current_coef: (float) Current coefficient the device is expected
        to report, compared within `COEFF_TOL`.
        @param enable_tactile: (bool) Whether to register the tactile callback.
        @param sdk_path: (str or None) Directory containing the `sharpa` package,
        used only when `sharpa` is not importable from the environment.
        Defaults to `SDK_PATH`.
        @param tac_uv_map_dir: (str) Directory containing the tactile UV maps
        used to place contact points on the fingertip surface.
        @param verify_every: (int) Re-read the control source and mode every
        this many `setCommands` calls. 0, the default, disables the periodic
        check, which is what a real control loop wants: each read is a
        synchronous round trip on the same connection the commands use. Set it
        to a small number while debugging a hand that drops out of SDK control.
        """
        assert hand_side in (0, 1), "hand_side must be 0 (left) or 1 (right)"
        assert verify_every >= 0, "verify_every must be 0 (off) or positive"
        self.hand_side = hand_side
        self.control_mode = control_mode
        self.speed_coef = speed_coef
        self.current_coef = current_coef
        self.enable_tactile = enable_tactile
        self.verify_every = verify_every
        self._verified_control_mode = None
        self._commands_since_verify = 0
        self.tactile_config_file = TACTILE_CONFIG_FILE
        self.tac_uv_map_dir = tac_uv_map_dir
        self.tac_uv_map = None
        self._tac_uv_map_loaded = False

        self._load_sdk(sdk_path)

        self.channel_offset = NUM_TACTILE_CHANNELS * (1 - self.hand_side)
        self.channels = list(
            range(self.channel_offset, self.channel_offset + NUM_TACTILE_CHANNELS)
        )
        self._frames_lock = threading.Lock()
        self.frames = {ch: (0.0, None) for ch in self.channels}

        self.hand = self._connect()
        self.hand_info = self.hand.get_device_info()
        self._initialize()
        self.hand.start()
        time.sleep(0.2)
        print("Hand Interface Running...")

    def _load_sdk(self, sdk_path):
        """
        Import the Sharpa SDK, extending `sys.path` with `sdk_path` if the
        `sharpa` package is not already installed in the environment.
        """
        try:
            import sharpa
        except ImportError:
            if sdk_path is None:
                raise ImportError(
                    "The `sharpa` package is not installed. Pass `sdk_path` "
                    "pointing to the directory that contains it."
                )
            sdk_path = os.path.expanduser(sdk_path)
            if sdk_path not in sys.path:
                sys.path.insert(0, sdk_path)
            try:
                import sharpa
            except ImportError:
                raise ImportError(
                    "The `sharpa` package is not installed and was not found in "
                    f"`{sdk_path}`. Pass `sdk_path` pointing to the directory "
                    "that contains it."
                )

        self.sdk = sharpa
        self.SharpaWaveManager = sharpa.SharpaWaveManager
        self.SharpaWaveConfig = sharpa.SharpaWaveConfig
        self.ControlMode = sharpa.ControlMode
        self.ControlSource = sharpa.ControlSource

    def _connect(self):
        """
        Discover the attached hands and connect to the first one reported by
        the SDK device manager.

        @return: The connected SDK hand handle.
        """
        print("Searching for devices...")
        self.manager = self.SharpaWaveManager.get_instance()
        time.sleep(1)
        while True:
            devices = self.manager.get_all_device_sn()
            if not devices:
                print("No available devices found")
                time.sleep(1)
                continue
            print(f"Device found: {devices[0]}")
            cfg = self.SharpaWaveConfig()
            cfg.tactile_config_file = self.tactile_config_file
            return self.manager.connect(devices[0], cfg)

    def _check(self, error, what):
        """
        Raise if an SDK call returned a non-zero error code.

        @param error: The SDK error object returned by the call.
        @param what: (str) Human readable name of the call.
        """
        if error.code != 0:
            raise RuntimeError(f"Failed to {what}: {error.message}")

    def _verify(self, what, getter, expected, tol=None):
        """
        Read a setting back from the hand and confirm it holds the expected
        value.

        @param what: (str) Human readable name of the setting.
        @param getter: (callable) SDK getter returning `(Error, value)`.
        @param expected: The value the setting was written with.
        @param tol: (float or None) Absolute tolerance used to compare numeric
        values. When None the comparison is an exact equality, which is what
        the enum valued settings need.
        @raise HandConfigError: When the read back fails or does not match.
        """
        error, value = getter()
        if error.code != 0:
            raise HandConfigError(f"Failed to read back {what}: {error.message}")
        matches = value == expected if tol is None else abs(value - expected) <= tol
        if not matches:
            raise HandConfigError(
                f"Unexpected {what}: expected {expected!r}, got {value!r}"
            )

    def _config_checks(self):
        """
        Describe the configuration this interface expects to find on the
        device, so that it can be checked at startup and again on demand.

        @return: (tuple) One `(what, getter, expected, tol)` entry per setting,
        in the order they are checked.
        """
        return (
            (
                "control mode",
                self.hand.get_control_mode,
                getattr(self.ControlMode, self.control_mode),
                None,
            ),
            (
                "control source",
                self.hand.get_control_source,
                self.ControlSource.SDK,
                None,
            ),
            ("speed coeff", self.hand.get_speed_coeff, self.speed_coef, COEFF_TOL),
            (
                "current coeff",
                self.hand.get_current_coeff,
                self.current_coef,
                COEFF_TOL,
            ),
        )

    def _initialize(self):
        """
        Confirm the connected hand is configured the way this interface
        expects, and register the tactile callback if enabled.

        Nothing is written: the control mode, control source and the
        speed/current coefficients are configured in SharpaPilot. This reads
        them back and refuses to run when they do not match, rather than
        quietly driving a hand that is in a different mode than the caller
        thinks.

        @raise HandConfigError: On the first setting that does not match.
        """
        try:
            self.verifyConfig()
        except HandConfigError as e:
            raise HandConfigError(
                f"{e}. This is not set from the SDK: configure it in "
                f"{CONFIG_SOURCE} to match, then run again."
            ) from e
        # Nothing has confirmed the mode since, so the first command re-checks.
        self._verified_control_mode = None
        self._commands_since_verify = 0
        if self.enable_tactile:
            self.hand.set_tactile_callback(self._tactile_callback)

    def verifyConfig(self):
        """
        Confirm that the hand still reports the configuration this interface
        expects, and that it is still enabled.

        @return: None
        @raise HandConfigError: On the first setting that does not match.
        """
        for what, getter, expected, tol in self._config_checks():
            self._verify(what, getter, expected, tol)
        self._verify("enable state", self.hand.get_enable_state, True)

    def _verifyControlAuthority(self):
        """
        Confirm the hand is still taking its commands from the SDK and is in
        the control mode this interface expects, before a command is sent.

        Each call is two synchronous round trips to the hand, so commands do
        not do this unconditionally. See `setCommands`.

        @raise HandConfigError: When the control source or mode does not match.
        """
        self._verify(
            "control source", self.hand.get_control_source, self.ControlSource.SDK
        )
        self._verify(
            "control mode",
            self.hand.get_control_mode,
            getattr(self.ControlMode, self.control_mode),
        )
        self._verified_control_mode = self.control_mode
        self._commands_since_verify = 0

    def _tactile_callback(self, frame):
        """
        Callback function that executes whenever a new tactile frame is
        received from the SDK.

        @param frame: (dict) The tactile frame published by the SDK.
        """
        ch = frame["channel"]
        content = frame.get("content", {})

        f6 = content.get("F6")
        f_norm = 0.0
        if f6 is not None:
            f_norm = float(np.linalg.norm(np.asarray(f6, dtype=np.float32)[:3]))

        points = content.get("CONTACT_POINT")
        if points is not None:
            points = np.asarray(points, dtype=np.float32).reshape(-1, 3)

        with self._frames_lock:
            self.frames[ch] = (f_norm, points)

    def getJointStates(self):
        """
        Get the current joint angles of the hand in the SDK joint ordering.

        @return: (dict) The joint state {'q': (numpy.ndarray, shape=(22,),
        dtype=numpy.float32)}.
        """
        states = self.hand.get_states()
        q = np.asarray(states.angles, dtype=np.float32).reshape(NUM_HAND_DOFS)
        return {"q": q}

    def _load_tac_uv_map(self):
        """
        Load the tactile UV maps once. The maps are set to None when they
        cannot be read, and no further attempt is made to load them.
        """
        self._tac_uv_map_loaded = True
        self.tac_uv_map = None
        try:
            four_finger = np.load(
                os.path.join(self.tac_uv_map_dir, "tactileSensor_map_4F_point.npy")
            )
            thumb = np.load(
                os.path.join(self.tac_uv_map_dir, "tactileSensor_map_TH_point.npy")
            )
        except (OSError, ValueError) as e:
            print(
                f"[SharpaHand] Could not load the tactile UV maps from "
                f"{self.tac_uv_map_dir} ({e}). Reporting raw contact points."
            )
            return
        self.tac_uv_map = [four_finger] * 4 + [thumb]

    def getTactile(self):
        """
        Get the latest tactile reading of every channel of this hand.

        @return: (dict) A mapping from the SDK channel index to
        {'force': (float), 'contact_pos': (numpy.ndarray, shape=(3,),
        dtype=numpy.float32)}. When the tactile UV maps are available,
        'contact_pos' is the contact location on the fingertip surface in
        meters; otherwise it is the raw SDK contact point, which is in the
        units of the sensor frame and not in meters.
        """
        if not self._tac_uv_map_loaded:
            self._load_tac_uv_map()

        with self._frames_lock:
            frames = {
                ch: (force, None if points is None else points.copy())
                for ch, (force, points) in self.frames.items()
            }

        tactile = {}
        for ch, (force, points) in frames.items():
            contact_pos = np.zeros(3, dtype=np.float32)
            if points is not None and points.shape[0] > 0:
                centroid = points[int(np.argmax(points[:, 2]))]
                if self.tac_uv_map is None:
                    contact_pos = centroid.astype(np.float32)
                else:
                    idx = ch - self.channel_offset
                    pos = self.tac_uv_map[idx][int(centroid[1]), int(centroid[0])]
                    contact_pos = np.asarray(pos[:3], dtype=np.float32) / 1000.0
            tactile[ch] = {"force": force, "contact_pos": contact_pos}
        return tactile

    def calibTactile(self):
        """
        Run the SDK tactile calibration routine.

        @return: (bool) True if the calibration succeeded.
        """
        return bool(self.hand.calib_tactile())

    def _authorityCheckDue(self):
        """
        Decide whether the control source and mode should be read back before
        the command that is about to be sent.

        The check runs on the first command after the control mode is written,
        and then only every `verify_every` commands. It is off by default
        because a control loop would otherwise pay two synchronous round trips
        per tick on the same connection that carries the commands. Drift that
        happens later is caught by the error the command itself returns and by
        `get_fault_code`, and `verifyConfig` is available to check on demand.

        @return: (bool) True when the check should run.
        """
        if self._verified_control_mode != self.control_mode:
            return True
        return self.verify_every > 0 and (
            self._commands_since_verify >= self.verify_every
        )

    def setCommands(self, cmd):
        """
        Send a joint position command to the hand in the SDK joint ordering.

        The command is clamped per joint to `JOINT_LIMITS_RAD` before it is
        sent.

        @param cmd: (numpy.ndarray, shape=(22,)) The joint command to send.
        @raise HandConfigError: When the control source or mode is checked and
        does not match. The command is not sent in that case.
        @raise RuntimeError: When the hand rejects the command.
        """
        cmd = np.asarray(cmd, dtype=np.float32)
        assert cmd.shape == (NUM_HAND_DOFS,), "Command shape must be (22,)"
        cmd = np.clip(cmd, JOINT_LIMITS_RAD[:, 0], JOINT_LIMITS_RAD[:, 1])
        self._commands_since_verify += 1
        if self._authorityCheckDue():
            self._verifyControlAuthority()
        self._check(self.hand.set_joint_position(cmd), "send joint position")
        self.cmd_log = cmd

    def goHome(self):
        """
        Drive every joint of the hand to zero and wait for the motion to settle.

        This is not on the hot path, so the control source and mode are always
        read back first.

        @raise HandConfigError: When the hand is no longer taking commands from
        the SDK, or is no longer in the expected control mode. The command is
        not sent in that case.
        @raise RuntimeError: When the hand rejects the command.
        """
        self._verifyControlAuthority()
        cmd = np.clip(
            np.zeros(NUM_HAND_DOFS, dtype=np.float32),
            JOINT_LIMITS_RAD[:, 0],
            JOINT_LIMITS_RAD[:, 1],
        )
        self._check(self.hand.set_joint_position(cmd), "send joint position")
        time.sleep(3)

    def close(self):
        """
        Stop the hand and disconnect every device held by the SDK manager,
        effectively shutting down the interface.
        """
        if self.hand is not None:
            self.hand.stop()
            self.hand = None
        self.manager.disconnect_all()
        print("Hand Interface Closed.")
