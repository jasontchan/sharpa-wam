"""Rigid transforms placing the Sharpa Wave hand on the FR3 flange.

Both transforms are ``(pos_xyz, quat_wxyz)`` and BOTH REQUIRE VISUAL
CALIBRATION against the real hardware before the model can be trusted. Run
``tools/calibrate_sharpa_mount.py`` to view the merged model and edit the values
here until the mount and palm sit where they do on the robot.
"""

FLANGE_TO_MOUNT = (
    (-0.055, 0.003, 0.1),
    (0.9238795, 0.0, 0.0, -0.3826834),
)

MOUNT_TO_PALM = (
    (0.04, 0.037, 0.03),
    (1.0, 0.0, 0.0, 0.0),
)
