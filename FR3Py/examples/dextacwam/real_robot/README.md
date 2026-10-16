# DexTacWAM on the real FR3 + Sharpa

The policy needs ~15 GB of GPU memory, so it runs on the L40 server, and this
client runs on the robot PC. The two talk over an SSH tunnel:

```
robot PC (FR3Py devcontainer)                         L40 server
real_robot_client.py ── 127.0.0.1:5008 ══ ssh -L ══> 127.0.0.1:5008  serve_policy.sh
  camera, tactile, joint state  ──────────────────>   DexTacWAM policy
  arm (100 Hz velocity), hand   <──────────────────   54-step chunk: palm poses + hand targets
```

The server binds to `127.0.0.1` only. The wire protocol is unauthenticated
pickle, so never expose the port directly.

## Run it

**1. L40 server** (DexTacWAM venv; checkpoint under `outputs/`):

```bash
cd sharpa-wam/DexTacWAM && STEP=4000 bash fr3_integration/serve_policy.sh
```

Wait for `listening on 127.0.0.1:5008 — ready`.

**2. Robot PC host**: open the tunnel and leave it running:

```bash
bash FR3Py/examples/dextacwam/real_robot/ssh_tunnel.sh <user>@<l40-host>
```

**3. Robot PC devcontainer**: start the `fr3_joint_velocity` bridge (not the
torque one), then work through the modes in order:

```bash
cd /home/FR3Py/examples/dextacwam/real_robot
python3 real_robot_client.py --mode check     # state, palm pose, per-finger force, one camera frame
python3 real_robot_client.py --mode align     # camera vs. the training camera view
python3 real_robot_client.py --mode shadow    # policy runs, robot does NOT move
python3 real_robot_client.py --mode execute   # moves the robot (asks for confirmation)
```

Logs, videos and images go to `logs/<timestamp>/`.

| Mode | Moves robot | What to look at |
|---|---|---|
| `check` | no | Press each fingertip and confirm the force shows up on the right finger. If not, fix `FINGER_CHANNELS`. |
| `align` | no | `align_*.png` shows real, sim and a 50/50 blend. Move the camera until the robot and table overlap the sim view. The training camera was at (1.15, −0.45, 0.80) m in the robot base frame, aimed at (0.45, 0, 0.30). |
| `shadow` | no | `shadow.mp4` shows the live camera next to the sim robot posed where the policy wants to go. `shadow_log.npz` holds every predicted chunk. |
| `execute` | **yes** | Start with defaults (half speed, 300 actions). Keep the e-stop in reach. |

`--backend sim` runs any mode against a MuJoCo stand-in with the same
interfaces (velocity-commanded arm, hand, camera), which is useful for
checking the full pipeline without hardware:

```bash
python3 real_robot_client.py --mode execute --backend sim --yes
```

## Scene the policy expects

It was trained in sim, so the real scene should match it:

- table top 0.20 m above the robot base;
- a 2.4 cm bar resting across two posts, 0.33 m above the base;
- bar at x 0.45–0.55 m, y ±0.06 m;
- the arm starts at FR3Py's home pose (`execute` offers to move there slowly).

## Safety in `execute`

- **Arm:** a 100 Hz thread streams joint velocities toward the IK target and
  keeps actively holding position while the policy is queried. This matters
  because the velocity bridge keeps executing a stale command for up to 1 s
  (`control_lag < 0.1*1e7` µs in `fr3_joint_velocity.cpp`; 0.1 s was probably
  intended).
- **Limits:** palm targets are clamped to a workspace box and to 8 mm per
  action. Joints are capped at 0.5 rad/s and kept 0.05 rad inside their
  limits. Hand targets are limited to 0.08 rad per action. All limits are
  constants at the top of `real_robot_client.py`.
- **Faults:** lost arm state → zero velocity and stop. Ctrl-C or SIGTERM →
  hold the current position, then stop.

## Known gaps for this sim-trained policy

- **Tactile:** `--tactile-source synthetic` (default) turns the measured force
  on each finger into the same blob images the policy saw in sim. `raw` and
  `deform` send the real sensor images; use those for policies trained on
  real data.
- **Camera images** are real, but the policy only ever saw MuJoCo renders.
  This is the main sim-to-real gap to measure in `shadow` mode.
- **Unverified:** the tactile channel → finger mapping (`FINGER_CHANNELS`)
  and the Sharpa mount transform in FR3Py's sim model, which defines the palm
  pose.
