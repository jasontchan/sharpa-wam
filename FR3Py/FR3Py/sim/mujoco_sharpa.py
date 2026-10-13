import time

import mujoco
import mujoco.viewer
import numpy as np

from FR3Py.sim.sharpa_constants import (
    DEFAULT_GRASP_POSE,
    FINGERTIP_GEOMS,
    JOINT_NAMES,
    JOINT_ORDER,
)
from FR3Py.sim.sharpa_spec import (
    HAND_ROOT_BODY,
    NUM_ARM_ACTUATORS,
    NUM_HAND_ACTUATORS,
    build_fr3_sharpa_spec,
)


class FR3SharpaSim:
    def __init__(
        self,
        interface_type="torque",
        render=True,
        dt=0.002,
    ):
        assert interface_type in [
            "torque",
            "velocity",
        ], "The interface should be velocity or torque"
        self.interface_type = interface_type
        self.spec = build_fr3_sharpa_spec()
        self.model = self.spec.compile()
        self.simulated = True
        self.data = mujoco.MjData(self.model)
        self.dt = dt
        _render_dt = 1 / 60
        self.render_ds_ratio = max(1, _render_dt // dt)

        if render:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.render = True
            self.viewer.cam.distance = 3.0
            self.viewer.cam.azimuth = 90
            self.viewer.cam.elevation = -45
            self.viewer.cam.lookat[:] = np.array([0.0, -0.25, 0.824])
        else:
            self.viewer = None
            self.render = False

        self.model.opt.gravity[2] = -9.81
        self.model.opt.timestep = dt
        self.renderer = None
        self.render = render
        self.step_counter = 0

        self.hand_joint_names = JOINT_NAMES
        self._resolve_indices()
        self._legacy_full_m = hasattr(self.data, "qM")

        self.q0 = np.array(
            [
                0.0,
                -0.785398163,
                0.0,
                -2.35619449,
                0.0,
                1.57079632679,
                0.785398163397,
            ]
        )
        self.hand_q0 = np.array(
            [DEFAULT_GRASP_POSE[name] for name in JOINT_ORDER], dtype=np.float64
        )
        self.hand_targets = np.clip(
            self.hand_q0, self.hand_ctrlrange[:, 0], self.hand_ctrlrange[:, 1]
        )

        self.reset()
        mujoco.mj_step(self.model, self.data)
        if self.render:
            self.viewer.sync()
        self.nv = self.model.nv
        self.jacp = np.zeros((3, self.nv))
        self.jacr = np.zeros((3, self.nv))
        self.M = np.zeros((self.nv, self.nv))
        self.latest_command_stamp = time.time()
        self.actuator_tau = np.zeros(NUM_ARM_ACTUATORS)
        self.tau_ff = np.zeros(NUM_ARM_ACTUATORS)
        self.dq_des = np.zeros(NUM_ARM_ACTUATORS)

    def _resolve_indices(self):
        hand_qpos, hand_qvel, hand_ctrlrange = [], [], []
        for name in JOINT_NAMES:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            assert jid >= 0, f"Hand joint {name} is missing from the model"
            hand_qpos.append(self.model.jnt_qposadr[jid])
            hand_qvel.append(self.model.jnt_dofadr[jid])

            aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            assert aid >= 0, f"Hand actuator {name} is missing from the model"
            hand_ctrlrange.append(self.model.actuator_ctrlrange[aid])

        self.hand_qpos_idx = np.array(hand_qpos, dtype=int)
        self.hand_qvel_idx = np.array(hand_qvel, dtype=int)
        self.hand_ctrlrange = np.array(hand_ctrlrange, dtype=np.float64)
        self.hand_ctrl_slice = slice(
            NUM_ARM_ACTUATORS, NUM_ARM_ACTUATORS + NUM_HAND_ACTUATORS
        )

        self.fingertip_geom_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            for name in FINGERTIP_GEOMS
        ]
        self.palm_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, HAND_ROOT_BODY
        )
        assert self.palm_body_id >= 0, f"Body {HAND_ROOT_BODY} is missing from the model"

    def reset(self):
        self.data.qpos[:NUM_ARM_ACTUATORS] = self.q0
        self.data.qvel[:NUM_ARM_ACTUATORS] = np.zeros(NUM_ARM_ACTUATORS)
        self.data.qpos[self.hand_qpos_idx] = self.hand_q0
        self.data.qvel[self.hand_qvel_idx] = np.zeros(NUM_HAND_ACTUATORS)
        self.data.ctrl[self.hand_ctrl_slice] = self.hand_targets

        mujoco.mj_step(self.model, self.data)
        if self.render and (self.step_counter % self.render_ds_ratio) == 0:
            self.viewer.sync()

    def getJointStates(self):
        return {
            "arm": {
                "q": self.data.qpos[:NUM_ARM_ACTUATORS],
                "dq": self.data.qvel[:NUM_ARM_ACTUATORS],
                "tau_est": (
                    self.data.qfrc_constraint.squeeze()
                    + self.data.qfrc_smooth.squeeze()
                )[0:NUM_ARM_ACTUATORS],
            },
            "hand": {
                "q": self.data.qpos[self.hand_qpos_idx],
                "dq": self.data.qvel[self.hand_qvel_idx],
            },
        }

    def setCommands(self, arm_cmd, hand_cmd=None):
        arm_cmd = np.asarray(arm_cmd, dtype=np.float64)
        assert arm_cmd.shape == (NUM_ARM_ACTUATORS,), "Command shape must be (7,)"
        self.dq_des = arm_cmd
        self.tau_ff = arm_cmd

        if hand_cmd is not None:
            hand_cmd = np.asarray(hand_cmd, dtype=np.float64)
            assert hand_cmd.shape == (
                NUM_HAND_ACTUATORS,
            ), "Hand command shape must be (22,)"
            self.hand_targets = np.clip(
                hand_cmd, self.hand_ctrlrange[:, 0], self.hand_ctrlrange[:, 1]
            )

        self.latest_command_stamp = time.time()
        self.step()

    def step(self):
        if self.interface_type == "torque":
            nle = self.getDynamicsParams()["nle"]
            tau = nle[:NUM_ARM_ACTUATORS].squeeze() + self.tau_ff
            self.actuator_tau = tau
        else:
            state = self.getJointStates()["arm"]
            dq = state["dq"]
            nle = self.getDynamicsParams()["nle"]
            tau = nle[:NUM_ARM_ACTUATORS].squeeze() + 20 * (self.dq_des - dq)
            self.actuator_tau = tau

        self.data.ctrl[:NUM_ARM_ACTUATORS] = tau.squeeze()
        self.data.ctrl[self.hand_ctrl_slice] = self.hand_targets
        self.step_counter += 1
        mujoco.mj_step(self.model, self.data)
        if self.render and (self.step_counter % self.render_ds_ratio) == 0:
            self.viewer.sync()

    def getSiteJacobian(self, site_name):
        id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        assert id > 0, "The requested site could not be found"
        mujoco.mj_jacSite(self.model, self.data, self.jacp, self.jacr, id)
        return self.jacp, self.jacr

    def getDynamicsParams(self):
        if self._legacy_full_m:
            mujoco.mj_fullM(self.model, self.M, self.data.qM)
        else:
            mujoco.mj_fullM(self.model, self.data, self.M)
        nle = self.data.qfrc_bias.reshape(self.nv, 1)
        return {"M": self.M, "nle": nle}

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
