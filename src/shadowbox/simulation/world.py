"""Shared MuJoCo state and model-independent accessors."""

import mujoco
import numpy as np

from ..config import load_config
from .model import make_model, JOINTS, GUARD


class Arena:
    def __init__(self, timestep=None, config=None):
        self.config = config or load_config()
        simulation = self.config["simulation"]
        if simulation.get('model') == 'unitree_g1':
            from .unitree import make_unitree_model
            self.model = make_unitree_model(timestep or simulation['timestep'], simulation)
        else:
            self.model = make_model(timestep or simulation['timestep'], simulation)
        self.data = mujoco.MjData(self.model)
        self._joints = {r: np.array([self.model.joint(f"{r}_{j}").qposadr[0]
                                    for j in JOINTS]) for r in ("a", "b")}
        self._actuators = {r: np.array([self.model.actuator(f"{r}_{j}").id
                                       for j in JOINTS]) for r in ("a", "b")}
        self.reset()

    def reset(self) -> None:
        if self.model.nkey:
            mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        else:
            mujoco.mj_resetData(self.model, self.data)
        spacing = self.config["simulation"]["robot_spacing"] / 2
        for robot, x, yaw in (("a", -spacing, 0), ("b", spacing, np.pi)):
            pose = np.array([x, 0, yaw, *GUARD])
            self.data.qpos[self._joints[robot]] = pose
            self.set_targets(robot, pose)
        mujoco.mj_forward(self.model, self.data)
        if (self.config['simulation'].get('model') == 'unitree_g1'
                and self.config.get('stance', {}).get('enabled', False)):
            from ..motion.stance import initialize_stance
            initialize_stance(self)

    def pose(self, robot: str) -> np.ndarray:
        return self.data.qpos[self._joints[robot]].copy()

    def targets(self, robot: str) -> np.ndarray:
        return self.data.ctrl[self._actuators[robot]].copy()

    def set_targets(self, robot: str, values) -> None:
        if self.config['simulation'].get('model') == 'unitree_g1' and isinstance(values, dict):
            # For Unitree G1, set all 29 joint targets
            self.set_full_targets(robot, values)
        else:
            self.data.ctrl[self._actuators[robot]] = values

    def position(self, robot: str) -> np.ndarray:
        return self.data.body(robot).xpos.copy()

    def glove_positions(self, robot: str) -> np.ndarray:
        return np.array([self.data.geom(f"{robot}_{side}_glove").xpos.copy()
                         for side in ("left", "right")])

    def step(self, count: int = 1) -> None:
        for _ in range(count):
            mujoco.mj_step(self.model, self.data)

    @property
    def time(self) -> float:
        return float(self.data.time)


    def full_pose(self, robot):
        from .joints import full_joint_state
        return full_joint_state(self, robot)

    def full_targets(self, robot):
        from .joints import full_joint_state
        return full_joint_state(self, robot, targets=True)

    def full_velocity(self, robot):
        from .joints import full_joint_velocity
        return full_joint_velocity(self, robot)

    def full_limits(self, robot):
        from .joints import full_joint_limits
        return full_joint_limits(self, robot)

    def set_full_targets(self, robot, targets):
        from .joints import set_full_joint_targets
        set_full_joint_targets(self, robot, targets)
