"""MorphMimic native/canonical bridge using its pinned NumPy observation code.

This exports measured diagnostic data from ShadowBox. It does not run a trained
world model or replace the existing gait/controller.
"""

from functools import lru_cache
import importlib.util
import json
from pathlib import Path
import sys

import mujoco
import numpy as np

from ..integrations.morphmimic import load_settings, require_checkout, ROBOTS


@lru_cache(maxsize=4)
def _observation_module(repository):
    path = Path(repository) / 'deploy/sim2sim/obs.py'
    name = '_shadowbox_morphmimic_obs_' + str(abs(hash(str(path))))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclass resolves its module during import
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


class MorphMimicProfile:
    def __init__(self, robot='g1', settings=None, deployment=None):
        self.settings = settings or load_settings()
        require_checkout(self.settings)
        if robot not in ROBOTS:
            raise ValueError(f'No verified MorphMimic profile for {robot}')
        self.obs = _observation_module(str(self.settings['repository']))
        if deployment is None:
            path = self.settings['repository'] / 'deploy/sim2sim/robots' / f'{robot}.json'
            constants = json.loads(path.read_text())
            self.morphology = None
        else:
            document = json.loads((Path(deployment) / f'{robot}.json').read_text())
            constants = document['robot']
            descriptor = document['inputs'].get('morph')
            self.morphology = None if descriptor is None else np.asarray(descriptor, np.float32)
            if self.morphology is not None and (self.morphology.shape != (125,)
                                                or not np.isfinite(self.morphology).all()):
                raise ValueError('Exported morphology descriptor must have 125 finite values')
        self.robot = self.obs.Robot.from_dict(constants)
        self.canonical_names = tuple(constants['canonical_joint_names'])
        n = self.robot.num_native
        if (self.robot.name != robot or len(self.canonical_names) != 29
                or len(set(self.canonical_names)) != 29
                or len(set(self.robot.native_joint_names)) != n
                or self.robot.perm.shape != (29,) or len(set(self.robot.perm)) != 29
                or np.any(self.robot.perm < 0) or np.any(self.robot.perm >= n)
                or self.robot.sign.shape != (29,) or not np.isin(self.robot.sign, (-1, 1)).all()
                or self.robot.offset.shape != (29,) or not np.isfinite(self.robot.offset).all()
                or self.robot.default_joint_pos.shape != (n,)
                or not np.isfinite(self.robot.default_joint_pos).all()
                or self.robot.action_scale.shape != (n,)
                or not np.isfinite(self.robot.action_scale).all()
                or np.any(self.robot.action_scale <= 0)):
            raise ValueError('Invalid MorphMimic native/canonical joint contract')

    def manifest(self):
        return dict(robot=self.robot.name, native_joint_names=list(self.robot.native_joint_names),
                    canonical_joint_names=list(self.canonical_names),
                    imu_body=self.robot.imu_body, control_dt=self.robot.control_dt,
                    proprioception_size=93, descriptor_available=self.morphology is not None,
                    upstream_revision=self.settings['revision'],
                    shadowbox_joint_mapping=self.robot.name == 'g1')

    def native_targets(self, canonical_action):
        action = np.asarray(canonical_action, dtype=float)
        if action.shape != (29,) or not np.isfinite(action).all():
            raise ValueError('Canonical action needs 29 finite values')
        return dict(zip(self.robot.native_joint_names, self.robot.targets(action).tolist()))

    def capture(self, arena, side):
        """Read encoders and the correct IMU frame without changing simulator state."""
        if (arena.config['simulation'].get('model') != 'unitree_g1'
                or self.robot.name != 'g1' or side not in ('a', 'b')):
            raise ValueError('ShadowBox Arena currently provides only G1 measurements; X2/T2 use sim2sim')
        qmap, dqmap, targets = (arena.full_pose(side), arena.full_velocity(side),
                                arena.full_targets(side))
        names = self.robot.native_joint_names
        if set(names) != set(qmap):
            raise ValueError('Arena joints do not match the exported robot profile')
        q = np.array([qmap[name] for name in names])
        dq = np.array([dqmap[name] for name in names])
        target = np.array([targets[name] for name in names])
        action = self.robot.to_canonical_delta((target - self.robot.default_joint_pos)
                                                / self.robot.action_scale)
        imu_name = side if self.robot.imu_body == 'pelvis' else f'{side}_{self.robot.imu_body}'
        imu = arena.model.body(imu_name).id
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(arena.model, arena.data, mujoco.mjtObj.mjOBJ_XBODY,
                                imu, velocity, 1)
        quat = arena.data.xquat[imu].copy()
        proprio = self.obs.proprio_frame(self.robot, q, dq, quat, velocity[:3], action)
        if not np.isfinite(proprio).all():
            raise ValueError('Nonfinite MorphMimic measurement')
        return dict(canonical_q=self.robot.to_canonical(q),
                    canonical_dq=self.robot.to_canonical_delta(dq),
                    target_action=action, proprioception=proprio,
                    imu_quat_wxyz=quat, imu_gyro=velocity[:3].copy())
