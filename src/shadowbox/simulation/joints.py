"""Native joint access alongside the seven-value boxing control interface."""

import math
import numpy as np

ALIAS_TO_NATIVE = {'left_shoulder': 'left_shoulder_pitch_joint',
                   'right_shoulder': 'right_shoulder_pitch_joint',
                   'left_elbow': 'left_elbow_joint', 'right_elbow': 'right_elbow_joint'}


def _entries(arena, robot):
    unitree = arena.config['simulation'].get('model') == 'unitree_g1'
    prefix = robot + '_'
    for index in range(arena.model.njnt):
        internal = arena.model.joint(index).name
        if not internal.startswith(prefix):
            continue
        suffix = internal[len(prefix):]
        if suffix in ('x', 'y', 'yaw'):
            continue
        external = ALIAS_TO_NATIVE.get(suffix, suffix) if unitree else suffix
        offset = math.pi/2 if unitree and suffix in ('left_elbow', 'right_elbow') else 0.0
        yield external, index, arena.model.actuator(internal).id, offset


def full_joint_state(arena, robot, targets=False):
    return {name: float((arena.data.ctrl[actuator] if targets else
                         arena.data.qpos[arena.model.jnt_qposadr[joint]]) + offset)
            for name, joint, actuator, offset in _entries(arena, robot)}


def full_joint_velocity(arena, robot):
    """Native encoder velocities; coordinate offsets apply only to position."""
    return {name: float(arena.data.qvel[arena.model.jnt_dofadr[joint]])
            for name, joint, _, _ in _entries(arena, robot)}


def full_joint_limits(arena, robot):
    return {name: (arena.model.jnt_range[joint] + offset).tolist()
            for name, joint, actuator, offset in _entries(arena, robot)}


def set_full_joint_targets(arena, robot, targets):
    """Validate the complete request before updating any native joint actuator."""
    entries = {name: (joint, actuator, offset) for name, joint, actuator, offset in _entries(arena, robot)}
    updates = {}
    for name, value in targets.items():
        if name not in entries:
            raise ValueError(f'Unknown {robot} joint: {name}')
        joint, actuator, offset = entries[name]
        internal = float(value) - offset
        lo, hi = arena.model.jnt_range[joint]
        if not np.isfinite(internal) or not lo <= internal <= hi:
            raise ValueError(f'{name} target is outside its native joint limits')
        updates[actuator] = internal
    for actuator, value in updates.items():
        arena.data.ctrl[actuator] = value
