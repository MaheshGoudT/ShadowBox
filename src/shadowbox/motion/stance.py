"""Staggered boxing feet solved with the real G1 leg kinematics."""

import math

import mujoco
import numpy as np

LEG_JOINTS = ('hip_pitch', 'hip_roll', 'hip_yaw', 'knee', 'ankle_pitch', 'ankle_roll')


class BoxingStance:
    """Solve joint targets; live motion still runs through the position servos.

    The planar pelvis is supported by the simulation. This is grounded
    kinematic foot placement, not a free-base balance controller.
    """

    def __init__(self, arena, config=None):
        self.arena = arena
        self.config = config or arena.config.get('stance', {})
        self.scratch = mujoco.MjData(arena.model)
        self.entries = {}
        self.last_errors = {}
        for robot in ('a', 'b'):
            for side in ('left', 'right'):
                names = [f'{side}_{joint}_joint' for joint in LEG_JOINTS]
                ids = np.array([arena.model.joint(f'{robot}_{name}').id for name in names])
                self.entries[robot, side] = (names, ids,
                    arena.model.site(f'{robot}_{side}_foot').id)

    def neutral_feet(self, robot):
        """Orthodox stance: left lead foot ahead, right rear foot behind."""
        root = self.arena.position(robot)
        yaw = self.arena.pose(robot)[2]
        forward, left = np.array([math.cos(yaw), math.sin(yaw)]), np.array([-math.sin(yaw), math.cos(yaw)])
        width = self.config.get('width', .22) / 2
        height = self.config.get('sole_height', .005)
        targets = {}
        for side, distance, lateral in (('left', self.config.get('lead_forward', .14), width),
                                        ('right', -self.config.get('rear_backward', .12), -width)):
            xy = root[:2] + distance * forward + lateral * left
            targets[side] = np.array([xy[0], xy[1], height])
        return targets

    def foot_positions(self, robot, data=None):
        data = data or self.arena.data
        return {side: data.site_xpos[self.entries[robot, side][2]].copy() for side in ('left', 'right')}

    def foot_headings(self, robot):
        yaw = self.arena.pose(robot)[2]
        return {'left': yaw, 'right': yaw + self.config.get('rear_toe_out', -.12)}

    def solve(self, robot, foot_targets, headings=None):
        """Damped least-squares IK on a scratch state; never teleport live joints."""
        model, data = self.arena.model, self.scratch
        data.qpos[:] = self.arena.data.qpos
        headings = headings or self.foot_headings(robot)
        output = {}
        for side, target in foot_targets.items():
            target = np.asarray(target, dtype=float)
            if side not in ('left', 'right') or target.shape != (3,) or not np.isfinite(target).all():
                raise ValueError('Foot targets require left/right finite XYZ positions')
            names, ids, site = self.entries[robot, side]
            addresses = model.jnt_qposadr[ids]
            velocities = model.jnt_dofadr[ids]
            limits = model.jnt_range[ids].copy()
            limits[3, 0] = max(.12, limits[3, 0])  # Keep a visible, stable knee bend.
            yaw = float(headings[side])
            if not math.isfinite(yaw):
                raise ValueError('Foot headings must be finite')
            c, s = math.cos(yaw), math.sin(yaw)
            desired_rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
            jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
            for _ in range(self.config.get('ik_iterations', 60)):
                mujoco.mj_kinematics(model, data)
                mujoco.mj_comPos(model, data)
                rotation = data.site_xmat[site].reshape(3, 3)
                relative = desired_rotation @ rotation.T
                orientation_error = .5 * np.array([relative[2, 1]-relative[1, 2],
                    relative[0, 2]-relative[2, 0], relative[1, 0]-relative[0, 1]])
                position_error = target - data.site_xpos[site]
                error = np.r_[position_error, .25 * orientation_error]
                if np.linalg.norm(error) < self.config.get('ik_tolerance', .0003):
                    break
                mujoco.mj_jacSite(model, data, jp, jr, site)
                jacobian = np.vstack((jp[:, velocities], .25 * jr[:, velocities]))
                delta = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + np.eye(6)*1e-6, error)
                data.qpos[addresses] = np.clip(data.qpos[addresses] + np.clip(delta, -.15, .15),
                                               limits[:, 0], limits[:, 1])
            mujoco.mj_kinematics(model, data)
            mujoco.mj_comPos(model, data)
            self.last_errors[side] = float(np.linalg.norm(target-data.site_xpos[site]))
            output.update(zip(names, map(float, data.qpos[addresses])))
        return output

    def apply(self, robot, foot_targets, headings=None):
        targets = self.solve(robot, foot_targets, headings)
        self.arena.set_full_targets(robot, targets)
        return targets


def initialize_stance(arena):
    """Set a reproducible initial pose during reset, before live stepping starts."""
    stance = BoxingStance(arena)
    for robot in ('a', 'b'):
        targets = stance.apply(robot, stance.neutral_feet(robot))
        for name, value in targets.items():
            joint = arena.model.joint(f'{robot}_{name}').id
            arena.data.qpos[arena.model.jnt_qposadr[joint]] = value
    mujoco.mj_forward(arena.model, arena.data)
