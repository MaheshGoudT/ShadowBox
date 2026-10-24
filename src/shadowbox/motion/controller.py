"""Smooth, nonblocking joint setpoint trajectories."""

import math
import numpy as np

from ..simulation import Arena, GUARD
from .primitives import Primitive


class MotionController:
    def __init__(self, arena: Arena, robot: str):
        self.arena, self.robot = arena, robot
        self.config = arena.config["motion"]
        from .supermimic import SuperMimic
        self.supermimic = SuperMimic(arena.config)
        if arena.config.get('natural_motion', {}).get('enabled', False):
            from .natural import NaturalMotion
            self.natural = NaturalMotion(arena.config.get('natural_motion', {}))
        else:
            self.natural = None
        self.reset()

    def reset(self) -> None:
        self.active = Primitive.IDLE
        self.busy = False
        self.elapsed = 0.0

    def plan(self, primitive: Primitive = None, motion_name: str = None) -> np.ndarray:
        if motion_name is not None:
            return self.supermimic.get_motion_targets(self.robot, motion_name)
        target = self.arena.targets(self.robot)
        target[3:] = GUARD
        yaw = self.arena.pose(self.robot)[2]
        forward = np.array([math.cos(yaw), math.sin(yaw)])
        left = np.array([-forward[1], forward[0]])
        if primitive == Primitive.STEP_FORWARD:
            target[:2] += self.config["step_distance"] * forward
        elif primitive == Primitive.STEP_BACKWARD:
            target[:2] -= self.config["step_distance"] * forward
        elif primitive == Primitive.SIDESTEP_LEFT:
            target[:2] += self.config["sidestep_distance"] * left
        elif primitive == Primitive.SIDESTEP_RIGHT:
            target[:2] -= self.config["sidestep_distance"] * left
        elif primitive == Primitive.TURN_TO_TARGET:
            other = "b" if self.robot == "a" else "a"
            delta = self.arena.position(other) - self.arena.position(self.robot)
            desired = math.atan2(delta[1], delta[0])
            target[2] = yaw + math.remainder(desired - yaw, 2 * math.pi)
        elif primitive in (Primitive.JAB, Primitive.CROSS):
            index = 3 if primitive == Primitive.JAB else 5
            target[index:index+2] = (-1.80, -0.08) if primitive == Primitive.JAB else (-1.4, 0.0)
        elif primitive == Primitive.BODY_GUARD:
            angles = (-0.8, -1.0) if self.arena.config['simulation']['model'] == 'unitree_g1' else (-0.5, -1.0)
            target[3:] = (*angles, *angles)
        elif primitive == Primitive.UPPERCUT:
            target[3:5] = (-1.7, -0.3)
        return target

    def start(self, primitive: Primitive, target=None, duration=None) -> bool:
        if self.busy:
            return False
        is_unitree = self.arena.config['simulation'].get('model') == 'unitree_g1'
        # Handle SuperMimic dict targets for Unitree G1
        if is_unitree and isinstance(target, dict):
            self.begin = self.arena.full_targets(self.robot)
            unknown = set(target) - set(self.begin)
            if unknown:
                raise ValueError(f"Unknown {self.robot} reference joints: {sorted(unknown)}")
            self.goal = {**self.begin, **target}
            # Preserve the model's full native joint order for both endpoints.
            self.begin_arr = np.array(list(self.begin.values()))
            self.goal_arr = np.array(list(self.goal.values()))
            self.waypoints = [self.begin_arr, self.goal_arr]
            self.segments = []
            if self.natural is not None:
                traj = self.natural.smooth_blend(self.begin_arr, self.goal_arr, duration or self.config["minimum_duration"])
                self.segments.append((traj, float(len(traj)) * 0.02))
            else:
                # For Unitree G1 without natural motion, use per-joint limits
                delta = np.abs(self.goal_arr - self.begin_arr)
                # Use joint-specific limits from the model
                joint_times = []
                for i, joint_name in enumerate(self.arena.full_targets(self.robot).keys()):
                    mujoco_name = joint_name.replace('_pitch_joint', '').replace('_joint', '')
                    if mujoco_name in ('left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow'):
                        mujoco_name = f'{self.robot}_{mujoco_name}'
                    else:
                        mujoco_name = f'{self.robot}_{joint_name}'
                    joint = self.arena.model.joint(mujoco_name)
                    limits = self.arena.model.jnt_range[joint.id]
                    # Calculate time based on joint limits
                    joint_delta = delta[i]
                    joint_speed = (limits[1] - limits[0]) / 0.1  # Assume 0.1s for full range
                    joint_accel = (limits[1] - limits[0]) / 0.05  # Assume 0.05s for full range
                    time_speed = 1.875 * joint_delta / joint_speed if joint_speed > 0 else 0
                    time_accel = np.sqrt(5.774 * joint_delta / joint_accel) if joint_accel > 0 else 0
                    joint_times.append(max(duration or self.config["minimum_duration"], time_speed, time_accel))
                seconds = max(joint_times)
                self.segments.append((self.begin_arr, self.goal_arr, seconds))
            self.duration = sum(seg[1] if len(seg) == 2 else seg[2] for seg in self.segments)
            self.elapsed = 0.0
            self.active, self.busy = primitive, True
            return True
        else:
            # Original capsule model handling
            self.begin = self.arena.targets(self.robot)
            self.goal = self.plan(primitive) if target is None else np.asarray(target).copy()
            waypoints = [self.begin, self.goal]
            if primitive == Primitive.CROSS:
                lowered = self.goal.copy()
                lowered[5:7] = (-0.25, -2.15)  # Lower the hand before extending below high guard.
                waypoints.insert(1, lowered)
            elif primitive == Primitive.UPPERCUT:
                loaded = self.goal.copy()
                loaded[3:5] = (-0.5, -1.5)
                waypoints.insert(1, loaded)
            speed = np.array(self.config["velocity_limits"])
            acceleration = np.array(self.config["acceleration_limits"])
            self.segments = []
            for begin, goal in zip(waypoints, waypoints[1:]):
                delta = np.abs(goal - begin)
                # Max first/second derivatives of the quintic blend are 1.875 and 5.774.
                minimum = duration or self.config["minimum_duration"]
                if (is_unitree and primitive in (Primitive.STEP_FORWARD, Primitive.STEP_BACKWARD,
                                                 Primitive.SIDESTEP_LEFT, Primitive.SIDESTEP_RIGHT,
                                                 Primitive.TURN_TO_TARGET)
                        and self.arena.config.get("footwork", {}).get("enabled")):
                    minimum = max(minimum, self.arena.config["footwork"].get("minimum_duration", 1.0))
                seconds = max(minimum,
                              float(np.max(1.875 * delta / speed)),
                              float(np.sqrt(np.max(5.774 * delta / acceleration))))
                self.segments.append((begin, goal, seconds))
            self.duration = sum(segment[2] for segment in self.segments)
            self.elapsed = 0.0
            self.active, self.busy = primitive, True
            return True

    def tick(self) -> None:
        from ..simulation.joints import _entries
        if not self.busy:
            return
        self.elapsed += self.arena.model.opt.timestep
        remaining = self.elapsed
        is_unitree = self.arena.config['simulation'].get('model') == 'unitree_g1'
        # Pre-compute joint limits for Unitree G1
        joint_limits = {}
        if is_unitree:
            for joint_name in self.arena.full_targets(self.robot).keys():
                mujoco_name = joint_name.replace('_pitch_joint', '').replace('_joint', '')
                if mujoco_name in ('left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow'):
                    mujoco_name = f'{self.robot}_{mujoco_name}'
                else:
                    mujoco_name = f'{self.robot}_{joint_name}'
                joint = self.arena.model.joint(mujoco_name)
                joint_limits[joint_name] = self.arena.model.jnt_range[joint.id]
        for seg in self.segments:
            if len(seg) == 2:
                traj, seg_duration = seg
                if remaining > seg_duration:
                    remaining -= seg_duration
                    continue
                frame = min(int(remaining / seg_duration * len(traj)), len(traj) - 1)
                target = traj[frame]
            else:
                begin, goal, seconds = seg
                if remaining > seconds:
                    remaining -= seconds
                    continue
                t = min(1.0, remaining / seconds)
                blend = 10*t**3 - 15*t**4 + 6*t**5
                target = begin + blend * (goal - begin)
            # For Unitree G1, clamp targets to joint limits
            if is_unitree:
                if isinstance(target, np.ndarray):
                    # Only clamp if target has 29 elements (Unitree G1)
                    if len(target) == 29:
                        clamped = {}
                        for i, joint_name in enumerate(self.arena.full_targets(self.robot).keys()):
                            # Get the offset for this joint
                            entries = [(n, j, a, o) for n, j, a, o in _entries(self.arena, self.robot) if n == joint_name]
                            if entries:
                                offset = entries[0][3]
                                internal = float(target[i]) - offset
                                # Clamp the internal value
                                clamped_internal = np.clip(internal, joint_limits[joint_name][0], joint_limits[joint_name][1])
                                # Convert back to target space
                                clamped[joint_name] = clamped_internal + offset
                        target = clamped
                elif isinstance(target, dict):
                    # Clamp dict values to joint limits, accounting for offset
                    for joint_name, value in target.items():
                        if joint_name in joint_limits:
                            # Get the offset for this joint
                            entries = [(n, j, a, o) for n, j, a, o in _entries(self.arena, self.robot) if n == joint_name]
                            if entries:
                                offset = entries[0][3]
                                internal = float(value) - offset
                                # Clamp the internal value
                                clamped_internal = np.clip(internal, joint_limits[joint_name][0], joint_limits[joint_name][1])
                                # Convert back to target space
                                target[joint_name] = clamped_internal + offset
            self.arena.set_targets(self.robot, target)
            break
        if self.elapsed >= self.duration:
            # Clamp goal to joint limits for Unitree G1, accounting for offset
            if is_unitree and isinstance(self.goal, dict):
                for joint_name in joint_limits:
                    if joint_name in self.goal:
                        # Get the offset for this joint
                        entries = [(n, j, a, o) for n, j, a, o in _entries(self.arena, self.robot) if n == joint_name]
                        if entries:
                            offset = entries[0][3]
                            internal = float(self.goal[joint_name]) - offset
                            # Clamp the internal value
                            clamped_internal = np.clip(internal, joint_limits[joint_name][0], joint_limits[joint_name][1])
                            # Convert back to target space
                            self.goal[joint_name] = clamped_internal + offset
            self.arena.set_targets(self.robot, self.goal)
            self.busy = False

    def run(self, primitive: Primitive, duration=None) -> None:
        """Headless convenience; the application uses start/tick for concurrent robots."""
        self.start(primitive, duration=duration)
        while self.busy:
            self.tick()
            self.arena.step()
        self.active = Primitive.IDLE

    def cancel(self, seconds=None) -> None:
        """Hold the last continuous setpoint so the position servos settle."""
        self.busy = False
        self.active = Primitive.IDLE
        self.goal = self.arena.targets(self.robot).copy()
        self.segments = []
        self.elapsed = 0.0
        self.duration = 0.0
        """Hold the last continuous setpoint so the position servos settle."""
        self.busy = False
        self.active = Primitive.IDLE
