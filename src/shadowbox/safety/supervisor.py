"""Conservative virtual-sparring root-path and state supervision."""

import numpy as np
from ..motion import Primitive
from ..simulation import Arena

PUNCHES = (Primitive.JAB, Primitive.CROSS, Primitive.UPPERCUT)


class Supervisor:
    """Checks the entire proposed root path and joint targets before execution."""

    def __init__(self, arena: Arena, defense_only: bool = False):
        self.arena = arena
        self.defense_only = defense_only
        self.config = arena.config["safety"]
        self.boundary = arena.config["simulation"]["arena_half_size"]
        self.stopped = False
        self.reason = "ready"
        self.rejections = 0
        self.reserved = {r: arena.targets(r)[:2] for r in ("a", "b")}

    def approve(self, robot: str, primitive: Primitive, target) -> bool:
        reason = "accepted"
        is_unitree = self.arena.config['simulation'].get('model') == 'unitree_g1'
        if self.stopped:
            reason = "emergency stop"
        elif robot == "b" and self.defense_only and primitive in (*PUNCHES, 'jab', 'cross', 'uppercut', 'hook_left', 'hook_right'):
            reason = "B is defense-only"
        elif is_unitree and isinstance(target, dict):
            limits = self.arena.full_limits(robot)
            if not target or any(name not in limits for name in target):
                reason = "unknown joint target"
            elif any(not np.isfinite(value) for value in target.values()):
                reason = "nonfinite target"
            elif any(not limits[name][0] <= value <= limits[name][1]
                     for name, value in target.items()):
                reason = "joint target limit"
            else:
                start_pose = self.arena.full_targets(robot)
                is_safe, swept_reason = self.check_swept_limb_safety(robot, start_pose, target)
                if not is_safe:
                    reason = swept_reason
                else:
                    self.reason = "accepted"
                    self.reserved[robot] = self.arena.position(robot)[:2].copy()
                    return True
        elif not np.isfinite(target).all():
            reason = "nonfinite target"
        else:
            joint_ids = [self.arena.model.joint(f"{robot}_{joint}").id for joint in
                         ("x", "y", "yaw", "left_shoulder", "left_elbow", "right_shoulder", "right_elbow")]
            limits = self.arena.model.jnt_range[joint_ids]
            limited = self.arena.model.jnt_limited[joint_ids].astype(bool)
            if np.any(target[limited] < limits[limited, 0]) or np.any(target[limited] > limits[limited, 1]):
                reason = "joint target limit"
            else:
                start = self.arena.position(robot)[:2]
                opponent = "b" if robot == "a" else "a"
                other = self.arena.position(opponent)[:2]
                path = np.linspace(start, target[:2], self.config["path_samples"])
                reserved = np.linspace(other, self.reserved[opponent], self.config["path_samples"])
                if np.max(np.abs(path)) > self.boundary - self.config["boundary_margin"]:
                    reason = "arena boundary"
                else:
                    baseline = min(self.config["min_separation"], float(np.min(np.linalg.norm(start - reserved, axis=1))))
                    if np.min(np.linalg.norm(path[:, None] - reserved[None, :], axis=2)) < baseline - 1e-6:
                        reason = "torso separation"
        self.reason = reason
        if reason != "accepted":
            self.rejections += 1
            return False
        self.reserved[robot] = target[:2].copy()
        return True

    def check_swept_limb_safety(self, robot: str, start_pose: dict, end_pose: dict) -> tuple[bool, str]:
        """Check if swept limb volumes collide with opponent or boundaries."""
        from ..simulation.joints import _entries
        reason = "swept_clear"
        entries = list(_entries(self.arena, robot))
        limb_joints = ['left_shoulder_pitch_joint', 'left_elbow_joint', 'right_shoulder_pitch_joint', 'right_elbow_joint']

        for joint_name in limb_joints:
            # Get joint entries
            joint_entries = [e for e in entries if e[0] == joint_name]
            if not joint_entries:
                continue
            _, joint, actuator, offset = joint_entries[0]

            # Get start and end values
            start_val = start_pose.get(joint_name, 0)
            end_val = end_pose.get(joint_name, 0)

            # Check if joint moves
            if abs(end_val - start_val) < 1e-6:
                continue

            # Sample the trajectory
            samples = self.config.get("path_samples", 25)
            trajectory = np.linspace(start_val - offset, end_val - offset, samples)

            # Check each sample against joint limits
            joint_id = joint.id if hasattr(joint, 'id') else joint
            limits = self.arena.model.jnt_range[joint_id]
            if np.any(trajectory < limits[0]) or np.any(trajectory > limits[1]):
                reason = f"swept_limb_limit_{joint_name}"
                return False, reason

        return True, reason

    def monitor(self) -> bool:
        positions = np.array([self.arena.position(r)[:2] for r in ("a", "b")])
        if not np.isfinite(self.arena.data.qpos).all():
            self.stopped, self.reason = True, "nonfinite state"
        elif np.max(np.abs(positions)) > self.boundary - self.config["boundary_override_margin"]:
            self.stopped, self.reason = True, "boundary override"
        elif np.linalg.norm(positions[0] - positions[1]) < self.config["monitor_separation"]:
            self.stopped, self.reason = True, "separation override"
        return not self.stopped
