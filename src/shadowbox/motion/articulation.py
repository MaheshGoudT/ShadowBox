"""Smooth native-joint exercises for a stabilized, fully articulated humanoid."""

import numpy as np


class ArticulationController:
    def __init__(self, arena, robot):
        self.arena, self.robot = arena, robot
        self.busy = False

    def start(self, targets, duration=None):
        if self.busy:
            return False
        current, limits = self.arena.full_targets(self.robot), self.arena.full_limits(self.robot)
        for joint, value in targets.items():
            if joint not in current or not np.isfinite(value) or not limits[joint][0] <= value <= limits[joint][1]:
                raise ValueError(f'Invalid articulation target: {joint}')
        self.start_pose = {joint: current[joint] for joint in targets}
        self.goal = dict(targets)
        delta = np.abs(np.array(list(self.goal.values())) - np.array(list(self.start_pose.values())))
        settings = self.arena.config.get('articulation') or {}
        speed = settings.get('velocity_limit', 1.5)
        acceleration = settings.get('acceleration_limit', 4.0)
        self.duration = max(duration or settings.get('duration', 1.6),
                            float(np.max(1.875*delta/speed) if np.any(delta) else 0),
                            float(np.sqrt(np.max(5.774*delta/acceleration)) if np.any(delta) else 0))
        self.elapsed, self.busy = 0.0, True
        return True

    def tick(self):
        if not self.busy:
            return
        self.elapsed += self.arena.model.opt.timestep
        t = min(1.0, self.elapsed/self.duration)
        blend = 10*t**3 - 15*t**4 + 6*t**5
        self.arena.set_full_targets(self.robot,
            {joint: self.start_pose[joint] + blend*(goal-self.start_pose[joint])
             for joint, goal in self.goal.items()})
        self.busy = t < 1.0


def joint_demo(arena):
    """Yield actual physics frames through waist, arm, wrist, and leg exercises.

    The planar roots support the robots throughout; this does not demonstrate
    learned locomotion, contact response, or free-base balance.
    """
    if arena.config['simulation'].get('model') != 'unitree_g1':
        raise ValueError('The joint demo requires the unitree_g1 model')
    robots = ('a', 'b')
    neutral = {robot: arena.full_targets(robot) for robot in robots}
    exercises = [dict(waist_yaw_joint=.25, waist_roll_joint=.10, waist_pitch_joint=.08),
                 dict(left_shoulder_roll_joint=.45, right_shoulder_roll_joint=-.45,
                      left_shoulder_yaw_joint=.25, right_shoulder_yaw_joint=-.25,
                      left_wrist_roll_joint=.30, right_wrist_roll_joint=-.30,
                      left_wrist_pitch_joint=.25, right_wrist_pitch_joint=.25,
                      left_wrist_yaw_joint=.35, right_wrist_yaw_joint=-.35),
                 dict(left_hip_pitch_joint=-.22, right_hip_pitch_joint=-.22,
                      left_knee_joint=.45, right_knee_joint=.45,
                      left_ankle_pitch_joint=-.23, right_ankle_pitch_joint=-.23,
                      left_hip_roll_joint=.05, right_hip_roll_joint=-.05,
                      left_ankle_roll_joint=-.05, right_ankle_roll_joint=.05)]
    controllers = {r: ArticulationController(arena, r) for r in robots}
    for exercise in [*exercises, None]:
        for robot in robots:
            controllers[robot].start(neutral[robot] if exercise is None else exercise)
        while any(controller.busy for controller in controllers.values()):
            for controller in controllers.values():
                controller.tick()
            arena.step()
            yield
