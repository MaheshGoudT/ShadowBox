"""Shared tactical actions and legality checks for every learner."""

import math

import numpy as np

from ..motion import Primitive


ACTIONS = (Primitive.GUARD, Primitive.TURN_TO_TARGET, Primitive.STEP_FORWARD,
           Primitive.STEP_BACKWARD, Primitive.SIDESTEP_LEFT, Primitive.SIDESTEP_RIGHT,
           Primitive.JAB, Primitive.CROSS, Primitive.BODY_GUARD, Primitive.UPPERCUT)


def allowed_actions(observation, parameters, motion, minimum_separation, rules,
                    defense_only=False, proactive_guard=False):
    if observation.stopped:
        return [0]
    delta = observation.opponent_position[:2] - observation.position[:2]
    error = abs(math.remainder(math.atan2(delta[1], delta[0]) - observation.yaw,
                               2 * math.pi))
    if error > .18:
        return [1]
    allowed = [0]
    forward = np.array([math.cos(observation.yaw), math.sin(observation.yaw)])
    left = np.array([-forward[1], forward[0]])
    bound = np.max(np.abs(observation.position[:2])) + observation.boundary_space
    # The controller executes the full step; masking a shorter imagined step
    # admits unsafe approaches, especially when both boxers advance together.
    moves = {2: motion['step_distance'] * forward, 3: -motion['step_distance'] * forward,
             4: motion['sidestep_distance'] * left,
             5: -motion['sidestep_distance'] * left}
    for action, displacement in moves.items():
        target = observation.position[:2] + displacement
        clearance = minimum_separation + (motion['step_distance'] if action == 2 else 0)
        if (np.linalg.norm(displacement) > 1e-6 and np.max(np.abs(target)) <= bound
                and np.linalg.norm(target - observation.opponent_position[:2]) >= clearance):
            allowed.append(action)
    scale = rules['distance_scale']
    if not defense_only:
        ranges = rules.get('punch_ranges', {'jab': 1.1 * scale, 'cross': 1.1 * scale})
        if .68 * scale <= observation.distance <= ranges['jab']:
            allowed.append(6)
        if .68 * scale <= observation.distance <= ranges['cross']:
            allowed.append(7)
        if minimum_separation <= observation.distance <= ranges.get('uppercut', min(ranges['cross'], .59)):
            allowed.append(9)
    if (observation.distance < .95 * scale
            and (proactive_guard or (getattr(observation, 'incoming_body', False)
                                     and observation.incoming_speed > .25 * scale))):
        allowed.append(8)
    return allowed
