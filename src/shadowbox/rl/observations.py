"""Policy-facing boxing state, shared by training and inference."""

import math

import numpy as np


OBSERVATION_SIZE = 12


def encode_observation(match, robot):
    state = match.observe(robot)
    delta = state.opponent_position[:2] - state.position[:2]
    c, s = math.cos(state.yaw), math.sin(state.yaw)
    forward, left = np.array([c, s]), np.array([-s, c])
    relative = state.relative_velocity[:2]
    other = 'b' if robot == 'a' else 'a'
    gloves = match.arena.glove_positions(other)
    head = match.arena.data.geom(f'{robot}_head').xpos[2]
    return np.asarray([np.dot(delta, forward), np.dot(delta, left),
                       state.distance, np.dot(relative, forward), np.dot(relative, left),
                       state.incoming_speed, float(state.incoming_body),
                       state.boundary_space, gloves[0, 2] - head, gloves[1, 2] - head,
                       match.stats[robot]['touches'] - match.stats[other]['touches'],
                       float(state.stopped)], dtype=np.float32)
