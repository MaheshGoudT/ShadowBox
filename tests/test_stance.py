"""Actual G1 foot placement and non-teleporting leg target checks."""

import numpy as np
import pytest
import math

from shadowbox.config import PROJECT_ROOT, load_config
from shadowbox.boxing import Match
from shadowbox.simulation import Arena
from shadowbox.motion.stance import BoxingStance


def boxing_arena():
    return Arena(config=load_config(PROJECT_ROOT / 'configs/unitree.yaml'))


def test_real_feet_form_mirrored_stagger_with_bent_knees():
    arena = boxing_arena()
    stance = BoxingStance(arena)
    for robot in ('a', 'b'):
        root, yaw = arena.position(robot), arena.pose(robot)[2]
        forward = np.array([np.cos(yaw), np.sin(yaw)])
        feet = stance.foot_positions(robot)
        lead = np.dot(feet['left'][:2]-root[:2], forward)
        rear = np.dot(feet['right'][:2]-root[:2], forward)
        assert lead > .13 and rear < -.11
        assert lead-rear > .25
        assert all(abs(position[2]-.005) < .001 for position in feet.values())
        native = arena.full_pose(robot)
        assert len(native) == 29
        assert native['left_knee_joint'] > .5 and native['right_knee_joint'] > .5
        for side in ('left', 'right'):
            rotation = arena.data.site(f'{robot}_{side}_foot').xmat.reshape(3, 3)
            assert np.dot(rotation[:, 2], [0, 0, 1]) > .999
            heading = math.atan2(rotation[1, 0], rotation[0, 0])
            expected = yaw if side == 'left' else yaw - .12
            assert abs(math.remainder(heading - expected, 2*math.pi)) < .01


def test_ik_only_changes_scratch_and_returns_native_bounded_targets():
    arena = boxing_arena()
    stance = BoxingStance(arena)
    before_qpos, before_ctrl = arena.data.qpos.copy(), arena.data.ctrl.copy()
    feet = stance.neutral_feet('a')
    feet['left'] += [.025, 0, .02]
    targets = stance.solve('a', feet)
    np.testing.assert_array_equal(arena.data.qpos, before_qpos)
    np.testing.assert_array_equal(arena.data.ctrl, before_ctrl)
    assert max(stance.last_errors.values()) < .001
    limits = arena.full_limits('a')
    assert len(targets) == 12
    assert all(limits[name][0] <= value <= limits[name][1] for name, value in targets.items())


def test_supported_stance_settles_near_the_ground_without_side_by_side_feet():
    arena = boxing_arena()
    stance = BoxingStance(arena)
    desired = stance.neutral_feet('a')
    for _ in range(100):
        stance.apply('a', desired)
        arena.step()
    actual = stance.foot_positions('a')
    assert all(np.linalg.norm(actual[side]-desired[side]) < .02 for side in desired)
    assert actual['left'][0]-actual['right'][0] > .22
    assert all(position[2] > -.01 for position in actual.values())


def test_stance_reset_repeats_and_invalid_targets_fail():
    arena = boxing_arena()
    first = arena.data.qpos.copy()
    arena.step(50)
    arena.reset()
    np.testing.assert_array_equal(arena.data.qpos, first)
    stance = BoxingStance(arena)
    with pytest.raises(ValueError):
        stance.solve('a', {'left': [np.nan, 0, 0]})


def test_both_robots_keep_feet_near_floor_during_match():
    match = Match(config=load_config(PROJECT_ROOT / 'configs/unitree.yaml'),
                  rounds=1, round_seconds=12)
    stance = BoxingStance(match.arena)
    planted = []
    saw_swing = False
    for _ in range(1200):
        match.step()
        for robot in ('a', 'b'):
            feet = stance.foot_positions(robot)
            swinging = match.footwork[robot].swinging
            saw_swing |= swinging is not None
            grounded = [abs(foot[2]-.005) < .01 for foot in feet.values()]
            assert any(grounded)
            for side, foot in feet.items():
                if side != swinging:
                    planted.append(abs(foot[2]-.005) < .01)
    assert saw_swing and np.mean(planted) >= .95
    assert not match.supervisor.stopped


def test_step_moves_lead_then_rear_while_the_other_foot_stays_planted():
    from shadowbox.motion import Primitive

    match = Match(config=load_config(PROJECT_ROOT / 'configs/unitree.yaml'),
                  rounds=1, round_seconds=5)
    start = match.arena.position('a').copy()
    assert match._request('a', Primitive.STEP_FORWARD)
    swings = []
    for _ in range(130):
        match.controllers['a'].tick()
        match.footwork['a'].tick(match.controllers['a'])
        match.arena.step()
        footwork = match.footwork['a']
        footwork.measure()
        if footwork.swinging is not None:
            swings.append(footwork.swinging)
    assert swings[0] == 'left' and swings[-1] == 'right'
    assert match.arena.position('a')[0] > start[0] + .05
    assert footwork.max_planted_slip < .05
