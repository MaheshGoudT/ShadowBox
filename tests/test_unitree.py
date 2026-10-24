"""Real Unitree geometry, native articulation, and compatibility checks."""

import math
import numpy as np
import pytest

from shadowbox.config import BASE_CONFIG, PROJECT_ROOT, load_config
from shadowbox.boxing import Match
from shadowbox.simulation import Arena
from shadowbox.motion import MotionController, Primitive
from shadowbox.motion.articulation import ArticulationController, joint_demo


@pytest.fixture
def arena():
    config = load_config(BASE_CONFIG)
    config['simulation'].update(model='unitree_g1', translation_damping=320)
    return Arena(config=config)


def test_real_model_has_two_independent_29_joint_trees(arena):
    assert arena.model.njnt == arena.model.nu == 64
    assert arena.model.na == len(arena.data.act) == 0
    assert len(arena.data.ctrl) == arena.model.nu
    assert arena.model.actuator('a_x').id != arena.model.actuator('b_x').id
    assert len(arena.full_pose('a')) == len(arena.full_pose('b')) == 29
    assert arena.model.nmesh >= 30
    np.testing.assert_allclose(arena.position('a')[:2], [-1.25, 0], atol=1e-12)
    np.testing.assert_allclose(arena.position('b')[:2], [1.25, 0], atol=1e-12)
    assert arena.data.geom('a_head').xpos[2] > arena.data.geom('a_torso').xpos[2]


def test_position_actuator_target_moves_g1_root(arena):
    start = arena.position('a')[0]
    other = arena.position('b').copy()
    target = arena.targets('a')
    target[0] += .1
    arena.set_targets('a', target)
    arena.step(100)
    assert arena.position('a')[0] > start + .05
    np.testing.assert_allclose(arena.position('a')[0], arena.pose('a')[0], atol=.002)
    np.testing.assert_allclose(arena.position('b'), other, atol=.01)


def test_native_elbow_api_hides_legacy_coordinate_offset(arena):
    limits = arena.full_limits('a')['left_elbow_joint']
    np.testing.assert_allclose(limits, [-1.0472, 2.0944], atol=1e-10)
    arena.set_full_targets('a', {'left_elbow_joint': 1.0})
    assert arena.targets('a')[4] == pytest.approx(1.0-math.pi/2)
    assert arena.full_targets('a')['left_elbow_joint'] == pytest.approx(1.0)
    arena.step(150)
    assert arena.full_pose('a')['left_elbow_joint'] == pytest.approx(1.0, abs=.02)


def test_native_target_validation_is_atomic(arena):
    before = arena.data.ctrl.copy()
    with pytest.raises(ValueError):
        arena.set_full_targets('a', {'left_knee_joint': .5, 'left_wrist_pitch_joint': 9})
    np.testing.assert_array_equal(arena.data.ctrl, before)


def test_boxing_primitives_move_actual_g1_hands(arena):
    controller = MotionController(arena, 'a')
    guard = arena.glove_positions('a').copy()
    controller.run(Primitive.JAB)
    assert np.linalg.norm(arena.glove_positions('a')[0]-guard[0]) > .20
    controller.run(Primitive.GUARD)
    controller.run(Primitive.CROSS)
    assert arena.glove_positions('a')[1, 2] < guard[1, 2]-.20
    assert np.isfinite(arena.data.qpos).all()


def test_joint_demo_exercises_waist_wrists_and_legs_with_stable_roots(arena):
    initial = arena.full_pose('a')
    moved = {joint: 0.0 for joint in initial}
    root = arena.position('a')[:2].copy()
    for _ in joint_demo(arena):
        for joint, value in arena.full_pose('a').items():
            moved[joint] = max(moved[joint], abs(value-initial[joint]))
        assert np.isfinite(arena.data.qpos).all()
    assert sum(amount > .03 for amount in moved.values()) >= 20
    for joint in ('waist_yaw_joint', 'left_wrist_pitch_joint', 'left_knee_joint', 'right_ankle_roll_joint'):
        assert moved[joint] > .03
    assert np.linalg.norm(arena.position('a')[:2]-root) < .03


def test_reset_restores_full_neutral_pose(arena):
    initial = arena.data.qpos.copy()
    for _ in joint_demo(arena):
        pass
    arena.reset()
    np.testing.assert_array_equal(arena.data.qpos, initial)
    for joint, value in arena.full_targets('a').items():
        lo, hi = arena.full_limits('a')[joint]
        assert lo <= value <= hi


def test_active_articulation_cannot_be_overwritten(arena):
    controller = ArticulationController(arena, 'a')
    assert controller.start({'waist_yaw_joint': .2})
    assert not controller.start({'waist_yaw_joint': -.2})


def test_unitree_boxers_approach_and_score_with_normal_profile():
    match = Match(config=load_config(PROJECT_ROOT / 'configs/unitree.yaml'),
                  rounds=1, round_seconds=15)
    for _ in range(1501):
        match.step()
    actions = [event.get('action') for event in match.events if event['event'] == 'action']
    assert 'step_forward' in actions
    assert match.stats['a']['touches'] + match.stats['b']['touches'] > 0
    assert not match.supervisor.stopped
