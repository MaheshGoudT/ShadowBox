"""Goal fixtures preserve support targets and finish finite attack cycles."""
import numpy as np
import pytest
from shadowbox.playground.skill_goals import BoxingGoals


def fixture(width=.1):
    bodies = ['torso_link', 'right_ankle_roll_link', 'left_ankle_roll_link',
              'left_rubber_hand', 'right_rubber_hand']
    metadata = {'body_names': bodies, 'masked_mimic': {
        'conditionable_bodies': bodies, 'num_masked_future_steps': 5},
        'policy_inputs': [dict(provider='masked_mimic', kind='anchor_rotation_mask',
            key='masked_mimic.anchor_rotation_mask', shape=[1, 5, 2], dtype='bool')]}
    goals = BoxingGoals(metadata, 2, defender_guard_width=width)
    positions = np.zeros((2, 5, 3), dtype=np.float32)
    rotations = np.zeros((2, 5, 4), dtype=np.float32)
    rotations[..., 3] = 1
    goals.reset(positions, rotations)
    return goals, positions, rotations


def test_attacks_recover_and_primitive_changes_do_not_rebase_support():
    goals, positions, rotations = fixture()
    opponent = np.array([[.6, 0, 0], [-.6, 0, 0]])
    guard = goals.compile([0, 0], positions, rotations, opponent)['masked_mimic.ref_pos']
    attack = goals.compile([6, 7], positions, rotations, opponent)['masked_mimic.ref_pos']
    assert attack[0, 2, 3, 0] > guard[0, 2, 3, 0]
    assert attack[1, 3, 4, 2] < guard[1, 3, 4, 2]
    goals.advance(1.2)
    recovered = goals.compile([6, 7], positions+5, rotations, opponent)['masked_mimic.ref_pos']
    np.testing.assert_allclose(recovered, guard)
    changed = goals.compile([8, 0], positions+5, rotations, opponent)['masked_mimic.ref_pos']
    np.testing.assert_array_equal(changed[:, :, :3], guard[:, :, :3])


def test_exposed_fixture_changes_only_defender_hands():
    normal, pos, rot = fixture()
    exposed, _, _ = fixture(.45)
    first = normal.compile([0, 0], pos, rot, pos[:, 0])['masked_mimic.ref_pos']
    second = exposed.compile([0, 0], pos, rot, pos[:, 0])['masked_mimic.ref_pos']
    np.testing.assert_array_equal(first[1], second[1])
    np.testing.assert_array_equal(first[0, :, :3], second[0, :, :3])
    assert second[0, 0, 3, 1] == pytest.approx(.45)
    with pytest.raises(ValueError, match='integer'):
        normal.compile([6.5, 0], pos, rot, pos[:, 0])
    with pytest.raises(ValueError, match='only'):
        normal.compile([9, 0], pos, rot, pos[:, 0])


def test_bad_geometry_and_time_fail_before_mutating_skill_clock():
    goals, pos, rot = fixture()
    goals.advance(.2)
    before = goals.clock.copy()
    with pytest.raises(ValueError, match='anchor'):
        goals.compile([6, 7], pos, rot, np.full((2, 3), np.nan))
    np.testing.assert_array_equal(goals.clock, before)
    with pytest.raises(ValueError, match='time'):
        goals.advance(-.1)
    with pytest.raises(ValueError, match='poses'):
        goals.reset(pos, np.zeros_like(rot))
    np.testing.assert_array_equal(goals.clock, before)
