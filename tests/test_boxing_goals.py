"""Goal fixtures preserve support targets and finish finite attack cycles."""
import numpy as np
import pytest
from shadowbox.playground.skill_goals import BoxingGoals


def fixture(width=.1, body_attack_arc_lateral=0., approach_speed_a=.27,
            approach_speed_b=.27, max_punch_reach_m=.48):
    bodies = ['torso_link', 'right_ankle_roll_link', 'left_ankle_roll_link',
              'left_rubber_hand', 'right_rubber_hand', 'head',
              'left_shoulder_pitch_link', 'right_shoulder_pitch_link']
    metadata = {'body_names': bodies, 'masked_mimic': {
        'conditionable_bodies': bodies[:5], 'num_masked_future_steps': 5},
        'policy_inputs': [dict(provider='masked_mimic', kind='anchor_rotation_mask',
            key='masked_mimic.anchor_rotation_mask', shape=[1, 5, 2], dtype='bool'),
            dict(provider='masked_mimic', kind='average_velocity',
            key='masked_mimic.average_velocity', shape=[1, 2], dtype='float32'),
            dict(provider='masked_mimic', kind='velocity_command_mask',
            key='masked_mimic.velocity_command_mask', shape=[1, 1], dtype='bool')]}
    goals = BoxingGoals(metadata, 2, defender_guard_width=width,
                        body_attack_arc_lateral=body_attack_arc_lateral,
                        approach_speed_a=approach_speed_a,
                        approach_speed_b=approach_speed_b,
                        max_punch_reach_m=max_punch_reach_m)
    positions = np.zeros((2, len(bodies), 3), dtype=np.float32)
    rotations = np.zeros((2, len(bodies), 4), dtype=np.float32)
    rotations[..., 3] = 1
    goals.reset(positions, rotations)
    return goals, positions, rotations


def test_attacks_recover_and_primitive_changes_do_not_rebase_support():
    goals, positions, rotations = fixture()
    positions[0, 5] = [-.6, 0, 0]
    positions[1, 5] = [.6, 0, 0]
    goals.reset(positions, rotations)
    opponent = np.array([[.6, 0, 0], [-.6, 0, 0]])
    guard = goals.compile([0, 0], positions, rotations, opponent)['masked_mimic.ref_pos']
    attack = goals.compile([6, 7], positions, rotations, opponent)['masked_mimic.ref_pos']
    assert attack[0, 2, 3, 0] > guard[0, 2, 3, 0]
    assert attack[1, 3, 4, 2] < guard[1, 3, 4, 2]
    goals.advance(1.2)
    recovered = goals.compile([6, 7], positions, rotations, opponent)['masked_mimic.ref_pos']
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


def test_cross_goal_can_swing_laterally_around_a_centerline_guard():
    centered, pos, rot = fixture()
    positive, _, _ = fixture(body_attack_arc_lateral=.15)
    negative, _, _ = fixture(body_attack_arc_lateral=-.15)
    opponent = np.array([[.6, 0, 0], [-.6, 0, 0]])
    center_goal = centered.compile([7, 7], pos, rot, opponent)['masked_mimic.ref_pos']
    positive_goal = positive.compile([7, 7], pos, rot, opponent)['masked_mimic.ref_pos']
    negative_goal = negative.compile([7, 7], pos, rot, opponent)['masked_mimic.ref_pos']
    assert positive_goal[0, 2, 4, 1] > center_goal[0, 2, 4, 1]
    assert negative_goal[0, 2, 4, 1] < center_goal[0, 2, 4, 1]
    with pytest.raises(ValueError, match='target offsets'):
        fixture(body_attack_arc_lateral=.26)


def test_jab_targets_fk_head_and_clamps_to_configured_arm_reach():
    goals, pos, rot = fixture()
    shoulder = np.array([0., 0., 1.], dtype=np.float32)
    pos[0, 6] = shoulder
    pos[1, 5] = [.8, .1, 1.8]
    opponent = np.array([[.8, 0., .8], [-.8, 0., .8]], dtype=np.float32)
    goals.reset(pos, rot)
    goal = goals.compile([6, 0], pos, rot, opponent)['masked_mimic.ref_pos']

    raw_target = pos[1, 5] + np.array([.16, 0., 0.], dtype=np.float32)
    direction = raw_target - shoulder
    expected = shoulder + .48 * direction / np.linalg.norm(direction)
    # At 0.3 s the authored jab is at full extension; its target is the actual
    # head point projected to the qualified reach limit.
    np.testing.assert_allclose(goal[0, 2, 3], expected, atol=1e-6)
    assert np.linalg.norm(goal[0, 2, 3] - shoulder) == pytest.approx(.48)


def test_forward_footwork_uses_side_configured_approach_speed():
    goals, pos, rot = fixture(approach_speed_a=.29, approach_speed_b=.24)
    opponent = np.array([[1., 0., 0.], [-1., 0., 0.]], dtype=np.float32)
    result = goals.compile([2, 2], pos, rot, opponent)
    np.testing.assert_allclose(result['masked_mimic.average_velocity'],
                               [[.29, 0.], [-.24, 0.]], atol=1e-6)
    assert result['masked_mimic.velocity_command_mask'].tolist() == [[True], [True]]
