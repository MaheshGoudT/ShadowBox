"""Learning actions must actually control a boxer and expose defense signals."""

import numpy as np
import pytest

from shadowbox.evaluation.qlearning import ACTIONS
from shadowbox.rl import BoxingEnv


def action(name):
    return next(i for i, primitive in enumerate(ACTIONS) if primitive.value == name)


def test_requested_action_changes_the_match_and_reward():
    env = BoxingEnv('configs/unitree.yaml')
    observation, info = env.reset(seed=3)
    assert observation.shape == env.observation_space_shape
    assert info['action_mask'].shape == (len(ACTIONS),)
    _, jab_reward, _, _, jab = env.step(action('jab'), trace=True)
    assert jab['requested'] == 'jab' and jab['action_seen']
    assert any(event['event'] == 'action' and event.get('action') == 'jab'
               and event['robot'] == 'a' for event in env.match.events)
    assert np.isfinite(jab_reward)
    assert jab['trace']['closest_center_m']['torso'] > 0
    assert jab['trace']['max_glove_speed_m_s'] >= 0
    env.reset(seed=3)
    _, guard_reward, _, _, guard = env.step(action('guard'))
    assert guard['requested'] == 'guard' and guard['action_seen']
    assert guard['own']['touches'] == 0
    assert jab['own']['touches'] >= guard['own']['touches']
    assert jab_reward > guard_reward


def test_opponent_parameter_and_start_spacing_curriculum():
    env = BoxingEnv('configs/unitree.yaml')
    env.reset(seed=4, options={'spacing': 1.2,
                               'opponent_parameters': {'aggression': .3, 'reaction': .3,
                                                       'desired_distance': .65}})
    assert abs(env.match.observe('a').distance - 1.2) < .01
    assert env.match.parameters['b'].aggression == .3
    assert env.action_mask('b').any()
    observation, _ = env.reset(seed=4, options={'spacing': 1.2, 'side': 'b'})
    assert observation[0] > 0  # Opponent is ahead in B's local frame.
    _, _, _, _, info = env.step(action('guard'))
    assert info['action_seen'] and info['requested'] == 'guard'


def test_action_mask_matches_full_step_and_allows_close_defense():
    env = BoxingEnv('configs/unitree.yaml')
    _, info = env.reset(seed=5, options={'spacing': .72})
    assert not info['action_mask'][action('step_forward')]
    with pytest.raises(ValueError, match='masked'):
        env.step(action('step_forward'))
    _, info = env.reset(seed=5, options={'spacing': .62})
    assert info['action_mask'][action('body_guard')]
