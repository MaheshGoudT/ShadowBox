"""Specialists must produce legal, measurable incoming threats."""

import pytest

from shadowbox.boxing import Parameters
from shadowbox.config import load_config
from shadowbox.motion import Primitive
from shadowbox.rl.actions import ACTIONS, allowed_actions
from shadowbox.rl.environment import BoxingEnv
from shadowbox.rl.opponents import AttackOpponent


@pytest.mark.parametrize('level,expected', [('head', Primitive.JAB), ('body', Primitive.CROSS)])
def test_attacker_generates_a_legal_measured_threat(level, expected):
    config = load_config()
    config['match']['round_seconds'] = 2
    config['scoring']['proximity_fallback'] = False
    env = BoxingEnv(config)
    attacker = AttackOpponent(45009, Parameters(**config['policies']['b']), config, level)
    env.reset(45000, dict(spacing=.62, opponent_policy=attacker))
    observation = env.match.observe('b')
    chosen = attacker.choose(observation)
    assert chosen == expected
    assert ACTIONS.index(chosen) in allowed_actions(observation, attacker.parameters,
        config['motion'], config['safety']['min_separation'], config['policy_rules'])
    while not env.match.finished:
        _, _, done, truncated, _ = env.step(0)
        assert not truncated
        if done:
            break
    assert env.match.stats['b']['attempts'] > 0
    assert not env.match.supervisor.stopped
