"""G1 body defense and uppercut use measured glove paths."""

import mujoco
from dataclasses import replace

from shadowbox.boxing import Match
from shadowbox.boxing.policy import Observation
from shadowbox.config import load_config
from shadowbox.motion import Primitive


def fixed_g1(distance):
    match = Match(config=load_config('configs/unitree.yaml'), rounds=1, round_seconds=5)
    for robot, x in (('a', -distance / 2), ('b', distance / 2)):
        match.arena.data.qpos[match.arena.model.joint(f'{robot}_x').qposadr[0]] = x
        target = match.arena.targets(robot)
        target[0] = x
        match.arena.set_targets(robot, target)
        match.supervisor.reserved[robot] = target[:2].copy()
    mujoco.mj_forward(match.arena.model, match.arena.data)
    remember(match)
    return match


def remember(match):
    match.previous_gloves = {r: match.arena.glove_positions(r).copy() for r in 'ab'}
    match.previous_positions = {r: match.arena.position(r).copy() for r in 'ab'}


def complete(match, robot):
    while match.controllers[robot].busy:
        match.controllers[robot].tick()
        match.arena.step()
        match._referee(robot, match.arena.glove_positions(robot))
        remember(match)
    match._finish_punch(robot)


def test_body_guard_blocks_cross_at_boxing_distance():
    match = fixed_g1(.6)
    assert match.controllers['b'].start(Primitive.BODY_GUARD)
    complete(match, 'b')
    assert match._request('a', Primitive.CROSS)
    complete(match, 'a')
    assert match.stats['b']['blocks'] == 1
    assert match.stats['a']['touches'] == 0


def test_uppercut_can_reach_head_over_body_guard():
    match = fixed_g1(.5)
    assert match.controllers['b'].start(Primitive.BODY_GUARD)
    complete(match, 'b')
    assert match._request('a', Primitive.UPPERCUT)
    complete(match, 'a')
    assert [e['defender'] for e in match.events if e['event'] == 'blocked'] == ['b']


def test_reactive_policy_uses_body_guard_for_low_incoming_glove():
    match = fixed_g1(.6)
    observation = match.observe('b')
    observation = Observation(**{**observation.__dict__, 'incoming_speed': .4,
                                 'incoming_body': True})
    assert match.policies['b'].choose(observation) == Primitive.BODY_GUARD


def test_incoming_body_shot_interrupts_a_step_for_guard(monkeypatch):
    match = fixed_g1(.6)
    assert match.controllers['b'].start(Primitive.STEP_BACKWARD)
    original = match.observe

    def threatened(robot):
        observation = original(robot)
        return (replace(observation, incoming_speed=.5, incoming_body=True)
                if robot == 'b' else observation)

    monkeypatch.setattr(match, 'observe', threatened)
    match.step()
    assert match.controllers['b'].active == Primitive.BODY_GUARD
    assert any(event['event'] == 'action' and event.get('action') == 'body_guard'
               for event in match.events)
