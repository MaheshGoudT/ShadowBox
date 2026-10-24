"""Behavior-level checks for seeded sparring and independent supervision."""

import mujoco
import numpy as np
import pytest

from shadowbox.boxing import Match, Parameters
from shadowbox.motion import Primitive
from shadowbox.config import BASE_CONFIG, load_config


def play(seed=0, defense_only=False, seconds=12, **kwargs):
    match = Match(seed=seed, defense_only=defense_only, rounds=1,
                  round_seconds=seconds, **kwargs)
    for _ in range(round((seconds + 1) / match.arena.model.opt.timestep)):
        match.step()
        if match.finished or match.supervisor.stopped:
            break
    return match


def test_seeded_match_repeats_and_records_actual_touches():
    first, second = play(), play()
    assert first.result() == second.result()
    assert first.finished and not first.supervisor.stopped
    assert sum(stats["attempts"] for stats in first.stats.values()) > 0
    assert sum(stats["touches"] for stats in first.stats.values()) > 0
    for stats in first.stats.values():
        assert stats["touches"] + stats["misses"] <= stats["attempts"]


def test_defense_only_never_requests_a_punch():
    match = play(defense_only=True)
    assert match.stats["b"]["attempts"] == 0
    assert match.stats["a"]["attempts"] > 0
    assert not any(event.get("robot") == "b" and event.get("action") in ("jab", "cross")
                   for event in match.events)
    assert np.isfinite(match.arena.data.qpos).all()


def test_supervisor_rejects_crossing_path_even_with_safe_endpoint():
    match = Match()
    target = match.arena.targets("a")
    target[0] = 2.3  # Endpoint is beyond B; the path crosses B's protected torso.
    assert not match.supervisor.approve("a", Primitive.STEP_FORWARD, target)
    assert match.supervisor.reason == "torso separation"


def test_defense_supervisor_rejects_offense_independently_of_policy():
    match = Match(defense_only=True)
    assert not match.supervisor.approve("b", Primitive.JAB, match.arena.targets("b"))
    assert match.supervisor.reason == "B is defense-only"


def test_supervisor_reserves_other_robots_inflight_path():
    match = Match()
    other_target = match.arena.targets("b")
    other_target[0] = -0.2
    match.arena.set_targets("b", other_target)
    match.supervisor.reserved["b"] = other_target[:2].copy()
    target = match.arena.targets("a")
    target[0] = 0.2
    assert not match.supervisor.approve("a", Primitive.STEP_FORWARD, target)


def test_supervisor_permits_recovery_out_of_close_range():
    match = Match()
    for robot, x in (("a", -0.32), ("b", 0.32)):
        pose = match.arena.pose(robot)
        pose[0] = x
        address = match.arena.model.joint(f"{robot}_x").qposadr[0]
        match.arena.data.qpos[address] = x
        match.arena.set_targets(robot, pose)
        match.supervisor.reserved[robot] = pose[:2].copy()
    mujoco.mj_forward(match.arena.model, match.arena.data)
    target = match.arena.targets("a")
    target[0] = -0.6
    assert match.supervisor.approve("a", Primitive.STEP_BACKWARD, target)


def test_stop_cancels_both_and_reset_is_recoverable():
    match = Match()
    for _ in range(50):
        match.step()
    match.stop()
    pose, time = match.arena.data.qpos.copy(), match.arena.time
    for _ in range(10):
        match.step()
    np.testing.assert_array_equal(match.arena.data.qpos, pose)
    assert match.arena.time == time
    assert all(not controller.busy for controller in match.controllers.values())
    match.reset()
    match.step()
    assert not match.supervisor.stopped and match.arena.time > 0


def test_match_remains_finite_for_sixty_seconds():
    match = play(seconds=60)
    assert match.finished and match.arena.time >= 59.99
    assert not match.supervisor.stopped
    assert np.isfinite(match.arena.data.qpos).all()


def test_reaction_delay_changes_behavior():
    config = load_config(BASE_CONFIG)
    fast = play(seconds=5, parameters_a=Parameters(1, 0.05, 0.78), config=config)
    slow = play(seconds=5, parameters_a=Parameters(1, 1.0, 0.78), config=config)
    decisions = lambda match: [event["time"] for event in match.events
                               if event["event"] == "action" and event["robot"] == "a"]
    assert decisions(slow)[1] - decisions(fast)[1] >= 0.9


@pytest.mark.parametrize("settings", [dict(aggression=float("nan")), dict(aggression=1.2),
                                       dict(reaction=0), dict(desired_distance=0.1)])
def test_invalid_policy_settings_fail_clearly(settings):
    with pytest.raises(ValueError):
        Parameters(**settings)


def test_multiple_rounds_pause_actions_during_breaks():
    match = Match(round_seconds=10, rounds=3)
    for _ in range(3500):
        match.step()
        if match.finished or match.supervisor.stopped:
            break
    assert match.finished and not match.supervisor.stopped
    assert len(match.scorecards) == 3
    ends = [e['time'] for e in match.events if e['event'] == 'round_end']
    starts = [e['time'] for e in match.events if e['event'] == 'round_start']
    for end, start in zip(ends, starts):
        assert not any(e['event'] == 'action' and end <= e['time'] < start for e in match.events)
