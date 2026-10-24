"""Fixed-pose scenarios use actual MuJoCo kinematics, not sampled hit odds."""

import mujoco
import numpy as np

from shadowbox.boxing import Match
from shadowbox.boxing.geometry import sphere_entry, capsule_entry
from shadowbox.motion import Primitive


def fixed_stance(distance=0.75):
    match = Match(rounds=1, round_seconds=10)
    for robot, x in (("a", -distance / 2), ("b", distance / 2)):
        pose = match.arena.pose(robot)
        pose[0] = x
        address = match.arena.model.joint(f"{robot}_x").qposadr[0]
        match.arena.data.qpos[address] = x
        match.arena.set_targets(robot, pose)
        match.supervisor.reserved[robot] = pose[:2].copy()
    mujoco.mj_forward(match.arena.model, match.arena.data)
    remember(match)
    return match


def remember(match):
    match.previous_gloves = {r: match.arena.glove_positions(r).copy() for r in ("a", "b")}
    match.previous_positions = {r: match.arena.position(r).copy() for r in ("a", "b")}


def throw(match, action, evade=False):
    assert match._request("a", action)
    while match.controllers["a"].busy:
        if evade and match.arena.time >= 0.3 and not match.controllers["b"].busy:
            match.controllers["b"].start(Primitive.SIDESTEP_RIGHT)
            evade = False
        for controller in match.controllers.values():
            controller.tick()
        match.arena.step()
        match._referee("a", match.arena.glove_positions("a"))
        remember(match)
    match._finish_punch("a")


def test_high_guard_does_not_overlap_at_minimum_planned_separation():
    match = fixed_stance(0.66)
    distances = np.linalg.norm(match.arena.glove_positions("a")[:, None] -
                               match.arena.glove_positions("b")[None, :], axis=2)
    assert distances.min() > 0.22


def test_high_guard_allows_jab_touch():
    match = fixed_stance()
    throw(match, Primitive.JAB)
    assert match.stats["a"]["touches"] == 1
    assert match.stats["b"]["blocks"] == 0
    assert [e['target'] for e in match.events if e['event'] == 'touch'] == ['head']


def test_body_cross_scores_torso_touch():
    match = fixed_stance()
    assert match.controllers['b'].start(Primitive.GUARD)
    throw(match, Primitive.CROSS)
    assert match.stats["a"]["touches"] == 1
    assert match.stats["b"]["blocks"] == 0
    assert [e['target'] for e in match.events if e['event'] == 'touch'] == ['torso']


def test_sidestep_scores_touch():
    match = fixed_stance()
    throw(match, Primitive.CROSS, evade=True)
    assert match.stats["a"]["touches"] == 1
    assert match.stats["b"]["blocks"] == 0
    assert [e['target'] for e in match.events if e['event'] == 'touch'] == ['torso']


def test_out_of_range_punch_is_a_miss():
    match = fixed_stance(1.5)
    throw(match, Primitive.JAB)
    assert match.stats["a"]["misses"] == 1
    assert match.stats["a"]["touches"] == match.stats["b"]["blocks"] == 0


def test_swept_entry_orders_guard_before_target_in_one_large_step():
    start, end = np.array([0., 0., 1.]), np.array([1., 0., 1.])
    guard = sphere_entry(start, end, np.array([0.4, 0., 1.]), 0.1)
    torso = capsule_entry(start, end, np.array([0.8, 0., 0.8]),
                          np.array([0.8, 0., 1.2]), 0.2)
    assert guard is not None and torso is not None and guard < torso


def test_lowering_cross_respects_motion_rate_limits():
    match = fixed_stance(1.5)
    controller = match.controllers["a"]
    history = [match.arena.targets("a")]
    controller.start(Primitive.CROSS)
    while controller.busy:
        controller.tick()
        history.append(match.arena.targets("a"))
        match.arena.step()
    timestep = match.arena.model.opt.timestep
    velocities = np.diff(history, axis=0) / timestep
    accelerations = np.diff(velocities, axis=0) / timestep
    assert np.all(np.abs(velocities) <= np.array(match.config["motion"]["velocity_limits"]) + 0.02)
    assert np.all(np.abs(accelerations) <= np.array(match.config["motion"]["acceleration_limits"]) + 0.2)
