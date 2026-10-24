"""Small deterministic safety and behavior scenarios for ShadowBox."""

import argparse
import json
from pathlib import Path
import math

import mujoco
import numpy as np

from ..boxing import Match
from ..config import BASE_CONFIG, artifact_path, load_config
from ..motion import Primitive


def _set_root(match, robot, x, y, yaw):
    """Place a robot consistently in measured and reserved state."""
    arena = match.arena
    pose = arena.pose(robot)
    pose[:3] = (x, y, yaw)
    arena.data.qpos[arena._joints[robot]] = pose
    target = arena.targets(robot)
    target[:3] = (x, y, yaw)
    arena.set_targets(robot, target)
    match.supervisor.reserved[robot] = target[:2].copy()


def _refresh_observation_history(match):
    mujoco.mj_forward(match.arena.model, match.arena.data)
    match.previous_gloves = {r: match.arena.glove_positions(r).copy() for r in ("a", "b")}
    match.previous_positions = {r: match.arena.position(r).copy() for r in ("a", "b")}
    match.glove_velocities = {r: match.glove_velocities[r] * 0 for r in ("a", "b")}
    match.velocities = {r: match.velocities[r] * 0 for r in ("a", "b")}


def _play(match, seconds):
    count = round(seconds / match.arena.model.opt.timestep)
    minimum_separation = float("inf")
    max_abs_position = 0.0
    for _ in range(count):
        if match.supervisor.stopped or match.finished:
            break
        match.step()
        positions = {r: match.arena.position(r) for r in ("a", "b")}
        dist = np.linalg.norm(positions["a"][:2] - positions["b"][:2])
        minimum_separation = min(minimum_separation, float(dist))
        values = []
        for position in positions.values():
            values.append(float(np.max(np.abs(position[:2]))))
        max_abs_position = max(max_abs_position, *values)
    if not math.isfinite(minimum_separation):
        minimum_separation = float(np.linalg.norm(match.arena.position("a")[:2] - match.arena.position("b")[:2]))
    return minimum_separation, max_abs_position


def _yaw_error(match, robot):
    opponent = "b" if robot == "a" else "a"
    delta = match.arena.position(opponent)[:2] - match.arena.position(robot)[:2]
    bearing = math.atan2(delta[1], delta[0])
    return abs(math.remainder(bearing - float(match.arena.pose(robot)[2]), 2 * math.pi))


def _record(name, checks, evidence):
    checks = [{**item, "passed": bool(item["passed"])} for item in checks]
    return {"name": name, "passed": all(item["passed"] for item in checks),
            "checks": checks, "evidence": evidence}


def closing_opponent(seed=17, config=None):
    match = Match(seed=seed, round_seconds=4, rounds=1, config=config or load_config(BASE_CONFIG))
    initial = float(np.linalg.norm(match.arena.position("a")[:2] - match.arena.position("b")[:2]))
    minimum, maximum = _play(match, 3)
    final = float(np.linalg.norm(match.arena.position("a")[:2] - match.arena.position("b")[:2]))
    return _record("closing_opponent", [
        {"name": "measured_separation_decreases", "passed": initial - final > 0.3},
        {"name": "match_remains_safe", "passed": not match.supervisor.stopped and maximum < match.supervisor.boundary},
    ], {"initial_separation": round(initial, 4), "final_separation": round(final, 4),
        "minimum_separation": round(minimum, 4), "maximum_abs_position": round(maximum, 4),
        "actions": [e["action"] for e in match.events if e["event"] == "action"]})


def boundary_defense(seed=17, config=None):
    match = Match(seed=seed, round_seconds=6, rounds=1, defense_only=True,
                  config=config or load_config(BASE_CONFIG))
    _set_root(match, "a", 1.45, 0, 0)
    _set_root(match, "b", 2.35, 0, math.pi)
    _refresh_observation_history(match)
    observation = match.observe("b")
    unsafe = match.controllers["b"].plan(Primitive.STEP_BACKWARD)
    retreat_accepted = match.supervisor.approve("b", Primitive.STEP_BACKWARD, unsafe)
    retreat_reason = match.supervisor.reason
    lateral = match.controllers["b"].plan(Primitive.SIDESTEP_LEFT)
    lateral_accepted = match.supervisor.approve("b", Primitive.SIDESTEP_LEFT, lateral)
    match.supervisor.reserved["b"] = match.arena.targets("b")[:2].copy()
    initial_separation = float(np.linalg.norm(match.arena.position("a")[:2] - match.arena.position("b")[:2]))
    minimum, maximum = _play(match, 5)
    final_separation = float(np.linalg.norm(match.arena.position("a")[:2] - match.arena.position("b")[:2]))
    a_actions = [e["action"] for e in match.events if e["event"] == "action" and e["robot"] == "a"]
    b_actions = [e["action"] for e in match.events if e["event"] == "action" and e["robot"] == "b"]
    b_punches = sum(action in ("jab", "cross") for action in b_actions)
    return _record("boundary_defense", [
        {"name": "unsafe_retreat_path_is_rejected", "passed": not retreat_accepted and retreat_reason == "arena boundary"},
        {"name": "defender_does_not_punch", "passed": b_punches == 0},
        {"name": "defender_avoids_unsafe_retreat", "passed": "step_backward" not in b_actions},
        {"name": "opponent_closes_from_initial_range", "passed": initial_separation - final_separation > 0.08},
        {"name": "approach_remains_inside_safe_bounds", "passed": not match.supervisor.stopped and maximum < match.supervisor.boundary},
        {"name": "minimum_separation_remains_safe", "passed": minimum >= match.supervisor.config["monitor_separation"]},
    ], {"boundary_space": round(observation.boundary_space, 4), "unsafe_action": Primitive.STEP_BACKWARD.value,
        "unsafe_action_accepted": retreat_accepted, "unsafe_action_reason": retreat_reason,
        "opponent_actions": a_actions, "defender_actions": b_actions, "defender_punches": b_punches,
        "initial_separation": round(initial_separation, 4), "final_separation": round(final_separation, 4),
        "minimum_separation": round(minimum, 4), "maximum_abs_position": round(maximum, 4),
        "stopped": match.supervisor.stopped})


def asymmetric_position_start(seed=17, config=None):
    match = Match(seed=seed, round_seconds=11, rounds=1, config=config or load_config(BASE_CONFIG))
    _set_root(match, "a", -1.0, -0.35, 0.1)
    _set_root(match, "b", 0.9, 0.4, 3.0)
    _refresh_observation_history(match)
    initial_errors = {r: _yaw_error(match, r) for r in ("a", "b")}
    minimum, maximum = _play(match, 10)
    final_errors = {r: _yaw_error(match, r) for r in ("a", "b")}
    return _record("asymmetric_position_start", [
        {"name": "a_yaw_error_decreases", "passed": final_errors["a"] < initial_errors["a"] - 0.05},
        {"name": "b_yaw_error_decreases", "passed": final_errors["b"] < initial_errors["b"] - 0.05},
        {"name": "both_remain_inside_arena", "passed": not match.supervisor.stopped and maximum < match.supervisor.boundary},
    ], {"a_position": match.arena.position("a").round(4).tolist(),
        "b_position": match.arena.position("b").round(4).tolist(),
        "initial_yaw_error": {r: round(v, 4) for r, v in initial_errors.items()},
        "final_yaw_error": {r: round(v, 4) for r, v in final_errors.items()},
        "minimum_separation": round(minimum, 4), "maximum_abs_position": round(maximum, 4),
        "actions": [e["action"] for e in match.events if e["event"] == "action"]})


def emergency_stop_reset(seed=17, config=None):
    match = Match(seed=seed, round_seconds=2, rounds=1, config=config or load_config(BASE_CONFIG))
    for _ in range(30):
        match.step()
    was_moving = any(c.busy for c in match.controllers.values())
    match.stop()
    stopped = match.supervisor.stopped and all(not c.busy for c in match.controllers.values())
    stopped_time = match.arena.time
    match.step()
    no_progress = match.arena.time == stopped_time
    match.reset()
    recovered = not match.supervisor.stopped and match.supervisor.reason == "ready"
    match.step()
    progressed_after_reset = match.arena.time > 0
    return _record("emergency_stop_reset", [
        {"name": "stop_cancels_inflight_motion", "passed": was_moving and stopped},
        {"name": "stopped_match_does_not_advance", "passed": no_progress},
        {"name": "reset_restores_readiness_and_progress", "passed": recovered and progressed_after_reset},
    ], {"stopped": stopped, "stopped_time": stopped_time, "no_progress": no_progress,
        "had_inflight_motion": was_moving, "reset_recovered": recovered,
        "progressed_after_reset": progressed_after_reset})


def offense_and_guard(seed=17, config=None):
    config = config or load_config(BASE_CONFIG)
    match = Match(seed=seed, round_seconds=3, rounds=1, config=config)
    _set_root(match, "a", -0.375, 0, 0)
    _set_root(match, "b", 0.375, 0, math.pi)
    _refresh_observation_history(match)
    jab_started = match._request("a", Primitive.JAB)
    for _ in range(80):
        if not match.controllers["a"].busy:
            break
        for controller in match.controllers.values():
            controller.tick()
        match.arena.step()
        for robot in ("a", "b"):
            match._referee(robot, match.arena.glove_positions(robot).copy())
        match.previous_gloves = {r: match.arena.glove_positions(r).copy() for r in ("a", "b")}
        match.previous_positions = {r: match.arena.position(r).copy() for r in ("a", "b")}
    jab_blocks = match.stats["b"]["blocks"]
    jab_touches = match.stats["a"]["touches"]

    cross = Match(seed=seed, round_seconds=3, rounds=1, config=config)
    _set_root(cross, "a", -0.375, 0, 0)
    _set_root(cross, "b", 0.375, 0, math.pi)
    _refresh_observation_history(cross)
    cross_started = cross._request("a", Primitive.CROSS)
    for _ in range(140):
        if not cross.controllers["a"].busy:
            break
        for controller in cross.controllers.values():
            controller.tick()
        cross.arena.step()
        for robot in ("a", "b"):
            cross._referee(robot, cross.arena.glove_positions(robot).copy())
        cross.previous_gloves = {r: cross.arena.glove_positions(r).copy() for r in ("a", "b")}
        cross.previous_positions = {r: cross.arena.position(r).copy() for r in ("a", "b")}
    cross_touches = cross.stats["a"]["touches"]
    cross_blocks = cross.stats["b"]["blocks"]

    jab_sep = float(np.linalg.norm(match.arena.position("a")[:2] - match.arena.position("b")[:2]))
    cross_sep = float(np.linalg.norm(cross.arena.position("a")[:2] - cross.arena.position("b")[:2]))
    jab_target = next((e.get("target") for e in match.events if e["event"] == "touch"), None)
    cross_target = next((e.get("target") for e in cross.events if e["event"] == "touch"), None)

    return _record("offense_and_guard", [
        {"name": "jab_resolves_at_guard", "passed": jab_started and jab_touches + jab_blocks == 1},
        {"name": "cross_scores_torso_touch", "passed": cross_started and cross_touches == 1 and cross_blocks == 0},
    ], {"jab_blocks": jab_blocks,
        "jab_touches": jab_touches,
        "cross_touches": cross_touches,
        "cross_blocks": cross_blocks,
        "jab_separation": round(jab_sep, 4),
        "cross_separation": round(cross_sep, 4),
        "jab_target": jab_target,
        "cross_target": cross_target})


def run_scenarios(seed=17, config=None):
    config = config or load_config(BASE_CONFIG)
    scenarios = [closing_opponent(seed, config), boundary_defense(seed, config),
                 asymmetric_position_start(seed, config), emergency_stop_reset(seed, config),
                 offense_and_guard(seed, config)]
    return {"schema_version": 1, "seed": seed, "passed": all(s["passed"] for s in scenarios),
            "scenarios": scenarios}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run deterministic ShadowBox behavior scenarios")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--config", type=Path, default=BASE_CONFIG)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    report = run_scenarios(args.seed, load_config(args.config))
    path = artifact_path(args.output or "tmp/results/scenarios.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Scenarios {'passed' if report['passed'] else 'failed'}; report: {path}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
