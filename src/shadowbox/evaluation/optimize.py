"""Seeded two-sided evaluation and small reactive-policy parameter search."""

import argparse
from dataclasses import asdict
import json
import logging
from pathlib import Path

import numpy as np

from ..boxing import Match, Parameters
from ..config import load_config, configure_logging, artifact_path, source_fingerprint


def evaluate(parameters: dict, opponents: list[dict], seeds: list[int],
             seconds: float = 12, config: dict | None = None) -> dict:
    """Play both assignments; reward clean touch differential, penalize stops."""
    if not opponents or not seeds or not np.isfinite(seconds) or seconds <= 0:
        raise ValueError("Evaluation needs opponents, seeds, and positive seconds.")
    matches = []
    for opponent_index, opponent in enumerate(opponents):
        for seed in seeds:
            for side in ("a", "b"):
                other = "b" if side == "a" else "a"
                setting = {side: parameters, other: opponent}
                settings = dict(seed=seed, round_seconds=seconds, rounds=1,
                                parameters_a=setting["a"], parameters_b=setting["b"])
                if config is not None:
                    settings["config"] = config
                match = Match(**settings)
                limit = int((seconds + 1) / match.arena.model.opt.timestep) + 1
                for _ in range(limit):
                    if match.finished or match.supervisor.stopped:
                        break
                    match.step()
                result = match.result()
                own, theirs = result["stats"][side], result["stats"][other]
                failed = result["stopped"] or not result["finished"]
                reward = own["touches"] - theirs["touches"] - 5 * failed
                matches.append(dict(seed=seed, candidate_side=side, opponent=opponent_index,
                                    opponent_parameters=dict(opponent), candidate_stats=dict(own),
                                    opponent_stats=dict(theirs), reward=reward,
                                    stopped=result["stopped"], finished=result["finished"],
                                    safety_reason=result["safety_reason"],
                                    safety_rejections=result["safety_rejections"],
                                    scores=result["scores"], simulated_seconds=result["time"],
                                    evasion_attempts=result.get(f"evasion_attempts_{side}", 0),
                                    evasion_success=result.get(f"evasion_success_{side}", 0),
                                    offense_attempts=result.get(f"offense_attempts_{side}", 0),
                                    offense_success=result.get(f"offense_success_{side}", 0)))
    return dict(mean_reward=float(np.mean([m["reward"] for m in matches])),
                touches_for=sum(m["candidate_stats"]["touches"] for m in matches),
                touches_against=sum(m["opponent_stats"]["touches"] for m in matches),
                safety_stops=sum(m["stopped"] for m in matches),
                incomplete_matches=sum(not m["finished"] for m in matches),
                mean_evasion_rate=float(np.mean([m["evasion_success"] / max(m["evasion_attempts"], 1) for m in matches])),
                mean_offense_rate=float(np.mean([m["offense_success"] / max(m["offense_attempts"], 1) for m in matches])),
                matches=matches)


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def search(iterations: int = 8, seeds: int = 3, seconds: float = 12,
           output: str | Path = "tmp/policies/candidate.json", seed: int = 0,
           config: dict | None = None) -> dict:
    """Optimize a small policy; this is parameter search, not neural RL."""
    if iterations < 0 or seeds < 1 or not np.isfinite(seconds) or seconds <= 0:
        raise ValueError("Use nonnegative iterations, positive seeds, and positive seconds.")
    config = config or load_config()
    settings = config["training"]
    rng = np.random.default_rng(seed)
    baseline = asdict(Parameters(**config["policies"]["a"]))
    best = dict(baseline)
    pool = [dict(baseline)]
    train_seeds = list(range(seeds))
    validation_seeds = list(range(settings["validation_seed_offset"] + seeds, settings["validation_seed_offset"] + 2 * seeds))
    path = artifact_path(output)
    snapshots = path.parent / (path.stem + "_snapshots")
    _write(snapshots / "baseline.json", dict(parameters=baseline, role="fixed baseline"))
    history = []

    for iteration in range(iterations):
        values = np.array(list(best.values())) + rng.normal(0, settings["perturbation_std"])
        values = np.clip(values, settings["lower_bounds"], settings["upper_bounds"])
        candidate = dict(zip(best, map(float, values)))
        incumbent_result = evaluate(best, pool, train_seeds, seconds, config)
        candidate_result = evaluate(candidate, pool, train_seeds, seconds, config)
        accepted = (candidate_result["mean_reward"] > incumbent_result["mean_reward"]
                    and candidate_result["safety_stops"] <= incumbent_result["safety_stops"]
                    and candidate_result["incomplete_matches"] <= incumbent_result["incomplete_matches"])
        entry = dict(iteration=iteration + 1, parameters=candidate, accepted=accepted,
                     incumbent=incumbent_result, candidate=candidate_result)
        history.append(entry)
        _write(snapshots / f"iteration_{iteration + 1:03d}.json", entry)
        if accepted:
            previous = dict(best)
            best = candidate
            if previous not in pool:
                pool.append(previous)
            pool = [baseline, *[p for p in pool[1:]][-2:]]
        print(f"Iteration {iteration + 1}/{iterations}: "
              f"candidate {candidate_result['mean_reward']:.3f}, "
              f"incumbent {incumbent_result['mean_reward']:.3f}, "
              f"{'accepted' if accepted else 'kept incumbent'}", flush=True)

    # Validation is consulted once after search and never selects another candidate.
    validation_opponents = [baseline, *settings["validation_opponents"]]
    baseline_result = evaluate(baseline, validation_opponents, validation_seeds, seconds, config)
    candidate_result = evaluate(best, validation_opponents, validation_seeds, seconds, config)
    improvement = candidate_result["mean_reward"] - baseline_result["mean_reward"]
    improved = (improvement > 0 and candidate_result["safety_stops"] == 0
                and candidate_result["incomplete_matches"] == 0)
    deployed = best if improved else baseline
    report = dict(schema_version=1, source_fingerprint=source_fingerprint(), method="seeded reactive-policy parameter search",
                  parameters=deployed, baseline=baseline, candidate=best, search_seed=seed,
                  promoted=improved, simulation_config=config,
                  iterations=iterations, seconds_per_round=seconds,
                  training_seeds=train_seeds, validation_seeds=validation_seeds,
                  training_opponents=pool, validation_opponents=validation_opponents,
                  history=history,
                  validation=dict(baseline=baseline_result, candidate=candidate_result,
                                  reward_difference=improvement, improved=improved),
                  limitation="Observed results on this small held-out set are not proof of generalization. "
                             "This searches reactive policy parameters; no neural policy was trained.")
    report_path = path.with_name(path.stem + "_report.json")
    logging.info("Search validation difference=%s promoted=%s report=%s", improvement, improved, report_path)
    _write(report_path, report)
    _write(path, dict(schema_version=1, parameters=deployed, promoted=improved,
                     method=report["method"], report=str(report_path)))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate and optimize reactive boxing policies")
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--seeds", type=int, help="number of training and held-out seeds")
    parser.add_argument("--seconds", type=float, help="simulated seconds per round")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, help="parameter-search random seed")
    parser.add_argument("--config", type=Path, help="simulation YAML configuration")
    args = parser.parse_args()
    try:
        config = load_config(args.config)
        configure_logging(config)
        settings = config['training']
        iterations = args.iterations if args.iterations is not None else settings['iterations']
        seeds = args.seeds if args.seeds is not None else settings['seeds']
        seconds = args.seconds if args.seconds is not None else settings['round_seconds']
        seed = args.seed if args.seed is not None else settings['search_seed']
        output = args.output or Path(settings['output'])
        report = search(iterations, seeds, seconds, output, seed, config)
    except ValueError as error:
        parser.error(str(error))
    validation = report["validation"]
    print(f"Held-out mean reward: baseline {validation['baseline']['mean_reward']:.3f}, "
          f"candidate {validation['candidate']['mean_reward']:.3f}")
    print(f"Promoted candidate: {validation['improved']}. Saved {output}")
    if not validation["improved"]:
        print("Validation did not justify promotion; the saved policy retains the baseline.")


if __name__ == "__main__":
    main()
