"""Measured bottlenecks for the next ShadowBox iteration."""

import argparse
from collections import Counter
import json
import math

import numpy as np

from ..boxing import Match
from ..config import artifact_path, load_config, source_fingerprint
from ..motion import Primitive


STEPS = {Primitive.STEP_FORWARD, Primitive.STEP_BACKWARD,
         Primitive.SIDESTEP_LEFT, Primitive.SIDESTEP_RIGHT}


def diagnose(config=None, seeds=(41, 42, 43), seconds=20, policy_path=None):
    """Run fresh matches and rank observable simulation problems."""
    config = config or load_config()
    if not seeds or seconds <= 0 or not math.isfinite(seconds):
        raise ValueError('Diagnostics need seeds and positive simulated seconds')
    matches = []
    for seed in seeds:
        match = Match(seed=seed, rounds=1, round_seconds=seconds, config=config)
        if policy_path:
            from .qlearning import load_learned_policy
            for index, robot in enumerate(('a', 'b')):
                match.policies[robot] = load_learned_policy(policy_path,
                    seed=seed + 1009 * index, config=config)
        stance = None
        if config['simulation']['model'] == 'unitree_g1' and config['stance']['enabled']:
            from ..motion.stance import BoxingStance
            stance = BoxingStance(match.arena)
        previous_feet = ({r: stance.foot_positions(r) for r in ('a', 'b')}
                         if stance else None)
        previous_swinging = {r: None for r in ('a', 'b')}
        grounded = samples = slip = max_planted_error = 0.0
        distances = []
        for _ in range(math.ceil((seconds + 1) / match.arena.model.opt.timestep)):
            if match.finished or match.supervisor.stopped:
                break
            match.step()
            distances.append(match.observe('a').distance)
            if stance:
                for robot in ('a', 'b'):
                    feet = stance.foot_positions(robot)
                    footwork = match.footwork.get(robot)
                    swinging = footwork.swinging if footwork else None
                    for side, foot in feet.items():
                        old = previous_feet[robot][side]
                        near_floor = abs(foot[2] - config['stance']['sole_height']) <= .01
                        grounded += near_floor
                        samples += 1
                        planted = side not in (swinging, previous_swinging[robot])
                        if planted and near_floor and abs(old[2] - config['stance']['sole_height']) <= .01:
                            slip += float(np.linalg.norm(foot[:2] - old[:2]))
                        if footwork and planted:
                            max_planted_error = max(max_planted_error,
                                float(np.linalg.norm(foot[:2] - footwork.targets[side][:2])))
                    previous_feet[robot] = feet
                    previous_swinging[robot] = swinging
        actions = Counter(e.get('action') for e in match.events if e['event'] == 'action')
        rejections = Counter(e.get('reason') for e in match.events if e['event'] == 'rejected')
        matches.append(dict(seed=seed, finished=match.finished, stopped=match.supervisor.stopped,
                            simulated_seconds=match.arena.time, stats=match.stats,
                            steps=sum(actions[action.value] for action in STEPS),
                            actions=dict(actions), rejections=dict(rejections),
                            mean_distance=float(np.mean(distances)) if distances else None,
                            grounded_fraction=grounded / samples if samples else None,
                            grounded_foot_slip_m=slip if stance else None,
                            max_planted_foot_error_m=max_planted_error if match.footwork else None))
    attempts = sum(m['stats'][r]['attempts'] for m in matches for r in ('a', 'b'))
    touches = sum(m['stats'][r]['touches'] for m in matches for r in ('a', 'b'))
    blocks = sum(m['stats'][r]['blocks'] for m in matches for r in ('a', 'b'))
    actions = Counter()
    for match in matches:
        actions.update(match['actions'])
    steps = sum(m['steps'] for m in matches)
    slip = sum(m['grounded_foot_slip_m'] or 0 for m in matches)
    stopped = sum(m['stopped'] for m in matches)
    next_steps = []
    if stopped:
        next_steps.append('Resolve safety stops before training.')
    if steps == 0 and np.mean([m['mean_distance'] for m in matches]) > 1.2 * config['policy_rules']['distance_scale']:
        next_steps.append('Fix approach policy and root motion; opponents remain out of range.')
    if attempts and not touches:
        next_steps.append('Measure glove-to-target reach and correct punch trajectories.')
    if policy_path and attempts >= 10 and (blocks == 0 or
            max(actions['jab'], actions['cross']) / max(1, actions['jab'] + actions['cross']) > .9):
        next_steps.append('Add body defense and punch-level variety; learned self-play repeats one attack.')
    if slip > .1:
        if config.get('footwork', {}).get('enabled'):
            next_steps.append('Tune foot placement and servo tracking; planted feet still drift.')
        else:
            next_steps.append('Add planted and swinging foot phases; grounded feet slide during root steps.')
    if attempts and touches and touches / attempts < .15:
        next_steps.append('Improve distance choice, target height, and defensive responses before more training.')
    if not next_steps:
        next_steps.append('Compare learned policies with the reactive baseline on held-out opponents and seeds.')
    return dict(source_fingerprint=source_fingerprint(), model=config['simulation']['model'],
                policy=str(policy_path) if policy_path else 'reactive',
                seeds=list(seeds), seconds=seconds, attempts=attempts, touches=touches,
                blocks=blocks, touch_per_attempt=touches / max(attempts, 1),
                actions=dict(actions), steps=steps, safety_stops=stopped,
                grounded_foot_slip_m=slip if stance else None,
                max_planted_foot_error_m=max((m['max_planted_foot_error_m'] or 0 for m in matches),
                                              default=0) if matches[0]['max_planted_foot_error_m'] is not None else None,
                next_steps=next_steps, matches=matches)


def main():
    parser = argparse.ArgumentParser(description='Measure ShadowBox behavior and rank next fixes')
    parser.add_argument('--config')
    parser.add_argument('--seconds', type=float, default=20)
    parser.add_argument('--seeds', type=int, nargs='+', default=[41, 42, 43])
    parser.add_argument('--output', default='tmp/results/diagnostics.json')
    parser.add_argument('--policy', help='promoted tabular policy artifact to test on fresh seeds')
    args = parser.parse_args()
    report = diagnose(load_config(args.config), args.seeds, args.seconds, args.policy)
    path = artifact_path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + '\n')
    print(f"{report['attempts']} attempts, {report['touches']} touches, {report['steps']} steps")
    print('Next:', report['next_steps'][0])
    print(path)


if __name__ == '__main__':
    main()
