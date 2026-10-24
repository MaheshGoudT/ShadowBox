"""Held-out tactical checks before a learned boxer can be promoted."""

import argparse
from collections import Counter
from copy import deepcopy
import json

import numpy as np
import torch

from ..boxing import Parameters, Policy
from ..config import artifact_path, load_config, source_fingerprint
from .actions import ACTIONS
from .environment import BoxingEnv
from .model import ActorCritic
from .observations import OBSERVATION_SIZE
from .opponents import AttackOpponent, GuardOpponent
from .policy import sample_action

OPPONENTS = ('reactive', 'head_guard', 'body_guard', 'head_attack', 'body_attack')


def _play(config, model, seed, side, spacing, opponent_kind):
    env = BoxingEnv(config)
    options = dict(side=side, spacing=spacing)
    other = 'b' if side == 'a' else 'a'
    if opponent_kind in ('head_attack', 'body_attack'):
        options['opponent_policy'] = AttackOpponent(seed + 1009,
            Parameters(**config['policies'][other]), config,
            'head' if opponent_kind == 'head_attack' else 'body')
    elif opponent_kind != 'reactive':
        options['opponent_policy'] = GuardOpponent(seed + 1009,
            Parameters(**config['policies'][other]), config['policy_rules'],
            'head' if opponent_kind == 'head_guard' else 'body')
    state, info = env.reset(seed, options)
    baseline = Policy(seed, Parameters(**config['policies'][side]),
                      rules=config['policy_rules'])
    choices, rejections, reasons = Counter(), 0, Counter()
    guard_blocks = evasion_misses = 0
    generator = torch.Generator().manual_seed(seed * 100 + (side == 'b') * 10
                                             + (spacing > .7) * 3
                                             + OPPONENTS.index(opponent_kind))
    for _ in range(200):
        if model is None:
            suggested = baseline.choose(env.match.observe(side))
            action = ACTIONS.index(suggested) if suggested in ACTIONS else 0
            if not info['action_mask'][action]:
                action = 0
        else:
            with torch.no_grad():
                distribution, _ = model(torch.as_tensor(state).unsqueeze(0),
                                        torch.as_tensor(info['action_mask']).unsqueeze(0))
                temperature = config.get('tactics', {}).get('inference_temperature', .5)
                action = sample_action(distribution, generator, temperature)
        state, _, done, truncated, info = env.step(action)
        choices[info['requested']] += 1
        rejections += int(info['rejected'])
        reasons.update(info['rejection_reasons'])
        if info['requested'] in ('guard', 'body_guard'):
            guard_blocks += info['own']['blocks']
        if info['requested'] in ('step_backward', 'sidestep_left', 'sidestep_right'):
            evasion_misses += info['opponent']['misses']
        if done or truncated:
            break
    match = env.match
    return dict(seed=seed, side=side, spacing=spacing, opponent_kind=opponent_kind,
                own=match.stats[side], opponent=match.stats[other],
                actions=dict(choices), rejections=rejections,
                rejection_reasons=dict(reasons),
                guard_blocks=guard_blocks, opponent_misses_during_evasion=evasion_misses,
                stopped=match.supervisor.stopped, finished=match.finished,
                seconds=match.arena.time)


def _summary(matches):
    actions, reasons = Counter(), Counter()
    for match in matches:
        actions.update(match['actions'])
        reasons.update(match['rejection_reasons'])
    attacks = {name: actions[name] for name in ('jab', 'cross', 'uppercut')}
    total = sum(attacks.values())
    return dict(matches=matches,
                mean_touch_difference=float(np.mean([
                    m['own']['touches'] - m['opponent']['touches'] for m in matches])),
                touches=sum(m['own']['touches'] for m in matches),
                blocks=sum(m['own']['blocks'] for m in matches),
                guard_blocks=sum(m.get('guard_blocks', 0) for m in matches),
                conceded_touches=sum(m['opponent']['touches'] for m in matches),
                opponent_misses_during_evasion=sum(m.get('opponent_misses_during_evasion', 0)
                                                  for m in matches),
                opponent_attempts=sum(m['opponent']['attempts'] for m in matches),
                attempts=sum(m['own']['attempts'] for m in matches),
                rejections=sum(m['rejections'] for m in matches),
                rejection_reasons=dict(reasons),
                safety_stops=sum(m['stopped'] for m in matches),
                incomplete=sum(not m['finished'] for m in matches),
                actions=dict(actions), attack_types=sum(count > 0 for count in attacks.values()),
                largest_attack_share=max(attacks.values(), default=0) / max(total, 1))


def evaluate_checkpoint(path, seeds=(20000, 20001, 20002), spacings=(.72, 1.2),
                        opponents=('reactive', 'head_guard', 'body_guard')):
    path = artifact_path(path)
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if (checkpoint['actions'] != [action.value for action in ACTIONS]
            or checkpoint['observation_shape'] != OBSERVATION_SIZE):
        raise ValueError('Tactical checkpoint action or observation schema changed')
    if set(seeds) & set(range(checkpoint['seed'], checkpoint['seed'] + checkpoint['episodes'])):
        raise ValueError('Evaluation seeds overlap training seeds')
    config = deepcopy(checkpoint['config'])
    model = ActorCritic()
    model.load_state_dict(checkpoint['model'])
    model.eval()
    cases = [(seed, side, spacing, opponent) for seed in seeds for side in ('a', 'b')
             for spacing in spacings for opponent in opponents]
    baseline = _summary([_play(config, None, *case) for case in cases])
    candidate = _summary([_play(config, model, *case) for case in cases])
    unchanged = checkpoint['source_fingerprint'] == source_fingerprint()
    side_ok = all(sum(m['own']['touches'] for m in candidate['matches'] if m['side'] == side) > 0
                  for side in ('a', 'b'))
    candidate_rejection_rate = candidate['rejections'] / max(sum(candidate['actions'].values()), 1)
    promoted = (unchanged and side_ok
                and candidate['mean_touch_difference'] > baseline['mean_touch_difference']
                and candidate['attempts'] >= len(cases)
                and candidate['blocks'] > 0 and candidate['attack_types'] >= 2
                and candidate['largest_attack_share'] <= .8
                and candidate_rejection_rate <= .05
                and candidate['safety_stops'] == candidate['incomplete'] == 0)
    report = dict(checkpoint=str(path), seeds=list(seeds), spacings=list(spacings),
                  opponents=list(opponents),
                  source_unchanged=unchanged, baseline=baseline, candidate=candidate,
                  candidate_rejection_rate=candidate_rejection_rate,
                  inference_temperature=config.get('tactics', {}).get('inference_temperature', .5),
                  promoted=promoted)
    output = path.with_name(path.stem + '_evaluation.json')
    output.write_text(json.dumps(report, indent=2) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description='Compare tactical PPO with reactive boxing')
    parser.add_argument('checkpoint')
    parser.add_argument('--seeds', type=int, nargs='+', default=[20000, 20001, 20002])
    args = parser.parse_args()
    result = evaluate_checkpoint(args.checkpoint, args.seeds)
    print(f"Held-out touch difference: reactive {result['baseline']['mean_touch_difference']:.2f}, "
          f"candidate {result['candidate']['mean_touch_difference']:.2f}; promoted={result['promoted']}")


if __name__ == '__main__':
    main()
