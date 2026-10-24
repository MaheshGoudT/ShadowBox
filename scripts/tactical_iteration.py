#!/usr/bin/env python3
"""Audit boxing actions, then train and evaluate only if the audit passes."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

import yaml

from shadowbox.boxing import Parameters, Policy
from shadowbox.config import PROJECT_ROOT, artifact_path, load_config, source_fingerprint
from shadowbox.evaluation.diagnostics import diagnose
from shadowbox.rl.actions import ACTIONS
from shadowbox.rl.environment import BoxingEnv
from shadowbox.rl.evaluate import evaluate_checkpoint
from shadowbox.rl.opponents import GuardOpponent
from shadowbox.rl.ppo import train


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


def _opponent(config, kind, side, seed):
    if kind == 'reactive':
        return None
    other = 'b' if side == 'a' else 'a'
    level = {'head_guard': 'head', 'body_guard': 'body'}[kind]
    return GuardOpponent(seed + 1009, Parameters(**config['policies'][other]),
                         config['policy_rules'], level)


def _diverse_examples(records, limit):
    """Keep measured successes across sides, range, opponents and sources."""
    remaining = list(records)
    chosen = []
    coverage = Counter()
    facets = ('side', 'spacing', 'opponent', 'source')
    while remaining and len(chosen) < limit:
        index = min(range(len(remaining)), key=lambda position: (
            max(coverage[(facet, remaining[position][facet])] for facet in facets),
            sum(coverage[(facet, remaining[position][facet])] for facet in facets),
            position))
        record = remaining.pop(index)
        chosen.append(record)
        coverage.update((facet, record[facet]) for facet in facets)
    return chosen


def audit(config, settings, prefix):
    """Try every legal primitive from fresh, reproducible starting states."""
    if settings.get('max_examples_per_action', 16) < settings['examples_per_action']:
        raise ValueError('max_examples_per_action must cover examples_per_action')
    env = BoxingEnv(config)
    records = []
    for seed in settings['seeds']:
        for spacing in settings['spacings']:
            for side in ('a', 'b'):
                for kind in settings['opponents']:
                    for index, primitive in enumerate(ACTIONS):
                        options = dict(side=side, spacing=spacing,
                                       proximity_fallback=False)
                        opponent = _opponent(config, kind, side, seed)
                        if opponent is not None:
                            options['opponent_policy'] = opponent
                        state, reset_info = env.reset(seed, options)
                        if not reset_info['action_mask'][index]:
                            continue
                        before_distance = env.match.observe(side).distance
                        _, reward, done, truncated, info = env.step(index, trace=True)
                        after_distance = env.match.observe(side).distance
                        record = dict(seed=seed, side=side, spacing=spacing,
                                      opponent=kind, source='probe',
                                      action=primitive.value, index=index,
                                      observation=state.tolist(),
                                      mask=reset_info['action_mask'].tolist(),
                                      reward=reward, accepted=info['action_seen'] and not info['rejected'],
                                      touches=info['own']['touches'], blocks=info['own']['blocks'],
                                      misses=info['own']['misses'],
                                      distance_change=before_distance - after_distance,
                                      rejection_reasons=info['rejection_reasons'],
                                      trace=info['trace'],
                                      seconds=info['seconds'], stopped=env.match.supervisor.stopped,
                                      terminated=done, truncated=truncated)
                        records.append(record)
                    # Full bouts expose defensive timing that a reset-only probe cannot.
                    options = dict(side=side, spacing=spacing, proximity_fallback=False)
                    opponent = _opponent(config, kind, side, seed)
                    if opponent is not None:
                        options['opponent_policy'] = opponent
                    state, reset_info = env.reset(seed + 50000, options)
                    teacher = Policy(seed, Parameters(**config['policies'][side]),
                                     rules=config['policy_rules'])
                    for _ in range(100):
                        chosen = teacher.choose(env.match.observe(side))
                        index = ACTIONS.index(chosen) if chosen in ACTIONS else 0
                        mask = reset_info['action_mask']
                        if not mask[index]:
                            index = 2 if mask[2] else int(mask.nonzero()[0][0])
                        before_distance = env.match.observe(side).distance
                        next_state, reward, done, truncated, info = env.step(index, trace=True)
                        record = dict(seed=seed + 50000, side=side, spacing=spacing,
                                      opponent=kind, source='teacher',
                                      action=ACTIONS[index].value, index=index,
                                      observation=state.tolist(), mask=mask.tolist(),
                                      reward=reward, accepted=info['action_seen'] and not info['rejected'],
                                      touches=info['own']['touches'], blocks=info['own']['blocks'],
                                      misses=info['own']['misses'],
                                      distance_change=before_distance - env.match.observe(side).distance,
                                      rejection_reasons=info['rejection_reasons'],
                                      trace=info['trace'],
                                      seconds=info['seconds'], stopped=env.match.supervisor.stopped,
                                      terminated=done, truncated=truncated)
                        records.append(record)
                        state, reset_info = next_state, info
                        if done or truncated or env.match.arena.time >= settings['demo_seconds']:
                            break
    log_path = prefix / 'audit.jsonl'
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open('w') as stream:
        for record in records:
            stream.write(json.dumps(record) + '\n')

    qualified = defaultdict(list)
    for record in records:
        action = record['action']
        if not record['accepted'] or record['stopped'] or record['truncated']:
            continue
        if action in ('jab', 'cross', 'uppercut') and record['touches']:
            qualified[action].append(record)
        elif action in ('guard', 'body_guard') and record['blocks']:
            qualified[action].append(record)
    required = settings['required_examples']
    count = min((len(qualified[action]) for action in required), default=0)
    selected = {action: _diverse_examples(qualified[action],
                settings.get('max_examples_per_action', 16)) for action in required}
    examples = [record for action in required for record in selected[action]]
    _write_json(prefix / 'demonstrations.json', examples)

    rejections = Counter()
    action_counts = Counter()
    for record in records:
        action_counts[record['action']] += 1
        rejections.update(record['rejection_reasons'])
    rejected = sum(not record['accepted'] for record in records)
    rejection_rate = rejected / max(len(records), 1)
    passed = (bool(records) and rejection_rate <= settings['max_rejection_rate']
              and count >= settings['examples_per_action']
              and not any(record['stopped'] for record in records))
    report = dict(passed=passed, source_fingerprint=source_fingerprint(),
                  audit_config=settings, cases=len(records), rejected=rejected,
                  rejection_rate=rejection_rate,
                  rejection_reasons=dict(rejections), actions=dict(action_counts),
                  qualified={action: len(qualified[action]) for action in required},
                  selected={action: len(selected[action]) for action in required},
                  demonstration_examples=len(examples), examples_per_action=count,
                  audit_log=str(log_path), demonstrations=str(prefix / 'demonstrations.json'))
    _write_json(prefix / 'audit-summary.json', report)
    return report


def run(audit_settings, train_settings, eval_settings, sim_settings, stage):
    prefix = artifact_path(audit_settings['output_dir'])
    config = load_config(PROJECT_ROOT / 'configs' / 'unitree.yaml')
    if stage in ('sim', 'all'):
        result = diagnose(config, sim_settings['seeds'], sim_settings['seconds'])
        path = artifact_path(sim_settings['output'])
        _write_json(path, result)
        print(f"Simulation: {result['attempts']} punches, {result['touches']} touches, "
              f"{result['safety_stops']} stops, {result['grounded_foot_slip_m']:.2f} m foot slip")
        print(f'Report: {path}')
    if stage in ('audit', 'all'):
        report = audit(config, audit_settings, prefix)
        print(f"Audit: {report['cases']} legal cases, {report['rejected']} rejected, "
              f"qualified={report['qualified']}, passed={report['passed']}")
        print(f"Report: {prefix / 'audit-summary.json'}")
        if not report['passed']:
            return 2
    if stage in ('train', 'all'):
        report_path = prefix / 'audit-summary.json'
        if not report_path.is_file():
            print('Run the audit before training.', file=sys.stderr)
            return 2
        audit_report = json.loads(report_path.read_text())
        if (not audit_report['passed']
                or audit_report['source_fingerprint'] != source_fingerprint()
                or audit_report['audit_config'] != audit_settings):
            print('Run a fresh passing audit with the current code and config.', file=sys.stderr)
            return 2
        if set(eval_settings['seeds']) & (set(audit_settings['seeds'])
                                          | {seed + 50000 for seed in audit_settings['seeds']}
                                          | set(range(train_settings['seed'], train_settings['seed'] + train_settings['episodes']))):
            print('Evaluation seeds must be separate from audit and training seeds.', file=sys.stderr)
            return 2
        checkpoint = artifact_path(train_settings['checkpoint'])
        train(config, episodes=train_settings['episodes'],
              seconds=train_settings['seconds'], seed=train_settings['seed'],
              output=checkpoint, demonstrations_path=prefix / 'demonstrations.json')
        result = evaluate_checkpoint(checkpoint, seeds=eval_settings['seeds'],
                                     spacings=eval_settings['spacings'],
                                     opponents=eval_settings['opponents'])
        print(f"Held-out baseline {result['baseline']['mean_touch_difference']:.2f}, "
              f"candidate {result['candidate']['mean_touch_difference']:.2f}; "
              f"promoted={result['promoted']}")
        print(f"Report: {checkpoint.with_name(checkpoint.stem + '_evaluation.json')}")
        return 0 if result['promoted'] else 3
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit-config', default='configs/tactical-audit.yaml')
    parser.add_argument('--train-config', default='configs/tactical-train.yaml')
    parser.add_argument('--eval-config', default='configs/tactical-eval.yaml')
    parser.add_argument('--sim-config', default='configs/simulation-diagnostics.yaml')
    parser.add_argument('--stage', choices=('sim', 'audit', 'train', 'all'), default='all')
    args = parser.parse_args()
    settings = []
    for name in ('audit_config', 'train_config', 'eval_config', 'sim_config'):
        path = artifact_path(getattr(args, name))
        value = yaml.safe_load(path.read_text())
        if not isinstance(value, dict):
            parser.error(f'{path} must contain a mapping')
        settings.append(value)
    return run(*settings, args.stage)


if __name__ == '__main__':
    raise SystemExit(main())
