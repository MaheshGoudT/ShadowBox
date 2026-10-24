"""Diagnostic active-threat gauntlet, separate from checkpoint promotion."""

import argparse
from copy import deepcopy
import json
import math

import torch

from ..config import artifact_path, source_fingerprint
from .actions import ACTIONS
from .evaluate import _play, _summary
from .model import ActorCritic
from .observations import OBSERVATION_SIZE


def evaluate_defense(checkpoint_path, seeds=(44000, 44001), spacings=(.58, .62), seconds=8):
    if not seeds or not spacings or not math.isfinite(seconds) or seconds <= 0:
        raise ValueError('Defense evaluation needs seeds, spacings and positive seconds')
    path = artifact_path(checkpoint_path)
    saved = torch.load(path, map_location='cpu', weights_only=True)
    if (saved['actions'] != [action.value for action in ACTIONS]
            or saved['observation_shape'] != OBSERVATION_SIZE):
        raise ValueError('Tactical checkpoint schema changed')
    if set(seeds) & set(range(saved['seed'], saved['seed'] + saved['episodes'])):
        raise ValueError('Defense seeds overlap training seeds')
    config = deepcopy(saved['config'])
    config['match']['round_seconds'] = seconds
    config['scoring']['proximity_fallback'] = False
    model = ActorCritic()
    model.load_state_dict(saved['model'])
    model.eval()
    cases = [(seed, side, distance, threat) for seed in seeds for side in ('a', 'b')
             for distance in spacings for threat in ('head_attack', 'body_attack')]
    report = dict(checkpoint=str(path), seeds=list(seeds), spacings=list(spacings),
                  seconds=seconds, diagnostic_only=True, promoted=False,
                  source_unchanged=saved['source_fingerprint'] == source_fingerprint(),
                  current_source_fingerprint=source_fingerprint())
    for label, actor in (('baseline', None), ('candidate', model)):
        matches = [_play(config, actor, *case) for case in cases]
        per_threat = {}
        for threat in ('head_attack', 'body_attack'):
            subset = [match for match in matches if match['opponent_kind'] == threat]
            summary = _summary(subset)
            summary['threat_coverage'] = sum(m['opponent']['attempts'] > 0 for m in subset) / len(subset)
            summary['conceded_per_attempt'] = summary['conceded_touches'] / max(summary['opponent_attempts'], 1)
            summary['guard_block_per_attempt'] = summary['guard_blocks'] / max(summary['opponent_attempts'], 1)
            per_threat[threat] = summary
        report[label] = dict(summary=_summary(matches), per_threat=per_threat)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint')
    parser.add_argument('--seeds', type=int, nargs='+', default=[44000, 44001])
    parser.add_argument('--spacings', type=float, nargs='+', default=[.58, .62])
    parser.add_argument('--seconds', type=float, default=8)
    parser.add_argument('--output', default='tmp/tactical-iteration/active-defense.json')
    args = parser.parse_args()
    output = artifact_path(args.output)
    if output.exists():
        parser.error(f'Report exists; choose a new output: {output}')
    report = evaluate_defense(args.checkpoint, args.seeds, args.spacings, args.seconds)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + '\n')
    for label in ('baseline', 'candidate'):
        for threat, metrics in report[label]['per_threat'].items():
            print(f"{label} {threat}: {metrics['opponent_attempts']} incoming attempts, "
                  f"{metrics['conceded_touches']} conceded, {metrics['guard_blocks']} guard blocks, "
                  f"coverage={metrics['threat_coverage']:.0%}")
    print(f'Diagnostic only; source_unchanged={report["source_unchanged"]}; {output}')


if __name__ == '__main__':
    main()
