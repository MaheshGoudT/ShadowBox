"""Run a held-out opponent, side, and starting-orientation matrix."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import time

import numpy as np
import yaml

from ..integrations.backend import run_backend
from ..integrations.playground import PlaygroundBackend

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def contact_totals(report, side):
    """Map referee event channels onto candidate and opponent totals."""
    values = report.get('contact_episode_counts_by_channel_and_glove')
    if values is None:
        return dict(candidate_head=0, candidate_body=0, opponent_head=0,
                    opponent_body=0, candidate_blocks=0,
                    candidate_attacks_blocked=0)
    values = [[float(value) for value in row] for row in values]
    if len(values) != 6 or any(len(row) != 2 for row in values):
        raise ValueError('Self-play evaluation report has an unexpected contact-channel shape')
    own = (3, 4, 2) if side == 'a' else (0, 1, 5)
    opponent = (0, 1) if side == 'a' else (3, 4)
    attacks_blocked = 5 if side == 'a' else 2
    return dict(candidate_head=sum(values[own[0]]),
                candidate_body=sum(values[own[1]]),
                opponent_head=sum(values[opponent[0]]),
                opponent_body=sum(values[opponent[1]]),
                candidate_blocks=sum(values[own[2]]),
                candidate_attacks_blocked=sum(values[attacks_blocked]))


def transition_action_counts(transitions, side):
    """Count sampled policy actions separately from executed skill goals."""
    side_index = 0 if side == 'a' else 1
    actions = np.asarray(transitions['action']).reshape(-1).astype(int)
    goals = np.asarray(transitions['requested_primitives'])[:, 0, side_index].astype(int)
    sampled = {str(action): int((actions == action).sum()) for action in range(10)
               if np.any(actions == action)}
    executed = {str(goal): int((goals == goal).sum()) for goal in range(9)
                if np.any(goals == goal)}
    return sampled, executed


def summarize_cases(results):
    """Aggregate completed held-out cases by opponent style or snapshot."""
    summaries = {}
    contact_keys = ('candidate_head', 'candidate_body', 'opponent_head',
                    'opponent_body', 'candidate_blocks', 'candidate_attacks_blocked')
    for result in results:
        name = result.get('opponent', 'unknown')
        summary = summaries.setdefault(name, dict(
            cases=0, completed_cases=0, failed_cases=0,
            candidate_contacts={key: 0 for key in contact_keys},
            sampled_policy_actions={}, executed_skill_goals={},
            completed_episodes=0, censored_episode_steps=0))
        summary['cases'] += 1
        if result.get('status') != 'completed':
            summary['failed_cases'] += 1
            continue
        summary['completed_cases'] += 1
        contacts = result.get('candidate_contacts', {})
        for key in contact_keys:
            summary['candidate_contacts'][key] += int(contacts.get(key, 0))
        for field in ('sampled_policy_actions', 'executed_skill_goals'):
            counts = summary[field]
            for action, count in result.get(field, {}).items():
                counts[action] = counts.get(action, 0) + int(count)
        completed = result.get('completed_episodes', 0)
        censored = result.get('censored_episode_steps', 0)
        summary['completed_episodes'] += sum(map(int, completed)) if isinstance(
            completed, list) else int(completed)
        summary['censored_episode_steps'] += sum(map(int, censored)) if isinstance(
            censored, list) else int(censored)

    for summary in summaries.values():
        contacts = summary['candidate_contacts']
        contacts['net_head_contacts'] = contacts['candidate_head'] - contacts['opponent_head']
        contacts['net_body_contacts'] = contacts['candidate_body'] - contacts['opponent_body']
        contacts['net_total_contacts'] = (
            contacts['candidate_head'] + contacts['candidate_body']
            - contacts['opponent_head'] - contacts['opponent_body'])
        summary['sampled_policy_actions'] = dict(sorted(
            summary['sampled_policy_actions'].items(), key=lambda item: int(item[0])))
        summary['executed_skill_goals'] = dict(sorted(
            summary['executed_skill_goals'].items(), key=lambda item: int(item[0])))
    return summaries


def evaluation_guard_width(settings, override=None):
    """Resolve the held-out opponent width independently of training target."""
    if override is not None:
        width = override
    else:
        width = settings.get('self_play', {}).get(
            'opponent_guard_width_end', settings.get('task_options', {}).get(
                'opponent_guard_width', .10))
    if isinstance(width, bool) or not isinstance(width, (int, float)) or not .10 <= width <= .50:
        raise ValueError('Evaluation opponent guard width must be in [.10, .50] m')
    return float(width)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--opponent', type=Path, action='append', default=[],
                        help='Frozen TorchScript opponent; repeat for more league members')
    parser.add_argument('--seeds', type=int, nargs='+', default=[101, 103])
    parser.add_argument('--steps', type=int, default=500)
    parser.add_argument('--opponent-guard-width', type=float,
                        help='held-out opponent guard width in metres (defaults to config target)')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.steps < 1 or not args.seeds or any(seed < 0 for seed in args.seeds):
        parser.error('steps must be positive and held-out seeds must be nonnegative')
    if len(set(args.seeds)) != len(args.seeds):
        parser.error('held-out seeds must be unique')

    config_path = args.config if args.config.is_absolute() else PROJECT_ROOT / args.config
    try:
        base = yaml.safe_load(config_path.read_text())
    except (OSError, yaml.YAMLError) as error:
        parser.error(f'Could not read configuration {config_path}: {error}')
    if not isinstance(base, dict) or not isinstance(base.get('playground'), dict):
        parser.error('Configuration must contain a playground section')
    settings = base['playground']
    if settings.get('task') != 'ShadowBoxG1SelfPlay':
        parser.error('Evaluation requires task: ShadowBoxG1SelfPlay')
    try:
        guard_width = evaluation_guard_width(settings, args.opponent_guard_width)
    except ValueError as error:
        parser.error(str(error))

    candidate = args.candidate if args.candidate.is_absolute() else PROJECT_ROOT / args.candidate
    candidate = candidate.resolve()
    if not candidate.is_file():
        parser.error(f'Candidate checkpoint does not exist: {candidate}')
    opponents = []
    for index, value in enumerate(args.opponent, start=1):
        path = value if value.is_absolute() else PROJECT_ROOT / value
        path = path.resolve()
        if not path.is_file():
            parser.error(f'Opponent snapshot does not exist: {path}')
        opponents.append((f'opponent-{index:02d}-{path.stem}', path))

    opponent_cases = [('scripted', None), *opponents]
    config_dir = PROJECT_ROOT / 'tmp/playground/selfplay-eval-configs'
    config_dir.mkdir(parents=True, exist_ok=True)
    cases_root = PROJECT_ROOT / 'tmp/playground/selfplay-eval-cases' / (
        f'{candidate.stem}-{time.time_ns()}')
    cases_root.mkdir(parents=True, exist_ok=False)
    results = []
    failed = False
    for opponent_name, opponent in opponent_cases:
        for seed in args.seeds:
            for side in ('a', 'b'):
                for swapped in (False, True):
                    label = f'{opponent_name}-s{seed}-{side}-swap{int(swapped)}'
                    safe = re.sub(r'[^A-Za-z0-9_.-]+', '_', label)
                    case_root = cases_root / safe
                    case_root.mkdir(parents=True, exist_ok=False)
                    config = dict(base)
                    config['playground'] = dict(settings)
                    current = config['playground']
                    current['checkpoint'] = str(candidate)
                    current['evaluation_seed'] = seed
                    current['smoke_steps'] = args.steps
                    current['output_dir'] = str(case_root)
                    current['gif'] = dict(current.get('gif', {}), enabled=False)
                    options = dict(current.get('task_options', {}))
                    options['controlled_side'] = side
                    options['swap_start_sides'] = swapped
                    options['opponent_guard_width'] = guard_width
                    if opponent is None:
                        options.pop('opponent_policy_script', None)
                    else:
                        options['opponent_policy_script'] = str(opponent)
                    current['task_options'] = options
                    config_path = config_dir / f'{safe}.yaml'
                    config_path.write_text(yaml.safe_dump(config, sort_keys=False))

                    try:
                        backend = PlaygroundBackend(config_path)
                        status = run_backend(backend, 'evaluate')
                        report_path = next(case_root.glob('run-*/evaluate.json'), None)
                        if status != 0 or report_path is None:
                            failed = True
                            result = dict(case=label, opponent=opponent_name, seed=seed,
                                          controlled_side=side, swapped_start=swapped,
                                          status='failed', exit_code=status)
                        else:
                            report = json.loads(report_path.read_text())
                            with np.load(report_path.parent / 'transitions.npz') as transitions:
                                sampled_actions, executed_goals = transition_action_counts(
                                    transitions, side)
                            result = dict(
                                case=label,
                                opponent=opponent_name,
                                seed=seed,
                                controlled_side=side,
                                swapped_start=swapped,
                                status='completed',
                                report=str(report_path.relative_to(PROJECT_ROOT)),
                                candidate_contacts=contact_totals(report, side),
                                completed_episodes=report['completed_episodes'],
                                censored_episode_steps=report['censored_episode_steps'],
                                sampled_policy_actions=sampled_actions,
                                executed_skill_goals=executed_goals,
                                mean_step_reward=report['mean_step_reward'],
                                transitions_per_second=report['transitions_per_second'],
                                checkpoint=report['checkpoint'])
                    except Exception as error:
                        failed = True
                        result = dict(case=label, opponent=opponent_name, seed=seed,
                                      controlled_side=side, swapped_start=swapped,
                                      status='failed', error=str(error))
                    results.append(result)

    output = args.output or PROJECT_ROOT / 'tmp/playground/selfplay-evaluation.json'
    output = output if output.is_absolute() else PROJECT_ROOT / output
    output = output.resolve()
    if not output.is_relative_to((PROJECT_ROOT / 'tmp').resolve()):
        parser.error('Evaluation reports must remain inside tmp/')
    report = dict(
        candidate=str(candidate),
        candidate_sha256=hashlib.sha256(candidate.read_bytes()).hexdigest(),
        steps=args.steps,
        seeds=args.seeds,
        sides=['a', 'b'],
        swapped_starts=[False, True],
        opponents=['scripted', *[str(path) for _, path in opponents]],
        evaluation_opponent_guard_width=guard_width,
        cases=results,
        summary_by_opponent=summarize_cases(results),
        qualification='held-out diagnostic only; contact metrics are not an official boxing score')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(output)
    if failed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
