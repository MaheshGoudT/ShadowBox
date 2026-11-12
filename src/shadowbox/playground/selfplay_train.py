"""Iterative RSL-RL training against frozen ShadowBox policy snapshots."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / 'src'))


def _resolved_options(settings):
    options = dict(settings.get('task_options', {}))
    for key in ('motion_bundle', 'motion_script', 'opponent_policy_script'):
        if options.get(key):
            path = Path(options[key]).expanduser()
            options[key] = str((path if path.is_absolute() else PROJECT_ROOT / path).resolve())
    return options


def next_guard_width(current, target, legal_head_contacts, legal_body_contacts):
    """Advance one curriculum stage only after an observed legal touch."""
    return target if legal_head_contacts + legal_body_contacts > 0 else current


def next_approach_start_distance(current, target, step, gate_passed):
    """Increase boxing range only after both sides pass the contact gate."""
    if (not all(isinstance(value, (int, float)) and not isinstance(value, bool)
                and math.isfinite(value) for value in (current, target, step))
            or not .5 <= current <= target <= 2.0 or not 0. <= step <= .4):
        raise ValueError('Approach start-distance curriculum values are invalid')
    return min(target, current + step) if gate_passed else current


def curriculum_guard_step(start, target, rounds, configured_step=None):
    """Use a fixed curriculum increment when retries need not change stage size."""
    if configured_step is None:
        return (start - target) / max(1, rounds - 1)
    if not .001 <= configured_step <= .4:
        raise ValueError('Configured guard-width step must be in [.001, .4]')
    return configured_step


def training_entropy_coef(configured, default):
    """Resolve a categorical exploration coefficient from self-play config."""
    value = default if configured is None else configured
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or not 0. <= value <= .1):
        raise ValueError('Self-play entropy_coef must be finite and in [0, 0.1]')
    return float(value)


def both_sides_touched(results):
    """Require a legal touch from both controlled roles before advancing."""
    return all(side in results and
               results[side]['legal_head_contacts'] + results[side]['legal_body_contacts'] > 0
               for side in ('a', 'b'))


def controlled_side_for_round(run_round, global_round, alternate=True,
                              configured_side='a', previous_side=None):
    """Keep role alternation correct when resuming an older qualified candidate."""
    if not alternate:
        return configured_side
    if previous_side in ('a', 'b'):
        first_side = 'b' if previous_side == 'a' else 'a'
        return first_side if run_round % 2 == 0 else (
            'a' if first_side == 'b' else 'b')
    return 'a' if global_round % 2 == 0 else 'b'


def _resume_run_state(run_path, opponent_pool_size):
    """Load the latest verified candidate and its frozen league snapshots."""
    run_path = Path(run_path).expanduser().resolve()
    tmp_root = (PROJECT_ROOT / 'tmp').resolve()
    if not run_path.is_relative_to(tmp_root):
        raise ValueError('Resume run must remain inside tmp/')
    league_path = run_path / 'league.json'
    if not league_path.is_file():
        raise ValueError(f'Resume run has no league.json: {run_path}')
    league = json.loads(league_path.read_text())
    rounds = league.get('rounds') if isinstance(league, dict) else None
    if not isinstance(rounds, list) or not rounds:
        raise ValueError('Resume league must contain at least one completed round')
    qualified_indices = [index for index, record in enumerate(rounds)
                         if record.get('curriculum_gate_passed') is True
                         or (isinstance(record.get('curriculum_evaluation'), dict)
                             and both_sides_touched(
                                 record['curriculum_evaluation'].get('by_side', {})))]
    selected_index = qualified_indices[-1] if qualified_indices else len(rounds) - 1
    selected = rounds[selected_index]
    eligible_rounds = rounds[:selected_index + 1]

    def checkpoint_for(record):
        value = Path(record.get('checkpoint', ''))
        path = (value if value.is_absolute() else PROJECT_ROOT / value).resolve()
        if not path.is_relative_to(tmp_root) or not path.is_file():
            raise ValueError(f'Resume checkpoint is missing or outside tmp/: {path}')
        return path

    checkpoint = checkpoint_for(selected)
    expected_hash = selected.get('checkpoint_sha256')
    if expected_hash and hashlib.sha256(checkpoint.read_bytes()).hexdigest() != expected_hash:
        raise ValueError('Resume candidate hash does not match league.json')
    snapshots = []
    for record in eligible_rounds[-opponent_pool_size:]:
        source = checkpoint_for(record)
        snapshot = source.parent / 'snapshot' / 'opponent.pt'
        if not snapshot.is_file():
            raise ValueError(f'Resume opponent snapshot is missing: {snapshot}')
        snapshots.append((int(record['round']), snapshot))
    next_width = selected.get('next_opponent_guard_width')
    if next_width is not None and (type(next_width) not in (int, float)
                                   or not .10 <= next_width <= .50):
        raise ValueError('Resume league has an invalid next opponent guard width')
    next_start_distance = selected.get('next_start_distance')
    if (next_start_distance is not None
            and (type(next_start_distance) not in (int, float)
                 or not .5 <= next_start_distance <= 2.0)):
        raise ValueError('Resume league has an invalid next approach start distance')
    return dict(run_path=run_path, checkpoint=checkpoint, snapshots=snapshots,
                rounds=rounds, selected_round=selected.get('round'),
                selected_controlled_side=selected.get('controlled_side'),
                next_guard_width=next_width,
                next_start_distance=next_start_distance,
                prior_round_count=len(rounds))


def _write_league(run_dir, task, rounds, resume_state, last_qualified_checkpoint,
                  last_qualified_next_width, opponent_pool_size):
    """Persist a complete league after every finished round for safe continuation."""
    payload = dict(task=task, rounds=rounds,
                   resumed_from=str(resume_state['run_path']) if resume_state else None,
                   last_qualified_checkpoint=(
                       str(last_qualified_checkpoint.relative_to(PROJECT_ROOT))
                       if last_qualified_checkpoint is not None else None),
                   last_qualified_next_guard_width=last_qualified_next_width,
                   initial_opponent_snapshot_rounds=(
                       [round_index for round_index, _ in resume_state['snapshots']]
                       if resume_state else []),
                   opponent_pool_size=opponent_pool_size,
                   qualification='candidate generation only; no held-out promotion decision')
    path = run_dir / 'league.json'
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(payload, indent=2) + '\n')
    temporary.replace(path)


def _evaluate_candidate(actor, env, raw, motion, steps, seed, controlled_side):
    """Measure scripted-opponent contacts at the current curriculum width."""
    import torch
    from mujoco_playground._src.rl import rng

    rng.seed(seed)
    torch.manual_seed(seed)
    motion.set_opponent_pool([])
    head_channel, body_channel = ((3, 4) if controlled_side == 'a' else (0, 1))
    head_contacts = body_contacts = completed_episodes = 0
    with torch.inference_mode():
        env.reset()
        env.episode_length_buf.zero_()
        observations = env.get_observations()
        for _ in range(steps):
            observations, _, done, _ = env.step(actor(observations))
            if done.any():
                completed_episodes += int(done.sum().item())
                continue
            events = raw._latest_features['boxing/events']
            head_contacts += int(events[:, head_channel].sum().item())
            body_contacts += int(events[:, body_channel].sum().item())
    return dict(steps=steps, seed=seed, controlled_side=controlled_side,
                legal_head_contacts=head_contacts,
                legal_body_contacts=body_contacts,
                completed_episodes=completed_episodes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--rounds', type=int, required=True)
    parser.add_argument('--updates-per-round', type=int, required=True)
    parser.add_argument('--opponent-pool-size', type=int, required=True)
    parser.add_argument('--rollout-steps', type=int, required=True)
    parser.add_argument('--curriculum-eval-steps', type=int, required=True)
    parser.add_argument('--resume-run', type=Path)
    parser.add_argument('--resume-guard-width', type=float)
    args = parser.parse_args()
    if min(args.rounds, args.updates_per_round, args.opponent_pool_size,
           args.rollout_steps, args.curriculum_eval_steps) < 1 or args.rollout_steps > 1024:
        parser.error('rounds, updates, pool size and rollout steps must be in range')
    if args.resume_guard_width is not None and not .10 <= args.resume_guard_width <= .50:
        parser.error('resume guard width must be in [.10, .50]')
    if args.resume_guard_width is not None and args.resume_run is None:
        parser.error('--resume-guard-width requires --resume-run')
    config = yaml.safe_load(args.config.read_text())
    settings = config['playground']
    if settings['task'] != 'ShadowBoxG1SelfPlay':
        parser.error('self-play requires task: ShadowBoxG1SelfPlay')
    if settings['num_envs'] != 1:
        parser.error('Tracker-backed self-play currently supports one paired world')

    import dataclasses
    import torch
    from mujoco_playground._src.rl import env_registry, rng
    from mujoco_playground._src.rl.rsl_rl.vecenv_wrapper import RslRlVecEnvWrapper
    from rsl_rl.runners import OnPolicyRunner
    from shadowbox.playground.registry import register_tasks

    register_tasks()
    entry = env_registry.get('ShadowBoxG1SelfPlay')
    base_options = _resolved_options(settings)
    self_play = settings.get('self_play', {})
    try:
        entropy_coef = training_entropy_coef(
            self_play.get('entropy_coef'), entry.runner_cfg.algorithm.entropy_coef)
    except ValueError as error:
        parser.error(str(error))
    resume_state = None
    if args.resume_run is not None:
        try:
            resume_state = _resume_run_state(args.resume_run, args.opponent_pool_size)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            parser.error(str(error))
    guard_start = float(
        args.resume_guard_width if args.resume_guard_width is not None else
        resume_state['next_guard_width'] if resume_state and resume_state['next_guard_width'] is not None else
        base_options.get('opponent_guard_width', .10))
    guard_end = float(self_play.get('opponent_guard_width_end', guard_start))
    if guard_start < guard_end:
        parser.error('initial opponent guard width must be at least the curriculum target')
    guard_step = curriculum_guard_step(
        guard_start, guard_end, args.rounds,
        self_play.get('opponent_guard_width_step'))
    current_guard_width = guard_start
    approach_target_distance = float(base_options.get('start_distance', .6))
    approach_start_distance = float(
        resume_state.get('next_start_distance') if resume_state and
        resume_state.get('next_start_distance') is not None else
        self_play.get('approach_start_distance', approach_target_distance))
    approach_distance_step = float(self_play.get('approach_start_distance_step', 0.))
    if not .5 <= approach_start_distance <= approach_target_distance <= 2.:
        parser.error('approach start distance must be within [.5, task_options.start_distance]')
    if approach_distance_step and 'opponent_guard_width_end' not in self_play:
        parser.error('approach distance curriculum requires opponent guard width evaluation gates')
    run_dir = args.run_dir.resolve()
    if not run_dir.is_relative_to((PROJECT_ROOT / 'tmp').resolve()):
        parser.error('self-play run directory must remain inside tmp/')
    run_dir.mkdir(parents=True, exist_ok=True)
    opponent_pool = []
    opponent_pool_rounds = []
    previous_checkpoint = None
    first_global_round = 0
    if resume_state:
        previous_checkpoint = resume_state['checkpoint']
        first_global_round = resume_state['prior_round_count']
        for previous_round, snapshot_path in resume_state['snapshots']:
            opponent_pool.append(torch.jit.load(
                str(snapshot_path), map_location=settings['device']).eval())
            opponent_pool_rounds.append(previous_round)
    last_qualified_checkpoint = previous_checkpoint
    last_qualified_next_width = (resume_state['next_guard_width'] if resume_state else None)
    rounds = list(resume_state['rounds']) if resume_state else []

    for round_index in range(args.rounds):
        global_round_index = first_global_round + round_index
        options = dict(base_options)
        options['start_distance'] = approach_start_distance
        if 'opponent_guard_width_end' in self_play:
            options['opponent_guard_width'] = current_guard_width
        training_side = controlled_side_for_round(
            round_index, global_round_index,
            alternate=self_play.get('alternate_controlled_side', False),
            configured_side=base_options.get('controlled_side', 'a'),
            previous_side=(resume_state.get('selected_controlled_side')
                           if resume_state else None))
        options['controlled_side'] = training_side
        seed = settings['seed'] + global_round_index
        rng.seed(seed)
        torch.manual_seed(seed)
        round_dir = run_dir / f'round-{round_index:03d}'
        round_dir.mkdir(parents=True, exist_ok=False)
        raw = entry.env_builder(nworld=1, device=settings['device'], **options)
        try:
            env = RslRlVecEnvWrapper(raw, cfg=entry.runner_cfg.vecenv_wrapper_cfg)
            runner_cfg = dataclasses.asdict(entry.runner_cfg)
            runner_cfg['nworld'] = 1
            runner_cfg['max_iterations'] = args.updates_per_round
            runner_cfg['num_steps_per_env'] = args.rollout_steps
            runner_cfg['seed'] = seed
            runner_cfg['torch_compile_mode'] = None
            runner_cfg['algorithm']['entropy_coef'] = entropy_coef
            runner = OnPolicyRunner(env, runner_cfg, log_dir=str(round_dir), device=settings['device'])
            training_source_checkpoint = previous_checkpoint
            if previous_checkpoint is not None:
                runner.load(str(previous_checkpoint), map_location=settings['device'])
            motion = raw.task_layer.get_component_by_name('motion')
            round_opponent_pool = opponent_pool[-args.opponent_pool_size:]
            opponent_pool_start_index = (global_round_index % (len(round_opponent_pool) + 1)
                                         if round_opponent_pool else 0)
            if opponent_pool:
                motion.set_opponent_pool(
                    round_opponent_pool, start_index=opponent_pool_start_index)
                env.reset()
                env.episode_length_buf.zero_()
            runner.learn(args.updates_per_round, init_at_random_ep_len=True)
            checkpoint = round_dir / 'candidate.pt'
            runner.save(str(checkpoint))
            snapshot_dir = round_dir / 'snapshot'
            runner.export_policy_to_jit(str(snapshot_dir), 'opponent.pt')
            snapshot_path = snapshot_dir / 'opponent.pt'
            snapshot = torch.jit.load(str(snapshot_path), map_location=settings['device']).eval()
            digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            snapshot_rounds = list(opponent_pool_rounds[-args.opponent_pool_size:])
            curriculum_eval = None
            next_width = current_guard_width
            next_start_distance = approach_start_distance
            gate_passed = None
            if 'opponent_guard_width_end' in self_play:
                actor = runner.get_inference_policy(device=settings['device'])
                by_side = {}
                for side_index, side in enumerate(('a', 'b')):
                    if side == training_side:
                        eval_env, eval_raw, eval_motion = env, raw, motion
                    else:
                        side_options = dict(options, controlled_side=side)
                        eval_raw = entry.env_builder(
                            nworld=1, device=settings['device'], **side_options)
                        eval_env = RslRlVecEnvWrapper(
                            eval_raw, cfg=entry.runner_cfg.vecenv_wrapper_cfg)
                        eval_motion = eval_raw.task_layer.get_component_by_name('motion')
                    try:
                        by_side[side] = _evaluate_candidate(
                            actor, eval_env, eval_raw, eval_motion,
                            args.curriculum_eval_steps,
                            settings['seed'] + 10000 + global_round_index * 2 + side_index,
                            side)
                    finally:
                        if side != training_side:
                            eval_raw.close()
                curriculum_eval = dict(steps=args.curriculum_eval_steps,
                                      by_side=by_side)
                target_width = max(guard_end, current_guard_width - guard_step)
                gate_passed = both_sides_touched(by_side)
                next_width = next_guard_width(
                    current_guard_width, target_width, int(gate_passed), 0)
                next_start_distance = next_approach_start_distance(
                    approach_start_distance, approach_target_distance,
                    approach_distance_step, gate_passed)
            opponent_pool.append(snapshot)
            opponent_pool_rounds.append(global_round_index)
            if gate_passed is True:
                last_qualified_checkpoint = checkpoint
                last_qualified_next_width = next_width
            # Keep learning from the latest candidate at a failed stage while
            # preserving the last qualified checkpoint separately for resumes.
            previous_checkpoint = checkpoint
            record = dict(round=global_round_index, run_round=round_index,
                          controlled_side=training_side, seed=seed,
                          updates=args.updates_per_round,
                          initialized_from_checkpoint=(
                              str(training_source_checkpoint.relative_to(PROJECT_ROOT))
                              if training_source_checkpoint is not None else None),
                          checkpoint=str(checkpoint.relative_to(PROJECT_ROOT)),
                          checkpoint_sha256=digest,
                          opponent_snapshot_rounds=snapshot_rounds,
                          opponent_snapshots_used=len(snapshot_rounds),
                          opponent_pool_start_index=opponent_pool_start_index,
                          entropy_coef=entropy_coef,
                          boxing_reward_weights={key: options[key] for key in (
                              'head_contact_reward', 'body_contact_reward',
                              'received_head_penalty', 'received_body_penalty',
                              'block_reward', 'approach_progress_reward') if key in options},
                          opponent_guard_width=options.get('opponent_guard_width', .10),
                          start_distance=approach_start_distance,
                          next_start_distance=next_start_distance,
                          approach_target_range_m=options.get('approach_target_range_m', .65),
                          approach_speed_a=options.get('approach_speed_a', .27),
                          approach_speed_b=options.get('approach_speed_b', .27),
                          curriculum_evaluation=curriculum_eval,
                          curriculum_gate_passed=gate_passed,
                          next_opponent_guard_width=next_width,
                          scripted_opponent=bool(options.get('opponent_enabled', True)),
                          status=('curriculum gate passed; held-out match evaluation required before promotion'
                                  if gate_passed is True else
                                  'curriculum gate failed; candidate remains diagnostic and width is held'
                                  if gate_passed is False else
                                  'trained; held-out match evaluation required before promotion'))
            (round_dir / 'round.json').write_text(json.dumps(record, indent=2) + '\n')
            rounds.append(record)
            _write_league(run_dir, settings['task'], rounds, resume_state,
                          last_qualified_checkpoint, last_qualified_next_width,
                          args.opponent_pool_size)
            print(f"round {round_index + 1}/{args.rounds}: {checkpoint} sha256={digest}", flush=True)
            if curriculum_eval is not None:
                contact_summary = ', '.join(
                    f"{side}=H{values['legal_head_contacts']}/B{values['legal_body_contacts']}"
                    for side, values in by_side.items())
                print(f"curriculum contacts: {contact_summary}; "
                      f"next opponent guard width={next_width:.3f} m", flush=True)
                current_guard_width = next_width
                print(f"approach start distance={approach_start_distance:.3f} m; "
                      f"next={next_start_distance:.3f} m", flush=True)
            approach_start_distance = next_start_distance
        finally:
            raw.close()

    _write_league(run_dir, settings['task'], rounds, resume_state,
                  last_qualified_checkpoint, last_qualified_next_width,
                  args.opponent_pool_size)


if __name__ == '__main__':
    main()
