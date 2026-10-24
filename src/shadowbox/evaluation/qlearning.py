"""CPU tabular Q learning over existing boxing primitives and observed state."""

import argparse
from copy import deepcopy
from dataclasses import asdict
import json
import math
from pathlib import Path

import numpy as np

from ..boxing import Match, Parameters, Policy
from ..config import artifact_path, configure_logging, load_config, source_fingerprint
from ..motion import Primitive
from ..rl.actions import ACTIONS, allowed_actions

LEGACY_ACTIONS = ACTIONS[:8]
DEFAULT_SETTINGS = dict(episodes=120, seconds=20, seed=0, alpha=0.15,
                        discount_per_second=0.97, epsilon_start=0.35, epsilon_end=0.05,
                        prior_bonus=0.05, time_cost=0.01, rejection_cost=0.05,
                        failure_cost=5.0, snapshot_every=20, pool_size=4,
                        validation_seeds=[20000, 20001, 20002],
                        selfplay_validation_seeds=[30000, 30001],
                        minimum_attempts_per_second=0.15,
                        idle_after_seconds=4.0, idle_cost=0.06,
                        approach_reward=0.75, far_time_cost=0.05,
                        output='tmp/policies/tabular.json', resume='')


def state_key(observation, distance_scale=1.0) -> str:
    delta = observation.opponent_position[:2] - observation.position[:2]
    bearing = math.atan2(delta[1], delta[0])
    error = abs(math.remainder(bearing - observation.yaw, 2 * math.pi))
    closing = float(np.dot(observation.relative_velocity[:2], delta) /
                    max(observation.distance, 1e-6))
    fields = (int(np.digitize(observation.distance, np.array([.68, .8, .95, 1.2, 1.8]) * distance_scale)),
              int(observation.incoming_speed > .25 * distance_scale),
              int(observation.boundary_space < .4 * distance_scale),
              int(error > .18), int(np.digitize(closing, [-.05, .05])))
    return ','.join(map(str, fields))


class TabularPolicy(Policy):
    """Learn action values; simulator supervision still approves every motion."""

    def __init__(self, seed=0, parameters=None, table=None, settings=None, learning=False,
                 epsilon=0.0, defense_only=False, motion=None, minimum_separation=.66,
                 rules=None):
        super().__init__(seed, parameters or Parameters(), defense_only, rules=rules)
        self.table = deepcopy(table or {})
        self.settings = {**DEFAULT_SETTINGS, **(settings or {})}
        self.learning, self.epsilon = learning, epsilon
        self.motion = motion or dict(step_distance=.22, sidestep_distance=.18)
        self.minimum_separation = minimum_separation
        self.previous = None
        self.reward = self.elapsed = 0.0
        self.updates = 0

    def allowed(self, observation) -> list[int]:
        return allowed_actions(observation, self.parameters, self.motion,
                               self.minimum_separation, self.rules, self.defense_only)

    def _values(self, key, observation, allowed):
        if key not in self.table:
            values = [0.0] * len(ACTIONS)
            suggested = super().choose(observation)
            index = ACTIONS.index(suggested) if suggested in ACTIONS else 0
            values[index if index in allowed else allowed[0]] = self.settings['prior_bonus']
            self.table[key] = values
        return self.table[key]

    def update(self, next_key=None, next_actions=None) -> None:
        if self.previous is None:
            return
        key, action = self.previous
        future = 0.0 if next_key is None else max(self.table[next_key][a] for a in next_actions)
        target = self.reward + self.settings['discount_per_second'] ** self.elapsed * future
        self.table[key][action] += self.settings['alpha'] * (target - self.table[key][action])
        self.updates += 1
        self.previous = None
        self.reward = self.elapsed = 0.0

    def choose(self, observation):
        key, allowed = state_key(observation, self.rules['distance_scale']), self.allowed(observation)
        if not self.learning and key not in self.table:
            suggested = super().choose(observation)
            index = ACTIONS.index(suggested) if suggested in ACTIONS else 0
            return ACTIONS[index if index in allowed else allowed[0]]
        values = self._values(key, observation, allowed)
        if self.learning:
            self.update(key, allowed)
        if self.learning and self.rng.random() < self.epsilon:
            action = int(self.rng.choice(allowed))
        else:
            action = max(allowed, key=lambda i: values[i])
        if self.learning:
            self.previous = key, action
        return ACTIONS[action]

    def add_reward(self, reward, seconds) -> None:
        if self.learning and self.previous is not None:
            self.reward += reward
            self.elapsed += seconds

    def finish(self) -> None:
        if self.learning:
            self.update()


def _episode(policy, side, opponent, seed, seconds, config, settings):
    other = 'b' if side == 'a' else 'a'
    match = Match(seed=seed, rounds=1, round_seconds=seconds, config=config)
    match.policies[side] = policy
    match.parameters[side] = policy.parameters
    if isinstance(opponent, TabularPolicy):
        match.policies[other] = opponent
        match.parameters[other] = opponent.parameters
    elif opponent is not None:
        match.parameters[other] = Parameters(**opponent)
        match.policies[other] = Policy(seed + 1009, match.parameters[other])
    previous_difference, event_cursor = 0, 0
    last_own_attempt = last_opponent_attempt = 0.0
    for _ in range(math.ceil((seconds + 1) / match.arena.model.opt.timestep)):
        if match.finished or match.supervisor.stopped:
            break
        before = match.arena.time
        before_distance = match.observe(side).distance
        match.step()
        difference = match.stats[side]['touches'] - match.stats[other]['touches']
        recent = match.events[event_cursor:]
        rejected = sum(e['event'] == 'rejected' and e['robot'] == side for e in recent)
        for event in recent:
            if event['event'] == 'action' and event.get('action') in ('jab', 'cross', 'uppercut'):
                if event['robot'] == side:
                    last_own_attempt = match.arena.time
                else:
                    last_opponent_attempt = match.arena.time
        duration = match.arena.time - before
        if isinstance(policy, TabularPolicy):
            distance = match.observe(side).distance
            engaging = distance <= 1.2 * policy.rules['distance_scale']
            idle = (engaging and match.arena.time - last_own_attempt > settings['idle_after_seconds']
                    and match.arena.time - last_opponent_attempt > 2.0)
            far = before_distance > policy.parameters.desired_distance + .08 * policy.rules['distance_scale']
            policy.add_reward(difference - previous_difference - settings['time_cost'] * duration
                              - settings['rejection_cost'] * rejected
                              - settings['idle_cost'] * duration * idle
                              - settings['far_time_cost'] * duration * far
                              + settings['approach_reward'] * (before_distance - distance) * far,
                              duration)
        previous_difference, event_cursor = difference, len(match.events)
    failed = match.supervisor.stopped or not match.finished
    if isinstance(policy, TabularPolicy):
        if failed:
            policy.add_reward(-settings['failure_cost'], 0)
        policy.finish()
    return dict(seed=seed, side=side, finished=match.finished, stopped=match.supervisor.stopped,
                own=dict(match.stats[side]), opponent=dict(match.stats[other]),
                touch_difference=previous_difference, reward=previous_difference - settings['failure_cost'] * failed,
                rejected=sum(e['event'] == 'rejected' and e['robot'] == side for e in match.events),
                attempts_per_second=match.stats[side]['attempts'] / max(match.arena.time, 1e-9),
                updates=policy.updates if isinstance(policy, TabularPolicy) else 0, time=match.arena.time)


def evaluate(table, parameters, opponents, seeds, seconds, config, settings):
    matches = []
    for opponent in opponents:
        for seed in seeds:
            for side in ('a', 'b'):
                policy = (Policy(seed, Parameters(**parameters)) if table is None else
                          TabularPolicy(seed, Parameters(**parameters), table, settings,
                                        motion=config['motion'], minimum_separation=config['safety']['min_separation'],
                                        rules=config.get('policy_rules')))
                matches.append(_episode(policy, side, opponent, seed, seconds, config, settings))
    return dict(mean_reward=float(np.mean([m['reward'] for m in matches])),
                attempts_per_second=float(np.mean([m['attempts_per_second'] for m in matches])),
                touches_for=sum(m['own']['touches'] for m in matches),
                touches_against=sum(m['opponent']['touches'] for m in matches),
                safety_stops=sum(m['stopped'] for m in matches),
                incomplete=sum(not m['finished'] for m in matches), matches=matches)


def evaluate_self_play(table, parameters, seeds, seconds, config, settings):
    """Reject learned policies that stop engaging each other."""
    matches = []
    for seed in seeds:
        match = Match(seed=seed, rounds=1, round_seconds=seconds, config=config)
        for index, side in enumerate(('a', 'b')):
            match.policies[side] = TabularPolicy(seed + 1009 * index,
                Parameters(**parameters), table, settings, motion=config['motion'],
                minimum_separation=config['safety']['min_separation'],
                rules=config.get('policy_rules'))
        for _ in range(math.ceil((seconds + 1) / match.arena.model.opt.timestep)):
            if match.finished or match.supervisor.stopped:
                break
            match.step()
        matches.append(dict(seed=seed, finished=match.finished, stopped=match.supervisor.stopped,
                            touches=sum(match.stats[r]['touches'] for r in ('a', 'b')),
                            attempt_rates={r: match.stats[r]['attempts'] / max(match.arena.time, 1e-9)
                                           for r in ('a', 'b')}))
    return dict(matches=matches, touches=sum(m['touches'] for m in matches),
                minimum_attempts_per_second=min(rate for m in matches
                                                for rate in m['attempt_rates'].values()),
                safety_stops=sum(m['stopped'] for m in matches),
                incomplete=sum(not m['finished'] for m in matches))


def _read_artifact(path):
    value = json.loads(artifact_path(path).read_text())
    version = value.get('schema_version')
    if value.get('kind') != 'tabular_q' or version not in (1, 2):
        raise ValueError('Expected a version-1 or version-2 tabular_q artifact')
    expected = LEGACY_ACTIONS if version == 1 else ACTIONS
    if value.get('actions') != [a.value for a in expected]:
        raise ValueError('Policy action schema does not match this controller')
    if not isinstance(value.get('table'), dict) or any(
            not isinstance(row, list) or len(row) != len(expected)
            or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in row)
            for row in value['table'].values()):
        raise ValueError(f'Q-table rows require {len(expected)} finite action values')
    if version == 1:
        value['legacy_schema'] = True
        padding = [0.0] * (len(ACTIONS) - len(LEGACY_ACTIONS))
        value['table'] = {key: row + padding for key, row in value['table'].items()}
        value['opponent_tables'] = [
            {key: row + padding for key, row in snapshot.items()}
            for snapshot in value.get('opponent_tables', [])]
    Parameters(**value['parameters'])
    return value


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


def train(config=None, settings=None):
    config = deepcopy(config or load_config())
    settings = {**DEFAULT_SETTINGS, **config.get('learning', {}), **(settings or {})}
    settings['output'], settings['resume'] = str(settings['output']), str(settings['resume'] or '')
    for name in ('episodes', 'snapshot_every', 'pool_size'):
        if not isinstance(settings[name], int) or isinstance(settings[name], bool) or settings[name] < 1:
            raise ValueError(f'{name} must be a positive integer')
    for name in ('seconds', 'alpha', 'discount_per_second', 'epsilon_start', 'epsilon_end',
                 'prior_bonus', 'time_cost', 'rejection_cost', 'failure_cost',
                 'minimum_attempts_per_second', 'idle_after_seconds', 'idle_cost',
                 'approach_reward', 'far_time_cost'):
        if not math.isfinite(settings[name]):
            raise ValueError(f'{name} must be finite')
    if (settings['seconds'] <= 0 or not settings['validation_seeds']
            or not settings['selfplay_validation_seeds']):
        raise ValueError('Learning needs positive seconds and evaluation seeds')
    if not 0 < settings['alpha'] <= 1 or not 0 < settings['discount_per_second'] <= 1:
        raise ValueError('Learning rate and discount must be in (0, 1]')
    if not 0 <= settings['epsilon_end'] <= settings['epsilon_start'] <= 1:
        raise ValueError('Exploration probabilities must be ordered within [0, 1]')
    for name in ('prior_bonus', 'time_cost', 'rejection_cost', 'failure_cost',
                 'minimum_attempts_per_second', 'idle_after_seconds', 'idle_cost',
                 'approach_reward', 'far_time_cost'):
        if settings[name] < 0:
            raise ValueError(f'{name} must be nonnegative')
    if any(not isinstance(seed, int) or seed < 0 for seed in
           [*settings['validation_seeds'], *settings['selfplay_validation_seeds']]):
        raise ValueError('Evaluation seeds must be nonnegative integers')

    parameters = asdict(Parameters(**config['policies']['a']))
    path = artifact_path(settings['output'])
    table, history, snapshots, previous_seeds, previous_evaluations = {}, [], [], [], []
    offset, resumed_from, resume_source = 0, None, None
    resume_config_changed = False
    if settings['resume']:
        resume_path = artifact_path(settings['resume'])
        prior = _read_artifact(resume_path)
        if asdict(Parameters(**prior['parameters'])) != parameters:
            raise ValueError('Resume parameters differ from configured policies.a; use the original settings')
        offset = prior.get('completed_episodes', prior['settings']['episodes'])
        if not isinstance(offset, int) or offset < 0:
            raise ValueError('Resume completed_episodes must be a nonnegative integer')
        table = deepcopy(prior['table'])
        report_path = resume_path.with_name(resume_path.stem + '_report.json')
        prior_report = json.loads(report_path.read_text()) if report_path.exists() else prior
        history = deepcopy(prior.get('history', prior_report.get('history', [])))
        previous_seeds = list(prior.get('training_seeds', prior_report.get('training_seeds', range(offset))))
        snapshots = deepcopy(prior.get('opponent_tables', [table]))[-settings['pool_size']:]
        previous_evaluations = list(prior.get('previous_evaluations', prior_report.get('previous_evaluations', [])))
        if 'candidate' in prior_report:
            previous_evaluations.append({key: prior_report[key] for key in
                                         ('source_fingerprint', 'baseline', 'candidate', 'promoted')})
        resumed_from, resume_source = str(resume_path), prior.get('source_fingerprint')
        resume_config_changed = prior.get('config') != config
    new_seeds = list(range(offset, offset + settings['episodes']))
    training_seeds = previous_seeds + new_seeds
    if set(training_seeds) & set([*settings['validation_seeds'],
                                *settings['selfplay_validation_seeds']]):
        raise ValueError('Training and validation seeds must be separate')
    current_source = source_fingerprint()
    checkpoints = path.parent / (path.stem + '_checkpoints')
    checkpoint_files = []

    def artifact(completed, promoted=False, activity_gate_passed=False):
        return dict(kind='tabular_q', schema_version=2, actions=[a.value for a in ACTIONS],
                    source_fingerprint=current_source, settings=settings, parameters=parameters,
                    table=table, promoted=promoted,
                    activity_gate_passed=activity_gate_passed,
                    completed_episodes=completed,
                    training_seeds=training_seeds[:completed], history=history,
                    opponent_tables=snapshots, config=config, resumed_from=resumed_from,
                    episode_offset=offset, resume_source=resume_source,
                    resume_config_changed=resume_config_changed,
                    resume_source_changed=resume_source is not None and resume_source != current_source,
                    previous_evaluations=previous_evaluations)

    rng = np.random.default_rng(settings['seed'] + offset)
    for local_episode, seed in enumerate(new_seeds):
        episode = offset + local_episode
        fraction = local_episode / max(1, settings['episodes'] - 1)
        epsilon = settings['epsilon_start'] + fraction * (settings['epsilon_end'] - settings['epsilon_start'])
        learner = TabularPolicy(settings['seed'] + seed, Parameters(**parameters), table, settings,
                               learning=True, epsilon=epsilon, motion=config['motion'],
                               minimum_separation=config['safety']['min_separation'],
                               rules=config.get('policy_rules'))
        opponent, opponent_index = None, None
        if snapshots and episode % 2:
            opponent_index = int(rng.integers(len(snapshots)))
            opponent = TabularPolicy(seed + 1009, Parameters(**parameters), snapshots[opponent_index],
                                     settings, motion=config['motion'],
                                     minimum_separation=config['safety']['min_separation'],
                                     rules=config.get('policy_rules'))
        result = _episode(learner, ('a', 'b')[episode % 2], opponent, seed, settings['seconds'], config, settings)
        table = learner.table
        history.append(dict(episode=episode+1, epsilon=epsilon, states=len(table),
                            opponent_kind='reactive' if opponent is None else 'frozen_self_snapshot',
                            snapshot_index=opponent_index, **result))
        completed = episode + 1
        if completed % settings['snapshot_every'] == 0 or local_episode == len(new_seeds) - 1:
            snapshots.append(deepcopy(table))
            snapshots = snapshots[-settings['pool_size']:]
            checkpoint_path = checkpoints / f'episode_{completed:06d}.json'
            _write_json(checkpoint_path, artifact(completed))
            checkpoint_files.append(str(checkpoint_path))
            print(f"Episode {completed}/{offset+settings['episodes']}: states={len(table)}, reward={result['reward']:.1f}", flush=True)
    opponents = [parameters, *config['training']['validation_opponents']]
    baseline = evaluate(None, parameters, opponents, settings['validation_seeds'], settings['seconds'], config, settings)
    candidate = evaluate(table, parameters, opponents, settings['validation_seeds'], settings['seconds'], config, settings)
    self_play = evaluate_self_play(table, parameters, settings['selfplay_validation_seeds'],
                                   settings['seconds'], config, settings)
    activity_gate_passed = (self_play['minimum_attempts_per_second'] >=
                            settings['minimum_attempts_per_second'] and self_play['touches'] > 0
                            and self_play['safety_stops'] == 0 and self_play['incomplete'] == 0)
    promoted = (candidate['mean_reward'] > baseline['mean_reward'] and candidate['safety_stops'] == 0
                and candidate['incomplete'] == 0 and candidate['touches_for'] > 0
                and activity_gate_passed)
    final = artifact(offset + settings['episodes'], promoted, activity_gate_passed)
    report = dict(**final, validation_seeds=settings['validation_seeds'],
                  checkpoint_files=checkpoint_files, baseline=baseline, candidate=candidate,
                  self_play=self_play,
                  continuation='Existing Q values and opponent snapshots continue; exploration schedule restarts for this chunk.',
                  limitation='Tabular reinforcement learning over stabilized motion primitives; no neural policy, imitation, or balance learning. Prior evaluation is historical; current promotion uses a fresh evaluation. Default guards are high, so improved body-shot selection does not establish sophisticated boxing.')
    _write_json(path.with_name(path.stem+'_report.json'), report)
    _write_json(path, final)
    return report


def load_learned_policy(path, seed=0, allow_unpromoted=False, config=None):
    value = _read_artifact(path)
    parameters = Parameters(**value['parameters'])
    if (value.get('legacy_schema') or not value['promoted']
            or not value.get('activity_gate_passed', False)) and not allow_unpromoted:
        return Policy(seed, parameters)
    config = config or load_config()
    if value.get('config') and (value['config']['simulation']['model'] != config['simulation']['model']
                                or value['config'].get('policy_rules') != config.get('policy_rules')):
        return Policy(seed, parameters)
    return TabularPolicy(seed, parameters, value['table'], value['settings'],
                         motion=config['motion'], minimum_separation=config['safety']['min_separation'],
                         rules=config.get('policy_rules'))


def main():
    parser = argparse.ArgumentParser(description='Train a CPU tabular boxing policy')
    parser.add_argument('--config', type=Path)
    parser.add_argument('--episodes', type=int)
    parser.add_argument('--seconds', type=float)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--resume', type=Path, help='continue Q values from a saved artifact or checkpoint')
    args = parser.parse_args()
    config = load_config(args.config)
    configure_logging(config)
    overrides = {key: value for key, value in vars(args).items() if key != 'config' and value is not None}
    for key in ('output', 'resume'):
        if key in overrides:
            overrides[key] = str(overrides[key])
    try:
        report = train(config, overrides)
    except ValueError as error:
        parser.error(str(error))
    print(f"Held-out reward baseline={report['baseline']['mean_reward']:.3f}, candidate={report['candidate']['mean_reward']:.3f}; promoted={report['promoted']}")


if __name__ == '__main__':
    main()
