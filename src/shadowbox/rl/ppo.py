"""Small CPU PPO trainer for boxing decisions, separate from motion tracking."""

import argparse
from copy import deepcopy
from collections import Counter
import json

import numpy as np
import torch
from torch import nn

from ..boxing import Parameters, Policy
from ..config import artifact_path, load_config, source_fingerprint
from .actions import ACTIONS
from .environment import BoxingEnv
from .model import ActorCritic
from .observations import OBSERVATION_SIZE
from .opponents import GuardOpponent
from .policy import FrozenOpponent


def _balanced_imitation_loss(distribution, actions):
    """Give each demonstrated action class equal weight without duplicating rows."""
    counts = torch.bincount(actions, minlength=len(ACTIONS)).float().clamp(min=1)
    weights = counts[actions].reciprocal()
    return -(distribution.log_prob(actions) * weights).sum() / weights.sum()


def _rollout(env, model, rng, seed, episode, snapshots, seconds):
    side = ('a', 'b')[episode % 2]
    other = 'b' if side == 'a' else 'a'
    spacing = float(rng.choice((.62, .72, .9, 1.2)))
    # Keep training and held-out scoring on the same measured glove geometry.
    options = dict(side=side, spacing=spacing, proximity_fallback=False)
    opponent_choice = int(rng.integers(4)) if snapshots else int(rng.integers(3))
    if snapshots and opponent_choice == 3:
        options['opponent_policy'] = FrozenOpponent(env, snapshots[int(rng.integers(len(snapshots)))],
                                                    other, seed + 1009)
        opponent_kind = 'frozen_self'
    elif opponent_choice == 0:
        settings = deepcopy(env.config['policies'][other])
        settings['aggression'] = float(rng.choice((.35, .72, .95)))
        settings['reaction'] = float(rng.choice((.1, .18, .3)))
        options['opponent_parameters'] = settings
        opponent_kind = 'reactive'
    else:
        level = 'head' if opponent_choice == 1 else 'body'
        options['opponent_policy'] = GuardOpponent(seed + 1009,
            Parameters(**env.config['policies'][other]), env.config['policy_rules'], level)
        opponent_kind = f'{level}_guard'
    state, info = env.reset(seed=seed, options=options)
    trajectory = []
    actions, rejection_reasons = [], Counter()
    for _ in range(200):
        mask = info['action_mask']
        with torch.no_grad():
            distribution, value = model(torch.as_tensor(state).unsqueeze(0),
                                        torch.as_tensor(mask).unsqueeze(0))
            action = distribution.sample()
            log_prob = distribution.log_prob(action)
        next_state, reward, done, truncated, next_info = env.step(int(action))
        trajectory.append((state, mask, int(action), float(log_prob), float(value),
                           reward, next_info['seconds']))
        actions.append(ACTIONS[int(action)].value)
        rejection_reasons.update(next_info['rejection_reasons'])
        state, info = next_state, next_info
        if done or truncated or env.match.arena.time >= seconds:
            break
    match = env.match
    summary = dict(seed=seed, side=side, spacing=spacing, opponent=opponent_kind,
                   proximity_fallback=False,
                   own=match.stats[side], opponent_stats=match.stats[other],
                   actions=actions, stopped=match.supervisor.stopped,
                   rejection_reasons=dict(rejection_reasons),
                   seconds=match.arena.time)
    return trajectory, summary


def _update(model, optimizer, trajectory, demonstrations=None, imitation_weight=.1,
            gamma=.97, lam=.95, epochs=4):
    if not trajectory:
        return
    states, masks, actions, old_log_probs, values, rewards, durations = zip(*trajectory)
    advantage, next_value = 0.0, 0.0
    advantages = []
    for reward, value, duration in reversed(list(zip(rewards, values, durations))):
        discount = gamma ** duration
        advantage = reward + discount * next_value - value + discount * lam * advantage
        advantages.append(advantage)
        next_value = value
    advantages.reverse()
    states = torch.as_tensor(np.asarray(states))
    masks = torch.as_tensor(np.asarray(masks))
    actions = torch.as_tensor(actions)
    old_log_probs = torch.as_tensor(old_log_probs)
    advantages = torch.as_tensor(advantages, dtype=torch.float32)
    returns = advantages + torch.as_tensor(values)
    if len(advantages) > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)
    if demonstrations:
        teacher_states = torch.as_tensor(np.asarray([item[0] for item in demonstrations]))
        teacher_masks = torch.as_tensor(np.asarray([item[1] for item in demonstrations]))
        teacher_actions = torch.as_tensor([item[2] for item in demonstrations])
    for _ in range(epochs):
        distribution, predicted = model(states, masks)
        ratio = (distribution.log_prob(actions) - old_log_probs).exp()
        actor = -torch.minimum(ratio * advantages,
                               ratio.clamp(.8, 1.2) * advantages).mean()
        loss = actor + .5 * (predicted - returns).square().mean() - .01 * distribution.entropy().mean()
        if demonstrations:
            teacher_distribution, _ = model(teacher_states, teacher_masks)
            loss += imitation_weight * _balanced_imitation_loss(
                teacher_distribution, teacher_actions)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()


def _teacher_data(env, seed, episodes, seconds):
    """Collect action labels from the working reactive boxer on both sides."""
    examples = []
    for episode in range(episodes):
        side = ('a', 'b')[episode % 2]
        other = 'b' if side == 'a' else 'a'
        options = dict(side=side, spacing=(.62, .72, .9, 1.2)[episode % 4])
        if episode % 3:
            level = 'head' if episode % 3 == 1 else 'body'
            options['opponent_policy'] = GuardOpponent(seed + 1009,
                Parameters(**env.config['policies'][other]), env.config['policy_rules'], level)
        state, info = env.reset(seed + 100000 + episode, options)
        teacher = Policy(seed + episode, Parameters(**env.config['policies'][side]),
                         rules=env.config['policy_rules'])
        for _ in range(100):
            mask = info['action_mask']
            if state[6] and mask[8]:
                action = 8  # Block an observed low incoming glove.
            elif state[8:10].mean() < -.1:
                action = next((index for index in (9, 6, 2) if mask[index]), 0)
            elif state[8:10].mean() > .02:
                action = next((index for index in (7, 2) if mask[index]), 0)
            else:
                chosen = teacher.choose(env.match.observe(side))
                action = ACTIONS.index(chosen) if chosen in ACTIONS else 0
            if not mask[action]:
                action = 2 if mask[2] else int(np.flatnonzero(mask)[0])
            examples.append((state, mask, action))
            state, _, done, truncated, info = env.step(action)
            if done or truncated or env.match.arena.time >= seconds:
                break
    return examples


def _imitate(model, optimizer, examples, epochs):
    states = torch.as_tensor(np.asarray([item[0] for item in examples]))
    masks = torch.as_tensor(np.asarray([item[1] for item in examples]))
    actions = torch.as_tensor([item[2] for item in examples])
    for _ in range(epochs):
        distribution, _ = model(states, masks)
        loss = _balanced_imitation_loss(distribution, actions)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()


def _load_demonstrations(path):
    records = json.loads(artifact_path(path).read_text())
    if not isinstance(records, list) or not records:
        raise ValueError('Demonstrations must be a nonempty list')
    examples = []
    for record in records:
        state = np.asarray(record['observation'], dtype=np.float32)
        mask = np.asarray(record['mask'], dtype=bool)
        action = record['index']
        if (state.shape != (OBSERVATION_SIZE,) or mask.shape != (len(ACTIONS),)
                or not isinstance(action, int) or not 0 <= action < len(ACTIONS)
                or record['action'] != ACTIONS[action].value or not mask[action]
                or not record['accepted'] or record['stopped'] or record['truncated']
                or not np.isfinite(state).all()):
            raise ValueError('Invalid qualified demonstration')
        examples.append((state, mask, action))
    return examples


def train(config=None, episodes=120, seconds=20, seed=0,
          output='tmp/policies/tactical-ppo.pt', demonstrations_path=None):
    """Train against varied reactive boxers and frozen snapshots; never auto-promote."""
    if episodes < 1 or seconds <= 0:
        raise ValueError('Training needs positive episodes and seconds')
    config = load_config(config) if not isinstance(config, dict) else deepcopy(config)
    config['match']['round_seconds'] = seconds
    config['scoring']['proximity_fallback'] = False  # Train on measured 3D glove paths.
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = ActorCritic()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    env = BoxingEnv(config)
    demonstrations = (_load_demonstrations(demonstrations_path) if demonstrations_path
                      else _teacher_data(env, seed, min(12, max(4, episodes // 4)), seconds))
    _imitate(model, optimizer, demonstrations, epochs=20)
    snapshots, history = [], []
    for episode in range(episodes):
        trajectory, result = _rollout(env, model, rng, seed + episode, episode,
                                      snapshots, seconds)
        _update(model, optimizer, trajectory, demonstrations)
        history.append(result)
        if (episode + 1) % 10 == 0:
            snapshots.append(deepcopy(model).eval())
            snapshots = snapshots[-4:]
        if (episode + 1) % 10 == 0 or episode + 1 == episodes:
            print(f"Episode {episode + 1}/{episodes}: "
                  f"touches {result['own']['touches']}:{result['opponent_stats']['touches']}", flush=True)
    path = artifact_path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(model=model.state_dict(), actions=[a.value for a in ACTIONS],
                    observation_shape=OBSERVATION_SIZE, source_fingerprint=source_fingerprint(),
                    config=config, seed=seed, episodes=episodes,
                    teacher_examples=len(demonstrations),
                    demonstrations_path=str(demonstrations_path) if demonstrations_path else None), path)
    report = dict(output=str(path), episodes=episodes, seed=seed, seconds=seconds,
                  teacher_examples=len(demonstrations),
                  demonstrations_path=str(demonstrations_path) if demonstrations_path else None,
                  promoted=False, reason='Held-out tactical evaluation is required',
                  history=history)
    path.with_suffix('.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description='Train tactical PPO against mixed opponents')
    parser.add_argument('--config')
    parser.add_argument('--episodes', type=int, default=120)
    parser.add_argument('--seconds', type=float, default=20)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output', default='tmp/policies/tactical-ppo.pt')
    parser.add_argument('--demonstrations', help='Qualified examples from tactical_iteration.py')
    args = parser.parse_args()
    train(args.config, args.episodes, args.seconds, args.seed, args.output,
          args.demonstrations)


if __name__ == '__main__':
    main()
