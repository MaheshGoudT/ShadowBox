"""Decision-step boxing environment for tactical learning."""

from copy import deepcopy
from collections import Counter
import math
from os import PathLike

import numpy as np

from ..boxing import Match, Parameters, Policy
from ..motion import Primitive
from ..config import load_config
from .actions import ACTIONS, allowed_actions
from .observations import OBSERVATION_SIZE, encode_observation
from .reward import tactical_reward


class _ControlledPolicy(Policy):
    def __init__(self, seed, parameters, rules):
        super().__init__(seed, parameters, rules=rules)
        self.pending = None

    def choose(self, observation):
        action, self.pending = self.pending, None
        return action or Primitive.GUARD


class BoxingEnv:
    """One learner controls A or B; the other boxer uses an opponent policy.

    Each step executes one requested decision through the real match loop,
    including motion completion and the opponent's concurrent actions.
    """

    def __init__(self, config=None):
        self.config = load_config(config) if isinstance(config, (str, PathLike, type(None))) else deepcopy(config)
        self.actions = ACTIONS
        self.action_space_n = len(ACTIONS)
        self.observation_space_shape = (OBSERVATION_SIZE,)
        self.match = None
        self.side = 'a'

    def reset(self, seed=0, options=None):
        config = deepcopy(self.config)
        options = options or {}
        self.side = options.get('side', 'a')
        if self.side not in ('a', 'b'):
            raise ValueError('Learner side must be a or b')
        other = 'b' if self.side == 'a' else 'a'
        if 'spacing' in options:
            spacing = float(options['spacing'])
            minimum = config['safety']['min_separation'] + .05
            if not math.isfinite(spacing) or not minimum < spacing < 2.5:
                raise ValueError(f'Reset spacing must be between {minimum:.2f} and 2.5 m')
            config['simulation']['robot_spacing'] = spacing
        if 'proximity_fallback' in options:
            if not isinstance(options['proximity_fallback'], bool):
                raise ValueError('proximity_fallback must be boolean')
            config['scoring']['proximity_fallback'] = options['proximity_fallback']
        self.match = Match(seed=seed, config=config, rounds=1,
                           round_seconds=config['match']['round_seconds'])
        if 'opponent_parameters' in options:
            opponent = Parameters(**options['opponent_parameters'])
            self.match.parameters[other] = opponent
            self.match.policies[other] = Policy(seed + 1009, opponent, rules=config['policy_rules'])
        if 'opponent_policy' in options:
            self.match.policies[other] = options['opponent_policy']
        parameters = Parameters(**config['policies'][self.side])
        self.learner = _ControlledPolicy(seed, parameters, config['policy_rules'])
        self.match.policies[self.side] = self.learner
        return self.observation(), {'action_mask': self.action_mask()}

    def observation(self, robot=None):
        if self.match is None:
            raise RuntimeError('Call reset before observation')
        return encode_observation(self.match, robot or self.side)

    def action_mask(self, robot=None):
        robot = robot or self.side
        mask = np.zeros(self.action_space_n, dtype=bool)
        config = self.match.config
        mask[allowed_actions(self.match.observe(robot), self.match.parameters[robot],
                             config['motion'], config['safety']['min_separation'],
                             config['policy_rules'], proactive_guard=config.get('tactics', {}).get(
                                 'proactive_body_guard', False))] = True
        return mask

    def step(self, action, trace=False):
        if self.match is None:
            raise RuntimeError('Call reset before step')
        if not isinstance(action, (int, np.integer)) or not 0 <= action < self.action_space_n:
            raise ValueError('Action index is outside the tactical action set')
        if self.match.finished or self.match.supervisor.stopped:
            raise RuntimeError('Reset after the episode ends')
        if not self.action_mask()[action]:
            raise ValueError('Action is masked in the current boxing state')
        side = self.side
        other = 'b' if side == 'a' else 'a'
        chosen = self.actions[action]
        observed = self.match.observe(side)
        self.learner.pending = chosen
        if trace:
            arena = self.match.arena
            path = dict(root_start=arena.position(side)[:2].tolist(),
                        joint_targets_start={key: float(value) for key, value in arena.full_targets(side).items()},
                        closest_center_m={target: float('inf') for target in ('head', 'torso')},
                        closest_glove={}, min_root_separation_m=float('inf'),
                        max_glove_speed_m_s=0.0)
        before = {side: self.match.stats[side].copy() for side in ('a', 'b')}
        event_cursor, start = len(self.match.events), self.match.arena.time
        seen = rejected = False
        rejection_reasons = Counter()
        max_ticks = math.ceil(5 / self.match.arena.model.opt.timestep)
        for _ in range(max_ticks):
            self.match.step()
            if trace:
                gloves = arena.glove_positions(side)
                for target in ('head', 'torso'):
                    center = arena.data.geom_xpos[arena.model.geom(f'{other}_{target}').id]
                    for hand, glove in enumerate(gloves):
                        distance = float(np.linalg.norm(glove - center))
                        if distance < path['closest_center_m'][target]:
                            path['closest_center_m'][target] = distance
                            path['closest_glove'][target] = dict(hand=hand,
                                glove=glove.tolist(), target_center=center.tolist())
                separation = float(np.linalg.norm(arena.position(side)[:2]
                                                  - arena.position(other)[:2]))
                path['min_root_separation_m'] = min(path['min_root_separation_m'], separation)
                path['max_glove_speed_m_s'] = max(path['max_glove_speed_m_s'],
                    float(np.max(np.linalg.norm(self.match.glove_velocities[side], axis=1))))
            for event in self.match.events[event_cursor:]:
                if event['robot'] == side and event['event'] in ('action', 'rejected'):
                    if event.get('action') == chosen.value:
                        seen = True
                        rejected |= event['event'] == 'rejected'
                        if event['event'] == 'rejected':
                            rejection_reasons[event.get('reason', 'unknown')] += 1
            event_cursor = len(self.match.events)
            if self.match.finished or self.match.supervisor.stopped:
                break
            if seen and not self.match.controllers[side].busy and not self.match.punch[side]:
                break
        self.learner.pending = None
        delta = {side: {key: self.match.stats[side][key] - before[side][key]
                        for key in ('attempts', 'touches', 'blocks', 'misses')} for side in ('a', 'b')}
        elapsed = self.match.arena.time - start
        reward = tactical_reward(delta[side], delta[other], elapsed, rejected,
                                 self.match.supervisor.stopped,
                                 self.match.config.get('tactics', {}).get('reward'),
                                 chosen.value, delta[side]['attempts'] if (seen and not rejected
                                     and chosen.value in ('jab', 'cross', 'uppercut')) else 0,
                                 unprovoked_retreat=(chosen.value == 'step_backward'
                                     and seen and not rejected and delta[other]['attempts'] == 0
                                     and not observed.incoming_body
                                     and observed.incoming_speed < .25 * self.match.config[
                                         'policy_rules']['distance_scale']))
        terminated = self.match.finished or self.match.supervisor.stopped
        truncated = not terminated and not seen
        info = dict(seconds=elapsed, requested=chosen.value, action_seen=seen,
                    rejected=rejected, own=delta[side], opponent=delta[other],
                    rejection_reasons=dict(rejection_reasons),
                    action_mask=self.action_mask())
        if trace:
            path['root_end'] = arena.position(side)[:2].tolist()
            path['joint_targets_end'] = {key: float(value) for key, value in arena.full_targets(side).items()}
            info['trace'] = path
        return self.observation(), float(reward), terminated, truncated, info

    def close(self):
        self.match = None
