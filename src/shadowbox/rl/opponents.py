"""Simple fixed opponents that expose one-trick tactical policies."""

from ..boxing import Policy
from ..motion import Primitive
from .actions import ACTIONS, allowed_actions


class GuardOpponent(Policy):
    def __init__(self, seed, parameters, rules, level):
        super().__init__(seed, parameters, rules=rules)
        if level not in ('head', 'body'):
            raise ValueError('Guard level must be head or body')
        self.level = level

    def choose(self, observation):
        if observation.distance > self.parameters.desired_distance + .1:
            return Primitive.STEP_FORWARD
        return Primitive.GUARD if self.level == 'head' else Primitive.BODY_GUARD


class AttackOpponent(Policy):
    """Repeated legal head/body threats for measuring actual defensive outcomes."""

    def __init__(self, seed, parameters, config, level):
        super().__init__(seed, parameters, rules=config['policy_rules'])
        if level not in ('head', 'body'):
            raise ValueError('Attack level must be head or body')
        self.level = level
        self.motion = config['motion']
        self.minimum_separation = config['safety']['min_separation']

    def choose(self, observation):
        allowed = allowed_actions(observation, self.parameters, self.motion,
                                  self.minimum_separation, self.rules)
        if 1 in allowed:
            return Primitive.TURN_TO_TARGET
        attack = 6 if self.level == 'head' else 7
        if attack in allowed:
            return ACTIONS[attack]
        if 2 in allowed:
            return Primitive.STEP_FORWARD
        return Primitive.GUARD
