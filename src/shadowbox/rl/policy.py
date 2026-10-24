"""Adapt a frozen tactical actor to the match policy interface."""

import torch

from ..boxing import Parameters, Policy
from .actions import ACTIONS


def sample_action(distribution, generator, temperature=.5):
    """Draw a reproducible legal action from the trained categorical actor."""
    if temperature <= 0:
        raise ValueError('Inference temperature must be positive')
    probabilities = torch.softmax(distribution.logits.squeeze(0) / temperature, dim=-1)
    return int(torch.multinomial(probabilities, 1, generator=generator).item())


class FrozenOpponent(Policy):
    def __init__(self, env, model, side, seed):
        super().__init__(seed, Parameters(**env.config['policies'][side]),
                         rules=env.config['policy_rules'])
        self.env, self.model, self.side = env, model, side
        self.generator = torch.Generator().manual_seed(seed + (side == 'b'))

    def choose(self, observation):
        state = torch.as_tensor(self.env.observation(self.side)).unsqueeze(0)
        mask = torch.as_tensor(self.env.action_mask(self.side)).unsqueeze(0)
        with torch.no_grad():
            distribution, _ = self.model(state, mask)
        temperature = self.env.config.get('tactics', {}).get('inference_temperature', .5)
        return ACTIONS[sample_action(distribution, self.generator, temperature)]
