"""Neural policy and value model; no simulator dependency."""

from torch import nn
from torch.distributions import Categorical

from .actions import ACTIONS
from .observations import OBSERVATION_SIZE


class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(OBSERVATION_SIZE, 64), nn.Tanh(),
                                  nn.Linear(64, 64), nn.Tanh())
        self.actor = nn.Linear(64, len(ACTIONS))
        self.critic = nn.Linear(64, 1)

    def forward(self, states, masks):
        hidden = self.body(states)
        logits = self.actor(hidden).masked_fill(~masks, -1e9)
        return Categorical(logits=logits), self.critic(hidden).squeeze(-1)
