"""RSL-RL distribution contract for the two qualified defensive primitives.

Network outputs are logits. Environment outputs are actual Primitive IDs (0/8),
never a continuous expectation or a rounded Gaussian sample.
"""
import torch
from torch import nn
from torch.distributions import Categorical, kl_divergence


class DefensiveArgmax(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('primitive_ids', torch.tensor([0., 8.]))

    def forward(self, logits):
        return self.primitive_ids[logits.argmax(dim=-1)].unsqueeze(-1)


class DefensivePrimitiveDistribution(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        if output_dim != 1:
            raise ValueError('Defensive primitive distribution requires one ID output')
        self.output_dim = output_dim
        self.register_buffer('primitive_ids', torch.tensor([0., 8.]))
        self._distribution = None

    @property
    def input_dim(self):
        return 2

    def init_mlp_weights(self, mlp):
        pass

    def update(self, mlp_output):
        self._distribution = Categorical(logits=mlp_output)

    def sample(self):
        return self.primitive_ids[self._distribution.sample()].unsqueeze(-1)

    def deterministic_output(self, mlp_output):
        return self.primitive_ids[mlp_output.argmax(dim=-1)].unsqueeze(-1)

    def as_deterministic_output_module(self):
        return DefensiveArgmax().to(self.primitive_ids.device)

    @property
    def mean(self):
        # Diagnostic expectation only; never sent to the environment.
        return (self._distribution.probs*self.primitive_ids).sum(-1, keepdim=True)

    @property
    def std(self):
        variance=(self._distribution.probs*(self.primitive_ids-self.mean).square()).sum(-1, keepdim=True)
        return variance.sqrt()

    @property
    def entropy(self):
        return self._distribution.entropy()

    @property
    def params(self):
        return (self._distribution.logits,)

    def log_prob(self, outputs):
        matches=outputs==self.primitive_ids
        if outputs.shape[-1]!=1 or not matches.any(-1).all():
            raise ValueError('Policy action is outside defensive Primitive IDs 0/8')
        return self._distribution.log_prob(matches.long().argmax(-1))

    def kl_divergence(self, old_params, new_params):
        return kl_divergence(Categorical(logits=old_params[0]), Categorical(logits=new_params[0]))
