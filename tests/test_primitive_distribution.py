import pytest
import torch
from shadowbox.playground.primitive_distribution import DefensivePrimitiveDistribution


def test_samples_and_exports_are_ids_while_likelihood_uses_category_indices():
    distribution=DefensivePrimitiveDistribution(1)
    logits=torch.tensor([[0., 1.], [2., -1.]], requires_grad=True)
    distribution.update(logits)
    for _ in range(10):
        sampled=distribution.sample()
        assert sampled.shape==(2,1)
        assert torch.isin(sampled, torch.tensor([0.,8.])).all()
    actions=torch.tensor([[8.],[0.]])
    expected=torch.log_softmax(logits,-1)[torch.arange(2),torch.tensor([1,0])]
    torch.testing.assert_close(distribution.log_prob(actions), expected)
    (-distribution.log_prob(actions).mean()).backward()
    assert torch.isfinite(logits.grad).all() and logits.grad.abs().sum()>0
    torch.testing.assert_close(distribution.deterministic_output(logits),actions)
    exported=torch.jit.script(distribution.as_deterministic_output_module())
    torch.testing.assert_close(exported(logits),actions)
    with pytest.raises(ValueError,match='0/8'):
        distribution.log_prob(torch.tensor([[1.],[0.]]))


def test_kl_and_entropy_match_torch_categorical_and_saved_mapping():
    distribution=DefensivePrimitiveDistribution(1)
    old=torch.tensor([[.1,.4]])
    new=torch.tensor([[.7,-.3]])
    distribution.update(old)
    params=distribution.params
    torch.testing.assert_close(distribution.kl_divergence(params,params),torch.zeros(1))
    expected=torch.distributions.kl_divergence(torch.distributions.Categorical(logits=old),
                                             torch.distributions.Categorical(logits=new))
    torch.testing.assert_close(distribution.kl_divergence(params,(new,)),expected)
    torch.testing.assert_close(distribution.entropy,torch.distributions.Categorical(logits=old).entropy())
    restored=DefensivePrimitiveDistribution(1)
    restored.load_state_dict(distribution.state_dict())
    torch.testing.assert_close(restored.deterministic_output(old),torch.tensor([[8.]]))
