import pytest
import torch
from shadowbox.playground.primitive_distribution import (BoxingPrimitiveDistribution,
    DefensivePrimitiveDistribution)


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


def test_boxing_distribution_trains_and_exports_all_authored_goal_ids():
    distribution=BoxingPrimitiveDistribution(1)
    logits=torch.arange(20,dtype=torch.float32).reshape(2,10).requires_grad_()
    distribution.update(logits)
    sampled=distribution.sample()
    assert sampled.shape==(2,1)
    assert ((sampled>=0)&(sampled<=9)).all()
    assert distribution.input_dim==10
    actions=torch.tensor([[9.],[0.]])
    expected=torch.log_softmax(logits,-1)[torch.arange(2),torch.tensor([9,0])]
    torch.testing.assert_close(distribution.log_prob(actions),expected)
    (-distribution.log_prob(actions).mean()).backward()
    assert torch.isfinite(logits.grad).all() and logits.grad.abs().sum()>0
    torch.testing.assert_close(distribution.deterministic_output(logits),torch.tensor([[9.],[9.]]))
    exported=torch.jit.script(distribution.as_deterministic_output_module())
    torch.testing.assert_close(exported(logits),torch.tensor([[9.],[9.]]))
    with pytest.raises(ValueError,match='action IDs 0 through 9'):
        BoxingPrimitiveDistribution(1,primitive_ids=tuple(range(11)))
    with pytest.raises(ValueError,match='configured boxing'):
        distribution.log_prob(torch.tensor([[10.],[0.]]))
