"""Tactical learning modules share interfaces without sharing training state."""

import numpy as np
import torch

from shadowbox.rl.actions import ACTIONS
from shadowbox.rl.model import ActorCritic
from shadowbox.rl.reward import tactical_reward
from shadowbox.rl.ppo import train
from shadowbox.rl.evaluate import evaluate_checkpoint


def test_mask_and_reward_preserve_defense_and_safety():
    model = ActorCritic()
    state = torch.zeros((1, 12))
    mask = torch.zeros((1, len(ACTIONS)), dtype=torch.bool)
    mask[0, 0] = True
    distribution, value = model(state, mask)
    assert distribution.probs[0, 0] == 1
    assert distribution.probs[0, 1:].sum() == 0
    assert value.shape == (1,)
    empty = dict(touches=0, blocks=0, misses=0)
    defense = dict(touches=0, blocks=1, misses=0)
    assert tactical_reward(defense, empty, 1, False, False) > tactical_reward(empty, empty, 1, False, False)
    assert tactical_reward(empty, empty, 1, False, True) < 0
    idle = tactical_reward(empty, empty, 1, False, False, action='body_guard')
    blocked = tactical_reward(defense, empty, 1, False, False, action='body_guard')
    assert idle < tactical_reward(empty, empty, 1, False, False)
    assert blocked > idle
    attempted = tactical_reward(empty, empty, 1, False, False,
                                action='jab', accepted_attempts=1)
    assert attempted > tactical_reward(empty, empty, 1, False, False)
    retreat = tactical_reward(empty, empty, 1, False, False,
                              action='step_backward', unprovoked_retreat=True)
    assert retreat < tactical_reward(empty, empty, 1, False, False,
                                     action='step_backward')


def test_short_run_saves_unpromoted_checkpoint_and_held_out_report(tmp_path):
    path = tmp_path / 'tactical.pt'
    report = train('configs/unitree.yaml', episodes=2, seconds=2, seed=9, output=path)
    assert path.is_file() and report['teacher_examples'] > 0
    assert {entry['side'] for entry in report['history']} == {'a', 'b'}
    assert not report['promoted']
    evaluated = evaluate_checkpoint(path, seeds=(20000,), spacings=(.72,))
    assert len(evaluated['candidate']['matches']) == 6
    assert evaluated['candidate']['safety_stops'] == 0
    assert (tmp_path / 'tactical_evaluation.json').is_file()


def test_balanced_imitation_uses_all_qualified_action_classes():
    from shadowbox.rl.ppo import _balanced_imitation_loss

    logits = torch.tensor([[2., 0.], [2., 0.], [2., 0.], [2., 0.]],
                          requires_grad=True)
    distribution = torch.distributions.Categorical(logits=logits)
    actions = torch.tensor([0, 0, 0, 1])
    loss = _balanced_imitation_loss(distribution, actions)
    log_probs = distribution.log_prob(actions)
    expected = -(log_probs[:3].mean() + log_probs[3]) / 2
    assert torch.allclose(loss, expected)
    loss.backward()
    assert logits.grad is not None


def test_diverse_examples_cover_both_sides_and_ranges():
    from scripts.tactical_iteration import _diverse_examples

    rows = [dict(side=side, spacing=spacing, opponent='reactive', source='probe')
            for side in ('a', 'b') for spacing in (.62, .72, .9)]
    chosen = _diverse_examples(rows, 4)
    assert {row['side'] for row in chosen} == {'a', 'b'}
    assert len({row['spacing'] for row in chosen}) >= 2


def test_seeded_actor_sampling_reproducible_and_masked():
    from shadowbox.rl.policy import sample_action

    logits = torch.tensor([[0., 0., 20.]])
    distribution = torch.distributions.Categorical(logits=logits.masked_fill(
        torch.tensor([[False, False, True]]), -1e9))
    first = torch.Generator().manual_seed(59)
    second = torch.Generator().manual_seed(59)
    a = [sample_action(distribution, first) for _ in range(20)]
    b = [sample_action(distribution, second) for _ in range(20)]
    assert a == b
    assert set(a) <= {0, 1}
