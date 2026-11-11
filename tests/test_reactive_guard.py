"""Causal threat detection, hold and reset behavior of the geometry baseline."""
import numpy as np
import pytest
from shadowbox.playground.tactical_policy import ReactiveGuard


def test_body_closing_threat_holds_then_expires_and_head_takes_priority():
    policy = ReactiveGuard(1)
    head = np.array([[0., 0., 1.5]])
    torso = np.array([[0., 0., 1.]])
    gloves = np.array([[[.5, 0., 1.], [1., 0., 1.5]]])
    assert policy.choose(gloves, head, torso).item() == 0
    gloves[0, 0, 0] = .35
    assert policy.choose(gloves, head, torso).item() == 8
    for _ in range(5):
        assert policy.choose(gloves, head, torso).item() == 8
    gloves[0, 1, 0] = .2
    assert policy.choose(gloves, head, torso).item() == 0
    gloves[0, 0, 0] = .3
    assert policy.choose(gloves, head, torso).item() == 8
    for _ in range(16):
        result = policy.choose(gloves, head, torso)
    assert result.item() == 0
    policy.reset()
    assert policy.choose(gloves, head, torso).item() == 0


def test_rejects_broadcast_target_geometry():
    with pytest.raises(ValueError, match='geometry'):
        ReactiveGuard(2).choose(np.zeros((2, 2, 3)), np.zeros(3), np.zeros((2, 3)))
