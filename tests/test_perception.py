import numpy as np
import pytest

from shadowbox.boxing.policy import Observation
from shadowbox.boxing.perception import Perception


def obs(x, *, state='guard', stopped=False):
    return Observation(np.array([x, 0., 0.]), np.array([2. + x, 1., 0.]), 0., 1.,
                       np.array([0., 1., 0.]), 0.5, 1.2, state, stopped)


def test_zero_noise_zero_delay_is_value_identity_without_mutation():
    original = obs(0.2)
    before = original.position.copy()
    reading = Perception(1).read(original, 0)
    assert np.array_equal(reading.position, original.position)
    assert reading is not original and reading.position is not original.position
    assert np.array_equal(original.position, before)


def test_seeded_noise_is_reproducible_and_keeps_distance_consistent():
    source = obs(0.2)
    a = Perception(7, 0.1, 0.2).read(source, 0)
    b = Perception(7, 0.1, 0.2).read(source, 0)
    assert np.array_equal(a.position, b.position)
    assert np.array_equal(a.opponent_position, b.opponent_position)
    assert np.array_equal(a.relative_velocity, b.relative_velocity)
    assert a.distance == pytest.approx(np.linalg.norm(a.position[:2] - a.opponent_position[:2]))


def test_delay_uses_latest_available_sample_at_or_before_target():
    sensor = Perception(1, delay_seconds=0.5)
    sensor.read(obs(0), 0.0)
    sensor.read(obs(1), 0.4)
    sensor.read(obs(2), 0.7)
    assert sensor.read(obs(3), 1.0).position[0] == 1


def test_first_sample_fallback_and_current_stopped_status():
    sensor = Perception(1, delay_seconds=10)
    sensor.read(obs(0), 0)
    current = obs(2, state='punch', stopped=True)
    result = sensor.read(current, 1)
    assert result.position[0] == 0
    assert result.state == 'punch'
    assert result.stopped is True


def test_sensor_history_remains_bounded():
    sensor = Perception(seed=1, delay_seconds=0.05)
    from shadowbox.boxing import Match
    match = Match()
    observation = match.observe('a')
    for step in range(1000):
        sensor.read(observation, step * 0.01)
    assert len(sensor.history) <= 8
