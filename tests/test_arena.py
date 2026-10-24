import numpy as np
import pytest

from shadowbox.motion import MotionController, Primitive
from shadowbox.simulation import Arena, LIMITS
from shadowbox.config import BASE_CONFIG, load_config


def test_reset_restores_both_humanoids_and_physics():
    arena = Arena(config=load_config(BASE_CONFIG))
    np.testing.assert_allclose(arena.position('a'), [-1.25, 0, 0])
    np.testing.assert_allclose(arena.position('b'), [1.25, 0, 0])
    arena.step(20)
    first = arena.data.qpos.copy()
    arena.reset()
    arena.step(20)
    np.testing.assert_allclose(arena.data.qpos, first, atol=1e-8)


def test_each_primitive_completes_with_bounded_setpoints():
    arena = Arena()
    controller = MotionController(arena, 'a')
    lower, upper = np.array(LIMITS).T
    for primitive in Primitive:
        controller.run(primitive)
        assert not controller.busy
        assert np.all(arena.targets('a') <= upper)
        assert np.all(arena.targets('a') >= lower)
        assert np.isfinite(arena.pose('a')).all()


def test_headless_simulation_runs_for_sixty_seconds():
    arena = Arena()
    arena.step(6000)
    assert arena.time == pytest.approx(60)
    assert np.isfinite(arena.data.qpos).all()


def test_motion_does_not_overwrite_active_request_and_cancel_holds_setpoint():
    arena = Arena()
    controller = MotionController(arena, 'a')
    assert controller.start(Primitive.JAB)
    assert not controller.start(Primitive.CROSS)
    for _ in range(10):
        controller.tick()
        arena.step()
    held = arena.targets('a')
    controller.cancel()
    controller.tick()
    assert not controller.busy
    np.testing.assert_array_equal(arena.targets('a'), held)


def test_smooth_motion_setpoint_velocity_and_acceleration():
    arena = Arena()
    controller = MotionController(arena, 'a')
    history = [arena.targets('a')]
    controller.start(Primitive.JAB)
    while controller.busy:
        controller.tick()
        history.append(arena.targets('a'))
        arena.step()
    velocity = np.diff(history, axis=0) / 0.01
    acceleration = np.diff(velocity, axis=0) / 0.01
    assert abs(velocity).max() <= 7.01
    assert abs(acceleration).max() <= 55.1
