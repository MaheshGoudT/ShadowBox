from shadowbox.simulation.timing import SimulationClock


def test_physics_clock_advances_same_time_at_different_frame_rates():
    fast, slow = SimulationClock(0.01), SimulationClock(0.01)
    assert sum(fast.steps(0.01) for _ in range(100)) == 100
    assert sum(slow.steps(0.1) for _ in range(10)) == 100


def test_pause_does_not_accumulate_a_physics_jump():
    clock = SimulationClock(0.01)
    assert clock.steps(20, active=False) == 0
    assert clock.steps(0.01) == 1
