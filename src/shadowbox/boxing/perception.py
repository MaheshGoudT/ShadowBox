"""Seeded noisy and delayed observations for a policy's private sensor feed."""

from copy import deepcopy
from dataclasses import replace
import numpy as np

from .policy import Observation


class Perception:
    """Cache timestamped observations and return policy-facing sensor readings.

    Boundary space is retained as an ideal boundary signal. Safety state and
    the observing robot's active state always come from the current observation.
    """

    def __init__(self, seed, position_noise=0.0, velocity_noise=0.0, delay_seconds=0.0):
        for name, value in (("position_noise", position_noise), ("velocity_noise", velocity_noise),
                            ("delay_seconds", delay_seconds)):
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        self.position_noise = float(position_noise)
        self.velocity_noise = float(velocity_noise)
        self.delay_seconds = float(delay_seconds)
        self.rng = np.random.default_rng(seed)
        self.history = []

    def _sensor_copy(self, observation):
        result = deepcopy(observation)
        if self.position_noise:
            for name in ("position", "opponent_position"):
                position = getattr(result, name)
                position[:2] += self.rng.normal(0, self.position_noise, 2)
        if self.velocity_noise:
            result.relative_velocity[:] += self.rng.normal(0, self.velocity_noise, result.relative_velocity.shape)
        delta = result.position[:2] - result.opponent_position[:2]
        return replace(result, distance=float(np.linalg.norm(delta)),
                       incoming_speed=max(0.0, result.incoming_speed + self.rng.normal(0, self.velocity_noise))
                       if self.velocity_noise else result.incoming_speed)

    def read(self, observation: Observation, time: float) -> Observation:
        """Return a fresh reading, delayed to the newest cached sample at/before target time."""
        time = float(time)
        if not np.isfinite(time):
            raise ValueError("time must be finite")
        if self.history and time == self.history[-1][0]:
            sample = self.history[-1][1]
        else:
            sample = self._sensor_copy(observation)
            self.history.append((time, sample))
            self.history.sort(key=lambda item: item[0])
        target = time - self.delay_seconds
        # Retain the sample just before the delay target and all newer samples.
        while len(self.history) > 1 and self.history[1][0] <= target:
            self.history.pop(0)
        delayed = self.history[0][1]
        return replace(deepcopy(delayed), state=observation.state,
                       stopped=observation.stopped, boundary_space=observation.boundary_space)
