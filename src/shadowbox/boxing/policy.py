"""Independent decisions from observed robot state."""

from dataclasses import dataclass
import math
import numpy as np
from ..motion import Primitive

PUNCHES = (Primitive.JAB, Primitive.CROSS, Primitive.UPPERCUT)

@dataclass(frozen=True)
class Parameters:
    aggression: float = 0.72
    reaction: float = 0.18
    desired_distance: float = 0.78

    def __post_init__(self):
        for name, lower, upper in (("aggression", 0, 1), ("reaction", 0.05, 2),
                                   ("desired_distance", 0.3, 1.5)):
            value = getattr(self, name)
            if not math.isfinite(value) or not lower <= value <= upper:
                raise ValueError(f"{name} must be finite and within [{lower}, {upper}]")


@dataclass(frozen=True)
class Observation:
    position: np.ndarray
    opponent_position: np.ndarray
    yaw: float
    distance: float
    relative_velocity: np.ndarray
    incoming_speed: float
    boundary_space: float
    state: str
    stopped: bool
    incoming_body: bool = False


class Policy:
    """Uses measured positions and glove motion, never another policy's plans."""

    def __init__(self, seed: int, parameters: Parameters, defense_only: bool = False, rules=None):
        self.rng = np.random.default_rng(seed)
        self.parameters = parameters
        self.defense_only = defense_only
        self.rules = rules or {"distance_scale": 1.0}

    def choose(self, observation: Observation, supermimic=None) -> Primitive:
        p, o = self.parameters, observation
        scale = self.rules["distance_scale"]
        delta = o.opponent_position[:2] - o.position[:2]
        bearing = math.atan2(delta[1], delta[0])
        if abs(math.remainder(bearing - o.yaw, 2 * math.pi)) > 0.18:
            return Primitive.TURN_TO_TARGET
        threatened = o.incoming_speed > 0.25 * scale and o.distance < 1.25 * scale
        if threatened and o.incoming_body and o.distance < 0.95 * scale:
            return Primitive.BODY_GUARD
        if o.distance < 0.68 * scale or (threatened and (self.defense_only or self.rng.random() > p.aggression * 0.4)):
            # Prefer defensive motions when threatened
            if supermimic and supermimic.enabled and supermimic.motions:
                # Use parry/slip for active defense
                if threatened and 'slip_left' in supermimic.motions:
                    return ('slip_left', 'slip_right')[int(self.rng.integers(2))]
                if 'parry' in supermimic.motions:
                    return 'parry'
                if 'bob_and_weave' in supermimic.motions:
                    return 'bob_and_weave'
                if 'duck' in supermimic.motions:
                    return 'duck'
            if o.boundary_space > 0.4 * scale and self.rng.random() < 0.55:
                return Primitive.STEP_BACKWARD
            return (Primitive.SIDESTEP_LEFT, Primitive.SIDESTEP_RIGHT)[int(self.rng.integers(2))]
        if self.defense_only:
            if o.distance < p.desired_distance and np.linalg.norm(o.relative_velocity) > 0.1:
                return Primitive.STEP_BACKWARD if o.boundary_space > 0.4 * scale else Primitive.GUARD
            return Primitive.GUARD
        if o.distance > p.desired_distance + 0.08 * scale:
            return Primitive.STEP_FORWARD
        if self.rng.random() < p.aggression:
            # Use SuperMimic motions including combos
            if supermimic and supermimic.enabled and supermimic.motions:
                motions = ['uppercut', 'hook_left', 'hook_right', 'combo_1_2', 'combo_1_1_2', 'combo_1_2_3', 'body_hook_left', 'body_hook_right', 'lead_jab', 'rear_cross', 'feint_jab', 'counter_right']
                available = [m for m in motions if m in supermimic.motions]
                if available:
                    return available[int(self.rng.integers(len(available)))]
            return PUNCHES[int(self.rng.integers(len(PUNCHES)))]
        return (Primitive.GUARD, Primitive.SIDESTEP_LEFT, Primitive.SIDESTEP_RIGHT)[int(self.rng.integers(3))]
