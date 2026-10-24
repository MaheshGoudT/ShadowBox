"""Shape helpers for the decision-step tactical environment."""

from .actions import ACTIONS
from .observations import OBSERVATION_SIZE


class ObservationSpace:
    @staticmethod
    def create():
        return (OBSERVATION_SIZE,)


class ActionSpace:
    @staticmethod
    def create():
        return len(ACTIONS)
