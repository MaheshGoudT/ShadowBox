"""Reinforcement learning environment for ShadowBox."""

__all__ = ['BoxingEnv', 'ObservationSpace', 'ActionSpace']


def __getattr__(name):
    if name == 'BoxingEnv':
        from .environment import BoxingEnv
        return BoxingEnv
    if name in ('ObservationSpace', 'ActionSpace'):
        from .state import ObservationSpace, ActionSpace
        return {'ObservationSpace': ObservationSpace, 'ActionSpace': ActionSpace}[name]
    raise AttributeError(name)
