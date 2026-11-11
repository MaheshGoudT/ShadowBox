"""Shared motion vocabulary; legacy controller imports only when requested."""
from .primitives import Primitive

__all__ = ['Primitive', 'MotionController']


def __getattr__(name):
    if name == 'MotionController':
        from .controller import MotionController
        return MotionController
    raise AttributeError(name)
