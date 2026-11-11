"""The bounded action vocabulary shared by both boxers."""

from enum import Enum

class Primitive(str, Enum):
    IDLE = "idle"
    GUARD = "guard"
    BODY_GUARD = "body_guard"
    STEP_FORWARD = "step_forward"
    STEP_BACKWARD = "step_backward"
    SIDESTEP_LEFT = "sidestep_left"
    SIDESTEP_RIGHT = "sidestep_right"
    TURN_TO_TARGET = "turn_to_target"
    JAB = "jab"
    CROSS = "cross"
    UPPERCUT = "uppercut"
    KICK = "kick"
    BOB_AND_WEAVE = "bob_and_weave"
    DUCK = "duck"
    RETURN_TO_GUARD = "return_to_guard"


# Stable tactical order shared by local and external execution backends.
TACTICAL_ACTIONS = (Primitive.GUARD, Primitive.TURN_TO_TARGET, Primitive.STEP_FORWARD,
                   Primitive.STEP_BACKWARD, Primitive.SIDESTEP_LEFT, Primitive.SIDESTEP_RIGHT,
                   Primitive.JAB, Primitive.CROSS, Primitive.BODY_GUARD, Primitive.UPPERCUT)
