"""Lead/rear boxing steps with planted world-space foot targets."""

import math

import numpy as np

from .primitives import Primitive
from .stance import BoxingStance

MOVES = (Primitive.STEP_FORWARD, Primitive.STEP_BACKWARD, Primitive.SIDESTEP_LEFT,
         Primitive.SIDESTEP_RIGHT, Primitive.TURN_TO_TARGET)


def blend(t):
    return 10*t**3 - 15*t**4 + 6*t**5


class FootworkController:
    """Kinematic foot placement while the simulated pelvis remains supported.

    The first foot swings, plants, then the following foot moves. A planted
    foot keeps its world position and heading rather than moving with the root.
    """

    def __init__(self, arena, robot, stance=None):
        self.arena, self.robot = arena, robot
        self.stance = stance or BoxingStance(arena)
        self.settings = arena.config.get('footwork', {})
        self.reset()

    def reset(self):
        self.targets = self.stance.foot_positions(self.robot)
        self.headings = self.stance.foot_headings(self.robot)
        self.busy, self.swinging = False, None
        self.elapsed = 0.0
        self._motion = None
        self.cancelled = False
        self.max_planted_slip = 0.0
        self.minimum_sole_height = min(point[2] for point in self.targets.values())

    def start(self, primitive, target_root, duration, start_root=None):
        if self.busy or primitive not in MOVES:
            return False
        start_root = self.arena.targets(self.robot) if start_root is None else start_root
        target_root = np.asarray(target_root)
        displacement = target_root[:2] - start_root[:2]
        turn = math.remainder(target_root[2] - start_root[2], 2*math.pi)
        if np.linalg.norm(displacement) < 1e-6 and abs(turn) < 1e-6:
            return False
        self.starts = self.stance.foot_positions(self.robot)
        for point in self.starts.values():
            point[2] = self.stance.config.get('sole_height', .005)
        self.ends = {side: point.copy() for side, point in self.starts.items()}
        self.start_headings = dict(self.headings)
        self.end_headings = dict(self.headings)
        if primitive == Primitive.TURN_TO_TARGET:
            c, s = math.cos(turn), math.sin(turn)
            rotation = np.array([[c, -s], [s, c]])
            for side in self.ends:
                relative = self.starts[side][:2] - start_root[:2]
                self.ends[side][:2] = target_root[:2] + rotation @ relative
                self.end_headings[side] += turn
        else:
            for point in self.ends.values():
                point[:2] += displacement
        self.order = (('right', 'left') if primitive in
                      (Primitive.STEP_BACKWARD, Primitive.SIDESTEP_RIGHT) else ('left', 'right'))
        self.duration, self.elapsed = float(duration), 0.0
        self.busy, self.cancelled = True, False
        return True

    def cancel(self):
        """Land a raised foot locally; reset rebuilds the complete stance."""
        if not self.busy:
            return
        self.starts = {side: point.copy() for side, point in self.targets.items()}
        self.ends = {side: point.copy() for side, point in self.starts.items()}
        for point in self.ends.values():
            point[2] = self.stance.config.get('sole_height', .005)
        self.duration = self.settings.get('landing_seconds', .25)
        self.elapsed, self.cancelled = 0.0, True
        self.start_headings, self.end_headings = dict(self.headings), dict(self.headings)

    def tick(self, controller=None):
        if controller is not None:
            motion = getattr(controller, 'segments', None)
            if controller.active in MOVES and controller.busy and motion is not self._motion:
                self.start(controller.active, controller.goal, controller.duration, controller.begin)
                self._motion = motion
            if controller.active == Primitive.IDLE and self.busy and not self.cancelled:
                self.cancel()
            # Cancel footwork when controller switches to non-footwork primitive
            if controller.active not in MOVES and self.busy and not self.cancelled:
                self.cancel()
        if self.busy:
            if controller is not None and not self.cancelled and controller.segments is self._motion:
                self.elapsed = controller.elapsed
            else:
                self.elapsed += self.arena.model.opt.timestep
            phase = min(1.0, self.elapsed / self.duration)
            if self.cancelled:
                for side in self.targets:
                    self.targets[side] = self.starts[side] + blend(phase) * (self.ends[side]-self.starts[side])
                self.swinging = None
            else:
                first = phase < .5
                self.swinging = self.order[0 if first else 1]
                progress = phase*2 if first else phase*2-1
                for side in self.targets:
                    fraction = progress if side == self.swinging else (0.0 if first else 1.0)
                    if side == self.order[1] and first:
                        fraction = 0.0
                    if side == self.order[0] and not first:
                        fraction = 1.0
                    self.targets[side] = self.starts[side] + blend(fraction) * (self.ends[side]-self.starts[side])
                    if side == self.swinging:
                        self.targets[side][2] += self.settings.get('lift_height', .025) * math.sin(math.pi*progress)**2
                    self.headings[side] = self.start_headings[side] + blend(fraction) * math.remainder(
                        self.end_headings[side]-self.start_headings[side], 2*math.pi)
            if phase >= 1:
                self.busy, self.swinging = False, None
                self.targets = {side: point.copy() for side, point in self.ends.items()}
                self.headings = dict(self.end_headings)
        self.stance.apply(self.robot, self.targets, self.headings)

    def measure(self):
        positions = self.stance.foot_positions(self.robot)
        for side, point in positions.items():
            self.minimum_sole_height = min(self.minimum_sole_height, float(point[2]))
            if side != self.swinging:
                slip = float(np.linalg.norm(point[:2]-self.targets[side][:2]))
                self.max_planted_slip = max(self.max_planted_slip, slip)
        return positions
