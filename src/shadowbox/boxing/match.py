"""Seeded boxing matches, virtual-touch referee, and metrics."""

from dataclasses import asdict
import logging
import math
import numpy as np

from ..config import load_config, source_fingerprint
from ..motion import MotionController, Primitive
from ..simulation import Arena
from ..safety import Supervisor
from .policy import Parameters, Observation, Policy, PUNCHES
from .perception import Perception
from .geometry import sphere_entry, capsule_entry


def _action_name(action):
    return action.value if isinstance(action, Primitive) else str(action)


class Match:
    """Seeded two-agent match. Scores are virtual touches, not impact or injury."""

    def __init__(self, seed: int = 0, defense_only: bool = False,
                 round_seconds: float | None = None, rounds: int | None = None,
                 parameters_a=None, parameters_b=None, config=None):
        self.seed = seed
        self.defense_only = defense_only
        self.config = config or load_config()
        self.round_seconds = round_seconds if round_seconds is not None else self.config["match"]["round_seconds"]
        self.rounds = rounds if rounds is not None else self.config["match"]["rounds"]
        self.break_seconds = self.config["match"]["break_seconds"]
        self.parameters = {"a": self._parameters(parameters_a if parameters_a is not None else self.config["policies"]["a"]),
                           "b": self._parameters(parameters_b if parameters_b is not None else self.config["policies"]["b"])}
        self.arena = Arena(config=self.config)
        self.reset()

    @staticmethod
    def _parameters(value):
        return value if isinstance(value, Parameters) else Parameters(**(value or {}))

    def reset(self) -> None:
        self.arena.reset()
        self.controllers = {r: MotionController(self.arena, r) for r in ("a", "b")}
        self.footwork = {}
        if (self.config.get("footwork", {}).get("enabled", False)
                and self.config["simulation"].get("model") == "unitree_g1"):
            from ..motion.footwork import FootworkController
            from ..motion.stance import BoxingStance
            stance = BoxingStance(self.arena)
            self.footwork = {r: FootworkController(self.arena, r, stance) for r in ("a", "b")}
        self.policies = {r: Policy(self.seed + (0 if r == "a" else 1009), self.parameters[r],
                                  self.defense_only and r == "b", rules=self.config.get("policy_rules")) for r in ("a", "b")}
        sensor_settings = self.config.get('perception', {})
        self.sensor_enabled = any(sensor_settings.values())
        self.sensors = {r: Perception(self.seed + 9000 + i, **sensor_settings)
                        for i, r in enumerate(('a', 'b'))}
        self.supervisor = Supervisor(self.arena, self.defense_only)
        self.stats = {r: dict(attempts=0, touches=0, blocks=0, misses=0,
                               evasion_attempts=0, evasion_success=0,
                               offense_attempts=0, offense_success=0) for r in ("a", "b")}
        self.events = []
        self.scorecards = []
        self.round_touches = {"a": 0, "b": 0}
        self.round = 1
        self.phase = "BOX"
        self.phase_started = 0.0
        self.finished = False
        self.next_decision = {"a": 0.0, "b": 0.12}
        self.punch = {"a": None, "b": None}
        self.previous_gloves = {r: self.arena.glove_positions(r).copy() for r in ("a", "b")}
        self.previous_positions = {r: self.arena.position(r).copy() for r in ("a", "b")}
        self.glove_velocities = {r: np.zeros((2, 3)) for r in ("a", "b")}
        self.velocities = {r: np.zeros(3) for r in ("a", "b")}
        self.punch_start = {"a": None, "b": None}
        self.punch_duration = {"a": 0.0, "b": 0.0}

    def observe(self, robot: str) -> Observation:
        other = "b" if robot == "a" else "a"
        position = self.arena.position(robot).copy()
        opponent = self.arena.position(other).copy()
        incoming = 0.0
        incoming_body = False
        head_height = float(self.arena.data.geom_xpos[self.arena.model.geom(f"{robot}_head").id, 2])
        for glove, velocity in zip(self.arena.glove_positions(other), self.glove_velocities[other]):
            direction = position[:2] - glove[:2]
            speed = float(np.dot(velocity[:2], direction) / max(np.linalg.norm(direction), 1e-6))
            if speed > incoming:
                incoming = speed
                incoming_body = glove[2] < head_height - 0.12
        return Observation(position, opponent, float(self.arena.pose(robot)[2]),
                           float(np.linalg.norm(position[:2] - opponent[:2])),
                           (self.velocities[other] - self.velocities[robot]).copy(), incoming,
                           float(self.supervisor.boundary - self.supervisor.config["boundary_margin"] - np.max(np.abs(position[:2]))),
                           _action_name(self.controllers[robot].active), self.supervisor.stopped, incoming_body)

    def _event(self, robot: str, event: str, **values) -> None:
        record = dict(time=round(self.arena.time, 3), robot=robot, event=event, **values)
        self.events.append(record)
        logging.getLogger(__name__).info("%s", record)

    def _request(self, robot: str, primitive) -> bool:
        controller = self.controllers[robot]
        # Handle SuperMimic motion names (strings) vs Primitive enum
        if isinstance(primitive, str) and not isinstance(primitive, Primitive):
            # SuperMimic motion name
            target = controller.plan(None, motion_name=primitive)
            if target is None:
                return False
            if not self.supervisor.approve(robot, primitive, target):
                self._event(robot, "rejected", action=primitive, reason=self.supervisor.reason)
                return False
            if not controller.start(primitive, target=target):
                return False
            self._event(robot, "action", action=primitive)
            # Track evasion for punch motions
            if primitive in ('jab', 'cross', 'uppercut', 'hook_left', 'hook_right', 'combo_1_2', 'combo_1_1_2', 'combo_1_2_3', 'body_hook_left', 'body_hook_right', 'lead_jab', 'rear_cross', 'feint_jab', 'counter_right'):
                other = "b" if robot == "a" else "a"
                self.stats[robot]["attempts"] += 1
                self.stats[other]["evasion_attempts"] += 1
                self.punch[robot] = dict(side=1 if primitive in ('cross', 'hook_right', 'uppercut', 'combo_1_2', 'combo_1_1_2', 'combo_1_2_3', 'rear_cross', 'counter_right') else 0,
                                         resolved=False)
                # Track punch start position and duration
                self.punch_start[robot] = self.arena.glove_positions(robot)[self.punch[robot]['side']].copy()
                self.punch_duration[robot] = 0.0
                # Start footwork for punch motions
                if robot in self.footwork:
                    start_root = self.arena.targets(robot)
                    target_root = np.array([start_root[0] + 0.5, start_root[1], start_root[2]])  # Step forward while punching
                    self.footwork[robot].start(Primitive.STEP_FORWARD, target_root, 0.8)
            return True
        else:
            # Original Primitive enum handling
            target = controller.plan(primitive)
            if primitive == Primitive.STEP_FORWARD:
                current = self.arena.targets(robot)[:2]
                delta = target[:2] - current
                distance = self.observe(robot).distance
                stride = min(float(np.linalg.norm(delta)), max(0, distance - self.parameters[robot].desired_distance))
                target[:2] = current + delta * stride / max(np.linalg.norm(delta), 1e-9)
            if not self.supervisor.approve(robot, primitive, target):
                self._event(robot, "rejected", action=primitive.value, reason=self.supervisor.reason)
                return False
            if not controller.start(primitive, target=target):
                return False
            self._event(robot, "action", action=primitive.value)
            if primitive in PUNCHES:
                other = "b" if robot == "a" else "a"
                self.stats[robot]["attempts"] += 1
                self.stats[other]["evasion_attempts"] += 1  # Opponent must evade
                self.punch[robot] = dict(side=1 if primitive == Primitive.CROSS else 0, resolved=False)
            return True

    def _finish_punch(self, robot: str) -> None:
        punch = self.punch[robot]
        if punch and not punch["resolved"]:
            self.stats[robot]["misses"] += 1
            self._event(robot, "miss")
        self.punch[robot] = None

    def _referee(self, robot: str, gloves: np.ndarray) -> None:
        punch = self.punch[robot]
        if not punch or punch["resolved"]:
            return
        other = "b" if robot == "a" else "a"
        side = punch["side"]
        start, end = self.previous_gloves[robot][side], gloves[side]
        position = self.arena.position(other)
        # Simple distance-based touch detection (fallback)
        glove_to_opponent = np.linalg.norm(end[:2] - position[:2])
        if self.config['scoring']['proximity_fallback'] and glove_to_opponent < 0.4:
            # Check for blocking
            for index, defending_glove in enumerate(self.arena.glove_positions(other)):
                name = ("left", "right")[index]
                glove_dist = np.linalg.norm(end[:2] - defending_glove[:2])
                if glove_dist < 0.15:  # Blocking distance
                    punch["resolved"] = True
                    self.controllers[robot].cancel()
                    self.stats[other]["blocks"] += 1
                    self.stats[other]["evasion_success"] += 1
                    self._event(robot, "blocked", defender=other)
                    return
            # No block, check for touch
            punch["resolved"] = True
            self.controllers[robot].cancel()
            # Determine target based on glove height
            head_height = float(self.arena.data.geom_xpos[self.arena.model.geom(f"{other}_head").id, 2])
            target = "head" if end[2] > head_height - 0.1 else "torso"
            self.stats[robot]["touches"] += 1
            self.stats[robot]["offense_success"] += 1
            self.round_touches[robot] += 1
            self._event(robot, "touch", target=target)
            return
        glove_name = ("left", "right")[side]
        glove_radius = self.arena.model.geom_size[self.arena.model.geom(f"{robot}_{glove_name}_glove").id, 0]
        collisions = []
        for index, defending_glove in enumerate(self.arena.glove_positions(other)):
            name = ("left", "right")[index]
            radius = self.arena.model.geom_size[self.arena.model.geom(f"{other}_{name}_glove").id, 0]
            relative_start = start - self.previous_gloves[other][index]
            relative_end = end - defending_glove
            entry = sphere_entry(relative_start, relative_end, np.zeros(3), glove_radius + radius)
            if entry is not None:
                collisions.append((entry, "blocked", name))
        # Shift the old hand into the target's current frame to include target translation.
        relative_start = start + position - self.previous_positions[other]
        head_id = self.arena.model.geom(f"{other}_head").id
        head = self.arena.data.geom_xpos[head_id]
        entry = sphere_entry(relative_start, end, head,
                             glove_radius + self.arena.model.geom_size[head_id, 0])
        if entry is not None:
            collisions.append((entry, "touch", "head"))
        torso_id = self.arena.model.geom(f"{other}_torso").id
        center = self.arena.data.geom_xpos[torso_id]
        axis = self.arena.data.geom_xmat[torso_id].reshape(3, 3)[:, 2]
        half_length = self.arena.model.geom_size[torso_id, 1]
        entry = capsule_entry(relative_start, end, center - half_length * axis,
                              center + half_length * axis,
                              glove_radius + self.arena.model.geom_size[torso_id, 0])
        if entry is not None:
            collisions.append((entry, "touch", "torso"))
        yaw = self.arena.pose(other)[2]
        forward = np.array([math.cos(yaw), math.sin(yaw)])
        for entry, kind, target in sorted(collisions):
            hit = start + entry * (end - start)
            if kind == "touch" and np.dot(hit[:2] - position[:2], forward) < -0.04:
                continue
            punch["resolved"] = True
            # Stop the extension at the first measured contact; next tick recovers guard.
            self.controllers[robot].cancel()
            if kind == "blocked":
                self.stats[other]["blocks"] += 1
                self.stats[other]["evasion_success"] += 1  # Successfully blocked = evasion success
                self._event(robot, "blocked", defender=other)
            else:
                self.stats[robot]["touches"] += 1
                self.stats[robot]["offense_success"] += 1  # Touch = offense success
                self.round_touches[robot] += 1
                self._event(robot, "touch", target=target)
            return

    def stop(self) -> None:
        self.supervisor.stopped = True
        self.supervisor.reason = "emergency stop"
        for robot, controller in self.controllers.items():
            self._finish_punch(robot)
            controller.cancel()
            if robot in self.footwork:
                self.footwork[robot].cancel()
        self._event("referee", "stop")

    def _round_clock(self) -> None:
        elapsed = self.arena.time - self.phase_started
        if self.phase == "BOX" and elapsed >= self.round_seconds:
            a, b = self.round_touches["a"], self.round_touches["b"]
            self.scorecards.append({"round": self.round, "touches": dict(self.round_touches),
                                    "a": 10 if a >= b else 9, "b": 10 if b >= a else 9})
            for robot, controller in self.controllers.items():
                self._finish_punch(robot)
                controller.cancel()
                if robot in self.footwork:
                    self.footwork[robot].cancel()
            self._event("referee", "round_end", round=self.round)
            self.phase, self.phase_started = "BREAK", self.arena.time
            if self.round >= self.rounds:
                self.phase, self.finished = "FINISHED", True
        elif self.phase == "BREAK" and elapsed >= self.break_seconds:
            self.round += 1
            self.round_touches = {"a": 0, "b": 0}
            self.phase, self.phase_started = "BOX", self.arena.time
            self._event("referee", "round_start", round=self.round)

    def step(self) -> None:
        if self.finished or self.supervisor.stopped:
            return
        self._round_clock()
        if self.finished:
            return
        readings = None
        if self.sensor_enabled:
            readings = {r: self.sensors[r].read(self.observe(r), self.arena.time) for r in ('a', 'b')}
        if self.phase == "BOX":
            for robot, controller in self.controllers.items():
                observation = readings[robot] if readings is not None else self.observe(robot)
                if controller.busy and controller.active in (Primitive.GUARD,
                        Primitive.RETURN_TO_GUARD, Primitive.STEP_FORWARD,
                        Primitive.STEP_BACKWARD, Primitive.SIDESTEP_LEFT,
                        Primitive.SIDESTEP_RIGHT, 'slip_left', 'slip_right', 'parry', 'bob_and_weave',
                        'combo_1_2', 'combo_1_1_2', 'combo_1_2_3', 'hook_left', 'hook_right', 'uppercut', 'body_hook_left', 'body_hook_right', 'lead_jab', 'rear_cross') and observation.incoming_body and observation.incoming_speed > .25 * self.config['policy_rules']['distance_scale'] and observation.distance < .95 * self.config['policy_rules']['distance_scale']:
                    controller.cancel()
                    if robot in self.footwork:
                        self.footwork[robot].cancel()
                    self._request(robot, Primitive.BODY_GUARD)
                if not controller.busy:
                    if self.punch[robot]:
                        self._finish_punch(robot)
                        self._request(robot, Primitive.RETURN_TO_GUARD)
                    elif self.arena.time >= self.next_decision[robot]:
                        policy = self.policies[robot]
                        mimic = controller.supermimic
                        primitive = (policy.choose(observation, mimic)
                                     if mimic.enabled and type(policy) is Policy
                                     else policy.choose(observation))
                        self._request(robot, primitive)
                        self.next_decision[robot] = self.arena.time + self.parameters[robot].reaction
                was_busy = controller.busy
                controller.tick()
                if robot in self.footwork:
                    self.footwork[robot].tick(controller)
                if was_busy and not controller.busy:
                    self.next_decision[robot] = self.arena.time + self.parameters[robot].reaction
        elif self.footwork:
            for footwork in self.footwork.values():
                footwork.tick()
        self.arena.step()
        dt = self.arena.model.opt.timestep
        for robot in ("a", "b"):
            gloves = self.arena.glove_positions(robot).copy()
            position = self.arena.position(robot).copy()
            self.glove_velocities[robot] = (gloves - self.previous_gloves[robot]) / dt
            self.velocities[robot] = (position - self.previous_positions[robot]) / dt
            if self.phase == "BOX":
                self._referee(robot, gloves)
            # Update punch duration
            if self.punch[robot] and not self.punch[robot]["resolved"]:
                self.punch_duration[robot] += dt
        self.previous_gloves = {r: self.arena.glove_positions(r).copy() for r in ("a", "b")}
        self.previous_positions = {r: self.arena.position(r).copy() for r in ("a", "b")}
        if not self.supervisor.monitor():
            for controller in self.controllers.values():
                controller.cancel()
            for footwork in self.footwork.values():
                footwork.cancel()
            self._event("referee", "override", reason=self.supervisor.reason)

    def status(self) -> list[str]:
        remaining = max(0.0, (self.round_seconds if self.phase == "BOX" else self.break_seconds)
                        - (self.arena.time - self.phase_started))
        if self.finished:
            remaining = 0.0
        totals = {r: sum(card[r] for card in self.scorecards) for r in ('a', 'b')}
        return [f"ShadowBox | round {self.round}/{self.rounds} | {self.phase} | {remaining:.1f}s",
                f"A: {_action_name(self.controllers['a'].active)} | B: {_action_name(self.controllers['b'].active)}",
                f"Touches A {self.stats['a']['touches']} : {self.stats['b']['touches']} B | "
                f"Blocks A {self.stats['a']['blocks']} : {self.stats['b']['blocks']} B",
                f"Round cards A {totals['a']} : {totals['b']} B | B mode: {'DEFENSE ONLY' if self.defense_only else 'SPARRING'} | "
                f"Safety: {self.supervisor.reason}",
                f"Policies A {type(self.policies['a']).__name__} / B {type(self.policies['b']).__name__}",
                "Virtual touches | stabilized bases | contact physics deferred"]

    def result(self) -> dict:
        totals = {r: sum(card[r] for card in self.scorecards) for r in ("a", "b")}
        return dict(seed=self.seed, time=round(self.arena.time, 3), finished=self.finished,
                    defense_only=self.defense_only, stats=self.stats, scorecards=self.scorecards,
                    scores=totals, safety_rejections=self.supervisor.rejections,
                    stopped=self.supervisor.stopped, safety_reason=self.supervisor.reason,
                    parameters={r: asdict(p) for r, p in self.parameters.items()},
                    config=self.config, source_fingerprint=source_fingerprint(), events=self.events,
                    # Add evasion and offense metrics
                    evasion_attempts_a=self.stats['a']['evasion_attempts'],
                    evasion_success_a=self.stats['a']['evasion_success'],
                    evasion_attempts_b=self.stats['b']['evasion_attempts'],
                    evasion_success_b=self.stats['b']['evasion_success'],
                    offense_attempts_a=self.stats['a']['attempts'],
                    offense_success_a=self.stats['a']['offense_success'],
                    offense_attempts_b=self.stats['b']['attempts'],
                    offense_success_b=self.stats['b']['offense_success'])
