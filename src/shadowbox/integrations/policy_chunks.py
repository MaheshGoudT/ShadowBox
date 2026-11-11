"""Optional serving/action-chunk benchmark; no ManiFlow checkpoint dependency.

Servers must explicitly decode their model output into ShadowBox tactical action
indices. Franka continuous actions are incompatible and are rejected here.
"""

import argparse
from collections import deque
import json
from pathlib import Path
import time
from typing import Protocol
from urllib.request import Request, urlopen

import numpy as np

from ..config import artifact_path

DEFENSIVE_ACTIONS = frozenset((0, 1, 3, 4, 5, 8))


class ChunkPolicy(Protocol):
    def __call__(self, request: dict) -> dict: ...


class HttpChunkPolicy:
    """JSON request/response transport for an explicitly compatible policy server."""

    def __init__(self, endpoint, timeout=5):
        if not endpoint.startswith(('http://', 'https://')):
            raise ValueError('Policy endpoint must use http or https')
        self.endpoint, self.timeout = endpoint, timeout

    def __call__(self, request):
        data = json.dumps(request, allow_nan=False).encode()
        req = Request(self.endpoint, data=data, headers={'Content-Type': 'application/json'})
        with urlopen(req, timeout=self.timeout) as response:
            body = response.read(1_048_577)
        if len(body) > 1_048_576:
            raise ValueError('Policy response exceeds 1 MiB')
        return json.loads(body)


class ActionChunkController:
    """Revalidate every queued action against current state; clear on reset."""

    def __init__(self, policy: ChunkPolicy, horizon=8):
        if type(horizon) is not int or not 1 <= horizon <= 64:
            raise ValueError('Chunk horizon must be 1..64')
        self.policy, self.horizon = policy, horizon
        self.reset()

    def reset(self):
        self.pending = deque()
        self.calls = self.masked = 0
        self.latencies = []

    def action(self, observation, action_mask):
        obs = np.asarray(observation, dtype=np.float32)
        mask = np.asarray(action_mask, dtype=bool)
        if obs.shape != (12,) or not np.isfinite(obs).all() or mask.shape != (10,):
            raise ValueError('Expected finite ShadowBox observation[12] and action_mask[10]')
        mask = mask & np.asarray([i in DEFENSIVE_ACTIONS for i in range(10)])
        if not mask.any():
            raise ValueError('No legal defensive action')
        if not self.pending:
            start = time.perf_counter()
            response = self.policy(dict(schema='shadowbox.tactical.v1', observation=obs.tolist(),
                                        action_mask=mask.tolist(), max_actions=self.horizon))
            self.latencies.append(time.perf_counter() - start)
            self.calls += 1
            if not isinstance(response, dict) or response.get('schema') != 'shadowbox.tactical.v1':
                raise ValueError('Policy response needs schema shadowbox.tactical.v1')
            actions = response.get('actions')
            if (not isinstance(actions, list) or not 1 <= len(actions) <= self.horizon
                    or any(type(a) is not int or a not in DEFENSIVE_ACTIONS for a in actions)):
                raise ValueError('Expected a chunk of defensive tactical indices; arm/joint actions are unsupported')
            self.pending.extend(actions)
        action = self.pending.popleft()
        if not mask[action]:
            self.pending.clear()
            self.masked += 1
            return int(np.flatnonzero(mask)[0])
        return action


def benchmark(env, controller, steps=32, seed=0):
    controller.reset()
    obs, info = env.reset(seed=seed)
    total_reward = 0.0
    executed, episodes = 0, 0
    try:
        for _ in range(steps):
            action = controller.action(obs, info['action_mask'])
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            executed += 1
            if terminated or truncated:
                episodes += 1
                break  # A new episode must start with a fresh chunk.
        return dict(scope='serving/action-chunk plumbing; no humanoid control or vision validation',
                    steps=executed, seed=seed, episodes=episodes, reward=total_reward,
                    policy_calls=controller.calls, masked_actions=controller.masked,
                    inference_seconds=controller.latencies)
    finally:
        controller.pending.clear()
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--endpoint', help='optional compatible JSON policy server')
    parser.add_argument('--config', type=Path)
    parser.add_argument('--steps', type=int, default=32)
    parser.add_argument('--horizon', type=int, default=4)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output', type=Path, default=Path('tmp/playground/policy-benchmark.json'))
    args = parser.parse_args()
    if args.steps < 1:
        parser.error('--steps must be positive')
    from ..rl.environment import BoxingEnv
    try:
        policy = HttpChunkPolicy(args.endpoint) if args.endpoint else lambda request: dict(
            schema=request['schema'], actions=[0] * request['max_actions'])
        controller = ActionChunkController(policy, args.horizon)
        report = benchmark(BoxingEnv(args.config), controller, args.steps, args.seed)
        report['policy'] = args.endpoint or 'synthetic guard chunks (no checkpoint)'
        output = artifact_path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report, indent=2))
        return 0
    except (OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    raise SystemExit(main())
