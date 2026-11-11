"""Measure native CPU standing/opponent failures before scaling training."""
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import torch
from mujoco_playground._src.rl import env_registry, rng
from reaf.simulation.torch import environment
from shadowbox.playground.registry import register_tasks
from shadowbox.playground.task import TASK_ID, contact_force, upright


def main():
    register_tasks()
    cases = []
    for distance in (0.6, 1.1):
        for attacking in (False, True):
            rng.seed(10001)
            env = env_registry.get(TASK_ID).env_builder(nworld=1, device='cpu', impl='cpp',
                start_distance=distance, opponent_enabled=attacking)
            try:
                env.auto_reset_mode = environment.AutoResetMode.DISABLED
                env.reset()
                lengths = []
                failures = {'a': 0, 'b': 0, 'bounds': 0}
                incoming = outgoing = 0.0
                count = 0
                with torch.inference_mode():
                    for _ in range(1000):
                        timestep = env.step({'action': torch.zeros(1, 29)})
                        count += 1
                        f = env._latest_features  # Confined to this pinned-runtime diagnostic.
                        incoming = max(incoming, contact_force(f, 'incoming').item())
                        outgoing = max(outgoing, contact_force(f, 'outgoing').item())
                        if not torch.isfinite(env.device_layer.sim.data.qpos).all():
                            raise ValueError('Nonfinite physics')
                        if timestep.last().any():
                            lengths.append(count)
                            count = 0
                            for side in ('a', 'b'):
                                failures[side] += int((f[f'{side}_root/xpos'][0, 2] < 0.45)
                                                      or (upright(f, side)[0] < 0.4))
                            failures['bounds'] += int(any((f[f'{side}_root/xpos'][0, :2].abs() > 3).any() for side in ('a', 'b')))
                            env.reset()
                cases.append(dict(distance=distance, opponent_enabled=attacking, failures=failures,
                    completed_episode_steps=lengths, censored_steps=count,
                    max_incoming_force=incoming, max_outgoing_force=outgoing))
            finally:
                env.close()
    output = Path('tmp/playground/standing-diagnostic.json')
    output.write_text(json.dumps(dict(seed=10001, steps_per_case=1000,
        task_sha256=hashlib.sha256(Path('src/shadowbox/playground/task.py').read_bytes()).hexdigest(),
        cases=cases), indent=2) + '\n')
    print(output.read_text())


if __name__ == '__main__':
    main()
