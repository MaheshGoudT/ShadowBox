"""Isolated Playground rollout worker with transition and checkpoint evidence."""

import argparse
import json
from pathlib import Path
import time
import sys
import dataclasses
import hashlib

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--env', required=True)
    parser.add_argument('--device', required=True)
    parser.add_argument('--env_overrides', required=True)
    parser.add_argument('--nworld', type=int, required=True)
    parser.add_argument('--steps', type=int, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--tactical-policy', choices=('guard','reactive'), default='guard')
    parser.add_argument('--zero_actions', '--zero-actions', dest='zero_actions', action='store_true')
    args = parser.parse_args()
    if args.zero_actions and args.checkpoint:
        parser.error('--zero-actions cannot be combined with --checkpoint')
    import torch
    import mujoco
    import numpy as np
    from mujoco_playground._src.rl import env_registry, rng
    from mujoco_playground._src.rl.rsl_rl.vecenv_wrapper import RslRlVecEnvWrapper

    from shadowbox.playground.registry import register_tasks
    register_tasks()
    rng.seed(args.seed)
    entry = env_registry.get(args.env)
    raw = entry.env_builder(nworld=args.nworld, device=args.device,
                            **json.loads(args.env_overrides))
    try:
        env = RslRlVecEnvWrapper(raw, cfg=entry.runner_cfg.vecenv_wrapper_cfg)
        action = torch.zeros((args.nworld, env.num_actions), device=args.device)
        policy = None
        identity = None
        if args.checkpoint:
            from rsl_rl.runners import OnPolicyRunner
            from mujoco_playground._src.rl.rsl_rl import checkpoint
            runner_config = dataclasses.asdict(entry.runner_cfg)
            runner_config['torch_compile_mode'] = None
            runner = OnPolicyRunner(env, runner_config, device=args.device)
            runner.load(checkpoint.read_checkpoint(str(args.checkpoint)), map_location=args.device)
            policy = runner.get_inference_policy(device=args.device)
            path = args.checkpoint
            if path.is_dir():
                path = Path(checkpoint.get_latest_checkpoint(path))
            identity = dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            manifest = path.parent.parent / 'run.json'
            if manifest.is_file():
                training = json.loads(manifest.read_text())
                hashes = training.get('provenance', {}).get('local_sources', {})
                project = Path(__file__).resolve().parents[3]
                changed = [name for name, expected in hashes.items()
                           if not (project/name).is_file() or
                           hashlib.sha256((project/name).read_bytes()).hexdigest() != expected]
                identity.update(training_manifest=str(manifest), training_source_changes=changed,
                                source_scope='diagnostic if recorded sources changed; no automatic promotion')
        reactive=None
        if args.tactical_policy=='reactive' and not args.zero_actions:
            if args.env!='ShadowBoxG1Motion' or args.checkpoint:
                raise ValueError('Reactive primitive policy requires ShadowBoxG1Motion without a checkpoint')
            from shadowbox.playground.tactical_policy import ReactiveGuard
            reactive=ReactiveGuard(args.nworld)
        obs = env.get_observations()
        start = time.monotonic()
        completed = 0
        traces = {key: [] for key in ('qpos', 'qvel', 'ctrl', 'action', 'qpos_next', 'reward', 'done', 'timeout')}
        sim = raw.device_layer.sim
        mujoco.mj_saveModel(sim.mj_model, str(args.output.with_name('model.mjb')), None)
        if args.env in ('ShadowBoxG1Defense','ShadowBoxG1Motion'):
            traces['boxing_events'] = []
        if args.env=='ShadowBoxG1Motion':
            traces['requested_primitives']=[]
            traces['inference_ms']=[]
        episode_steps = torch.zeros(args.nworld, device=args.device, dtype=torch.long)
        episode_lengths = []
        with torch.inference_mode():
            for _ in range(args.steps):
                if policy is not None:
                    action = policy(obs)
                elif reactive is not None:
                    model=sim.mj_model
                    ids=[model.geom('b/'+hand+'_glove').id for hand in ('left','right')]
                    gloves=sim.data.geom_xpos[:,ids].cpu().numpy()
                    head=sim.data.geom_xpos[:,model.geom('a/head_collision').id].cpu().numpy()
                    body=sim.data.geom_xpos[:,model.geom('a/torso_collision').id].cpu().numpy()
                    action=torch.as_tensor(reactive.choose(gloves,head,body),device=args.device)
                traces['qpos'].append(sim.data.qpos.cpu().numpy().copy())
                traces['qvel'].append(sim.data.qvel.cpu().numpy().copy())
                obs, reward, done, extras = env.step(action)
                traces['ctrl'].append(sim.data.ctrl.cpu().numpy().copy())
                traces['timeout'].append(extras.get('time_outs',torch.zeros_like(done)).cpu().numpy().copy())
                traces['action'].append(action.cpu().numpy().copy())
                traces['qpos_next'].append(sim.data.qpos.cpu().numpy().copy())
                traces['reward'].append(reward.cpu().numpy().copy())
                traces['done'].append(done.cpu().numpy().copy())
                if 'requested_primitives' in traces:
                    f=raw._latest_features
                    traces['requested_primitives'].append(np.stack([f['motion/a_primitive'].cpu().numpy(),f['motion/b_primitive'].cpu().numpy()],axis=-1))
                    traces['inference_ms'].append(f['motion/inference_ms'].cpu().numpy().copy())
                if 'boxing_events' in traces:
                    # Terminal rows are post-reset; they are excluded from counts below.
                    traces['boxing_events'].append(raw._latest_features['boxing/events'].cpu().numpy().copy())
                if not torch.isfinite(reward).all() or not all(
                        torch.isfinite(value).all() for value in obs.values()):
                    raise ValueError('Nonfinite Playground observations or rewards')
                completed += int(done.sum().item())
                if reactive is not None and done.any():
                    reactive.reset(done.cpu().numpy().astype(bool))
                episode_steps += 1
                episode_lengths.extend(episode_steps[done.bool()].cpu().tolist())
                episode_steps[done.bool()] = 0
        if args.device.startswith('cuda'):
            torch.cuda.synchronize()
        seconds = time.monotonic() - start
        motion_identity = None
        if args.env=='ShadowBoxG1Motion':
            motion_identity=raw.task_layer.get_component_by_name('motion').policy.provenance()
        contact_counts = None
        if 'boxing_events' in traces:
            events = np.stack(traces['boxing_events'])
            valid = ~np.stack(traces['done']).astype(bool)
            contact_counts = (events * valid[..., None, None]).sum(axis=(0, 1)).tolist()
        np.savez_compressed(args.output.with_name('transitions.npz'), **{k: np.stack(v) for k, v in traces.items()})
        args.output.write_text(json.dumps(dict(task=args.env, nworld=args.nworld,
            steps=args.steps, seed=args.seed, completed_episodes=completed,
            seconds=seconds, transitions_per_second=args.nworld * args.steps / seconds,
            transitions='transitions.npz',
            motion_policy=motion_identity,
            checkpoint=identity, policy=('checkpoint' if policy is not None else
                'zero' if args.zero_actions else
                args.tactical_policy+'_primitive' if args.env=='ShadowBoxG1Motion' else 'zero'),
            contact_episode_counts_by_channel_and_glove=contact_counts,
            contact_channel_order=['incoming_head', 'incoming_body', 'incoming_block', 'outgoing_head', 'outgoing_body'],
            completed_episode_steps=episode_lengths,
            censored_episode_steps=episode_steps.cpu().tolist(),
            mean_step_reward=float(np.stack(traces['reward']).mean()),
            transition_semantics='qpos_next is post-reset when done; exclude done rows from dynamics fitting',
            scope='diagnostic rollout; no automatic policy promotion'), indent=2) + '\n')
    finally:
        raw.close()


if __name__ == '__main__':
    main()
