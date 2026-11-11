"""Bounded native Metatron qualification, before any Playground policy adapter.

Uses the pinned ProtoMotions public configuration/environment/agent lifecycle.
This is a one-robot transfer baseline, not the two-player boxing environment.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=root/'tmp/playground/protomotions-source')
    parser.add_argument('--bundle', type=Path, default=Path('/home/mahesh/Desktop/huggingfacehub/models/nvidia/Metatron-v1/ada816723d62c194c82ffd8fab97a3d8a001f94d/G1'))
    parser.add_argument('--eager', action='store_true', help='explicitly disable torch.compile for CPU qualification')
    parser.add_argument('--onnx', type=Path, help='drive native physics with the exported graph')
    parser.add_argument('--record-all-feeds', action='store_true')
    parser.add_argument('--compare-native', action='store_true', help='compare graph actions with the native policy each step')
    parser.add_argument('--steps', type=int, default=300)
    parser.add_argument('--output', type=Path, default=root/'tmp/playground/metatron/native-idle.json')
    args = parser.parse_args()
    if args.onnx:
        args.onnx = args.onnx.resolve()
    args.source = args.source.resolve()
    args.bundle = args.bundle.resolve()
    args.output = args.output.resolve()
    if args.steps <= 0 or not args.output.is_relative_to(root/'tmp'):
        parser.error('Require positive steps and output under ShadowBox tmp/')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = args.bundle/'inference_last.ckpt'
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    expected = 'e5fc2e47a8f9ee9b96a674cabbf236b3c510f887427645a14bc1779b10f0b43a'
    if digest != expected:
        raise ValueError('Metatron checkpoint differs from the verified official bundle')
    source = subprocess.run(['git', '-C', str(args.source), 'rev-parse', 'HEAD'], check=True, capture_output=True, text=True).stdout.strip()
    if source != '13de0cccd06f53ce5c9cf9ec7d1932c7da73610f':
        raise ValueError('ProtoMotions source differs from qualification pin')
    os.chdir(args.source)
    if args.eager:
        os.environ['TORCH_COMPILE_DISABLE'] = '1'
    import torch
    import numpy as np
    torch.set_num_threads(8)
    from lightning.fabric import Fabric
    from protomotions.launch.inference import create_inference_parser
    from protomotions.utils.experiment_configs import load_experiment_configs_from_checkpoint
    from protomotions.utils.inference_utils import derive_inference_configs, recipe_hooks
    from protomotions.utils.component_builder import build_env_from_configs
    from protomotions.utils.hydra_replacement import get_class
    from protomotions.utils.fabric_config import FabricConfig
    options = create_inference_parser().parse_args(['--checkpoint', str(checkpoint), '--simulator', 'mujoco', '--headless', '--num-envs', '1', '--inference-config', str(args.source/'examples/inference/metatron.py')])
    configs = load_experiment_configs_from_checkpoint(checkpoint)
    view = derive_inference_configs(configs, options, {'simulator.num_envs': 1, 'simulator.headless': True}, recipe_hooks(args.bundle, Path(options.inference_config)))
    view.robot.asset.asset_root = str(root/'tmp/playground/protomotions-assets')
    fabric = Fabric(**FabricConfig(accelerator='cpu', devices=1, num_nodes=1, loggers=[], callbacks=[]).as_kwargs())
    fabric.launch()
    env = None
    report = dict(source=source, checkpoint_sha256=digest, hub_revision='ada816723d62c194c82ffd8fab97a3d8a001f94d', status='running', steps=args.steps, torch_compile=not args.eager,
                  scope='native single G1 idle; no Playground transfer or boxing claim')
    try:
        env = build_env_from_configs(view, fabric.device)
        agent = get_class(view.agent._target_)(config=view.agent, env=env, fabric=fabric, root_dir=args.output.parent)
        agent.setup(training=False)
        agent.load(str(checkpoint), load_env=False, load_training_state=False, for_inference=True)
        agent.eval()
        session = None
        feed_specs = []
        if args.onnx:
            import yaml
            import onnxruntime as ort
            onnx = args.onnx.resolve()
            metadata = yaml.safe_load(onnx.with_suffix('.yaml').read_text())
            options = ort.SessionOptions()
            options.intra_op_num_threads = 8
            options.inter_op_num_threads = 1
            session = ort.InferenceSession(str(onnx), sess_options=options, providers=['CPUExecutionProvider'])
            feed_specs = metadata['policy_inputs']
            report['onnx_sha256'] = hashlib.sha256(onnx.read_bytes()).hexdigest()
        elif args.compare_native:
            raise ValueError('--compare-native requires --onnx')
        done_indices = None
        completed = 0
        traces = {name: [] for name in ('root_pos', 'root_rot', 'action', 'done')}
        latencies = []
        parity_errors = []
        target_errors = []
        feeds = []
        start = time.monotonic()
        with torch.inference_mode():
            for step in range(args.steps):
                obs, _ = env.reset(done_indices)
                agent.pre_collect_step(step)
                obs = agent.add_agent_info_to_obs(obs)
                inputs = agent.obs_dict_to_tensordict(obs)
                tick = time.monotonic()
                if session is None:
                    action = agent.evaluator._policy_action(inputs)
                else:
                    feed = {spec['name']: np.asarray(env.context[spec['key']].cpu().numpy(), dtype=spec['dtype'])
                            for spec in feed_specs}
                    outputs = dict(zip((s.name for s in session.get_outputs()), session.run(None, feed)))
                    action = torch.from_numpy(outputs['actions'])
                    if args.record_all_feeds or step in (0, 1, 8, 16, args.steps-1):
                        feeds.append((step, {key: value.copy() for key, value in feed.items()}))
                latencies.append(time.monotonic()-tick)
                if args.compare_native:
                    native = agent.evaluator._policy_action(inputs)
                    parity_errors.append(float((native-action).abs().max()))
                if not torch.isfinite(action).all():
                    raise ValueError('Nonfinite native Metatron actions')
                _, _, dones, _, _ = env.step(action)
                if session is not None:
                    actual = env.context['current_processed_action'].cpu().numpy()
                    target_errors.append(float(np.max(np.abs(actual-outputs['joint_pos_targets']))))
                state = env.simulator.get_root_state()
                if not torch.isfinite(state.root_pos).all():
                    raise ValueError('Nonfinite native MuJoCo state')
                for key, value in (('root_pos',state.root_pos),('root_rot',state.root_rot),('action',action),('done',dones)):
                    traces[key].append(value.cpu().numpy().copy())
                completed += int(dones.sum())
                done_indices = dones.nonzero(as_tuple=False).squeeze(-1)
        np.savez_compressed(args.output.with_suffix('.npz'), **{k: np.stack(v) for k,v in traces.items()})
        if feeds:
            np.savez_compressed(args.output.with_name(args.output.stem+'-feeds.npz'),
                                **{f'step_{step}_{key}': value for step, feed in feeds for key,value in feed.items()})
        height = np.stack(traces['root_pos'])[:,0,2]
        quat = np.stack(traces['root_rot'])[:,0]
        upright = 1-2*(quat[:,0]**2+quat[:,1]**2)
        warmup = min(100, args.steps//3)
        report.update(status='completed', seconds=time.monotonic()-start, completed_episodes=completed,
                      minimum_root_height=float(np.stack(traces['root_pos'])[...,2].min()),
                      mean_inference_ms=1000*float(np.mean(latencies)), p95_inference_ms=1000*float(np.quantile(latencies,.95)),
                      max_action_parity_error=max(parity_errors, default=None),
                      max_target_conversion_error=max(target_errors, default=None), quaternion_order='xyzw',
                      warmup_steps=warmup, minimum_height_after_warmup=float(height[warmup:].min()),
                      min_upright_after_warmup=float(upright[warmup:].min()),
                      unsafe_frames_after_warmup=int(((height[warmup:]<.65)|(upright[warmup:]<.75)).sum()),
                      termination_scope='Native task disables fall termination; posture is checked independently')
    except Exception as error:
        report.update(status='failed',error=str(error))
        raise
    finally:
        args.output.write_text(json.dumps(report, indent=2)+'\n')
        if env is not None:
            env.simulator.shutdown()
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
