"""Pinned Playground 1.0 adapter: primary scalable simulation/RL integration."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import zipfile

import yaml

from ..config import PROJECT_ROOT, artifact_path
from .backend import run_backend

DEFAULT_CONFIG = PROJECT_ROOT / 'configs/playground.yaml'


class PlaygroundBackend:
    name = 'playground'

    def __init__(self, config=None):
        try:
            value = yaml.safe_load(Path(config or DEFAULT_CONFIG).read_text())
        except (OSError, yaml.YAMLError) as error:
            raise ValueError(f'Could not read Playground configuration: {config or DEFAULT_CONFIG}') from error
        if not isinstance(value, dict) or set(value) != {'backend', 'playground'} or value['backend'] != self.name:
            raise ValueError('Playground requires separate backend/playground config sections')
        settings = value['playground']
        required = {'python', 'checkout', 'revision', 'task', 'device', 'num_envs',
                    'iterations', 'smoke_steps', 'seed', 'trajectory', 'checkpoint',
                    'timeout_seconds', 'output_dir'}
        if (not isinstance(settings, dict) or not required <= set(settings)
                or set(settings) - required - {'task_options','tactical_policy','viewer','play_steps','gif'}):
            raise ValueError('Playground config has missing or unknown settings')
        if type(settings.get('viewer', False)) is not bool:
            raise ValueError('playground.viewer must be boolean')
        if type(settings.get('play_steps', 500)) is not int or not 1 <= settings.get('play_steps', 500) <= 100000:
            raise ValueError('playground.play_steps must be an integer in [1, 100000]')
        gif = settings.get('gif', {})
        if (not isinstance(gif, dict) or set(gif) - {'enabled', 'directory'}):
            raise ValueError('playground.gif accepts only enabled and directory')
        if type(gif.get('enabled', True)) is not bool:
            raise ValueError('playground.gif.enabled must be boolean')
        if 'directory' in gif and (not isinstance(gif['directory'], str) or not gif['directory'].strip()):
            raise ValueError('playground.gif.directory must be a nonempty path')
        if settings.get('tactical_policy','guard') not in ('guard','reactive'):
            raise ValueError('tactical_policy must be guard or reactive')
        if settings.get('tactical_policy')=='reactive' and settings['task']!='ShadowBoxG1Motion':
            raise ValueError('Reactive primitive policy requires ShadowBoxG1Motion')
        self.config_path = Path(config or DEFAULT_CONFIG).resolve()
        self.settings = settings.copy()
        options = settings.get('task_options', {})
        if not isinstance(options, dict) or set(options) - {'start_distance', 'opponent_enabled', 'episode_length_secs', 'terminate_opponent',
                'motion_bundle', 'motion_script', 'motion_runtime', 'motion_device', 'defender_guard_width',
                'opponent_guard_width', 'head_target_z', 'body_target_z', 'head_overshoot', 'body_overshoot',
                'opponent_attack_target', 'opponent_attack_delay_secs', 'controlled_side',
                'reset_yaw_radians', 'reset_lateral_m', 'retreat_speed_a', 'retreat_speed_b',
                'swap_start_sides'}:
            raise ValueError('Unknown Playground task_options')
        if options and settings['task'] not in ('ShadowBoxG1Defense','ShadowBoxG1Motion'):
            raise ValueError('task_options apply only to ShadowBox G1 boxing tasks')
        for key, lower, upper in (('start_distance', 0.5, 2.0), ('episode_length_secs', 0.1, 120),
                ('defender_guard_width', .10, .50),('opponent_guard_width',.10,.50),
                ('head_target_z',.25,.55),('body_target_z',-.05,.35),
                ('head_overshoot',0.,.30),('body_overshoot',0.,.30),
                ('opponent_attack_delay_secs',0.,1.5),
                ('reset_yaw_radians',0,.15), ('reset_lateral_m',0,.05),
                ('retreat_speed_a',0.,.4),('retreat_speed_b',0.,.4)):
            if key in options and (type(options[key]) not in (int, float) or not lower <= options[key] <= upper):
                raise ValueError(f'task_options.{key} must be in [{lower}, {upper}]')
        for key in ('opponent_enabled', 'terminate_opponent','swap_start_sides'):
            if key in options and type(options[key]) is not bool:
                raise ValueError(f'task_options.{key} must be boolean')
        if ('opponent_attack_target' in options
                and options['opponent_attack_target'] not in ('alternating','head','body')):
            raise ValueError('task_options.opponent_attack_target must be alternating, head or body')
        if 'controlled_side' in options and options['controlled_side'] not in ('a','b'):
            raise ValueError('task_options.controlled_side must be a or b')
        motion_keys={'motion_bundle','motion_script','motion_runtime','motion_device','defender_guard_width',
            'opponent_guard_width','head_target_z','body_target_z','head_overshoot','body_overshoot',
            'opponent_attack_target','opponent_attack_delay_secs','controlled_side',
            'retreat_speed_a','retreat_speed_b'}
        if motion_keys & set(options) and settings['task']!='ShadowBoxG1Motion':
            raise ValueError('Motion bundle options require ShadowBoxG1Motion')
        if settings['task']=='ShadowBoxG1Motion':
            if settings['num_envs']!=1 or not options.get('motion_bundle'):
                raise ValueError('Tracker-backed boxing requires one world and a qualified motion_bundle')
            for key in ('motion_bundle','motion_script'):
                if options.get(key) is not None and (not isinstance(options[key],str) or not options[key].strip()):
                    raise ValueError(f'task_options.{key} must be a path')
            if options.get('motion_runtime','torchscript') not in ('torchscript','torchscript_graph','onnx'):
                raise ValueError('motion_runtime must be torchscript, torchscript_graph or onnx')
            if options.get('motion_device','cuda:0') not in ('cpu','cuda:0'):
                raise ValueError('Qualified motion devices are cpu and cuda:0')
        for key in ('python', 'checkout', 'revision', 'task', 'device', 'output_dir'):
            if not isinstance(settings[key], str) or not settings[key].strip():
                raise ValueError(f'playground.{key} must be a nonempty string')
        for key in ('num_envs', 'iterations', 'smoke_steps', 'timeout_seconds', 'seed'):
            minimum = 0 if key == 'seed' else 1
            if type(settings[key]) is not int or settings[key] < minimum:
                raise ValueError(f'playground.{key} must be an integer >= {minimum}')
        if not re.fullmatch(r'[0-9a-f]{40}', settings['revision']):
            raise ValueError('Playground revision must be a full Git commit')
        if settings['task'] not in ('ShadowBoxG1Defense', 'ShadowBoxG1Motion', 'G1WholeBodyTracking', 'CartpoleBalance'):
            raise ValueError('Supported Playground tasks: ShadowBoxG1Defense, ShadowBoxG1Motion, G1WholeBodyTracking, CartpoleBalance')
        if not re.fullmatch(r'cpu|cuda:\d+', settings['device']):
            raise ValueError('playground.device must be cpu or cuda:N')
        if settings['device'] == 'cpu' and settings['num_envs'] != 1:
            raise ValueError('Pinned Playground cpp CPU backend supports num_envs: 1 only')
        for key in ('trajectory', 'checkpoint'):
            if settings[key] is not None and (not isinstance(settings[key], str) or not settings[key].strip()):
                raise ValueError(f'playground.{key} must be a path or null')
        # abspath preserves the virtualenv interpreter symlink.
        self.python = Path(os.path.abspath(artifact_path(Path(settings['python']).expanduser())))
        self.checkout = artifact_path(Path(settings['checkout']).expanduser()).resolve()
        self.output_dir = artifact_path(Path(settings['output_dir']).expanduser()).resolve()
        if not self.output_dir.is_relative_to((PROJECT_ROOT / 'tmp').resolve()):
            raise ValueError('Playground output_dir must be inside ShadowBox tmp/')
        gif = settings.get('gif', {})
        self.gif_enabled = gif.get('enabled', True)
        self.gif_dir = artifact_path(Path(gif.get('directory', 'tmp/playground/gifs')).expanduser()).resolve()
        if not self.gif_dir.is_relative_to((PROJECT_ROOT / 'tmp').resolve()):
            raise ValueError('Playground gif.directory must be inside ShadowBox tmp/')
        self.timeout_seconds = settings['timeout_seconds']

    def provenance(self):
        return {'checkout': str(self.checkout), 'revision': self.settings['revision'],
                'task': self.settings['task'], 'device': self.settings['device'],
                'local_sources': {str(p.relative_to(PROJECT_ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in sorted((PROJECT_ROOT / 'src/shadowbox/playground').glob('*.py'))},
                'scope': 'two free-base G1 defensive task; trained boxing capability unqualified'}

    def capture_sources(self, run_dir):
        archive = run_dir / 'source.zip'
        files = sorted((PROJECT_ROOT / 'src/shadowbox').rglob('*.py'))
        files += sorted((PROJECT_ROOT / 'requirements').glob('*.txt'))
        with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as bundle:
            for path in files:
                bundle.write(path, str(path.relative_to(PROJECT_ROOT)))
            bundle.writestr('run-config.yaml', self.config_path.read_text())
        return {'path': archive.name, 'sha256': hashlib.sha256(archive.read_bytes()).hexdigest()}

    def _path(self, key):
        value = self.settings[key]
        return artifact_path(Path(value).expanduser()).resolve() if value else None

    def probe(self):
        state = dict(ready=False, **self.provenance())
        try:
            head = subprocess.run(['git', '-C', str(self.checkout), 'rev-parse', 'HEAD'],
                                  capture_output=True, text=True, timeout=10, check=True).stdout.strip()
            if head != self.settings['revision']:
                raise ValueError('Playground checkout differs from configured revision')
            dirty = subprocess.run(['git', '-C', str(self.checkout), 'status', '--porcelain',
                                    '--untracked-files=no'], capture_output=True, text=True,
                                   timeout=10, check=True).stdout.strip()
            if dirty:
                raise ValueError('Playground checkout has tracked changes; restore or deliberately update the pin')
            if not self.python.is_file():
                raise ValueError('Separate Playground environment is not installed; see docs/playground.md')
            code = (
                'import json,sys,torch,mujoco_playground,rsl_rl,reaf; '
                + ('import onnxruntime; ' if self.settings['task']=='ShadowBoxG1Motion' else '') +
                f'sys.path.insert(0, {str(PROJECT_ROOT / "src")!r}); '
                'from shadowbox.playground.registry import register_tasks; register_tasks(); '
                'from mujoco_playground._src.rl import env_registry; '
                f'env_registry.get({self.settings["task"]!r}); '
                'print("SHADOWBOX_PROBE="+json.dumps(dict(prefix=sys.prefix,base_prefix=sys.base_prefix,'
                'python=list(sys.version_info[:2]),module=mujoco_playground.__file__,'
                'cuda=torch.cuda.is_available(),devices=torch.cuda.device_count())))'
            )
            env = os.environ.copy()
            env.pop('PYTHONPATH', None)
            env['PYTHONNOUSERSITE'] = '1'
            result = subprocess.run([str(self.python), '-c', code], cwd=PROJECT_ROOT,
                                    env=env, capture_output=True, text=True, timeout=45)
            lines = [line.removeprefix('SHADOWBOX_PROBE=') for line in result.stdout.splitlines()
                     if line.startswith('SHADOWBOX_PROBE=')]
            if result.returncode or not lines:
                raise ValueError((result.stderr or result.stdout or 'Runtime probe failed')[-2000:])
            state.update(json.loads(lines[-1]))
            if (state['prefix'] == state['base_prefix'] or
                    Path(state['prefix']).resolve() == (PROJECT_ROOT / '.venv').resolve()):
                raise ValueError('Playground requires an isolated virtual environment')
            if state['python'] not in ([3, 12], [3, 13]):
                raise ValueError('Pinned Playground requires Python 3.12 or 3.13')
            if not Path(state['module']).resolve().is_relative_to(self.checkout):
                raise ValueError('Install the pinned checkout editable; unrelated Playground package detected')
            if self.settings['device'].startswith('cuda'):
                index = int(self.settings['device'].split(':')[1])
                if not state['cuda'] or index >= state['devices']:
                    raise ValueError('Configured CUDA device is unavailable; no CPU fallback')
            if self.settings['task'] == 'G1WholeBodyTracking':
                trajectory = self._path('trajectory')
                if trajectory is None or not trajectory.is_file():
                    raise ValueError('G1 tracking requires a converted reference trajectory; see docs/playground.md')
            if self.settings['task']=='ShadowBoxG1Motion':
                options=self.settings['task_options']
                runtime=options.get('motion_runtime','torchscript')
                device=options.get('motion_device','cuda:0')
                if runtime=='onnx' and device!='cpu':
                    raise ValueError('Qualified ONNX motion runtime requires CPU')
                if runtime!='onnx' and (device!='cuda:0' or not state['cuda'] or state['devices']<1):
                    raise ValueError('Qualified TorchScript motion policy requires CUDA:0; no fallback')
                bundle=artifact_path(Path(options['motion_bundle'])).resolve()
                required=[bundle/'unified_pipeline.yaml',bundle/'pelvis_fk/fk.yaml',bundle/'pelvis_fk/fk.onnx']
                if runtime=='onnx':
                    required.append(bundle/'unified_pipeline.onnx')
                else:
                    if not options.get('motion_script'):
                        raise ValueError('TorchScript motion runtime requires an explicit motion_script')
                    script=artifact_path(Path(options['motion_script'])).resolve()
                    required.extend((script,script.with_suffix('.json')))
                missing=[str(path) for path in required if not path.is_file()]
                if missing:
                    raise ValueError('Missing motion artifacts: '+', '.join(missing))
                if runtime!='onnx':
                    qualification=json.loads(script.with_suffix('.json').read_text())
                    if (not isinstance(qualification,dict)
                            or not isinstance(qualification.get('qualified_batch_sizes'),list)
                            or not isinstance(qualification.get('cuda_graph_qualification',{}),dict)):
                        raise ValueError('Malformed motion artifact qualification')
                    if (not qualification.get('passed') or 2 not in qualification.get('qualified_batch_sizes',[])
                            or qualification.get('script_sha256')!=hashlib.sha256(script.read_bytes()).hexdigest()
                            or qualification.get('sidecar_sha256')!=hashlib.sha256((bundle/'unified_pipeline.yaml').read_bytes()).hexdigest()
                            or (runtime=='torchscript_graph' and not qualification.get('cuda_graph_qualification',{}).get('passed'))):
                        raise ValueError('Motion artifact has no matching successful batch-two qualification')
                state['motion_artifact_scope']='Files and recorded export qualification checked; live robot/goal contracts checked during task construction'
            state['ready'] = True
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            state['error'] = str(error)
        return state

    def command(self, stage, run_dir):
        if stage not in ('smoke', 'train', 'play', 'evaluate'):
            raise ValueError('Playground stages: smoke, train, play, evaluate')
        s = self.settings
        overrides = {'impl': 'cpp' if s['device'] == 'cpu' else 'warp'}
        overrides.update(s.get('task_options', {}))
        for key in ('motion_bundle','motion_script'):
            if overrides.get(key):
                overrides[key]=str(artifact_path(Path(overrides[key]).expanduser()).resolve())
        if s['task'] == 'G1WholeBodyTracking':
            path = self._path('trajectory')
            if path is None:
                raise ValueError('G1 tracking requires trajectory')
            overrides['trajectory_path'] = str(path)
        common = [f'--env={s["task"]}', f'--device={s["device"]}',
                  '--env_overrides=' + json.dumps(overrides)]
        viewer = getattr(self, 'viewer', s.get('viewer', False))
        headless_play = stage == 'play' and not viewer
        checkpoint = self._path('checkpoint')
        zero_actions = getattr(self, 'zero_actions', False)
        if stage == 'play':
            if zero_actions and checkpoint is not None:
                raise ValueError('Choose either --zero-actions or a configured checkpoint')
            reactive_policy = (not viewer and s.get('tactical_policy', 'guard') == 'reactive'
                               and s['task'] == 'ShadowBoxG1Motion')
            if checkpoint is None and not zero_actions and not reactive_policy:
                raise ValueError('Playground play requires checkpoint or explicit --zero-actions')
            if checkpoint is not None and not checkpoint.exists():
                raise ValueError(f'Checkpoint is missing: {checkpoint}')
        if stage in ('smoke', 'evaluate') or headless_play:
            steps = s.get('play_steps', 500) if headless_play else s['smoke_steps']
            seed = (s['seed'] if stage == 'smoke' else
                    s['seed'] + 20000 if headless_play else s['seed'] + 10000)
            command = [str(self.python), str(Path(__file__).with_name('playground_worker.py')),
                    *common, f'--nworld={s["num_envs"]}', f'--steps={steps}',
                    f'--seed={seed}',
                    f'--output={run_dir / (stage + ".json")}',
                    f'--tactical-policy={s.get("tactical_policy","guard")}']
            if (stage == 'evaluate' or (stage == 'play' and not getattr(self, 'zero_actions', False))) and checkpoint is not None:
                command.append(f'--checkpoint={checkpoint}')
            if stage == 'play' and zero_actions:
                command.append('--zero_actions')
            return command
        command = [str(self.python), str(PROJECT_ROOT / 'src/shadowbox/playground/launch.py'), stage, *common,
                   f'--wp_kernel_cache_dir={PROJECT_ROOT / "tmp/playground/warp-cache"}']
        if stage == 'train':
            command += [f'--seed={s["seed"]}', f'--logdir={run_dir / "checkpoints"}',
                        '--overrides=' + json.dumps({'nworld': s['num_envs'],
                                                    'max_iterations': s['iterations']})]
        else:
            command += [f'--nworld={s["num_envs"]}', '--gfx=web']
            if zero_actions:
                command += ['--zero_actions']
            else:
                command += [f'--resume_path={checkpoint}']
        return command

    def finalize(self, stage, run_dir):
        """Render saved headless physics states to an animated preview GIF."""
        if (not self.gif_enabled or
                (stage not in ('smoke', 'evaluate') and not (stage == 'play' and not getattr(self, 'viewer', False)))):
            return {}
        trace = run_dir / 'transitions.npz'
        model = run_dir / 'model.mjb'
        if not trace.is_file() or not model.is_file():
            return {}
        steps = self.settings.get('play_steps', 500) if stage == 'play' else self.settings['smoke_steps']
        stride = max(1, min(50, (steps + 199) // 200))
        frames = max(2, min(200, (steps + stride - 1) // stride))
        self.gif_dir.mkdir(parents=True, exist_ok=True)
        gif = self.gif_dir / f'{run_dir.name}-{stage}.gif'
        command = [str(self.python), str(PROJECT_ROOT / 'src/shadowbox/playground/replay.py'),
                   str(run_dir), f'--frames={frames}', f'--stride={stride}',
                   '--width=640', '--height=480', f'--output={gif}']
        env = os.environ.copy()
        env.pop('PYTHONPATH', None)
        env['PYTHONNOUSERSITE'] = '1'
        subprocess.run(command, cwd=PROJECT_ROOT, env=env, check=True,
                       timeout=min(300, self.timeout_seconds))
        return {'visualization_gif': str(gif.relative_to(PROJECT_ROOT))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('doctor', 'smoke', 'train', 'play', 'evaluate'))
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--zero-actions', action='store_true', help='inspect zero task actions in play; no learned tactical actor')
    parser.add_argument('--viewer', action='store_true', default=None,
                        help='open the NVIDIA-backed browser viewer for play')
    args = parser.parse_args()
    try:
        backend = PlaygroundBackend(args.config)
        if args.zero_actions and args.stage != 'play':
            raise ValueError('--zero-actions applies only to play')
        if args.viewer and args.stage != 'play':
            raise ValueError('--viewer applies only to play')
        backend.zero_actions = args.zero_actions
        backend.viewer = backend.settings.get('viewer', False) if args.viewer is None else args.viewer
        if args.stage == 'doctor':
            state = backend.probe()
            print(json.dumps(state, indent=2))
            return 0 if state['ready'] else 1
        return run_backend(backend, args.stage, args.dry_run)
    except (ValueError, OSError, yaml.YAMLError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    sys.exit(main())
