"""Isolated Isaac Lab 3.0 G1 environment, selected independently of MuJoCo."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import threading
import uuid

import yaml

from ..config import PROJECT_ROOT, artifact_path

DEFAULT_CONFIG = PROJECT_ROOT / 'configs/isaaclab.yaml'


def load_isaaclab_config(path=DEFAULT_CONFIG):
    """Load a standalone config; never merge it into MuJoCo's config."""
    try:
        value = yaml.safe_load(Path(path).read_text())
    except (OSError, yaml.YAMLError) as error:
        raise ValueError(f'Could not read Isaac Lab config: {path}') from error
    if not isinstance(value, dict) or set(value) != {'backend', 'isaaclab'}:
        raise ValueError('Isaac Lab config needs only backend and isaaclab sections')
    if value['backend'] != 'isaaclab' or not isinstance(value['isaaclab'], dict):
        raise ValueError('Isaac Lab config must select backend: isaaclab')
    settings = value['isaaclab'].copy()
    required = {'python', 'executable', 'version', 'task', 'physics', 'num_envs',
                'steps', 'visualizer', 'timeout_seconds', 'output_dir'}
    if set(settings) != required:
        raise ValueError('Isaac Lab config has missing or unknown settings: '
                         f'{sorted(set(settings) ^ required)}')
    for name in ('python', 'executable', 'version', 'task', 'physics', 'visualizer', 'output_dir'):
        if not isinstance(settings[name], str) or not settings[name].strip():
            raise ValueError(f'isaaclab.{name} must be a nonempty string')
    if settings['version'] != '3.0.0':
        raise ValueError('This integration targets Isaac Lab 3.0.0')
    if not re.fullmatch(r'[A-Za-z][A-Za-z0-9_-]*', settings['task']):
        raise ValueError('isaaclab.task must be a registered task name')
    if settings['physics'] != 'newton_mjwarp':
        raise ValueError('Only the Newton/MuJoCo-Warp physics preset is wired')
    if settings['visualizer'] not in ('none', 'newton_gl'):
        raise ValueError('isaaclab.visualizer must be none or newton_gl')
    for name in ('num_envs', 'steps', 'timeout_seconds'):
        if isinstance(settings[name], bool) or not isinstance(settings[name], int) or settings[name] < 1:
            raise ValueError(f'isaaclab.{name} must be a positive integer')
    for name in ('python', 'executable'):
        # Preserve the venv symlink: resolving it would bypass pyvenv.cfg.
        settings[name] = Path(os.path.abspath(artifact_path(Path(settings[name]).expanduser())))
    if settings['python'].parent != settings['executable'].parent:
        raise ValueError('Isaac Lab Python and CLI must belong to the same environment')
    settings['output_dir'] = artifact_path(Path(settings['output_dir']).expanduser()).resolve()
    try:
        settings['output_dir'].relative_to((PROJECT_ROOT / 'tmp').resolve())
    except ValueError as error:
        raise ValueError('isaaclab.output_dir must be inside ShadowBox tmp/') from error
    return settings


def detect_backend(path):
    if path is None:
        return None
    try:
        value = yaml.safe_load(Path(path).read_text())
    except (OSError, yaml.YAMLError) as error:
        raise ValueError(f'Could not read configuration {path}') from error
    if isinstance(value, dict) and 'backend' in value:
        if value['backend'] not in ('mujoco', 'isaaclab'):
            raise ValueError('backend must be mujoco or isaaclab')
        return value['backend']
    return None


def build_command(settings, headless=False):
    return [str(settings['executable']), 'zero_agent', '--task', settings['task'],
            '--num_envs', str(settings['num_envs']), '--max_steps', str(settings['steps']),
            '--visualizer', 'none' if headless else settings['visualizer'],
            f"physics={settings['physics']}"]


def probe(settings):
    state = {'python': str(settings['python']), 'executable': str(settings['executable']),
             'python_exists': settings['python'].is_file(),
             'executable_exists': settings['executable'].is_file(),
             'expected_version': settings['version'], 'ready': False}
    if not state['python_exists'] or not state['executable_exists']:
        state['error'] = 'Separate Isaac Lab environment is not installed; see docs/isaaclab.md'
        return state
    code = ("import importlib.metadata,json,sys,torch; "
            "print('SHADOWBOX_ISAAC_PROBE='+json.dumps(dict("
            "python_version=list(sys.version_info[:2]),prefix=sys.prefix,base_prefix=sys.base_prefix,"
            "isaaclab=importlib.metadata.version('isaaclab'),"
            "torch_cuda=torch.version.cuda,cuda_available=torch.cuda.is_available())))")
    try:
        result = subprocess.run([str(settings['python']), '-c', code], capture_output=True,
                                text=True, timeout=45, check=False)
        marker = 'SHADOWBOX_ISAAC_PROBE='
        lines = [line[len(marker):] for line in result.stdout.splitlines() if line.startswith(marker)]
        if result.returncode or not lines:
            state['error'] = (result.stderr or result.stdout or 'Environment probe failed').strip()[-1500:]
            return state
        state.update(json.loads(lines[-1]))
        isolated = (state['prefix'] != state['base_prefix']
                    and Path(state['prefix']).resolve() != (PROJECT_ROOT / '.venv').resolve())
        state['ready'] = (state['python_version'] == [3, 12] and isolated
                          and state['isaaclab'] == settings['version']
                          and state['torch_cuda'] == '13.0' and state['cuda_available'])
        if not state['ready']:
            state['error'] = ('Requires an isolated Python 3.12 Isaac Lab 3.0.0 environment '
                              'with CUDA 13 Torch and a visible GPU')
        return state
    except (OSError, subprocess.TimeoutExpired, ValueError) as error:
        state['error'] = str(error)
        return state


def run_backend(settings, dry_run=False, headless=False):
    command = build_command(settings, headless)
    if dry_run:
        print('Isaac Lab G1 physics smoke task (not the ShadowBox boxing match):')
        print(shlex.join(command))
        print(f"Run logs will be saved below {settings['output_dir']}")
        return 0
    state = probe(settings)
    if not state['ready']:
        raise ValueError(state['error'])
    run_dir = settings['output_dir'] / ('run-' + uuid.uuid4().hex[:12])
    run_dir.mkdir(parents=True, exist_ok=False)
    metadata = {'backend': 'isaaclab', 'scope': 'upstream G1 zero-action physics smoke; not boxing',
                'started_utc': datetime.now(timezone.utc).isoformat(), 'command': command,
                'task': settings['task'], 'physics': settings['physics'],
                'version': settings['version'], 'status': 'running'}
    manifest = run_dir / 'run.json'
    manifest.write_text(json.dumps(metadata, indent=2) + '\n')
    env = os.environ.copy()
    env.pop('PYTHONPATH', None)
    env.update(PYTHONUNBUFFERED='1', PYTHONNOUSERSITE='1')
    print(f'Isaac Lab G1 physics smoke; logs: {run_dir}', flush=True)
    timed_out = threading.Event()
    process = None
    timer = None
    try:
        with (run_dir / 'simulation.log').open('w') as log:
            process = subprocess.Popen(command, cwd=run_dir, env=env, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, bufsize=1)
            def timeout():
                try:
                    process.kill()
                    timed_out.set()
                except ProcessLookupError:
                    pass
            timer = threading.Timer(settings['timeout_seconds'], timeout)
            timer.start()
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line, end='', flush=True)
            code = process.wait()
        metadata.update(status='timeout' if timed_out.is_set() else ('completed' if code == 0 else 'failed'),
                        exit_code=124 if timed_out.is_set() else code)
        return metadata['exit_code']
    except (OSError, KeyboardInterrupt) as error:
        if process and process.poll() is None:
            process.terminate()
            process.wait()
        metadata.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                        error=str(error))
        raise
    finally:
        if timer:
            timer.cancel()
        metadata['finished_utc'] = datetime.now(timezone.utc).isoformat()
        manifest.write_text(json.dumps(metadata, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('doctor', 'run'))
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    try:
        settings = load_isaaclab_config(args.config)
        if args.command == 'doctor':
            state = probe(settings)
            print(json.dumps(state, indent=2))
            return 0 if state['ready'] else 1
        return run_backend(settings, args.dry_run)
    except (ValueError, OSError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    sys.exit(main())
