"""Pinned MorphMimic tooling, isolated from ShadowBox's simulation environment."""

import argparse
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys

import yaml

from ..config import PROJECT_ROOT, artifact_path

DEFAULT_CONFIG = PROJECT_ROOT / 'configs/morphmimic.yaml'
ROBOTS = ('g1', 'x2', 't2')
STAGES = {
    'convert': 'morphmimic.scripts.convert_seed',
    'subsets': 'morphmimic.scripts.build_subsets',
    'train': 'morphmimic.scripts.train',
    'export': 'deploy.export_onnx',
    'assets': 'deploy.sim2sim.export_assets',
}
PATH_KEYS = ('repository', 'training_python', 'runtime_python', 'deployment',
             'assets', 'data', 'output')


def load_settings(path=DEFAULT_CONFIG):
    settings = yaml.safe_load(Path(path).read_text())
    required = {*PATH_KEYS, 'revision', 'robot'}
    if (not isinstance(settings, dict) or required - settings.keys()
            or any(not isinstance(settings[k], str) or not settings[k].strip()
                   for k in required)):
        raise ValueError('MorphMimic config needs nonempty path, revision and robot settings')
    if not re.fullmatch(r'[0-9a-f]{40}', settings['revision']):
        raise ValueError('MorphMimic revision must be a full Git commit')
    if settings['robot'] not in ROBOTS:
        raise ValueError(f'MorphMimic robot must be one of {ROBOTS}')
    for key in PATH_KEYS:
        settings[key] = artifact_path(Path(settings[key]).expanduser()).resolve()
    return settings


def checkout_revision(settings):
    repo = settings['repository']
    if not (repo / 'deploy/sim2sim/obs.py').is_file():
        return None
    result = subprocess.run(['git', '-C', str(repo), 'rev-parse', 'HEAD'],
                            capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def require_checkout(settings):
    if checkout_revision(settings) != settings['revision']:
        raise ValueError('Initialize external/MorphMimic at the configured pinned revision')
    result = subprocess.run(['git', '-C', str(settings['repository']), 'diff',
                             '--quiet', 'HEAD', '--', 'deploy/sim2sim/obs.py'], check=False)
    if result.returncode:
        raise ValueError('MorphMimic observation contract has local changes; restore or repin it')


def build_stage(settings, stage, extra):
    if extra and extra[0] == '--':
        extra = extra[1:]
    replacements = {key: str(value) for key, value in settings.items()}
    try:
        extra = [value.format_map(replacements) for value in extra]
    except (KeyError, ValueError) as error:
        raise ValueError(f'Invalid MorphMimic config placeholder: {error}') from error
    robot = settings['robot']
    for index, argument in enumerate(extra):
        if argument == '--robot' and index + 1 < len(extra):
            robot = extra[index + 1]
        elif argument.startswith('--robot='):
            robot = argument.split('=', 1)[1]
    if robot not in ROBOTS:
        raise ValueError(f'Unsupported MorphMimic robot: {robot}')
    # Upstream defaults write logs/assets into its source tree. Redirect those.
    defaults = {
        'train': ('--log-dir', str(settings['output'] / 'training')),
        'assets': ('--out', str(settings['assets'])),
        'convert': ('--out-dir', str(settings['data'] / robot)),
        'subsets': ('--out-dir', str(settings['data'] / settings['robot'] / 'subsets')),
    }
    if stage in defaults:
        flag, value = defaults[stage]
        if not any(arg == flag or arg.startswith(flag + '=') for arg in extra):
            extra += [flag, value]
    return [str(settings['training_python']), '-m', STAGES[stage], *extra]


def build_simulation(settings, robot, clip, interface, delay_ms, viewer, summary):
    if not clip or Path(clip).name != clip or clip in ('.', '..'):
        raise ValueError('clip must be an exported clip name, without a directory')
    if not 0 <= delay_ms <= 200:
        raise ValueError('delay-ms must be in [0, 200]')
    return [str(settings['runtime_python']), '-m', 'deploy.run',
            '--deploy', str(settings['deployment']), '--assets', str(settings['assets']),
            '--robot', robot, '--clip', clip, '--interface', interface,
            '--delay-ms', str(delay_ms), '--viewer', viewer, '--summary', str(summary)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('doctor', help='inspect source and optional environment paths')
    commands.add_parser('robots', help='inspect official canonical joint/IMU profiles')
    run = commands.add_parser('run', help='launch an upstream data/training/export stage')
    run.add_argument('--dry-run', action='store_true', help='place before stage')
    run.add_argument('stage', choices=STAGES)
    run.add_argument('stage_args', nargs=argparse.REMAINDER)
    sim = commands.add_parser('simulate', help='run official CPU MuJoCo sim2sim')
    sim.add_argument('--robot', choices=ROBOTS)
    sim.add_argument('--clip', required=True)
    sim.add_argument('--interface', choices=('motion', 'keypoint'), default='motion')
    sim.add_argument('--delay-ms', type=float, default=0)
    sim.add_argument('--viewer', choices=('none', 'native'), default='none')
    sim.add_argument('--output', type=Path, help='summary JSON (default under tmp/morphmimic)')
    sim.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    try:
        settings = load_settings(args.config)
        if args.command == 'doctor':
            print(json.dumps({
                'revision': checkout_revision(settings), 'pinned_revision': settings['revision'],
                'revision_matches': checkout_revision(settings) == settings['revision'],
                'training_python_exists': settings['training_python'].is_file(),
                'runtime_python_exists': settings['runtime_python'].is_file(),
                'deployment_exists': (settings['deployment'] / 'policy.onnx').is_file(),
                'source_only': not (settings['deployment'] / 'policy.onnx').is_file(),
                'hardware_execution': False,
            }, indent=2))
            require_checkout(settings)
            return 0
        require_checkout(settings)
        if args.command == 'robots':
            from ..motion.morphology import MorphMimicProfile
            print(json.dumps([MorphMimicProfile(robot, settings).manifest()
                              for robot in ROBOTS], indent=2))
            return 0
        if args.command == 'run':
            command = build_stage(settings, args.stage, args.stage_args)
        else:
            robot = args.robot or settings['robot']
            summary = artifact_path(args.output).resolve() if args.output else (
                settings['output'] / 'results' / f'{robot}-{args.clip}-{args.interface}.json')
            command = build_simulation(settings, robot, args.clip, args.interface,
                                       args.delay_ms, args.viewer, summary)
            if not args.dry_run:
                if summary.exists():
                    raise ValueError(f'Result exists; choose a new --output: {summary}')
                for file in (settings['deployment'] / 'policy.onnx',
                             settings['deployment'] / f'{robot}.json',
                             settings['assets'] / f'{robot}.xml',
                             settings['assets'] / 'clips' / robot / f'{args.clip}.npz'):
                    if not file.is_file():
                        raise ValueError(f'Missing exported simulation artifact: {file}')
                summary.parent.mkdir(parents=True, exist_ok=True)
        if args.dry_run:
            print(shlex.join(command))
            return 0
        env = os.environ.copy()
        env['PYTHONDONTWRITEBYTECODE'] = '1'
        return subprocess.run(command, cwd=settings['repository'], env=env, check=False).returncode
    except (OSError, ValueError, yaml.YAMLError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    sys.exit(main())
