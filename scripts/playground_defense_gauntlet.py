"""Bounded distance/policy sweep; retain each run's source and physics evidence.

This is a diagnostic comparison, not policy promotion. Default seeds vary only
lane translation; optional independent yaw/lateral ranges are recorded explicitly.
"""
import argparse
import copy
from datetime import datetime, timezone
import json
import subprocess
import hashlib
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from shadowbox.integrations.backend import run_backend
from shadowbox.integrations.playground import PlaygroundBackend, PROJECT_ROOT
import yaml


def runtime_sources(backend):
    # Saved-model rendering and interactive launch do not execute in this worker.
    sources={name: digest for name,digest in backend.provenance()['local_sources'].items()
             if Path(name).name not in ('replay.py','launch.py')}
    for name in ('src/shadowbox/integrations/playground_worker.py','src/shadowbox/motion/primitives.py'):
        sources[name]=hashlib.sha256((PROJECT_ROOT/name).read_bytes()).hexdigest()
    return sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('configs/playground-motion-defense.yaml'))
    parser.add_argument('--distances', type=float, nargs='+', default=[.55, .60, .70])
    parser.add_argument('--steps', type=int, default=500)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    if not 1 <= args.steps <= 3000 or not 0 <= args.seed:
        parser.error('Require 1..3000 steps and nonnegative seed')
    if not args.distances or len(args.distances)>10 or any(not .5<=d<=2 for d in args.distances):
        parser.error('Require 1..10 distances in [.5, 2]')
    template = yaml.safe_load(args.config.read_text())
    if template['playground']['task']!='ShadowBoxG1Motion':
        parser.error('Gauntlet requires ShadowBoxG1Motion')
    root = PROJECT_ROOT/'tmp/playground/gauntlets'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    root.mkdir(parents=True)
    options=template['playground'].get('task_options',{})
    source=Path(__file__).read_bytes()
    (root/'gauntlet-source.py').write_bytes(source)
    report = {'scope': 'seeded distance/reset diagnostic; no automatic promotion',
              'runner_sha256': hashlib.sha256(source).hexdigest(),
              'reset_variation': {key: options.get(key,0.) for key in ('reset_yaw_radians','reset_lateral_m')},
              'cases': []}
    initial_sources=runtime_sources(PlaygroundBackend(args.config))
    report['local_sources']=initial_sources
    for distance in args.distances:
        for policy in ('guard','reactive'):
            if runtime_sources(PlaygroundBackend(args.config))!=initial_sources:
                raise RuntimeError('Task sources changed during comparison; retain prior runs as diagnostics and restart the sweep')
            config = copy.deepcopy(template)
            settings = config['playground']
            settings.update(seed=args.seed, smoke_steps=args.steps, tactical_policy=policy)
            settings['task_options'].update(start_distance=distance, opponent_enabled=True)
            if settings['task_options'].get('defender_guard_width', .1)!=.1:
                parser.error('Comparison requires normal defender guard width')
            case_dir = root/f'{distance:.3f}-{policy}'
            case_dir.mkdir()
            settings['output_dir'] = str(case_dir/'runs')
            path = case_dir/'config.yaml'
            path.write_text(yaml.safe_dump(config, sort_keys=False))
            backend=PlaygroundBackend(path)
            code = run_backend(backend, 'evaluate')
            runs = list((case_dir/'runs').glob('run-*'))
            case = dict(distance_m=distance, policy=policy, exit_code=code)
            if len(runs)==1:
                case['run_dir'] = str(runs[0].relative_to(PROJECT_ROOT))
                result = runs[0]/'evaluate.json'
                if result.is_file():
                    evidence = json.loads(result.read_text())
                    with (case_dir/'posture.log').open('w') as log:
                        measured=subprocess.run([str(backend.python), str(PROJECT_ROOT/'scripts/playground_rollout_metrics.py'),
                            str(runs[0])],stdout=log,stderr=subprocess.STDOUT,timeout=120)
                    if measured.returncode:
                        raise RuntimeError(f'Posture measurement failed: {case_dir / "posture.log"}')
                    case['posture_metrics']=json.loads((runs[0]/'posture-metrics.json').read_text())
                    for key in ('completed_episodes','contact_episode_counts_by_channel_and_glove',
                                'contact_channel_order','seconds','steps'):
                        case[key] = evidence[key]
            case['runtime_sources_consistent']=runtime_sources(backend)==initial_sources
            report['cases'].append(case)
            (root/'summary.json').write_text(json.dumps(report, indent=2)+'\n')
            print(json.dumps(case), flush=True)
            if not case['runtime_sources_consistent']:
                raise RuntimeError('Runtime sources changed during rollout; retain diagnostics and restart the comparison')
            if code:
                raise SystemExit(code)
    print(root/'summary.json')


if __name__=='__main__':
    main()
