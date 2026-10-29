"""Run the visible arena or a reproducible headless boxing match."""

import argparse
import json
import logging
import math
import os
from pathlib import Path
import sys
import time

from .config import load_config, configure_logging, artifact_path


def load_policy(path):
    if path is None:
        return None
    value = json.loads(artifact_path(path).read_text())
    return value.get('parameters', value)


def main() -> None:
    parser = argparse.ArgumentParser(description='ShadowBox: two humanoids in a MuJoCo ring')
    parser.add_argument('--demo', action='store_true', help='visible sparring demo (default)')
    parser.add_argument('--view', action='store_true', help='open the 3D arena (default)')
    parser.add_argument('--joint-demo', action='store_true', help='exercise G1 waist, wrists, arms, and legs')
    parser.add_argument('--loop', action='store_true', help='repeat matches or joint exercises until closed')
    parser.add_argument('--headless', action='store_true', help='run without a graphics window')
    parser.add_argument('--config', type=Path, help='partial YAML override of configs/simulation.yaml')
    parser.add_argument('--backend', choices=('mujoco', 'isaaclab'),
                        help='select a simulation environment (default: MuJoCo)')
    parser.add_argument('--backend-dry-run', action='store_true',
                        help='preview the Isaac Lab command without launching it')
    parser.add_argument('--seconds', type=float, help='limit simulated time and close the window')
    parser.add_argument('--round-seconds', type=float)
    parser.add_argument('--rounds', type=int)
    parser.add_argument('--seed', type=int, default=0)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--defense-only', action='store_true', default=None, help='B never punches')
    modes.add_argument('--sparring', dest='defense_only', action='store_false', help='enable both players to attack')
    rendering = parser.add_mutually_exclusive_group()
    rendering.add_argument('--software', dest='software', action='store_true', default=None)
    rendering.add_argument('--hardware', dest='software', action='store_false')
    parser.add_argument('--capture', type=Path, help='save the final visible framebuffer as PNG')
    parser.add_argument('--output', type=Path, help='override the configured match result path')
    parser.add_argument('--policy-a', type=Path)
    parser.add_argument('--policy-b', type=Path)
    parser.add_argument('--learned-a', type=Path, help='load a validated tabular policy')
    parser.add_argument('--learned-b', type=Path, help='load a validated tabular policy')
    parser.add_argument('--supermimic', action='store_true', help='enable SuperMimic motion imitation')
    args = parser.parse_args()
    try:
        from .integrations.isaaclab import detect_backend
        configured_backend = detect_backend(args.config)
        if args.backend and configured_backend and args.backend != configured_backend:
            raise ValueError('CLI backend conflicts with the selected config')
        backend = args.backend or configured_backend or 'mujoco'
        if backend == 'isaaclab':
            from .integrations.isaaclab import DEFAULT_CONFIG, load_isaaclab_config, run_backend
            if args.config and configured_backend != 'isaaclab':
                raise ValueError('Isaac Lab requires its separate configs/isaaclab.yaml')
            unsupported = (args.demo or args.view or args.joint_demo or args.loop or args.seconds is not None
                           or args.round_seconds is not None or args.rounds is not None
                           or args.defense_only is not None or args.software is not None or args.capture
                           or args.output or args.policy_a or args.policy_b or args.learned_a or args.learned_b
                           or args.supermimic or args.seed != 0)
            if unsupported:
                raise ValueError('Isaac Lab currently runs its configured G1 smoke task; '
                                 'boxing, match policies and timing flags are not yet supported')
            return run_backend(load_isaaclab_config(args.config or DEFAULT_CONFIG),
                               dry_run=args.backend_dry_run, headless=args.headless)
        if args.backend_dry_run:
            raise ValueError('--backend-dry-run is only available for Isaac Lab')
        config = load_config(args.config)
        if args.supermimic:
            config['supermimic']['enabled'] = True
        for name in ('round_seconds', 'rounds', 'defense_only'):
            if getattr(args, name) is not None:
                config['match'][name] = getattr(args, name)
        if not math.isfinite(config['match']['round_seconds']) or config['match']['round_seconds'] <= 0 or config['match']['rounds'] < 1:
            raise ValueError('Round durations and count must be positive')
        if args.seconds is not None and (not math.isfinite(args.seconds) or args.seconds <= 0):
            raise ValueError('--seconds must be positive')
        if (args.policy_a and args.learned_a) or (args.policy_b and args.learned_b):
            raise ValueError('Choose either parameter overrides or a learned policy for each player')
        if args.joint_demo and config['simulation']['model'] != 'unitree_g1':
            raise ValueError('--joint-demo requires --config configs/unitree.yaml')
        if args.capture and args.headless:
            raise ValueError('--capture requires the visible viewer')
        logfile = configure_logging(config)
    except (ValueError, OSError) as error:
        parser.error(str(error))

    # Select libraries before importing MuJoCo/GLFW; software rendering works on this host.
    software = config['viewer']['software'] if args.software is None else args.software
    if sys.platform.startswith('linux'):
        os.environ.setdefault('PYGLFW_LIBRARY_VARIANT', 'x11')
        if software:
            os.environ.update(LIBGL_ALWAYS_SOFTWARE='1', GALLIUM_DRIVER='llvmpipe',
                              __GLX_VENDOR_LIBRARY_NAME='mesa')
            mesa = '/usr/share/glvnd/egl_vendor.d/50_mesa.json'
            if Path(mesa).exists():
                os.environ['__EGL_VENDOR_LIBRARY_FILENAMES'] = mesa

    from .boxing import Match
    try:
        match = Match(seed=args.seed, defense_only=config['match']['defense_only'], config=config,
                      parameters_a=load_policy(args.policy_a), parameters_b=load_policy(args.policy_b))
    except (ValueError, OSError) as error:
        parser.error(str(error))
    def install_learned():
        from .evaluation.qlearning import load_learned_policy
        for side, path in (('a', args.learned_a), ('b', args.learned_b)):
            if path is not None:
                policy = load_learned_policy(path, seed=args.seed + (1009 if side == 'b' else 0), config=config)
                match.policies[side] = policy
                match.parameters[side] = policy.parameters

    try:
        install_learned()
    except (ValueError, OSError) as error:
        parser.error(str(error))
    logging.info('Match started seed=%s config=%s', args.seed, config)
    demo_frames = None
    if args.joint_demo:
        from .motion.articulation import joint_demo
        demo_frames = joint_demo(match.arena)

    simulated_total = 0.0

    def advance():
        nonlocal simulated_total
        before = match.arena.time
        if demo_frames is None:
            match.step()
        elif not match.supervisor.stopped:
            try:
                next(demo_frames)
            except StopIteration:
                match.finished = True
        simulated_total += match.arena.time - before

    limit = args.seconds if args.seconds is not None else float('inf')
    if args.headless:
        while not match.finished and not match.supervisor.stopped and simulated_total < limit:
            advance()
    else:
        from .ui.viewer import Viewer
        from .simulation.timing import SimulationClock
        clock = SimulationClock(match.arena.model.opt.timestep, config['viewer']['max_catchup_seconds'])
        try:
            with Viewer(match.arena) as viewer:
                previous_wall = time.monotonic()
                finished_wall = None
                match_number = 1
                while viewer.running and simulated_total < limit:
                    start = time.monotonic()
                    elapsed, previous_wall = start - previous_wall, start
                    repeat = args.loop or args.demo
                    if match.finished and finished_wall is None:
                        finished_wall = start
                        history = artifact_path('tmp/results/matches') / logfile.stem
                        history.mkdir(parents=True, exist_ok=True)
                        (history / f'match-{match_number:04d}.json').write_text(json.dumps(match.result(), indent=2))
                    automatic_reset = (finished_wall is not None and repeat
                                       and start-finished_wall >= config['viewer']['repeat_delay_seconds'])
                    if viewer.reset_requested or automatic_reset:
                        if automatic_reset:
                            match.seed += 1
                            match_number += 1
                        match.reset()
                        install_learned()
                        if args.joint_demo:
                            demo_frames = joint_demo(match.arena)
                        viewer.reset_requested = False
                        finished_wall = None
                        elapsed = 0
                    if viewer.stopped and not match.supervisor.stopped:
                        match.stop()
                    active = not viewer.paused and not match.supervisor.stopped and not match.finished
                    for _ in range(clock.steps(elapsed, active)):
                        if simulated_total < limit:
                            advance()
                    lines = match.status()
                    if args.joint_demo:
                        lines = ['ShadowBox | Unitree G1 full-joint exercise',
                                 '29 native joints per robot: legs, waist, shoulders, elbows, wrists',
                                 f'Simulation {match.arena.time:.2f}s | stabilized roots',
                                 'Joint articulation demonstration; free-base balance is deferred']
                    if match.finished:
                        lines.append('COMPLETE - restarting shortly' if repeat else 'COMPLETE - R restarts')
                    viewer.render(lines)
                    if match.finished and args.seconds is not None and not repeat:
                        break
                    time.sleep(max(0, 1/config['viewer']['frames_per_second'] - (time.monotonic()-start)))
                if args.capture:
                    viewer.capture(artifact_path(args.capture))
        except RuntimeError as error:
            parser.exit(1, f'{error}\nUse --headless when no desktop display is available.\n')

    result = match.result()
    result['mode'] = 'joint_demo' if args.joint_demo else 'boxing'
    result['model_profile'] = config['simulation']['model']
    result['actuators'] = match.arena.model.nu
    result["log_file"] = str(logfile)
    result['policy_artifacts'] = {side: str(artifact_path(path)) if path else None
                                for side, path in (('a', args.learned_a or args.policy_a),
                                                   ('b', args.learned_b or args.policy_b))}
    output = artifact_path(args.output or config['logging']['result'])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n')
    logging.info('Match result %s', output)
    print(f"Simulation: {result['time']:.2f}s | touches A {result['stats']['a']['touches']} "
          f": {result['stats']['b']['touches']} B | scores {result['scores']} | {output}")


if __name__ == '__main__':
    sys.exit(main())
