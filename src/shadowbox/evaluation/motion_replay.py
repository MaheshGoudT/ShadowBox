"""Replay a motion reference through G1 servos and measure its physical fit."""

import argparse
import json
import math

import numpy as np

from ..config import artifact_path, load_config
from ..motion.clip import load_motion_clip
from ..motion.stance import BoxingStance
from ..simulation import Arena


def replay_motion(path, config=None, robot='a', transitions=None,
                  morphmimic=False, morphmimic_config=None, morphmimic_deployment=None):
    config = config or load_config()
    if config['simulation']['model'] != 'unitree_g1' or robot not in ('a', 'b'):
        raise ValueError('Motion replay requires Unitree G1 robot a or b')
    arena = Arena(config=config)
    profile = None
    if morphmimic:
        if not transitions:
            raise ValueError('MorphMimic diagnostics require a transition output')
        from ..integrations.morphmimic import load_settings, DEFAULT_CONFIG
        from ..motion.morphology import MorphMimicProfile
        settings = load_settings(morphmimic_config or DEFAULT_CONFIG)
        profile = MorphMimicProfile(settings['robot'], settings, morphmimic_deployment)
    clip = load_motion_clip(path, arena)
    stance = BoxingStance(arena)
    start = arena.pose(robot).copy()
    forward = np.array([math.cos(start[2]), math.sin(start[2])])
    left = np.array([-forward[1], forward[0]])
    previous_feet = stance.foot_positions(robot)
    previous_planted = set()
    squared_error = samples = slide = height_error = foot_samples = 0.0
    dt = arena.model.opt.timestep
    records = {key: [] for key in ('qpos', 'qvel', 'ctrl', 'next_qpos', 'next_qvel')} if transitions else None
    canonical = {key: [] for key in ('canonical_q', 'canonical_dq', 'canonical_action',
                                     'proprioception', 'next_proprioception')} if profile else None

    for index in range(math.ceil(clip.duration / dt) + 1):
        # The observation sees the previously held target, then we choose this
        # interval's command. This preserves the last-action timing contract.
        before_canonical = profile.capture(arena, robot) if profile else None
        joints, root, planted = clip.sample(min(index * dt, clip.duration))
        xy = start[:2] + root[0] * forward + root[1] * left
        if np.max(np.abs(xy)) > config['simulation']['arena_half_size'] - config['safety']['boundary_margin']:
            raise ValueError('Motion clip root leaves the ring')
        target = arena.targets(robot)
        target[:3] = [*xy, start[2] + root[2]]
        arena.set_targets(robot, target)
        arena.set_full_targets(robot, joints)
        if records is not None:
            records['qpos'].append(arena.data.qpos.copy())
            records['qvel'].append(arena.data.qvel.copy())
            records['ctrl'].append(arena.data.ctrl.copy())
        if profile:
            canonical['canonical_q'].append(before_canonical['canonical_q'])
            canonical['canonical_dq'].append(before_canonical['canonical_dq'])
            canonical['proprioception'].append(before_canonical['proprioception'])
            canonical['canonical_action'].append(profile.capture(arena, robot)['target_action'])
        arena.step()
        if profile:
            canonical['next_proprioception'].append(profile.capture(arena, robot)['proprioception'])
        if records is not None:
            records['next_qpos'].append(arena.data.qpos.copy())
            records['next_qvel'].append(arena.data.qvel.copy())
        actual = arena.full_pose(robot)
        squared_error += sum((actual[name] - angle) ** 2 for name, angle in joints.items())
        samples += len(joints)
        feet = stance.foot_positions(robot)
        for side in planted:
            foot = feet[side]
            height_error += abs(foot[2] - config['stance'].get('sole_height', .005))
            foot_samples += 1
            if side in previous_planted:
                slide += float(np.linalg.norm(foot[:2] - previous_feet[side][:2]))
        previous_feet, previous_planted = feet, set(planted)

    if records is not None:
        output = artifact_path(transitions)
        if output.suffix != '.npz':
            raise ValueError('Transition output must end in .npz')
        output.parent.mkdir(parents=True, exist_ok=True)
        arrays = {key: np.asarray(value) for key, value in records.items()}
        if profile:
            arrays.update({key: np.asarray(value) for key, value in canonical.items()})
            arrays['morphmimic_metadata'] = json.dumps({
                **profile.manifest(), 'sampling_dt': dt,
                'scope': 'ShadowBox stabilized-root diagnostic trace; not an upstream training dataset',
                'last_action': 'reconstructed from previously held native joint targets',
            })
            if profile.morphology is not None:
                arrays['morphology_descriptor'] = profile.morphology
        np.savez_compressed(output, **arrays, timestep=dt, source=json.dumps(clip.source))
    return dict(source=clip.source, robot=robot, seconds=clip.duration,
                frames=len(clip.frames), joint_rmse_rad=math.sqrt(squared_error / samples),
                planted_foot_slide_m=slide,
                mean_planted_foot_height_error_m=float(height_error / max(foot_samples, 1)),
                planted_foot_samples=int(foot_samples),
                transitions=str(artifact_path(transitions)) if transitions else None,
                morphmimic=profile.manifest() if profile else None)


def main():
    parser = argparse.ArgumentParser(description='Replay and measure a G1 motion clip')
    parser.add_argument('clip')
    parser.add_argument('--config')
    parser.add_argument('--robot', choices=('a', 'b'), default='a')
    parser.add_argument('--output', default='tmp/results/motion-replay.json')
    parser.add_argument('--transitions', help='optional NPZ state/action transitions for model learning')
    parser.add_argument('--morphmimic', action='store_true', help='include canonical measured-state diagnostics')
    parser.add_argument('--morphmimic-config', help='robot and pinned upstream configuration')
    parser.add_argument('--morphmimic-deployment', help='optional exported bundle with fitted morphology inputs')
    args = parser.parse_args()
    result = replay_motion(args.clip, load_config(args.config), args.robot, args.transitions,
                           args.morphmimic, args.morphmimic_config, args.morphmimic_deployment)
    output = artifact_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(f"Joint RMSE {result['joint_rmse_rad']:.3f} rad; "
          f"planted foot slide {result['planted_foot_slide_m']:.3f} m")
    print(output)


if __name__ == '__main__':
    main()
