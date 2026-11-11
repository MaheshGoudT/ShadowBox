"""Measure scripted legal head/body contacts across ranges, timing and sides.

This is a fixed-motion positive-control experiment, not policy evaluation or
promotion. Each case exposes both guards and records contact-classifier events.
Run it from the isolated Playground CUDA environment; physics remains CPU.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))

from shadowbox.integrations.playground import PlaygroundBackend, PROJECT_ROOT

DT=.02
CHANNELS=('incoming_head','incoming_body','incoming_block','outgoing_head','outgoing_body')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_case(entry, options, *, seed, distance, attacker, target, delay, steps,
             head_z, body_z, head_overshoot, body_overshoot, approach_secs,
             reset_yaw_radians, reset_lateral_m, out_dir):
    import mujoco
    import torch
    from mujoco_playground._src.rl.rsl_rl.vecenv_wrapper import RslRlVecEnvWrapper
    from mujoco_playground._src.rl import rng

    if attacker not in ('a','b') or target not in ('head','body'):
        raise ValueError('Invalid attacker or target')
    settings=dict(options)
    settings.update(start_distance=distance,opponent_enabled=False,controlled_side=attacker,
        terminate_opponent=True,defender_guard_width=.45,opponent_guard_width=.45,
        head_target_z=head_z,body_target_z=body_z,
        head_overshoot=head_overshoot,body_overshoot=body_overshoot,
        opponent_attack_target=target,opponent_attack_delay_secs=delay,
        reset_yaw_radians=reset_yaw_radians,reset_lateral_m=reset_lateral_m,
        episode_length_secs=10.)

    case_name=f'{attacker}-{target}-d{distance:.2f}-t{delay:.2f}'
    if approach_secs:
        case_name=f'{attacker}-{target}-d{distance:.2f}-a{approach_secs:.2f}-t{delay:.2f}'
    out_dir.mkdir(parents=True,exist_ok=True)
    rng.seed(seed)
    raw=entry.env_builder(nworld=1,device='cpu',impl='cpp',**settings)
    try:
        env=RslRlVecEnvWrapper(raw,cfg=entry.runner_cfg.vecenv_wrapper_cfg)
        motion=raw.task_layer.get_component_by_name('motion')
        motion_identity=motion.policy.provenance()
        sim=raw.device_layer.sim
        model=sim.mj_model
        mujoco.mj_saveModel(model,str(out_dir/'model.mjb'),None)
        root_addresses={}
        for side in ('a','b'):
            joint=next(j_id for j_id in range(model.njnt)
                if model.joint(j_id).name.startswith(f'{side}/')
                and model.jnt_type[j_id]==mujoco.mjtJoint.mjJNT_FREE)
            root_addresses[side]=int(model.jnt_qposadr[joint])
        initial_qpos=sim.data.qpos[0].detach().cpu().numpy().copy()
        start_xy={side:initial_qpos[address:address+2].copy()
                  for side,address in root_addresses.items()}
        approach_end_xy={side:value.copy() for side,value in start_xy.items()}
        channels=np.zeros((steps,len(CHANNELS),2),dtype=np.float32)
        requested=np.zeros(steps,dtype=np.int32)
        heights={'a':[],'b':[]}
        upright={'a':[],'b':[]}
        done_steps=[]
        attack_duration=.7 if target=='head' else 1.1
        primitive=6 if target=='head' else 7
        started=time.monotonic()
        for step in range(steps):
            elapsed=step*DT
            if elapsed<approach_secs:
                action_id=2
            else:
                elapsed-=approach_secs+delay
                phase=elapsed%1.6 if elapsed>=0 else -1.
                action_id=primitive if elapsed>=0 and phase<attack_duration else 0
            requested[step]=action_id
            action=torch.full((1,1),float(action_id),device='cpu')
            _,_,done,_=env.step(action)
            f=raw._latest_features
            is_done=bool(done.any().item())
            if not is_done:
                channels[step]=f['boxing/events'].detach().cpu().numpy()[0]
            state=sim.data.qpos[0].detach().cpu().numpy()
            if step+1==round(approach_secs/DT):
                approach_end_xy={side:state[address:address+2].copy()
                                 for side,address in root_addresses.items()}
            for side in ('a','b'):
                root=f[f'{side}_root/xpos'][0]
                q=f[f'{side}_root/xquat'][0]
                heights[side].append(float(root[2]))
                upright[side].append(float(1-2*(q[1]**2+q[2]**2)))
            if is_done:
                done_steps.append(step)
                break
        elapsed_wall=time.monotonic()-started
        sim_steps=step+1
        valid_channels=channels[:sim_steps]
        sums=valid_channels.sum(axis=(0,2)).astype(int)
        target_channel=(0 if attacker=='b' else 3)+(target=='body')
        target_events=valid_channels[:,target_channel]
        event_frames=np.flatnonzero(target_events.sum(axis=-1)>0).tolist()
        case=dict(name=case_name,seed=seed,distance_m=distance,attacker=attacker,
        target=target,delay_secs=delay,approach_secs=approach_secs,
        reset_yaw_radians=reset_yaw_radians,reset_lateral_m=reset_lateral_m,
        configured_retreat_speeds_m_s={
            'a':float(settings.get('retreat_speed_a',.18)),
            'b':float(settings.get('retreat_speed_b',.18))},
        swap_start_sides=bool(settings.get('swap_start_sides',False)),
        steps=sim_steps,simulated_secs=sim_steps*DT,
            wall_secs=elapsed_wall,opponent_guard_width=.45,defender_guard_width=.45,
            head_target_z=head_z,body_target_z=body_z,
            head_overshoot=head_overshoot,body_overshoot=body_overshoot,
            contact_episode_counts=dict(zip(CHANNELS,sums.tolist())),
            expected_target_events=int(target_events.sum()),
            expected_target_event_frames=event_frames,
            minimum_root_height_m={k:min(v) for k,v in heights.items()},
            minimum_upright_z={k:min(v) for k,v in upright.items()},
            terminated=bool(done_steps),termination_step=done_steps[0] if done_steps else None,
            positive_control=bool(target_events.sum()>0),motion_policy=motion_identity)
        case['approach_displacement_m']=float(np.linalg.norm(
            approach_end_xy[attacker]-start_xy[attacker]))
        np.savez_compressed(out_dir/'trace.npz',boxing_events=channels[:sim_steps],
            requested_primitive=requested[:sim_steps])
        (out_dir/'case.json').write_text(json.dumps(case,indent=2)+'\n')
        return case
    finally:
        raw.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path('configs/playground-motion-exposed.yaml'))
    parser.add_argument('--distances',type=float,nargs='+',default=[.55,.60,.70])
    parser.add_argument('--delays',type=float,nargs='+',default=[0.,.20])
    parser.add_argument('--sides',choices=('a','b'),nargs='+',default=['a','b'])
    parser.add_argument('--targets',choices=('head','body'),nargs='+',default=['head','body'])
    parser.add_argument('--steps',type=int,default=210)
    parser.add_argument('--seed',type=int,default=10020)
    parser.add_argument('--head-target-z',type=float,default=.385)
    parser.add_argument('--body-target-z',type=float,default=.15)
    parser.add_argument('--head-overshoot',type=float,default=.16)
    parser.add_argument('--body-overshoot',type=float,default=.08)
    parser.add_argument('--approach-secs',type=float,default=0.)
    parser.add_argument('--reset-yaw-radians',type=float,default=0.)
    parser.add_argument('--reset-lateral-m',type=float,default=0.)
    parser.add_argument('--swap-start-sides',action='store_true')
    args=parser.parse_args()
    if not 40<=args.steps<=1000 or args.seed<0:
        parser.error('Require 40..1000 steps and a nonnegative seed')
    if not args.distances or any(not .5<=d<=2. for d in args.distances):
        parser.error('Distances must be in [.5, 2] meters')
    if not args.delays or any(not 0.<=d<=1.5 for d in args.delays):
        parser.error('Delays must be in [0, 1.5] seconds')
    if not 0.<=args.approach_secs<=1.5:
        parser.error('approach-secs must be in [0, 1.5]')
    if not 0.<=args.reset_yaw_radians<=.15 or not 0.<=args.reset_lateral_m<=.05:
        parser.error('reset yaw/lateral must be in [0, .15] radians and [0, .05] meters')
    if not .25<=args.head_target_z<=.55 or not -.05<=args.body_target_z<=.35:
        parser.error('Target Z offsets are outside the task bounds')
    if not 0.<=args.head_overshoot<=.30 or not 0.<=args.body_overshoot<=.30:
        parser.error('Target overshoot is outside the task bounds')

    backend=PlaygroundBackend(args.config)
    if backend.settings['task']!='ShadowBoxG1Motion':
        parser.error('Contact controls require ShadowBoxG1Motion')
    options=dict(backend.settings.get('task_options',{}))
    if args.swap_start_sides:
        options['swap_start_sides']=True
    for key in ('motion_bundle','motion_script'):
        if options.get(key):
            options[key]=str((PROJECT_ROOT/Path(options[key])).resolve())
    from mujoco_playground._src.rl import env_registry
    from shadowbox.playground.registry import register_tasks
    register_tasks()
    entry=env_registry.get('ShadowBoxG1Motion')

    root=PROJECT_ROOT/'tmp/playground/contact-controls'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    root.mkdir(parents=True)
    local_sources={name:sha256(PROJECT_ROOT/name) for name in sorted(
        str(p.relative_to(PROJECT_ROOT)) for p in (PROJECT_ROOT/'src/shadowbox/playground').glob('*.py'))}
    local_sources['scripts/playground_contact_controls.py']=sha256(__file__)
    template=dict(scope='scripted contact positive controls; exposed targets; no policy evaluation or promotion',
        config=str(args.config),config_sha256=sha256(args.config),seed=args.seed,
        distances_m=args.distances,delays_secs=args.delays,sides=args.sides,targets=args.targets,
        steps=args.steps,approach_secs=args.approach_secs,
        reset_yaw_radians=args.reset_yaw_radians,reset_lateral_m=args.reset_lateral_m,
        configured_retreat_speeds_m_s={
            'a':float(options.get('retreat_speed_a',.18)),
            'b':float(options.get('retreat_speed_b',.18))},
        swap_start_sides=bool(options.get('swap_start_sides',False)),
        expected_case_count=len(args.distances)*len(args.sides)*len(args.targets)*len(args.delays),
        completed_case_count=0,complete=False,all_positive_controls=False,
        local_sources=local_sources,cases=[])
    (root/'summary.json').write_text(json.dumps(template,indent=2)+'\n')

    for distance in args.distances:
        for attacker in args.sides:
            for target in args.targets:
                for delay in args.delays:
                    if dict(local_sources)!={name:sha256(PROJECT_ROOT/name) for name in local_sources}:
                        raise RuntimeError('Runtime sources changed during contact-control run')
                    case_name=f'{attacker}-{target}-d{distance:.2f}-t{delay:.2f}'
                    if args.approach_secs:
                        case_name=f'{attacker}-{target}-d{distance:.2f}-a{args.approach_secs:.2f}-t{delay:.2f}'
                    case_dir=root/case_name
                    case=run_case(entry,options,seed=args.seed,distance=distance,
                        attacker=attacker,target=target,delay=delay,steps=args.steps,
                        head_z=args.head_target_z,body_z=args.body_target_z,
                        head_overshoot=args.head_overshoot,body_overshoot=args.body_overshoot,
                        approach_secs=args.approach_secs,reset_yaw_radians=args.reset_yaw_radians,
                        reset_lateral_m=args.reset_lateral_m,
                        out_dir=case_dir)
                    template['cases'].append(case)
                    template['completed_case_count']=len(template['cases'])
                    template['complete']=template['completed_case_count']==template['expected_case_count']
                    template['all_positive_controls']=template['complete'] and all(
                        c['positive_control'] for c in template['cases'])
                    (root/'summary.json').write_text(json.dumps(template,indent=2)+'\n')
                    print(json.dumps(case),flush=True)
    print(root/'summary.json')


if __name__=='__main__':
    main()
