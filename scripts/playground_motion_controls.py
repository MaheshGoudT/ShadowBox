"""Measure authored turn and footwork commands through the frozen G1 tracker.

These fixed-command rollouts check direction, support, recovery and contact;
they are not policy evaluation or a learned-action vocabulary promotion.
Run in the isolated Playground GPU environment (CPU physics, CUDA tracker).
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))

from shadowbox.integrations.playground import PlaygroundBackend, PROJECT_ROOT

DT=.02
ACTIONS={'turn':1,'forward':2,'backward':3,'left':4,'right':5}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_case(entry, options, *, seed, side, action_name, action_secs, settle_secs,
             start_distance, out_dir):
    import mujoco
    import torch
    from mujoco_playground._src.rl.rsl_rl.vecenv_wrapper import RslRlVecEnvWrapper
    from mujoco_playground._src.rl import rng

    action_id=ACTIONS[action_name]
    settings=dict(options)
    settings.update(start_distance=start_distance,opponent_enabled=False,terminate_opponent=True,
        controlled_side=side,reset_yaw_radians=.15,reset_lateral_m=.05,
        swap_start_sides=settings.get('swap_start_sides',False),
        episode_length_secs=10.)
    out_dir.mkdir(parents=True,exist_ok=True)
    rng.seed(seed)
    raw=entry.env_builder(nworld=1,device='cpu',impl='cpp',**settings)
    try:
        env=RslRlVecEnvWrapper(raw,cfg=entry.runner_cfg.vecenv_wrapper_cfg)
        motion=raw.task_layer.get_component_by_name('motion')
        identity=motion.policy.provenance()
        sim=raw.device_layer.sim
        model=sim.mj_model
        mujoco.mj_saveModel(model,str(out_dir/'model.mjb'),None)
        root_joint=next(j_id for j_id in range(model.njnt)
            if model.joint(j_id).name.startswith(f'{side}/') and model.jnt_type[j_id]==mujoco.mjtJoint.mjJNT_FREE)
        root_address=model.jnt_qposadr[root_joint]
        other_side='b' if side=='a' else 'a'
        other_joint=next(j_id for j_id in range(model.njnt)
            if model.joint(j_id).name.startswith(f'{other_side}/') and model.jnt_type[j_id]==mujoco.mjtJoint.mjJNT_FREE)
        other_address=model.jnt_qposadr[other_joint]
        traces={key:[] for key in ('qpos','qvel','ctrl','action','qpos_next','reward','done','timeout',
                                  'requested_primitives','inference_ms','boxing_events')}
        action_steps=round(action_secs/DT)
        total_steps=round((action_secs+settle_secs)/DT)
        completed=[]
        started=time.monotonic()
        with torch.inference_mode():
            for step in range(total_steps):
                command=action_id if step<action_steps else 0
                current_qpos=sim.data.qpos.cpu().numpy().copy()
                current_qvel=sim.data.qvel.cpu().numpy().copy()
                traces['qpos'].append(current_qpos)
                traces['qvel'].append(current_qvel)
                action=torch.full((1,1),float(command),device='cpu')
                _,reward,done,extras=env.step(action)
                traces['ctrl'].append(sim.data.ctrl.cpu().numpy().copy())
                traces['qpos_next'].append(sim.data.qpos.cpu().numpy().copy())
                traces['action'].append(action.cpu().numpy().copy())
                traces['reward'].append(reward.cpu().numpy().copy())
                traces['done'].append(done.cpu().numpy().copy())
                traces['timeout'].append(extras.get('time_outs',torch.zeros_like(done)).cpu().numpy().copy())
                feature=raw._latest_features
                traces['requested_primitives'].append(np.stack([
                    feature['motion/a_primitive'].cpu().numpy(),
                    feature['motion/b_primitive'].cpu().numpy()],axis=-1))
                traces['inference_ms'].append(feature['motion/inference_ms'].cpu().numpy().copy())
                traces['boxing_events'].append(feature['boxing/events'].cpu().numpy().copy())
                if bool(done.any().item()):
                    completed.append(step)
                    break
        sim_steps=len(traces['qpos'])
        arrays={key:np.stack(value) for key,value in traces.items()}
        np.savez_compressed(out_dir/'transitions.npz',**arrays)
        # The live feature tensors are sampled at the control rate. Root poses
        # below are extracted from the saved MuJoCo transition arrays instead.
        states=arrays['qpos'][:,0]
        root_start=states[0,root_address:root_address+3]
        root_after=states[min(action_steps,sim_steps-1),root_address:root_address+3]
        root_end=states[-1,root_address:root_address+3]
        root_during=states[:min(action_steps+1,sim_steps),root_address:root_address+3]
        start_state=states[0]
        after_state=states[min(action_steps,sim_steps-1)]
        def torso_pose(state,which):
            data=mujoco.MjData(model)
            data.qpos[:]=state
            data.qvel[:]=0
            mujoco.mj_forward(model,data)
            body=model.body(f'{which}/torso_link').id
            return data.xpos[body].copy(),data.xquat[body].copy()
        own_start,quat_start=torso_pose(start_state,side)
        own_after,quat_after=torso_pose(after_state,side)
        other_start,_=torso_pose(start_state,other_side)
        other_after,_=torso_pose(after_state,other_side)
        def quaternion_yaw(q):
            w,x,y,z=q
            return float(np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z)))
        yaw=quaternion_yaw(quat_start)
        final_yaw=quaternion_yaw(quat_after)
        target_yaw_start=float(np.arctan2(other_start[1]-own_start[1],other_start[0]-own_start[0]))
        target_yaw_after=float(np.arctan2(other_after[1]-own_after[1],other_after[0]-own_after[0]))
        wrap=lambda angle:float(np.arctan2(np.sin(angle),np.cos(angle)))
        heading_error_start=abs(wrap(target_yaw_start-yaw))
        heading_error_after=abs(wrap(target_yaw_after-final_yaw))
        toward=other_start[:2]-own_start[:2]
        toward=toward/np.linalg.norm(toward)
        forward=np.array([np.cos(yaw),np.sin(yaw)])
        left=np.array([-np.sin(yaw),np.cos(yaw)])
        command_vec=(root_after[:2]-root_start[:2])
        case=dict(name=f'{side}-{action_name}',seed=seed,side=side,primitive=action_name,
            primitive_id=action_id,action_secs=action_secs,settle_secs=settle_secs,
            configured_retreat_speeds_m_s={
                'a':float(settings.get('retreat_speed_a',.18)),
                'b':float(settings.get('retreat_speed_b',.18))},
            configured_approach_speeds_m_s={
                'a':float(settings.get('approach_speed_a',.27)),
                'b':float(settings.get('approach_speed_b',.27))},
            start_distance_m=start_distance,
            swap_start_sides=bool(settings.get('swap_start_sides',False)),
            steps=sim_steps,simulated_secs=sim_steps*DT,wall_secs=time.monotonic()-started,
            root_displacement_during_command_m=command_vec.tolist(),
            displacement_forward_m=float(command_vec@forward),
            displacement_toward_opponent_m=float(command_vec@toward),
            displacement_left_m=float(command_vec@left),
            torso_heading_error_start_rad=heading_error_start,
            torso_heading_error_after_command_rad=heading_error_after,
            torso_heading_error_reduction_rad=heading_error_start-heading_error_after,
            torso_yaw_change_command_rad=wrap(final_yaw-yaw),
            settle_drift_m=float(np.linalg.norm(root_end[:2]-root_after[:2])),
            maximum_command_root_offset_m=float(np.linalg.norm(root_during[:,:2]-root_start[:2],axis=-1).max()),
            ended_early=bool(completed),termination_steps=completed,
            boxing_events_by_side_channel_glove=arrays['boxing_events'].sum(axis=(0,1)).astype(int).tolist(),
            motion_policy=identity)
        (out_dir/'case.json').write_text(json.dumps(case,indent=2)+'\n')
        return case
    finally:
        raw.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path('configs/playground-motion-exposed.yaml'))
    parser.add_argument('--sides',choices=('a','b'),nargs='+',default=['a','b'])
    parser.add_argument('--actions',choices=tuple(ACTIONS),nargs='+',default=list(ACTIONS))
    parser.add_argument('--action-secs',type=float,default=1.0)
    parser.add_argument('--settle-secs',type=float,default=2.0)
    parser.add_argument('--start-distance',type=float,default=.9,
                        help='distance between the two robot roots in metres')
    parser.add_argument('--seed',type=int,default=17031)
    parser.add_argument('--swap-start-sides',action='store_true')
    args=parser.parse_args()
    if (not .2<=args.action_secs<=3 or not .2<=args.settle_secs<=5
            or not .5<=args.start_distance<=2. or args.seed<0):
        parser.error('Require action_secs in [.2, 3], settle_secs in [.2, 5], start_distance in [.5, 2], and a nonnegative seed')
    backend=PlaygroundBackend(args.config)
    if backend.settings['task']!='ShadowBoxG1Motion':
        parser.error('Motion controls require ShadowBoxG1Motion')
    options=dict(backend.settings.get('task_options',{}))
    if args.swap_start_sides:
        options['swap_start_sides']=True
    for key in ('motion_bundle','motion_script'):
        if options.get(key):options[key]=str((PROJECT_ROOT/Path(options[key])).resolve())
    from mujoco_playground._src.rl import env_registry
    from shadowbox.playground.registry import register_tasks
    register_tasks()
    entry=env_registry.get('ShadowBoxG1Motion')
    root=PROJECT_ROOT/'tmp/playground/motion-controls'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    root.mkdir(parents=True)
    sources={name:sha256(PROJECT_ROOT/name) for name in sorted(
        str(p.relative_to(PROJECT_ROOT)) for p in (PROJECT_ROOT/'src/shadowbox/playground').glob('*.py'))}
    sources['scripts/playground_motion_controls.py']=sha256(__file__)
    summary=dict(scope='authored fixed-command turn and footwork diagnostics; no policy evaluation or promotion',
        config=str(args.config),config_sha256=sha256(args.config),seed=args.seed,
        configured_retreat_speeds_m_s={
            'a':float(options.get('retreat_speed_a',.18)),
            'b':float(options.get('retreat_speed_b',.18))},
        configured_approach_speeds_m_s={
            'a':float(options.get('approach_speed_a',.27)),
            'b':float(options.get('approach_speed_b',.27))},
        start_distance_m=args.start_distance,
        swap_start_sides=bool(options.get('swap_start_sides',False)),
        sides=args.sides,actions=args.actions,action_secs=args.action_secs,settle_secs=args.settle_secs,
        source_sha256=sources,cases=[])
    for side in args.sides:
        for action_index,action_name in enumerate(args.actions):
            case_dir=root/f'{side}-{action_name}'
            # Reuse each action seed on both sides for paired mirrored trials.
            case=run_case(entry,options,seed=args.seed+action_index,
                side=side,action_name=action_name,action_secs=args.action_secs,
                settle_secs=args.settle_secs,start_distance=args.start_distance,
                out_dir=case_dir)
            summary['cases'].append(case)
            (root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
            print(json.dumps(case,indent=2),flush=True)
    print(f'Motion-control evidence: {root}')


if __name__=='__main__':main()
