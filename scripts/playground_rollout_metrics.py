"""Post-hoc posture, drift, ankle travel and glove-distance diagnostics.

Uses each run's saved compiled MuJoCo model. Center distances are diagnostic;
only the live contact classifier supplies contact episode counts.
"""
import argparse
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir',type=Path)
    parser.add_argument('--world',type=int,default=0)
    args=parser.parse_args()
    import mujoco
    import numpy as np
    model=mujoco.MjModel.from_binary_path(str(args.run_dir/'model.mjb'))
    data=mujoco.MjData(model)
    archive=np.load(args.run_dir/'transitions.npz')
    states=archive['qpos'][:,args.world]
    done=archive['done'][:,args.world].astype(bool)
    report={'world':args.world,'steps':len(states),'sides':{},
            'scope':'recorded posture/ankle travel/center distances; ankle travel includes foot rotation and is not sole slip; center distance is not contact scoring'}
    timeout=archive['timeout'][:,args.world].astype(bool) if 'timeout' in archive else np.zeros_like(done)
    report['episode_endings']=dict(total=int(done.sum()),time_limit=int((done&timeout).sum()),non_time_limit=int((done&~timeout).sum()))
    values={side:dict(root=[],upright=[],feet=[],planted=[],head_distance=[],body_distance=[]) for side in ('a','b')}
    for index,qpos in enumerate(states):
        data.qpos[:]=qpos
        if 'qvel' in archive:data.qvel[:]=archive['qvel'][index,args.world]
        if 'ctrl' in archive:data.ctrl[:]=archive['ctrl'][index,args.world]
        mujoco.mj_forward(model,data)
        for side,other in (('a','b'),('b','a')):
            out=values[side]
            root=model.body(f'{side}/pelvis').id
            out['root'].append(data.xpos[root].copy())
            q=data.xquat[root]
            out['upright'].append(1-2*(q[1]**2+q[2]**2))
            feet=[];planted=[]
            for hand in ('left','right'):
                foot_body=model.body(f'{side}/{hand}_ankle_roll_link').id
                foot_ids={g for g in range(model.ngeom) if model.geom_bodyid[g]==foot_body
                          and model.geom_contype[g]!=0}
                feet.append(data.xpos[foot_body].copy())
                planted.append(any(
                    (int(c.geom1) in foot_ids and model.geom_bodyid[int(c.geom2)]==0
                     or int(c.geom2) in foot_ids and model.geom_bodyid[int(c.geom1)]==0)
                    and c.dist<=.005 for c in data.contact))
            out['feet'].append(feet);out['planted'].append(planted)
            # Sample actual floor-contact point velocity, rather than ankle drift.
            # Multiple contact points are averaged per foot per recorded control frame.
            contact_speeds=[]
            for hand in ('left','right'):
                foot_body=model.body(f'{side}/{hand}_ankle_roll_link').id
                foot_ids={g for g in range(model.ngeom) if model.geom_bodyid[g]==foot_body
                          and model.geom_contype[g]!=0}
                speeds=[]
                for contact in data.contact:
                    g1,g2=int(contact.geom1),int(contact.geom2)
                    foot=g1 if g1 in foot_ids and model.geom_bodyid[g2]==0 else (
                         g2 if g2 in foot_ids and model.geom_bodyid[g1]==0 else None)
                    if foot is None or contact.dist>0:continue
                    jac=np.zeros((3,model.nv))
                    mujoco.mj_jac(model,data,jac,None,contact.pos,int(model.geom_bodyid[foot]))
                    velocity=jac@data.qvel
                    normal=contact.frame[:3]
                    tangent=velocity-normal*np.dot(velocity,normal)
                    speeds.append(float(np.linalg.norm(tangent)))
                contact_speeds.append(float(np.mean(speeds)) if speeds else float('nan'))
            out.setdefault('contact_speeds',[]).append(contact_speeds)
            head=data.geom_xpos[model.geom(f'{side}/head_collision').id]
            torso=data.geom_xpos[model.geom(f'{side}/torso_collision').id]
            gloves=np.stack([data.geom_xpos[model.geom(f'{other}/{hand}_glove').id] for hand in ('left','right')])
            out['head_distance'].append(np.linalg.norm(gloves-head,axis=-1))
            out['body_distance'].append(np.linalg.norm(gloves-torso,axis=-1))
            for target,geom_name in (('head','head_collision'),('body','torso_collision')):
                gaps=[mujoco.mj_geomDistance(model,data,model.geom(f'{other}/{hand}_glove').id,
                    model.geom(f'{side}/{geom_name}').id,10.,None) for hand in ('left','right')]
                out.setdefault(target+'_gap',[]).append(gaps)
    for side,v in values.items():
        root=np.asarray(v['root']);up=np.asarray(v['upright']);feet=np.asarray(v['feet']);plant=np.asarray(v['planted'])
        stable=plant[1:]&plant[:-1]&~done[:-1,None]
        slide=np.linalg.norm(np.diff(feet[...,:2],axis=0),axis=-1)
        starts=np.maximum.accumulate(np.where(np.r_[True,done[:-1]],np.arange(len(root)),0))
        drift=np.linalg.norm(root[:,:2]-root[starts,:2],axis=-1)
        report['sides'][side]=dict(minimum_height_m=float(root[:,2].min()),minimum_upright_z=float(up.min()),
            maximum_root_drift_m=float(drift.max()),unsafe_posture_frames=int(((root[:,2]<.65)|(up<.75)).sum()),
            near_floor_ankle_xy_travel_m=(slide*stable).sum(axis=0).tolist(),
            planted_foot_fraction=plant.mean(axis=0).tolist(),
            closest_incoming_glove_to_head_center_m=np.asarray(v['head_distance']).min(axis=0).tolist(),
            closest_incoming_glove_to_torso_center_m=np.asarray(v['body_distance']).min(axis=0).tolist())
        for target in ('head','body'):
            gaps=np.asarray(v[target+'_gap'])
            report['sides'][side][f'minimum_sampled_incoming_{target}_surface_gap_m']=gaps.min(axis=0).tolist()
            report['sides'][side][f'closest_incoming_{target}_frames']=gaps.argmin(axis=0).tolist()
        if 'qvel' in archive:
            speeds=np.asarray(v['contact_speeds'])
            report['sides'][side]['sampled_floor_contact_tangential_travel_m']=(np.nan_to_num(speeds)*.02).sum(axis=0).tolist()
            report['sides'][side]['sampled_floor_contact_fraction']=np.isfinite(speeds).mean(axis=0).tolist()
    report['contact_velocity_scope']='Contact-point Jacobian velocity sampled every 20 ms, averaged across each foot contact patch; travel is an approximation, not substep-integrated sole slip'
    if 'requested_primitives' in archive:
        requests=archive['requested_primitives'][:,args.world]
        previous=np.vstack([np.zeros((1,2)),requests[:-1]])
        report['commanded_attacks_by_side']={side:int(((requests[:,i]!=previous[:,i])&np.isin(requests[:,i],[6,7])).sum()) for i,side in enumerate(('a','b'))}
        report['attack_scope']='Commanded attacks, not proof of observed active threats'
    if 'inference_ms' in archive:
        latency=archive['inference_ms'][:,args.world]
        report['motion_inference_ms']=dict(mean=float(latency.mean()),p95=float(np.quantile(latency,.95)))
    (args.run_dir/'posture-metrics.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
