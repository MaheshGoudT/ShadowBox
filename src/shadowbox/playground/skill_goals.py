"""Initial boxing skill-to-sparse-goal compiler for the qualified G1 layout.

Guard, body guard, jab, cross and conservative footwork are authored goal
fixtures. Exposed-target head/body contacts have scripted positive controls;
this does not qualify a learned policy or closed-guard defensive performance.
"""
import numpy as np
from ..motion.primitives import TACTICAL_ACTIONS, Primitive

PUNCH_DURATIONS_SECS = {6: .7, 7: 1.1}
APPROACH_DURATION_SECS = .4
STRAIGHT_JAB_ARM_JOINTS = (
    'left_shoulder_pitch_joint', 'left_shoulder_roll_joint',
    'left_shoulder_yaw_joint', 'left_elbow_joint')
# Measured G1 joint-space head-jab baseline: roughly 13 degrees of elbow bend
# at full extension. Keep these values configurable for later motion calibration.
STRAIGHT_JAB_ARM_TARGET = (-1.54810, -0.01048, 0.04930, 1.49)
JAB_ARM_RISE_SECS = .25
JAB_ARM_RECOVERY_SECS = .25


class BoxingGoals:
    SUPPORTED = (0, 1, 2, 3, 4, 5, 6, 7, 8)
    LOCOMOTION = (2, 3, 4, 5)
    REBASE = (*LOCOMOTION, 1)

    def __init__(self, metadata, batch_size, defender_guard_width=.10,
                 opponent_guard_width=.10, head_target_z=.385, body_target_z=.15,
                 head_overshoot=.16, body_overshoot=.08,
                 body_attack_arc_lateral=0.,
                 retreat_speed_a=.18, retreat_speed_b=.18,
                 approach_speed_a=.27, approach_speed_b=.27,
                 max_punch_reach_m=.48):
        if not .10 <= defender_guard_width <= .50 or not .10 <= opponent_guard_width <= .50:
            raise ValueError("Diagnostic guard widths must be in [.10, .50]")
        values=(head_target_z,body_target_z,head_overshoot,body_overshoot,
                body_attack_arc_lateral)
        if (not np.isfinite(values).all() or not .25<=head_target_z<=.55
                or not -.05<=body_target_z<=.35
                or not 0.<=head_overshoot<=.30 or not 0.<=body_overshoot<=.30
                or not -.25<=body_attack_arc_lateral<=.25):
            raise ValueError('Invalid boxing target offsets')
        retreat_speeds=np.asarray((retreat_speed_a,retreat_speed_b),dtype=np.float32)
        if not np.isfinite(retreat_speeds).all() or np.any((retreat_speeds<0)|(retreat_speeds>.4)):
            raise ValueError('Retreat speeds must be finite and in [0, .4] m/s')
        approach_speeds=np.asarray((approach_speed_a,approach_speed_b),dtype=np.float32)
        if not np.isfinite(approach_speeds).all() or np.any((approach_speeds<0)|(approach_speeds>.4)):
            raise ValueError('Approach speeds must be finite and in [0, .4] m/s')
        if not np.isfinite(max_punch_reach_m) or not .3<=max_punch_reach_m<=.7:
            raise ValueError('Maximum punch reach must be finite and in [.3, .7] m')
        self.defender_guard_width = defender_guard_width
        self.opponent_guard_width = opponent_guard_width
        self.head_target_z=float(head_target_z)
        self.body_target_z=float(body_target_z)
        self.head_overshoot=float(head_overshoot)
        self.body_overshoot=float(body_overshoot)
        self.body_attack_arc_lateral=float(body_attack_arc_lateral)
        self.retreat_speeds=retreat_speeds
        self.approach_speeds=approach_speeds
        self.max_punch_reach_m=float(max_punch_reach_m)
        self.metadata = metadata
        self.batch_size = batch_size
        self.bodies = metadata['body_names']
        self.head_index=self.bodies.index('head') if 'head' in self.bodies else None
        self.shoulder_indices={hand:self.bodies.index(f'{hand}_shoulder_pitch_link')
            for hand in ('left','right') if f'{hand}_shoulder_pitch_link' in self.bodies}
        self.conditionable = metadata['masked_mimic']['conditionable_bodies']
        self.steps = metadata['masked_mimic']['num_masked_future_steps']
        if (self.steps != 5 or self.conditionable != ['torso_link','right_ankle_roll_link',
                'left_ankle_roll_link','left_rubber_hand','right_rubber_hand']):
            raise ValueError('Boxing goals require the qualified G1 five-point layout')
        self.last = np.full(batch_size,-1,dtype=int)
        self.clock = np.zeros(batch_size,dtype=np.float32)
        self.origin_pos = None
        self.origin_rot = None
        self.initialized = False

    def reset(self, positions, rotations, mask=None):
        positions=np.asarray(positions,dtype=np.float32)
        rotations=np.asarray(rotations,dtype=np.float32)
        if (positions.shape!=(self.batch_size,len(self.bodies),3)
                or rotations.shape!=(self.batch_size,len(self.bodies),4)
                or not np.isfinite(positions).all() or not np.isfinite(rotations).all()
                or not np.allclose(np.linalg.norm(rotations,axis=-1),1,atol=1e-3)):
            raise ValueError('Invalid body poses for a boxing goal reset')
        if mask is None:
            self.origin_pos=positions.copy();self.origin_rot=rotations.copy()
            self.last[:]=-1;self.clock[:]=0
        else:
            mask=np.asarray(mask,dtype=bool)
            if not self.initialized or mask.shape!=(self.batch_size,):
                raise ValueError('Partial goal reset requires initialized goals and a batch mask')
            self.origin_pos[mask]=positions[mask];self.origin_rot[mask]=rotations[mask]
            self.last[mask]=-1;self.clock[mask]=0
        self.initialized=True

    def advance(self, dt):
        if not np.isfinite(dt) or dt<0:
            raise ValueError('Goal time must advance by a finite nonnegative interval')
        self.clock+=dt

    def compile(self, actions, positions, rotations, opponent_anchor):
        if not self.initialized:raise ValueError('Reset goal compiler before use')
        actions=np.asarray(actions)
        if not np.isfinite(actions).all() or not np.all(actions==np.rint(actions)):
            raise ValueError('Boxing primitives must be integer action IDs')
        opponent_anchor=np.asarray(opponent_anchor,dtype=np.float32)
        if opponent_anchor.shape!=(self.batch_size,3) or not np.isfinite(opponent_anchor).all():
            raise ValueError('Invalid observed opponent anchor')
        actions=actions.astype(int)
        if actions.shape!=(self.batch_size,) or any(int(a) not in self.SUPPORTED for a in actions):
            raise ValueError('Goal fixtures support guard, turn, footwork, jab, cross and body guard only')
        changed=actions!=self.last
        self.clock[changed]=0
        # Re-anchor only when entering or leaving an authored movement skill.
        # Arm-skill changes retain the episode support reference for drift checks.
        for row in np.flatnonzero(changed):
            old=int(self.last[row])
            if actions[row] in self.REBASE or old in self.REBASE:
                self.origin_pos[row]=positions[row]
                self.origin_rot[row]=rotations[row]
        self.last[:]=actions
        n=self.batch_size;k=self.steps
        result={}
        for spec in self.metadata['policy_inputs']:
            if spec['provider']!='masked_mimic':continue
            value=np.zeros((n,*spec['shape'][1:]),dtype=spec['dtype'])
            if spec['kind'].endswith('ref_rot'):value[...,3]=1
            result[spec['key']]=value
        pos=np.repeat(self.origin_pos[:,None],k,axis=1)
        rot=np.repeat(self.origin_rot[:,None],k,axis=1)
        times=np.array([.10,.20,.30,.40,.50],dtype=np.float32)
        anchor=self.bodies.index('torso_link')
        base=self.origin_pos[:,anchor]
        q=self.origin_rot[:,anchor]
        yaw=np.arctan2(2*(q[:,3]*q[:,2]+q[:,0]*q[:,1]),1-2*(q[:,1]**2+q[:,2]**2))
        # A transient pose must not rotate guard/recovery goals while an arm
        # skill retains its support frame.
        facing_yaw=np.arctan2(opponent_anchor[:,1]-base[:,1],
                              opponent_anchor[:,0]-base[:,0])
        facing_delta=np.arctan2(np.sin(facing_yaw-yaw),np.cos(facing_yaw-yaw))
        heading_goal_yaw=facing_yaw.copy()
        turn_rows=actions==1
        # A direct 0.1 s heading target over-rotates this tracker when its
        # command is refreshed at 50 Hz. Use a conservative partial correction.
        heading_goal_yaw[turn_rows]=yaw[turn_rows]+.35*facing_delta[turn_rows]
        forward=np.stack((np.cos(facing_yaw),np.sin(facing_yaw),np.zeros(n)),axis=-1)
        left=np.stack((-np.sin(facing_yaw),np.cos(facing_yaw),np.zeros(n)),axis=-1)
        turn_forward=np.stack((np.cos(heading_goal_yaw),np.sin(heading_goal_yaw),np.zeros(n)),axis=-1)
        turn_left=np.stack((-np.sin(heading_goal_yaw),np.cos(heading_goal_yaw),np.zeros(n)),axis=-1)
        velocity=np.zeros((n,2),dtype=np.float32)
        movement=np.isin(actions,self.LOCOMOTION)
        for row,action in enumerate(actions):
            if action in self.LOCOMOTION:
                if action==2:
                    speed=float(self.approach_speeds[row])
                elif action==3:
                    speed=float(self.retreat_speeds[row])
                else:
                    speed=.12
                direction=(forward[row] if action==2 else -forward[row] if action==3
                           else left[row] if action==4 else -left[row])
                velocity[row]=speed*direction[:2]
        if 'masked_mimic.average_velocity' in result:
            result['masked_mimic.average_velocity']=velocity
        if 'masked_mimic.velocity_command_mask' in result:
            result['masked_mimic.velocity_command_mask']=movement[:,None]
        for row,action in enumerate(actions):
            primitive=TACTICAL_ACTIONS[action]
            if action==1 or action in self.LOCOMOTION:
                desired=heading_goal_yaw[row]
                rot[row,0,anchor]=[0.,0.,np.sin(desired/2),np.cos(desired/2)]
            if movement[row]:
                # Match the published Metatron steering skill: velocity and
                # heading goals plus alternating timed foot swings.
                speed=float(np.linalg.norm(velocity[row]))
                frequency=min(2.,.83+.93*speed)
                amplitude=.13+.065*speed
                phase=2*np.pi*frequency*(self.clock[row]+times)
                move_direction=velocity[row]/speed
                for hand,shift in (('right',0.),('left',np.pi)):
                    body=self.bodies.index(f'{hand}_ankle_roll_link')
                    angle=phase+shift
                    foot_offset=self.origin_pos[row,body]-base[row]
                    travel=np.zeros((k,3),dtype=np.float32)
                    travel[:,:2]=velocity[row][None,:]*(self.clock[row]+times)[:,None]
                    target=(base[row][None,:]+foot_offset[None,:]+travel+
                            (amplitude*np.cos(angle))[:,None]*np.r_[move_direction,0.][None,:])
                    target[:,2]+=.08*np.maximum(0.,-np.sin(angle))
                    pos[row,:,body]=target
            for hand,sign in (('left',1),('right',-1)):
                body=self.bodies.index(f'{hand}_rubber_hand')
                width=self.defender_guard_width if row==0 else self.opponent_guard_width
                if movement[row]:
                    hand_travel=np.zeros((k,3),dtype=np.float32)
                    hand_travel[:,:2]=velocity[row][None,:]*times[:,None]
                    guard_base=positions[row,anchor][None,:]+hand_travel
                    guard=guard_base+.18*forward[row]+sign*width*left[row]+[0,0,.32 if primitive!=Primitive.BODY_GUARD else .16]
                elif primitive==Primitive.TURN_TO_TARGET:
                    guard_base=base[row][None,:]
                    guard_forward=np.broadcast_to(turn_forward[row],(k,3))
                    guard_left=np.broadcast_to(turn_left[row],(k,3))
                    guard=(guard_base+.18*guard_forward+sign*width*guard_left+
                           [0,0,.32 if primitive!=Primitive.BODY_GUARD else .16])
                else:
                    guard=base[row]+.18*forward[row]+sign*width*left[row]+[0,0,.32 if primitive!=Primitive.BODY_GUARD else .16]
                pos[row,:,body]=guard
            if primitive in (Primitive.JAB,Primitive.CROSS):
                hand='left' if primitive==Primitive.JAB else 'right'
                body=self.bodies.index(f'{hand}_rubber_hand')
                if primitive==Primitive.JAB and self.head_index is not None:
                    # Target the measured tracker head point, including its
                    # posture and orientation, rather than estimating it from
                    # a fixed torso height.
                    target=positions[1-row,self.head_index].copy()
                else:
                    target=np.asarray(opponent_anchor[row],dtype=np.float32).copy()
                    target[2]+=self.head_target_z if primitive==Primitive.JAB else self.body_target_z
                # Aim through the target to account for tracking lag during contact.
                target+=(self.head_overshoot if primitive==Primitive.JAB else self.body_overshoot)*forward[row]
                shoulder_index=self.shoulder_indices.get(hand)
                if shoulder_index is not None:
                    shoulder=positions[row,shoulder_index]
                    reach=target-shoulder
                    distance=np.linalg.norm(reach)
                    if distance>self.max_punch_reach_m:
                        target=shoulder+(self.max_punch_reach_m/distance)*reach
                # A finite extend/retract cycle; requests do not become perpetual extension.
                future=self.clock[row]+times
                duration=PUNCH_DURATIONS_SECS[action]
                rise=.25 if primitive==Primitive.JAB else .35
                fall=.25 if primitive==Primitive.JAB else .35
                extension=np.minimum(np.clip(future/rise,0,1),np.clip((duration-future)/fall,0,1))
                extension=extension*extension*(3-2*extension)
                guard=pos[row,0,body].copy()
                lateral_arc=np.zeros(k,dtype=np.float32)
                if primitive==Primitive.CROSS:
                    progress=np.clip(future/duration,0,1)
                    lateral_arc=(self.body_attack_arc_lateral*np.sin(np.pi*progress)
                                 * extension)
                pos[row,:,body]=(guard+extension[:,None]*(target-guard)
                                 +lateral_arc[:,None]*left[row])
                # A small physical weight shift; feet stay planted targets, roots remain free.
                pos[row,:,anchor]+=(.05*extension)[:,None]*forward[row]
        masks=np.zeros((n,k,len(self.conditionable),2),dtype=bool)
        for index,name in enumerate(self.conditionable):
            masks[:,:,index,0]=True
            if name=='torso_link':masks[:,:,index,1]=True
        turn_rows=actions==1
        for row in np.flatnonzero(movement | turn_rows):
            masks[row,:,self.conditionable.index('torso_link'),0]=False
        heading_actions=np.isin(actions,(1,*self.LOCOMOTION))
        masks[heading_actions,:,self.conditionable.index('torso_link'),1]=False
        result['masked_mimic.ref_pos']=pos.astype(np.float32)
        result['masked_mimic.ref_rot']=rot.astype(np.float32)
        result['masked_mimic.time_offsets']=np.broadcast_to(times,(n,k)).copy()
        result['masked_mimic.target_bodies_masks']=masks.reshape(n,-1)
        result['masked_mimic.target_poses_masks']=masks.any(axis=(2,3))
        result['masked_mimic.anchor_translation_mask']=np.ones((n,k,2),dtype=bool)
        result['masked_mimic.anchor_translation_mask'][movement,:,0]=False
        result['masked_mimic.anchor_translation_mask'][turn_rows,:,0]=False
        result['masked_mimic.anchor_rotation_mask'][...,0]=True
        result['masked_mimic.anchor_rotation_mask'][heading_actions,:,0]=False
        result['masked_mimic.anchor_rotation_mask'][heading_actions,0,1]=True
        return result
