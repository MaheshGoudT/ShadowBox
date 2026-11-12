"""Reaf command/feature component executing tactical primitives through a tracker."""
from pathlib import Path
import time

from dm_env import specs
from gdm_robotics.interfaces import types as command_types
import numpy as np
from reaf.core import commands_processor
from reaf.simulation.torch import feature_producer
import torch

from .motion_policy import MotionKinematics, MotionPolicy
from .primitive_distribution import CONTINUE_ACTION_ID
from .skill_goals import (APPROACH_DURATION_SECS, BoxingGoals, JAB_ARM_RECOVERY_SECS,
                          JAB_ARM_RISE_SECS, PUNCH_DURATIONS_SECS,
                          STRAIGHT_JAB_ARM_JOINTS, STRAIGHT_JAB_ARM_TARGET)
from .task import player_observation


def resolve_primitive(action, previous, active_phase=0.,
                      approach_duration=APPROACH_DURATION_SECS):
    """Resolve CONTINUE and let an already launched punch finish its phase."""
    duration = PUNCH_DURATIONS_SECS.get(
        int(previous), approach_duration if int(previous)==2 else None)
    if duration is not None:
        return int(previous) if active_phase < duration else 0
    if action == CONTINUE_ACTION_ID:
        return int(previous)
    if not 0 <= action < CONTINUE_ACTION_ID:
        raise ValueError('Boxing action must be a goal ID 0–8 or CONTINUE (9)')
    return int(action)


def queue_primitive(action, previous, active_phase=0., pending_action=0,
                    approach_duration=APPROACH_DURATION_SECS):
    """Finish active punches, then run the latched next punch or approach."""
    action = int(action)
    previous = int(previous)
    pending_action = int(pending_action)
    if not 0 <= action <= CONTINUE_ACTION_ID:
        raise ValueError('Boxing action must be a goal ID 0–8 or CONTINUE (9)')
    if pending_action not in (0, 2, 6, 7):
        raise ValueError('Pending action must be approach (2), jab (6), cross (7), or empty (0)')
    if not np.isfinite(active_phase) or active_phase < 0:
        raise ValueError('Active skill phase must be finite and nonnegative')

    if (previous==2 and (not np.isfinite(approach_duration)
                         or not .1<=approach_duration<=1.5)):
        raise ValueError('Approach duration must be finite and in [.1, 1.5] seconds')
    if previous==2 and active_phase<approach_duration:
        if action in PUNCH_DURATIONS_SECS:
            pending_action=action
        return previous,pending_action
    if previous==2:
        action=pending_action or action
        previous=0
        pending_action=0

    duration = PUNCH_DURATIONS_SECS.get(previous)
    if duration is not None:
        if active_phase < duration:
            if pending_action == 2:
                # Preserve a sampled approach until the punch finishes. The
                # policy is sampled every control step, so later guard samples
                # must not erase the only approach command before its release.
                if action in PUNCH_DURATIONS_SECS and action != previous:
                    pending_action = action
                return previous, pending_action
            if action in (1, CONTINUE_ACTION_ID):
                # Target turns and CONTINUE keep an already queued follow-up.
                return previous, pending_action
            if action == 2:
                return previous, 2
            if action in PUNCH_DURATIONS_SECS and action != previous and not pending_action:
                pending_action = action
            elif action not in PUNCH_DURATIONS_SECS and action != previous:
                pending_action = 0
            return previous, pending_action
        action = pending_action or action
        previous = 0
        pending_action = 0

    return resolve_primitive(action, previous, active_phase, approach_duration), pending_action


def guard_widths_by_side(controlled_side, defender_width, opponent_width):
    """Return A/B widths with the defender role attached to the learner."""
    if controlled_side == 'a':
        return defender_width, opponent_width
    if controlled_side == 'b':
        return opponent_width, defender_width
    raise ValueError('Controlled side must be a or b')


def forward_approach_progress(previous_roots, current_roots, requests, target_range):
    """Measure each active approacher's own closing distance, capped at range."""
    previous=np.asarray(previous_roots,dtype=np.float32)
    current=np.asarray(current_roots,dtype=np.float32)
    requests=np.asarray(requests)
    if (previous.shape!=(2,2) or current.shape!=(2,2)
            or not np.isfinite(previous).all() or not np.isfinite(current).all()):
        raise ValueError('Approach root positions must be finite planar positions for both boxers')
    if (requests.shape!=(2,) or not np.isfinite(requests).all()
            or not np.isfinite(target_range) or not .5<=target_range<=1.5):
        raise ValueError('Approach requests and target range are invalid')
    progress=np.zeros(2,dtype=np.float32)
    for index in (0,1):
        if requests[index]!=2:
            continue
        toward=previous[1-index]-previous[index]
        distance=float(np.linalg.norm(toward))
        if distance<=target_range:
            continue
        own_motion=current[index]-previous[index]
        forward=float(np.dot(own_motion,toward/max(distance,1e-8)))
        progress[index]=min(distance-target_range,max(0.,forward))
    return progress


def apply_jab_arm_targets(joint_targets, primitives, phases, joint_indices,
                          arm_target=STRAIGHT_JAB_ARM_TARGET,
                          rise_secs=JAB_ARM_RISE_SECS,
                          recovery_secs=JAB_ARM_RECOVERY_SECS):
    """Blend a calibrated, near-straight arm pose through an active left jab."""
    values=np.asarray(joint_targets,dtype=np.float32)
    actions=np.asarray(primitives)
    phases=np.asarray(phases,dtype=np.float32)
    indices=np.asarray(joint_indices,dtype=int)
    target=np.asarray(arm_target,dtype=np.float32)
    duration=PUNCH_DURATIONS_SECS[6]
    if (values.ndim!=2 or not np.isfinite(values).all()
            or actions.shape!=(values.shape[0],) or not np.isfinite(actions).all()
            or not np.all(actions==np.rint(actions))
            or phases.shape!=(values.shape[0],) or not np.isfinite(phases).all()
            or np.any(phases<0) or indices.shape!=(4,)
            or len(set(indices.tolist()))!=4 or np.any(indices<0)
            or np.any(indices>=values.shape[1]) or target.shape!=(4,)
            or not np.isfinite(target).all()
            or not np.isfinite((rise_secs,recovery_secs)).all()
            or not .05<=rise_secs<=duration or not .05<=recovery_secs<=duration):
        raise ValueError('Invalid near-straight jab arm targets or phases')
    result=values.copy()
    for row in np.flatnonzero(actions.astype(int)==6):
        phase=min(float(phases[row]),duration)
        if duration-phase <= 2*np.finfo(np.float32).eps:
            continue
        extension=min(phase/rise_secs,(duration-phase)/recovery_secs,1.)
        extension=float(np.clip(extension,0.,1.))
        extension=extension*extension*(3.-2.*extension)
        if extension > 0.:
            result[row,indices]+=extension*(target-result[row,indices])
    return result


class TrackedBoxingCommands(commands_processor.CommandsProcessor, feature_producer.FeatureProducer):
    def __init__(self, name='', nworld=1, xpu_device='cpu', bundle='', script=None,
                 runtime='torchscript', policy_device='cuda:0', opponent_enabled=False, defender_guard_width=.10,
                 opponent_guard_width=.10, head_target_z=.385, body_target_z=.15,
                 head_overshoot=.16, body_overshoot=.08,
                 body_attack_arc_lateral=0.,
                 retreat_speed_a=.18, retreat_speed_b=.18,
                 approach_speed_a=.27, approach_speed_b=.27,
                 max_punch_reach_m=.48, approach_target_range_m=.65,
                 approach_duration_secs=APPROACH_DURATION_SECS,
                 jab_arm_target=STRAIGHT_JAB_ARM_TARGET,
                 opponent_attack_target='alternating', opponent_attack_delay_secs=0.,
                 controlled_side='a', opponent_policy_script=None,
                 stiffness=(), damping=(), effort=(), joint_names=()):
        if nworld!=1:
            raise ValueError('Tracker-backed boxing currently qualifies one two-robot world only')
        self._name=name;self.nworld=nworld;self.device=xpu_device
        self.policy=MotionPolicy(Path(bundle)/'unified_pipeline.yaml',batch_size=2,
            runtime=runtime,device=policy_device,script=script)
        metadata=self.policy.metadata
        for actual,expected,label in ((joint_names,metadata['joint_names'],'joint order'),
                (stiffness,metadata['control']['stiffness'],'stiffness'),
                (damping,metadata['control']['damping'],'damping'),
                (effort,metadata['control']['effort_limits'],'effort limits')):
            equal=list(actual)==list(expected) if label=='joint order' else np.allclose(actual,expected,atol=1e-5)
            if not equal:raise ValueError(f'Tracker/Playground robot mismatch: {label}')
        self.kinematics=MotionKinematics(Path(bundle)/'pelvis_fk',metadata)
        width_a,width_b=guard_widths_by_side(
            controlled_side,defender_guard_width,opponent_guard_width)
        self.goals=BoxingGoals(metadata,2,defender_guard_width=width_a,
            opponent_guard_width=width_b,head_target_z=head_target_z,
            body_target_z=body_target_z,head_overshoot=head_overshoot,
            body_overshoot=body_overshoot,
            body_attack_arc_lateral=body_attack_arc_lateral,
            retreat_speed_a=retreat_speed_a,
            retreat_speed_b=retreat_speed_b,
            approach_speed_a=approach_speed_a,
            approach_speed_b=approach_speed_b,
            max_punch_reach_m=max_punch_reach_m)
        self.opponent_enabled=opponent_enabled
        if opponent_attack_target not in ('alternating','head','body'):
            raise ValueError('Opponent attack target must be alternating, head or body')
        if not np.isfinite(opponent_attack_delay_secs) or not 0.<=opponent_attack_delay_secs<=1.5:
            raise ValueError('Opponent attack delay must be in [0, 1.5] seconds')
        if controlled_side not in ('a','b'):
            raise ValueError('controlled_side must be a or b')
        self.opponent_attack_target=opponent_attack_target
        self.opponent_attack_delay_secs=float(opponent_attack_delay_secs)
        if not np.isfinite(approach_target_range_m) or not .5<=approach_target_range_m<=1.5:
            raise ValueError('Approach target range must be finite and in [.5, 1.5] m')
        self.approach_target_range_m=float(approach_target_range_m)
        if not np.isfinite(approach_duration_secs) or not .1<=approach_duration_secs<=1.5:
            raise ValueError('Approach duration must be finite and in [.1, 1.5] seconds')
        self.approach_duration_secs=float(approach_duration_secs)
        self.jab_arm_target=np.asarray(jab_arm_target,dtype=np.float32)
        if (self.jab_arm_target.shape!=(4,) or not np.isfinite(self.jab_arm_target).all()
                or np.any(np.abs(self.jab_arm_target)>3.2)):
            raise ValueError('Jab arm target must contain four finite joint angles within +/-3.2 rad')
        self.jab_arm_joint_indices=np.asarray(
            [metadata['joint_names'].index(name) for name in STRAIGHT_JAB_ARM_JOINTS],dtype=int)
        self.previous_roots=None
        self.approach_progress=np.zeros(2,dtype=np.float32)
        self.controlled_index=0 if controlled_side=='a' else 1
        self.opponent_index=1-self.controlled_index
        self.clock=0.;self.latency=0.;self.current=None
        self.requests=np.zeros(2,dtype=int)
        self.pending_requests=np.zeros(2,dtype=int)
        self.features=None
        self.opponent_pool=[]
        self.opponent_pool_index=0
        self.opponent_policy=None
        self.scripted_opponent_policy=None
        if opponent_policy_script:
            self.scripted_opponent_policy=torch.jit.load(str(opponent_policy_script),map_location=xpu_device).eval()
            self.opponent_policy=self.scripted_opponent_policy

    @property
    def name(self):return self._name

    def consumed_commands_spec(self):
        return {'action':command_types.UnboundedArraySpec(shape=(1,),dtype=np.float32)}

    def produced_commands_keys(self):return {'a/ctrl','b/ctrl'}

    def required_features_keys(self):
        return {f'{side}/{key}' for side in ('a','b') for key in ('joint_pos','joint_vel')} | {
            f'{side}_root/{key}' for side in ('a','b') for key in ('xpos','xquat','joint_vel')}

    def produced_features_spec(self):
        return {key:specs.Array(shape=(),dtype=np.float32) for key in
                ('motion/inference_ms','motion/a_primitive','motion/b_primitive',
                 'motion/a_phase','motion/b_phase','motion/a_approach_progress',
                 'motion/b_approach_progress')}

    def _measure(self,f):
        self.features=f
        def stack(key):return np.concatenate([f[key.format(side=side)].detach().cpu().numpy() for side in ('a','b')],axis=0)
        q=stack('{side}_root/xquat')[:,[1,2,3,0]]
        self.current,self.positions,self.rotations=self.kinematics.measure(
            stack('{side}/joint_pos'),stack('{side}/joint_vel'),stack('{side}_root/xpos'),q,
            stack('{side}_root/joint_vel')[:,3:6])

    def on_begin_stepping(self,required_features,env_mask=None):
        self._measure(required_features)
        self.previous_roots=self._root_positions()
        self.approach_progress.fill(0.)
        mask=None if env_mask is None else np.tile(env_mask.detach().cpu().numpy(),2)
        if not self.policy.history.ready:mask=None
        self.policy.reset(self.current,mask)
        self.goals.reset(self.positions,self.rotations,mask)
        self.clock=0.

    def on_step(self,required_features):
        self._measure(required_features)
        self.clock+=.02
        self.goals.advance(.02)
        roots=self._root_positions()
        if self.previous_roots is not None:
            self.approach_progress=forward_approach_progress(
                self.previous_roots,roots,self.requests,self.approach_target_range_m)
        self.previous_roots=roots

    def _root_positions(self):
        anchor=self.policy.metadata['robot']['anchor_body_index']
        return self.positions[:,anchor,:2].copy()

    def produce_features(self,required_features):
        phases=np.clip(self.goals.clock,0,2)/2
        approach_progress={side:float(self.approach_progress[index])
                           for index,side in enumerate(('a','b'))}
        return {key:torch.full((self.nworld,),value,device=self.device,dtype=torch.float32)
                for key,value in (('motion/inference_ms',self.latency),
                                  ('motion/a_primitive',float(self.requests[0])),
                                  ('motion/b_primitive',float(self.requests[1])),
                                  ('motion/a_phase',float(phases[0])),
                                  ('motion/b_phase',float(phases[1])),
                                  ('motion/a_approach_progress',approach_progress['a']),
                                  ('motion/b_approach_progress',approach_progress['b']))}

    def process_commands(self,consumed_commands):
        raw=consumed_commands['action'].detach().cpu().numpy()
        if raw.shape!=(1,1) or not np.isfinite(raw).all() or not np.all(raw==np.rint(raw)):
            raise ValueError('Tracker task requires integer Primitive action IDs, not joint vectors')
        (self.requests[self.controlled_index],
         self.pending_requests[self.controlled_index]) = queue_primitive(
            int(raw[0,0]), self.requests[self.controlled_index],
            self.goals.clock[self.controlled_index],
            self.pending_requests[self.controlled_index],
            self.approach_duration_secs)
        opponent_action=0
        if self.opponent_policy is not None:
            opponent_side='b' if self.controlled_index==0 else 'a'
            observation=player_observation(
                self.features, opponent_side,
                active_primitive=self.requests[self.opponent_index],
                active_phase=self.goals.clock[self.opponent_index])
            with torch.inference_mode():
                if isinstance(self.opponent_policy,torch.jit.ScriptModule):
                    output=self.opponent_policy(observation)
                else:
                    from tensordict import TensorDict
                    obs=TensorDict({'policy':observation},batch_size=[self.nworld])
                    output=self.opponent_policy(obs)
            selected=output.detach().reshape(-1).cpu().numpy()
            if (selected.shape!=(1,) or not np.isfinite(selected).all()
                    or selected[0]!=np.rint(selected[0]) or not 0<=selected[0]<=9):
                raise ValueError('Frozen opponent policy returned an invalid boxing primitive')
            opponent_action = int(selected[0])
        elif self.opponent_enabled:
            elapsed=self.clock-self.opponent_attack_delay_secs
            if elapsed>=0:
                if self.opponent_attack_target=='alternating':
                    cycle=elapsed%3.2
                    if cycle<.7:opponent_action=6
                    elif 1.6<=cycle<2.7:opponent_action=7
                else:
                    head=self.opponent_attack_target=='head'
                    cycle=elapsed%1.6
                    if cycle<(.7 if head else 1.1):
                        opponent_action=6 if head else 7
        (self.requests[self.opponent_index],
         self.pending_requests[self.opponent_index]) = queue_primitive(
            opponent_action, self.requests[self.opponent_index],
            self.goals.clock[self.opponent_index],
            self.pending_requests[self.opponent_index],
            self.approach_duration_secs)
        anchor=self.policy.metadata['robot']['anchor_body_index']
        other=self.positions[::-1,anchor].copy()
        goals=self.goals.compile(self.requests,self.positions,self.rotations,other)
        tick=time.monotonic()
        output=self.policy.predict(self.current,goals)
        output['joint_pos_targets']=apply_jab_arm_targets(
            output['joint_pos_targets'],self.requests,self.goals.clock,
            self.jab_arm_joint_indices,self.jab_arm_target)
        self.latency=1000*(time.monotonic()-tick)
        # Feedback records the exact tracker command accepted by the task.
        self.policy.accept(self.current,output)
        return {f'{side}/ctrl':torch.as_tensor(output['joint_pos_targets'][index:index+1],
                 device=self.device,dtype=torch.float32) for index,side in enumerate(('a','b'))}

    def reset(self,env_mask=None):
        # Fresh measurements initialize each reset row in on_begin_stepping.
        self.requests[:]=0
        self.pending_requests[:]=0
        self.latency=0.
        if self.opponent_pool:
            self.opponent_policy=self.opponent_pool[self.opponent_pool_index%len(self.opponent_pool)]
            self.opponent_pool_index+=1

    def set_opponent_pool(self, policies, start_index=0):
        """Mix the scripted schedule and frozen policies across episodes."""
        if not all(isinstance(policy,torch.nn.Module) for policy in policies):
            raise ValueError('Opponent league entries must be Torch policies')
        if type(start_index) is not int or start_index < 0:
            raise ValueError('Opponent pool start index must be a nonnegative integer')
        # A None entry selects the timed jab/cross schedule in process_commands.
        scripted = (self.scripted_opponent_policy
                    if self.scripted_opponent_policy is not None else None)
        self.opponent_pool = [scripted, *policies]
        self.opponent_pool_index = start_index % len(self.opponent_pool)
        self.opponent_policy = self.opponent_pool[self.opponent_pool_index]
