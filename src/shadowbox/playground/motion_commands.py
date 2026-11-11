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
from .skill_goals import BoxingGoals


class TrackedBoxingCommands(commands_processor.CommandsProcessor, feature_producer.FeatureProducer):
    def __init__(self, name='', nworld=1, xpu_device='cpu', bundle='', script=None,
                 runtime='torchscript', policy_device='cuda:0', opponent_enabled=False, defender_guard_width=.10,
                 opponent_guard_width=.10, head_target_z=.385, body_target_z=.15,
                 head_overshoot=.16, body_overshoot=.08,
                 retreat_speed_a=.18, retreat_speed_b=.18,
                 opponent_attack_target='alternating', opponent_attack_delay_secs=0.,
                 controlled_side='a',
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
        self.goals=BoxingGoals(metadata,2,defender_guard_width=defender_guard_width,
            opponent_guard_width=opponent_guard_width,head_target_z=head_target_z,
            body_target_z=body_target_z,head_overshoot=head_overshoot,
            body_overshoot=body_overshoot,retreat_speed_a=retreat_speed_a,
            retreat_speed_b=retreat_speed_b)
        self.opponent_enabled=opponent_enabled
        if opponent_attack_target not in ('alternating','head','body'):
            raise ValueError('Opponent attack target must be alternating, head or body')
        if not np.isfinite(opponent_attack_delay_secs) or not 0.<=opponent_attack_delay_secs<=1.5:
            raise ValueError('Opponent attack delay must be in [0, 1.5] seconds')
        if controlled_side not in ('a','b'):
            raise ValueError('controlled_side must be a or b')
        self.opponent_attack_target=opponent_attack_target
        self.opponent_attack_delay_secs=float(opponent_attack_delay_secs)
        self.controlled_index=0 if controlled_side=='a' else 1
        self.opponent_index=1-self.controlled_index
        self.clock=0.;self.latency=0.;self.current=None
        self.requests=np.zeros(2,dtype=int)

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
                ('motion/inference_ms','motion/a_primitive','motion/b_primitive')}

    def _measure(self,f):
        def stack(key):return np.concatenate([f[key.format(side=side)].detach().cpu().numpy() for side in ('a','b')],axis=0)
        q=stack('{side}_root/xquat')[:,[1,2,3,0]]
        self.current,self.positions,self.rotations=self.kinematics.measure(
            stack('{side}/joint_pos'),stack('{side}/joint_vel'),stack('{side}_root/xpos'),q,
            stack('{side}_root/joint_vel')[:,3:6])

    def on_begin_stepping(self,required_features,env_mask=None):
        self._measure(required_features)
        mask=None if env_mask is None else np.tile(env_mask.detach().cpu().numpy(),2)
        if not self.policy.history.ready:mask=None
        self.policy.reset(self.current,mask)
        self.goals.reset(self.positions,self.rotations,mask)
        self.clock=0.

    def on_step(self,required_features):
        self._measure(required_features)
        self.clock+=.02
        self.goals.advance(.02)

    def produce_features(self,required_features):
        return {key:torch.full((self.nworld,),value,device=self.device,dtype=torch.float32)
                for key,value in (('motion/inference_ms',self.latency),
                                  ('motion/a_primitive',float(self.requests[0])),
                                  ('motion/b_primitive',float(self.requests[1])))}

    def process_commands(self,consumed_commands):
        raw=consumed_commands['action'].detach().cpu().numpy()
        if raw.shape!=(1,1) or not np.isfinite(raw).all() or not np.all(raw==np.rint(raw)):
            raise ValueError('Tracker task requires integer Primitive action IDs, not joint vectors')
        self.requests[self.controlled_index]=int(raw[0,0])
        self.requests[self.opponent_index]=0
        if self.opponent_enabled:
            elapsed=self.clock-self.opponent_attack_delay_secs
            if elapsed>=0:
                if self.opponent_attack_target=='alternating':
                    cycle=elapsed%3.2
                    if cycle<.7:self.requests[self.opponent_index]=6
                    elif 1.6<=cycle<2.7:self.requests[self.opponent_index]=7
                else:
                    head=self.opponent_attack_target=='head'
                    cycle=elapsed%1.6
                    if cycle<(.7 if head else 1.1):
                        self.requests[self.opponent_index]=6 if head else 7
        anchor=self.policy.metadata['robot']['anchor_body_index']
        other=self.positions[::-1,anchor].copy()
        goals=self.goals.compile(self.requests,self.positions,self.rotations,other)
        tick=time.monotonic()
        output=self.policy.predict(self.current,goals)
        self.latency=1000*(time.monotonic()-tick)
        # No editing of a prediction after feedback: the actual accepted command is recorded.
        self.policy.accept(self.current,output)
        return {f'{side}/ctrl':torch.as_tensor(output['joint_pos_targets'][index:index+1],
                 device=self.device,dtype=torch.float32) for index,side in enumerate(('a','b'))}

    def reset(self,env_mask=None):
        # Fresh measurements initialize each reset row in on_begin_stepping.
        self.requests[:]=0
        self.latency=0.
