"""Contact episode metrics for the Playground task; no planned-action inputs.

Force episodes are not official boxing scores or a one-event-per-punch referee.
A guard contact is reported only if the incoming glove did not also touch a
legal target in that frame. Collapse contacts are excluded by posture gates.
"""
from dm_env import specs
import numpy as np
from reaf.simulation.torch import flat_scene as cfg
from reaf.simulation.torch import feature_producer
import torch

CHANNELS = ('incoming_head', 'incoming_body', 'incoming_block', 'outgoing_head', 'outgoing_body')


def sensor_components(history_length=4):
    from reaf.simulation import torch as devices
    result = {}
    for direction, attacker, defender in (('incoming', 'b', 'a'), ('outgoing', 'a', 'b')):
        gloves = (f'{attacker}/left_glove', f'{attacker}/right_glove')
        for target in ('head', 'body'):
            geom = 'head_collision' if target == 'head' else 'torso_collision'
            result[f'{direction}_{target}'] = cfg.Component(devices.ContactSensorDevice,
                fields=('found', 'force'), history_length=history_length).configure_spec(
                    objtype='geom', objname=gloves, reftype='geom', refname=f'{defender}/{geom}', reduce='netforce')
    for hand in ('left', 'right'):
        result[f'incoming_block_{hand}'] = cfg.Component(devices.ContactSensorDevice,
            fields=('found', 'force'), history_length=history_length).configure_spec(
                objtype='geom', objname=('b/left_glove', 'b/right_glove'),
                reftype='xbody', refname=f'a/{hand}_elbow_link', reduce='netforce')
    return result


def contact_count(features, channel):
    return features['boxing/counts'][:, channel].sum(-1)


class ContactEpisodes(feature_producer.FeatureProducer):
    def __init__(self, name='', nworld=1, xpu_device='cpu', threshold=5.0, cooldown=0.3, dt=0.02):
        self._name = name
        self.threshold = threshold
        self.cooldown = cooldown
        self.dt = dt
        shape = (nworld, len(CHANNELS), 2)
        self.counts = torch.zeros(shape, device=xpu_device)
        self.events = torch.zeros_like(self.counts)
        self.previous = torch.zeros(shape, dtype=torch.bool, device=xpu_device)
        self.remaining = torch.zeros_like(self.counts)

    @property
    def name(self):
        return self._name

    def required_features_keys(self):
        sensors = ('incoming_head', 'incoming_body', 'outgoing_head', 'outgoing_body',
                   'incoming_block_left', 'incoming_block_right')
        return ({f'{key}/force_history' for key in sensors}
                | {f'{side}_root/{field}' for side in ('a', 'b') for field in ('xpos', 'xquat')})

    def produced_features_spec(self):
        return {f'boxing/{key}': specs.Array(shape=(len(CHANNELS), 2), dtype=np.float32)
                for key in ('events', 'counts')}

    def _contacts(self, f):
        def contact(key):
            # Keep the two attacking gloves separate, reduce over substeps.
            return f[f'{key}/force_history'].norm(dim=-1).amax(dim=2) >= self.threshold
        ih, ib = contact('incoming_head'), contact('incoming_body')
        oh, ob = contact('outgoing_head'), contact('outgoing_body')
        block = contact('incoming_block_left') | contact('incoming_block_right')
        values = torch.stack((ih, ib & ~ih, block & ~ih & ~ib, oh, ob & ~oh), dim=1)
        legal = torch.ones(values.shape[0], dtype=torch.bool, device=values.device)
        for side in ('a', 'b'):
            q = f[f'{side}_root/xquat']
            upright = 1 - 2 * (q[:, 1].square() + q[:, 2].square())
            legal &= (f[f'{side}_root/xpos'][:, 2] > 0.65) & (upright > 0.75)
        return values & legal[:, None, None]

    def on_step(self, required_features):
        contacts = self._contacts(required_features)
        self.remaining.sub_(self.dt).clamp_(min=0)
        event = contacts & ~self.previous & (self.remaining <= 0)
        self.events.copy_(event)
        self.counts.add_(self.events)
        self.remaining[:] = torch.where(event, self.cooldown, self.remaining)
        self.previous.copy_(contacts)

    def on_begin_stepping(self, required_features, env_mask=None):
        contacts = self._contacts(required_features)
        if env_mask is None:
            self.previous.copy_(contacts)
        else:
            self.previous[env_mask] = contacts[env_mask]

    def produce_features(self, required_features):
        return {'boxing/events': self.events.clone(), 'boxing/counts': self.counts.clone()}

    def reset(self, env_mask=None):
        if env_mask is None:
            for value in (self.counts, self.events, self.previous, self.remaining):
                value.zero_()
        else:
            for value in (self.counts, self.events, self.previous, self.remaining):
                value[env_mask] = 0
