"""Observed-geometry defensive baseline for the tracker-backed task."""
import numpy as np


class ReactiveGuard:
    """Choose high/body guard from measured closing gloves, with a short hold.

    No opponent action, goal, clock or planned trajectory is accepted here.
    """
    def __init__(self, batch_size, dt=.02):
        self.dt=dt
        self.remaining=np.zeros(batch_size,dtype=np.float32)
        self.previous_head=None
        self.previous_body=None

    def reset(self, mask=None):
        if mask is None:
            self.remaining[:]=0;self.previous_head=None;self.previous_body=None
        else:
            self.remaining[np.asarray(mask,dtype=bool)]=0
            # The next frame reinitializes velocities, avoiding a reset teleport threat.
            self.previous_head=None;self.previous_body=None

    def choose(self, gloves, head, torso):
        gloves=np.asarray(gloves,dtype=np.float32)
        head=np.asarray(head,dtype=np.float32);torso=np.asarray(torso,dtype=np.float32)
        if (gloves.shape!=(len(self.remaining),2,3) or head.shape!=(len(self.remaining),3)
                or torso.shape!=(len(self.remaining),3) or not all(np.isfinite(v).all() for v in (gloves,head,torso))):
            raise ValueError('Invalid observed glove/target geometry')
        head_distance=np.linalg.norm(gloves-head[:,None],axis=-1)
        body_distance=np.linalg.norm(gloves-torso[:,None],axis=-1)
        self.remaining=np.maximum(0,self.remaining-self.dt)
        if self.previous_head is not None:
            closing_head=(self.previous_head-head_distance)/self.dt
            closing_body=(self.previous_body-body_distance)/self.dt
            body=np.any((body_distance<head_distance-.03)&(body_distance<.45)&(closing_body>.1),axis=-1)
            urgent_head=np.any((head_distance<.23)&(closing_head>.1),axis=-1)
            self.remaining[body]=.3
            self.remaining[urgent_head]=0
        self.previous_head=head_distance.copy();self.previous_body=body_distance.copy()
        return np.where(self.remaining>0,8,0).astype(np.float32)[:,None]
