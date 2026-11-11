"""Optional quintic pose blending with configured velocity limits."""

import numpy as np

from ..config import load_config


class NaturalMotion:
    def __init__(self, config=None):
        self.config = config or load_config()

    def smooth_blend(self, start_pose, end_pose, duration=1.0):
        """Generate a 50 Hz quintic blend and clip configured joint speeds."""
        start_pose=np.asarray(start_pose,dtype=np.float64)
        end_pose=np.asarray(end_pose,dtype=np.float64)
        if (start_pose.shape!=end_pose.shape or not np.isfinite(start_pose).all()
                or not np.isfinite(end_pose).all() or not np.isfinite(duration)
                or duration<=0):
            raise ValueError('Pose endpoints and duration must be finite and compatible')
        times=np.linspace(0,1,max(2,int(duration/.02)))
        blend=(10*times**3-15*times**4+6*times**5)[:,None]
        trajectory=start_pose[None,:]+blend*(end_pose-start_pose)[None,:]
        limits=self.config.get('velocity_limits')
        if limits is None:
            limits=self.config.get('motion',{}).get('velocity_limits',[])
        if limits:
            trajectory=self._apply_velocity_limits(trajectory,limits)
        return trajectory

    @staticmethod
    def _apply_velocity_limits(trajectory, limits):
        dt=.02
        velocity=np.diff(trajectory,axis=0)/dt
        width=min(velocity.shape[1],len(limits))
        maximum=np.asarray(limits[:width],dtype=np.float64)
        if not np.isfinite(maximum).all() or np.any(maximum<=0):
            raise ValueError('Velocity limits must be finite and positive')
        velocity[:,:width]=np.clip(velocity[:,:width],-maximum,maximum)
        limited=np.zeros_like(trajectory)
        limited[0]=trajectory[0]
        for index in range(1,len(trajectory)):
            limited[index,:width]=limited[index-1,:width]+velocity[index-1,:width]*dt
            limited[index,width:]=trajectory[index,width:]
        return limited
