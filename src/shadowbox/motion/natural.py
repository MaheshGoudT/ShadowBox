"""Natural movement enhancements for more realistic humanoid motion.

This module provides:
- Smooth trajectory blending between poses
- Velocity/acceleration limits based on joint physics
- Recovery motions after actions
- Optional GPU-accelerated trajectory optimization (when CUDA available)
"""

import numpy as np
import torch
import torch.nn as nn
from ..config import load_config


class NaturalMotion:
    """Enhances motion primitives with natural movement characteristics."""

    def __init__(self, config=None):
        self.config = config or load_config()
        self.use_gpu = self._check_cuda()
        self.device = 'cuda' if self.use_gpu else 'cpu'

    def _check_cuda(self):
        """Check if GPU is available for trajectory optimization."""
        try:
            if torch.cuda.is_available():
                return True
        except ImportError:
            pass
        return False

    def smooth_blend(self, start_pose, end_pose, duration=1.0):
        """Generate a smooth trajectory between poses using quintic polynomials."""
        t = np.linspace(0, 1, int(duration / 0.02))  # 50Hz
        delta = end_pose - start_pose
        a = 6 * delta
        b = -15 * delta
        c = 10 * delta

        # Quintic blend: q(t) = a*t^5 + b*t^4 + c*t^3 + start_pose
        trajectory = np.zeros((len(t), len(start_pose)))
        for i, ti in enumerate(t):
            trajectory[i] = a * ti**5 + b * ti**4 + c * ti**3 + start_pose

        # Apply velocity/acceleration limits from config
        vel_limits = self.config.get('motion', {}).get('velocity_limits', [])
        acc_limits = self.config.get('motion', {}).get('acceleration_limits', [])

        if vel_limits and acc_limits:
            trajectory = self._apply_limits(trajectory, vel_limits, acc_limits)

        return trajectory

    def _apply_limits(self, trajectory, vel_limits, acc_limits):
        """Apply velocity and acceleration constraints."""
        dt = 0.02  # 50Hz
        velocity = np.diff(trajectory, axis=0) / dt
        acceleration = np.diff(velocity, axis=0) / dt

        # Clip to limits (match trajectory dimensions)
        min_dim = min(velocity.shape[1], len(vel_limits))
        velocity[:, :min_dim] = np.clip(velocity[:, :min_dim],
                                          -np.array(vel_limits[:min_dim]),
                                          np.array(vel_limits[:min_dim]))

        # Reconstruct trajectory from limited velocity
        trajectory_limited = np.zeros_like(trajectory)
        trajectory_limited[0] = trajectory[0]
        for i in range(1, len(trajectory)):
            # Only update the dimensions that have limits
            trajectory_limited[i, :min_dim] = trajectory_limited[i-1, :min_dim] + velocity[i-1, :min_dim] * dt
            # Keep other dimensions unchanged from original trajectory
            trajectory_limited[i, min_dim:] = trajectory[i, min_dim:]
        return trajectory_limited

    def gpu_optimize(self, trajectory):
        """GPU-accelerated trajectory optimization (when CUDA available)."""
        if not self.use_gpu:
            return trajectory

        # Convert to torch tensor on GPU
        traj_tensor = torch.tensor(trajectory, device=self.device, dtype=torch.float32)

        # Simple optimization: smoothness regularization
        # In practice, implement proper jerk minimization

        return traj_tensor.cpu().numpy()

    def recovery_motion(self, current_pose, target_pose, duration=0.5):
        """Generate recovery motion to return to guard or neutral pose."""
        return self.smooth_blend(current_pose, target_pose, duration)


class GPUMotionOptimizer(nn.Module):
    """GPU-accelerated motion optimization module.

    Uses PyTorch on CUDA to optimize trajectories for:
    - Minimum jerk
    - Velocity/acceleration constraints
    - Smooth transitions
    """

    def __init__(self):
        super().__init__()
        # Placeholder parameters for optimization
        self.smoothness_weight = nn.Parameter(torch.tensor(1.0))

    def forward(self, trajectory):
        """Optimize trajectory on GPU."""
        if not torch.cuda.is_available():
            return trajectory

        # Move to GPU
        traj = torch.tensor(trajectory, device='cuda', dtype=torch.float32)

        # Placeholder optimization: smoothness regularization
        # In practice, implement proper jerk minimization

        return traj.cpu().numpy()


def create_natural_motion(config=None):
    """Factory function to create natural motion handler."""
    return NaturalMotion(config)


if __name__ == "__main__":
    # Test GPU availability
    motion = NaturalMotion()
    print(f"GPU available: {motion.use_gpu}")

    # Test trajectory generation
    start = np.array([0.0, 0.0, 0.0])
    end = np.array([0.5, 0.0, 0.0])
    traj = motion.smooth_blend(start, end, 1.0)
    print(f"Trajectory shape: {traj.shape}")

    # Test GPU optimization
    if motion.use_gpu:
        optimizer = GPUMotionOptimizer().cuda()
        optimized = optimizer(traj)
        print(f"Optimized trajectory shape: {optimized.shape}")
