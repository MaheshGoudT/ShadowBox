"""Motion references must be safe to replay and useful for model fitting."""

import json

import numpy as np
import pytest

from shadowbox.config import PROJECT_ROOT, load_config
from shadowbox.evaluation.motion_replay import replay_motion


EXAMPLE = PROJECT_ROOT / 'assets/motions/boxing-jab-example.json'


def test_replay_reports_tracking_and_aligned_transitions(tmp_path):
    output = tmp_path / 'transitions.npz'
    config = load_config('configs/unitree.yaml')
    for robot in ('a', 'b'):
        result = replay_motion(EXAMPLE, config, robot, output)
        assert result['frames'] == 3
        assert 0 < result['joint_rmse_rad'] < 1
        assert result['planted_foot_samples'] > 0
        assert result['mean_planted_foot_height_error_m'] < .02
        with np.load(output) as data:
            assert data['qpos'].shape == data['next_qpos'].shape
            assert data['qvel'].shape == data['next_qvel'].shape
            assert data['ctrl'].shape[0] == data['qpos'].shape[0]
            assert np.isfinite(data['next_qpos']).all()
            assert np.any(data['next_qpos'] != data['qpos'])


def test_replay_rejects_out_of_limit_joint_and_ring_exit(tmp_path):
    clip = json.loads(EXAMPLE.read_text())
    path = tmp_path / 'bad.json'
    clip['frames'][1]['joints']['left_elbow_joint'] = 9
    path.write_text(json.dumps(clip))
    with pytest.raises(ValueError, match='joint target'):
        replay_motion(path, load_config('configs/unitree.yaml'))
    clip['frames'][1]['joints']['left_elbow_joint'] = 1.49
    clip['frames'][1]['root'] = [9, 0, 0]
    path.write_text(json.dumps(clip))
    with pytest.raises(ValueError, match='leaves the ring'):
        replay_motion(path, load_config('configs/unitree.yaml'))
