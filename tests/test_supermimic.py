"""The optional reference-pose path must be explicit and reproducible."""

import pytest

from shadowbox.config import load_config
from shadowbox.motion.supermimic import SuperMimic


def test_reference_library_loads_from_project_root(monkeypatch, tmp_path):
    config = load_config('configs/unitree.yaml')
    config['supermimic']['enabled'] = True
    monkeypatch.chdir(tmp_path)
    motion = SuperMimic(config)
    assert 'jab' in motion.motions
    assert isinstance(motion.get_motion_targets('a', 'jab'), dict)


def test_missing_reference_library_fails_fast():
    config = load_config('configs/unitree.yaml')
    config['supermimic'].update(enabled=True, motion_library='tmp/no-such-motion.json')
    with pytest.raises(ValueError, match='Could not load'):
        SuperMimic(config)


def test_unimplemented_model_inference_fails_fast():
    config = load_config('configs/unitree.yaml')
    config['supermimic'].update(enabled=True, model_path='tmp/model.onnx')
    with pytest.raises(ValueError, match='not implemented'):
        SuperMimic(config)


def test_controller_accepts_partial_named_pose():
    from shadowbox.motion.controller import MotionController
    from shadowbox.simulation import Arena

    config = load_config('configs/unitree.yaml')
    config['supermimic']['enabled'] = True
    arena = Arena(config=config)
    arena.reset()
    controller = MotionController(arena, 'a')
    current = arena.full_targets('a')
    joint = 'left_shoulder_pitch_joint'
    target = {joint: current[joint] + 0.05}
    assert controller.start('test_pose', target=target)
    assert len(controller.goal) == len(current) == 29
    assert controller.goal[joint] == pytest.approx(target[joint])
    assert controller.goal['right_shoulder_pitch_joint'] == current['right_shoulder_pitch_joint']
    controller.tick()


def test_named_pose_runs_through_match_and_status():
    from shadowbox.boxing import Match

    config = load_config('configs/unitree.yaml')
    config['supermimic']['enabled'] = True
    match = Match(config=config)
    assert match._request('a', 'hook_left')
    match.step()
    assert match.observe('a').state == 'hook_left'
    assert 'hook_left' in match.status()[1]
    assert match.stats['a']['attempts'] == 1
    assert not match._request('b', 'missing_motion')
