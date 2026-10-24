"""Canonical mapping and command boundaries without the optional training stack."""

import json
import sys

import mujoco
import numpy as np
import pytest

from shadowbox.config import PROJECT_ROOT, load_config
from shadowbox.evaluation.motion_replay import replay_motion
from shadowbox.integrations import morphmimic as integration
from shadowbox.motion.morphology import MorphMimicProfile
from shadowbox.simulation import Arena


@pytest.fixture
def settings():
    result = integration.load_settings()
    if not (result['repository'] / 'deploy/sim2sim/obs.py').exists():
        pytest.skip('initialize optional external/MorphMimic submodule')
    return result


@pytest.mark.parametrize('name', integration.ROBOTS)
def test_profiles_translate_canonical_actions_and_hold_extra_joints(name, settings):
    profile = MorphMimicProfile(name, settings)
    robot = profile.robot
    action = np.linspace(-.2, .2, 29)
    targets = profile.native_targets(action)
    native = np.array([targets[joint] for joint in robot.native_joint_names])
    recovered = robot.to_canonical_delta((native - robot.default_joint_pos) / robot.action_scale)
    assert np.allclose(recovered, action)
    extras = sorted(set(range(robot.num_native)) - set(robot.perm))
    assert np.allclose(native[extras], robot.default_joint_pos[extras])
    assert profile.manifest()['proprioception_size'] == 93


def test_g1_measurements_use_native_encoder_offsets_and_velocities(settings):
    arena = Arena(config=load_config())
    profile = MorphMimicProfile('g1', settings)
    for side in ('a', 'b'):
        elbow = arena.model.joint(f'{side}_left_elbow').id
        arena.data.qvel[arena.model.jnt_dofadr[elbow]] = .23
        mujoco.mj_forward(arena.model, arena.data)
        measured = profile.capture(arena, side)
        names = profile.robot.native_joint_names
        assert np.allclose(measured['canonical_q'], [arena.full_pose(side)[n] for n in names])
        assert measured['proprioception'].shape == (93,)
        assert np.isfinite(measured['proprioception']).all()
        # The elbow encoder offset is already restored by Arena.full_pose.
        index = names.index('left_elbow_joint')
        raw = arena.data.qpos[arena.model.jnt_qposadr[elbow]]
        assert measured['canonical_q'][index] == pytest.approx(raw + np.pi / 2)
        assert measured['canonical_dq'][index] == pytest.approx(.23)
    with pytest.raises(ValueError, match='only G1'):
        MorphMimicProfile('t2', settings).capture(arena, 'a')


def test_replay_exports_aligned_previous_action_diagnostics(tmp_path, settings):
    output = tmp_path / 'canonical.npz'
    result = replay_motion(PROJECT_ROOT / 'assets/motions/boxing-jab-example.json',
                           load_config(), transitions=output, morphmimic=True)
    with np.load(output) as data:
        count = len(data['qpos'])
        assert data['canonical_q'].shape == data['canonical_action'].shape == (count, 29)
        assert data['proprioception'].shape == (count, 93)
        assert np.allclose(data['next_proprioception'][:, -29:], data['canonical_action'])
        assert np.allclose(data['proprioception'][1:, -29:], data['canonical_action'][:-1])
        metadata = json.loads(str(data['morphmimic_metadata']))
        assert metadata['upstream_revision'] == settings['revision']
        assert metadata['descriptor_available'] is False
        assert 'not an upstream training dataset' in metadata['scope']
    assert result['morphmimic']['robot'] == 'g1'


def test_launcher_expands_paths_and_redirects_upstream_defaults(settings):
    command = integration.build_stage(settings, 'train', ['--', '--robot', 'g1',
        '--subset-yaml', '{data}/g1/subsets/1h.yaml'])
    assert command[:3] == [str(settings['training_python']), '-m', 'morphmimic.scripts.train']
    assert str(settings['data'] / 'g1/subsets/1h.yaml') in command
    assert command[-2:] == ['--log-dir', str(settings['output'] / 'training')]
    converted = integration.build_stage(settings, 'convert', ['--robot', 'x2'])
    assert converted[-2:] == ['--out-dir', str(settings['data'] / 'x2')]
    with pytest.raises(ValueError, match='placeholder'):
        integration.build_stage(settings, 'train', ['{typo}'])


def test_simulator_preview_and_missing_artifacts_fail_before_launch(settings, monkeypatch, capsys):
    monkeypatch.setattr(sys, 'argv', ['morphmimic', 'simulate', '--robot', 'x2',
                                     '--clip', 'test-clip', '--dry-run'])
    assert integration.main() == 0
    assert '-m deploy.run' in capsys.readouterr().out
    monkeypatch.setattr(sys, 'argv', ['morphmimic', 'simulate', '--clip', 'test-clip'])
    with pytest.raises(SystemExit, match='2'):
        integration.main()


def test_pin_mismatch_and_invalid_action_are_rejected(settings):
    with pytest.raises(ValueError, match='pinned revision'):
        integration.require_checkout({**settings, 'revision': '0' * 40})
    profile = MorphMimicProfile('g1', settings)
    with pytest.raises(ValueError, match='29 finite'):
        profile.native_targets(np.full(29, np.nan))
