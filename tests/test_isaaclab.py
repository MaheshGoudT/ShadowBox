import json
from pathlib import Path

import pytest
import yaml

from shadowbox.integrations import isaaclab


def test_separate_config_and_command():
    settings = isaaclab.load_isaaclab_config()
    assert isaaclab.detect_backend(isaaclab.DEFAULT_CONFIG) == 'isaaclab'
    assert isaaclab.detect_backend(None) is None
    assert settings['python'].name == 'python'
    assert 'tmp/isaaclab/venv/bin/python' in str(settings['python'])
    assert isaaclab.build_command(settings, headless=True)[1:] == [
        'zero_agent', '--task', 'Isaac-Velocity-Flat-G1', '--num_envs', '1',
        '--max_steps', '200', '--visualizer', 'none', 'physics=newton_mjwarp']


def test_config_rejects_invalid_task_and_output(tmp_path):
    data = yaml.safe_load(isaaclab.DEFAULT_CONFIG.read_text())
    config = tmp_path / 'config.yaml'
    data['isaaclab']['task'] = 'bad; command'
    config.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match='registered task'):
        isaaclab.load_isaaclab_config(config)
    data['isaaclab']['task'] = 'Isaac-Velocity-Flat-G1'
    data['isaaclab']['output_dir'] = '/tmp/outside-shadowbox'
    config.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match='inside ShadowBox tmp'):
        isaaclab.load_isaaclab_config(config)


def test_missing_environment_never_falls_back(tmp_path):
    settings = isaaclab.load_isaaclab_config()
    settings['python'] = tmp_path / 'missing-python'
    settings['executable'] = tmp_path / 'missing-isaaclab'
    assert not isaaclab.probe(settings)['ready']
    with pytest.raises(ValueError, match='not installed'):
        isaaclab.run_backend(settings)
    assert isaaclab.run_backend(settings, dry_run=True) == 0


def test_runner_records_actual_subprocess_exit(tmp_path, monkeypatch):
    executable = tmp_path / 'fake-cli'
    executable.write_text('#!/bin/sh\nprintf "plumbing test: %s\\n" "$*"\nexit 7\n')
    executable.chmod(0o755)
    settings = isaaclab.load_isaaclab_config()
    settings['executable'] = executable
    settings['output_dir'] = tmp_path / 'runs'
    monkeypatch.setattr(isaaclab, 'probe', lambda _: {'ready': True})
    assert isaaclab.run_backend(settings) == 7
    run_dir = next(settings['output_dir'].iterdir())
    result = json.loads((run_dir / 'run.json').read_text())
    assert result['status'] == 'failed' and result['exit_code'] == 7
    assert 'zero_agent' in (run_dir / 'simulation.log').read_text()
