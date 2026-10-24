"""Launcher contract tests; never import the optional upstream training stack."""
import json
import sys
from pathlib import Path

import pytest

from shadowbox.integrations import holoboxing as h


def invoke(monkeypatch, *args):
    monkeypatch.setattr(sys, 'argv', ['holoboxing', *args])
    return h.main()


def test_dry_run_expands_config_without_runtime_probe(monkeypatch, capsys):
    monkeypatch.setattr(h, 'revision', lambda _: h.load_settings(h.DEFAULT_CONFIG)['revision'])
    monkeypatch.setattr(h, 'probe', lambda _: pytest.fail('dry run probed runtime'))
    assert invoke(monkeypatch, 'run', '--dry-run', 'supervised', '--', '--data', '{data}', '--device', '{device}') == 0
    output = capsys.readouterr().out
    assert 'holoboxing.training.train_sl' in output
    assert str(h.ROOT / 'tmp/holoboxing/data/processed/combined/prepared') in output
    assert 'cuda:0' in output


def test_evaluator_uses_configured_image_and_output(monkeypatch, capsys):
    monkeypatch.setattr(h, 'revision', lambda _: h.load_settings(h.DEFAULT_CONFIG)['revision'])
    assert invoke(monkeypatch, 'validate', 'tmp/policy.onnx', '--benchmark', '--dry-run') == 0
    output = capsys.readouterr().out
    assert 'holoboxing/run.sh --validate' in output
    assert '--image horizonrobotics/holomotion:holoboxing-eval-v12-amd64' in output
    assert str(h.ROOT / 'tmp/holoboxing/results/validation.json') in output
    assert '--benchmark' in output


def test_mismatched_revision_blocks_launch(monkeypatch):
    monkeypatch.setattr(h, 'revision', lambda _: 'wrong')
    with pytest.raises(SystemExit, match='2'):
        invoke(monkeypatch, 'run', '--dry-run', 'export')


def test_unsupported_host_blocks_execution(monkeypatch):
    monkeypatch.setattr(h, 'revision', lambda _: h.load_settings(h.DEFAULT_CONFIG)['revision'])
    monkeypatch.setattr(h.platform, 'machine', lambda: 'aarch64')
    with pytest.raises(SystemExit, match='2'):
        invoke(monkeypatch, 'run', 'rl')


def test_doctor_reports_not_ready(monkeypatch, capsys):
    monkeypatch.setattr(h, 'probe', lambda _: {'ready': False, 'error': 'absent'})
    assert invoke(monkeypatch, 'doctor') == 1
    assert json.loads(capsys.readouterr().out)['ready'] is False


def test_missing_config_fields_rejected(tmp_path):
    config = tmp_path / 'config.yaml'
    config.write_text('repository: external/HoloMotion\n')
    with pytest.raises(ValueError, match='Missing'):
        h.load_settings(config)


def test_retarget_launch_uses_separate_interpreter_and_preserves_exit(monkeypatch):
    settings = h.load_settings(h.DEFAULT_CONFIG)
    monkeypatch.setattr(h, 'revision', lambda _: settings['revision'])
    monkeypatch.setattr(h.platform, 'machine', lambda: 'x86_64')
    monkeypatch.setattr(h.platform, 'system', lambda: 'Linux')
    monkeypatch.setattr(h, 'probe', lambda _: {'ready': True})
    calls = []

    def execute(command, **kwargs):
        calls.append((command, kwargs))
        return type('Result', (), {'returncode': 7})()

    monkeypatch.setattr(h.subprocess, 'run', execute)
    assert invoke(monkeypatch, 'run', 'retarget', '--', '--output', '{output}') == 7
    command, kwargs = calls[0]
    assert command[:4] == [str(settings['retarget_python']), '-m', 'holoboxing.data', 'retarget']
    assert kwargs['cwd'] == settings['repository']
    assert kwargs['env']['HOLOBOXING_LOWER'] == str(settings['lower'])
