import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import yaml

from shadowbox.integrations.backend import detect_backend, load_backend, run_backend
from shadowbox.integrations.playground import DEFAULT_CONFIG, PlaygroundBackend


def config_file(tmp_path, **changes):
    value = yaml.safe_load(DEFAULT_CONFIG.read_text())
    value['playground'].update(changes)
    path = tmp_path / 'playground.yaml'
    path.write_text(yaml.safe_dump(value))
    return path


def test_pinned_commands_and_lazy_backend(tmp_path):
    backend = load_backend('playground')
    assert detect_backend(DEFAULT_CONFIG) == 'playground'
    command = backend.command('train', tmp_path)
    assert command[1].endswith('/playground/launch.py')
    assert command[2] == 'train'
    assert '--env=ShadowBoxG1Defense' in command
    overrides = json.loads(next(a.split('=', 1)[1] for a in command if a.startswith('--overrides=')))
    assert overrides['nworld'] == 64
    assert 'trajectory_path' not in next(a for a in command if a.startswith('--env_overrides='))
    cpu = PlaygroundBackend(config_file(tmp_path, device='cpu', num_envs=1, task='CartpoleBalance', trajectory=None))
    assert '"impl": "cpp"' in ' '.join(cpu.command('smoke', tmp_path))
    assert 'trajectory_path' not in ' '.join(cpu.command('train', tmp_path))


@pytest.mark.parametrize('changes', [dict(num_envs=True), dict(seed=-1), dict(device='auto'),
                                    dict(task='G1JoystickFlatTerrain'), dict(revision='main'),
                                    dict(output_dir='/tmp/outside-shadowbox')])
def test_invalid_configuration(tmp_path, changes):
    with pytest.raises(ValueError):
        PlaygroundBackend(config_file(tmp_path, **changes))


def test_missing_runtime_fails_without_fallback(tmp_path):
    backend = PlaygroundBackend(config_file(tmp_path, python=str(tmp_path / 'missing')))
    assert not backend.probe()['ready']
    with pytest.raises(ValueError, match='not installed'):
        run_backend(backend, 'smoke')
    assert run_backend(backend, 'smoke', dry_run=True) == 0


def test_generic_runner_records_failure_and_timeout(tmp_path):
    class FakeBackend:
        name = 'test'
        output_dir = tmp_path
        timeout_seconds = 1
        code = 'import sys; print("failure log"); sys.exit(7)'

        def probe(self):
            return {'ready': True}

        def provenance(self):
            return {'revision': 'test'}

        def command(self, stage, run_dir):
            return [sys.executable, '-c', self.code]

    backend = FakeBackend()
    assert run_backend(backend, 'smoke') == 7
    folder = next(tmp_path.iterdir())
    assert json.loads((folder / 'run.json').read_text())['status'] == 'failed'
    assert 'failure log' in (folder / 'runtime.log').read_text()
    backend.code = 'import time; time.sleep(5)'
    assert run_backend(backend, 'smoke') == 124
    assert any(json.loads(p.read_text())['status'] == 'timeout' for p in tmp_path.glob('*/run.json'))


def test_main_routes_playground_without_loading_arena(monkeypatch):
    from shadowbox import app
    from shadowbox.integrations import backend
    calls = []
    monkeypatch.setattr(sys, 'argv', ['shadowbox', '--backend', 'playground', '--backend-dry-run'])
    monkeypatch.setattr(backend, 'run_backend', lambda b, stage, dry: calls.append((b.name, stage, dry)) or 0)
    assert app.main() == 0
    assert calls == [('playground', 'smoke', True)]


def test_main_rejects_incompatible_arena_flags(monkeypatch):
    from shadowbox import app
    monkeypatch.setattr(sys, 'argv', ['shadowbox', '--backend', 'playground', '--defense-only'])
    with pytest.raises(SystemExit) as error:
        app.main()
    assert error.value.code == 2


@pytest.mark.parametrize('failure', ['wrong-source', 'no-gpu', 'missing-motion', None])
def test_runtime_probe_requires_matching_source_gpu_and_motion(tmp_path, monkeypatch, failure):
    from shadowbox.integrations import playground
    motion = tmp_path / 'motion.npz'
    if failure != 'missing-motion':
        motion.write_bytes(b'trajectory placeholder; probe checks existence only')
    backend = PlaygroundBackend(config_file(tmp_path, python=sys.executable, task='G1WholeBodyTracking', trajectory=str(motion)))
    payload = dict(prefix=str(tmp_path / 'isolated'), base_prefix='/usr', python=[3, 12],
                   module=str(backend.checkout / 'mujoco_playground/__init__.py'),
                   cuda=True, devices=1)
    if failure == 'wrong-source':
        payload['module'] = '/unrelated/mujoco_playground/__init__.py'
    if failure == 'no-gpu':
        payload['cuda'] = False
    original = playground.subprocess.run

    def probe_command(command, **kwargs):
        if command[0] == str(backend.python):
            return SimpleNamespace(returncode=0, stdout='SHADOWBOX_PROBE=' + json.dumps(payload), stderr='')
        return original(command, **kwargs)

    monkeypatch.setattr(playground.subprocess, 'run', probe_command)
    result = backend.probe()
    assert result['ready'] is (failure is None)


def test_source_archive_contains_task_and_exact_config(tmp_path):
    import hashlib
    import zipfile
    backend = PlaygroundBackend()
    identity = backend.capture_sources(tmp_path)
    archive = tmp_path / identity['path']
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == identity['sha256']
    with zipfile.ZipFile(archive) as bundle:
        assert bundle.read('run-config.yaml') == DEFAULT_CONFIG.read_bytes()
        assert 'src/shadowbox/playground/task.py' in bundle.namelist()


@pytest.mark.parametrize('options', [dict(start_distance=0.1), dict(start_distance=float('nan')),
    dict(episode_length_secs=-1), dict(reset_yaw_radians=.2), dict(reset_lateral_m=float('nan')),
    dict(opponent_enabled='false'), dict(terminate_opponent=1), dict(unknown=0)])
def test_invalid_task_options(tmp_path, options):
    with pytest.raises(ValueError):
        PlaygroundBackend(config_file(tmp_path, task_options=options))


def test_viewer_requires_an_explicit_policy_choice(tmp_path):
    backend = PlaygroundBackend()
    with pytest.raises(ValueError, match='checkpoint'):
        backend.command('play', tmp_path)
    backend.zero_actions = True
    assert '--zero_actions' in backend.command('play', tmp_path)


def test_motion_training_and_checkpoint_commands_keep_categorical_task(tmp_path):
    backend=PlaygroundBackend(config_file(tmp_path, task='ShadowBoxG1Motion', num_envs=1,
        device='cpu', task_options=dict(motion_bundle='tmp/qualified-bundle')))
    train=backend.command('train',tmp_path)
    assert '--env=ShadowBoxG1Motion' in train
    overrides=json.loads(next(arg.split('=',1)[1] for arg in train if arg.startswith('--overrides=')))
    assert overrides['max_iterations']==backend.settings['iterations']
    checkpoint=tmp_path/'categorical.pt'
    checkpoint.touch()
    backend.settings['checkpoint']=str(checkpoint)
    evaluate=backend.command('evaluate',tmp_path)
    assert f'--checkpoint={checkpoint}' in evaluate


@pytest.mark.parametrize('failure',[None,'missing-fk','tampered-script','no-policy-gpu','uncaptured','malformed'])
def test_motion_doctor_rejects_missing_or_unqualified_artifacts(tmp_path,monkeypatch,failure):
    import hashlib
    from shadowbox.integrations import playground
    bundle=tmp_path/'bundle'
    (bundle/'pelvis_fk').mkdir(parents=True)
    for name in ('unified_pipeline.yaml','pelvis_fk/fk.yaml','pelvis_fk/fk.onnx'):
        (bundle/name).write_bytes(b'unit readiness fixture; live contracts are checked by task build')
    script=bundle/'model.ts';script.write_bytes(b'unit readiness fixture')
    qualification=dict(passed=True,qualified_batch_sizes=[2],
        script_sha256=hashlib.sha256(script.read_bytes()).hexdigest(),
        sidecar_sha256=hashlib.sha256((bundle/'unified_pipeline.yaml').read_bytes()).hexdigest(),
        cuda_graph_qualification=dict(passed=failure!='uncaptured'))
    script.with_suffix('.json').write_text(json.dumps([] if failure=='malformed' else qualification))
    if failure=='missing-fk':(bundle/'pelvis_fk/fk.onnx').unlink()
    if failure=='tampered-script':script.write_bytes(b'changed')
    backend=PlaygroundBackend(config_file(tmp_path,python=sys.executable,task='ShadowBoxG1Motion',
        device='cpu',num_envs=1,task_options=dict(motion_bundle=str(bundle),motion_script=str(script),
        motion_runtime='torchscript_graph',motion_device='cuda:0')))
    payload=dict(prefix=str(tmp_path/'isolated'),base_prefix='/usr',python=[3,12],
        module=str(backend.checkout/'mujoco_playground/__init__.py'),cuda=failure!='no-policy-gpu',devices=1)
    original=playground.subprocess.run
    def command(argv,**kwargs):
        if argv[0]==str(backend.python):
            return SimpleNamespace(returncode=0,stdout='SHADOWBOX_PROBE='+json.dumps(payload),stderr='')
        return original(argv,**kwargs)
    monkeypatch.setattr(playground.subprocess,'run',command)
    result=backend.probe()
    assert result['ready'] is (failure is None)
    if failure is None:assert 'task construction' in result['motion_artifact_scope']
