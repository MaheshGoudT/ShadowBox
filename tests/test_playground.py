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


def selfplay_config_file(tmp_path, self_play_changes=None, task_changes=None):
    source = Path(__file__).resolve().parents[1] / 'configs/playground-motion-selfplay.yaml'
    value = yaml.safe_load(source.read_text())
    value['playground']['self_play'].update(self_play_changes or {})
    value['playground']['task_options'].update(task_changes or {})
    path = tmp_path / 'selfplay.yaml'
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


@pytest.mark.parametrize('seed', [-1, True, 1.5])
def test_evaluation_seed_must_be_a_nonnegative_integer(tmp_path, seed):
    with pytest.raises(ValueError, match='evaluation_seed'):
        PlaygroundBackend(config_file(tmp_path, evaluation_seed=seed))


@pytest.mark.parametrize('value', [-.26, .26, True])
def test_body_attack_arc_lateral_is_bounded_and_numeric(tmp_path, value):
    options={'body_attack_arc_lateral':value}
    with pytest.raises(ValueError, match='body_attack_arc_lateral'):
        PlaygroundBackend(config_file(tmp_path, task_options=options))


@pytest.mark.parametrize(('key','value'), [
    ('approach_target_range_m', .49),
    ('approach_target_range_m', 1.51),
    ('approach_duration_secs', .09),
    ('approach_duration_secs', 1.51),
    ('approach_progress_reward', -0.1),
    ('approach_progress_reward', 20.1),
])
def test_approach_curriculum_options_are_bounded(tmp_path, key, value):
    with pytest.raises(ValueError, match=key):
        PlaygroundBackend(config_file(tmp_path, task_options={key:value}))


def test_selfplay_start_distance_curriculum_config_is_validated(tmp_path):
    backend = PlaygroundBackend(selfplay_config_file(tmp_path))
    assert backend.settings['self_play']['approach_start_distance'] == .6
    with pytest.raises(ValueError, match='cannot exceed'):
        PlaygroundBackend(selfplay_config_file(
            tmp_path, {'approach_start_distance': .95}))
    path = selfplay_config_file(tmp_path)
    value = yaml.safe_load(path.read_text())
    del value['playground']['self_play']['opponent_guard_width_end']
    path.write_text(yaml.safe_dump(value))
    with pytest.raises(ValueError, match='requires opponent_guard_width_end'):
        PlaygroundBackend(path)


@pytest.mark.parametrize('target', [
    [-1.5, 0.0, 0.0],
    [-1.5, 0.0, 0.0, True],
    [-1.5, 0.0, 0.0, 3.21],
])
def test_straight_jab_arm_target_is_validated(tmp_path, target):
    with pytest.raises(ValueError, match='jab_arm_target'):
        PlaygroundBackend(selfplay_config_file(tmp_path, task_changes={'jab_arm_target':target}))


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
    dict(opponent_enabled='false'), dict(terminate_opponent=1),
    dict(body_contact_reward=21.), dict(approach_speed_a=.41),
    dict(approach_speed_b=-.01), dict(max_punch_reach_m=.29),
    dict(max_punch_reach_m=.71), dict(unknown=0)])
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
    assert '--seed=10000' in evaluate
    backend.settings['evaluation_seed'] = 17
    assert '--seed=17' in backend.command('evaluate',tmp_path)


def test_selfplay_stage_routes_to_iterative_playground_runner(tmp_path):
    backend=PlaygroundBackend(config_file(tmp_path,task='ShadowBoxG1SelfPlay',device='cuda:0',num_envs=1,
        task_options=dict(motion_bundle='tmp/playground/metatron/compiled',
            motion_script='tmp/playground/metatron/compiled/unified_pipeline.cuda.ts',
            motion_runtime='torchscript',motion_device='cuda:0',opponent_enabled=True,
            body_contact_reward=4.),
        self_play=dict(rounds=3,updates_per_round=7,opponent_pool_size=2,entropy_coef=.02)))
    command=backend.command('selfplay',tmp_path)
    assert command[1].endswith('/shadowbox/playground/launch.py')
    assert command[2]=='selfplay'
    assert '--rounds=3' in command
    assert '--updates-per-round=7' in command
    assert '--opponent-pool-size=2' in command
    assert '--rollout-steps=64' in command
    assert '--curriculum-eval-steps=300' in command
    assert '--env=ShadowBoxG1SelfPlay' not in command
    backend.resume_run=tmp_path/'previous-league'
    backend.resume_guard_width=.15
    continued=backend.command('selfplay',tmp_path)
    assert f'--resume-run={backend.resume_run}' in continued
    assert '--resume-guard-width=0.15' in continued


def test_selfplay_settings_cannot_be_attached_to_a_different_task(tmp_path):
    with pytest.raises(ValueError,match='require ShadowBoxG1SelfPlay'):
        PlaygroundBackend(config_file(tmp_path,self_play=dict(rounds=2)))


def test_selfplay_guard_width_step_is_bounded(tmp_path):
    with pytest.raises(ValueError,match='opponent_guard_width_step'):
        PlaygroundBackend(config_file(tmp_path,task='ShadowBoxG1SelfPlay',num_envs=1,
            task_options=dict(motion_bundle='tmp/qualified-bundle'),
            self_play=dict(opponent_guard_width_step=.5)))


@pytest.mark.parametrize('entropy_coef',[True,-.01,.101,float('nan')])
def test_selfplay_entropy_coefficient_is_bounded(tmp_path,entropy_coef):
    with pytest.raises(ValueError,match='entropy_coef'):
        PlaygroundBackend(config_file(tmp_path,task='ShadowBoxG1SelfPlay',num_envs=1,
            task_options=dict(motion_bundle='tmp/qualified-bundle'),
            self_play=dict(entropy_coef=entropy_coef)))


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
