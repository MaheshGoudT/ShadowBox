import json

import numpy as np
import pytest

from shadowbox.boxing import Observation, Parameters, Policy
from shadowbox.config import PROJECT_ROOT, load_config
from shadowbox.evaluation.qlearning import ACTIONS, LEGACY_ACTIONS, TabularPolicy, load_learned_policy, train
from shadowbox.motion import Primitive


def observation(distance=.8, x=0, yaw=0, stopped=False, incoming_body=False, incoming_speed=0.):
    return Observation(np.array([x, 0., 0.]), np.array([x+distance, 0., 0.]), yaw,
                       distance, np.zeros(3), incoming_speed, 2.55-abs(x), 'guard', stopped,
                       incoming_body)


def test_td_update_uses_real_accumulated_return_and_time_discount():
    policy = TabularPolicy(learning=True, settings=dict(alpha=.5, discount_per_second=.5))
    policy.table = {'old': [0.] * len(ACTIONS), 'next': [2.] * len(ACTIONS)}
    policy.previous = ('old', 0)
    policy.add_reward(1., 2.)
    policy.update('next', [0, 1])
    assert policy.table['old'][0] == pytest.approx(.75)
    assert policy.previous is None and policy.updates == 1


def test_mask_turns_before_attacking_and_respects_defense_and_boundaries():
    policy = TabularPolicy(defense_only=True)
    assert policy.allowed(observation(yaw=np.pi)) == [1]
    assert not {6, 7} & set(policy.allowed(observation()))
    allowed = policy.allowed(observation(x=2.54))
    assert 2 not in allowed
    assert policy.allowed(observation(stopped=True)) == [0]


def test_g1_punch_mask_uses_measured_reach():
    config = load_config(PROJECT_ROOT / 'configs/unitree.yaml')
    policy = TabularPolicy(parameters=Parameters(desired_distance=.58),
                           rules=config['policy_rules'], motion=config['motion'],
                           minimum_separation=config['safety']['min_separation'])
    assert {6, 7} <= set(policy.allowed(observation(distance=.60)))
    assert 9 not in policy.allowed(observation(distance=.60))
    assert 9 in policy.allowed(observation(distance=.50))
    assert 6 in policy.allowed(observation(distance=.70))
    assert 7 not in policy.allowed(observation(distance=.70))
    assert not {6, 7} & set(policy.allowed(observation(distance=.80)))


def test_body_guard_is_available_only_for_near_incoming_body_shots():
    policy = TabularPolicy()
    assert 8 in policy.allowed(observation(incoming_body=True, incoming_speed=.5))
    assert 8 not in policy.allowed(observation(incoming_body=False, incoming_speed=.5))
    assert 8 not in policy.allowed(observation(incoming_body=True, incoming_speed=0.))


def test_untrained_reactive_uppercut_suggestion_does_not_break_tabular_policy(monkeypatch):
    monkeypatch.setattr(Policy, 'choose', lambda self, observation: Primitive.UPPERCUT)
    policy = TabularPolicy()
    assert policy.choose(observation()) in ACTIONS


def test_terminal_update_does_not_bootstrap():
    policy = TabularPolicy(learning=True, settings=dict(alpha=1.))
    policy.choose(observation())
    key, action = policy.previous
    policy.add_reward(2., .1)
    policy.finish()
    assert policy.table[key][action] == 2.


def test_saved_policy_roundtrip_and_unpromoted_fallback(tmp_path):
    path = tmp_path / 'policy.json'
    value = dict(kind='tabular_q', schema_version=2, actions=[a.value for a in ACTIONS], parameters=vars(Parameters()), settings={},
                 table={'2,0,0,0,1': [0.] * len(ACTIONS)}, promoted=False)
    path.write_text(json.dumps(value))
    assert type(load_learned_policy(path)) is Policy
    loaded = load_learned_policy(path, allow_unpromoted=True)
    assert isinstance(loaded, TabularPolicy)
    assert loaded.table == value['table']
    value['table']['broken'] = [float('nan')] * len(ACTIONS)
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        load_learned_policy(path, allow_unpromoted=True)


def test_short_training_repeats_updates_and_uses_held_out_seeds(tmp_path):
    settings = dict(episodes=4, seconds=4., snapshot_every=2, validation_seeds=[30000],
                    output=str(tmp_path / 'learned.json'))
    first = train(load_config(), settings)
    second = train(load_config(), settings)
    assert first['table'] == second['table']
    assert first['history'] == second['history']
    assert sum(e['updates'] for e in first['history']) > 0
    assert not set(first['training_seeds']) & set(first['validation_seeds'])
    assert len(first['candidate']['matches']) == 6
    assert 'self_play' in first and 'activity_gate_passed' in first
    assert (tmp_path / 'learned_report.json').is_file()
    assert all(m['finished'] for m in first['candidate']['matches'])
    if not first['promoted']:
        assert type(load_learned_policy(settings['output'])) is Policy


def test_legacy_promoted_table_without_activity_evidence_falls_back(tmp_path):
    path = tmp_path / 'legacy.json'
    path.write_text(json.dumps(dict(kind='tabular_q', schema_version=1,
                                    actions=[a.value for a in LEGACY_ACTIONS],
                                    parameters=vars(Parameters()), settings={},
                                    table={}, promoted=True)))
    assert type(load_learned_policy(path)) is Policy


def test_legacy_policy_rows_migrate_for_explicit_inspection_but_default_falls_back(tmp_path):
    path = tmp_path / 'legacy.json'
    path.write_text(json.dumps(dict(kind='tabular_q', schema_version=1,
                                    actions=[a.value for a in LEGACY_ACTIONS],
                                    parameters=vars(Parameters()), settings={},
                                    table={'state': [0.2] * 8}, promoted=True,
                                    activity_gate_passed=True)))
    assert type(load_learned_policy(path)) is Policy
    inspected = load_learned_policy(path, allow_unpromoted=True)
    assert isinstance(inspected, TabularPolicy)
    assert inspected.table['state'] == [0.2] * 8 + [0., 0.]


@pytest.mark.parametrize('settings', [dict(episodes=1.5), dict(snapshot_every=0),
                                       dict(alpha=float('nan')), dict(epsilon_start=1.2)])
def test_invalid_learning_settings_fail_before_rollouts(settings):
    with pytest.raises(ValueError):
        train(settings=settings)


def test_resume_preserves_values_history_and_uses_new_training_seeds(tmp_path):
    output = tmp_path / 'first.json'
    settings = dict(episodes=2, seconds=3., snapshot_every=1, validation_seeds=[30000], output=str(output))
    first = train(settings=settings)
    checkpoint = tmp_path / 'first_checkpoints' / 'episode_000002.json'
    saved = json.loads(checkpoint.read_text())
    assert saved['table'] and not saved['promoted'] and saved['completed_episodes'] == 2
    prior = json.loads(output.read_text())
    retained_state = '5,1,1,1,2'
    prior['table'][retained_state] = [0.123] * len(ACTIONS)
    output.write_text(json.dumps(prior))
    resumed = train(settings={**settings, 'resume': str(output), 'output': str(tmp_path / 'second.json')})
    assert resumed['episode_offset'] == 2 and resumed['completed_episodes'] == 4
    assert resumed['training_seeds'] == [0, 1, 2, 3]
    assert resumed['history'][:2] == first['history']
    assert set(first['table']) <= set(resumed['table'])
    assert resumed['table'][retained_state] == [0.123] * len(ACTIONS)
    assert any(e['updates'] for e in resumed['history'][2:])
    assert resumed['resumed_from'] == str(output)
    assert resumed['previous_evaluations'][0]['candidate'] == first['candidate']
    assert (tmp_path / 'second_checkpoints' / 'episode_000004.json').is_file()
    assert isinstance(load_learned_policy(checkpoint, allow_unpromoted=True), TabularPolicy)


def test_resume_rejects_changed_parameters_and_seed_overlap(tmp_path):
    output = tmp_path / 'first.json'
    settings = dict(episodes=2, seconds=2., snapshot_every=1, validation_seeds=[30000], output=str(output))
    train(settings=settings)
    config = load_config()
    config['policies']['a']['reaction'] = .5
    with pytest.raises(ValueError, match='parameters differ'):
        train(config, {**settings, 'resume': str(output)})
    with pytest.raises(ValueError, match='separate'):
        train(settings={**settings, 'resume': str(output), 'validation_seeds': [3]})
