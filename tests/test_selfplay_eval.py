import numpy as np
import pytest

from shadowbox.playground.selfplay_eval import (
    contact_totals, evaluation_guard_width, summarize_cases,
    transition_action_counts)


def test_contact_totals_map_candidate_and_opponent_by_controlled_side():
    report={'contact_episode_counts_by_channel_and_glove':[
        [1,0],[0,2],[3,0],[4,0],[0,5],[0,6]]}
    assert contact_totals(report,'a')==dict(candidate_head=4,candidate_body=5,
        opponent_head=1,opponent_body=2,candidate_blocks=3,
        candidate_attacks_blocked=6)
    assert contact_totals(report,'b')==dict(candidate_head=1,candidate_body=2,
        opponent_head=4,opponent_body=5,candidate_blocks=6,
        candidate_attacks_blocked=3)


def test_contact_totals_accept_empty_reports_without_claiming_contacts():
    assert contact_totals({},'a')==dict(candidate_head=0,candidate_body=0,
        opponent_head=0,opponent_body=0,candidate_blocks=0,
        candidate_attacks_blocked=0)


def test_transition_counts_distinguish_continue_actions_from_executed_goals():
    transitions={
        'action':np.array([[[6.]],[[9.]],[[9.]],[[7.]]]),
        'requested_primitives':np.array([[[6.,0.]],[[6.,0.]],[[6.,0.]],[[7.,0.]]]),
    }
    assert transition_action_counts(transitions,'a')==(
        {'6':1,'7':1,'9':2},{'6':3,'7':1})


def test_held_out_guard_width_can_target_a_measured_curriculum_stage():
    settings={'self_play':{'opponent_guard_width_end':.1},
              'task_options':{'opponent_guard_width':.45}}
    assert evaluation_guard_width(settings)==.1
    assert evaluation_guard_width(settings,.12)==.12
    with pytest.raises(ValueError,match='guard width'):
        evaluation_guard_width(settings,.08)


def test_held_out_summary_aggregates_completed_cases_by_opponent():
    cases=[
        dict(opponent='scripted',status='completed',
             candidate_contacts=dict(candidate_head=2,candidate_body=1,
                 opponent_head=0,opponent_body=3,candidate_blocks=4,
                 candidate_attacks_blocked=1),
             sampled_policy_actions={'7':2,'9':3},executed_skill_goals={'7':4},
             completed_episodes=1,censored_episode_steps=[8,5]),
        dict(opponent='scripted',status='completed',
             candidate_contacts=dict(candidate_head=3,candidate_body=0,
                 opponent_head=1,opponent_body=2,candidate_blocks=2,
                 candidate_attacks_blocked=0),
             sampled_policy_actions={'7':1},executed_skill_goals={'7':2},
             completed_episodes=0,censored_episode_steps=[5]),
        dict(opponent='scripted',status='failed'),
    ]
    summary=summarize_cases(cases)['scripted']
    assert summary['cases']==3
    assert summary['completed_cases']==2
    assert summary['failed_cases']==1
    assert summary['candidate_contacts']==dict(
        candidate_head=5,candidate_body=1,opponent_head=1,opponent_body=5,
        candidate_blocks=6,candidate_attacks_blocked=1,net_head_contacts=4,
        net_body_contacts=-4,net_total_contacts=0)
    assert summary['sampled_policy_actions']=={'7':3,'9':3}
    assert summary['executed_skill_goals']=={'7':6}
    assert summary['completed_episodes']==1
    assert summary['censored_episode_steps']==18
