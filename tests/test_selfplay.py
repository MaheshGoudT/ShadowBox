import pytest
import numpy as np
import torch

pytest.importorskip('mujoco_playground')
pytest.importorskip('reaf')

from shadowbox.playground.motion_commands import (
    TrackedBoxingCommands, apply_jab_arm_targets, forward_approach_progress,
    guard_widths_by_side, queue_primitive, resolve_primitive)
from shadowbox.playground import selfplay_train
from shadowbox.playground.selfplay_train import (
    both_sides_touched, controlled_side_for_round, curriculum_guard_step,
    next_approach_start_distance, next_guard_width, training_entropy_coef)
from shadowbox.playground.task import competitive_reward, player_observation


def features(n=2):
    out={}
    for index,side in enumerate(('a','b')):
        out[f'{side}/joint_pos']=torch.full((n,29),float(index+1))
        out[f'{side}/joint_vel']=torch.full((n,29),float(index+3))
        out[f'{side}_root/xquat']=torch.zeros(n,4)
        out[f'{side}_root/xquat'][:,0]=1
        out[f'{side}_root/joint_vel']=torch.full((n,6),float(index+5))
        out[f'{side}_root/xpos']=torch.zeros(n,3)
        out[f'{side}_root/xpos'][:,0]=float(index)
        out[f'{side}_root/xpos'][:,2]=.76
    out['b_root/xquat'][:,0]=0
    out['b_root/xquat'][:,3]=1
    out['boxing/events']=torch.zeros(n,6,2)
    out['motion/a_primitive']=torch.full((n,),6.)
    out['motion/b_primitive']=torch.full((n,),8.)
    out['motion/a_phase']=torch.full((n,),.5)
    out['motion/b_phase']=torch.full((n,),1.)
    return out


def test_player_observation_swaps_own_and_opponent_state():
    f=features()
    a=player_observation(f,'a')
    b=player_observation(f,'b')
    assert a.shape==b.shape==(2,149)
    torch.testing.assert_close(a[:,:29],f['a/joint_pos'])
    torch.testing.assert_close(b[:,:29],f['b/joint_pos'])
    torch.testing.assert_close(a[:,68:71],b[:,68:71])
    torch.testing.assert_close(a[:,58:62],b[:,58:62])
    torch.testing.assert_close(a[:,71:75],b[:,71:75])
    torch.testing.assert_close(a[:,81:110],b[:,:29])
    assert a[:,139:148].argmax(-1).tolist()==[6,6]
    assert b[:,139:148].argmax(-1).tolist()==[8,8]
    torch.testing.assert_close(a[:,148],torch.full((2,),.25))
    torch.testing.assert_close(b[:,148],torch.full((2,),.5))
    with pytest.raises(ValueError,match='side'):
        player_observation(f,'c')


def test_continue_action_preserves_the_active_skill():
    assert resolve_primitive(9,6)==6
    assert resolve_primitive(9,8)==8
    assert resolve_primitive(7,6)==6
    assert resolve_primitive(9,0)==0
    assert resolve_primitive(2,6,.35)==6  # Punch phases cannot be preempted.
    assert resolve_primitive(2,6,.7)==0   # Completed punch returns to guard.
    assert resolve_primitive(2,7,1.09)==7
    assert resolve_primitive(2,7,1.1)==0
    with pytest.raises(ValueError,match='CONTINUE'):
        resolve_primitive(10,0)


def test_straight_jab_blends_calibrated_arm_pose_only_during_jab():
    targets=np.zeros((4,29),dtype=np.float32)
    indices=[15,16,17,18]
    arm_target=[-1.54810,-0.01048,0.04930,1.49]
    result=apply_jab_arm_targets(targets,[6,6,6,7],[0.,.35,.7,.35],indices,arm_target)
    np.testing.assert_array_equal(result[0],targets[0])
    np.testing.assert_allclose(result[1,indices],arm_target,atol=1e-6)
    np.testing.assert_array_equal(result[2],targets[2])
    np.testing.assert_array_equal(result[3],targets[3])
    with pytest.raises(ValueError,match='jab arm targets'):
        apply_jab_arm_targets(targets,[6,0],[float('nan'),0.],indices,arm_target)


def test_new_punch_is_queued_until_the_active_punch_finishes():
    resolved,pending=queue_primitive(7,6,.2)
    assert (resolved,pending)==(6,7)
    assert queue_primitive(6,resolved,.4,pending)==(6,7)
    assert queue_primitive(6,resolved,.7,pending)==(7,0)
    assert queue_primitive(9,6,.2,7)==(6,7)
    assert queue_primitive(1,6,.2,7)==(6,7)
    assert queue_primitive(1,6,.7,7)==(7,0)
    assert queue_primitive(4,6,.2,7)==(6,0)
    assert queue_primitive(4,6,.7,7)==(7,0)
    assert queue_primitive(0,6,.2,7)==(6,0)
    assert queue_primitive(8,6,.2,7)==(6,0)
    with pytest.raises(ValueError,match='phase'):
        queue_primitive(7,6,float('nan'))


def test_approach_sampled_during_a_punch_runs_after_the_punch_finishes():
    resolved,pending=queue_primitive(2,6,.2)
    assert (resolved,pending)==(6,2)
    assert queue_primitive(0,6,.4,pending)==(6,2)
    assert queue_primitive(9,6,.5,pending)==(6,2)
    assert queue_primitive(0,6,.7,pending)==(2,0)
    # A new explicit punch may replace the queued approach.
    assert queue_primitive(7,6,.4,2)==(6,7)
    with pytest.raises(ValueError,match='Pending action'):
        queue_primitive(2,6,.2,4)


def test_approach_holds_long_enough_for_footwork_then_releases_to_next_skill():
    assert resolve_primitive(0,2,.2)==2
    assert resolve_primitive(0,2,.4)==0
    resolved,pending=queue_primitive(0,2,.2)
    assert (resolved,pending)==(2,0)
    assert queue_primitive(6,2,.3,pending)==(2,6)
    assert queue_primitive(0,2,.35,6)==(2,6)
    assert queue_primitive(0,2,.4,6)==(6,0)
    assert queue_primitive(0,2,.19,approach_duration=.2)==(2,0)
    assert queue_primitive(0,2,.2,approach_duration=.2)==(0,0)


def test_guard_widths_follow_controlled_role_when_sides_swap():
    assert guard_widths_by_side('a',.1,.45)==(.1,.45)
    assert guard_widths_by_side('b',.1,.45)==(.45,.1)
    with pytest.raises(ValueError,match='Controlled side'):
        guard_widths_by_side('c',.1,.45)


def test_approach_progress_credits_only_own_forward_motion_until_target_range():
    previous=torch.tensor([[0.,0.],[.9,0.]]).numpy()
    current=torch.tensor([[.02,.2],[.85,0.]]).numpy()
    np.testing.assert_allclose(
        forward_approach_progress(previous,current,[2,0],.65),[.02,0.],atol=1e-6)
    np.testing.assert_allclose(
        forward_approach_progress(previous,current,[0,2],.65),[0.,.05],atol=1e-6)
    # The opponent moving closer does not give side A progress credit.
    np.testing.assert_allclose(
        forward_approach_progress(previous,[[0,0],[.85,0]],[2,0],.65),[0.,0.],atol=1e-6)
    np.testing.assert_allclose(
        forward_approach_progress([[0,0],[.66,0]],[[.04,0],[.66,0]],[2,0],.65),
        [.01,0.],atol=1e-6)
    with pytest.raises(ValueError,match='positions'):
        forward_approach_progress([[0,0]],[[0,0]],[2,0],.65)


def test_snapshot_pool_also_samples_builtin_scripted_opponent():
    class Actor(torch.nn.Module):
        def forward(self, observation):
            return observation

    actor = Actor()
    motion = object.__new__(TrackedBoxingCommands)
    motion.scripted_opponent_policy = None
    motion.opponent_policy = None
    motion.opponent_pool = []
    motion.opponent_pool_index = 0
    motion.requests = torch.zeros(2, dtype=torch.int64)
    motion.pending_requests = torch.zeros(2, dtype=torch.int64)
    motion.latency = 0.
    motion.set_opponent_pool([actor])
    assert motion.opponent_pool == [None, actor]
    motion.reset()
    assert motion.opponent_policy is None
    motion.reset()
    assert motion.opponent_policy is actor
    motion.set_opponent_pool([actor], start_index=1)
    assert motion.opponent_policy is actor
    motion.set_opponent_pool([actor], start_index=2)
    assert motion.opponent_pool_index == 0
    motion.reset()
    assert motion.opponent_policy is None
    with pytest.raises(ValueError, match='start index'):
        motion.set_opponent_pool([actor], start_index=True)


def test_guard_curriculum_advances_only_after_a_legal_touch():
    assert next_guard_width(.45,.333,0,0)==.45
    assert next_guard_width(.45,.333,1,0)==.333


def test_approach_distance_curriculum_advances_only_after_both_sides_touch():
    assert next_approach_start_distance(.65,.9,.05,False)==.65
    assert next_approach_start_distance(.65,.9,.05,True)==pytest.approx(.70)
    assert next_approach_start_distance(.89,.9,.05,True)==pytest.approx(.9)
    with pytest.raises(ValueError,match='curriculum'):
        next_approach_start_distance(.9,.65,.05,True)
    assert next_guard_width(.333,.216,0,2)==.216


def test_guard_curriculum_requires_a_touch_from_both_controlled_sides():
    a={'legal_head_contacts':1,'legal_body_contacts':0}
    b={'legal_head_contacts':0,'legal_body_contacts':1}
    assert not both_sides_touched({'a':a})
    assert not both_sides_touched({'a':a,'b':{'legal_head_contacts':0,'legal_body_contacts':0}})
    assert both_sides_touched({'a':a,'b':b})


def test_guard_curriculum_can_keep_a_fixed_step_while_retrying():
    assert curriculum_guard_step(.12,.10,10)==pytest.approx(.02/9)
    assert curriculum_guard_step(.12,.10,10,.01)==.01
    with pytest.raises(ValueError,match='guard-width step'):
        curriculum_guard_step(.12,.10,10,.5)


def test_resume_side_alternation_starts_opposite_the_selected_checkpoint():
    assert controlled_side_for_round(0,0,alternate=True)=='a'
    assert controlled_side_for_round(1,1,alternate=True)=='b'
    assert controlled_side_for_round(0,14,alternate=True,previous_side='a')=='b'
    assert controlled_side_for_round(1,15,alternate=True,previous_side='a')=='a'
    assert controlled_side_for_round(0,15,alternate=True,previous_side='b')=='a'
    assert controlled_side_for_round(0,14,alternate=False,configured_side='a',
                                    previous_side='a')=='a'


def test_resume_run_selects_latest_qualified_candidate_and_pool_snapshots(tmp_path,monkeypatch):
    import hashlib
    import json

    project=tmp_path/'project'; project.mkdir()
    monkeypatch.setattr(selfplay_train,'PROJECT_ROOT',project)
    run=project/'tmp'/'league-run'; round_records=[]
    for index in range(2):
        round_dir=run/f'round-{index:03d}'; round_dir.mkdir(parents=True)
        checkpoint=round_dir/'candidate.pt'; checkpoint.write_bytes(f'candidate-{index}'.encode())
        snapshot=round_dir/'snapshot'/'opponent.pt'; snapshot.parent.mkdir()
        snapshot.write_bytes(f'snapshot-{index}'.encode())
        round_records.append(dict(
            round=index,
            controlled_side='a' if index==0 else 'b',
            checkpoint=str(checkpoint.relative_to(project)),
            checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            next_opponent_guard_width=.15 if index==1 else .2,
            curriculum_gate_passed=index==0,
            curriculum_evaluation={'by_side':{
                'a':{'legal_head_contacts':int(index==0),'legal_body_contacts':0},
                'b':{'legal_head_contacts':0,'legal_body_contacts':int(index==0)}}}))
    (run/'league.json').write_text(json.dumps({'rounds':round_records}))

    state=selfplay_train._resume_run_state(run,opponent_pool_size=1)
    assert state['checkpoint']==run/'round-000'/'candidate.pt'
    assert state['snapshots']==[(0,run/'round-000'/'snapshot'/'opponent.pt')]
    assert state['next_guard_width']==.2
    assert state['selected_round']==0
    assert state['selected_controlled_side']=='a'
    assert state['prior_round_count']==2


def test_league_checkpoint_is_atomic_and_records_latest_qualification(tmp_path,monkeypatch):
    import json

    project=tmp_path/'project'; project.mkdir()
    monkeypatch.setattr(selfplay_train,'PROJECT_ROOT',project)
    run=project/'tmp'/'league-run'; run.mkdir(parents=True)
    checkpoint=run/'round-000'/'candidate.pt'
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b'candidate')
    record={'round':0,'checkpoint':str(checkpoint.relative_to(project))}
    selfplay_train._write_league(
        run,'ShadowBoxG1SelfPlay',[record],None,checkpoint,.12,1)

    league=json.loads((run/'league.json').read_text())
    assert league['rounds']==[record]
    assert league['last_qualified_checkpoint']==str(checkpoint.relative_to(project))
    assert league['last_qualified_next_guard_width']==.12
    assert not (run/'league.json.tmp').exists()


def test_player_observation_is_invariant_to_world_yaw_rotation():
    f=features(1)
    f['a_root/xpos'][0]=torch.tensor([-.5,0,.76])
    f['b_root/xpos'][0]=torch.tensor([.5,0,.76])
    f['a_root/joint_vel'][0]=torch.tensor([.4,.1,0,.01,.02,.03])
    f['b_root/joint_vel'][0]=torch.tensor([-.4,-.1,0,.04,.05,.06])
    rotated={key:value.clone() for key,value in f.items()}
    angle=torch.tensor(.73)
    cosine,sine=torch.cos(angle),torch.sin(angle)
    for side in ('a','b'):
        position=rotated[f'{side}_root/xpos'][0]
        position[:2]=torch.stack((cosine*position[0]-sine*position[1],
                                  sine*position[0]+cosine*position[1]))
        quaternion=rotated[f'{side}_root/xquat'][0]
        half=angle/2
        cz,sz=torch.cos(half),torch.sin(half)
        w,x,y,z=quaternion
        rotated[f'{side}_root/xquat'][0]=torch.stack((cz*w-sz*z,cz*x-sz*y,
                                                       cz*y+sz*x,cz*z+sz*w))
        velocity=rotated[f'{side}_root/joint_vel'][0]
        vx,vy=velocity[:2].clone()
        velocity[:2]=torch.stack((cosine*vx-sine*vy,sine*vx+cosine*vy))
    torch.testing.assert_close(player_observation(f,'a'),
                               player_observation(rotated,'a'),atol=1e-6,rtol=1e-6)


def test_competitive_contact_and_fall_rewards_are_zero_sum():
    f=features(1)
    f['boxing/events'][0,3,0]=1  # A lands a legal head contact on B.
    reward_a=competitive_reward(f,'a')
    reward_b=competitive_reward(f,'b')
    torch.testing.assert_close(reward_a,-reward_b)
    f['boxing/events'].zero_()
    f['boxing/events'][0,2,1]=1  # A blocks B's incoming glove.
    torch.testing.assert_close(competitive_reward(f,'a'),-competitive_reward(f,'b'))
    f['boxing/events'].zero_()
    f['a_root/xpos'][0,2]=.3
    torch.testing.assert_close(competitive_reward(f,'a'),-competitive_reward(f,'b'))


def test_body_contact_reward_and_penalty_are_configurable():
    f=features(1)
    f['boxing/events'][0,4,0]=1  # A lands a body contact.
    baseline=competitive_reward(f,'a')
    torch.testing.assert_close(
        competitive_reward(f,'a',body_contact_reward=5.),baseline+3.)
    f['boxing/events'].zero_()
    f['boxing/events'][0,1,0]=1  # A receives a body contact.
    baseline=competitive_reward(f,'a')
    torch.testing.assert_close(
        competitive_reward(f,'a',received_body_penalty=5.),baseline-3.)


def test_approach_progress_reward_only_rewards_positive_closing_motion():
    f=features(1)
    baseline=competitive_reward(f,'a')
    f['motion/a_approach_progress']=torch.tensor([.2])
    torch.testing.assert_close(
        competitive_reward(f,'a',approach_progress_reward=2.),baseline+.4)
    f['motion/a_approach_progress']=torch.tensor([-.2])
    torch.testing.assert_close(
        competitive_reward(f,'a',approach_progress_reward=2.),baseline)


def test_training_entropy_coefficient_is_configurable_and_bounded():
    assert training_entropy_coef(None,.005)==.005
    assert training_entropy_coef(.02,.005)==.02
    for value in (True,-.01,.101,float('nan')):
        with pytest.raises(ValueError,match='entropy_coef'):
            training_entropy_coef(value,.005)
