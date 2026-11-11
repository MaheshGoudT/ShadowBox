"""Executable task gates; run with the isolated Playground interpreter.

Core tests skip these when Playground is absent. No mocked simulator results.
"""
import pytest

pytest.importorskip('mujoco_playground')
torch = pytest.importorskip('torch')
import mujoco

from shadowbox.playground.registry import register_tasks
from shadowbox.playground.task import TASK_ID, boxing_env, BoxingCommands, guard_pose


def test_scene_has_independent_free_bases_and_physical_contacts():
    config = boxing_env()
    model = config.base_scene.compile()
    assert model.nq == 72
    assert model.nv == 70
    assert model.nu == 58
    assert (model.jnt_type == mujoco.mjtJoint.mjJNT_FREE).sum() == 2
    assert model.opt.gravity[2] < 0
    assert not model.body_gravcomp.any()
    for side in ('a', 'b'):
        for hand in ('left', 'right'):
            glove = model.geom(f'{side}/{hand}_glove')
            assert glove.contype[0] and glove.conaffinity[0]
    assert model.nsensor >= 4


def test_native_cpu_world_is_finite_and_resettable():
    from mujoco_playground._src.rl import env_registry
    from mujoco_playground._src.rl.rsl_rl.vecenv_wrapper import RslRlVecEnvWrapper
    register_tasks()
    raw = env_registry.get(TASK_ID).env_builder(nworld=1, device='cpu', impl='cpp')
    try:
        env = RslRlVecEnvWrapper(raw)
        assert env.num_actions == 29
        with torch.inference_mode():
            for _ in range(12):
                obs, reward, done, _ = env.step(torch.zeros(1, 29))
                assert obs['policy'].shape == (1, 162)
                assert torch.isfinite(obs['policy']).all()
                assert torch.isfinite(reward).all()
            sim = raw.device_layer.sim
            assert torch.isfinite(sim.data.qpos).all()
            sim.data.qvel[:] = 1
            from gdm_robotics.interfaces import environment
            raw.reset_with_options(options={}, env_mask=torch.tensor([True]))
            assert torch.equal(sim.data.qvel, torch.zeros_like(sim.data.qvel))
    finally:
        raw.close()


def test_opponent_does_not_depend_on_learner_and_partial_reset():
    from mujoco_playground.robots.g1.sim import device as g1
    common = dict(nworld=2, xpu_device='cpu', lower=[-3]*29, upper=[3]*29)
    left, right = BoxingCommands(**common), BoxingCommands(**common)
    for _ in range(20):
        a = left.process_commands({'action': torch.zeros(2, 29)})
        b = right.process_commands({'action': torch.ones(2, 29)})
        assert torch.equal(a['b/ctrl'], b['b/ctrl'])
        assert not torch.equal(a['a/ctrl'], b['a/ctrl'])
    old = left.phase.clone()
    left.reset(torch.tensor([True, False]))
    assert left.phase[0] == 0
    assert left.phase[1] == old[1]
    assert a['b/ctrl'][0, g1.JOINT_NAMES.index('left_elbow_joint')] > guard_pose()[g1.JOINT_NAMES.index('left_elbow_joint')]


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires a real CUDA runtime')
def test_cuda_batch_partial_reset_preserves_other_world():
    from mujoco_playground._src.rl import env_registry
    from mujoco_playground._src.rl.rsl_rl.vecenv_wrapper import RslRlVecEnvWrapper
    register_tasks()
    raw = env_registry.get(TASK_ID).env_builder(nworld=2, device='cuda:0', impl='warp',
        reset_yaw_radians=.1,reset_lateral_m=.03)
    try:
        env = RslRlVecEnvWrapper(raw)
        with torch.inference_mode():
            for _ in range(12):
                obs, reward, _, _ = env.step(torch.zeros(2, 29, device='cuda:0'))
                assert torch.isfinite(reward).all() and torch.isfinite(obs['policy']).all()
            before_q = raw.device_layer.sim.data.qpos.clone()
            before_v = raw.device_layer.sim.data.qvel.clone()
            raw.reset_with_options(options=raw.default_reset_options(),
                                   env_mask=torch.tensor([True, False], device='cuda:0'))
            assert torch.equal(raw.device_layer.sim.data.qpos[1], before_q[1])
            assert torch.equal(raw.device_layer.sim.data.qvel[1], before_v[1])
            assert torch.equal(raw.device_layer.sim.data.qvel[0], torch.zeros_like(before_v[0]))
            model=raw.device_layer.sim.mj_model
            for joint in range(model.njnt):
                if model.jnt_type[joint]!=mujoco.mjtJoint.mjJNT_FREE:continue
                address=int(model.jnt_qposadr[joint])
                root=raw.device_layer.sim.data.qpos[0,address:address+7]
                torch.testing.assert_close(root[3:].norm(),torch.ones_like(root[3:].norm()))
                assert root[4:6].abs().max()==0  # Reset changes yaw, not roll/pitch.
                assert root[1].abs()<=.13
                deviation=2*torch.atan2(root[6].abs(),root[3].abs())
                deviation=torch.minimum(deviation,torch.pi-deviation)
                assert deviation<=.10001
    finally:
        raw.close()


def test_contact_episodes_do_not_count_holds_collapses_or_duplicate_blocks():
    from shadowbox.playground.referee import ContactEpisodes
    referee = ContactEpisodes(nworld=2)
    f = {}
    for key in referee.required_features_keys():
        if key.endswith('force_history'):
            f[key] = torch.zeros(2, 2, 4, 3)
        elif key.endswith('xquat'):
            f[key] = torch.tensor([[1., 0, 0, 0], [1., 0, 0, 0]])
        else:
            f[key] = torch.tensor([[0., 0, .76], [0., 0, .76]])
    referee.on_begin_stepping(f)
    f['incoming_head/force_history'][:, 0, :, 0] = 10
    f['incoming_block_left/force_history'][:, 0, :, 0] = 20
    f['a_root/xpos'][1, 2] = .4  # The collapsed world cannot generate legal touches.
    referee.on_step(f)
    assert referee.counts[0, 0, 0] == 1
    assert referee.counts[0, 2, 0] == 0  # A block cannot erase an actual head contact.
    assert not referee.counts[1].any()
    for _ in range(20):
        referee.on_step(f)
    assert referee.counts[0, 0, 0] == 1
    before = referee.counts[0].clone()
    referee.reset(torch.tensor([False, True]))
    assert torch.equal(referee.counts[0], before)
    for key in f:
        if key.endswith('force_history'):
            f[key].zero_()
    referee.on_step(f)
    f['incoming_block_right/force_history'][0, 1, :, 0] = 20
    referee.on_step(f)
    assert referee.counts[0, 2, 1] == 1


def test_real_head_contact_is_classified_from_mujoco_force():
    from mujoco_playground._src.rl import env_registry
    register_tasks()
    raw = env_registry.get(TASK_ID).env_builder(nworld=1, device='cpu', impl='cpp')
    try:
        raw.reset()
        sim = raw.device_layer.sim
        model = sim.mj_model
        from mujoco_playground.robots.g1.sim.device import JOINT_NAMES
        # Controlled penetration fixture, not evidence of autonomous punch skill.
        values = {'left_shoulder_pitch_joint': -1.54810, 'left_shoulder_roll_joint': -0.01048,
                  'left_shoulder_yaw_joint': 0.04930, 'left_elbow_joint': 1.03735,
                  'waist_pitch_joint': 0.15}
        for name, value in values.items():
            address = int(model.joint('b/'+name).qposadr[0])
            sim.data.qpos[0, address] = value
        raw.device_layer.forward()
        head = sim.data.geom_xpos[0, model.geom('a/head_collision').id].clone()
        glove = sim.data.geom_xpos[0, model.geom('b/left_glove').id].clone()
        sim.data.qpos[0, 36:39] += head + torch.tensor([.11, 0, 0]) - glove
        raw.device_layer.forward()
        raw.step({'action': torch.zeros(1, 29)})
        features = raw._latest_features
        measured = features['incoming_head/force_history'].norm(dim=-1).max()
        assert measured > 5
        assert features['boxing/counts'][0, 0, 0] == 1
    finally:
        raw.close()


def test_action_history_is_the_same_instance_for_command_and_observation():
    from mujoco_playground._src.rl import env_registry
    register_tasks()
    raw=env_registry.get(TASK_ID).env_builder(nworld=1,device='cpu',impl='cpp')
    try:
        raw.reset()
        first=torch.linspace(-.1,.1,29).reshape(1,29)
        second=-first
        raw.step({'action':first})
        torch.testing.assert_close(raw._latest_features['action'],first)
        raw.step({'action':second})
        torch.testing.assert_close(raw._latest_features['action'],second)
        torch.testing.assert_close(raw._latest_features['action_history'][:,0],first)
    finally:
        raw.close()
