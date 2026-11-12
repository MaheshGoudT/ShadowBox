"""Register local tasks without changing the pinned upstream checkout."""


from dataclasses import dataclass


@dataclass
class DefensiveDistributionCfg:
    class_name: str = "shadowbox.playground.primitive_distribution:DefensivePrimitiveDistribution"


@dataclass
class BoxingDistributionCfg:
    class_name: str = "shadowbox.playground.primitive_distribution:BoxingPrimitiveDistribution"


def register_tasks():
    from mujoco_playground._src.rl import env_registry
    from mujoco_playground._src.rl.rsl_rl.env_rl_config import G1WholeBodyTrackingRslRlOnPolicyRunnerCfg
    from shadowbox.playground.task import TASK_ID, boxing_env
    try:
        env_registry.get(TASK_ID)
    except ValueError:
        runner = G1WholeBodyTrackingRslRlOnPolicyRunnerCfg()
        runner.actor.hidden_dims = (128, 128)
        runner.critic.hidden_dims = (128, 128)
        runner.save_interval = 10
        runner.actor.distribution_cfg.init_std = 0.3
        env_registry.register(TASK_ID, runner, boxing_env)

    def motion_env(**kwargs):
        if not kwargs.get('motion_bundle'):
            raise ValueError('ShadowBoxG1Motion requires a qualified exported motion bundle')
        return boxing_env(**kwargs)

    try:
        env_registry.get('ShadowBoxG1Motion')
    except ValueError:
        motion_runner=G1WholeBodyTrackingRslRlOnPolicyRunnerCfg()
        motion_runner.actor.hidden_dims=(128,128)
        motion_runner.critic.hidden_dims=(128,128)
        motion_runner.actor.distribution_cfg=DefensiveDistributionCfg()
        motion_runner.torch_compile_mode=None
        motion_runner.save_interval=10
        env_registry.register('ShadowBoxG1Motion',motion_runner,motion_env)

    def selfplay_env(**kwargs):
        if not kwargs.get('motion_bundle'):
            raise ValueError('ShadowBoxG1SelfPlay requires a qualified exported motion bundle')
        kwargs['self_play']=True
        return boxing_env(**kwargs)

    try:
        env_registry.get('ShadowBoxG1SelfPlay')
    except ValueError:
        selfplay_runner=G1WholeBodyTrackingRslRlOnPolicyRunnerCfg()
        selfplay_runner.actor.hidden_dims=(128,128)
        selfplay_runner.critic.hidden_dims=(128,128)
        selfplay_runner.actor.distribution_cfg=BoxingDistributionCfg()
        selfplay_runner.torch_compile_mode=None
        selfplay_runner.save_interval=10
        env_registry.register('ShadowBoxG1SelfPlay',selfplay_runner,selfplay_env)
