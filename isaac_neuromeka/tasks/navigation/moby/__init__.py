import gymnasium as gym

from . import env_cfg, learning

##
# Register Gym environments.
##

# gym.register(
#     id="Indy-Reach",
#     entry_point="isaac_neuromeka.env.rl_task_custom_env:CustomManagerBasedRLEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": env_cfg.Indy7ReachEnvCfg,
#         "rsl_rl_cfg_entry_point": f"{learning.__name__}.rsl_rl_cfg:ReachPPORunnerCfg",
#         "nrmk_rl_cfg_entry_point": f"{learning.__name__}.nrmk_rl_cfg:ReachPPORunnerCfg",
#     },
# )

gym.register(
    id="Moby-Navigation",
    entry_point="isaac_neuromeka.env.rl_task_custom_env:CustomManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": env_cfg.NavigationEnvCfg,  # 모바일 로봇 네비게이션 환경으로 변경
        "rsl_rl_cfg_entry_point": f"{learning.__name__}.rsl_rl_cfg:NavigationPPORunnerCfg",  # PPO 학습 설정 변경
        "nrmk_rl_cfg_entry_point": f"{learning.__name__}.nrmk_rl_cfg:NavigationPPORunnerCfg",  # NRMK 학습 설정 변경
    },
)