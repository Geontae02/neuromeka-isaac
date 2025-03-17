from __future__ import annotations

import pdb
from dataclasses import MISSING

import numpy as np
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.managers import ActionTermCfg as ActionTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import ManagerTermBase
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaac_neuromeka.utils.etc import EmptyCfg
import isaac_neuromeka.mdp as mdp
from isaaclab.sensors import ContactSensor, ContactSensorCfg, FrameTransformer, FrameTransformerCfg

# Import common environment configuration
from isaac_neuromeka.tasks.manipulation.common.env_cfg_common import *

##
# Scene definition
##
@configclass
class NavigationSceneCfg(InteractiveSceneCfg):
    """Configuration for the scene with a mobile robot."""

    # world
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )

    # 모바일 로봇 추가
    robot: ArticulationCfg = None  # 여기에 Moby 로봇의 설정을 적용해야 함
    
    # 장애물 설정 (예: 정적 장애물 추가 가능)
    obstacle = None

    # 센서 설정 (예: 모바일 로봇에 필요한 라이다 센서 추가 가능)
    contact_sensors = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_link",  # 모바일 로봇의 기본 링크를 감지 대상으로 설정
            update_period=0.0, debug_vis=False, track_pose=True, track_air_time=False,
        )
    
    # 조명 설정
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=2500.0),
    )

# @configclass
# class ReachSceneCfg(InteractiveSceneCfg):
#     """Configuration for the scene with a robotic arm."""

#     # world
#     ground = AssetBaseCfg(
#         prim_path="/World/ground",
#         spawn=sim_utils.GroundPlaneCfg(),
#         init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
#     )

#     # table = AssetBaseCfg(
#     #     prim_path="{ENV_REGEX_NS}/Table",
#     #     spawn=sim_utils.UsdFileCfg(
#     #         usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd",
#     #     ),
#     #     init_state=AssetBaseCfg.InitialStateCfg(pos=(0.55, 0.0, 0.0), rot=(0.70711, 0.0, 0.0, 0.70711)),
#     # )

#     # robots
#     robot: ArticulationCfg = None
    
#     # target object
#     obstacle = None

#     # contact sensor
#     contact_sensors = ContactSensorCfg(
#             prim_path="{ENV_REGEX_NS}/Robot/link[2-6]",
#             update_period=0.0, debug_vis=False, track_pose=True, track_air_time=False,
#         )
    
#     # lights
#     light = AssetBaseCfg(
#         prim_path="/World/light",
#         spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=2500.0),
#     )

##
# Environment configuration
##


@configclass
class NavigationEnvCfg(NrmkRLEnvCfg): 
    """Configuration for the navigation task of the mobile robot."""

    # Scene settings
    scene: NavigationSceneCfg = NavigationSceneCfg(num_envs=4096, env_spacing=3.0)
    
    # 기본 설정
    observations: ObservationsCfg = ObservationsCfg() 
    actions: ActionsCfg = ActionsCfg()  
    commands: CommandsCfg = CommandsCfg()  
    
    # MDP 설정
    rewards: RewardsCfg | EmptyCfg = RewardsCfg()  
    terminations: TerminationsCfg = TerminationsCfg() 
    events: EventCfg | EmptyCfg = EventCfg()
    curriculum = EmptyCfg()  # 미사용
    
    # CMDP 설정 (안전 관련 비용 설정, 미사용)
    costs: CostsCfg | EmptyCfg = EmptyCfg()
    
    # 관찰 목록 설정 (예: LiDAR, 속도, 위치 정보 등 포함 가능)
    #actor_obs_list: list = ["proprioception", "lidar"]  # 로봇의 자기 감각과 라이다 센서 정보 포함
    critic_obs_list: list | None = None  # None: actor_obs_list와 동일
    teacher_obs_list: list | None = None  # None: actor_obs_list와 동일

    def __post_init__(self):
        """Post initialization."""
        # task settings
        self.decimation = 10  # 10 Hz 주행 주파수
        self.episode_length_s = 20.0  # 에피소드 길이를 20초로 설정
        
        # viewer settings (카메라 시점 변경 가능)
        self.viewer.eye = (5.0, 5.0, 3.0)  # 더 넓은 시야 확보

        
# class NavigationEnvCfg
# @configclass
# class ReachEnvCfg(NrmkRLEnvCfg): 
#     """Configuration for the reach end-effector pose tracking environment."""

#     #NavigationSceneCfg
#     # Scene settings
#     scene: ReachSceneCfg = ReachSceneCfg(num_envs=4096, env_spacing=3.0)
#     # Basic settings
#     observations: ObservationsCfg = ObservationsCfg()
#     actions: ActionsCfg = ActionsCfg()
#     commands: CommandsCfg = CommandsCfg()
#     # MDP settings
#     rewards: RewardsCfg | EmptyCfg = RewardsCfg()
#     terminations: TerminationsCfg = TerminationsCfg()
#     events: EventCfg | EmptyCfg = EventCfg()
#     curriculum = EmptyCfg() # Not used for now
#     # CMDP settings
#     costs: CostsCfg | EmptyCfg = EmptyCfg() # Not used for now
    
#     # 
#     actor_obs_list: list = ["policy"] # ["proprioception", "point_cloud", "privileged"]
#     critic_obs_list: list | None = None # None: same as actor_obs_list
#     teacher_obs_list: list | None = None # None: same as actor_obs_list


#     def __post_init__(self):
#         """Post initialization."""
#         # task settings
#         self.decimation = 24  # 5hz # 30 Hz (4)
#         self.episode_length_s = 8.0
#         # viewer settings
#         self.viewer.eye = (2.5, 2.5, 2.5)
