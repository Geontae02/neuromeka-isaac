from __future__ import annotations

import math
import numpy as np

from isaaclab.utils import configclass
import isaac_neuromeka.mdp as mdp

##
# Pre-defined configs
##
from isaac_neuromeka.assets import MOBY_CFG  # Moby 로봇 설정으로 변경
from isaac_neuromeka.tasks.navigation.navigation_env_cfg import NavigationEnvCfg, ObservationsCfg
from isaac_neuromeka.tasks.navigation.navigation_env_cfg import TeacherObsCfg
from isaac_neuromeka.tasks.navigation.navigation_env_cfg import CostsCfg
#from isaac_neuromeka.mdp.actions import CustomVelocityAction
from isaac_neuromeka.mdp.actions import CustomJointPositionAction
from isaac_neuromeka.utils.etc import EmptyCfg

##
# Environment configuration
##

# @configclass
# class MobyNavigationEnvCfg(NavigationEnvCfg):
#     def __post_init__(self):
#         # post init of parent
#         super().__post_init__()

#         # switch robot to Moby
#         self.scene.robot = MOBY_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
#         # override rewards (예: 주행 속도, 목표 도달)
#         self.rewards.robot_speed_tracking.params["asset_cfg"].body_names = ["base_link"]
#         self.rewards.goal_reaching.params["asset_cfg"].body_names = ["base_link"]
        
#         # override actions (이동 방식 설정)
#         self.actions.motion_action = mdp.VelocityActionCfg(
#             class_type=CustomVelocityAction,
#             asset_name="robot", joint_names=["fl_tract_joint", "fr_tract_joint", "rl_tract_joint", "rr_tract_joint"],
#             scale=1.0, use_default_offset=True
#         )
        
#         # override command generator body (목표 위치 설정)
#         self.commands.goal_pose.body_name = "base_link"

#########################################################################
# @configclass
# class MobyNavigationEnvCfg(NavigationEnvCfg):
#     def __post_init__(self):
#         # post init of parent
#         super().__post_init__()

#         # switch robot to indy
#         self.scene.robot = MOBY_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
#         # override rewards
#         self.rewards.end_effector_position_tracking.params["asset_cfg"].body_names = ["base_link"]
#         self.rewards.end_effector_orientation_tracking.params["asset_cfg"].body_names = ["base_link"]
#         # self.rewards.end_effector_pose_tracking.params["asset_cfg"].body_names = ["tcp"]
#         self.rewards.end_effector_speed.params["asset_cfg"].body_names = ["base_link"]
#         # override actions
#         self.actions.arm_action = mdp.JointPositionActionCfg(
#             class_type=CustomJointPositionAction,
#             asset_name="robot", joint_names=["fl_tract_joint", "fr_tract_joint", "rl_tract_joint", "rr_tract_joint"], scale=0.5, use_default_offset=True
#         )
#         # override command generator body
#         # end-effector is along z-direction
#         self.commands.ee_pose.body_name = "base_link"
##############################################################

@configclass
class MobyNavigationEnvCfg(NavigationEnvCfg):
    def __post_init__(self):
        # 부모 클래스 초기화
        super().__post_init__()

        # Moby 로봇 설정 적용
        self.scene.robot = MOBY_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # 행동(Action) 설정 - Joint Position Control 사용
        self.actions.motion_action = mdp.JointPositionActionCfg(
            class_type=CustomJointPositionAction,
            asset_name="robot", 
            joint_names=["fl_tract_joint", "fr_tract_joint", "rl_tract_joint", "rr_tract_joint"],
            scale=0.5, 
            use_default_offset=True
        )

        # 보상(Reward) 설정
        self.rewards.robot_speed_tracking = mdp.RewardTermCfg(
            func=mdp.robot_speed_tracking,  # 속도 추적 보상
            weight=1.0,
            params={"asset_cfg": mdp.SceneEntityCfg("robot", body_names=["base_link"])}
        )
        
        self.rewards.goal_reaching = mdp.RewardTermCfg(
            func=mdp.goal_reaching,  # 목표 지점 도달 보상
            weight=5.0,
            params={"asset_cfg": mdp.SceneEntityCfg("robot", body_names=["base_link"])}
        )


        # 명령(Command) 설정
        self.commands.goal_pose.body_name = "base_link"

################################################################
# @configclass
# class MobyNavigationEnvCfg(NavigationEnvCfg):
#     def __post_init__(self):
#         # post init of parent
#         super().__post_init__()

#         # switch robot to Moby
#         self.scene.robot = MOBY_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        
#         # override rewards (예: 주행 속도, 목표 도달)
#         self.rewards.robot_speed_tracking.params["asset_cfg"].body_names = ["base_link"]
#         self.rewards.goal_reaching.params["asset_cfg"].body_names = ["base_link"]

#         # override actions (이동 방식 설정)
#         self.actions.motion_action = mdp.JointPositionActionCfg(
#             class_type=CustomJointPositionAction,
#             asset_name="robot",
#             joint_names=["fl_tract_joint", "fr_tract_joint", "rl_tract_joint", "rr_tract_joint"],
#             scale=0.1,  # 작은 값으로 설정 (각도 변화가 급격하지 않도록)
#             use_default_offset=True
#         )
        
#         # override command generator body (목표 위치 설정)
#         self.commands.goal_pose.body_name = "base_link"

# @configclass
# class MobyNavigationTeacherEnvCfg(MobyNavigationEnvCfg):
#     observations = TeacherObsCfg()
    
#     actor_obs_list: list = ["proprioception", "lidar", "privileged"]
#     critic_obs_list: list | None = None # None: same as actor_obs_list
#     teacher_obs_list: list | None = None # unused. No teacher for teacher

# @configclass
# class MobyNavigationStudentEnvCfg(MobyNavigationTeacherEnvCfg):
#     actor_obs_list: list = ["proprioception", "lidar"]
#     teacher_obs_list: list = ["proprioception", "lidar", "privileged"]

# @configclass
# class MobyNavigationCMDPEnvCfg(MobyNavigationEnvCfg):
#     observations = TeacherObsCfg()
#     costs = CostsCfg()
    
#     actor_obs_list: list = ["proprioception", "lidar", "privileged"]
#     critic_obs_list: list | None = None 
#     teacher_obs_list: list | None = None 

#     def __post_init__(self):
#         super().__post_init__()

################################################
        # 추가적인 비용 또는 보상 조정 가능


# @configclass
# class ActionsCfg:
#     """Action specifications for the MDP."""

#     arm_action: ActionTerm = mdp.JointPositionActionCfg(
#             class_type=CustomJointPositionAction,
#             asset_name="robot", joint_names=[".*_arm_.*"], scale=0.5, use_default_offset=True
#         )
#     base_action: ActionTerm = mdp.JointPositionActionCfg(
#             class_type=CustomJointPositionAction,
#             asset_name="robot", joint_names=["base_joint"], scale=0.5, use_default_offset=True
#         )


# @configclass
# class MobyNavigationEnvCfg(NavigationEnvConfg):
#     action = ActionsCfg()
#     observation = ObservationCfg()
