from __future__ import annotations

import math
import numpy as np

from omni.isaac.lab.utils import configclass
# from omni.isaac.lab.assets import RigidObjectCfg
# import omni.isaac.lab.sim as sim_utils
# from omni.isaac.lab.sensors import CameraCfg, ContactSensorCfg
import isaac_neuromeka.mdp as mdp

##
# Pre-defined configs
##
from isaac_neuromeka.assets import INDY7_CFG
from isaac_neuromeka.tasks.manipulation.reach.reach_env_cfg import ReachEnvCfg, ObservationsCfg
from isaac_neuromeka.tasks.manipulation.reach.reach_env_cfg import TeacherObsCfg
from isaac_neuromeka.tasks.manipulation.reach.reach_env_cfg import CostsCfg
from isaac_neuromeka.mdp.actions import CustomJointPositionAction
from isaac_neuromeka.utils.etc import EmptyCfg

##
# Environment configuration
##


@configclass
class Indy7ReachEnvCfg(ReachEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # switch robot to indy
        self.scene.robot = INDY7_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # override rewards
        self.rewards.end_effector_position_tracking.params["asset_cfg"].body_names = ["tcp"]
        self.rewards.end_effector_orientation_tracking.params["asset_cfg"].body_names = ["tcp"]
        # self.rewards.end_effector_pose_tracking.params["asset_cfg"].body_names = ["tcp"]
        self.rewards.end_effector_speed.params["asset_cfg"].body_names = ["tcp"]
        # override actions
        self.actions.arm_action = mdp.JointPositionActionCfg(
            class_type=CustomJointPositionAction,
            asset_name="robot", joint_names=["joint[0-5]"], scale=0.2, use_default_offset=True
        )
        # override command generator body
        # end-effector is along z-direction
        self.commands.ee_pose.body_name = "tcp"
        self.commands.ee_pose.ranges.pitch = (math.pi, math.pi)
        # self.observations.policy.enable_corruption = False



@configclass
class Indy7ReachTeacherEnvCfg(Indy7ReachEnvCfg):
    observations = TeacherObsCfg()
        
    actor_obs_list: list = ["proprioception", "privileged"]
    critic_obs_list: list | None = None # None: same as actor_obs_list
    teacher_obs_list: list | None = None # unused. No teacher for teacher

@configclass
class Indy7ReachStudentEnvCfg(Indy7ReachTeacherEnvCfg):
    actor_obs_list: list = ["proprioception"]
    teacher_obs_list: list = ["proprioception", "privileged"]

@configclass
class Indy7ReachCMDPEnvCfg(Indy7ReachEnvCfg):
    observations = TeacherObsCfg()
    costs = CostsCfg()
        
    actor_obs_list: list = ["proprioception", "privileged"]
    critic_obs_list: list | None = None 
    teacher_obs_list: list | None = None 

    def __post_init__(self):
        super().__post_init__()
        # self.rewards.joint_vel.weight *= 0.1

