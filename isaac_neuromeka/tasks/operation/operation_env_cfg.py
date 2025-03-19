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
class OperationSceneCfg(InteractiveSceneCfg):
    """Configuration for the scene with a robotic arm."""

    # world
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )

    # robots
    robot: ArticulationCfg = None
    
    # target object
    obstacle = None

    # contact sensor
    # contact_sensors = ContactSensorCfg(
    #         prim_path="{ENV_REGEX_NS}/Robot/link[2-6]",
    #         update_period=0.0, debug_vis=False, track_pose=True, track_air_time=False,
    #     )
    
    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=2500.0),
    )

##
# Environment configuration
##

@configclass
class OperationEnvCfg(NrmkRLEnvCfg): 
    """Configuration for the reach end-effector pose tracking environment."""

    # Scene settings
    scene: OperationSceneCfg = OperationSceneCfg(num_envs=4096, env_spacing=3.0)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg | EmptyCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg | EmptyCfg = EventCfg()
    curriculum = EmptyCfg() # Not used for now
    # CMDP settings
    costs: CostsCfg | EmptyCfg = EmptyCfg() # Not used for now
    
    # 
    actor_obs_list: list = ["policy"] # ["proprioception", "point_cloud", "privileged"]
    critic_obs_list: list | None = None # None: same as actor_obs_list
    teacher_obs_list: list | None = None # None: same as actor_obs_list


    def __post_init__(self):
        """Post initialization."""
        # task settings
        self.decimation = 24  # 5hz # 30 Hz (4)
        self.episode_length_s = 8.0
        # viewer settings
        self.viewer.eye = (2.5, 2.5, 2.5)
