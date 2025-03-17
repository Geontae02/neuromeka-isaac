import math
import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaac_neuromeka.assets.articulation import FiniteArticulation

##
# Configuration
##

#MOBY_CFG 만들기  /model/usd/moby/moby.usd
MOBY_CFG = ArticulationCfg(
    class_type=FiniteArticulation,
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{os.path.dirname(os.path.abspath(__file__))}/model/usd/moby/moby.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,  # (Indy control framework already includes gravity compensation) 모비는?
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "fl_rot_joint": 0.0,
            "fr_rot_joint": 0.0,
            "rl_rot_joint": 0.0,
            "rr_rot_joint": 0.0,
            "fl_tract_joint": 0.0,
            "fr_tract_joint": 0.0,
            "rl_tract_joint": 0.0,
            "rr_tract_joint": 0.0,
        },
    ),
    actuators={
        "front_left_wheel": ImplicitActuatorCfg(
            joint_names_expr=["fl_tract_joint"],
            velocity_limit=9.424778,  # 3π rad/s
            effort_limit=60.0,
            stiffness=100.0,
            damping=20.0,
        ),
        "front_right_wheel": ImplicitActuatorCfg(
            joint_names_expr=["fr_tract_joint"],
            velocity_limit=9.424778,
            effort_limit=60.0,
            stiffness=100.0,
            damping=20.0,
        ),
        "rear_left_wheel": ImplicitActuatorCfg(
            joint_names_expr=["rl_tract_joint"],
            velocity_limit=9.424778,
            effort_limit=60.0,
            stiffness=100.0,
            damping=20.0,
        ),
        "rear_right_wheel": ImplicitActuatorCfg(
            joint_names_expr=["rr_tract_joint"],
            velocity_limit=9.424778,
            effort_limit=60.0,
            stiffness=100.0,
            damping=20.0,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)

# INDY7_CFG = ArticulationCfg(
#     class_type=FiniteArticulation,
#     spawn=sim_utils.UsdFileCfg(
#         usd_path=f"{os.path.dirname(os.path.abspath(__file__))}/model/usd/indy7/indy7.usd",
#         # usd_path=f"{os.path.dirname(os.path.abspath(__file__))}/model/usd/indy7_simplified/indy7_simplified.usd",
#         activate_contact_sensors=True,
#         rigid_props=sim_utils.RigidBodyPropertiesCfg(
#             disable_gravity=True,  # (Indy control framework already includes gravity compensation)
#             max_depenetration_velocity=5.0,
#         ),
#         articulation_props=sim_utils.ArticulationRootPropertiesCfg(
#             enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=0
#         ),
#     ),
#     init_state=ArticulationCfg.InitialStateCfg(
#         joint_pos={
#             "joint0": 0.0,
#             "joint1": 0.0,
#             "joint2": -1.57079,
#             "joint3": 0.0,
#             "joint4": -1.57079,
#             "joint5": 0.0,
#         },
#     ),
#     actuators={
#         "arm0": ImplicitActuatorCfg(
#             joint_names_expr=["joint[0-1]"],
#             velocity_limit=2.775073510670984,
#             effort_limit=431.97,
#             stiffness=100.0,
#             damping=20.0,
#         ),
#         "arm1": ImplicitActuatorCfg(
#             joint_names_expr=["joint2"],
#             velocity_limit=2.775073510670984,
#             effort_limit=197.23,
#             stiffness=100.0,
#             damping=20.0,
#         ),
#         "arm2": ImplicitActuatorCfg(
#             joint_names_expr=["joint[3-5]"],
#             velocity_limit=3.2986722862692828,
#             effort_limit=79.79,
#             stiffness=100.0,
#             damping=20.0,
#         ),
#     },
#     soft_joint_pos_limit_factor=1.0,
# )


# INDY7_ORBIT_ALLEGRO_CFG = ArticulationCfg(
#     class_type=FiniteArticulation,
#     spawn=sim_utils.UsdFileCfg(
#         usd_path=f"{os.path.dirname(os.path.abspath(__file__))}/model/usd/indy7_orbit_allegro_hand/indy7_orbit_allegro_hand.usd",
#         activate_contact_sensors=False,
#         rigid_props=sim_utils.RigidBodyPropertiesCfg(
#             disable_gravity=True,
#             retain_accelerations=False,
#             enable_gyroscopic_forces=False,
#             angular_damping=0.01,
#             max_linear_velocity=1000.0,
#             max_angular_velocity=64 / math.pi * 180.0,
#             max_depenetration_velocity=1000.0,
#             max_contact_impulse=1e32,
#         ),
#         # disable self-collision
#         articulation_props=sim_utils.ArticulationRootPropertiesCfg(
#             enabled_self_collisions=True,
#             solver_position_iteration_count=8,
#             solver_velocity_iteration_count=0,
#             sleep_threshold=0.005,
#             stabilization_threshold=0.0005,
#         ),
#     ),
#     init_state=ArticulationCfg.InitialStateCfg(
#         joint_pos={
#             # indy7
#             "joint0": 0.0,
#             "joint1": -0.649,
#             "joint2": -2.064,
#             "joint3": 0.0,
#             "joint4": 1.11199,
#             "joint5": 2.356194490192345,
#             # allegro hand
#             "index_joint_[0-3]": 0.0,
#             "middle_joint_[0-3]": 0.0,
#             "ring_joint_[0-3]": 0.0,
#             "thumb_joint_0": 0.28,
#             "thumb_joint_[1-3]": 0.0,
#         },
#     ),
#     actuators={
#         # indy7
#         "arm0": ImplicitActuatorCfg(
#             joint_names_expr=["joint[0-1]"],
#             velocity_limit=2.775073510670984,
#             effort_limit=431.97,
#             stiffness=100.0,
#             damping=20.0,
#         ),
#         "arm1": ImplicitActuatorCfg(
#             joint_names_expr=["joint2"],
#             velocity_limit=2.775073510670984,
#             effort_limit=197.23,
#             stiffness=100.0,
#             damping=20.0,
#         ),
#         "arm2": ImplicitActuatorCfg(
#             joint_names_expr=["joint[3-5]"],
#             velocity_limit=3.2986722862692828,
#             effort_limit=79.79,
#             stiffness=100.0,
#             damping=20.0,
#         ),
#         # allegro hand
#         "fingers": ImplicitActuatorCfg(
#             joint_names_expr=["index_joint_[0-3]", "middle_joint_[0-3]", "ring_joint_[0-3]", "thumb_joint_[0-3]"],
#             effort_limit=0.5,
#             velocity_limit=100.0,
#             stiffness=3.0,
#             damping=0.1,
#             friction=0.01,
#         ),
#     },
#     soft_joint_pos_limit_factor=1.0,
# )