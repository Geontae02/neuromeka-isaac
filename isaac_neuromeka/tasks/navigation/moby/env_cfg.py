@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    arm_action: ActionTerm = mdp.JointPositionActionCfg(
            class_type=CustomJointPositionAction,
            asset_name="robot", joint_names=[".*_arm_.*"], scale=0.5, use_default_offset=True
        )
    base_action: ActionTerm = mdp.JointPositionActionCfg(
            class_type=CustomJointPositionAction,
            asset_name="robot", joint_names=["base_joint"], scale=0.5, use_default_offset=True
        )


@configclass
class MobyNavigationEnvCfg(NavigationEnvConfg):
    action = ActionsCfg()
    observation = ObservationCfg()
