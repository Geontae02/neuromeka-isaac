from __future__ import annotations

import pdb
from typing import TYPE_CHECKING

import torch
from omni.isaac.lab.managers import SceneEntityCfg


if TYPE_CHECKING:
    from omni.isaac.lab.envs import RLTaskEnv
    from isaac_neuromeka.env.rl_task_custom_env import CustomRLTaskEnv

from isaac_neuromeka.assets.articulation import FiniteArticulation


def finite_joint_vel(
        env: RLTaskEnv,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: FiniteArticulation = env.scene[asset_cfg.name]
    return asset._finite_joint_vel[:, asset_cfg.joint_ids]

def joint_pos_history(
        env: RLTaskEnv,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: FiniteArticulation = env.scene[asset_cfg.name]
    return torch.cat((asset._prevprev_joint_pos[:, asset_cfg.joint_ids], asset._prev_joint_pos[:, asset_cfg.joint_ids]), dim=-1)

def joint_vel_history(
        env: RLTaskEnv,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: FiniteArticulation = env.scene[asset_cfg.name]
    return torch.cat((asset._prevprev_finite_joint_vel[:, asset_cfg.joint_ids], asset._prev_finite_joint_vel[:, asset_cfg.joint_ids]), dim=-1)

def action_history(env: RLTaskEnv) -> torch.Tensor:
    return torch.cat((env.action_manager.prev_action, env.action_manager.action), dim=-1)

def position_error(env: RLTaskEnv) -> torch.Tensor:
    return env.command_manager.get_term("ee_pose").metrics["position_error"].unsqueeze(-1)

def orientation_error(env: RLTaskEnv) -> torch.Tensor:
    return env.command_manager.get_term("ee_pose").metrics["orientation_error"].unsqueeze(-1)

def op_state(
        env: RLTaskEnv,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: FiniteArticulation = env.scene[asset_cfg.name]
    return asset._op_state


from omni.isaac.lab.utils.math import subtract_frame_transforms, matrix_from_quat

def body_pose_b(
        env: RLTaskEnv,
        body_name: str,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: FiniteArticulation = env.scene[asset_cfg.name]
    body_idx = asset.find_bodies(body_name)[0][0]
    body_pose_w = asset.data.body_state_w[:, body_idx, :7]
    body_pos_b, body_quat_b = subtract_frame_transforms(
        asset.data.root_pos_w,   # w -> r
        asset.data.root_quat_w,
        body_pose_w[:, :3],  # w -> b
        body_pose_w[:, 3:]
    )  # r -> b
    body_pose_b = torch.cat((body_pos_b, body_quat_b), dim=-1)  # position + orientation (quaternion)
    return body_pose_b

def body_vel_b(
        env: RLTaskEnv,
        body_name: str,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.tensor:
    asset: FiniteArticulation = env.scene[asset_cfg.name]
    body_idx = asset.find_bodies(body_name)[0][0]
    body_vel_w = asset.data.body_state_w[:, body_idx, 7:]
    body_vel_b = torch.zeros_like(body_vel_w)
    R_BW = torch.transpose(matrix_from_quat(asset.data.root_quat_w), 1, 2)
    body_vel_b[:, :3] = torch.einsum("bij, bj->bi", R_BW, body_vel_w[:, :3])
    body_vel_b[:, 3:] = torch.einsum("bij, bj->bi", R_BW, body_vel_w[:, 3:])
    return body_vel_b


"""
Privileged information.
"""

def joint_friction(env: CustomRLTaskEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """The joint friction of the asset.

    Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their friction returned.
    """
    # extract the used quantities (to enable type-hinting)
    asset: FiniteArticulation = env.scene[asset_cfg.name]
    return asset.data.joint_friction[:, asset_cfg.joint_ids]

def joint_damping(env: CustomRLTaskEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """The joint damping of the asset.

    Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their damping returned.
    """
    # extract the used quantities (to enable type-hinting)
    asset: FiniteArticulation = env.scene[asset_cfg.name]
    return asset.data.joint_damping[:, asset_cfg.joint_ids]

def action_delay_steps(env: CustomRLTaskEnv) -> torch.Tensor:
    
    if hasattr(env, "delay_steps"):
        return env.delay_steps.reshape(-1, 1)
    else:
        return  torch.zeros((env.num_envs, 1), device=env.device, dtype=torch.long)
