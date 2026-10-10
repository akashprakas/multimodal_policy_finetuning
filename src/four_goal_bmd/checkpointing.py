from __future__ import annotations

from pathlib import Path

import torch
from omegaconf import DictConfig, OmegaConf

from four_goal_bmd.algorithms.bmd import BMDTrainingResult
from four_goal_bmd.factory import build_diffusion_model, build_frozen_policy
from four_goal_bmd.models.steering import InferenceModel, LatentCritic, SteeringPolicy


def save_bc_checkpoint(path: Path, frozen_policy, cfg: DictConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 1,
            "config": OmegaConf.to_container(cfg, resolve=True),
            "model_state": frozen_policy.model.state_dict(),
            "action_scale": frozen_policy.action_scale,
        },
        path,
    )


def save_flow_bc_checkpoint(path: Path, model, action_scale: float, cfg: DictConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 1,
            "backend": "reinflow",
            "config": OmegaConf.to_container(cfg, resolve=True),
            "model_state": model.state_dict(),
            "action_scale": action_scale,
        },
        path,
    )


def save_reinflow_checkpoint(
    path: Path, result, action_scale: float, cfg: DictConfig, training_seed: int
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 1,
            "backend": "reinflow",
            "config": OmegaConf.to_container(cfg, resolve=True),
            "training_seed": training_seed,
            "action_scale": action_scale,
            "actor_state": result.actor.state_dict(),
            "critic_state": result.critic.state_dict(),
        },
        path,
    )


def save_training_checkpoint(
    path: Path,
    frozen_policy,
    result: BMDTrainingResult,
    cfg: DictConfig,
    training_seed: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 1,
            "config": OmegaConf.to_container(cfg, resolve=True),
            "training_seed": training_seed,
            "bc_model_state": frozen_policy.model.state_dict(),
            "action_scale": frozen_policy.action_scale,
            "steering_state": result.steering_policy.state_dict(),
            "critic_state": result.critic.state_dict(),
            "inference_state": result.inference_model.state_dict(),
        },
        path,
    )


def load_training_checkpoint(path: str | Path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("backend") == "reinflow":
        raise ValueError(f"{path} is a ReinFlow checkpoint.")
    cfg = OmegaConf.create(payload["config"])
    model = build_diffusion_model(cfg)
    model.load_state_dict(payload["bc_model_state"])
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    frozen_policy = build_frozen_policy(model, payload["action_scale"], cfg)

    steering_policy = SteeringPolicy(
        observation_dim=cfg.environment.observation_dim,
        k_latent=cfg.experiment.k_latent,
        noise_dim=frozen_policy.noise_dim,
        hidden_dims=cfg.experiment.steering_dims,
        w_bound=cfg.experiment.w_bound,
        min_log_std=cfg.experiment.min_log_std,
    )
    critic = LatentCritic(
        cfg.environment.observation_dim,
        cfg.experiment.k_latent,
        cfg.experiment.critic_dims,
    )
    inference_model = InferenceModel(
        cfg.environment.observation_dim,
        cfg.experiment.k_latent,
        cfg.experiment.inference_dims,
    )
    steering_policy.load_state_dict(payload["steering_state"])
    critic.load_state_dict(payload["critic_state"])
    inference_model.load_state_dict(payload["inference_state"])
    return cfg, frozen_policy, steering_policy, critic, inference_model, payload


def load_reinflow_checkpoint(path: str | Path):
    from four_goal_bmd.algorithms.reinflow import build_actor_from_bc
    from four_goal_bmd.factory import build_flow_model

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("backend") != "reinflow":
        raise ValueError(f"{path} is not a ReinFlow checkpoint.")
    cfg = OmegaConf.create(payload["config"])
    actor = build_actor_from_bc(
        build_flow_model(cfg),
        cfg.experiment,
        condition_dim=cfg.environment.observation_dim,
        time_embedding_dim=cfg.policy.time_embedding_dim,
    )
    actor.load_state_dict(payload["actor_state"])
    critic = LatentCritic(
        cfg.environment.observation_dim, 1, cfg.experiment.critic_dims
    )
    critic.load_state_dict(payload["critic_state"])
    return cfg, actor, critic, payload["action_scale"], payload
