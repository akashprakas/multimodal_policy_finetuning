from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from omegaconf import DictConfig, open_dict

from four_goal_bmd.anchors import ANCHOR_TYPES


@dataclass
class EnvironmentConfig:
    name: str = "four_goal"
    observation_dim: int = 2
    action_dim: int = 2
    n_goals: int = 4
    goal_radius: float = 1.0
    reward_sigma: float = 0.25
    success_distance: float = 0.30


@dataclass
class PolicyConfig:
    name: str = "single_step"
    chunk_length: int = 1
    episode_steps: int = 14
    denoising_steps: int = 10
    ddim_steps: int = 8
    network_dims: List[int] = field(default_factory=lambda: [32, 32])


@dataclass
class ExperimentConfig:
    name: str = "bmd"
    k_latent: int = 8
    total_epochs: int = 300
    warmup_epochs: int = 120
    intrinsic_scale: float = 2.0
    finetune_intrinsic_scale: float = 0.2


def backend(cfg: DictConfig) -> str:
    """Return the training backend selected by the experiment config."""
    return cfg.experiment.get("backend", "steering_ppo")


def _validate_reinflow(cfg: DictConfig) -> None:
    experiment = cfg.experiment
    if experiment.flow_steps < 1:
        raise ValueError("flow_steps must be positive.")
    if not 0.0 < experiment.sigma_min <= experiment.sigma_max:
        raise ValueError("require 0 < sigma_min <= sigma_max.")
    if experiment.min_sampling_std != experiment.sigma_min:
        raise ValueError("min_sampling_std must equal sigma_min.")
    if experiment.denoised_clip_value != experiment.act_limit:
        raise ValueError("denoised_clip_value must equal act_limit.")
    if not experiment.clip_intermediate_actions:
        raise ValueError("clip_intermediate_actions=false is unsupported.")
    if experiment.k_latent != 1:
        raise ValueError("ReinFlow carries no discrete latent; k_latent must be 1.")
    if experiment.total_epochs > 0 and (
        experiment.critic_warmup_epochs >= experiment.total_epochs
    ):
        raise ValueError("critic_warmup_epochs must be less than total_epochs.")
    if experiment.time_scale is None:
        raise ValueError("apply_run_profile must resolve time_scale before validation.")
    rollout_size = experiment.n_envs * (
        cfg.policy.episode_steps // cfg.policy.chunk_length
    )
    if experiment.ppo_batch_size > rollout_size:
        raise ValueError("ppo_batch_size exceeds the number of rollout samples.")
    if experiment.noise_scheduler not in {"none", "learn_decay"}:
        raise ValueError("noise_scheduler must be 'none' or 'learn_decay'.")
    anchor_type = experiment.get("anchor_type", "none")
    if anchor_type not in ANCHOR_TYPES:
        raise ValueError(f"anchor_type must be one of {ANCHOR_TYPES}.")
    if float(experiment.get("anchor_coef", 0.0)) < 0.0:
        raise ValueError("anchor_coef must be non-negative.")
    if anchor_type != "none" and not experiment.learn_velocity_field:
        raise ValueError("an anchor requires a trainable velocity field.")


def validate_config(cfg: DictConfig) -> None:
    if cfg.environment.name != "four_goal":
        raise ValueError("This reproduction supports only FourGoalEnv.")
    if cfg.environment.observation_dim != 2 or cfg.environment.action_dim != 2:
        raise ValueError("FourGoalEnv requires 2D observations and actions.")
    if cfg.environment.n_goals != 4:
        raise ValueError("This reproduction fixes the evaluation to four goals.")
    if cfg.policy.episode_steps % cfg.policy.chunk_length != 0:
        raise ValueError("episode_steps must be divisible by chunk_length.")
    if cfg.experiment.warmup_epochs > cfg.experiment.total_epochs:
        raise ValueError("warmup_epochs cannot exceed total_epochs.")
    if cfg.experiment.k_latent < 1:
        raise ValueError("k_latent must be positive.")
    if backend(cfg) == "reinflow":
        _validate_reinflow(cfg)
    else:
        if not (1 <= cfg.policy.ddim_steps <= cfg.policy.denoising_steps):
            raise ValueError("ddim_steps must be between 1 and denoising_steps.")
        if cfg.experiment.inference_steps < 1:
            raise ValueError("inference_steps must be positive.")
    shift = cfg.get("shift", None)
    if shift is not None:
        if shift.get("goal_radius") is not None and shift.goal_radius <= 0:
            raise ValueError("shift.goal_radius must be positive when provided.")
        if shift.get("success_distance") is not None and shift.success_distance <= 0:
            raise ValueError("shift.success_distance must be positive when provided.")
    if not cfg.seeds.training or not cfg.seeds.evaluation:
        raise ValueError("At least one training and evaluation seed is required.")


def apply_run_profile(cfg: DictConfig) -> None:
    """Resolve smoke/final caps into the actual run configuration."""
    with open_dict(cfg):
        cfg.seeds.training = list(cfg.run.training_seeds)
        cfg.seeds.evaluation = list(cfg.run.evaluation_seeds)
        cfg.evaluation.n_episodes = cfg.run.n_evaluation_episodes
        cfg.logging.log_every = cfg.run.log_every
        cfg.logging.save_checkpoints = cfg.run.save_checkpoints

        caps = [
            (cfg.run.max_n_demonstrations, cfg.policy, "n_demonstrations"),
            (cfg.run.max_bc_steps, cfg.policy, "bc_steps"),
            (cfg.run.max_total_epochs, cfg.experiment, "total_epochs"),
            (cfg.run.max_warmup_epochs, cfg.experiment, "warmup_epochs"),
            (cfg.run.get("max_inference_steps"), cfg.experiment, "inference_steps"),
            (cfg.run.max_n_envs, cfg.experiment, "n_envs"),
            (cfg.run.max_ppo_batch_size, cfg.experiment, "ppo_batch_size"),
            (cfg.run.get("max_flow_steps"), cfg.experiment, "flow_steps"),
        ]
        for cap, section, key in caps:
            if cap is not None and key in section:
                section[key] = min(section[key], cap)

        if "critic_warmup_epochs" in cfg.experiment:
            cfg.experiment.critic_warmup_epochs = min(
                cfg.experiment.critic_warmup_epochs,
                max(0, cfg.experiment.total_epochs - 1),
            )
        if backend(cfg) == "reinflow":
            rollout_size = cfg.experiment.n_envs * (
                cfg.policy.episode_steps // cfg.policy.chunk_length
            )
            cfg.experiment.ppo_batch_size = min(
                cfg.experiment.ppo_batch_size, rollout_size
            )
            if cfg.experiment.time_scale is None:
                cfg.experiment.time_scale = float(cfg.experiment.flow_steps)
