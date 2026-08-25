from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from omegaconf import DictConfig, open_dict


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


def validate_config(cfg: DictConfig) -> None:
    if cfg.environment.name != "four_goal":
        raise ValueError("This reproduction supports only FourGoalEnv.")
    if cfg.environment.observation_dim != 2 or cfg.environment.action_dim != 2:
        raise ValueError("FourGoalEnv requires 2D observations and actions.")
    if cfg.environment.n_goals != 4:
        raise ValueError("This reproduction fixes the evaluation to four goals.")
    if cfg.policy.episode_steps % cfg.policy.chunk_length != 0:
        raise ValueError("episode_steps must be divisible by chunk_length.")
    if not (1 <= cfg.policy.ddim_steps <= cfg.policy.denoising_steps):
        raise ValueError("ddim_steps must be between 1 and denoising_steps.")
    if cfg.experiment.warmup_epochs > cfg.experiment.total_epochs:
        raise ValueError("warmup_epochs cannot exceed total_epochs.")
    if cfg.experiment.k_latent < 1:
        raise ValueError("k_latent must be positive.")
    if cfg.experiment.inference_steps < 1:
        raise ValueError("inference_steps must be positive.")
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
            (cfg.run.max_inference_steps, cfg.experiment, "inference_steps"),
            (cfg.run.max_n_envs, cfg.experiment, "n_envs"),
            (cfg.run.max_ppo_batch_size, cfg.experiment, "ppo_batch_size"),
        ]
        for cap, section, key in caps:
            if cap is not None:
                section[key] = min(section[key], cap)
