from __future__ import annotations

import torch

from four_goal_bmd.algorithms.diffusion_bc import train_diffusion_bc
from four_goal_bmd.data.demonstrations import generate_demonstrations
from four_goal_bmd.envs.four_goals import make_goal_centers
from four_goal_bmd.models.diffusion import (
    DiffusionMLP,
    DiffusionModel,
    FrozenDiffusionPolicy,
)


def build_diffusion_model(cfg) -> DiffusionModel:
    output_dim = cfg.environment.action_dim * cfg.policy.chunk_length
    network = DiffusionMLP(
        output_dim=output_dim,
        condition_dim=cfg.environment.observation_dim,
        time_embedding_dim=cfg.policy.time_embedding_dim,
        hidden_dims=cfg.policy.network_dims,
    )
    return DiffusionModel(
        network=network,
        output_dim=output_dim,
        denoising_steps=cfg.policy.denoising_steps,
        denoised_clip_value=cfg.policy.denoised_clip_value,
        device=cfg.device,
    )


def build_frozen_policy(
    model: DiffusionModel, action_scale: float, cfg
) -> FrozenDiffusionPolicy:
    return FrozenDiffusionPolicy(
        model=model,
        action_scale=action_scale,
        action_dim=cfg.environment.action_dim,
        chunk_length=cfg.policy.chunk_length,
        ddim_steps=cfg.policy.ddim_steps,
    )


def pretrain_frozen_policy(cfg, verbose: bool = True):
    """Train one BC model on unrotated four-goal demonstrations."""
    training_goals = make_goal_centers(
        n_goals=cfg.environment.n_goals,
        radius=cfg.environment.goal_radius,
        rotation=0.0,
    )
    dataset = generate_demonstrations(
        goal_centers=training_goals,
        n_demonstrations=cfg.policy.n_demonstrations,
        episode_steps=cfg.policy.episode_steps,
        chunk_length=cfg.policy.chunk_length,
        seed=cfg.seeds.bc,
        action_noise_std=cfg.policy.expert_action_noise_std,
    )
    # Historical runs fixed model/optimizer randomness separately from the demo seed.
    torch.manual_seed(42)
    model = build_diffusion_model(cfg)
    history = train_diffusion_bc(
        model=model,
        dataset=dataset,
        training_steps=cfg.policy.bc_steps,
        batch_size=cfg.policy.bc_batch_size,
        learning_rate=cfg.policy.bc_learning_rate,
        log_every=max(1, cfg.policy.bc_steps // 5) if verbose else 0,
    )
    return build_frozen_policy(model, dataset.action_scale, cfg), history
