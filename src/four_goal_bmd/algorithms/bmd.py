from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from four_goal_bmd.algorithms.ppo import ppo_update
from four_goal_bmd.envs.four_goals import FourGoalEnv
from four_goal_bmd.models.diffusion import FrozenDiffusionPolicy
from four_goal_bmd.models.steering import InferenceModel, LatentCritic, SteeringPolicy
from four_goal_bmd.rollouts import collect_rollout


@dataclass
class BMDTrainingResult:
    steering_policy: SteeringPolicy
    critic: LatentCritic
    inference_model: InferenceModel
    history: list[dict[str, float]]


def horizon_units(
    epoch: int, start: int, maximum: int, warmup_epochs: int, enabled: bool
) -> int:
    if not enabled or warmup_epochs <= 1:
        return maximum
    fraction = min(1.0, epoch / (warmup_epochs - 1))
    return int(round(start + fraction * (maximum - start)))


def train_bmd(
    frozen_policy: FrozenDiffusionPolicy,
    goal_centers: np.ndarray,
    environment_cfg,
    experiment_cfg,
    policy_cfg,
    seed: int,
    environment_seed: int,
    log_every: int = 0,
) -> BMDTrainingResult:
    torch.manual_seed(seed)
    steering_policy = SteeringPolicy(
        observation_dim=2,
        k_latent=experiment_cfg.k_latent,
        noise_dim=frozen_policy.noise_dim,
        hidden_dims=experiment_cfg.steering_dims,
        w_bound=experiment_cfg.w_bound,
        min_log_std=experiment_cfg.min_log_std,
    )
    critic = LatentCritic(
        observation_dim=2,
        k_latent=experiment_cfg.k_latent,
        hidden_dims=experiment_cfg.critic_dims,
    )
    inference_model = InferenceModel(
        observation_dim=2,
        k_latent=experiment_cfg.k_latent,
        hidden_dims=experiment_cfg.inference_dims,
    )
    steering_optimizer = torch.optim.AdamW(
        steering_policy.parameters(), lr=experiment_cfg.steering_lr
    )
    critic_optimizer = torch.optim.AdamW(
        critic.parameters(), lr=experiment_cfg.critic_lr
    )
    inference_optimizer = torch.optim.AdamW(
        inference_model.parameters(), lr=experiment_cfg.inference_lr
    )

    environment = FourGoalEnv(
        n_envs=experiment_cfg.n_envs,
        goal_centers=goal_centers,
        horizon=policy_cfg.episode_steps,
        reward_sigma=environment_cfg.reward_sigma,
        success_distance=environment_cfg.success_distance,
        seed=environment_seed,
    )

    # Dedicated generator for anchor eps draws: anchor sampling must not perturb
    # the rollout RNG.
    anchor_generator = None
    if (
        experiment_cfg.get("dsrl_anchor_type", "none") != "none"
        and float(experiment_cfg.get("dsrl_anchor_coef", 0.0)) > 0.0
    ):
        anchor_generator = torch.Generator().manual_seed(
            int(experiment_cfg.get("dsrl_anchor_seed", 20260824)) + seed
        )

    maximum_units = policy_cfg.episode_steps // policy_cfg.chunk_length
    start_units = policy_cfg.curriculum_start_units
    history: list[dict[str, float]] = []

    for epoch in range(experiment_cfg.total_epochs):
        units = horizon_units(
            epoch=epoch,
            start=start_units,
            maximum=maximum_units,
            warmup_epochs=experiment_cfg.warmup_epochs,
            enabled=experiment_cfg.curriculum,
        )
        environment.horizon = units * policy_cfg.chunk_length
        use_environment_reward = epoch >= experiment_cfg.warmup_epochs
        intrinsic_scale = (
            experiment_cfg.finetune_intrinsic_scale
            if use_environment_reward
            else experiment_cfg.intrinsic_scale
        )
        rollout = collect_rollout(
            frozen_policy=frozen_policy,
            steering_policy=steering_policy,
            inference_model=inference_model,
            environment=environment,
            n_units=units,
            k_latent=experiment_cfg.k_latent,
            intrinsic_scale=intrinsic_scale,
            use_environment_reward=use_environment_reward,
            environment_reward_scale=experiment_cfg.environment_reward_scale,
        )
        update = ppo_update(
            steering_policy,
            critic,
            inference_model,
            steering_optimizer,
            critic_optimizer,
            inference_optimizer,
            rollout,
            experiment_cfg,
            frozen_policy=frozen_policy,
            anchor_generator=anchor_generator,
        )
        # per-env total env reward this rollout, to see how compounded noise scales with horizon
        r_env_per_env = rollout["r_env"].sum(axis=0)
        record = {
            "epoch": epoch,
            "horizon_units": units,
            "phase": "fine_tune" if use_environment_reward else "discovery",
            "r_int": float(rollout["r_int"].mean()),
            "r_env": float(rollout["r_env"].mean()),
            "r_env_var_across_envs": float(r_env_per_env.var()),
            "success": float(rollout["success"].max(axis=0).mean()),
            "log_std": float(steering_policy.log_std.detach().mean()),
            **update,
        }
        history.append(record)
        if log_every and (
            epoch % log_every == 0 or epoch == experiment_cfg.total_epochs - 1
        ):
            print(
                f"epoch={epoch:3d} phase={record['phase']:9s} units={units:2d} "
                f"r_int={record['r_int']:+.3f} r_env={record['r_env']:+.4f} "
                f"r_env_var={record['r_env_var_across_envs']:.4f} "
                f"nll={record['nll']:.3f} log_std={record['log_std']:+.2f}"
            )

    environment.horizon = policy_cfg.episode_steps
    return BMDTrainingResult(
        steering_policy=steering_policy,
        critic=critic,
        inference_model=inference_model,
        history=history,
    )
