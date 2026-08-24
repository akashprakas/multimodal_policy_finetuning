from __future__ import annotations

import numpy as np
import torch

from four_goal_bmd.envs.four_goals import FourGoalEnv
from four_goal_bmd.metrics import compute_metrics
from four_goal_bmd.models.diffusion import FrozenDiffusionPolicy
from four_goal_bmd.models.steering import SteeringPolicy


def evaluate_policy(
    frozen_policy: FrozenDiffusionPolicy,
    goal_centers: np.ndarray,
    environment_cfg,
    policy_cfg,
    n_episodes: int,
    evaluation_seed: int,
    min_attempts: int,
    steering_policy: SteeringPolicy | None = None,
    k_latent: int | None = None,
    deterministic_steering: bool = False,
    hold_noise_steps: int | None = None,
    steering_hold_units: int = 1,
) -> dict:
    """Evaluate steered or prior-noise behavior under a declared convention."""
    torch.manual_seed(evaluation_seed)
    environment = FourGoalEnv(
        n_envs=n_episodes,
        goal_centers=goal_centers,
        horizon=policy_cfg.episode_steps,
        reward_sigma=environment_cfg.reward_sigma,
        success_distance=environment_cfg.success_distance,
        seed=evaluation_seed,
    )
    observation = environment.reset()
    n_units = policy_cfg.episode_steps // policy_cfg.chunk_length
    successes = np.zeros(n_episodes, dtype=bool)
    done = np.zeros(n_episodes, dtype=bool)
    final_positions = np.zeros((n_episodes, 2), dtype=np.float32)
    latents = None
    if steering_policy is not None:
        if k_latent is None:
            raise ValueError("k_latent is required for steered evaluation.")
        latents = torch.randint(0, k_latent, (n_episodes,))

    held_noise = None
    held_until_step = -1
    held_steered = None
    for unit in range(n_units):
        state_tensor = torch.from_numpy(observation).float()
        with torch.no_grad():
            if steering_policy is not None:
                # Must match the training convention, or the evaluation measures a
                # different policy than the one that was trained.
                if held_steered is None or unit % max(1, steering_hold_units) == 0:
                    held_steered, _ = steering_policy.sample(
                        state_tensor, latents, deterministic=deterministic_steering
                    )
                noise = held_steered
            elif hold_noise_steps is None:
                noise = torch.randn(n_episodes, frozen_policy.noise_dim)
            else:
                action_step = unit * policy_cfg.chunk_length
                if held_noise is None or action_step >= held_until_step:
                    held_noise = torch.randn(n_episodes, frozen_policy.noise_dim)
                    held_until_step = action_step + hold_noise_steps
                noise = held_noise
            actions = frozen_policy(state_tensor, noise)
        _, _, success, truncated, end_position = environment.apply_action_chunk(
            actions.numpy()
        )
        newly_done = (success | truncated) & ~done
        final_positions[newly_done] = end_position[newly_done]
        done |= success | truncated
        successes |= success
        observation = environment.state.copy()

    return compute_metrics(
        goal_centers=goal_centers,
        final_positions=final_positions,
        successes=successes,
        min_attempts=min_attempts,
        latents=None if latents is None else latents.numpy(),
        k_latent=k_latent,
    )
