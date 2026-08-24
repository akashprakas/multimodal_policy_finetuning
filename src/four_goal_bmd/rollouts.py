from __future__ import annotations

import math

import numpy as np
import torch

from four_goal_bmd.envs.four_goals import FourGoalEnv
from four_goal_bmd.models.diffusion import FrozenDiffusionPolicy
from four_goal_bmd.models.steering import InferenceModel, SteeringPolicy


def collect_rollout(
    frozen_policy: FrozenDiffusionPolicy,
    steering_policy: SteeringPolicy,
    inference_model: InferenceModel,
    environment: FourGoalEnv,
    n_units: int,
    k_latent: int,
    intrinsic_scale: float,
    use_environment_reward: bool,
    environment_reward_scale: float = 1.0,
) -> dict[str, np.ndarray]:
    """Collect steps or chunks while holding `z` fixed and resampling `w`."""
    n_envs = environment.n_envs
    chunk_length = frozen_policy.chunk_length
    observation_dim = environment.state.shape[-1]
    noise_dim = frozen_policy.noise_dim
    log_prior = -math.log(k_latent)

    observation = environment.reset()
    latent = torch.randint(0, k_latent, (n_envs,))

    observations = np.zeros((n_units, n_envs, observation_dim), dtype=np.float32)
    visited_states = np.zeros(
        (n_units, n_envs, chunk_length, observation_dim), dtype=np.float32
    )
    noises = np.zeros((n_units, n_envs, noise_dim), dtype=np.float32)
    latents = np.zeros((n_units, n_envs), dtype=np.int64)
    environment_rewards = np.zeros((n_units, n_envs), dtype=np.float32)
    intrinsic_rewards = np.zeros((n_units, n_envs), dtype=np.float32)
    successes = np.zeros((n_units, n_envs), dtype=np.float32)
    dones = np.zeros((n_units, n_envs), dtype=np.float32)

    for unit in range(n_units):
        state_tensor = torch.from_numpy(observation).float()
        with torch.no_grad():
            noise, _ = steering_policy.sample(state_tensor, latent)
            action_chunk = frozen_policy(state_tensor, noise)
        visited, rewards, success, truncated, _ = environment.apply_action_chunk(
            action_chunk.numpy()
        )

        with torch.no_grad():
            flat_states = torch.from_numpy(visited.reshape(-1, observation_dim)).float()
            repeated_latent = latent.repeat_interleave(chunk_length)
            log_q = inference_model.log_prob(flat_states, repeated_latent).reshape(
                n_envs, chunk_length
            )
            intrinsic = intrinsic_scale * (log_q - log_prior).mean(dim=-1)

        observations[unit] = observation
        visited_states[unit] = visited
        noises[unit] = noise.numpy()
        latents[unit] = latent.numpy()
        environment_rewards[unit] = rewards.mean(axis=-1)
        intrinsic_rewards[unit] = intrinsic.numpy()
        successes[unit] = success.astype(np.float32)
        dones[unit] = truncated.astype(np.float32)
        observation = environment.state.copy()

    total_rewards = intrinsic_rewards.copy()
    if use_environment_reward:
        total_rewards += environment_reward_scale * environment_rewards

    return {
        "obs": observations,
        "visited_state": visited_states,
        "w": noises,
        "z": latents,
        "r_env": environment_rewards,
        "r_int": intrinsic_rewards,
        "r_tot": total_rewards,
        "success": successes,
        "done": dones,
        "last_obs": observation,
    }
