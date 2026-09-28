"""Rollout collection for the ReinFlow backend.

Mirrors `rollouts.collect_rollout` exactly -- same shapes, same `environment
.reset()` at the top, same `r_env = rewards.mean(axis=-1)` -- and returns a
superset of its keys, so `algorithms.ppo.compute_gae` is reused unchanged.

Two differences from the steering path:

  * `log_prob` is *stored* at collection time rather than recomputed. The
    steering path can recompute because nothing has changed yet; here the noise
    head's detached-vs-differentiable forward makes the sampling-time value the
    only correct `pi_old`.
  * `z` is an all-zeros int64 array. It exists so `compute_gae` and
    `LatentCritic` need no adapter, and so a later MI hook has somewhere to go.
"""

from __future__ import annotations

import numpy as np
import torch

from four_goal_bmd.algorithms.reinflow_ppo import sample_flow_chain
from four_goal_bmd.envs.four_goals import FourGoalEnv
from four_goal_bmd.models.flow import NoisyFlowPolicy


def collect_flow_rollout(
    actor: NoisyFlowPolicy,
    environment: FourGoalEnv,
    n_units: int,
    action_scale: float,
    chunk_length: int,
    action_dim: int,
    cfg,
) -> dict[str, np.ndarray]:
    n_envs = environment.n_envs
    observation_dim = environment.state.shape[-1]
    action_dim_total = actor.model.output_dim

    observation = environment.reset()
    clip_statistics: dict = {}

    observations = np.zeros((n_units, n_envs, observation_dim), dtype=np.float32)
    visited_states = np.zeros(
        (n_units, n_envs, chunk_length, observation_dim), dtype=np.float32
    )
    chains = np.zeros(
        (n_units, n_envs, cfg.flow_steps + 1, action_dim_total), dtype=np.float32
    )
    log_probs = np.zeros((n_units, n_envs), dtype=np.float32)
    environment_rewards = np.zeros((n_units, n_envs), dtype=np.float32)
    successes = np.zeros((n_units, n_envs), dtype=np.float32)
    dones = np.zeros((n_units, n_envs), dtype=np.float32)
    end_positions = np.zeros((n_units, n_envs, observation_dim), dtype=np.float32)

    for unit in range(n_units):
        state_tensor = torch.from_numpy(observation).float()
        action, chain, log_prob = sample_flow_chain(
            actor=actor,
            condition=state_tensor,
            flow_steps=cfg.flow_steps,
            act_limit=cfg.act_limit,
            denoised_clip_value=cfg.denoised_clip_value,
            min_sampling_std=cfg.min_sampling_std,
            randn_clip_value=cfg.randn_clip_value,
            clip_intermediate_actions=cfg.clip_intermediate_actions,
            deterministic=False,
            account_for_initial_stochasticity=cfg.account_for_initial_stochasticity,
            normalize_denoising_horizon=cfg.normalize_denoising_horizon,
            normalize_action_dimension=cfg.normalize_action_dimension,
            clip_statistics=clip_statistics,
        )
        action_chunk = (action * action_scale).reshape(-1, chunk_length, action_dim)
        visited, rewards, success, truncated, end_position = (
            environment.apply_action_chunk(action_chunk.numpy())
        )

        observations[unit] = observation
        visited_states[unit] = visited
        chains[unit] = chain.numpy()
        log_probs[unit] = log_prob.numpy()
        environment_rewards[unit] = rewards.mean(axis=-1)
        successes[unit] = success.astype(np.float32)
        dones[unit] = truncated.astype(np.float32)
        end_positions[unit] = end_position
        observation = environment.state.copy()

    return {
        "obs": observations,
        "visited_state": visited_states,
        "chain": chains,
        "log_prob": log_probs,
        "z": np.zeros((n_units, n_envs), dtype=np.int64),
        "r_env": environment_rewards,
        "r_int": np.zeros((n_units, n_envs), dtype=np.float32),
        "r_tot": cfg.environment_reward_scale * environment_rewards,
        "success": successes,
        "done": dones,
        "end_position": end_positions,
        "last_obs": observation,
        "clip_frac_randn": clip_statistics.get("randn_clipped", 0.0)
        / max(clip_statistics.get("randn_total", 1.0), 1.0),
        "clip_frac_act_limit": clip_statistics.get("act_clipped", 0.0)
        / max(clip_statistics.get("act_total", 1.0), 1.0),
    }
