from __future__ import annotations

import numpy as np
import torch

from four_goal_bmd.models.steering import InferenceModel, LatentCritic, SteeringPolicy


def compute_gae(
    critic: LatentCritic,
    observations: np.ndarray,
    latents: np.ndarray,
    rewards: np.ndarray,
    dones: np.ndarray,
    last_observation: np.ndarray,
    last_latent: torch.Tensor,
    gamma: float,
    gae_lambda: float,
):
    time_steps, n_envs = rewards.shape
    with torch.no_grad():
        observation_tensor = torch.from_numpy(observations).float()
        latent_tensor = torch.from_numpy(latents).long()
        values = (
            critic(
                observation_tensor.reshape(time_steps * n_envs, -1),
                latent_tensor.reshape(time_steps * n_envs),
            )
            .numpy()
            .reshape(time_steps, n_envs)
        )
        last_value = critic(
            torch.from_numpy(last_observation).float(), last_latent
        ).numpy()

    advantages = np.zeros_like(rewards)
    last_advantage = 0.0
    for index in reversed(range(time_steps)):
        next_value = last_value if index == time_steps - 1 else values[index + 1]
        nonterminal = 1.0 - dones[index]
        delta = rewards[index] + gamma * next_value * nonterminal - values[index]
        last_advantage = delta + gamma * gae_lambda * nonterminal * last_advantage
        advantages[index] = last_advantage
    return advantages, advantages + values


def ppo_update(
    steering_policy: SteeringPolicy,
    critic: LatentCritic,
    inference_model: InferenceModel,
    steering_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    inference_optimizer: torch.optim.Optimizer,
    rollout: dict[str, np.ndarray],
    cfg,
) -> dict[str, float]:
    time_steps, n_envs = rollout["r_tot"].shape
    latent_tensor = torch.from_numpy(rollout["z"]).long()
    advantages, returns = compute_gae(
        critic,
        rollout["obs"],
        rollout["z"],
        rollout["r_tot"],
        rollout["done"],
        rollout["last_obs"],
        latent_tensor[-1],
        cfg.gamma,
        cfg.gae_lambda,
    )

    observations = torch.from_numpy(
        rollout["obs"].reshape(time_steps * n_envs, -1)
    ).float()
    noises = torch.from_numpy(rollout["w"].reshape(time_steps * n_envs, -1)).float()
    latents = torch.from_numpy(rollout["z"].reshape(time_steps * n_envs)).long()
    advantage_tensor = torch.from_numpy(advantages.reshape(-1)).float()
    return_tensor = torch.from_numpy(returns.reshape(-1)).float()
    with torch.no_grad():
        old_log_prob = steering_policy.log_prob(observations, latents, noises)
    advantage_tensor = (advantage_tensor - advantage_tensor.mean()) / (
        advantage_tensor.std() + 1e-8
    )

    total = time_steps * n_envs
    policy_losses: list[float] = []
    value_losses: list[float] = []
    approximate_kls: list[float] = []
    for _ in range(cfg.ppo_update_epochs):
        permutation = torch.randperm(total)
        n_batches = max(1, total // cfg.ppo_batch_size)
        for batch_index in range(n_batches):
            indices = permutation[
                batch_index * cfg.ppo_batch_size : (batch_index + 1)
                * cfg.ppo_batch_size
            ]
            distribution = steering_policy.distribution(
                observations[indices], latents[indices]
            )
            new_log_prob = distribution.log_prob(noises[indices]).sum(dim=-1)
            log_ratio = new_log_prob - old_log_prob[indices]
            ratio = log_ratio.exp()
            with torch.no_grad():
                approximate_kls.append(float(((ratio - 1) - log_ratio).mean().item()))

            unclipped_loss = -advantage_tensor[indices] * ratio
            clipped_loss = -advantage_tensor[indices] * torch.clamp(
                ratio, 1 - cfg.clip_epsilon, 1 + cfg.clip_epsilon
            )
            policy_loss = torch.maximum(unclipped_loss, clipped_loss).mean()
            entropy = distribution.entropy().sum(dim=-1).mean()
            value_loss = (
                0.5
                * (
                    critic(observations[indices], latents[indices])
                    - return_tensor[indices]
                )
                .pow(2)
                .mean()
            )
            total_loss = (
                policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy
            )

            steering_optimizer.zero_grad()
            critic_optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                steering_policy.parameters(), cfg.max_grad_norm
            )
            steering_optimizer.step()
            critic_optimizer.step()
            policy_losses.append(float(policy_loss.item()))
            value_losses.append(float(value_loss.item()))

        if (
            cfg.target_kl is not None
            and np.mean(approximate_kls[-n_batches:]) > cfg.target_kl
        ):
            break

    visited = rollout["visited_state"]
    chunk_length = visited.shape[2]
    flat_visited = torch.from_numpy(visited.reshape(-1, visited.shape[-1])).float()
    inference_latents = latents.repeat_interleave(chunk_length)
    n_inference_samples = flat_visited.shape[0]
    inference_batch_size = min(cfg.inference_batch_size, n_inference_samples)
    nll = 0.0
    for _ in range(cfg.inference_steps):
        if inference_batch_size < n_inference_samples:
            indices = torch.randint(0, n_inference_samples, (inference_batch_size,))
        else:
            indices = slice(None)
        inference_loss = -inference_model.log_prob(
            flat_visited[indices],
            inference_latents[indices],
            training_noise_std=cfg.inference_input_noise_std,
        ).mean()
        inference_optimizer.zero_grad()
        inference_loss.backward()
        inference_optimizer.step()
        nll = float(inference_loss.item())

    return {
        "policy_loss": float(np.mean(policy_losses)),
        "value_loss": float(np.mean(value_losses)),
        "approx_kl": float(np.mean(approximate_kls)),
        "nll": nll,
    }
