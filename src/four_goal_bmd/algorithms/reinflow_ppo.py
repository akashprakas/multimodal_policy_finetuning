"""ReinFlow chain sampling, the chain log-probability surrogate, and PPO.

Reference: upstream `model/flow/ft_ppo/ppoflow.py:195-527` and
`agent/finetune/reinflow/train_ppo_flow_agent.py`.

The injected noise turns each deterministic Euler step into a proper Gaussian
transition, so the joint density of the *unclipped, un-normalized* chain has a
closed form:

    log pi(x_0..x_K | s) = log N(x_0 | 0, I)
                           + sum_k log N(x_{k+1} | mu_k, sigma_k^2 I)
    mu_k    = clip(x_k + v(x_k, t_k, s) * dt)
    sigma_k = sigma_phi(s, t_k)

summed over the chunk*action dimensions and then over the K transitions, giving
one scalar per environment step.

**This is a surrogate, not the exact likelihood of the sampling process.** Call
it the upstream ReinFlow chain-log-probability surrogate. Three reasons, all
inherited from upstream:

  * sampling truncates each Gaussian draw at `randn_clip_value` standard
    deviations, but scoring uses the untruncated Normal;
  * the final chain point is clipped to the action range, which puts an atom at
    the boundary that the density ignores (upstream has no Jacobian correction);
  * the result is divided by the chain length and the action dimension, so the
    PPO ratio is a *tempered* joint-density ratio rather than the raw chain
    likelihood ratio.

`ratio_sanity` proves that the sampling path and the differentiable
recomputation implement the *same* surrogate. It does not prove the surrogate is
the exact likelihood of the clipped sampling process. `clip_frac_randn` and
`clip_frac_act_limit` measure how often the two clips actually bind.

Four consistency requirements bind the sampling and recomputation paths. Any
violation shows up as `ratio != 1` on the first minibatch, which is checked at
run time rather than left to a test:

  1. `denoised_clip_value == act_limit`, because sampling clamps the sampled
     point to the action range on the last step while recomputation only clamps
     the mean.
  2. `min_sampling_std == sigma_min`, because sampling clamps the std and
     recomputation does not.
  3. `randn_clip_value` truncates the sample but the density is the untruncated
     Normal -- a deliberate upstream approximation, worse at small K or large
     sigma.
  4. Chain entries must be stored as copies; `x = x + v*dt` must not be in-place.
"""

from __future__ import annotations

import numpy as np
import torch
from four_goal_bmd.anchors import anchor_loss, gradient_norm_ratio, velocity_parameters
from four_goal_bmd.models.flow import NoisyFlowPolicy
from torch.distributions import Normal

from four_goal_bmd.algorithms.ppo import compute_gae
from four_goal_bmd.models.steering import LatentCritic


def _logprob_steps(flow_steps: int, account_for_initial_stochasticity: bool) -> int:
    return flow_steps + 1 if account_for_initial_stochasticity else flow_steps


@torch.no_grad()
def sample_flow_chain(
    actor: NoisyFlowPolicy,
    condition: torch.Tensor,
    flow_steps: int,
    act_limit: float,
    denoised_clip_value: float,
    min_sampling_std: float,
    randn_clip_value: float = 3.0,
    clip_intermediate_actions: bool = True,
    deterministic: bool = False,
    account_for_initial_stochasticity: bool = True,
    normalize_denoising_horizon: bool = True,
    normalize_action_dimension: bool = True,
    initial_noise: torch.Tensor | None = None,
    clip_statistics: dict | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample a stochastic denoising chain.

    Returns (action [B, D], chain [B, K+1, D], log_prob [B]).

    Pass a dict as `clip_statistics` to receive the fraction of scalar draws the
    `randn_clip_value` truncation bound and the fraction the action-range clip
    bound. Both are places where the scored surrogate departs from the sampling
    process, so their magnitude is what says whether that matters.
    """
    batch_size = condition.shape[0]
    action_dim_total = actor.model.output_dim
    device = condition.device
    step_size = 1.0 / flow_steps

    if initial_noise is None:
        sample = torch.randn(batch_size, action_dim_total, device=device)
    else:
        sample = initial_noise.reshape(batch_size, action_dim_total)

    chain = torch.zeros(batch_size, flow_steps + 1, action_dim_total, device=device)
    chain[:, 0] = sample

    log_prob = torch.zeros(batch_size, device=device)
    if account_for_initial_stochasticity:
        initial_distribution = Normal(torch.zeros_like(sample), torch.ones_like(sample))
        log_prob = log_prob + initial_distribution.log_prob(sample).sum(-1)

    for index in range(flow_steps):
        t = torch.full((batch_size,), index * step_size, device=device)
        velocity, sigma = actor.forward(
            sample, t, condition, learn_exploration_noise=False
        )
        mean = sample + velocity * step_size
        if clip_intermediate_actions:
            mean = mean.clamp(-denoised_clip_value, denoised_clip_value)
        std = sigma.clamp(min=min_sampling_std)
        distribution = Normal(mean, std)
        if deterministic:
            sample = mean
        else:
            raw = distribution.sample()
            sample = raw.clamp(
                mean - randn_clip_value * std, mean + randn_clip_value * std
            )
            if clip_statistics is not None:
                clip_statistics.setdefault("randn_clipped", 0.0)
                clip_statistics.setdefault("randn_total", 0.0)
                clip_statistics["randn_clipped"] += float((raw != sample).sum())
                clip_statistics["randn_total"] += float(raw.numel())
        if index == flow_steps - 1:
            clipped = sample.clamp(-act_limit, act_limit)
            if clip_statistics is not None:
                clip_statistics.setdefault("act_clipped", 0.0)
                clip_statistics.setdefault("act_total", 0.0)
                clip_statistics["act_clipped"] += float((clipped != sample).sum())
                clip_statistics["act_total"] += float(sample.numel())
            sample = clipped
        log_prob = log_prob + distribution.log_prob(sample).sum(-1)
        chain[:, index + 1] = sample

    if normalize_denoising_horizon:
        log_prob = log_prob / _logprob_steps(
            flow_steps, account_for_initial_stochasticity
        )
    if normalize_action_dimension:
        log_prob = log_prob / action_dim_total
    return sample, chain, log_prob


def flow_chain_logprob(
    actor: NoisyFlowPolicy,
    condition: torch.Tensor,
    chain: torch.Tensor,
    flow_steps: int,
    denoised_clip_value: float,
    clip_intermediate_actions: bool = True,
    account_for_initial_stochasticity: bool = True,
    normalize_denoising_horizon: bool = True,
    normalize_action_dimension: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Differentiable recomputation on a stored chain.

    Returns (log_prob [B], entropy_rate [B], sigma [B, K, D]).
    """
    batch_size = chain.shape[0]
    action_dim_total = actor.model.output_dim
    device = chain.device
    step_size = 1.0 / flow_steps

    log_prob = torch.zeros(batch_size, device=device)
    entropy = torch.zeros(batch_size, device=device)
    if account_for_initial_stochasticity:
        initial_point = chain[:, 0]
        initial_distribution = Normal(
            torch.zeros_like(initial_point), torch.ones_like(initial_point)
        )
        log_prob = log_prob + initial_distribution.log_prob(initial_point).sum(-1)
        entropy = entropy + initial_distribution.entropy().sum(-1)

    velocities: list[torch.Tensor] = []
    sigmas: list[torch.Tensor] = []
    for index in range(flow_steps):
        t = torch.full((batch_size,), index * step_size, device=device)
        velocity, sigma = actor.forward(
            chain[:, index], t, condition, learn_exploration_noise=True
        )
        velocities.append(velocity)
        sigmas.append(sigma)
    # Stacked rather than written into a preallocated zeros buffer: same
    # numerics, unambiguous autograd.
    velocity_chain = torch.stack(velocities, dim=1)
    sigma_chain = torch.stack(sigmas, dim=1)

    mean = chain[:, :-1] + velocity_chain * step_size
    if clip_intermediate_actions:
        mean = mean.clamp(-denoised_clip_value, denoised_clip_value)
    distribution = Normal(mean, sigma_chain)
    log_prob = log_prob + distribution.log_prob(chain[:, 1:]).sum(-1).sum(-1)
    entropy = entropy + distribution.entropy().sum(-1).sum(-1)

    steps = _logprob_steps(flow_steps, account_for_initial_stochasticity)
    entropy_rate = entropy / steps
    if normalize_denoising_horizon:
        log_prob = log_prob / steps
    if normalize_action_dimension:
        log_prob = log_prob / action_dim_total
        entropy_rate = entropy_rate / action_dim_total
    return log_prob, entropy_rate, sigma_chain


def reinflow_update(
    actor: NoisyFlowPolicy,
    critic: LatentCritic,
    actor_optimizer: torch.optim.Optimizer | None,
    critic_optimizer: torch.optim.Optimizer,
    rollout: dict[str, np.ndarray],
    cfg,
    update_actor: bool = True,
    reference_model=None,
    anchor_generator: torch.Generator | None = None,
) -> dict[str, float]:
    """PPO update on the flow policy and its noise head.

    `reference_model` is a frozen copy of the pretrained velocity field. When
    `cfg.anchor_type != "none"` and `cfg.anchor_coef > 0`, a drift penalty is
    added; at `anchor_coef == 0` the loss is bitwise the plain objective, which is
    asserted by `tests/test_anchors.py`.
    """
    n_units, n_envs = rollout["r_tot"].shape
    latent_tensor = torch.zeros(n_units, n_envs, dtype=torch.long)
    advantages, returns = compute_gae(
        critic=critic,
        observations=rollout["obs"],
        latents=rollout["z"],
        rewards=rollout["r_tot"],
        dones=rollout["done"],
        last_observation=rollout["last_obs"],
        last_latent=latent_tensor[-1],
        gamma=cfg.gamma,
        gae_lambda=cfg.gae_lambda,
    )

    total = n_units * n_envs
    observations = torch.from_numpy(rollout["obs"].reshape(total, -1)).float()
    chains = torch.from_numpy(
        rollout["chain"].reshape(total, *rollout["chain"].shape[2:])
    ).float()
    old_log_prob = torch.from_numpy(rollout["log_prob"].reshape(total)).float()
    advantage_tensor = torch.from_numpy(advantages.reshape(total)).float()
    return_tensor = torch.from_numpy(returns.reshape(total)).float()
    latents = torch.zeros(total, dtype=torch.long)

    logprob_kwargs = dict(
        flow_steps=cfg.flow_steps,
        denoised_clip_value=cfg.denoised_clip_value,
        clip_intermediate_actions=cfg.clip_intermediate_actions,
        account_for_initial_stochasticity=cfg.account_for_initial_stochasticity,
        normalize_denoising_horizon=cfg.normalize_denoising_horizon,
        normalize_action_dimension=cfg.normalize_action_dimension,
    )

    records: list[dict[str, float]] = []
    ratio_sanity = float("nan")
    # Computed on the first minibatch only, and kept: initializing it inside the
    # loop overwrote it with nan on every subsequent minibatch, so the stored
    # value was always nan.
    anchor_ratio = float("nan")
    stopped_epoch = cfg.ppo_update_epochs
    for update_epoch in range(cfg.ppo_update_epochs):
        permutation = torch.randperm(total)
        epoch_kl: list[float] = []
        for start in range(0, total, cfg.ppo_batch_size):
            batch = permutation[start : start + cfg.ppo_batch_size]
            new_log_prob, entropy_rate, sigma = flow_chain_logprob(
                actor=actor,
                condition=observations[batch],
                chain=chains[batch],
                **logprob_kwargs,
            )

            clamped_new = new_log_prob.clamp(cfg.logprob_min, cfg.logprob_max)
            clamped_old = old_log_prob[batch].clamp(cfg.logprob_min, cfg.logprob_max)
            with torch.no_grad():
                clamp_binds = (
                    (
                        (new_log_prob < cfg.logprob_min)
                        | (new_log_prob > cfg.logprob_max)
                        | (old_log_prob[batch] < cfg.logprob_min)
                        | (old_log_prob[batch] > cfg.logprob_max)
                    )
                    .float()
                    .mean()
                    .item()
                )

            batch_advantages = advantage_tensor[batch]
            batch_advantages = (batch_advantages - batch_advantages.mean()) / (
                batch_advantages.std() + 1e-8
            )

            log_ratio = clamped_new - clamped_old
            ratio = log_ratio.exp()

            if update_epoch == 0 and start == 0:
                with torch.no_grad():
                    ratio_sanity = float((ratio - 1.0).abs().max())
                if not ratio_sanity < cfg.ratio_sanity_tolerance:
                    raise ValueError(
                        "ReinFlow sampling and recomputation disagree: "
                        f"max|ratio-1|={ratio_sanity:.3e} exceeds "
                        f"{cfg.ratio_sanity_tolerance:.1e}. Check that "
                        "denoised_clip_value == act_limit, "
                        "min_sampling_std == sigma_min, and that the stored "
                        "chain is not aliased."
                    )

            unclipped = -batch_advantages * ratio
            clipped = -batch_advantages * ratio.clamp(
                1.0 - cfg.clip_ploss_coef, 1.0 + cfg.clip_ploss_coef
            )
            policy_loss = torch.maximum(unclipped, clipped).mean()

            values = critic(observations[batch], latents[batch])
            value_loss = 0.5 * ((values - return_tensor[batch]) ** 2).mean()
            entropy_term = entropy_rate.mean()

            policy_objective = policy_loss - cfg.entropy_coef * entropy_term
            loss = policy_objective + cfg.value_coef * value_loss

            anchor_type = cfg.get("anchor_type", "none")
            anchor_coef = float(cfg.get("anchor_coef", 0.0))
            anchor_value = 0.0
            # `anchor_coef == 0` must leave the objective untouched, not merely
            # add a zero -- adding `0.0 * term` would still change the autograd
            # graph and the RNG draw.
            if anchor_type != "none" and anchor_coef > 0.0:
                if reference_model is None:
                    raise ValueError(
                        f"anchor_type={anchor_type} needs a reference_model."
                    )
                anchor_term = anchor_loss(
                    anchor_type=anchor_type,
                    actor=actor,
                    reference_model=reference_model,
                    observations=observations[batch],
                    flow_steps=cfg.flow_steps,
                    generator=anchor_generator,
                )
                anchor_value = float(anchor_term.detach())
                if update_epoch == 0 and start == 0:
                    anchor_ratio = gradient_norm_ratio(
                        anchor_coef * anchor_term,
                        policy_objective,
                        velocity_parameters(actor),
                    )
                loss = loss + anchor_coef * anchor_term

            if actor_optimizer is not None:
                actor_optimizer.zero_grad()
            critic_optimizer.zero_grad()
            loss.backward()
            critic_grad_norm = torch.nn.utils.clip_grad_norm_(
                critic.parameters(), cfg.max_grad_norm
            )
            critic_optimizer.step()
            trainable = [
                parameter for parameter in actor.parameters() if parameter.requires_grad
            ]
            actor_grad_norm = (
                torch.nn.utils.clip_grad_norm_(trainable, cfg.max_grad_norm)
                if trainable
                else torch.tensor(0.0)
            )
            if update_actor and actor_optimizer is not None:
                actor_optimizer.step()

            with torch.no_grad():
                approx_kl = float(((ratio - 1.0) - log_ratio).mean())
                records.append(
                    {
                        "policy_loss": float(policy_loss),
                        "value_loss": float(value_loss),
                        "entropy_rate": float(entropy_term),
                        "approx_kl": approx_kl,
                        "clipfrac": float(
                            ((ratio - 1.0).abs() > cfg.clip_ploss_coef).float().mean()
                        ),
                        "ratio_mean": float(ratio.mean()),
                        "sigma_mean": float(sigma.mean()),
                        "sigma_min": float(sigma.min()),
                        "sigma_max": float(sigma.max()),
                        "logprob_clamp_frac": clamp_binds,
                        "actor_grad_norm": float(actor_grad_norm),
                        "critic_grad_norm": float(critic_grad_norm),
                        "anchor_loss": anchor_value,
                    }
                )
                epoch_kl.append(approx_kl)
        if epoch_kl and float(np.mean(epoch_kl)) > cfg.target_kl:
            stopped_epoch = update_epoch + 1
            break

    summary = {
        key: float(np.mean([record[key] for record in records])) for key in records[0]
    }
    # Extremes, not averages of extremes: the question is whether sigma ever
    # touched a bound.
    summary["sigma_min"] = float(np.min([r["sigma_min"] for r in records]))
    summary["sigma_max"] = float(np.max([r["sigma_max"] for r in records]))
    summary["logprob_clamp_frac"] = float(
        np.max([r["logprob_clamp_frac"] for r in records])
    )
    summary["ratio_sanity"] = ratio_sanity
    summary["anchor_grad_ratio"] = anchor_ratio
    summary["stopped_epoch"] = float(stopped_epoch)
    summary["n_minibatches"] = float(len(records))
    return summary
