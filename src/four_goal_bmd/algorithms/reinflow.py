"""ReinFlow fine-tuning of a flow policy on FourGoalEnv.

ReinFlow-plain is the single-factor analogue of `experiment/plain_dsrl.yaml`: no
latent, no MI reward, no curriculum, task reward only. What differs from the
steering path is *what* is optimized -- the velocity field and its noise head,
rather than a steering distribution over a frozen decoder's input noise.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np
import torch
from four_goal_bmd.flow_diagnostics import (
    endpoint_spread,
    noise_sensitivity_probe,
    sigma_field_probe,
)
from four_goal_bmd.flow_rollouts import collect_flow_rollout
from four_goal_bmd.models.flow import FlowModel, FlowPolicyActor, NoisyFlowPolicy

from four_goal_bmd.algorithms.bmd import horizon_units
from four_goal_bmd.algorithms.reinflow_ppo import reinflow_update
from four_goal_bmd.envs.four_goals import FourGoalEnv
from four_goal_bmd.metrics import compute_metrics
from four_goal_bmd.models.steering import LatentCritic


def actor_fingerprint(actor: NoisyFlowPolicy) -> list[float]:
    """Cheap identity used to verify per-seed actor initialization."""
    return [float(parameter.detach().sum()) for parameter in actor.parameters()]


@dataclass
class ReinFlowTrainingResult:
    actor: NoisyFlowPolicy
    critic: LatentCritic
    history: list[dict[str, float]]
    bc_gate: dict = field(default_factory=dict)
    # This seed's actor measured immediately before its first update, under the
    # SAME stochastic convention as the post-RL evaluation. The BC gate uses
    # deterministic integration and a different noise head, so it is not the
    # right within-policy baseline.
    pre_rl: dict = field(default_factory=dict)


def build_actor_from_bc(
    bc_model: FlowModel, experiment_cfg, condition_dim: int, time_embedding_dim: int
) -> NoisyFlowPolicy:
    """Deep-copy the BC model, then wrap it with a fresh noise head.

    The copy is load-bearing and differs from the diffusion path. That BC model
    is frozen and can safely be shared across training seeds by reference; this
    one is *trained*, so sharing the object would make seed k+1 silently continue
    seed k's fine-tuning.
    """
    model = copy.deepcopy(bc_model)
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    model.train()
    return NoisyFlowPolicy(
        model=model,
        flow_steps=experiment_cfg.flow_steps,
        sigma_min=experiment_cfg.sigma_min,
        sigma_max=experiment_cfg.sigma_max,
        condition_dim=condition_dim,
        time_embedding_dim=time_embedding_dim,
        noise_hidden_dims=list(experiment_cfg.noise_hidden_dims),
        time_independent_noise=experiment_cfg.time_independent_noise,
        noise_activation=experiment_cfg.noise_activation,
    )


def make_flow_actor(
    actor: NoisyFlowPolicy,
    action_scale: float,
    environment_cfg,
    policy_cfg,
    experiment_cfg,
    deterministic: bool = False,
) -> FlowPolicyActor:
    return FlowPolicyActor(
        actor=actor,
        action_scale=action_scale,
        action_dim=environment_cfg.action_dim,
        chunk_length=policy_cfg.chunk_length,
        flow_steps=experiment_cfg.flow_steps,
        act_limit=experiment_cfg.act_limit,
        denoised_clip_value=experiment_cfg.denoised_clip_value,
        min_sampling_std=experiment_cfg.min_sampling_std,
        randn_clip_value=experiment_cfg.randn_clip_value,
        clip_intermediate_actions=experiment_cfg.clip_intermediate_actions,
        deterministic=deterministic,
    )


def train_reinflow(
    bc_model: FlowModel,
    action_scale: float,
    goal_centers: np.ndarray,
    environment_cfg,
    experiment_cfg,
    policy_cfg,
    evaluation_cfg,
    seed: int,
    environment_seed: int,
    log_every: int = 0,
) -> ReinFlowTrainingResult:
    torch.manual_seed(seed)
    actor = build_actor_from_bc(
        bc_model,
        experiment_cfg,
        condition_dim=environment_cfg.observation_dim,
        time_embedding_dim=policy_cfg.time_embedding_dim,
    )
    critic = LatentCritic(
        observation_dim=environment_cfg.observation_dim,
        k_latent=1,
        hidden_dims=experiment_cfg.critic_dims,
    )

    # Frozen reference for the drift anchors: a separate deep copy of the same
    # pretrained weights, never trained, never in an optimizer.
    anchor_type = experiment_cfg.get("anchor_type", "none")
    reference_model = None
    anchor_generator = None
    if anchor_type != "none" and float(experiment_cfg.get("anchor_coef", 0.0)) > 0:
        reference_model = copy.deepcopy(bc_model)
        for parameter in reference_model.parameters():
            parameter.requires_grad_(False)
        reference_model.eval()
        # Dedicated generator: anchor x_0 draws must not perturb the rollout RNG.
        anchor_generator = torch.Generator().manual_seed(
            int(experiment_cfg.get("anchor_seed", 20260823)) + seed
        )

    # Two independent switches, so three controls are expressible:
    #   both true   -- plain ReinFlow
    #   field false -- noise-only MECHANISM control (the head still adapts)
    #   both false  -- FULLY FROZEN anti-freezing falsification control
    learn_noise = experiment_cfg.get("learn_noise_head", True)
    if not experiment_cfg.learn_velocity_field:
        for parameter in actor.model.parameters():
            parameter.requires_grad_(False)
    if not learn_noise:
        for parameter in actor.explore_noise_net.parameters():
            parameter.requires_grad_(False)
    actor_parameters = [
        parameter for parameter in actor.parameters() if parameter.requires_grad
    ]
    fully_frozen = not actor_parameters
    actor_optimizer = (
        torch.optim.AdamW(actor_parameters, lr=experiment_cfg.actor_lr)
        if not fully_frozen
        else None
    )
    critic_optimizer = torch.optim.AdamW(
        critic.parameters(), lr=experiment_cfg.critic_lr
    )

    environment = FourGoalEnv(
        n_envs=experiment_cfg.n_envs,
        goal_centers=goal_centers,
        horizon=policy_cfg.episode_steps,
        reward_sigma=environment_cfg.reward_sigma,
        success_distance=environment_cfg.success_distance,
        seed=environment_seed,
        goal_weights=environment_cfg.get("goal_weights"),
    )
    maximum_units = policy_cfg.episode_steps // policy_cfg.chunk_length
    origin = torch.zeros(environment_cfg.observation_dim)
    history: list[dict[str, float]] = []

    # Pre-RL baseline for THIS seed's actor, stochastic, before any update.
    pre_rl_policy = make_flow_actor(
        actor, action_scale, environment_cfg, policy_cfg, experiment_cfg
    )
    pre_rl = noise_sensitivity_probe(
        policy=pre_rl_policy,
        goal_centers=goal_centers,
        environment_cfg=environment_cfg,
        policy_cfg=policy_cfg,
        n_episodes=evaluation_cfg.n_episodes,
        seed=9_000,
        min_attempts=evaluation_cfg.min_attempts,
    )
    pre_rl_record = {
        "actor_fingerprint": actor_fingerprint(actor),
        "goal_centers": goal_centers.tolist(),
        "SR": pre_rl["metrics_a"]["SR"],
        "SR_M": pre_rl["metrics_a"]["SR_M"],
        "mc@0.80_floored": pre_rl["metrics_a"]["mc@0.80_floored"],
        "H_norm": pre_rl["metrics_a"]["H_norm"],
        "H_succ_norm": pre_rl["metrics_a"]["H_succ_norm"],
        "mean_final_radius": pre_rl["metrics_a"]["mean_final_radius"],
        "attempts_per_mode": pre_rl["metrics_a"]["attempts_per_mode"],
        "mode_flip_rate": pre_rl["mode_flip_rate"],
        "endpoint_l2": pre_rl["endpoint_l2"],
        **{
            f"spread_{key}": value
            for key, value in endpoint_spread(
                pre_rl_policy,
                origin,
                n_samples=experiment_cfg.n_spread_samples,
                seed=9_001,
            ).items()
        },
        **{
            f"sigma_{key}": value
            for key, value in sigma_field_probe(
                actor,
                torch.from_numpy(goal_centers.astype("float32")) * 0.5,
                experiment_cfg.flow_steps,
            ).items()
            if key in ("sigma_min_bound", "sigma_max_bound")
        },
        "sigma_mean": float(
            sum(
                sigma_field_probe(
                    actor,
                    torch.from_numpy(goal_centers.astype("float32")) * 0.5,
                    experiment_cfg.flow_steps,
                )["per_step_mean"]
            )
            / experiment_cfg.flow_steps
        ),
    }

    for epoch in range(experiment_cfg.total_epochs):
        units = horizon_units(
            epoch=epoch,
            start=policy_cfg.curriculum_start_units,
            maximum=maximum_units,
            warmup_epochs=experiment_cfg.warmup_epochs,
            enabled=experiment_cfg.curriculum,
        )
        environment.horizon = units * policy_cfg.chunk_length
        rollout = collect_flow_rollout(
            actor=actor,
            environment=environment,
            n_units=units,
            action_scale=action_scale,
            chunk_length=policy_cfg.chunk_length,
            action_dim=environment_cfg.action_dim,
            cfg=experiment_cfg,
        )
        update = reinflow_update(
            actor=actor,
            critic=critic,
            actor_optimizer=actor_optimizer,
            critic_optimizer=critic_optimizer,
            rollout=rollout,
            cfg=experiment_cfg,
            update_actor=(
                not fully_frozen and epoch >= experiment_cfg.critic_warmup_epochs
            ),
            reference_model=reference_model,
            anchor_generator=anchor_generator,
        )

        # Rollout-derived metrics are curriculum-confounded; they carry a
        # `train_` prefix so they are never mistaken for the `eval_` numbers.
        train_metrics = compute_metrics(
            goal_centers=goal_centers,
            final_positions=rollout["end_position"][-1],
            successes=rollout["success"].max(axis=0).astype(bool),
            min_attempts=evaluation_cfg.min_attempts,
        )
        stochastic_policy = make_flow_actor(
            actor, action_scale, environment_cfg, policy_cfg, experiment_cfg
        )
        spread = endpoint_spread(
            stochastic_policy,
            origin,
            n_samples=experiment_cfg.n_spread_samples,
            seed=epoch,
        )

        record = {
            "epoch": epoch,
            "horizon_units": units,
            "r_env": float(rollout["r_env"].mean()),
            "clip_frac_randn": float(rollout["clip_frac_randn"]),
            "clip_frac_act_limit": float(rollout["clip_frac_act_limit"]),
            "success": float(rollout["success"].max(axis=0).mean()),
            "train_SR": train_metrics["SR"],
            "train_SR_M": train_metrics["SR_M"],
            "train_mcf": train_metrics["mc@0.80_floored"],
            "train_H_succ_norm": train_metrics["H_succ_norm"],
            "train_mean_final_radius": train_metrics["mean_final_radius"],
            **spread,
            **update,
        }

        is_last = epoch == experiment_cfg.total_epochs - 1
        if experiment_cfg.diagnostic_every and (
            epoch % experiment_cfg.diagnostic_every == 0 or is_last
        ):
            probe = noise_sensitivity_probe(
                policy=stochastic_policy,
                goal_centers=goal_centers,
                environment_cfg=environment_cfg,
                policy_cfg=policy_cfg,
                n_episodes=experiment_cfg.n_diagnostic_episodes,
                seed=10_000 + epoch,
                min_attempts=evaluation_cfg.min_attempts,
            )
            evaluated = probe["metrics_a"]
            record.update(
                {
                    "eval_SR": evaluated["SR"],
                    "eval_SR_M": evaluated["SR_M"],
                    "eval_mcf": evaluated["mc@0.80_floored"],
                    "eval_H_norm": evaluated["H_norm"],
                    "eval_H_succ_norm": evaluated["H_succ_norm"],
                    "eval_mean_final_radius": evaluated["mean_final_radius"],
                    "eval_mode_flip_rate": probe["mode_flip_rate"],
                    "eval_endpoint_l2": probe["endpoint_l2"],
                }
            )

        history.append(record)
        if log_every and (epoch % log_every == 0 or is_last):
            evaluation_note = ""
            if "eval_SR" in record:
                evaluation_note = (
                    f" eval_SR={record['eval_SR']:.3f}"
                    f" mcf={record['eval_mcf']:.2f}"
                    f" Hs={record['eval_H_succ_norm']:.3f}"
                    f" flip={record['eval_mode_flip_rate']:.3f}"
                )
            print(
                f"epoch={epoch:3d} units={units:2d} "
                f"r_env={record['r_env']:+.4f} "
                f"sigma=[{record['sigma_min']:.3f},{record['sigma_max']:.3f}] "
                f"conc={record['direction_concentration']:.3f} "
                f"kl={record['approx_kl']:.5f} "
                f"ratio={record['ratio_mean']:.4f}{evaluation_note}"
            )

    environment.horizon = policy_cfg.episode_steps
    return ReinFlowTrainingResult(
        actor=actor, critic=critic, history=history, pre_rl=pre_rl_record
    )
