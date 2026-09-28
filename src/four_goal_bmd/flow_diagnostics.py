"""Diagnostics that separate sigma-collapse from mode-collapse.

Why this module exists. In this toy, actions are normalized by a single scalar
(`action_scale ~ 0.0517`), so a sigma of 0.10 displaces one executed action by
0.0052 position units against a goal separation of 1.41 and a success radius of
0.30. Injected sigma therefore cannot walk an episode from one goal to another
by addition. It reaches mode selection only indirectly, as a perturbation of
x_0 of magnitude ~ sigma * sqrt(K) against x_0 ~ N(0, I).

So the sigma floor keeps mode *selection* stochastic only while the velocity
field's x_0 response spans more than one direction -- and ReinFlow trains that
field. `sigma` pinned at its ceiling while `mode_flip_rate` goes to zero is the
signature the floor cannot prevent. Without this measurement a coverage failure
is uninterpretable.

Note on interpretation. A collapsed field is NOT insensitive to x_0: measured over
many states and x_0 draws, its ||d(a)/d(x_0)|| median is slightly *larger* than the
pretrained field's. What collapses is the response's direction -- the Jacobian
becomes near rank-one (leading singular fraction 0.78 -> 0.99). A Jacobian norm
cannot distinguish "stopped responding" from "responds along one ray", which is
why `jacobian_distribution` reports the leading singular fraction. Treat rank-one
geometry as a measured signature that co-occurs with collapse, not as a
demonstrated cause.
"""

from __future__ import annotations

import numpy as np
import torch

from four_goal_bmd.envs.four_goals import FourGoalEnv
from four_goal_bmd.metrics import compute_metrics
from four_goal_bmd.models.flow import FlowPolicyActor, NoisyFlowPolicy


def _rollout_with_fixed_initial_noise(
    policy: FlowPolicyActor,
    goal_centers: np.ndarray,
    environment_cfg,
    policy_cfg,
    n_episodes: int,
    initial_noises: list[torch.Tensor],
    injected_noise_seed: int,
    min_attempts: int,
) -> dict:
    """Evaluate with caller-supplied x_0 draws and a chosen injected-noise seed.

    Accounting mirrors `evaluation.evaluate_policy` exactly: first
    success-or-truncation position wins, so the environment's in-place reset at
    truncation cannot corrupt the recorded endpoint.
    """
    torch.manual_seed(injected_noise_seed)
    environment = FourGoalEnv(
        n_envs=n_episodes,
        goal_centers=goal_centers,
        horizon=policy_cfg.episode_steps,
        reward_sigma=environment_cfg.reward_sigma,
        success_distance=environment_cfg.success_distance,
        seed=injected_noise_seed,
        goal_weights=environment_cfg.get("goal_weights"),
    )
    observation = environment.reset()
    n_units = policy_cfg.episode_steps // policy_cfg.chunk_length
    successes = np.zeros(n_episodes, dtype=bool)
    done = np.zeros(n_episodes, dtype=bool)
    final_positions = np.zeros((n_episodes, 2), dtype=np.float32)

    for unit in range(n_units):
        state_tensor = torch.from_numpy(observation).float()
        with torch.no_grad():
            actions = policy(state_tensor, initial_noises[unit])
        _, _, success, truncated, end_position = environment.apply_action_chunk(
            actions.numpy()
        )
        newly_done = (success | truncated) & ~done
        final_positions[newly_done] = end_position[newly_done]
        done |= success | truncated
        successes |= success
        observation = environment.state.copy()

    metrics = compute_metrics(
        goal_centers=goal_centers,
        final_positions=final_positions,
        successes=successes,
        min_attempts=min_attempts,
    )
    difference = final_positions[:, None, :] - goal_centers[None, :, :]
    metrics["_assignments"] = np.argmin(np.sum(difference**2, axis=-1), axis=-1)
    metrics["_final_positions"] = final_positions
    return metrics


def noise_sensitivity_probe(
    policy: FlowPolicyActor,
    goal_centers: np.ndarray,
    environment_cfg,
    policy_cfg,
    n_episodes: int,
    seed: int,
    min_attempts: int,
) -> dict:
    """Paired-x_0 mode-flip rate.

    Two full evaluations share the *same* per-unit x_0 draws but use independent
    injected noise. Pairing x_0 is what isolates sigma's contribution: a bare
    "spread at a fixed state" cannot tell the two noise sources apart.

    Diagnostics must be observational. `_rollout_with_fixed_initial_noise` calls
    `torch.manual_seed`, so without `fork_rng` this probe would reseed the global
    generator: `diagnostic_every` would change the learned policy, and because
    the probe seed does not depend on the training seed, every training seed
    would continue from one shared randomness stream. `fork_rng` restores the
    caller's state on exit, including on exception.
    """
    with torch.random.fork_rng(devices=[]):
        return _noise_sensitivity_probe_unguarded(
            policy,
            goal_centers,
            environment_cfg,
            policy_cfg,
            n_episodes,
            seed,
            min_attempts,
        )


def _noise_sensitivity_probe_unguarded(
    policy: FlowPolicyActor,
    goal_centers: np.ndarray,
    environment_cfg,
    policy_cfg,
    n_episodes: int,
    seed: int,
    min_attempts: int,
) -> dict:
    n_units = policy_cfg.episode_steps // policy_cfg.chunk_length
    generator = torch.Generator().manual_seed(seed)
    initial_noises = [
        torch.randn(n_episodes, policy.noise_dim, generator=generator)
        for _ in range(n_units)
    ]

    first = _rollout_with_fixed_initial_noise(
        policy,
        goal_centers,
        environment_cfg,
        policy_cfg,
        n_episodes,
        initial_noises,
        seed + 1,
        min_attempts,
    )
    second = _rollout_with_fixed_initial_noise(
        policy,
        goal_centers,
        environment_cfg,
        policy_cfg,
        n_episodes,
        initial_noises,
        seed + 2,
        min_attempts,
    )

    flips = first["_assignments"] != second["_assignments"]
    endpoint_l2 = np.linalg.norm(
        first["_final_positions"] - second["_final_positions"], axis=-1
    )
    for metrics in (first, second):
        metrics.pop("_assignments")
        metrics.pop("_final_positions")
    return {
        "mode_flip_rate": float(flips.mean()),
        "endpoint_l2": float(endpoint_l2.mean()),
        "metrics_a": first,
        "metrics_b": second,
    }


def sigma_field_probe(
    actor: NoisyFlowPolicy, states: torch.Tensor, flow_steps: int
) -> dict:
    """sigma statistics per chain step, to confirm the floor was never touched."""
    per_step_mean: list[float] = []
    per_step_min: list[float] = []
    per_step_max: list[float] = []
    with torch.no_grad():
        dummy = torch.zeros(states.shape[0], actor.model.output_dim)
        for index in range(flow_steps):
            t = torch.full((states.shape[0],), index / flow_steps)
            _, sigma = actor.forward(dummy, t, states, learn_exploration_noise=False)
            per_step_mean.append(float(sigma.mean()))
            per_step_min.append(float(sigma.min()))
            per_step_max.append(float(sigma.max()))
        origin = torch.zeros(1, states.shape[-1])
        _, origin_sigma = actor.forward(
            torch.zeros(1, actor.model.output_dim),
            torch.zeros(1),
            origin,
            learn_exploration_noise=False,
        )
    return {
        "per_step_mean": per_step_mean,
        "per_step_min": per_step_min,
        "per_step_max": per_step_max,
        "origin_sigma": origin_sigma[0].tolist(),
        "sigma_min_bound": actor.sigma_min,
        "sigma_max_bound": actor.sigma_max,
    }


def source_attribution_probe(
    actor: NoisyFlowPolicy,
    action_scale: float,
    goal_centers: np.ndarray,
    environment_cfg,
    policy_cfg,
    experiment_cfg,
    n_samples: int,
    seed: int,
    n_jacobian_states: int = 32,
    n_jacobian_noises: int = 64,
    state_radius: float = 0.8,
) -> dict:
    """Attribute output diversity to x_0 versus injected noise, separately.

    `noise_sensitivity_probe` fixes x_0 and varies injected noise, which shows
    that injected noise does not change the mode. It does not, on its own,
    isolate the *other* arm. This does:

      * arm A -- vary x_0, injected noise DISABLED (deterministic integration).
        Measures how much the velocity field still depends on its input noise.
      * arm B -- fix x_0 at a single draw, vary injected noise only.

    Also reports the Frobenius norm of the endpoint Jacobian d(action)/d(x_0),
    which measures x_0-sensitivity directly rather than through an outcome.

    Checkpoint-only: no retraining, no rollout, and no effect on the caller's
    RNG state (the stochastic arms draw from the global generator, so the body
    runs inside `fork_rng`).
    """
    with torch.random.fork_rng(devices=[]):
        return _source_attribution_probe_unguarded(
            actor,
            action_scale,
            goal_centers,
            environment_cfg,
            policy_cfg,
            experiment_cfg,
            n_samples,
            seed,
            n_jacobian_states,
            n_jacobian_noises,
            state_radius,
        )


def _source_attribution_probe_unguarded(
    actor,
    action_scale,
    goal_centers,
    environment_cfg,
    policy_cfg,
    experiment_cfg,
    n_samples,
    seed,
    n_jacobian_states,
    n_jacobian_noises,
    state_radius,
):
    from four_goal_bmd.algorithms.reinflow import make_flow_actor

    state = torch.zeros(1, environment_cfg.observation_dim)
    generator = torch.Generator().manual_seed(seed)
    noises = torch.randn(n_samples, actor.model.output_dim, generator=generator)

    def summarize(actions: torch.Tensor) -> dict:
        first = actions[:, 0, :]
        angles = torch.atan2(first[:, 1], first[:, 0])
        concentration = float(
            torch.sqrt(angles.cos().mean() ** 2 + angles.sin().mean() ** 2)
        )
        # Direction sectors corresponding to goals: which goal's bearing the
        # single-step action points nearest to. This is NOT a goal-reaching
        # histogram -- no episode is rolled out.
        goal_angles = torch.from_numpy(
            np.arctan2(goal_centers[:, 1], goal_centers[:, 0]).astype("float32")
        )
        difference = (angles[:, None] - goal_angles[None, :] + np.pi) % (
            2 * np.pi
        ) - np.pi
        assignment = difference.abs().argmin(dim=-1)
        histogram = torch.bincount(assignment, minlength=len(goal_centers)).tolist()
        return {
            "endpoint_std": float(first.std()),
            "endpoint_norm_std": float(first.norm(dim=-1).std()),
            "direction_concentration": concentration,
            "direction_sector_histogram": histogram,
            "distinct_direction_sectors": int(sum(1 for c in histogram if c > 0)),
        }

    deterministic = make_flow_actor(
        actor,
        action_scale,
        environment_cfg,
        policy_cfg,
        experiment_cfg,
        deterministic=True,
    )
    stochastic = make_flow_actor(
        actor, action_scale, environment_cfg, policy_cfg, experiment_cfg
    )

    with torch.no_grad():
        # Arm A: x_0 varies, no injected noise.
        vary_x0 = summarize(deterministic(state.repeat(n_samples, 1), noises))
        # Arm B: x_0 fixed, injected noise varies.
        fixed = noises[:1].repeat(n_samples, 1)
        vary_noise = summarize(stochastic(state.repeat(n_samples, 1), fixed))
        # Both, for reference.
        vary_both = summarize(stochastic(state.repeat(n_samples, 1), noises))

    jacobian = jacobian_distribution(
        actor,
        action_scale,
        experiment_cfg.flow_steps,
        observation_dim=environment_cfg.observation_dim,
        n_states=n_jacobian_states,
        n_noises=n_jacobian_noises,
        state_radius=state_radius,
        seed=seed + 1,
    )

    return {
        "vary_x0_only": vary_x0,
        "vary_injected_noise_only": vary_noise,
        "vary_both": vary_both,
        "jacobian_action_wrt_x0_fro": jacobian,
    }


def jacobian_distribution(
    actor: NoisyFlowPolicy,
    action_scale: float,
    flow_steps: int,
    observation_dim: int,
    n_states: int = 32,
    n_noises: int = 64,
    state_radius: float = 0.8,
    seed: int = 0,
) -> dict:
    """Distribution of ||d(action)/d(x_0)||_F over many states and x_0 draws.

    A single-point Jacobian could be a local accident. States are drawn uniformly
    from the disc of radius `state_radius` (the region episodes actually visit;
    goals sit at radius 1.0), and x_0 from the standard normal the policy uses.
    """
    generator = torch.Generator().manual_seed(seed)
    radius = state_radius * torch.rand(n_states, 1, generator=generator).sqrt()
    angle = 2 * np.pi * torch.rand(n_states, 1, generator=generator)
    states = torch.cat([radius * angle.cos(), radius * angle.sin()], dim=-1)
    if observation_dim != 2:
        states = states[:, :observation_dim]

    norms: list[float] = []
    leading_fractions: list[float] = []
    degenerate = 0
    for state_index in range(n_states):
        state = states[state_index : state_index + 1]
        noises = torch.randn(n_noises, actor.model.output_dim, generator=generator)
        for noise_index in range(n_noises):
            probe = noises[noise_index : noise_index + 1].clone()
            jacobian = torch.autograd.functional.jacobian(
                lambda z: actor.model.euler_sample_differentiable(z, state, flow_steps),
                probe,
            )
            matrix = jacobian.reshape(actor.model.output_dim, -1)
            norms.append(float(matrix.norm()) * action_scale)
            # Anisotropy. The norm alone cannot distinguish "insensitive to x_0"
            # from "sensitive to x_0 but only along one direction" -- and it is
            # the second that a collapsed policy exhibits. The leading fraction
            # is s_1^2 / sum(s_i^2): 1/output_dim means isotropic, 1.0 means the
            # Jacobian is rank one, so every x_0 perturbation moves the action
            # along a single ray.
            singular = torch.linalg.svdvals(matrix)
            total = float((singular**2).sum())
            if total > 1e-24:
                leading_fractions.append(float(singular[0] ** 2) / total)
            else:
                # An exactly-zero Jacobian has no orientation, so the leading
                # fraction is undefined rather than 1.0. Excluded and counted;
                # without this guard one such point made the whole summary nan.
                degenerate += 1

    def summarize(values: np.ndarray) -> dict:
        return {
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "p10": float(np.percentile(values, 10)),
            "p90": float(np.percentile(values, 90)),
            "min": float(values.min()),
            "max": float(values.max()),
        }

    leading = (
        summarize(np.asarray(leading_fractions))
        if leading_fractions
        else {
            key: float("nan") for key in ("mean", "median", "p10", "p90", "min", "max")
        }
    )
    return {
        **summarize(np.asarray(norms)),
        "leading_singular_fraction": leading,
        "degenerate_jacobians": degenerate,
        "isotropic_leading_fraction": 1.0 / actor.model.output_dim,
        "n_states": n_states,
        "n_noises_per_state": n_noises,
        "state_radius": state_radius,
    }


def retention_probe(
    actor,
    reference_model,
    action_scale: float,
    goal_centers: np.ndarray,
    observation_dim: int,
    flow_steps: int,
    n_states: int = 32,
    n_noises: int = 64,
    state_radius: float = 0.8,
    seed: int = 0,
) -> dict:
    """Paired-x_0 retention: does the same x_0 still choose the same goal?

    For each (state, x_0), integrate deterministically under the fine-tuned field
    and under the frozen pretrained field, and compare the resulting actions.
    Measures whether paired initial-noise samples retain their reference action
    sectors while the fine-tuned map adapts scale:

        retention      P( goal(a_post(s,x0)) == goal(a_BC(s,x0)) )
        angular_change |angle(a_post) - angle(a_BC)|, radians
        radial_change  ||a_post|| / ||a_BC||

    High retention with a small angular change and a radial change away from 1.0 is
    the signature the anchor is designed to produce.
    """
    generator = torch.Generator().manual_seed(seed)
    radius = state_radius * torch.rand(n_states, 1, generator=generator).sqrt()
    angle = 2 * np.pi * torch.rand(n_states, 1, generator=generator)
    states = torch.cat([radius * angle.cos(), radius * angle.sin()], dim=-1)
    if observation_dim != 2:
        states = states[:, :observation_dim]

    goal_angles = torch.from_numpy(
        np.arctan2(goal_centers[:, 1], goal_centers[:, 0]).astype("float32")
    )

    def sector(actions: torch.Tensor) -> torch.Tensor:
        bearing = torch.atan2(actions[:, 1], actions[:, 0])
        difference = (bearing[:, None] - goal_angles[None, :] + np.pi) % (
            2 * np.pi
        ) - np.pi
        return difference.abs().argmin(dim=-1)

    retained, angular, radial = [], [], []
    with torch.no_grad():
        for index in range(n_states):
            state = states[index : index + 1].repeat(n_noises, 1)
            noises = torch.randn(n_noises, actor.model.output_dim, generator=generator)
            post = (
                actor.model.euler_sample_differentiable(noises, state, flow_steps)[
                    :, :2
                ]
                * action_scale
            )
            before = (
                reference_model.euler_sample_differentiable(noises, state, flow_steps)[
                    :, :2
                ]
                * action_scale
            )
            retained.append((sector(post) == sector(before)).float())
            post_angle = torch.atan2(post[:, 1], post[:, 0])
            before_angle = torch.atan2(before[:, 1], before[:, 0])
            delta = (post_angle - before_angle + np.pi) % (2 * np.pi) - np.pi
            angular.append(delta.abs())
            radial.append(post.norm(dim=-1) / (before.norm(dim=-1) + 1e-8))

    retained = torch.cat(retained).numpy()
    angular = torch.cat(angular).numpy()
    radial = torch.cat(radial).numpy()
    return {
        "retention": float(retained.mean()),
        "angular_change_mean_rad": float(angular.mean()),
        "angular_change_median_rad": float(np.median(angular)),
        "angular_change_p90_rad": float(np.percentile(angular, 90)),
        "radial_change_mean": float(radial.mean()),
        "radial_change_median": float(np.median(radial)),
        "n_states": n_states,
        "n_noises_per_state": n_noises,
    }


def endpoint_spread(
    policy: FlowPolicyActor, state: torch.Tensor, n_samples: int, seed: int
) -> dict:
    """Spread of the executed action at one fixed state, over fresh x_0 draws.

    Cheap enough to log every epoch. `direction_concentration` is the circular
    concentration |mean(exp(i*theta))| of the action directions: 1.0 means every
    x_0 produces the same heading (collapsed), 0.0 means uniformly spread.
    """
    generator = torch.Generator().manual_seed(seed)
    repeated = state.reshape(1, -1).repeat(n_samples, 1)
    noise = torch.randn(n_samples, policy.noise_dim, generator=generator)
    # The policy samples injected noise from the global generator; keep that off
    # the caller's stream.
    with torch.random.fork_rng(devices=[]), torch.no_grad():
        torch.manual_seed(seed)
        actions = policy(repeated, noise)[:, 0, :]
    angles = torch.atan2(actions[:, 1], actions[:, 0])
    concentration = float(
        torch.sqrt(angles.cos().mean() ** 2 + angles.sin().mean() ** 2)
    )
    return {
        "endpoint_std": float(actions.std()),
        "direction_concentration": concentration,
    }
