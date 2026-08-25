from __future__ import annotations

import numpy as np
import torch

from four_goal_bmd.envs.four_goals import FourGoalEnv
from four_goal_bmd.evaluation import evaluate_policy
from four_goal_bmd.metrics import format_metrics


def expert_replay(
    goal_centers: np.ndarray,
    environment_cfg,
    policy_cfg,
    n_episodes: int = 256,
    seed: int = 777,
) -> float:
    environment = FourGoalEnv(
        n_episodes,
        goal_centers,
        policy_cfg.episode_steps,
        environment_cfg.reward_sigma,
        environment_cfg.success_distance,
        seed,
    )
    environment.reset()
    rng = np.random.default_rng(0)
    targets = goal_centers[rng.integers(0, 4, n_episodes)]
    successes = np.zeros(n_episodes, dtype=bool)
    for step in range(policy_cfg.episode_steps):
        action = (targets - environment.state) / (policy_cfg.episode_steps - step)
        _, _, success, _ = environment.step(action)
        successes |= success
    return float(successes.mean())


def grid_search_reachability(
    frozen_policy,
    goal_centers: np.ndarray,
    environment_cfg,
    policy_cfg,
    grid_size: int,
    w_limit: float,
) -> list[dict]:
    if policy_cfg.chunk_length != 1 or frozen_policy.noise_dim != 2:
        raise ValueError(
            "Cartesian grid search is defined for the 2D single-step policy."
        )
    axis = torch.linspace(-w_limit, w_limit, grid_size)
    grid = torch.stack(torch.meshgrid(axis, axis, indexing="ij"), dim=-1).reshape(-1, 2)
    results = []
    for goal_index, goal in enumerate(goal_centers):
        environment = FourGoalEnv(
            1,
            goal_centers,
            policy_cfg.episode_steps,
            environment_cfg.reward_sigma,
            environment_cfg.success_distance,
            seed=1,
        )
        observation = environment.reset()
        reached = False
        final_position = np.zeros(2, dtype=np.float32)
        for _ in range(policy_cfg.episode_steps):
            repeated_state = torch.from_numpy(observation).float().repeat(len(grid), 1)
            with torch.no_grad():
                actions = frozen_policy(repeated_state, grid)[:, 0, :]
            next_positions = repeated_state + actions
            best = (next_positions - torch.from_numpy(goal)).norm(dim=-1).argmin()
            chosen = actions[best : best + 1].numpy()
            final_position = environment.state[0] + chosen[0]
            observation, _, success, _ = environment.step(chosen)
            reached |= bool(success[0])
        results.append(
            {
                "goal": goal_index,
                "reached": reached,
                "final_position": final_position.tolist(),
                "final_distance": float(np.linalg.norm(final_position - goal)),
            }
        )
    return results


def hold_noise_probe(
    frozen_policy,
    goal_centers: np.ndarray,
    environment_cfg,
    policy_cfg,
    hold_lengths: list[int],
    n_episodes: int,
    evaluation_seed: int,
    min_attempts: int,
) -> list[dict]:
    if policy_cfg.chunk_length != 1:
        raise ValueError("The hold-w probe is intended for the single-step policy.")
    rows = []
    for hold_length in hold_lengths:
        metrics = evaluate_policy(
            frozen_policy=frozen_policy,
            goal_centers=goal_centers,
            environment_cfg=environment_cfg,
            policy_cfg=policy_cfg,
            n_episodes=n_episodes,
            evaluation_seed=evaluation_seed,
            min_attempts=min_attempts,
            hold_noise_steps=hold_length,
        )
        rows.append({"hold_length": hold_length, **metrics})
        print(f"hold={hold_length:2d} {format_metrics(metrics)}")
    return rows
