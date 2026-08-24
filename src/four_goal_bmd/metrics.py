from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np

METRIC_KEYS = (
    "SR",
    "SR_M",
    "mc@0.80",
    "mc@0.80_floored",
    "H",
    "H_norm",
    "H_succ",
    "H_succ_norm",
    "mean_final_radius",
)


def compute_metrics(
    goal_centers: np.ndarray,
    final_positions: np.ndarray,
    successes: np.ndarray,
    min_attempts: int,
    latents: np.ndarray | None = None,
    k_latent: int | None = None,
) -> dict:
    n_goals = len(goal_centers)
    difference = final_positions[:, None, :] - goal_centers[None, :, :]
    assignments = np.argmin(np.sum(difference**2, axis=-1), axis=-1)
    attempts = np.bincount(assignments, minlength=n_goals)
    success_counts = np.bincount(assignments[successes], minlength=n_goals).astype(
        np.float64
    )
    per_mode_success = np.zeros(n_goals, dtype=np.float64)
    for mode in range(n_goals):
        mask = assignments == mode
        per_mode_success[mode] = successes[mask].mean() if mask.any() else 0.0

    attempt_probability = attempts[attempts > 0] / attempts.sum()
    entropy = float(-(attempt_probability * np.log(attempt_probability)).sum())
    if success_counts.sum() > 0:
        success_probability = success_counts[success_counts > 0] / success_counts.sum()
        success_entropy = float(
            -(success_probability * np.log(success_probability)).sum()
        )
    else:
        success_entropy = float("nan")

    result = {
        "SR": float(successes.mean()),
        "SR_M": float(per_mode_success.mean()),
        "mc@0.80": float((per_mode_success >= 0.80).sum()),
        "mc@0.80_floored": float(
            ((per_mode_success >= 0.80) & (attempts >= min_attempts)).sum()
        ),
        "H": entropy,
        "H_norm": entropy / math.log(n_goals),
        "H_succ": success_entropy,
        "H_succ_norm": success_entropy / math.log(n_goals),
        "mean_final_radius": float(np.linalg.norm(final_positions, axis=-1).mean()),
        "attempts_per_mode": attempts.astype(int).tolist(),
        "successes_per_mode": success_counts.astype(int).tolist(),
        "sr_per_mode": per_mode_success.round(3).tolist(),
        "episodes": int(len(successes)),
        "n_goals": n_goals,
    }
    if latents is not None:
        if k_latent is None:
            raise ValueError("k_latent is required when latents are supplied.")
        table = np.zeros((k_latent, n_goals), dtype=int)
        for latent, mode in zip(latents, assignments):
            table[int(latent), int(mode)] += 1
        result["z_to_mode"] = table.tolist()
    return result


def aggregate_metrics(metrics: Iterable[dict]) -> tuple[dict, dict]:
    rows = list(metrics)
    mean = {}
    std = {}
    for key in METRIC_KEYS:
        values = np.asarray([row[key] for row in rows], dtype=np.float64)
        if np.isnan(values).all():
            mean[key] = float("nan")
            std[key] = float("nan")
        else:
            mean[key] = float(np.nanmean(values))
            std[key] = float(np.nanstd(values))
    return mean, std


def format_metrics(metrics: dict) -> str:
    return (
        f"SR={metrics['SR']:.3f} SR_M={metrics['SR_M']:.3f} "
        f"mcf={metrics['mc@0.80_floored']:.2f}/{metrics['n_goals']} "
        f"H/logM={metrics['H_norm']:.3f} "
        f"Hs/logM={metrics['H_succ_norm']:.3f} "
        f"radius={metrics['mean_final_radius']:.3f}"
    )
