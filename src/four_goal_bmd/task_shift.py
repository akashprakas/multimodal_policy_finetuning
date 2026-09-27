from __future__ import annotations

import copy

from omegaconf import DictConfig, open_dict


def shifted_environment(
    environment_cfg: DictConfig,
    goal_radius: float | None,
    success_distance: float | None,
) -> DictConfig:
    """Return an environment copy with a fine-tuning/evaluation shift applied.

    Call this only after BC pretraining. Keeping the input immutable is important:
    the frozen BC policy must always be trained on the original demonstration task.
    """
    shifted = copy.deepcopy(environment_cfg)
    with open_dict(shifted):
        if goal_radius is not None:
            shifted.goal_radius = float(goal_radius)
        if success_distance is not None:
            shifted.success_distance = float(success_distance)
    return shifted


def finetuning_environment(cfg: DictConfig) -> DictConfig:
    """Resolve the environment used by RL and evaluation from the full config."""
    shift = cfg.get("shift", None)
    if shift is None:
        return copy.deepcopy(cfg.environment)
    return shifted_environment(
        cfg.environment,
        shift.get("goal_radius"),
        shift.get("success_distance"),
    )
