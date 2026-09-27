from pathlib import Path

import numpy as np
from hydra import compose, initialize_config_dir

from four_goal_bmd.envs.four_goals import make_goal_centers
from four_goal_bmd.task_shift import finetuning_environment, shifted_environment

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _config(overrides):
    with initialize_config_dir(
        version_base="1.3", config_dir=str(PROJECT_ROOT / "configs")
    ):
        return compose(config_name="config", overrides=overrides)


def test_none_shift_is_inert_and_does_not_mutate_base_environment():
    cfg = _config(["shift=none"])
    environment_cfg = finetuning_environment(cfg)

    assert cfg.environment.goal_radius == 1.0
    assert environment_cfg.goal_radius == 1.0
    assert environment_cfg.success_distance == cfg.environment.success_distance
    assert environment_cfg is not cfg.environment


def test_dsrl_shift_changes_rl_radius_but_not_bc_radius():
    cfg = _config(["shift=dsrl_v1", "landscape=g5"])
    environment_cfg = finetuning_environment(cfg)

    assert cfg.environment.goal_radius == 1.0
    assert environment_cfg.goal_radius == 1.16
    goals = make_goal_centers(
        environment_cfg.n_goals,
        environment_cfg.goal_radius,
        cfg.landscape.rotation,
    )
    np.testing.assert_allclose(np.linalg.norm(goals, axis=1), 1.16)


def test_shifted_environment_returns_an_independent_copy():
    cfg = _config([])
    shifted = shifted_environment(cfg.environment, 1.2, 0.25)

    assert shifted.goal_radius == 1.2
    assert shifted.success_distance == 0.25
    assert cfg.environment.goal_radius == 1.0
    assert cfg.environment.success_distance == 0.30
