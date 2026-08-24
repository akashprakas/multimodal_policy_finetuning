import numpy as np

from four_goal_bmd.envs.four_goals import make_goal_centers
from four_goal_bmd.metrics import compute_metrics


def test_collapsed_success_has_one_covered_mode_and_zero_entropy():
    goals = make_goal_centers()
    positions = np.repeat(goals[0][None, :], 20, axis=0)
    successes = np.ones(20, dtype=bool)
    metrics = compute_metrics(goals, positions, successes, min_attempts=5)
    assert metrics["SR"] == 1.0
    assert metrics["mc@0.80_floored"] == 1.0
    assert abs(metrics["H_norm"]) < 1e-12


def test_balanced_success_has_maximum_entropy():
    goals = make_goal_centers()
    positions = np.repeat(goals, 5, axis=0)
    successes = np.ones(20, dtype=bool)
    metrics = compute_metrics(goals, positions, successes, min_attempts=5)
    assert metrics["mc@0.80_floored"] == 4.0
    assert abs(metrics["H_norm"] - 1.0) < 1e-12
    assert abs(metrics["H_succ_norm"] - 1.0) < 1e-12
