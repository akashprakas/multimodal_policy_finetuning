import numpy as np

from four_goal_bmd.envs.four_goals import FourGoalEnv, make_goal_centers


def test_goal_centers_are_on_unit_circle():
    goals = make_goal_centers()
    assert goals.shape == (4, 2)
    np.testing.assert_allclose(np.linalg.norm(goals, axis=-1), 1.0, atol=1e-6)


def test_environment_resets_on_truncation_and_keeps_pre_reset_position():
    env = FourGoalEnv(1, make_goal_centers(), horizon=1)
    observation, _, _, truncated = env.step(np.array([[0.1, 0.0]], np.float32))
    assert truncated.tolist() == [True]
    np.testing.assert_allclose(env.last_pre_reset, [[0.1, 0.0]])
    np.testing.assert_allclose(observation, [[0.0, 0.0]])


def test_action_chunk_records_each_visited_state():
    env = FourGoalEnv(1, make_goal_centers(), horizon=4)
    actions = np.array([[[0.1, 0.0], [0.1, 0.0]]], dtype=np.float32)
    visited, rewards, _, _, end = env.apply_action_chunk(actions)
    assert visited.shape == (1, 2, 2)
    assert rewards.shape == (1, 2)
    np.testing.assert_allclose(visited[0, :, 0], [0.1, 0.2])
    np.testing.assert_allclose(end, [[0.2, 0.0]])
