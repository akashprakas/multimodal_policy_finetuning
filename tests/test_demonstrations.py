from four_goal_bmd.data.demonstrations import generate_demonstrations
from four_goal_bmd.envs.four_goals import make_goal_centers


def test_single_step_demonstration_shapes():
    dataset = generate_demonstrations(
        make_goal_centers(), 10, episode_steps=14, chunk_length=1, seed=2
    )
    assert dataset.states.shape == (140, 2)
    assert dataset.action_targets.shape == (140, 2)
    assert dataset.action_scale > 0


def test_chunked_demonstration_shapes():
    dataset = generate_demonstrations(
        make_goal_centers(), 10, episode_steps=16, chunk_length=4, seed=2
    )
    assert dataset.states.shape == (40, 2)
    assert dataset.action_targets.shape == (40, 8)
    assert dataset.chunk_length == 4
