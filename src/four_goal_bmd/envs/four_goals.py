import numpy as np


def make_goal_centers(
    n_goals: int = 4, radius: float = 1.0, rotation: float = 0.0
) -> np.ndarray:
    """Make goal centers in a circle.

    Args:
        n_goals: Number of goals.
        radius: Radius of the circle.
        rotation: Rotation of the circle in radians.

    Returns:
        Goal centers as a numpy array of shape (n_goals, 2).
    """
    angles = np.linspace(0, 2 * np.pi, n_goals, endpoint=False) + rotation
    x = radius * np.cos(angles)
    y = radius * np.sin(angles)
    return np.stack([x, y], axis=1)


class FourGoalEnv:
    """Vectorized 2D navigation environment with four Gaussian goals regions"""

    def __init__(
        self,
        n_envs: int,
        goal_centers: np.ndarray,
        horizon: int,
        reward_sigma: float = 0.25,
        success_distance: float = 0.30,
        seed: int = 0,
        goal_weights: np.ndarray | None = None,
    ):
        """Initialize the environment.

        Args:
            n_envs: Number of parallel environments.
            goal_centers: Goal centers as a numpy array of shape (n_goals, 2).
            horizon: Maximum number of steps per episode.
            reward_sigma: Standard deviation of the Gaussian reward function.
            success_distance: Distance threshold for success.
            seed: Random seed for reproducibility.
        """
        self.n_envs = n_envs
        self.goal_centers = goal_centers
        self.goal_weights = (
            None if goal_weights is None else np.asarray(goal_weights, dtype=np.float32)
        )
        if self.goal_weights is not None:
            if self.goal_weights.shape != (len(goal_centers),):
                raise ValueError(
                    f"goal_weights must have shape {(len(goal_centers),)}, "
                    f"got {self.goal_weights.shape}"
                )
            if (self.goal_weights <= 0).any():
                raise ValueError("goal_weights must be strictly positive.")
        self.horizon = horizon
        self.reward_sigma = reward_sigma
        self.success_distance = success_distance
        self.rng = np.random.default_rng(seed)
        self.state: np.ndarray
        self.steps: np.ndarray
        self.last_pre_reset: np.ndarray
        self.reset()

    def reset(self) -> np.ndarray:
        self.state = np.zeros((self.n_envs, 2), dtype=np.float32)
        self.steps = np.zeros(self.n_envs, dtype=np.int32)
        self.last_pre_reset = self.state.copy()
        return self.state.copy()

    def reward(self, position: np.ndarray) -> np.ndarray:
        difference = position[:, None, :] - self.goal_centers[None, :, :]
        squared_distance = np.sum(difference**2, axis=-1)
        bumps = np.exp(-squared_distance / (2 * self.reward_sigma**2))
        if self.goal_weights is None:
            return bumps.sum(axis=-1)
        return (bumps * self.goal_weights[None, :]).sum(axis=-1)

    def nearest_goal(self, position: np.ndarray) -> np.ndarray:
        difference = position[:, None, :] - self.goal_centers[None, :, :]
        return np.argmin(np.sum(difference**2, axis=-1), axis=-1)

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (self.n_envs, 2):
            raise ValueError(
                f"Expected action shape {(self.n_envs, 2)}, got {action.shape}"
            )
        self.state = self.state + action
        self.last_pre_reset = self.state.copy()
        reward = self.reward(self.state)
        distances = np.linalg.norm(
            self.state[:, None, :] - self.goal_centers[None, :, :], axis=-1
        )
        success = distances.min(axis=-1) < self.success_distance
        self.steps += 1
        truncated = self.steps >= self.horizon
        if truncated.any():
            self.state[truncated] = 0.0
            self.steps[truncated] = 0
        return self.state.copy(), reward, success, truncated

    def apply_action_chunk(self, actions: np.ndarray):
        """Execute `[n_envs, chunk_length, 2]` actions open-loop."""
        actions = np.asarray(actions, dtype=np.float32)
        if (
            actions.ndim != 3
            or actions.shape[0] != self.n_envs
            or actions.shape[2] != 2
        ):
            raise ValueError("Action chunk must have shape [n_envs, chunk_length, 2].")
        chunk_length = actions.shape[1]
        visited = np.zeros((self.n_envs, chunk_length, 2), dtype=np.float32)
        rewards = np.zeros((self.n_envs, chunk_length), dtype=np.float32)
        success_any = np.zeros(self.n_envs, dtype=bool)
        truncated_any = np.zeros(self.n_envs, dtype=bool)
        end_position = np.zeros((self.n_envs, 2), dtype=np.float32)
        ended = np.zeros(self.n_envs, dtype=bool)

        for index in range(chunk_length):
            _, reward, success, truncated = self.step(actions[:, index, :])
            visited[:, index, :] = self.last_pre_reset
            rewards[:, index] = reward
            newly_ended = (success | truncated) & ~ended
            end_position[newly_ended] = self.last_pre_reset[newly_ended]
            ended |= success | truncated
            success_any |= success
            truncated_any |= truncated

        end_position[~ended] = visited[~ended, -1, :]
        return visited, rewards, success_any, truncated_any, end_position
