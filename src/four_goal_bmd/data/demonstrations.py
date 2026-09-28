from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class DemonstrationDataset:
    states: torch.Tensor
    action_targets: torch.Tensor
    goal_indices: np.ndarray
    action_scale: float
    chunk_length: int
    validation_states: torch.Tensor | None = None
    validation_action_targets: torch.Tensor | None = None

    def sample(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        indices = torch.randint(0, self.states.shape[0], (batch_size,))
        return self.states[indices], self.action_targets[indices]


def generate_demonstrations(
    goal_centers: np.ndarray,
    n_demonstrations: int,
    episode_steps: int,
    chunk_length: int,
    seed: int,
    action_noise_std: float = 0.01,
    validation_fraction: float = 0.0,
) -> DemonstrationDataset:
    """Generate the procedural expert used by the historical experiment."""
    if episode_steps % chunk_length != 0:
        raise ValueError("episode_steps must be divisible by chunk_length.")
    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError("validation_fraction must lie in [0, 1).")
    rng = np.random.default_rng(seed)
    goal_indices = rng.integers(0, len(goal_centers), size=n_demonstrations)
    targets = goal_centers[goal_indices]
    positions = np.zeros((n_demonstrations, 2), dtype=np.float32)
    boundary_states: list[np.ndarray] = []
    action_chunks: list[np.ndarray] = []

    for chunk_index in range(episode_steps // chunk_length):
        boundary_states.append(positions.copy())
        actions = []
        for substep in range(chunk_length):
            global_step = chunk_index * chunk_length + substep
            remaining = episode_steps - global_step
            action = (targets - positions) / remaining
            action += rng.normal(scale=action_noise_std, size=action.shape).astype(
                np.float32
            )
            actions.append(action.copy())
            positions = positions + action
        action_chunks.append(np.concatenate(actions, axis=-1))

    states = np.stack(boundary_states, axis=1).reshape(-1, 2).astype(np.float32)
    targets_flat = (
        np.stack(action_chunks, axis=1).reshape(-1, chunk_length * 2).astype(np.float32)
    )
    action_scale = float(targets_flat.std())
    state_tensor = torch.from_numpy(states)
    normalized_targets = torch.from_numpy(targets_flat) / action_scale
    n_validation = int(round(validation_fraction * n_demonstrations))
    if validation_fraction > 0.0 and n_validation < 1:
        raise ValueError(
            "validation_fraction is too small to hold out one demonstration."
        )
    if n_validation:
        n_chunks = episode_steps // chunk_length
        split = (n_demonstrations - n_validation) * n_chunks
        return DemonstrationDataset(
            states=state_tensor[:split],
            action_targets=normalized_targets[:split],
            goal_indices=goal_indices[: n_demonstrations - n_validation],
            action_scale=action_scale,
            chunk_length=chunk_length,
            validation_states=state_tensor[split:],
            validation_action_targets=normalized_targets[split:],
        )
    return DemonstrationDataset(
        states=state_tensor,
        action_targets=normalized_targets,
        goal_indices=goal_indices,
        action_scale=action_scale,
        chunk_length=chunk_length,
    )
