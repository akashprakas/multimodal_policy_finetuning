from __future__ import annotations

from collections.abc import Callable

import torch

from four_goal_bmd.data.demonstrations import DemonstrationDataset
from four_goal_bmd.models.diffusion import DiffusionModel


def train_diffusion_bc(
    model: DiffusionModel,
    dataset: DemonstrationDataset,
    training_steps: int,
    batch_size: int,
    learning_rate: float,
    log_every: int = 0,
    logger: Callable[[str], None] = print,
) -> list[float]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    history: list[float] = []
    for step in range(training_steps):
        states, normalized_actions = dataset.sample(batch_size)
        loss = model.loss(normalized_actions, states)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        history.append(float(loss.item()))
        if log_every and (step % log_every == 0 or step == training_steps - 1):
            logger(f"BC step={step:5d} loss={loss.item():.6f}")

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    return history
