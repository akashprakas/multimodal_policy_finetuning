"""Behavior cloning for the rectified-flow policy.

Mirrors `train_diffusion_bc` -- same optimizer, no LR schedule, no gradient
clipping, no EMA -- with two deliberate differences:

  * it does NOT freeze the model afterwards, because ReinFlow fine-tunes it;
  * it optionally reports a deterministic held-out loss.

The held-out number is a within-model overfitting check only. Diffusion BC
minimizes epsilon-MSE and flow BC minimizes velocity-MSE; the two differ in
target and scale and are not cross-comparable. Cross-model fairness is
established behaviorally, by the BC gate in `flow_diagnostics`.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from four_goal_bmd.models.flow import FlowModel

from four_goal_bmd.data.demonstrations import DemonstrationDataset


def flow_validation_loss(
    model: FlowModel,
    states: torch.Tensor,
    targets: torch.Tensor,
    t_grid: int = 8,
    seed: int = 0,
) -> float:
    """Deterministic held-out velocity MSE.

    A fixed generator draws one x_0 per (row, t) and `t` is the fixed grid
    {(j + 0.5) / t_grid}, so repeated calls during training are comparable rather
    than differing by resampling noise.
    """
    generator = torch.Generator().manual_seed(seed)
    total = 0.0
    with torch.no_grad():
        for index in range(t_grid):
            t = torch.full((targets.shape[0],), (index + 0.5) / t_grid)
            initial_noise = torch.randn(
                targets.shape, generator=generator, dtype=targets.dtype
            )
            total += float(model.loss_with(targets, states, t, initial_noise))
    return total / t_grid


def train_flow_bc(
    model: FlowModel,
    dataset: DemonstrationDataset,
    training_steps: int,
    batch_size: int,
    learning_rate: float,
    validation_every: int = 0,
    validation_t_grid: int = 8,
    validation_seed: int = 0,
    cosine_schedule: bool = False,
    minimum_learning_rate: float = 1e-5,
    log_every: int = 0,
    logger: Callable[[str], None] = print,
) -> dict:
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = None
    if cosine_schedule:
        # A schedule is a deviation from `train_diffusion_bc`, which has none.
        # It is on the BC-gate tuning grid because the flow objective is
        # noticeably learning-rate limited at the step counts this toy needs;
        # the deviation is reported alongside the results.
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=training_steps, eta_min=minimum_learning_rate
        )
    has_validation = dataset.validation_states is not None
    history: list[float] = []
    validation: list[dict[str, float]] = []

    def record_validation(step: int) -> None:
        loss = flow_validation_loss(
            model,
            dataset.validation_states,
            dataset.validation_action_targets,
            t_grid=validation_t_grid,
            seed=validation_seed,
        )
        validation.append({"step": step, "loss": loss})

    for step in range(training_steps):
        states, normalized_actions = dataset.sample(batch_size)
        loss = model.loss(normalized_actions, states)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        history.append(float(loss.item()))
        is_last = step == training_steps - 1
        if (
            has_validation
            and validation_every
            and (step % validation_every == 0 or is_last)
        ):
            record_validation(step)
        if log_every and (step % log_every == 0 or is_last):
            suffix = ""
            if validation and validation[-1]["step"] == step:
                suffix = f" val={validation[-1]['loss']:.6f}"
            logger(f"flow BC step={step:5d} loss={loss.item():.6f}{suffix}")

    # Deliberately no requires_grad_(False) and no .eval(): the velocity field is
    # the thing ReinFlow optimizes.
    return {"train_loss": history, "validation": validation}
