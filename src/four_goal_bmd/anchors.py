"""Drift anchors: keep the fine-tuned velocity field near its pretrained self.

Two arms of the preservation study (`docs/PRESERVATION_PREREG.md`). Both share
one interface and differ only in `anchor_type`:

    upstream    ||a_theta(s, x0=0) - sg a_BC(s, x0=0)||^2
                Exactly upstream ReinFlow's optional regularizer: deterministic
                outputs compared at a single fixed x0. Deliberately NOT called
                Wasserstein-2 -- it compares two points, not two distributions.

    functional  E_{x0 ~ N(0,I)} ||a_theta(s, x0) - sg a_BC(s, x0)||^2
                Anchors the whole noise-to-action map rather than one point.

Common conventions, all deliberate:

  * evaluated on the **current rollout observations**, not a fixed probe set, so
    the anchor constrains the field where the policy actually operates;
  * **deterministic** Euler integration, so the anchor constrains the map rather
    than the sampling distribution;
  * **normalized** action space (before `action_scale`), so a given coefficient
    means the same thing regardless of the environment's action units;
  * gradients flow **only** through the fine-tuned field; the reference is a
    frozen deep copy and every use of it is wrapped in `no_grad`;
  * `x_0` draws come from a **dedicated generator**, so anchor sampling cannot
    perturb the rollout RNG.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

ANCHOR_TYPES = ("none", "upstream", "functional")
EPSILON = 1e-8


def _integrate(
    model, initial_noise: torch.Tensor, condition: torch.Tensor, flow_steps: int
) -> torch.Tensor:
    return model.euler_sample_differentiable(
        initial_noise, condition, flow_steps, clip_intermediate=True
    )


def _reference(
    model, initial_noise: torch.Tensor, condition: torch.Tensor, flow_steps: int
) -> torch.Tensor:
    with torch.no_grad():
        return _integrate(model, initial_noise, condition, flow_steps)


def anchor_loss(
    anchor_type: str,
    actor,
    reference_model,
    observations: torch.Tensor,
    flow_steps: int,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Drift penalty in normalized action space. Zero when theta == theta_BC."""
    if anchor_type not in ANCHOR_TYPES:
        raise ValueError(f"anchor_type must be one of {ANCHOR_TYPES}")
    if anchor_type == "none":
        return torch.zeros((), device=observations.device)

    batch_size = observations.shape[0]
    action_dim_total = actor.model.output_dim
    device = observations.device

    if anchor_type == "upstream":
        # A single fixed x_0 = 0, exactly as upstream does.
        initial = torch.zeros(batch_size, action_dim_total, device=device)
        theta = _integrate(actor.model, initial, observations, flow_steps)
        target = _reference(reference_model, initial, observations, flow_steps)
        return F.mse_loss(theta, target)

    initial = torch.randn(
        batch_size, action_dim_total, generator=generator, device=device
    )
    theta = _integrate(actor.model, initial, observations, flow_steps)
    target = _reference(reference_model, initial, observations, flow_steps)
    return F.mse_loss(theta, target)


def velocity_parameters(actor) -> list[torch.Tensor]:
    """Trainable parameters of the velocity field only.

    The gradient ratio is measured over these alone. Including the noise head in
    the denominator would understate the anchor's relative influence, because the
    anchor does not constrain that head at all.
    """
    return [
        parameter for parameter in actor.model.parameters() if parameter.requires_grad
    ]


def gradient_norm_ratio(
    anchor_term: torch.Tensor, policy_term: torch.Tensor, parameters: list[torch.Tensor]
) -> float:
    """||grad(anchor)|| / ||grad(policy)|| over the velocity field.

    Diagnostic for pass-by-freezing: a coefficient large enough to dominate the
    policy gradient produces a policy that barely moves, and this makes that
    visible directly rather than inferred from a flat learning curve.
    """
    if not parameters:
        return float("nan")

    def norm(term: torch.Tensor) -> float:
        if not term.requires_grad:
            return 0.0
        grads = torch.autograd.grad(
            term, parameters, retain_graph=True, allow_unused=True
        )
        total = sum(float((grad**2).sum()) for grad in grads if grad is not None)
        return total**0.5

    return norm(anchor_term) / (norm(policy_term) + EPSILON)
