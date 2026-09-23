from __future__ import annotations

import torch
import torch.nn.functional as F

DSRL_ANCHOR_TYPES = ("none", "functional")
EPSILON = 1e-8


def _decode(
    frozen_policy, states: torch.Tensor, noise: torch.Tensor, differentiable: bool
) -> torch.Tensor:
    """Decode steering noise into a flat action, optionally with gradient.

    Returns the *normalized* action (before `action_scale`) so a coefficient means
    the same thing regardless of the environment's action units.
    """
    model = frozen_policy.model
    if differentiable:
        decoded = model.ddim_sample_differentiable(
            noise, states, inference_steps=frozen_policy.ddim_steps, eta=0.0
        )
    else:
        with torch.no_grad():
            decoded = model.ddim_sample(
                noise, states, inference_steps=frozen_policy.ddim_steps, eta=0.0
            )
    return decoded


def dsrl_anchor_loss(
    anchor_type: str,
    steering_policy,
    frozen_policy,
    states: torch.Tensor,
    latents: torch.Tensor,
    generator: torch.Generator | None = None,
    n_draws: int = 2,
) -> torch.Tensor:
    """Behavior-level drift penalty for the steering policy.

    `eps` draws come from a dedicated generator so anchor sampling cannot perturb
    the rollout RNG. The reference branch is decoded under `no_grad`, so no
    gradient reaches it and the decoder is never updated.
    """
    if anchor_type not in DSRL_ANCHOR_TYPES:
        raise ValueError(f"anchor_type must be one of {DSRL_ANCHOR_TYPES}")
    if anchor_type == "none":
        return torch.zeros((), device=states.device)

    batch_size = states.shape[0]
    noise_dim = frozen_policy.noise_dim
    device = states.device
    distribution = steering_policy.distribution(states, latents)

    def steered(epsilon: torch.Tensor) -> torch.Tensor:
        # Reparameterized, so the gradient reaches mu_psi and log_std.
        w = distribution.mean + distribution.stddev * epsilon
        return _decode(frozen_policy, states, w, differentiable=True)

    def reference(epsilon: torch.Tensor) -> torch.Tensor:
        # Identity steering: the frozen decoder on its own training prior.
        return _decode(frozen_policy, states, epsilon, differentiable=False)

    if anchor_type == "functional":
        epsilon = torch.randn(batch_size, noise_dim, generator=generator, device=device)
        return F.mse_loss(steered(epsilon), reference(epsilon).detach())
