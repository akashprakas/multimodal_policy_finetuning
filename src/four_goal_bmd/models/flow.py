"""Rectified-flow policy with ReinFlow's learnable bounded noise injection.

Reference: Zhang et al., "ReinFlow: Fine-tuning Flow Matching Policy with Online
Reinforcement Learning" (arXiv 2505.22094). Upstream sources consulted:
`model/flow/reflow.py`, `model/flow/mlp_flow.py`, `model/flow/ft_ppo/ppoflow.py`.

Convention: t=1 is data, t=0 is noise (upstream `reflow.py:88`).

The velocity field reuses `DiffusionMLP` unchanged, so a flow policy built from
the same `policy` config has exactly the same parameter count as the diffusion
policy it is compared against.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from four_goal_bmd.models.diffusion import DiffusionMLP


def build_velocity_network(
    output_dim: int,
    condition_dim: int,
    time_embedding_dim: int,
    hidden_dims: Sequence[int],
) -> DiffusionMLP:
    """The velocity field v(x, t, s).

    `DiffusionMLP.forward(x, t, cond)` already has the required signature and
    returns a tensor shaped like `x`; it is the same architecture upstream calls
    `FlowMLP`, minus a condition encoder this toy does not have.
    """
    return DiffusionMLP(
        output_dim=output_dim,
        condition_dim=condition_dim,
        time_embedding_dim=time_embedding_dim,
        hidden_dims=hidden_dims,
    )


class FlowModel(nn.Module):
    """Rectified flow: x_t = t*x_1 + (1-t)*x_0, target velocity x_1 - x_0."""

    def __init__(
        self,
        network: DiffusionMLP,
        output_dim: int,
        flow_steps: int,
        act_limit: float = 3.0,
        time_scale: float | None = None,
        sample_t_type: str = "uniform",
        device: str = "cpu",
    ):
        super().__init__()
        self.network = network.to(device)
        self.output_dim = output_dim
        self.flow_steps = flow_steps
        self.act_limit = act_limit
        if sample_t_type not in {"uniform", "logitnormal", "beta"}:
            raise ValueError(f"unsupported sample_t_type {sample_t_type!r}")
        self.sample_t_type = sample_t_type
        # `SinusoidalPositionEmbedding` uses frequency_scale = log(10000)/(half-1),
        # which is calibrated for integer step indices. Feeding raw t in [0,1)
        # would make the embedding near-affine and nearly constant across the K
        # grid points, so scale t onto {0, ..., K-1} -- the same index semantics
        # the diffusion path enjoys. `time_scale=1.0` reproduces upstream.
        self.time_scale = float(flow_steps) if time_scale is None else float(time_scale)
        self.device = device

    def time_embedding(self, t: torch.Tensor) -> torch.Tensor:
        return self.network.time_embedding((t * self.time_scale).float())

    def velocity(
        self, x: torch.Tensor, t: torch.Tensor, condition: torch.Tensor
    ) -> torch.Tensor:
        return self.network(x, t * self.time_scale, condition)

    def sample_time(self, batch_size: int, device) -> torch.Tensor:
        """Upstream `reflow.py:101` offers three t distributions.

        With independent coupling the velocity field is nearly mode-averaging at
        small t and sharp near t=1, so where the budget is spent matters:
        `beta` concentrates samples near t=1, `logitnormal` near t=0.5.
        """
        if self.sample_t_type == "uniform":
            return torch.rand(batch_size, device=device)
        if self.sample_t_type == "logitnormal":
            return torch.sigmoid(torch.randn(batch_size, device=device))
        beta = torch.distributions.Beta(
            torch.tensor(1.5, device=device), torch.tensor(1.0, device=device)
        )
        return 0.999 * (1.0 - beta.sample((batch_size,)))

    def loss(self, clean_action: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """Flow-matching MSE. Argument order matches `DiffusionModel.loss`."""
        t = self.sample_time(len(clean_action), clean_action.device)
        initial_noise = torch.randn_like(clean_action)
        return self.loss_with(clean_action, condition, t, initial_noise)

    def loss_with(
        self,
        clean_action: torch.Tensor,
        condition: torch.Tensor,
        t: torch.Tensor,
        initial_noise: torch.Tensor,
    ) -> torch.Tensor:
        """Deterministic variant, so held-out loss is comparable across steps."""
        t_column = t.reshape(-1, 1)
        interpolated = t_column * clean_action + (1.0 - t_column) * initial_noise
        target = clean_action - initial_noise
        return F.mse_loss(self.velocity(interpolated, t, condition), target)

    def euler_sample_differentiable(
        self,
        initial_noise: torch.Tensor,
        condition: torch.Tensor,
        inference_steps: int,
        clip_intermediate: bool = True,
    ) -> torch.Tensor:
        """Left-endpoint Euler on {0, 1/K, ..., 1-1/K} with dt = 1/K.

        Gradient-carrying, so d(action)/d(x_0) can be measured directly.
        """
        step_size = 1.0 / inference_steps
        sample = initial_noise
        for index in range(inference_steps):
            t = torch.full((sample.shape[0],), index * step_size, device=sample.device)
            # Not in-place: `sample` may alias a caller's tensor.
            sample = sample + self.velocity(sample, t, condition) * step_size
            if clip_intermediate or index == inference_steps - 1:
                sample = sample.clamp(-self.act_limit, self.act_limit)
        return sample

    @torch.no_grad()
    def euler_sample(
        self,
        initial_noise: torch.Tensor,
        condition: torch.Tensor,
        inference_steps: int,
        clip_intermediate: bool = True,
    ) -> torch.Tensor:
        return self.euler_sample_differentiable(
            initial_noise, condition, inference_steps, clip_intermediate
        )


class ExploreNoiseNet(nn.Module):
    """Learnable sigma(s, t), bounded in [sigma_min, sigma_max] by construction.

    Upstream `model/flow/mlp_flow.py:189-214`:

        logvar = logvar_min + (logvar_max - logvar_min) * (tanh(MLP(f)) + 1) / 2
        sigma  = exp(0.5 * logvar)

    Tanh rather than `clamp` because clamp has zero gradient at the boundary: a
    sigma that reached the floor could never come back up. Interpolating in
    log-variance space (not std) makes a linear change in the raw output a linear
    change in Gaussian entropy, which conditions the optimization better.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        sigma_min: float,
        sigma_max: float,
        hidden_dims: Sequence[int] = (64, 64),
        activation: str = "tanh",
    ):
        super().__init__()
        if not 0.0 < sigma_min <= sigma_max:
            raise ValueError(
                f"require 0 < sigma_min <= sigma_max, got {sigma_min}, {sigma_max}"
            )
        activations = {"tanh": nn.Tanh, "mish": nn.Mish, "relu": nn.ReLU}
        if activation not in activations:
            raise ValueError(f"unsupported noise activation {activation!r}")
        make_activation = activations[activation]
        dimensions = [input_dim, *hidden_dims, output_dim]
        layers: list[nn.Module] = []
        for index in range(len(dimensions) - 1):
            layers.append(nn.Linear(dimensions[index], dimensions[index + 1]))
            if index < len(dimensions) - 2:
                layers.append(make_activation())
        self.mlp = nn.Sequential(*layers)
        # Buffers rather than requires_grad=False parameters: the bounds must
        # never reach an optimizer, and `set_sigma_range` must not be able to
        # silently create a fresh leaf mid-training. They stay in state_dict.
        self.register_buffer("logvar_min", torch.tensor(0.0))
        self.register_buffer("logvar_max", torch.tensor(0.0))
        self.set_sigma_range(sigma_min, sigma_max)

    def set_sigma_range(self, sigma_min: float, sigma_max: float) -> None:
        if not 0.0 < sigma_min <= sigma_max:
            raise ValueError(
                f"require 0 < sigma_min <= sigma_max, got {sigma_min}, {sigma_max}"
            )
        self.logvar_min.fill_(math.log(sigma_min**2))
        self.logvar_max.fill_(math.log(sigma_max**2))

    @property
    def sigma_min(self) -> float:
        return float(torch.exp(0.5 * self.logvar_min))

    @property
    def sigma_max(self) -> float:
        return float(torch.exp(0.5 * self.logvar_max))

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        raw = torch.tanh(self.mlp(feature))
        log_variance = (
            self.logvar_min + (self.logvar_max - self.logvar_min) * (raw + 1.0) / 2.0
        )
        return torch.exp(0.5 * log_variance)


class NoisyFlowPolicy(nn.Module):
    """Velocity field plus learnable noise head. The ReinFlow actor.

    The noise head deliberately never sees `x`: conditioning sigma on the sample
    it is perturbing tangles the noise gradient with the trajectory producing it
    (upstream note, `mlp_flow.py`). It sees the detached time embedding and the
    condition only.
    """

    def __init__(
        self,
        model: FlowModel,
        flow_steps: int,
        sigma_min: float,
        sigma_max: float,
        condition_dim: int,
        time_embedding_dim: int,
        noise_hidden_dims: Sequence[int] = (64, 64),
        time_independent_noise: bool = False,
        noise_activation: str = "tanh",
    ):
        super().__init__()
        self.model = model
        self.flow_steps = flow_steps
        self.time_independent_noise = time_independent_noise
        noise_input_dim = condition_dim
        if not time_independent_noise:
            noise_input_dim += time_embedding_dim
        self.explore_noise_net = ExploreNoiseNet(
            input_dim=noise_input_dim,
            output_dim=model.output_dim,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            hidden_dims=noise_hidden_dims,
            activation=noise_activation,
        )

    @property
    def sigma_min(self) -> float:
        return self.explore_noise_net.sigma_min

    @property
    def sigma_max(self) -> float:
        return self.explore_noise_net.sigma_max

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        condition: torch.Tensor,
        learn_exploration_noise: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (velocity, sigma).

        sigma is detached unless `learn_exploration_noise`. Rollouts must treat
        sigma as fixed data or the PPO ratio is not well defined; the loss
        recomputation must let gradients through, because that is the only signal
        the noise head ever receives.
        """
        velocity = self.model.velocity(x, t, condition)
        if self.time_independent_noise:
            feature = condition
        else:
            feature = torch.cat(
                [self.model.time_embedding(t).detach(), condition], dim=-1
            )
        sigma = self.explore_noise_net(feature)
        return velocity, sigma if learn_exploration_noise else sigma.detach()


class FlowPolicyActor:
    """Duck-type twin of `FrozenDiffusionPolicy`, where `noise` IS x_0.

    `evaluation.py`, `diagnostics.py` and `checkpointing.py` depend only on
    `policy(state, noise)`, `.noise_dim`, `.chunk_length`, `.action_scale` and
    `.model`, so satisfying this contract reuses the entire measurement harness
    without modification. The prior-noise evaluator's per-unit
    `torch.randn(n, noise_dim)` is exactly the fresh x_0 a flow needs.
    """

    def __init__(
        self,
        actor: NoisyFlowPolicy,
        action_scale: float,
        action_dim: int,
        chunk_length: int,
        flow_steps: int,
        act_limit: float,
        denoised_clip_value: float,
        min_sampling_std: float,
        randn_clip_value: float = 3.0,
        clip_intermediate_actions: bool = True,
        deterministic: bool = False,
    ):
        self.actor = actor
        self.model = actor.model
        self.action_scale = action_scale
        self.action_dim = action_dim
        self.chunk_length = chunk_length
        self.flow_steps = flow_steps
        self.act_limit = act_limit
        self.denoised_clip_value = denoised_clip_value
        self.min_sampling_std = min_sampling_std
        self.randn_clip_value = randn_clip_value
        self.clip_intermediate_actions = clip_intermediate_actions
        self.deterministic = deterministic
        self.noise_dim = action_dim * chunk_length

    def __call__(self, state: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        from four_goal_bmd.algorithms.reinflow_ppo import sample_flow_chain

        action, _, _ = sample_flow_chain(
            actor=self.actor,
            condition=state,
            flow_steps=self.flow_steps,
            act_limit=self.act_limit,
            denoised_clip_value=self.denoised_clip_value,
            min_sampling_std=self.min_sampling_std,
            randn_clip_value=self.randn_clip_value,
            clip_intermediate_actions=self.clip_intermediate_actions,
            deterministic=self.deterministic,
            initial_noise=noise,
        )
        return (action * self.action_scale).reshape(
            -1, self.chunk_length, self.action_dim
        )
