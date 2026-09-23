from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def cosine_beta_schedule(timesteps: int, offset: float = 0.008) -> torch.Tensor:
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alpha_cumprod = np.cos(((x / steps) + offset) / (1 + offset) * np.pi * 0.5) ** 2
    alpha_cumprod = alpha_cumprod / alpha_cumprod[0]
    betas = 1 - alpha_cumprod[1:] / alpha_cumprod[:-1]
    return torch.tensor(np.clip(betas, 0, 0.999), dtype=torch.float32)


def extract(
    schedule: torch.Tensor,
    timesteps: torch.Tensor,
    target_shape: torch.Size | tuple[int, ...],
) -> torch.Tensor:
    return schedule.gather(-1, timesteps).reshape(
        timesteps.shape[0], *((1,) * (len(target_shape) - 1))
    )


class SinusoidalPositionEmbedding(nn.Module):
    def __init__(self, dimension: int):
        super().__init__()
        self.dimension = dimension

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half = self.dimension // 2
        frequency_scale = math.log(10000) / (half - 1)
        frequencies = torch.exp(
            torch.arange(half, device=timesteps.device) * -frequency_scale
        )
        phase = timesteps.reshape(-1, 1) * frequencies.reshape(1, -1)
        return torch.cat([phase.sin(), phase.cos()], dim=-1)


class DiffusionMLP(nn.Module):
    def __init__(
        self,
        output_dim: int,
        condition_dim: int,
        time_embedding_dim: int,
        hidden_dims: Sequence[int],
    ):
        super().__init__()
        self.time_embedding = nn.Sequential(
            SinusoidalPositionEmbedding(time_embedding_dim),
            nn.Linear(time_embedding_dim, time_embedding_dim * 2),
            nn.Mish(),
            nn.Linear(time_embedding_dim * 2, time_embedding_dim),
        )
        dimensions = [
            time_embedding_dim + output_dim + condition_dim,
            *hidden_dims,
            output_dim,
        ]
        layers: list[nn.Module] = []
        for index in range(len(dimensions) - 1):
            layers.append(nn.Linear(dimensions[index], dimensions[index + 1]))
            if index < len(dimensions) - 2:
                layers.append(nn.Mish())
        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        noisy_action: torch.Tensor,
        timestep: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        time_embedding = self.time_embedding(timestep.float())
        return self.mlp(torch.cat([noisy_action, time_embedding, condition], dim=-1))


class DiffusionModel(nn.Module):
    def __init__(
        self,
        network: DiffusionMLP,
        output_dim: int,
        denoising_steps: int,
        denoised_clip_value: float = 1.5,
        device: str = "cpu",
    ):
        super().__init__()
        self.network = network.to(device)
        self.output_dim = output_dim
        self.denoising_steps = denoising_steps
        self.denoised_clip_value = denoised_clip_value
        self.device = device
        betas = cosine_beta_schedule(denoising_steps)
        self.register_buffer("betas", betas)
        alpha_cumprod = torch.cumprod(1.0 - betas, dim=0)
        self.register_buffer("alphas_cumprod", alpha_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alpha_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1 - alpha_cumprod)
        )
        self.register_buffer(
            "sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alpha_cumprod)
        )
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alpha_cumprod - 1)
        )

    def q_sample(
        self,
        clean_action: torch.Tensor,
        timestep: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        noise = torch.randn_like(clean_action) if noise is None else noise
        return (
            extract(self.sqrt_alphas_cumprod, timestep, clean_action.shape)
            * clean_action
            + extract(self.sqrt_one_minus_alphas_cumprod, timestep, clean_action.shape)
            * noise
        )

    def loss(self, clean_action: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        timestep = torch.randint(
            0, self.denoising_steps, (len(clean_action),), device=clean_action.device
        )
        noise = torch.randn_like(clean_action)
        noisy_action = self.q_sample(clean_action, timestep, noise)
        predicted_noise = self.network(noisy_action, timestep, condition)
        return F.mse_loss(predicted_noise, noise)

    def x0_from_epsilon(
        self, sample: torch.Tensor, timestep: torch.Tensor, epsilon: torch.Tensor
    ) -> torch.Tensor:
        clean = (
            extract(self.sqrt_recip_alphas_cumprod, timestep, sample.shape) * sample
            - extract(self.sqrt_recipm1_alphas_cumprod, timestep, sample.shape)
            * epsilon
        )
        return clean.clamp(-self.denoised_clip_value, self.denoised_clip_value)

    def ddim_sample_differentiable(
        self,
        initial_noise: torch.Tensor,
        condition: torch.Tensor,
        inference_steps: int,
        eta: float = 0.0,
    ) -> torch.Tensor:
        """`ddim_sample` without the no-grad guard.

        The decoder's own parameters stay frozen; this exists so gradients can
        reach whatever produced `initial_noise` -- in DSRL, the steering policy.
        The composite map eps -> D(s, w_psi(s, eps)) is trainable through psi even
        though D is not, which is what makes a behavior-level anchor possible here.
        Verified numerically identical to `ddim_sample` in tests/test_dsrl_anchors.py.
        """
        return self._ddim_sample_impl(initial_noise, condition, inference_steps, eta)

    @torch.no_grad()
    def ddim_sample(
        self,
        initial_noise: torch.Tensor,
        condition: torch.Tensor,
        inference_steps: int,
        eta: float = 0.0,
    ) -> torch.Tensor:
        return self._ddim_sample_impl(initial_noise, condition, inference_steps, eta)

    def _ddim_sample_impl(
        self,
        initial_noise: torch.Tensor,
        condition: torch.Tensor,
        inference_steps: int,
        eta: float = 0.0,
    ) -> torch.Tensor:
        schedule = (
            np.linspace(0, self.denoising_steps - 1, inference_steps)
            .round()
            .astype(int)[::-1]
        )
        sample = initial_noise
        batch_size = sample.shape[0]
        for index, timestep_value in enumerate(schedule):
            timestep = torch.full(
                (batch_size,), int(timestep_value), device=self.device, dtype=torch.long
            )
            epsilon = self.network(sample, timestep, condition)
            clean = self.x0_from_epsilon(sample, timestep, epsilon)
            alpha_t = self.alphas_cumprod[timestep_value]
            alpha_previous = (
                self.alphas_cumprod[schedule[index + 1]]
                if index + 1 < len(schedule)
                else torch.tensor(1.0, device=self.device)
            )
            sigma = eta * torch.sqrt(
                (1 - alpha_previous) / (1 - alpha_t) * (1 - alpha_t / alpha_previous)
            )
            direction = (
                torch.sqrt(torch.clamp(1 - alpha_previous - sigma**2, min=0.0))
                * epsilon
            )
            reverse_noise = torch.randn_like(sample) if eta > 0 else 0.0
            sample = (
                torch.sqrt(alpha_previous) * clean + direction + sigma * reverse_noise
            )
        return sample


class FrozenDiffusionPolicy:
    """Decode steering noise into one action or an open-loop action chunk."""

    def __init__(
        self,
        model: DiffusionModel,
        action_scale: float,
        action_dim: int,
        chunk_length: int,
        ddim_steps: int,
    ):
        self.model = model
        self.action_scale = action_scale
        self.action_dim = action_dim
        self.chunk_length = chunk_length
        self.ddim_steps = ddim_steps
        self.noise_dim = action_dim * chunk_length

    def __call__(self, state: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        decoded = self.model.ddim_sample(
            noise, state, inference_steps=self.ddim_steps, eta=0.0
        )
        return (decoded * self.action_scale).reshape(
            -1, self.chunk_length, self.action_dim
        )
