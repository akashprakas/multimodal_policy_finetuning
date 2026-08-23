from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


def build_mlp(
    dimensions: Sequence[int], final_activation: bool = False
) -> nn.Sequential:
    layers: list[nn.Module] = []
    for index in range(len(dimensions) - 1):
        layers.append(nn.Linear(dimensions[index], dimensions[index + 1]))
        if index < len(dimensions) - 2 or final_activation:
            layers.append(nn.Mish())
    return nn.Sequential(*layers)


class InferenceModel(nn.Module):
    def __init__(self, observation_dim: int, k_latent: int, hidden_dims: Sequence[int]):
        super().__init__()
        self.network = build_mlp([observation_dim, *hidden_dims, k_latent])

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)

    def log_prob(
        self, state: torch.Tensor, latent: torch.Tensor, training_noise_std: float = 0.0
    ) -> torch.Tensor:
        if training_noise_std > 0:
            state = state + training_noise_std * torch.randn_like(state)
        log_probabilities = F.log_softmax(self.forward(state), dim=-1)
        return log_probabilities.gather(-1, latent.reshape(-1, 1)).squeeze(-1)


class SteeringPolicy(nn.Module):
    def __init__(
        self,
        observation_dim: int,
        k_latent: int,
        noise_dim: int,
        hidden_dims: Sequence[int],
        w_bound: float = 3.0,
        log_std_init: float = 0.0,
        min_log_std: float = -20.0,
    ):
        super().__init__()
        self.k_latent = k_latent
        self.noise_dim = noise_dim
        self.w_bound = w_bound
        self.min_log_std = min_log_std
        self.mean_network = build_mlp(
            [observation_dim + k_latent, *hidden_dims, noise_dim]
        )
        self.log_std = nn.Parameter(torch.full((noise_dim,), log_std_init))

    def _input(self, state: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        one_hot = F.one_hot(latent, self.k_latent).float()
        return torch.cat([state, one_hot], dim=-1)

    def distribution(self, state: torch.Tensor, latent: torch.Tensor) -> Normal:
        mean = torch.tanh(self.mean_network(self._input(state, latent))) * self.w_bound
        std = self.log_std.clamp(min=self.min_log_std).exp().expand_as(mean)
        return Normal(mean, std)

    def sample(
        self, state: torch.Tensor, latent: torch.Tensor, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        distribution = self.distribution(state, latent)
        noise = distribution.mean if deterministic else distribution.rsample()
        return noise, distribution.log_prob(noise).sum(dim=-1)

    def log_prob(
        self, state: torch.Tensor, latent: torch.Tensor, noise: torch.Tensor
    ) -> torch.Tensor:
        return self.distribution(state, latent).log_prob(noise).sum(dim=-1)


class LatentCritic(nn.Module):
    def __init__(self, observation_dim: int, k_latent: int, hidden_dims: Sequence[int]):
        super().__init__()
        self.k_latent = k_latent
        self.network = build_mlp([observation_dim + k_latent, *hidden_dims, 1])

    def forward(self, state: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        one_hot = F.one_hot(latent, self.k_latent).float()
        return self.network(torch.cat([state, one_hot], dim=-1)).squeeze(-1)
