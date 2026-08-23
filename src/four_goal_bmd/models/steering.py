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
