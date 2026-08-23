import torch

from four_goal_bmd.models.diffusion import (
    DiffusionMLP,
    DiffusionModel,
    FrozenDiffusionPolicy,
)


def test_single_step_and_chunked_output_shapes():
    for chunk_length in (1, 4):
        output_dim = 2 * chunk_length
        network = DiffusionMLP(output_dim, 2, 8, [8, 8])
        model = DiffusionModel(network, output_dim, denoising_steps=10)
        policy = FrozenDiffusionPolicy(model, 0.05, 2, chunk_length, ddim_steps=2)
        actions = policy(torch.zeros(3, 2), torch.randn(3, output_dim))
        assert actions.shape == (3, chunk_length, 2)
