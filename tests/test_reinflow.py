from __future__ import annotations

import copy
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir

from four_goal_bmd.algorithms.reinflow import build_actor_from_bc, train_reinflow
from four_goal_bmd.algorithms.reinflow_ppo import (
    flow_chain_logprob,
    sample_flow_chain,
)
from four_goal_bmd.anchors import anchor_loss
from four_goal_bmd.config_schema import apply_run_profile, validate_config
from four_goal_bmd.envs.four_goals import make_goal_centers
from four_goal_bmd.factory import pretrain_flow_policy
from four_goal_bmd.models.flow import FlowModel, NoisyFlowPolicy, build_velocity_network
from four_goal_bmd.task_shift import finetuning_environment


def make_actor(flow_steps: int = 4) -> NoisyFlowPolicy:
    network = build_velocity_network(2, 2, 8, [8, 8])
    model = FlowModel(
        network,
        output_dim=2,
        flow_steps=flow_steps,
        act_limit=3.0,
        time_scale=1.0,
    )
    return NoisyFlowPolicy(
        model=model,
        flow_steps=flow_steps,
        sigma_min=0.1,
        sigma_max=0.4,
        condition_dim=2,
        time_embedding_dim=8,
        noise_hidden_dims=[8],
    )


def smoke_config(project_root: Path, experiment: str = "reinflow_plain"):
    overrides = [
        f"experiment={experiment}",
        "policy=single_step_matched",
        "landscape=g5",
        "shift=adaptation_v1",
        "run=smoke",
        "run.max_bc_steps=4",
        "run.max_total_epochs=1",
        "run.max_n_demonstrations=32",
        "run.max_n_envs=8",
        "run.max_ppo_batch_size=8",
        "run.n_evaluation_episodes=16",
        "experiment.diagnostic_every=0",
        "experiment.n_spread_samples=8",
    ]
    with initialize_config_dir(
        version_base="1.3", config_dir=str(project_root / "configs")
    ):
        cfg = compose(config_name="config", overrides=overrides)
    apply_run_profile(cfg)
    validate_config(cfg)
    return cfg


def test_flow_chain_logprob_recomputation_matches_sampling():
    torch.manual_seed(0)
    actor = make_actor()
    condition = torch.randn(32, 2)
    _, chain, sampled = sample_flow_chain(
        actor=actor,
        condition=condition,
        flow_steps=4,
        act_limit=3.0,
        denoised_clip_value=3.0,
        min_sampling_std=0.1,
        randn_clip_value=3.0,
        clip_intermediate_actions=True,
    )
    recomputed, _, _ = flow_chain_logprob(
        actor=actor,
        condition=condition,
        chain=chain,
        flow_steps=4,
        denoised_clip_value=3.0,
    )
    assert float((recomputed - sampled).abs().max().detach()) < 1e-4


def test_functional_anchor_is_zero_at_initialization_and_stops_reference_gradients():
    actor = make_actor()
    reference = copy.deepcopy(actor.model)
    for parameter in reference.parameters():
        parameter.requires_grad_(False)
    observations = torch.randn(16, 2)
    loss = anchor_loss(
        "functional",
        actor,
        reference,
        observations,
        flow_steps=4,
        generator=torch.Generator().manual_seed(7),
    )
    assert float(loss.detach()) < 1e-10

    with torch.no_grad():
        next(actor.model.parameters()).add_(0.1)
    loss = anchor_loss(
        "functional",
        actor,
        reference,
        observations,
        flow_steps=4,
        generator=torch.Generator().manual_seed(7),
    )
    loss.backward()
    assert float(loss.detach()) > 0.0
    assert all(parameter.grad is None for parameter in reference.parameters())


def test_reinflow_smoke_preserves_bc_and_uses_shifted_environment():
    project_root = Path(__file__).resolve().parents[1]
    cfg = smoke_config(project_root)
    bc_model, action_scale, report = pretrain_flow_policy(cfg, verbose=False)
    assert len(report["train_loss"]) == 4
    before = [parameter.detach().clone() for parameter in bc_model.parameters()]

    environment_cfg = finetuning_environment(cfg)
    assert environment_cfg.goal_radius == 1.22
    result = train_reinflow(
        bc_model=bc_model,
        action_scale=action_scale,
        goal_centers=make_goal_centers(
            4, environment_cfg.goal_radius, cfg.landscape.rotation
        ),
        environment_cfg=environment_cfg,
        experiment_cfg=cfg.experiment,
        policy_cfg=cfg.policy,
        evaluation_cfg=cfg.evaluation,
        seed=0,
        environment_seed=1,
    )
    assert len(result.history) == 1
    assert result.history[0]["ratio_sanity"] < cfg.experiment.ratio_sanity_tolerance
    for original, current in zip(before, bc_model.parameters()):
        assert torch.equal(original, current)
