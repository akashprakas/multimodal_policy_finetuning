from __future__ import annotations

from pathlib import Path

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from four_goal_bmd.algorithms.bmd import train_bmd
from four_goal_bmd.algorithms.reinflow import (
    build_actor_from_bc,
    make_flow_actor,
    train_reinflow,
)
from four_goal_bmd.checkpointing import (
    save_bc_checkpoint,
    save_flow_bc_checkpoint,
    save_reinflow_checkpoint,
    save_training_checkpoint,
)
from four_goal_bmd.config_schema import apply_run_profile, backend, validate_config
from four_goal_bmd.envs.four_goals import make_goal_centers
from four_goal_bmd.evaluation import evaluate_policy
from four_goal_bmd.factory import pretrain_flow_policy, pretrain_frozen_policy
from four_goal_bmd.flow_diagnostics import noise_sensitivity_probe, sigma_field_probe
from four_goal_bmd.metrics import aggregate_metrics, format_metrics
from four_goal_bmd.task_shift import finetuning_environment
from four_goal_bmd.utils import configure_cpu, metadata, save_json, save_resolved_config


def run_reinflow(cfg: DictConfig, output_dir: Path) -> None:
    """Pretrain the flow prior, fine-tune independent actors, and evaluate them."""
    print("Pretraining one flow BC model shared by all training seeds...")
    bc_model, action_scale, bc_report = pretrain_flow_policy(cfg, verbose=True)
    print(
        f"action_scale={action_scale:.6f} "
        f"n_parameters={bc_report['n_parameters']}"
    )
    save_json(output_dir / "bc_report.json", bc_report)
    if cfg.logging.save_checkpoints:
        save_flow_bc_checkpoint(
            output_dir / "flow_bc_checkpoint.pt", bc_model, action_scale, cfg
        )

    # BC is learned at the original radius. The shift is introduced only here.
    environment_cfg = finetuning_environment(cfg)
    if cfg.shift.goal_radius is not None or cfg.shift.success_distance is not None:
        print(
            f"task shift: goal_radius={environment_cfg.goal_radius} "
            f"success_distance={environment_cfg.success_distance}"
        )
    goal_centers = make_goal_centers(
        n_goals=environment_cfg.n_goals,
        radius=environment_cfg.goal_radius,
        rotation=cfg.landscape.rotation,
    )

    # Flow analogue of the BC-prior evaluation: deterministic ODE integration,
    # while retaining fresh initial x_0 noise for every action chunk.
    bc_actor = build_actor_from_bc(
        bc_model,
        cfg.experiment,
        condition_dim=environment_cfg.observation_dim,
        time_embedding_dim=cfg.policy.time_embedding_dim,
    )
    bc_policy = make_flow_actor(
        bc_actor,
        action_scale,
        environment_cfg,
        cfg.policy,
        cfg.experiment,
        deterministic=True,
    )
    bc_evaluations = []
    for evaluation_seed in cfg.seeds.evaluation:
        bc_evaluations.append(
            evaluate_policy(
                frozen_policy=bc_policy,
                goal_centers=goal_centers,
                environment_cfg=environment_cfg,
                policy_cfg=cfg.policy,
                n_episodes=cfg.evaluation.n_episodes,
                evaluation_seed=evaluation_seed,
                min_attempts=cfg.evaluation.min_attempts,
            )
        )
    bc_mean, bc_std = aggregate_metrics(bc_evaluations)
    print("BC prior:", format_metrics({**bc_mean, "n_goals": 4}))

    if cfg.experiment.total_epochs == 0:
        summary = {
            "experiment": cfg.experiment.name,
            "backend": "reinflow",
            "policy": cfg.policy.name,
            "landscape": cfg.landscape.name,
            "shift": cfg.shift.name,
            "task": {
                "bc_goal_radius": float(cfg.environment.goal_radius),
                "finetuning_goal_radius": float(environment_cfg.goal_radius),
                "success_distance": float(environment_cfg.success_distance),
                "goal_centers": goal_centers.tolist(),
            },
            "bc_only": True,
            "bc": {
                "report": {
                    key: value
                    for key, value in bc_report.items()
                    if key != "train_loss"
                },
                "mean": bc_mean,
                "std": bc_std,
                "evaluations": bc_evaluations,
            },
        }
        save_json(output_dir / "summary.json", summary)
        print("total_epochs=0: flow BC evaluation only, no RL fine-tuning.")
        return

    runs = []
    for training_seed in cfg.seeds.training:
        print(f"\nTraining seed {training_seed}")
        result = train_reinflow(
            bc_model=bc_model,
            action_scale=action_scale,
            goal_centers=goal_centers,
            environment_cfg=environment_cfg,
            experiment_cfg=cfg.experiment,
            policy_cfg=cfg.policy,
            evaluation_cfg=cfg.evaluation,
            seed=training_seed,
            environment_seed=training_seed + cfg.seeds.environment_offset,
            log_every=cfg.logging.log_every,
        )
        seed_dir = output_dir / f"seed_{training_seed:03d}"
        save_json(seed_dir / "history.json", result.history)

        trained_policy = make_flow_actor(
            result.actor,
            action_scale,
            environment_cfg,
            cfg.policy,
            cfg.experiment,
        )
        evaluations = []
        for evaluation_seed in cfg.seeds.evaluation:
            metrics = evaluate_policy(
                frozen_policy=trained_policy,
                goal_centers=goal_centers,
                environment_cfg=environment_cfg,
                policy_cfg=cfg.policy,
                n_episodes=cfg.evaluation.n_episodes,
                evaluation_seed=evaluation_seed,
                min_attempts=cfg.evaluation.min_attempts,
            )
            metrics["evaluation_seed"] = int(evaluation_seed)
            evaluations.append(metrics)

        probe = noise_sensitivity_probe(
            policy=trained_policy,
            goal_centers=goal_centers,
            environment_cfg=environment_cfg,
            policy_cfg=cfg.policy,
            n_episodes=cfg.evaluation.n_episodes,
            seed=cfg.seeds.evaluation[0],
            min_attempts=cfg.evaluation.min_attempts,
        )
        sigma_field = sigma_field_probe(
            actor=result.actor,
            states=torch.from_numpy(goal_centers.astype("float32")) * 0.5,
            flow_steps=cfg.experiment.flow_steps,
        )
        metrics_mean, metrics_std = aggregate_metrics(evaluations)
        run = {
            "training_seed": int(training_seed),
            "evaluations": evaluations,
            "metrics": metrics_mean,
            "evaluation_std": metrics_std,
            "final_training": result.history[-1],
            "pre_rl": result.pre_rl,
            "mode_flip_rate": probe["mode_flip_rate"],
            "endpoint_l2": probe["endpoint_l2"],
            "sigma_field": sigma_field,
        }
        runs.append(run)
        save_json(seed_dir / "metrics.json", run)
        if cfg.logging.save_checkpoints:
            save_reinflow_checkpoint(
                seed_dir / "checkpoint.pt",
                result,
                action_scale,
                cfg,
                training_seed,
            )
        print(
            "result:",
            format_metrics({**metrics_mean, "n_goals": 4}),
            f"flip={probe['mode_flip_rate']:.3f}",
        )

    across_seed_mean, across_seed_std = aggregate_metrics(
        [run["metrics"] for run in runs]
    )
    summary = {
        "experiment": cfg.experiment.name,
        "backend": "reinflow",
        "policy": cfg.policy.name,
        "landscape": cfg.landscape.name,
        "shift": cfg.shift.name,
        "task": {
            "bc_goal_radius": float(cfg.environment.goal_radius),
            "finetuning_goal_radius": float(environment_cfg.goal_radius),
            "success_distance": float(environment_cfg.success_distance),
            "goal_centers": goal_centers.tolist(),
        },
        "training_seeds": [int(seed) for seed in cfg.seeds.training],
        "evaluation_seeds": [int(seed) for seed in cfg.seeds.evaluation],
        "bc": {
            "report": {key: value for key, value in bc_report.items() if key != "train_loss"},
            "mean": bc_mean,
            "std": bc_std,
            "evaluations": bc_evaluations,
        },
        "mean": across_seed_mean,
        "std": across_seed_std,
        "runs": runs,
    }
    save_json(output_dir / "summary.json", summary)
    print("\nAggregate:", format_metrics({**across_seed_mean, "n_goals": 4}))


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    configure_cpu()
    apply_run_profile(cfg)
    validate_config(cfg)
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    project_dir = Path(__file__).resolve().parent
    save_resolved_config(output_dir / "resolved_config.yaml", cfg)
    save_json(output_dir / "metadata.json", metadata(project_dir))

    print(f"output_dir={output_dir}")
    if backend(cfg) == "reinflow":
        run_reinflow(cfg, output_dir)
        return

    print("Pretraining one frozen BC policy shared by all training seeds...")
    frozen_policy, bc_history = pretrain_frozen_policy(cfg, verbose=True)
    print(f"action_scale={frozen_policy.action_scale:.6f}")
    save_json(output_dir / "bc_history.json", bc_history)
    if cfg.logging.save_checkpoints:
        save_bc_checkpoint(output_dir / "bc_checkpoint.pt", frozen_policy, cfg)

    # Apply the task shift only after BC has been trained on the original goals.
    environment_cfg = finetuning_environment(cfg)
    if cfg.shift.goal_radius is not None or cfg.shift.success_distance is not None:
        print(
            f"task shift: goal_radius={environment_cfg.goal_radius} "
            f"success_distance={environment_cfg.success_distance}"
        )
    goal_centers = make_goal_centers(
        n_goals=environment_cfg.n_goals,
        radius=environment_cfg.goal_radius,
        rotation=cfg.landscape.rotation,
    )
    bc_evaluations = []
    for evaluation_seed in cfg.seeds.evaluation:
        bc_evaluations.append(
            evaluate_policy(
                frozen_policy=frozen_policy,
                goal_centers=goal_centers,
                environment_cfg=environment_cfg,
                policy_cfg=cfg.policy,
                n_episodes=cfg.evaluation.n_episodes,
                evaluation_seed=evaluation_seed,
                min_attempts=cfg.evaluation.min_attempts,
            )
        )
    bc_mean, bc_std = aggregate_metrics(bc_evaluations)
    print("BC prior:", format_metrics({**bc_mean, "n_goals": 4}))

    runs = []
    for training_seed in cfg.seeds.training:
        print(f"\nTraining seed {training_seed}")
        result = train_bmd(
            frozen_policy=frozen_policy,
            goal_centers=goal_centers,
            environment_cfg=environment_cfg,
            experiment_cfg=cfg.experiment,
            policy_cfg=cfg.policy,
            seed=training_seed,
            environment_seed=training_seed + cfg.seeds.environment_offset,
            log_every=cfg.logging.log_every,
        )
        seed_dir = output_dir / f"seed_{training_seed:03d}"
        save_json(seed_dir / "history.json", result.history)

        evaluations = []
        for evaluation_seed in cfg.seeds.evaluation:
            metrics = evaluate_policy(
                frozen_policy=frozen_policy,
                steering_policy=result.steering_policy,
                k_latent=cfg.experiment.k_latent,
                goal_centers=goal_centers,
                environment_cfg=environment_cfg,
                policy_cfg=cfg.policy,
                n_episodes=cfg.evaluation.n_episodes,
                evaluation_seed=evaluation_seed,
                min_attempts=cfg.evaluation.min_attempts,
                deterministic_steering=cfg.evaluation.deterministic_steering,
            )
            metrics["evaluation_seed"] = int(evaluation_seed)
            evaluations.append(metrics)

        metrics_mean, metrics_std = aggregate_metrics(evaluations)
        run = {
            "training_seed": int(training_seed),
            "evaluations": evaluations,
            "metrics": metrics_mean,
            "evaluation_std": metrics_std,
            "final_training": result.history[-1],
        }
        runs.append(run)
        save_json(seed_dir / "metrics.json", run)
        if cfg.logging.save_checkpoints:
            save_training_checkpoint(
                seed_dir / "checkpoint.pt", frozen_policy, result, cfg, training_seed
            )
        print("result:", format_metrics({**metrics_mean, "n_goals": 4}))

    across_seed_mean, across_seed_std = aggregate_metrics(
        [run["metrics"] for run in runs]
    )
    summary = {
        "experiment": cfg.experiment.name,
        "policy": cfg.policy.name,
        "landscape": cfg.landscape.name,
        "shift": cfg.shift.name,
        "task": {
            "bc_goal_radius": float(cfg.environment.goal_radius),
            "finetuning_goal_radius": float(environment_cfg.goal_radius),
            "success_distance": float(environment_cfg.success_distance),
            "goal_centers": goal_centers.tolist(),
        },
        "training_seeds": [int(seed) for seed in cfg.seeds.training],
        "evaluation_seeds": [int(seed) for seed in cfg.seeds.evaluation],
        "bc": {"mean": bc_mean, "std": bc_std, "evaluations": bc_evaluations},
        "mean": across_seed_mean,
        "std": across_seed_std,
        "runs": runs,
    }
    save_json(output_dir / "summary.json", summary)
    print("\nAggregate:", format_metrics({**across_seed_mean, "n_goals": 4}))


if __name__ == "__main__":
    main()
