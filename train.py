from __future__ import annotations

from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from four_goal_bmd.algorithms.bmd import train_bmd
from four_goal_bmd.checkpointing import save_bc_checkpoint, save_training_checkpoint
from four_goal_bmd.config_schema import apply_run_profile, validate_config
from four_goal_bmd.envs.four_goals import make_goal_centers
from four_goal_bmd.evaluation import evaluate_policy
from four_goal_bmd.factory import pretrain_frozen_policy
from four_goal_bmd.metrics import aggregate_metrics, format_metrics
from four_goal_bmd.utils import configure_cpu, metadata, save_json, save_resolved_config


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
    print("Pretraining one frozen BC policy shared by all training seeds...")
    frozen_policy, bc_history = pretrain_frozen_policy(cfg, verbose=True)
    print(f"action_scale={frozen_policy.action_scale:.6f}")
    save_json(output_dir / "bc_history.json", bc_history)
    if cfg.logging.save_checkpoints:
        save_bc_checkpoint(output_dir / "bc_checkpoint.pt", frozen_policy, cfg)

    goal_centers = make_goal_centers(
        n_goals=cfg.environment.n_goals,
        radius=cfg.environment.goal_radius,
        rotation=cfg.landscape.rotation,
    )
    bc_evaluations = []
    for evaluation_seed in cfg.seeds.evaluation:
        bc_evaluations.append(
            evaluate_policy(
                frozen_policy=frozen_policy,
                goal_centers=goal_centers,
                environment_cfg=cfg.environment,
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
            environment_cfg=cfg.environment,
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
                environment_cfg=cfg.environment,
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
