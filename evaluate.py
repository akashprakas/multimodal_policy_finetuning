from __future__ import annotations

from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from four_goal_bmd.checkpointing import load_training_checkpoint
from four_goal_bmd.config_schema import apply_run_profile
from four_goal_bmd.envs.four_goals import make_goal_centers
from four_goal_bmd.evaluation import evaluate_policy
from four_goal_bmd.metrics import aggregate_metrics, format_metrics
from four_goal_bmd.task_shift import finetuning_environment
from four_goal_bmd.utils import configure_cpu, save_json


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    configure_cpu()
    apply_run_profile(cfg)
    if not cfg.checkpoint_path:
        raise ValueError("Set checkpoint_path=/path/to/seed_xxx/checkpoint.pt")
    saved_cfg, frozen_policy, steering_policy, _, _, payload = load_training_checkpoint(
        cfg.checkpoint_path
    )
    environment_cfg = finetuning_environment(saved_cfg)
    goal_centers = make_goal_centers(
        environment_cfg.n_goals,
        environment_cfg.goal_radius,
        saved_cfg.landscape.rotation,
    )
    evaluations = []
    for evaluation_seed in cfg.seeds.evaluation:
        row = evaluate_policy(
            frozen_policy=frozen_policy,
            steering_policy=steering_policy,
            k_latent=saved_cfg.experiment.k_latent,
            goal_centers=goal_centers,
            environment_cfg=environment_cfg,
            policy_cfg=saved_cfg.policy,
            n_episodes=cfg.evaluation.n_episodes,
            evaluation_seed=evaluation_seed,
            min_attempts=cfg.evaluation.min_attempts,
            deterministic_steering=cfg.evaluation.deterministic_steering,
        )
        row["evaluation_seed"] = int(evaluation_seed)
        evaluations.append(row)
    mean, std = aggregate_metrics(evaluations)
    result = {
        "checkpoint": str(Path(cfg.checkpoint_path).resolve()),
        "training_seed": payload["training_seed"],
        "shift": saved_cfg.shift.name if saved_cfg.get("shift") is not None else None,
        "goal_radius": float(environment_cfg.goal_radius),
        "mean": mean,
        "std": std,
        "evaluations": evaluations,
    }
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    save_json(output_dir / "evaluation.json", result)
    print(format_metrics({**mean, "n_goals": 4}))


if __name__ == "__main__":
    main()
