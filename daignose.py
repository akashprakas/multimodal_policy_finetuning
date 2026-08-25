from __future__ import annotations

from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from four_goal_bmd.config_schema import apply_run_profile, validate_config
from four_goal_bmd.diagnostics import (
    expert_replay,
    grid_search_reachability,
    hold_noise_probe,
)
from four_goal_bmd.envs.four_goals import make_goal_centers
from four_goal_bmd.evaluation import evaluate_policy
from four_goal_bmd.factory import pretrain_frozen_policy
from four_goal_bmd.metrics import format_metrics
from four_goal_bmd.utils import configure_cpu, metadata, save_json, save_resolved_config


@hydra.main(version_base="1.3", config_path="configs", config_name="diagnose")
def main(cfg: DictConfig) -> None:
    configure_cpu()
    apply_run_profile(cfg)
    validate_config(cfg)
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    save_resolved_config(output_dir / "resolved_config.yaml", cfg)
    save_json(output_dir / "metadata.json", metadata(Path(__file__).resolve().parent))
    frozen_policy, bc_history = pretrain_frozen_policy(cfg, verbose=True)
    print(f"action_scale={frozen_policy.action_scale:.6f}")
    save_json(output_dir / "bc_history.json", bc_history)
    goals = make_goal_centers(
        cfg.environment.n_goals, cfg.environment.goal_radius, cfg.landscape.rotation
    )

    direct = evaluate_policy(
        frozen_policy,
        goals,
        cfg.environment,
        cfg.policy,
        cfg.evaluation.n_episodes,
        cfg.seeds.evaluation[0],
        cfg.evaluation.min_attempts,
    )
    print("BC prior:", format_metrics(direct))

    result = {"diagnostic": cfg.diagnostic.name, "bc_prior": direct}
    if cfg.diagnostic.name == "bc_prior":
        pass
    elif cfg.diagnostic.name == "grid_search":
        oracle_success = expert_replay(
            goals,
            cfg.environment,
            cfg.policy,
            n_episodes=cfg.evaluation.n_episodes,
            seed=cfg.seeds.evaluation[0],
        )
        reachability = grid_search_reachability(
            frozen_policy,
            goals,
            cfg.environment,
            cfg.policy,
            cfg.diagnostic.grid_size,
            cfg.diagnostic.w_limit,
        )
        result.update(
            {"expert_replay_sr": oracle_success, "reachability": reachability}
        )
        print(f"expert replay SR={oracle_success:.3f}")
        for row in reachability:
            print(
                f"goal={row['goal']} reached={row['reached']} "
                f"final_distance={row['final_distance']:.3f}"
            )
    elif cfg.diagnostic.name == "hold_w":
        result["rows"] = hold_noise_probe(
            frozen_policy,
            goals,
            cfg.environment,
            cfg.policy,
            list(cfg.diagnostic.hold_lengths),
            cfg.diagnostic.n_episodes,
            cfg.seeds.evaluation[0],
            cfg.evaluation.min_attempts,
        )
    else:
        raise ValueError(f"Unknown diagnostic: {cfg.diagnostic.name}")
    save_json(output_dir / "diagnostic.json", result)


if __name__ == "__main__":
    main()
