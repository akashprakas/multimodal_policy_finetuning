"""Inspect aggregate and per-seed results produced by train.py."""

import argparse
import json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("summary")
    parser.add_argument("--runs", action="store_true")
    args = parser.parse_args()
    with open(args.summary) as f:
        summary = json.load(f)

    print(f"{summary['experiment']} / {summary['policy']} / {summary['landscape']}")
    mean, std = summary["mean"], summary["std"]
    print(
        f"mean SR={mean['SR']:.3f}±{std['SR']:.3f} "
        f"SR_M={mean['SR_M']:.3f}±{std['SR_M']:.3f} "
        f"mcf={mean['mc@0.80_floored']:.2f}±{std['mc@0.80_floored']:.2f}/4 "
        f"H/logM={mean['H_norm']:.3f}±{std['H_norm']:.3f}"
    )
    if args.runs:
        for run in summary["runs"]:
            metrics = run["metrics"]
            attempts = run["evaluations"][0]["attempts_per_mode"]
            print(
                f"seed={run['training_seed']:3d} SR={metrics['SR']:.3f} "
                f"SR_M={metrics['SR_M']:.3f} "
                f"mcf={metrics['mc@0.80_floored']:.1f}/4 "
                f"H/logM={metrics['H_norm']:.3f} attempts={attempts}"
            )


if __name__ == "__main__":
    main()
