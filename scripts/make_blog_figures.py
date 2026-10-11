"""Figures for research documentation and analysis, drawn from the run summaries.

Run from the repo root:  uv run python scripts/make_blog_figures.py
Writes PNG + SVG into notes/figures/. Coverage is the floored count
(`mc@0.80_floored`), matching the draft's minimum-attempt rule.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notes" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

MODE_COLORS = ["#4C78A8", "#F58518", "#54A24B", "#B279A2"]
PLAIN = "#9A9A9A"
ANCHOR = "#1F77B4"
WARN = "#D62728"

plt.rcParams.update(
    {
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "figure.dpi": 120,
    }
)


def load(path: Path) -> dict:
    return json.loads((path / "summary.json").read_text())


def per_seed(summary: dict, key: str) -> list[float]:
    return [run["evaluations"][0][key] for run in summary["runs"]]


def coverage_per_seed(
    summary: dict,
    *,
    min_attempts: int,
    success_threshold: float = 0.8,
) -> list[float]:
    """Recompute coverage with one shared floor across older sweep configs."""
    coverage = []
    for run in summary["runs"]:
        evaluation = run["evaluations"][0]
        attempts = np.asarray(evaluation["attempts_per_mode"], dtype=float)
        successes = np.asarray(evaluation["successes_per_mode"], dtype=float)
        success_rates = np.divide(
            successes,
            attempts,
            out=np.zeros_like(successes),
            where=attempts > 0,
        )
        solved = (attempts >= min_attempts) & (success_rates >= success_threshold)
        coverage.append(float(solved.sum()))
    return coverage


def save(fig, name: str) -> None:
    fig.savefig(OUT / f"{name}.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / f"{name}.png")


def rotation_degrees(landscape: str) -> float:
    text = (ROOT / "configs" / "landscape" / f"{landscape}.yaml").read_text()
    radians = float(re.search(r"^rotation:\s*([0-9.eE+-]+)", text, re.M).group(1))
    return float(np.degrees(radians))


# --------------------------------------------------------------------------
# Figure 0: success hides collapse
# --------------------------------------------------------------------------
def fig_hook() -> None:
    final = ROOT / "outputs" / "blog_final"
    arms = [
        ("Plain RL", "reinflow_plain", PLAIN),
        ("Functional\nanchor", "reinflow_functional", ANCHOR),
    ]
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    width = 0.34
    for i, (label, folder, color) in enumerate(arms):
        summary = load(final / folder)
        for j, (metric, hatch) in enumerate([("SR", ""), ("SR_M", "//")]):
            values = per_seed(summary, metric)
            x = i + (j - 0.5) * width
            ax.bar(
                x,
                np.mean(values),
                width,
                color=color,
                hatch=hatch,
                edgecolor="white",
                alpha=0.9,
            )
            ax.scatter(
                np.full(len(values), x),
                values,
                s=10,
                color="black",
                zorder=3,
                alpha=0.6,
            )
            ax.text(
                x,
                np.mean(values) + 0.03,
                f"{np.mean(values):.2f}",
                ha="center",
                fontsize=9,
            )
    ax.set_xticks(range(len(arms)), [a[0] for a in arms])
    ax.set_ylim(0, 1.18)
    ax.set_ylabel("score (8 seeds, dots = seeds)")
    ax.set_title("Same task success, very different behavior")
    ax.bar([], [], color="#DDDDDD", edgecolor="black", label="SR (episodes)")
    ax.bar(
        [],
        [],
        color="#DDDDDD",
        edgecolor="black",
        hatch="//",
        label="SR_M (mode-balanced)",
    )
    ax.legend(
        frameon=False,
        loc="upper center",
        ncol=2,
        fontsize=9,
        bbox_to_anchor=(0.5, -0.14),
    )
    save(fig, "fig0_hook")


# --------------------------------------------------------------------------
# Figure 1: goal geometry
# --------------------------------------------------------------------------
def fig_geometry() -> None:
    bc = np.array(
        load(ROOT / "multirun/reinflow_rotation/functional_g0")["task"]["goal_centers"]
    )
    shifted = np.array(
        load(ROOT / "multirun/reinflow_lambda_g5/lambda_0.03")["task"]["goal_centers"]
    )
    rotated = np.array(
        load(ROOT / "multirun/reinflow_rotation/functional_g2")["task"]["goal_centers"]
    )
    success = load(ROOT / "multirun/reinflow_rotation/functional_g0")["task"][
        "success_distance"
    ]

    panels = [
        ("BC demonstrations", bc, None),
        ("Compatible shift\nradius 1.22, rotated 7.9°", shifted, bc),
        ("45° rotation", rotated, bc),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.9), sharex=True, sharey=True)
    for ax, (title, goals, reference) in zip(axes, panels):
        if reference is not None:
            for goal in reference:
                ax.add_patch(
                    Circle(goal, success, fill=False, ls="--", lw=1, color="#888888")
                )
        for index, goal in enumerate(goals):
            ax.add_patch(Circle(goal, success, color=MODE_COLORS[index], alpha=0.25))
            ax.scatter(*goal, color=MODE_COLORS[index], s=60, zorder=3)
            ax.add_patch(
                FancyArrowPatch(
                    (0, 0),
                    goal * 0.88,
                    arrowstyle="-|>",
                    mutation_scale=10,
                    color=MODE_COLORS[index],
                    lw=1.4,
                    alpha=0.8,
                )
            )
        ax.scatter(0, 0, color="black", s=25, zorder=4)
        ax.set_title(title, fontsize=11)
        ax.set_aspect("equal")
        ax.set_xlim(-1.7, 1.7)
        ax.set_ylim(-1.7, 1.7)
        ax.grid(False)
    axes[1].text(
        0,
        -1.62,
        "dashed = original success regions",
        ha="center",
        fontsize=8.5,
        color="#555555",
    )
    axes[2].text(
        0,
        -1.62,
        "new goals sit between old ones",
        ha="center",
        fontsize=8.5,
        color="#555555",
    )
    fig.suptitle("Four goals; shaded discs are the success regions", y=1.0, fontsize=12)
    save(fig, "fig1_geometry")


# --------------------------------------------------------------------------
# Figure 2: paired-noise anchor (method diagram)
# --------------------------------------------------------------------------
def box(ax, xy, w, h, text, face, edge="#333333", fs=10, weight="normal"):
    ax.add_patch(
        FancyBboxPatch(
            xy,
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.06",
            fc=face,
            ec=edge,
            lw=1.3,
        )
    )
    ax.text(
        xy[0] + w / 2,
        xy[1] + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fs,
        weight=weight,
    )


def arrow(ax, start, end, color="#333333", ls="-"):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=14,
            lw=1.5,
            color=color,
            linestyle=ls,
        )
    )


def fig_anchor() -> None:
    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    ax.set_xlim(0, 10.5)
    ax.set_ylim(0, 4.6)
    ax.axis("off")

    box(ax, (0.2, 1.85), 1.9, 0.9, "same noise\nx₀ ~ N(0, I)", "#EEF3FA", weight="bold")
    box(
        ax, (3.3, 3.0), 2.3, 0.95, "frozen BC flow\n(reference, no gradient)", "#F0F0F0"
    )
    box(ax, (3.3, 0.65), 2.3, 0.95, "fine-tuned flow\n(trainable)", "#DCEBFA")
    box(ax, (6.5, 3.0), 1.4, 0.95, "a₀", "#F0F0F0", fs=12)
    box(ax, (6.5, 0.65), 1.4, 0.95, "a_θ", "#DCEBFA", fs=12)
    box(
        ax,
        (8.55, 1.75),
        1.75,
        1.1,
        "anchor loss\n‖a_θ − a₀‖²",
        "#FDE9D9",
        weight="bold",
    )

    arrow(ax, (2.1, 2.5), (3.3, 3.45))
    arrow(ax, (2.1, 2.1), (3.3, 1.15))
    arrow(ax, (5.6, 3.45), (6.5, 3.45))
    arrow(ax, (5.6, 1.15), (6.5, 1.15))
    arrow(ax, (7.9, 3.35), (8.7, 2.7))
    arrow(ax, (7.9, 1.25), (8.7, 1.9))
    ax.text(
        5.4,
        2.1,
        "gradient flows only through\nthe trainable branch",
        fontsize=9,
        color=ANCHOR,
        ha="center",
        style="italic",
    )
    ax.text(
        9.43,
        1.35,
        "loss = PPO loss\n+ λ · anchor loss",
        fontsize=9.5,
        ha="center",
        color="#7A3E00",
    )
    ax.text(
        0.2,
        0.2,
        "Functional anchor: a fresh x₀ is drawn each step, so the whole noise → action map is constrained.\n"
        "'Upstream' anchor: x₀ = 0 only, so a single point is constrained.",
        fontsize=9.5,
        color="#444444",
        va="bottom",
    )
    ax.set_title("The paired-noise functional anchor", fontsize=12, loc="left")
    save(fig, "fig2_anchor")


# --------------------------------------------------------------------------
# Figure 3: preservation degrades with rotation
# --------------------------------------------------------------------------
def fig_rotation() -> None:
    base = ROOT / "multirun" / "reinflow_rotation"
    landscapes = ["g0", "g5", "g3", "g1", "g4", "g2"]
    degrees = np.array([rotation_degrees(g) for g in landscapes])
    min_attempts = 26  # Same 10%-of-256 floor used in the lambda comparison.
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.9))
    panels = [
        ("mc@0.80_floored", "modes covered (of 4)", (-0.2, 4.4)),
        ("SR_M", "mode-balanced success SR_M", (0, 1.08)),
    ]
    for ax, (metric, ylabel, ylim) in zip(axes, panels):
        for arm, color, label, offset in [
            ("plain", PLAIN, "plain RL", -0.5),
            ("functional", ANCHOR, "functional anchor, λ = 0.03", 0.5),
        ]:
            summaries = [load(base / f"{arm}_{g}") for g in landscapes]
            if metric == "mc@0.80_floored":
                seeds = np.array(
                    [
                        coverage_per_seed(summary, min_attempts=min_attempts)
                        for summary in summaries
                    ]
                )
            else:
                seeds = np.array([per_seed(summary, metric) for summary in summaries])
            ax.plot(
                degrees, seeds.mean(axis=1), "-o", color=color, label=label, lw=2, ms=5
            )
            for x, values in zip(degrees, seeds):
                ax.scatter(
                    np.full(len(values), x + offset),
                    values,
                    s=9,
                    color=color,
                    alpha=0.5,
                    zorder=3,
                )
        ax.set_xlabel("goal rotation relative to demonstrations (degrees)")
        ax.set_ylabel(ylabel)
        ax.set_ylim(*ylim)
        # Stagger neighbouring labels: 7.9 and 11.25 are too close to share a row.
        labels = [f"{d:.2f}".rstrip("0").rstrip(".") for d in degrees]
        ax.set_xticks(
            degrees, [("\n" if i % 2 else "") + text for i, text in enumerate(labels)]
        )
    axes[0].set_yticks(range(5))
    axes[0].legend(frameon=False, fontsize=9, loc="upper right")
    fig.suptitle(
        "The anchor's coverage advantage disappears under larger rotations "
        "(3 seeds, development sweep)",
        fontsize=11.5,
        y=1.02,
    )
    save(fig, "fig3_rotation")


# --------------------------------------------------------------------------
# Figure 4: the lambda window across DSRL and ReinFlow
# --------------------------------------------------------------------------
def fig_lambda() -> None:
    sweeps = [
        {
            "name": "DSRL",
            "base": ROOT / "multirun" / "radius_shift" / "lambda_dev",
            "lambdas": ["0.0", "0.3", "0.5", "0.7", "1.0", "1.5", "3.0"],
            "window": (3.65, 5.25),
        },
        {
            "name": "ReinFlow",
            "base": ROOT / "multirun" / "reinflow_lambda_g5",
            "lambdas": ["0.0", "0.01", "0.03", "0.1", "0.3", "1.0"],
            "window": (1.65, 3.35),
        },
    ]
    min_attempts = 26  # 10% of the 256 evaluation episodes, rounded up.

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.0))
    for row_index, sweep in enumerate(sweeps):
        summaries = [
            load(sweep["base"] / f"lambda_{value}") for value in sweep["lambdas"]
        ]
        xs = np.arange(len(summaries), dtype=float)
        tick_labels = [
            "0" if float(value) == 0 else f"{float(value):g}"
            for value in sweep["lambdas"]
        ]
        bc_sr = summaries[0]["bc"]["mean"]["SR"]
        coverage_ax, success_ax = axes[row_index]

        for ax in (coverage_ax, success_ax):
            ax.axvspan(*sweep["window"], color=ANCHOR, alpha=0.09)
            ax.set_xticks(xs, tick_labels)
            ax.set_xlabel(f"{sweep['name']} anchor coefficient λ")

        coverage = np.array(
            [
                coverage_per_seed(summary, min_attempts=min_attempts)
                for summary in summaries
            ]
        )
        coverage_ax.plot(xs, coverage.mean(axis=1), "-o", color=ANCHOR, lw=2)
        for x, values in zip(xs, coverage):
            coverage_ax.scatter(
                np.full(len(values), x), values, s=11, color=ANCHOR, alpha=0.5, zorder=3
            )
        coverage_ax.set_ylabel("modes covered (of 4)")
        coverage_ax.set_ylim(-0.2, 4.4)
        coverage_ax.set_yticks(range(5))
        coverage_ax.text(
            np.mean(sweep["window"]),
            4.15,
            "useful window",
            ha="center",
            fontsize=8.5,
            color=ANCHOR,
        )

        for metric, label, style, color in [
            ("SR", "SR", "-o", ANCHOR),
            ("SR_M", "SR_M", "--s", "#FF7F0E"),
        ]:
            data = np.array([per_seed(summary, metric) for summary in summaries])
            success_ax.plot(
                xs, data.mean(axis=1), style, color=color, lw=2, label=label
            )
        success_ax.axhline(bc_sr, color="#555555", ls=":", lw=1.3)
        success_ax.text(
            xs[-1] - 0.05,
            bc_sr - 0.065,
            f"BC prior SR = {bc_sr:.3f}",
            fontsize=8,
            color="#555555",
            ha="right",
        )
        success_ax.set_ylabel("success")
        success_ax.set_ylim(0, 1.1)
        if row_index == 0:
            success_ax.legend(frameon=False, fontsize=9, loc="lower left")

    axes[0, 0].set_title("Floored mode coverage", fontsize=11)
    axes[0, 1].set_title("Task and mode-balanced success", fontsize=11)
    fig.suptitle(
        "The same preservation–adaptation trade-off appears at different λ scales",
        fontsize=12,
        y=0.99,
    )
    fig.text(
        0.5,
        0.01,
        "Three development seeds per setting; coverage requires ≥26 of 256 attempts and conditional success ≥0.8.",
        ha="center",
        fontsize=9,
        color="#555555",
    )
    fig.tight_layout(rect=(0.0, 0.045, 1.0, 0.96), h_pad=2.0)
    save(fig, "fig4_lambda")


if __name__ == "__main__":
    fig_hook()
    fig_geometry()
    fig_anchor()
    fig_rotation()
    fig_lambda()
