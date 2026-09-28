# ReinFlow g2 Study

Date: 2026-09-28

## Question

Does functional anchoring work when the fine-tuning goals are rotated 45° away
from the cardinal directions represented in the BC demonstrations?

The answer from this study is **no**. The ReinFlow functional anchor reproduces
the same basic failure previously observed with DSRL:

- weak anchoring permits task adaptation but collapses to one goal;
- strong anchoring preserves the old multimodal visitation pattern but prevents
  successful adaptation to the rotated goals;
- none of the tested coefficients achieves useful multi-goal coverage.

Two settings were evaluated:

1. **Standard g2:** 45° rotation, goal radius 1.0, matching the earlier DSRL g2
   comparison.
2. **g2 + radius shift:** 45° rotation and radius 1.22, a harder stress test.

All results use training seeds `[0, 1, 2]`, 20,000 flow-BC updates, 300 RL
epochs, and 256 evaluation episodes per training seed.

## Standard g2: rotation only

### Main ablation table

| Method | λ | SR | SR_M | mc@0.8 | H/logM | Hs/logM | Radius |
|---|---:|---:|---:|---:|---:|---:|---:|
| Plain ReinFlow | — | 1.000 | 0.250 | 1/4 | 0.000 | 0.000 | 0.797 |
| Fully frozen | — | 0.000 | 0.000 | 0/4 | 0.985 | undefined | 0.960 |
| Noise-only | — | 0.000 | 0.000 | 0/4 | 0.993 | undefined | 0.961 |
| Upstream anchor | 0.01 | 1.000 | 0.250 | 1/4 | 0.000 | 0.000 | 0.773 |
| Functional anchor | 0.03 | 0.974 | 0.244 | 1/4 | 0.011 | 0.000 | 0.750 |

The BC prior has `SR=0`: it still produces diverse cardinal-axis behaviors, but
none reaches a diagonal g2 goal within the success threshold.

### Commands

Plain ReinFlow:

```bash
uv run python train.py \
  run=diagnostic policy=single_step_matched \
  experiment=reinflow_plain landscape=g2 shift=none \
  hydra.run.dir=outputs/reinflow_g2/plain
```

Fully frozen:

```bash
uv run python train.py \
  run=diagnostic policy=single_step_matched \
  experiment=reinflow_frozen landscape=g2 shift=none \
  hydra.run.dir=outputs/reinflow_g2/frozen
```

Noise-only:

```bash
uv run python train.py \
  run=diagnostic policy=single_step_matched \
  experiment=reinflow_noise_only landscape=g2 shift=none \
  hydra.run.dir=outputs/reinflow_g2/noise_only
```

Upstream anchor:

```bash
uv run python train.py \
  run=diagnostic policy=single_step_matched \
  experiment=reinflow_anchor_upstream experiment.anchor_coef=0.01 \
  landscape=g2 shift=none \
  hydra.run.dir=outputs/reinflow_g2/upstream_lambda_0.01
```

Functional anchor:

```bash
uv run python train.py \
  run=diagnostic policy=single_step_matched \
  experiment=reinflow_anchor_functional experiment.anchor_coef=0.03 \
  landscape=g2 shift=none \
  hydra.run.dir=outputs/reinflow_g2/functional_lambda_0.03
```

The corresponding summaries are under
[`outputs/reinflow_g2`](../outputs/reinflow_g2).

## Functional-anchor coefficient sweep

The selected g5 coefficient failing on g2 is not sufficient evidence by itself.
Therefore, the functional coefficient was swept over the same grid used in the
earlier DSRL investigation.

| λ | SR | SR_M | mc@0.8 | H/logM | Hs/logM | Interpretation |
|---:|---:|---:|---:|---:|---:|---|
| 0.01 | 1.000 | 0.250 | 1/4 | 0.000 | 0.000 | Full adaptation, complete collapse |
| 0.03 | 0.974 | 0.244 | 1/4 | 0.011 | 0.000 | Near-full adaptation, complete collapse |
| 0.10 | 0.142 | 0.047 | 0/4 | 0.569 | 0.000 | Partial visitation diversity, almost no useful adaptation |
| 0.30 | 0.008 | 0.006 | 0/4 | 0.986 | 0.167 | Prior largely preserved; essentially no success |
| 1.00 | 0.000 | 0.000 | 0/4 | 0.994 | undefined | Frozen near the unsuccessful prior |

Commands followed this template:

```bash
uv run python train.py \
  run=diagnostic policy=single_step_matched \
  experiment=reinflow_anchor_functional \
  experiment.anchor_coef=<0.01|0.03|0.1|0.3|1.0> \
  landscape=g2 shift=none \
  hydra.run.dir=outputs/reinflow_g2/functional_lambda_<lambda>
```

### Per-seed mode behavior

At `λ=0.01`, every seed sends all 256 trajectories to one goal:

```text
seed 0: [0, 0, 0, 256]
seed 1: [0, 0, 0, 256]
seed 2: [0, 0, 0, 256]
```

At `λ=0.03`, two seeds still send every trajectory to one goal; seed 1 sends
only two unsuccessful trajectories elsewhere:

```text
seed 0: [0, 0, 0, 256]
seed 1: [2, 0, 0, 254]
seed 2: [0, 0, 0, 256]
```

At `λ=0.3` and `λ=1.0`, visitation becomes balanced again, but nearly all or all
trajectories fail because the resulting actions remain aligned with the old BC
behavior rather than the rotated goals.

## Why functional anchoring fails on g2

For g5, the desired goals are only approximately 7.88° from the BC directions.
The policy mainly needs to increase action magnitude and make a small directional
correction. Functional anchoring can allow that local adjustment while preserving
the noise-to-mode mapping.

For g2, every desired direction is rotated 45° from the BC modes. Successful
adaptation requires a large, coordinated change to the noise-to-action function.
The task and anchor objectives therefore conflict:

- with small `λ`, reward optimization wins and all noise inputs are reorganized
  around one discovered diagonal goal;
- with large `λ`, the anchor wins and preserves diverse but unsuccessful
  cardinal-axis behavior;
- intermediate `λ` produces a compromise that is neither successful nor
  mode-covering.

The sweep therefore supports the interpretation that the functional anchor is a
**preservation mechanism**, not a multimodal discovery mechanism.

## Harder stress test: g2 plus radius 1.22

The same study was also run with `shift=adaptation_v1`, combining the 45°
rotation with an outward radius shift.

| Method | λ | SR | SR_M | mc@0.8 | H/logM | Hs/logM |
|---|---:|---:|---:|---:|---:|---:|
| Plain ReinFlow | — | 1.000 | 0.250 | 1/4 | 0.000 | 0.000 |
| Fully frozen | — | 0.000 | 0.000 | 0/4 | 0.985 | undefined |
| Noise-only | — | 0.000 | 0.000 | 0/4 | 0.993 | undefined |
| Upstream anchor | 0.01 | 1.000 | 0.250 | 1/4 | 0.000 | 0.000 |
| Functional anchor | 0.03 | 0.876 | 0.219 | 1/4 | 0.000 | 0.000 |

The functional sweep under this harder shift shows the same transition:

| λ | SR | mc@0.8 | H/logM | Outcome |
|---:|---:|---:|---:|---|
| 0.01 | 0.999 | 1/4 | 0.000 | Collapse |
| 0.03 | 0.876 | 1/4 | 0.000 | Collapse with reduced success |
| 0.10 | 0.001 | 0/4 | 0.726 | Anchor blocks adaptation |
| 0.30 | 0.000 | 0/4 | 0.986 | Prior preserved, no success |
| 1.00 | 0.000 | 0/4 | 0.989 | Prior preserved, no success |

Commands are identical to the standard g2 commands except:

```bash
shift=adaptation_v1
hydra.run.dir=outputs/reinflow_ablations_g2/<run_name>
```

## Comparison with the previous DSRL finding

The ReinFlow result agrees with the earlier DSRL conclusion. Changing the
fine-tuning backend from noise steering to direct flow-policy optimization does
not make functional anchoring discover four new diagonal modes.

ReinFlow improves the ability to optimize task reward: weakly anchored or plain
policies reliably find one successful diagonal goal. It does not supply pressure
to divide different noise regions among all four goals. BMD's mutual-information
objective provides exactly that missing specialization pressure.

## Research conclusion

The combined g5 and g2 evidence supports a narrower and more defensible claim:

> Functional anchoring can preserve multimodality during local or
> structure-compatible adaptation, but it does not independently discover a new
> set of out-of-distribution behavioral modes.

For g2, the next scientifically useful experiment is not another anchor-only
coefficient search. It is ReinFlow combined with an explicit mode-discovery
objective, or an anchor that factors out a learned global transformation before
penalizing changes to the internal mode geometry.
