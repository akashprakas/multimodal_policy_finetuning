# ReinFlow Ablation Study

Date: 2026-09-28

The corresponding 45° out-of-distribution study is documented in
[ReinFlow g2 Study](reinflow_g2_study.md).

This report evaluates the five ReinFlow mechanisms retained in this
repository:

1. plain ReinFlow;
2. fully frozen policy;
3. trainable exploration-noise head with a frozen velocity field;
4. upstream single-point output anchor;
5. functional anchor.

An exploratory relational variant was tested and then removed from the supported method
because it did not improve the conclusions.
The purpose is to separate three properties that can otherwise be confused:

- task adaptation;
- preservation of multiple behavioral modes;
- stochasticity without useful task adaptation.

## Shared experimental setting

All runs used:

```text
policy                 single_step_matched
landscape              g5 (rotation approximately 7.88 degrees)
BC goal radius         1.00
fine-tuning radius     1.22
BC updates             20,000
RL epochs              300
training seeds         [0, 1, 2]
evaluation seed        777
evaluation episodes    256 per training seed
minimum attempts       5 per mode for mc@0.8_floored
flow steps             4
```

The deterministic flow-BC evaluation was:

```text
SR=0.625  SR_M=0.613  mcf=1/4  H_norm=0.983  H_succ_norm=0.956
```

The stochastic actor immediately before RL, averaged over the three actor
initialization seeds, was:

```text
SR=0.598  SR_M=0.594  mcf=0/4  H_norm=0.999  H_succ_norm=0.981
```

Thus, the initial policy is highly diverse but under-reaches the shifted goals.
A successful method must improve task success without collapsing those modes.

## Aggregate results

Values are mean ± population standard deviation over training seeds 0–2.

| Method | λ | SR | SR_M | mc@0.8 | H/logM | Hs/logM | Radius |
|---|---:|---:|---:|---:|---:|---:|---:|
| Plain ReinFlow | — | 1.000 ± 0.000 | 0.250 ± 0.000 | 1.00 ± 0.00 | 0.000 ± 0.000 | 0.000 ± 0.000 | 1.000 ± 0.002 |
| Fully frozen | — | 0.578 ± 0.013 | 0.564 ± 0.010 | 0.00 ± 0.00 | 0.978 ± 0.006 | 0.954 ± 0.007 | 0.941 ± 0.001 |
| Noise-only | — | 0.555 ± 0.000 | 0.543 ± 0.000 | 0.00 ± 0.00 | 0.994 ± 0.000 | 0.966 ± 0.000 | 0.927 ± 0.000 |
| Upstream output anchor | 0.01 | 1.000 ± 0.000 | 0.917 ± 0.118 | 2.33 ± 0.47 | 0.562 ± 0.048 | 0.562 ± 0.048 | 0.978 ± 0.002 |
| Functional anchor | 0.03 | **1.000 ± 0.000** | **1.000 ± 0.000** | **4.00 ± 0.00** | **0.959 ± 0.018** | **0.959 ± 0.018** | 0.975 ± 0.001 |

The main result is clear: functional anchoring is the only retained method that
simultaneously achieves complete adaptation and preserves all four modes.

## 1. Plain ReinFlow

### Command

```bash
uv run python train.py \
  run=diagnostic \
  policy=single_step_matched \
  experiment=reinflow_plain \
  landscape=g5 \
  shift=adaptation_v1 \
  hydra.run.dir=outputs/reinflow_ablations/plain
```

Result: [summary.json](../outputs/reinflow_ablations/plain/summary.json)

### Per-seed result

| Seed | SR | SR_M | mc@0.8 | Hs/logM | Attempts per mode |
|---:|---:|---:|---:|---:|---|
| 0 | 1.000 | 0.250 | 1/4 | 0.000 | `[0, 0, 0, 256]` |
| 1 | 1.000 | 0.250 | 1/4 | 0.000 | `[0, 0, 0, 256]` |
| 2 | 1.000 | 0.250 | 1/4 | 0.000 | `[0, 0, 0, 256]` |

### Interpretation

Plain ReinFlow solves the reward-maximization problem perfectly, but every
rollout goes to the same goal. This is complete mode collapse, not a failure to
learn the task. It demonstrates why raw success rate alone is insufficient for
multimodal policy evaluation.

The paired-noise mode-flip rate is exactly zero: after fine-tuning, changing the
injected transition noise no longer changes the selected mode.

## 2. Fully frozen control

### Command

```bash
uv run python train.py \
  run=diagnostic \
  policy=single_step_matched \
  experiment=reinflow_frozen \
  landscape=g5 \
  shift=adaptation_v1 \
  hydra.run.dir=outputs/reinflow_ablations/frozen
```

Result: [summary.json](../outputs/reinflow_ablations/frozen/summary.json)

### Per-seed result

| Seed | SR | SR_M | mc@0.8 | Hs/logM | Attempts per mode |
|---:|---:|---:|---:|---:|---|
| 0 | 0.578 | 0.564 | 0/4 | 0.959 | `[62, 73, 44, 77]` |
| 1 | 0.594 | 0.576 | 0/4 | 0.944 | `[60, 79, 37, 80]` |
| 2 | 0.562 | 0.552 | 0/4 | 0.959 | `[61, 74, 41, 80]` |

### Interpretation

Freezing preserves a balanced mode distribution, but the policy continues to
under-reach the radius-1.22 goals. No mode reaches the required 80% success rate.
This is the anti-freezing control: diversity by refusing to adapt is not a
successful solution.

## 3. Noise-head-only adaptation

### Command

```bash
uv run python train.py \
  run=diagnostic \
  policy=single_step_matched \
  experiment=reinflow_noise_only \
  landscape=g5 \
  shift=adaptation_v1 \
  hydra.run.dir=outputs/reinflow_ablations/noise_only
```

Result: [summary.json](../outputs/reinflow_ablations/noise_only/summary.json)

### Per-seed result

All three seeds produced the same evaluation result:

```text
SR=0.555  SR_M=0.543  mcf=0/4  H_norm=0.994  H_succ_norm=0.966
attempts=[59, 69, 54, 74]
successes=[27, 37, 25, 53]
```

### Interpretation

The policy remains diverse, and changing injected noise frequently changes its
mode (`mode_flip_rate=0.582`). Nevertheless, task performance is slightly worse
than the pre-RL actor. Exploration variance cannot compensate for a velocity
field that still produces actions appropriate for the old radius.

This distinguishes stochasticity from useful multimodal adaptation.

## 4. Upstream single-point output anchor

This uses ReinFlow's single-point regularizer at its strongest previously
observed development setting, `λ=0.01`. It compares deterministic outputs only
at `x0=0`.

### Command

```bash
uv run python train.py \
  run=diagnostic \
  policy=single_step_matched \
  experiment=reinflow_anchor_upstream \
  experiment.anchor_coef=0.01 \
  landscape=g5 \
  shift=adaptation_v1 \
  hydra.run.dir=outputs/reinflow_ablations/upstream_lambda_0.01
```

Result: [summary.json](../outputs/reinflow_ablations/upstream_lambda_0.01/summary.json)

### Per-seed result

| Seed | SR | SR_M | mc@0.8 | Hs/logM | Attempts per mode |
|---:|---:|---:|---:|---:|---|
| 0 | 1.000 | 1.000 | 3/4 | 0.627 | `[7, 125, 4, 120]` |
| 1 | 1.000 | 0.750 | 2/4 | 0.514 | `[0, 116, 1, 139]` |
| 2 | 1.000 | 1.000 | 2/4 | 0.545 | `[2, 125, 1, 128]` |

### Interpretation

The upstream anchor improves substantially over plain ReinFlow, but it does not
preserve a balanced four-mode policy. Most trajectories occupy only two modes.

This result also illustrates an important `SR_M` caveat. Seeds 0 and 2 report
`SR_M=1.0` because every *attempted* mode succeeds, even though some modes receive
only one to seven attempts. Entropy, the attempt counts, and floored coverage
correctly expose the imbalance.

Constraining the decoder at one initial-noise point does not constrain the full
noise-to-action map. The remaining noise directions can still be reorganized or
folded into a smaller number of modes.

## 5. Functional anchor

The functional anchor compares fine-tuned and reference outputs using the same
random `x0` samples. It therefore constrains the complete sampled
noise-to-action function rather than only `x0=0`.

### Command

```bash
uv run python train.py \
  run=diagnostic \
  policy=single_step_matched \
  experiment=reinflow_anchor_functional \
  experiment.anchor_coef=0.03 \
  landscape=g5 \
  shift=adaptation_v1 \
  hydra.run.dir=outputs/reinflow_ablations/functional_lambda_0.03
```

Result: [summary.json](../outputs/reinflow_ablations/functional_lambda_0.03/summary.json)

### Per-seed result

| Seed | SR | SR_M | mc@0.8 | Hs/logM | Attempts per mode | Final anchor-gradient ratio |
|---:|---:|---:|---:|---:|---|---:|
| 0 | 1.000 | 1.000 | 4/4 | 0.974 | `[74, 49, 46, 87]` | 0.190 |
| 1 | 1.000 | 1.000 | 4/4 | 0.934 | `[49, 76, 30, 101]` | 0.263 |
| 2 | 1.000 | 1.000 | 4/4 | 0.969 | `[53, 66, 43, 94]` | 0.213 |

### Interpretation

The functional anchor completely solves the adaptation-preservation trade-off
in this setting. Every trajectory succeeds, every goal has 100% conditional
success, and every goal receives enough attempts to satisfy mode coverage.

The mean final anchor-gradient ratio is approximately `0.222`. The anchor is a
meaningful but minority component of the velocity-field gradient, so the result
is not explained by freezing the actor.

## Cross-ablation conclusions

### 1. Unconstrained reward optimization causes mode collapse

Plain ReinFlow learns the shifted task but maps every rollout to one goal. This
is the central failure the anchors must repair.

### 2. Diversity alone is not adaptation

Frozen and noise-only policies retain high entropy but stay close to the old
action radius and fail the shifted task. A useful preservation method must allow
the velocity field to change.

### 3. A single-point constraint is insufficient

The upstream anchor is better than no anchor, but it preserves only two dominant
modes. Its apparently strong `SR_M` demonstrates why mode visitation and entropy
must be reported alongside per-mode conditional success.

### 4. Anchoring the paired noise-to-action function works

The functional anchor achieves:

```text
SR=1.0, SR_M=1.0, mc@0.8=4/4 on all three seeds.
```

This is strong evidence that collapse is associated with destructive
reorganization of the noise-to-action map, not merely insufficient action-level
randomness.

## Implementation checks

Across all trainable ReinFlow runs, the final flow-chain PPO ratio-sanity error
was between approximately `1e-7` and `2.4e-7`, far below the configured `1e-4`
tolerance. This confirms that sampling-time and update-time flow-chain
log-probabilities remain synchronized.

The repository test suite also passes:

```bash
uv run python -m pytest -q
# 14 passed
```

## Scope and recommended next experiment

These are development results: three training seeds and 256 evaluation episodes
per seed, with `min_attempts=5`. They are appropriate for implementation
validation and the main mechanistic comparison, but not yet the strongest
publication protocol.

For the blog's final table, rerun the selected methods with held-out seeds 3–10,
1024 evaluation episodes, and `evaluation.min_attempts=103`. At minimum this
should include plain, frozen, noise-only, and functional `λ=0.03`. The upstream
arm can remain a development-only negative result if the selection rule is
documented in advance.
