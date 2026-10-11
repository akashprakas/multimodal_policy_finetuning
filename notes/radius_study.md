# Functional Anchoring Under a Radius Shift

## Experimental protocol

All commands were run from the repository root with `uv run python train.py`.
The development runs use training seeds 0–2, 256 evaluation episodes, and an
attempt floor of 26. Final runs use held-out training seeds 3–10, 1024 evaluation
episodes, and an attempt floor of 103. The BC policy is trained on unrotated goals
at radius 1.0; `landscape=g5 shift=dsrl_v1` applies rotation 0.137571 radians and
radius 1.16 only during RL fine-tuning and evaluation. Every command used to
produce a result is included immediately before that result.

The study ran correctly and strongly reproduces the functional-anchor result. All histories completed without NaNs, the configurations are consistent, and the held-out evaluation uses 8 seeds × 1024 episodes with the correct 103-attempt floor.

## Main conclusion

At radius 1.16:

| Method | SR | SR_M | mcf | H_succ | Outcome |
|---|---:|---:|---:|---:|---|
| Initialized actor | 0.551 | 0.533 | 0.00 | 0.936 | Diverse but insufficiently adapted |
| Plain DSRL | 1.000 | 0.250 | 1.00 | 0.000 | Complete collapse |
| K=8, no MI | 0.998 | 0.406 | 1.50 | 0.191 | Latent capacity alone fails |
| BMD schedule, no MI | 0.973 | 0.545 | 2.00 | 0.456 | Curriculum alone is insufficient |
| Full BMD | 0.973 | 0.895 | 3.50 | 0.886 | Strong average, but seed failures |
| Functional anchor λ=1 | 0.964 | 0.963 | 4.00 | 0.948 | Best and most reliable |

The clean result is:

> Plain RL achieves perfect task success by collapsing to one goal. The functional anchor retains all four modes across every held-out seed while raising success from approximately 55% before RL to 96%.

The result closely matches the completed reproduction:

```text
Completed repository: SR=0.963, SR_M=0.962
Current repository:   SR=0.964, SR_M=0.963
```

---

# 1. Lambda development sweep

```bash
uv run python train.py -m \
  run=diagnostic \
  run.n_evaluation_episodes=256 \
  evaluation.min_attempts=26 \
  policy=single_step_matched \
  experiment=plain_dsrl \
  landscape=g5 \
  shift=dsrl_v1 \
  experiment.dsrl_anchor_type=functional \
  experiment.dsrl_anchor_coef=0.0,0.3,0.5,0.7,1.0,1.5,3.0 \
  hydra.sweep.dir='multirun/radius_shift/lambda_dev' \
  'hydra.sweep.subdir=lambda_${experiment.dsrl_anchor_coef}'
```

Results from [lambda_dev](../multirun/radius_shift/lambda_dev/multirun.yaml):

| λ | SR | SR_M | mcf | H_succ | Interpretation |
|---:|---:|---:|---:|---:|---|
| 0.0 | 0.974 | 0.321 | 1.33 | 0.167 | Mostly collapsed |
| 0.3 | 0.993 | 0.249 | 1.00 | Complete collapse |
| 0.5 | 1.000 | 0.250 | 1.00 | Complete collapse |
| 0.7 | 0.957 | 0.815 | 2.33 | Partial transition |
| 1.0 | 0.947 | 0.943 | 3.67 | Best success/diversity balance |
| 1.5 | 0.897 | 0.904 | 4.00 | Stronger preservation, less adaptation |
| 3.0 | 0.760 | 0.757 | 0.67 | Over-constrained |

### Interpretation

There is a sharp transition between λ=0.5 and λ=1.0:

```text
weak anchor → collapse
moderate anchor → adaptation + preservation
strong anchor → preservation but under-adaptation
```

λ=3 has high entropy (`H_succ=0.964`) but low per-mode success. This is important: balanced visitation does not mean the task has been solved.

λ=1 was the correct selection because it has substantially higher `SR` and `SR_M` than λ=1.5. Its development `mcf=3.67` was caused by one goal receiving only 19 attempts against the 26-attempt floor; raw `mc@0.80` was still 4/4.

The held-out run confirms λ=1 was not overfit to development seeds.

---

# 2. Initialized actor

```bash
uv run python train.py \
  run=final run.n_evaluation_episodes=1024 evaluation.min_attempts=103 \
  policy=single_step_matched experiment=plain_dsrl landscape=g5 shift=dsrl_v1 \
  experiment.total_epochs=1 experiment.steering_lr=0.0 \
  hydra.run.dir='outputs/radius_shift/final_initialized_actor'
```

[Result](../outputs/radius_shift/final_initialized_actor/summary.json):

```text
SR=0.551 ± 0.035
SR_M=0.533 ± 0.016
mcf=0/4
H_succ=0.936
```

The initialized actor is diverse but not sufficiently successful. This establishes that the final functional-anchor result is not obtained by simply freezing the initial policy.

The functional anchor improves mode-balanced success by:

```text
ΔSR_M = 0.963 - 0.533 = +0.429
normalized adaptation gain ≈ 0.920
```

So it achieves approximately 92% of the possible improvement between initialization and perfect mode-balanced success.

---

# 3. Plain DSRL

```bash
uv run python train.py \
  run=final run.n_evaluation_episodes=1024 evaluation.min_attempts=103 \
  policy=single_step_matched experiment=plain_dsrl landscape=g5 shift=dsrl_v1 \
  hydra.run.dir='outputs/radius_shift/final_plain_dsrl'
```

[Result](../outputs/radius_shift/final_plain_dsrl/summary.json):

```text
SR=1.000
SR_M=0.250
mcf=1/4
H_succ=0.000
```

Every seed converges to exactly one goal.

This is the clearest baseline in the study:

- PPO successfully adapts to the shifted reward.
- The policy is not failing optimization.
- The reward is satisfied by specializing in one goal.
- Without a diversity-preserving constraint, all stochastic modes collapse.

---

# 4. K=8 capacity without MI

```bash
uv run python train.py \
  run=final run.n_evaluation_episodes=1024 evaluation.min_attempts=103 \
  policy=single_step_matched experiment=plain_dsrl experiment.k_latent=8 \
  landscape=g5 shift=dsrl_v1 \
  hydra.run.dir='outputs/radius_shift/final_k8_capacity_only'
```

[Result](../outputs/radius_shift/final_k8_capacity_only/summary.json):

```text
SR=0.998
SR_M=0.406
mcf=1.50/4
H_succ=0.191
```

Simply giving the actor eight latent codes does not create reliable diversity.

Individual seeds solve only one or two modes. Some technically reach a second goal a few times, but not enough to satisfy the attempt floor.

Therefore:

> Diversity is not caused merely by increasing latent capacity.

A latent becomes useful only when training provides a reason for different latent values to represent different behavior.

---

# 5. BMD without MI

```bash
uv run python train.py \
  run=final run.n_evaluation_episodes=1024 evaluation.min_attempts=103 \
  policy=single_step_matched experiment=bmd landscape=g5 shift=dsrl_v1 \
  experiment.intrinsic_scale=0.0 experiment.finetune_intrinsic_scale=0.0 \
  hydra.run.dir='outputs/radius_shift/final_bmd_no_mi'
```

[Result](../outputs/radius_shift/final_bmd_no_mi/summary.json):

```text
SR=0.973
SR_M=0.545
mcf=2.00/4
H_succ=0.456
```

This retains:

- K=8
- curriculum
- 120 warmup epochs
- classifier training

but removes intrinsic MI reward.

It performs much worse than full BMD:

```text
SR_M: 0.545 → 0.895
mcf:  2.00 → 3.50
Hsucc: 0.456 → 0.886
```

Its mode-balanced success is almost unchanged from initialization:

```text
initialized SR_M = 0.533
no-MI SR_M       = 0.545
```

This is strong evidence that BMD’s benefit is coming from the MI reward—not merely K=8, the curriculum, or the classifier existing.

---

# 6. Full BMD

```bash
uv run python train.py \
  run=final run.n_evaluation_episodes=1024 evaluation.min_attempts=103 \
  policy=single_step_matched experiment=bmd landscape=g5 shift=dsrl_v1 \
  hydra.run.dir='outputs/radius_shift/final_bmd'
```

[Result](../outputs/radius_shift/final_bmd/summary.json):

```text
SR=0.973 ± 0.036
SR_M=0.895 ± 0.116
mcf=3.50 ± 0.71
H_succ=0.886 ± 0.099
```

Full BMD is clearly effective, but less reliable than the functional anchor.

Five of eight seeds retain 4/4 modes. Three fail:

- Seed 7: one goal receives only 13 attempts and zero successes.
- Seed 8: one mode has 27.3% success and another 70.7%.
- Seed 9: one goal receives one attempt and zero successes.

This is not merely an attempt-floor artifact for BMD: seeds 7 and 9 effectively lose one mode entirely.

Comparison against functional anchoring:

```text
Functional − BMD:
SR:      -0.009
SR_M:    +0.068
mcf:     +0.50 modes
H_succ:  +0.062
```

The functional anchor sacrifices approximately one percentage point of raw success but is dramatically more consistent across training seeds.

The appropriate claim is:

> Functional anchoring was more reliable than BMD on this controlled radius-shift task, not necessarily universally better than BMD.

BMD still provides addressable modes through `z`; the functional anchor preserves unlabelled stochastic modes.

---

# 7. Functional anchor λ=1

```bash
uv run python train.py \
  run=final run.n_evaluation_episodes=1024 evaluation.min_attempts=103 \
  policy=single_step_matched experiment=plain_dsrl landscape=g5 shift=dsrl_v1 \
  experiment.dsrl_anchor_type=functional experiment.dsrl_anchor_coef=1.0 \
  hydra.run.dir='outputs/radius_shift/final_functional_lambda_1'
```

[Result](../outputs/radius_shift/final_functional_lambda_1/summary.json):

```text
SR=0.964 ± 0.009
SR_M=0.963 ± 0.009
mcf=4.00 ± 0.00
H_succ=0.948 ± 0.025
```

This is an exceptionally stable result:

- 8/8 seeds retain all four modes.
- Every seed has `SR_M > 0.94`.
- Every seed has `SR > 0.95`.
- Low standard deviation.
- Attempts remain reasonably balanced.
- The final radius increases from approximately 0.872 to 0.913.

The anchor therefore does not simply freeze the original behavior. It permits enough functional change to reach farther goals while preventing the noise-to-action map from reorganizing into one dominant direction.

---

# 8. Radius robustness

| Goal radius | BC SR_M | Anchor SR | Anchor SR_M | mcf |
|---:|---:|---:|---:|---:|
| 1.16 | 0.562 | 0.964 | 0.963 | 4.00 |
| 1.18 | 0.490 | 0.954 | 0.952 | 3.875 |
| 1.20 | 0.376 | 0.937 | 0.935 | 4.00 |

### Radius 1.18

The development-selected λ=1.0 is reused without retuning.

```bash
uv run python train.py \
  run=final run.n_evaluation_episodes=1024 evaluation.min_attempts=103 \
  policy=single_step_matched experiment=plain_dsrl landscape=g5 \
  shift=dsrl_v1_r118 \
  experiment.dsrl_anchor_type=functional experiment.dsrl_anchor_coef=1.0 \
  hydra.run.dir='outputs/radius_shift/robustness_r118'
```

[Result](../outputs/radius_shift/robustness_r118/summary.json):

Seven seeds pass the floored 4/4 criterion. Seed 9 reports 3/4 only because one mode has 91 attempts, below the required 103:

```text
attempts: [91, 192, 381, 360]
per-mode SR: [0.945, 0.948, 0.969, 0.947]
```

All four modes are actually competent. This is a visitation-balance/floor failure, not mode failure.

### Radius 1.20

The same λ=1.0 is again reused without retuning.

```bash
uv run python train.py \
  run=final run.n_evaluation_episodes=1024 evaluation.min_attempts=103 \
  policy=single_step_matched experiment=plain_dsrl landscape=g5 \
  shift=dsrl_v1_r120 \
  experiment.dsrl_anchor_type=functional experiment.dsrl_anchor_coef=1.0 \
  hydra.run.dir='outputs/radius_shift/robustness_r120'
```

[Result](../outputs/radius_shift/robustness_r120/summary.json):

```text
SR=0.937
SR_M=0.935
mcf=4/4 for all eight seeds
H_succ=0.964
```

Despite the larger shift and much weaker BC prior, the same λ=1 works without retuning.

That is strong evidence that the result is not specific to radius 1.16.

---

# 9. BC quality study

```bash
uv run python train.py -m \
  run=diagnostic \
  evaluation.min_attempts=26 \
  policy=single_step_matched \
  policy.bc_steps=2000,10000,20000 \
  experiment=plain_dsrl \
  landscape=g5 \
  shift=dsrl_v1 \
  experiment.dsrl_anchor_type=functional \
  experiment.dsrl_anchor_coef=1.0 \
  hydra.sweep.dir='multirun/radius_shift/bc_quality' \
  'hydra.sweep.subdir=bc_${policy.bc_steps}'
```

| BC steps | BC SR | BC SR_M | Anchored SR | Anchored SR_M | mcf |
|---:|---:|---:|---:|---:|---:|
| 2k | 0.012 | 0.011 | 0.076 | 0.038 | 0.00 |
| 10k | 0.293 | 0.258 | 0.888 | 0.753 | 2.67 |
| 20k | 0.551 | 0.544 | 0.947 | 0.943 | 3.67 |

This is a clean monotonic result.

The functional anchor does not create useful modes from an incompetent reference:

- At 2k, the BC barely reaches the shifted task; anchoring fails.
- At 10k, the reference contains partially useful behavior; anchoring produces partial coverage.
- At 20k, the reference is diverse and competent enough; anchoring achieves near-complete adaptation.

The correct claim is:

> Functional anchoring preserves and adapts useful multimodality already represented by the pretrained policy. It cannot manufacture missing behavioral competence.


---

# Overall conclusions

The experiments support four strong conclusions:

1. Reward-only RL fine-tuning causes complete mode collapse despite perfect task success.
2. Latent capacity and curriculum without MI do not reliably prevent collapse.
3. BMD substantially improves diversity, confirming that its MI objective is doing meaningful work.
4. A functional anchor can preserve diversity without discrete latents or MI when the pretrained policy already contains competent multimodal behavior.

The strongest result is reliability:

```text
Functional anchor: 8/8 full-coverage seeds
Full BMD:          5/8 full-coverage seeds
Plain DSRL:        0/8
```

