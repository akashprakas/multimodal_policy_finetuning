## Plain DSRL (K=1) -> full BMD, single-step, landscape=g2

One factor changed at a time. `mcf` = `mc@0.80_floored` = count of goals (out
of 4) with per-goal success rate >= 80% AND >= min_attempts; reported as a
mean across 8 training seeds (3-10), so it is a fraction even though any one
seed's own mcf is always a whole number.

## 1. Plain DSRL baseline (K=1)

command: `train.py run=final policy=single_step experiment=plain_dsrl landscape=g2`

BC prior: SR=0.035 SR_M=0.028 mcf=0.00/4 H/logM=0.965 Hs/logM=0.459 radius=0.494

Aggregate: SR=1.000 SR_M=0.250 mcf=1.00/4 H/logM=0.000 Hs/logM=0.000 radius=0.760

With `k_latent=1` there is only one possible latent, so there is nothing *to*
collapse from -- every rollout shares the same output slot by construction.
Result: unconstrained task-reward-only training reaches one physical goal
every time (H=0.000 across all 8 seeds independently), despite 14 free
timesteps and no signal telling it which goal to prefer.

## 2. + K=8, no MI, no curriculum

command: `uv run python train.py run=final policy=single_step experiment=plain_dsrl landscape=g2 experiment.entropy_coef=0.0 experiment.k_latent=8`

Aggregate: SR=0.960 SR_M=0.474 mcf=1.83/4 H/logM=0.304 Hs/logM=0.292 radius=0.769

Real improvement despite zero MI reward. K=8 does not add stochasticity -- it
adds *capacity* (8 separate one-hot-conditioned "slots" through the same
shared network) for stochasticity that already existed (random init +
exploration noise) to resolve into distinguishable outcomes, instead of being
forced through one shared output and averaged into a single behavior.

## 3. + curriculum flag, warmup_epochs still 0 (inherited, unmodified)

command: `uv run python train.py run=final policy=single_step experiment=plain_dsrl landscape=g2 experiment.entropy_coef=0.0 experiment.k_latent=8 experiment.curriculum=true`

Result: identical to row 2 (not separately re-run here). `horizon_units()`
returns `maximum` whenever `warmup_epochs <= 1`, regardless of the `curriculum`
flag -- the ramp needs a nonzero warmup window to do anything. Confirmed
directly later (row 5b).

## 4. + curriculum, k=1 (control: does curriculum help without capacity?)

command: `uv run python train.py run=final policy=single_step experiment=plain_dsrl landscape=g2 experiment.entropy_coef=0.0 experiment.curriculum=true experiment.warmup_epochs=120` (k_latent left at 1, default)

Aggregate: SR=1.000 SR_M=0.250 mcf=1.00/4 H/logM=0.000 Hs/logM=0.000 radius=0.761

Identical collapse to row 1. Curriculum has nothing to preserve distinctness
*among* when there is only one latent to begin with -- confirms curriculum's
benefit (row 5) is conditional on K>1 giving it something to act on, not an
independent diversity mechanism.

## 5. + curriculum, k=8, real warmup_epochs=120, still no MI

command: `uv run python train.py run=final policy=single_step experiment=plain_dsrl landscape=g2 experiment.entropy_coef=0.0 experiment.k_latent=8 experiment.curriculum=true experiment.warmup_epochs=120`

Aggregate: SR=0.948 SR_M=0.553 mcf=2.00/4 H/logM=0.539 Hs/logM=0.527 radius=0.763

**Important correction found by tracing the code**: with `intrinsic_scale=0`
(inherited from plain_dsrl, "no MI" by definition) and `warmup_epochs=120`,
`r_tot` is *exactly* zero for the entire discovery phase (epochs 0-119) --
`use_environment_reward=False` there, and `intrinsic_scale=0` zeroes the MI
term too. So this is not "easier learning during a short-horizon discovery
phase" (my first, wrong explanation) -- there is no learning signal at all
for the steering policy during discovery here. What *does* keep training
during the reward-free window: the critic (regressing toward V~0, a
well-defined if trivial target) and the classifier (NLL training runs
unconditionally, independent of reward). Matched-seed follow-up (180 total
epochs, no curriculum, no idle period) got the *same* mcf (1.67/4 vs 1.67/4 on
seeds 3-5) but *lower* H_norm, driven by one seed fully collapsing without the
idle window and not collapsing with it -- suggestive that the idle
critic/classifier pre-training preserves something, but not confirmed (n=3).
Verified directly: `r_env_var_across_envs` (added to `bmd.py` logging) grows
monotonically with `units` during the ramp (0.00000 at units=3 up to 1.25608
at units=14) -- the GAE-variance/credit-assignment mechanism is real, it just
isn't what explains *this* result, since there's no reward for it to apply to.

Also confirmed: `curriculum=true` with `warmup_epochs` left at 0 (default)
gives the exact same trace as `curriculum=false` (units=14, fine_tune from
epoch 0) -- `warmup_epochs`, not the boolean flag, is what actually matters.

## 6. + MI (intrinsic_scale=2.0 discovery / 0.20 fine-tune), entropy_coef=0.0

command: `uv run python train.py run=final policy=single_step experiment=plain_dsrl landscape=g2 experiment.entropy_coef=0.0 experiment.k_latent=8 experiment.curriculum=true experiment.warmup_epochs=120 experiment.intrinsic_scale=2.0 experiment.finetune_intrinsic_scale=0.20`

Aggregate: SR=0.881 SR_M=0.773 mcf=2.75/4 H/logM=0.811 Hs/logM=0.784 radius=0.783

First row where the steering policy gets a *real*, nonzero gradient during
discovery: `r_tot = intrinsic_scale * (log_q + log K)` is no longer
identically zero, since `intrinsic_scale=2.0` now. Big jump in every coverage
metric over row 5, without touching curriculum/entropy at all -- MI itself is
doing real work here, not just an artifact of idle pre-training.

## 7. + entropy regularization, entropy_coef sweep (MI still on)

commands (base is row 6's command, only `entropy_coef` changes):
```
uv run python train.py run=final policy=single_step experiment=plain_dsrl landscape=g2 experiment.entropy_coef=0.00 experiment.k_latent=8 experiment.curriculum=true experiment.warmup_epochs=120 experiment.intrinsic_scale=2.0 experiment.finetune_intrinsic_scale=0.20

uv run python train.py run=final policy=single_step experiment=plain_dsrl landscape=g2 experiment.entropy_coef=0.01 experiment.k_latent=8 experiment.curriculum=true experiment.warmup_epochs=120 experiment.intrinsic_scale=2.0 experiment.finetune_intrinsic_scale=0.20

uv run python train.py run=final policy=single_step experiment=plain_dsrl landscape=g2 experiment.entropy_coef=0.05 experiment.k_latent=8 experiment.curriculum=true experiment.warmup_epochs=120 experiment.intrinsic_scale=2.0 experiment.finetune_intrinsic_scale=0.20

uv run python train.py run=final policy=single_step experiment=plain_dsrl landscape=g2 experiment.entropy_coef=0.1 experiment.k_latent=8 experiment.curriculum=true experiment.warmup_epochs=120 experiment.intrinsic_scale=2.0 experiment.finetune_intrinsic_scale=0.20   # pending, not yet logged

```

| entropy_coef | SR (mean) | SR_M | mcf | H/logM | Hs/logM |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0.00 | 0.881 | 0.773 | 2.75/4 | 0.811 | 0.784 |
| 0.01 | 0.976 | 0.797 | 3.12/4 | 0.793 | 0.793 |
| 0.05 | 0.966 | 0.788 | 3.25/4 | 0.781 | 0.781 |

Per-seed (seeds 3-10, in order):

```
ent=0.00 mcf: 4  1  2  4  3  4  2  2   std=1.09
ent=0.01 mcf: 3  4  4  4  3  3  2  2   std=0.78
ent=0.05 mcf: 2  4  2  4  4  3  4  3   std=0.83

ent=0.00 SR:  1.000  0.727  0.617  0.996  0.988  0.992  0.727  1.000
ent=0.01 SR:  0.992  1.000  1.000  1.000  0.980  0.996  0.844  0.992
ent=0.05 SR:  0.949  0.992  0.930  0.949  0.961  0.977  0.988  0.980
```


## 8. Ablation: classifier input noise (`inference_input_noise_std`)

Paired comparison, both using `experiment=bmd` (full recipe) directly, only
`inference_input_noise_std` differs:

```
uv run python train.py run=final experiment=bmd policy=single_step landscape=g2 experiment.inference_input_noise_std=0.0
uv run python train.py run=final experiment=bmd policy=single_step landscape=g2
```

| | inference_input_noise_std=0.0 | =0.30 (default) |
| --- | ---: | ---: |
| SR | 0.213 | 0.976 |
| SR_M | 0.206 | 0.797 |
| mcf | 0.25/4 | 3.12/4 |
| H/logM | 0.965 | 0.793 |
| Hs/logM | 0.206 | 0.793 |
| radius | 0.375 | 0.764 |

The `0.30`-noise result matches the equivalent `plain_dsrl`+overrides row from
section 7 almost exactly (SR=0.976, mcf=3.12/4 both ways) -- good cross-check
that the two construction paths are truly equivalent.

**The noise regularizer is load-bearing, and the `H` vs `Hs` gap is the
clearest evidence why.** Without it: `radius=0.375` collapses to roughly the
same magnitude as the historical "modes that don't travel" trace in section 16
(radius ~0.23-0.31 during that failure). And `H/logM=0.965` (near-maximum,
attempts spread almost evenly across all 4 goal-bins) sits right next to
`Hs/logM=0.206` (successes concentrated on essentially one goal) -- this is
Trap B from `lab_guide.md` reproduced directly: raw `H` looks great because
*failures* scatter roughly evenly across all 4 nearest-goal bins by chance,
not because the policy is meaningfully covering all 4 goals. `SR=0.213` (most
episodes fail) confirms the "attempts" and "successes" distributions are
telling very different stories here, which raw `H` alone would hide.

Mechanism: with `training_noise_std=0`, the classifier can reward the steering
policy for arbitrarily small state differences -- including jitter that never
leaves the immediate area around the origin. `q_phi` learns to tell 8 such
jitter patterns apart perfectly well (hence high raw `H`), but none of that
motion is goal-directed, so the intrinsic reward stops correlating with actual
task progress and training collapses toward useless, non-traveling behavior.
Smoothing the classifier's input with noise of scale 0.30 makes any
distinction smaller than that noise floor unrewardable, forcing the policy to
create separations large enough to correspond to real, distinct locations.


**Conclusion, corrected after looking at per-seed data, not just aggregates:**
entropy cleanly raises the *worst-case SR floor* across seeds (0.617 -> 0.844
-> 0.930, monotonic) -- this is the clean, defensible finding. It does **not**
cleanly improve `mcf`/`H_norm`: seed 5 scored a perfect 4/4 at entropy=0.01 but
dropped to 2/4 at entropy=0.05 -- "more entropy" is not uniformly better.
`H/logM` actually *decreases* slightly as entropy_coef rises (0.811 -> 0.793 ->
0.781), the opposite of naive intuition -- entropy_coef regularizes the
steering policy's own noise spread (`log_std`), not goal-assignment diversity
directly; that's MI's job. Matches the guide's own framing (section 16):
entropy's documented benefit is reduced cross-seed variance, not a uniform
boost to every metric's mean. `entropy_coef=0.1` run in progress, not yet
logged.

## Open questions / next steps

- Confirm the classifier-NLL-during-idle-discovery hypothesis (row 5) by
  checking `nll` in `history.json` across epochs 0-119 -- is it actually
  falling despite the frozen policy, or flat?
- Log the pending `entropy_coef=0.1` result once it finishes.
- Full bridge still needs: re-ablate `environment_reward_scale` (row 7 of the
  original 7-step plan), then repeat on landscape=g0 and policy=chunked.



## plain dsrl

### g2

```text

behavior-mode) (base) ap@ap:~/Documents/behavior_mode$ uv run python train.py \
  run=diagnostic \
  experiment=plain_dsrl \
  policy=single_step \
  landscape=g2
output_dir=/home/ap/Documents/behavior_mode/outputs/2026-09-24/21-18-45_plain_dsrl_single_step_g2
Pretraining one frozen BC policy shared by all training seeds...
BC step=    0 loss=0.978687
BC step=  400 loss=0.400577
BC step=  800 loss=0.355607
BC step= 1200 loss=0.365017
BC step= 1600 loss=0.208683
BC step= 1999 loss=0.246500
action_scale=0.051692
BC prior: SR=0.035 SR_M=0.028 mcf=0.00/4 H/logM=0.965 Hs/logM=0.459 radius=0.494

Training seed 0
result: SR=1.000 SR_M=0.250 mcf=1.00/4 H/logM=0.000 Hs/logM=0.000 radius=0.754

Training seed 1
result: SR=1.000 SR_M=0.250 mcf=1.00/4 H/logM=0.000 Hs/logM=0.000 radius=0.763

Training seed 2
result: SR=1.000 SR_M=0.250 mcf=1.00/4 H/logM=0.000 Hs/logM=0.000 radius=0.743

Aggregate: SR=1.000 SR_M=0.250 mcf=1.00/4 H/logM=0.000 Hs/logM=0.000 radius=0.753

```


### plain dsrl g0

```text

behavior-mode) (base) ap@ap:~/Documents/behavior_mode$  uv run python train.py \
  run=diagnostic \
  experiment=plain_dsrl \
  policy=single_step \
  landscape=g0
output_dir=/home/ap/Documents/behavior_mode/outputs/2026-09-24/21-19-09_plain_dsrl_single_step_g0
Pretraining one frozen BC policy shared by all training seeds...
BC step=    0 loss=0.978687
BC step=  400 loss=0.400577
BC step=  800 loss=0.355607
BC step= 1200 loss=0.365017
BC step= 1600 loss=0.208683
BC step= 1999 loss=0.246500
action_scale=0.051692
BC prior: SR=0.059 SR_M=0.055 mcf=0.00/4 H/logM=0.982 Hs/logM=0.777 radius=0.492

Training seed 0
result: SR=1.000 SR_M=0.250 mcf=1.00/4 H/logM=0.000 Hs/logM=0.000 radius=0.774

Training seed 1
result: SR=1.000 SR_M=0.250 mcf=1.00/4 H/logM=0.000 Hs/logM=0.000 radius=0.774

Training seed 2
result: SR=1.000 SR_M=0.250 mcf=1.00/4 H/logM=0.000 Hs/logM=0.000 radius=0.772

Aggregate: SR=1.000 SR_M=0.250 mcf=1.00/4 H/logM=0.000 Hs/logM=0.000 radius=0.773


```



# Funcitonal anchor

The functional anchor can loosely be viewed as behavior distillation from the unsteered BC policy

# Notes

what i observed is that if i  reduce the BC steps then BMD can do better in g2, but if 
i increase the bc steps then BMD doesnt do that well.

## bmd with 2000 steps

```text

(behavior-mode) (base) ap@ap:~/Documents/behavior_mode$ uv run python train.py \
  run=diagnostic \
  policy=single_step \          
  policy.bc_steps=2000 \
  experiment=bmd \
  landscape=g2
output_dir=/home/ap/Documents/behavior_mode/outputs/2026-09-24/20-35-47_bmd_single_step_g2
Pretraining one frozen BC policy shared by all training seeds...
BC step=    0 loss=0.978687
BC step=  400 loss=0.400577
BC step=  800 loss=0.355607
BC step= 1200 loss=0.365017
BC step= 1600 loss=0.208683
BC step= 1999 loss=0.246500
action_scale=0.051692
BC prior: SR=0.035 SR_M=0.028 mcf=0.00/4 H/logM=0.965 Hs/logM=0.459 radius=0.494

Training seed 0
result: SR=0.984 SR_M=0.738 mcf=3.00/4 H/logM=0.650 Hs/logM=0.648 radius=0.771

Aggregate: SR=0.984 SR_M=0.738 mcf=3.00/4 H/logM=0.650 Hs/logM=0.648 radius=0.771

```

## bmd with 20k steps

```text

behavior-mode) (base) ap@ap:~/Documents/behavior_mode$ uv run python train.py \
  run=diagnostic \
  policy=single_step \
  policy.bc_steps=20000 \
  experiment=bmd \
  landscape=g2
output_dir=/home/ap/Documents/behavior_mode/outputs/2026-09-24/20-27-18_bmd_single_step_g2
Pretraining one frozen BC policy shared by all training seeds...
BC step=    0 loss=0.978687
BC step= 4000 loss=0.225510
BC step= 8000 loss=0.161122
BC step=12000 loss=0.177359
BC step=16000 loss=0.207526
BC step=19999 loss=0.117472
action_scale=0.051692
BC prior: SR=0.000 SR_M=0.000 mcf=0.00/4 H/logM=0.980 Hs/logM=nan radius=0.887

Training seed 0
result: SR=0.559 SR_M=0.397 mcf=0.00/4 H/logM=0.740 Hs/logM=0.690 radius=0.854

Aggregate: SR=0.559 SR_M=0.397 mcf=0.00/4 H/logM=0.740 Hs/logM=0.690 radius=0.854

```


## function anchor g0

```text

(behavior-mode) (base) ap@ap:~/Documents/behavior_mode$ uv run python train.py \
  run=diagnostic \
  experiment=plain_dsrl \
  policy=single_step \
  policy.bc_steps=20000 \
  landscape=g0 \
  landscape.name=g5 \
  landscape.rotation=0.1375710828025018 \
  +experiment.dsrl_anchor_type=functional \
  +experiment.dsrl_anchor_coef=1.0 \
  +experiment.dsrl_anchor_seed=20260824
output_dir=/home/ap/Documents/behavior_mode/outputs/2026-09-24/21-13-39_plain_dsrl_single_step_g5
Pretraining one frozen BC policy shared by all training seeds...
BC step=    0 loss=0.978687
BC step= 4000 loss=0.225510
BC step= 8000 loss=0.161122
BC step=12000 loss=0.177359
BC step=16000 loss=0.207526
BC step=19999 loss=0.117472
action_scale=0.051692
BC prior: SR=0.871 SR_M=0.881 mcf=4.00/4 H/logM=0.973 Hs/logM=0.981 radius=0.762

Training seed 0
result: SR=0.984 SR_M=0.985 mcf=4.00/4 H/logM=0.958 Hs/logM=0.959 radius=0.754

Training seed 1
result: SR=0.992 SR_M=0.988 mcf=4.00/4 H/logM=0.942 Hs/logM=0.940 radius=0.756

Training seed 2
result: SR=0.977 SR_M=0.973 mcf=4.00/4 H/logM=0.940 Hs/logM=0.938 radius=0.754

Aggregate: SR=0.984 SR_M=0.982 mcf=4.00/4 H/logM=0.947 Hs/logM=0.945 radius=0.755

```

# Final


## No Mi no ciriucullum
```text

plain_dsrl + k_latent=8 + entropy=0.

uv run python train.py   run=final   policy=single_step   experiment=bmd   landscape=g2   experiment.intrinsic_scale=0.0   experiment.finetune_intrinsic_scale=0.0   experiment.curriculum=false   experiment.entropy_coef=0.0   hydra.run.dir='outputs/experiments/no_mi_no_curriculum_ent0_${now:%Y-%m-%d_%H-%M-%S}'

Aggregate: SR=0.953 SR_M=0.559 mcf=2.00/4 H/logM=0.530 Hs/logM=0.507 radius=0.767


```
| Seed | SR | SR_M | mcf / 4 | $H / \log M$ | $H_s / \log M$ | Radius | Attempts Per Goal | Successes Per Goal |
|---|---|---|---|---|---|---|---|---|
| `003` | 1.000 | 0.750 | 3 | 0.776 | 0.776 | 0.751 | `[110, 0, 80, 66]` | `[110, 0, 80, 66]` |
| `004` | 0.926 | 0.442 | 1 | 0.441 | 0.408 | 0.740 | `[77, 0, 179, 0]` | `[60, 0, 177, 0]` |
| `005` | 1.000 | 1.000 | 4 | 0.946 | 0.946 | 0.776 | `[92, 62, 28, 74]` | `[92, 62, 28, 74]` |
| `006` | 1.000 | 0.500 | 2 | 0.399 | 0.399 | 0.759 | `[62, 0, 194, 0]` | `[62, 0, 194, 0]` |
| `007` | 0.996 | 0.498 | 2 | 0.499 | 0.499 | 0.765 | `[135, 121, 0, 0]` | `[134, 121, 0, 0]` |
| `008` | 1.000 | 0.500 | 2 | 0.272 | 0.272 | 0.761 | `[224, 0, 0, 32]` | `[224, 0, 0, 32]` |
| `009` | 0.891 | 0.397 | 1 | 0.418 | 0.335 | 0.795 | `[0, 68, 188, 0]` | `[0, 40, 188, 0]` |
| `010` | 0.809 | 0.383 | 1 | 0.488 | 0.421 | 0.788 | `[151, 105, 0, 0]` | `[151, 56, 0, 0]` |



### Interpretation

1. **Severe Mode Collapse Without MI**:
   - Despite having $K=8$ latent slots, without MI reward (`intrinsic_scale=0.0`), curriculum, or entropy bonus, the policy collapses to only 1–2 goals on most seeds (seeds 4, 9, 10 only reliably reach 1 goal; seeds 6, 7, 8 reach 2 goals).
   - Only a single lucky seed (`005`) happened to cover all 4 goals by random initialization.
2. **SR vs. Coverage Divergence**:
   - `SR = 0.953` looks high, but `mcf = 2.00/4` and `SR_M = 0.559` reveal that it is almost entirely exploiting the easiest 1 or 2 goals found during RL, rather than discovering diverse behavioral modes.




 ## Different bc steps for BMD 


 ```text

  uv run python train.py -m   run=diagnostic   run.training_seeds='[0,1,2]'   policy=single_step   policy.bc_steps=2000,5000,10000,20000   experiment=bmd   landscape=g2   hydra.sweep.dir='multirun/bmd_bc_budget_g2'   'hydra.sweep.subdir=bc_${policy.bc_steps}'

```

| BC steps | BC prior SR | Trained SR | SR_M | mcf /4 | H/logM | Hs/logM |
|---:|---:|---:|---:|---:|---:|---:|
| 2,000 | 0.035 | 0.945 | 0.875 | 3.33 | 0.810 | 0.815 |
| 5,000 | 0.012 | 0.954 | 0.794 | 3.00 | 0.800 | 0.797 |
| 10,000 | 0.000 | 0.921 | **0.920** | **3.67** | **0.920** | **0.914** |
| 20,000 | 0.000 | 0.712 | 0.584 | 1.33 | 0.848 | 0.715 |




## BC  Epoch based diagnoistics NO RL 


### Overview of the Three Diagnostic Runs

These three sweeps run `diagnose.py` using `diagnostic=bc_prior` across 8 BC training budgets (`bc_steps`: 250, 500, 1000, 2000, 5000, 10000, 20000, 60000) evaluated on **1024 episodes** each.

Because the BC model is **always trained on unrotated demonstrations** (`rotation=0.0`), the underlying BC policy is identical across all three sweeps at each step count. The difference is solely the evaluation landscape:
1. **`g0`**: $\text{rotation} = 0.0$ ($0^\circ$) — perfect alignment with training demonstrations.
2. **`g5`**: $\text{rotation} = 0.1376\text{ rad} \approx 7.88^\circ$ — slight angular offset.
3. **`g2`**: $\text{rotation} = \pi/4 \approx 0.7854\text{ rad} = 45.0^\circ$ — maximum offset (halfway between adjacent demonstration axes).

---

### Stats Comparison Table

| BC Steps | BC Loss | `g0` (0°) SR | `g0` Radius | `g5` (7.9°) SR | `g5` Radius | `g2` (45°) SR | `g2` Radius |
|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **250** | 0.3652 | 0.000 | 0.401 | 0.000 | 0.401 | 0.003 | 0.401 |
| **500** | 0.4562 | 0.000 | 0.380 | 0.000 | 0.380 | 0.002 | 0.380 |
| **1,000** | 0.3542 | 0.000 | 0.418 | 0.000 | 0.418 | 0.013 | 0.417 |
| **2,000** | 0.2465 | 0.063 | 0.514 | 0.072 | 0.514 | **0.028** | 0.517 |
| **5,000** | 0.2078 | 0.451 | 0.688 | 0.423 | 0.692 | 0.021 | 0.729 |
| **10,000** | 0.1668 | 0.744 | 0.733 | 0.676 | 0.747 | 0.006 | 0.828 |
| **20,000** | 0.1175 | **0.905** | 0.741 | **0.868** | 0.758 | **0.000** | 0.888 |
| **60,000** | 0.1321 | **0.982** | 0.734 | **0.979** | 0.757 | **0.000** | 0.925 |



## No Mi schedule



### Overview of `no_mi_schedule`

This Hydra sweep tested the impact of the **entropy bonus** when **Mutual Information (MI) reward is completely disabled** (`intrinsic_scale=0.0`, `finetune_intrinsic_scale=0.0`), with:
- Landscape: `g2` ($45^\circ$ rotated goals)
- Policy: `single_step` ($K=8$ latent modes)
- Evaluation: `run=final` (evaluated across **8 seeds**: `3, 4, 5, 6, 7, 8, 9, 10`, 256 episodes each)
- Horizon: `curriculum=false` (full horizon $= 14$ steps throughout training)
- Sweep parameter: `experiment.entropy_coef` $\in \{0.0, 0.01\}$

---

### Comparative Statistics Table

| Sweep Job | Entropy Coef | Mean SR | Mean SR_M | Mean mcf / 4 | Mean $H / \log M$ | Mean $H_s / \log M$ | Mean Radius |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `curriculum_off_entropy_0.0` | `0.00` | 0.953 | 0.559 | 2.00 | 0.530 | 0.507 | 0.767 |
| `curriculum_off_entropy_0.01` | `0.01` | **0.981** | **0.654** | **2.63** | **0.610** | **0.582** | 0.762 |

*(For comparison, the shared unsteered BC prior has $\text{SR}=0.035$, $\text{mcf}=0.00/4$, and $\text{radius}=0.494$.)*

---

### Per-Seed Results Breakdown (Seeds 3 to 10)

#### 1. `entropy_coef=0.00` (No MI, No Curriculum, No Entropy)
- **mcf per seed**: `[3, 1, 4, 2, 2, 2, 1, 1]` $\to$ **Mean: 2.00 / 4** ($\pm 1.00$)
- **SR per seed**: `[1.000, 0.926, 1.000, 1.000, 0.996, 1.000, 0.891, 0.809]` $\to$ **Mean: 0.953**
- **Pattern**:
  - 3 out of 8 seeds only cover **1 goal** reliably (`mcf=1`).
  - 3 seeds cover **2 goals**.
  - Only 1 seed reached 3 goals, and 1 lucky seed (`005`) covered all 4.
  - The policy readily falls into mode collapse, committing all or most latent codes to 1 or 2 easy goals.

#### 2. `entropy_coef=0.01` (No MI, No Curriculum, With Entropy Bonus)
- **mcf per seed**: `[2, 4, 4, 2, 2, 3, 2, 2]` $\to$ **Mean: 2.63 / 4** ($\pm 0.86$)
- **SR per seed**: `[1.000, 0.996, 1.000, 0.867, 1.000, 0.996, 1.000, 0.992]` $\to$ **Mean: 0.981**
- **Pattern**:
  - **Zero seeds** collapse to just 1 goal (every seed covers at least 2 goals reliably).
  - 2 seeds (`004` and `005`) achieve **full 4/4 coverage** with $\text{SR\_M} \approx 1.0$.
  - 1 seed reaches **3/4 coverage**.
  - 5 seeds reach **2/4 coverage**.

---

### Key Analysis & Interpretations

1. **Entropy bonus prevents worst-case single-goal collapse**:
   - Going from `entropy_coef=0.0` to `0.01` raises the floor on mode coverage: no seed collapses to 1 goal (the minimum `mcf` becomes 2 instead of 1).
   - Mean goal coverage increases from **2.00 / 4** to **2.63 / 4**, and average success rate improves from **0.953** to **0.981**.

2. **Entropy helps, but cannot replace Mutual Information (MI)**:
   - When MI was enabled (`intrinsic_scale=2.0`), `entropy_coef=0.01` achieved **`mcf = 3.12 / 4`** and `entropy_coef=0.05` reached **`mcf = 3.25 / 4`** on this same landscape.
   - Without MI, the steering policy has no explicit incentive to map distinct $z$ values to distinct states. Any multi-modal coverage that emerges under `entropy_coef=0.01` is purely driven by random initialization + action-space exploration noise. Once the agent finds rewarding goals, it still concentrates on just 2 goals for the majority of seeds.

3. **Comparison with `outputs/experiments/no_mi_no_curriculum_ent0_*`**:
   - The first subfolder (`curriculum_off_entropy_0.0`) has the exact same configuration and identical results to the runs in `experiments` analyzed earlier, confirming full determinism and reproducibility across seeds.




## BC epoch with RL (BMD)


### Overview of `bmd_bc_budget_g2`

This Hydra sweep tested how the **training duration of the frozen Behavioral Cloning (BC) prior** affects downstream BMD reinforcement learning.

* **Environment**: `landscape=g2` ($45^\circ$ rotated goals relative to demonstration axes)
* **RL Policy**: `single_step` ($K=8$ latent modes, full BMD recipe with curriculum, MI reward, and entropy bonus)
* **Training Seeds**: `0, 1, 2` (3 seeds per budget), evaluated on 256 episodes each
* **Sweep Parameter**: `policy.bc_steps` $\in \{2000, 5000, 10000, 20000\}$

---

### Comparative Statistics Table

| BC Steps | Unsteered BC SR | Unsteered BC Radius | Trained Mean SR | Trained Mean SR_M | Trained Mean mcf / 4 | Trained $H/\log M$ | Trained $H_s/\log M$ | Trained Radius |
|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **2,000** | 0.035 | 0.494 | **0.945** | 0.875 | 3.33 | 0.810 | 0.815 | 0.775 |
| **5,000** | 0.012 | 0.729 | **0.954** | 0.794 | 3.00 | 0.800 | 0.797 | 0.789 |
| **10,000** | 0.000 | 0.828 | 0.921 | **0.920** | **3.67** | **0.920** | **0.914** | 0.785 |
| **20,000** | 0.000 | 0.887 | 0.712 | 0.584 | **1.33** | 0.848 | 0.715 | 0.786 |

---


## BMD with more more total epochs (more env transitions)

### Overview of `bmd_training_budget_g2`

This Hydra sweep tested whether **training the BMD steering policy for more RL epochs** improves success rate or goal coverage on `g2` ($45^\circ$ rotated goals).

* **Base setup**: `experiment=bmd`, `policy=single_step` ($K=8$ latents), `landscape=g2`
* **BC policy**: fixed at default `bc_steps=2000`
* **Discovery schedule**: `warmup_epochs=120`, `curriculum=true`
* **Training seeds**: `0, 1, 2` (3 seeds per setting), evaluated on 256 episodes each
* **Sweep parameter**: `experiment.total_epochs` $\in \{300, 450, 600\}$ (fine-tuning phase after epoch 120 gets 180, 330, or 480 epochs)

---

### Comparative Statistics Table

| Total Epochs | Fine-tuning Epochs | Mean SR | Mean SR_M | Mean mcf / 4 | Mean $H / \log M$ | Mean $H_s / \log M$ | Mean Radius |
|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **300** (default) | 180 | 0.945 | **0.875** | **3.33** | 0.810 | 0.815 | 0.775 |
| **450** | 330 | 0.905 | 0.835 | 3.00 | **0.823** | 0.815 | 0.791 |
| **600** | 480 | **0.990** | 0.827 | **3.33** | 0.742 | 0.744 | 0.769 |

*(Shared unsteered BC prior: $\text{SR}=0.035$, $\text{mcf}=0.00/4$, $\text{radius}=0.494$.)*

---



### Key Analysis & Interpretations

1. **Training longer optimizes success rate, not diversity/coverage**:
   - At **600 epochs**, average success rate reaches **$99.0\%$** ($\pm 0.9\%$) across all seeds. Almost every episode hits a goal.
   - However, **mode coverage does not improve**: mean `mcf` is $3.33/4$ at 300 epochs and stays $3.33/4$ at 600 epochs.
   - Mean entropy actually **drops** slightly from $0.810 \to 0.742$, indicating that extended fine-tuning sharpens exploitation of existing modes rather than discovering new ones.

2. **Persistent seed-level attractor basin**:
   - Look at `seed_000` across all three budgets:
     - 300 epochs: attempts = `[156, 72,  0, 28]`
     - 450 epochs: attempts = `[156, 72,  0, 28]`
     - 600 epochs: attempts = `[156, 72,  0, 28]`
   - The assignment counts are virtually **identical** across budgets. Mode 2 is never visited (0 attempts), while the policy simply hones its accuracy on the other 3 modes ($155 \to 156$ successes).
   - This demonstrates that **which modes a seed covers is determined during the initial discovery phase (first 120 epochs)**. Longer fine-tuning on task reward does not escape unvisited modes; it only polishes performance on already-discovered ones.

3. **Practical Takeaway**:
   - 300 epochs is already sufficient for BMD on this problem. Doubling training to 600 epochs pushes SR from $94.5\% \to 99.0\%$, but yields zero gain in diversity or goal coverage (`mcf`).



## BMD classifier noise 


### Overview of `bmd_classifier_noise_anchor_g2`

This Hydra sweep explores the **2D interaction grid** between:
1. **Classifier Input Noise (`inference_input_noise_std`)** $\in \{0.0, 0.1, 0.3, 0.5\}$ (smoothing regularizer on the skill inference model $q_\phi(z \mid s)$).
2. **Functional Anchor Coefficient (`dsrl_anchor_coef` / $\lambda$)** $\in \{0.0, 0.03, 0.1, 0.3\}$ (drift penalty keeping steered outputs anchored to the unsteered BC prior).

- **Environment**: `landscape=g2` ($45^\circ$ diagonal goals)
- **Base Policy**: `single_step` with $K=8$ latents, full BMD schedule, `bc_steps=2000`
- **Seeds**: 3 training seeds (`0, 1, 2`), 256 evaluation episodes per seed ($4 \times 4 = 16$ total configurations).

---

### Full 4×4 Matrix of Results

| Classifier Noise | Anchor $\lambda$ | Mean SR | Mean SR_M | Mean mcf / 4 | Mean $H/\log M$ | Mean $H_s/\log M$ | Radius | Seed mcf `[s0, s1, s2]` |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **0.0** | 0.0 | 0.314 | 0.312 | 0.00 | 0.952 | 0.508 | 0.412 | `[0, 0, 0]` |
| **0.0** | 0.03 | 0.663 | 0.588 | 1.67 | 0.902 | 0.741 | 0.602 | `[2, 1, 2]` |
| **0.0** | 0.1 | 0.694 | 0.721 | 1.67 | 0.952 | 0.893 | 0.699 | `[1, 3, 1]` |
| **0.0** | 0.3 | 0.290 | 0.247 | 0.33 | 0.890 | 0.504 | 0.650 | `[0, 0, 1]` |
| **0.1** | 0.0 | 0.868 | 0.882 | 3.00 | 0.957 | 0.950 | 0.700 | `[3, 4, 2]` |
| **0.1** | 0.03 | 0.870 | 0.865 | 2.67 | 0.975 | 0.959 | 0.718 | `[4, 1, 3]` |
| **0.1** | **0.1** | **0.991** | **0.992** | **4.00** | **0.971** | **0.972** | **0.756** | **`[4, 4, 4]`** |
| **0.1** | 0.3 | 0.595 | 0.593 | 0.67 | 0.929 | 0.863 | 0.718 | `[0, 1, 1]` |
| **0.3** | 0.0 | 0.945 | 0.875 | 3.33 | 0.810 | 0.815 | 0.775 | `[3, 3, 4]` |
| **0.3** | 0.03 | **1.000** | 0.917 | 3.67 | 0.849 | 0.849 | 0.759 | `[4, 4, 3]` |
| **0.3** | **0.1** | **1.000** | **1.000** | **4.00** | **0.955** | **0.955** | **0.751** | **`[4, 4, 4]`** |
| **0.3** | 0.3 | 0.965 | 0.891 | 3.67 | 0.830 | 0.836 | 0.755 | `[3, 4, 4]` |
| **0.5** | 0.0 | 0.919 | 0.552 | 2.00 | 0.523 | 0.443 | 0.758 | `[2, 2, 2]` |
| **0.5** | 0.03 | **1.000** | 0.750 | 3.00 | 0.715 | 0.715 | 0.757 | `[2, 3, 4]` |
| **0.5** | 0.1 | 0.999 | 0.666 | 2.67 | 0.671 | 0.671 | 0.753 | `[2, 3, 3]` |
| **0.5** | 0.3 | 0.960 | 0.471 | 2.00 | 0.443 | 0.432 | 0.758 | `[2, 2, 2]` |

---

### Key Takeaways & Detailed Analysis

#### 1. The Undisputed "Sweet Spot": `noise ∈ {0.1, 0.3}` with `λ = 0.1`
Across all 16 configurations, **only two settings achieved a perfect 4.00 / 4 goal coverage across every single seed**:
- **`noise = 0.1, λ = 0.1`**: $\text{SR}=0.991$, $\text{SR\_M}=0.992$, $\text{mcf}=4.00/4$, $H_s/\log M = 0.972$.
- **`noise = 0.3, λ = 0.1`**: $\text{SR}=1.000$, $\text{SR\_M}=1.000$, $\text{mcf}=4.00/4$, $H_s/\log M = 0.955$.

In both of these configurations, the policy attains virtually 100% success while partitioning the 8 latent codes evenly to reach all four $45^\circ$ diagonal goals reliably across all seeds.

#### 2. The Anchor Cannot Rescue Zero Noise (`noise = 0.0`)
- Without classifier noise (`noise = 0.0`), the unregularized classifier overfits to micro-jitter near the origin (radius is only $0.412$ at $\lambda=0.0$, $\text{SR}=31.4\%$).
- Adding the anchor $\lambda=0.1$ helps travel distance (radius rises to $0.699$) and increases success ($\text{SR}=69.4\%$), but goal coverage remains broken ($\text{mcf}=1.67/4$).
- **Conclusion**: Classifier smoothing is strictly necessary. The anchor alone cannot replace it.

#### 3. Too Much Classifier Noise Destroys Mode Diversity (`noise = 0.5`)
- When input noise is too large ($\sigma = 0.50$), the classifier cannot resolve states that are relatively close to each other.
- The intrinsic reward becomes washed out, so the policy suffers from mode collapse:
  - At $\lambda=0.0$, `SR = 0.919` but `SR_M = 0.552` and `mcf = 2.00/4` (all 3 seeds collapse to only 2 goals).
  - Even with $\lambda = 0.1$, `mcf` only reaches $2.67/4$, and $H_s/\log M$ remains low ($0.671$).

#### 4. The Anchor Tolerance Envelope Depends on Noise
- At **`noise = 0.1`**:
  - The anchor helps significantly at $\lambda=0.1$, but becomes destructive at $\lambda=0.3$ ($\text{SR}$ drops to $59.5\%$, $\text{mcf}=0.67/4$).
- At **`noise = 0.3`**:
  - The model is much more resilient across all anchor strengths ($\text{mcf} \ge 3.33/4$ for all $\lambda \in [0.0, 0.3]$).
  - Even at $\lambda=0.3$, `mcf` is $3.67/4$ and `SR = 96.5%`.

### Summary Recommendation
- **Classifier noise = 0.30** and **Anchor $\lambda = 0.10$** is the most robust and highest-performing configuration overall (100% success, 4/4 goals covered across all seeds, and robust across nearby hyperparameter choices).




# FUNCTIONAL ANCHOR

## landscape g5 sweep


### Overview of `functional_anchor_lambda_g5`

This sweep is an extension of the earlier `functional_anchor_g5` experiment, testing the **functional anchor coefficient** $\lambda \in \{0.0, 0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 3.0\}$ on the **`g5` landscape** ($\text{rotation} = 0.1376\text{ rad} \approx 7.88^\circ$).

- **Base Algorithm**: Plain DSRL ($K=1$ latent mode, pure task reward, no MI reward, no curriculum)
- **BC Prior**: 20,000 steps on unrotated demonstrations (unsteered BC prior already reaches $\text{SR}=87.1\%$, $\text{mcf}=4/4$ on `g5` due to the small angle).
- **Seeds**: 3 training seeds (`0, 1, 2`), 256 evaluation episodes each.

---

### Comparative Statistics Table

| Anchor $\lambda$ | Mean SR | Mean SR_M | Mean mcf / 4 | Mean $H / \log M$ | Mean $H_s / \log M$ | Mean Radius | Per-Seed mcf `[s0, s1, s2]` | State of the Policy |
|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **0.0** | **1.000** | 0.250 | 1.00 | 0.000 | 0.000 | 0.721 | `[1, 1, 1]` | **Complete Mode Collapse** (1 goal) |
| **0.1** | **0.999** | 0.333 | 1.00 | 0.011 | 0.011 | 0.724 | `[1, 1, 1]` | **Complete Mode Collapse** (1 goal) |
| **0.3** | **1.000** | 0.250 | 1.00 | 0.000 | 0.000 | 0.713 | `[1, 1, 1]` | **Complete Mode Collapse** (1 goal) |
| **0.5** | **1.000** | 0.250 | 1.00 | 0.000 | 0.000 | 0.716 | `[1, 1, 1]` | **Complete Mode Collapse** (1 goal) |
| **0.7** | **0.997** | **0.997** | **4.00** | 0.909 | 0.909 | 0.754 | `[4, 4, 4]` | **Full 4-Goal Coverage Restored** |
| **1.0** | 0.984 | 0.982 | **4.00** | 0.947 | 0.945 | 0.755 | `[4, 4, 4]` | **Full 4-Goal Coverage Restored** |
| **1.5** | 0.973 | 0.977 | **4.00** | 0.948 | 0.951 | 0.756 | `[4, 4, 4]` | **Full 4-Goal Coverage Restored** |
| **3.0** | 0.917 | 0.928 | **4.00** | **0.968** | **0.976** | 0.757 | `[4, 4, 4]` | **Conservative Drift Penalty** |

*(Shared 20k BC Prior: $\text{SR}=0.871$, $\text{SR\_M}=0.881$, $\text{mcf}=4.00/4$, $H/\log M=0.973$, $\text{radius}=0.762$.)*

---

### Detailed Breakdown Across the Sweep

#### 1. The Collapse Regime: $\lambda \in [0.0, 0.5]$
- With low anchor weight ($\lambda \le 0.5$), task reward gradient completely overwhelms the anchor.
- Because there is only a single latent slot ($K=1$) and no diversity incentive, the policy maps **every input** to whichever single goal is discovered earliest:
  - $\lambda = 0.0$: `seed 0` sends 256/256 episodes to Goal 3; `seed 1` sends 256/256 to Goal 0; `seed 2` sends 256/256 to Goal 2.
  - $\lambda = 0.1$: `seed 0` sends 254 to Goal 3 and 2 to Goal 2; `seed 1` sends all to Goal 0; `seed 2` sends all to Goal 2.
  - $\lambda = 0.3, 0.5$: All 3 seeds collapse 100% of their rollouts to Goal 3 (`[0, 0, 0, 256]`).
- Across all these settings:
  $$\text{SR} \approx 1.000, \quad \text{SR\_M} \approx 0.250, \quad \text{mcf} = 1.00/4, \quad H = 0.0$$

#### 2. The Phase Transition: Between $\lambda = 0.5$ and $\lambda = 0.7$
- Between $\lambda = 0.5$ and $\lambda = 0.7$, there is a sharp binary transition:
  - At $\lambda = 0.5$, $\text{mcf} = 1.00/4$.
  - At $\lambda = 0.7$, $\text{mcf} = 4.00/4$ across every single seed.
- At $\lambda = 0.7$, the anchor penalty for steering away from the unsteered BC policy becomes large enough to stop the policy from collapsing all modes into one.

#### 3. The Stable Preservation Regime: $\lambda \in [0.7, 3.0]$
- Across $\lambda \in \{0.7, 1.0, 1.5, 3.0\}$, **every single seed achieves $\text{mcf} = 4/4$**.
- **$\lambda = 0.7$**: Delivers the highest overall success rate ($\text{SR} = 99.7\%$, $\text{SR\_M} = 0.997$) while preserving all 4 modes.
- **$\lambda = 1.0$**: Provides a slightly more balanced entropy across goals ($H/\log M = 0.947$) with $\text{SR} = 98.4\%$.
- **$\lambda = 3.0$**: The anchor is very rigid; success rate drops to $91.7\%$, closely mimicking the unsteered BC baseline ($87.1\%$) with nearly identical entropy ($0.968$ vs $0.973$).

---

### Core Insight

This sweep provides clear empirical proof for the role of the functional anchor:
1. **Without the anchor ($\lambda < 0.7$)**: Plain DSRL inevitably collapses to a single goal mode on multi-modal tasks, even when initialized with a multi-modal BC policy.
2. **With sufficient anchor ($\lambda \ge 0.7$)**: The anchor acts as an effective behavioral regularizer, forcing the policy to remain close to the multi-modal demonstration distribution while still optimizing task success along all 4 modes simultaneously.


## Functional anchor with different BC g5



### Overview of `functional_anchor_bc_quality_g5`

This sweep investigates the core interaction between **BC Prior Quality** (pretraining steps: 2k, 5k, 10k, 20k) and the **Functional Anchor** ($\lambda=0.0$ vs $\lambda=1.0$) on the **`g5` landscape** ($7.88^\circ$ rotation) using Plain DSRL ($K=1$ latent, pure task reward RL):

- **Hypothesis tested**: Does the functional anchor require a well-trained BC model to preserve diversity, or can it work with a loose/underfitted prior? What happens if you anchor to a poor BC model?
- **Setup**: $4 \text{ BC budgets} \times 2 \text{ anchor settings} = 8 \text{ runs}$, 3 seeds each (`0, 1, 2`), 256 evaluation episodes.

---

### Comparative Statistics Table

| BC Steps | Unsteered Prior SR | Unsteered Prior Radius | Anchor $\lambda$ | Trained Mean SR | Trained Mean SR_M | Trained Mean mcf / 4 | Trained $H_s / \log M$ | Trained Radius | Mode Coverage Outcome |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **2,000** | 0.051 | 0.492 | **0.0** | **1.000** | 0.250 | 1.00 | 0.000 | 0.712 | **Collapsed to 1 goal** |
| **2,000** | 0.051 | 0.492 | **1.0** | 0.284 | 0.170 | **0.00** | 0.316 | 0.599 | **Severe Underfitting Trap** |
| **5,000** | 0.430 | 0.693 | **0.0** | **1.000** | 0.250 | 1.00 | 0.000 | 0.713 | **Collapsed to 1 goal** |
| **5,000** | 0.430 | 0.693 | **1.0** | 0.823 | 0.791 | 2.00 | 0.734 | 0.746 | Partial coverage (2/4) |
| **10,000** | 0.699 | 0.747 | **0.0** | **1.000** | 0.250 | 1.00 | 0.000 | 0.716 | **Collapsed to 1 goal** |
| **10,000** | 0.699 | 0.747 | **1.0** | 0.957 | 0.948 | 3.33 | 0.703 | 0.754 | Strong coverage (3.33/4) |
| **20,000** | 0.871 | 0.762 | **0.0** | **1.000** | 0.250 | 1.00 | 0.000 | 0.721 | **Collapsed to 1 goal** |
| **20,000** | 0.871 | 0.762 | **1.0** | 0.984 | **0.982** | **4.00** | **0.945** | 0.755 | **Full 4/4 Goal Coverage** |

---

### Detailed Breakdown

#### 1. Without Anchor ($\lambda = 0.0$): Mode Collapse is Invariant to BC Quality
Regardless of whether BC was trained for 2,000 steps or 20,000 steps:
- **Every single seed across all 4 BC budgets completely collapses to 1 goal** ($\text{mcf} = 1.00/4$, $\text{SR\_M} = 0.250$, $H/\log M = 0.000$).
- Standard task reward RL with $K=1$ has no incentive to preserve diversity; it relentlessly collapses into the single easiest goal path found during exploration.

#### 2. With Anchor ($\lambda = 1.0$): Performance is Strictly Monotonic in BC Quality
When the anchor is active, the policy is penalized for deviating from the unsteered BC decoder. The outcome depends directly on the quality of that anchor:

1. **At 2,000 BC steps ("Anchoring to Garbage")**:
   - The unsteered BC policy hasn't learned to travel outward yet (radius only $0.49$, $\text{SR}=5\%$).
   - A strong anchor ($\lambda = 1.0$) prevents the RL policy from deviating from this weak prior.
   - The agent gets trapped near the origin: **`SR` drops to $28.4\%$**, and `mcf = 0.00/4` across all seeds.
   - *Conclusion*: You cannot regularize toward a prior that doesn't yet have the desired behaviors.

2. **At 5,000 & 10,000 BC steps**:
   - As BC improves ($\text{SR}=43\% \to 70\%$, radius $0.69 \to 0.75$), the anchored RL policy successfully leverages the emerging multi-modal structure.
   - `mcf` rises from $0.00 \to 2.00 \to 3.33 / 4$, and `SR` climbs from $28.4\% \to 82.3\% \to 95.7\%$.

3. **At 20,000 BC steps**:
   - The BC prior is well-converged: trajectories cleanly radiate outward along all 4 demonstration directions ($\text{SR}=87\%$, radius $0.76$).
   - Anchoring to this high-quality prior allows task RL to fine-tune the final $7.88^\circ$ heading error without losing any modes:
     $$\text{SR} = 98.4\%, \quad \text{SR\_M} = 0.982, \quad \text{mcf} = 4.00 / 4, \quad H_s/\log M = 0.945$$
   - Every seed achieves full 4/4 coverage with nearly uniform episode distribution across all 4 goals (`[48, 40, 72, 96]`).

---

### Core Insight: What This Teaches Us About Behavioral Anchoring

1. **The Functional Anchor is Behavioral Distillation / Regularization**:
   - The anchor's job is **preservation of prior competence**, not discovery.
   - If the BC prior contains multi-modal competence (as it does at 20k steps on `g5`), the anchor successfully forces RL to keep all modes alive ($\text{mcf} = 4/4$).
   - If the BC prior is incompetent or underfitted (2k steps), the anchor actively restrains RL from exploring outward, crippling task learning.

2. **Contrast with `g2` ($45^\circ$ rotation)**:
   - On **`g5`** ($7.88^\circ$), high BC steps (20k) is **optimal** for the anchor because the demonstration directions already land inside the goal balls.
   - On **`g2`** ($45^\circ$), earlier runs showed that 20k BC steps hurts steering because the prior is rigidly aligned with the cardinal axes and steering cannot easily bend the trajectories $45^\circ$.
   - **Takeaway**: When the downstream task aligns with or is close to the demonstration distribution (`g5`), a highly converged BC model + functional anchor is ideal. When the task requires significant departure from the demonstration axes (`g2`), an overly rigid BC prior restricts steering leverage.


## Functional anchor g2 with different BC



### Overview of `functional_anchor_g2`

This sweep evaluates whether the **Functional Anchor** can prevent mode collapse on **`landscape=g2`** ($45.0^\circ$ rotated goals, completely diagonal to the unrotated cardinal demonstration axes) using **Plain DSRL** ($K=1$ latent, pure task reward RL):

- **Hypothesis tested**: Can the functional anchor preserve multi-goal coverage when the environment goals are **far away ($45^\circ$) from the demonstration axes**? Does BC quality (2k vs 20k steps) change this?
- **Configurations**: $2 \text{ BC budgets (2k, 20k)} \times 5 \text{ anchor weights } \lambda \in \{0.0, 0.03, 0.1, 0.3, 1.0\} = 10 \text{ runs}$.
- **Evaluation**: 3 training seeds (`0, 1, 2`), 256 evaluation episodes each.

---

### Comparative Statistics Table

| BC Steps | Anchor $\lambda$ | Unsteered Prior SR | Trained Mean SR | Trained Mean SR_M | Trained Mean mcf / 4 | Trained $H/\log M$ | Trained $H_s/\log M$ | Trained Radius | Mode Coverage Outcome |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **2,000** | 0.035 | 0.0 | **1.000** | 0.250 | 1.00 | 0.000 | 0.000 | 0.753 | **Complete Collapse (1 goal)** |
| **2,000** | 0.035 | 0.03 | **1.000** | 0.250 | 1.00 | 0.000 | 0.000 | 0.740 | **Complete Collapse (1 goal)** |
| **2,000** | 0.035 | 0.1 | **1.000** | 0.250 | 1.00 | 0.000 | 0.000 | 0.746 | **Complete Collapse (1 goal)** |
| **2,000** | 0.035 | 0.3 | **1.000** | 0.250 | 1.00 | 0.000 | 0.000 | 0.739 | **Complete Collapse (1 goal)** |
| **2,000** | 0.035 | 1.0 | 0.262 | 0.102 | **0.00** | 0.620 | 0.106 | 0.616 | **Severe Underfitting Trap** |
| **20,000** | 0.000 | 0.0 | **1.000** | 0.250 | 1.00 | 0.000 | 0.000 | 0.731 | **Complete Collapse (1 goal)** |
| **20,000** | 0.000 | 0.03 | 0.999 | 0.250 | 1.00 | 0.000 | 0.000 | 0.743 | **Complete Collapse (1 goal)** |
| **20,000** | 0.000 | 0.1 | 0.932 | 0.233 | 0.67 | 0.000 | 0.000 | 0.750 | **Complete Collapse (1 goal)** |
| **20,000** | 0.000 | 0.3 | 0.832 | 0.208 | 0.67 | 0.000 | 0.000 | 0.781 | **Complete Collapse (1 goal)** |
| **20,000** | 0.000 | 1.0 | 0.341 | 0.102 | **0.00** | 0.241 | 0.045 | 0.892 | **Severe Prior Conflict Trap** |

---

### Per-Seed Behavior: The Anatomy of Failure

Looking directly at the attempts and successes per goal across seeds highlights the failure mode:

#### 1. Low-to-moderate anchor ($\lambda \le 0.3$):
Every single seed sends **100% of its rollouts to exactly one goal**:
* `bc_2000_lambda_0.0`:
  * `seed 0`: attempts = `[0, 0, 256, 0]`, succ = `[0, 0, 256, 0]`
  * `seed 1`: attempts = `[256, 0, 0, 0]`, succ = `[256, 0, 0, 0]`
  * `seed 2`: attempts = `[0, 0, 256, 0]`, succ = `[0, 0, 256, 0]`
* `bc_20000_lambda_0.1`:
  * `seed 0`: attempts = `[0, 0, 256, 0]`, succ = `[0, 0, 256, 0]`
  * `seed 1`: attempts = `[0, 0, 256, 0]`, succ = `[0, 0, 256, 0]`
  * `seed 2`: attempts = `[0, 0, 256, 0]`, succ = `[0, 0, 204, 0]`

In Plain DSRL ($K=1$), task reward RL has no diversity mechanism. The anchor weight $\lambda \le 0.3$ is not strong enough to counteract task reward, so the agent collapses into whichever single diagonal goal it encounters first.

#### 2. High anchor ($\lambda = 1.0$):
* At $\lambda = 1.0$, the anchor heavily penalizes the policy whenever steered actions deviate from unsteered BC actions.
* But unsteered BC actions point along the **cardinal axes ($0^\circ, 90^\circ, 180^\circ, 270^\circ$)**, while the goals are at **$45^\circ$**.
* Because the task reward and the anchor pull in directly conflicting directions:
  - The agent is prevented from steering $45^\circ$ toward the rotated goals.
  - Success drops from **$100\% \to 26.2\%$** (at 2k BC) and **$34.1\%$** (at 20k BC).
  - Goal coverage is $\mathbf{0.00 / 4}$ across all seeds.

---

### Key Takeaways: Functional Anchor vs. BMD

Comparing this sweep on `g2` with the earlier sweeps yields a fundamental conclusion:

| Algorithmic Setup | Environment Landscape | BC Steps | Result | Why? |
|---|:---:|:---:|:---:|---|
| **Plain DSRL + Anchor ($\lambda=1.0$)** | **`g5`** ($7.88^\circ$ rotation) | 20,000 | **$\text{mcf} = 4.00 / 4$**, $\text{SR}=98.4\%$ | **Success**: The BC prior already points near the goals. The anchor successfully preserves multi-modal coverage. |
| **Plain DSRL + Anchor (any $\lambda$)** | **`g2`** ($45.0^\circ$ rotation) | 2k or 20k | **$\text{mcf} \le 1.00 / 4$**, $\text{SR} \le 34\%$ (at $\lambda=1$) | **Failure**: Goals are far from demonstration axes. Low $\lambda \to$ collapse to 1 goal. High $\lambda \to$ trapped on wrong axes. |
| **BMD (MI Discovery + $K=8$)** | **`g2`** ($45.0^\circ$ rotation) | 2,000 | **$\text{mcf} = 3.33 / 4$**, $\text{SR}=94.5\%$ | **Success**: Active skill discovery forces latent modes $z$ to explore separate, non-demonstrated regions of the state space. |
| **BMD + Anchor ($\lambda=0.1$)** | **`g2`** ($45.0^\circ$ rotation) | 2,000 | **$\text{mcf} = 4.00 / 4$**, $\text{SR}=100.0\%$ | **Best Overall**: MI drives discovery into the $45^\circ$ directions, while a mild anchor regularizes steering locally. |

### Conclusion
**The functional anchor is a preservation mechanism, not a discovery mechanism.**
- It works when the desired skills are already contained within the demonstration distribution (`g5`).
- It **fails** when the agent must invent or adapt to out-of-distribution skills (`g2`). For out-of-distribution mode discovery, **BMD's mutual information objective is necessary**.


## Functional anchor More Epochs /steps g5

### Overview of `functional_anchor_training_budget_g5`

This sweep tests the effect of **RL training budget** (`total_epochs` $\in \{150, 300, 450, 600\}$) with and without the **Functional Anchor** ($\lambda=0.0$ vs $\lambda=1.0$) on **`landscape=g5`** ($7.88^\circ$ rotation) using **Plain DSRL** ($K=1$ latent, 20k BC steps).

- **Core question**: Does training for more or fewer RL epochs help Plain DSRL preserve modes on its own, or does longer unconstrained RL exacerbate instability/mode collapse? How stable is the functional anchor across training durations?
- **Setup**: $4 \text{ epoch budgets} \times 2 \text{ anchor weights} = 8 \text{ jobs}$, 3 seeds (`0, 1, 2`), 256 evaluation episodes each.

---

### Comparative Statistics Table

| Epochs | Anchor $\lambda$ | Trained Mean SR | Trained Mean SR_M | Trained Mean mcf / 4 | Trained $H / \log M$ | Trained $H_s / \log M$ | Trained Radius | Mode Coverage Outcome |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **150** | **0.0** | **1.000** | 0.250 | 1.00 | 0.000 | 0.000 | 0.713 | **Collapsed to 1 goal immediately** |
| **150** | **1.0** | 0.973 | 0.971 | **4.00** | **0.952** | **0.952** | 0.760 | **Full 4/4 Goal Coverage** |
| **300** | **0.0** | **1.000** | 0.250 | 1.00 | 0.000 | 0.000 | 0.721 | **Collapsed to 1 goal** |
| **300** | **1.0** | **0.984** | **0.982** | **4.00** | 0.947 | 0.945 | 0.755 | **Full 4/4 Goal Coverage** |
| **450** | **0.0** | 0.706 | 0.247 | 1.00 | 0.089 | 0.000 | 0.823 | **Policy instability & collapse** |
| **450** | **1.0** | **0.995** | **0.994** | **4.00** | 0.937 | 0.936 | 0.757 | **Full 4/4 Goal Coverage (Peak SR)** |
| **600** | **0.0** | 0.710 | 0.243 | 1.00 | 0.098 | 0.000 | 0.821 | **Policy instability & collapse** |
| **600** | **1.0** | **0.987** | **0.989** | **4.00** | 0.923 | 0.924 | 0.756 | **Full 4/4 Goal Coverage** |

*(Shared 20k BC Prior: $\text{SR}=0.871$, $\text{SR\_M}=0.881$, $\text{mcf}=4.00/4$, $H/\log M=0.973$, $\text{radius}=0.762$.)*

---

### Key Observations & Detailed Analysis

#### 1. Unconstrained RL ($\lambda = 0.0$): Fast Collapse Followed by Policy Instability
- **At 150 & 300 epochs**:
  - The policy collapses into a single goal almost instantly ($\text{mcf} = 1.00/4$, $H=0.0$).
  - Every seed reaches 100% success on its single chosen goal (Seed 0 $\to$ Goal 3, Seed 1 $\to$ Goal 0, Seed 2 $\to$ Goal 2).
- **At 450 & 600 epochs**:
  - Training longer does **not** recover any other goals; instead, the policy starts to **destabilize**.
  - In `seed_1`, unconstrained policy updates push actions too far outward (radius increases to $0.82$), causing the agent to overshoot Goal 3 into empty space:
    - `seed_1` SR drops from $100\% \to \mathbf{11.7\%}$ at 450 epochs, and $\mathbf{12.9\%}$ at 600 epochs (`attempts=[0, 0, 31, 225]`, `successes=[0, 0, 30, 0]`).
  - Average SR falls from $1.000 \to 0.706$.

#### 2. With Functional Anchor ($\lambda = 1.0$): Remarkable Stability Across All Epoch Budgets
- **$100\%$ Consistency**: At every training length (150, 300, 450, and 600 epochs), **every single seed achieves $\text{mcf} = 4.00 / 4$**.
- **Fast Convergence**: Even at just **150 epochs**, the anchor has already fine-tuned the policy to $\text{SR} = 97.3\%$ with near-perfect goal entropy ($H_s/\log M = 0.952$).
- **Peak Performance at 450 epochs**:
  - $\text{SR} = \mathbf{99.5\%}$
  - $\text{SR\_M} = \mathbf{0.994}$
  - $\text{mcf} = \mathbf{4.00 / 4}$
  - Per-seed successes:
    - Seed 0: `[55, 36, 80, 83]` (254/256)
    - Seed 1: `[31, 27, 81, 116]` (255/256)
    - Seed 2: `[36, 56, 84, 79]` (255/256)
- **Prevents Overshooting / Policy Drift**: While unconstrained RL overshoots and degrades at 450+ epochs (radius rising to $0.82$), the anchored policy stays locked at radius $\approx 0.756 - 0.760$, preventing policy degradation.

---

### Takeaway

1. **More RL training does not fix mode collapse in Plain DSRL**: Unanchored RL collapses to a single mode within the first 150 epochs, and training for 450–600 epochs only introduces overshooting and reward instability.
2. **The functional anchor makes training robust to budget**: Any budget between 150 and 600 epochs works reliably with $\lambda=1.0$, reaching full 4-goal coverage ($\text{mcf}=4/4$) and up to $99.5\%$ success rate.



## Functional anchor entropy check


### Overview of `functional_anchor_entropy_g5`

This sweep tests the interaction between the **PPO policy entropy coefficient** (`entropy_coef` $\in \{0.0, 0.01, 0.05\}$) and the **Functional Anchor** ($\lambda = 0.0$ vs $\lambda = 1.0$) on **`landscape=g5`** ($7.88^\circ$ rotation) using **Plain DSRL** ($K=1$ latent, 20,000 BC steps, 300 epochs):

- **Core question**: Can an action-entropy bonus alone prevent mode collapse in Plain DSRL on a multi-modal task? Or does it still require the functional anchor?
- **Setup**: $3 \text{ entropy levels} \times 2 \text{ anchor weights} = 6 \text{ jobs}$, 3 seeds (`0, 1, 2`), 256 evaluation episodes each.

---

### Comparative Statistics Table

| Entropy Coef | Anchor $\lambda$ | Trained Mean SR | Trained Mean SR_M | Trained Mean mcf / 4 | Trained $H / \log M$ | Trained $H_s / \log M$ | Trained Radius | Mode Coverage Outcome |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **0.00** | **0.0** | 0.667 | 0.167 | 0.67 | 0.000 | 0.000 | 0.840 | **Collapse + 1 seed complete failure** |
| **0.00** | **1.0** | **0.997** | **0.998** | **4.00** | **0.961** | **0.961** | 0.755 | **Full 4/4 Coverage (Peak Performance)** |
| **0.01** | **0.0** | **1.000** | 0.250 | 1.00 | 0.000 | 0.000 | 0.721 | **Complete Collapse (1 goal)** |
| **0.01** | **1.0** | 0.984 | 0.982 | **4.00** | 0.947 | 0.945 | 0.755 | **Full 4/4 Goal Coverage** |
| **0.05** | **0.0** | **1.000** | 0.250 | 1.00 | 0.000 | 0.000 | 0.737 | **Complete Collapse (1 goal)** |
| **0.05** | **1.0** | 0.983 | 0.981 | **4.00** | 0.949 | 0.947 | 0.757 | **Full 4/4 Goal Coverage** |

*(Shared 20k BC Prior: $\text{SR}=0.871$, $\text{SR\_M}=0.881$, $\text{mcf}=4.00/4$, $H/\log M=0.973$, $\text{radius}=0.762$.)*

---

### Key Observations & Detailed Breakdown

#### 1. Entropy Alone ($\lambda = 0.0$) Cannot Prevent Mode Collapse
- Increasing `entropy_coef` from `0.00` to `0.01` and `0.05` **does nothing to preserve multi-modal goal coverage**:
  - `entropy=0.01`: every single seed collapses 100% of its episodes into a single goal (`mcf = 1.00/4`, $H=0.0$).
  - `entropy=0.05`: even with 5× higher entropy penalty, every seed still collapses 100% to a single goal (`mcf = 1.00/4`, $H=0.0$).
- **Why?**
  - Action entropy regularizes the spread of the steering policy's Gaussian noise $\pi_\psi(w \mid s) = \mathcal{N}(\mu, \sigma^2)$. It forces noise samples to be slightly broader, but because there is only one latent mode ($K=1$) and pure task reward, the policy's mean $\mu(s)$ simply steers all trajectories into whichever single goal has the steepest gradient.
  - Action entropy is **not** mode entropy.

#### 2. Entropy = 0.0 Without Anchor Causes Policy Collapse
- At `entropy=0.0, λ=0.0`:
  - `seed_0`: 256/256 successes on Goal 3.
  - `seed_2`: 256/256 successes on Goal 2.
  - `seed_1`: **0% success** (`attempts=[0, 0, 0, 256]`, `successes=[0, 0, 0, 0]`, radius $= 0.98$). Without any entropy regularization, the policy's variance shrank to the minimum bound, overshooting Goal 3 entirely into empty space.
  - This pulled mean SR down to $66.7\%$.

#### 3. With Functional Anchor ($\lambda = 1.0$): Flawless Performance Across All Entropy Levels
- Across all three entropy settings (`0.0, 0.01, 0.05`), **every single seed achieves $\text{mcf} = 4.00 / 4$**.
- Interestingly, **`entropy_coef = 0.0` with $\lambda = 1.0$ achieved the highest performance**:
  - $\text{SR} = \mathbf{99.7\%}$
  - $\text{SR\_M} = \mathbf{0.998}$
  - $H_s / \log M = \mathbf{0.961}$
  - Per-seed successes:
    - `seed_0`: `[51, 49, 67, 89]` (256/256)
    - `seed_1`: `[22, 56, 95, 82]` (255/256)
    - `seed_2`: `[45, 68, 83, 59]` (255/256)
- Because the functional anchor already acts as a strong behavioral regularizer keeping actions close to the multi-modal BC distribution, additional action entropy is not needed to stabilize the policy.

---

### Core Takeaway
- **Policy entropy does not solve mode collapse** in Plain DSRL. It only controls local noise spread, leaving the global multi-modal collapse completely unaddressed.
- **The functional anchor is the component that preserves multi-modal coverage** ($\text{mcf} = 4/4$). Once the anchor is active, the policy is robust and achieves $\ge 98.3\%$ success across all tested entropy values.


## Functional anchor and BMD 



### Overview of `bmd_functional_anchor_g2`

This sweep evaluates the **Functional Anchor** ($\lambda \in \{0.0, 0.01, 0.03, 0.1, 0.3\}$) applied directly on top of **full BMD** on **`landscape=g2`** ($45^\circ$ diagonal rotation):

- **Base Algorithm**: Full BMD ($K=8$ latents, mutual information reward, curriculum, entropy bonus $= 0.01$, and classifier input noise $\sigma = 0.30$)
- **BC Prior**: 2,000 steps on unrotated demonstrations (unsteered BC prior has $\text{SR}=3.5\%$, radius $=0.494$ on `g2`)
- **Seeds**: 3 training seeds (`0, 1, 2`), 256 evaluation episodes each

---

### Comparative Statistics Table

| Anchor $\lambda$ | Trained Mean SR | Trained Mean SR_M | Trained Mean mcf / 4 | Trained $H / \log M$ | Trained $H_s / \log M$ | Trained Radius | Seed mcf `[s0, s1, s2]` | Goal Coverage Quality |
|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **0.00** (Pure BMD) | 0.945 | 0.875 | 3.33 | 0.810 | 0.815 | 0.775 | `[3, 3, 4]` | Good (1 seed misses a goal) |
| **0.01** | 0.917 | 0.778 | 3.00 | 0.803 | 0.790 | 0.789 | `[2, 4, 3]` | High variance across seeds |
| **0.03** | **1.000** | 0.917 | 3.67 | 0.849 | 0.849 | 0.759 | `[4, 4, 3]` | Very strong (100% SR) |
| **0.10** | **1.000** | **1.000** | **4.00** | **0.955** | **0.955** | 0.751 | **`[4, 4, 4]`** | **Optimal: 100% success & 4/4 coverage** |
| **0.30** | 0.965 | 0.891 | 3.67 | 0.830 | 0.836 | 0.755 | `[3, 4, 4]` | Very strong |

*(Shared 2k BC Prior: $\text{SR}=0.035$, $\text{SR\_M}=0.028$, $\text{mcf}=0.00/4$, $H/\log M=0.965$, $\text{radius}=0.494$.)*

---

### Per-Seed Results Breakdown (Seeds 0, 1, 2)

#### 1. $\lambda = 0.00$ (Standard BMD Baseline)
- $\text{SR} = 0.945 \pm 0.061$, $\text{mcf} = 3.33 / 4$
- `seed_0`: $\text{SR}=0.984$, $\text{mcf}=3/4$ | attempts: `[156, 72, 0, 28]` $\to$ Goal 2 completely unvisited
- `seed_1`: $\text{SR}=0.859$, $\text{mcf}=3/4$ | attempts: `[49, 28, 82, 97]` $\to$ Goal 1 SR is lower
- `seed_2`: $\text{SR}=0.992$, $\text{mcf}=4/4$ | attempts: `[133, 28, 28, 67]`

#### 2. $\lambda = 0.01$
- $\text{SR} = 0.917 \pm 0.110$, $\text{mcf} = 3.00 / 4$
- `seed_0`: $\text{SR}=0.762$, $\text{mcf}=2/4$ | attempts: `[62, 0, 99, 95]`
- `seed_1`: $\text{SR}=1.000$, $\text{mcf}=4/4$ | attempts: `[112, 40, 34, 70]`
- `seed_2`: $\text{SR}=0.988$, $\text{mcf}=3/4$ | attempts: `[122, 100, 0, 34]`

#### 3. $\lambda = 0.03$
- $\text{SR} = \mathbf{1.000} \pm 0.000$, $\text{mcf} = 3.67 / 4$
- `seed_0`: $\text{SR}=1.000$, $\text{mcf}=4/4$ | attempts: `[91, 28, 102, 35]`
- `seed_1`: $\text{SR}=1.000$, $\text{mcf}=4/4$ | attempts: `[60, 28, 51, 117]`
- `seed_2`: $\text{SR}=1.000$, $\text{mcf}=3/4$ | attempts: `[0, 62, 60, 134]`

#### 4. $\lambda = 0.10$ *(Peak Performance / Optimum)*
- **$\text{SR} = 1.000 \pm 0.000$**
- **$\text{SR\_M} = 1.000 \pm 0.000$**
- **$\text{mcf} = 4.00 / 4 \pm 0.00$**
- **$H_s / \log M = 0.955$**
- Every seed achieves 256/256 successes with an exceptionally balanced spread across all 4 diagonal goals:
  - `seed_0`: attempts = `[34, 70, 100, 52]`, succ = `[34, 70, 100, 52]`
  - `seed_1`: attempts = `[67, 45, 110, 34]`, succ = `[67, 45, 110, 34]`
  - `seed_2`: attempts = `[51, 56, 82, 67]`, succ = `[51, 56, 82, 67]`

#### 5. $\lambda = 0.30$
- $\text{SR} = 0.965 \pm 0.041$, $\text{mcf} = 3.67 / 4$
- `seed_0`: $\text{SR}=0.992$, $\text{mcf}=3/4$ | attempts: `[69, 32, 155, 0]` $\to$ misses Goal 3
- `seed_1`: $\text{SR}=0.996$, $\text{mcf}=4/4$ | attempts: `[76, 57, 88, 35]`
- `seed_2`: $\text{SR}=0.906$, $\text{mcf}=4/4$ | attempts: `[43, 75, 120, 18]`

---

### Key Takeaways & Analysis

1. **Synergy Between BMD and a Moderate Anchor ($\lambda = 0.1$)**:
   - In Plain DSRL ($K=1$), an anchor on `g2` completely failed ($\text{mcf} \le 1.0$) because there was no active mechanism to discover the $45^\circ$ diagonal goals.
   - In BMD, **Mutual Information (MI) provides the outward discovery drive**, while the **functional anchor ($\lambda = 0.1$) provides local behavioral regularization**.
   - This combination eliminates missing-mode variance across seeds: standard BMD has $\text{mcf} = 3.33/4$ (seeds often miss 1 goal), whereas adding $\lambda = 0.1$ brings every seed to a flawless **$4.00/4$** coverage and **$100\%$ success rate**.

2. **The Effective Anchor Window on `g2` is Narrower ($\lambda \approx 0.03 - 0.10$)**:
   - On `g5` (where goals are nearly unrotated), the anchor can be much stronger ($\lambda = 0.7 - 1.5$) because the unsteered prior is already pointed at the goals.
   - On `g2` (where goals require a $45^\circ$ deflection), anchor weights $\lambda > 0.1$ begin to fight the required steering direction, causing $\text{SR}$ and coverage to decline at $\lambda \ge 0.3$.

   ## BMD anchor g2 noise

   
### Overview of `bmd_anchor_g2_noise03_lambda01`

This is a **full 8-seed benchmark (`run=final`, seeds 3–10)** testing the best-performing combination identified from earlier diagnostic sweeps on **`landscape=g2`** ($45.0^\circ$ diagonal rotation):

- **Algorithm**: Full BMD ($K=8$ latents, 120-epoch warmup curriculum, MI intrinsic reward, `entropy_coef=0.01`)
- **BC Prior**: `bc_steps=2000` (soft, steerable prior)
- **Classifier Input Noise**: `inference_input_noise_std = 0.30`
- **Functional Anchor**: `dsrl_anchor_type = functional`, `dsrl_anchor_coef = 0.10`
- **Evaluation**: 8 independent training seeds (`3, 4, 5, 6, 7, 8, 9, 10`), 256 evaluation episodes each (2,048 total episodes).

---

### Benchmark Aggregate Statistics

| Metric | BC Prior Mean | Trained Mean | Trained Std |
|---|:---:|:---:|:---:|
| **Success Rate (`SR`)** | 0.0352 (3.5%) | **0.9995 (99.95%)** | 0.0013 |
| **Macro Success Rate (`SR_M`)** | 0.0281 | **0.9676** | 0.0823 |
| **Goal Coverage (`mcf` / 4)** | 0.00 / 4 | **3.88 / 4 (7 out of 8 seeds hit 4/4)** | 0.33 |
| **Assignment Entropy ($H / \log M$)** | 0.9652 | **0.9107** | 0.0841 |
| **Success Entropy ($H_s / \log M$)** | 0.4591 | **0.9104** | 0.0840 |
| **Mean Final Radius** | 0.4944 | **0.7538** | 0.0038 |

---

### Per-Seed Results Breakdown (Seeds 3 to 10)

| Seed | SR | SR_M | mcf / 4 | $H / \log M$ | $H_s / \log M$ | Radius | Attempts Per Goal `[G0, G1, G2, G3]` | Successes Per Goal `[G0, G1, G2, G3]` |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `003` | **1.000** | 1.000 | **4 / 4** | 0.997 | 0.997 | 0.748 | `[59, 63, 74, 60]` | `[59, 63, 74, 60]` |
| `004` | **1.000** | 1.000 | **4 / 4** | 0.774 | 0.774 | 0.761 | `[151, 17, 60, 28]` | `[151, 17, 60, 28]` |
| `005` | **1.000** | 1.000 | **4 / 4** | 0.987 | 0.987 | 0.750 | `[45, 74, 63, 74]` | `[45, 74, 63, 74]` |
| `006` | **1.000** | 1.000 | **4 / 4** | 0.923 | 0.923 | 0.755 | `[68, 111, 45, 32]` | `[68, 111, 45, 32]` |
| `007` | **1.000** | 1.000 | **4 / 4** | 0.937 | 0.937 | 0.756 | `[59, 28, 68, 101]` | `[59, 28, 68, 101]` |
| `008` | **1.000** | 0.750 | **3 / 4** | 0.768 | 0.768 | 0.756 | `[66, 0, 73, 117]` | `[66, 0, 73, 117]` |
| `009` | **0.996** | 0.991 | **4 / 4** | 0.931 | 0.929 | 0.751 | `[28, 49, 97, 82]` | `[27, 49, 97, 82]` |
| `010` | **1.000** | 1.000 | **4 / 4** | 0.967 | 0.967 | 0.755 | `[56, 40, 67, 93]` | `[56, 40, 67, 93]` |

---

### Comparison with Prior Configurations on `g2` (8-seed `run=final`)

| Method / Setup | SR | SR_M | mcf / 4 | $H_s / \log M$ | Key Limitation |
|---|:---:|:---:|:---:|:---:|---|
| **Plain DSRL Baseline** ($K=1$) | 1.000 | 0.250 | 1.00 | 0.000 | Total mode collapse (1 goal only) |
| **No MI, No Curriculum, Ent=0** | 0.953 | 0.559 | 2.00 | 0.507 | Severe collapse to 1–2 goals |
| **Full BMD Baseline** ($\lambda=0.0$, noise=0.3) | 0.976 | 0.797 | 3.12 | 0.793 | High variance (seeds miss 1–2 goals) |
| **BMD + Anchor** ($\lambda=0.1$, noise=0.3) [**This Run**] | **0.999** | **0.968** | **3.88** | **0.910** | **Highest SR, lowest variance, 7/8 seeds reach 4/4** |

---

### Key Takeaways

1. **Near-Perfect Task Success**:
   Across 2,048 total test episodes across 8 seeds, **only 1 single episode failed** (`seed_009` had 255/256), yielding an overall success rate of **99.95%**.
2. **Robust Multi-Modal Coverage**:
   - **7 out of 8 seeds covered all 4 diagonal goals** at $\ge 80\%$ success and $\ge \text{min\_attempts}$.
   - Only `seed_008` missed Goal 1 (`[66, 0, 73, 117]`).
   - Seeds like `003` achieved an almost perfectly uniform division across all four $45^\circ$ quadrants (`[59, 63, 74, 60]`, $H_s/\log M = 0.997$).
3. **Synergy Confirmed at Scale**:
   The diagnostic finding holds up across all 8 standard evaluation seeds: **Mutual Information provides the active discovery pressure** to steer toward unseen diagonal modes, while the **functional anchor ($\lambda = 0.10$) stabilizes and regularizes local steering**, eliminating the mode dropouts common in pure BMD.



## final_bmd_bc_budget_g2

### 1. Executive Summary & Core Finding

* **Over-training the BC prior degrades downstream BMD performance.**
* In the initial diagnostic sweep (`bmd_bc_budget_g2`, 3 seeds: 0–2), `bc_10000` appeared slightly better ($3.67/4$ vs $3.33/4$ `mcf`). However, the full 8-seed final run disproves this artifact:
  * At **10,000 BC steps**, floored mode coverage drops from **$3.12/4$** (the 2,000-step baseline) to **$2.38/4$**, with increased variance ($\sigma = 1.32$) and pooled success falling to $0.889$.
  * At **20,000 BC steps**, performance collapses: floored mode coverage drops to **$1.12/4$**, no seed achieves $4/4$ coverage, two seeds drop to $0/4$, and pooled success drops to $0.724$.

---

### 2. Aggregate Comparison: BC Steps Sweep (Seeds 3–10)

| Setting | BC Prior SR | BC Prior Radius | RL Mean SR | RL $\text{SR}_M$ | RL `mcf` (`mc@0.80_floored`) | RL $H/\log 4$ | RL $H_{\text{succ}}/\log 4$ | RL Final Radius |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`bc_2000`** (baseline) | $0.035$ | $0.494$ | **$0.976 \pm 0.054$** | **$0.797$** | **$3.12 \pm 0.78 / 4$** | $0.793$ | $0.793$ | $0.764$ |
| **`bc_10000`** | $0.000$ | $0.827$ | $0.889 \pm 0.106$ | $0.745 \pm 0.190$ | $2.38 \pm 1.32 / 4$ | $0.814 \pm 0.164$ | $0.800 \pm 0.155$ | $0.791 \pm 0.045$ |
| **`bc_20000`** | $0.000$ | $0.887$ | $0.724 \pm 0.125$ | $0.605 \pm 0.074$ | $1.12 \pm 0.93 / 4$ | $0.841 \pm 0.084$ | $0.782 \pm 0.111$ | $0.787 \pm 0.031$ |

*(Baseline numbers taken from section 7 and 8 of `four_goal_exp.md:110-145`).*

---

### 3. Per-Seed Breakdown (Seeds 3–10)

#### `bc_10000` (`summary.json`)
* **Per-seed `mcf`:** `[2, 2, 4, 1, 3, 3, 4, 0]` ($\text{mean} = 2.38, \text{std} = 1.32$)
* **Per-seed SR:** `[1.000, 0.902, 1.000, 0.773, 0.887, 0.887, 0.980, 0.680]`
* **Key Observations:**
  * Highly bimodal behavior across seeds: Seeds 5 and 9 achieve perfect $4/4$ coverage with $\text{SR} \ge 0.98$.
  * In contrast, Seed 10 completely drops out with $0/4$ modes and an SR floor of $0.680$. Seed 6 manages only $1/4$ mode.
  * Several seeds abandon entire quadrants during evaluation (e.g., Seed 3: `attempts = [134, 0, 122, 0]`; Seed 4: `attempts = [114, 0, 70, 72]`).

#### `bc_20000` ([multirun/final_bmd_bc_budget_g2/bc_20000/summary.json](multirun/final_bmd_bc_budget_g2/bc_20000/summary.json))
* **Per-seed `mcf`:** `[1, 1, 3, 0, 1, 1, 2, 0]` ($\text{mean} = 1.12, \text{std} = 0.93$)
* **Per-seed SR:** `[0.777, 0.672, 0.930, 0.559, 0.789, 0.652, 0.848, 0.566]`
* **Key Observations:**
  * Comprehensive collapse across all seeds: **zero** seeds reach $4/4$ modes.
  * Only a single seed (Seed 5) clears 3 modes. Seeds 6 and 10 score $0/4$.
  * Three of eight seeds drop below $0.68$ success rate (Seeds 6, 8, 10).
  * High entropy ($H/\log 4 = 0.841$) paired with low coverage ($1.12/4$) and lower success entropy ($H_{\text{succ}}/\log 4 = 0.782$) indicates Trap B: attempts wander broadly, but execution fails to reliably land inside goal radii.

---

### 4. Mechanism: Why Over-Training the BC Diffusion Prior Hurts BMD

1. **Diffusion Prior Loss of Steerability**:
   * As seen in the standalone learning curve diagnostics in [multirun/bc_learning_curve_g2](multirun/bc_learning_curve_g2), at 2,000 steps the diffusion prior has moderate MSE loss ($\sim 0.24$) and average final rollout radius of $0.50$. The decoder produces a smooth, pliable velocity field that can be readily redirected in action space.
   * By 10,000 steps (loss $\sim 0.16$, radius $0.83$) and 20,000 steps (loss $\sim 0.12$, radius $0.89$), the model overfits to the deterministic demo trajectories. The score function $\nabla_x \log p_t(x|s)$ develops deep attractor basins.
2. **Exhaustion of Steering Authority ($w_{\text{bound}} = 3.0$)**:
   * In BMD, the steering policy outputs bounded latent noise adjustments $w \in [-3.0, 3.0]$ that perturb the DDIM reverse-diffusion input.
   * When the diffusion model's attractor basins become excessively sharp, the bounded perturbation $w$ lacks sufficient control authority to pull trajectories across modes or fine-tune terminal coordinates accurately.
3. **Plateau During Fine-Tuning**:
   * Tracing the training histories in [multirun/final_bmd_bc_budget_g2/bc_10000/seed_003/history.json](multirun/final_bmd_bc_budget_g2/bc_10000/seed_003/history.json) vs [multirun/final_bmd_bc_budget_g2/bc_20000/seed_003/history.json](multirun/final_bmd_bc_budget_g2/bc_20000/seed_003/history.json):
     * During the discovery phase (epochs 0–119), both policies discover intrinsic rewards ($r_{\text{int}} \approx 1.8$).
     * However, during the fine-tuning phase (epochs 120–299), environment reward for `bc_20000` plateaus at $r_{\text{env}} = 0.191$ with rollout success at $0.723$, compared to $r_{\text{env}} = 0.266$ and success at $0.918$ for `bc_10000`. The frozen diffusion network resists task-directed policy updates.

### Conclusion & Recommendation
For single-step BMD on `landscape=g2`, the optimal BC budget remains around the default of **2,000 steps**. Higher BC budgets constrain steering flexibility and significantly reduce multimodal coverage.



## Final functional anchor

### Analysis of `final_functional_anchor_g5`

The multirun sweep in `final_functional_anchor_g5` evaluates the **Functional Anchor** regularization loss under the full 8-seed final evaluation protocol (seeds 3–10, 256 evaluation episodes each) on **`landscape=g5`** ($\text{rotation} = 0.1376\text{ rad} \approx 7.88^\circ$). 

The experiment uses **Plain DSRL** ($K=1$ latent, pure task reward, no MI, no curriculum) with a well-trained diffusion prior (**20,000 BC steps**) across three anchor regularization strengths: $\lambda \in \{0.0, 0.7, 1.0\}$.

---

### 1. Executive Summary & Core Finding

* **Mode collapse is total without the anchor ($\lambda = 0.0$):** Plain DSRL ($K=1$) reaches $100\%$ pooled success, but collapses completely onto a single physical goal across all 8 seeds ($H/\log 4 \approx 0.000$, $\text{mcf} = 1.00/4$).
* **$\lambda = 0.7$ partially prevents collapse:** Recovers mode coverage to $\text{mcf} = 3.50/4$, with 6 of 8 seeds maintaining all 4 modes, but 2 seeds still drop to 2 modes ($\text{std} = 0.87$).
* **$\lambda = 1.0$ achieves complete, zero-variance mode preservation:** Every single seed achieves perfect $4/4$ mode coverage ($\text{mcf} = 4.00 \pm 0.00/4$) with pooled success $\text{SR} = 99.3\%$ and normalized entropy $H/\log 4 = 0.978$.

---

### 2. Aggregate Results (Seeds 3–10)

| Setting ($\lambda$) | BC Prior SR | BC Prior `mcf` | RL Mean SR | RL $\text{SR}_M$ | RL `mcf` (`mc@0.80_floored`) | RL $H/\log 4$ | RL $H_{\text{succ}}/\log 4$ | RL Final Radius |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$\lambda = 0.0$** | $0.871$ | $4.00/4$ | **$1.000 \pm 0.001$** | $0.250 \pm 0.000$ | **$1.00 \pm 0.00 / 4$** | $0.002 \pm 0.006$ | $0.000 \pm 0.000$ | $0.712 \pm 0.001$ |
| **$\lambda = 0.7$** | $0.871$ | $4.00/4$ | **$0.992 \pm 0.007$** | $0.866 \pm 0.212$ | **$3.50 \pm 0.87 / 4$** | $0.831 \pm 0.194$ | $0.828 \pm 0.197$ | $0.756 \pm 0.003$ |
| **$\lambda = 1.0$** | $0.871$ | $4.00/4$ | **$0.993 \pm 0.003$** | $0.993 \pm 0.003$ | **$4.00 \pm 0.00 / 4$** | $0.978 \pm 0.013$ | $0.978 \pm 0.013$ | $0.756 \pm 0.001$ |

---

### 3. Per-Seed Breakdown (Seeds 3–10)

#### $\lambda = 0.0$ (No Anchor Baseline)
* **Per-seed `mcf`:** `[1, 1, 1, 1, 1, 1, 1, 1]` ($\text{mean} = 1.00, \text{std} = 0.00$)
* **Per-seed SR:** `[1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 0.996, 1.000]`
* **Goal Distribution:**
  * Seed 3: Goal 1 ($256/256$)
  * Seed 4: Goal 3 ($256/256$)
  * Seed 5: Goal 3 ($256/256$)
  * Seed 6: Goal 1 ($256/256$)
  * Seed 7: Goal 2 ($256/256$)
  * Seed 8: Goal 1 ($256/256$)
  * Seed 9: Goal 3 ($255/256$)
  * Seed 10: Goal 3 ($256/256$)
* **Observation:** Classic unconstrained RL collapse. The steering policy deterministically shifts latent noise to steer every rollout to whichever single goal was first encountered or most easily reached.

#### $\lambda = 0.7$ (Intermediate Anchor)
* **Per-seed `mcf`:** `[4, 2, 4, 4, 4, 4, 4, 2]` ($\text{mean} = 3.50, \text{std} = 0.87$)
* **Per-seed SR:** `[0.977, 0.992, 0.996, 0.992, 0.992, 0.996, 0.988, 1.000]`
* **Observation:** $\lambda=0.7$ is strong enough to keep all 4 modes alive in 6 of the 8 seeds. However, it is not consistently stiff enough across all initializations: Seeds 4 and 10 collapsed two of their modes (Seed 4: attempts `[1, 0, 159, 96]`; Seed 10: attempts `[0, 0, 127, 129]`).

#### $\lambda = 1.0$ (Full Anchor)
* **Per-seed `mcf`:** `[4, 4, 4, 4, 4, 4, 4, 4]` ($\text{mean} = 4.00, \text{std} = 0.00$)
* **Per-seed SR:** `[0.992, 0.992, 0.988, 0.996, 0.988, 0.996, 0.992, 0.996]`
* **Observation:** Perfect stability across every seed. Attempts are uniformly balanced across all four modes (each mode receiving roughly $40\text{--}90$ attempts out of 256), with per-mode success rates between $96.7\%$ and $100.0\%$.

---

### 4. Key Mechanisms & Insights

1. **Why Functional Anchoring Works on `g5`:**
   * On `g5`, the goals are rotated by only $7.88^\circ$ from the axes. The unsteered 20,000-step BC prior already reaches $\text{SR}=87.1\%$ with full 4-mode coverage ($\text{mcf}=4/4$).
   * The functional anchor loss penalizes deviations in the decoded action space between steered noise $w_\psi(s, \epsilon)$ and identity noise $\epsilon$:
     $$\mathcal{L}_{\text{anchor}} = \mathbb{E}_{s, \epsilon} \left[ \| \mathcal{D}(s, w_\psi(s, \epsilon)) - \mathcal{D}(s, \epsilon) \|_2^2 \right]$$
   * At $\lambda = 1.0$, this regularizer acts as a behavioral tether that prevents RL from collapsing the action manifold into a single attractor, while still allowing the steering policy enough local flexibility to elevate success from $87.1\%$ up to $99.3\%$.

2. **Contrast with `g2` ($45^\circ$ rotation):**
   * On `g2`, the goals require a substantial $45^\circ$ deviation from the training demonstrations. A strong anchor to the unsteered prior on `g2` anchors the policy to non-rewarding directions, forcing a strict trade-off between mode preservation and task reward.
   * On `g5`, because the prior already encompasses all 4 modes, anchoring to the prior preserves the multimodal capability without sacrificing task success.

3. **Training Dynamics:**
   * Tracing policy exploration std ($\sigma = e^{\log \sigma}$):
     * $\lambda = 0.0$: std drops from $1.00$ to $0.57$ early in training, reflecting policy collapse and deterministic exploitation.
     * $\lambda = 1.0$: std expands from $1.00$ to $1.36$, maintaining exploratory breadth throughout all 300 epochs because the anchor prevents the policy distribution from narrowing down to a single mode.
