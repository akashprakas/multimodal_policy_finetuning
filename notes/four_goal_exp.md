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

