# When and Why a Functional Anchor Preserves Modes

Date: 2026-09-28

Status: evidence-backed working theory. Section 3 and the one-step statement in
Section 4 are standard consequences of coupling and decision-boundary geometry.
Sections 5–7 are mechanistic hypotheses motivated by the experiments, not
trajectory-level theorems. The g2 failure prediction has now been tested with DSRL
and ReinFlow; the full rotation-scaling law and the tightness of the margin bound
remain untested.

---

## 1. Novelty and prior work

### 1.1 The loss is not new

| Work | Relation |
|---|---|
| Flow Q-Learning, Park, Li, Levine (ICML 2025, arXiv:2502.02538) | Eq. 7 is `E_{s,z}‖μ_ω(s,z) − μ_θ(s,z)‖²`, the same paired-noise loss. Eq. 8 states the W₂ upper bound. Offline setting, dataset states, one-step distillation, used as a behavior regularizer. |
| ReinFlow, Zhang et al. (NeurIPS 2025, arXiv:2505.22094) | The paper's Eq. 12 is `E ½‖a − a_old‖²` from the same starting noise, i.e. the functional anchor. **The released code uses `x0 = 0` only**, which is what our "upstream" arm reproduces. It is not framed as mode preservation, and the paper reports it as weaker than the entropy regularizer. |
| LP-DS, Simsir & Oguz (ICML 2026, arXiv:2606.01151) | The closest work to our DSRL setting. It targets DSRL mode collapse with a noise-space trust region `E‖Δw‖² ≤ δ`. |
| ORW-CFM-W2, Fan et al. (2025, arXiv:2502.06061) | Velocity-matching W₂ regularizer for online flow fine-tuning of image models. Lemma 1: collapse to a Dirac without regularization. Theorem 3: `W₂² ≤ e^{2L} ∫ E‖Δv‖²`. |
| DSRL, Wagenmaker et al. (2025, arXiv:2506.15799) | Only a box constraint on the noise action. No anchor. |
| BMD, Longhini et al. (ICML 2026, arXiv:2605.11387) | MI-based mode discovery (DIAYN-style). No paired-noise regularizer. |

Function-space regularization is also an established idea in continual learning:
Learning without Forgetting (arXiv:1606.09282), Titsias et al. (arXiv:1901.11356),
L2-SP (arXiv:1802.01483), and Laria et al. (arXiv:2410.14159) for diffusion.

### 1.2 What is plausibly new

1. **An action-space anchor for noise steering (DSRL):**
   `‖D(s, μ + σε) − D(s, ε)‖²`. LP-DS regularizes the noise itself. This loss regularizes
   the decoded behavior.
2. **A mechanistic account of why paired noise preserves modes** (Sections 4–6), with a
   prediction for where it breaks, and the direct contrast of paired noise against
   `x0 = 0` on the same setup.
3. **A head-to-head comparison of mode preservation (anchor) and mode discovery (MI, BMD)**,
   including seed reliability and the dependence on BC quality.
**Positioning statement.** The loss itself and its W₂ interpretation are not new, and
ReinFlow already applies paired-output regularization online against a frozen old policy.
Our plausible contribution is its extension to latent-noise steering, a controlled
comparison with MI-based mode discovery, and a mechanistic mode-preservation analysis.

**Blog note.** Say explicitly that our "upstream" arm matches the ReinFlow *code*
(`x0 = 0`), not its Eq. 12.

---

## 2. Setup and notation

Fix a state `s`. We drop `s` from the notation where it is unambiguous.

- `ε ~ N(0, I)`: the input noise.
- `f₀(ε)`: the frozen BC map from noise to action (deterministic DDIM or Euler integration).
- `f_θ(ε)`: the fine-tuned deterministic noise-to-action map.
  - **ReinFlow:** `f_θ` is obtained by integrating the updated velocity field with
    deterministic Euler steps.
  - **DSRL:** `f_θ(ε) = D(μ(s) + σ ε)`. The decoder `D` is frozen and the steering
    policy is trained.
- The BC action distribution `p₀ = f₀ # N(0, I)` has `K` modes with centers `c_k`.
- `B_k = {ε : f₀(ε) is assigned to mode k}` is the **noise basin** of mode `k`, and
  `π_k = P(ε ∈ B_k)` is its probability mass. Here `K = 4` and `π_k ≈ 1/4`.
- `m(a)` assigns an action to the nearest mode. `Δ(ε)` is the distance from `f₀(ε)` to
  the nearest decision boundary of `m`, and `Δ = ess inf_ε Δ(ε)` is the **margin**.
- The anchor is

  ```text
  L(θ) = E_s E_ε ‖ f_θ(s, ε) − f₀(s, ε) ‖²
  ```

  evaluated on rollout states, in normalized action space, with `f₀` under stop-gradient.

**Geometry of the four-goal task.** The goals lie at radius `R` and are 90° apart, so the
decision boundaries are the 45° bisectors. An endpoint at a goal center is
`Δ = R sin 45° ≈ 0.707 R` from the nearest boundary.

**Scope caveat.** The anchor acts on per-step actions evaluated at states from the current
rollout, while the reported behavioral modes are defined by trajectory endpoints. The
endpoint-level geometry below is therefore a useful proxy, not a proof. A trajectory
result would require coupling complete noise sequences and controlling how action drift
changes future states under the environment dynamics.

---

## 3. Standard fact: the paired loss upper-bounds W₂

Using the same noise in both models is one particular coupling of `p_θ` and `p₀`. Since W₂
is the infimum over all couplings,

```text
L(s) = E_ε ‖f_θ(ε) − f₀(ε)‖²  ≥  W₂²(p_θ(·|s), p₀(·|s)).
```

References: FQL Eq. 8, ReinFlow Eq. 12, and Peyré & Cuturi (arXiv:1803.00567).

So the anchor is a cheap, sample-based upper bound on the distance between the
deterministic-map action distributions at a fixed state, with no optimal-transport solve
needed. For ReinFlow, this concerns the deterministic initial-noise pushforwards used by
the anchor. The deployed actor also injects learned transition noise, so this is not
directly a W₂ bound on its complete stochastic action distribution.

The paired loss constrains a particular coupling and can therefore be larger than the
optimal W₂ cost. Unlike distribution-only W₂, it penalizes relabelling which noise sample
produces which output. If the paired displacement is small relative to the relevant
decision-boundary margins, this also keeps samples inside their original mode basins.

---

## 4. Proposition 1: an exact one-step mode-switch bound

Let `S = {ε : m(f_θ(ε)) ≠ m(f₀(ε))}` be the set of noise draws whose instantaneous
action-sector assignment changes. For every sample with `Δ(ε) > 0`,

```text
1{ε ∈ S} ≤ ‖f_θ(ε) − f₀(ε)‖² / Δ(ε)².
```

**Proof.** A switched sample must cross a decision boundary, so its displacement is at
least `Δ(ε)`. The right-hand side is therefore at least one for a switched sample and
non-negative otherwise. Taking expectations gives

```text
P(S) ≤ E[‖f_θ − f₀‖² / Δ(ε)²].  ∎
```

If a strictly positive global margin `Δ = ess inf Δ(ε) > 0` exists, the looser bound

```text
P(S) ≤ L / Δ²
```

follows. For a continuous distribution whose support approaches a decision boundary,
however, this essential infimum is typically zero and the global bound is vacuous. The
pointwise-margin form is the practically relevant statement.

**Conditional corollaries when `Δ > 0`.**

- **Mass kept per mode:** `p_θ(mode k) ≥ π_k − L/Δ²`.
- **Collapse is expensive under the one-step assignment:** collapsing onto one mode moves
  at least `1 − π_max ≈ 3/4` of the mass, requiring
  `L ≥ (1 − π_max) Δ² ≈ 0.75 Δ²`.
- **Why `x0 = 0` has no distributional guarantee:** it constrains a single point, which
  has probability zero under `N(0, I)`, and therefore gives no bound on `P(S)`. Empirically,
  the ReinFlow upstream arm retains `2.33/4` modes on average with `Hs/logM = 0.562`,
  versus `4/4` for paired-noise anchoring.

This proposition is exact for a fixed-state action assignment. It does not by itself
prove preservation of endpoint trajectory modes.

---

## 5. Geometric hypothesis: local adaptation can be cheaper than mode switching

Let `δ` be how far each action must move to solve the shifted task. For a rigid shift, the
cheapest adaptation moves every sample by about `δ`, so

```text
L_adapt ≈ δ².
```

Switching one sample's mode costs at least `Δ²`. The key quantity is

```text
ρ = δ / Δ    (adaptation required ÷ margin between modes).
```

When `ρ ≪ 1`, this proxy says that changing assignment is much more expensive than adapting
within a basin. The anchor can then act as a **gate**: local movement is relatively cheap,
while crossing a boundary is relatively expensive.

For a simultaneous rotation `φ` and radius ratio `r`, the displacement between old and
new goal centers is

```text
δ/R = sqrt(1 + r² − 2r cos φ).
```

The rotation-only and radius-only expressions are special cases. Using the endpoint
margin `Δ/R = sin 45°` gives the following **heuristic diagnostic**:

| Shift | δ / R | proxy ρ² | Prediction | Observed (functional anchor) |
|---|---:|---:|---|---|
| g5, radius 1.00 | 0.137 | 0.038 | preserve | 4/4 |
| g3 (11.25°), radius 1.00 | 0.196 | 0.077 | preserve | 4/4 modes, 3/3 seeds |
| g5, radius 1.16 | 0.218 | 0.095 | preserve | 4/4 modes, 8/8 seeds |
| g5, radius 1.18 | 0.234 | 0.109 | preserve | 3.875/4 mean (attempt-floor artifact) |
| g5, radius 1.20 | 0.250 | 0.125 | preserve | 4/4 modes, 8/8 seeds |
| g5, radius 1.22 (ReinFlow) | 0.267 | 0.143 | preserve | 4/4 modes, 3/3 seeds |
| g2 (45°) | 0.765 | 1.17 | **fail** | fails |

**Why g2 is a limiting case.** A 45° rotation places every new goal on an old endpoint
decision boundary. The preservation objective and the required adaptation are therefore
in strong conflict. The completed DSRL and ReinFlow sweeps support this qualitative
prediction: weak anchors adapt but collapse, while strong anchors preserve diverse old
directions but do not solve the new task. The additional 11.25° probe also preserves all
four modes (`SR=0.983`, `SR_M=0.980`, `Hs/logM=0.966`). Three tested angles still do not
establish a sharp dependence on `ρ`; that requires the rotation sweep in Section 9.

The quantities in this table are measured in endpoint-position units, whereas the actual
loss is applied to normalized per-step actions. They explain the observed pattern but do
not constitute a trajectory theorem.

---

## 6. Toy model for a λ working window

This is a per-sample trade-off argument between reward and penalty, and it is heuristic.

**Lower threshold (preventing collapse).** Moving mass `m` from a lagging mode `j` to a
leading mode `i` gains about `m G_ij` in reward and costs at least `λ m Δ²` in
penalty. In this toy linearization, collapse is favored when

```text
λ < G_ij / Δ².
```

During training one mode can get ahead first, producing rich-get-richer symmetry breaking.
In this simplified model the transition is sharp. The DSRL sweep is qualitatively
consistent with it (`0.5` collapses, `0.7` is partial, `1.0` preserves), but PPO clipping,
advantage normalization, entropy, minibatching, and gradient clipping make an exact
all-or-nothing threshold unlikely in the real optimizer.

**Upper threshold (over-constraining).** Adapting gains about `G` in reward and costs
`λ δ²`. It stops being worthwhile when

```text
λ > G / δ².
```

This matches λ = 3 under-adapting (SR 0.76, high entropy).

**Candidate width of the window in the toy model.**

```text
λ ∈ ( G_ij/Δ² ,  G/δ² ),      upper/lower = (G / G_ij) · (1/ρ²).
```

This expression suggests a wider feasible interval for small shifts and a narrower one as
`ρ` grows. It is an interpretation of the g2 failure, not a fitted scaling law.

**Caveats.**
- PPO advantage normalization and the joint gradient clipping of the anchor and PPO terms
  (`max_grad_norm = 1.0` in `ppo.py`) change the effective scale of `λ`.
- The absolute thresholds and a guaranteed working window are not predicted. The model
  only suggests that preservation and adaptation become harder to balance as `ρ` grows.
- Raw λ values cannot be compared across backends (DSRL 1.0, ReinFlow 0.03). The ratio of
  anchor-gradient norm to PPO-gradient norm is the natural common scale.

---

## 7. Consequences

### 7.1 In DSRL the anchor is a trust region shaped by the decoder

Linearizing the decoder around `ε`, with `w − ε = μ + (σ − 1) ε`,

```text
‖D(w) − D(ε)‖² ≈ (w − ε)ᵀ J(ε)ᵀ J(ε) (w − ε),      J = ∂D/∂ε.
```

Compared with LP-DS's isotropic `‖w − ε‖²`:

- directions in the local null space of `J` cost zero to first order;
- decoder-sensitive directions cost more.

The metric is therefore shaped by the decoder. We do not currently have evidence that
large decoder Jacobian norm coincides with mode boundaries. Whether this makes a practical
difference over the noise-space penalty is an **open empirical question** addressed by the
proposed baseline in Section 9 (test 4).

### 7.2 Why the anchor can be more reliable than MI on local shifts

- **The anchor's minimum is the BC map.** On sampled states and noises its gradient points
  toward the reference outputs. This becomes a restoring force for coverage only when the
  reference is competent and the samples represent the relevant regions.
- **The MI reward only asks that latents be distinguishable.** It has no preference for the
  4 goals and no restoring force toward a lost mode. A seed can settle into covering 3
  modes with 8 distinguishable latents. This matches BMD seeds 7 and 9 at r = 1.16.

This is consistent with the g5 radius-study reliability counts (functional anchor 8/8,
BMD 5/8, plain DSRL 0/8). The mechanism is a hypothesis; it has not been measured directly.

The g2 results also show that the mechanisms are complementary. On held-out seeds,
anchor-only DSRL cannot discover the diagonal modes, BMD reaches `3.12/4` modes on average,
and BMD plus a moderate functional anchor reaches `3.88/4` (`7/8` seeds at `4/4`) with
`SR = 0.9995`. MI supplies specialization; the anchor appears to stabilize it.

### 7.3 Why BC quality matters

The one-step bound concerns **assignment mass**, not the **competence** of each mode. If
the BC's modes do not reach their goals, the anchor can faithfully preserve failing modes.
This matches the BC-quality study: 2k BC steps give 0 modes, 10k give 2.67, 20k give 3.67.

> The anchor preserves and adapts multimodality that the pretrained policy already has.
> By itself, it does not provide a mechanism for creating missing competence.

---

## 8. Summary

| Observation | Explanation |
|---|---|
| Plain RL collapses while SR = 1.0 | Nothing penalizes switching mode, and one mode wins by symmetry breaking |
| `x0 = 0` keeps about 2 modes | Constraining one point gives no bound on how much noise mass switches mode |
| Paired noise keeps 4/4 modes on local shifts | Consistent with the pointwise switch bound and reference restoration |
| Adaptation still happens on local shifts | Endpoint proxy has `δ² ≪ Δ²`; this is a heuristic explanation |
| λ sweep has a transition region | Consistent with the toy reward-versus-penalty model |
| Too large a λ under-adapts | Anchor gradient competes with the adaptation objective |
| Fails on g2 | `ρ² ≈ 1.17`: new goals sit on the old decision boundaries |
| Needs a good BC | The anchor preserves the mass of each mode, not its competence |
| More reliable than BMD on the g5 radius study | Reference restoration is a plausible mechanism |
| BMD + anchor is strongest on g2 | Discovery and preservation pressures are complementary |

---

## 9. Experimental status and next tests

Completed reports:

- [DSRL radius study](radius_study.md): local g5/radius shifts, BC quality, λ, and
  reliability ablations.
- [ReinFlow ablations](reinflow_ablations.md): plain, frozen, noise-only, upstream, and
  functional arms.
- [ReinFlow g2 study](reinflow_g2_study.md): standard g2 and g2 plus radius 1.22,
  including a functional-anchor coefficient sweep.
- Exploratory DSRL rotation probe at 11.25°:
  [summary](../outputs/rotation_sweep/functional_g3/summary.json). Functional anchoring with `λ=1`
  retains `4/4` modes on all three seeds (`SR=0.983`, `SR_M=0.980`,
  `Hs/logM=0.966`). Command:

  ```bash
  uv run python train.py \
    run=diagnostic policy=single_step_matched experiment=plain_dsrl \
    landscape=g3 shift=none \
    experiment.dsrl_anchor_type=functional experiment.dsrl_anchor_coef=1.0 \
    hydra.run.dir=outputs/rotation_sweep/functional_g3
  ```

The completed g2 study validates the predicted qualitative failure regime. It does not
validate the full `ρ` scaling law. The following experiments would strengthen a paper but
are not required for the narrower claims summarized above.

### Test 1: retention against the bound (no retraining)

On the final checkpoints (`outputs/radius_shift/final_functional_lambda_1`,
`outputs/reinflow_ablations/*`), sample states along rollouts and measure:

- the anchor loss `L(s)`;
- the pointwise margin `Δ(ε)` from BC samples;
- the observed switch rate `P(S)`. `retention_probe` in `flow_diagnostics.py` already
  computes this for ReinFlow; a DSRL version is needed.

Plot the observed `P(S)` against `E[‖f_θ − f₀‖² / Δ(ε)²]`, excluding or explicitly
regularizing zero-margin samples. The pointwise inequality must hold by construction; its
tightness tells us whether it meaningfully explains preservation. Also compare the
retention of plain, upstream, and functional runs.

### Test 2: rotation sweep (highest-priority missing figure)

Run the functional anchor (DSRL, λ = 1, 20k BC, seeds 0–2) at
`φ ∈ {0, 5, 10, 15, 20, 25, 30, 38, 45}°` and plot mc@0.8 and Hs/logM against
`ρ² = (2 sin(φ/2) / sin 45°)²`.

Prediction: preservation becomes harder as the endpoint proxy `ρ²` approaches 1. The
location and sharpness of any transition are empirical questions.

Also run plain DSRL and BMD at each angle. The expected pattern is that the anchor is best
at small `ρ` and BMD is relatively better at large `ρ`, which motivates combining the two.

### Test 3: the λ window against δ

Using the radius shift (δ = 0.16, 0.18, 0.20, 0.24), run a coarse λ sweep
(0.3, 0.5, 0.7, 1, 1.5, 3, 5) on 3 seeds. Prediction: the upper threshold falls roughly as
`1/δ²`, and the lower threshold stays about the same.

At the same time, log the anchor/PPO gradient ratio and the gradient-clip fraction in
`ppo.py`, so thresholds can be reported on a scale that is comparable across backends.

### Test 4: noise-space baseline (separates the method from LP-DS)

In DSRL, replace the anchor with `λ' E‖w − ε‖²` (equivalently `‖μ‖² + d(σ − 1)²`), sweep
λ', and compare the best result against the functional anchor.

- If the noise-space penalty matches the functional anchor, the DSRL contribution reduces
  to "a trust region to the prior works". The theory in Sections 4–6 still holds with `Δ`
  measured in noise space.
- If it falls short, Section 7.1 (decoder-shaped trust region) is supported.

### Test 5: the mechanism behind reliability (optional)

For BMD failure seeds 7 and 9 and for anchored runs, track per-mode mass over training.
The prediction from Section 7.2: under the anchor, a mode that dips recovers; under BMD,
it does not.

---

## 10. Blog outline

**Working title:** *Anchoring Where the Noise Goes: When Paired-Noise Regularization
Preserves Modes in Generative Policy Fine-Tuning*

1. **Hook.** RL fine-tuning reaches SR = 1.0 by collapsing onto one goal. Show endpoint
   plots for the initialized actor against plain DSRL.
2. **Background.** Noise steering (DSRL), velocity-field fine-tuning (ReinFlow), mode
   discovery (BMD). Credit FQL, ReinFlow Eq. 12 and LP-DS up front.
3. **Method.** One loss in two settings, with a figure showing the same ε sent through the
   BC map and the fine-tuned map.
4. **Results.**
   - The ablation ladder: plain → K=8 → BMD without MI → BMD → anchor.
   - Reliability counts: 8/8, 5/8, 0/8.
   - ReinFlow: `x0 = 0` against paired-noise anchoring.
   - Radius robustness and the g2 counterexample.
5. **Why it works.** The exact one-step bound plus the trajectory-level geometric and
   optimization hypotheses, clearly separated.
6. **Where it breaks.** g2 (`ρ ≈ 1`) and weak BC, both as predictions of the theory.
   Show that combining the anchor with MI reaches `3.88/4` modes on g2, versus `3.12/4`
   for BMD alone.
7. **Limitations.**
   - A single 2D environment family, 8 seeds.
   - Endpoint-level geometry used as a proxy for per-step actions.
   - Metric caveats: SR_M is inflated by modes with few attempts, and mcf depends on the
     attempt floor.

**Claims to avoid:**
- "a new loss";
- "solves mode collapse";
- "better than BMD" in general (say "more reliable on this task");
- "optimal" or "proof" for any empirical result.
