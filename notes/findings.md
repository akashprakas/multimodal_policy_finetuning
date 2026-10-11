
> Functional anchoring and BMD solve different failure modes. Anchoring preserves useful pretrained structure; MI discovers distinct behaviors; combining them gives the strongest result when adaptation requires both discovery and stability.

The numerical results and resolved configurations are internally consistent. Several duplicated configurations reproduce exactly, which is a good implementation check.

## Headline results

| Setting | SR | SR_M | mcf | Hs/logM |
|---|---:|---:|---:|---:|
| Plain DSRL, g2 | 1.000 | 0.250 | 1.00 | 0.000 |
| Full BMD, g2 | 0.976 | 0.797 | 3.12 | 0.793 |
| **BMD + anchor, g2** | **0.9995** | **0.968** | **3.88** | **0.910** |
| Plain DSRL, g5 | 0.9995 | 0.250 | 1.00 | 0.000 |
| **Functional anchor, g5** | **0.993** | **0.993** | **4.00** | **0.978** |

The final BMD+anchor result is available in [summary.json](../multirun/bmd_functional_anchor_g2/lambda_0.1/summary.json), and the final anchor-only results are under [final_functional_anchor_g5](../multirun/final_functional_anchor_g5).

## Finding 1: reward-only RL collapses multimodality

Plain DSRL obtains nearly perfect overall success but selects one physical goal:

```text
SR=1.0
SR_M=0.25
mcf=1/4
Hs=0
```

This is the clearest motivation for the project: raw success can completely conceal behavioral collapse.

The important evaluation metrics are therefore:

- `SR`: task completion.
- `SR_M`: mode-balanced success.
- `mcf`: reliably solved modes.
- `Hs/logM`: balance among successful behaviors.

## Finding 2: a functional anchor preserves competent modes

On g5 with a competent 20k BC prior:

```text
BC prior: SR=0.871, mcf=4/4
```

Reward-only fine-tuning collapses:

```text
λ=0:
SR=1.000
SR_M=0.250
mcf=1/4
```

A sufficiently strong functional anchor preserves everything:

```text
λ=1:
SR=0.993
SR_M=0.993
mcf=4/4 on all 8 seeds
Hs=0.978
```

There is a sharp threshold:

```text
λ≤0.5 → collapse
λ=0.7 → partial reliability
λ=1.0 → reliable preservation
```

The anchor also remains stable from 150 to 600 RL epochs, while unanchored training remains collapsed and eventually becomes less successful.

This supports:

> Functional anchoring can prevent catastrophic behavioral collapse during RL fine-tuning without latents or MI.

However, g5 is a mild shift whose success regions overlap the original behaviors. This serves as strong mode preservation evidence rather than out-of-distribution adaptation.

## Finding 3: anchor success depends on reference quality

At fixed `λ=1` on g5:

```text
2k BC  → SR_M=0.170, mcf=0
5k BC  → SR_M=0.791, mcf=2
10k BC → SR_M=0.948, mcf=3.33
20k BC → SR_M=0.982, mcf=4
```

An anchor preserves whatever its reference contains. If the reference is undertrained, anchoring can trap the policy near poor behavior.

The right claim is:

> Functional anchors are effective when the reference is competent and compatible with the downstream task.

## Finding 4: functional anchoring alone cannot discover g2 modes

On g2, the pretrained cardinal modes are approximately 45° away from the diagonal goals.

Anchor-only DSRL exhibits a trade-off:

```text
small λ → adapts but collapses to one goal
large λ → remains close to unsuitable BC behavior and fails
```

Neither 2k nor 20k BC resolves this. Functional anchoring is therefore not a mode-discovery objective.

## Finding 5: BMD discovers modes, but remains variable

Full BMD substantially improves held-out g2 diversity:

```text
SR=0.976
SR_M=0.797
mcf=3.12
Hs=0.793
```

The ablations show:

- K=1 collapses.
- K=8 alone gives partial diversity.
- The training schedule and entropy help.
- MI improves held-out coverage and balance.
- Classifier input noise is load-bearing.

Without classifier smoothing, the discriminator can distinguish small near-origin jitter. This creates high attempt entropy without useful successful behavior. Noise around `0.1–0.3` prevents this shortcut; `0.5` is too destructive.

## Finding 6: BMD and the anchor are complementary

This is the most compelling result.

On held-out seeds 3–10:

```text
Full BMD:
SR=0.976
SR_M=0.797
mcf=3.12
Hs=0.793

BMD + functional anchor, λ=0.1:
SR=0.9995
SR_M=0.968
mcf=3.88
Hs=0.910
```

Seven of eight anchored BMD seeds achieve 4/4 coverage. The remaining seed covers 3/4.

The likely division of labour is:

```text
MI:
pushes different latent codes toward distinct diagonal behaviors

Functional anchor:
prevents excessive local drift and missing-mode instability
```

This updates the earlier conclusion:

- Functional anchor alone fails on g2.
- BMD alone discovers g2 modes but occasionally loses one.
- BMD plus a moderate anchor is the strongest tested configuration.

The anchor does not eliminate all failures (seed 8 still misses one goal), but substantially improves reliability across seeds.

## Finding 7: stronger BC is not always better for downstream steering

Held-out BMD results:

| BC steps | SR | SR_M | mcf |
|---:|---:|---:|---:|
| 2k | **0.976** | **0.797** | **3.12** |
| 10k | 0.889 | 0.745 | 2.38 |
| 20k | 0.724 | 0.605 | 1.12 |

Meanwhile, more BC training improves in-distribution g0/g5 performance but drives g2 prior success to zero.

The evidence supports:

> Greater BC convergence can produce a more specialized decoder that is less useful for downstream steering under a large distribution shift.

While deeper attractor basins provide a plausible mechanism, this remains a hypothesis pending direct Jacobian or reachability measurements.

## Finding 8: more RL training does not discover missing modes

Increasing BMD from 300 to 600 epochs:

```text
SR:   0.945 → 0.990
SR_M: 0.875 → 0.827
Hs:   0.815 → 0.744
```

Additional fine-tuning improves exploitation and raw success but not diversity. Mode discovery happens early; longer task-reward optimization does not reliably recover absent modes.
