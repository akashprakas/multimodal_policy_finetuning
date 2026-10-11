# Multimodal Policy Fine-Tuning

An independent reimplementation and controlled study of [Behavioral Mode Discovery (BMD)](https://behav-mod-dis.github.io/) in a four-goal environment. **The related blog can be found here** https://akashprakas.github.io/akashBlog/posts/2026-03-30-mulitmodal-generative-rl-finetuning/

The repository studies two complementary approaches to behavioral diversity during reinforcement-learning fine-tuning:

- **BMD / mutual information:** discovers distinct behaviors through latent skills.
- **Functional anchoring:** preserves useful behaviors already represented by a pretrained generative policy.

The central finding is that preservation and discovery solve different problems. Anchoring works well when the pretrained modes remain compatible with the downstream task, but it can obstruct large behavioral changes. BMD is more useful when the policy must reorganize or discover modes.

This is an independent research implementation, not an official implementation of BMD, DSRL, FQL, or ReinFlow.

## What is included

| Backend | Available experiments |
|---|---|
| DSRL / diffusion noise steering | Plain fine-tuning, BMD, functional anchoring, and BMD-plus-anchor experiments |
| ReinFlow / direct flow fine-tuning | Plain fine-tuning, frozen and noise-head controls, upstream anchoring, and paired-noise functional anchoring |

The environment supports two useful task shifts:

- `landscape=g5` with a radius shift: existing modes remain useful, testing preservation.
- `landscape=g2`: goals rotate by 45°, testing mode discovery and behavioral reorganization.

Evaluation reports overall success, mode-balanced success, successful mode coverage, entropy, and per-mode attempt counts.

## Quick start

Install the environment and run the tests:

```bash
uv sync
uv run pytest
```

Run a short smoke test:

```bash
uv run python train.py \
  run=smoke \
  policy=single_step \
  experiment=bmd \
  landscape=g2 \
  shift=none
```

Use `run=diagnostic` for three development seeds and `run=final` for the larger evaluation profile.

## Example experiments

### BMD on the 45° rotation

```bash
uv run python train.py \
  run=diagnostic \
  policy=single_step \
  experiment=bmd \
  landscape=g2 \
  shift=none
```

Plain DSRL is the corresponding task-reward-only baseline:

```bash
uv run python train.py \
  run=diagnostic \
  policy=single_step \
  experiment=plain_dsrl \
  landscape=g2 \
  shift=none
```

### Functional anchoring with DSRL

```bash
uv run python train.py \
  run=diagnostic \
  policy=single_step_matched \
  experiment=plain_dsrl \
  landscape=g5 \
  shift=dsrl_v1 \
  experiment.dsrl_anchor_type=functional \
  experiment.dsrl_anchor_coef=1.0
```

### Functional anchoring with ReinFlow

```bash
uv run python train.py \
  run=diagnostic \
  policy=single_step_matched \
  experiment=reinflow_anchor_functional \
  experiment.anchor_coef=0.03 \
  landscape=g5 \
  shift=adaptation_v1
```

Replace `experiment=reinflow_anchor_functional` with `experiment=reinflow_plain` for the unanchored baseline.


## Suggested reading path

1. [Consolidated findings](notes/findings.md) — headline results and interpretation.
2. [DSRL radius study](notes/radius_study.md) — preservation, BC quality, and anchor-strength ablations.
3. [ReinFlow ablations](notes/reinflow_ablations.md) — frozen, noise-only, upstream, and functional controls.
4. [ReinFlow rotation study](notes/reinflow_g2_study.md) — failure of anchor-only discovery under a large shift.

Regenerate the blog figures with:

```bash
uv run python scripts/make_blog_figures.py
```

## Related work

- [Behavioral Mode Discovery for Fine-tuning Multimodal Generative Policies](https://behav-mod-dis.github.io/)
- [Flow Q-Learning](https://arxiv.org/abs/2502.02538)
- [Steering Your Diffusion Policy with Latent Space Reinforcement Learning](https://arxiv.org/abs/2506.15799)
- [ReinFlow: Fine-tuning Flow Matching Policy with Online Reinforcement Learning](https://arxiv.org/abs/2505.22094)
