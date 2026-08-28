# Stage 1: Minari seed-dataset pilot results

Stage 1 completed successfully on 2026-08-28 at commit `45a902e`. Both pilot
gates passed. These are single-seed results on a small seed dataset and should
not be interpreted as the final chunk-length scaling study.

## Protocol

- Data: Minari `atari/pong/expert-v0`, with the frozen 8-episode training and
  2-episode validation split in `data/pong/v1`.
- Training: 3,000 optimizer updates, batch size 256, seed 0, and one shared
  vision-encoder architecture across policies.
- Checkpoint selection: best validation first-action accuracy. D3PM uses the
  first action of a complete deterministic reverse sample for this metric.
- Online evaluation: 20 fixed seeds (`10000` through `10019`) in
  `ALE/Pong-v5`, deterministic frame skip 4, and one-action execution (`E=1`).
- Hardware: one NVIDIA H100 80 GB GPU.

The majority-action offline accuracy was 0.2375. The random-policy online mean
return was -20.25.

## Models compared

Both learned policies condition on the same four consecutive 84x84 grayscale
frames and use the same three-layer convolutional vision encoder. They differ
after the vision encoder and are therefore different complete models.

The behavioral-cloning baseline is a direct classifier:

```text
four frames -> convolutional encoder -> two-layer MLP -> six action logits
```

It is trained with ordinary cross-entropy on the next expert action and chooses
an action with one forward pass.

The D3PM policy is a vision-conditioned categorical denoiser:

```text
four frames + noisy action tokens + diffusion timestep
    -> convolutional encoder + token/time/position embeddings
    -> three-layer Transformer -> clean-action logits
```

It is trained by corrupting expert action chunks at randomly sampled diffusion
timesteps and predicting the clean chunk with the hybrid D3PM objective. At
inference it begins with random action tokens and runs a 20-step reverse chain.
Only the first predicted action is executed before observing and replanning.

For `H=1`, BC and D3PM both ultimately output one action, but D3PM still uses
the Transformer denoiser, diffusion objective, and 20 network evaluations. It
cannot exploit temporal structure at this horizon. Consequently, `H=1` is a
useful control for the cost of diffusion, but it is not an architecture-matched
comparison. A future non-diffusion Transformer baseline would be needed to
separate backbone effects from diffusion-objective effects.

## Results

| Policy | H | Best step | Validation first-action accuracy | Mean return | Median | Range | Win rate | Batched ms / env step |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BC | 1 | 2250 | 0.4350 | 20.00 | 20.0 | 20 to 20 | 1.00 | 0.050 |
| D3PM | 1 | 2750 | 0.3070 | -17.85 | -18.5 | -21 to -12 | 0.00 | 2.521 |
| D3PM | 4 | 2500 | 0.2296 | -19.45 | -20.0 | -21 to -16 | 0.00 | 2.327 |
| D3PM | 16 | 1 | 0.2171 | -21.00 | -21.0 | -21 to -21 | 0.00 | 1.957 |
| D3PM | 64 | 2250 | 0.1935 | -20.45 | -21.0 | -21 to -19 | 0.00 | 2.320 |

The latency values are throughput-normalized measurements from batched
20-environment evaluation, not single-environment response latency.

## Gate outcomes

- BC beats the offline majority-action baseline: **pass** (0.4350 versus
  0.2375).
- At least one learned policy beats random online: **pass**. BC does so by a
  large margin, and D3PM `H=1` also improves over random (-17.85 versus -20.25).

## Interpretation

The pilot validates the full data-to-rollout path, but it does not yet show a
benefit from diffusing longer action chunks. D3PM performance is strongest at
`H=1` and generally weakens as the prediction horizon grows. This is consistent
with future expert actions becoming less identifiable from only the current
four-frame observation.

The direct BC result shows that the observations and demonstrations contain
enough information for successful imitation. Its large advantage over D3PM at
`H=1` also shows that iterative categorical diffusion is unnecessary for this
single-action prediction problem under the current setup.

The direct BC policy's perfect online record despite only 43.5% exact held-out
action agreement also shows that token accuracy and control return can diverge:
Pong has redundant or behaviorally equivalent actions in many states.

The `H=16` sampled-accuracy criterion selected the step-1 checkpoint even though
its max-noise denoising accuracy improved later. Before the larger Stage 3
sweep, checkpoint selection should be revisited on Stage 2 data—for example by
measuring sampled first-action accuracy with more samples per observation or by
using an additional online-validation protocol fixed in advance. This pilot
result should remain unchanged rather than being retuned post hoc.

Raw generated checkpoints and JSON summaries live under `runs/stage1/` on the
experiment machine and are excluded from Git because of their size.
