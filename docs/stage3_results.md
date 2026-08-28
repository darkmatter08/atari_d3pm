# Stage 3: action-chunk horizon sweep

Stage 3 completed successfully on 2026-08-28 using the accepted stochastic
`data/pong/v3` dataset. The experiment trained three independent seeds of the
one-step behavioral-cloning baseline and three seeds at every D3PM horizon in
`H = {1, 2, 4, 8, 16, 32, 64}`.

## Protocol

- Training data: 100 v3 training episodes.
- Checkpoint selection: sampled first-action accuracy on 10 validation
  episodes.
- Final offline evaluation: 20 test episodes, opened only after all 24 models
  finished training and checkpoint selection.
- Training budget: 3,000 optimizer updates per model.
- Batch size: 1,024; sample stride: 1.
- Model: shared vision encoder, width 128, three Transformer layers, four
  attention heads, and 20 diffusion steps.
- Training seeds: 0, 1, and 2.
- Hardware: one NVIDIA H100 80 GB GPU.

The 24 training runs took 1,062 seconds in total, or 17.7 minutes. Offline-test
accuracy is descriptive of imitation quality; online Pong return remains a
separate Stage 4 outcome.

## Utilization benchmark

The worst-case `H=64` model was benchmarked before the sweep for 50 updates with
one validation batch at steps 1 and 50:

| Batch size | Elapsed seconds |
|---:|---:|
| 512 | 2.34 |
| 1,024 | 2.29 |
| 2,048 | 3.62 |
| 4,096 | 6.58 |

Batch 1,024 reached 100% sampled GPU utilization and doubled examples per
update at essentially the same wall time as batch 512. Larger batches slowed a
fixed-update experiment. At batch 1,024, eight loader workers took 1.49 seconds
for the same benchmark, compared with 2.15 seconds for four workers and 2.29
seconds for 16 workers. The primary sweep therefore used batch 1,024 and eight
workers.

## Held-out results

The training-set majority-action accuracy is 0.2218. Values below are means and
population standard deviations across the three training seeds. Denoising
accuracy is measured from maximally corrupted chunks; sample accuracy uses a
complete 20-step reverse chain.

| Policy | H | Test first-action accuracy | Test token accuracy | Test max-noise denoised first-action accuracy | Selected steps |
|---|---:|---:|---:|---:|---|
| BC | 1 | 0.4165 ± 0.0007 | 0.4165 | 0.4165 | 3000, 3000, 3000 |
| D3PM | 1 | 0.2677 ± 0.0021 | 0.2677 | 0.3878 | 3000, 2750, 3000 |
| D3PM | 2 | 0.2634 ± 0.0022 | 0.2591 | 0.3830 | 3000, 3000, 2750 |
| D3PM | 4 | 0.2602 ± 0.0013 | 0.2524 | 0.3836 | 2750, 3000, 3000 |
| D3PM | 8 | 0.2515 ± 0.0012 | 0.2429 | 0.3682 | 3000, 3000, 3000 |
| D3PM | 16 | 0.2215 ± 0.0018 | 0.2206 | 0.2830 | 3000, 1, 3000 |
| D3PM | 32 | 0.1988 ± 0.0131 | 0.1990 | 0.1849 | 1, 1, 1 |
| D3PM | 64 | 0.1949 ± 0.0159 | 0.1942 | 0.2128 | 1, 1, 3000 |

## Interpretation

The main result is a clear negative scaling trend: increasing the diffused
action horizon does not improve held-out imitation accuracy under the current
four-frame conditioning setup. Performance declines gradually from `H=1`
through `H=8`, reaches the majority-action baseline at `H=16`, and falls below
it at `H=32` and `H=64`.

The gap between max-noise denoising and full reverse sampling is also important.
For `H=1`, the model denoises the first action at 38.8% accuracy but produces it
at only 26.8% accuracy after the reverse chain. This suggests that reverse
sampling, rather than representation capacity alone, is a major source of
error. The gap persists through `H=16`.

At long horizons, the checkpoint-selection signal becomes pathological. All
three `H=32` runs and two `H=64` runs select their randomly initialized step-1
checkpoint because later full-chain samples approach an action-frequency
baseline. The third `H=64` seed selects step 3000 but has similarly weak test
accuracy. This behavior is consistent across a much larger leakage-free dataset
and should be treated as a substantive failure mode of this formulation.

The direct BC baseline remains much stronger than D3PM at `H=1` (41.65% versus
26.77%). This comparison includes architectural and objective differences, so
it demonstrates that diffusion is unnecessary for one-step prediction here; it
does not by itself isolate which D3PM component causes the gap.

## Engineering audit

- Every checkpoint records Stage 3 metadata and its resolved hyperparameters.
- Seeded validation sampling uses isolated generators and cannot perturb the
  subsequent training RNG stream.
- Training and checkpoint selection completed before any test loader was
  created.
- The test pass initially exposed persistent evaluation-worker file-descriptor
  leakage. Evaluation workers were made non-persistent, all 18 tests passed,
  and the resumable runner completed from cached checkpoints without retraining
  or changing model selection.
- Generated checkpoints and summaries remain under `runs/stage3` on the H100
  and are excluded from Git because of their size.

The next step is Stage 4 online evaluation on fixed, previously unused Pong
seeds. Offline window accuracy must not be interpreted as the final control
result because behaviorally equivalent actions can produce different labels.
