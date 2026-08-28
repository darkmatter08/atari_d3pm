# Stage 5: diagnostics and targeted remedies

Stage 5 completed successfully on 2026-08-28. Stage 5A diagnosed the Stage 4
offline-online disconnect. Stage 5B collected recovery states, trained 27 new
checkpoints, selected a model family on 20 fixed online seeds, and evaluated the
selected family once on 100 untouched seeds.

## Stage 5A findings

### Action aliases are real, but not the primary control failure

Mapping `RIGHTFIRE -> RIGHT` and `LEFTFIRE -> LEFT` raised the three BC
checkpoints' held-out action accuracy from 41.55%-41.73% to 53.65%-54.23%.
Pairwise checkpoint agreement rose from 76.47%-77.77% with six raw actions to
82.68%-84.38% with four canonical actions. Roughly 6.2%-6.6% of decisions were
raw disagreements that vanished under canonicalization.

The later intervention showed that this cleaner label space was not sufficient:
canonical BC reached 55.29% mean offline accuracy but scored -18.20 during
online selection. Action aliasing explains part of the low raw accuracy, but it
does not explain closed-loop success.

### Action differences precede trajectory differences

Across synchronized seeds 75000-75009, the three BC policies first chose
different raw actions within 0-8 steps. Their observations first differed 1-71
steps into the rollout. The delayed observation separation on most seeds is
consistent with redundant raw actions initially producing equivalent behavior,
followed by a motion-relevant disagreement that changes later states.

This supports trajectory drift as a mechanism, but the experiment also found a
substantial evaluation-protocol confound.

### The Stage 4 wrapper materially affected BC seed 0

On ten diagnostic seeds, BC seed 0 scored -21.0 under the Stage 4 environment
but +16.8 under the exact Stage 2 collection wrapper, winning nine of ten games.
BC seeds 1 and 2 scored +21.0 under both wrappers. On the 100 final
collection-matched seeds, raw BC seed 0 scored +9.66 with a 73% win rate; seeds
1 and 2 again won every game.

Therefore Stage 4 did not isolate covariate shift cleanly. The original seed-0
failure was partly caused by environment mismatch, especially reset/no-op and
preprocessing semantics. Residual seed instability remains under the corrected
wrapper, but it is much smaller than the original `-21/+21/+21` split.

### The D3PM reverse chain often degrades the first action

The mean predicted-clean first-action accuracy at reverse step `T`, at step 1,
and after final sampling was:

| H | Predicted x0 at T | Predicted x0 at 1 | Final sample |
|---:|---:|---:|---:|
| 1 | 39.09% | 27.05% | 27.05% |
| 2 | 38.54% | 26.86% | 26.86% |
| 4 | 38.81% | 26.47% | 26.47% |
| 8 | 36.56% | 25.85% | 25.85% |
| 16 | 28.65% | 21.88% | 21.88% |
| 32 | 19.34% | 20.42% | 20.42% |
| 64 | 21.80% | 20.46% | 20.46% |

For `H <= 16`, the denoiser already identifies a better first action from the
initial maximally noisy state than the full reverse chain ultimately returns.
The sampler is degrading useful conditional predictions rather than refining
them. At `H >= 32`, both direct predictions and samples are poor.

## Stage 5B recovery data

One round of DAgger executed each frozen BC checkpoint on ten new seeds and
queried the deterministic expert on every student-visited state. The resulting
30 episodes contain 47,653 labeled recovery states. Student and expert actions
disagree on 30.12% of them. These examples were aggregated with the original v3
training split only for the `H=1` DAgger models.

## Offline remedy results

| Family | H | Mean best validation first-action accuracy |
|---|---:|---:|
| Chunk BC | 1 | 41.17% |
| Chunk BC | 2 | 40.36% |
| Chunk BC | 4 | 40.51% |
| Chunk BC | 8 | 39.70% |
| Chunk BC | 16 | 38.47% |
| Chunk BC | 32 | 34.33% |
| Chunk BC | 64 | 35.12% |
| Canonical BC | 1 | 55.29% |
| DAgger BC | 1 | 41.42% |

Direct chunk prediction has a negative horizon trend, but it is much gentler
than D3PM's. DAgger preserves original expert-state accuracy. Canonical accuracy
is not directly comparable numerically to raw six-action accuracy, but its
online failure shows that the simpler label space did not produce stable control.

The 27 training runs took 1,890 seconds (31.5 minutes) in total. Each used batch
size 1,024 and 3,000 optimizer steps. Two persistent spawned data workers avoided
unsafe JAX/fork interaction while reducing per-run startup time.

## Predeclared online selection

All families used collection-matched Pong and seeds 80000-80019. Values below
are means over three training seeds, 20 episodes per seed.

| Family | Mean return | Between-seed SD | Win rate | Per-seed means |
|---|---:|---:|---:|---|
| Chunk BC H=1 | 21.00 | 0.00 | 100.0% | 21.00, 21.00, 21.00 |
| Chunk BC H=4 | 19.60 | 1.98 | 96.7% | 16.80, 21.00, 21.00 |
| Raw BC | 19.60 | 1.98 | 96.7% | 16.80, 21.00, 21.00 |
| DAgger BC | 16.18 | 6.81 | 88.3% | 21.00, 21.00, 6.55 |
| Chunk BC H=2 | 16.15 | 4.25 | 88.3% | 16.80, 10.65, 21.00 |
| Chunk BC H=8 | 9.22 | 13.80 | 71.7% | 21.00, -10.15, 16.80 |
| Chunk BC H=16 | 0.12 | 7.41 | 50.0% | -3.85, -6.30, 10.50 |
| Chunk BC H=64 | -18.03 | 3.71 | 0% | -21.00, -20.30, -12.80 |
| Canonical BC | -18.20 | 1.98 | 6.7% | -21.00, -16.80, -16.80 |
| Chunk BC H=32 | -19.85 | 1.31 | 0% | -20.90, -20.65, -18.00 |

The predeclared rule selected `chunk_bc_h1`. Neither canonicalization nor one
round of DAgger improved on raw BC. Longer direct chunks show a strong online
horizon penalty even though execution remains `E=1`.

## Untouched final evaluation

The selected family, raw BC reference, and random policy were evaluated on seeds
90000-90099. Each learned-policy row aggregates three checkpoints and 300
episodes.

| Family | Mean return | Hierarchical 95% CI | Win rate | Per-seed means |
|---|---:|---:|---:|---|
| Chunk BC H=1 | 21.00 | [21.00, 21.00] | 100% | 21.00, 21.00, 21.00 |
| Raw BC | 17.22 | [10.08, 21.00] | 91% | 9.66, 21.00, 21.00 |
| Random | -20.25 | -- | 0% | -- |

Chunk BC exceeds random by +41.25 points with paired 95% CI [41.14, 41.35]. Raw
BC exceeds random by +37.47 points with paired 95% CI [30.52, 41.32]. The paired
hierarchical difference between chunk BC and raw BC is +3.78 with 95% CI
[0.00, 11.06]; its bootstrap probability of being above zero is 0.694. Thus the
observed perfect robustness is encouraging, but three training seeds do not
resolve a population-level mean advantage over raw BC.

## Matched-wrapper D3PM control

Because Stage 5A found environment mismatch, all frozen D3PM checkpoints were
reevaluated post-selection on the 20 collection-matched selection seeds. These
runs were diagnostic and were not eligible to alter the selected family.

| H | Mean return | Hierarchical 95% CI | Win rate |
|---:|---:|---:|---:|
| 1 | -19.38 | [-20.10, -18.57] | 0% |
| 2 | -19.70 | [-20.25, -18.95] | 0% |
| 4 | -19.95 | [-20.47, -19.33] | 0% |
| 8 | -20.03 | [-20.50, -19.52] | 0% |
| 16 | -20.42 | [-21.00, -19.80] | 0% |
| 32 | -21.00 | [-21.00, -21.00] | 0% |
| 64 | -20.78 | [-21.00, -20.38] | 0% |

The corrected wrapper does not rescue D3PM: no checkpoint wins any of the 420
matched episodes. The direct chunk control therefore attributes the failure to
discrete diffusion and its sampling chain, rather than to action chunking or the
shared vision/transformer architecture alone.

## Conclusion

The main empirical result is now sharper than the original horizon study:

- direct chunk prediction works well at short horizons and degrades with H;
- D3PM degrades much more severely and fails online under either wrapper;
- canonical action labels improve offline accuracy but hurt online control;
- one DAgger round does not eliminate training-seed instability;
- evaluation-wrapper parity is essential for interpreting BC failures.

For this Pong imitation-learning setup, `H=1` direct transformer chunk BC is the
best tested policy. Discrete diffusion adds substantial sampling cost and a
reverse-chain failure mode without an observed control benefit.
