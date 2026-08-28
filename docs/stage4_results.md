# Stage 4: final online evaluation

Stage 4 completed successfully on 2026-08-28. It evaluated all 24 frozen Stage
3 checkpoints for 100 complete Pong episodes each, producing 2,400 learned-policy
rollouts plus a 100-episode random-policy baseline.

## Protocol

- Environment: `ALE/Pong-v5`, six-action space, frame skip 4, and no sticky
  actions.
- Evaluation seeds: 70000-70099, unused by Stage 1, expert verification, or v3
  dataset collection.
- Execution horizon: `E=1`; every policy observes and replans after each action.
- Checkpoints: selected exclusively by Stage 3 validation metrics before this
  online evaluation began.
- Replicates: training seeds 0, 1, and 2 for every policy and horizon.
- Uncertainty: 10,000-sample hierarchical bootstrap, resampling training seeds
  and then episodes within each selected seed.
- Random comparisons: paired by evaluation seed, with the same hierarchical
  resampling of training seeds.

Three checkpoints were evaluated concurrently. This overlaps their CPU-heavy
ALE stepping while keeping one H100 available for batched inference. Each
checkpoint result was written immediately, making the run resumable.

## Results

The random policy scored a mean return of -20.40 with a 95% episode-bootstrap
interval of [-20.56, -20.23] and zero wins.

| Policy | H | Mean return | Hierarchical 95% CI | Per-training-seed means | Win rate | Difference from random (95% CI) |
|---|---:|---:|---:|---|---:|---:|
| BC | 1 | 7.00 | [-21.00, 21.00] | -21.00, 21.00, 21.00 | 66.67% | 27.40 [-0.58, 41.47] |
| D3PM | 1 | -20.12 | [-20.53, -19.69] | -20.06, -19.79, -20.51 | 0% | 0.28 [-0.14, 0.73] |
| D3PM | 2 | -20.50 | [-20.68, -20.29] | -20.36, -20.47, -20.67 | 0% | -0.10 [-0.30, 0.14] |
| D3PM | 4 | -20.16 | [-20.40, -19.90] | -20.02, -20.16, -20.31 | 0% | 0.24 [-0.02, 0.50] |
| D3PM | 8 | -20.48 | [-20.65, -20.29] | -20.39, -20.41, -20.63 | 0% | -0.08 [-0.27, 0.14] |
| D3PM | 16 | -20.71 | [-21.00, -20.51] | -20.58, -21.00, -20.56 | 0% | -0.31 [-0.58, -0.08] |
| D3PM | 32 | -21.00 | [-21.00, -21.00] | -21.00, -21.00, -21.00 | 0% | -0.60 [-0.70, -0.51] |
| D3PM | 64 | -20.80 | [-21.00, -20.41] | -21.00, -21.00, -20.39 | 0% | -0.40 [-0.67, -0.02] |

Rollout-normalized inference timing:

| Policy | H | Mean inference ms / environment step |
|---|---:|---:|
| BC | 1 | 0.012 |
| D3PM | 1 | 0.913 |
| D3PM | 2 | 0.706 |
| D3PM | 4 | 0.869 |
| D3PM | 8 | 0.722 |
| D3PM | 16 | 0.539 |
| D3PM | 32 | 0.408 |
| D3PM | 64 | 0.475 |

These latency values come from three concurrent CUDA processes and dynamically
shrinking active batches. They describe the evaluation workload, not isolated
single-environment response latency, so the non-monotonic horizon trend should
not be interpreted as an architectural speedup.

## Interpretation

No D3PM checkpoint won a single game in 2,100 online episodes. `H=1` and `H=4`
have small positive mean-return differences from random, but their paired 95%
intervals include zero. `H=16`, `H=32`, and `H=64` are detectably worse than
random under the paired bootstrap. The Stage 3 negative horizon trend therefore
survives online evaluation and becomes stronger in actual control.

BC exposes a different failure mode. Its three checkpoints have almost
identical held-out action accuracy (41.55%-41.73%), yet seed 0 loses every game
and seeds 1 and 2 win every game. The pooled mean return of 7.0 is consequently
not a description of a typical policy. The hierarchical interval correctly
spans the complete return range because three training seeds provide limited
information about this bimodal training outcome.

This offline-online disconnect is the most important Stage 4 finding. Exact
expert-action accuracy does not identify whether a learned Pong policy is
closed-loop stable. Small differences in which redundant actions a model learns
can compound into either perfect play or complete failure.

Our leading interpretation is trajectory drift caused by closed-loop covariate
shift, not conventional supervised overfitting. BC is trained and validated on
states visited by the expert, but at evaluation time it controls which states it
will observe next. A small early action error can move the policy onto a state
trajectory that is rare or absent in the expert dataset; predictions on those
states can then become less reliable and compound the deviation. This explains
how checkpoints with nearly indistinguishable held-out accuracy can have
opposite returns. It remains a hypothesis until a trace-level diagnostic locates
the first action divergence and measures how quickly the resulting observations
depart from the expert and successful-policy distributions.

The current evidence does not support a benefit from discrete diffusion over
action chunks for this setup. The next useful experiments would target the
failure mechanism rather than extend the same horizon sweep: canonicalize
behaviorally equivalent actions, add an architecture-matched non-diffusion
chunk predictor, inspect reverse-chain degradation, and use interactive data
collection or a predeclared online-validation protocol to address covariate
shift.

## Integrity checks

- All checkpoint hashes and evaluation seeds are recorded in the Stage 4 run
  manifest.
- Every policy used exactly the same 100 environment seeds.
- Results are cached per checkpoint and the final summary contains all 24
  learned policies.
- Paired bootstrap comparisons preserve evaluation-seed alignment.
- All 22 repository tests pass locally and on the H100.
- Generated JSON results remain under `runs/stage4` on the H100 and are excluded
  from Git.
