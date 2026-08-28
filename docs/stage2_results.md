# Stage 2: controlled expert-rollout dataset

Stage 2 completed successfully on 2026-08-28. The primary dataset is the
stochastic, reproducible `data/pong/v3` collection. It contains 100 training,
10 validation, and 20 offline-test episodes.

## Why v2 was replaced

The first Stage 2 collection used deterministic argmax actions. Pong has no
sticky actions in this setup, so environment seeds mostly select one of a small
number of reset no-op counts. The resulting `v2` data contained only 29 unique
trajectories among 130 nominal episodes:

| Split | Episodes | Unique trajectories |
|---|---:|---:|
| Training | 100 | 28 |
| Validation | 10 | 8 |
| Test | 20 | 16 |

There were 7 exact train-validation overlaps, 15 exact train-test overlaps,
and 5 exact validation-test overlaps. That leakage invalidates `v2` as the
primary source for a held-out horizon study. The files remain on the H100 as a
deterministic-policy ablation and debugging artifact, but Stage 3 must not train
or select models with them.

The corrected `v3` collector samples from the expert's categorical policy. It
resets the policy RNG from a recorded seed at the start of every episode, which
makes stochastic collection independent of collection order, resumable, and
exactly replayable. Dataset finalization requires at least 95% unique
trajectories within each split and zero exact trajectory overlap across splits.

## Expert provenance

The collector uses the exact expert named by the Minari `atari/pong/expert-v0`
metadata:

- Repository: [`cleanrl/Pong-v5-cleanba_ppo_envpool_impala_atari_wrapper-seed1`](https://huggingface.co/cleanrl/Pong-v5-cleanba_ppo_envpool_impala_atari_wrapper-seed1)
- Repository revision: `f2ad2531c78cc639f2a54511aa8716765c33499d`
- Checkpoint: `cleanba_ppo_envpool_impala_atari_wrapper.cleanrl_model`
- Checkpoint SHA-256: `7b76801eff3153a6f55a87c2a6e221d96b6a428fb76c2c29fdedc93f204a96e4`
- Algorithm: CleanBA PPO with the IMPALA-style Atari network
- Collection policy: categorical sampling from the expert logits

## Frozen environment and preprocessing

- Environment: `ALE/Pong-v5`
- Minimal six-action space
- Base frame skip 1; policy frame skip 4
- Sticky-action probability 0
- Up to 30 no-op reset actions
- No episodic-life termination during evaluation or collection
- Expert input: grayscale 84x84 Atari preprocessing with four-frame stacking
- Stored student input: current raw ALE decision frame converted to grayscale
  84x84 with the repository's frozen `preprocess_frame`
- Maximum: 108,000 raw frames, equivalent to 27,000 policy decisions

The Gymnasium preprocessing is parameter-matched to the expert's published
EnvPool setup. Compatibility is established empirically by the expert
verification below.

## Seeds and splits

Environment and policy seed ranges are disjoint by purpose. The policy seed is
the environment seed plus 1,000,000 and is recorded in every episode manifest.

| Purpose | Environment seeds | Policy seeds | Episodes |
|---|---|---|---:|
| Expert verification | 20000-20019 | 1020000-1020019 | 20 |
| Training | 30000-30099 | 1030000-1030099 | 100 |
| Validation | 40000-40009 | 1040000-1040009 | 10 |
| Offline test | 50000-50019 | 1050000-1050019 | 20 |

The Stage 1 online seeds, 10000-10019, are also disjoint from every Stage 2
range.

## Expert verification and diversity gates

The sampled expert was evaluated before collection:

| Metric | Result |
|---|---:|
| Mean return | 21.0 |
| Standard deviation | 0.0 |
| Minimum / maximum | 21.0 / 21.0 |
| Win rate | 100% |

This passes the predeclared mean-return threshold of 18 and win-rate threshold
of 90%. The completed dataset also passed every diversity gate:

| Split | Episodes | Unique | Unique fraction |
|---|---:|---:|---:|
| Training | 100 | 100 | 100% |
| Validation | 10 | 10 | 100% |
| Test | 20 | 20 | 100% |
| Overall | 130 | 130 | 100% |

Exact train-validation, train-test, and validation-test overlap counts are all
zero.

## Dataset EDA

| Property | Result |
|---|---:|
| Episodes | 130 |
| Decisions | 216,972 |
| Return range | 21 to 21 |
| Episode-length range | 1,629 to 1,734 |
| Mean episode length | 1,669.02 |
| Terminations | 130 |
| Truncations | 0 |
| Identical consecutive frames | 0.160% |
| Majority-action baseline | 22.19% |
| On-disk size | approximately 1.5 GB |

Action counts:

| ID | Action | Count | Fraction |
|---:|---|---:|---:|
| 0 | NOOP | 48,151 | 22.19% |
| 1 | FIRE | 44,389 | 20.46% |
| 2 | RIGHT | 28,681 | 13.22% |
| 3 | LEFT | 36,067 | 16.62% |
| 4 | RIGHTFIRE | 25,772 | 11.88% |
| 5 | LEFTFIRE | 33,912 | 15.63% |

All actions occur frequently. At `H=64` and stride 1, the finalized loader
provides 160,535 training windows, 16,213 validation windows, and 32,034 test
windows.

## Integrity checks

- The exact checkpoint hash is verified before it can be loaded.
- Collection is resumable through one compressed file per episode.
- Episode metadata must match its frozen ID, split, environment seed, policy
  seed, and policy mode.
- Final arrays are memory-mappable and content-hashed.
- One stochastic episode from every split was replayed from both seeds. Every
  stored frame, sampled action, reward, terminal flag, and length matched
  exactly.
- All 130 trajectories were content-hashed before acceptance.
- `PongActionChunkDataset` was exercised at `H=64` on all three splits.
- All 14 repository tests passed locally and on the H100.

Finalized array hashes:

| File | SHA-256 |
|---|---|
| `frames.npy` | `95f1d6986c62478e0bae879165dbe3911eece6ce8ee6383efc5d5e6b3e5b0125` |
| `actions.npy` | `22367c39dfc7f80281c707a2b73648e9779eb6a082ed1f5cfcae63f846b529aa` |
| `rewards.npy` | `887579ad02b3698a0d0068e84b00ed57c74af0259e9c0e6b0ac85dee07756ad4` |
| `terminations.npy` | `ce4c180b5855654f4eb5db7d9e7a2420f6a538e60a81d7788ec01bc42f64a17c` |
| `truncations.npy` | `6c3ba74741121ca7d8890a6f380fbfdbb8c6ef0e636a9d241d4385963d013b3f` |
| `episode_offsets.npy` | `c12d7f5c9a93f2f40aeee013113b8396f9462e75b30e307f126dde1e9d7b1997` |
| `splits.json` | `3116682bfa6cbb1d0a511409d7f7ea6aabf6bd32385096cadbd86fe457263731` |

The generated primary dataset, per-episode recovery files, EDA figures, and
JSON reports remain on the experiment machine under `data/pong/v3` and
`reports/pong_v3`. They are excluded from Git because of their size.
