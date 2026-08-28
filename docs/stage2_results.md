# Stage 2: controlled expert-rollout dataset

Stage 2 completed successfully on 2026-08-28. It produced the frozen
`data/pong/v2` dataset for the main horizon sweep.

## Expert provenance

The collector uses the exact expert named by the Minari `atari/pong/expert-v0`
metadata:

- Repository: [`cleanrl/Pong-v5-cleanba_ppo_envpool_impala_atari_wrapper-seed1`](https://huggingface.co/cleanrl/Pong-v5-cleanba_ppo_envpool_impala_atari_wrapper-seed1)
- Repository revision: `f2ad2531c78cc639f2a54511aa8716765c33499d`
- Checkpoint: `cleanba_ppo_envpool_impala_atari_wrapper.cleanrl_model`
- Checkpoint SHA-256: `7b76801eff3153a6f55a87c2a6e221d96b6a428fb76c2c29fdedc93f204a96e4`
- Algorithm: CleanBA PPO with the IMPALA-style Atari network
- Collection policy: deterministic argmax

Deterministic collection is intentionally identified as a different mode from
the categorical sampling used in CleanRL's published evaluation helper. The
checkpoint itself is unchanged.

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
EnvPool setup. It is not claimed to be pixel-identical to EnvPool; compatibility
is established empirically by the expert verification result below.

## Seeds and splits

All seed ranges are disjoint:

| Purpose | Seeds | Episodes |
|---|---|---:|
| Expert verification | 20000-20019 | 20 |
| Training | 30000-30099 | 100 |
| Validation | 40000-40009 | 10 |
| Offline test | 50000-50019 | 20 |

The Stage 1 online seeds, 10000-10019, are also disjoint from every Stage 2
range.

## Expert verification

The frozen expert was evaluated before dataset collection:

| Metric | Result |
|---|---:|
| Mean return | 21.0 |
| Standard deviation | 0.0 |
| Minimum / maximum | 21.0 / 21.0 |
| Win rate | 100% |

This passes the predeclared mean-return threshold of 18 and win-rate threshold
of 90%.

## Dataset EDA

| Property | Result |
|---|---:|
| Episodes | 130 |
| Decisions | 217,127 |
| Return range | 21 to 21 |
| Episode-length range | 1,629 to 1,734 |
| Mean episode length | 1,670.21 |
| Terminations | 130 |
| Truncations | 0 |
| Identical consecutive frames | 0.175% |
| Majority-action baseline | 26.54% |
| On-disk size | approximately 1.5 GB |

Action counts:

| ID | Action | Count | Fraction |
|---:|---|---:|---:|
| 0 | NOOP | 57,629 | 26.54% |
| 1 | FIRE | 50,408 | 23.22% |
| 2 | RIGHT | 23,162 | 10.67% |
| 3 | LEFT | 32,095 | 14.78% |
| 4 | RIGHTFIRE | 23,112 | 10.64% |
| 5 | LEFTFIRE | 30,721 | 14.15% |

All actions occur frequently. At `H=64` and stride 1, the finalized loader
provides 160,672 training windows, 16,214 validation windows, and 32,051 test
windows.

## Integrity checks

- The exact checkpoint hash is verified before it can be loaded.
- Collection is resumable through one compressed file per episode.
- Episode metadata must match its frozen ID, split, seed, and policy mode.
- Final arrays are written as memory-mappable files and hashed.
- One episode from every split was replayed from its seed. All stored frames,
  actions, rewards, terminal flags, and lengths matched exactly.
- `PongActionChunkDataset` was exercised at `H=64` on all three splits; every
  returned action remained in `[0, 5]`.
- All 14 repository tests passed locally and on the H100.

Finalized array hashes:

| File | SHA-256 |
|---|---|
| `frames.npy` | `00f1ecf060120f1877bf34093025412f340678e0912ba4fa23163ef38b0cc7ae` |
| `actions.npy` | `fc783832a8ea18c93a298c7fc24e5c70bdf518170ef29cb0c427bc1037047eb5` |
| `rewards.npy` | `5cf6c04a8e43be2564e6e1d400a3a851fb1e942883d0aef317739d2a2e6927cc` |
| `terminations.npy` | `c11ffee54f7ac12f42ac43fb1a0aef3cf253746b475243508d78eb3e16d3e893` |
| `truncations.npy` | `a5f2106f11814b7d194d7995484b31d046f5e80327cc9395e8a39f4229354bba` |
| `episode_offsets.npy` | `b02cf9e7b838ebd09d1ca5d9439116c64ec1e48615a60412cff5c7eb056a714f` |
| `splits.json` | `3116682bfa6cbb1d0a511409d7f7ea6aabf6bd32385096cadbd86fe457263731` |

The generated dataset, per-episode recovery files, EDA figures, and JSON
reports remain on the experiment machine under `data/pong/v2` and
`reports/pong_v2`. They are excluded from Git because of their size.
