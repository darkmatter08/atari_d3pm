# Pong expert-v0 EDA findings

This is the checked-in summary of the generated `reports/pong_eda.md` and
`reports/pong_eda.json` outputs for Minari `atari/pong/expert-v0`, converted by
this repository on 2026-08-28.

## Dataset health

- 10 episodes and 14,790 aligned decision steps were recovered.
- Runtime ALE reported the expected action meanings: `NOOP`, `FIRE`, `RIGHT`,
  `LEFT`, `RIGHTFIRE`, and `LEFTFIRE`.
- Processed observations have shape `[14790, 84, 84]`, dtype `uint8`, and range
  62-242.
- Only 0.704% of within-episode consecutive processed frame pairs are exactly
  identical.
- Visual inspection confirmed that grayscale/resize processing preserves the
  score, paddles, ball, and playfield.
- Episode offsets, action lengths, reward lengths, and terminal markers passed
  the converter's consistency checks.

## Returns

Episode lengths:

```text
[1864, 824, 1202, 1752, 1941, 1962, 1804, 880, 792, 1769]
```

Episode returns:

```text
[19, -21, -20, 20, 19, 14, 19, -21, -21, 20]
```

The nominal expert dataset is bimodal: six episodes are strong positive
trajectories and four are near-complete losses. Stage 0 deliberately retains
the published dataset unchanged. The 8/2 seed split is return-stratified so the
validation set contains one positive and one negative episode; this avoids the
initial random split that placed only losses in validation. Before Stage 1, we
must still decide whether the research target is imitation of this behavior
distribution or filtering/reweighting toward strong episodes.

## Actions

| Action | Count | Percent |
|---|---:|---:|
| `NOOP` | 3,499 | 23.66% |
| `FIRE` | 2,930 | 19.81% |
| `RIGHT` | 1,632 | 11.03% |
| `LEFT` | 2,850 | 19.27% |
| `RIGHTFIRE` | 1,273 | 8.61% |
| `LEFTFIRE` | 2,606 | 17.62% |

The majority-action baseline is 23.66%. No class is absent, so the initial
six-action vocabulary remains appropriate.

## Available stride-1 windows

| Horizon | Train windows | Validation windows |
|---:|---:|---:|
| 1 | 12,158 | 2,632 |
| 2 | 12,150 | 2,630 |
| 4 | 12,134 | 2,626 |
| 8 | 12,102 | 2,618 |
| 16 | 12,038 | 2,602 |
| 32 | 11,910 | 2,570 |
| 64 | 11,654 | 2,506 |

The longest proposed horizon still retains more than 96% of the training
decision points, so episode-end truncation is not a serious sample-size
confound for this seed dataset.
