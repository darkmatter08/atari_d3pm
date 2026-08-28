# Pong Dataset Plan

## Goal

Build a small, reproducible expert-imitation dataset for a vision-conditioned
D3PM policy. Each training example conditions on four Atari frames and predicts
a chunk of future discrete actions:

```text
condition: [o[t-3], o[t-2], o[t-1], o[t]]  -> uint8 [4, 84, 84]
target:    [a[t], a[t+1], ..., a[t+H-1]]   -> int64 [H]
```

The action-chunk horizon `H` is a hyperparameter, not a property baked into the
stored dataset. Initial sweep:

```text
H = 1, 2, 4, 8, 16, 32, 64
```

Prediction horizon and execution horizon are separate. The first experiments
will execute one action and then replan (`E=1`).

## Seed dataset

Start with Minari's [`atari/pong/expert-v0`](https://minari.farama.org/datasets/atari/pong/expert-v0/).
It currently contains:

- 10 expert episodes;
- 14,790 environment steps;
- raw RGB observations with shape `[210, 160, 3]` and dtype `uint8`;
- `Discrete(6)` actions;
- trajectories produced by a CleanRL PPO-Impala expert;
- `ALE/Pong-v5` with frame skip 4 and no sticky actions.

This is a seed dataset for pipeline development, not necessarily the final
dataset for the scaling study.

## Action vocabulary

The single-player ALE Pong minimal action set has six values, documented by
the [Arcade Learning Environment](https://ale.farama.org/environments/pong/):

| ID | ALE meaning |
|---:|---|
| 0 | `NOOP` |
| 1 | `FIRE` |
| 2 | `RIGHT` |
| 3 | `LEFT` |
| 4 | `RIGHTFIRE` |
| 5 | `LEFTFIRE` |

Although the Pong paddle moves vertically on screen, ALE retains joystick
direction names. We will preserve all six raw action IDs initially. We will not
collapse fire variants until EDA shows whether doing so is justified. The
download/conversion script must also query `env.unwrapped.get_action_meanings()`
and fail if the runtime mapping differs from this recorded mapping.

## Canonical preprocessing

For each raw observation:

1. Convert RGB to grayscale.
2. Resize to `84 x 84` with area interpolation.
3. Keep values as `uint8` in `[0, 255]` on disk.
4. Normalize only when a batch enters the model.

Store individual processed frames, not materialized four-frame stacks. This
avoids storing the same frame four times and permits changing the context length
later.

The Minari dataset records observations returned by an environment already
configured with frame skip 4. We will not apply a second frame skip. Any online
evaluation environment must use the same base environment settings and the same
grayscale/resize/stack code as the offline converter.

## Sample construction and overlap

The default sample stride is one environment step:

```text
sample t:   frames [1, 2, 3, 4] -> actions [4, 5, ..., 4+H-1]
sample t+1: frames [2, 3, 4, 5] -> actions [5, 6, ..., 5+H-1]
```

Thus consecutive samples overlap, but the underlying frames are not duplicated
on disk. This is deliberate: 10 episodes is small, and stride 1 extracts all
available supervised state-action pairs.

`sample_stride` will be an independent data-loader hyperparameter with default
`1`. This lets us later compare strides such as `1`, `4`, or `H` without
confounding stride with action horizon.

At the start of an episode, missing historical observations are filled by
repeating that episode's first observation, matching common Atari frame-stack
behavior. Targets are never padded: a starting index is valid only if its full
action chunk ends before the episode boundary.

## Episode splits

Do not randomly split sliding windows. Neighboring windows share frames and
actions, which would leak nearly identical examples across splits.

The earlier 7/1/2 proposal was simply a 70/10/20 convention. With only ten
episodes, one validation episode is too noisy. For the seed dataset use:

```text
8 training episodes
2 validation episodes
0 offline test episodes
```

The primary test metric is return from new online Pong rollouts on fixed,
unseen evaluation seeds. Using eight training episodes makes better use of the
small seed dataset while two validation episodes provide a less brittle signal
for checkpoint selection. Once the dataset is expanded, freeze a separate
episode-level offline test set.

The split must be deterministic, recorded in the manifest, and reused for every
value of `H`.

## Processed storage format

```text
data/pong/v1/
  frames.npy           # uint8 [N, 84, 84] decision observations, memory-mappable
  actions.npy          # uint8 [N]
  rewards.npy          # float32 [N]
  terminations.npy     # bool [N]
  episode_offsets.npy  # int64 [num_episodes + 1]
  splits.json          # episode IDs assigned to train/validation
  metadata.json        # provenance and preprocessing contract
```

Minari episodes normally contain one more observation than action: the initial
observation plus the observation reached by every transition. The converter
must verify `len(observations) == len(actions) + 1`, store only decision
observations `o[0:T]` in `frames.npy`, and align them one-to-one with actions
`a[0:T]`. The final post-transition observation is not a policy decision input;
its presence and shape are still checked during EDA.

`metadata.json` should include:

- Minari dataset ID and version;
- environment ID and full environment specification;
- action meanings;
- source episode IDs;
- preprocessing parameters;
- split seed and episode assignments;
- converter version or Git commit;
- hashes and shapes of generated arrays.

The dataset itself remains Git-ignored. Conversion code, metadata schemas, EDA
code, and small summary reports belong in Git.

## Required EDA after download

The first download is not considered complete until an EDA command produces a
human-readable report and machine-readable summary.

### Integrity and provenance

- Dataset/version and recovered environment specification.
- Number of episodes, steps, and terminal markers.
- Episode lengths and returns.
- Observation shape, dtype, value range, and missing/non-finite values.
- Action range and agreement with the recovered action meanings.
- Checks that episode offsets and terminal boundaries agree.

### Action behavior

- Global and per-episode action counts and percentages.
- Majority-action baseline accuracy.
- Action-transition matrix.
- Action run-length distribution.
- Frequency and timing of `FIRE`, `RIGHTFIRE`, and `LEFTFIRE`.
- Counts after any proposed action canonicalization, without applying it yet.

### Visual and temporal behavior

- Contact sheets of raw frames, processed frames, and four-frame stacks.
- Mean and standard-deviation images.
- Fraction of identical or nearly identical consecutive processed frames.
- Frame-difference distribution to detect preprocessing or alignment mistakes.
- Examples around rewards, serves, and episode termination.

### Chunk availability

For every requested `H`, report:

- valid train and validation sample counts;
- fraction of steps lost near episode ends;
- target action distribution by position within the chunk;
- duplicate chunk frequency;
- effective sample count at alternate strides.

Expected EDA outputs:

```text
reports/pong_eda.md
reports/pong_eda.json
reports/figures/*.png
```

## Loader interface

The planned PyTorch interface is:

```python
PongActionChunkDataset(
    root="data/pong/v1",
    split="train",
    horizon=16,
    frame_stack=4,
    sample_stride=1,
)
```

The returned tensors are:

```text
frames:  uint8/float tensor [4, 84, 84]
actions: long tensor        [H]
```

The trainer will expose `--horizon H` and record it in every checkpoint and run
manifest. The model will output logits shaped `[batch, H, 6]`. Dataset arrays
will not be regenerated when `H` changes.

## Dataset expansion

After the Minari seed dataset can train and evaluate an end-to-end policy:

1. Retrieve the exact CleanRL expert checkpoint linked by the Minari dataset,
   or use a separately verified RL-Zoo Pong expert.
2. Reproduce its observation wrappers exactly.
3. Verify expert return over fixed seeds before collecting labels.
4. Collect at least 50-100 training episodes, 10 validation episodes, and 20
   held-out offline test episodes.
5. Keep collection seeds disjoint across splits.
6. Record deterministic and stochastic expert modes as distinct dataset sources.
7. Version the expanded dataset separately rather than overwriting `v1`.

## Training schedule and datasets

Training is deliberately staged. We should not launch the full horizon sweep
until the data alignment, loss, and sampler have each passed a cheaper test.

### Stage 0: implementation smoke tests

**When:** After the loader and vision-conditioned `x0_model` exist, before any
meaningful experiment.

**Data:** Fixed windows taken from one training episode of processed Minari
`atari/pong/expert-v0` (`data/pong/v1`). No validation or research result is
reported from this stage.

**Runs:**

1. Overfit one fixed, maximally corrupted batch of 32 well-spaced windows from
   one episode with `H=1`.
2. Overfit the same 32 decision points with `H=16`.
3. Sample action chunks from both checkpoints and verify shape, action range,
   determinism under a fixed random seed, and completion of the reverse chain.

Maximum corruption makes this a deterministic and demanding check that the
visual condition can identify the clean actions. The production training
objective still samples diffusion timesteps uniformly. A 32-window batch keeps
this a fast CPU smoke test; a larger 256-window memorization run is an optional
accelerator-backed diagnostic rather than a Stage-0 gate.

**Gate:** Training loss must fall sharply and first-action accuracy on the fixed
batch must approach 100%. Failure here is treated as an implementation bug, not
a modeling result.

### Stage 1: Minari seed-dataset pilot

**Status:** Completed on 2026-08-28. See
[`stage1_results.md`](stage1_results.md).

**When:** After Stage 0 passes and the Minari EDA report has no unresolved data
or alignment problems.

**Data:** Minari `atari/pong/expert-v0` only, using the frozen 8-episode training
and 2-episode validation split in `data/pong/v1`.

**Runs:**

1. A non-diffusion one-step behavioral-cloning baseline with `H=1`.
2. D3PM pilots with `H in {1, 4, 16, 64}` and one training seed.
3. Online evaluation of each pilot on the same 20 fixed, unseen Pong seeds,
   always executing one action before replanning (`E=1`).

All runs use the same vision encoder capacity, optimizer, number of gradient
updates, training episodes, and checkpoint-selection rule. Holding gradient
updates fixed is preferable to holding epochs fixed because the number of valid
windows changes slightly with `H`.

**Purpose:** Establish that vision-conditioned imitation works, diffusion
sampling works in the environment, and longer chunks are not obviously broken.
These are pilot results, not the final scaling study.

**Gate:** The one-step baseline must beat the majority-action baseline offline,
and at least one learned policy must achieve meaningfully better online return
than a random policy. If it does not, debug data coverage/modeling before
collecting a larger dataset.

**Implementation:** `atari-d3pm-stage1` runs the complete pilot. Each run writes
its resolved configuration, append-only validation metrics, best/last
checkpoints, a training summary, and online returns beneath `runs/stage1/`.
The default training budget is 3,000 optimizer updates per policy, with
validation every 250 updates. The CLI exposes `--horizons`, so the chunk-length
sweep does not require code changes. The rollout evaluator batches inference
over the 20 environments while stepping each environment independently and
replanning after every action.

### Stage 2: controlled expert-rollout dataset

**Status:** Corrected stochastic collection completed on 2026-08-28. See
[`stage2_results.md`](stage2_results.md).

**When:** After the Minari pilot passes.

**Data collection:** Use one verified expert checkpoint with one frozen wrapper
configuration to create `data/pong/v3`:

```text
100 training episodes
10 validation episodes
20 offline test episodes
```

These are newly collected episodes with disjoint environment and policy seeds.
The stochastic policy RNG is reset from a recorded per-episode seed, so
collection is reproducible and resumable. `v3` does not silently
merge the Minari episodes; Minari remains a separately identified pilot source.
If we later test mixed-source training, that will be an explicit ablation.

The deterministic `v2` collection is retained as an ablation only. It contained
29 unique trajectories among 130 episodes and exact trajectories crossed the
train, validation, and test boundaries. The primary `v3` build must pass a 95%
within-split uniqueness threshold and have zero exact cross-split overlap.

Before student training, evaluate the expert itself on the collection and test
seeds and record its return distribution. Run the complete EDA suite again on
`v3` and freeze its manifest and split assignments.

### Stage 3: main horizon sweep

**Status:** Completed on 2026-08-28. See
[`stage3_results.md`](stage3_results.md).

**When:** After `data/pong/v3` is frozen and audited.

**Training data:** The 100 `v3` training episodes only.

**Checkpoint selection:** Offline metrics on the 10 `v3` validation episodes.

**Final offline evaluation:** The 20 `v3` test episodes, touched only after
model and checkpoint-selection decisions are frozen.

**Runs:**

```text
H = 1, 2, 4, 8, 16, 32, 64
3 training seeds per H
E = 1 for every online rollout
sample_stride = 1 for every primary run
```

Also train the non-diffusion `H=1` behavioral-cloning baseline with three seeds.
If compute permits, add a direct autoregressive or parallel chunk-prediction
baseline at the same horizons; this is secondary to completing the D3PM sweep.

Every horizon uses identical episode splits, preprocessing, vision encoder,
transformer width/depth, diffusion schedule, batch size, number of gradient
updates, and evaluation seeds. `H` is the intended primary independent variable.

**Implementation:** `atari-d3pm-stage3` creates one resumable run directory per
policy, horizon, and training seed. It trains and selects every checkpoint
before beginning a separate offline-test pass, preventing accidental use of the
test split for model selection. The run manifest pins the v3 dataset identity
and array hashes. H100 defaults are batch size 1,024 and eight data-loader workers;
these settings are benchmarked before the primary sweep and remain explicit
hyperparameters in every saved run configuration.

### Stage 4: final online evaluation

**Status:** Implementation complete; evaluation in progress.

**When:** After the Stage 3 checkpoints and evaluation protocol are frozen.

**Data:** No offline dataset is used for the primary outcome. Each policy is
rolled out in `ALE/Pong-v5` on 100 fixed evaluation seeds that are disjoint from
all collection seeds.

Report mean return, median return, bootstrap confidence intervals, success/win
rate where appropriate, and inference latency. Aggregate across training seeds;
do not treat overlapping offline windows as independent experimental replicates.

**Implementation:** `atari-d3pm-stage4` evaluates the 24 frozen checkpoints on
environment seeds 70000-70099, which are disjoint from the Stage 1, expert
verification, and dataset-collection ranges. It is resumable per checkpoint.
Confidence intervals use a hierarchical bootstrap that resamples training seeds
and then episodes within each selected seed.

### Optional follow-up ablations

Run these only after the primary sweep:

- `sample_stride in {1, 4, H}` to measure sensitivity to correlated windows;
- execution horizon `E in {1, 2, 4}` while holding prediction horizon fixed;
- raw six-action vocabulary versus a justified canonicalized vocabulary;
- Minari-only, fresh-rollout-only, and explicitly mixed-source training;
- dataset-size scaling using fixed subsets of the 100 training episodes.

## Implementation order

1. Add dependency/configuration files and data directories to `.gitignore`.
2. Implement Minari download and conversion.
3. Implement EDA and inspect its outputs before accepting the data.
4. Implement the episode-aware, horizon-parameterized dataset loader.
5. Add boundary, alignment, split-leakage, and shape tests.
6. Adapt the D3PM `x0_model` to accept `[B, H]` tokens conditioned on visual
   features and emit `[B, H, 6]` logits.
7. Train a one-step baseline (`H=1`) and verify it can overfit a tiny subset.
8. Run the full horizon sweep with identical splits and training budgets.

## Acceptance criteria for the dataset milestone

- Conversion is deterministic from a documented upstream dataset version.
- EDA contains no unexplained action IDs, episode-boundary errors, or frame
  alignment anomalies.
- No action chunk crosses an episode boundary.
- Split membership is by episode and identical for every horizon.
- `H` and `sample_stride` are independent runtime hyperparameters.
- A small batch for every target horizon has the expected shapes and can pass
  through a placeholder model.
