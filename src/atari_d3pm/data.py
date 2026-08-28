"""Conversion and loading utilities for episode-aligned Pong trajectories."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


PONG_ACTION_MEANINGS = [
    "NOOP",
    "FIRE",
    "RIGHT",
    "LEFT",
    "RIGHTFIRE",
    "LEFTFIRE",
]


def return_stratified_split(
    episode_ids: list[int],
    episode_returns: list[float],
    train_episodes: int,
    seed: int,
) -> dict[str, list[int]]:
    """Create a deterministic split with both return modes in validation."""
    validation_count = len(episode_ids) - train_episodes
    if train_episodes <= 0 or validation_count <= 0:
        raise ValueError("train_episodes must leave at least one validation episode")
    rng = np.random.default_rng(seed)
    positive = [ep for ep, ret in zip(episode_ids, episode_returns) if ret >= 0]
    negative = [ep for ep, ret in zip(episode_ids, episode_returns) if ret < 0]
    rng.shuffle(positive)
    rng.shuffle(negative)

    validation: list[int] = []
    if validation_count >= 2 and positive and negative:
        validation.extend([positive.pop(), negative.pop()])
    remaining = positive + negative
    rng.shuffle(remaining)
    validation.extend(remaining[: validation_count - len(validation)])
    validation_set = set(validation)
    training = [ep for ep in episode_ids if ep not in validation_set]
    rng.shuffle(training)
    rng.shuffle(validation)
    return {"train": training, "validation": validation}


def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    """Convert one raw Atari RGB frame to 84x84 grayscale uint8."""
    frame = np.asarray(frame)
    if frame.shape != (210, 160, 3) or frame.dtype != np.uint8:
        raise ValueError(f"Expected uint8 RGB frame [210, 160, 3], got {frame.shape} {frame.dtype}")
    image = Image.fromarray(frame, mode="RGB").convert("L")
    image = image.resize((84, 84), resample=Image.Resampling.BOX)
    return np.asarray(image, dtype=np.uint8)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _episode_arrays(episode: Any) -> tuple[np.ndarray, ...]:
    observations = np.asarray(episode.observations)
    actions = np.asarray(episode.actions).reshape(-1)
    rewards = np.asarray(episode.rewards, dtype=np.float32).reshape(-1)
    terminations = np.asarray(episode.terminations, dtype=bool).reshape(-1)
    truncations = np.asarray(episode.truncations, dtype=bool).reshape(-1)
    steps = actions.shape[0]
    if observations.shape[0] != steps + 1:
        raise ValueError(
            f"Episode {episode.id}: expected T+1 observations, got {observations.shape[0]} for T={steps}"
        )
    for name, values in {
        "rewards": rewards,
        "terminations": terminations,
        "truncations": truncations,
    }.items():
        if values.shape[0] != steps:
            raise ValueError(f"Episode {episode.id}: {name} length {values.shape[0]} != {steps}")
    if actions.size and (actions.min() < 0 or actions.max() >= len(PONG_ACTION_MEANINGS)):
        raise ValueError(f"Episode {episode.id}: action IDs are outside [0, 5]")
    return observations, actions.astype(np.uint8), rewards, terminations, truncations


def convert_minari_dataset(
    dataset: Any,
    output_dir: str | Path,
    split_seed: int = 0,
    train_episodes: int = 8,
    verify_environment: bool = True,
) -> dict[str, Any]:
    """Convert a Minari dataset into compact, memory-mappable decision arrays."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    action_space_n = getattr(dataset.action_space, "n", None)
    if action_space_n != len(PONG_ACTION_MEANINGS):
        raise ValueError(f"Expected Discrete(6), got {dataset.action_space!r}")

    runtime_action_meanings = None
    if verify_environment:
        try:
            env = dataset.recover_environment()
            runtime_action_meanings = list(env.unwrapped.get_action_meanings())
            env.close()
        except Exception as exc:
            raise RuntimeError(
                "Could not recover ALE/Pong to verify action meanings. Install the Atari ROM "
                "dependencies or rerun with --skip-environment-check."
            ) from exc
        if runtime_action_meanings != PONG_ACTION_MEANINGS:
            raise ValueError(
                f"Runtime action mapping {runtime_action_meanings} != expected {PONG_ACTION_MEANINGS}"
            )

    frame_parts: list[np.ndarray] = []
    action_parts: list[np.ndarray] = []
    reward_parts: list[np.ndarray] = []
    termination_parts: list[np.ndarray] = []
    truncation_parts: list[np.ndarray] = []
    offsets = [0]
    episode_ids: list[int] = []
    episode_returns: list[float] = []
    raw_preview: list[np.ndarray] = []

    for episode in dataset.iterate_episodes():
        observations, actions, rewards, terminations, truncations = _episode_arrays(episode)
        decision_observations = observations[:-1]
        processed = np.stack([preprocess_frame(frame) for frame in decision_observations])
        frame_parts.append(processed)
        action_parts.append(actions)
        reward_parts.append(rewards)
        termination_parts.append(terminations)
        truncation_parts.append(truncations)
        offsets.append(offsets[-1] + len(actions))
        episode_ids.append(int(episode.id))
        episode_returns.append(float(rewards.sum()))
        if len(raw_preview) < 12:
            remaining = 12 - len(raw_preview)
            raw_preview.extend(list(decision_observations[:remaining]))

    if not episode_ids:
        raise ValueError("The source dataset contains no episodes")
    arrays = {
        "frames.npy": np.concatenate(frame_parts),
        "actions.npy": np.concatenate(action_parts),
        "rewards.npy": np.concatenate(reward_parts),
        "terminations.npy": np.concatenate(termination_parts),
        "truncations.npy": np.concatenate(truncation_parts),
        "episode_offsets.npy": np.asarray(offsets, dtype=np.int64),
    }
    for filename, values in arrays.items():
        np.save(output / filename, values, allow_pickle=False)

    splits = return_stratified_split(
        episode_ids, episode_returns, train_episodes, split_seed
    )
    (output / "splits.json").write_text(json.dumps(splits, indent=2) + "\n")

    metadata = {
        "format_version": 1,
        "dataset_id": str(getattr(dataset, "id", "unknown")),
        "minari_version": str(getattr(dataset, "minari_version", "unknown")),
        "environment_spec": str(getattr(dataset, "env_spec", "unknown")),
        "observation_space": str(dataset.observation_space),
        "action_space": str(dataset.action_space),
        "action_meanings": PONG_ACTION_MEANINGS,
        "runtime_action_meanings": runtime_action_meanings,
        "episode_ids": episode_ids,
        "num_episodes": len(episode_ids),
        "num_steps": int(offsets[-1]),
        "preprocessing": {
            "input": "RGB uint8 [210, 160, 3]",
            "grayscale": "Pillow L conversion",
            "resize": "84x84 Pillow BOX",
            "output": "uint8 [84, 84]",
            "additional_frame_skip": 1,
        },
        "alignment": "frames[t] is decision observation o_t for actions[t] = a_t; final o_T omitted",
        "split_seed": split_seed,
        "split_strategy": "return-stratified when validation has at least two episodes",
        "splits": splits,
    }
    paths_to_hash = list(arrays) + ["splits.json"]
    metadata["files"] = {
        name: {
            "sha256": _sha256(output / name),
            "bytes": (output / name).stat().st_size,
        }
        for name in paths_to_hash
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    np.save(output / "raw_preview.npy", np.asarray(raw_preview), allow_pickle=False)
    return metadata


class PongActionChunkDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Episode-aware sliding windows over processed Pong trajectories."""

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        horizon: int = 1,
        frame_stack: int = 4,
        sample_stride: int = 1,
    ) -> None:
        if horizon < 1 or frame_stack < 1 or sample_stride < 1:
            raise ValueError("horizon, frame_stack, and sample_stride must be positive")
        self.root = Path(root)
        self.horizon = horizon
        self.frame_stack = frame_stack
        self.frames = np.load(self.root / "frames.npy", mmap_mode="r")
        self.actions = np.load(self.root / "actions.npy", mmap_mode="r")
        self.offsets = np.load(self.root / "episode_offsets.npy")
        metadata = json.loads((self.root / "metadata.json").read_text())
        splits = json.loads((self.root / "splits.json").read_text())
        if split not in splits:
            raise ValueError(f"Unknown split {split!r}; choose from {sorted(splits)}")
        episode_ids = metadata["episode_ids"]
        id_to_index = {int(episode_id): index for index, episode_id in enumerate(episode_ids)}
        self.indices: list[tuple[int, int]] = []
        for episode_id in splits[split]:
            episode_index = id_to_index[int(episode_id)]
            start = int(self.offsets[episode_index])
            end = int(self.offsets[episode_index + 1])
            last_start = end - horizon
            self.indices.extend((index, start) for index in range(start, last_start + 1, sample_stride))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor]:
        decision_index, episode_start = self.indices[item]
        first_context = decision_index - self.frame_stack + 1
        frame_indices = [max(episode_start, index) for index in range(first_context, decision_index + 1)]
        frames = np.asarray(self.frames[frame_indices]).copy()
        actions = np.asarray(self.actions[decision_index : decision_index + self.horizon]).copy()
        return torch.from_numpy(frames), torch.from_numpy(actions.astype(np.int64))
