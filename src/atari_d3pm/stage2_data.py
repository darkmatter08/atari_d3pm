"""Resumable expert rollout collection and v2 dataset finalization."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Mapping

import numpy as np
from numpy.lib.format import open_memmap

from .data import PONG_ACTION_MEANINGS, preprocess_frame
from .expert import (
    EXPERT_FILENAME,
    EXPERT_REPOSITORY,
    EXPERT_REVISION,
    EXPERT_SHA256,
    current_rgb_frame,
    make_expert_env,
    sha256,
)


@dataclass(frozen=True)
class CollectionSpec:
    train_episodes: int = 100
    validation_episodes: int = 10
    test_episodes: int = 20
    train_seed_base: int = 30_000
    validation_seed_base: int = 40_000
    test_seed_base: int = 50_000
    policy_mode: str = "deterministic"
    max_steps: int = 27_000

    def __post_init__(self) -> None:
        counts = (self.train_episodes, self.validation_episodes, self.test_episodes)
        if any(count < 1 for count in counts):
            raise ValueError("Every split must contain at least one episode")
        if self.policy_mode not in {"deterministic", "stochastic"}:
            raise ValueError("Unknown policy mode")
        if self.max_steps < 1:
            raise ValueError("max_steps must be positive")
        all_seeds = [seed for seeds in self.seeds().values() for seed in seeds]
        if len(all_seeds) != len(set(all_seeds)):
            raise ValueError("Collection seeds must be disjoint across splits")

    def seeds(self) -> dict[str, list[int]]:
        return {
            "train": list(
                range(self.train_seed_base, self.train_seed_base + self.train_episodes)
            ),
            "validation": list(
                range(
                    self.validation_seed_base,
                    self.validation_seed_base + self.validation_episodes,
                )
            ),
            "test": list(
                range(self.test_seed_base, self.test_seed_base + self.test_episodes)
            ),
        }


def episode_manifest(spec: CollectionSpec) -> list[dict]:
    manifest = []
    episode_id = 0
    for split, seeds in spec.seeds().items():
        for seed in seeds:
            manifest.append({"episode_id": episode_id, "split": split, "seed": seed})
            episode_id += 1
    return manifest


def episode_path(root: str | Path, episode_id: int) -> Path:
    return Path(root) / "episodes" / f"episode_{episode_id:05d}.npz"


def collect_episode(
    expert,
    seed: int,
    max_steps: int,
    env_factory: Callable = make_expert_env,
    raw_frame_reader: Callable = current_rgb_frame,
) -> dict[str, np.ndarray]:
    env = env_factory()
    observation, _ = env.reset(seed=int(seed))
    frames = []
    raw_preview = []
    actions = []
    rewards = []
    terminations = []
    truncations = []
    try:
        for _ in range(max_steps):
            raw = raw_frame_reader(env)
            if len(raw_preview) < 12:
                raw_preview.append(raw)
            action = expert.action(np.asarray(observation))
            frames.append(preprocess_frame(raw))
            actions.append(action)
            observation, reward, terminated, truncated, _ = env.step(action)
            rewards.append(float(reward))
            terminations.append(bool(terminated))
            truncations.append(bool(truncated))
            if terminated or truncated:
                break
        else:
            truncations[-1] = True
    finally:
        env.close()

    return {
        "frames": np.asarray(frames, dtype=np.uint8),
        "actions": np.asarray(actions, dtype=np.uint8),
        "rewards": np.asarray(rewards, dtype=np.float32),
        "terminations": np.asarray(terminations, dtype=bool),
        "truncations": np.asarray(truncations, dtype=bool),
        "raw_preview": np.asarray(raw_preview, dtype=np.uint8),
    }


def save_episode(path: str | Path, arrays: Mapping[str, np.ndarray], metadata: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, metadata=json.dumps(metadata), **arrays)
    temporary.replace(path)


def load_episode(path: str | Path) -> tuple[dict[str, np.ndarray], dict]:
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"]))
        arrays = {
            name: np.asarray(archive[name])
            for name in (
                "frames",
                "actions",
                "rewards",
                "terminations",
                "truncations",
                "raw_preview",
            )
        }
    length = len(arrays["actions"])
    for name in ("frames", "rewards", "terminations", "truncations"):
        if len(arrays[name]) != length:
            raise RuntimeError(f"{path}: {name} is not action-aligned")
    return arrays, metadata


def audit_episode_replay(
    expert,
    path: str | Path,
    max_steps: int,
    env_factory: Callable = make_expert_env,
    raw_frame_reader: Callable = current_rgb_frame,
) -> dict:
    """Replay a deterministic episode and require exact decision alignment."""
    expected, metadata = load_episode(path)
    if metadata["policy_mode"] != "deterministic":
        raise ValueError("Exact replay audit is only defined for deterministic experts")
    env = env_factory()
    observation, _ = env.reset(seed=int(metadata["seed"]))
    checked = 0
    try:
        for index in range(min(len(expected["actions"]), max_steps)):
            frame = preprocess_frame(raw_frame_reader(env))
            action = expert.action(np.asarray(observation))
            if not np.array_equal(frame, expected["frames"][index]):
                raise RuntimeError(f"Replay frame mismatch at step {index}: {path}")
            if action != int(expected["actions"][index]):
                raise RuntimeError(f"Replay action mismatch at step {index}: {path}")
            observation, reward, terminated, truncated, _ = env.step(action)
            actual = (float(reward), bool(terminated), bool(truncated))
            wanted = (
                float(expected["rewards"][index]),
                bool(expected["terminations"][index]),
                bool(expected["truncations"][index]),
            )
            if actual != wanted:
                raise RuntimeError(
                    f"Replay transition mismatch at step {index}: {actual} != {wanted}"
                )
            checked += 1
            if terminated or truncated:
                break
    finally:
        env.close()
    if checked != len(expected["actions"]):
        raise RuntimeError(
            f"Replay length mismatch for {path}: {checked} != {len(expected['actions'])}"
        )
    return {
        "episode_id": metadata["episode_id"],
        "split": metadata["split"],
        "seed": metadata["seed"],
        "steps_checked": checked,
        "return": float(expected["rewards"].sum()),
        "exact_match": True,
    }


def finalize_dataset(
    root: str | Path,
    spec: CollectionSpec,
    verification: dict,
) -> dict:
    root = Path(root)
    manifest = episode_manifest(spec)
    episode_data = []
    lengths = []
    returns = []
    for expected in manifest:
        arrays, metadata = load_episode(episode_path(root, expected["episode_id"]))
        for key in ("episode_id", "split", "seed"):
            if metadata[key] != expected[key]:
                raise RuntimeError(
                    f"Episode {expected['episode_id']} metadata mismatch for {key}"
                )
        if metadata["policy_mode"] != spec.policy_mode:
            raise RuntimeError("Episode policy mode differs from collection spec")
        episode_data.append(arrays)
        lengths.append(len(arrays["actions"]))
        returns.append(float(arrays["rewards"].sum()))

    offsets = np.concatenate(([0], np.cumsum(lengths, dtype=np.int64)))
    total = int(offsets[-1])
    schemas = {
        "frames.npy": ((total, 84, 84), np.uint8, "frames"),
        "actions.npy": ((total,), np.uint8, "actions"),
        "rewards.npy": ((total,), np.float32, "rewards"),
        "terminations.npy": ((total,), bool, "terminations"),
        "truncations.npy": ((total,), bool, "truncations"),
    }
    for filename, (shape, dtype, key) in schemas.items():
        output = open_memmap(root / filename, mode="w+", dtype=dtype, shape=shape)
        for index, arrays in enumerate(episode_data):
            output[offsets[index] : offsets[index + 1]] = arrays[key]
        output.flush()
        del output
    np.save(root / "episode_offsets.npy", offsets, allow_pickle=False)
    np.save(root / "raw_preview.npy", episode_data[0]["raw_preview"], allow_pickle=False)

    splits = {
        split: [item["episode_id"] for item in manifest if item["split"] == split]
        for split in spec.seeds()
    }
    (root / "splits.json").write_text(json.dumps(splits, indent=2) + "\n")
    metadata = {
        "format_version": 2,
        "dataset_id": "atari/pong/cleanrl-expert-v1",
        "num_episodes": len(manifest),
        "num_steps": total,
        "episode_ids": [item["episode_id"] for item in manifest],
        "episode_manifest": manifest,
        "episode_lengths": lengths,
        "episode_returns": returns,
        "splits": splits,
        "collection_spec": asdict(spec),
        "expert": {
            "repository": EXPERT_REPOSITORY,
            "revision": EXPERT_REVISION,
            "filename": EXPERT_FILENAME,
            "sha256": EXPERT_SHA256,
            "policy_mode": spec.policy_mode,
            "verification": verification,
        },
        "environment": {
            "id": "ALE/Pong-v5",
            "base_frameskip": 1,
            "policy_frame_skip": 4,
            "repeat_action_probability": 0.0,
            "full_action_space": False,
            "noop_max": 30,
            "terminal_on_life_loss": False,
            "max_raw_frames": 108_000,
            "action_meanings": PONG_ACTION_MEANINGS,
        },
        "preprocessing": {
            "expert_input": "Gymnasium AtariPreprocessing, grayscale 84x84, stack 4",
            "stored_input": "raw current ALE RGB screen converted by preprocess_frame",
            "stored_shape": [84, 84],
            "stored_dtype": "uint8",
        },
        "alignment": "frames[t] is the decision observation used to choose actions[t]",
    }
    hashed = list(schemas) + ["episode_offsets.npy", "splits.json"]
    metadata["files"] = {
        name: {"sha256": sha256(root / name), "bytes": (root / name).stat().st_size}
        for name in hashed
    }
    (root / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata
