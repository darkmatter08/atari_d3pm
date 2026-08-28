"""One-round DAgger recovery-state collection for the Pong BC baseline."""

from __future__ import annotations

import hashlib
import json
from collections import deque
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from numpy.lib.format import open_memmap

from .data import preprocess_frame
from .expert import CleanRLPongExpert, current_rgb_frame, make_expert_env
from .rollout import _policy_actions
from .training import choose_device, load_checkpoint


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@torch.no_grad()
def collect_recovery_episode(
    student_checkpoint: str | Path,
    expert_checkpoint: str | Path,
    seed: int,
    expert_seed: int,
    max_steps: int,
    device_name: str = "auto",
) -> dict[str, np.ndarray]:
    """Execute the student while labeling each visited state with the expert."""
    device = choose_device(device_name)
    config, model, diffusion, _ = load_checkpoint(student_checkpoint, device)
    if config.policy_type != "bc" or config.action_vocabulary != "raw6":
        raise ValueError("Recovery collection expects a raw6 one-step BC student")
    expert = CleanRLPongExpert(expert_checkpoint, mode="deterministic")
    expert.reset(expert_seed)
    return _collect_loaded_recovery_episode(
        config, model, diffusion, expert, seed, expert_seed, max_steps, device
    )


@torch.no_grad()
def _collect_loaded_recovery_episode(
    config,
    model,
    diffusion,
    expert,
    seed: int,
    expert_seed: int,
    max_steps: int,
    device: torch.device,
) -> dict[str, np.ndarray]:
    expert.reset(expert_seed)
    env = make_expert_env()
    observation, _ = env.reset(seed=int(seed))
    initial_frame = preprocess_frame(current_rgb_frame(env))
    stack: deque[np.ndarray] = deque(
        [initial_frame.copy() for _ in range(4)], maxlen=4
    )
    frames = []
    expert_actions = []
    student_actions = []
    rewards = []
    terminations = []
    truncations = []
    try:
        for step in range(max_steps):
            frame = stack[-1]
            expert_action = expert.action(np.asarray(observation))
            student_action = int(
                _policy_actions(
                    torch.from_numpy(np.stack(stack)[None]),
                    config.policy_type,
                    config.horizon,
                    model,
                    diffusion,
                    device,
                    sample_seed=config.seed + step,
                    action_vocabulary=config.action_vocabulary,
                )[0]
            )
            frames.append(frame.copy())
            expert_actions.append(expert_action)
            student_actions.append(student_action)
            observation, reward, terminated, truncated, _ = env.step(student_action)
            rewards.append(float(reward))
            terminations.append(bool(terminated))
            truncations.append(bool(truncated))
            if terminated or truncated:
                break
            stack.append(preprocess_frame(current_rgb_frame(env)))
        else:
            truncations[-1] = True
    finally:
        env.close()
    return {
        "frames": np.asarray(frames, dtype=np.uint8),
        "actions": np.asarray(expert_actions, dtype=np.uint8),
        "student_actions": np.asarray(student_actions, dtype=np.uint8),
        "rewards": np.asarray(rewards, dtype=np.float32),
        "terminations": np.asarray(terminations, dtype=bool),
        "truncations": np.asarray(truncations, dtype=bool),
    }


def collect_recovery_episodes(
    student_checkpoint: str | Path,
    expert_checkpoint: str | Path,
    seeds: Sequence[int],
    expert_seed_offset: int,
    max_steps: int,
    device_name: str = "auto",
) -> list[dict[str, np.ndarray]]:
    """Collect several episodes while loading and compiling each policy once."""
    device = choose_device(device_name)
    config, model, diffusion, _ = load_checkpoint(student_checkpoint, device)
    if config.policy_type != "bc" or config.action_vocabulary != "raw6":
        raise ValueError("Recovery collection expects a raw6 one-step BC student")
    expert = CleanRLPongExpert(expert_checkpoint, mode="deterministic")
    return [
        _collect_loaded_recovery_episode(
            config,
            model,
            diffusion,
            expert,
            int(seed),
            int(seed) + expert_seed_offset,
            max_steps,
            device,
        )
        for seed in seeds
    ]


def save_recovery_episode(path: str | Path, arrays: dict, metadata: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, metadata=json.dumps(metadata), **arrays)
    temporary.replace(path)


def finalize_recovery_dataset(
    root: str | Path, episode_paths: Sequence[Path], manifest: list[dict]
) -> dict:
    root = Path(root)
    lengths = []
    disagreements = []
    for path in episode_paths:
        with np.load(path, allow_pickle=False) as archive:
            lengths.append(len(archive["actions"]))
            disagreements.append(
                float((archive["actions"] != archive["student_actions"]).mean())
            )
    offsets = np.concatenate(([0], np.cumsum(lengths, dtype=np.int64)))
    total = int(offsets[-1])
    schemas = {
        "frames.npy": ((total, 84, 84), np.uint8, "frames"),
        "actions.npy": ((total,), np.uint8, "actions"),
        "student_actions.npy": ((total,), np.uint8, "student_actions"),
        "rewards.npy": ((total,), np.float32, "rewards"),
        "terminations.npy": ((total,), bool, "terminations"),
        "truncations.npy": ((total,), bool, "truncations"),
    }
    for filename, (shape, dtype, key) in schemas.items():
        output = open_memmap(root / filename, mode="w+", dtype=dtype, shape=shape)
        for index, path in enumerate(episode_paths):
            with np.load(path, allow_pickle=False) as archive:
                output[offsets[index] : offsets[index + 1]] = archive[key]
        output.flush()
        del output
    np.save(root / "episode_offsets.npy", offsets, allow_pickle=False)
    splits = {"train": list(range(len(episode_paths)))}
    (root / "splits.json").write_text(json.dumps(splits, indent=2) + "\n")
    metadata = {
        "format_version": 1,
        "dataset_id": "atari/pong/dagger-recovery-v1",
        "num_episodes": len(episode_paths),
        "num_steps": total,
        "episode_ids": list(range(len(episode_paths))),
        "episode_lengths": lengths,
        "mean_student_expert_disagreement": float(np.mean(disagreements)),
        "per_episode_student_expert_disagreement": disagreements,
        "episode_manifest": manifest,
        "alignment": (
            "frames[t] is a student-visited decision state; actions[t] is the "
            "deterministic expert label and student_actions[t] generated the transition"
        ),
    }
    hashed = list(schemas) + ["episode_offsets.npy", "splits.json"]
    metadata["files"] = {
        name: {"sha256": _hash_file(root / name), "bytes": (root / name).stat().st_size}
        for name in hashed
    }
    (root / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata
