"""Exploratory data analysis for processed Pong trajectories."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np

from .data import PONG_ACTION_MEANINGS


def _run_lengths(actions: np.ndarray) -> list[int]:
    if not len(actions):
        return []
    changes = np.flatnonzero(actions[1:] != actions[:-1]) + 1
    boundaries = np.concatenate(([0], changes, [len(actions)]))
    return np.diff(boundaries).astype(int).tolist()


def _save_figures(root: Path, report_dir: Path, offsets: np.ndarray) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures = report_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    frames = np.load(root / "frames.npy", mmap_mode="r")
    actions = np.load(root / "actions.npy", mmap_mode="r")

    fig, axis = plt.subplots(figsize=(8, 4))
    counts = np.bincount(actions, minlength=6)
    axis.bar(PONG_ACTION_MEANINGS, counts)
    axis.set_title("Pong expert action counts")
    axis.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(figures / "action_counts.png", dpi=150)
    plt.close(fig)

    sample_ids = np.linspace(0, len(frames) - 1, 12, dtype=int)
    fig, axes = plt.subplots(3, 4, figsize=(8, 7))
    for axis, index in zip(axes.flat, sample_ids):
        axis.imshow(frames[index], cmap="gray", vmin=0, vmax=255)
        axis.set_title(f"t={index}, a={int(actions[index])}")
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(figures / "processed_contact_sheet.png", dpi=150)
    plt.close(fig)

    sample_count = min(len(frames), 5000)
    sampled = np.asarray(frames[np.linspace(0, len(frames) - 1, sample_count, dtype=int)], dtype=np.float32)
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(sampled.mean(axis=0), cmap="gray")
    axes[0].set_title("Mean processed frame")
    axes[1].imshow(sampled.std(axis=0), cmap="magma")
    axes[1].set_title("Frame standard deviation")
    for axis in axes:
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(figures / "frame_mean_std.png", dpi=150)
    plt.close(fig)

    raw_preview_path = root / "raw_preview.npy"
    if raw_preview_path.exists():
        raw = np.load(raw_preview_path)
        fig, axes = plt.subplots(2, min(6, len(raw)), figsize=(12, 4))
        for column in range(axes.shape[1]):
            axes[0, column].imshow(raw[column])
            axes[1, column].imshow(frames[column], cmap="gray", vmin=0, vmax=255)
            axes[0, column].axis("off")
            axes[1, column].axis("off")
        axes[0, 0].set_ylabel("raw")
        axes[1, 0].set_ylabel("processed")
        fig.tight_layout()
        fig.savefig(figures / "raw_vs_processed.png", dpi=150)
        plt.close(fig)


def run_eda(
    root: str | Path,
    report_dir: str | Path,
    horizons: Iterable[int] = (1, 2, 4, 8, 16, 32, 64),
    strides: Iterable[int] = (1, 4),
) -> dict:
    root = Path(root)
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    frames = np.load(root / "frames.npy", mmap_mode="r")
    actions = np.load(root / "actions.npy", mmap_mode="r")
    rewards = np.load(root / "rewards.npy", mmap_mode="r")
    terminations = np.load(root / "terminations.npy", mmap_mode="r")
    truncations = np.load(root / "truncations.npy", mmap_mode="r")
    offsets = np.load(root / "episode_offsets.npy")
    metadata = json.loads((root / "metadata.json").read_text())
    splits = json.loads((root / "splits.json").read_text())

    episode_lengths = np.diff(offsets)
    episode_returns = np.asarray([rewards[start:end].sum() for start, end in zip(offsets[:-1], offsets[1:])])
    counts = np.bincount(actions, minlength=6)
    transitions = np.zeros((6, 6), dtype=np.int64)
    run_lengths: list[int] = []
    identical_pairs = 0
    pair_count = 0
    mean_abs_diffs: list[float] = []
    for start, end in zip(offsets[:-1], offsets[1:]):
        episode_actions = np.asarray(actions[start:end])
        if len(episode_actions) > 1:
            np.add.at(transitions, (episode_actions[:-1], episode_actions[1:]), 1)
            first = np.asarray(frames[start : end - 1], dtype=np.int16)
            second = np.asarray(frames[start + 1 : end], dtype=np.int16)
            differences = np.abs(second - first)
            identical_pairs += int(np.all(differences == 0, axis=(1, 2)).sum())
            pair_count += len(differences)
            mean_abs_diffs.extend(differences.mean(axis=(1, 2)).tolist())
        run_lengths.extend(_run_lengths(episode_actions))

    episode_ids = metadata["episode_ids"]
    id_to_length = {int(ep): int(length) for ep, length in zip(episode_ids, episode_lengths)}
    id_to_return = {int(ep): float(value) for ep, value in zip(episode_ids, episode_returns)}
    split_returns = {
        split: [id_to_return[int(ep)] for ep in ids] for split, ids in splits.items()
    }
    chunk_availability = {}
    for horizon in horizons:
        per_split = {}
        for split, ids in splits.items():
            per_split[split] = {}
            for stride in set([*strides, int(horizon)]):
                valid = sum(max(0, (id_to_length[int(ep)] - int(horizon)) // int(stride) + 1) for ep in ids)
                per_split[split][f"stride_{stride}"] = int(valid)
        chunk_availability[str(horizon)] = per_split

    summary = {
        "dataset_id": metadata["dataset_id"],
        "num_episodes": int(len(episode_lengths)),
        "num_steps": int(len(actions)),
        "episode_lengths": episode_lengths.astype(int).tolist(),
        "episode_returns": episode_returns.astype(float).tolist(),
        "observation": {
            "shape": list(frames.shape),
            "dtype": str(frames.dtype),
            "min": int(frames.min()),
            "max": int(frames.max()),
            "identical_consecutive_fraction": float(identical_pairs / max(pair_count, 1)),
            "mean_absolute_frame_difference": float(np.mean(mean_abs_diffs)),
        },
        "actions": {
            "meanings": PONG_ACTION_MEANINGS,
            "counts": counts.astype(int).tolist(),
            "percentages": (counts / max(len(actions), 1)).tolist(),
            "majority_baseline_accuracy": float(counts.max() / max(len(actions), 1)),
            "transition_counts": transitions.tolist(),
            "run_length_mean": float(np.mean(run_lengths)),
            "run_length_median": float(np.median(run_lengths)),
        },
        "terminals": {
            "terminations": int(terminations.sum()),
            "truncations": int(truncations.sum()),
        },
        "chunk_availability": chunk_availability,
        "splits": splits,
        "split_episode_returns": split_returns,
    }
    (report_dir / "pong_eda.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = [
        "# Pong dataset EDA",
        "",
        f"- Dataset: `{summary['dataset_id']}`",
        f"- Episodes: {summary['num_episodes']}",
        f"- Steps: {summary['num_steps']}",
        f"- Episode lengths: {summary['episode_lengths']}",
        f"- Episode returns: {summary['episode_returns']}",
        f"- Observation range: {summary['observation']['min']} to {summary['observation']['max']}",
        f"- Identical consecutive frames: {summary['observation']['identical_consecutive_fraction']:.3%}",
        f"- Action counts: {dict(zip(PONG_ACTION_MEANINGS, summary['actions']['counts']))}",
        f"- Majority-action baseline: {summary['actions']['majority_baseline_accuracy']:.3%}",
        f"- Split returns: {summary['split_episode_returns']}",
        "",
        "## Chunk availability",
        "",
        "```json",
        json.dumps(chunk_availability, indent=2),
        "```",
        "",
        "Figures are stored in `reports/figures/`.",
    ]
    (report_dir / "pong_eda.md").write_text("\n".join(lines) + "\n")
    _save_figures(root, report_dir, offsets)
    return summary
