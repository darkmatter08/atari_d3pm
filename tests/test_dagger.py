from __future__ import annotations

import json

import numpy as np

from atari_d3pm.dagger import finalize_recovery_dataset
from atari_d3pm.data import PongActionChunkDataset


def test_finalize_recovery_dataset_is_loadable(tmp_path):
    episode_dir = tmp_path / "episodes"
    episode_dir.mkdir()
    paths = []
    for episode, length in enumerate((3, 4)):
        path = episode_dir / f"episode_{episode}.npz"
        actions = np.arange(length, dtype=np.uint8) % 6
        np.savez_compressed(
            path,
            metadata=json.dumps({"episode": episode}),
            frames=np.full((length, 84, 84), episode, dtype=np.uint8),
            actions=actions,
            student_actions=(actions + 1) % 6,
            rewards=np.zeros(length, dtype=np.float32),
            terminations=np.zeros(length, dtype=bool),
            truncations=np.zeros(length, dtype=bool),
        )
        paths.append(path)
    metadata = finalize_recovery_dataset(tmp_path, paths, [{}, {}])
    dataset = PongActionChunkDataset(tmp_path, split="train", horizon=1)
    assert metadata["num_steps"] == 7
    assert metadata["mean_student_expert_disagreement"] == 1.0
    assert len(dataset) == 7
