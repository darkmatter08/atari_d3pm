from __future__ import annotations

import numpy as np

from atari_d3pm.data import PongActionChunkDataset
from atari_d3pm.stage2_data import (
    CollectionSpec,
    episode_manifest,
    episode_path,
    finalize_dataset,
    save_episode,
)


def _episode_arrays(value: int) -> dict[str, np.ndarray]:
    return {
        "frames": np.full((3, 84, 84), value, dtype=np.uint8),
        "actions": np.asarray([value % 6, 0, 1], dtype=np.uint8),
        "rewards": np.asarray([0, 0, 1], dtype=np.float32),
        "terminations": np.asarray([False, False, True]),
        "truncations": np.asarray([False, False, False]),
        "raw_preview": np.full((2, 210, 160, 3), value, dtype=np.uint8),
    }


def test_collection_manifest_has_disjoint_fixed_splits():
    spec = CollectionSpec(
        train_episodes=2,
        validation_episodes=1,
        test_episodes=1,
        train_seed_base=100,
        validation_seed_base=200,
        test_seed_base=300,
    )
    manifest = episode_manifest(spec)

    assert [item["split"] for item in manifest] == [
        "train",
        "train",
        "validation",
        "test",
    ]
    assert [item["seed"] for item in manifest] == [100, 101, 200, 300]


def test_finalize_stage2_dataset_is_loadable_for_all_splits(tmp_path):
    spec = CollectionSpec(
        train_episodes=2,
        validation_episodes=1,
        test_episodes=1,
        train_seed_base=100,
        validation_seed_base=200,
        test_seed_base=300,
    )
    manifest = episode_manifest(spec)
    for item in manifest:
        arrays = _episode_arrays(item["episode_id"])
        save_episode(
            episode_path(tmp_path, item["episode_id"]),
            arrays,
            {**item, "policy_mode": "deterministic"},
        )

    metadata = finalize_dataset(tmp_path, spec, verification={"passed": True})

    assert metadata["num_episodes"] == 4
    assert metadata["num_steps"] == 12
    assert len(PongActionChunkDataset(tmp_path, "train", horizon=2)) == 4
    assert len(PongActionChunkDataset(tmp_path, "validation", horizon=2)) == 2
    assert len(PongActionChunkDataset(tmp_path, "test", horizon=2)) == 2
