from __future__ import annotations

import numpy as np

from atari_d3pm.data import PongActionChunkDataset
from atari_d3pm.stage2_data import (
    CollectionSpec,
    audit_episode_replay,
    collect_episode,
    episode_manifest,
    episode_path,
    finalize_dataset,
    save_episode,
)


class _FakeExpert:
    def action(self, observation):
        return int(observation[0, 0, 0] % 6)


class _FakeEnv:
    def __init__(self):
        self.index = 0

    def reset(self, seed):
        self.index = 0
        return np.zeros((4, 84, 84), dtype=np.uint8), {}

    def step(self, action):
        self.index += 1
        observation = np.full((4, 84, 84), self.index, dtype=np.uint8)
        terminated = self.index == 3
        return observation, float(terminated), terminated, False, {}

    def close(self):
        pass


def _fake_raw_frame(env):
    return np.full((210, 160, 3), env.index, dtype=np.uint8)


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


def test_collected_episode_passes_exact_replay_audit(tmp_path):
    arrays = collect_episode(
        _FakeExpert(),
        seed=123,
        max_steps=10,
        env_factory=_FakeEnv,
        raw_frame_reader=_fake_raw_frame,
    )
    path = tmp_path / "episode.npz"
    save_episode(
        path,
        arrays,
        {
            "episode_id": 0,
            "split": "train",
            "seed": 123,
            "policy_mode": "deterministic",
        },
    )

    result = audit_episode_replay(
        _FakeExpert(),
        path,
        max_steps=10,
        env_factory=_FakeEnv,
        raw_frame_reader=_fake_raw_frame,
    )

    assert result["exact_match"] is True
    assert result["steps_checked"] == 3
