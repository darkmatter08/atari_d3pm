from __future__ import annotations

import json

import numpy as np
import pytest


@pytest.fixture
def processed_dataset(tmp_path):
    root = tmp_path / "pong"
    root.mkdir()
    lengths = [8, 10, 12]
    offsets = np.concatenate(([0], np.cumsum(lengths))).astype(np.int64)
    frames = np.stack(
        [np.full((84, 84), index, dtype=np.uint8) for index in range(offsets[-1])]
    )
    actions = (np.arange(offsets[-1]) % 6).astype(np.uint8)
    np.save(root / "frames.npy", frames)
    np.save(root / "actions.npy", actions)
    np.save(root / "episode_offsets.npy", offsets)
    (root / "metadata.json").write_text(
        json.dumps({"episode_ids": [10, 20, 30], "action_meanings": [str(i) for i in range(6)]})
    )
    (root / "splits.json").write_text(
        json.dumps({"train": [10, 20], "validation": [30]})
    )
    return root
