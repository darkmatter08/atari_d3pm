from __future__ import annotations

import json

import pytest

from atari_d3pm.cli.stage3 import _aggregate, _validate_dataset


def test_stage3_requires_frozen_v3_identity_and_split_sizes(tmp_path):
    metadata = {
        "dataset_id": "atari/pong/cleanrl-stochastic-expert-v1",
        "num_episodes": 130,
        "num_steps": 123,
        "files": {"frames.npy": {"sha256": "abc"}},
    }
    splits = {
        "train": list(range(100)),
        "validation": list(range(100, 110)),
        "test": list(range(110, 130)),
    }
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))
    (tmp_path / "splits.json").write_text(json.dumps(splits))

    identity = _validate_dataset(tmp_path)
    assert identity["split_counts"] == {
        "train": 100,
        "validation": 10,
        "test": 20,
    }

    metadata["dataset_id"] = "atari/pong/cleanrl-deterministic-expert-v1"
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))
    with pytest.raises(RuntimeError, match="stochastic v3"):
        _validate_dataset(tmp_path)


def test_stage3_aggregates_test_metrics_across_training_seeds():
    results = [
        {
            "policy_type": "d3pm",
            "horizon": 4,
            "seed": seed,
            "test": {"metrics": {"sample_first_action_accuracy": value}},
        }
        for seed, value in enumerate([0.2, 0.3, 0.4])
    ]
    aggregate = _aggregate(results)[0]
    assert aggregate["seeds"] == [0, 1, 2]
    assert aggregate["test_metrics"]["sample_first_action_accuracy"]["mean"] == pytest.approx(0.3)
