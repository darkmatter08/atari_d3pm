from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from atari_d3pm.training import (
    TrainConfig,
    _loader,
    evaluate_checkpoint_offline,
    load_checkpoint,
    train_policy,
)
from atari_d3pm.data import PongActionChunkDataset


def _write_tiny_dataset(root) -> None:
    episode_lengths = [6, 6]
    frames = np.zeros((sum(episode_lengths), 84, 84), dtype=np.uint8)
    actions = np.asarray([0, 1, 2, 3, 4, 5] * 2, dtype=np.uint8)
    np.save(root / "frames.npy", frames)
    np.save(root / "actions.npy", actions)
    np.save(root / "episode_offsets.npy", np.asarray([0, 6, 12], dtype=np.int64))
    (root / "metadata.json").write_text(json.dumps({"episode_ids": [10, 11]}))
    (root / "splits.json").write_text(
        json.dumps({"train": [10], "validation": [11], "test": [11]})
    )


def test_train_config_rejects_non_one_step_bc():
    with pytest.raises(ValueError, match="must use H=1"):
        TrainConfig(policy_type="bc", horizon=4)


def test_chunk_bc_allows_long_horizons_and_canonical_actions():
    config = TrainConfig(
        policy_type="chunk_bc", horizon=8, action_vocabulary="canonical4"
    )
    assert config.horizon == 8
    assert config.action_vocabulary == "canonical4"


def test_only_training_loader_keeps_workers_persistent(tmp_path):
    _write_tiny_dataset(tmp_path)
    dataset = PongActionChunkDataset(tmp_path, split="train", horizon=1)
    config = TrainConfig(
        policy_type="bc", horizon=1, num_workers=1, batch_size=2
    )
    assert _loader(dataset, config, shuffle=True).persistent_workers is True
    assert _loader(dataset, config, shuffle=False).persistent_workers is False


def test_stage5_loader_uses_spawn_after_jax_expert_loading(tmp_path):
    _write_tiny_dataset(tmp_path)
    dataset = PongActionChunkDataset(tmp_path, split="train", horizon=1)
    config = TrainConfig(
        policy_type="bc", horizon=1, num_workers=1, batch_size=2,
        checkpoint_stage=5,
    )
    assert _loader(dataset, config, shuffle=True).multiprocessing_context.get_start_method() == "spawn"


def test_behavior_cloning_training_saves_loadable_checkpoint(tmp_path):
    data_root = tmp_path / "data"
    output = tmp_path / "run"
    data_root.mkdir()
    _write_tiny_dataset(data_root)
    config = TrainConfig(
        policy_type="bc",
        horizon=1,
        data_root=str(data_root),
        output_dir=str(output),
        train_steps=1,
        batch_size=2,
        validation_every=1,
        d_model=16,
        num_workers=0,
        device="cpu",
    )

    summary = train_policy(config)
    loaded_config, model, diffusion, checkpoint = load_checkpoint(
        output / "best.pt", torch.device("cpu")
    )

    assert summary["best_metrics"] is not None
    assert loaded_config == config
    assert diffusion is None
    assert checkpoint["step"] == 1
    assert model(torch.zeros((1, 4, 84, 84), dtype=torch.uint8)).shape == (1, 6)

    offline = evaluate_checkpoint_offline(
        output / "best.pt", split="test", device_name="cpu"
    )
    assert offline["split_windows"] == 6
    assert offline["metrics"]["evaluated_windows"] == 6
