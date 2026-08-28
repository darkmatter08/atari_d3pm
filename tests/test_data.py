from __future__ import annotations

import numpy as np

from atari_d3pm.data import PongActionChunkDataset, return_stratified_split


def test_return_stratified_split_represents_both_modes():
    split = return_stratified_split(
        list(range(10)), [19, -21, -20, 20, 19, 14, 19, -21, -21, 20], 8, 0
    )
    returns = {0: 19, 1: -21, 2: -20, 3: 20, 4: 19, 5: 14, 6: 19, 7: -21, 8: -21, 9: 20}
    validation_returns = [returns[episode] for episode in split["validation"]]
    assert len(split["train"]) == 8
    assert len(split["validation"]) == 2
    assert any(value >= 0 for value in validation_returns)
    assert any(value < 0 for value in validation_returns)


def test_horizon_is_runtime_parameter(processed_dataset):
    h1 = PongActionChunkDataset(processed_dataset, split="train", horizon=1)
    h4 = PongActionChunkDataset(processed_dataset, split="train", horizon=4)
    assert len(h1) == 18
    assert len(h4) == (8 - 4 + 1) + (10 - 4 + 1)
    assert h1[0][1].shape == (1,)
    assert h4[0][1].shape == (4,)


def test_context_overlaps_and_pads_only_at_episode_start(processed_dataset):
    dataset = PongActionChunkDataset(processed_dataset, split="train", horizon=2)
    frames0, actions0 = dataset[0]
    frames1, actions1 = dataset[1]
    assert frames0[:, 0, 0].tolist() == [0, 0, 0, 0]
    assert frames1[:, 0, 0].tolist() == [0, 0, 0, 1]
    assert actions0.tolist() == [0, 1]
    assert actions1.tolist() == [1, 2]
    assert np.array_equal(frames0[1:].numpy(), frames1[:3].numpy())


def test_chunks_do_not_cross_episode_boundaries(processed_dataset):
    dataset = PongActionChunkDataset(processed_dataset, split="train", horizon=4)
    _, final_first_episode_actions = dataset[4]
    _, first_second_episode_actions = dataset[5]
    assert final_first_episode_actions.tolist() == [4, 5, 0, 1]
    assert first_second_episode_actions.tolist() == [2, 3, 4, 5]


def test_sample_stride_is_independent_of_horizon(processed_dataset):
    dataset = PongActionChunkDataset(
        processed_dataset, split="train", horizon=2, sample_stride=4
    )
    assert [index for index, _ in dataset.indices] == [0, 4, 8, 12, 16]
