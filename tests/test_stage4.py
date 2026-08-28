from __future__ import annotations

import json

import pytest

from atari_d3pm.cli.stage4 import (
    _check_unused_seeds,
    aggregate_online_results,
    hierarchical_bootstrap,
)


def test_hierarchical_bootstrap_is_seeded_and_finite():
    returns = [[-1, 0, 1, 2], [0, 1, 2, 3], [-2, -1, 0, 1]]
    first = hierarchical_bootstrap(returns, samples=200, seed=7)
    second = hierarchical_bootstrap(returns, samples=200, seed=7)
    assert first == second
    assert first["mean_return_ci95"][0] <= first["mean_return_ci95"][1]
    assert 0 <= first["win_rate_ci95"][0] <= first["win_rate_ci95"][1] <= 1


def test_online_aggregation_uses_training_seeds_as_replicates():
    results = []
    for seed, returns in enumerate(([1, 2], [2, 3], [3, 4])):
        results.append(
            {
                "policy_type": "d3pm",
                "horizon": 4,
                "training_seed": seed,
                "online": {
                    "returns": returns,
                    "inference_ms_per_environment_step": 2.0 + seed,
                },
            }
        )
    aggregate = aggregate_online_results(results, 100, 0)[0]
    assert aggregate["per_seed_mean_returns"] == [1.5, 2.5, 3.5]
    assert aggregate["mean_return"] == pytest.approx(2.5)
    assert aggregate["between_seed_std_return"] == pytest.approx(0.81649658)
    assert aggregate["mean_inference_ms_per_environment_step"] == 3.0


def test_stage4_rejects_dataset_collection_seeds(tmp_path):
    (tmp_path / "metadata.json").write_text(
        json.dumps({"episode_manifest": [{"seed": 30_000}, {"seed": 40_000}]})
    )
    summary = {"data": str(tmp_path)}
    _check_unused_seeds(summary, [70_000, 70_001])
    with pytest.raises(ValueError, match="overlap"):
        _check_unused_seeds(summary, [30_000])
