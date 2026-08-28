from __future__ import annotations

from atari_d3pm.cli.stage5b import aggregate_families, select_family


def _result(family, seed, returns, latency=1.0):
    return {
        "family": family,
        "training_seed": seed,
        "online": {
            "returns": returns,
            "inference_ms_per_environment_step": latency,
        },
    }


def test_family_selection_aggregates_training_seeds_before_selection():
    results = [
        _result("stable", 0, [2, 2]),
        _result("stable", 1, [2, 2]),
        _result("unstable", 0, [10, 10]),
        _result("unstable", 1, [-10, -10]),
    ]
    aggregates = aggregate_families(results)
    assert select_family(aggregates)["family"] == "stable"


def test_family_selection_uses_predeclared_latency_tiebreak():
    aggregates = aggregate_families(
        [_result("slow", 0, [1], 2.0), _result("fast", 0, [1], 1.0)]
    )
    assert select_family(aggregates)["family"] == "fast"
