"""Evaluate frozen Stage-3 checkpoints online on previously unused Pong seeds."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import multiprocessing
from collections import defaultdict
from pathlib import Path

import numpy as np

from atari_d3pm.rollout import (
    evaluate_checkpoint,
    evaluate_random_policy,
    write_rollout_summary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage3", type=Path, default=Path("runs/stage3"))
    parser.add_argument("--output", type=Path, default=Path("runs/stage4"))
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed-base", type=int, default=70_000)
    parser.add_argument("--max-steps", type=int, default=27_000)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--include", nargs="+")
    parser.add_argument("--parallel-runs", type=int, default=3)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def hierarchical_bootstrap(
    returns_by_training_seed: list[list[float]],
    samples: int,
    seed: int,
) -> dict:
    """Bootstrap training seeds, then episodes within each sampled seed."""
    if samples < 1 or not returns_by_training_seed:
        raise ValueError("Bootstrap requires samples and at least one training seed")
    if any(not values for values in returns_by_training_seed):
        raise ValueError("Every training seed must contain at least one episode")
    rng = np.random.default_rng(seed)
    model_count = len(returns_by_training_seed)
    means = np.empty(samples, dtype=np.float64)
    win_rates = np.empty(samples, dtype=np.float64)
    arrays = [np.asarray(values, dtype=np.float64) for values in returns_by_training_seed]
    for sample_index in range(samples):
        selected_models = rng.integers(0, model_count, size=model_count)
        sampled = []
        for model_index in selected_models:
            values = arrays[int(model_index)]
            sampled.append(values[rng.integers(0, len(values), size=len(values))])
        values = np.concatenate(sampled)
        means[sample_index] = values.mean()
        win_rates[sample_index] = (values > 0).mean()
    return {
        "samples": samples,
        "seed": seed,
        "mean_return_ci95": np.quantile(means, [0.025, 0.975]).tolist(),
        "win_rate_ci95": np.quantile(win_rates, [0.025, 0.975]).tolist(),
    }


def aggregate_online_results(
    results: list[dict], bootstrap_samples: int, bootstrap_seed: int
) -> list[dict]:
    grouped: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for result in results:
        grouped[(result["policy_type"], result["horizon"])].append(result)

    aggregates = []
    for group_index, ((policy_type, horizon), members) in enumerate(grouped.items()):
        members.sort(key=lambda item: item["training_seed"])
        returns_by_seed = [member["online"]["returns"] for member in members]
        per_seed_means = np.asarray(
            [np.mean(values) for values in returns_by_seed], dtype=np.float64
        )
        per_seed_wins = np.asarray(
            [np.mean(np.asarray(values) > 0) for values in returns_by_seed],
            dtype=np.float64,
        )
        pooled = np.concatenate(
            [np.asarray(values, dtype=np.float64) for values in returns_by_seed]
        )
        bootstrap = hierarchical_bootstrap(
            returns_by_seed,
            samples=bootstrap_samples,
            seed=bootstrap_seed + group_index,
        )
        aggregates.append(
            {
                "policy_type": policy_type,
                "horizon": horizon,
                "training_seeds": [member["training_seed"] for member in members],
                "episodes_per_seed": [len(values) for values in returns_by_seed],
                "per_seed_mean_returns": per_seed_means.tolist(),
                "mean_return": float(per_seed_means.mean()),
                "between_seed_std_return": float(per_seed_means.std()),
                "median_pooled_return": float(np.median(pooled)),
                "min_return": float(pooled.min()),
                "max_return": float(pooled.max()),
                "per_seed_win_rates": per_seed_wins.tolist(),
                "mean_win_rate": float(per_seed_wins.mean()),
                "mean_inference_ms_per_environment_step": float(
                    np.mean(
                        [
                            member["online"]["inference_ms_per_environment_step"]
                            for member in members
                        ]
                    )
                ),
                "bootstrap": bootstrap,
            }
        )
    return sorted(aggregates, key=lambda item: (item["policy_type"], item["horizon"]))


def _load_stage3_runs(stage3: Path, include: list[str] | None) -> tuple[dict, list[dict]]:
    summary = json.loads((stage3 / "summary.json").read_text())
    if not summary.get("passed") or summary.get("stage") != 3:
        raise RuntimeError("Stage 4 requires a completed Stage 3 summary")
    results = summary["results"]
    if include is not None:
        unknown = set(include).difference(result["name"] for result in results)
        if unknown:
            raise ValueError(f"Unknown Stage 3 runs: {sorted(unknown)}")
        results = [result for result in results if result["name"] in include]
    runs = []
    for result in results:
        checkpoint = stage3 / result["name"] / "best.pt"
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        runs.append(
            {
                "name": result["name"],
                "policy_type": result["policy_type"],
                "horizon": result["horizon"],
                "training_seed": result["seed"],
                "checkpoint": checkpoint,
                "checkpoint_sha256": _sha256(checkpoint),
            }
        )
    return summary, runs


def _check_unused_seeds(stage3_summary: dict, seeds: list[int]) -> None:
    if len(seeds) != len(set(seeds)):
        raise ValueError("Evaluation seeds must be unique")
    data_root = Path(stage3_summary["data"])
    metadata = json.loads((data_root / "metadata.json").read_text())
    collection_seeds = {
        int(item["seed"]) for item in metadata.get("episode_manifest", [])
    }
    overlap = collection_seeds.intersection(seeds)
    if overlap:
        raise ValueError(f"Online seeds overlap dataset collection: {sorted(overlap)}")


def _evaluate_run_worker(
    checkpoint: Path,
    seeds: list[int],
    device: str,
    max_steps: int,
) -> dict:
    """Process-pool entry point for one independent checkpoint rollout."""
    return evaluate_checkpoint(
        checkpoint,
        seeds=seeds,
        device_name=device,
        max_steps=max_steps,
    )


def main() -> None:
    args = parse_args()
    if args.episodes < 1 or args.max_steps < 1:
        raise ValueError("episodes and max-steps must be positive")
    if args.parallel_runs < 1:
        raise ValueError("parallel-runs must be positive")
    seeds = list(range(args.seed_base, args.seed_base + args.episodes))
    stage3_summary, runs = _load_stage3_runs(args.stage3, args.include)
    _check_unused_seeds(stage3_summary, seeds)
    args.output.mkdir(parents=True, exist_ok=True)

    run_config = {
        "stage": 4,
        "stage3": str(args.stage3),
        "runs": runs,
        "evaluation_seeds": seeds,
        "max_steps": args.max_steps,
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.bootstrap_seed,
        "device": args.device,
    }
    serializable_config = {
        **run_config,
        "runs": [
            {**run, "checkpoint": str(run["checkpoint"])} for run in runs
        ],
    }
    config_path = args.output / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != serializable_config:
        raise RuntimeError(f"{args.output} has a different Stage 4 configuration")
    config_path.write_text(json.dumps(serializable_config, indent=2) + "\n")

    random_path = args.output / "random.json"
    if random_path.exists() and not args.force:
        random_online = json.loads(random_path.read_text())
        if random_online.get("seeds") != seeds or random_online.get(
            "max_steps"
        ) != args.max_steps:
            raise RuntimeError(f"Cached random settings differ: {random_path}")
    else:
        print("Evaluating random policy", flush=True)
        random_online = evaluate_random_policy(seeds, max_steps=args.max_steps)
        write_rollout_summary(random_path, random_online)

    online_by_name = {}
    pending = []
    for run in runs:
        path = args.output / f"{run['name']}.json"
        if path.exists() and not args.force:
            online = json.loads(path.read_text())
            if online.get("seeds") != seeds or online.get("max_steps") != args.max_steps:
                raise RuntimeError(f"Cached online settings differ: {path}")
            online_by_name[run["name"]] = online
        else:
            pending.append(run)

    if pending and args.parallel_runs == 1:
        for run in pending:
            print(f"Evaluating {run['name']} on {len(seeds)} seeds", flush=True)
            online = _evaluate_run_worker(
                run["checkpoint"], seeds, args.device, args.max_steps
            )
            online["checkpoint_sha256"] = run["checkpoint_sha256"]
            path = args.output / f"{run['name']}.json"
            write_rollout_summary(path, online)
            online_by_name[run["name"]] = online
    elif pending:
        context = multiprocessing.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=args.parallel_runs,
            mp_context=context,
        ) as executor:
            futures = {
                executor.submit(
                    _evaluate_run_worker,
                    run["checkpoint"],
                    seeds,
                    args.device,
                    args.max_steps,
                ): run
                for run in pending
            }
            print(
                f"Evaluating {len(pending)} checkpoints with "
                f"{args.parallel_runs} parallel workers",
                flush=True,
            )
            for future in concurrent.futures.as_completed(futures):
                run = futures[future]
                online = future.result()
                online["checkpoint_sha256"] = run["checkpoint_sha256"]
                path = args.output / f"{run['name']}.json"
                write_rollout_summary(path, online)
                online_by_name[run["name"]] = online
                print(f"Completed {run['name']}", flush=True)

    results = []
    for run in runs:
        online = online_by_name[run["name"]]
        results.append(
            {
                "name": run["name"],
                "policy_type": run["policy_type"],
                "horizon": run["horizon"],
                "training_seed": run["training_seed"],
                "online": online,
            }
        )

    random_bootstrap = hierarchical_bootstrap(
        [random_online["returns"]], args.bootstrap_samples, args.bootstrap_seed
    )
    summary = {
        "stage": 4,
        "passed": True,
        "evaluation_seeds": seeds,
        "max_steps": args.max_steps,
        "random": {**random_online, "bootstrap": random_bootstrap},
        "results": results,
        "aggregates": aggregate_online_results(
            results, args.bootstrap_samples, args.bootstrap_seed
        ),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Stage 4 complete: {args.output / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
