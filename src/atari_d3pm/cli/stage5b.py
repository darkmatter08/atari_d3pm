"""Stage 5B: gated Pong remedies and predeclared online model selection."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import multiprocessing
from pathlib import Path

import numpy as np

from atari_d3pm.cli.stage3 import _train_or_load
from atari_d3pm.cli.stage4 import (
    _check_unused_seeds,
    _load_stage3_runs,
    hierarchical_bootstrap,
    paired_hierarchical_bootstrap,
)
from atari_d3pm.dagger import (
    collect_recovery_episodes,
    finalize_recovery_dataset,
    save_recovery_episode,
)
from atari_d3pm.expert import EXPERT_FILENAME, download_expert
from atari_d3pm.rollout import (
    evaluate_checkpoint_collection_env,
    evaluate_random_policy_collection_env,
)
from atari_d3pm.training import TrainConfig, train_policy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage3", type=Path, default=Path("runs/stage3"))
    parser.add_argument("--stage5a", type=Path, default=Path("runs/stage5a"))
    parser.add_argument("--data", type=Path, default=Path("data/pong/v3"))
    parser.add_argument("--output", type=Path, default=Path("runs/stage5b"))
    parser.add_argument(
        "--expert", type=Path, default=Path("data/expert") / EXPERT_FILENAME
    )
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--train-steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--validation-every", type=int, default=250)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--recovery-episodes-per-student", type=int, default=10)
    parser.add_argument("--recovery-seed-base", type=int, default=76_000)
    parser.add_argument("--selection-episodes", type=int, default=20)
    parser.add_argument("--selection-seed-base", type=int, default=80_000)
    parser.add_argument("--test-episodes", type=int, default=100)
    parser.add_argument("--test-seed-base", type=int, default=90_000)
    parser.add_argument("--max-steps", type=int, default=27_000)
    parser.add_argument("--parallel-runs", type=int, default=3)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--force-all-remedies", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def select_family(aggregates: list[dict]) -> dict:
    """Predeclared selection: return, win rate, latency, then family name."""
    if not aggregates:
        raise ValueError("Cannot select from no policy families")
    return sorted(
        aggregates,
        key=lambda item: (
            -item["mean_return"],
            -item["mean_win_rate"],
            item["mean_inference_ms_per_environment_step"],
            item["family"],
        ),
    )[0]


def aggregate_families(
    results: list[dict], baseline_returns: list[float] | None = None
) -> list[dict]:
    families = sorted({result["family"] for result in results})
    aggregates = []
    for family in families:
        members = sorted(
            [result for result in results if result["family"] == family],
            key=lambda item: item["training_seed"],
        )
        per_seed_returns = [
            float(np.mean(member["online"]["returns"])) for member in members
        ]
        per_seed_wins = [
            float((np.asarray(member["online"]["returns"]) > 0).mean())
            for member in members
        ]
        returns_by_seed = [member["online"]["returns"] for member in members]
        aggregate = {
                "family": family,
                "training_seeds": [member["training_seed"] for member in members],
                "per_seed_mean_returns": per_seed_returns,
                "mean_return": float(np.mean(per_seed_returns)),
                "between_seed_std_return": float(np.std(per_seed_returns)),
                "per_seed_win_rates": per_seed_wins,
                "mean_win_rate": float(np.mean(per_seed_wins)),
                "mean_inference_ms_per_environment_step": float(
                    np.mean(
                        [
                            member["online"]["inference_ms_per_environment_step"]
                            for member in members
                        ]
                    )
                ),
                "bootstrap": hierarchical_bootstrap(
                    returns_by_seed,
                    samples=10_000,
                    seed=10_000 + len(aggregates),
                ),
            }
        if baseline_returns is not None:
            aggregate["random_comparison"] = paired_hierarchical_bootstrap(
                returns_by_seed,
                baseline_returns,
                samples=10_000,
                seed=20_000 + len(aggregates),
            )
        aggregates.append(aggregate)
    return aggregates


def paired_family_bootstrap(
    candidate_by_seed: list[list[float]],
    baseline_by_seed: list[list[float]],
    samples: int = 10_000,
    seed: int = 0,
) -> dict:
    """Paired bootstrap across matching training and environment seeds."""
    if len(candidate_by_seed) != len(baseline_by_seed) or not candidate_by_seed:
        raise ValueError("Policy families must have matching training seeds")
    candidate = [np.asarray(values, dtype=np.float64) for values in candidate_by_seed]
    baseline = [np.asarray(values, dtype=np.float64) for values in baseline_by_seed]
    if any(len(left) != len(right) for left, right in zip(candidate, baseline)):
        raise ValueError("Policy families must have matching evaluation seeds")
    rng = np.random.default_rng(seed)
    differences = np.empty(samples, dtype=np.float64)
    for sample_index in range(samples):
        selected_seeds = rng.integers(0, len(candidate), size=len(candidate))
        values = []
        for seed_index in selected_seeds:
            episode_indices = rng.integers(
                0, len(candidate[seed_index]), size=len(candidate[seed_index])
            )
            values.append(
                candidate[seed_index][episode_indices]
                - baseline[seed_index][episode_indices]
            )
        differences[sample_index] = np.concatenate(values).mean()
    observed = float(
        np.mean([values.mean() for values in candidate])
        - np.mean([values.mean() for values in baseline])
    )
    return {
        "mean_return_difference": observed,
        "mean_return_difference_ci95": np.quantile(
            differences, [0.025, 0.975]
        ).tolist(),
        "probability_difference_above_zero": float((differences > 0).mean()),
    }


def _evaluation_worker(checkpoint: Path, seeds: list[int], device: str, max_steps: int):
    return evaluate_checkpoint_collection_env(
        checkpoint, seeds, device_name=device, max_steps=max_steps
    )


def evaluate_runs(
    runs: list[dict],
    output: Path,
    seeds: list[int],
    device: str,
    max_steps: int,
    parallel_runs: int,
    force: bool,
) -> list[dict]:
    output.mkdir(parents=True, exist_ok=True)
    completed = {}
    pending = []
    for run in runs:
        path = output / f"{run['name']}.json"
        if path.exists() and not force:
            result = json.loads(path.read_text())
            if result["seeds"] != seeds or result["max_steps"] != max_steps:
                raise RuntimeError(f"Cached evaluation settings differ: {path}")
            completed[run["name"]] = result
        else:
            pending.append(run)
    if pending:
        context = multiprocessing.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=parallel_runs, mp_context=context
        ) as executor:
            futures = {
                executor.submit(
                    _evaluation_worker, run["checkpoint"], seeds, device, max_steps
                ): run
                for run in pending
            }
            for future in concurrent.futures.as_completed(futures):
                run = futures[future]
                result = future.result()
                (output / f"{run['name']}.json").write_text(
                    json.dumps(result, indent=2) + "\n"
                )
                completed[run["name"]] = result
                print(f"Completed online evaluation {run['name']}", flush=True)
    return [
        {
            "name": run["name"],
            "family": run["family"],
            "training_seed": run["training_seed"],
            "online": completed[run["name"]],
        }
        for run in runs
    ]


def _collect_recovery_data(
    root: Path,
    bc_runs: list[dict],
    expert_path: Path,
    episodes_per_student: int,
    seed_base: int,
    max_steps: int,
    device: str,
) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    episode_paths = []
    manifest = []
    episode_id = 0
    for student_index, run in enumerate(bc_runs):
        seeds = list(
            range(
                seed_base + student_index * episodes_per_student,
                seed_base + (student_index + 1) * episodes_per_student,
            )
        )
        missing = []
        for seed in seeds:
            path = root / "episodes" / f"episode_{episode_id:05d}.npz"
            item = {
                "episode_id": episode_id,
                "seed": seed,
                "expert_seed": seed + 1_000_000,
                "student_name": run["name"],
                "student_training_seed": run["training_seed"],
                "student_checkpoint": str(run["checkpoint"]),
            }
            episode_paths.append(path)
            manifest.append(item)
            if not path.exists():
                missing.append((path, item))
            episode_id += 1
        if missing:
            arrays = collect_recovery_episodes(
                run["checkpoint"], expert_path, [item[1]["seed"] for item in missing],
                expert_seed_offset=1_000_000, max_steps=max_steps, device_name=device
            )
            for values, (path, item) in zip(arrays, missing):
                save_recovery_episode(path, values, item)
                print(f"Collected recovery episode {item['episode_id']}", flush=True)
    return finalize_recovery_dataset(root, episode_paths, manifest)


def main() -> None:
    args = parse_args()
    if args.parallel_runs < 1:
        raise ValueError("parallel-runs must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    stage5a = json.loads((args.stage5a / "summary.json").read_text())
    if stage5a.get("stage") != "5A" or not stage5a.get("passed"):
        raise RuntimeError("Stage 5B requires a completed Stage 5A")
    stage3_summary, stage3_runs = _load_stage3_runs(args.stage3, include=None)
    bc_runs = sorted(
        [run for run in stage3_runs if run["policy_type"] == "bc"],
        key=lambda item: item["training_seed"],
    )
    recommendations = stage5a["recommendations"]
    run_canonical = args.force_all_remedies or recommendations["test_canonical4"]
    run_dagger = args.force_all_remedies or recommendations["collect_dagger_recovery_data"]

    selection_seeds = list(
        range(args.selection_seed_base, args.selection_seed_base + args.selection_episodes)
    )
    test_seeds = list(range(args.test_seed_base, args.test_seed_base + args.test_episodes))
    recovery_seeds = list(
        range(
            args.recovery_seed_base,
            args.recovery_seed_base + args.recovery_episodes_per_student * len(bc_runs),
        )
    )
    _check_unused_seeds(stage3_summary, selection_seeds + test_seeds + recovery_seeds)
    if len(set(selection_seeds + test_seeds + recovery_seeds)) != (
        len(selection_seeds) + len(test_seeds) + len(recovery_seeds)
    ):
        raise ValueError("Recovery, selection, and final-test seeds must be disjoint")

    run_config = {
        "stage": "5B",
        "stage3": str(args.stage3),
        "stage5a": str(args.stage5a),
        "data": str(args.data),
        "horizons": args.horizons,
        "training_seeds": args.seeds,
        "train_steps": args.train_steps,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "validation_every": args.validation_every,
        "num_workers": args.num_workers,
        "recovery_seeds": recovery_seeds,
        "selection_seeds": selection_seeds,
        "test_seeds": test_seeds,
        "max_steps": args.max_steps,
        "gates": {"canonical4": run_canonical, "dagger": run_dagger},
        "device": args.device,
    }
    config_path = args.output / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != run_config:
        raise RuntimeError(f"{args.output} has a different Stage 5B configuration")
    config_path.write_text(json.dumps(run_config, indent=2) + "\n")

    recovery_root = args.output / "recovery_data"
    recovery_metadata = None
    if run_dagger:
        expert_path = download_expert(args.expert)
        recovery_metadata = _collect_recovery_data(
            recovery_root, bc_runs, expert_path,
            args.recovery_episodes_per_student, args.recovery_seed_base,
            args.max_steps, args.device
        )

    candidate_specs = []
    for horizon in args.horizons:
        for seed in args.seeds:
            candidate_specs.append(
                (f"chunk_bc_h{horizon}", "chunk_bc", horizon, seed, "raw6", None)
            )
    if run_canonical:
        for seed in args.seeds:
            candidate_specs.append(
                ("canonical4_bc", "bc", 1, seed, "canonical4", None)
            )
    if run_dagger:
        for seed in args.seeds:
            candidate_specs.append(
                ("dagger_bc", "bc", 1, seed, "raw6", str(recovery_root))
            )

    trained_runs = []
    for family, policy_type, horizon, seed, vocabulary, recovery in candidate_specs:
        name = f"{family}_seed{seed}"
        config = TrainConfig(
            policy_type=policy_type,
            horizon=horizon,
            data_root=str(args.data),
            output_dir=str(args.output / "training" / name),
            train_steps=args.train_steps,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            validation_every=args.validation_every,
            num_workers=args.num_workers,
            checkpoint_stage=5,
            seed=seed,
            device=args.device,
            action_vocabulary=vocabulary,
            recovery_data_root=recovery,
        )
        _train_or_load(config, args.force)
        trained_runs.append(
            {"name": name, "family": family, "training_seed": seed,
             "checkpoint": Path(config.output_dir) / "best.pt"}
        )

    comparison_runs = [
        {"name": run["name"], "family": "raw_bc", "training_seed": run["training_seed"],
         "checkpoint": run["checkpoint"]}
        for run in bc_runs
    ]
    all_selection_runs = comparison_runs + trained_runs
    selection_results = evaluate_runs(
        all_selection_runs, args.output / "selection", selection_seeds,
        args.device, args.max_steps, args.parallel_runs, args.force
    )
    selection_aggregates = aggregate_families(selection_results)
    selected = select_family(selection_aggregates)

    selected_runs = [
        run for run in all_selection_runs if run["family"] == selected["family"]
    ]
    if selected["family"] != "raw_bc":
        final_runs = comparison_runs + selected_runs
    else:
        final_runs = comparison_runs
    final_results = evaluate_runs(
        final_runs, args.output / "final", test_seeds,
        args.device, args.max_steps, args.parallel_runs, args.force
    )
    random_path = args.output / "final" / "random.json"
    if args.force or not random_path.exists():
        random_result = evaluate_random_policy_collection_env(
            test_seeds, max_steps=args.max_steps
        )
        random_path.write_text(json.dumps(random_result, indent=2) + "\n")
    else:
        random_result = json.loads(random_path.read_text())

    # Post-selection diagnostic control: evaluate frozen D3PM checkpoints with
    # the same wrapper and seeds as direct chunk BC. These runs cannot
    # retroactively participate in model-family selection.
    d3pm_control_runs = [
        {
            "name": run["name"],
            "family": f"d3pm_h{run['horizon']}",
            "training_seed": run["training_seed"],
            "checkpoint": run["checkpoint"],
        }
        for run in stage3_runs
        if run["policy_type"] == "d3pm"
    ]
    d3pm_control_results = evaluate_runs(
        d3pm_control_runs,
        args.output / "matched_d3pm",
        selection_seeds,
        args.device,
        args.max_steps,
        args.parallel_runs,
        args.force,
    )

    final_aggregates = aggregate_families(
        final_results, baseline_returns=random_result["returns"]
    )
    ordered_final = sorted(final_results, key=lambda item: item["training_seed"])
    selected_returns = [
        item["online"]["returns"]
        for item in ordered_final
        if item["family"] == selected["family"]
    ]
    raw_returns = [
        item["online"]["returns"]
        for item in ordered_final
        if item["family"] == "raw_bc"
    ]
    selected_vs_raw = (
        paired_family_bootstrap(selected_returns, raw_returns)
        if selected["family"] != "raw_bc"
        else None
    )

    summary = {
        "stage": "5B",
        "passed": True,
        "gates": {"canonical4": run_canonical, "dagger": run_dagger,
                  "chunk_bc_control": True},
        "recovery_dataset": recovery_metadata,
        "selection_protocol": {
            "environment": "stage2_collection",
            "seeds": selection_seeds,
            "rule": "mean return, win rate, lower latency, lexicographic family",
            "aggregates": selection_aggregates,
            "selected_family": selected["family"],
        },
        "final_protocol": {
            "environment": "stage2_collection",
            "seeds": test_seeds,
            "random": random_result,
            "results": final_results,
            "aggregates": final_aggregates,
            "selected_vs_raw_bc": selected_vs_raw,
        },
        "matched_d3pm_control": {
            "role": "post-selection diagnostic; not eligible for family selection",
            "seeds": selection_seeds,
            "results": d3pm_control_results,
            "aggregates": aggregate_families(d3pm_control_results),
        },
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Stage 5B complete: {args.output / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
