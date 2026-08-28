"""Run the multi-seed Stage-3 horizon sweep on the frozen v3 dataset."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np

from atari_d3pm.data import PongActionChunkDataset
from atari_d3pm.training import (
    TrainConfig,
    evaluate_checkpoint_offline,
    train_policy,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/pong/v3"))
    parser.add_argument("--output", type=Path, default=Path("runs/stage3"))
    parser.add_argument(
        "--horizons", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64]
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--train-steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--validation-every", type=int, default=250)
    parser.add_argument("--validation-max-batches", type=int)
    parser.add_argument("--test-max-batches", type=int)
    parser.add_argument("--diffusion-steps", type=int, default=20)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--sample-stride", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _validate_dataset(root: Path) -> dict:
    metadata = json.loads((root / "metadata.json").read_text())
    splits = json.loads((root / "splits.json").read_text())
    if metadata.get("dataset_id") != "atari/pong/cleanrl-stochastic-expert-v1":
        raise RuntimeError("Stage 3 requires the accepted stochastic v3 dataset")
    expected_counts = {"train": 100, "validation": 10, "test": 20}
    actual_counts = {split: len(splits.get(split, [])) for split in expected_counts}
    if actual_counts != expected_counts:
        raise RuntimeError(
            f"Stage 3 requires split counts {expected_counts}, got {actual_counts}"
        )
    return {
        "dataset_id": metadata["dataset_id"],
        "num_episodes": metadata["num_episodes"],
        "num_steps": metadata["num_steps"],
        "split_counts": actual_counts,
        "files": metadata["files"],
    }


def _train_or_load(config: TrainConfig, force: bool) -> dict:
    output = Path(config.output_dir)
    summary_path = output / "training_summary.json"
    checkpoint_path = output / "best.pt"
    if not force and summary_path.exists() and checkpoint_path.exists():
        saved_values = json.loads((output / "config.json").read_text())
        if asdict(TrainConfig(**saved_values)) != asdict(config):
            raise RuntimeError(
                f"Run {output} has a different configuration; choose another "
                "--output or pass --force"
            )
        print(f"Using completed run {output}", flush=True)
        return json.loads(summary_path.read_text())
    return train_policy(config)


def _aggregate(results: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for result in results:
        grouped[(result["policy_type"], result["horizon"])].append(result)

    aggregates = []
    for (policy_type, horizon), members in grouped.items():
        metric_names = members[0]["test"]["metrics"].keys()
        metrics = {}
        for name in metric_names:
            values = np.asarray(
                [member["test"]["metrics"][name] for member in members],
                dtype=np.float64,
            )
            metrics[name] = {
                "mean": float(values.mean()),
                "std": float(values.std()),
                "values": values.tolist(),
            }
        aggregates.append(
            {
                "policy_type": policy_type,
                "horizon": horizon,
                "seeds": [member["seed"] for member in members],
                "test_metrics": metrics,
            }
        )
    return sorted(aggregates, key=lambda item: (item["policy_type"], item["horizon"]))


def main() -> None:
    args = parse_args()
    if len(set(args.horizons)) != len(args.horizons) or min(args.horizons) < 1:
        raise ValueError("Horizons must be unique positive integers")
    if len(set(args.seeds)) != len(args.seeds) or min(args.seeds) < 0:
        raise ValueError("Seeds must be unique non-negative integers")
    args.output.mkdir(parents=True, exist_ok=True)
    dataset = _validate_dataset(args.data)

    run_config = {
        "stage": 3,
        "data": str(args.data),
        "dataset": dataset,
        "horizons": args.horizons,
        "seeds": args.seeds,
        "train_steps": args.train_steps,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "validation_every": args.validation_every,
        "validation_max_batches": args.validation_max_batches,
        "test_max_batches": args.test_max_batches,
        "diffusion_steps": args.diffusion_steps,
        "d_model": args.d_model,
        "n_layers": args.n_layers,
        "n_heads": args.n_heads,
        "num_workers": args.num_workers,
        "sample_stride": args.sample_stride,
        "device": args.device,
    }
    config_path = args.output / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != run_config:
        raise RuntimeError(
            f"{args.output} has a different sweep configuration; choose another output"
        )
    config_path.write_text(json.dumps(run_config, indent=2) + "\n")

    train_h1 = PongActionChunkDataset(
        args.data, split="train", horizon=1, sample_stride=args.sample_stride
    )
    first_actions = np.asarray(
        [train_h1.actions[index] for index, _ in train_h1.indices],
        dtype=np.int64,
    )
    majority_accuracy = float(
        np.bincount(first_actions, minlength=6).max() / len(first_actions)
    )

    specs = [("bc", 1, seed) for seed in args.seeds]
    specs.extend(
        ("d3pm", horizon, seed)
        for horizon in args.horizons
        for seed in args.seeds
    )
    trained = []
    for policy_type, horizon, seed in specs:
        name = f"{policy_type}_h{horizon}_seed{seed}"
        config = TrainConfig(
            policy_type=policy_type,
            horizon=horizon,
            data_root=str(args.data),
            output_dir=str(args.output / name),
            train_steps=args.train_steps,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            validation_every=args.validation_every,
            diffusion_steps=args.diffusion_steps,
            d_model=args.d_model,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            num_workers=args.num_workers,
            sample_stride=args.sample_stride,
            validation_max_batches=args.validation_max_batches,
            checkpoint_stage=3,
            seed=seed,
            device=args.device,
        )
        training = _train_or_load(config, args.force)
        trained.append(
            {
                "name": name,
                "policy_type": policy_type,
                "horizon": horizon,
                "seed": seed,
                "config": config,
                "training": training,
            }
        )

    # Do not touch the test split until every model is trained and selected.
    results = []
    for item in trained:
        run_dir = Path(item["config"].output_dir)
        test_path = run_dir / "offline_test.json"
        if test_path.exists() and not args.force:
            test = json.loads(test_path.read_text())
        else:
            print(f"Testing frozen checkpoint {item['name']}", flush=True)
            test = evaluate_checkpoint_offline(
                run_dir / "best.pt",
                split="test",
                device_name=args.device,
                max_batches=args.test_max_batches,
            )
            test_path.write_text(json.dumps(test, indent=2) + "\n")
        results.append(
            {
                "name": item["name"],
                "policy_type": item["policy_type"],
                "horizon": item["horizon"],
                "seed": item["seed"],
                "training": item["training"],
                "test": test,
            }
        )

    summary = {
        "stage": 3,
        "passed": True,
        "data": str(args.data),
        "dataset": dataset,
        "majority_action_accuracy": majority_accuracy,
        "results": results,
        "aggregates": _aggregate(results),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Stage 3 complete: {args.output / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
