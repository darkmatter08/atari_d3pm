"""Run the complete Minari seed-dataset Stage-1 pilot."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from atari_d3pm.data import PongActionChunkDataset
from atari_d3pm.rollout import (
    evaluate_checkpoint,
    evaluate_random_policy,
    write_rollout_summary,
)
from atari_d3pm.training import TrainConfig, train_policy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/pong/v1"))
    parser.add_argument("--output", type=Path, default=Path("runs/stage1"))
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 4, 16, 64])
    parser.add_argument("--train-steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--validation-every", type=int, default=250)
    parser.add_argument("--diffusion-steps", type=int, default=20)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--eval-seed-base", type=int, default=10_000)
    parser.add_argument("--max-eval-steps", type=int, default=27_000)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _train_or_load(config: TrainConfig, force: bool) -> dict:
    summary_path = Path(config.output_dir) / "training_summary.json"
    checkpoint_path = Path(config.output_dir) / "best.pt"
    if not force and summary_path.exists() and checkpoint_path.exists():
        saved_values = json.loads(
            (Path(config.output_dir) / "config.json").read_text()
        )
        saved_config = asdict(TrainConfig(**saved_values))
        if saved_config != asdict(config):
            raise RuntimeError(
                f"Run {config.output_dir} has a different configuration; "
                "choose another --output or pass --force"
            )
        print(f"Using completed run {config.output_dir}", flush=True)
        return json.loads(summary_path.read_text())
    return train_policy(config)


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.eval_episodes < 1 or args.max_eval_steps < 1:
        raise ValueError("eval-episodes and max-eval-steps must be positive")
    seeds = list(range(args.eval_seed_base, args.eval_seed_base + args.eval_episodes))

    train_dataset = PongActionChunkDataset(args.data, split="train", horizon=1)
    first_actions = np.asarray(
        [train_dataset.actions[index] for index, _ in train_dataset.indices], dtype=np.int64
    )
    majority_accuracy = float(np.bincount(first_actions, minlength=6).max() / len(first_actions))

    random_path = args.output / "random_online.json"
    if args.force or not random_path.exists():
        print("Evaluating random-policy baseline", flush=True)
        random_online = evaluate_random_policy(seeds, max_steps=args.max_eval_steps)
        write_rollout_summary(random_path, random_online)
    else:
        random_online = json.loads(random_path.read_text())
        if random_online.get("seeds") != seeds or random_online.get(
            "max_steps"
        ) != args.max_eval_steps:
            raise RuntimeError(
                f"{random_path} used different evaluation settings; "
                "choose another --output or pass --force"
            )

    run_specs = [("bc", 1)] + [("d3pm", horizon) for horizon in args.horizons]
    results = []
    for policy_type, horizon in run_specs:
        name = f"{policy_type}_h{horizon}"
        run_dir = args.output / name
        config = TrainConfig(
            policy_type=policy_type,
            horizon=horizon,
            data_root=str(args.data),
            output_dir=str(run_dir),
            train_steps=args.train_steps,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            validation_every=args.validation_every,
            diffusion_steps=args.diffusion_steps,
            d_model=args.d_model,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            num_workers=args.num_workers,
            seed=args.seed,
            device=args.device,
        )
        training = _train_or_load(config, args.force)
        online_path = run_dir / "online.json"
        if args.force or not online_path.exists():
            print(f"Evaluating {name} online", flush=True)
            online = evaluate_checkpoint(
                run_dir / "best.pt",
                seeds=seeds,
                device_name=args.device,
                max_steps=args.max_eval_steps,
            )
            write_rollout_summary(online_path, online)
        else:
            online = json.loads(online_path.read_text())
            if online.get("seeds") != seeds or online.get(
                "max_steps"
            ) != args.max_eval_steps:
                raise RuntimeError(
                    f"{online_path} used different evaluation settings; "
                    "choose another --output or pass --force"
                )
        results.append({"name": name, "training": training, "online": online})

    bc_result = next(result for result in results if result["name"] == "bc_h1")
    bc_offline = bc_result["training"]["best_metrics"]["first_action_accuracy"]
    best_learned_return = max(result["online"]["mean_return"] for result in results)
    summary = {
        "stage": 1,
        "data": str(args.data),
        "seed": args.seed,
        "evaluation_seeds": seeds,
        "max_evaluation_steps": args.max_eval_steps,
        "majority_action_accuracy": majority_accuracy,
        "random_online": random_online,
        "results": results,
        "gates": {
            "bc_beats_majority_offline": bc_offline > majority_accuracy,
            "learned_policy_beats_random_online": best_learned_return
            > random_online["mean_return"],
        },
    }
    summary["passed"] = all(summary["gates"].values())
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Stage 1 summary: {args.output / 'summary.json'}", flush=True)
    print(f"Stage 1 gates: {summary['gates']}", flush=True)
    if not summary["passed"]:
        raise SystemExit("Stage 1 did not pass all pilot gates")


if __name__ == "__main__":
    main()
