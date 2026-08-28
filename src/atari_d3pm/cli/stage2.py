"""Verify the pinned Pong expert and collect the controlled Stage-2 dataset."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from atari_d3pm.eda import run_eda
from atari_d3pm.expert import CleanRLPongExpert, download_expert, make_expert_env
from atari_d3pm.stage2_data import (
    CollectionSpec,
    audit_episode_replay,
    collect_episode,
    episode_manifest,
    episode_path,
    finalize_dataset,
    load_episode,
    save_episode,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("data/expert/cleanrl_pong/expert.cleanrl_model"),
    )
    parser.add_argument("--output", type=Path, default=Path("data/pong/v2"))
    parser.add_argument("--report", type=Path, default=Path("reports/pong_v2"))
    parser.add_argument("--policy-mode", choices=["deterministic", "stochastic"], default="deterministic")
    parser.add_argument("--train-episodes", type=int, default=100)
    parser.add_argument("--validation-episodes", type=int, default=10)
    parser.add_argument("--test-episodes", type=int, default=20)
    parser.add_argument("--train-seed-base", type=int, default=30_000)
    parser.add_argument("--validation-seed-base", type=int, default=40_000)
    parser.add_argument("--test-seed-base", type=int, default=50_000)
    parser.add_argument("--verification-episodes", type=int, default=20)
    parser.add_argument("--verification-seed-base", type=int, default=20_000)
    parser.add_argument("--minimum-expert-return", type=float, default=18.0)
    parser.add_argument("--minimum-win-rate", type=float, default=0.9)
    parser.add_argument("--max-steps", type=int, default=27_000)
    parser.add_argument("--force-verification", action="store_true")
    parser.add_argument("--force-episodes", action="store_true")
    return parser.parse_args()


def evaluate_expert(expert, seeds: list[int], max_steps: int) -> dict:
    returns = []
    lengths = []
    for index, seed in enumerate(seeds):
        env = make_expert_env()
        observation, _ = env.reset(seed=seed)
        episode_return = 0.0
        length = 0
        try:
            while length < max_steps:
                action = expert.action(np.asarray(observation))
                observation, reward, terminated, truncated, _ = env.step(action)
                episode_return += float(reward)
                length += 1
                if terminated or truncated:
                    break
        finally:
            env.close()
        returns.append(episode_return)
        lengths.append(length)
        print(
            f"verify {index + 1}/{len(seeds)} seed={seed} "
            f"return={episode_return:.1f} length={length}",
            flush=True,
        )
    values = np.asarray(returns)
    return {
        "seeds": seeds,
        "returns": returns,
        "lengths": lengths,
        "mean_return": float(values.mean()),
        "median_return": float(np.median(values)),
        "std_return": float(values.std()),
        "min_return": float(values.min()),
        "max_return": float(values.max()),
        "win_rate": float((values > 0).mean()),
        "max_steps": max_steps,
    }


def main() -> None:
    args = parse_args()
    if args.verification_episodes < 1:
        raise ValueError("verification-episodes must be positive")
    spec = CollectionSpec(
        train_episodes=args.train_episodes,
        validation_episodes=args.validation_episodes,
        test_episodes=args.test_episodes,
        train_seed_base=args.train_seed_base,
        validation_seed_base=args.validation_seed_base,
        test_seed_base=args.test_seed_base,
        policy_mode=args.policy_mode,
        max_steps=args.max_steps,
    )
    verification_seeds = list(
        range(
            args.verification_seed_base,
            args.verification_seed_base + args.verification_episodes,
        )
    )
    collection_seeds = {seed for seeds in spec.seeds().values() for seed in seeds}
    if collection_seeds.intersection(verification_seeds):
        raise ValueError("Verification and collection seeds must be disjoint")

    checkpoint = download_expert(args.checkpoint)
    expert = CleanRLPongExpert(checkpoint, mode=args.policy_mode)
    args.output.mkdir(parents=True, exist_ok=True)
    args.report.mkdir(parents=True, exist_ok=True)
    run_config = {
        "collection": asdict(spec),
        "verification_seeds": verification_seeds,
        "minimum_expert_return": args.minimum_expert_return,
        "minimum_win_rate": args.minimum_win_rate,
    }
    config_path = args.output / "collection_config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != run_config:
        raise RuntimeError("Existing v2 output has a different collection configuration")
    config_path.write_text(json.dumps(run_config, indent=2) + "\n")

    verification_path = args.report / "expert_verification.json"
    if verification_path.exists() and not args.force_verification:
        verification = json.loads(verification_path.read_text())
        if verification.get("seeds") != verification_seeds:
            raise RuntimeError("Existing expert verification used different seeds")
        print("Using completed expert verification", flush=True)
    else:
        verification = evaluate_expert(expert, verification_seeds, args.max_steps)
        verification.update(
            {
                "policy_mode": args.policy_mode,
                "minimum_mean_return": args.minimum_expert_return,
                "minimum_win_rate": args.minimum_win_rate,
            }
        )
        verification["passed"] = bool(
            verification["mean_return"] >= args.minimum_expert_return
            and verification["win_rate"] >= args.minimum_win_rate
        )
        verification_path.write_text(json.dumps(verification, indent=2) + "\n")
    if not verification.get("passed", False):
        raise SystemExit(f"Expert verification failed: {verification}")

    manifest = episode_manifest(spec)
    for ordinal, item in enumerate(manifest, start=1):
        path = episode_path(args.output, item["episode_id"])
        if path.exists() and not args.force_episodes:
            _, metadata = load_episode(path)
            if any(metadata[key] != item[key] for key in ("episode_id", "split", "seed")):
                raise RuntimeError(f"Existing episode metadata mismatch: {path}")
            print(f"episode {ordinal}/{len(manifest)} already complete", flush=True)
            continue
        arrays = collect_episode(expert, item["seed"], args.max_steps)
        episode_metadata = {
            **item,
            "policy_mode": args.policy_mode,
            "length": len(arrays["actions"]),
            "return": float(arrays["rewards"].sum()),
        }
        save_episode(path, arrays, episode_metadata)
        print(
            f"episode {ordinal}/{len(manifest)} split={item['split']} "
            f"seed={item['seed']} return={episode_metadata['return']:.1f} "
            f"length={episode_metadata['length']}",
            flush=True,
        )

    metadata = finalize_dataset(args.output, spec, verification)
    replay_audits = []
    if spec.policy_mode == "deterministic":
        for split in spec.seeds():
            item = next(item for item in manifest if item["split"] == split)
            audit = audit_episode_replay(
                expert,
                episode_path(args.output, item["episode_id"]),
                args.max_steps,
            )
            replay_audits.append(audit)
            print(
                f"replay audit split={split} seed={audit['seed']} "
                f"steps={audit['steps_checked']} exact_match=True",
                flush=True,
            )
    replay_path = args.report / "replay_audit.json"
    replay_path.write_text(json.dumps(replay_audits, indent=2) + "\n")
    eda = run_eda(args.output, args.report)
    stage_summary = {
        "stage": 2,
        "passed": True,
        "dataset": str(args.output),
        "num_episodes": metadata["num_episodes"],
        "num_steps": metadata["num_steps"],
        "split_returns": eda["split_episode_returns"],
        "expert_verification": verification,
        "replay_audits": replay_audits,
    }
    (args.report / "summary.json").write_text(json.dumps(stage_summary, indent=2) + "\n")
    print(
        f"Stage 2 complete: {metadata['num_episodes']} episodes / "
        f"{metadata['num_steps']} steps",
        flush=True,
    )


if __name__ == "__main__":
    main()
