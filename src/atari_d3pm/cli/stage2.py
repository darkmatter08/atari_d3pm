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
    audit_trajectory_uniqueness,
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
    parser.add_argument("--output", type=Path, default=Path("data/pong/v3"))
    parser.add_argument("--report", type=Path, default=Path("reports/pong_v3"))
    parser.add_argument(
        "--policy-mode",
        choices=["deterministic", "stochastic"],
        default="stochastic",
    )
    parser.add_argument("--train-episodes", type=int, default=100)
    parser.add_argument("--validation-episodes", type=int, default=10)
    parser.add_argument("--test-episodes", type=int, default=20)
    parser.add_argument("--train-seed-base", type=int, default=30_000)
    parser.add_argument("--validation-seed-base", type=int, default=40_000)
    parser.add_argument("--test-seed-base", type=int, default=50_000)
    parser.add_argument("--expert-seed-offset", type=int, default=1_000_000)
    parser.add_argument("--verification-episodes", type=int, default=20)
    parser.add_argument("--verification-seed-base", type=int, default=20_000)
    parser.add_argument("--minimum-expert-return", type=float, default=18.0)
    parser.add_argument("--minimum-win-rate", type=float, default=0.9)
    parser.add_argument("--minimum-unique-fraction", type=float, default=0.95)
    parser.add_argument("--max-steps", type=int, default=27_000)
    parser.add_argument("--force-verification", action="store_true")
    parser.add_argument("--force-episodes", action="store_true")
    return parser.parse_args()


def evaluate_expert(
    expert, seeds: list[int], max_steps: int, expert_seed_offset: int = 1_000_000
) -> dict:
    returns = []
    lengths = []
    expert_seeds = []
    for index, seed in enumerate(seeds):
        expert_seed = seed + expert_seed_offset
        expert.reset(expert_seed)
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
        expert_seeds.append(expert_seed)
        print(
            f"verify {index + 1}/{len(seeds)} seed={seed} "
            f"return={episode_return:.1f} length={length}",
            flush=True,
        )
    values = np.asarray(returns)
    return {
        "seeds": seeds,
        "expert_seeds": expert_seeds,
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
    if not 0 < args.minimum_unique_fraction <= 1:
        raise ValueError("minimum-unique-fraction must be in (0, 1]")
    spec = CollectionSpec(
        train_episodes=args.train_episodes,
        validation_episodes=args.validation_episodes,
        test_episodes=args.test_episodes,
        train_seed_base=args.train_seed_base,
        validation_seed_base=args.validation_seed_base,
        test_seed_base=args.test_seed_base,
        expert_seed_offset=args.expert_seed_offset,
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
        "minimum_unique_fraction": args.minimum_unique_fraction,
    }
    config_path = args.output / "collection_config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != run_config:
        raise RuntimeError("Existing output has a different collection configuration")
    config_path.write_text(json.dumps(run_config, indent=2) + "\n")

    verification_path = args.report / "expert_verification.json"
    if verification_path.exists() and not args.force_verification:
        verification = json.loads(verification_path.read_text())
        if verification.get("seeds") != verification_seeds:
            raise RuntimeError("Existing expert verification used different seeds")
        expected_expert_seeds = [
            seed + args.expert_seed_offset for seed in verification_seeds
        ]
        if verification.get("expert_seeds") != expected_expert_seeds:
            raise RuntimeError("Existing expert verification used different policy seeds")
        if verification.get("policy_mode") != args.policy_mode:
            raise RuntimeError("Existing expert verification used a different policy mode")
        print("Using completed expert verification", flush=True)
    else:
        verification = evaluate_expert(
            expert,
            verification_seeds,
            args.max_steps,
            expert_seed_offset=args.expert_seed_offset,
        )
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
            if any(
                metadata[key] != item[key]
                for key in ("episode_id", "split", "seed", "expert_seed")
            ):
                raise RuntimeError(f"Existing episode metadata mismatch: {path}")
            if metadata.get("policy_mode") != args.policy_mode:
                raise RuntimeError(f"Existing episode policy mode mismatch: {path}")
            print(f"episode {ordinal}/{len(manifest)} already complete", flush=True)
            continue
        arrays = collect_episode(
            expert,
            item["seed"],
            item["expert_seed"],
            args.max_steps,
        )
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
            f"expert_seed={audit['expert_seed']} "
            f"steps={audit['steps_checked']} exact_match=True",
            flush=True,
        )
    replay_path = args.report / "replay_audit.json"
    replay_path.write_text(json.dumps(replay_audits, indent=2) + "\n")
    uniqueness = audit_trajectory_uniqueness(args.output, spec)
    uniqueness["minimum_unique_fraction"] = args.minimum_unique_fraction
    uniqueness["passed_unique_fraction"] = all(
        result["unique_fraction"] >= args.minimum_unique_fraction
        for result in uniqueness["per_split"].values()
    )
    uniqueness["passed_zero_cross_split_overlap"] = all(
        overlap == 0
        for overlap in uniqueness["cross_split_exact_overlaps"].values()
    )
    uniqueness["passed"] = bool(
        uniqueness["passed_unique_fraction"]
        and uniqueness["passed_zero_cross_split_overlap"]
    )
    (args.report / "trajectory_uniqueness.json").write_text(
        json.dumps(uniqueness, indent=2) + "\n"
    )
    eda = run_eda(args.output, args.report)
    stage_summary = {
        "stage": 2,
        "passed": bool(verification["passed"] and uniqueness["passed"]),
        "dataset": str(args.output),
        "num_episodes": metadata["num_episodes"],
        "num_steps": metadata["num_steps"],
        "split_returns": eda["split_episode_returns"],
        "expert_verification": verification,
        "replay_audits": replay_audits,
        "trajectory_uniqueness": uniqueness,
    }
    (args.report / "summary.json").write_text(json.dumps(stage_summary, indent=2) + "\n")
    print(
        f"Stage 2 complete: {metadata['num_episodes']} episodes / "
        f"{metadata['num_steps']} steps",
        flush=True,
    )
    if not stage_summary["passed"]:
        raise SystemExit(f"Stage 2 diversity gates failed: {uniqueness}")


if __name__ == "__main__":
    main()
