"""Stage 5A: diagnose BC trajectory drift, action aliases, and D3PM chains."""

from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from atari_d3pm.cli.stage4 import _check_unused_seeds, _load_stage3_runs
from atari_d3pm.data import PongActionChunkDataset, encode_actions, preprocess_frame
from atari_d3pm.expert import current_rgb_frame, make_expert_env
from atari_d3pm.rollout import evaluate_checkpoint, evaluate_checkpoint_collection_env
from atari_d3pm.training import choose_device, load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage3", type=Path, default=Path("runs/stage3"))
    parser.add_argument("--output", type=Path, default=Path("runs/stage5a"))
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed-base", type=int, default=75_000)
    parser.add_argument("--max-steps", type=int, default=27_000)
    parser.add_argument("--reverse-max-batches", type=int, default=4)
    parser.add_argument("--trace-after-divergence", type=int, default=32)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def pairwise_agreement(predictions: list[np.ndarray]) -> dict:
    """Raw and canonical pairwise agreement for aligned action predictions."""
    result = {}
    for left in range(len(predictions)):
        for right in range(left + 1, len(predictions)):
            raw = predictions[left] == predictions[right]
            left_canonical = encode_actions(predictions[left], "canonical4")
            right_canonical = encode_actions(predictions[right], "canonical4")
            result[f"{left}__{right}"] = {
                "raw": float(raw.mean()),
                "canonical4": float((left_canonical == right_canonical).mean()),
                "alias_only": float((~raw & (left_canonical == right_canonical)).mean()),
            }
    return result


@torch.no_grad()
def offline_bc_alias_diagnostic(
    checkpoints: list[Path], data_root: Path, device_name: str
) -> dict:
    device = choose_device(device_name)
    dataset = PongActionChunkDataset(data_root, split="test", horizon=1)
    loader = DataLoader(dataset, batch_size=1024, shuffle=False, num_workers=4)
    loaded = [load_checkpoint(path, device) for path in checkpoints]
    prediction_parts: list[list[np.ndarray]] = [[] for _ in checkpoints]
    target_parts = []
    for frames, actions in loader:
        frames = frames.to(device, non_blocking=True)
        target_parts.append(actions[:, 0].numpy())
        for index, (config, model, _, _) in enumerate(loaded):
            if config.policy_type != "bc" or config.action_vocabulary != "raw6":
                raise RuntimeError("BC alias diagnostic requires raw6 one-step BC")
            prediction_parts[index].append(model(frames).argmax(dim=-1).cpu().numpy())
    targets = np.concatenate(target_parts)
    predictions = [np.concatenate(parts) for parts in prediction_parts]
    target_canonical = encode_actions(targets, "canonical4")
    policies = []
    for path, values in zip(checkpoints, predictions):
        confusion = np.zeros((6, 6), dtype=np.int64)
        np.add.at(confusion, (targets, values), 1)
        policies.append(
            {
                "checkpoint": str(path),
                "raw_accuracy": float((values == targets).mean()),
                "canonical4_accuracy": float(
                    (encode_actions(values, "canonical4") == target_canonical).mean()
                ),
                "predicted_action_counts": np.bincount(values, minlength=6).tolist(),
                "confusion_target_rows_prediction_columns": confusion.tolist(),
            }
        )
    return {
        "windows": len(targets),
        "target_action_counts": np.bincount(targets, minlength=6).tolist(),
        "policies": policies,
        "pairwise_prediction_agreement": pairwise_agreement(predictions),
    }


@torch.no_grad()
def trace_bc_divergence(
    checkpoints: list[Path],
    seeds: list[int],
    device_name: str,
    max_steps: int,
    after_divergence: int,
) -> dict:
    """Run BC policies in lockstep and retain a compact first-divergence trace."""
    device = choose_device(device_name)
    loaded = [load_checkpoint(path, device) for path in checkpoints]
    if any(config.policy_type != "bc" for config, _, _, _ in loaded):
        raise RuntimeError("Divergence tracing requires BC checkpoints")
    traces = []
    for seed in seeds:
        envs = [make_expert_env() for _ in loaded]
        stacks = []
        for env in envs:
            env.reset(seed=int(seed))
            frame = preprocess_frame(current_rgb_frame(env))
            stacks.append(deque([frame.copy() for _ in range(4)], maxlen=4))
        first_raw = None
        first_canonical = None
        first_observation = None
        records = []
        rewards = [0.0] * len(envs)
        action_counts = [np.zeros(6, dtype=np.int64) for _ in envs]
        try:
            for step in range(max_steps):
                frame_arrays = [np.stack(stack) for stack in stacks]
                reference = frame_arrays[0].astype(np.int16)
                frame_mae = [
                    float(np.abs(values.astype(np.int16) - reference).mean())
                    for values in frame_arrays
                ]
                if first_observation is None and any(value > 0 for value in frame_mae[1:]):
                    first_observation = step

                actions = []
                probabilities = []
                for frames, (config, model, _, _) in zip(frame_arrays, loaded):
                    logits = model(torch.from_numpy(frames[None]).to(device))
                    probs = torch.softmax(logits.float(), dim=-1)[0]
                    actions.append(int(probs.argmax().cpu()))
                    probabilities.append([round(float(value), 5) for value in probs.cpu()])
                canonical = encode_actions(np.asarray(actions), "canonical4").tolist()
                if first_raw is None and len(set(actions)) > 1:
                    first_raw = step
                if first_canonical is None and len(set(canonical)) > 1:
                    first_canonical = step
                if first_raw is not None and step < first_raw + after_divergence:
                    records.append(
                        {
                            "step": step,
                            "actions": actions,
                            "canonical4_actions": canonical,
                            "frame_mae_from_policy0": frame_mae,
                            "action_probabilities": probabilities,
                        }
                    )

                done = False
                for index, (env, action) in enumerate(zip(envs, actions)):
                    _, reward, terminated, truncated, _ = env.step(action)
                    rewards[index] += float(reward)
                    action_counts[index][action] += 1
                    done = done or terminated or truncated
                    if not (terminated or truncated):
                        stacks[index].append(preprocess_frame(current_rgb_frame(env)))
                if done:
                    break
        finally:
            for env in envs:
                env.close()
        traces.append(
            {
                "seed": seed,
                "steps_until_first_policy_finished": step + 1,
                "partial_returns": rewards,
                "first_raw_action_divergence": first_raw,
                "first_canonical_action_divergence": first_canonical,
                "first_observation_divergence": first_observation,
                "action_counts": [values.tolist() for values in action_counts],
                "first_divergence_trace": records,
            }
        )
    return {"checkpoint_order": [str(path) for path in checkpoints], "traces": traces}


@torch.no_grad()
def reverse_chain_diagnostic(
    checkpoint: Path, device_name: str, max_batches: int
) -> dict:
    device = choose_device(device_name)
    config, model, diffusion, _ = load_checkpoint(checkpoint, device)
    if config.policy_type != "d3pm" or diffusion is None:
        raise RuntimeError("Reverse-chain diagnostic requires D3PM")
    dataset = PongActionChunkDataset(
        config.data_root,
        split="validation",
        horizon=config.horizon,
        sample_stride=config.sample_stride,
        action_vocabulary=config.action_vocabulary,
    )
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)
    totals = {
        step: {"tokens": 0, "first": 0, "x_accuracy": 0.0, "x0_accuracy": 0.0,
               "x0_first_accuracy": 0.0, "entropy": 0.0}
        for step in range(diffusion.n_steps, 0, -1)
    }
    final_correct = 0
    final_first_correct = 0
    final_tokens = 0
    final_windows = 0
    generator = torch.Generator(device=device).manual_seed(config.seed + 50_000_000)
    for batch_index, (frames, actions) in enumerate(loader):
        if batch_index >= max_batches:
            break
        frames = frames.to(device)
        actions = actions.to(device)
        batch = len(actions)
        x = torch.randint(
            diffusion.num_classes,
            actions.shape,
            device=device,
            generator=generator,
        )
        for step in range(diffusion.n_steps, 0, -1):
            t = torch.full((batch,), step, device=device, dtype=torch.long)
            logits = model(x, t, frames)
            predictions = logits.argmax(dim=-1)
            probs = torch.softmax(logits.float(), dim=-1)
            values = totals[step]
            values["tokens"] += actions.numel()
            values["first"] += batch
            values["x_accuracy"] += float((x == actions).sum().cpu())
            values["x0_accuracy"] += float((predictions == actions).sum().cpu())
            values["x0_first_accuracy"] += float(
                (predictions[:, 0] == actions[:, 0]).sum().cpu()
            )
            values["entropy"] += float(
                (-(probs * torch.log(probs.clamp_min(1e-8))).sum(dim=-1)).sum().cpu()
            )
            posterior = diffusion.q_posterior_logits(logits, x, t)
            noise = torch.rand(
                posterior.shape, device=device, dtype=posterior.dtype, generator=generator
            ).clamp(diffusion.eps, 1.0)
            gumbel = -torch.log(-torch.log(noise))
            x = torch.argmax(
                posterior + gumbel * (t != 1).reshape(batch, 1, 1), dim=-1
            )
        final_correct += int((x == actions).sum().cpu())
        final_first_correct += int((x[:, 0] == actions[:, 0]).sum().cpu())
        final_tokens += actions.numel()
        final_windows += batch

    chain = []
    for step in range(diffusion.n_steps, 0, -1):
        values = totals[step]
        chain.append(
            {
                "reverse_t": step,
                "xt_token_accuracy": values["x_accuracy"] / values["tokens"],
                "predicted_x0_token_accuracy": values["x0_accuracy"] / values["tokens"],
                "predicted_x0_first_accuracy": (
                    values["x0_first_accuracy"] / values["first"]
                ),
                "predicted_x0_entropy": values["entropy"] / values["tokens"],
            }
        )
    return {
        "checkpoint": str(checkpoint),
        "horizon": config.horizon,
        "training_seed": config.seed,
        "evaluated_windows": final_windows,
        "final_sample_token_accuracy": final_correct / final_tokens,
        "final_sample_first_accuracy": final_first_correct / final_windows,
        "chain": chain,
    }


def main() -> None:
    args = parse_args()
    if args.episodes < 1 or args.reverse_max_batches < 1:
        raise ValueError("episodes and reverse-max-batches must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    seeds = list(range(args.seed_base, args.seed_base + args.episodes))
    stage3_summary, runs = _load_stage3_runs(args.stage3, include=None)
    _check_unused_seeds(stage3_summary, seeds)
    bc_runs = sorted(
        [run for run in runs if run["policy_type"] == "bc"],
        key=lambda item: item["training_seed"],
    )
    d3pm_runs = [run for run in runs if run["policy_type"] == "d3pm"]
    bc_checkpoints = [run["checkpoint"] for run in bc_runs]

    config = {
        "stage": "5A",
        "stage3": str(args.stage3),
        "diagnostic_seeds": seeds,
        "max_steps": args.max_steps,
        "reverse_max_batches": args.reverse_max_batches,
        "trace_after_divergence": args.trace_after_divergence,
        "device": args.device,
    }
    config_path = args.output / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise RuntimeError(f"{args.output} has a different Stage 5A configuration")
    config_path.write_text(json.dumps(config, indent=2) + "\n")

    alias_path = args.output / "bc_offline_aliases.json"
    if args.force or not alias_path.exists():
        alias = offline_bc_alias_diagnostic(
            bc_checkpoints, Path(stage3_summary["data"]), args.device
        )
        alias_path.write_text(json.dumps(alias, indent=2) + "\n")
    else:
        alias = json.loads(alias_path.read_text())

    trace_path = args.output / "bc_divergence_traces.json"
    if args.force or not trace_path.exists():
        traces = trace_bc_divergence(
            bc_checkpoints, seeds, args.device, args.max_steps, args.trace_after_divergence
        )
        trace_path.write_text(json.dumps(traces, indent=2) + "\n")
    else:
        traces = json.loads(trace_path.read_text())

    environment_results = []
    for run in bc_runs:
        for protocol, evaluator in (
            ("stage4", evaluate_checkpoint),
            ("stage2_collection", evaluate_checkpoint_collection_env),
        ):
            path = args.output / f"{run['name']}_{protocol}.json"
            if args.force or not path.exists():
                result = evaluator(
                    run["checkpoint"], seeds, device_name=args.device,
                    max_steps=args.max_steps
                )
                path.write_text(json.dumps(result, indent=2) + "\n")
            else:
                result = json.loads(path.read_text())
            environment_results.append(
                {"name": run["name"], "training_seed": run["training_seed"],
                 "protocol": protocol, "online": result}
            )

    reverse_results = []
    for run in d3pm_runs:
        path = args.output / f"reverse_{run['name']}.json"
        if args.force or not path.exists():
            result = reverse_chain_diagnostic(
                run["checkpoint"], args.device, args.reverse_max_batches
            )
            path.write_text(json.dumps(result, indent=2) + "\n")
        else:
            result = json.loads(path.read_text())
        reverse_results.append(result)

    trace_values = traces["traces"]
    alias_gains = [
        item["canonical4_accuracy"] - item["raw_accuracy"] for item in alias["policies"]
    ]
    recommendations = {
        "test_canonical4": bool(np.mean(alias_gains) > 0.01),
        "collect_dagger_recovery_data": any(
            item["first_observation_divergence"] is not None for item in trace_values
        ),
        "run_chunk_bc_control": True,
        "selection_rule": (
            "Use fixed seeds 80000-80019 for model-family selection; evaluate the "
            "selected family once on untouched seeds 90000-90099."
        ),
    }
    summary = {
        "stage": "5A",
        "passed": True,
        "diagnostic_seeds": seeds,
        "offline_aliases": alias,
        "bc_divergence": traces,
        "environment_protocol_results": environment_results,
        "reverse_chain_results": reverse_results,
        "recommendations": recommendations,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Stage 5A complete: {args.output / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
