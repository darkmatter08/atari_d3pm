"""Stage-0 fixed-batch overfit checks for H=1 and H=16."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from atari_d3pm.data import PongActionChunkDataset
from atari_d3pm.diffusion import D3PM
from atari_d3pm.model import PongActionDenoiser


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/pong/v1"))
    parser.add_argument("--output", type=Path, default=Path("reports/stage0"))
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 16])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--target-accuracy", type=float, default=0.95)
    parser.add_argument("--diffusion-steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--log-every", type=int, default=50)
    return parser.parse_args()


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def fixed_batch(
    dataset: PongActionChunkDataset, batch_size: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    first_episode_start = dataset.indices[0][1]
    first_episode_items = [
        item for item, (_, episode_start) in enumerate(dataset.indices)
        if episode_start == first_episode_start
    ]
    if len(first_episode_items) < batch_size:
        raise ValueError(
            f"First training episode has {len(first_episode_items)} windows, "
            f"fewer than batch size {batch_size}"
        )
    selected = np.linspace(0, len(first_episode_items) - 1, batch_size, dtype=int)
    item_ids = [first_episode_items[index] for index in selected]
    frames, actions = zip(*(dataset[index] for index in item_ids))
    return torch.stack(frames).to(device), torch.stack(actions).to(device)


@torch.no_grad()
def evaluate_fixed_noise(
    diffusion: D3PM,
    frames: torch.Tensor,
    actions: torch.Tensor,
    seed: int,
) -> dict[str, float]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.rand(
        (*actions.shape, diffusion.num_classes), generator=generator, device="cpu"
    ).to(actions.device)
    t = torch.full(
        (actions.shape[0],), diffusion.n_steps, device=actions.device, dtype=torch.long
    )
    noisy = diffusion.q_sample(actions, t, noise=noise)
    logits = diffusion.x0_model(noisy, t, frames)
    predictions = logits.argmax(dim=-1)
    return {
        "token_accuracy": float((predictions == actions).float().mean().cpu()),
        "first_action_accuracy": float((predictions[:, 0] == actions[:, 0]).float().mean().cpu()),
    }


def overfit_horizon(args: argparse.Namespace, horizon: int, device: torch.device) -> dict:
    seed_everything(args.seed + horizon)
    dataset = PongActionChunkDataset(args.data, split="train", horizon=horizon)
    frames, actions = fixed_batch(dataset, args.batch_size, device)
    model = PongActionDenoiser(
        horizon=horizon,
        diffusion_steps=args.diffusion_steps,
        d_model=64,
        n_layers=2,
        n_heads=4,
    ).to(device)
    diffusion = D3PM(
        model,
        n_steps=args.diffusion_steps,
        num_classes=6,
        hybrid_loss_coeff=1e-3,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.0)
    started = time.perf_counter()
    history = []
    best = {"token_accuracy": 0.0, "first_action_accuracy": 0.0}
    fixed_t = torch.full(
        (actions.shape[0],), diffusion.n_steps, device=device, dtype=torch.long
    )
    fixed_generator = torch.Generator(device="cpu").manual_seed(args.seed)
    fixed_noise = torch.rand(
        (*actions.shape, diffusion.num_classes), generator=fixed_generator, device="cpu"
    ).to(device)
    fixed_noisy_actions = diffusion.q_sample(actions, fixed_t, noise=fixed_noise)

    for step in range(1, args.steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss, metrics = diffusion.denoising_loss(
            actions, fixed_noisy_actions, fixed_t, frames
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step == 1 or step % args.log_every == 0 or step == args.steps:
            model.eval()
            fixed_metrics = evaluate_fixed_noise(diffusion, frames, actions, args.seed)
            best = max(best, fixed_metrics, key=lambda item: item["token_accuracy"])
            row = {
                "step": step,
                "loss": float(metrics["loss"].cpu()),
                "ce": float(metrics["ce"].cpu()),
                "vb": float(metrics["vb"].cpu()),
                **fixed_metrics,
            }
            history.append(row)
            print(
                f"H={horizon:>2} step={step:>4} loss={row['loss']:.4f} "
                f"token_acc={row['token_accuracy']:.3f} first_acc={row['first_action_accuracy']:.3f}",
                flush=True,
            )
            if (
                row["token_accuracy"] >= args.target_accuracy
                and row["first_action_accuracy"] >= args.target_accuracy
            ):
                break

    model.eval()
    torch.manual_seed(args.seed)
    sampled = diffusion.sample(frames[:4], horizon=horizon)
    torch.manual_seed(args.seed)
    repeated_sample = diffusion.sample(frames[:4], horizon=horizon)
    if not torch.equal(sampled, repeated_sample):
        raise AssertionError("Reverse diffusion is not reproducible under a fixed seed")
    if sampled.shape != (4, horizon) or sampled.min() < 0 or sampled.max() >= 6:
        raise AssertionError("Reverse diffusion returned invalid action chunks")

    result = {
        "horizon": horizon,
        "batch_size": args.batch_size,
        "steps_run": history[-1]["step"],
        "target_accuracy": args.target_accuracy,
        "passed": bool(
            history[-1]["token_accuracy"] >= args.target_accuracy
            and history[-1]["first_action_accuracy"] >= args.target_accuracy
        ),
        "elapsed_seconds": time.perf_counter() - started,
        "final": history[-1],
        "history": history,
        "sample_shape": list(sampled.shape),
        "sample_min": int(sampled.min().cpu()),
        "sample_max": int(sampled.max().cpu()),
        "fixed_seed_sampling_reproducible": True,
    }
    checkpoint = {
        "model": model.state_dict(),
        "horizon": horizon,
        "num_actions": 6,
        "diffusion_steps": args.diffusion_steps,
        "stage": 0,
        "result": result,
    }
    torch.save(checkpoint, args.output / f"horizon_{horizon}.pt")
    return result


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)
    print(f"Stage 0 device: {device}", flush=True)
    results = [overfit_horizon(args, horizon, device) for horizon in args.horizons]
    summary = {
        "stage": 0,
        "data": str(args.data),
        "device": str(device),
        "seed": args.seed,
        "passed": all(result["passed"] for result in results),
        "results": results,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if not summary["passed"]:
        raise SystemExit("Stage 0 failed its fixed-batch accuracy threshold")
    print(f"Stage 0 passed; summary written to {args.output / 'summary.json'}")


if __name__ == "__main__":
    main()
