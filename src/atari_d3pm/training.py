"""Training and validation for behavioral-cloning and D3PM policies."""

from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .data import PongActionChunkDataset
from .diffusion import D3PM
from .model import PongActionDenoiser, PongBehaviorCloner


PolicyType = Literal["bc", "d3pm"]


@dataclass(frozen=True)
class TrainConfig:
    policy_type: PolicyType
    horizon: int
    data_root: str = "data/pong/v1"
    output_dir: str = "runs/stage1"
    train_steps: int = 3000
    batch_size: int = 256
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    validation_every: int = 250
    diffusion_steps: int = 20
    d_model: int = 128
    n_layers: int = 3
    n_heads: int = 4
    num_workers: int = 4
    sample_stride: int = 1
    validation_max_batches: int | None = None
    checkpoint_stage: int = 1
    seed: int = 0
    device: str = "auto"

    def __post_init__(self) -> None:
        if self.horizon < 1:
            raise ValueError("horizon must be positive")
        if self.policy_type == "bc" and self.horizon != 1:
            raise ValueError("The Stage-1 behavioral-cloning baseline must use H=1")
        if self.train_steps < 1 or self.validation_every < 1:
            raise ValueError("train_steps and validation_every must be positive")
        if self.batch_size < 1 or self.num_workers < 0:
            raise ValueError("batch_size must be positive and num_workers non-negative")
        if self.sample_stride < 1:
            raise ValueError("sample_stride must be positive")
        if self.validation_max_batches is not None and self.validation_max_batches < 1:
            raise ValueError("validation_max_batches must be positive when set")


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
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_policy(config: TrainConfig, device: torch.device):
    if config.policy_type == "bc":
        model = PongBehaviorCloner(d_model=config.d_model).to(device)
        return model, None
    model = PongActionDenoiser(
        horizon=config.horizon,
        diffusion_steps=config.diffusion_steps,
        d_model=config.d_model,
        n_layers=config.n_layers,
        n_heads=config.n_heads,
    ).to(device)
    diffusion = D3PM(
        model,
        n_steps=config.diffusion_steps,
        num_classes=6,
        hybrid_loss_coeff=1e-3,
    ).to(device)
    return model, diffusion


def _loader(dataset, config: TrainConfig, shuffle: bool) -> DataLoader:
    generator = torch.Generator().manual_seed(config.seed)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=config.num_workers > 0,
        drop_last=shuffle,
        generator=generator,
    )


def _infinite_batches(loader: DataLoader):
    while True:
        yield from loader


def _move_batch(batch, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    frames, actions = batch
    return (
        frames.to(device, non_blocking=True),
        actions.to(device, non_blocking=True),
    )


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    diffusion: D3PM | None,
    loader: DataLoader,
    config: TrainConfig,
    device: torch.device,
    max_batches: int | None = None,
) -> dict[str, float]:
    model.eval()
    total_tokens = 0
    total_first = 0
    denoise_correct = 0
    denoise_first_correct = 0
    sample_correct = 0
    sample_first_correct = 0
    total_ce = 0.0

    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        frames, actions = _move_batch(batch, device)
        if config.policy_type == "bc":
            logits = model(frames)
            targets = actions[:, 0]
            total_ce += float(F.cross_entropy(logits, targets, reduction="sum").cpu())
            predictions = logits.argmax(dim=-1)
            denoise_first_correct += int((predictions == targets).sum().cpu())
            total_first += len(targets)
            continue

        assert diffusion is not None
        batch_size = actions.shape[0]
        t = torch.full(
            (batch_size,), diffusion.n_steps, device=device, dtype=torch.long
        )
        generator = torch.Generator(device=device).manual_seed(
            config.seed + batch_index
        )
        noise = torch.rand(
            (*actions.shape, diffusion.num_classes),
            device=device,
            generator=generator,
        )
        noisy = diffusion.q_sample(actions, t, noise=noise)
        logits = model(noisy, t, frames)
        total_ce += float(
            F.cross_entropy(logits.flatten(0, -2), actions.flatten(), reduction="sum").cpu()
        )
        denoised = logits.argmax(dim=-1)
        denoise_correct += int((denoised == actions).sum().cpu())
        denoise_first_correct += int((denoised[:, 0] == actions[:, 0]).sum().cpu())

        sample_generator = torch.Generator(device=device).manual_seed(
            config.seed + 10_000_000 + batch_index
        )
        sampled = diffusion.sample(
            frames,
            horizon=config.horizon,
            generator=sample_generator,
        )
        sample_correct += int((sampled == actions).sum().cpu())
        sample_first_correct += int((sampled[:, 0] == actions[:, 0]).sum().cpu())
        total_tokens += actions.numel()
        total_first += batch_size

    if config.policy_type == "bc":
        return {
            "ce": total_ce / total_first,
            "first_action_accuracy": denoise_first_correct / total_first,
            "evaluated_windows": total_first,
        }
    return {
        "ce_at_max_noise": total_ce / total_tokens,
        "denoise_token_accuracy": denoise_correct / total_tokens,
        "denoise_first_action_accuracy": denoise_first_correct / total_first,
        "sample_token_accuracy": sample_correct / total_tokens,
        "sample_first_action_accuracy": sample_first_correct / total_first,
        "evaluated_windows": total_first,
    }


def checkpoint_score(metrics: dict[str, float], policy_type: PolicyType) -> float:
    if policy_type == "bc":
        return metrics["first_action_accuracy"]
    return metrics["sample_first_action_accuracy"]


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    config: TrainConfig,
    step: int,
    metrics: dict[str, float],
) -> None:
    torch.save(
        {
            "format_version": 1,
            "stage": config.checkpoint_stage,
            "config": asdict(config),
            "step": step,
            "metrics": metrics,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        path,
    )


def train_policy(config: TrainConfig) -> dict:
    seed_everything(config.seed)
    device = choose_device(config.device)
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(asdict(config), indent=2) + "\n")
    metrics_path = output / "metrics.jsonl"
    metrics_path.unlink(missing_ok=True)

    train_data = PongActionChunkDataset(
        config.data_root,
        split="train",
        horizon=config.horizon,
        sample_stride=config.sample_stride,
    )
    validation_data = PongActionChunkDataset(
        config.data_root,
        split="validation",
        horizon=config.horizon,
        sample_stride=config.sample_stride,
    )
    train_loader = _loader(train_data, config, shuffle=True)
    validation_loader = _loader(validation_data, config, shuffle=False)
    batches = _infinite_batches(train_loader)
    model, diffusion = build_policy(config, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    use_amp = device.type == "cuda" and torch.cuda.is_bf16_supported()
    best_score = -float("inf")
    best_metrics: dict[str, float] | None = None
    history = []
    started = time.perf_counter()

    print(
        f"Training {config.policy_type} H={config.horizon} on {device}: "
        f"{len(train_data)} train / {len(validation_data)} validation windows",
        flush=True,
    )
    for step in range(1, config.train_steps + 1):
        model.train()
        frames, actions = _move_batch(next(batches), device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=use_amp,
        ):
            if config.policy_type == "bc":
                logits = model(frames)
                loss = F.cross_entropy(logits, actions[:, 0])
            else:
                assert diffusion is not None
                loss, _ = diffusion.loss(actions, frames)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        should_validate = (
            step == 1
            or step % config.validation_every == 0
            or step == config.train_steps
        )
        if should_validate:
            metrics = validate(
                model,
                diffusion,
                validation_loader,
                config,
                device,
                max_batches=config.validation_max_batches,
            )
            score = checkpoint_score(metrics, config.policy_type)
            row = {"step": step, "train_loss": float(loss.detach().cpu()), **metrics}
            history.append(row)
            with metrics_path.open("a") as stream:
                stream.write(json.dumps(row) + "\n")
            print(f"step={step} train_loss={row['train_loss']:.4f} val={metrics}", flush=True)
            save_checkpoint(output / "last.pt", model, optimizer, config, step, metrics)
            if score > best_score:
                best_score = score
                best_metrics = metrics
                save_checkpoint(output / "best.pt", model, optimizer, config, step, metrics)

    summary = {
        "policy_type": config.policy_type,
        "horizon": config.horizon,
        "device": str(device),
        "train_windows": len(train_data),
        "validation_windows": len(validation_data),
        "best_score": best_score,
        "best_metrics": best_metrics,
        "elapsed_seconds": time.perf_counter() - started,
        "history": history,
    }
    (output / "training_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def load_checkpoint(path: str | Path, device: torch.device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = TrainConfig(**checkpoint["config"])
    model, diffusion = build_policy(config, device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    if diffusion is not None:
        diffusion.eval()
    return config, model, diffusion, checkpoint


@torch.no_grad()
def evaluate_checkpoint_offline(
    path: str | Path,
    split: str = "test",
    device_name: str = "auto",
    max_batches: int | None = None,
) -> dict:
    """Evaluate a frozen checkpoint on one offline split."""
    device = choose_device(device_name)
    config, model, diffusion, checkpoint = load_checkpoint(path, device)
    dataset = PongActionChunkDataset(
        config.data_root,
        split=split,
        horizon=config.horizon,
        sample_stride=config.sample_stride,
    )
    loader = _loader(dataset, config, shuffle=False)
    metrics = validate(
        model,
        diffusion,
        loader,
        config,
        device,
        max_batches=max_batches,
    )
    return {
        "checkpoint": str(path),
        "checkpoint_step": int(checkpoint["step"]),
        "split": split,
        "split_windows": len(dataset),
        "metrics": metrics,
    }
