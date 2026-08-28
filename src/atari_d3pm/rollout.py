"""Online ALE/Pong evaluation for trained policies."""

from __future__ import annotations

import json
import time
from collections import deque
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from .data import preprocess_frame
from .training import choose_device, load_checkpoint


def make_pong_env():
    import ale_py
    import gymnasium as gym

    gym.register_envs(ale_py)
    env = gym.make(
        "ALE/Pong-v5",
        obs_type="rgb",
        frameskip=4,
        repeat_action_probability=0.0,
        full_action_space=False,
    )
    meanings = list(env.unwrapped.get_action_meanings())
    expected = ["NOOP", "FIRE", "RIGHT", "LEFT", "RIGHTFIRE", "LEFTFIRE"]
    if meanings != expected:
        env.close()
        raise RuntimeError(f"Unexpected Pong action mapping: {meanings}")
    return env


@torch.no_grad()
def _policy_actions(
    frames: torch.Tensor,
    policy_type: str,
    horizon: int,
    model,
    diffusion,
    device: torch.device,
    sample_seed: int,
) -> np.ndarray:
    frames = frames.to(device, non_blocking=True)
    use_amp = device.type == "cuda" and torch.cuda.is_bf16_supported()
    torch.manual_seed(sample_seed)
    with torch.autocast(
        device_type=device.type, dtype=torch.bfloat16, enabled=use_amp
    ):
        if policy_type == "bc":
            actions = model(frames).argmax(dim=-1)
        else:
            chunks = diffusion.sample(frames, horizon=horizon)
            actions = chunks[:, 0]
    return actions.cpu().numpy()


def _summary(returns: list[float], lengths: list[int], inference_seconds: float) -> dict:
    values = np.asarray(returns, dtype=np.float64)
    return {
        "episodes": len(returns),
        "returns": returns,
        "lengths": lengths,
        "mean_return": float(values.mean()),
        "median_return": float(np.median(values)),
        "std_return": float(values.std()),
        "min_return": float(values.min()),
        "max_return": float(values.max()),
        "win_rate": float((values > 0).mean()),
        "inference_seconds": inference_seconds,
        "inference_ms_per_environment_step": float(
            1000 * inference_seconds / max(sum(lengths), 1)
        ),
    }


def evaluate_random_policy(
    seeds: Sequence[int], max_steps: int = 27_000
) -> dict:
    returns = []
    lengths = []
    for seed in seeds:
        env = make_pong_env()
        _, _ = env.reset(seed=int(seed))
        rng = np.random.default_rng(seed)
        episode_return = 0.0
        length = 0
        while length < max_steps:
            _, reward, terminated, truncated, _ = env.step(int(rng.integers(0, 6)))
            episode_return += float(reward)
            length += 1
            if terminated or truncated:
                break
        env.close()
        returns.append(episode_return)
        lengths.append(length)
    summary = _summary(returns, lengths, inference_seconds=0.0)
    summary.update(
        {
            "policy_type": "random",
            "seeds": [int(seed) for seed in seeds],
            "max_steps": max_steps,
        }
    )
    return summary


def evaluate_checkpoint(
    checkpoint_path: str | Path,
    seeds: Sequence[int],
    device_name: str = "auto",
    max_steps: int = 27_000,
) -> dict:
    device = choose_device(device_name)
    config, model, diffusion, checkpoint = load_checkpoint(checkpoint_path, device)

    envs = [make_pong_env() for _ in seeds]
    stacks: list[deque[np.ndarray]] = []
    for env, seed in zip(envs, seeds):
        observation, _ = env.reset(seed=int(seed))
        frame = preprocess_frame(observation)
        stacks.append(deque([frame.copy() for _ in range(4)], maxlen=4))

    active = list(range(len(envs)))
    episode_returns = [0.0 for _ in envs]
    episode_lengths = [0 for _ in envs]
    inference_seconds = 0.0
    policy_step = 0
    while active:
        frame_batch = torch.from_numpy(
            np.stack([np.stack(stacks[index]) for index in active])
        )
        started = time.perf_counter()
        actions = _policy_actions(
            frame_batch,
            config.policy_type,
            config.horizon,
            model,
            diffusion,
            device,
            sample_seed=config.seed + policy_step,
        )
        if device.type == "cuda":
            torch.cuda.synchronize()
        inference_seconds += time.perf_counter() - started

        still_active = []
        for batch_index, env_index in enumerate(active):
            observation, reward, terminated, truncated, _ = envs[env_index].step(
                int(actions[batch_index])
            )
            episode_returns[env_index] += float(reward)
            episode_lengths[env_index] += 1
            done = terminated or truncated or episode_lengths[env_index] >= max_steps
            if not done:
                stacks[env_index].append(preprocess_frame(observation))
                still_active.append(env_index)
        active = still_active
        policy_step += 1

    for env in envs:
        env.close()
    summary = _summary(episode_returns, episode_lengths, inference_seconds)
    summary.update(
        {
            "checkpoint": str(checkpoint_path),
            "checkpoint_step": int(checkpoint["step"]),
            "policy_type": config.policy_type,
            "horizon": config.horizon,
            "device": str(device),
            "seeds": [int(seed) for seed in seeds],
            "max_steps": max_steps,
        }
    )
    return summary


def write_rollout_summary(path: str | Path, summary: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2) + "\n")
