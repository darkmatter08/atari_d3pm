from __future__ import annotations

import torch

from atari_d3pm.model import PongActionDenoiser, PongBehaviorCloner


def test_model_emits_action_logits_for_each_chunk_position():
    model = PongActionDenoiser(
        horizon=8, diffusion_steps=10, d_model=32, n_layers=1, n_heads=4
    )
    actions = torch.randint(0, 6, (2, 8))
    steps = torch.tensor([1, 10])
    frames = torch.randint(0, 256, (2, 4, 84, 84), dtype=torch.uint8)
    logits = model(actions, steps, frames)
    assert logits.shape == (2, 8, 6)
    assert torch.isfinite(logits).all()


def test_behavior_cloner_emits_one_action_distribution():
    model = PongBehaviorCloner(d_model=32)
    frames = torch.randint(0, 256, (2, 4, 84, 84), dtype=torch.uint8)
    logits = model(frames)
    assert logits.shape == (2, 6)
    assert torch.isfinite(logits).all()
