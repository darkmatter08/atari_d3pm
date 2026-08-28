from __future__ import annotations

import torch
from torch import nn

from atari_d3pm.diffusion import D3PM


class TinyDenoiser(nn.Module):
    def __init__(self, classes: int):
        super().__init__()
        self.embedding = nn.Embedding(classes, classes)

    def forward(self, x, t, condition):
        del t, condition
        return self.embedding(x)


def test_diffusion_loss_and_sampling_are_finite():
    diffusion = D3PM(TinyDenoiser(6), n_steps=4, num_classes=6)
    actions = torch.randint(0, 6, (3, 5))
    condition = torch.zeros(3, 4, 84, 84, dtype=torch.uint8)
    loss, metrics = diffusion.loss(actions, condition)
    assert torch.isfinite(loss)
    assert all(torch.isfinite(value) for value in metrics.values())
    samples = diffusion.sample(condition, horizon=5)
    assert samples.shape == (3, 5)
    assert samples.min() >= 0 and samples.max() < 6


def test_final_forward_distribution_is_nearly_uniform():
    diffusion = D3PM(TinyDenoiser(6), n_steps=20, num_classes=6)
    final = diffusion.q_mats[-1]
    expected = torch.full_like(final, 1 / 6)
    assert torch.allclose(final, expected, atol=2e-3)
