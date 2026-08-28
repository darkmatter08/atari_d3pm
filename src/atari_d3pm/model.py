"""Vision-conditioned denoiser for Pong action chunks."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class PongVisionEncoder(nn.Module):
    """Encode four 84x84 grayscale frames while retaining spatial information."""

    def __init__(self, frame_stack: int = 4, d_model: int = 128) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(frame_stack, 16, kernel_size=8, stride=4),
            nn.SiLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.SiLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1),
            nn.SiLU(),
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        if frames.ndim != 4:
            raise ValueError(f"Expected frames [B, C, H, W], got {tuple(frames.shape)}")
        return self.network(frames.float().div(255.0))


class PongBehaviorCloner(nn.Module):
    """One-step non-diffusion behavioral-cloning baseline."""

    def __init__(
        self, num_actions: int = 6, frame_stack: int = 4, d_model: int = 128
    ) -> None:
        super().__init__()
        self.vision = PongVisionEncoder(frame_stack=frame_stack, d_model=d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_actions),
        )

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        return self.head(self.vision(frames))


class PongActionDenoiser(nn.Module):
    """Predict clean action tokens from noisy tokens and a frame stack."""

    def __init__(
        self,
        horizon: int,
        num_actions: int = 6,
        diffusion_steps: int = 20,
        frame_stack: int = 4,
        d_model: int = 128,
        n_layers: int = 3,
        n_heads: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.horizon = horizon
        self.num_actions = num_actions

        self.vision = PongVisionEncoder(frame_stack=frame_stack, d_model=d_model)
        self.action_embedding = nn.Embedding(num_actions, d_model)
        self.time_embedding = nn.Embedding(diffusion_steps + 1, d_model)
        self.position_embedding = nn.Parameter(torch.randn(1, horizon, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.final_norm = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, num_actions)

    def forward(
        self, noisy_actions: torch.Tensor, t: torch.Tensor, frames: torch.Tensor
    ) -> torch.Tensor:
        if noisy_actions.ndim != 2 or noisy_actions.shape[1] != self.horizon:
            raise ValueError(
                f"Expected noisy actions [B, {self.horizon}], got {tuple(noisy_actions.shape)}"
            )
        vision = self.vision(frames).unsqueeze(1)
        time = self.time_embedding(t).unsqueeze(1)
        tokens = self.action_embedding(noisy_actions) + self.position_embedding + vision + time
        tokens = self.transformer(tokens)
        logits = self.output(self.final_norm(tokens))
        return logits + 0.1 * F.one_hot(noisy_actions, self.num_actions).to(logits.dtype)
