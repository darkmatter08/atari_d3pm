"""Categorical diffusion adapted from the repository's known-good D3PM."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class D3PM(nn.Module):
    """Uniform categorical D3PM with an x0-predicting denoiser.

    The denoiser receives ``(x_t, t, condition)`` and returns logits over clean
    categories with shape ``x_t.shape + (num_classes,)``.
    """

    def __init__(
        self,
        x0_model: nn.Module,
        n_steps: int,
        num_classes: int,
        hybrid_loss_coeff: float = 1e-3,
    ) -> None:
        super().__init__()
        if n_steps < 2:
            raise ValueError("n_steps must be at least 2")
        if num_classes < 2:
            raise ValueError("num_classes must be at least 2")

        self.x0_model = x0_model
        self.n_steps = n_steps
        self.num_classes = num_classes
        self.hybrid_loss_coeff = hybrid_loss_coeff
        self.eps = 1e-6

        steps = torch.arange(n_steps + 1, dtype=torch.float64) / n_steps
        alpha_bar = torch.cos((steps + 0.008) / 1.008 * torch.pi / 2)
        betas = torch.minimum(
            1 - alpha_bar[1:] / alpha_bar[:-1],
            torch.full_like(alpha_bar[1:], 0.999),
        )

        one_step = []
        for beta in betas:
            matrix = torch.full(
                (num_classes, num_classes), beta / num_classes, dtype=torch.float64
            )
            matrix.diagonal().fill_(1 - (num_classes - 1) * beta / num_classes)
            one_step.append(matrix)
        one_step_mats = torch.stack(one_step).float()

        cumulative = []
        current = one_step_mats[0]
        cumulative.append(current)
        for index in range(1, n_steps):
            current = current @ one_step_mats[index]
            cumulative.append(current)

        self.register_buffer("q_one_step_transposed", one_step_mats.transpose(1, 2))
        self.register_buffer("q_mats", torch.stack(cumulative))

    @staticmethod
    def _broadcast_t(t: torch.Tensor, ndim: int) -> torch.Tensor:
        return t.reshape(t.shape[0], *([1] * (ndim - 1)))

    def _at(self, matrices: torch.Tensor, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        step = self._broadcast_t(t, x.ndim)
        return matrices[step - 1, x, :]

    def q_sample(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Sample q(x_t | x_0) using the closed-form cumulative transition."""
        logits = torch.log(self._at(self.q_mats, t, x0) + self.eps)
        if noise is None:
            noise = torch.rand_like(logits)
        noise = noise.clamp(self.eps, 1.0)
        gumbel = -torch.log(-torch.log(noise))
        return torch.argmax(logits + gumbel, dim=-1)

    def q_posterior_logits(
        self, x0_or_logits: torch.Tensor, xt: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """Compute q(x_{t-1} | x_t, x_0), corresponding to D3PM Eq. 3."""
        if not x0_or_logits.is_floating_point():
            x0_logits = torch.log(
                F.one_hot(x0_or_logits.long(), self.num_classes).float() + self.eps
            )
        else:
            x0_logits = x0_or_logits

        expected = xt.shape + (self.num_classes,)
        if x0_logits.shape != expected:
            raise ValueError(f"Expected x0 logits {expected}, got {x0_logits.shape}")

        fact1 = self._at(self.q_one_step_transposed, t, xt)
        x0_probs = torch.softmax(x0_logits, dim=-1)
        previous_index = (t - 2).clamp(min=0)
        previous_q = self.q_mats[previous_index].to(x0_probs.dtype)
        fact2 = torch.einsum("b...c,bcd->b...d", x0_probs, previous_q)
        posterior = torch.log(fact1 + self.eps) + torch.log(fact2 + self.eps)
        is_first = self._broadcast_t(t, xt.ndim + 1) == 1
        return torch.where(is_first, x0_logits, posterior)

    @staticmethod
    def variational_bound(true_logits: torch.Tensor, pred_logits: torch.Tensor) -> torch.Tensor:
        true_logits = true_logits.flatten(0, -2)
        pred_logits = pred_logits.flatten(0, -2)
        true_probs = torch.softmax(true_logits, dim=-1)
        return (
            true_probs
            * (torch.log_softmax(true_logits, dim=-1) - torch.log_softmax(pred_logits, dim=-1))
        ).sum(dim=-1).mean()

    def denoising_loss(
        self,
        x0: torch.Tensor,
        xt: torch.Tensor,
        t: torch.Tensor,
        condition: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Evaluate the hybrid objective for an explicitly supplied corruption."""
        x0_logits = self.x0_model(xt, t, condition)
        true_posterior = self.q_posterior_logits(x0, xt, t)
        pred_posterior = self.q_posterior_logits(x0_logits, xt, t)
        vb = self.variational_bound(true_posterior, pred_posterior)
        ce = F.cross_entropy(x0_logits.flatten(0, -2), x0.flatten())
        total = ce + self.hybrid_loss_coeff * vb
        accuracy = (x0_logits.argmax(dim=-1) == x0).float().mean()
        return total, {"loss": total.detach(), "ce": ce.detach(), "vb": vb.detach(), "accuracy": accuracy.detach()}

    def loss(
        self, x0: torch.Tensor, condition: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        batch = x0.shape[0]
        t = torch.randint(1, self.n_steps + 1, (batch,), device=x0.device)
        xt = self.q_sample(x0, t)
        return self.denoising_loss(x0, xt, t, condition)

    def forward(self, x0: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        return self.loss(x0, condition)[0]

    @torch.no_grad()
    def p_sample(
        self,
        xt: torch.Tensor,
        t: torch.Tensor,
        condition: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x0_logits = self.x0_model(xt, t, condition)
        posterior = self.q_posterior_logits(x0_logits, xt, t)
        if noise is None:
            noise = torch.rand_like(posterior)
        noise = noise.clamp(self.eps, 1.0)
        gumbel = -torch.log(-torch.log(noise))
        not_first = self._broadcast_t(t != 1, xt.ndim + 1)
        return torch.argmax(posterior + gumbel * not_first, dim=-1)

    @torch.no_grad()
    def sample(self, condition: torch.Tensor, horizon: int) -> torch.Tensor:
        batch = condition.shape[0]
        x = torch.randint(
            self.num_classes, (batch, horizon), device=condition.device
        )
        for step in range(self.n_steps, 0, -1):
            t = torch.full((batch,), step, device=x.device, dtype=torch.long)
            x = self.p_sample(x, t, condition)
        return x
