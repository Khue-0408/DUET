# path: src/models/domain/grl.py
from __future__ import annotations

import torch
from torch.autograd import Function


class _GRL(Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambd: float) -> torch.Tensor:
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.lambd * grad_output, None


class GradientReversalLayer(torch.nn.Module):
    """
    Gradient Reversal Layer for domain-adversarial training (Sec.3.1, Eq.(3)).
    """

    def __init__(self, lambd: float = 1.0) -> None:
        super().__init__()
        self.lambd = float(lambd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _GRL.apply(x, self.lambd)
