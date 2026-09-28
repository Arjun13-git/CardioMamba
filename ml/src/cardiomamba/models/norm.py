"""RMSNorm (Zhang & Sennrich, 2019), computed in float32."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class RMSNorm(nn.Module):
    """y = x / sqrt(mean(x², last dim) + eps) · weight.

    No mean-centring and no bias (unlike LayerNorm). The statistic is computed in float32 so
    the normalisation is stable under bf16 autocast; the output is float32.
    """

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        x = x.float()
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps) * self.weight
