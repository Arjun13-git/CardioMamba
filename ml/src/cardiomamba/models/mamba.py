"""Mamba-1 mixer (S6) and the bidirectional residual block used by CardioMamba."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from cardiomamba.models.config import CardioMambaConfig, ScanMode
from cardiomamba.models.norm import RMSNorm
from cardiomamba.models.selective_scan import selective_scan


class MambaMixer(nn.Module):
    """Mamba-1 selective state-space mixer (Gu & Dao, 2023), causal over time.

    [B, L, D] → in_proj → x, z [B, L, E]
      x → depthwise causal Conv1d(k=d_conv) → SiLU
        → x_proj → (Δ_raw [B, L, R], B [B, L, N], C [B, L, N])
        → Δ = softplus(dt_proj(Δ_raw)) [B, L, E]
        → selective scan (FP32) with A = −exp(A_log) [E, N], D [E] → y [B, L, E]
      y ⊙ SiLU(z) → out_proj → [B, L, D]
    """

    def __init__(self, d_model: int, d_state: int, d_conv: int, expand: int, dt_rank: int,
                 dt_min: float = 1e-3, dt_max: float = 1e-1, dt_init_floor: float = 1e-4,
                 scan_mode: ScanMode = "parallel") -> None:
        super().__init__()
        d_inner = expand * d_model
        self.d_inner, self.d_state, self.dt_rank = d_inner, d_state, dt_rank
        self.scan_mode = scan_mode

        self.in_proj = nn.Linear(d_model, 2 * d_inner, bias=False)
        # Depthwise conv; left padding d_conv-1 then truncation to L keeps it causal.
        self.conv1d = nn.Conv1d(d_inner, d_inner, kernel_size=d_conv, groups=d_inner,
                                padding=d_conv - 1, bias=True)
        self.x_proj = nn.Linear(d_inner, dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(dt_rank, d_inner, bias=True)
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

        # S4D-real initialisation: A[e, n] = -(n + 1), stored as A_log = log(n + 1).
        a_init = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_inner, 1)
        self.A_log = nn.Parameter(torch.log(a_init))              # [E, N]
        self.D = nn.Parameter(torch.ones(d_inner))                # [E]
        self.A_log._no_weight_decay = True                        # read by a later optimizer setup
        self.D._no_weight_decay = True

        # Δ initialisation (reference Mamba-1): dt_proj.weight ~ U(±R^-0.5) and bias chosen so
        # softplus(bias) ~ LogUniform[dt_min, dt_max] (bias = inverse softplus of dt).
        nn.init.uniform_(self.dt_proj.weight, -dt_rank**-0.5, dt_rank**-0.5)
        dt = torch.exp(torch.rand(d_inner) * (math.log(dt_max) - math.log(dt_min))
                       + math.log(dt_min)).clamp(min=dt_init_floor)
        with torch.no_grad():
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))

    def forward(self, hidden: Tensor) -> Tensor:
        length = hidden.shape[1]
        x, z = self.in_proj(hidden).chunk(2, dim=-1)              # [B, L, E] each
        x = self.conv1d(x.transpose(1, 2))                        # [B, E, L] -> [B, E, L+k-1]
        x = F.silu(x[..., :length].transpose(1, 2))               # causal crop -> [B, L, E]
        dt_raw, b, c = self.x_proj(x).split([self.dt_rank, self.d_state, self.d_state], dim=-1)
        delta = F.softplus(self.dt_proj(dt_raw).float())          # [B, L, E], FP32, > 0
        a = -torch.exp(self.A_log.float())                        # [E, N], strictly negative
        y = selective_scan(x, delta, a, b, c, self.D, mode=self.scan_mode)  # FP32 [B, L, E]
        return self.out_proj(y * F.silu(z))                       # [B, L, D]


class BiMambaBlock(nn.Module):
    """Pre-norm residual block with a forward and a time-reversed Mamba mixer.

        u = RMSNorm(x)
        m = Mixer_fwd(u) + flip_t(Mixer_rev(flip_t(u)))      (separate parameters)
        out = x + Dropout(m)

    The reverse mixer is causal in reversed time, i.e. anti-causal in original time; its
    output is flipped back so both terms are aligned position-by-position before the sum.
    Summation keeps the width at d_model (no concatenation / extra projection).
    """

    def __init__(self, config: CardioMambaConfig) -> None:
        super().__init__()
        self.norm = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.forward_mixer = self._mixer(config)
        self.reverse_mixer = self._mixer(config) if config.bidirectional else None
        self.dropout = nn.Dropout(config.dropout)

    @staticmethod
    def _mixer(config: CardioMambaConfig) -> MambaMixer:
        return MambaMixer(config.d_model, config.d_state, config.d_conv, config.expand,
                          config.dt_rank, config.dt_min, config.dt_max, config.dt_init_floor,
                          config.scan_mode)

    def mix(self, u: Tensor) -> Tensor:
        """Bidirectional mixing of an already-normalised sequence [B, L, D] -> [B, L, D]."""
        out = self.forward_mixer(u)                               # left -> right
        if self.reverse_mixer is not None:
            out = out + self.reverse_mixer(u.flip(1)).flip(1)     # right -> left, re-aligned
        return out

    def forward(self, x: Tensor) -> Tensor:
        return x + self.dropout(self.mix(self.norm(x)))           # [B, L, D]
