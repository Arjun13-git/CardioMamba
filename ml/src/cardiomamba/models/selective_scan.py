"""Mamba-1 selective scan (S6 recurrence) in pure PyTorch.

Notation (per batch element, time step t, inner channel e, state index n):
    u_t ∈ R^E            input sequence (after in_proj, causal conv, SiLU)
    Δ_t ∈ R^E            input-dependent step size (softplus already applied, Δ > 0)
    A   ∈ R^{E×N}        state matrix, diagonal per channel, A = −exp(A_log) < 0
    B_t, C_t ∈ R^N       input-dependent input/output projections (shared across channels)
    D   ∈ R^E            skip connection

Discretisation (as in the reference Mamba-1 `selective_scan_ref`):
    Ā_t = exp(Δ_t ⊗ A)                 zero-order hold for A        [E, N]
    B̄_t u_t = (Δ_t ⊙ u_t) ⊗ B_t         Euler step for B             [E, N]

Recurrence and readout (state updated *before* the readout at the same step):
    h_t = Ā_t ⊙ h_{t−1} + B̄_t u_t,     h_{−1} = 0                   [E, N]
    y_t = h_t · C_t + D ⊙ u_t                                        [E]

Two implementations of the same recurrence are provided:
    * `selective_scan_reference` — explicit Python loop over t with autograd (correctness oracle);
    * `selective_scan_parallel`  — chunked associative scan (no Python loop over the sequence)
      with an exact custom autograd backward (the same scan run in reverse time).

Numerical policy: the scan ALWAYS runs in float32 with autocast disabled, whatever the dtype
of its inputs (bf16 autocast in the surrounding model is fine). Every factor Ā_t lies in
(0, 1], so products of Ā never overflow; there is no log-space division that could.

Layouts: the public functions take batch-major [B, L, E] / [B, L, N]. The reference scan
works on those directly; the parallel scan works time-major ([L, B, E, N], time on dim 0) so
that every time slice is one contiguous block (see `linear_recurrence`).
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor


def discretize(u: Tensor, delta: Tensor, A: Tensor, B: Tensor) -> tuple[Tensor, Tensor]:
    """Return (Ā, B̄u), both [B, L, E, N]."""
    delta_a = torch.exp(delta.unsqueeze(-1) * A)                  # [B,L,E,1]*[E,N] -> [B,L,E,N]
    delta_b_u = (delta * u).unsqueeze(-1) * B.unsqueeze(2)        # [B,L,E,1]*[B,L,1,N]
    return delta_a, delta_b_u


def selective_scan_reference(u: Tensor, delta: Tensor, A: Tensor, B: Tensor, C: Tensor,
                             D: Tensor) -> Tensor:
    """Sequential S6 scan: one explicit state update and readout per time step.

    Shapes: u, delta [B, L, E]; A [E, N]; B, C [B, L, N]; D [E]  ->  y [B, L, E].
    """
    delta_a, delta_b_u = discretize(u, delta, A, B)
    batch, length, d_inner = u.shape
    h = u.new_zeros(batch, d_inner, A.shape[1])                  # h_{-1} = 0, [B, E, N]
    ys = []
    for t in range(length):
        h = delta_a[:, t] * h + delta_b_u[:, t]                   # h_t = Ā_t h_{t-1} + B̄_t u_t
        ys.append(torch.einsum("ben,bn->be", h, C[:, t]))         # y_t = h_t · C_t
    y = torch.stack(ys, dim=1)                                    # [B, L, E]
    return y + u * D


MAX_CHUNK = 16


def chunk_size(length: int, max_chunk: int = MAX_CHUNK) -> int:
    """Largest divisor of `length` not exceeding `max_chunk` (L = 250 -> 10; prime L -> 1)."""
    return max(k for k in range(1, min(length, max_chunk) + 1) if length % k == 0)


def _hillis_steele(a: Tensor, b: Tensor) -> Tensor:
    """Inclusive scan of h_t = a_t h_{t−1} + b_t along dim 0 in ⌈log₂ L⌉ doubling steps.

    Uses the associative composition (a₁, b₁) ∘ (a₂, b₂) = (a₁a₂, a₂b₁ + b₂): after the step
    with offset k every position holds the composition of its last 2k inputs.
    """
    a, h = a.clone(), b.clone()
    length, offset = a.shape[0], 1
    while offset < length:
        n = length - offset
        h[offset:] = torch.addcmul(h[offset:], a[offset:], h[:n])        # uses old a
        if 2 * offset < length:                                         # a unused afterwards
            a[offset:] = a[offset:] * a[:n]
        offset *= 2
    return h


@torch.no_grad()
def linear_recurrence(a: Tensor, b: Tensor) -> Tensor:
    """Solve h_t = a_t · h_{t−1} + b_t with h_{−1} = 0 along dim 0 (time-major). Exact.

    Chunked two-level associative scan, L = n_chunks · K (K = chunk_size(L) ≤ 16):
      1. inside every chunk, K vectorised steps (over all chunks at once) give the local state
         H (chunk started from 0) and the running product P = a_{start}···a_t;
      2. the chunk summaries (P_end, H_end) are combined across chunks with a log-depth
         Hillis–Steele scan, giving the true state S_c at the end of every chunk;
      3. h_t = H_t + P_t · S_{c−1} (one broadcast pass; S_{−1} = 0).
    There is no Python loop over the sequence length. Memory traffic is a few passes over the
    [L, ...] tensor, unlike a plain log-depth scan which needs ~log₂ L passes.
    No autograd graph is built (see `_LinearRecurrence` for the backward).
    """
    length = a.shape[0]
    k = chunk_size(length)
    n_chunks, rest = length // k, a.shape[1:]
    a = a.reshape(n_chunks, k, *rest)                              # [nC, K, ...]
    b = b.reshape(n_chunks, k, *rest)
    local = torch.empty_like(b)                                    # H
    prod = torch.empty_like(a)                                     # P
    local[:, 0], prod[:, 0] = b[:, 0], a[:, 0]
    for i in range(1, k):                                          # K <= 16 steps, not L
        torch.addcmul(b[:, i], a[:, i], local[:, i - 1], out=local[:, i])
        torch.mul(prod[:, i - 1], a[:, i], out=prod[:, i])
    chunk_end = _hillis_steele(prod[:, -1].contiguous(), local[:, -1].contiguous())  # [nC, ...]
    carry = torch.zeros_like(chunk_end)
    carry[1:] = chunk_end[:-1]                                     # state entering chunk c
    return local.addcmul_(prod, carry.unsqueeze(1)).reshape(length, *rest)


class _LinearRecurrence(torch.autograd.Function):
    """h = scan(a, b) along dim 0 (time-major) with an exact reverse-scan backward.

    Backward of h_t = a_t h_{t−1} + b_t with upstream gradient ∂L/∂h_t = g̃_t:
        g_t = g̃_t + a_{t+1} g_{t+1}      (reverse-time recurrence, g_L = 0)
        ∂L/∂b_t = g_t,    ∂L/∂a_t = g_t · h_{t−1}
    Only a and h are saved (b is not needed).
    """

    @staticmethod
    def forward(ctx, a: Tensor, b: Tensor) -> Tensor:
        h = linear_recurrence(a, b)
        ctx.save_for_backward(a, h)
        return h

    @staticmethod
    def backward(ctx, grad_h: Tensor) -> tuple[Tensor, Tensor]:
        a, h = ctx.saved_tensors                                   # time-major [L, ...]
        a_next = torch.zeros_like(a)
        a_next[:-1] = a[1:]                                        # a_{t+1}, zero past the end
        g = linear_recurrence(a_next.flip(0), grad_h.flip(0)).flip(0)
        h_prev = torch.zeros_like(h)
        h_prev[1:] = h[:-1]                                        # h_{t-1}, h_{-1} = 0
        return g * h_prev, g


def selective_scan_parallel(u: Tensor, delta: Tensor, A: Tensor, B: Tensor, C: Tensor,
                            D: Tensor) -> Tensor:
    """Parallel S6 scan; mathematically identical to `selective_scan_reference`.

    Same shapes as the reference ([B, L, E] in/out); internally time-major.
    """
    u_t = u.transpose(0, 1).contiguous()                           # [B,L,E] -> [L,B,E]
    delta_a, delta_b_u = discretize(u_t, delta.transpose(0, 1).contiguous(), A,
                                    B.transpose(0, 1).contiguous())  # [L, B, E, N] each
    h = _LinearRecurrence.apply(delta_a, delta_b_u)                # [L, B, E, N]
    y = torch.einsum("lben,lbn->lbe", h, C.transpose(0, 1))        # [L, B, E]
    return (y + u_t * D).transpose(0, 1)                           # [B, L, E]


_IMPLEMENTATIONS = {
    "reference": selective_scan_reference,
    "parallel": selective_scan_parallel,
}


def selective_scan(u: Tensor, delta: Tensor, A: Tensor, B: Tensor, C: Tensor, D: Tensor,
                   mode: Literal["parallel", "reference"] = "parallel") -> Tensor:
    """S6 selective scan in float32 (autocast disabled). Returns float32 y [B, L, E]."""
    impl = _IMPLEMENTATIONS[mode]
    with torch.autocast(device_type=u.device.type, enabled=False):
        return impl(u.float(), delta.float(), A.float(), B.float(), C.float(), D.float())
