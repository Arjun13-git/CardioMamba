"""Reference (sequential) vs parallel selective scan: forward and gradient equivalence.

Errors are normalised by the reference magnitude: rel = max|x - ref| / max|ref|. Observed
values are ~1e-7 (float32 rounding); the 1e-5 bound leaves margin for summation-order effects
while any real recurrence bug produces O(1) errors.
"""

import pytest
import torch
import torch.nn.functional as F

from cardiomamba.models.selective_scan import (
    _LinearRecurrence,
    chunk_size,
    linear_recurrence,
    selective_scan,
    selective_scan_parallel,
    selective_scan_reference,
)

REL_TOL_FP32 = 1e-5
SHAPES = [  # (batch, length, d_inner, d_state)
    (1, 1, 4, 2),
    (2, 7, 8, 4),
    (3, 64, 16, 8),
    (2, 64, 32, 16),
    (4, 250, 64, 16),
    (1, 250, 256, 16),   # the CardioMamba mixer size
]


def make_inputs(batch, length, d_inner, d_state, seed=0, dt_shift=-2.0):
    g = torch.Generator().manual_seed(seed)
    u = torch.randn(batch, length, d_inner, generator=g)
    delta = F.softplus(torch.randn(batch, length, d_inner, generator=g) + dt_shift)
    a_log = torch.log(torch.arange(1, d_state + 1).float()).repeat(d_inner, 1)
    A = -torch.exp(a_log + 0.1 * torch.randn(d_inner, d_state, generator=g))
    B = torch.randn(batch, length, d_state, generator=g)
    C = torch.randn(batch, length, d_state, generator=g)
    D = torch.randn(d_inner, generator=g)
    return [u, delta, A, B, C, D]


def rel_err(x: torch.Tensor, ref: torch.Tensor) -> float:
    return ((x.double() - ref.double()).abs().max() / ref.double().abs().max().clamp_min(1e-30)).item()


def grads(fn, inputs, weight):
    inputs = [t.detach().clone().requires_grad_() for t in inputs]
    y = fn(*inputs)
    return y, torch.autograd.grad((y * weight).sum(), inputs)


@pytest.mark.parametrize("shape", SHAPES)
def test_forward_and_gradients_match_reference_fp32(shape):
    inputs = make_inputs(*shape)
    weight = torch.randn(shape[0], shape[1], shape[2], generator=torch.Generator().manual_seed(9))
    y_ref, g_ref = grads(selective_scan_reference, inputs, weight)
    y_par, g_par = grads(selective_scan_parallel, inputs, weight)
    assert rel_err(y_par, y_ref) < REL_TOL_FP32
    for name, gp, gr in zip(["u", "delta", "A", "B", "C", "D"], g_par, g_ref, strict=True):
        assert rel_err(gp, gr) < REL_TOL_FP32, name


@pytest.mark.parametrize("shape", [(2, 7, 8, 4), (2, 64, 16, 16), (1, 250, 64, 16)])
def test_both_paths_close_to_float64_oracle(shape):
    inputs = make_inputs(*shape)
    oracle = selective_scan_reference(*[t.double() for t in inputs])
    assert rel_err(selective_scan_reference(*inputs), oracle) < REL_TOL_FP32
    assert rel_err(selective_scan_parallel(*inputs), oracle) < REL_TOL_FP32


@pytest.mark.parametrize("shape", [(2, 64, 16, 16), (2, 250, 64, 16)])
def test_bf16_inputs_are_scanned_in_fp32(shape):
    """bf16 inputs are upcast; both paths then agree to FP32 precision on the same values."""
    inputs = [t.to(torch.bfloat16) for t in make_inputs(*shape)]
    y_ref = selective_scan(*inputs, mode="reference")
    y_par = selective_scan(*inputs, mode="parallel")
    assert y_ref.dtype == y_par.dtype == torch.float32
    assert rel_err(y_par, y_ref) < REL_TOL_FP32
    oracle = selective_scan_reference(*[t.double() for t in inputs])
    assert rel_err(y_par, oracle) < REL_TOL_FP32


def test_bf16_gradients_are_fp32_gradients_rounded():
    """With bf16 leaves, autograd casts gradients back to bf16 at the end; everything inside
    the scan is FP32. So bf16-leaf gradients must equal FP32-leaf gradients (on the same
    bf16 values) rounded to bf16, bit for bit, for both paths; and the two paths must agree
    to FP32 precision before that final rounding."""
    values = [t.to(torch.bfloat16) for t in make_inputs(2, 250, 64, 16)]
    weight = torch.randn(2, 250, 64, generator=torch.Generator().manual_seed(9))
    fp32 = {}
    for mode in ("parallel", "reference"):
        low = [t.clone().requires_grad_() for t in values]
        g_low = torch.autograd.grad((selective_scan(*low, mode=mode) * weight).sum(), low)
        high = [t.float().requires_grad_() for t in values]
        g_high = torch.autograd.grad((selective_scan(*high, mode=mode) * weight).sum(), high)
        for gl, gh in zip(g_low, g_high, strict=True):
            assert gl.dtype == torch.bfloat16 and torch.equal(gl, gh.to(torch.bfloat16))
        fp32[mode] = g_high
    for gp, gr in zip(fp32["parallel"], fp32["reference"], strict=True):
        assert rel_err(gp, gr) < REL_TOL_FP32


def test_scan_ignores_surrounding_autocast():
    inputs = make_inputs(2, 32, 16, 8)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        y = selective_scan(*inputs, mode="parallel")
    assert y.dtype == torch.float32
    assert rel_err(y, selective_scan_reference(*inputs)) < REL_TOL_FP32


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_parallel_matches_cpu_reference():
    inputs = make_inputs(2, 250, 256, 16)
    y_ref = selective_scan_reference(*inputs)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        y_gpu = selective_scan(*[t.cuda() for t in inputs], mode="parallel")
    assert y_gpu.dtype == torch.float32
    assert rel_err(y_gpu.cpu(), y_ref) < REL_TOL_FP32


def test_large_step_sizes_stay_finite_and_equal():
    """Stress: large Δ (strong decay) and long sequence; Ā ∈ (0, 1] keeps products bounded."""
    inputs = make_inputs(2, 250, 32, 16, dt_shift=3.0)   # Δ ≈ 3 → Ā as small as e^-50
    y_ref = selective_scan_reference(*inputs)
    y_par = selective_scan_parallel(*inputs)
    assert torch.isfinite(y_par).all()
    assert rel_err(y_par, y_ref) < REL_TOL_FP32


@pytest.mark.parametrize("length", [1, 2, 3, 5, 16, 17, 40])
def test_linear_recurrence_custom_backward_gradcheck(length):
    g = torch.Generator().manual_seed(length)
    a = (torch.rand(length, 2, 3, 4, generator=g, dtype=torch.float64) * 0.9 + 0.05)
    b = torch.randn(length, 2, 3, 4, generator=g, dtype=torch.float64)
    assert torch.autograd.gradcheck(_LinearRecurrence.apply,
                                    (a.requires_grad_(), b.requires_grad_()))


@pytest.mark.parametrize("length", [1, 2, 7, 16, 37, 97, 250, 256])
def test_linear_recurrence_matches_explicit_loop(length):
    """Covers chunk sizes K = 1 (prime L -> pure log-depth), 2, 7, 10 and 16."""
    g = torch.Generator().manual_seed(length)
    a = torch.rand(length, 2, 5, generator=g, dtype=torch.float64)
    b = torch.randn(length, 2, 5, generator=g, dtype=torch.float64)
    h, expected = torch.zeros(2, 5, dtype=torch.float64), []
    for t in range(length):
        h = a[t] * h + b[t]
        expected.append(h)
    torch.testing.assert_close(linear_recurrence(a, b), torch.stack(expected, 0),
                               rtol=1e-12, atol=1e-12)


def test_chunk_size():
    assert chunk_size(250) == 10 and chunk_size(256) == 16 and chunk_size(97) == 1
    assert chunk_size(1) == 1


def test_state_update_precedes_readout():
    """y_0 must already include the contribution of u_0 (h_0 = B̄_0 u_0, not h_{-1})."""
    u, delta, A, B, C, D = make_inputs(1, 3, 2, 2)
    D = torch.zeros_like(D)
    y = selective_scan_reference(u, delta, A, B, C, D)
    h0 = (delta[0, 0] * u[0, 0])[:, None] * B[0, 0][None, :]
    torch.testing.assert_close(y[0, 0], h0 @ C[0, 0])
