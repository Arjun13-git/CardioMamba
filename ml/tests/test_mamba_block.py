"""Mamba mixer and bidirectional block: initialisation, causality and direction handling."""

import math

import pytest
import torch
import torch.nn.functional as F

from cardiomamba.models import BiMambaBlock, CardioMambaConfig, MambaMixer

CFG = CardioMambaConfig()


@pytest.fixture
def mixer() -> MambaMixer:
    torch.manual_seed(0)
    return MambaMixer(d_model=128, d_state=16, d_conv=4, expand=2, dt_rank=8).eval()


@pytest.fixture
def block() -> BiMambaBlock:
    torch.manual_seed(0)
    return BiMambaBlock(CFG).eval()


def test_mixer_parameter_shapes_and_init(mixer):
    assert mixer.in_proj.weight.shape == (512, 128)
    assert mixer.conv1d.weight.shape == (256, 1, 4)          # depthwise
    assert mixer.x_proj.weight.shape == (8 + 2 * 16, 256)
    assert mixer.dt_proj.weight.shape == (256, 8)
    assert mixer.out_proj.weight.shape == (128, 256)
    assert mixer.A_log.shape == (256, 16) and mixer.D.shape == (256,)
    # S4D-real: A = -[1..N] for every channel
    torch.testing.assert_close(-torch.exp(mixer.A_log[0]), -torch.arange(1.0, 17.0))
    assert torch.equal(mixer.D, torch.ones(256))
    dt = F.softplus(mixer.dt_proj.bias)
    assert dt.min() >= 1e-4 and dt.max() <= 0.1 + 1e-6 and dt.min() < 0.01


def test_mixer_shape(mixer):
    assert mixer(torch.randn(3, 250, 128)).shape == (3, 250, 128)


def test_mixer_is_causal(mixer):
    """Changing the input at t0 must not change any output at t < t0."""
    x = torch.randn(2, 60, 128)
    x2 = x.clone()
    x2[:, 40] += 5.0
    y, y2 = mixer(x), mixer(x2)
    assert torch.equal(y[:, :40], y2[:, :40])
    assert (y[:, 40:] - y2[:, 40:]).abs().max() > 1e-3


def test_block_structure_has_distinct_directions(block):
    assert block.reverse_mixer is not None
    fwd = dict(block.forward_mixer.named_parameters())
    rev = dict(block.reverse_mixer.named_parameters())
    for name, param in fwd.items():
        assert param.data_ptr() != rev[name].data_ptr(), f"{name} is shared"
    assert not torch.equal(fwd["in_proj.weight"], rev["in_proj.weight"])


def test_forward_direction_is_left_to_right(block):
    """Forward branch: a perturbation at t0 only affects positions t >= t0."""
    u = torch.randn(1, 50, 128)
    u2 = u.clone()
    u2[:, 30] += 5.0
    f, f2 = block.forward_mixer(u), block.forward_mixer(u2)
    assert torch.equal(f[:, :30], f2[:, :30]) and (f[:, 30:] - f2[:, 30:]).abs().max() > 1e-3


def test_reverse_direction_is_right_to_left_and_realigned(block):
    """Reverse branch: flip -> mixer -> flip is anti-causal and aligned with the input."""
    u = torch.randn(1, 50, 128)
    u2 = u.clone()
    u2[:, 20] += 5.0
    r = block.reverse_mixer(u.flip(1)).flip(1)
    r2 = block.reverse_mixer(u2.flip(1)).flip(1)
    assert torch.equal(r[:, 21:], r2[:, 21:])                 # nothing after t0 changes
    assert (r[:, :21] - r2[:, :21]).abs().max() > 1e-3        # t <= t0 changes
    # the block's mix() is exactly forward + re-aligned reverse
    torch.testing.assert_close(block.mix(u), block.forward_mixer(u) + r)


def test_bidirectional_block_sees_both_directions(block):
    """A perturbation in the middle changes every position with both branches, but leaves
    earlier positions bit-identical with the forward branch alone. (At initialisation the
    state decays quickly, so distant effects are small but non-zero.)"""
    x = torch.randn(1, 50, 128)
    x2 = x.clone()
    x2[:, 25] += 5.0
    diff = (block(x) - block(x2)).abs().amax(dim=-1)[0]      # [L]
    assert (diff > 0).all()
    u, u2 = block.norm(x), block.norm(x2)
    fwd_diff = (block.forward_mixer(u) - block.forward_mixer(u2)).abs().amax(dim=-1)[0]
    assert (fwd_diff[:25] == 0).all() and (fwd_diff[25:] > 0).all()


def test_reverse_equals_forward_on_flipped_sequence_when_weights_tied():
    """If the reverse mixer had the forward weights, its output on x equals flip(fwd(flip(x)))."""
    torch.manual_seed(1)
    blk = BiMambaBlock(CFG).eval()
    blk.reverse_mixer.load_state_dict(blk.forward_mixer.state_dict())
    u = torch.randn(2, 40, 128)
    torch.testing.assert_close(blk.mix(u), blk.forward_mixer(u) + blk.forward_mixer(u.flip(1)).flip(1))
    # and the result is NOT two identical forward passes
    assert not torch.allclose(blk.mix(u), 2 * blk.forward_mixer(u))


def test_block_residual_and_width(block):
    x = torch.randn(2, 250, 128)
    out = block(x)
    assert out.shape == x.shape
    torch.testing.assert_close(out - x, block.mix(block.norm(x)))   # eval: dropout is identity


def test_unidirectional_config_has_no_reverse_mixer():
    blk = BiMambaBlock(CardioMambaConfig(bidirectional=False))
    assert blk.reverse_mixer is None


def test_dt_init_is_log_uniform_in_range():
    torch.manual_seed(0)
    m = MambaMixer(128, 16, 4, 2, 8)
    log_dt = torch.log(F.softplus(m.dt_proj.bias))
    assert log_dt.min() >= math.log(1e-3) - 1e-4 and log_dt.max() <= math.log(1e-1) + 1e-4
