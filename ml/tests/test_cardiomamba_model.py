"""Full CardioMamba model: shapes, gradients, precision, overfit sanity, real PTB-XL forward."""

import pytest
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from cardiomamba.models import CardioMamba, CardioMambaConfig, count_parameters
from conftest import STATS_PATH, requires_ptbxl

EXPECTED_PARAMS = 939_397
cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


@pytest.fixture(scope="module")
def model() -> CardioMamba:
    torch.manual_seed(0)
    return CardioMamba().eval()


def test_default_config_is_locked():
    c = CardioMambaConfig()
    assert (c.input_channels, c.input_length, c.patch_size, c.patch_stride) == (12, 1000, 4, 4)
    assert (c.d_model, c.n_layers, c.d_state, c.d_conv, c.expand, c.dt_rank) == (
        128, 4, 16, 4, 2, 8)
    assert (c.dropout, c.num_classes, c.bidirectional) == (0.1, 5, True)
    assert c.seq_len == 250 and c.d_inner == 256


def test_parameter_count(model):
    assert count_parameters(model) == EXPECTED_PARAMS


def test_no_positional_parameters(model):
    names = [n for n, _ in model.named_parameters()]
    assert not any("pos" in n or "embed" in n for n in names)
    assert model.stem.proj.padding == (0,)


@pytest.mark.parametrize("batch", [1, 2, 4])
def test_stage_shapes(model, batch):
    x = torch.randn(batch, 1000, 12)
    with torch.no_grad():
        h = model.stem(x)
        assert h.shape == (batch, 250, 128)
        h = h.float()
        for blk in model.blocks:
            h = blk(h)
            assert h.shape == (batch, 250, 128)
        pooled = model.norm_f(h).mean(dim=1)
        assert pooled.shape == (batch, 128)
        logits = model(x)
    assert logits.shape == (batch, 5) and logits.dtype == torch.float32
    torch.testing.assert_close(model.head(pooled), logits)


def test_rejects_wrong_input_layout(model):
    with pytest.raises(ValueError, match="Expected input"):
        model(torch.randn(2, 12, 1000))   # [B, C, T] is not accepted


def test_outputs_are_logits_and_predict_proba_is_sigmoid(model):
    x = torch.randn(4, 1000, 12) * 3
    logits = model(x)
    torch.testing.assert_close(model.predict_proba(x), torch.sigmoid(logits))


def test_reference_and_parallel_models_agree(model):
    ref = CardioMamba(CardioMambaConfig(scan_mode="reference")).eval()
    ref.load_state_dict(model.state_dict())
    x = torch.randn(2, 1000, 12)
    with torch.no_grad():
        a, b = model(x), ref(x)
    assert ((a - b).abs().max() / b.abs().max()).item() < 1e-5


def test_all_parameters_receive_finite_gradients():
    torch.manual_seed(0)
    m = CardioMamba().train()
    m(torch.randn(2, 1000, 12)).sum().backward()
    for name, p in m.named_parameters():
        assert p.grad is not None, f"{name} has no gradient"
        assert torch.isfinite(p.grad).all(), f"{name} has non-finite gradient"
    mixer = m.blocks[0].reverse_mixer
    for p in (mixer.A_log, mixer.D, mixer.dt_proj.weight, mixer.dt_proj.bias, mixer.x_proj.weight,
              mixer.conv1d.weight, m.stem.proj.weight, m.head.weight):
        assert p.grad.abs().sum() > 0


@pytest.mark.parametrize("autocast", [False, True])
def test_finite_outputs_and_gradients_over_random_inputs(autocast):
    torch.manual_seed(0)
    m = CardioMamba().train()
    for seed in range(3):
        x = torch.randn(2, 1000, 12, generator=torch.Generator().manual_seed(seed)) * (seed + 1)
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
            out = m(x)
        assert out.dtype == torch.float32 and torch.isfinite(out).all()
        m.zero_grad()
        out.sum().backward()
        assert all(torch.isfinite(p.grad).all() for p in m.parameters())


def test_activation_checkpointing_is_numerically_identical():
    torch.manual_seed(0)
    plain = CardioMamba(CardioMambaConfig(dropout=0.0)).train()
    ckpt = CardioMamba(CardioMambaConfig(dropout=0.0, checkpoint_blocks=True)).train()
    ckpt.load_state_dict(plain.state_dict())
    x = torch.randn(2, 1000, 12)
    a, b = plain(x), ckpt(x)
    torch.testing.assert_close(a, b)
    a.sum().backward()
    b.sum().backward()
    for (n, p), q in zip(plain.named_parameters(), ckpt.parameters(), strict=True):
        torch.testing.assert_close(p.grad, q.grad, msg=n)


def _overfit(model: CardioMamba, x: torch.Tensor, y: torch.Tensor, steps: int) -> list[float]:
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    losses = []
    model.train()
    for _ in range(steps):
        loss = F.binary_cross_entropy_with_logits(model(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses


def _synthetic(n, cfg, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, cfg.input_length, cfg.input_channels, generator=g)
    y = (torch.rand(n, cfg.num_classes, generator=g) < 0.4).float()
    return x, y


def test_tiny_overfit_small_config_cpu():
    """Architecture sanity only (not the real training setup): loss must collapse."""
    torch.manual_seed(0)
    cfg = CardioMambaConfig(input_length=64, d_model=32, n_layers=2, d_state=8, dt_rank=4,
                            dropout=0.0)
    model = CardioMamba(cfg)
    x, y = _synthetic(12, cfg)
    before = [p.detach().clone() for p in model.parameters()]
    losses = _overfit(model, x, y, steps=150)
    assert losses[-1] < 0.1 * losses[0], (losses[0], losses[-1])
    assert all(not torch.equal(b, p) for b, p in zip(before, model.parameters(), strict=True))


@cuda_only
def test_tiny_overfit_full_config_cuda():
    torch.manual_seed(0)
    model = CardioMamba().cuda()
    x, y = _synthetic(16, model.config)
    losses = _overfit(model, x.cuda(), y.cuda(), steps=150)
    assert losses[-1] < 0.1 * losses[0], (losses[0], losses[-1])


@cuda_only
def test_cuda_bf16_autocast_forward_backward():
    torch.manual_seed(0)
    model = CardioMamba().cuda().train()
    x = torch.randn(4, 1000, 12, device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(x)
    assert out.dtype == torch.float32 and torch.isfinite(out).all()
    out.sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters())
    fp32 = model.eval()(x)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        bf16 = model(x)
    assert (fp32 - bf16).abs().max() < 0.1   # bf16 matmuls; scan itself stays FP32


@requires_ptbxl
def test_real_ptbxl_batch_forward_backward(metadata, ptbxl_dir):
    from cardiomamba.data import NormalizationStats, PTBXLMultilabelDataset, PTBXLPreprocessor

    pre = PTBXLPreprocessor(ptbxl_dir, NormalizationStats.load(STATS_PATH))
    ds = PTBXLMultilabelDataset(metadata.split("val").iloc[:4], pre)
    batch = next(iter(DataLoader(ds, batch_size=4)))
    assert batch["signal"].shape == (4, 1000, 12)
    torch.manual_seed(0)
    model = CardioMamba().train()
    logits = model(batch["signal"])
    assert logits.shape == (4, 5) and torch.isfinite(logits).all()
    loss = F.binary_cross_entropy_with_logits(logits, batch["target"])
    loss.backward()
    assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in model.parameters())
