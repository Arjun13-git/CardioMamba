"""Phase 3 evidence: CardioMamba architecture, S6 scan equivalence and real PTB-XL sanity.

Reports reference-vs-parallel scan errors (forward and gradients) for several shapes and
dtypes, the parameter breakdown, and a forward/backward pass on real PTB-XL records coming
from the Phase 2 Dataset. No training is performed.

Usage:  uv run python scripts/validate_model.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from cardiomamba.data import (
    NormalizationStats,
    PTBXLMetadata,
    PTBXLMultilabelDataset,
    PTBXLPreprocessor,
)
from cardiomamba.models import (
    CardioMamba,
    CardioMambaConfig,
    parameter_breakdown,
    selective_scan,
    selective_scan_reference,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = REPO_ROOT / "ml" / "data" / "raw" / "ptb-xl-1.0.3"
STATS_PATH = REPO_ROOT / "ml" / "configs" / "normalization_stats_prep-v1.json"
SHAPES = [(1, 1, 4, 2), (2, 7, 8, 4), (3, 64, 16, 16), (4, 97, 64, 16), (4, 250, 256, 16),
          (2, 1000, 64, 16)]
NAMES = ["u", "delta", "A", "B", "C", "D"]


def scan_inputs(batch, length, d_inner, d_state, device, dtype, seed=0):
    g = torch.Generator().manual_seed(seed)
    tensors = [
        torch.randn(batch, length, d_inner, generator=g),
        F.softplus(torch.randn(batch, length, d_inner, generator=g) - 2.0),
        -torch.exp(torch.log(torch.arange(1, d_state + 1).float()).repeat(d_inner, 1)
                   + 0.1 * torch.randn(d_inner, d_state, generator=g)),
        torch.randn(batch, length, d_state, generator=g),
        torch.randn(batch, length, d_state, generator=g),
        torch.randn(d_inner, generator=g),
    ]
    return [t.to(device=device, dtype=dtype) for t in tensors]


def errors(x: torch.Tensor, ref: torch.Tensor) -> tuple[float, float]:
    diff = (x.double() - ref.double()).abs().max().item()
    return diff, diff / max(ref.double().abs().max().item(), 1e-30)


def scan_report(device: str) -> bool:
    ok = True
    dtypes = [torch.float32, torch.bfloat16]
    print(f"\nScan equivalence on {device} (rel = max|par - ref| / max|ref|; reference = "
          "sequential scan on the same inputs)")
    print("bf16 rows: inputs hold bf16 values; gradients are compared in FP32 (the scan's own "
          "precision), and 'bf16 cast' checks that bf16-leaf gradients are exactly the FP32 "
          "gradients rounded to bf16 (autograd's final cast back to the leaf dtype).")
    print(f"{'dtype':>9} {'B':>2} {'L':>5} {'E':>4} {'N':>3} | {'fwd abs':>9} {'fwd rel':>9} | "
          f"{'max grad rel':>12} (param) | {'par vs fp64':>11} | bf16 cast")
    for dtype in dtypes:
        for shape in SHAPES:
            values = scan_inputs(*shape, device, dtype)
            inputs = [t.float().requires_grad_() for t in values]   # bf16 values, FP32 leaves
            w = torch.randn(shape[:3], generator=torch.Generator().manual_seed(1)).to(device)
            y_par = selective_scan(*inputs, mode="parallel")
            g_par = torch.autograd.grad((y_par * w).sum(), inputs)
            y_ref = selective_scan(*inputs, mode="reference")
            g_ref = torch.autograd.grad((y_ref * w).sum(), inputs)
            oracle = selective_scan_reference(*[t.detach().double() for t in inputs])
            fwd_abs, fwd_rel = errors(y_par, y_ref)
            grad_rels = [errors(a, b)[1] for a, b in zip(g_par, g_ref, strict=True)]
            worst = max(range(6), key=lambda i: grad_rels[i])
            oracle_rel = errors(y_par, oracle)[1]
            cast = "n/a"
            if dtype != torch.float32:
                leaves = [t.clone().requires_grad_() for t in values]      # bf16 leaves
                y_low = selective_scan(*leaves, mode="parallel")
                g_low = torch.autograd.grad((y_low * w).sum(), leaves)
                exact = torch.equal(y_low, y_par) and all(
                    torch.equal(gl, gp.to(dtype)) for gl, gp in zip(g_low, g_par, strict=True))
                ok &= exact
                cast = "exact" if exact else "MISMATCH"
            ok &= fwd_rel < 1e-5 and max(grad_rels) < 1e-5 and oracle_rel < 1e-5
            ok &= y_par.dtype == torch.float32
            print(f"{str(dtype).split('.')[-1]:>9} {shape[0]:>2} {shape[1]:>5} {shape[2]:>4} "
                  f"{shape[3]:>3} | {fwd_abs:9.2e} {fwd_rel:9.2e} | {grad_rels[worst]:12.2e} "
                  f"({NAMES[worst]:>5}) | {oracle_rel:11.2e} | {cast}")
    return ok


def main() -> int:
    torch.manual_seed(0)
    cfg = CardioMambaConfig()
    model = CardioMamba(cfg)
    print("Configuration:", {k: v for k, v in cfg.to_dict().items()
                             if k not in ("dt_min", "dt_max", "dt_init_floor", "rms_norm_eps")})
    print("\nTrainable parameters")
    for name, n in parameter_breakdown(model).items():
        if name.startswith(("block1", "block2", "block3")):
            continue
        print(f"  {name:24s}{n:>10,}")
    print("  (blocks 1-3 identical to block0)")

    ok = scan_report("cpu")
    if torch.cuda.is_available():
        ok &= scan_report("cuda")

    print("\nReal PTB-XL sanity (Phase 2 Dataset -> CardioMamba)")
    meta = PTBXLMetadata.load(DATASET_DIR)
    pre = PTBXLPreprocessor(DATASET_DIR, NormalizationStats.load(STATS_PATH))
    ds = PTBXLMultilabelDataset(meta.split("val").iloc[:8], pre)
    batch = next(iter(DataLoader(ds, batch_size=8)))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).train()
    x, y = batch["signal"].to(device), batch["target"].to(device)
    for label, autocast in (("fp32", False), ("bf16 autocast", True)):
        model.zero_grad(set_to_none=True)
        with torch.autocast(device, dtype=torch.bfloat16, enabled=autocast):
            logits = model(x)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        loss.backward()
        finite_grads = all(torch.isfinite(p.grad).all() for p in model.parameters())
        good = (logits.shape == (8, 5) and bool(torch.isfinite(logits).all())
                and bool(torch.isfinite(loss)) and finite_grads)
        ok &= good
        print(f"  {label:14s} input {list(x.shape)} -> logits {list(logits.shape)} "
              f"{logits.dtype}; loss {loss.item():.4f}; finite grads {finite_grads} -> "
              f"{'PASS' if good else 'FAIL'}")
    print(f"  ecg_ids: {batch['ecg_id'].tolist()}")

    print(f"\nOverall: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
