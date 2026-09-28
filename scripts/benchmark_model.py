"""Phase 3 GPU benchmark of the locked CardioMamba configuration (no training loop).

For each batch size / precision / activation-checkpointing setting it measures:
  * forward only (eval, no grad): peak allocated memory and time;
  * forward + backward (train mode, BCE loss on random targets): peak memory and time.
A setting is "safe" when its peak *reserved* memory stays below the free VRAM measured at
start-up minus a safety margin (the desktop session also uses the GPU).

Usage:  uv run python scripts/benchmark_model.py [--batch-sizes 64 48 32 16 8 4]
"""

from __future__ import annotations

import argparse
import sys
import time

import torch
import torch.nn.functional as F

from cardiomamba.models import CardioMamba, CardioMambaConfig, count_parameters

MiB = 2**20


def timed(fn, warmup: int = 2, iters: int = 5) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000


def measure(batch: int, autocast: bool, ckpt: bool) -> dict:
    torch.manual_seed(0)
    model = CardioMamba(CardioMambaConfig(checkpoint_blocks=ckpt)).cuda()
    x = torch.randn(batch, 1000, 12, device="cuda")
    y = (torch.rand(batch, 5, device="cuda") < 0.3).float()
    def dtype_ctx() -> torch.autocast:
        return torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast)

    def fwd():
        with torch.no_grad(), dtype_ctx():
            model(x)

    def step():
        model.zero_grad(set_to_none=True)
        with dtype_ctx():
            loss = F.binary_cross_entropy_with_logits(model(x), y)
        loss.backward()

    out = {}
    model.eval()
    torch.cuda.reset_peak_memory_stats()
    out["fwd_ms"] = timed(fwd)
    out["fwd_peak"] = torch.cuda.max_memory_allocated() / MiB
    model.train()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    out["step_ms"] = timed(step)
    out["step_peak"] = torch.cuda.max_memory_allocated() / MiB
    out["step_reserved"] = torch.cuda.max_memory_reserved() / MiB
    return out   # model / tensors are freed on return; caller empties the cache


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[64, 48, 32, 16, 8, 4])
    parser.add_argument("--margin-mib", type=int, default=512)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        print("CUDA not available")
        return 1

    props = torch.cuda.get_device_properties(0)
    free, total = (v / MiB for v in torch.cuda.mem_get_info())
    budget = free - args.margin_mib
    print(f"GPU: {props.name} (cc {props.major}.{props.minor}), total {total:.0f} MiB, "
          f"free at start {free:.0f} MiB, safety budget {budget:.0f} MiB")
    print(f"torch {torch.__version__}, CUDA {torch.version.cuda}; "
          f"parameters {count_parameters(CardioMamba()):,}")
    print(f"\n{'precision':>13} {'ckpt':>5} {'B':>3} | {'fwd ms':>7} {'fwd MiB':>8} | "
          f"{'step ms':>8} {'step MiB':>9} {'reserved':>9} | {'ms/sample':>9} | status")
    largest_safe: dict[tuple[str, bool], int] = {}
    for autocast in (False, True):
        prec = "bf16 autocast" if autocast else "fp32"
        for ckpt in (False, True):
            for batch in args.batch_sizes:
                try:
                    r = measure(batch, autocast, ckpt)
                except torch.OutOfMemoryError:
                    r = None
                torch.cuda.empty_cache()
                if r is None:
                    print(f"{prec:>13} {ckpt!s:>5} {batch:>3} | {'OOM':>7}")
                    continue
                safe = r["step_reserved"] <= budget
                if safe:
                    key = (prec, ckpt)
                    largest_safe[key] = max(largest_safe.get(key, 0), batch)
                print(f"{prec:>13} {ckpt!s:>5} {batch:>3} | {r['fwd_ms']:7.1f} "
                      f"{r['fwd_peak']:8.0f} | {r['step_ms']:8.1f} {r['step_peak']:9.0f} "
                      f"{r['step_reserved']:9.0f} | {r['step_ms'] / batch:9.2f} | "
                      f"{'safe' if safe else 'UNSAFE'}")
    print("\nLargest safe training batch (forward+backward):")
    for (prec, ckpt), batch in largest_safe.items():
        print(f"  {prec:>13}, checkpointing={ckpt!s:5}: {batch}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
