"""CardioMamba: patch stem → 4 × bidirectional Mamba-1 → RMSNorm → mean pool → linear head."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from cardiomamba.models.config import CardioMambaConfig
from cardiomamba.models.mamba import BiMambaBlock
from cardiomamba.models.norm import RMSNorm


class PatchStem(nn.Module):
    """Non-overlapping temporal patches: [B, T, C] -> [B, T/stride, D].

    Conv1d(C → D, kernel=patch_size, stride=patch_stride, no padding). Each token embeds one
    40 ms window (4 samples at 100 Hz) of all 12 leads. No positional encoding is added: token
    order is modelled by the Mamba recurrence.
    """

    def __init__(self, config: CardioMambaConfig) -> None:
        super().__init__()
        self.proj = nn.Conv1d(config.input_channels, config.d_model,
                              kernel_size=config.patch_size, stride=config.patch_stride)

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(x.transpose(1, 2)).transpose(1, 2)   # [B,T,C]->[B,C,T]->[B,D,L]->[B,L,D]


class CardioMamba(nn.Module):
    """Multi-label ECG classifier. forward: [B, 1000, 12] -> logits [B, 5] (no sigmoid)."""

    def __init__(self, config: CardioMambaConfig | None = None) -> None:
        super().__init__()
        self.config = config or CardioMambaConfig()
        cfg = self.config
        self.stem = PatchStem(cfg)
        self.blocks = nn.ModuleList(BiMambaBlock(cfg) for _ in range(cfg.n_layers))
        self.norm_f = RMSNorm(cfg.d_model, eps=cfg.rms_norm_eps)
        self.head_dropout = nn.Dropout(cfg.dropout)
        self.head = nn.Linear(cfg.d_model, cfg.num_classes)

    def encode(self, x: Tensor) -> Tensor:
        """[B, T, C] -> normalised token features [B, L, D]."""
        cfg = self.config
        if x.ndim != 3 or tuple(x.shape[1:]) != (cfg.input_length, cfg.input_channels):
            raise ValueError(f"Expected input [B, {cfg.input_length}, {cfg.input_channels}], "
                             f"got {list(x.shape)}")
        h = self.stem(x).float()                          # [B, L, D]; residual stream in FP32
        use_ckpt = cfg.checkpoint_blocks and self.training and torch.is_grad_enabled()
        for block in self.blocks:
            h = checkpoint(block, h, use_reentrant=False) if use_ckpt else block(h)
        return self.norm_f(h)                             # [B, L, D]

    def forward(self, x: Tensor) -> Tensor:
        pooled = self.encode(x).mean(dim=1)               # [B, L, D] -> [B, D]
        return self.head(self.head_dropout(pooled)).float()   # [B, num_classes] logits

    @torch.no_grad()
    def predict_proba(self, x: Tensor) -> Tensor:
        """Inference helper: independent sigmoid probabilities [B, num_classes]."""
        return torch.sigmoid(self(x))


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)


def parameter_breakdown(model: CardioMamba) -> dict[str, int]:
    """Trainable parameters per component (stem, each block, per-mixer, final norm, head)."""
    out = {"stem": count_parameters(model.stem)}
    for i, block in enumerate(model.blocks):
        out[f"block{i}.norm"] = count_parameters(block.norm)
        out[f"block{i}.forward_mixer"] = count_parameters(block.forward_mixer)
        if block.reverse_mixer is not None:
            out[f"block{i}.reverse_mixer"] = count_parameters(block.reverse_mixer)
    out["norm_f"] = count_parameters(model.norm_f)
    out["head"] = count_parameters(model.head)
    out["total"] = count_parameters(model)
    return out
