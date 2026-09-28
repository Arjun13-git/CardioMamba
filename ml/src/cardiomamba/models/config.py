"""CardioMamba model configuration (locked defaults; see ADR-004 and 04-ml-architecture)."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Literal

ScanMode = Literal["parallel", "reference"]


@dataclass(frozen=True)
class CardioMambaConfig:
    """Hyper-parameters of the CardioMamba classifier.

    The defaults are the locked project configuration. Changing any value defines a new
    experiment configuration and must be recorded with the run.
    """

    # input: [B, input_length, input_channels] = [B, 1000, 12] (100 Hz, 10 s, 12 leads)
    input_channels: int = 12
    input_length: int = 1000
    # patch stem: Conv1d(12 -> d_model, kernel=patch_size, stride=patch_stride), no padding
    patch_size: int = 4
    patch_stride: int = 4
    # bidirectional Mamba-1 backbone
    d_model: int = 128
    n_layers: int = 4
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    dt_rank: int = 8
    bidirectional: bool = True
    dropout: float = 0.1
    num_classes: int = 5
    # Δt initialisation (reference Mamba-1 values): softplus(dt_bias) ~ LogUniform[dt_min, dt_max]
    dt_min: float = 1e-3
    dt_max: float = 1e-1
    dt_init_floor: float = 1e-4
    rms_norm_eps: float = 1e-5
    # implementation switches (do not change the mathematical model)
    scan_mode: ScanMode = "parallel"
    checkpoint_blocks: bool = False

    def __post_init__(self) -> None:
        if (self.input_length - self.patch_size) % self.patch_stride != 0:
            raise ValueError("input_length must tile exactly into patches (no padding is used)")
        if self.scan_mode not in ("parallel", "reference"):
            raise ValueError(f"Unknown scan_mode {self.scan_mode!r}")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

    @property
    def d_inner(self) -> int:
        """Mamba inner (expanded) width E = expand · d_model."""
        return self.expand * self.d_model

    @property
    def seq_len(self) -> int:
        """Number of tokens produced by the patch stem (250 for the defaults)."""
        return (self.input_length - self.patch_size) // self.patch_stride + 1

    @property
    def dt_init_std(self) -> float:
        """Std of the uniform init of dt_proj.weight (reference: dt_rank ** -0.5)."""
        return 1.0 / math.sqrt(self.dt_rank)

    def to_dict(self) -> dict:
        return asdict(self)
