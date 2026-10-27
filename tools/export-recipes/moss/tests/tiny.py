"""The tiny `MOSSImageTransformer` the DiT patch and export tests share."""

from __future__ import annotations

from types import ModuleType
from typing import Any

import torch

#: 24 blocks (the negative branch skips block 22), head_dim 16 (two RoPE bands per axis), freq_dim
#: 8; no extent equals the tests' symbol maximum (512).
TINY: dict[str, Any] = {
    "dim": 32,
    "ffn_dim": 16,
    "num_heads": 2,
    "num_layers": 24,
    "text_dim": 16,
    "freq_dim": 8,
    "latent_channels": 8,
    "rope_base": 10000.0,
}


def tiny_model(module: ModuleType) -> Any:
    """The tiny DiT of the pinned pipeline code, with the RMSNorm weights moved off 1 (so a norm
    read on the wrong rows shows)."""
    torch.manual_seed(0)
    model = module.MOSSImageTransformer(**TINY).eval()
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.dim() == 1:
                parameter.add_(0.1 * torch.randn_like(parameter))
    return model
