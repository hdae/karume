"""Fixtures shared by the Wan recipe tests (used by both the DiT-side and the VAE-side tests).

Tests that need real weights take {@link wan_snapshot} (Wan2.1 T2V-1.3B) or {@link wan22_snapshot}
(Wan2.2 TI2V-5B). On a machine without the components of the pinned revision in the HF cache, or
in an environment that has not synced the `wan` group, they **SKIP** (with a reason) — so the
default sync alone never makes pytest fail at collection (the same direction as
`tests/test_optional_group_imports.py`). Fetch with
`uv run --group wan python -m wan.sources [--model ti2v-5b] --fetch`.

{@link ti2v_synthetic_vae} is a synthetic Wan2.2 VAE that reads no real weights; it exists so that
the structural assertions and fault injection for Wan2.2 always run, even on a machine without the
snapshot (ADR 0121 step 4).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

#: The config of the synthetic Wan2.2 VAE — upstream `AutoencoderKLWan` with the same structure as
#: the 5B VAE (the pinned `vae/config.json`), with only the channels shrunk. The decoder dims are
#: [64, 64, 64, 32, 16] (the 5B has [1024, 1024, 1024, 512, 256]), and the shortcuts (DupUp3D) are
#: the same three as the 5B (up0 / up1 = ft 2, rep 8; up2 = ft 1, rep 2). `clip_output` is a key
#: `from_pretrained` ignores, so it is not passed.
TI2V_SYNTHETIC_CONFIG: dict[str, Any] = {
    "base_dim": 8,
    "decoder_base_dim": 16,
    "z_dim": 48,
    "dim_mult": [1, 2, 4, 4],
    "num_res_blocks": 2,
    "temperal_downsample": [False, True, True],
    "is_residual": True,
    "in_channels": 12,
    "out_channels": 12,
    "patch_size": 2,
    "scale_factor_spatial": 16,
    "latents_mean": [0.0] * 48,
    "latents_std": [1.0] * 48,
}


@pytest.fixture(scope="session")
def wan_snapshot() -> Path:
    """The fetched snapshot of the pinned revision (with `transformer` / `vae` / `scheduler`)."""
    pytest.importorskip("huggingface_hub")
    from wan.sources import WanSourceError, local_snapshot

    try:
        return local_snapshot()
    except WanSourceError as error:
        pytest.skip(f"the Wan2.1 upstream checkpoint is not available locally: {error}")


@pytest.fixture(scope="session")
def wan22_snapshot() -> Path:
    """The fetched snapshot of the pinned revision of Wan2.2 TI2V-5B (ADR 0121 decision 1)."""
    pytest.importorskip("huggingface_hub")
    from wan.sources import WanSourceError, local_snapshot

    try:
        return local_snapshot("ti2v-5b")
    except WanSourceError as error:
        pytest.skip(f"the Wan2.2 TI2V-5B upstream checkpoint is not available locally: {error}")


@pytest.fixture(scope="session")
def ti2v_synthetic_vae():
    """A randomly initialised synthetic Wan2.2 VAE (CPU f32, eval — {@link TI2V_SYNTHETIC_CONFIG}).

    The initialisation uses seed 0 and leaves the global random state untouched. Tests do not modify
    it (`copy.deepcopy` it first if a test must).
    """
    pytest.importorskip("diffusers")
    import torch
    from diffusers import AutoencoderKLWan

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        return AutoencoderKLWan(**TI2V_SYNTHETIC_CONFIG).eval()
