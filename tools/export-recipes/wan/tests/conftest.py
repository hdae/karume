"""Wan recipe のテストが共有するフィクスチャ（DiT 側・VAE 側のテストの両方が使う）。

実重みを要するテストは {@link wan_snapshot}（Wan2.1 T2V-1.3B）か {@link wan22_snapshot}（Wan2.2
TI2V-5B）を受ける。pin した revision の部品が HF キャッシュに無い機・`wan` グループを同期して
いない環境では **SKIP**（理由つき）にする — 既定の sync だけで pytest が collection ごと落ちない形を
保つ（`tests/test_optional_group_imports.py` と同じ向き）。取得は
`uv run --group wan python -m wan.sources [--model ti2v-5b] --fetch`。

{@link ti2v_synthetic_vae} は実重みを読まない合成の Wan2.2 VAE で、snapshot の無い機でも Wan2.2 の
構造の主張と故障注入を常に走らせるためにある（ADR 0121 段 4）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

#: 合成の Wan2.2 VAE の config — 上流 `AutoencoderKLWan` を 5B の VAE（pin した
#: `vae/config.json`）と同じ構造で、チャネルだけを縮めたもの。decoder の dims は
#: [64, 64, 64, 32, 16]（5B は [1024, 1024, 1024, 512, 256]）で、ショートカット（DupUp3D）は 5B と
#: 同じ 3 本（up0 / up1 = ft 2・rep 8、up2 = ft 1・rep 2）。`clip_output` は `from_pretrained` が
#: 無視するキーなので渡さない。
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
    """pin した revision の取得済み snapshot（`transformer` / `vae` / `scheduler` を持つ）。"""
    pytest.importorskip("huggingface_hub")
    from wan.sources import WanSourceError, local_snapshot

    try:
        return local_snapshot()
    except WanSourceError as error:
        pytest.skip(f"Wan2.1 の上流 checkpoint が手元に無い: {error}")


@pytest.fixture(scope="session")
def wan22_snapshot() -> Path:
    """Wan2.2 TI2V-5B の pin した revision の取得済み snapshot（ADR 0121 決定 1）。"""
    pytest.importorskip("huggingface_hub")
    from wan.sources import WanSourceError, local_snapshot

    try:
        return local_snapshot("ti2v-5b")
    except WanSourceError as error:
        pytest.skip(f"Wan2.2 TI2V-5B の上流 checkpoint が手元に無い: {error}")


@pytest.fixture(scope="session")
def ti2v_synthetic_vae():
    """乱数初期化した合成の Wan2.2 VAE（CPU f32・eval — {@link TI2V_SYNTHETIC_CONFIG}）。

    初期化の乱数は seed 0 で、グローバルの乱数の状態は汚さない。テストは書き換えない（書き換える
    なら `copy.deepcopy` してから）。
    """
    pytest.importorskip("diffusers")
    import torch
    from diffusers import AutoencoderKLWan

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        return AutoencoderKLWan(**TI2V_SYNTHETIC_CONFIG).eval()
