"""Wan2.1 recipe のテストが共有するフィクスチャ（DiT 側・VAE 側のテストの両方が使う）。

実重みを要するテストは {@link wan_snapshot} を受ける。pin した revision の部品が HF キャッシュに
無い機・`wan` グループを同期していない環境では **SKIP**（理由つき）にする — 既定の sync だけで
pytest が collection ごと落ちない形を保つ（`tests/test_optional_group_imports.py` と同じ向き）。
取得は `uv run --group wan python -m wan.sources --fetch`。
"""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def wan_snapshot() -> Path:
    """pin した revision の取得済み snapshot（`transformer` / `vae` / `scheduler` を持つ）。"""
    pytest.importorskip("huggingface_hub")
    from wan.sources import WanSourceError, local_snapshot

    try:
        return local_snapshot()
    except WanSourceError as error:
        pytest.skip(f"Wan2.1 の上流 checkpoint が手元に無い: {error}")
