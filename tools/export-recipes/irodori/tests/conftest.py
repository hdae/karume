"""Irodori のモデル実装 clone を import 可能にする（`irodori.export` の `--source-dir` 既定）。

**在れば** `sys.path` へ載せる — {@link irodori.patch} の Irodori 側パッチは `irodori_tts` を
差し替えるので、同値テストにはこの clone が要る。git 追跡外なので無い環境ではテスト側が
skip する（`pytest.importorskip("irodori_tts")`）。

置き場が `irodori/tests/` なのは、**この 4 本のテストだけが必要とするから**（pytest は
rootdir から対象ファイルのディレクトリまでの conftest を、そのディレクトリのテストを
import する前に読む）。export-recipes 直下に置くと、clone の有無が他 family の収集にも
効いてしまう — 依存の射程は要求元の隣に置く。

NOTE: 元は core 側の `tools/exporter/tests/conftest.py` に居た（ADR 0065 Context が名指しで
挙げた逆流の 1 つ）。family の移動と一緒にここへ降りてきた（同 段 4）。
"""

from __future__ import annotations

import sys

import pytest

from _shared.paths import REPO_ROOT

#: `irodori.export.DEFAULT_SOURCE_DIR` と同じ置き場（綴りが割れると片方だけ空振りする）。
IRODORI_SOURCE_DIR = REPO_ROOT / "inputs" / "irodori" / "Irodori-TTS"
if IRODORI_SOURCE_DIR.is_dir() and str(IRODORI_SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(IRODORI_SOURCE_DIR))


@pytest.fixture
def restore_forward():
    """`irodori.patch.apply_patches` のクラス属性の差し替えをテスト後に戻す。

    差し替えはプロセス全域なので、戻さないと後続のテストがパッチ後の実装で回る。

    `apply_patches` は `irodori_tts`（git 追跡外の clone — 上で `sys.path` へ足す）も
    差し替えるので、無い環境ではこのフィクスチャを使うテストだけを skip する。パッチの同値
    （`test_patch.py`）と DiT の分割の同値（`test_export.py`）の両方が使うのでここに置く。
    """
    from irodori import patch as patch_irodori

    modernbert = pytest.importorskip("transformers.models.modernbert.modeling_modernbert")
    irodori_model = pytest.importorskip("irodori_tts.model")
    original = modernbert.ModernBertAttention.forward
    original_rope = irodori_model.apply_rotary_emb
    original_norm = irodori_model.RMSNorm.forward
    original_adaln = irodori_model.LowRankAdaLN.forward
    applied = patch_irodori._APPLIED
    try:
        yield irodori_model
    finally:
        modernbert.ModernBertAttention.forward = original
        irodori_model.apply_rotary_emb = original_rope
        irodori_model.RMSNorm.forward = original_norm
        irodori_model.LowRankAdaLN.forward = original_adaln
        patch_irodori._APPLIED = applied
        patch_irodori._ORIGINAL_APPLY_ROTARY_EMB = None
