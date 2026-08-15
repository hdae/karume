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

from _shared.paths import REPO_ROOT

#: `irodori.export.DEFAULT_SOURCE_DIR` と同じ置き場（綴りが割れると片方だけ空振りする）。
IRODORI_SOURCE_DIR = REPO_ROOT / "inputs" / "irodori" / "Irodori-TTS"
if IRODORI_SOURCE_DIR.is_dir() and str(IRODORI_SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(IRODORI_SOURCE_DIR))
