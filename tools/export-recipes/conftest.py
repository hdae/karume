"""recipe のテストから core 側のテストヘルパ（`ir_fixtures`）を引くための path 張り。

`ir_fixtures`（正当な最小 IR コンテナの合成 — `tools/exporter/tests/ir_fixtures.py`）は core と
recipe の**両方**のテストが使う。置き場を 1 つに保つのは、safetensors の綴りと IR v1 の規則の
写しを 2 つ持たないため（片方だけ古びると「テストは緑・実物だけ落ちる」に戻る）。依存の向きは
recipe → core の一方向のままで（ADR 0065 決定 3）、wheel（`src/karume/`）には 1 バイトも
入れない。

MUST: 追加は**末尾**（`insert(0, …)` にしない）— recipe 側の綴り（`dist` / `_shared` /
family パッケージ）を core 側のテストディレクトリが先に解決してしまう形を作らない。
"""

from __future__ import annotations

import sys
from pathlib import Path

#: `tools/export-recipes/conftest.py` → `tools/` → core のテスト置き場。
CORE_TESTS = Path(__file__).resolve().parent.parent / "exporter" / "tests"

if str(CORE_TESTS) not in sys.path:
    sys.path.append(str(CORE_TESTS))
