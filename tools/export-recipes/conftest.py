"""recipe のテストが共有するテストヘルパ（`ir_fixtures` / `shard_series`）への path 張り。

`ir_fixtures`（正当な最小 IR コンポーネントの合成 — `tools/exporter/tests/ir_fixtures.py`）は
core と recipe の**両方**のテストが使う。置き場を 1 つに保つのは、safetensors の綴りと IR v1 の
規則の写しを 2 つ持たないため（片方だけ古びると「テストは緑・実物だけ落ちる」に戻る）。依存の
向きは recipe → core の一方向のままで（ADR 0065 決定 3）、wheel（`src/karume/`）には 1 バイトも
入れない。

`shard_series`（`tools/export-recipes/tests/shard_series.py`）は recipe 側だけの糊で、shard 列を
系列 / 配布形として扱う 3 手（書く / 期待 path を並べる / 読む）を 7 家族で共有する。同じ
ディレクトリの test モジュールは pytest が自力で解決するが、family サブパッケージ配下
（`<family>/tests/` は `__init__.py` を持つ）からは解決されないので、ここで明示的に張る。

MUST: 追加は**末尾**（`insert(0, …)` にしない）— recipe 側の綴り（`dist` / `_shared` /
family パッケージ）を core 側のテストディレクトリが先に解決してしまう形を作らない。
"""

from __future__ import annotations

import sys
from pathlib import Path

#: `tools/export-recipes/conftest.py` → `tools/` → core のテスト置き場。
CORE_TESTS = Path(__file__).resolve().parent.parent / "exporter" / "tests"

#: recipe 側の共有テストヘルパ（`shard_series`）の置き場。
RECIPE_TESTS = Path(__file__).resolve().parent / "tests"

for helpers in (CORE_TESTS, RECIPE_TESTS):
    if str(helpers) not in sys.path:
        sys.path.append(str(helpers))
