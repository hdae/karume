"""上流 checkpoint の出所（`_shared.upstream`）をテストで再現する糊。

2 つの席を合成する:

- **手元の checkpoint の取得記録**（{@link write_snapshot}）— `hf download --local-dir` が残す
  `.cache/huggingface/download/config.json.metadata` と、`README.md` の front matter。台本が
  容器へ焼く出所と、組み立てが突き合わせる revision の出どころ。
- **フィクスチャ容器が名乗る出所**（{@link stamp_fixture_provenance}）— `ir_fixtures.ir_container`
  は出所を引数に取らず、モジュール定数 `FIXTURE_PROVENANCE` を焼く（core のテストヘルパで、
  recipe 側からは編集しない）。出所の門を持つ family の組み立てテストは、その定数をテストの間だけ
  差し替えて「台本が焼いた形」の容器を作る。

`tools/export-recipes/conftest.py` がこのディレクトリを sys.path へ張る（`container_series` と同じ
経路）。
"""

from __future__ import annotations

from pathlib import Path

import ir_fixtures
import pytest

from _shared.upstream import SNAPSHOT_RECORD_DIR, SNAPSHOT_RECORD_FILE
from karume.container import Provenance

#: 合成の commit SHA（実在のどの revision とも一致しない綴り）。
FIXTURE_REVISION = "0123456789abcdef0123456789abcdef01234567"

#: 取り違えの故障注入に使う、もう 1 つの commit SHA。
OTHER_REVISION = "fedcba9876543210fedcba9876543210fedcba98"


def write_snapshot(
    model_dir: Path, *, license: str, revision: str = FIXTURE_REVISION, extra: str = ""
) -> None:
    """`hf download --local-dir` の取得記録と README の front matter を合成する。

    `extra` は front matter へ足す行（`license_name: …` など）。
    """
    record = model_dir / SNAPSHOT_RECORD_DIR / f"{SNAPSHOT_RECORD_FILE}.metadata"
    record.parent.mkdir(parents=True, exist_ok=True)
    record.write_text(f'{revision}\n"etag-fixture"\n1786862475.0\n', encoding="utf-8")
    (model_dir / "README.md").write_text(
        f"---\nlicense: {license}\n{extra}tags:\n- fixture\n---\n\n# fixture\n", encoding="utf-8"
    )


def stamp_fixture_provenance(monkeypatch: pytest.MonkeyPatch, provenance: Provenance) -> None:
    """このテストの間だけ、`ir_fixtures.ir_container` が焼く出所を差し替える。"""
    monkeypatch.setattr(ir_fixtures, "FIXTURE_PROVENANCE", provenance)
