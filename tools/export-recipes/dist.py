"""リポジトリ用の dist ドライバ — core の組み立てエンジンに recipe 側 pipeline を合成する。

`karume.dist` が持つ {@link karume.dist.PIPELINES} は「core wheel だけで組める pipeline」で
あって全量ではない（ADR 0065 決定 2 — モデル別 recipe は wheel の外）。受理集合の**全量**は
ここが決める: core の表に、リポ専用 recipe が公開する `PIPELINE` を足したもの。

    uv run python dist.py                                  # 既定 = anima（export-recipes で）
    uv run python dist.py --pipeline sbv2 --card-profile fn
    uv run python dist.py --pipeline sbv2 --card-profile jvnv \\
        --model F1 --model F2 --out ../../models/karume-sbv2-jvnv

NOTE: family の移行が 1 つ進むたびに、合成元が core 側から recipe 側へ 1 行ずつ移る
（`karume.dist.PIPELINES` から消え、`<family>.distribution.PIPELINE` として足される）。
最後の family が出た時点で core の表は空になり、この辞書だけが受理集合の正本になる。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from anima import distribution as anima_distribution
from karume.dist import PIPELINES as CORE_PIPELINES
from karume.dist import Pipeline
from karume.dist import main as dist_main
from sbv2 import distribution as sbv2_distribution

#: 受理集合の全量。並びは `--help` の並びでもあるので、既定を先頭に置く。
PIPELINES: Mapping[str, Pipeline] = {
    "anima": anima_distribution.PIPELINE,
    "sbv2": sbv2_distribution.PIPELINE,
    **CORE_PIPELINES,
}

#: 旧 `karume dist`（引数なし）の UX をドライバ側で維持する。
DEFAULT_PIPELINE = "anima"


def main(argv: Sequence[str] | None = None) -> None:
    dist_main(argv, pipelines=PIPELINES, default_pipeline=DEFAULT_PIPELINE)


if __name__ == "__main__":
    main()
