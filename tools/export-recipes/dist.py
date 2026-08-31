"""リポジトリ用の dist ドライバ — core の組み立てエンジンに recipe 側 pipeline を合成する。

`karume.dist` が持つ {@link karume.dist.PIPELINES} は「core wheel だけで組める pipeline」で
あって全量ではない（ADR 0065 決定 2 — モデル別 recipe は wheel の外）。family の移行が全部
終わった今、core の表は**空**なので、受理集合の正本はこの辞書だけになった。

    uv run python dist.py                                  # 既定 = anima（素の base 系）
    uv run python dist.py --pipeline anima-turbo           # 蒸留済み（LoRA 焼き込み）の別リポ
    uv run python dist.py --pipeline irodori
    uv run python dist.py --pipeline sbv2 --card-profile fn
    uv run python dist.py --pipeline sbv2 --card-profile jvnv \\
        --model F1 --model F2 --out ../../models/karume-sbv2-jvnv

置き場の既定（`--series` / `--out`）もここが渡す — リポの `outputs/series/` と `models/` は
repo topology で、core は綴りを持たない（ADR 0065 Consequences・`karume.dist` の同 MUST）。

NOTE: `**CORE_PIPELINES` の展開は残す — core が「表を受け取る側」であって「表を持たない側」で
はないことは変わっておらず、core wheel だけで組める pipeline が将来生えたら黙って合流する。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from _shared.paths import DIST_ROOT, SERIES_ROOT
from anima import distribution as anima_distribution
from birefnet import distribution as birefnet_distribution
from depth_anything import distribution as depth_anything_distribution
from gemma4 import distribution as gemma4_distribution
from irodori import distribution as irodori_distribution
from karume.dist import PIPELINES as CORE_PIPELINES
from karume.dist import DistError, Pipeline
from karume.dist import main as dist_main
from sbv2 import distribution as sbv2_distribution
from siglip2 import distribution as siglip2_distribution
from vowel_detector import distribution as vowel_detector_distribution

#: 受理集合の全量。並びは `--help` の並びでもあるので、既定を先頭に置く。
PIPELINES: Mapping[str, Pipeline] = {
    "anima": anima_distribution.BASE_PIPELINE,
    "anima-turbo": anima_distribution.TURBO_PIPELINE,
    "sbv2": sbv2_distribution.PIPELINE,
    "irodori": irodori_distribution.PIPELINE,
    "siglip2": siglip2_distribution.PIPELINE,
    "birefnet": birefnet_distribution.PIPELINE,
    "depth-anything": depth_anything_distribution.PIPELINE,
    "vowel-detector": vowel_detector_distribution.PIPELINE,
    "gemma4": gemma4_distribution.PIPELINE,
    **CORE_PIPELINES,
}

#: 旧 `karume dist`（引数なし）の UX をドライバ側で維持する。
DEFAULT_PIPELINE = "anima"


def default_out_dir(pipeline: Pipeline, models: Sequence[str]) -> Path:
    """`--out` 省略時の出力先（`models/<リポ名>/` = 1 ディレクトリ 1 HF リポ）。

    複数モデルのリポ名は**導出できない**（`karume-sbv2-jvnv` のようなファミリー名は命名の
    決定であって、モデル名の並びからは決まらない）ので、明示を求めて落とす。
    """
    if len(models) != 1:
        raise DistError(
            f"モデルを {len(models)} 個組む場合はリポ名を導出できない — --out で出力先を指定する"
        )
    return DIST_ROOT / pipeline.repo_name(models[0])


def main(argv: Sequence[str] | None = None) -> None:
    dist_main(
        argv,
        pipelines=PIPELINES,
        default_pipeline=DEFAULT_PIPELINE,
        default_out_dir=default_out_dir,
        default_series=SERIES_ROOT,
    )


if __name__ == "__main__":
    main()
