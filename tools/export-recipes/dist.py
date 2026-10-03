"""リポジトリ用の dist ドライバ — core の組み立てエンジンに recipe 側 pipeline を合成する。

`karume.dist` が持つ {@link karume.dist.PIPELINES} は「core wheel だけで組める pipeline」で
あって全量ではない（ADR 0065 決定 2 — モデル別 recipe は wheel の外）。family の移行が全部
終わった今、core の表は**空**なので、受理集合の正本はこの辞書だけになった。

    uv run python dist.py                                  # 既定 = anima（公式 5 変種 — ADR 0087）
    uv run python dist.py --pipeline anima-extra           # 追加学習系（越境参照 — リリース時）
    uv run python dist.py --pipeline irodori
    uv run python dist.py --pipeline sbv2-fn               # FN 系（HF 公開は保留）
    uv run python dist.py --pipeline sbv2 \\
        --model F1 --model F2 --out ../../models/karume-sbv2-jvnv
    uv run python dist.py --pipeline siglip2 \\
        --model base --model so400m --out ../../models/karume-siglip2
    uv run python dist.py --pipeline lucida                # BiRefNet_HR の派生（別リポ）
    uv run python dist.py --pipeline umt5                  # umT5-XXL encoder（Wan の参照先）
    uv run python dist.py --pipeline wan \\
        --ref-repo hdae/karume-umt5-xxl --ref-revision <karume-umt5-xxl の main の SHA> \\
        --ref-dist ../../models/karume-umt5-xxl --ref-model xxl --ref-role text_encoder
                                                           # Wan2.1 T2V 1.3B（HF 公開は未）

置き場の既定（`--series` / `--out`）もここが渡す — リポの `outputs/series/` と `models/` は
repo topology で、core は綴りを持たない（ADR 0065 Consequences・`karume.dist` の同 MUST）。

## 仮の SHA の門（ADR 0119 追記「段 10d の設計」E）

越境参照の参照先が未公開の間は、開発用のミラーを仮の commit SHA {@link PLACEHOLDER_REVISION}
（40 桁の 0）で焼く（ローカルの取得元は `crossRepo` の明示 mapping で解き、revision を見ない）。
仮の SHA は形の上では正当な 40 桁 hex なので、hub の parse も core の組み立ても通す — 公開へ漏れると
HF からの取得が存在しない commit を引きに行く。そこでこのドライバは `--ref-revision` が仮の SHA の
ときに {@link ALLOW_PLACEHOLDER_FLAG} の明示を要求し、無ければ 1 バイトも書く前に落とす。公開の
焼き直し（docs/release-runbook.md §0 — bump の後に必ず焼き直す）は明示を付けないので、参照先の
実 SHA を渡さない限り通らない:

    uv run python dist.py --pipeline wan --ref-revision 0000000000000000000000000000000000000000 \\
        --allow-placeholder-ref --ref-repo … --ref-dist … --ref-model xxl --ref-role text_encoder

NOTE: `**CORE_PIPELINES` の展開は残す — core が「表を受け取る側」であって「表を持たない側」で
はないことは変わっておらず、core wheel だけで組める pipeline が将来生えたら黙って合流する。
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

from _shared.paths import DIST_ROOT, SERIES_ROOT
from anima import distribution as anima_distribution
from birefnet import distribution as birefnet_distribution
from depth_anything import distribution as depth_anything_distribution
from gemma4 import distribution as gemma4_distribution
from gemma4_qat import distribution as gemma4_qat_distribution
from irodori import distribution as irodori_distribution
from karume.dist import PIPELINES as CORE_PIPELINES
from karume.dist import DistError, Pipeline, build_parser
from karume.dist import main as dist_main
from sbv2 import distribution as sbv2_distribution
from siglip2 import distribution as siglip2_distribution
from vowel_detector import distribution as vowel_detector_distribution
from wan import distribution as wan_distribution
from wan import umt5_distribution

#: 受理集合の全量。並びは `--help` の並びでもあるので、既定を先頭に置く。
PIPELINES: Mapping[str, Pipeline] = {
    "anima": anima_distribution.OFFICIAL_PIPELINE,
    "anima-extra": anima_distribution.EXTRA_PIPELINE,
    "sbv2": sbv2_distribution.PIPELINE,
    # 声のファミリーはライセンスが違う（JVNV = CC BY-SA 4.0・FN = Booth の頒布条件）ので別席。
    "sbv2-fn": sbv2_distribution.FN_PIPELINE,
    "irodori": irodori_distribution.PIPELINE,
    "siglip2": siglip2_distribution.PIPELINE,
    "birefnet": birefnet_distribution.PIPELINE,
    # 派生は別リポ（ADR 0092 決定 1）— リポ直下の著作権表示が違うので Pipeline も別席。
    "lucida": birefnet_distribution.LUCIDA_PIPELINE,
    "depth-anything": depth_anything_distribution.PIPELINE,
    "vowel-detector": vowel_detector_distribution.PIPELINE,
    "gemma4": gemma4_distribution.PIPELINE,
    "gemma4-qat": gemma4_qat_distribution.PIPELINE,
    "wan": wan_distribution.PIPELINE,
    # Wan の text_encoder の参照先（ADR 0119 追記 D — 部品だけの別リポ・読む TS の家族は無い）。
    "umt5": umt5_distribution.PIPELINE,
    **CORE_PIPELINES,
}

#: 旧 `karume dist`（引数なし）の UX をドライバ側で維持する。
DEFAULT_PIPELINE = "anima"

#: 越境参照の参照先が未公開の間に焼く仮の commit SHA（ADR 0119 追記 E — 40 桁の 0）。
PLACEHOLDER_REVISION = "0" * 40

#: 仮の SHA で焼くことの明示（開発用のミラーだけが付ける — モジュール doc の「仮の SHA の門」）。
ALLOW_PLACEHOLDER_FLAG = "--allow-placeholder-ref"


def assert_placeholder_intent(revision: str | None, *, allowed: bool) -> None:
    """`--ref-revision` が仮の SHA なら、明示（{@link ALLOW_PLACEHOLDER_FLAG}）を要求する。

    MUST: 明示が無ければ fail loudly — 仮の SHA は 40 桁 hex の形を満たすので、hub の parse も
    core の組み立ても `verify_dist` も通し、公開した配布形が存在しない commit を指したまま据わる。
    """
    if revision == PLACEHOLDER_REVISION and not allowed:
        raise DistError(
            f"--ref-revision が仮の SHA（{PLACEHOLDER_REVISION}）— 公開する配布形は参照先の main の"
            f" 実 SHA で焼く。開発用のミラーとして焼くなら {ALLOW_PLACEHOLDER_FLAG} を明示する"
            "（docs/release-runbook.md §0）"
        )


def default_out_dir(pipeline: Pipeline, models: Sequence[str]) -> Path:
    """`--out` 省略時の出力先（`models/<リポ名>/` = 1 ディレクトリ 1 HF リポ）。

    リポ名は Pipeline の宣言（`Pipeline.repo_name` — ファミリーのリポ名。sbv2 も
    `karume-sbv2-jvnv` を宣言する）から引く。複数モデルでは `--out` の明示を求めて落とす —
    この関数は 1 モデルの宣言だけを引き、並べたモデルの宣言が揃うかの突合を持たない。
    """
    if len(models) != 1:
        raise DistError(
            f"モデルを {len(models)} 個組む場合はリポ名を導出できない — --out で出力先を指定する"
        )
    return DIST_ROOT / pipeline.repo_name(models[0])


def build_driver_parser() -> argparse.ArgumentParser:
    """core の parser に {@link ALLOW_PLACEHOLDER_FLAG} を足したもの（門が引数を読むための 1 本）。

    省略形は受けない（`allow_abbrev`）。明示の綴りは文字列の一致で取り除いてから core へ渡すので、
    省略形（`--allow-placeholder` など）を門だけが受けると、core が知らない引数として落とす —
    綴り切らせて、門と core が同じ引数を 1 通りに読む形にする。
    """
    parser = build_parser(
        PIPELINES, DEFAULT_PIPELINE, default_series=SERIES_ROOT, has_default_out_dir=True
    )
    parser.allow_abbrev = False
    parser.add_argument(
        ALLOW_PLACEHOLDER_FLAG,
        dest="allow_placeholder_ref",
        action="store_true",
        help=f"--ref-revision に仮の SHA（{PLACEHOLDER_REVISION}）を許す — 参照先が未公開の間の"
        "開発用ミラーだけ（公開する配布形には付けない）",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    arguments = list(sys.argv[1:] if argv is None else argv)
    args = build_driver_parser().parse_args(arguments)
    # 1 バイトも書く前に落とす（core の組み立ては staging を経て out_dir を丸ごと差し替える）。
    assert_placeholder_intent(args.ref_revision, allowed=args.allow_placeholder_ref)
    dist_main(
        [argument for argument in arguments if argument != ALLOW_PLACEHOLDER_FLAG],
        pipelines=PIPELINES,
        default_pipeline=DEFAULT_PIPELINE,
        default_out_dir=default_out_dir,
        default_series=SERIES_ROOT,
    )


if __name__ == "__main__":
    main()
