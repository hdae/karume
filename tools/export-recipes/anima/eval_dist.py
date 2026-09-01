"""視認評価専用の組み立て（**配布しない**）— i4 の量子化感度実験変種を手元で見るための台本。

素版 3 モデルの i4 は視認裁定で配布スキップになり（research
`2026-08-24-gptq-expansion-quality.md` §5）、{@link anima.distribution.ANIMA_MODELS} からも i4 席が
外れている。改善候補（block 内 adaLN + block 外を i8 格納へ回す変種 — `anima.export` の
`--i4-adaln-i8`）を A/B で見るには i4 席の載った配布形が要るが、その席を配布の受理集合へ戻す
判断はまだ無い。ここはその 1 点だけを埋める口で、**配布経路には 1 バイトも触らない**:

- 席は {@link anima.distribution.anima_plan} の `spec` 差し替えで戻す（i4 を足した spec を渡す
  だけ — `ANIMA_MODELS` は動かさない）。
- i4 系列は変種の綴り（{@link EVAL_SERIES}）から引く。配布条件で焼いた `*-i4-dyn` は
  **見に行かない** — 同じ席へ別の丸め方の系列が入る余地を作らない。
- 要求する丸め方式は {@link anima.distribution.ADALN_I8_CALIB_METHOD}。配布経路の
  `gptq` とは綴りが違うので、変種を `dist.py` で組もうとしても方式一致で落ちるし、逆に
  配布条件で焼いた系列をここへ渡しても落ちる（**両方向**で取り違えが止まる）。
- 出力先の既定は {@link EVAL_ROOT}（`outputs/bench/` 側 = 消して安全な席）。`models/` は
  配布形だけの場所（`_shared.paths` の DECIDED）なので、既定では 1 度も触らない。

    # ① 変種の i4 系列を焼く（GPTQ 校正込み・実測 ~3h）。系列名の綴りは {@link EVAL_SERIES}。
    uv run python -m anima.export --dtype i4 --dit-graph dyn --i4-adaln-i8 \\
        --model anima-v1.0 --out ../../outputs/series/anima-v1.0-i4-adaln8-dyn
    # ② 視認用に組む（既定 out = outputs/bench/karume-anima-v1.0-adaln8/<日付>_eval/）
    uv run python -m anima.eval_dist --model anima-v1.0

MUST: 出来上がった配布形を HF へ上げない・`models/` へ移さない。quant 席名は配布と同じ
`f16+dit4`（ADR 0074 の文法）のままで、**中身のバイトだけが違う** — 名前からは判別できないので、
置き場を分けることが唯一の区別になる。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from datetime import date
from functools import partial
from pathlib import Path

from _shared.paths import BENCH_ROOT, SERIES_ROOT
from anima.card import render_base_card
from anima.distribution import (
    ADALN_I8_CALIB_METHOD,
    ADALN_I8_TAG,
    ANIMA_BASE_MODEL_NAME,
    ANIMA_MODELS,
    ANIMA_QUANT_ABBREVIATIONS,
    OFFICIAL_NOTICE_MARKDOWN,
    anima_model,
    anima_plan,
    anima_sources,
    root_files,
)
from karume.dist import DistError, ModelPlan, Pipeline
from karume.dist import main as dist_main

#: 変種の i4 系列のディレクトリ名（`--series` の親からの相対）。`anima.export --i4-adaln-i8` の
#: 既定 out（`anima-i4-adaln8-dyn`）とはモデル名の有無だけが違う — 配布の系列と同じく、素版は
#: モデル名で分かれる（{@link anima.distribution.anima_sources} の綴りに合わせる）。
EVAL_SERIES = f"{{model}}-i4-{ADALN_I8_TAG}-dyn"

#: 評価用の組み立て先の親（`<EVAL_ROOT>/<リポ名>/<日付>_eval/`）。**`models/` へは置かない**
#: （配布形だけの場所 — `_shared.paths`）。ベンチ生成物なので消して安全な席に置く。
EVAL_ROOT = BENCH_ROOT

#: `--pipeline` の綴り（受理集合はこの 1 つきり — 配布の表〈`tools/export-recipes/dist.py`〉に
#: 混ぜないのは、あちらが「HF へ上げてよい pipeline の全量」だから）。
EVAL_PIPELINE = "anima-i4-adaln8-eval"


def eval_plan(series_dir: Path, model: str) -> ModelPlan:
    """視認評価用の計画を組む（i4 席を戻し、変種の系列と丸め方式を要求する）。

    MUST: 受理するのは**配布で i4 席を持たない** Anima モデルだけ（2026-09-01 の全モデル
    i4 なし裁定後は全員が該当）。将来 i4 席が配布へ戻ったモデルは `dist.py` の配布経路が
    そのまま使えるので、ここへ通すと「配布と同じ席名・違う中身」が理由も無く増える —
    下の storages 検査が落とす。
    """
    if model not in ANIMA_MODELS:
        raise DistError(f"モデル {model!r} は視認評価の対象外（対象: {', '.join(ANIMA_MODELS)}）")
    spec = anima_model(model)
    if "i4" in spec.storages:
        raise DistError(
            f"モデル {model!r} は既に配布で i4 席を持っている — 視認評価の口ではなく"
            " dist.py で組む（席が戻ったならこの台本の役目は終わっている）"
        )
    sources = anima_sources(series_dir, model)
    return anima_plan(
        replace(
            sources,
            transformer={
                **sources.transformer,
                "i4": series_dir / EVAL_SERIES.format(model=model),
            },
        ),
        model,
        spec=replace(
            spec,
            storages=(*spec.storages, "i4"),
            calib_method=ADALN_I8_CALIB_METHOD,
        ),
    )


PIPELINE = Pipeline(
    default_model=ANIMA_BASE_MODEL_NAME,
    repo_name=lambda model: f"karume-{model}-{ADALN_I8_TAG}",
    plan=eval_plan,
    card_profiles={"anima": partial(render_base_card, abbreviations=ANIMA_QUANT_ABBREVIATIONS)},
    # 法的テキストは配布形と同じものを入れる — 手元でしか開かないとはいえ、改変告知が
    # 抜けた配布形を作る練習にはしない（うっかり上げたときに落ちるのはここではない）。
    root_files=root_files(OFFICIAL_NOTICE_MARKDOWN),
)


def default_out_dir(pipeline: Pipeline, models: Sequence[str]) -> Path:
    """`--out` 省略時の出力先（`outputs/bench/<リポ名>/<日付>_eval/`）。

    複数モデルのリポ名は導出できない（`dist.py` の同名関数と同じ理由）— 明示を求めて落とす。
    """
    if len(models) != 1:
        raise DistError(
            f"モデルを {len(models)} 個組む場合はリポ名を導出できない — --out で出力先を指定する"
        )
    return EVAL_ROOT / pipeline.repo_name(models[0]) / f"{date.today().isoformat()}_eval"


def main(argv: Sequence[str] | None = None) -> None:
    dist_main(
        argv,
        pipelines={EVAL_PIPELINE: PIPELINE},
        default_pipeline=EVAL_PIPELINE,
        default_out_dir=default_out_dir,
        default_series=SERIES_ROOT,
    )


if __name__ == "__main__":
    main()
