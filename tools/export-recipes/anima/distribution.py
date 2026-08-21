"""Anima の配布 recipe — 系列レイアウト・出力 path 表・quant 表・カードの選択（ADR 0065 決定 2）。

汎用の組み立てエンジン（配置・共有席の畳み込み・sha256・manifest・staging/swap・検証）は
`karume.dist` が持つ。ここが持つのは **Anima 固有の事実**だけ: どの系列ディレクトリから
何を拾い、配布形のどの path へ、どの dtype ラベルで並べ、どの quant を既定にするか。

公開面は {@link PIPELINE} 1 つ（`karume.dist.Pipeline`）— リポの dist ドライバ
（`tools/export-recipes/dist.py`）がこれを core の PIPELINES へ合成する。
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from anima.card import ATTRIBUTION_NOTICE, LORA_NAME, LORA_SHA256, LORA_SOURCE, render_model_card
from karume.dist import (
    Artifact,
    DistError,
    ModelPlan,
    Pipeline,
    WeightFiles,
    assert_model_name,
    assert_storage,
    assert_storage_absent,
    complete_quant_weights,
    sha256_file,
)

#: 既定のモデル名（= 単一モデルなら既定のリポ名でもある）。turbo LoRA を焼き込んだ配布物で
#: あることを名前に出す。
ANIMA_MODEL_NAME = "anima-turbo"

#: パイプライン契約（ADR 0041 §2 — モデル単位）。
ANIMA_PIPELINE = "anima/1"

#: モデル名に依らない系列（base の text 経路 / VAE と、tokenizer を書く台本の出力）。
ANIMA_BASE_SERIES = "anima-f16"
ANIMA_TOKENIZER_SERIES = "anima-demo"

#: 焼き込んだ LoRA の帰属を残すファイル（系列のターゲット直下）。系列レイアウトの綴りは
#: 読み手（ここ）が持ち、書き手（`anima/export.py`）はここから引く — 2 箇所で独立に動かさない。
LORA_PROVENANCE_FILE = "lora_provenance.json"

#: 上流の重みライセンス原文（この recipe の隣に逐語で置いてある）。配布は Derivative の
#: Distribution なので §3(a)（ライセンスのコピーを第三者へ提供する）が掛かる — 要約や
#: 書き換えでは条件を満たさないため、**1 バイトも変えずに**配布リポ直下の `LICENSE.md` として
#: 出す。`Path(__file__)` 基準で引くのは、cwd にも系列の置き場にも依存しないため。
LICENSE_SOURCE_PATH = Path(__file__).parent / "circlestone_license.txt"

#: 配布リポ直下の `NOTICE.md`。§3(b)（Attribution Notice の掲示）+ §3(d)(i)（改変した旨を
#: **Attribution Notice の中に**含める）+ §3(d)(iii)（公式製品と誤認させない）を 1 枚で満たす。
#: 逐語ブロックは {@link ATTRIBUTION_NOTICE}（`anima/card.py` が正本）で、残りは Karume 側の
#: 事実の記述。改変記載を独立節にせず Notice 節の内側へ置くのは §3(d)(i) の文言
#: （"include in the Attribution Notice"）に厳格に合わせるため。
NOTICE_MARKDOWN = (
    "\n".join(
        [
            "# Notice",
            "",
            "## Attribution Notice",
            "",
            ATTRIBUTION_NOTICE,
            "",
            "As required by the license, this Attribution Notice also states that the",
            "applicable CircleStone Model has been modified: this distribution is a Derivative",
            "of the CircleStone Anima model, modified as follows:",
            "",
            f"- The official {LORA_NAME} ({LORA_SOURCE}) was baked into the weights at export.",
            "- The weights were converted into the container format of the WebGPU inference"
            " runtime Karume",
            "  (a single safetensors file holding the weights plus an inference graph in"
            " `__metadata__`).",
            "- An int8-quantized series and an int4-quantized series of the transformer were added",
            "  alongside the f16 one.",
            "",
            "## Not an official product",
            "",
            "This is not an official product of CircleStone Labs LLC, and it is not endorsed,",
            "approved or validated by CircleStone Labs LLC.",
            "",
            "The full license text is distributed alongside this repository as LICENSE.md.",
        ]
    )
    + "\n"
)

#: 出力の相対 path（**モデルサブツリー内**）— 配置表と manifest が共有する 1 箇所。
#: 役割名でだけ引くので、綴りが 2 箇所で独立に動くことは起きない。
OUTPUT_PATHS: Mapping[str, str] = {
    "text_encoder": "text_encoder/model.safetensors",
    "text_conditioner": "text_conditioner/model.safetensors",
    "transformer_f16": "transformer/model.f16.safetensors",
    "transformer_i8": "transformer/model.i8.safetensors",
    "transformer_i4": "transformer/model.i4.safetensors",
    "rope_base": "transformer/rope_base.safetensors",
    "vae_decoder": "vae_decoder/model.safetensors",
    "tokenizer": "tokenizer/qwen2-tokenizer.json",
    "tokenizer_2": "tokenizer_2/t5-tokenizer.json",
}

#: quant 表（v1 の `presets` — ADR 0041 §3 で改名）。`session` の語彙は manifest 所有。
#: **dtype ラベルが 1 つしかない weights は書かない** — {@link complete_quant_weights} が
#: 完全写像へ埋める（写せば済む席を quant の数だけ複製しない）。
ANIMA_QUANTS: Mapping[str, Any] = {
    "f16": {"weights": {"transformer": "f16"}, "session": {}},
    "i8": {"weights": {"transformer": "i8"}, "session": {}},
    "w8a8": {"weights": {"transformer": "i8"}, "session": {"linearCompute": "i8a8"}},
    "w8a8-a8": {
        "weights": {"transformer": "i8"},
        "session": {"linearCompute": "i8a8", "attentionCompute": "i8a8"},
    },
    "w8a8-s16": {
        "weights": {"transformer": "i8"},
        "session": {
            "linearCompute": "i8a8",
            "attentionCompute": "i8a8",
            "attentionScoreStorage": "f16",
        },
    },
    # i4 常駐の 2 席（波 J-4a — 低 VRAM 席）。
    #
    # MUST: **`linearCompute` を宣言しない**。**理由は 2026-08-21 に入れ替わった**ので注意 —
    # 以前は「i8a8 の述語が i8 常駐を必要条件に含むので宣言しても 1 バイトも変わらない嘘の席に
    # なる」だったが、w4a8 の実装（ADR 0076）で i4 常駐も整数内積の経路に乗るようになった。
    # 今の理由は**品質**: 宣言すると linear の活性が per-token i8 になり、実 GPU の画で
    # 「細部に破綻・線がラフ」というユーザー視認裁定が出た（2026-08-21・研究記録
    # `docs/research/2026-08-21-anima-i4-seat-speed.md` §6）。速度は 1,640 → 955 ms/step と
    # 大きく戻るが、この席の存在理由は**サイズと VRAM** であって速度ではない（速度が要るなら
    # 既定の `w8a8-s16` が 823 ms/step で上）。attention 側の 2 つは重みスロットを見ないので
    # i4 常駐でもそのまま効き、視認でも劣化は出ていないので宣言する。
    "w4": {"weights": {"transformer": "i4"}, "session": {}},
    "w4-a8-s16": {
        "weights": {"transformer": "i4"},
        "session": {"attentionCompute": "i8a8", "attentionScoreStorage": "f16"},
    },
    "f16-c16": {
        "weights": {"transformer": "f16"},
        "session": {"linearCompute": "f16", "attentionCompute": "f16"},
        "gpuFeatures": {"shaderF16": True},
    },
}

ANIMA_DEFAULT_QUANT = "w8a8-s16"

#: パイプライン所有の設定（hub は素通し — ADR 0041 §2）。値は移行元の実装定数と一致する:
#: `shift` / `numTrainTimesteps` は sampler の `ANIMA_SHIFT` / `ANIMA_NUM_TRAIN_TIMESTEPS`
#: （エクスポータ側の `SHIFT` / `NUM_TRAIN_TIMESTEPS` = scheduler_config.json と同値）、
#: `steps` / `guidanceScale` は turbo 既定（8 step / cfg 1 — ADR 0038 Examples が正。品質目視
#: ゲート・最終ベンチ・PNG 参照 sha の採取は全て 8 step で行われており、配布既定はそれに揃える。
#: 移行元 CLI の 10 は検証履歴を持たない値）。`negativePrompt` は既定ネガティブプロンプト。
#: `resolution` だけは移行元 CLI の既定（512）を採らない — あちらの 512 は「静的資産の最小」
#: であって推奨値ではなく、配布形は S 形 1 本（ADR 0038 §4）で解像度に依存しない。配布の
#: 推奨既定は ADR 0038 Examples のとおり 1024²。
ANIMA_PIPELINE_CONFIG: Mapping[str, Any] = {
    "scheduler": {"shift": 3, "numTrainTimesteps": 1000},
    "defaults": {
        "steps": 8,
        "guidanceScale": 1,
        "resolution": {"width": 1024, "height": 1024},
        "negativePrompt": "low quality, worst quality, blurry, bad anatomy, jpeg artifacts",
    },
}

#: 各役割の safetensors ヘッダに**要求する格納 dtype**（存在検査）。実測の事故が根拠:
#: f16 系列のつもりで `--dtype` を付け忘れた素の F32 資産は、組み立て・ロード・実行の全てを
#: 通って**PNG の参照一致まで露見しなかった**。格納形は series ディレクトリ名でなくヘッダが正。
#: f16 系列は fake-quant 対象だけが F16 になる（norm/bias 等は F32 のまま）ので「F16 を含む」
#: を要求する。rope_base（F32 のみ）と tokenizer（JSON）はここに載せない。
#: i4 系列は**混成**（F32 + I8 + I4 が同居する）なので **I4 を要求する** — {@link assert_storage}
#: は「要求 dtype がヘッダに在る」を見るので、I8 を要求すると i8 系列が i4 席へ入っても素通りし、
#: 席の取り違えが沈黙する（sbv2 の i4 席と同じ規律）。
STORAGE_REQUIREMENTS: Mapping[str, str] = {
    "text_encoder": "F16",
    "text_conditioner": "F16",
    "transformer_f16": "F16",
    "transformer_i8": "I8",
    "transformer_i4": "I4",
    "vae_decoder": "F16",
}

#: 各役割の safetensors ヘッダに**あってはならない**格納 dtype（{@link assert_storage_absent}）。
#: 存在検査だけでは**圧縮席どうしの取り違え**が素通りする — i4 系列は混成で、既定格納が i8
#: （`anima/export.py` の `BASE_WEIGHT_DTYPES`）なので **必ず I8 を含む**。したがって i4 系列を
#: `transformer_i8` へ挿し込む取り違えは「I8 を含む」を満たしてしまい、組み立ても verify_dist も
#: ロードも通る。実害は既定 quant `w8a8-s16` に出る: 宣言した `linearCompute: "i8a8"` の述語は
#: i8 常駐を必要条件に含むので、常駐が i4 だと fail loudly せず f32 計算経路へ黙って落ちる
#: （`ANIMA_QUANTS` の w4 席が「嘘の席」として避けた挙動が、既定席で沈黙して起きる）。
#: MUST: 禁止は**役割ごとに集合**で持つ（1 つだけだと 4 本目の系列が生えた日に、名指ししなかった
#: ほうが黙って素通りする — irodori と同じ規律）。f16 席は I8 / I4 の不在で二重に締まる。
ANIMA_STORAGE_FORBIDDEN: Mapping[str, tuple[str, ...]] = {
    "transformer_f16": ("I8", "I4"),
    "transformer_i8": ("I4",),
}

#: weights の宣言（dtype ラベル → 役割名）。ラベルは**格納 dtype 語彙**で、
#: {@link STORAGE_REQUIREMENTS} が要求する格納形と 1:1（ADR 0041 §3）。`i4` は**混成の系列**を
#: 指すラベルで、実体は「i4 適格な重みが i4 group32・残りは i8」（`anima/export.py`）。
ANIMA_WEIGHTS: Mapping[str, Mapping[str, WeightFiles]] = {
    "text_encoder": {"f16": WeightFiles("text_encoder")},
    "text_conditioner": {"f16": WeightFiles("text_conditioner")},
    "transformer": {
        "f16": WeightFiles("transformer_f16", {"rope_base": "rope_base"}),
        "i8": WeightFiles("transformer_i8", {"rope_base": "rope_base"}),
        "i4": WeightFiles("transformer_i4", {"rope_base": "rope_base"}),
    },
    "vae_decoder": {"f16": WeightFiles("vae_decoder")},
}

#: assets の宣言（quant 選択に依存しない無条件ファイル — ADR 0041 §3）。
ANIMA_ASSETS: Mapping[str, str] = {"tokenizer": "tokenizer", "tokenizer_2": "tokenizer_2"}


@dataclass(frozen=True)
class AnimaSources:
    """組み立ての入力となる系列ディレクトリ群。

    テキスト経路と VAE は DiT の格納 dtype に依らないので f16 系列 1 本を共有する
    （ADR 0019）。transformer だけが f16 / i8 / i4 の 3 系列に分かれる。
    """

    transformer_f16: Path
    transformer_i8: Path
    transformer_i4: Path
    base: Path
    tokenizers: Path


def anima_sources(series_dir: Path, model: str = ANIMA_MODEL_NAME) -> AnimaSources:
    """系列の親ディレクトリ（`outputs/series/`）から Anima の 5 系列を引く。

    モデル名は transformer の 3 系列にだけ掛かる — base（text 経路 / VAE）と tokenizer は
    素のアーキテクチャ側の出力で、turbo かどうかに依らない。
    """
    return AnimaSources(
        transformer_f16=series_dir / f"{model}-f16-dyn",
        transformer_i8=series_dir / f"{model}-i8-dyn",
        transformer_i4=series_dir / f"{model}-i4-dyn",
        base=series_dir / ANIMA_BASE_SERIES,
        tokenizers=series_dir / ANIMA_TOKENIZER_SERIES / "text",
    )


def transformer_series(sources: AnimaSources) -> tuple[Path, ...]:
    """格納 dtype 別の transformer 系列（**系列横断の突合はこの 1 本から引く** MUST）。

    rope 素表のバイト同一検査と LoRA 帰属の突合はどちらも「全系列を舐める」検査で、列挙を
    2 箇所に持つと格納席が増えた日に片方だけ更新される — 網から漏れた系列は検査を素通りし、
    どちらの綻びも実行時には沈黙する（幾何違いは絵だけ壊れ、帰属違いは README だけが嘘になる）。
    """
    return (sources.transformer_f16, sources.transformer_i8, sources.transformer_i4)


def shared_rope_base(sources: AnimaSources) -> Path:
    """全 transformer 系列の rope 素表がバイト同一であることを確かめ、1 本化する元を返す。

    MUST: `rope_base.safetensors` は f16 / i8 / i4 の各系列に同名で並ぶ。全てのバイト同一を
    sha256 で確かめてから 1 本化する — 食い違ったまま 1 つを選ぶと、選ばれなかった系列の
    quant が「別の幾何の rope 表で走る」形になり、ロードも実行も通って絵だけが静かに壊れる。
    """
    candidates = [
        series / "transformer" / "rope_base.safetensors" for series in transformer_series(sources)
    ]
    for path in candidates:
        if not path.is_file():
            raise DistError(f"組み立ての入力が無い: {path}")
    digests = {path: sha256_file(path) for path in candidates}
    if len(set(digests.values())) != 1:
        listing = "\n".join(f"  {digest}  {path}" for path, digest in digests.items())
        raise DistError(
            "rope_base.safetensors が系列間でバイト同一でない — 1 本化できない。"
            f"どちらが正かはここでは決められないので組み立てを止める:\n{listing}"
        )
    return candidates[0]


def assert_lora_provenance(sources: AnimaSources) -> None:
    """transformer 系列が記録した LoRA の sha256 が、カードの宣言と一致することを確かめる。

    MUST: カードの帰属節（`anima/card.py` の `LORA_SHA256`）は HF に公開される事実なのに、
    融合後の重みからは焼いた LoRA を復元できない。突き合わせが無いと、LoRA を差し替えて
    再エクスポートしても古い / 誤った sha256 がそのまま印字される — 値は 64 桁 hex として
    形式が妥当なので `verify_dist` の構造検査も通り、**沈黙する**。「別々の台本が持つ同じ
    事実は組み立て時に必ず突き合わせる」（rope_base のバイト同一検査と同じ規律）。
    """
    for series in transformer_series(sources):
        path = series / "transformer" / LORA_PROVENANCE_FILE
        if not path.is_file():
            raise DistError(
                f"焼き込んだ LoRA の記録が無い: {path}"
                "（`python -m anima.export --lora …` で再エクスポートすると書かれる）"
            )
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except ValueError as cause:
            raise DistError(f"LoRA の記録を解析できない: {path} — {cause}") from cause
        digest = record.get("sha256") if isinstance(record, dict) else None
        if digest != LORA_SHA256:
            raise DistError(
                f"焼き込んだ LoRA がカードの宣言と違う: {path} は {digest!r}、"
                f"anima/card.py は {LORA_SHA256!r} — どちらが正かはここでは決められない"
            )


def anima_placements(sources: AnimaSources) -> dict[str, Path]:
    """役割名 → 出所のファイル。出力の path は {@link OUTPUT_PATHS} が持つ。

    この表に無いものは出力へ入らない（`io.*.safetensors` を落とす仕掛けはこれで足りる）。
    """
    return {
        "text_encoder": sources.base / "text_encoder" / "model.safetensors",
        "text_conditioner": sources.base / "text_conditioner" / "model.safetensors",
        "transformer_f16": sources.transformer_f16 / "transformer" / "model.safetensors",
        "transformer_i8": sources.transformer_i8 / "transformer" / "model.safetensors",
        "transformer_i4": sources.transformer_i4 / "transformer" / "model.safetensors",
        "rope_base": shared_rope_base(sources),
        "vae_decoder": sources.base / "vae_decoder" / "model.safetensors",
        "tokenizer": sources.tokenizers / "qwen2-tokenizer.json",
        "tokenizer_2": sources.tokenizers / "t5-tokenizer.json",
    }


def anima_plan(sources: AnimaSources, model: str = ANIMA_MODEL_NAME) -> ModelPlan:
    """Anima 1 モデルぶんの計画を組む（検査はここで全部済ませる — 1 バイトも書かない）。"""
    assert_model_name(model)
    assert_lora_provenance(sources)
    placements = anima_placements(sources)
    for role, source in placements.items():
        assert_storage(role, source, STORAGE_REQUIREMENTS)
        assert_storage_absent(role, source, ANIMA_STORAGE_FORBIDDEN)
    return ModelPlan(
        name=model,
        pipeline=ANIMA_PIPELINE,
        artifacts={
            role: Artifact(OUTPUT_PATHS[role], source=source) for role, source in placements.items()
        },
        weights=ANIMA_WEIGHTS,
        assets=ANIMA_ASSETS,
        quants=complete_quant_weights(ANIMA_WEIGHTS, ANIMA_QUANTS),
        default_quant=ANIMA_DEFAULT_QUANT,
        pipeline_config=ANIMA_PIPELINE_CONFIG,
    )


def anima_dist_plan(series_dir: Path, model: str) -> ModelPlan:
    """`--series` の親から Anima 1 モデルの計画を組む（CLI のディスパッチ先）。"""
    return anima_plan(anima_sources(series_dir, model), model)


def root_files() -> dict[str, str]:
    """配布リポ直下へ入れる法的テキスト（`karume.dist.Pipeline.root_files`）。

    ライセンス原文は recipe に置いた現物（{@link LICENSE_SOURCE_PATH}）を**逐語で**読む —
    ここで整形や差し替えをすると §3(a) の「このライセンスのコピー」ではなくなる。
    """
    return {
        "LICENSE.md": LICENSE_SOURCE_PATH.read_text(encoding="utf-8"),
        "NOTICE.md": NOTICE_MARKDOWN,
    }


#: `--pipeline anima` の 1 行（ドライバが core の PIPELINES へ合成する）。
PIPELINE = Pipeline(
    default_model=ANIMA_MODEL_NAME,
    # `karume-` prefix はリポ名裁定（2026-08-09）— HF org の代わりの名前空間。
    repo_name=lambda model: f"karume-{model}",
    plan=anima_dist_plan,
    # 帰属は 1 通りだけ（LoRA を焼いた base 1 本）— 選択肢が無いので省略で通る。
    card_profiles={"anima": render_model_card},
    # 上流ライセンスの再配布条件（§3）は配布リポ 1 つに掛かるので、読みも組み立ての回数に
    # よらず**ここで 1 回**。
    root_files=root_files(),
)
