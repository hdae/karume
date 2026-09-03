"""SBV2 配布形のモデルカード（`README.md`）— manifest から機械導出する純関数。

汎用の描画部品（frontmatter・モデル一覧・quant 表・節の組み立て）は
`karume.modelcard` が持つ。ここが持つのは **SBV2 固有の事実**だけ: 帰属（声の出所・
ライセンス・引用・再配布する text encoder）と、この pipeline のカードに何を書くか。

**帰属はテンプレートと別の軸**（{@link Sbv2CardProfile}）。同じ SBV2 のテンプレートでも、
どのファミリーの重みを配るかで出所・ライセンス・引用が丸ごと変わるので、そこだけを
プロファイルに分けて**呼び出し側に明示させる**。

MUST: **数値・ダウンロード量・quant 表・dtype ラベル・スタイル表・話者表は 1 つ残らず manifest
から導出する**（`karume.modelcard` の同 MUST がそのまま掛かる）。ここが持ってよい定数は、
manifest に**存在しない事実**だけ。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import partial
from typing import Any

from karume.modelcard import (
    CardMetadata,
    default_model,
    from_pretrained,
    frontmatter,
    knob,
    model_sections,
    models,
    quants,
    render,
    require_pipeline,
)

#: このテンプレートが説明できるパイプライン契約（ADR 0041 §2 — モデル単位）。
SBV2_SUPPORTED_PIPELINE = "sbv2/1"

SBV2_PIPELINE_TAG = "text-to-speech"

#: アーキテクチャ（どのファミリーも JP-Extra 系 — 違うのは重みの出所であって形ではない）。
SBV2_ARCHITECTURE = "Style-Bert-VITS2 JP-Extra"

#: `text_encoder` の素になった日本語 BERT（`deberta/export.py` の `MODEL_ID`）。manifest は
#: 役割名しか持たないので、帰属はカード側が負う。ライセンスは実地確認（2026-08-07）。
#: **どのファミリーでも同じ 1 本**を再配布するので、プロファイルの席ではなくここが持つ。
SBV2_TEXT_ENCODER_MODEL = "ku-nlp/deberta-v2-large-japanese-char-wwm"
SBV2_TEXT_ENCODER_LICENSE = "cc-by-sa-4.0"

#: 使い方スニペットのデモ文（日本語 TTS の入力なので日本語のまま — CLAUDE.md の言語規約）。
SBV2_DEMO_TEXT = "こんにちは、これはテストです。"

#: quant ごとの**丸め方**（manifest に無い事実 — 表が持つのは「どの席がどの格納 dtype か」まで
#: で、その dtype をどう作ったか〈per-tensor RTN / GPTQ 校正 / group-32〉は台本の知識）。
#: 値は箇条 1 つぶんの行の並び。先頭行は quant 名に続く本文で、2 行目以降は Markdown の
#: 継続行なので自分でインデントを持つ（`attribution` と同じ「行そのもの」の持ち方）。
#:
#: 既定マークは書かない — 既定は manifest の `defaultQuant` から引く（{@link _sbv2_quants}）。
#: ここに書くと、既定が動いたときに表と説明が別々に嘘をつく。
SBV2_QUANT_ROUNDING: Mapping[str, tuple[str, ...]] = {
    "i8": ("every weight in `i8`, rounded per tensor (plain RTN).",),
    "i8+bert4": (
        "`text_encoder` in `i4` group-32: its linear layers rounded with **GPTQ",
        "  calibration** (a 48-sentence Japanese corpus), its embedding table plain RTN.",
        "  `front` / `voice` stay `i8`.",
    ),
    "i4": (
        "the same text encoder as `i8+bert4`, plus `front` / `voice` linear and conv1d",
        "  weights in `i4` group-32 with plain RTN. Smallest download and fastest warm",
        "  synthesis; the output is very close to `i8+bert4`, with slightly lower tension",
        "  than the source checkpoint's unquantized f32 reference.",
    ),
}


@dataclass(frozen=True)
class Sbv2CardProfile:
    """SBV2 カードの**帰属プロファイル** — 声のファミリーごとに違う「manifest に無い事実」。

    SBV2 のテンプレートは 1 つでも、帰属（出所・ライセンス・再配布の条件・引用）はファミリー
    ごとに**別の法的事実**になる。決め打ちのまま別ファミリーのリポへ描くと、表も使い方も
    正しいのに帰属だけが前のファミリーのまま残る — 配ってからでないと誰も気づけない誤りなので、
    ここが席として分けて持つ（選択は {@link SBV2_CARD_PROFILES} 経由で**明示**）。

    `attribution` が行の並びそのものなのは、この席に入るのが導出できない散文（ライセンス条項の
    要約・引用・再配布条件）だから。機械的に組める 2 行（Voices / Architecture）だけは
    {@link _sbv2_base_weights} が `source_dirs` / `source_version` から組む。
    """

    metadata: CardMetadata
    title: str
    #: 出所リポジトリの中で、この配布形の声になったディレクトリ群。
    source_dirs: tuple[str, ...]
    #: 出所の `config.json` が名乗る version 文字列。
    source_version: str
    #: Voices / Architecture に続く帰属の箇条（Markdown の行そのもの）。
    attribution: tuple[str, ...]


# ---- FN 系（単一モデル） -------------------------------------------------

#: ライセンス（実地確認 2026-08-20）: 出所の HF リポジトリは **license を宣言しておらず、
#: モデルカードも無い**。利用条件が書かれているのは作者の Booth 頒布ページなので、SPDX
#: 識別子は当てられず `other` を採り、`license_link` はそのページを指す。
#: DECIDED: 帰属は**最小記述**に留める（「Booth のこのモデルを変換した」程度 — 条件の引用や
#: 頒布者の詳細は書かない。2026-08-20 ユーザー裁定・FN の HF 公開自体も保留中 — backlog）。
SBV2_FN_METADATA = CardMetadata(
    pipeline_tag=SBV2_PIPELINE_TAG,
    base_model=("rufflet17/voice_models",),
    base_model_relation="quantized",
    license="other",
    license_name="upstream-distribution-terms",
    license_link="https://booth.pm/ja/items/6695672",
    tags=("text-to-speech", "webgpu", "japanese"),
)

SBV2_FN_PROFILE = Sbv2CardProfile(
    metadata=SBV2_FN_METADATA,
    title="Style-Bert-VITS2 — Karume",
    source_dirs=("FN/",),
    source_version="2.6.1-JP-Extra",
    attribution=(
        "- **Source**: converted from the author's Style-Bert-VITS2 voice-model set distributed",
        f"  on [Booth]({SBV2_FN_METADATA.license_link}) so it runs on this stack. See that page",
        "  for the distribution terms.",
        f"- **Text encoder**: [{SBV2_TEXT_ENCODER_MODEL}]"
        f"(https://huggingface.co/{SBV2_TEXT_ENCODER_MODEL}),",
        f"  licensed **{SBV2_TEXT_ENCODER_LICENSE}** (as of retrieval). It is redistributed here",
        "  in the container format as the `text_encoder` component, so that license travels with",
        "  this repository too.",
    ),
)


# ---- JVNV 系（ファミリー） -----------------------------------------------

#: ライセンス（実地確認 2026-08-09 — https://huggingface.co/litagin/style_bert_vits2_jvnv）:
#: 出所のリポジトリは「ライセンスは JVNV コーパスの cc-by-sa-4.0 を引き継ぐ」と明言している。
#: SPDX 標準タグが当たるので `license_name` / `license_link` は持たない（HF が識別子から解決
#: する席で、独自名を名乗らせると「名前の無い独自ライセンス」に読める）。
#: `base_model` に text encoder を併記するのは、この配布形が**両方を再配布している**ため。
#: 両者とも cc-by-sa-4.0 なので、SA（同一ライセンス継承）はリポジトリ全体で矛盾しない。
SBV2_JVNV_METADATA = CardMetadata(
    pipeline_tag=SBV2_PIPELINE_TAG,
    base_model=("litagin/style_bert_vits2_jvnv", SBV2_TEXT_ENCODER_MODEL),
    base_model_relation="quantized",
    license="cc-by-sa-4.0",
    tags=("text-to-speech", "webgpu", "japanese"),
)

#: CC BY-SA 4.0 の条文（BY の「ライセンス URL を示す」を、カード自身が満たすための 1 本）。
SBV2_JVNV_LICENSE_URL = "https://creativecommons.org/licenses/by-sa/4.0/"

SBV2_JVNV_PROFILE = Sbv2CardProfile(
    metadata=SBV2_JVNV_METADATA,
    title="Style-Bert-VITS2 JVNV — Karume",
    source_dirs=("jvnv-F1-jp/", "jvnv-F2-jp/", "jvnv-M1-jp/", "jvnv-M2-jp/"),
    source_version="2.0-JP-Extra",
    attribution=(
        f"- **Terms**: **[CC BY-SA 4.0]({SBV2_JVNV_LICENSE_URL})**, inherited from the JVNV"
        " corpus the",
        "  voices were trained on — the source repository states the corpus license carries over",
        "  to the models. Redistribution here keeps it: credit the authors, name the source and",
        "  the license URL, state that the weights were **converted to another format and"
        " quantized**",
        "  to `i8` (`i4` for the group-quantized weights),",
        "  license any derivative work under CC BY-SA 4.0 as well, and impose no further",
        "  restrictions. There is no NonCommercial and no NoDerivatives clause — commercial use",
        "  and modification are both allowed.",
        "- **Changes made here**: conversion into the Karume container format and **quantization**",
        "  of the weights — the voices in `f16` / `i8` / mixed, the text encoder in `i8` / mixed,",
        "  where the mixed form stores the group-quantizable weights as `i4` and everything else",
        "  as `i8` (the quant table below says which storage each quant selects). No retraining,",
        "  no fine-tuning — the voices are the source checkpoints in a different storage form.",
        f"- **Text encoder**: [{SBV2_TEXT_ENCODER_MODEL}]"
        f"(https://huggingface.co/{SBV2_TEXT_ENCODER_MODEL}),",
        f"  licensed **{SBV2_TEXT_ENCODER_LICENSE}** (as of retrieval). It is redistributed here",
        "  in the container format as the `text_encoder` component — the same license as the",
        "  voices, so the share-alike term is consistent across everything in this repository.",
        "- **Training data**: the JVNV corpus, licensed CC BY-SA 4.0. Detai Xin, Junfeng Jiang,",
        "  Shinnosuke Takamichi, Yuki Saito, Akiko Aizawa, Hiroshi Saruwatari,",
        # 論文題は 1 行に収める（折り返すと、題での検索が本文にあるのに当たらなくなる）。
        "  *JVNV: A Corpus of Japanese Emotional Speech with Verbal Content and"
        " Nonverbal Expressions*,",
        "  [arXiv:2310.06072](https://arxiv.org/abs/2310.06072). Corpus page:",
        "  <https://sites.google.com/site/shinnosuketakamichi/research-topics/jvnv_corpus>",
        "- **Training implementation**:"
        " [Style-Bert-VITS2](https://github.com/litagin02/Style-Bert-VITS2)",
        "  (AGPL-3.0). Karume's runtime contains **none of that code**: it is an independent",
        "  implementation that reads these weights from its own container format, so the AGPL",
        "  terms govern the training implementation, not the runtime that plays these files.",
    ),
)

#: 選べる帰属プロファイル。**既定は置かない** — 省略時に片方を黙って選ぶと、新しいファミリーを
#: 配るたびに前のファミリーの帰属が沈黙で再発する（`karume.dist` が明示を要求する）。
SBV2_CARD_PROFILES: Mapping[str, Sbv2CardProfile] = {
    "fn": SBV2_FN_PROFILE,
    "jvnv": SBV2_JVNV_PROFILE,
}


def _sbv2_overview(manifest: Mapping[str, Any]) -> list[str]:
    return [
        "## What is this",
        "",
        f"A Japanese text-to-speech distribution: **{SBV2_ARCHITECTURE}** voices converted into",
        "the WebGPU inference runtime **Karume**'s container format (a single safetensors file =",
        "weights + a graph JSON embedded in `__metadata__`). Runs as-is in the browser and in"
        " Deno.",
        "",
        "- The acoustic chain is shipped as fused graphs: `text_encoder` (a Japanese DeBERTa),",
        "  `front` (phoneme encoder + duration predictors) and `voice` (flow + HiFi-GAN decoder).",
        "- Style and speaker are **looked up at run time** from the shipped tables — the names in",
        "  the tables below index the rows of `style_vectors` / `speaker_embeddings`.",
        "- Not readable by Style-Bert-VITS2 (it's a different container with an embedded graph);"
        f" the reader is a pipeline that implements `{SBV2_SUPPORTED_PIPELINE}`.",
        f"- Exporter used for the conversion: `{manifest['generator']}`. The distribution manifest"
        f" is `karume.json` (`{manifest['format']}`).",
    ]


def _sbv2_base_weights(profile: Sbv2CardProfile) -> list[str]:
    """帰属節。声の出所リポジトリは `base_model` の先頭（= 重みの出所そのもの）から引く。"""
    base_model = profile.metadata.base_model[0]
    dirs = " / ".join(f"`{name}`" for name in profile.source_dirs)
    return [
        "## Base weights and attribution",
        "",
        "Converted into the container format — the original checkpoints are not distributed here.",
        "",
        f"- **Voices**: {dirs} of [{base_model}](https://huggingface.co/{base_model})",
        f"- **Architecture**: {SBV2_ARCHITECTURE} (`version: {profile.source_version}`)",
        *profile.attribution,
    ]


def _sbv2_quants(model: Mapping[str, Any], *, abbreviations: Mapping[str, str]) -> list[str]:
    """汎用の quant 表に、SBV2 の**丸め方**の備考を足した節。

    表は「どの席がどの格納 dtype か」しか言えない（manifest がそこまでしか持たない）ので、
    同じ `i4` でも GPTQ 校正付きか素の RTN かが読めない — 既定が品質で選ばれている理由が
    カードから消える。備考は**このモデルが宣言している quant だけ**に付ける（`_sbv2_knob` と
    同じ規律 — 配布形が持たない席を説明すると、カードが配れる値を超えて喋る）。

    表示欄（`label` / `description`）とは責務が別 — あちらは「席 1 つが何か」を manifest から
    出す 1 行で、こちらは「その格納をどう作ったか」という台本側の知識（ADR 0075 決定 5）。
    """
    default = model["defaultQuant"]
    table = quants(model, abbreviations=abbreviations)
    notes: list[str] = []
    for name in model["quants"]:
        rounding = SBV2_QUANT_ROUNDING.get(name)
        if rounding is None:
            continue
        mark = " (default)" if name == default else ""
        head, *rest = rounding
        notes += [f"- `{name}`{mark} — {head}", *rest]
    if not notes:
        return table
    return [*table, "", "How the stored weights were rounded:", "", *notes]


def _name_id_rows(table: Mapping[str, int]) -> list[str]:
    """`名前 → ID` の表を manifest の並びのまま行にする（ID は配る表の行番号そのもの）。"""
    return [f"| `{name}` | {identifier} |" for name, identifier in table.items()]


def _sbv2_styles(model: Mapping[str, Any]) -> list[str]:
    """スタイル一覧（利用者が `style` に何を渡せるかをカードだけで知れるようにする節）。"""
    path = model["assets"]["style_vectors"]["path"]
    return [
        "### Styles",
        "",
        "| Style | ID |",
        "| ----- | -- |",
        *_name_id_rows(model["pipelineConfig"]["styles"]),
        "",
        f"`style` takes one of these names — the ID is the row it selects in `{path}`.",
        "`styleWeight` blends between the average style (`0`) and the named one (`1`).",
    ]


def _sbv2_speakers(model: Mapping[str, Any]) -> list[str]:
    """話者一覧（`speaker` の受理集合 — スタイルと同じく行番号の対応表）。"""
    path = model["assets"]["speaker_embeddings"]["path"]
    return [
        "### Speakers",
        "",
        "| Speaker | ID |",
        "| ------- | -- |",
        *_name_id_rows(model["pipelineConfig"]["speakers"]),
        "",
        f"`speaker` takes one of these names — the ID is the row it selects in `{path}`.",
    ]


def _sbv2_knob(defaults: Mapping[str, Any], key: str, note: str) -> list[str]:
    """`generate()` の optional ノブ 1 つをコメント行にする（既定が manifest に在るものだけ）。

    manifest が持たないキーは**綴らない** — `defaults` に無いノブを勧めると、カードが
    「この配布形で選べる値」を超えて喋る（`### Defaults` が同じ理由で `defaults` の写しを
    持たないのと同じ判断）。
    """
    if key not in defaults:
        return []
    value = defaults[key]
    literal = f'"{value}"' if isinstance(value, str) else f"{value}"
    return [f"  // {key}: {literal}, // default — {note}"]


def _sbv2_usage(manifest: Mapping[str, Any], repo: str) -> list[str]:
    """Usage 例の方針: 動く最小形は生かし、**普通のユースケースで使いそうな optional は
    コメントアウトで併記**する（選べる値も同じ行のコメントに列挙 — manifest から機械導出する
    ので、声 / スタイル / quant が増えれば列挙も追従する）。読者がコメントを外すだけで次の
    一歩へ進める形。

    NOTE: `fromAssets` は**案内しない**（2026-08-29 裁定）。分割配布形も読めるようになった
    （X2-101）が、あちらはバイト列を自分で持っている前提のローカルデバッグ向けの面で、HF から
    使う読者の普通の入口は `fromPretrained`。両方を並べると「どちらを使うのか」を読者に
    判断させることになる。
    """
    model_name = manifest["defaultModel"]
    model = default_model(manifest)
    quant = model["defaultQuant"]
    config = model["pipelineConfig"]
    defaults = config["defaults"]
    model_names = " / ".join(sorted(manifest["models"]))
    quant_names = " / ".join(sorted(model["quants"]))
    style_names = " / ".join(sorted(config["styles"]))
    speaker_names = " / ".join(sorted(config["speakers"]))
    return [
        "## Usage",
        "",
        "```ts",
        'import { encodeWav, Sbv2Pipeline } from "jsr:@karume/models";',
        "",
        *from_pretrained(
            "Sbv2Pipeline",
            repo,
            [
                f'  // model: "{model_name}", // default — available: {model_names}',
                f'  // quant: "{quant}", // default — available: {quant_names}',
            ],
        ),
        "",
        "const audio = await pipeline.generate({",
        f'  text: "{SBV2_DEMO_TEXT}",',
        "",
        "  // Voice — the names below come from the tables further down; every model in",
        "  // this repository brings its own set:",
        *_sbv2_knob(defaults, "style", f"available: {style_names}"),
        *_sbv2_knob(defaults, "styleWeight", "0 = the average style, 1 = the named one"),
        *_sbv2_knob(defaults, "speaker", f"available: {speaker_names}"),
        "",
        "  // Delivery:",
        *_sbv2_knob(defaults, "lengthScale", "larger is slower"),
        *_sbv2_knob(defaults, "sdpRatio", "1 = stochastic duration, 0 = deterministic"),
        *_sbv2_knob(defaults, "noiseScale", "sampling noise on z_p"),
        *_sbv2_knob(defaults, "noiseScaleW", "sampling noise inside the stochastic predictor"),
        "",
        "  seed: 42, // same seed + same knobs → same waveform",
        "});",
        'await Deno.writeFile("sbv2.wav", encodeWav(audio.data, audio.sampleRate));',
        "```",
        "",
        "`generate()` returns `{ sampleRate, data }`, where `data` is an f32 mono waveform —",
        "exactly what `encodeWav` takes.",
        "Weights are fetched once and cached (verified against `karume.json`'s `size` / `sha256`).",
        "",
        "The Japanese analyzer dictionary the text front-end needs is **not** part of this",
        "repository: the pipeline fetches it on the first `generate()` and keeps it for the rest",
        "of the instance's life (pass your own through the `dictionary` option to skip the fetch).",
    ]


def _sbv2_defaults(model: Mapping[str, Any]) -> list[str]:
    defaults = model["pipelineConfig"]["defaults"]
    return [
        "### Defaults",
        "",
        "Any knob not passed to `generate()` is filled in from the manifest's defaults.",
        "",
        *(f"- **{key}**: {knob(value)}" for key, value in defaults.items()),
        "",
        "`seed` is the one knob the manifest does not carry — it defaults to `0`, and the same"
        " seed with the same knobs gives the same waveform.",
    ]


def render_sbv2_model_card(
    manifest: Mapping[str, Any],
    repo: str,
    profile: Sbv2CardProfile,
    abbreviations: Mapping[str, str],
) -> str:
    """SBV2 配布形の `README.md` 本文を組み立てる（純関数・末尾改行つき）。

    `profile` に既定を置かないのは MUST — 帰属はファミリーごとに違う事実で、既定を持たせた
    瞬間に「別ファミリーのリポへ前のファミリーの帰属を描く」経路が黙って生える。

    `abbreviations` は席名の部品上書きトークンの対応表（正本は `sbv2.distribution` —
    ADR 0074 決定 4）。manifest に無い事実なので、定数として写さず引数で受ける。
    """
    require_pipeline(manifest, SBV2_SUPPORTED_PIPELINE)
    return render(
        (
            frontmatter(profile.metadata),
            ["", f"# {profile.title}", ""],
            _sbv2_overview(manifest),
            [""],
            _sbv2_base_weights(profile),
            [""],
            models(manifest),
            [""],
            _sbv2_usage(manifest, repo),
            *model_sections(
                manifest,
                (
                    partial(_sbv2_quants, abbreviations=abbreviations),
                    _sbv2_styles,
                    _sbv2_speakers,
                    _sbv2_defaults,
                ),
            ),
        )
    )
