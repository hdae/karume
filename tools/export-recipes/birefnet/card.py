"""BiRefNet 系配布形のモデルカード（`README.md`）— manifest から機械導出する純関数。

汎用の描画部品（frontmatter・モデル一覧・quant 表・節の組み立て）は
`karume.modelcard` が持つ。ここが持つのは **BiRefNet 系固有の事実**だけ: 帰属（出所・
ライセンス・引用・学習データ）と、この pipeline のカードに何を書くか。

帰属は**checkpoint から一意に決まる**（{@link BIREFNET_CHECKPOINTS}）ので、プロファイルの軸
には割らない — 分けると「Lucida を BiRefNet_HR の帰属で配る」取り違えを操作者が起こせるように
なる。checkpoint は配布リポの軸そのもの（1 リポ = 1 checkpoint）なので、manifest からは読まず
**呼び出し側の pipeline 席が束ねて渡す**（`birefnet.distribution._birefnet_pipeline`）。
manifest に並ぶモデル名は**解像度**（`"1024"` / `"2048"`）で、帰属の軸ではない。

MUST: **数値・ダウンロード量・quant 表・dtype ラベルは 1 つ残らず manifest から導出する**
（`karume.modelcard` の同 MUST がそのまま掛かる）。ここが持ってよい定数は、manifest に
**存在しない事実**だけ。
"""

from __future__ import annotations

import textwrap
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from typing import Any

from karume.modelcard import (
    CardMetadata,
    FileRefKey,
    default_model,
    from_pretrained,
    frontmatter,
    model_sections,
    models,
    quants,
    render,
    require_pipeline,
)

#: このテンプレートが説明できるパイプライン契約（ADR 0041 §2 — モデル単位）。
BIREFNET_SUPPORTED_PIPELINE = "birefnet/2"

#: HF の pipeline tag（上流 2 リポとも同じ）。出すのは α マット 1 枚で、クラス分類も
#: インスタンス分割も持たない。
BIREFNET_PIPELINE_TAG = "image-segmentation"


@dataclass(frozen=True)
class BirefnetResources:
    """1 モデル（= 1 解像度）ぶんの実行資源の実測。

    冒頭の MUST の例外ではない: `requiredLimits` は常駐分（重み・state）しか数えないので、
    中間テンソルが要求する binding の大きさも総確保も **manifest に存在しない事実**にあたる
    （ADR 0089 決定 3 の意味論）。読み手が最初に確かめたい制約なので、カードが持つ。
    """

    #: GPU 総確保 = runtime の見積り（`PreparedModel.estimate` の `peakAccountedBytes`）: 常駐の
    #: 重み + 中間の領域 + 入力 + 出力の readback。
    total: str
    #: 中間テンソルに要る領域。**常駐の重みは持たない** — その数は manifest の container
    #: （part 列）側にあり（quant 表が導出する）、ここに写すと同じ量の正本が 2 つになる。
    intermediates: str
    #: WebGPU 既定の `maxStorageBufferBindingSize`（128MiB）を超える binding を 1 本、
    #: **実測されたその正体ごと**名乗る句（読み手が「既定スペックでは走らない」と読める形）。
    binding: str
    #: 画像 1 枚あたりの実行時間。
    run: str


@dataclass(frozen=True)
class BirefnetResourceMeasurement:
    """1 checkpoint ぶんの実行資源の実測 — 測った quant 席・条件・解像度ごとの数。

    checkpoint ごとに持つのは、測った席と条件が checkpoint ごとに違うから（HR は f16 席・Lucida は
    f32 席 — 1 つの表を両方のカードへ写すと、片方が測っていない席の数を名乗る）。
    値は ADR 0093 の静的 liveness パッキングと recipe のパッチ ⑨（decoder 末尾の 1×1 conv と
    bilinear upsample の順序交換）の後の実測で、正本は docs/limitations.md の BiRefNet 節。
    """

    #: 実測した quant 席。MUST: カードは総確保を**この席の数として**名乗る — 他の席の総確保は
    #: 推し量らない（冒頭の MUST）。この席が manifest に無いモデルのカードは描かない。
    quant: str
    #: 実測の条件（日付・機 — 実測の性格そのものなので、`pipelineConfig` からは導出しない）。
    conditions: str
    #: モデル名（= 解像度）→ 数。MUST: 実測していない解像度の席は置かない — カードが名乗る数は
    #: 1 つ残らず実測に紐づく。引くのは manifest の `imageWidth` なので
    #: （{@link _birefnet_resources}）、実測していない解像度で描こうとすると落ちる。
    by_resolution: Mapping[str, BirefnetResources]


#: 後処理の手順（manifest の `pipelineConfig.postprocess` — TS 側 `MATTE_PROCEDURES` の綴り）→
#: カードの説明（補間・段の順序・量子化）。手順の名前は manifest にあるが、その中身の説明は
#: manifest に**存在しない事実**なので、ここが持つ（冒頭の MUST の例外ではない）。
#:
#: MUST: 知らない手順は描かない — 推し量った説明を載せると、配布形の実際の後処理とカードが
#: 食い違ったまま公開される。手順の正本は ADR 0125 と `packages/models/src/birefnet/matte.ts`。
BIREFNET_POSTPROCESS_DESCRIPTIONS: Mapping[str, tuple[str, ...]] = {
    "u8-pil-bicubic": (
        "- **post-processing** (`u8-pil-bicubic`, the order upstream uses for this checkpoint):",
        "  the sigmoid of the logits is rounded to 8 bits at the graph's resolution, then resized",
        "  back to the image's size with 8-bit bicubic resampling that matches Pillow's `BICUBIC`",
        "  bit for bit. Rounding is half-up (the upstream code truncates).",
    ),
    "float-bilinear": (
        "- **post-processing** (`float-bilinear`, the order upstream uses for this checkpoint):",
        "  the sigmoid of the logits is resized back to the image's size in floating point with",
        "  antialiased bilinear interpolation, then rounded to 8 bits. Rounding is half-up (the",
        "  upstream code truncates).",
    ),
}

#: 上流の論文（BiRefNet — 系列の共通の出典）。
BIREFNET_PAPER = "arxiv.org/abs/2401.03407"


@dataclass(frozen=True)
class BirefnetCheckpoint:
    """1 checkpoint ぶんの**manifest に無い事実**（出所・呼び名・帰属の追加行）。

    **この 1 表が「どのリポが何の重みか」の唯一の事実**で、`birefnet.distribution` は系列名（=
    上流リポ名から導く綴り — `birefnet/export.py` の流儀）と帰属をここから引く。載っていない
    checkpoint は出所を名乗れないので、カードは描かずに落ちる。
    """

    #: 上流チェックポイントの HF リポ ID。
    repo: str
    #: 配る重みの上流 revision（HF の commit SHA）。系列の容器が名乗る出所
    #: （`provenance.upstreamRevision` — 台本が `--model-dir` の取得記録から焼く）を組み立てが
    #: これと突き合わせる（`birefnet.distribution`）。固定値で見るので、別の checkpoint・別の
    #: revision から焼いた系列は名前が合っていても組めない。値は `inputs/birefnet/<名前>` の
    #: `hf download` の記録（`_shared.upstream.snapshot_revision`）で確かめた（突合は
    #: `birefnet/tests/test_export.py` が取得記録のある機で持つ）。
    revision: str
    #: カードの見出し（上流が名前で売っているモデルは、その名前で呼ぶ）。
    title: str
    #: 概要の 1 行目（何ができる重みか）。
    tagline: str
    #: 帰属節の追加行（学習データのライセンス等 — 上流ごとに違う事実）。
    attribution: tuple[str, ...]
    #: 上流 checkpoint の浮動小数の格納 dtype（safetensors ヘッダの実測 2026-09-25 —
    #: HR は F16 687 本・Lucida は F32 687 本）。配る quant 席（この格納より広い席は出さない・
    #: 既定は同じ型の席 — `birefnet.distribution.birefnet_seat_dtypes`）、どの席が値を変えずに
    #: 持てるか（{@link birefnet_storage_lines}）と `base_model_relation`
    #: （{@link _birefnet_relation}）がここから決まる。export 時の品質の門は
    #: 同じ事実をヘッダから直接読む（`birefnet.export.checkpoint_dtypes`）ので、2 つの突合は
    #: `birefnet/tests/test_export.py` が実重みのある機で持つ。
    stored_dtype: str
    #: カードが名乗る実行資源の実測（{@link BirefnetResourceMeasurement}）。
    resources: BirefnetResourceMeasurement


#: 実地確認（2026-08-13 — HF の model API）: どちらも `license: mit`。
BIREFNET_LICENSE = "mit"

BIREFNET_CHECKPOINTS: Mapping[str, BirefnetCheckpoint] = {
    "hr": BirefnetCheckpoint(
        repo="ZhengPeng7/BiRefNet_HR",
        revision="a7a562f6fd16021180f2f4348f4de003a2d3d1e1",
        title="BiRefNet HR — Karume",
        tagline="the high-resolution general-purpose BiRefNet checkpoint",
        attribution=(
            "- **Upstream training data**: see the upstream model card and the BiRefNet"
            " repository. Several of the datasets it draws on (DIS5K, COD10K, HRSOD and"
            " others) are distributed for research purposes — check them against your own use"
            " case.",
        ),
        stored_dtype="f16",
        # 総確保 = runtime の見積りの `peakAccountedBytes`（重み + 中間の領域 + 入力 + 出力の
        # readback — 1024²: 509.2 + 749.2 + 12 + 4 = 1,274.4 MiB・2048²: 706.2 + 2,948.0 + 48 +
        # 16 = 3,718.2 MiB）を 1 GiB = 1,024 MiB で換算して丸めた値。run はウォームアップの 1 回
        # （計画の当たらない run）を除いた 5 回の範囲で、5 回の 1 回目は中間の領域の確保を含む。
        # 中間の領域と最大の binding は f32 席と同じ（活性値は f16 席でも f32）。
        resources=BirefnetResourceMeasurement(
            quant="f16",
            conditions="measured 2026-10-11 on an RTX 3080 Ti",
            by_resolution={
                "1024": BirefnetResources(
                    total="about 1.2 GiB",
                    intermediates="749 MiB",
                    binding="the largest single storage buffer is 320 MiB",
                    run="about 1.4 s",
                ),
                "2048": BirefnetResources(
                    total="about 3.6 GiB",
                    intermediates="2,948 MiB",
                    binding="the largest single storage buffer is 1,280 MiB",
                    run="7.3 – 7.9 s",
                ),
            },
        ),
    ),
    "lucida": BirefnetCheckpoint(
        repo="egeorcun/lucida",
        revision="6cbedc9722652dc9a3df91dd871f0c4f3334e922",
        title="Lucida (BiRefNet) — Karume",
        tagline=(
            "a BiRefNet fine-tune aimed at camouflage, transparency, text/logos and illustration"
        ),
        attribution=(
            "- **Fine-tune of**:"
            " [ZhengPeng7/BiRefNet_HR](https://huggingface.co/ZhengPeng7/BiRefNet_HR)"
            " (**mit**) — architecture and initial weights; the upstream copyright notice is"
            " preserved.",
            "- **Illustration training data** includes"
            " [ToonOut](https://huggingface.co/datasets/joelseytre/toonout) (**CC-BY-4.0**);"
            " other sets it draws on (P3M-10k, COD10K, DIS5K) are distributed for research"
            " purposes — check them against your own use case.",
            "- **Not included**: `lucida-m35-comfy.safetensors`. That experimental variant has"
            " the `Normalize` step folded into its first convolution (it expects raw `[0, 1]`"
            " pixels), so the preprocessing this pipeline applies would be counted twice, and"
            " upstream ships it to run inside a larger ComfyUI pipeline rather than bare.",
        ),
        stored_dtype="f32",
        # 総確保 = runtime の見積りの `peakAccountedBytes`（重み + 中間の領域 + 入力 + 出力の
        # readback — 1024²: 919.2 + 749.2 + 12 + 4 = 1,684.5 MiB・2048²: 1,116.2 + 2,948.0 + 48 +
        # 16 = 4,128.2 MiB）を 1 GiB = 1,024 MiB で換算して丸めた値。run はウォームアップの 1 回
        # （計画の当たらない run）を除いた 5 回の範囲で、5 回の 1 回目は中間の領域の確保を含む。
        resources=BirefnetResourceMeasurement(
            quant="f32",
            conditions="measured 2026-10-11 on an RTX 3080 Ti",
            by_resolution={
                "1024": BirefnetResources(
                    total="about 1.6 GiB",
                    intermediates="749 MiB",
                    binding="the largest single storage buffer is 320 MiB",
                    run="about 1.4 s",
                ),
                "2048": BirefnetResources(
                    total="about 4.0 GiB",
                    intermediates="2,948 MiB",
                    binding="the largest single storage buffer is 1,280 MiB",
                    run="6.6 – 6.7 s",
                ),
            },
        ),
    ),
}

#: checkpoint → 上流リポ ID。**{@link BIREFNET_CHECKPOINTS} からの導出**で、2 表にしない
#: （片方だけ動いたときに「別のリポの重みを別の上流として帰属する」形が黙って作れる）。
BIREFNET_UPSTREAM: Mapping[str, str] = {
    name: entry.repo for name, entry in BIREFNET_CHECKPOINTS.items()
}

#: manifest の weights キー（= `birefnet.distribution.BIREFNET_ROLE`）。distribution がこの
#: モジュールを import するので、循環を避けてここでも綴る — 一致は
#: `birefnet/tests/test_card.py` が見る。
BIREFNET_MATTE_WEIGHTS = "matte"

#: 格納 dtype → ビット幅。配る席の規則（checkpoint の格納より広い席は出さない —
#: `birefnet.distribution.birefnet_seat_dtypes`）と、席が checkpoint の値を変えずに持てるかの
#: 判定（狭い席は丸める）がここから決まる。
#: MUST: 知らない dtype は描かない（段 2 の i8 は席と一緒にここへ足す — 推し量った文面を
#: 法的テキストへ載せない）。
_STORAGE_BITS: Mapping[str, int] = {"f32": 32, "f16": 16}


def birefnet_storage_bits(dtype: str) -> int:
    """格納 dtype のビット幅（{@link _STORAGE_BITS} — 知らない dtype は落とす）。"""
    bits = _STORAGE_BITS.get(dtype)
    if bits is None:
        raise ValueError(
            f"格納 dtype '{dtype}' の文面を持たない（既知: {' / '.join(_STORAGE_BITS)}）"
            " — 推し量った文面を NOTICE / カードへ載せない"
        )
    return bits


#: 格納の説明を折り返す幅（NOTICE / カードのテンプレートの他の行と同じ約 100 字）。
_STORAGE_LINE_WIDTH = 100


def _wrapped(text: str, indent: str) -> str:
    """Markdown の箇条 1 つを折り返す（続きの行は箇条の本文の位置へ字下げする）。"""
    return textwrap.fill(
        text,
        width=_STORAGE_LINE_WIDTH,
        initial_indent=indent,
        subsequent_indent=" " * len(indent),
        break_long_words=False,
        break_on_hyphens=False,
    )


#: f16 格納になる重み（emit の適格判定が重みスロットとして圧縮する側 — 相対位置表は embedding の
#: 重みスロット・deform の offset / modulator conv も通常の conv）。
_F16_STORED = "the linear, convolution and embedding (relative-position) weights"

#: f32 格納に残るもの。deform の sampling weights（`regular_conv` — `deform_conv2d` に渡る重みで、
#: 重みスロットを持たない）と bias / norm。⑦ の BatchNorm は α / β を統計から f32 で導く派生定数
#: なので、「広げた値」ではなく「導いた値」と書き分ける（2026-09-26 レビュー A-3 / F4）。
_F32_KEPT = (
    "the deformable convolutions' sampling weights (`regular_conv`), the biases and the"
    " remaining normalization parameters"
)
_BATCHNORM_AFFINE = "inference-time BatchNorm is a derived per-channel affine in f32"


def _seat_storage_line(dtype: str, stored: str) -> str:
    """quant 席 1 つぶんの格納の説明（checkpoint の格納 `stored` との関係で文面が決まる）。

    checkpoint の格納より広い席（f16 の checkpoint の f32 席）の文面は持たない — 値を正確に
    広げるだけの席は配らない（`birefnet.distribution.birefnet_seat_dtypes` の規則）ので、
    そういう席を持つ manifest は描かずに落とす。
    """
    if dtype == "f32" and stored == "f32":
        return _wrapped("`f32`: the checkpoint's own f32 values, unchanged.", "  - ")
    if dtype == "f16" and stored == "f16":
        return _wrapped(
            f"`f16`: the checkpoint's own f16 values, kept as f16 in {_F16_STORED}; {_F32_KEPT}"
            f" are widened exactly to f32, and {_BATCHNORM_AFFINE}.",
            "  - ",
        )
    if dtype == "f16" and birefnet_storage_bits(stored) > birefnet_storage_bits(dtype):
        return _wrapped(
            "`f16`: **quantized** — every weight was rounded to the nearest f16 value before"
            f" export; {_F16_STORED} are stored as f16, {_F32_KEPT} keep f32 storage,"
            f" {_BATCHNORM_AFFINE}, and all computation stays in f32.",
            "  - ",
        )
    raise ValueError(f"格納 {dtype} の席を {stored} の checkpoint について描く文面を持たない")


def birefnet_storage_lines(checkpoint: str, dtypes: Sequence[str]) -> list[str]:
    """quant 席ごとの格納の説明（NOTICE とカードの帰属節が**同じ 1 か所**から引く）。

    `dtypes` は配布形が持つ席の格納 dtype（カードは manifest から・NOTICE は配布 recipe の表から
    渡す）。見出しは「全ての席が checkpoint の値を変えずに持つか」で決まる — HR は checkpoint
    自身が f16 なので f16 席も無損失で **No quantization** のまま、Lucida は f32 なので f16 席が
    量子化になる（{@link BirefnetCheckpoint.stored_dtype}）。

    MUST: 文面は配布形の中身と対応していること（NOTICE の MUST — 散文としては妥当なまま
    `verify_dist` も manifest 検査も素通りするので、事実から組む）。
    """
    stored = _birefnet_entry(checkpoint).stored_dtype
    if not dtypes:
        raise ValueError("quant 席が 1 つも無い — 格納の説明を組めない")
    lossless = all(
        birefnet_storage_bits(dtype) >= birefnet_storage_bits(stored) for dtype in dtypes
    )
    # 席が 1 つ（HR）なら単数で書く — 「every quant seat」は比べる席が無いリポで読み手を迷わせる。
    seats = "every quant seat stores" if len(dtypes) > 1 else "the quant seat stores"
    heading = (
        f"**No quantization**: {seats} the source checkpoint's own values"
        f" (the checkpoint itself is stored in {stored}):"
        if lossless
        else f"**Quantization**: the source checkpoint is stored in {stored}; per quant seat:"
    )
    return [_wrapped(heading, "- "), *(_seat_storage_line(dtype, stored) for dtype in dtypes)]


def _birefnet_entry(checkpoint: str) -> BirefnetCheckpoint:
    """このリポが配る checkpoint の帰属を引く（1 リポ 1 checkpoint）。

    MUST: 未知の checkpoint では描かない — 帰属の表に無い重みを黙って落とすと、出所を名乗って
    いない再配布が静かに出る。
    """
    entry = BIREFNET_CHECKPOINTS.get(checkpoint)
    if entry is None:
        raise ValueError(
            f"checkpoint '{checkpoint}' の上流が帰属表に無い"
            f"（既知: {sorted(BIREFNET_CHECKPOINTS)}）— 出所を名乗れないカードは描かない"
        )
    return entry


def _birefnet_resources(
    measurement: BirefnetResourceMeasurement, config: Mapping[str, Any]
) -> BirefnetResources:
    """このモデルの解像度の実測を引く（{@link BirefnetResourceMeasurement.by_resolution}）。

    キーは manifest 側の `imageWidth` — モデル名の綴りではなく**宣言された解像度**で引くので、
    名前を付け替えても実測と対応が切れない。
    """
    resolution = str(config["imageWidth"])
    resources = measurement.by_resolution.get(resolution)
    if resources is None:
        raise ValueError(
            f"入力 {resolution}² の実行資源は実測していない"
            f"（実測済み: {' / '.join(measurement.by_resolution)}）— 実測していない数は名乗らない"
        )
    return resources


def _seat_dtypes(model: Mapping[str, Any]) -> list[str]:
    """モデルの quant 席が指すマットの格納 dtype（manifest の並び・重複なし）。"""
    seen: list[str] = []
    for quant in model["quants"].values():
        dtype = quant["weights"][BIREFNET_MATTE_WEIGHTS]
        if dtype not in seen:
            seen.append(dtype)
    return seen


def _birefnet_relation(manifest: Mapping[str, Any], entry: BirefnetCheckpoint) -> str | None:
    """`base_model_relation` を導く（既定 quant の DL 実体が値を丸めているときだけ `quantized`）。

    irodori / sbv2 / anima と同じ「既定 quant の DL 実体」の規則。値を変えない席（f32、HR の
    f16）が既定なら 4 値のどれでもないので置かない（CardMetadata の doc — 4 値から一番近いものを
    当てると、カードが事実でない主張を持つ）。
    """
    model = default_model(manifest)
    dtype = model["quants"][model["defaultQuant"]]["weights"][BIREFNET_MATTE_WEIGHTS]
    if birefnet_storage_bits(dtype) < birefnet_storage_bits(entry.stored_dtype):
        return "quantized"
    return None


def _birefnet_metadata(manifest: Mapping[str, Any], entry: BirefnetCheckpoint) -> CardMetadata:
    """frontmatter を組む（`base_model` はこのリポが再配布する上流 1 本）。

    モデルが 2 つ並んでも上流は 1 つ — 同居しているのは同じ checkpoint の**解像度違い**で、
    別の重みではない。
    """
    return CardMetadata(
        pipeline_tag=BIREFNET_PIPELINE_TAG,
        base_model=(entry.repo,),
        base_model_relation=_birefnet_relation(manifest, entry),
        license=BIREFNET_LICENSE,
        tags=(BIREFNET_PIPELINE_TAG, "background-removal", "webgpu", "birefnet"),
    )


def _birefnet_overview(manifest: Mapping[str, Any], entry: BirefnetCheckpoint) -> list[str]:
    config = default_model(manifest)["pipelineConfig"]
    default_name = manifest["defaultModel"]
    return [
        "## What is this",
        "",
        "A background-removal distribution, converted into the WebGPU inference runtime",
        "**Karume**'s container format (a `.krm` part sequence whose first part carries the graph",
        "and model descriptors). Runs as-is in the browser and in Deno.",
        "",
        f"The weights are {entry.tagline}.",
        "",
        "- One graph, one call: pixels in, an **8-bit alpha matte** out, at the same size as the",
        "  image you handed over.",
        "- **One model per input resolution.** The resolution is baked into the graph (window",
        "  masks and padding constants are per-resolution), so it is picked with `model`, not at",
        "  call time. Each model declares its own resize target below; the bigger one holds more",
        "  detail at the edges and costs more GPU memory and time.",
        "- **Pre- and post-processing are included.** The pipeline resizes to"
        f" {config['imageWidth']} × {config['imageHeight']} for the default model"
        f" `{default_name}`,",
        "  normalizes with the constants below, then takes the sigmoid of the logits, scales the",
        "  matte back to the original resolution and rounds it to 8 bits — in the order and with",
        "  the interpolation upstream uses for this checkpoint (listed per model below). Decoding",
        "  PNG / JPEG is *not* part of this — use `createImageBitmap` in the browser, or any",
        "  decoder in Deno.",
        "- **Compositing is yours.** The pipeline returns the matte, not a cut-out: whether alpha",
        "  goes into an RGBA buffer, gets composited over a flat colour, or is fed to a colour",
        "  decontamination pass is a decision this repository should not make for you.",
        "- Not readable by transformers (it's a different container with an embedded graph); the"
        f" reader is a pipeline that implements `{BIREFNET_SUPPORTED_PIPELINE}`.",
        f"- Exporter used for the conversion: `{manifest['generator']}`. The distribution manifest"
        f" is `karume.json` (`{manifest['format']}`).",
    ]


def _birefnet_base_weights(manifest: Mapping[str, Any], checkpoint: str) -> list[str]:
    """帰属節。変更点の列挙と、quant 席ごとの格納（{@link birefnet_storage_lines}）。

    席の格納 dtype は manifest から引く（モジュール doc の MUST — dtype ラベルは手で持たない）。
    """
    entry = _birefnet_entry(checkpoint)
    return [
        "## Base weights and attribution",
        "",
        f"- **Weights**: [{entry.repo}](https://huggingface.co/{entry.repo}), licensed"
        f" **{BIREFNET_LICENSE}** (as of retrieval) — a verbatim copy of the license, carrying"
        " the copyright notices that apply to this repository, is in `LICENSE.md`.",
        *entry.attribution,
        f"- **Architecture**: BiRefNet ([{BIREFNET_PAPER}](https://{BIREFNET_PAPER})).",
        "- **Changes made here** (also listed in `NOTICE.md`): conversion into the Karume",
        "  container format. No retraining and no fine-tuning; the stored values per quant seat",
        "  are listed below. The graph is the upstream",
        "  `forward` with layout-only rewrites (windowing, the shifted-window roll, spatial",
        "  padding and the patch merges were folded into equivalent operations — bit-exact),",
        "  plus three rewrites that are equivalent up to floating-point rounding:",
        "  inference-time `BatchNorm2d` became a per-channel affine, the ASPP image-level",
        "  pooling became a two-stage sum, and the decoder tail's 1×1 convolution was swapped",
        "  with the bilinear upsample it used to follow (both are linear, so they commute — this",
        "  removes two full-resolution intermediates).",
        *birefnet_storage_lines(checkpoint, _seat_dtypes(default_model(manifest))),
    ]


def _birefnet_usage(manifest: Mapping[str, Any], repo: str) -> list[str]:
    """Usage 例の方針: 動く最小形は生かし、**普通のユースケースで使いそうな optional は
    コメントアウトで併記**する（選べる値も同じ行のコメントに列挙 — manifest から機械導出する
    ので、モデル / quant が増えれば列挙も追従する）。読者がコメントを外すだけで次の一歩へ
    進める形。

    `segment()` は画像 1 枚しか受けない（解像度はグラフに焼かれていて実行時に選べない —
    `BirefnetPipeline` の公開面）ので、この pipeline の optional ノブは model / quant の
    2 つだけである。

    NOTE: `fromAssets` は**案内しない**（2026-08-29 裁定）。分割配布形も読めるようになった
    （X2-101）が、あちらはバイト列を自分で持っている前提のローカルデバッグ向けの面で、HF から
    使う読者の普通の入口は `fromPretrained`。両方を並べると「どちらを使うのか」を読者に
    判断させることになる。
    """
    model_name = manifest["defaultModel"]
    model = default_model(manifest)
    quant = model["defaultQuant"]
    model_names = " / ".join(sorted(manifest["models"]))
    quant_names = " / ".join(sorted(model["quants"]))
    return [
        "## Usage",
        "",
        "```ts",
        'import { BirefnetPipeline } from "jsr:@karume/models";',
        "",
        *from_pretrained(
            "BirefnetPipeline",
            repo,
            [
                f'  // model: "{model_name}", // default — available: {model_names}',
                f'  // quant: "{quant}", // default — available: {quant_names}',
            ],
            disposable="await using",
        ),
        "",
        "// RGB8, row-major, 3 bytes per pixel. Decoding is the caller's job.",
        "const image = { data: pixels, width, height };",
        "const matte = await pipeline.segment(image);",
        "",
        "// matte.data is one alpha byte per pixel, same width/height as the input.",
        "// Straight alpha into an RGBA buffer (canvas, ImageData, an RGBA encoder):",
        "const rgba = new Uint8Array(matte.width * matte.height * 4);",
        "for (let i = 0; i < matte.data.length; i += 1) {",
        "  for (let c = 0; c < 3; c += 1) rgba[i * 4 + c] = image.data[i * 3 + c];",
        "  rgba[i * 4 + 3] = matte.data[i];",
        "}",
        "```",
        "",
        "`segment()` keeps one GPU session alive for the lifetime of the pipeline, so processing",
        "many images uploads the weights once; concurrent calls are queued rather than run side by",
        "side. Weights are fetched once and cached (verified against `karume.json`'s `size` /",
        "`sha256`).",
    ]


def _birefnet_postprocess(config: Mapping[str, Any]) -> tuple[str, ...]:
    """このモデルの後処理の手順の説明を引く（{@link BIREFNET_POSTPROCESS_DESCRIPTIONS}）。

    欄が無い manifest も、説明を持たない手順と同じ文言で落とす（素の `KeyError` では何が欠けたかが
    読めない）。
    """
    procedure = config.get("postprocess")
    description = BIREFNET_POSTPROCESS_DESCRIPTIONS.get(procedure)
    if description is None:
        raise ValueError(
            f"後処理の手順 {procedure!r} の説明を持たない"
            f"（既知: {' / '.join(BIREFNET_POSTPROCESS_DESCRIPTIONS)}）— 推し量った説明は載せない"
        )
    return description


def _birefnet_shape(model: Mapping[str, Any], *, entry: BirefnetCheckpoint) -> list[str]:
    """前処理の定数と入出力の形（利用者が渡すもの・受け取るものがここで読める）。

    実行資源はモデルごと（= 解像度ごと）に別の実測なので、この節に載る — 1024² と 2048² で
    総確保が 2 倍以上違い、「どちらを選ぶか」の判断材料そのものである。実測はこのリポの
    checkpoint のもの（{@link BirefnetCheckpoint.resources}）。
    """
    config = model["pipelineConfig"]
    measurement = entry.resources
    resources = _birefnet_resources(measurement, config)
    if measurement.quant not in model["quants"]:
        raise ValueError(
            f"実行資源を実測した quant '{measurement.quant}' がこのモデルの席に無い"
            f"（席: {sorted(model['quants'])}）— 実測していない席の数は名乗らない"
        )
    mean = " / ".join(str(value) for value in config["imageMean"])
    std = " / ".join(str(value) for value in config["imageStd"])
    return [
        "### Input and output",
        "",
        "The resize target is the exported graph's own input shape; the normalization constants",
        "are the upstream preprocessor's (ImageNet statistics).",
        "",
        f"- **input**: RGB8 pixels, resized to {config['imageWidth']} ×"
        f" {config['imageHeight']} ({config['interpolation']}, antialiased). The aspect ratio is",
        "  **not** preserved — there is no crop and no padding, matching the upstream processor.",
        f"- **normalization**: `(pixel / 255 - mean) / std`, mean {mean}, std {std}",
        "- **output**: one alpha byte per pixel at the size of the image you passed in (the graph",
        "  itself emits pre-sigmoid logits; the sigmoid and the resize back happen on the host).",
        *_birefnet_postprocess(config),
        f"- **required GPU memory**: {resources.total} allocated in total with the"
        f" `{measurement.quant}` quant — its",
        f"  weights listed above, resident, plus {resources.intermediates} for the intermediate"
        " tensors — and",
        f"  {resources.binding} ({measurement.conditions}); one image takes {resources.run}.",
        "  WebGPU's default `maxStorageBufferBindingSize` is 128 MiB, so this is in practice a",
        "  desktop-class GPU requirement. `karume.json` does not declare it: the declared limits",
        "  cover the resident weights and state, not the intermediate tensors a run allocates.",
    ]


def render_birefnet_model_card(
    manifest: Mapping[str, Any],
    repo: str,
    checkpoint: str,
    host_assets: Mapping[FileRefKey, int] = {},
) -> str:
    """BiRefNet 系配布形の `README.md` 本文を組み立てる（純関数・末尾改行つき）。

    `checkpoint` はこのリポが配る重み（`"hr"` / `"lucida"`）— manifest に並ぶモデル名は
    解像度なので、帰属はそこからは決まらない。渡すのは pipeline 席（モジュール doc）。
    """
    require_pipeline(manifest, BIREFNET_SUPPORTED_PIPELINE)
    entry = _birefnet_entry(checkpoint)
    return render(
        (
            frontmatter(_birefnet_metadata(manifest, entry)),
            ["", f"# {entry.title}", ""],
            _birefnet_overview(manifest, entry),
            [""],
            _birefnet_base_weights(manifest, checkpoint),
            [""],
            models(manifest),
            [""],
            _birefnet_usage(manifest, repo),
            *model_sections(
                manifest,
                (partial(quants, host_assets=host_assets), partial(_birefnet_shape, entry=entry)),
            ),
        )
    )
