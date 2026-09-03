"""BiRefNet 系配布形のモデルカード（`README.md`）— manifest から機械導出する純関数。

汎用の描画部品（frontmatter・モデル一覧・quant 表・節の組み立て）は
`karume.modelcard` が持つ。ここが持つのは **BiRefNet 系固有の事実**だけ: 帰属（出所・
ライセンス・引用・学習データ）と、この pipeline のカードに何を書くか。

帰属は**モデル名から一意に決まる**（{@link BIREFNET_MODELS}）ので、プロファイルの軸には
割らない — 分けると「Lucida を BiRefNet_HR の帰属で配る」取り違えを操作者が起こせるように
なる。

MUST: **数値・ダウンロード量・quant 表・dtype ラベルは 1 つ残らず manifest から導出する**
（`karume.modelcard` の同 MUST がそのまま掛かる）。ここが持ってよい定数は、manifest に
**存在しない事実**だけ。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from karume.modelcard import (
    CardMetadata,
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
BIREFNET_SUPPORTED_PIPELINE = "birefnet/1"

#: HF の pipeline tag（上流 2 リポとも同じ）。出すのは α マット 1 枚で、クラス分類も
#: インスタンス分割も持たない。
BIREFNET_PIPELINE_TAG = "image-segmentation"

#: 上流の論文（BiRefNet — 系列の共通の出典）。
BIREFNET_PAPER = "arxiv.org/abs/2401.03407"


@dataclass(frozen=True)
class BirefnetModel:
    """1 モデルぶんの**manifest に無い事実**（出所・呼び名・帰属の追加行）。

    **この 1 表が「どのモデルが何の重みか」の唯一の事実**で、`birefnet.distribution` は系列名（=
    上流リポ名から導く綴り — `birefnet/export.py` の流儀）と帰属をここから引く。載っていない
    モデル名は出所を名乗れないので、カードは描かずに落ちる。
    """

    #: 上流チェックポイントの HF リポ ID。
    repo: str
    #: カードの見出し（上流が名前で売っているモデルは、その名前で呼ぶ）。
    title: str
    #: 概要の 1 行目（何ができる重みか）。
    tagline: str
    #: 帰属節の追加行（学習データのライセンス等 — 上流ごとに違う事実）。
    attribution: tuple[str, ...]


#: 実地確認（2026-08-13 — HF の model API）: どちらも `license: mit`。
BIREFNET_LICENSE = "mit"

BIREFNET_MODELS: Mapping[str, BirefnetModel] = {
    "hr": BirefnetModel(
        repo="ZhengPeng7/BiRefNet_HR",
        title="BiRefNet HR — Karume",
        tagline="the high-resolution general-purpose BiRefNet checkpoint",
        attribution=(
            "- **Upstream training data**: see the upstream model card and the BiRefNet"
            " repository. Several of the datasets it draws on (DIS5K, COD10K, HRSOD and"
            " others) are distributed for research purposes — check them against your own use"
            " case.",
        ),
    ),
    "lucida": BirefnetModel(
        repo="egeorcun/lucida",
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
    ),
}

#: モデル名 → 上流リポ ID。**{@link BIREFNET_MODELS} からの導出**で、2 表にしない（片方だけ
#: 動いたときに「別のモデルの重みを別のモデルとして帰属する」形が黙って作れる）。
BIREFNET_UPSTREAM: Mapping[str, str] = {name: entry.repo for name, entry in BIREFNET_MODELS.items()}


def _birefnet_entry(manifest: Mapping[str, Any]) -> BirefnetModel:
    """既定モデルの帰属を引く（1 リポ 1 モデル — {@link birefnet.distribution} の冒頭）。

    MUST: 未知のモデル名では描かない — 帰属の表に無いモデルを黙って落とすと、出所を名乗って
    いない再配布が静かに出る。
    """
    name = manifest["defaultModel"]
    entry = BIREFNET_MODELS.get(name)
    if entry is None:
        raise ValueError(
            f"モデル '{name}' の上流が帰属表に無い（既知: {sorted(BIREFNET_MODELS)}）"
            " — 出所を名乗れないカードは描かない"
        )
    return entry


def _birefnet_metadata(manifest: Mapping[str, Any]) -> CardMetadata:
    """frontmatter を manifest に並んだモデルから組む（`base_model` は再配布する上流の全部）。"""
    upstream: list[str] = []
    for name in manifest["models"]:
        entry = BIREFNET_MODELS.get(name)
        if entry is None:
            raise ValueError(
                f"モデル '{name}' の上流が帰属表に無い（既知: {sorted(BIREFNET_MODELS)}）"
                " — 出所を名乗れないカードは描かない"
            )
        upstream.append(entry.repo)
    return CardMetadata(
        pipeline_tag=BIREFNET_PIPELINE_TAG,
        base_model=tuple(upstream),
        # `base_model_relation` は置かない — 格納形を変えず（f32 のまま）コンテナだけを移した
        # もので、adapter / merge / quantized / finetune のどれでもない（CardMetadata の doc）。
        license=BIREFNET_LICENSE,
        tags=(BIREFNET_PIPELINE_TAG, "background-removal", "webgpu", "birefnet"),
    )


def _birefnet_overview(manifest: Mapping[str, Any]) -> list[str]:
    entry = _birefnet_entry(manifest)
    config = default_model(manifest)["pipelineConfig"]
    return [
        "## What is this",
        "",
        "A background-removal distribution, converted into the WebGPU inference runtime",
        "**Karume**'s container format (a single safetensors file = weights + a graph JSON",
        "embedded in `__metadata__`). Runs as-is in the browser and in Deno.",
        "",
        f"The weights are {entry.tagline}.",
        "",
        "- One graph, one call: pixels in, an **8-bit alpha matte** out, at the same size as the",
        "  image you handed over.",
        "- **Pre- and post-processing are included.** The pipeline resizes to"
        f" {config['imageWidth']} × {config['imageHeight']}, normalizes with the constants below,",
        "  then takes the sigmoid of the logits and scales the matte back to the original",
        "  resolution. Decoding PNG / JPEG is *not* part of this — use `createImageBitmap` in the",
        "  browser, or any decoder in Deno.",
        "- **Compositing is yours.** The pipeline returns the matte, not a cut-out: whether alpha",
        "  goes into an RGBA buffer, gets composited over a flat colour, or is fed to a colour",
        "  decontamination pass is a decision this repository should not make for you.",
        "- Not readable by transformers (it's a different container with an embedded graph); the"
        f" reader is a pipeline that implements `{BIREFNET_SUPPORTED_PIPELINE}`.",
        f"- Exporter used for the conversion: `{manifest['generator']}`. The distribution manifest"
        f" is `karume.json` (`{manifest['format']}`).",
    ]


def _birefnet_base_weights(manifest: Mapping[str, Any]) -> list[str]:
    """帰属節。格納形を変えていないので「変換したもの」としてだけ主張する。"""
    entry = _birefnet_entry(manifest)
    return [
        "## Base weights and attribution",
        "",
        f"- **Weights**: [{entry.repo}](https://huggingface.co/{entry.repo}), licensed"
        f" **{BIREFNET_LICENSE}** (as of retrieval).",
        *entry.attribution,
        f"- **Architecture**: BiRefNet ([{BIREFNET_PAPER}](https://{BIREFNET_PAPER})).",
        "- **Changes made here**: conversion into the Karume container format. No retraining, no",
        "  fine-tuning and **no quantization** — the weights are the source checkpoint's own f32",
        "  values. The graph is the upstream `forward` with layout-only rewrites (windowing, the",
        "  shifted-window roll, spatial padding and the patch merges were folded into equivalent",
        "  operations — bit-exact), plus two module rewrites that are equivalent up to",
        "  floating-point rounding: inference-time `BatchNorm2d` became a per-channel affine, and",
        "  the ASPP image-level pooling became a two-stage sum.",
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


def _birefnet_shape(model: Mapping[str, Any]) -> list[str]:
    """前処理の定数と入出力の形（利用者が渡すもの・受け取るものがここで読める）。"""
    config = model["pipelineConfig"]
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
    ]


def render_birefnet_model_card(manifest: Mapping[str, Any], repo: str) -> str:
    """BiRefNet 系配布形の `README.md` 本文を組み立てる（純関数・末尾改行つき）。"""
    require_pipeline(manifest, BIREFNET_SUPPORTED_PIPELINE)
    return render(
        (
            frontmatter(_birefnet_metadata(manifest)),
            ["", f"# {_birefnet_entry(manifest).title}", ""],
            _birefnet_overview(manifest),
            [""],
            _birefnet_base_weights(manifest),
            [""],
            models(manifest),
            [""],
            _birefnet_usage(manifest, repo),
            *model_sections(manifest, (quants, _birefnet_shape)),
        )
    )
