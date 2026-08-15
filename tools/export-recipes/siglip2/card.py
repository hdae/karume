"""SigLIP2 配布形のモデルカード（`README.md`）— manifest から機械導出する純関数。

汎用の描画部品（frontmatter・モデル一覧・ファイル表・quant 表・節の組み立て）は
`karume.modelcard` が持つ。ここが持つのは **SigLIP2 固有の事実**だけ: 帰属（出所・
ライセンス・引用）と、この pipeline のカードに何を書くか。

帰属は**モデル名から一意に決まる**（{@link SIGLIP2_UPSTREAM}）ので、プロファイルの軸には
割らない — 分けると「so400m を base の帰属で配る」取り違えを操作者が起こせるようになる。

MUST: **数値・ファイル一覧・quant 表・dtype ラベルは 1 つ残らず manifest から導出する**
（`karume.modelcard` の同 MUST がそのまま掛かる）。ここが持ってよい定数は、manifest に
**存在しない事実**だけ。
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from karume.modelcard import (
    CardMetadata,
    default_model,
    files,
    frontmatter,
    model_sections,
    models,
    quants,
    render,
    require_pipeline,
)

#: このテンプレートが説明できるパイプライン契約（ADR 0041 §2 — モデル単位）。
SIGLIP2_SUPPORTED_PIPELINE = "siglip2/1"

#: HF の pipeline tag。この配布形が持つのは **vision tower だけ**なので、上流カードの
#: `zero-shot-image-classification` は名乗れない（text tower が無いと成立しない）。
SIGLIP2_PIPELINE_TAG = "image-feature-extraction"

SIGLIP2_TITLE = "SigLIP2 Vision Tower — Karume"

#: モデル名 → 上流チェックポイントの HF リポ ID。**この 1 表が「どのモデルが何の重みか」の
#: 唯一の事実**で、`siglip2.distribution` は系列名 / 入力素材のディレクトリ名（どちらも上流
#: リポ名そのもの — `siglip2/export.py` の綴り）をここから導く。載っていないモデル名は帰属を
#: 書けないので、カードは描かずに落ちる。
#:
#: ライセンスは実地確認（2026-08-13 — https://huggingface.co/google/siglip2-base-patch16-224）:
#: `license: apache-2.0`。so400m 側も同じ。
SIGLIP2_UPSTREAM: Mapping[str, str] = {
    "base": "google/siglip2-base-patch16-224",
    "so400m": "google/siglip2-so400m-patch14-384",
}

SIGLIP2_LICENSE = "apache-2.0"

#: MAP head の差し替え（`siglip2.patch` の段 ③）が持ち込む差の実測幅。**ビット同一では
#: ない**ので帰属節が明示する（`siglip2/export.py --verify` が 2 系列とも毎回実測する値）。
SIGLIP2_MAP_HEAD_DIFF = "1.1e-6 to 2.9e-6"


def _siglip2_metadata(manifest: Mapping[str, Any]) -> CardMetadata:
    """frontmatter を manifest に並んだモデルから組む（`base_model` は再配布する上流の全部）。

    MUST: 未知のモデル名では描かない — 帰属の表に無いモデルを黙って落とすと、`base_model` が
    1 つ足りないカード（= 出所を名乗っていない再配布）が静かに出る。
    """
    upstream: list[str] = []
    for name in manifest["models"]:
        if name not in SIGLIP2_UPSTREAM:
            raise ValueError(
                f"モデル '{name}' の上流が帰属表に無い（既知: {sorted(SIGLIP2_UPSTREAM)}）"
                " — 出所を名乗れないカードは描かない"
            )
        upstream.append(SIGLIP2_UPSTREAM[name])
    return CardMetadata(
        pipeline_tag=SIGLIP2_PIPELINE_TAG,
        base_model=tuple(upstream),
        # `base_model_relation` は置かない — 格納形を変えず（f32 のまま）コンテナだけを移した
        # もので、adapter / merge / quantized / finetune のどれでもない（CardMetadata の doc）。
        license=SIGLIP2_LICENSE,
        tags=(SIGLIP2_PIPELINE_TAG, "webgpu", "siglip", "vision"),
    )


def _siglip2_overview(manifest: Mapping[str, Any]) -> list[str]:
    config = default_model(manifest)["pipelineConfig"]
    return [
        "## What is this",
        "",
        "An image embedding distribution: the **vision tower** of SigLIP2, converted into the",
        "WebGPU inference runtime **Karume**'s container format (a single safetensors file =",
        "weights + a graph JSON embedded in `__metadata__`). Runs as-is in the browser and in"
        " Deno.",
        "",
        "- One graph, one call: pixels in, `pooler_output` out — the pooled `[1,"
        f" {config['hiddenDim']}]` vector from the attention-pooling (MAP) head.",
        "- **Preprocessing is included.** The pipeline resizes to"
        f" {config['imageWidth']} × {config['imageHeight']}, rescales and normalizes with the",
        "  constants below, so callers hand over raw RGB8 pixels. Decoding PNG / JPEG is *not*",
        "  part of this — use `createImageBitmap` in the browser, or any decoder in Deno.",
        "- **The text tower is not here.** Zero-shot classification and image/text similarity need",
        "  it (plus the logit scale and bias that go with it), so this repository cannot do them;",
        "  what it does is turn an image into a vector.",
        "- Not readable by transformers (it's a different container with an embedded graph); the"
        f" reader is a pipeline that implements `{SIGLIP2_SUPPORTED_PIPELINE}`.",
        f"- Exporter used for the conversion: `{manifest['generator']}`. The distribution manifest"
        f" is `karume.json` (`{manifest['format']}`).",
    ]


def _siglip2_base_weights(manifest: Mapping[str, Any]) -> list[str]:
    """帰属節。格納形を変えていないので「変換したもの」としてだけ主張する。"""
    lines = [
        "## Base weights and attribution",
        "",
        "Converted into the container format — the original checkpoints are not distributed here.",
        "",
    ]
    for name in manifest["models"]:
        repo = SIGLIP2_UPSTREAM[name]
        lines.append(
            f"- **`{name}`**: [{repo}](https://huggingface.co/{repo}), licensed"
            f" **{SIGLIP2_LICENSE}** (as of retrieval)."
        )
    lines += [
        "- **Changes made here**: conversion into the Karume container format, vision tower only.",
        "  No retraining, no fine-tuning and **no quantization** — the weights are the source",
        "  checkpoint's own f32 values. Two rewrites were needed to export the graph: the patch",
        "  embedding's padding and the position embedding lookup were folded into equivalent",
        "  operations (bit-exact), and the pooling head's attention was rewritten with explicit",
        "  q/k/v projections, which is equivalent up to floating-point rounding"
        f" ({SIGLIP2_MAP_HEAD_DIFF}",
        "  measured on the pooled vector, whose L2 norm is around 13).",
    ]
    return lines


def _siglip2_usage(manifest: Mapping[str, Any], repo: str) -> list[str]:
    model_name = manifest["defaultModel"]
    model = default_model(manifest)
    quant = model["defaultQuant"]
    model_names = " / ".join(sorted(manifest["models"]))
    return [
        "## Usage",
        "",
        "```ts",
        'import { Siglip2Pipeline } from "jsr:@karume/models";',
        "",
        f'await using pipeline = await Siglip2Pipeline.fromPretrained("{repo}", {{',
        f'  // model: "{model_name}", // default — available: {model_names}',
        f'  // quant: "{quant}", // the only one this repository ships',
        "});",
        "",
        "// RGB8, row-major, 3 bytes per pixel. Decoding is the caller's job.",
        "const embedding = await pipeline.embed({ data: pixels, width, height });",
        "",
        "// pooler_output is not L2-normalized — normalize it yourself for cosine similarity:",
        "const norm = Math.sqrt(embedding.reduce((sum, value) => sum + value * value, 0));",
        "const unit = embedding.map((value) => value / norm);",
        "```",
        "",
        "`embed()` returns the pooled vector as an f32 array, exactly as the graph produces it.",
        "It keeps one GPU session alive for the lifetime of the pipeline, so embedding many images",
        "uploads the weights once; concurrent calls are queued rather than run side by side.",
        "Weights are fetched once and cached (verified against `karume.json`'s `size` / `sha256`).",
        "You can also build from bytes you fetched yourself (`Siglip2Pipeline.fromAssets`).",
    ]


def _siglip2_shape(model: Mapping[str, Any]) -> list[str]:
    """前処理の定数と入出力の形（利用者が渡すもの・受け取るものがここで読める）。"""
    config = model["pipelineConfig"]
    mean = " / ".join(str(value) for value in config["imageMean"])
    std = " / ".join(str(value) for value in config["imageStd"])
    return [
        "### Input and output",
        "",
        "Derived from the checkpoint's own `preprocessor_config.json` and the exported graph, and",
        "checked against each other when this repository was assembled.",
        "",
        f"- **input**: RGB8 pixels, resized to {config['imageWidth']} ×"
        f" {config['imageHeight']} ({config['interpolation']}, antialiased). The aspect ratio is",
        "  **not** preserved — there is no crop and no padding, matching the upstream processor.",
        f"- **normalization**: `(pixel / 255 - mean) / std`, mean {mean}, std {std}",
        f"- **output**: `pooler_output`, {config['hiddenDim']} f32 values, not L2-normalized",
    ]


def render_siglip2_model_card(manifest: Mapping[str, Any], repo: str) -> str:
    """SigLIP2 配布形の `README.md` 本文を組み立てる（純関数・末尾改行つき）。"""
    require_pipeline(manifest, SIGLIP2_SUPPORTED_PIPELINE)
    return render(
        (
            frontmatter(_siglip2_metadata(manifest)),
            ["", f"# {SIGLIP2_TITLE}", ""],
            _siglip2_overview(manifest),
            [""],
            _siglip2_base_weights(manifest),
            [""],
            models(manifest),
            [""],
            _siglip2_usage(manifest, repo),
            *model_sections(manifest, (files, quants, _siglip2_shape)),
        )
    )
