"""Depth Anything V2 配布形のモデルカード（`README.md`）— manifest から機械導出する純関数。

汎用の描画部品（frontmatter・モデル一覧・quant 表・節の組み立て）は
`karume.modelcard` が持つ。ここが持つのは **Depth Anything V2 固有の事実**だけ: 帰属（出所・
ライセンス・引用）と、この pipeline のカードに何を書くか。

帰属は**サイズから一意に決まる**（{@link DEPTH_ANYTHING_UPSTREAM}）ので、プロファイルの軸には
割らない — 分けると「Base を Small の帰属で配る」取り違えを操作者が起こせるようになる
（しかも Base は CC BY-NC 4.0）。

MUST: **数値・ダウンロード量・quant 表・dtype ラベルは 1 つ残らず manifest から導出する**
（`karume.modelcard` の同 MUST がそのまま掛かる）。ここが持ってよい定数は、manifest に
**存在しない事実**だけ。
"""

from __future__ import annotations

from collections.abc import Mapping
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

from .measurements import convt_diff_text

#: このテンプレートが説明できるパイプライン契約（ADR 0041 §2 — モデル単位）。
DEPTH_ANYTHING_SUPPORTED_PIPELINE = "depth-anything/1"

#: HF の pipeline tag（上流と同じ）。出すのは相対深度の地図 1 枚で、metric depth ではない。
DEPTH_ANYTHING_PIPELINE_TAG = "depth-estimation"

#: 上流の論文（Depth Anything V2）。
DEPTH_ANYTHING_PAPER = "arxiv.org/abs/2406.09414"

#: モデル名 → 上流チェックポイントの HF リポ ID。**この 1 表が「どのモデルが何の重みか」の
#: 唯一の事実**で、`depth_anything.distribution` は系列名 / 入力素材のディレクトリ名（どちらも
#: 上流リポ名そのもの — `depth_anything/export.py` の綴り）をここから導く。載っていないモデル名
#: は帰属を書けないので、カードは描かずに落ちる。
#:
#: MUST: **Small だけが Apache-2.0**（実地確認 2026-08-14 — HF の model API:
#: `depth-anything/Depth-Anything-V2-Base-hf` と `-Large-hf` はどちらも `license:
#: cc-by-nc-4.0`）。サイズ軸そのものは配布形の軸（1 サイズ = 1 リポ）だが、**この表に
#: 載っているものしか配れない**のが唯一の門で、NC の重みを Apache-2.0 のカードで再配布する
#: 事故はここで止まる。Base / Large を足すときは、ライセンス欄をモデル単位へ割る改修と
#: セット（{@link DEPTH_ANYTHING_LICENSE} は今 1 値しかない）。
DEPTH_ANYTHING_UPSTREAM: Mapping[str, str] = {
    "small": "depth-anything/Depth-Anything-V2-Small-hf",
}

#: 実地確認（2026-08-14 — HF の model API）: `license: apache-2.0`。**表に載っている全モデルが
#: この 1 値**である前提で frontmatter を書く（上の MUST）。
DEPTH_ANYTHING_LICENSE = "apache-2.0"

#: 原文の在処（配布リポ直下の `LICENSE.md` と同じテキスト — Apache 2.0 §4(a)）。
DEPTH_ANYTHING_LICENSE_TEXT_LINK = "https://www.apache.org/licenses/LICENSE-2.0"

#: `ConvTranspose2d` → 1×1 conv + pixel shuffle の差し替え（`depth_anything.patch` の
#: ③）が持ち込む差の実測幅。**ビット同一ではない**ので帰属節が明示する
#: （`depth_anything/export.py --verify` が毎回実測する値・深度の RMS はおよそ 1.0）。
#: 数も書式も持たない — 正本は {@link depth_anything.measurements.convt_diff_text}。
DEPTH_ANYTHING_CONVT_DIFF = convt_diff_text()


def _depth_anything_repo(name: str) -> str:
    """モデル名 → 上流リポ ID（帰属表に無ければ描かない）。"""
    repo = DEPTH_ANYTHING_UPSTREAM.get(name)
    if repo is None:
        raise ValueError(
            f"モデル '{name}' の上流が帰属表に無い（既知: {sorted(DEPTH_ANYTHING_UPSTREAM)}）"
            " — 出所を名乗れないカードは描かない"
        )
    return repo


def _depth_anything_title(manifest: Mapping[str, Any]) -> str:
    """見出し（`Depth Anything V2 Small — Karume`）。

    サイズの綴りは**上流リポ名から導く**（1 リポ 1 サイズなので、既定モデルの上流が
    そのままリポの名前）— 2 つ目の表を持つと、片方だけ動いたときに「Base の重みを Small
    として売る」カードが黙って作れる。
    """
    checkpoint = _depth_anything_repo(manifest["defaultModel"]).split("/", 1)[1]
    return f"{checkpoint.removesuffix('-hf').replace('-', ' ')} — Karume"


def _depth_anything_metadata(manifest: Mapping[str, Any]) -> CardMetadata:
    """frontmatter を manifest に並んだモデルから組む（`base_model` は再配布する上流の全部）。

    MUST: 未知のモデル名では描かない — 帰属の表に無いモデルを黙って落とすと、`base_model` が
    1 つ足りないカード（= 出所を名乗っていない再配布）が静かに出る。
    """
    return CardMetadata(
        pipeline_tag=DEPTH_ANYTHING_PIPELINE_TAG,
        base_model=tuple(_depth_anything_repo(name) for name in manifest["models"]),
        # `base_model_relation` は置かない — 格納形を変えず（f32 のまま）コンテナだけを移した
        # もので、adapter / merge / quantized / finetune のどれでもない（CardMetadata の doc）。
        license=DEPTH_ANYTHING_LICENSE,
        tags=(DEPTH_ANYTHING_PIPELINE_TAG, "depth", "monocular-depth-estimation", "webgpu"),
    )


def _depth_anything_overview(manifest: Mapping[str, Any]) -> list[str]:
    config = default_model(manifest)["pipelineConfig"]
    return [
        "## What is this",
        "",
        "A monocular **relative depth** distribution, converted into the WebGPU inference runtime",
        "**Karume**'s container format (a single safetensors file = weights + a graph JSON",
        "embedded in `__metadata__`). Runs as-is in the browser and in Deno.",
        "",
        "- One graph, one call: pixels in, one **f32 depth value per pixel** out, at the same size",
        "  as the image you handed over.",
        "- **Relative depth has no unit and no origin** — larger means nearer, and only the",
        "  ordering carries meaning. This is not a metric-depth checkpoint: the numbers are not",
        "  metres and are not comparable across images.",
        "- **Pre- and post-processing are included.** The pipeline resizes to"
        f" {config['imageWidth']} × {config['imageHeight']}, normalizes with the constants below,",
        "  runs the graph, and scales the depth map back to the original resolution. Decoding",
        "  PNG / JPEG is *not* part of this — use `createImageBitmap` in the browser, or any",
        "  decoder in Deno.",
        "- **Normalization and colouring are yours.** The pipeline returns the raw f32 map, not a",
        "  `[0, 1]` image: folding it with min/max would throw the scale away, and which colour",
        "  map to use is a decision this repository should not make for you.",
        "- Not readable by transformers (it's a different container with an embedded graph); the"
        f" reader is a pipeline that implements `{DEPTH_ANYTHING_SUPPORTED_PIPELINE}`.",
        f"- Exporter used for the conversion: `{manifest['generator']}`. The distribution manifest"
        f" is `karume.json` (`{manifest['format']}`).",
    ]


def _depth_anything_base_weights(manifest: Mapping[str, Any]) -> list[str]:
    """帰属節。格納形を変えていないので「変換したもの」としてだけ主張する。"""
    lines = [
        "## Base weights and attribution",
        "",
        "Converted into the container format — the original checkpoints are not distributed here.",
        "",
    ]
    for name in manifest["models"]:
        repo = _depth_anything_repo(name)
        lines.append(
            f"- **`{name}`**: [{repo}](https://huggingface.co/{repo}), licensed"
            f" **{DEPTH_ANYTHING_LICENSE}** (as of retrieval;"
            f" [full text]({DEPTH_ANYTHING_LICENSE_TEXT_LINK}) — a verbatim copy is in"
            " `LICENSE.md`)."
        )
    lines += [
        "- **Only the Small checkpoint is Apache-2.0.** Upstream ships Base and Large under"
        " **CC BY-NC 4.0**, so they are not converted or redistributed here.",
        "- **Training data**: see the upstream model card. Depth Anything V2 is distilled from a",
        "  teacher trained on synthetic data and then trained on pseudo-labelled real images —",
        "  check the upstream sources against your own use case.",
        f"- **Architecture**: DINOv2 backbone + DPT head"
        f" ([{DEPTH_ANYTHING_PAPER}](https://{DEPTH_ANYTHING_PAPER})).",
        "- **Changes made here** (also listed in `NOTICE.md`, per Apache 2.0 §4(b)):",
        "  conversion into the Karume container format. No retraining, no fine-tuning and",
        "  **no quantization** — the weights are the source checkpoint's own f32 values. The",
        "  graph is the upstream `forward` with two layout-only rewrites (the last fusion",
        "  stage's upsample takes an explicit output size instead of a scale factor, and the",
        "  position-embedding interpolation is pinned to the pretraining resolution where it",
        "  is the identity — both bit-exact), plus one module rewrite that is equivalent up",
        "  to floating-point rounding: the DPT reassemble stage's transposed convolutions",
        "  became a 1×1 convolution followed by a pixel shuffle (they have",
        "  `kernel == stride`, so the two are exactly the same sum in a different order —",
        f"  measured max {DEPTH_ANYTHING_CONVT_DIFF} on depth values whose RMS is around 1).",
    ]
    return lines


def _depth_anything_usage(manifest: Mapping[str, Any], repo: str) -> list[str]:
    """Usage 例の方針: 動く最小形は生かし、**普通のユースケースで使いそうな optional は
    コメントアウトで併記**する（選べる値も同じ行のコメントに列挙 — manifest から機械導出する
    ので、サイズ / quant が増えれば列挙も追従する）。読者がコメントを外すだけで次の一歩へ
    進める形。

    `estimate()` は画像 1 枚しか受けない（解像度は事前学習の正方形でグラフに焼かれていて実行時
    に選べない — `DepthAnythingPipeline` の公開面）ので、この pipeline の optional ノブは
    model / quant の 2 つだけである。

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
        'import { DepthAnythingPipeline } from "jsr:@karume/models";',
        "",
        *from_pretrained(
            "DepthAnythingPipeline",
            repo,
            [
                f'  // model: "{model_name}", // default — available: {model_names}',
                f'  // quant: "{quant}", // default — available: {quant_names}',
            ],
            disposable="await using",
        ),
        "",
        "// RGB8, row-major, 3 bytes per pixel. Decoding is the caller's job.",
        "const depth = await pipeline.estimate({ data: pixels, width, height });",
        "",
        "// depth.data is one f32 per pixel, same width/height as the input, larger = nearer.",
        "// Fold it to [0, 1] yourself when you want to look at it:",
        "let min = Infinity;",
        "let max = -Infinity;",
        "for (const value of depth.data) {",
        "  if (value < min) min = value;",
        "  if (value > max) max = value;",
        "}",
        "const span = max - min;",
        "const gray = depth.data.map((value) => (span > 0 ? (value - min) / span : 0));",
        "```",
        "",
        "`estimate()` keeps one GPU session alive for the lifetime of the pipeline, so processing",
        "many images uploads the weights once; concurrent calls are queued rather than run side by",
        "side. Weights are fetched once and cached (verified against `karume.json`'s `size` /",
        "`sha256`).",
    ]


def _depth_anything_shape(model: Mapping[str, Any]) -> list[str]:
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
        f" {config['imageHeight']} ({config['interpolation']}, antialiased). Note the filter:"
        " this checkpoint asks for **bicubic**, unlike most image towers.",
        "- The aspect ratio is **not** preserved. The graph is baked at the single square",
        "  pretraining resolution (the position embeddings are tied to the patch grid), so a",
        "  non-square photo is stretched rather than letterboxed. Crop it yourself first if that",
        "  matters for your images.",
        f"- **normalization**: `(pixel / 255 - mean) / std`, mean {mean}, std {std}",
        "- **output**: one f32 per pixel at the size of the image you passed in — relative depth,",
        "  non-negative (the head ends in a ReLU, so far regions sit at exactly 0), larger =",
        "  nearer. The graph itself emits the map at the resized resolution; scaling it back to",
        "  your image happens on the host, bilinearly.",
    ]


def render_depth_anything_model_card(manifest: Mapping[str, Any], repo: str) -> str:
    """Depth Anything V2 配布形の `README.md` 本文を組み立てる（純関数・末尾改行つき）。"""
    require_pipeline(manifest, DEPTH_ANYTHING_SUPPORTED_PIPELINE)
    return render(
        (
            frontmatter(_depth_anything_metadata(manifest)),
            ["", f"# {_depth_anything_title(manifest)}", ""],
            _depth_anything_overview(manifest),
            [""],
            _depth_anything_base_weights(manifest),
            [""],
            models(manifest),
            [""],
            _depth_anything_usage(manifest, repo),
            *model_sections(manifest, (quants, _depth_anything_shape)),
        )
    )
