"""Wan2.1 配布形のモデルカード（`README.md`）— manifest から機械導出する純関数。

汎用の描画部品（frontmatter・モデル一覧・quant 表・節の組み立て）は `karume.modelcard` が持つ。
ここが持つのは **Wan2.1 固有の事実**だけ: 帰属（出所・pin した revision・ライセンス）と、この
pipeline のカードに何を書くか（第 1 段の固定プロンプト・受理する入力・数値の門・実行資源の実測）。

MUST: **数値・ダウンロード量・quant 表・dtype ラベル・既定値は 1 つ残らず manifest から導出する**
（`karume.modelcard` の同 MUST がそのまま掛かる — text_encoder の取得量・越境参照の先・宣言された
device limit も manifest から引く）。ここが持ってよい定数は manifest に**存在しない事実**だけ —
上流の取得元と pin（`wan.sources.SOURCES` が正本）・固定プロンプトの表（`wan.prompts.FIXED_PROMPTS`
が正本 — 組み立ての門 `wan.distribution.assert_text_embeds` が資産のメタとの一致を見るので、ここに
描く本文と配る資産は食い違わない）・TS 側の受理集合（`packages/models/src/wan/pipeline.ts` の
`ACCEPTED_SIZES` / `MIN_FRAMES` / `MAX_FRAMES`・テキストの経路の選択 `textEncoder` — ADR 0119
追記 B・プロンプトの受理規則 — ADR 0119 決定 1 / 2 / 4 と追記 10a）・実行資源の実測
（`_wan_resources` — ADR 0089 決定 3）。

MUST: torch を import しない（`import dist` が torch を読まない —
`tests/test_dist_driver.py` の `TestImportingTheDriver`）。
"""

from __future__ import annotations

from collections.abc import Mapping
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
from wan.prompts import FIXED_PROMPTS
from wan.sources import SOURCES, UMT5_SOURCES, WAN21_MODELS
from wan.umt5_distribution import UMT5_DEFAULT_MODEL, UMT5_ROLE

#: このテンプレートが説明できるパイプライン契約（ADR 0041 §2 — モデル単位）。
WAN_SUPPORTED_PIPELINE = "wan/1"

#: HF の pipeline tag（上流と同じ）。
WAN_PIPELINE_TAG = "text-to-video"

#: 上流の技術レポート。
WAN_PAPER = "arxiv.org/abs/2503.20314"

#: 原文の在処（配布リポ直下の `LICENSE.md` と同じテキスト — Apache 2.0 §4(a)）。
WAN_LICENSE_TEXT_LINK = "https://www.apache.org/licenses/LICENSE-2.0"

#: TS 側が受理する寸法とフレーム数（`packages/models/src/wan/pipeline.ts` の受理集合の写し —
#: manifest に無い事実）。
#: MUST: TS 側と同じ値（`packages/models/tests/fixtures/wan-card-limits.json` を挟んで両側の
#: テストが突き合わせる — 片側だけ変えると赤）。
WAN_ACCEPTED_SIZES: tuple[tuple[int, int], ...] = ((832, 480), (480, 832))
WAN_FRAMES = (5, 81)

#: テキストの経路を選ぶ構築時のオプション（TS 側 `WanPipeline.fromPretrained` の第 2 引数 —
#: ADR 0119 追記 B。manifest に無い事実）。既定は GPU 経路（決定 7）。
WAN_TEXT_ENCODER_OPTION = "textEncoder"
WAN_TEXT_ENCODER_PATHS = ("gpu", "precomputed")

#: GPU 経路が受けるプロンプトの token 数（末尾の `</s>` を含む — ADR 0119 決定 4。TS 側の
#: 受理集合で、上限はトークナイザ資産の `maxLength`）。
WAN_PROMPT_TOKENS = (2, 512)

#: umT5 の部品名（manifest の weights のキー — `wan.distribution.WAN_TEXT_ENCODER_ROLE` と
#: 同じ正本）。
WAN_TEXT_ENCODER_COMPONENT = UMT5_ROLE

#: text encoder の上流（本家の encoder — ADR 0122 決定 1 / 3。この checkpoint の `text_encoder` とは
#: f32 でビット一致するが、出所は本家を名乗る — umT5 の配布形のカードと食い違わせない）。
WAN_TEXT_ENCODER_UPSTREAM = UMT5_SOURCES[UMT5_DEFAULT_MODEL].source.repo

#: 通しの実行資源（``_wan_resources`` の段ごとの表）を実測した quant 席。MUST: カードは数を
#: **この席の数として**名乗る — 席に無い配布形では描かない（実測していない席の数は名乗らない —
#: BiRefNet のカードと同じ）。
WAN_RESOURCE_QUANT = "f16"

#: 席ごとの transformer の実測（ADR 0120 追記「段 3 / 4 / 5 の結果」〈DiT 単体・通常モード・
#: B570〉と「段 6 の素材」〈実用席の 50 ステップ・33 フレームの通し・precomputed の経路〉・
#: 「50 ステップの実用席の sha 行 2 本」〈実用席の 81 フレームの通し: 壁 3,703 s = DiT 段 3,383 s
#: + VAE 段 319 s〉）。
#: 行は `(フレーム数, 1 forward, DiT 単体の確保, 50 ステップの通し)`。`f16` の行は比べる基準で、
#: 通しの欄は上の段ごとの表と同じ通し（ADR 0118 段 6 / 8）。参照席 `f16+dit8` の 1 forward は
#: r 門の計測（S = 14,040）で `f16` と同じ時間 — 確保と 81 フレームは計測していない。
#: MUST: 計測していない欄は「not measured」/「not run」と書き、推し量った数で埋めない。
#: 表に無い席は「未計測」と名乗る（席の並びは manifest のまま）。
WAN_QUANT_TRANSFORMER: Mapping[str, tuple[tuple[int, str, str, str], ...]] = {
    "f16": (
        (33, "17.1 s", "5.15 GiB", "~30 minutes"),
        (81, "68.5 s", "6.17 GiB", "~2 hours"),
    ),
    "f16+dit8": (
        (33, "same as `f16`", "not measured", "not run"),
        (81, "not measured", "not measured", "not run"),
    ),
    "f16+dit8-a8-attn8-s16": (
        (33, "8.5 s", "4.02 GiB", "952 s (~16 minutes)"),
        (81, "34.0 s", "5.46 GiB", "3,703 s (~62 minutes)"),
    ),
}

#: 実用席の step 1 の潜在の相対 RMS 誤差（参照席 `f16+dit8` に対して — ADR 0120 段 4 の自機
#: A/B 門の実測）。`(席, 参照席, 33 フレーム, 81 フレーム)`。カードが併記する視認の結果は ADR 0120
#: 裁定 2026-10-04 の 1（段 6 の 12 対 — `f16` と実用席・50 ステップ・33 フレーム）。
WAN_PRACTICAL_QUANT_ERROR = ("f16+dit8-a8-attn8-s16", "f16+dit8", "0.107", "0.210")


def _upstream(name: str) -> Any:
    """モデル名 → 上流の取得元（Wan2.1 のモデルでなければ描かない — このカードは Wan2.1 の
    配布の事実だけを書く。取得元の表は Wan2.2 の行も持つので、表に有ることでは通さない）。"""
    if name not in WAN21_MODELS:
        raise ValueError(
            f"モデル '{name}' は Wan2.1 のモデル（wan.sources.WAN21_MODELS）でない"
            f"（既知: {list(WAN21_MODELS)}）— Wan2.1 のカードに別の世代の出所を書かない"
        )
    return SOURCES[name]


def _wan_metadata(manifest: Mapping[str, Any]) -> CardMetadata:
    """frontmatter を manifest に並んだモデルから組む（`base_model` は再配布する上流の全部）。"""
    licenses = {_upstream(name).license for name in manifest["models"]}
    if len(licenses) != 1:
        raise ValueError(
            f"モデルごとにライセンスが割れている（{sorted(licenses)}）— 1 値で書けない"
        )
    return CardMetadata(
        pipeline_tag=WAN_PIPELINE_TAG,
        base_model=tuple(_upstream(name).repo for name in manifest["models"]),
        # f32 の上流を f16 / i8 へ落とし直した配布形（`CardMetadata` の doc — f16 / i8 の配布形は
        # `quantized`）。
        base_model_relation="quantized",
        license=next(iter(licenses)),
        tags=(WAN_PIPELINE_TAG, "video-generation", "wan", "webgpu"),
    )


def _text_encoder(manifest: Mapping[str, Any]) -> tuple[int, tuple[str, str] | None]:
    """既定モデルの既定 quant が取る text_encoder の容器の `(バイト数, 越境参照の先)`。

    越境参照の先は `(repo, revision)`（容器単位で全 part が同じ — ADR 0109 決定 3）で、自リポに
    置いた配布形なら `None`。カードは「どこから取るか」を manifest のこの事実から描く。
    """
    model = default_model(manifest)
    weights = model["weights"].get(WAN_TEXT_ENCODER_COMPONENT)
    if weights is None:
        raise ValueError(
            f"manifest の weights に '{WAN_TEXT_ENCODER_COMPONENT}' が無い — テキストの GPU 経路を"
            " 描けない"
        )
    label = model["quants"][model["defaultQuant"]]["weights"][WAN_TEXT_ENCODER_COMPONENT]
    parts = weights[label]["container"]["parts"]
    borrowed = {(ref["repo"], ref["revision"]) for ref in parts if "repo" in ref}
    if len(borrowed) > 1:
        raise ValueError(f"text_encoder の越境参照の先が 1 つでない: {sorted(borrowed)}")
    return sum(ref["size"] for ref in parts), next(iter(borrowed), None)


def _gib(size: int) -> str:
    """バイト数を GiB の 3 桁で綴る（このカードの資源の数と同じ書式）。"""
    return f"{size / (1 << 30):.2f} GiB"


def _wan_overview(manifest: Mapping[str, Any]) -> list[str]:
    _, borrowed = _text_encoder(manifest)
    where = (
        "stored in this repository"
        if borrowed is None
        else f"referenced from [`{borrowed[0]}`](https://huggingface.co/{borrowed[0]})"
    )
    return [
        "## What is this",
        "",
        "A **text-to-video** distribution of Wan2.1 T2V 1.3B, converted into the WebGPU",
        "inference runtime **Karume**'s container format (a `.krm` part sequence whose first part",
        "carries the graph and model descriptors).",
        "",
        f"- Four graphs: `{WAN_TEXT_ENCODER_COMPONENT}` (the umT5-XXL text encoder in int8,"
        f" {where}),",
        "  `transformer` (the diffusion transformer, on patchified latent tokens) and",
        "  `vae_decoder_first` / `vae_decoder_next` (the video VAE decoder, one latent frame per",
        "  call, with the causal cache passed in and out).",
        "- The rest runs on the host in TypeScript: the prompt cleaning and the tokenizer (or,",
        f'  with `{WAN_TEXT_ENCODER_OPTION}: "{WAN_TEXT_ENCODER_PATHS[1]}"`, the lookup of the',
        "  precomputed embeddings), the relative-position buckets, the patchify and RoPE tables,",
        "  classifier-free guidance as two batch-1 passes, the flow-matching UniPC scheduler, and",
        "  the tiled VAE decode. The output is `[3, frames, height, width]` float32 in `[-1, 1]`.",
        "- Verified end to end in Deno (Intel Arc B570, Deno 2.9.6): with the text encoder on the",
        "  GPU in 2-step runs (a fixed prompt and a free prompt, 832 × 480, 33 frames, each pinned",
        "  by the SHA-256 of its frames), and with the precomputed embeddings in full 50-step",
        "  runs. A 50-step run through the text encoder has not been done yet.",
        "- In a browser, Chrome on an NVIDIA GeForce RTX 5070 Ti finished one 50-step run (the",
        "  `f16` quant with the precomputed embeddings, 832 × 480, 81 frames) in 46.1 minutes.",
        "  The text encoder on the GPU and the int8 quants have not been run in a browser yet.",
        "- Not readable by diffusers (it's a different container with an embedded graph); the"
        f" reader is a pipeline that implements `{WAN_SUPPORTED_PIPELINE}`.",
        f"- Exporter used for the conversion: `{manifest['generator']}`. The distribution manifest"
        f" is `karume.json` (`{manifest['format']}`).",
    ]


def _wan_base_weights(manifest: Mapping[str, Any]) -> list[str]:
    """帰属節。上流の revision は pin した 40 桁を全部出す（容器の provenance と同じ値）。"""
    lines = [
        "## Base weights and attribution",
        "",
        "Converted into the container format — the original checkpoint is not distributed here.",
        "",
    ]
    for name in manifest["models"]:
        source = _upstream(name)
        lines.append(
            f"- **`{name}`**: [{source.repo}](https://huggingface.co/{source.repo}) at commit"
            f" `{source.revision}`, licensed **{source.license}** (as of retrieval;"
            f" [full text]({WAN_LICENSE_TEXT_LINK}) — a verbatim copy is in `LICENSE.md`)."
        )
    lines += [
        f"- **Authors**: Wan2.1 is released by Wan-AI ([technical report](https://{WAN_PAPER})).",
        "- **Changes made here** (listed in full in `NOTICE.md`, per Apache 2.0 §4(b)):",
        "  conversion into the Karume container format; every parameter of the float16",
        "  transformer and the VAE rounded to the nearest float16 value (weight matrices and",
        "  kernels stored as float16, computation in float32); a second, int8 copy of the",
        "  transformer whose linear weights are quantized per output channel (biases and",
        "  normalization weights kept in float32 at the source values);",
        "  the transformer graph re-expressed on patchified tokens, with the RoPE tables built on",
        "  the host from per-axis base tables and applied in a pair-swap form;",
        "  the VAE decoder re-expressed as two one-frame graphs with an explicit causal cache,",
        "  always decoded in overlapping tiles; the tokenizer converted into one table together",
        "  with lookup tables for the upstream prompt cleaning. No retraining and no fine-tuning.",
    ]
    _, borrowed = _text_encoder(manifest)
    if borrowed is None:
        lines.append(
            f"- **The text encoder** (the encoder of `{WAN_TEXT_ENCODER_UPSTREAM}`, bit-identical"
            " in float32 to this checkpoint's `text_encoder`, in int8) is stored in this"
            " repository."
        )
    else:
        repo, revision = borrowed
        lines += [
            "- **The text encoder is not stored here.** `karume.json` references the encoder of",
            f"  `{WAN_TEXT_ENCODER_UPSTREAM}` (bit-identical in float32 to this checkpoint's",
            "  `text_encoder`), converted to int8, at commit"
            f" `{revision[:16]}…` of [`{repo}`](https://huggingface.co/{repo})",
            "  (with the size and the SHA-256 of every part); that repository's `NOTICE.md` lists",
            "  the changes made to it.",
        ]
    lines += [
        "- **Precomputed embeddings**: the repository also ships the encoder's outputs for a",
        "  fixed set of prompts (see below), computed with the upstream encoder in bfloat16 and",
        "  stored as float32, for use without the text encoder.",
    ]
    return lines


def _wan_usage(manifest: Mapping[str, Any], repo: str) -> list[str]:
    """Usage 例: 動く最小形 + 普通に使いそうな optional はコメントアウトで併記する。

    NOTE: `fromAssets` は案内しない（Depth Anything のカードと同じ裁定 — HF から使う読者の入口は
    `fromPretrained`）。
    """
    model_name = manifest["defaultModel"]
    model = default_model(manifest)
    quant = model["defaultQuant"]
    model_names = " / ".join(manifest["models"])
    quant_names = " / ".join(model["quants"])
    return [
        "## Usage",
        "",
        "```ts",
        'import { encodePng, wanFrameToRgba, WanPipeline } from "jsr:@karume/models";',
        "",
        *from_pretrained(
            "WanPipeline",
            repo,
            [
                f'  // model: "{model_name}", // default — available: {model_names}',
                f'  // quant: "{quant}", // default — available: {quant_names}',
                f'  // {WAN_TEXT_ENCODER_OPTION}: "{WAN_TEXT_ENCODER_PATHS[0]}", // default —'
                f' "{WAN_TEXT_ENCODER_PATHS[1]}" skips the text encoder (see Prompts below)',
            ],
            disposable="await using",
        ),
        "",
        "const video = await pipeline.generate({",
        '  prompt: "A cat walks on the grass, realistic style.",',
        "  seed: 42,",
        '  // negativePrompt: "low quality, blurry", // default: the official negative prompt',
        f"  // frames: 33, // 4n+1, {WAN_FRAMES[0]} to {WAN_FRAMES[1]}",
        "  // width: 832, height: 480, // or 480 × 832",
        "});",
        "",
        "for (let frame = 0; frame < video.frames; frame += 1) {",
        "  const rgba = wanFrameToRgba(video, frame);",
        "  const png = await encodePng(rgba, video.width, video.height);",
        '  await Deno.writeFile(`frame-${String(frame).padStart(2, "0")}.png`, png);',
        "}",
        "```",
        "",
        "`generate()` opens one stage at a time — the text encoder session is disposed before the",
        "transformer session is opened, and the transformer session before the VAE sessions — so",
        "no two stages are resident together. Concurrent calls are queued. Weights are fetched",
        "once and cached (verified against `karume.json`'s `size` / `sha256`). GPU memory and time",
        "per clip are listed under Resources below.",
    ]


def _wan_prompts() -> list[str]:
    """プロンプトの受理集合（GPU 経路の規則と、precomputed の経路の固定プロンプトの表 — 原文は
    逐語で出す）。"""
    gpu, precomputed = WAN_TEXT_ENCODER_PATHS
    low, high = WAN_PROMPT_TOKENS
    lines = [
        "## Prompts",
        "",
        f'With the text encoder (`{WAN_TEXT_ENCODER_OPTION}: "{gpu}"`, the default), `generate()`',
        "accepts free text. The prompt goes through a port of the upstream prompt cleaning and",
        "the umT5 tokenizer on the host, and these are rejected with `ModelInputError` before",
        "anything runs on the GPU, where the upstream pipeline would silently truncate them,",
        "repair them or map them to an unknown token instead:",
        "",
        f"- fewer than {low} or more than {high} tokens (the end token included) — an empty prompt,"
        " or",
        "  one too long;",
        "- characters outside the tokenizer's vocabulary, and special tokens such as `</s>`",
        "  written in the text;",
        "- what the cleaning would treat as an HTML character reference (write `R & D`, not",
        "  `R&D`) or as mojibake;",
        "- C1 control characters and code points unassigned in Unicode 16.0.0.",
        "",
        "When `negativePrompt` is omitted, the official negative prompt (the `negative` row",
        "below) is used, and it goes through the text encoder too.",
        "",
        f'With `{WAN_TEXT_ENCODER_OPTION}: "{precomputed}"`, the text encoder is not fetched: the'
        " repository's",
        "umT5-XXL outputs of the prompts below are used, and `generate()` accepts exactly these",
        "strings (the original text or its normalized form — `pipeline.prompts` lists both). Any",
        "other string is rejected with `ModelInputError`. The texts are shown here for reading;",
        "pass the strings from `pipeline.prompts`, since the match is exact (`ferret`, for one,",
        "starts and ends with a line break).",
        "",
        "| Name | Role | Source |",
        "| ---- | ---- | ------ |",
    ]
    lines += [
        f"| `{prompt.name}` | {prompt.role} | [source]({prompt.url}) |" for prompt in FIXED_PROMPTS
    ]
    for prompt in FIXED_PROMPTS:
        lines += ["", f"`{prompt.name}`:", "", "```text", prompt.text.strip("\n"), "```"]
    return lines


def _wan_inputs() -> list[str]:
    """受理する入力と、数値の門（ビット同一・参照照合）の説明。"""
    sizes = " or ".join(f"{width} × {height}" for width, height in WAN_ACCEPTED_SIZES)
    low, high = WAN_FRAMES
    return [
        "## Accepted inputs",
        "",
        "- **prompt** / **negativePrompt**: see Prompts above.",
        f"- **size**: {sizes}.",
        f"- **frames**: 4n+1 from {low} to {high}. Only 832 × 480 has been checked end to end on",
        f"  the GPU: 33 and {high} frames with the precomputed embeddings, and 33 frames with the",
        "  text encoder.",
        "- **steps** ≥ 1, **guidance** ≥ 1 (1 turns classifier-free guidance off and the negative",
        "  prompt is then rejected), **shift** > 0.",
        "- **seed** (the host noise generator — not torch's `randn`) or the initial noise as",
        "  `latents`.",
        "",
        "Anything outside these sets throws `ModelInputError` before any weight reaches the GPU.",
        "",
        "## Determinism and verification",
        "",
        "- **Bit-identical runs**: the same request (prompt, seed or latents, knobs) produces the",
        "  same bytes on the same GPU and driver. Release verification pins the SHA-256 of the",
        "  8-bit frames per test environment and fails on any change — the check is never relaxed",
        "  to a tolerance.",
        "- **Against the upstream reference** (the `f16` quant, with the precomputed",
        "  embeddings): a 2-step run with injected noise is compared with diffusers on CPU in",
        "  float32 (the same f16-rounded weights, the same tiled decode), and one transformer",
        "  forward at each full size (33 and 81 frames) against a float64 reference. Differences",
        "  stay within tolerances measured on separate decision cases; the remaining gap is",
        "  float32 rounding in the GPU matrix products, not a porting difference.",
        "- **Tiled decode**: the VAE always decodes in overlapping tiles, so the frames differ",
        "  slightly from the upstream untiled decode.",
    ]


def _wan_resources(manifest: Mapping[str, Any]) -> list[str]:
    """実行資源の目安（ADR 0089 決定 3 — 中間テンソルの確保と束縛の大きさは manifest の
    `requiredLimits` が数えない、**manifest に存在しない事実**。BiRefNet のカードと同じ扱い）。

    数は B570・Deno の 50 ステップの通しの実測 — 33 フレームは 2026-10-02（ADR 0118 段 6 の検収）、
    81 フレームは 2026-10-03（段 8）: VRAM は fdinfo の `drm-total-vram0` の段ごとの山・時間は
    transformer 1 パス（batch 1）と VAE のタイル decode 全体。
    1 パス × 100 + VAE は通しと合う（33 フレーム: 16.8 × 100 + 129 ≈ 1,809 s / 1,811 s・
    81 フレーム: 68.6 × 100 + 315 ≈ 7,175 s / 7,173 s）。中間テンソルの大きさは形からの計算
    （FFN の `[S, 8960]` f32 は S = 14,040 で 503,193,600 B・S = 32,760 で 1,174,118,400 B。
    self-attention のスコアの行ブロック 1 枚は 81 フレームで 2,146,435,200 B —
    B570 の束縛上限 2,147,483,644 B の内）。

    text 段（GPU 経路）の数は ADR 0119 追記「段 10c の GPU の門と段 10d-4 の結果」の実測
    （2 ステップ・33 フレームの通し 1 回と、umT5 単体の 1 forward）。席ごとの transformer の数は
    `WAN_QUANT_TRANSFORMER` の出所のとおり。

    MUST: 実測していない条件の数は載せない — 受理集合の別の寸法・フレーム数・別の GPU の数を推し
    量って書かない。ブラウザは RTX 5070 Ti の Chrome で `f16` 席・資産の経路の 81 フレームの
    通しだけを確かめた（ADR 0118 追記「段 9 の結果」）ので、それ以外は未確認と書く。
    """
    for name, model in manifest["models"].items():
        if WAN_RESOURCE_QUANT not in model["quants"]:
            raise ValueError(
                f"実行資源を実測した quant '{WAN_RESOURCE_QUANT}' がモデル '{name}' の席に無い"
                f"（席: {sorted(model['quants'])}）— 実測していない席の数は名乗らない"
            )
    encoder_bytes, borrowed = _text_encoder(manifest)
    source = "" if borrowed is None else f" from `{borrowed[0]}`"
    # 席の並びは manifest のまま。表に無い席は数を推し量らず、未計測と名乗る。
    seats = dict.fromkeys(
        quant for model in manifest["models"].values() for quant in model["quants"]
    )
    measured = [quant for quant in seats if quant in WAN_QUANT_TRANSFORMER]
    unmeasured = [f"`{quant}`" for quant in seats if quant not in WAN_QUANT_TRANSFORMER]
    practical, reference, error_33, error_81 = WAN_PRACTICAL_QUANT_ERROR
    return [
        "## Resources",
        "",
        "Measured on an Intel Arc B570 in Deno 2.9.6, with the"
        f" `{WAN_RESOURCE_QUANT}` quant at 832 × 480,",
        "50 steps and guidance 5 (classifier-free guidance runs two transformer passes per step,",
        "so 100 passes); 33 frames on 2026-10-02 and 81 frames on 2026-10-03:",
        "",
        "| Frames | Transformer peak | VAE peak | Pass   | VAE decode | Clip        |",
        "| ------ | ---------------- | -------- | ------ | ---------- | ----------- |",
        "| 33     | 6.19 GiB         | 3.32 GiB | 16.8 s | 129 s      | ~30 minutes |",
        "| 81     | 7.31 GiB         | 3.78 GiB | 68.6 s | 315 s      | ~2 hours    |",
        "",
        "Pass is one transformer pass (batch 1); VAE decode is the tiled decode of the whole clip.",
        "These runs used the precomputed embeddings.",
        "",
        f'With the text encoder (`{WAN_TEXT_ENCODER_OPTION}: "{WAN_TEXT_ENCODER_PATHS[0]}"`),'
        " measured on 2026-10-03 in a 2-step run at",
        "832 × 480 and 33 frames: the text encoder stage (the umT5-XXL encoder run for the",
        "positive and the negative prompt, including building its session) took 10.4 s and",
        "peaked at 6.30 GiB; after it was disposed, 0.08 GiB more than before the stage remained",
        "allocated. In the same run the transformer stage peaked at 5.94 GiB and the VAE stage at",
        "3.32 GiB. One forward of the text encoder alone takes 0.14 s (8 tokens) to 1.73 s",
        "(488 tokens). A 50-step run through the text encoder has not been done yet.",
        "",
        "Per quant, at 832 × 480 (the 50-step clips with the precomputed embeddings and",
        "guidance 5):",
        "",
        "| Quant | Frames | Pass | Transformer allocation | 50-step clip |",
        "| ----- | ------ | ---- | ---------------------- | ------------ |",
        *(
            f"| `{quant}` | {frames} | {forward} | {allocation} | {clip} |"
            for quant in measured
            for frames, forward, allocation, clip in WAN_QUANT_TRANSFORMER[quant]
        ),
        "",
        "Pass and allocation here are of the transformer run alone (one pass, batch 1); the",
        "allocation is the runtime's own count of what it allocated, not the driver's total.",
        "",
        *(
            [f"The other quants ({' / '.join(unmeasured)}) have not been measured yet.", ""]
            if unmeasured
            else []
        ),
        *(
            [
                f"- **Quality of `{practical}`**: after the first step its latent differs from",
                f"  `{reference}`'s (the same int8 weights, computed in float32) by a relative",
                f"  RMS error of {error_33} at 33 frames and {error_81} at 81 frames, mostly",
                "  from the int8 attention. Side by side with `f16` on twelve 50-step clips at",
                "  33 frames (seeds 42 to 45 with the three fixed prompts), no clear degradation",
                "  was seen.",
            ]
            if practical in seats and reference in seats
            else []
        ),
        f"- **Download**: the quant table's Download column includes the text encoder"
        f" ({_gib(encoder_bytes)}{source});",
        f'  with `{WAN_TEXT_ENCODER_OPTION}: "{WAN_TEXT_ENCODER_PATHS[1]}"` it is not fetched.',
        "- **GPU memory**: the peaks are of the total allocation (the driver's fdinfo). The",
        "  stages are never resident together, so a clip's peak is the largest stage peak.",
        "- **Storage buffer size**: at these sizes some of the transformer's intermediate tensors",
        "  are larger than WebGPU's default `maxStorageBufferBindingSize` (128 MiB) — the",
        "  feed-forward activation `[S, 8960]` in float32 alone is about 480 MiB at 33 frames",
        "  (S = 14,040) and about 1.09 GiB at 81 frames (S = 32,760), and one row block of the",
        "  self-attention scores at 81 frames is 2.00 GiB, just under the 2 GiB binding limit the",
        "  adapter reports.",
        "  The runtime requests the adapter's own limits, which Deno grants on the B570; in a",
        "  browser the environment has to grant the adapter's limits as well. Chrome on the",
        "  RTX 5070 Ti granted them (2 GiB buffers) and ran the 81-frame clip above; other",
        "  browsers and GPUs have not been checked.",
        "- `karume.json` does not declare these figures: its declared limits cover the resident",
        "  weights and state, not the intermediate tensors a run allocates.",
        *_declared_limits(manifest),
    ]


def _declared_limits(manifest: Mapping[str, Any]) -> list[str]:
    """manifest が quant ごとに宣言した device limit（`requiredLimits` — 無ければ行も無い）。

    Wan の宣言は text_encoder の i8 の語彙埋め込み（1 バッファ）で決まる — どの席も同じ
    text_encoder を選ぶ（weights は完全写像）ので、全席が同じ宣言を持つ。席ごとに違う manifest は
    この 1 行では描けないので落とす（推し量って 1 つに畳まない）。
    """
    declared = [quant.get("requiredLimits") for quant in default_model(manifest)["quants"].values()]
    if not any(declared):
        return []
    if any(limits != declared[0] for limits in declared):
        raise ValueError(
            f"席ごとに requiredLimits が違う（{declared}）— 全席に同じ宣言が掛かる形しか描かない"
        )
    spelled = " and ".join(f"`{key}` ≥ {value:,} bytes" for key, value in declared[0].items())
    return [
        f"- **Declared limits**: every quant declares {spelled} in `karume.json` — the largest",
        "  resident weight is the text encoder's int8 vocabulary embedding, held as one buffer.",
        "  The declaration belongs to the quant, so it applies whether or not the text encoder is",
        "  used.",
    ]


def _wan_defaults(model: Mapping[str, Any]) -> list[str]:
    """`pipelineConfig` が持つ生成の既定（manifest から）。"""
    config = model["pipelineConfig"]
    defaults = config["defaults"]
    return [
        "### Defaults",
        "",
        "Any of these knobs not passed to `generate()` is filled in from the manifest:",
        "",
        f"- **steps**: {knob(defaults['steps'])}",
        f"- **guidance**: {knob(defaults['guidance'])}",
        f"- **shift** (flow-matching shift): {knob(config['scheduler']['shift'])}",
    ]


def render_wan_model_card(
    manifest: Mapping[str, Any],
    repo: str,
    abbreviations: Mapping[str, str],
    host_assets: Mapping[str, int] = {},
) -> str:
    """Wan2.1 配布形の `README.md` 本文を組み立てる（純関数・末尾改行つき）。

    `abbreviations` は席名の部品上書きトークンの対応表（正本は `wan.distribution` の
    `WAN_QUANT_ABBREVIATIONS` — ADR 0074 決定 4）。manifest に無い事実なので、定数として写さず
    引数で受ける（anima のカードと同じ形）。
    """
    require_pipeline(manifest, WAN_SUPPORTED_PIPELINE)
    return render(
        (
            frontmatter(_wan_metadata(manifest)),
            ["", "# Wan2.1 T2V 1.3B — Karume", ""],
            _wan_overview(manifest),
            [""],
            _wan_base_weights(manifest),
            [""],
            models(manifest),
            [""],
            _wan_usage(manifest, repo),
            [""],
            _wan_prompts(),
            [""],
            _wan_inputs(),
            [""],
            _wan_resources(manifest),
            *model_sections(
                manifest,
                (
                    partial(quants, abbreviations=abbreviations, host_assets=host_assets),
                    _wan_defaults,
                ),
            ),
        )
    )
