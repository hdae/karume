"""Wan2.1 配布形のモデルカード（`README.md`）— manifest から機械導出する純関数。

汎用の描画部品（frontmatter・モデル一覧・quant 表・節の組み立て）は `karume.modelcard` が持つ。
ここが持つのは **Wan2.1 固有の事実**だけ: 帰属（出所・pin した revision・ライセンス）と、この
pipeline のカードに何を書くか（第 1 段の固定プロンプト・受理する入力・数値の門・実行資源の実測）。

MUST: **数値・ダウンロード量・quant 表・dtype ラベル・既定値は 1 つ残らず manifest から導出する**
（`karume.modelcard` の同 MUST がそのまま掛かる）。ここが持ってよい定数は manifest に**存在しない
事実**だけ — 上流の取得元と pin（`wan.sources.SOURCES` が正本）・固定プロンプトの表
（`wan.prompts.FIXED_PROMPTS` が正本 — 組み立ての門 `wan.distribution.assert_text_embeds` が資産の
メタとの一致を見るので、ここに描く本文と配る資産は食い違わない）・TS 側の受理集合（
`packages/models/src/wan/pipeline.ts` の `ACCEPTED_SIZES` / `MIN_FRAMES` / `MAX_FRAMES`）・
実行資源の実測（`_wan_resources` — ADR 0089 決定 3）。

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
from wan.sources import SOURCES

#: このテンプレートが説明できるパイプライン契約（ADR 0041 §2 — モデル単位）。
WAN_SUPPORTED_PIPELINE = "wan/1"

#: HF の pipeline tag（上流と同じ）。
WAN_PIPELINE_TAG = "text-to-video"

#: 上流の技術レポート。
WAN_PAPER = "arxiv.org/abs/2503.20314"

#: 原文の在処（配布リポ直下の `LICENSE.md` と同じテキスト — Apache 2.0 §4(a)）。
WAN_LICENSE_TEXT_LINK = "https://www.apache.org/licenses/LICENSE-2.0"

#: TS 側が受理する寸法とフレーム数（`packages/models/src/wan/pipeline.ts` の受理集合の写し —
#: manifest に無い事実）。81 フレームまでの解禁は ADR 0118 段 8。
#: MUST: TS 側と同じ値（`packages/models/tests/fixtures/wan-card-limits.json` を挟んで両側の
#: テストが突き合わせる — 片側だけ変えると赤）。
WAN_ACCEPTED_SIZES: tuple[tuple[int, int], ...] = ((832, 480), (480, 832))
WAN_FRAMES = (5, 33)

#: 実行資源（``_wan_resources``）を実測した quant 席。MUST: カードは数を**この席の数として**
#: 名乗る — 席に無い配布形では描かない（実測していない席の数は名乗らない — BiRefNet の
#: カードと同じ）。
WAN_RESOURCE_QUANT = "f16"


def _upstream(name: str) -> Any:
    """モデル名 → 上流の取得元（表に無ければ描かない — 出所を名乗れないカードは出さない）。"""
    source = SOURCES.get(name)
    if source is None:
        raise ValueError(
            f"モデル '{name}' の上流が取得元の表（wan.sources.SOURCES）に無い"
            f"（既知: {sorted(SOURCES)}）— 出所を名乗れないカードは描かない"
        )
    return source


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
        # f32 の上流を f16 へ落とし直した配布形（`CardMetadata` の doc — f16 / i8 の配布形は
        # `quantized`）。
        base_model_relation="quantized",
        license=next(iter(licenses)),
        tags=(WAN_PIPELINE_TAG, "video-generation", "wan", "webgpu"),
    )


def _wan_overview(manifest: Mapping[str, Any]) -> list[str]:
    return [
        "## What is this",
        "",
        "A **text-to-video** distribution of Wan2.1 T2V 1.3B, converted into the WebGPU",
        "inference runtime **Karume**'s container format (a `.krm` part sequence whose first part",
        "carries the graph and model descriptors).",
        "",
        "- Three graphs: `transformer` (the diffusion transformer, on patchified latent tokens)",
        "  and `vae_decoder_first` / `vae_decoder_next` (the video VAE decoder, one latent frame",
        "  per call, with the causal cache passed in and out).",
        "- The rest runs on the host in TypeScript: the prompt lookup, the patchify and RoPE",
        "  tables, classifier-free guidance as two batch-1 passes, the flow-matching UniPC",
        "  scheduler, and the tiled VAE decode. The output is `[3, frames, height, width]`",
        "  float32 in `[-1, 1]`.",
        "- Verified end to end in Deno (Intel Arc B570, Deno 2.9.6). Browsers are not verified",
        "  yet.",
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
        "  conversion into the Karume container format; every parameter rounded to the nearest",
        "  float16 value (weight matrices and kernels stored as float16, computation in float32);",
        "  the transformer graph re-expressed on patchified tokens with real-valued RoPE tables;",
        "  the VAE decoder re-expressed as two one-frame graphs with an explicit causal cache,",
        "  always decoded in overlapping tiles. No retraining and no fine-tuning.",
        "- **The text encoder (umT5-XXL) is not included.** The repository ships its outputs for",
        "  a fixed set of prompts instead (see below), computed with the upstream encoder in",
        "  bfloat16 and stored as float32.",
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
            ],
            disposable="await using",
        ),
        "",
        "// Only the precomputed prompts are accepted (see Prompts below).",
        'const prompt = pipeline.prompts.find((entry) => entry.name === "boxing-cats")!;',
        "const video = await pipeline.generate({",
        "  prompt: prompt.prompt,",
        "  seed: 42,",
        "  // frames: 33, // 4n+1, 5 to 33",
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
        "`generate()` opens one stage at a time — the transformer session is disposed before the",
        "VAE sessions are opened — so the transformer and the VAE are never resident together.",
        "Concurrent calls are queued. Weights are fetched once and cached (verified against",
        "`karume.json`'s `size` / `sha256`). GPU memory and time per clip are listed under",
        "Resources below.",
    ]


def _wan_prompts() -> list[str]:
    """第 1 段の受理集合（固定プロンプトの表 — 原文は逐語で出す）。"""
    lines = [
        "## Prompts",
        "",
        "The text encoder is not part of this distribution yet: the repository carries the",
        "umT5-XXL outputs of the prompts below, and `generate()` accepts exactly these strings",
        "(the original text or its normalized form — `pipeline.prompts` lists both). Any other",
        "string is rejected with `ModelInputError` before anything runs on the GPU. When",
        "`negativePrompt` is omitted, the `negative` row is used. The texts are shown here for",
        "reading; pass the strings from `pipeline.prompts`, since the match is exact (`ferret`,",
        "for one, starts and ends with a line break).",
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
        f"- **size**: {sizes}.",
        f"- **frames**: 4n+1 from {low} to {high}. Only 832 × 480 with {high} frames has been",
        "  checked end to end on the GPU; longer clips (up to 81 frames) are planned.",
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
        "- **Against the upstream reference**: a 2-step run with injected noise is compared with",
        "  diffusers on CPU in float32 (the same f16-rounded weights, the same tiled decode), and",
        "  one transformer forward at full size against a float64 reference. Differences stay",
        "  within tolerances measured on separate decision cases; the remaining gap is float32",
        "  rounding in the GPU matrix products, not a porting difference.",
        "- **Tiled decode**: the VAE always decodes in overlapping tiles, so the frames differ",
        "  slightly from the upstream untiled decode.",
    ]


def _wan_resources(manifest: Mapping[str, Any]) -> list[str]:
    """実行資源の目安（ADR 0089 決定 3 — 中間テンソルの確保と束縛の大きさは manifest の
    `requiredLimits` が数えない、**manifest に存在しない事実**。BiRefNet のカードと同じ扱い）。

    数は 2026-10-02 の B570・Deno の 50 ステップの通し（ADR 0118 段 6 の検収）の実測: VRAM は
    fdinfo の `drm-total-vram0` の段ごとの山・時間は transformer 1 パス（batch 1）と VAE の
    タイル decode 全体。
    1 パス × 100 + VAE ≈ 1,809 s は通しの 1,811 s と合う。中間テンソルの大きさは形からの計算
    （FFN の `[14040, 8960]` f32 = 503,193,600 B）。

    MUST: 実測していない条件の数は載せない — 受理集合の別の寸法・フレーム数・別の GPU の数を推し
    量って書かない。ブラウザで動くかは未確認（ADR 0118 の後段）なので、そう書く。
    """
    for name, model in manifest["models"].items():
        if WAN_RESOURCE_QUANT not in model["quants"]:
            raise ValueError(
                f"実行資源を実測した quant '{WAN_RESOURCE_QUANT}' がモデル '{name}' の席に無い"
                f"（席: {sorted(model['quants'])}）— 実測していない席の数は名乗らない"
            )
    return [
        "## Resources",
        "",
        "Measured on 2026-10-02 on an Intel Arc B570 in Deno 2.9.6, with the"
        f" `{WAN_RESOURCE_QUANT}` quant at",
        "832 × 480, 33 frames, 50 steps and guidance 5 (classifier-free guidance runs two",
        "transformer passes per step, so 100 passes):",
        "",
        "- **GPU memory**: the peak of the total allocation (the driver's fdinfo) is 6.19 GiB in",
        "  the transformer stage and 3.32 GiB in the VAE stage. The two stages are never resident",
        "  together, so the transformer stage's peak is the peak of a clip.",
        "- **Time**: about 30 minutes per clip — 16.8 s per transformer pass and 129 s for the",
        "  tiled VAE decode.",
        "- **Storage buffer size**: at this size some of the transformer's intermediate",
        "  tensors are larger than WebGPU's default `maxStorageBufferBindingSize` (128 MiB) — the",
        "  feed-forward activation `[14040, 8960]` in float32 alone is about 480 MiB. The runtime",
        "  requests the adapter's own limits, which Deno grants on the B570; in a browser the",
        "  environment has to grant the adapter's limits as well. Browsers have not been checked",
        "  yet, so whether they run this model is not known.",
        "- `karume.json` does not declare these figures: its declared limits cover the resident",
        "  weights and state, not the intermediate tensors a run allocates.",
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
    manifest: Mapping[str, Any], repo: str, host_assets: Mapping[str, int] = {}
) -> str:
    """Wan2.1 配布形の `README.md` 本文を組み立てる（純関数・末尾改行つき）。"""
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
            *model_sections(manifest, (partial(quants, host_assets=host_assets), _wan_defaults)),
        )
    )
