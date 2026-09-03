"""Irodori 配布形のモデルカード（`README.md`）— manifest から機械導出する純関数。

汎用の描画部品（frontmatter・モデル一覧・quant 表・節の組み立て）は
`karume.modelcard` が持つ。ここが持つのは **Irodori 固有の事実**だけ: 帰属（重みの出所・
同梱するコーデックと text backbone・ライセンス）と、この pipeline のカードに何を書くか。

帰属プロファイルは 1 つだけ（上流 1 リポの重みを格納形へ落とし直したもの）— SBV2 のように
声のファミリーで出所が割れる軸を持たないので、{@link irodori.distribution.PIPELINE} は
`card_profiles` に 1 席しか置かない。

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
    knob,
    model_sections,
    models,
    quants,
    render,
    require_pipeline,
)

#: このテンプレートが説明できるパイプライン契約（ADR 0041 §2 — モデル単位）。
IRODORI_SUPPORTED_PIPELINE = "irodori/1"

IRODORI_PIPELINE_TAG = "text-to-speech"


@dataclass(frozen=True)
class IrodoriUpstream:
    """配布モデル 1 つぶんの上流の事実（manifest に存在しないのでここが持つ — 冒頭の MUST）。"""

    #: 重みの出所（`inputs/irodori/<モデル名>/` に手で置く HF リポ）。
    repo: str
    #: タイトルと概要の太字に使う表示名（上流の名乗りに合わせる）。
    display: str


#: 配布モデル名 → 上流の事実。v4.1-small は duration predictor のみ再学習した後継
#: （他 683 テンソルは v4 とビット同一・tokenizer もバイト同一 — 2026-09-01 実測）。
#: 表に無いモデル名は fail loudly — 黙って別の版の帰属を書くと出所表記が現物と食い違う。
#: ライセンスは実地確認（2026-08-12 / v4.1 は 2026-09-01・HF の models API の `tags`）:
#: いずれも `license: mit`。
IRODORI_UPSTREAMS: Mapping[str, IrodoriUpstream] = {
    "v4-small": IrodoriUpstream(
        repo="Aratako/Irodori-TTS-v4-Small", display="Irodori-TTS v4 Small"
    ),
    "v4.1-small": IrodoriUpstream(
        repo="Aratako/Irodori-TTS-v4.1-Small", display="Irodori-TTS v4.1 Small"
    ),
}

#: text backbone の素になった日本語 ModernBERT（チェックポイントの `config_json` が
#: `text_tokenizer_repo` / `text_encoder_revision` で名指ししている 1 本）。上流の版に依らず
#: 同一（v4 → v4.1 で text encoder は凍結 — 上の実測と同根）。再配布しているので帰属に並べる。
#: ライセンスは実地確認（2026-08-12）: `license: mit`。
IRODORI_TEXT_BACKBONE_MODEL = "sbintuitions/modernbert-ja-310m"


def irodori_upstream(model_name: str) -> IrodoriUpstream:
    """配布モデル名から上流の事実を引く（未知の名前は fail loudly）。"""
    if model_name not in IRODORI_UPSTREAMS:
        known = " / ".join(sorted(IRODORI_UPSTREAMS))
        raise ValueError(
            f"モデル名 '{model_name}' の上流が {__name__}.IRODORI_UPSTREAMS に無い"
            f"（既知: {known}）— 新しい版はまず出所と表示名をこの表に足す"
        )
    return IRODORI_UPSTREAMS[model_name]


#: 波形へ落とすコーデック（上流では別リポ・別重み — この配布形には**同梱している**）。
#: ライセンスは MIT（`docs/research/2026-08-11-irodori-source-recon.md` の実地確認）。
#: 再配布しているので `base_model` にも帰属節にも並べる。
IRODORI_CODEC_MODEL = "Aratako/Semantic-DACVAE-Japanese-32dim"


def _irodori_metadata(upstream: IrodoriUpstream) -> CardMetadata:
    return CardMetadata(
        pipeline_tag=IRODORI_PIPELINE_TAG,
        base_model=(upstream.repo, IRODORI_TEXT_BACKBONE_MODEL, IRODORI_CODEC_MODEL),
        # 既定 quant（`i8-a8`）の DL 実体が int8 系列なので `quantized` を宣言する（sbv2 /
        # anima と同型。
        # 旧「f32 のまま・quantized ではない」は i8 系列同梱前の陳腐化した前提だった）。
        base_model_relation="quantized",
        license="mit",
        tags=("text-to-speech", "webgpu", "japanese"),
    )


#: 使い方スニペットのデモ入力（日本語 TTS の入力なので日本語のまま — CLAUDE.md の言語規約）。
IRODORI_DEMO_TEXT = "こんにちは、これはテストです。"
IRODORI_DEMO_CAPTION = "落ち着いた女性の声で、ゆっくりと丁寧に話している。"


def _irodori_overview(manifest: Mapping[str, Any], upstream: IrodoriUpstream) -> list[str]:
    sample_rate = default_model(manifest)["pipelineConfig"]["sampleRate"]
    return [
        "## What is this",
        "",
        f"A Japanese text-to-speech distribution: the **{upstream.display}** rectified-flow DiT",
        "converted into the WebGPU inference runtime **Karume**'s container format (a single",
        "safetensors file = weights + a graph JSON embedded in `__metadata__`). Runs as-is in the",
        "browser and in Deno.",
        "",
        "- Eight graphs make up the chain: a shared Japanese ModernBERT `backbone`, the",
        "  `text_proj` and `caption_proj` condition projectors, the reference-latent `speaker`",
        "  encoder, the `duration` predictor, the `dit` itself (one forward per Euler step, plus",
        "  one per classifier-free-guidance branch), and the DACVAE codec"
        f" ([{IRODORI_CODEC_MODEL}](https://huggingface.co/{IRODORI_CODEC_MODEL}))",
        "  as `codec_decoder` / `codec_encoder`.",
        "- **Text in, waveform out.** `generate()` returns f32 mono samples at the codec's sample",
        "  rate, ready for `encodeWav`. The decoder is run in tiles so that it also fits GPUs with",
        "  the default 128MiB storage-buffer limit; the tiling is bit-exact against a single-shot",
        "  decode.",
        "- **Voice cloning is wired up both ways**: a reference speaker is passed either as audio",
        "  (mono f32 — what `decodeWav` returns), which `codec_encoder` turns into a DACVAE",
        "  latent, or as such a latent directly. Reference audio must already be at this",
        f"  distribution's own {sample_rate} Hz: there is no resampler, and a mismatch is refused",
        "  rather than silently converted. Unlike the decoder, `codec_encoder` is not tiled, so a",
        "  long reference can exceed the default 128MiB storage-buffer limit.",
        "- Not readable by the upstream implementation (it's a different container with an embedded"
        f" graph); the reader is a pipeline that implements `{IRODORI_SUPPORTED_PIPELINE}`.",
        f"- Exporter used for the conversion: `{manifest['generator']}`. The distribution manifest"
        f" is `karume.json` (`{manifest['format']}`).",
    ]


def _irodori_base_weights(upstream: IrodoriUpstream) -> list[str]:
    """帰属節。変換 + 量子化（int8 系列の生成）を主張する — 再学習・fine-tune はしていない。"""
    return [
        "## Base weights and attribution",
        "",
        "Converted into the container format — the original checkpoint is not distributed here.",
        "",
        f"- **Weights**: [{upstream.repo}](https://huggingface.co/{upstream.repo}),"
        " licensed **MIT**",
        "  (as of retrieval). The training / inference implementation it comes with",
        "  ([Aratako/Irodori-TTS](https://github.com/Aratako/Irodori-TTS)) is MIT as well;",
        "  Karume's runtime contains none of that code — it is an independent implementation that",
        "  reads these weights from its own container format.",
        f"- **Text backbone**: fine-tuned from"
        f" [{IRODORI_TEXT_BACKBONE_MODEL}](https://huggingface.co/{IRODORI_TEXT_BACKBONE_MODEL}),",
        "  licensed **MIT** (as of retrieval). It is redistributed here in the container format as",
        "  the `backbone` component, so that license travels with this repository too.",
        f"- **Codec**: [{IRODORI_CODEC_MODEL}](https://huggingface.co/{IRODORI_CODEC_MODEL}),"
        " licensed **MIT**",
        "  (as of retrieval). Upstream ships it as a separate repository; it is redistributed here",
        "  in the container format as the `codec_decoder` / `codec_encoder` components so that",
        "  text-to-audio runs from this repository alone.",
        "- **Changes made here**: conversion into the Karume container format and **quantization**",
        "  of the weights — every component is stored as `f32` / `f16` / `i8` series, and `dit`",
        "  adds an `i4` series rounded with GPTQ calibration (the quant table below says which",
        "  storage each quant selects). No retraining, no fine-tuning — the `f32` series is the",
        "  source checkpoint's own values, re-laid out per graph.",
    ]


def _irodori_usage(manifest: Mapping[str, Any], repo: str) -> list[str]:
    """Usage 例の方針: 動く最小形は生かし、**普通のユースケースで使いそうな optional は
    コメントアウトで併記**する（選べる値も同じ行のコメントに列挙 — manifest から機械導出する
    ので、系列が増えれば列挙も追従する）。読者がコメントを外すだけで次の一歩へ進める形。

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
    sample_rate = model["pipelineConfig"]["sampleRate"]
    return [
        "## Usage",
        "",
        "```ts",
        'import { decodeWav, encodeWav, IrodoriPipeline } from "jsr:@karume/models";',
        "",
        *from_pretrained(
            "IrodoriPipeline",
            repo,
            [
                f'  // model: "{model_name}", // default — available: {model_names}',
                f'  // quant: "{quant}", // default — available: {quant_names}',
            ],
        ),
        "",
        "const audio = await pipeline.generate({",
        f'  text: "{IRODORI_DEMO_TEXT}",',
        "",
        "  // Voice Design — describe the voice in Japanese prose:",
        f'  // caption: "{IRODORI_DEMO_CAPTION}",',
        "",
        "  // Voice cloning — condition on a reference speaker. The WAV must already be",
        f"  // {sample_rate} Hz mono or stereo (there is no resampler; a mismatch is refused):",
        '  // speaker: { audio: decodeWav(await Deno.readFile("reference.wav")) },',
        "  // ...or pass a DACVAE latent you saved earlier instead of the audio:",
        "  // speaker: { latent: savedLatent },",
        "",
        "  seed: 42, // same seed + same inputs → same audio",
        "  // durationSeconds: 5, // override the predicted utterance length (seconds)",
        "});",
        "",
        "await Deno.writeFile(",
        '  "out.wav",',
        "  encodeWav(audio.data, audio.sampleRate),",
        ");",
        "```",
        "",
        "`generate()` returns `{ data, sampleRate, frames, seed, forwards }`, where `data` is f32",
        "mono already trimmed to the predicted length. `generateLatent()` is the same run stopped",
        "one stage earlier: it returns `{ data, frames, latentDim, seed, forwards }` with the",
        "patched DACVAE latent, for callers that want the embedding rather than audio.",
        "`caption` and `speaker` are both optional: without them the voice is picked by the model",
        "alone, and the guidance branches for the missing conditions are skipped.",
        "Weights are fetched once and cached (verified against `karume.json`'s `size` / `sha256`).",
    ]


def _irodori_shape(model: Mapping[str, Any]) -> list[str]:
    """モデル固有の数（グラフの宣言と噛み合う側）— 利用者の入力上限がここで読める。"""
    config = model["pipelineConfig"]
    frame_rate = config["frameRate"]
    return [
        "### Shape",
        "",
        "Derived from the checkpoint's own config, and checked against the exported graphs when",
        "this repository was assembled.",
        "",
        f"- **text**: up to {config['maxTextLen']} tokens (BOS included), width"
        f" {config['textDim']}",
        f"- **caption**: up to {config['maxCaptionLen']} tokens (BOS included), width"
        f" {config['captionDim']}",
        f"- **reference speaker**: up to {config['speakerRows'] - 1} patched rows"
        f" ({config['speakerPatchSize']} latent frames each ="
        f" {(config['speakerRows'] - 1) * config['speakerPatchSize'] // frame_rate}s of audio),"
        f" width {config['speakerDim']}",
        f"- **latent**: up to {config['ditSymMax']} frames at {frame_rate} Hz"
        f" ({config['ditSymMax'] // frame_rate}s), width {config['latentDim']}",
        f"- **audio**: {config['sampleRate']} Hz mono, {config['hopLength']} samples per latent"
        " frame",
    ]


def _irodori_defaults(model: Mapping[str, Any]) -> list[str]:
    """実行時ノブ（`generate()` に渡さなかったものを埋める側）。"""
    config = model["pipelineConfig"]
    scales = config["cfgScales"]
    return [
        "### Defaults",
        "",
        "The sampler knobs are fixed by the manifest — `generate()` takes none of them.",
        "",
        f"- **steps**: {config['steps']} Euler steps (`initScale` {config['initScale']})",
        "- **guidance**: "
        + " / ".join(f"{name} {knob(scale)}" for name, scale in scales.items())
        + f", applied for t in [{config['cfgMinT']}, {config['cfgMaxT']}]",
        f"- **duration**: clamped to [{config['minSeconds']}, {config['maxSeconds']}] seconds"
        " (the duration predictor decides within that, unless `durationSeconds` is passed)",
        "",
        "`seed` is the one knob the manifest does not carry — it defaults to `0`, and the same",
        "seed with the same request gives the same audio.",
    ]


def render_irodori_model_card(manifest: Mapping[str, Any], repo: str) -> str:
    """Irodori 配布形の `README.md` 本文を組み立てる（純関数・末尾改行つき）。"""
    require_pipeline(manifest, IRODORI_SUPPORTED_PIPELINE)
    upstream = irodori_upstream(manifest["defaultModel"])
    return render(
        (
            frontmatter(_irodori_metadata(upstream)),
            ["", f"# {upstream.display} — Karume", ""],
            _irodori_overview(manifest, upstream),
            [""],
            _irodori_base_weights(upstream),
            [""],
            models(manifest),
            [""],
            _irodori_usage(manifest, repo),
            *model_sections(manifest, (quants, _irodori_shape, _irodori_defaults)),
        )
    )
