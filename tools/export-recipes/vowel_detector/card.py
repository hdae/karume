"""母音検出配布形のモデルカード（`README.md`）— manifest から機械導出する純関数。

汎用の描画部品（frontmatter・モデル一覧・ファイル表・quant 表・節の組み立て）は
`karume.modelcard` が持つ。ここが持つのは **母音検出固有の事実**だけ: 帰属（出所・
ライセンス・学習素材）と、この pipeline のカードに何を書くか。

帰属は 1 通りだけ（上流 1 リポ・1 ライセンス）なのでプロファイルの軸を持たない。

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
    from_pretrained,
    frontmatter,
    model_sections,
    models,
    quants,
    render,
    require_pipeline,
)

#: このテンプレートが説明できるパイプライン契約（ADR 0041 §2 — モデル単位）。
VOWEL_DETECTOR_SUPPORTED_PIPELINE = "vowel-detector/1"

#: HF の pipeline tag。フレーム単位の 8 クラス分類なので `audio-classification` が最も近い
#: （HF の語彙にフレーム分類の席は無い）。何を出すかは本文が説明する。
VOWEL_DETECTOR_PIPELINE_TAG = "audio-classification"

VOWEL_DETECTOR_TITLE = "Vowel Detector for Lip-sync — Karume"

#: 上流の配布（同じ重みの ONNX 版）。**再配布しているのはこの重み**なので `base_model` に置く。
#: ライセンスは上流 `LICENSE` / `NOTICE.txt`（MIT・(c) 2026 Spectopathy）。
VOWEL_DETECTOR_UPSTREAM = "hdae/vowel-detector"
VOWEL_DETECTOR_LICENSE = "mit"

#: 学習パイプラインの帰属（上流 `NOTICE.txt` の逐語）。**MIT は NOTICE の再配布を要求しないが、
#: 上流が「配布するなら帰属を残してほしい」と明記している**ので、カードが機械的に持ち回る。
#: 表を 1 つにしてあるのは、片方だけ動いたときに帰属が静かに欠けるのを避けるため。
VOWEL_DETECTOR_ATTRIBUTION: tuple[str, ...] = (
    "- **Teacher backbone (not distributed)**:"
    " [reazon-research/japanese-hubert-base-k2]"
    "(https://huggingface.co/reazon-research/japanese-hubert-base-k2)"
    " (**Apache-2.0**), © Reazon Human Interaction Laboratory. Used only to train an internal"
    " aligner/classifier; the teacher is **not** part of these weights — the CRNN is distilled"
    " from it.",
    "- **Reading scripts**:"
    " [ROHAN4600](https://github.com/mmorise/rohan4600) (**CC0-1.0**, © Masanori Morise) and the"
    " [ITA corpus](https://github.com/mmorise/ita-corpus) (public domain).",
    "- **Real speech**:"
    " [Common Voice ja v25.0](https://commonvoice.mozilla.org/ja/datasets) (**CC0-1.0**),"
    " © Mozilla Foundation and Common Voice contributors — weak g2p labels,"
    " confidence-filtered.",
    "- **Synthesized speech**: generated with"
    " [Style-Bert-VITS2](https://github.com/litagin02/Style-Bert-VITS2) (**AGPL-3.0**, used as a"
    " tool only — not distributed with, and not part of, these weights) from"
    " [AivisHub](https://hub.aivis-project.com/) voice models under"
    " [ACML 1.0](https://github.com/Aivis-Project/ACML/blob/master/ACML-1.0.md) or CC0:"
    " まお / コハク / morioki / fumifumi / 阿井田茂 / にせ / ろてじん / らせつん / 観測症"
    " (ACML 1.0) and zonoko (CC0). ACML 1.0 permits every use outside its listed prohibitions,"
    " and machine-learning training is not among them.",
    "- **Evaluation only (does not influence the weights)**: JSUT basic5000, with reference"
    " boundaries from automatic forced alignment.",
)


def _vowel_detector_metadata() -> CardMetadata:
    return CardMetadata(
        pipeline_tag=VOWEL_DETECTOR_PIPELINE_TAG,
        base_model=(VOWEL_DETECTOR_UPSTREAM,),
        # `base_model_relation` は置かない — 格納形を変えず（f32 のまま）コンテナだけを移した
        # もので、adapter / merge / quantized / finetune のどれでもない（CardMetadata の doc）。
        license=VOWEL_DETECTOR_LICENSE,
        tags=(VOWEL_DETECTOR_PIPELINE_TAG, "lip-sync", "japanese", "webgpu"),
    )


#: 入力フレームの hop（10ms 格子）。特徴抽出の契約そのもので、`pipelineConfig` には無い
#: （実行時に選べない数を宣言だけ持たせない — `src/vowel-detector/config.ts` の判断）。
#: ここが要るのは**秒に直して説明する**ためだけ。
_VOWEL_DETECTOR_HOP = 160


def _vowel_detector_seconds(config: Mapping[str, Any], frames: int) -> str:
    """フレーム数 → 秒（`sampleRate` から導く — 秒を宣言として持たない）。"""
    per_second = config["sampleRate"] / _VOWEL_DETECTOR_HOP
    return f"{frames / per_second:.1f}"


def _vowel_detector_overview(manifest: Mapping[str, Any]) -> list[str]:
    config = default_model(manifest)["pipelineConfig"]
    longest = _vowel_detector_seconds(config, config["maxFrames"])
    return [
        "## What is this",
        "",
        "A small Japanese **vowel-sequence detector for lip-sync**, converted into the WebGPU",
        "inference runtime **Karume**'s container format (a single safetensors file = weights + a",
        "graph JSON embedded in `__metadata__`). Runs as-is in the browser and in Deno.",
        "",
        "- Audio in, a **`.lab` timeline** out: `start end label` lines over the 7 lip-sync"
        " classes (`a` / `i` / `u` / `e` / `o` / `N` / `pau`), on a 20 ms grid.",
        "- **Feature extraction and post-processing are included.** The pipeline computes the"
        f" {config['featureDim']}-dimensional features (log-mel + DSP) with the mel filterbank"
        " shipped here, then runs Viterbi smoothing, short-run merging and consonant absorption"
        " on the logits.",
        "- **Decoding and resampling are yours.** The entry point is a"
        f" `Float32Array` of {config['sampleRate']} Hz mono samples; this repository ships no"
        " WAV parser and no resampler.",
        "- **One graph, any length.** The time axis is symbolic, so a clip of any duration runs"
        " through the same graph with no padding and no length buckets. Audio longer than"
        f" {longest} s is rejected rather than silently truncated — split it and call twice.",
        "- Not readable by transformers (it's a different container with an embedded graph); the"
        f" reader is a pipeline that implements `{VOWEL_DETECTOR_SUPPORTED_PIPELINE}`.",
        f"- Exporter used for the conversion: `{manifest['generator']}`. The distribution manifest"
        f" is `karume.json` (`{manifest['format']}`).",
    ]


def _vowel_detector_base_weights() -> list[str]:
    """帰属節。格納形を変えていないので「変換したもの」としてだけ主張する。"""
    return [
        "## Base weights and attribution",
        "",
        f"- **Weights**: [{VOWEL_DETECTOR_UPSTREAM}]"
        f"(https://huggingface.co/{VOWEL_DETECTOR_UPSTREAM}), licensed"
        f" **{VOWEL_DETECTOR_LICENSE}** — the same trained CRNN this repository re-exports.",
        "- **Architecture**: two 1-D convolutions (the second halves the time axis) followed by a",
        "  2-layer bidirectional GRU and a linear head. Bidirectional means the model is",
        "  **offline** — it needs the whole utterance, not a stream.",
        "- **Changes made here**: conversion into the Karume container format. No retraining, no",
        "  fine-tuning and **no quantization** — the weights are the source checkpoint's own f32",
        "  values, byte for byte. The graph is the upstream `forward`, with the GRU expressed as",
        "  one scan node per layer and direction so that the time axis stays symbolic (bit-exact",
        "  against eager `nn.GRU`, checked on every export).",
        "",
        "The upstream project asks that the attributions below travel with the model, so they are",
        "reproduced here in full.",
        "",
        *VOWEL_DETECTOR_ATTRIBUTION,
    ]


def _vowel_detector_usage(manifest: Mapping[str, Any], repo: str) -> list[str]:
    """Usage 例の方針: 動く最小形は生かし、**普通のユースケースで使いそうな optional は
    コメントアウトで併記**する（選べる値も同じ行のコメントに列挙 — manifest から機械導出する
    ので、モデル / quant が増えれば列挙も追従する）。読者がコメントを外すだけで次の一歩へ
    進める形。

    `detect()` は波形 1 本しか受けない（後処理の閾値も含めて実行時に選べる欄が無い —
    `VowelDetectorPipeline` の公開面）ので、この pipeline の optional ノブは model / quant の
    2 つだけである。
    """
    model_name = manifest["defaultModel"]
    model = default_model(manifest)
    config = model["pipelineConfig"]
    model_names = " / ".join(sorted(manifest["models"]))
    quant_names = " / ".join(sorted(model["quants"]))
    return [
        "## Usage",
        "",
        "```ts",
        'import { VowelDetectorPipeline } from "jsr:@karume/models";',
        "",
        *from_pretrained(
            "VowelDetectorPipeline",
            repo,
            [
                f'  // model: "{model_name}", // default — available: {model_names}',
                f'  // quant: "{model["defaultQuant"]}", // default — available: {quant_names}',
            ],
            disposable="await using",
        ),
        "",
        f"// {config['sampleRate']} Hz mono samples in [-1, 1]. Decoding and resampling are the",
        "// caller's job (`decodeWav` from the same package reads a WAV, but never resamples).",
        "const { segments, lab } = await pipeline.detect(samples);",
        "",
        "// lab is the ready-to-write file body; segments is the same timeline as objects:",
        '// [{ start: 0.04, end: 0.4, label: "a" }, ...]',
        'await Deno.writeTextFile("voice.lab", lab);',
        "```",
        "",
        "`detect()` builds a GPU session per call and tears it down afterwards; concurrent calls",
        "are queued rather than run side by side.",
        "Weights are fetched once and cached (verified against `karume.json`'s `size` /",
        "`sha256`). You can also build from bytes you fetched yourself",
        "(`VowelDetectorPipeline.fromAssets`).",
    ]


def _vowel_detector_shape(model: Mapping[str, Any]) -> list[str]:
    """入出力と運用上限（利用者が渡すもの・受け取るものがここで読める）。"""
    config = model["pipelineConfig"]
    limit = config["maxFrames"]
    return [
        "### Input, output and limits",
        "",
        "The feature contract comes from the upstream feature configuration; the shapes are the",
        "input and output declarations of the exported graph, checked against it when this",
        "repository was assembled.",
        "",
        f"- **input**: `Float32Array`, {config['sampleRate']} Hz, mono, samples in [-1, 1].",
        f"- **features** (computed for you): {config['featureDim']} dimensions per 10 ms frame —"
        " log-mel normalized per utterance, plus voicing, log-energy and zero-crossing rate.",
        f"- **classes**: {', '.join(f'`{name}`' for name in config['classes'])} — in this order"
        " (the order *is* the class id). `cons` is absorbed into the neighbouring vowel during",
        "  post-processing, so it never reaches the `.lab`.",
        f"- **length**: any, up to **{limit} frames of 10 ms**"
        f" ({_vowel_detector_seconds(config, limit)} s). The clip runs at its own length — there",
        "  is no padding and no bucketing, so the numbers do not depend on how long the clip is.",
        "  An odd number of frames drops the last one (the output grid is 20 ms).",
        "- **output**: one `.lab` line per run of frames, at 20 ms resolution.",
        "",
        "The limit is the symbolic upper bound the graph was exported with, not a property of the",
        "weights: past it the pipeline refuses the clip instead of truncating it. Split longer",
        "recordings and call `detect` per part.",
    ]


def render_vowel_detector_model_card(manifest: Mapping[str, Any], repo: str) -> str:
    """母音検出配布形の `README.md` 本文を組み立てる（純関数・末尾改行つき）。"""
    require_pipeline(manifest, VOWEL_DETECTOR_SUPPORTED_PIPELINE)
    return render(
        (
            frontmatter(_vowel_detector_metadata()),
            ["", f"# {VOWEL_DETECTOR_TITLE}", ""],
            _vowel_detector_overview(manifest),
            [""],
            _vowel_detector_base_weights(),
            [""],
            models(manifest),
            [""],
            _vowel_detector_usage(manifest, repo),
            *model_sections(manifest, (files, quants, _vowel_detector_shape)),
        )
    )
