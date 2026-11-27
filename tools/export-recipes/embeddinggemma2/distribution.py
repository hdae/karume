"""The EmbeddingGemma 2 distribution recipe — series layout, output path table, quant table,
`pipelineConfig` and the card choice (ADR 0065 decision 2).

The general-purpose assembly engine (placement, folding shared files, sha256, manifest,
staging/swap, verification) lives in `karume.dist`. What lives here is **the facts specific to
EmbeddingGemma 2** only.

The repository `karume-embeddinggemma-2` holds one model today, `text` — the text tower of
`google/embeddinggemma-2`, its vision tower and its audio tower. Its contract is
`embeddinggemma/1`, the TypeScript family `embeddinggemma`. The model ships three graphs and one
asset:

- `model` — the text tower, mean pooling and L2 normalisation as one graph, `input_ids` /
  `pool_mask` `[1, T]` and the soft rows of a tower (`soft_embeds` `[1, T, 512]`, `soft_mask`
  `[1, T, 1]` — all false for a text) in, a unit vector `[1, 768]` out, under two dtype labels,
  each the container of its own series ({@link EMBEDDINGGEMMA2_SERIES}):
  - `bf16` — `embeddinggemma2/export.py --dtype bf16`: every weight in a weight slot is the
    checkpoint's own bf16 bits; the rest is f32 holding the same values widened;
  - `i8` — `embeddinggemma2/export.py --dtype i8`: the same weight slots (the token table and every
    linear) rounded to per-row symmetric int8, each with its F32 scale; the rest as in bf16.

  Both containers carry the same graph ({@link assert_same_graph}) — only the storage differs.
- `vision` — the vision tower with its projection (`embeddinggemma2/export_vision.py`, series
  {@link EMBEDDINGGEMMA2_VISION_SERIES}): an image's patch rows and positions in, its soft tokens
  `[1, n, 512]` out — the text graph's soft rows. The text graph's two dtype labels: `bf16`
  (`--dtype bf16` — the checkpoint's own values; the norms and the folded rotary tables f32) and
  `i8` (`--dtype i8` — the same weight slots, the linears and the two learned position tables,
  rounded to per-row int8). Both carry the same graph. The pipeline fetches it only when asked for
  image input.
- `audio` — the audio tower with its projection (`embeddinggemma2/export_audio.py`, series
  {@link EMBEDDINGGEMMA2_AUDIO_SERIES}): an audio's log-mel frames and their frame mask in, its
  soft tokens `[1, n, 512]` out — the text graph's soft rows. The same two labels: `bf16` (the
  checkpoint's own values; the norms, the biases and the folded constants f32) and `i8` (the
  linears and the depthwise convolutions rounded to per-row int8; the two subsampling convolutions
  and the window selector stay bf16 — the export's declared exceptions, held here by
  {@link assert_embeddinggemma2_audio_i8_kept}). Both carry the same graph. The pipeline fetches it
  only when asked for audio input; the host computes the frames itself, with the feature
  extractor's settings held here to the pinned `preprocessor_config.json`
  ({@link EMBEDDINGGEMMA2_AUDIO_FEATURE_EXTRACTOR}).
- `tokenizer` — the compiled tokenizer asset (`embeddinggemma2/tokenizer.py`), held to the pinned
  `tokenizer.json` it was compiled from.

Two quant presets, both with an empty `session` (f32 compute on the reference path), each mapping
every graph to its own storage: `bf16` (the original weights, the default — ADR 0128 decision 4)
and `i8` (the int8 storage of the three graphs; the maintainer's choices of 2026-11-22 for the text
tower, after the torch-CPU screening of `docs/research/2026-11-22-embeddinggemma2-quant-screen.md`,
and of 2026-11-23 for the towers, after the screening of
`docs/research/2026-11-23-embeddinggemma2-tower-quant-screen.md`).

`pipelineConfig` carries **the model's own values**, read from their sources at build time and
checked against each other ({@link embeddinggemma2_pipeline_config}):

- `prompts` — prompt name → prefix, verbatim from the pinned `config_sentence_transformers.json`;
- `dimensions` — the Matryoshka dimensions, the full one first (upstream's model card; the first
  is checked against the graph's output width);
- `maxTokens` — the most tokens one text may encode to (`<bos>` and `<eos>` included): the upper
  bound of the graph's symbol `T`, read from the graph's folded constants and held to the export's
  `SYM_MAX`;
- `placeholderTokens` — modality → the id of its placeholder token in the vocabulary, from the
  pinned `config.json` (`image_token_id` / `audio_token_id` / `video_token_id`). The upstream
  composite model replaces such an id with the pad id before the embedding; the text tower here
  does not, so a reader refuses a text that encodes to one.
- `vision` — what the host needs for an image ({@link embeddinggemma2_vision_config}), in the
  shape of the Gemma 4 family's `pipelineConfig.vision`: the soft-token budget
  (`processor_config.json`'s `max_soft_tokens`), the ids of the soft token and of the two markers
  around an image (`config.json`'s `image_token_id` / `boi_token_id` / `eoi_token_id`) and the
  width of a soft token (`text_config.hidden_size`, held to both graphs).
- `audio` — what the host needs for an audio ({@link embeddinggemma2_audio_config}): the ids of
  the soft token and of the two markers around an audio (`config.json`'s `audio_token_id` /
  `boa_token_id` / `eoa_token_index`), the width of a soft token, the most soft tokens one audio
  takes (the audio graph's bound, held to the soft tokens of the feature extractor's longest audio
  and to the series' record) and the sample rate (`preprocessor_config.json`'s `sampling_rate`).

The public surface is the one {@link PIPELINE} (`karume.dist.Pipeline`) — the repository's dist
driver (`tools/export-recipes/dist.py`) merges it into the core's PIPELINES.

MUST: torch-free (`import dist` must not load torch — `tests/test_dist_driver.py`). The writer's
constants this module needs are mirrored here and held equal by the tests.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

from _shared.container_read import read_layouts
from _shared.licenses import apache_license_2_0, license_markdown
from _shared.upstream import assert_upstream_provenance
from embeddinggemma2 import sources as upstream_sources
from embeddinggemma2.tokenizer import TOKENIZER_SERIES, TokenizerScriptError, assert_prompts
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
    ir_graph,
)

from .card import (
    EMBEDDINGGEMMA2_LICENSE,
    EMBEDDINGGEMMA2_MODEL_ROLE,
    EMBEDDINGGEMMA2_USE_POLICY_SENTENCE,
    render_embeddinggemma2_model_card,
)

#: The pipeline contract (ADR 0041 §2 — per model): the TypeScript family `embeddinggemma`, major 1.
EMBEDDINGGEMMA_PIPELINE = "embeddinggemma/1"

#: The one model of the repository (the text tower), also the default.
EMBEDDINGGEMMA2_MODEL = "text"

#: The distribution repository name (`karume-<family>` of ADR 0092 decision 2 — one repository for
#: this checkpoint, whichever of its towers it carries).
EMBEDDINGGEMMA2_REPO_NAME = "karume-embeddinggemma-2"

#: The weights key of the graph (= the container's graph name — container-v1 §2.1) and the role of
#: the tokenizer asset.
EMBEDDINGGEMMA2_ROLE = EMBEDDINGGEMMA2_MODEL_ROLE
EMBEDDINGGEMMA2_TOKENIZER_ROLE = "tokenizer"

#: The storage of the original weights (its dtype label and the default preset's name).
EMBEDDINGGEMMA2_DTYPE = "bf16"
#: The int8 storage (its dtype label and its preset's name).
EMBEDDINGGEMMA2_I8_DTYPE = "i8"

#: Dtype label → the series its container is read from — the mirror of
#: `embeddinggemma2.export.DEFAULT_OUT_DIRS` (that module imports torch; the tests hold the two
#: equal). The default preset's storage comes first.
EMBEDDINGGEMMA2_SERIES: Mapping[str, str] = {
    EMBEDDINGGEMMA2_DTYPE: "embeddinggemma-2-text-bf16",
    EMBEDDINGGEMMA2_I8_DTYPE: "embeddinggemma-2-text-i8",
}

#: The weights role of the vision tower (= the tower graph's name, `export_vision.GRAPH_NAME`) and,
#: per dtype label, the series its container is read from (`export_vision.DEFAULT_OUT_DIRS` — that
#: module imports torch; the tests hold the mirrors equal). The labels are the text graph's: each
#: preset maps the tower to the storage it gives the text graph.
EMBEDDINGGEMMA2_VISION_ROLE = "vision"
EMBEDDINGGEMMA2_VISION_SERIES: Mapping[str, str] = {
    EMBEDDINGGEMMA2_DTYPE: "embeddinggemma-2-vision-bf16",
    EMBEDDINGGEMMA2_I8_DTYPE: "embeddinggemma-2-vision-i8",
}

#: The weights role of the audio tower (= the tower graph's name, `export_audio.GRAPH_NAME`) and,
#: per dtype label, the series its container is read from (`export_audio.DEFAULT_OUT_DIRS`,
#: mirrored likewise).
EMBEDDINGGEMMA2_AUDIO_ROLE = "audio"
EMBEDDINGGEMMA2_AUDIO_SERIES: Mapping[str, str] = {
    EMBEDDINGGEMMA2_DTYPE: "embeddinggemma-2-audio-bf16",
    EMBEDDINGGEMMA2_I8_DTYPE: "embeddinggemma-2-audio-i8",
}

#: Weights role → dtype label → series, for the three graphs (one storage label set, the text
#: graph's first).
EMBEDDINGGEMMA2_GRAPH_SERIES: Mapping[str, Mapping[str, str]] = {
    EMBEDDINGGEMMA2_ROLE: EMBEDDINGGEMMA2_SERIES,
    EMBEDDINGGEMMA2_VISION_ROLE: EMBEDDINGGEMMA2_VISION_SERIES,
    EMBEDDINGGEMMA2_AUDIO_ROLE: EMBEDDINGGEMMA2_AUDIO_SERIES,
}

#: What the audio tower's i8 container keeps bf16 — the mirror of
#: `export_audio.I8_KEPT_BF16_WEIGHTS` (its two subsampling convolutions; the reason is declared
#: there, the tests hold the two equal) — and the suffix of the one more tensor it keeps bf16, the
#: window selector (a weight slot of no weight, one buffer shared by every layer:
#: `export_audio.WINDOW_SELECTOR`).
EMBEDDINGGEMMA2_AUDIO_I8_KEPT_BF16: tuple[str, ...] = ("layer0.conv.weight", "layer1.conv.weight")
EMBEDDINGGEMMA2_AUDIO_SELECTOR_SUFFIX = ".self_attn.window_selector"


def embeddinggemma2_role(dtype: str, role: str = EMBEDDINGGEMMA2_ROLE) -> str:
    """The artifact role of a graph (the text graph unless named) in one storage (`model_bf16`,
    `vision_i8` — the one word the placements, the output paths and the storage checks share)."""
    return f"{role}_{dtype}"


#: The representative container path inside a series and the asset file inside the tokenizer
#: series (mirrors of `embeddinggemma2.export.MODEL_FILE` and the name of
#: `embeddinggemma2.tokenizer.ASSET_PATH` — held equal by the tests).
EMBEDDINGGEMMA2_MODEL_FILE = "model.krm"
EMBEDDINGGEMMA2_TOKENIZER_FILE = "tokenizer.json"

#: The relative output paths (**inside the model subtree**) — the storage dtype in the file name, as
#: in the other families, so the two containers side by side cannot be mistaken.
EMBEDDINGGEMMA2_OUTPUT_PATHS: Mapping[str, str] = {
    **{
        embeddinggemma2_role(dtype, role): f"{role}/model.{dtype}.krm"
        for role, series in EMBEDDINGGEMMA2_GRAPH_SERIES.items()
        for dtype in series
    },
    EMBEDDINGGEMMA2_TOKENIZER_ROLE: (
        f"{EMBEDDINGGEMMA2_TOKENIZER_ROLE}/{EMBEDDINGGEMMA2_TOKENIZER_FILE}"
    ),
}

#: The storage each graph's binding table **must** hold ({@link assert_storage}) — the f32 series
#: holds none and fails here.
EMBEDDINGGEMMA2_STORAGE_REQUIREMENTS: Mapping[str, str] = {
    embeddinggemma2_role(dtype, role): dtype
    for role, series in EMBEDDINGGEMMA2_GRAPH_SERIES.items()
    for dtype in series
}

#: The storage each graph's binding table **must not** hold ({@link assert_storage_absent}). MUST:
#: every layout of the codec ledger but f32 / i32 and the role's own — a series in another
#: compressed storage also holds f32 parts (the norms, the folded constants, the int8 scales), so
#: the requirement alone lets it through, and only this table tells the series apart (a mixed-up
#: series root cannot be told apart by the numeric gates — ADR 0027 / 0029). Each bf16 role
#: forbidding i8 and each i8 role requiring it (and forbidding bf16) is what keeps the two presets
#: from swapping. The audio tower's i8 role alone holds bf16 beside i8 — its declared exceptions —
#: so it cannot forbid it; {@link assert_embeddinggemma2_audio_i8_kept} holds its bf16 tensors to
#: exactly those.
EMBEDDINGGEMMA2_STORAGE_FORBIDDEN: Mapping[str, tuple[str, ...]] = {
    **{
        embeddinggemma2_role(EMBEDDINGGEMMA2_DTYPE, role): ("f16", "i8", "i4", "i4f16", "i2")
        for role in EMBEDDINGGEMMA2_GRAPH_SERIES
    },
    embeddinggemma2_role(EMBEDDINGGEMMA2_I8_DTYPE): ("f16", "bf16", "i4", "i4f16", "i2"),
    embeddinggemma2_role(EMBEDDINGGEMMA2_I8_DTYPE, EMBEDDINGGEMMA2_VISION_ROLE): (
        "f16",
        "bf16",
        "i4",
        "i4f16",
        "i2",
    ),
    embeddinggemma2_role(EMBEDDINGGEMMA2_I8_DTYPE, EMBEDDINGGEMMA2_AUDIO_ROLE): (
        "f16",
        "i4",
        "i4f16",
        "i2",
    ),
}

#: The weights declaration (weights role → dtype label → artifact role). Every graph has the two
#: labels, so every preset names its own for each ({@link EMBEDDINGGEMMA2_QUANTS}).
EMBEDDINGGEMMA2_WEIGHTS: Mapping[str, Mapping[str, WeightFiles]] = {
    role: {dtype: WeightFiles(embeddinggemma2_role(dtype, role)) for dtype in series}
    for role, series in EMBEDDINGGEMMA2_GRAPH_SERIES.items()
}

#: The assets declaration (asset name → role).
EMBEDDINGGEMMA2_ASSETS: Mapping[str, str] = {
    EMBEDDINGGEMMA2_TOKENIZER_ROLE: EMBEDDINGGEMMA2_TOKENIZER_ROLE
}

#: The presets (ADR 0074 names: the storage alone, no knob). Both compute in f32 on the reference
#: path (`session` empty — ADR 0104), so neither needs a self A/B check (ADR 0147 decision 2).
#:
#: - `bf16` — the original weights (ADR 0128 decision 4), the checkpoint's own bf16 values in the
#:   reference summation order, for every graph. The label is the original-weights label of ADR
#:   0131 / 0154.
#: - `i8` — the same weight slots stored as per-row int8, in every graph (the maintainer's choices
#:   of 2026-11-22 for the text graph — one int8 preset, no activation quantization, no 4-bit
#:   preset — and of 2026-11-23 for the towers, the audio tower keeping its declared exceptions
#:   bf16).
#:
#: `label` / `description` are the selection UI's fields (ADR 0075 decision 1 — 64 / 200
#: characters; ADR 0128 C3 needs both once a model has two presets). They state the storage and
#: the compute, not the quality (the screening's numbers are the card's, with their source) and
#: not which preset is the default (`defaultQuant` says it — ADR 0075 decision 3).
EMBEDDINGGEMMA2_QUANTS: Mapping[str, Any] = {
    EMBEDDINGGEMMA2_DTYPE: {
        "weights": {role: EMBEDDINGGEMMA2_DTYPE for role in EMBEDDINGGEMMA2_GRAPH_SERIES},
        "session": {},
        "label": "Original weights (bf16)",
        "description": "The checkpoint's own bf16 weights with f32 compute.",
    },
    EMBEDDINGGEMMA2_I8_DTYPE: {
        "weights": {role: EMBEDDINGGEMMA2_I8_DTYPE for role in EMBEDDINGGEMMA2_GRAPH_SERIES},
        "session": {},
        "label": "Half size (int8)",
        "description": "The token table, image position tables and every linear layer and"
        " convolution stored as int8 (one scale per row), computed in f32; the audio tower's two"
        " subsampling convolutions stay bf16.",
    },
}
EMBEDDINGGEMMA2_DEFAULT_QUANT = EMBEDDINGGEMMA2_DTYPE

#: The graph's inputs in order (the writer's `GRAPH_INPUTS`, mirrored — held equal by the tests)
#: with their dtypes and the axes after `[1, T]` ({@link EMBEDDINGGEMMA2_HIDDEN} stands for the
#: token table's width: a soft row replaces a scaled lookup, so the two have one width), and the
#: token-length symbol.
EMBEDDINGGEMMA2_SYMBOL = "T"
EMBEDDINGGEMMA2_HIDDEN = "hidden"
EMBEDDINGGEMMA2_GRAPH_INPUTS: tuple[tuple[str, str, tuple[int | str, ...]], ...] = (
    ("input_ids", "i32", ()),
    ("pool_mask", "f32", ()),
    ("soft_embeds", "f32", (EMBEDDINGGEMMA2_HIDDEN,)),
    ("soft_mask", "bool", (1,)),
)

#: The most tokens one text may encode to — the mirror of `embeddinggemma2.export.SYM_MAX` (held
#: equal by the tests; the upstream model's context), and checked against the bound the graph's
#: folded constants carry ({@link embeddinggemma2_graph_max_tokens}).
EMBEDDINGGEMMA2_MAX_TOKENS = 8192

#: The Matryoshka dimensions, the full one first. The upstream model card at the pinned revision
#: states them ("The supported dimensions are 768, 512, 256, and 128"); no configuration file
#: declares them, so they are this recipe's constant, and the first is checked against the graph.
EMBEDDINGGEMMA2_DIMENSIONS: tuple[int, ...] = (768, 512, 256, 128)

#: `pipelineConfig.placeholderTokens`' keys → the `config.json` fields they are read from.
EMBEDDINGGEMMA2_PLACEHOLDER_FIELDS: tuple[tuple[str, str], ...] = (
    ("image", "image_token_id"),
    ("audio", "audio_token_id"),
    ("video", "video_token_id"),
)

#: `pipelineConfig.vision`'s id fields → the `config.json` fields they are read from (the soft
#: token is the image placeholder — the reader holds the two equal).
EMBEDDINGGEMMA2_VISION_ID_FIELDS: tuple[tuple[str, str], ...] = (
    ("imageTokenId", "image_token_id"),
    ("beginImageTokenId", "boi_token_id"),
    ("endImageTokenId", "eoi_token_id"),
)

#: The tower graph's inputs in order with their dtypes, and its symbol — the mirror of
#: `export_vision.GRAPH_INPUTS` / `SYMBOL` (held equal by the tests). The patch axis is
#: `pooling² · n` and a patch row `patch_size² · 3` values, both read from `processor_config.json`.
EMBEDDINGGEMMA2_VISION_INPUTS: tuple[tuple[str, str], ...] = (
    ("pixels", "f32"),
    ("position_x", "i32"),
    ("position_y", "i32"),
)
EMBEDDINGGEMMA2_VISION_SYMBOL = "n"

#: The vision series' record (`export_vision.REFERENCE_FILE`, mirrored).
EMBEDDINGGEMMA2_VISION_REFERENCE_FILE = "reference.json"

#: What the host's image preprocessing implements (`packages/models/src/embeddinggemma/vision.ts`):
#: the processor class and the settings it reads them with. MUST: the pinned
#: `processor_config.json` states exactly these — a resize filter, a rescale or a normalisation
#: the host does not apply would give other pixels with every check passing.
EMBEDDINGGEMMA2_IMAGE_PROCESSOR: Mapping[str, Any] = {
    "image_processor_type": "Gemma4ImageProcessor",
    "do_convert_rgb": True,
    "do_resize": True,
    "resample": 3,
    "do_rescale": True,
    "rescale_factor": 1 / 255,
    "do_normalize": False,
}

#: `pipelineConfig.audio`'s id fields → the `config.json` fields they are read from (the soft
#: token is the audio placeholder — the reader holds the two equal; upstream spells the end
#: marker's field `eoa_token_index`).
EMBEDDINGGEMMA2_AUDIO_ID_FIELDS: tuple[tuple[str, str], ...] = (
    ("audioTokenId", "audio_token_id"),
    ("beginAudioTokenId", "boa_token_id"),
    ("endAudioTokenId", "eoa_token_index"),
)

#: The audio tower graph's inputs in order with their dtypes, its symbol, and the mel frames per
#: soft token — the mirror of `export_audio.GRAPH_INPUTS` / `SYMBOL` / `FRAMES_PER_TOKEN` (held
#: equal by the tests). The frame axis is `4n` and a frame row `feature_size` values.
EMBEDDINGGEMMA2_AUDIO_INPUTS: tuple[tuple[str, str], ...] = (
    ("features", "f32"),
    ("frame_mask", "f32"),
)
EMBEDDINGGEMMA2_AUDIO_SYMBOL = "n"
EMBEDDINGGEMMA2_AUDIO_FRAMES_PER_TOKEN = 4

#: The audio series' record (`export_audio.REFERENCE_FILE`, mirrored).
EMBEDDINGGEMMA2_AUDIO_REFERENCE_FILE = "reference.json"

#: The longest audio upstream's feature extractor reads (its `max_length` default — not a setting
#: of the file, so the mirror of `export_audio.MAX_SAMPLES`, which `export_audio` holds to the
#: signature; the host refuses a longer audio, `AUDIO_MAX_SAMPLES` of
#: `packages/models/src/embeddinggemma/audio.ts`). Its soft tokens bound the audio graph's symbol.
EMBEDDINGGEMMA2_AUDIO_MAX_SAMPLES = 480_000

#: The `audio_seq_length` of the pinned `processor_config.json`, as recorded on 2026-11-23 (its
#: processor's default is 750). Upstream's serving helper caps an audio's soft tokens there (280 ×
#: 40 ms = 11.2 s); its processor path, which sentence-transformers and the host follow, does not —
#: up to 750 for 30 s ({@link embeddinggemma2_audio_seq_length}; `docs/limitations.md`).
EMBEDDINGGEMMA2_AUDIO_SEQ_LENGTH = 280

#: What the host's audio front end implements (`AUDIO_FEATURE_EXTRACTOR` of
#: `packages/models/src/embeddinggemma/audio.ts`): the feature extractor's class and every setting
#: that changes its frames. MUST: the pinned `preprocessor_config.json` — and the same block of
#: `processor_config.json` — state exactly these: a preemphasis, a dither, a per-bin normalisation
#: or another mel range the host does not apply would give other frames with every check passing.
EMBEDDINGGEMMA2_AUDIO_FEATURE_EXTRACTOR: Mapping[str, Any] = {
    "feature_extractor_type": "Gemma4AudioFeatureExtractor",
    "sampling_rate": 16_000,
    "feature_size": 128,
    "frame_length": 320,
    "hop_length": 160,
    "fft_length": 512,
    "fft_overdrive": False,
    "min_frequency": 0.0,
    "max_frequency": 8000.0,
    "mel_floor": 0.001,
    "preemphasis": 0.0,
    "dither": 0.0,
    "input_scale_factor": 1.0,
    "per_bin_mean": None,
    "per_bin_stddev": None,
}

#: The snapshot files the plan reads (each checked against the pin before it is read).
EMBEDDINGGEMMA2_READ_FILES: tuple[str, ...] = (
    "config.json",
    "config_sentence_transformers.json",
    "tokenizer.json",
    "processor_config.json",
    "preprocessor_config.json",
)

#: The format of the compiled tokenizer asset (the writer is `_shared/gemma_tokenizer.py`) and the
#: post-processor the graph's pooling was traced with (`<bos>` … `<eos>` — the pooled tokens
#: include both).
EMBEDDINGGEMMA2_TOKENIZER_FORMAT = "karume-gemma-tokenizer/1"
EMBEDDINGGEMMA2_POST_PROCESSOR = "bos-eos"

#: The upstream's similarity function (`config_sentence_transformers.json`) — the card says the dot
#: product of the unit vectors is the cosine, which holds for this one only.
EMBEDDINGGEMMA2_SIMILARITY = "cosine"


def embeddinggemma2_repo_name(_model: str) -> str:
    """The distribution repository name (the same for every model of the checkpoint)."""
    return EMBEDDINGGEMMA2_REPO_NAME


@dataclass(frozen=True)
class EmbeddingGemma2Sources:
    """The assembly inputs: the series of each storage (dtype label → directory) of the text
    graph, of the vision tower and of the audio tower, the tokenizer series and the pinned upstream
    snapshot (`inputs/` — not a generated artifact; only its small configuration files and
    `tokenizer.json` are read)."""

    series: Mapping[str, Path]
    vision: Mapping[str, Path]
    audio: Mapping[str, Path]
    tokenizer: Path
    snapshot: Path

    def of(self, role: str) -> Mapping[str, Path]:
        """The series of the graph of a weights role."""
        return {
            EMBEDDINGGEMMA2_ROLE: self.series,
            EMBEDDINGGEMMA2_VISION_ROLE: self.vision,
            EMBEDDINGGEMMA2_AUDIO_ROLE: self.audio,
        }[role]


def embeddinggemma2_sources(series_dir: Path) -> EmbeddingGemma2Sources:
    """Resolve the inputs from the series parent directory (`outputs/series/`) and the snapshot's
    place (`embeddinggemma2.sources.SNAPSHOT_DIR`)."""
    resolved = {
        role: {dtype: series_dir / name for dtype, name in series.items()}
        for role, series in EMBEDDINGGEMMA2_GRAPH_SERIES.items()
    }
    return EmbeddingGemma2Sources(
        series=resolved[EMBEDDINGGEMMA2_ROLE],
        vision=resolved[EMBEDDINGGEMMA2_VISION_ROLE],
        audio=resolved[EMBEDDINGGEMMA2_AUDIO_ROLE],
        tokenizer=series_dir / TOKENIZER_SERIES,
        snapshot=upstream_sources.SNAPSHOT_DIR,
    )


def embeddinggemma2_placements(sources: EmbeddingGemma2Sources) -> dict[str, Path]:
    """Role → source file. Anything not here stays out (the series' `io.*.safetensors` and the
    towers' series records).

    MUST: exactly the storages of {@link EMBEDDINGGEMMA2_GRAPH_SERIES} for each graph — a missing
    one would leave a preset pointing at no file, an extra one would ship a container no preset
    reads.
    """
    placements: dict[str, Path] = {}
    for role, series in EMBEDDINGGEMMA2_GRAPH_SERIES.items():
        given = sources.of(role)
        if set(given) != set(series):
            raise DistError(
                f"the {role} series are given for {sorted(given)}, not for the storages"
                f" {sorted(series)}"
            )
        for dtype in series:
            placements[embeddinggemma2_role(dtype, role)] = (
                given[dtype] / EMBEDDINGGEMMA2_MODEL_FILE
            )
    placements[EMBEDDINGGEMMA2_TOKENIZER_ROLE] = sources.tokenizer / EMBEDDINGGEMMA2_TOKENIZER_FILE
    return placements


def _read_json(path: Path, what: str) -> Any:
    if not path.is_file():
        raise DistError(f"missing assembly input ({what}): {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise DistError(f"{path} cannot be read as JSON ({what})") from error


def _positive_int(value: Any, where: str) -> int:
    # bool is a subclass of int: `true` must not pass as 1.
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise DistError(f"{where} is not a positive integer ({value!r})")
    return value


def embeddinggemma2_graph_max_tokens(
    graph: Mapping[str, Any], path: Path, symbol: str = EMBEDDINGGEMMA2_SYMBOL
) -> int:
    """The upper bound of a symbol (the token symbol `T` unless named), read from the graph's
    folded constants.

    The IR's `symbols` lists names only; the bound lives in the constants traced at the maximum and
    sliced by `sym_prefix_slice` (the text graph's RoPE tables — the sliding window is a declared
    band, not a constant; the audio graph's window bias of `n`). Every slice of the symbol must read
    a prefix of one extent — that extent is the most rows the graph can bind.

    MUST: fail on a slice whose form leaves the bound unreadable (a coefficient or an offset), on
    disagreeing extents and on a graph without one — `maxTokens` / `maxSoftTokens` would otherwise
    be a number nothing in the asset backs.
    """
    values = graph.get("values")
    nodes = graph.get("nodes")
    if not isinstance(values, dict) or not isinstance(nodes, list):
        raise DistError(f"{path}: the IR metadata has no values / nodes")
    extents: set[int] = set()
    for node in nodes:
        if not isinstance(node, dict) or node.get("op") != "sym_prefix_slice":
            continue
        attrs = node.get("attrs", {})
        if attrs.get("sym") != symbol:
            continue
        source = values.get(node["ins"][0], {})
        shape = source.get("shape") if isinstance(source, dict) else None
        for cut in attrs.get("slices", []):
            if cut.get("coeff") != 1 or cut.get("offset") != 0:
                raise DistError(
                    f"{path}: a sym_prefix_slice of {node['ins'][0]} cuts {cut} — the bound of"
                    f" {symbol} cannot be read from it"
                )
            if not isinstance(shape, list) or not 0 <= cut.get("dim", -1) < len(shape):
                raise DistError(f"{path}: the sliced constant {node['ins'][0]} has no shape")
            extents.add(_positive_int(shape[cut["dim"]], f"{path}: {node['ins'][0]}'s extent"))
    if len(extents) != 1:
        raise DistError(
            f"{path}: the folded constants sliced by {symbol} have the extents"
            f" {sorted(extents)} — exactly one bound is expected"
        )
    return extents.pop()


def assert_embeddinggemma2_graph(graph: Mapping[str, Any], path: Path) -> tuple[int, int, int]:
    """Check the graph's contract before placement and return `(vocabulary size, output width,
    soft-row width)` (the soft rows are the token table's width).

    The reader binds `input_ids` (i32) and `pool_mask` (f32) `[1, T]`, `soft_embeds` (f32)
    `[1, T, H]` with H the token table's width, and `soft_mask` (bool) `[1, T, 1]` by name and
    reads one f32 `[1, D]` vector: a graph of another form (a mask left as an input, a fixed
    length, the per-token states, soft rows of another width) has a valid manifest and fails only
    on the user's machine.
    """
    values = graph.get("values")
    lookups = [
        node
        for node in graph.get("nodes", [])
        if isinstance(node, dict) and node.get("op") == "embedding"
    ]
    if len(lookups) != 1:
        raise DistError(
            f"{path}: {len(lookups)} embedding lookups — the text tower has one (the token table)"
        )
    table = values.get(lookups[0]["ins"][0], {}) if isinstance(values, dict) else {}
    table_shape = table.get("shape") if isinstance(table, dict) else None
    if not isinstance(table_shape, list) or len(table_shape) != 2:
        raise DistError(f"{path}: the token table's shape is {table_shape!r}, not [vocab, hidden]")
    hidden = _positive_int(table_shape[1], f"{path}: the token table's width")
    inputs = graph.get("inputs")
    declared = (
        tuple((item.get("name"), item.get("dtype"), item.get("shape")) for item in inputs)
        if isinstance(inputs, list)
        else None
    )
    expected = tuple(
        (
            name,
            dtype,
            [
                1,
                EMBEDDINGGEMMA2_SYMBOL,
                *(hidden if axis == EMBEDDINGGEMMA2_HIDDEN else axis for axis in trailing),
            ],
        )
        for name, dtype, trailing in EMBEDDINGGEMMA2_GRAPH_INPUTS
    )
    if declared != expected:
        raise DistError(
            f"{path}: the graph inputs are {declared}, not {expected} — the reader binds them by"
            " name, dtype and shape"
        )
    if graph.get("symbols") != [EMBEDDINGGEMMA2_SYMBOL]:
        raise DistError(
            f"{path}: the graph's symbols are {graph.get('symbols')!r}, not"
            f" [{EMBEDDINGGEMMA2_SYMBOL!r}]"
        )
    outputs = graph.get("outputs")
    if not isinstance(outputs, list) or len(outputs) != 1 or not isinstance(values, dict):
        raise DistError(f"{path}: the graph outputs are {outputs!r} — an embedding has one")
    output = values.get(outputs[0], {})
    shape = output.get("shape") if isinstance(output, dict) else None
    if (
        not isinstance(output, dict)
        or output.get("dtype") != "f32"
        or not isinstance(shape, list)
        or len(shape) != 2
        or shape[0] != 1
    ):
        raise DistError(f"{path}: the graph output '{outputs[0]}' is {output!r}, not f32 [1, D]")
    width = _positive_int(shape[1], f"{path}: the output width")
    return _positive_int(table_shape[0], f"{path}: the vocabulary size"), width, hidden


def assert_same_graph(
    graph: Mapping[str, Any], path: Path, reference: Mapping[str, Any], reference_path: Path
) -> None:
    """The container at `path` carries the same graph as the one at `reference_path`.

    MUST: every preset runs one graph — the contract, `pipelineConfig` (the bound of T, the width,
    the vocabulary) and the tokenizer checks are read from the default preset's container alone, so
    a container exported from another graph (another bound, another band, another release of the
    modeling code) would ship under a declaration nothing checked. The storage lives in the
    container's binding table, not in the graph (IR v2), so the two graphs are equal value for
    value.
    """
    if graph != reference:
        raise DistError(
            f"{path}: its graph is not the graph of {reference_path} — every preset of the model"
            " runs one graph, and only the storage of the weights may differ"
        )


def assert_embeddinggemma2_tokenizer(
    path: Path, *, vocab_size: int, tokenizer_sha256: str, text_config: Mapping[str, Any]
) -> Mapping[str, Any]:
    """Check the compiled tokenizer asset against the graph and the pinned upstream, and return it.

    MUST: the asset was compiled from the pinned `tokenizer.json` (its recorded SHA-256), has the
    graph's vocabulary, wraps a text in `<bos>` … `<eos>` (what the graph was traced and pooled
    with) with the ids upstream's `text_config` names — a tokenizer of the same vocabulary that
    differs in any of these gives other id sequences with every id in range.
    """
    raw = _read_json(path, "the compiled tokenizer asset")
    if not isinstance(raw, dict) or raw.get("format") != EMBEDDINGGEMMA2_TOKENIZER_FORMAT:
        raise DistError(
            f"{path}: not a compiled tokenizer asset ({EMBEDDINGGEMMA2_TOKENIZER_FORMAT}) — compile"
            " it with `python -m embeddinggemma2.tokenizer`"
        )
    source = raw.get("source")
    recorded = source.get("sha256") if isinstance(source, dict) else None
    if recorded != tokenizer_sha256:
        raise DistError(
            f"{path} was compiled from a tokenizer.json with SHA-256 {recorded}, not the pinned"
            f" {tokenizer_sha256} — compile it again from the pinned snapshot"
        )
    vocab = raw.get("vocab")
    if not isinstance(vocab, list) or len(vocab) != vocab_size:
        rows = len(vocab) if isinstance(vocab, list) else None
        raise DistError(f"{path}: {rows} vocabulary rows, not the graph's {vocab_size}")
    spec = raw.get("spec")
    post = spec.get("postProcessor") if isinstance(spec, dict) else None
    if post != EMBEDDINGGEMMA2_POST_PROCESSOR:
        raise DistError(
            f"{path}: the post-processor is {post!r}, not {EMBEDDINGGEMMA2_POST_PROCESSOR!r} —"
            " the graph pools `<bos>` … `<eos>`"
        )
    for field, upstream in (("bosId", "bos_token_id"), ("eosId", "eos_token_id")):
        if raw.get(field) != text_config.get(upstream):
            raise DistError(
                f"{path}: {field} is {raw.get(field)!r}, upstream's text_config.{upstream} is"
                f" {text_config.get(upstream)!r}"
            )
    return raw


def embeddinggemma2_placeholder_tokens(
    config: Mapping[str, Any], tokenizer: Mapping[str, Any], vocab_size: int, where: str
) -> dict[str, int]:
    """`pipelineConfig.placeholderTokens` from the pinned `config.json`.

    MUST: every id is a special token of the tokenizer asset (an added token, split out before
    BPE — so a text holding the placeholder's spelling encodes to exactly that id) and inside the
    graph's vocabulary, and the ids differ — an id outside that would never be seen, and the
    reader's refusal would test nothing.
    """
    special = tokenizer.get("specialIds")
    special_ids = set(special) if isinstance(special, list) else set()
    placeholders: dict[str, int] = {}
    for kind, field in EMBEDDINGGEMMA2_PLACEHOLDER_FIELDS:
        token = config.get(field)
        if isinstance(token, bool) or not isinstance(token, int) or not 0 <= token < vocab_size:
            raise DistError(f"{where}: {field} is {token!r}, not an id of the {vocab_size} rows")
        if token not in special_ids:
            raise DistError(
                f"{where}: {field} {token} is not a special token of the tokenizer asset"
            )
        placeholders[kind] = token
    if len(set(placeholders.values())) != len(placeholders):
        raise DistError(f"{where}: the placeholder ids repeat ({placeholders})")
    return placeholders


def embeddinggemma2_prompts(config: Mapping[str, Any], where: str) -> dict[str, str]:
    """`pipelineConfig.prompts` — upstream's prompt names and prefixes, verbatim and in order.

    MUST: upstream declares no default prompt (`default_prompt_name` null) — "no prompt, no prefix"
    is then upstream's own behaviour; with a default, a call without a prompt would differ from
    upstream silently. The similarity function must be the cosine (the card's statement). Every
    prefix has a tokenizer parity case ({@link embeddinggemma2.tokenizer.assert_prompts}).
    """
    prompts = config.get("prompts")
    if (
        not isinstance(prompts, dict)
        or not prompts
        or not all(
            isinstance(name, str) and name and isinstance(prefix, str)
            for name, prefix in prompts.items()
        )
    ):
        raise DistError(f"{where}: prompts is not a non-empty object of names to prefixes")
    if config.get("default_prompt_name") is not None:
        raise DistError(
            f"{where}: default_prompt_name is {config.get('default_prompt_name')!r} — the reader"
            " adds no prefix without a prompt, which is upstream's behaviour only without a default"
        )
    if config.get("similarity_fn_name") != EMBEDDINGGEMMA2_SIMILARITY:
        raise DistError(
            f"{where}: similarity_fn_name is {config.get('similarity_fn_name')!r}, not"
            f" {EMBEDDINGGEMMA2_SIMILARITY!r}"
        )
    try:
        assert_prompts(config)
    except TokenizerScriptError as cause:
        raise DistError(f"{where}: {cause}") from cause
    return dict(prompts)


@dataclass(frozen=True)
class EmbeddingGemma2ImageProcessor:
    """What the pinned `processor_config.json` says of images: the soft-token budget, the patch
    side and the pooling (patches per side of one soft token)."""

    budget: int
    patch_size: int
    pooling: int


def embeddinggemma2_image_processor(
    processor: Mapping[str, Any], where: str
) -> EmbeddingGemma2ImageProcessor:
    """Read the image processor's settings, held to what the host implements
    ({@link EMBEDDINGGEMMA2_IMAGE_PROCESSOR})."""
    image = processor.get("image_processor")
    if not isinstance(image, dict):
        raise DistError(f"{where}: no image_processor object")
    for key, expected in EMBEDDINGGEMMA2_IMAGE_PROCESSOR.items():
        if image.get(key) != expected:
            raise DistError(
                f"{where}: image_processor.{key} is {image.get(key)!r}, not {expected!r} — the"
                " host's image preprocessing implements that setting only"
            )
    return EmbeddingGemma2ImageProcessor(
        budget=_positive_int(image.get("max_soft_tokens"), f"{where}: max_soft_tokens"),
        patch_size=_positive_int(image.get("patch_size"), f"{where}: patch_size"),
        pooling=_positive_int(image.get("pooling_kernel_size"), f"{where}: pooling_kernel_size"),
    )


def assert_embeddinggemma2_vision_graph(
    graph: Mapping[str, Any], path: Path, *, hidden: int, processor: EmbeddingGemma2ImageProcessor
) -> None:
    """Check the tower graph's contract before placement.

    The reader binds `pixels` f32 `[1, pooling² · n, patch² · 3]` and `position_x` / `position_y`
    i32 `[1, pooling² · n]` by name and reads one f32 `[1, n, hidden]` output — the text graph's
    soft rows; one symbol, no state. MUST: both position inputs gather a table (the reader requires
    the same), and the tables gathered by the positions cover the longest side the resize rule
    gives at the budget (`pooling · budget` patches — the reader checks the same), or a long thin
    image would read past their end on the user's machine. MUST: a table gathered by both positions
    (the folded RoPE `cos` / `sin`, one table serving both axes) is stored f32 — its values are
    upstream's rotary module, not bf16 checkpoint values, and a bf16 copy would move every rotation
    while every shape still holds (the learned position tables, one per axis, are bf16 in the bf16
    tower).
    """
    symbol = EMBEDDINGGEMMA2_VISION_SYMBOL
    rows = f"{processor.pooling * processor.pooling}{symbol}"
    row_width = processor.patch_size * processor.patch_size * 3
    expected = [
        {"name": name, "dtype": dtype, "shape": [1, rows, row_width] if index == 0 else [1, rows]}
        for index, (name, dtype) in enumerate(EMBEDDINGGEMMA2_VISION_INPUTS)
    ]
    inputs = graph.get("inputs")
    found = (
        [{key: item.get(key) for key in ("name", "dtype", "shape")} for item in inputs]
        if isinstance(inputs, list)
        else None
    )
    if found != expected:
        raise DistError(f"{path}: the vision graph's inputs are {found}, not {expected}")
    if graph.get("symbols") != [symbol]:
        raise DistError(f"{path}: the vision graph's symbols are {graph.get('symbols')!r}")
    if graph.get("states"):
        raise DistError(f"{path}: the vision graph holds state slots")
    outputs = graph.get("outputs")
    values = graph.get("values")
    if not isinstance(outputs, list) or len(outputs) != 1 or not isinstance(values, dict):
        raise DistError(f"{path}: the vision graph does not have one output ({outputs!r})")
    output = values.get(outputs[0], {})
    if output.get("dtype") != "f32" or output.get("shape") != [1, symbol, hidden]:
        raise DistError(
            f"{path}: the vision graph's output is {output!r}, not f32 [1, {symbol!r}, {hidden}]"
            " — the text graph's soft rows"
        )
    position_inputs = {name for name, _ in EMBEDDINGGEMMA2_VISION_INPUTS[1:]}
    gathers = [
        (node["ins"][0], node["ins"][1])
        for node in graph.get("nodes", [])
        if isinstance(node, dict)
        and node.get("op") == "embedding"
        and node["ins"][1] in position_inputs
    ]
    positions = [values.get(table, {}).get("shape", [None])[0] for table, _ in gathers]
    if not positions or not all(isinstance(count, int) for count in positions):
        raise DistError(
            f"{path}: the vision graph gathers no table of static rows by the positions"
        )
    gathered = {index for _, index in gathers}
    if gathered != position_inputs:
        raise DistError(
            f"{path}: the vision graph gathers tables by {sorted(gathered)} only, not by both"
            f" {sorted(position_inputs)}"
        )
    axes: dict[str, set[str]] = {}
    for table, index in gathers:
        axes.setdefault(table, set()).add(index)
    layouts = read_layouts(path)
    rounded = {
        table: layouts.get(table)
        for table, indices in sorted(axes.items())
        if indices == position_inputs and layouts.get(table) != "f32"
    }
    if rounded:
        raise DistError(
            f"{path}: the vision graph's tables gathered by both positions (the folded RoPE) are"
            f" stored {rounded}, not f32"
        )
    longest = processor.pooling * processor.budget
    if min(positions) < longest:
        raise DistError(
            f"{path}: the vision graph's position tables cover {min(positions)} positions, but an"
            f" image side reaches {longest} patches at the budget of {processor.budget}"
        )


def assert_embeddinggemma2_audio_i8_kept(path: Path) -> None:
    """The audio tower's i8 container keeps exactly its declared exceptions bf16: the weights of
    {@link EMBEDDINGGEMMA2_AUDIO_I8_KEPT_BF16} and one window selector
    ({@link EMBEDDINGGEMMA2_AUDIO_SELECTOR_SUFFIX}).

    MUST: its role cannot forbid bf16 (it holds these), so this is what tells the shipped form from
    an audio series rounded otherwise — one with a linear left bf16, or with a kept convolution
    rounded, which the card's statement of the exception would not describe.
    """
    bf16 = {key for key, layout in read_layouts(path).items() if layout == "bf16"}
    selectors = {key for key in bf16 if key.endswith(EMBEDDINGGEMMA2_AUDIO_SELECTOR_SUFFIX)}
    expected = set(EMBEDDINGGEMMA2_AUDIO_I8_KEPT_BF16)
    if len(selectors) != 1 or bf16 - selectors != expected:
        raise DistError(
            f"{path}: the audio tower's i8 container keeps {sorted(bf16)} bf16, not its two"
            f" subsampling convolutions {sorted(expected)} and one window selector"
        )


def assert_embeddinggemma2_vision_series(
    series: Path, name: str, budget: int, pin: upstream_sources.SourcePin
) -> None:
    """A vision series is a full one, taken at the budget this distribution declares, from the
    pinned checkpoint.

    MUST: its record ({@link EMBEDDINGGEMMA2_VISION_REFERENCE_FILE}) names the series `name` (the
    one of its storage), the budget `pipelineConfig.vision` declares and the pin — goldens taken at
    another budget hold the host to rows it never builds, and a series of another revision would
    ship under this pin.
    """
    record_path = series / EMBEDDINGGEMMA2_VISION_REFERENCE_FILE
    record = _read_json(record_path, "the vision series' record")
    if not isinstance(record, dict):
        raise DistError(f"{record_path}: the top level is not an object")
    for key, expected in (
        ("series", name),
        ("softTokenBudget", budget),
        ("upstream", pin.label),
    ):
        if record.get(key) != expected:
            raise DistError(f"{record_path}: {key} is {record.get(key)!r}, not {expected!r}")


def embeddinggemma2_vision_config(
    config: Mapping[str, Any],
    tokenizer: Mapping[str, Any],
    *,
    vocab_size: int,
    hidden: int,
    budget: int,
    where: str,
) -> dict[str, int]:
    """`pipelineConfig.vision` (the Gemma 4 family's shape — read by that family's TS reader).

    MUST: each id is a special token of the tokenizer asset inside the graph's vocabulary and the
    three differ — an id outside that would place an image the model never saw, with every integer
    valid. The soft token is read from the field `placeholderTokens.image` is read from, so the two
    agree by construction (the reader holds them equal: it refuses that id in text, which keeps a
    soft row from being spelt without an image).
    """
    special = tokenizer.get("specialIds")
    special_ids = set(special) if isinstance(special, list) else set()
    declared: dict[str, int] = {"softTokenBudget": budget}
    for field, key in EMBEDDINGGEMMA2_VISION_ID_FIELDS:
        token = config.get(key)
        if isinstance(token, bool) or not isinstance(token, int) or not 0 <= token < vocab_size:
            raise DistError(f"{where}: {key} is {token!r}, not an id of the {vocab_size} rows")
        if token not in special_ids:
            raise DistError(f"{where}: {key} {token} is not a special token of the tokenizer asset")
        declared[field] = token
    ids = [declared[field] for field, _ in EMBEDDINGGEMMA2_VISION_ID_FIELDS]
    if len(set(ids)) != len(ids):
        raise DistError(f"{where}: the image token ids repeat ({ids})")
    declared["hiddenSize"] = hidden
    return declared


#: What a missing setting reads as in a refusal (no setting of the file has this value).
_MISSING = "missing"


def embeddinggemma2_audio_feature_extractor(
    preprocessor: Mapping[str, Any], processor: Mapping[str, Any], where: str
) -> int:
    """Hold the feature extractor's settings to what the host implements
    ({@link EMBEDDINGGEMMA2_AUDIO_FEATURE_EXTRACTOR}) in both places upstream states them —
    `preprocessor_config.json` (what the feature extractor loads) and the `feature_extractor` block
    of `processor_config.json` (what the processor loads) — and return the sample rate."""
    block = processor.get("feature_extractor")
    if not isinstance(block, dict):
        raise DistError(f"{where}: processor_config.json has no feature_extractor object")
    for name, settings in (
        ("preprocessor_config.json", preprocessor),
        ("processor_config.json feature_extractor", block),
    ):
        for key, expected in EMBEDDINGGEMMA2_AUDIO_FEATURE_EXTRACTOR.items():
            found = settings.get(key, _MISSING)
            # `0` and `0.0` are one setting (upstream writes both); `True` is not `1`.
            if found != expected or isinstance(found, bool) != isinstance(expected, bool):
                raise DistError(
                    f"{where}: {name}: {key} is {found!r}, not {expected!r} — the host's audio"
                    " front end implements that setting only"
                )
    return EMBEDDINGGEMMA2_AUDIO_FEATURE_EXTRACTOR["sampling_rate"]


def embeddinggemma2_audio_seq_length(processor: Mapping[str, Any], where: str) -> int:
    """Read `processor_config.json`'s `audio_seq_length` and hold it to the value this recipe
    recorded ({@link EMBEDDINGGEMMA2_AUDIO_SEQ_LENGTH}); return it.

    Nothing Karume builds reads it: upstream's processor path (`replace_audio_token`, the one
    sentence-transformers runs) counts an audio's soft tokens from the frame mask without a cap, and
    the host follows that path, up to the graph's bound. Only upstream's serving helper
    (`_compute_audio_num_tokens`) caps its count there. MUST: a snapshot that states another value
    fails, so the change is seen and the documented difference (`docs/limitations.md`) is checked
    again rather than going stale.
    """
    found = processor.get("audio_seq_length", _MISSING)
    if isinstance(found, bool) or found != EMBEDDINGGEMMA2_AUDIO_SEQ_LENGTH:
        raise DistError(
            f"{where}: audio_seq_length is {found!r}, not the recorded"
            f" {EMBEDDINGGEMMA2_AUDIO_SEQ_LENGTH} — upstream's serving helper caps an audio's soft"
            " tokens there while its processor path (which Karume follows) does not; check the"
            " documented difference again before recording the new value"
        )
    return found


def embeddinggemma2_audio_soft_tokens(samples: int) -> int:
    """The soft tokens of an audio of `samples` samples (`waveforms.valid_frames` and
    `soft_tokens`, mirrored torch-free: `ceil(max(0, ceil(samples / 160) − 1) / 4)`)."""
    hop = EMBEDDINGGEMMA2_AUDIO_FEATURE_EXTRACTOR["hop_length"]
    frames = max(0, -(-samples // hop) - 1)
    return -(-frames // EMBEDDINGGEMMA2_AUDIO_FRAMES_PER_TOKEN)


def assert_embeddinggemma2_audio_graph(
    graph: Mapping[str, Any], path: Path, *, hidden: int, mel_bins: int
) -> int:
    """Check the audio tower graph's contract before placement and return the bound of its
    symbol (its most soft tokens).

    The reader binds `features` f32 `[1, 4n, mel_bins]` and `frame_mask` f32 `[1, 4n]` by name and
    reads one f32 `[1, n, hidden]` output — the text graph's soft rows; one symbol, no state.
    MUST: the bound of `n` (its folded window bias) is the soft tokens of the feature extractor's
    longest audio ({@link EMBEDDINGGEMMA2_AUDIO_MAX_SAMPLES}) — a lower one would refuse audios the
    front end reads, and the reader holds `maxSoftTokens` to the same bound.
    """
    symbol = EMBEDDINGGEMMA2_AUDIO_SYMBOL
    frames = f"{EMBEDDINGGEMMA2_AUDIO_FRAMES_PER_TOKEN}{symbol}"
    expected = [
        {
            "name": name,
            "dtype": dtype,
            "shape": [1, frames, mel_bins] if index == 0 else [1, frames],
        }
        for index, (name, dtype) in enumerate(EMBEDDINGGEMMA2_AUDIO_INPUTS)
    ]
    inputs = graph.get("inputs")
    found = (
        [{key: item.get(key) for key in ("name", "dtype", "shape")} for item in inputs]
        if isinstance(inputs, list)
        else None
    )
    if found != expected:
        raise DistError(f"{path}: the audio graph's inputs are {found}, not {expected}")
    if graph.get("symbols") != [symbol]:
        raise DistError(f"{path}: the audio graph's symbols are {graph.get('symbols')!r}")
    if graph.get("states"):
        raise DistError(f"{path}: the audio graph holds state slots")
    outputs = graph.get("outputs")
    values = graph.get("values")
    if not isinstance(outputs, list) or len(outputs) != 1 or not isinstance(values, dict):
        raise DistError(f"{path}: the audio graph does not have one output ({outputs!r})")
    output = values.get(outputs[0], {})
    if output.get("dtype") != "f32" or output.get("shape") != [1, symbol, hidden]:
        raise DistError(
            f"{path}: the audio graph's output is {output!r}, not f32 [1, {symbol!r}, {hidden}]"
            " — the text graph's soft rows"
        )
    bound = embeddinggemma2_graph_max_tokens(graph, path, symbol)
    longest = embeddinggemma2_audio_soft_tokens(EMBEDDINGGEMMA2_AUDIO_MAX_SAMPLES)
    if bound != longest:
        raise DistError(
            f"{path}: the audio graph runs {symbol} up to {bound}, but the feature extractor's"
            f" longest audio ({EMBEDDINGGEMMA2_AUDIO_MAX_SAMPLES} samples) has {longest} soft"
            " tokens"
        )
    return bound


def assert_embeddinggemma2_audio_series(
    series: Path,
    name: str,
    *,
    sample_rate: int,
    max_soft_tokens: int,
    pin: upstream_sources.SourcePin,
) -> None:
    """An audio series is a full one, taken at the sample rate and the bound this distribution
    declares, from the pinned checkpoint.

    MUST: its record ({@link EMBEDDINGGEMMA2_AUDIO_REFERENCE_FILE}) names the series `name` (the
    one of its storage), the pin, the sample rate and the bound — goldens taken at another rate or
    bound hold the host to frames it never builds, and a series of another revision would ship
    under this pin.
    """
    record_path = series / EMBEDDINGGEMMA2_AUDIO_REFERENCE_FILE
    record = _read_json(record_path, "the audio series' record")
    if not isinstance(record, dict):
        raise DistError(f"{record_path}: the top level is not an object")
    for key, expected in (
        ("series", name),
        ("upstream", pin.label),
        ("sampleRate", sample_rate),
        ("softTokensMax", max_soft_tokens),
    ):
        if record.get(key) != expected:
            raise DistError(f"{record_path}: {key} is {record.get(key)!r}, not {expected!r}")


def embeddinggemma2_audio_config(
    config: Mapping[str, Any],
    tokenizer: Mapping[str, Any],
    *,
    vocab_size: int,
    hidden: int,
    max_soft_tokens: int,
    sample_rate: int,
    where: str,
) -> dict[str, int]:
    """`pipelineConfig.audio` (read by the EmbeddingGemma family's TS reader).

    MUST: each id is a special token of the tokenizer asset inside the graph's vocabulary and the
    three differ, for the image ids' reason ({@link embeddinggemma2_vision_config}); the soft token
    is read from the field `placeholderTokens.audio` is read from.
    """
    special = tokenizer.get("specialIds")
    special_ids = set(special) if isinstance(special, list) else set()
    declared: dict[str, int] = {}
    for field, key in EMBEDDINGGEMMA2_AUDIO_ID_FIELDS:
        token = config.get(key)
        if isinstance(token, bool) or not isinstance(token, int) or not 0 <= token < vocab_size:
            raise DistError(f"{where}: {key} is {token!r}, not an id of the {vocab_size} rows")
        if token not in special_ids:
            raise DistError(f"{where}: {key} {token} is not a special token of the tokenizer asset")
        declared[field] = token
    ids = [declared[field] for field, _ in EMBEDDINGGEMMA2_AUDIO_ID_FIELDS]
    if len(set(ids)) != len(ids):
        raise DistError(f"{where}: the audio token ids repeat ({ids})")
    declared["hiddenSize"] = hidden
    declared["maxSoftTokens"] = max_soft_tokens
    declared["sampleRate"] = sample_rate
    return declared


def embeddinggemma2_pipeline_config(
    *,
    prompts: Mapping[str, str],
    width: int,
    graph_max_tokens: int,
    placeholders: Mapping[str, int],
    vision: Mapping[str, int],
    audio: Mapping[str, int],
    where: str,
) -> dict[str, Any]:
    """Build `pipelineConfig` (the six fields of the TypeScript schema, in this order).

    MUST: the full dimension equals the graph's output width, the dimensions descend, and the
    token bound equals both the graph's and the writer's ({@link EMBEDDINGGEMMA2_MAX_TOKENS}) — a
    declaration the asset does not back fails here, not on the user's machine.
    """
    dimensions = list(EMBEDDINGGEMMA2_DIMENSIONS)
    if dimensions[0] != width:
        raise DistError(
            f"{where}: the full dimension {dimensions[0]} is not the graph's output width {width}"
        )
    if any(later >= earlier for earlier, later in pairwise(dimensions)):
        raise DistError(f"the dimensions {dimensions} do not descend")
    if graph_max_tokens != EMBEDDINGGEMMA2_MAX_TOKENS:
        raise DistError(
            f"{where}: the graph binds at most {graph_max_tokens} tokens, the declaration says"
            f" {EMBEDDINGGEMMA2_MAX_TOKENS} — the series was exported with another bound"
        )
    return {
        "prompts": dict(prompts),
        "dimensions": dimensions,
        "maxTokens": EMBEDDINGGEMMA2_MAX_TOKENS,
        "placeholderTokens": dict(placeholders),
        "vision": dict(vision),
        "audio": dict(audio),
    }


def embeddinggemma2_plan(
    sources: EmbeddingGemma2Sources,
    model: str = EMBEDDINGGEMMA2_MODEL,
    pin: upstream_sources.SourcePin | None = None,
) -> ModelPlan:
    """Build the plan of the one model (every check and read is done here; `pin` is the upstream
    pin the snapshot and the container are held to — `embeddinggemma2.sources.PIN` when omitted,
    read at call time)."""
    pin = upstream_sources.PIN if pin is None else pin
    assert_model_name(model)
    if model != EMBEDDINGGEMMA2_MODEL:
        raise DistError(
            f"unknown EmbeddingGemma 2 model '{model}' (known: {EMBEDDINGGEMMA2_MODEL})"
        )
    placements = embeddinggemma2_placements(sources)
    for role, source in placements.items():
        assert_storage(role, source, EMBEDDINGGEMMA2_STORAGE_REQUIREMENTS)
        assert_storage_absent(role, source, EMBEDDINGGEMMA2_STORAGE_FORBIDDEN)
    try:
        upstream_sources.assert_pinned(sources.snapshot, EMBEDDINGGEMMA2_READ_FILES, pin)
    except upstream_sources.SourcePinError as cause:
        raise DistError(str(cause)) from cause
    # The contract is read from the original weights' container; every other storage must carry
    # the same graph.
    container = placements[embeddinggemma2_role(EMBEDDINGGEMMA2_DTYPE)]
    graph = ir_graph(container)
    vocab_size, width, soft_width = assert_embeddinggemma2_graph(graph, container)
    for dtype in EMBEDDINGGEMMA2_SERIES:
        if dtype != EMBEDDINGGEMMA2_DTYPE:
            other = placements[embeddinggemma2_role(dtype)]
            assert_same_graph(ir_graph(other), other, graph, container)
    config_path = sources.snapshot / "config.json"
    config = _read_json(config_path, "the upstream config")
    text_config = config.get("text_config") if isinstance(config, dict) else None
    if not isinstance(text_config, dict):
        raise DistError(f"{config_path}: no text_config object")
    tokenizer = assert_embeddinggemma2_tokenizer(
        placements[EMBEDDINGGEMMA2_TOKENIZER_ROLE],
        vocab_size=vocab_size,
        tokenizer_sha256=pin.sha256["tokenizer.json"],
        text_config=text_config,
    )
    prompts_path = sources.snapshot / "config_sentence_transformers.json"
    prompts_config = _read_json(prompts_path, "the upstream prompts")
    if not isinstance(prompts_config, dict):
        raise DistError(f"{prompts_path}: the top level is not an object")
    processor_path = sources.snapshot / "processor_config.json"
    processor_config = _read_json(processor_path, "the upstream processor")
    if not isinstance(processor_config, dict):
        raise DistError(f"{processor_path}: the top level is not an object")
    processor = embeddinggemma2_image_processor(processor_config, str(processor_path))
    preprocessor_path = sources.snapshot / "preprocessor_config.json"
    preprocessor_config = _read_json(preprocessor_path, "the upstream feature extractor")
    if not isinstance(preprocessor_config, dict):
        raise DistError(f"{preprocessor_path}: the top level is not an object")
    sample_rate = embeddinggemma2_audio_feature_extractor(
        preprocessor_config, processor_config, str(sources.snapshot)
    )
    embeddinggemma2_audio_seq_length(processor_config, str(processor_path))
    # The soft rows' width is the token table's (a soft row replaces a scaled lookup) and the
    # upstream text config's hidden size; the tower's output must be that width too.
    hidden = _positive_int(
        text_config.get("hidden_size"), f"{config_path}: text_config.hidden_size"
    )
    if soft_width != hidden:
        raise DistError(
            f"{container}: soft_embeds is {soft_width} wide, upstream's text_config.hidden_size"
            f" {hidden}"
        )
    # Each tower's contract is read from its original weights' container; its other storage must
    # carry the same graph (as the text graph's).
    vision_container = placements[
        embeddinggemma2_role(EMBEDDINGGEMMA2_DTYPE, EMBEDDINGGEMMA2_VISION_ROLE)
    ]
    vision_graph = ir_graph(vision_container)
    assert_embeddinggemma2_vision_graph(
        vision_graph, vision_container, hidden=hidden, processor=processor
    )
    audio_container = placements[
        embeddinggemma2_role(EMBEDDINGGEMMA2_DTYPE, EMBEDDINGGEMMA2_AUDIO_ROLE)
    ]
    audio_graph = ir_graph(audio_container)
    max_soft_tokens = assert_embeddinggemma2_audio_graph(
        audio_graph,
        audio_container,
        hidden=hidden,
        mel_bins=EMBEDDINGGEMMA2_AUDIO_FEATURE_EXTRACTOR["feature_size"],
    )
    for dtype, name in EMBEDDINGGEMMA2_VISION_SERIES.items():
        if dtype != EMBEDDINGGEMMA2_DTYPE:
            other = placements[embeddinggemma2_role(dtype, EMBEDDINGGEMMA2_VISION_ROLE)]
            assert_same_graph(ir_graph(other), other, vision_graph, vision_container)
        assert_embeddinggemma2_vision_series(sources.vision[dtype], name, processor.budget, pin)
    for dtype, name in EMBEDDINGGEMMA2_AUDIO_SERIES.items():
        if dtype != EMBEDDINGGEMMA2_DTYPE:
            other = placements[embeddinggemma2_role(dtype, EMBEDDINGGEMMA2_AUDIO_ROLE)]
            assert_same_graph(ir_graph(other), other, audio_graph, audio_container)
        assert_embeddinggemma2_audio_series(
            sources.audio[dtype],
            name,
            sample_rate=sample_rate,
            max_soft_tokens=max_soft_tokens,
            pin=pin,
        )
    assert_embeddinggemma2_audio_i8_kept(
        placements[embeddinggemma2_role(EMBEDDINGGEMMA2_I8_DTYPE, EMBEDDINGGEMMA2_AUDIO_ROLE)]
    )
    # Every container names the license and the commit it was exported from (the export bakes
    # them).
    for role in placements:
        if role == EMBEDDINGGEMMA2_TOKENIZER_ROLE:
            continue
        assert_upstream_provenance(
            placements[role], license=EMBEDDINGGEMMA2_LICENSE, revision=pin.revision
        )
    placeholders = embeddinggemma2_placeholder_tokens(
        config, tokenizer, vocab_size, str(config_path)
    )
    pipeline_config = embeddinggemma2_pipeline_config(
        prompts=embeddinggemma2_prompts(prompts_config, str(prompts_path)),
        width=width,
        graph_max_tokens=embeddinggemma2_graph_max_tokens(graph, container),
        placeholders=placeholders,
        vision=embeddinggemma2_vision_config(
            config,
            tokenizer,
            vocab_size=vocab_size,
            hidden=hidden,
            budget=processor.budget,
            where=str(config_path),
        ),
        audio=embeddinggemma2_audio_config(
            config,
            tokenizer,
            vocab_size=vocab_size,
            hidden=hidden,
            max_soft_tokens=max_soft_tokens,
            sample_rate=sample_rate,
            where=str(config_path),
        ),
        where=str(container),
    )
    return ModelPlan(
        name=model,
        pipeline=EMBEDDINGGEMMA_PIPELINE,
        artifacts={
            role: Artifact(EMBEDDINGGEMMA2_OUTPUT_PATHS[role], source=source)
            for role, source in placements.items()
        },
        weights=EMBEDDINGGEMMA2_WEIGHTS,
        assets=EMBEDDINGGEMMA2_ASSETS,
        # requiredLimits is not written — the core derives and bakes it during assembly.
        quants=complete_quant_weights(EMBEDDINGGEMMA2_WEIGHTS, EMBEDDINGGEMMA2_QUANTS),
        default_quant=EMBEDDINGGEMMA2_DEFAULT_QUANT,
        pipeline_config=pipeline_config,
    )


def embeddinggemma2_dist_plan(series_dir: Path, model: str) -> ModelPlan:
    """Build the plan from the parent of `--series` (the CLI's dispatch target)."""
    return embeddinggemma2_plan(embeddinggemma2_sources(series_dir), model)


#: The modification notice (Apache 2.0 §4(b)) — **the changes this repository made to the upstream
#: weights**.
#:
#: MUST: the text matches the distribution's content — it is valid prose as a value, so it passes
#: `verify_dist` and the manifest checks, and a mismatch would only be noticed after distribution.
#: On the day a tower or a storage is added or dropped ({@link EMBEDDINGGEMMA2_GRAPH_SERIES}), this
#: text changes with it.
#: SPEC: docs/spec/distribution-cards.md#noticemd
EMBEDDINGGEMMA2_NOTICE_MARKDOWN = (
    """# NOTICE

This repository redistributes a modified form of `google/embeddinggemma-2`, which is licensed under
the Apache License, Version 2.0 (see `LICENSE.md`). The following changes were made:

- The **text tower, the vision tower and the audio tower** (each tower with its projection into
  the text tower's width) were extracted.
- The text tower, together with the mean pooling and the L2 normalisation of the
  SentenceTransformer pipeline, was re-expressed as one graph in the Karume container format (a
  `.krm` part sequence whose first part carries the graph and model descriptors). The
  sliding-window attention mask was replaced by a band declared on the attention of each sliding
  layer, which the runtime applies from the token positions; the rotary position tables are
  constants of the graph, which bound the input length it accepts (stated in `README.md`).
- The text tower's weights ship in two forms.
  **The `bf16` form keeps the checkpoint's BF16 values unchanged** (the parameters stored in
  float32, the norms among others, hold the same values widened). **The `i8` form is an
  int8-quantized copy**: the token embedding table and every linear layer of the text tower were
  rounded to symmetric int8 with one float32 scale per row; the other parameters are those of the
  `bf16` form.
- The vision tower and its projection were re-expressed as a second graph: it reads an image's
  real patches only, ordered so that the patches pooled into one soft token are adjacent, and its
  two-dimensional rotary embedding is folded into tables of the graph gathered by the patch
  positions. Its weights ship in the same two forms: the `bf16` form keeps the checkpoint's BF16
  values unchanged, and in the `i8` form every linear layer and the two learned position tables
  were rounded to symmetric int8 with one float32 scale per row (its norms and the folded tables
  are stored in float32 in both).
- The audio tower and its projection were re-expressed as a third graph: it reads an audio's
  log-mel frames (computed from the samples outside the graph) with their frame mask, its
  chunked local attention is computed over each query's window of 12 keys with the relative
  position term folded into constants, and the bounds of its clipped linear layers are constants
  of the graph. Its weights ship in the same two forms: the `bf16` form keeps the checkpoint's
  BF16 values unchanged, and in the `i8` form every linear layer and depthwise convolution was
  rounded to symmetric int8 with one float32 scale per row, while its two subsampling
  convolutions keep their BF16 values (its norms, biases and folded constants are stored in
  float32 in both).
- The tokenizer ships as a compiled form of the checkpoint's `tokenizer.json`, and the prompt
  prefixes of its `config_sentence_transformers.json` are carried in `karume.json`.

No retraining and no fine-tuning were performed. The original checkpoint is not distributed here.
"""
    + (
        # The use policy: upstream's sentence quoted from the one value the card quotes too.
        '\nThe upstream model card states, under "Ethics & Safety":\n\n'
        f"> {EMBEDDINGGEMMA2_USE_POLICY_SENTENCE}\n\n"
        "The policy is upstream's document: read it at that link. It is not reproduced here.\n"
    )
)


def embeddinggemma2_root_files() -> dict[str, str]:
    """The legal texts at the repository root (`karume.dist.Pipeline.root_files`).

    Upstream is Apache License 2.0: §4(a) asks for a copy of the license (the shared original,
    {@link _shared.licenses.apache_license_2_0}, read verbatim), §4(b) for a notice of the changes.
    Upstream has no `NOTICE` file (§4(d) does not apply), so `NOTICE.md` holds the §4(b) notice
    only — the Gemma 4 distribution's form.
    """
    return {
        "LICENSE.md": license_markdown(apache_license_2_0()),
        "NOTICE.md": EMBEDDINGGEMMA2_NOTICE_MARKDOWN,
    }


#: The one `--pipeline embeddinggemma-2` entry (the driver merges it into the core's PIPELINES).
PIPELINE = Pipeline(
    default_model=EMBEDDINGGEMMA2_MODEL,
    repo_name=embeddinggemma2_repo_name,
    plan=embeddinggemma2_dist_plan,
    # Attribution is determined by the model name alone (`EMBEDDINGGEMMA2_UPSTREAM`).
    card_profiles={"embeddinggemma2": render_embeddinggemma2_model_card},
    root_files=embeddinggemma2_root_files(),
)
