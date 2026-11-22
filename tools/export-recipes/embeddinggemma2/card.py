"""The model card (`README.md`) of the EmbeddingGemma 2 distribution — a pure function derived by
machine from the manifest.

The general-purpose rendering parts (frontmatter, model list, quant table, assembling the
sections) live in `karume.modelcard`. What lives here is **the facts specific to EmbeddingGemma 2**
only: attribution (source, license) and what this pipeline's card says.

Attribution is **determined by the model name alone** ({@link EMBEDDINGGEMMA2_UPSTREAM}), so it is
not split into a profile axis (the gemma4 / siglip2 rule). The files with the redistribution terms
(`LICENSE.md` / `NOTICE.md`) are placed at the repository root by
`embeddinggemma2.distribution.embeddinggemma2_root_files`, not by the card. What the model can and
cannot do is left to the upstream model card, which the card links.

MUST: **every number, download size, quant table and dtype label is derived from the manifest**
(the same MUST of `karume.modelcard` applies unchanged). The only constants this module may hold
are facts that **do not exist in the manifest** (the upstream's context window and its statement on
precision, what each storage form holds, and the measured distance of the int8 form from the
original weights with its source).
"""

from __future__ import annotations

from collections.abc import Mapping
from functools import partial
from typing import Any

from _shared.upstream import hf_upstreams
from karume.modelcard import (
    CardMetadata,
    FileRefKey,
    default_model,
    from_pretrained,
    frontmatter,
    introduction,
    model_sections,
    models,
    quants,
    render,
    require_pipeline,
)

#: The pipeline contract this template can describe (ADR 0041 §2 — per model). The TypeScript family
#: `embeddinggemma` reads it.
EMBEDDINGGEMMA2_SUPPORTED_PIPELINE = "embeddinggemma/1"

#: The HF pipeline tag: the upstream card's own (`feature-extraction`). The distribution carries the
#: text tower only, and a text embedding is feature extraction too.
EMBEDDINGGEMMA2_PIPELINE_TAG = "feature-extraction"

#: The title names only what this distribution carries (the text tower — no image, video or audio).
EMBEDDINGGEMMA2_TITLE = "EmbeddingGemma 2 Text Embedding — Karume"

#: Model name → the HF repository ID of the upstream checkpoint. **This one table is the only record
#: of "which model is which weights"**; a model name it does not list cannot be attributed, so the
#: card fails instead of rendering. The one model is the text tower (`text`); later keys of the
#: same repository would add the other towers of the same checkpoint.
EMBEDDINGGEMMA2_UPSTREAM: Mapping[str, str] = {"text": "google/embeddinggemma-2"}

#: License (the README front matter of the pinned snapshot says `license: apache-2.0` —
#: `embeddinggemma2.sources.PIN`; not gated).
EMBEDDINGGEMMA2_LICENSE = "apache-2.0"
EMBEDDINGGEMMA2_LICENSE_TEXT_LINK = "https://www.apache.org/licenses/LICENSE-2.0"

#: The storage of the upstream checkpoint's weights (BF16). The card states that the default preset
#: ships exactly this one ({@link _storage_lines}).
EMBEDDINGGEMMA2_CHECKPOINT_DTYPE = "bf16"

#: The int8 storage's dtype label (the runtime's storage vocabulary).
EMBEDDINGGEMMA2_I8_DTYPE = "i8"

#: Dtype label of the `model` role → (the word the precision line uses, what that storage holds).
#: MUST: the card describes these storages only — a manifest that stores the text tower otherwise
#: is not drawn ({@link _storage_lines}).
EMBEDDINGGEMMA2_STORAGES: Mapping[str, tuple[str, str]] = {
    EMBEDDINGGEMMA2_CHECKPOINT_DTYPE: ("bf16", "the checkpoint's weights unchanged (bf16)."),
    EMBEDDINGGEMMA2_I8_DTYPE: (
        "int8",
        "the token table and every linear layer rounded to int8, one f32 scale per row; the"
        " norms and the per-layer scalars are the checkpoint's values, as in `bf16`.",
    ),
}

#: Where the int8 form's measured distance from the original weights comes from (a record of the
#: Karume repository).
EMBEDDINGGEMMA2_I8_SCREENING_SOURCE = "docs/research/2026-11-22-embeddinggemma2-quant-screen.md"

#: The measured distance of the int8 form from the original weights (the screening above,
#: configuration `i8-all` — the rounding the `i8` files store; numbers copied from its tables).
#: MUST: stated as one measurement on that small set, never as a benchmark, and only for a manifest
#: with an `i8` preset ({@link _i8_screening}).
EMBEDDINGGEMMA2_I8_SCREENING: tuple[str, ...] = (
    "One measurement, not a benchmark: a screening on torch (CPU, f32 arithmetic) with the weights",
    "rounded exactly as the `i8` files store them, against the checkpoint's own weights, over 58",
    "texts — the ten test texts of this conversion (12 to 8,192 tokens) and 48 short ones, 10 of",
    "them in Japanese (2026-11-22). At the full 768 values:",
    "",
    "- the cosine of each text's embedding to the original weights' is 0.999396 at the lowest and",
    "  0.999842 on average;",
    "- ranking 20 queries against 20 documents, every query still ranks its own document first,",
    "  and 0.74% of the 3,800 (query, document, document) orders are flipped;",
    "- the shorter widths are no further off (lowest cosine 0.999411 / 0.999474 / 0.999620 at",
    "  512 / 256 / 128 values).",
    "",
    f"Source: `{EMBEDDINGGEMMA2_I8_SCREENING_SOURCE}` in the Karume repository.",
)

#: The weights role of the text tower (the spelling `embeddinggemma2.distribution` imports — the
#: card reads the storage of the main model from it).
EMBEDDINGGEMMA2_MODEL_ROLE = "model"

#: The upstream model's context window in tokens (upstream model card at the pinned revision:
#: "Context Window | 8,192 tokens"). A fact of the upstream, not of this distribution — the
#: distribution's own limit is `pipelineConfig.maxTokens`.
EMBEDDINGGEMMA2_UPSTREAM_CONTEXT_TOKENS = 8192

#: The upstream model card's statement on precision (pinned revision, "Run inference in `bfloat16`
#: or `float32`. Do not use `float16`." — its activations exceed float16's range). The card pairs
#: it with what this repository's presets compute in ({@link _precision_lines}).
EMBEDDINGGEMMA2_PRECISION_STATEMENT = (
    "upstream says to run the model in bfloat16 or float32, never float16: its activations exceed"
    " float16's range, and float16 returns NaN or silently degraded embeddings"
)


def _metadata(manifest: Mapping[str, Any]) -> CardMetadata:
    """The frontmatter, from the models the manifest lists (`base_model` = every redistributed
    upstream).

    MUST: do not render for an unknown model name — dropping it silently would emit a card that
    does not name its source.
    """
    upstream: list[str] = []
    for name in manifest["models"]:
        if name not in EMBEDDINGGEMMA2_UPSTREAM:
            raise ValueError(
                f"model '{name}' has no upstream in the attribution table"
                f" (known: {sorted(EMBEDDINGGEMMA2_UPSTREAM)}) — a card that cannot name its"
                " source is not drawn"
            )
        repo = EMBEDDINGGEMMA2_UPSTREAM[name]
        if repo not in upstream:
            upstream.append(repo)
    return CardMetadata(
        pipeline_tag=EMBEDDINGGEMMA2_PIPELINE_TAG,
        base_model=tuple(upstream),
        # No `base_model_relation`: the default preset's download keeps the checkpoint's storage
        # (bf16), so the distribution is none of adapter / merge / quantized / finetune — the
        # "default quant's downloaded file" rule of the other cards (birefnet, irodori, sbv2,
        # anima); the `i8` preset is an option beside it ({@link _storage_lines} refuses a
        # manifest where that is not so). The gemma4 card follows another rule: `quantized` as
        # soon as any stored weight differs from the checkpoint's, default or not
        # (`_gemma4_relation`).
        license=EMBEDDINGGEMMA2_LICENSE,
        tags=(EMBEDDINGGEMMA2_PIPELINE_TAG, "sentence-similarity", "webgpu", "embedding", "gemma"),
    )


def _spell(values: Any) -> str:
    return " / ".join(str(value) for value in values)


def _input_length(max_tokens: int) -> list[str]:
    """The input-length line: the distribution's limit, against the upstream model's context.

    MUST: a limit above the upstream context is not drawn (the card would promise lengths the
    model was not trained for); "for now" is said only while the limit is below it.
    """
    if max_tokens > EMBEDDINGGEMMA2_UPSTREAM_CONTEXT_TOKENS:
        raise ValueError(
            f"maxTokens {max_tokens} is above the upstream model's context of"
            f" {EMBEDDINGGEMMA2_UPSTREAM_CONTEXT_TOKENS} tokens"
        )
    if max_tokens == EMBEDDINGGEMMA2_UPSTREAM_CONTEXT_TOKENS:
        return [
            f"- **Input length**: at most {max_tokens:,} tokens including `<bos>` and `<eos>` (the",
            "  upstream model's whole context). A longer text is refused, never truncated.",
        ]
    return [
        f"- **Input length**: at most {max_tokens:,} tokens including `<bos>` and `<eos>` for now",
        f"  (the upstream model's context is {EMBEDDINGGEMMA2_UPSTREAM_CONTEXT_TOKENS:,} tokens). A"
        " longer text is refused, never",
        "  truncated.",
    ]


def _overview(manifest: Mapping[str, Any]) -> list[str]:
    config = default_model(manifest)["pipelineConfig"]
    dimensions = config["dimensions"]
    return [
        "## What is this",
        "",
        "A text embedding distribution: the **text tower** of"
        f" `{EMBEDDINGGEMMA2_UPSTREAM[manifest['defaultModel']]}`, with its mean pooling and",
        "L2 normalisation, converted into the WebGPU inference runtime **Karume**'s container",
        "format (a `.krm` part sequence whose first part carries the graph and model descriptors).",
        "Runs as-is in the browser and in Deno — a string in, a unit vector out.",
        "",
        "- **The tokenizer and the prompts ship with the weights.** `embed()` prepends the prefix",
        "  of the prompt you name, tokenizes on the host and runs one graph.",
        f"- **Output**: a unit vector of {dimensions[0]} values. The model was trained with",
        "  Matryoshka Representation Learning, so a shorter vector of"
        f" {_spell(dimensions[1:])} values can be asked for:",
        "  the leading values, re-normalised to unit length.",
        *_input_length(config["maxTokens"]),
        "- **Text only.** The upstream checkpoint also carries vision and audio towers; this",
        "  distribution contains neither, and cannot embed images, video or audio. A text holding",
        "  the image, audio or video placeholder token is refused.",
        "- Not readable by transformers or sentence-transformers (it's a different container with",
        "  an embedded graph); the reader is a pipeline that implements"
        f" `{EMBEDDINGGEMMA2_SUPPORTED_PIPELINE}`.",
        f"- Exporter used for the conversion: `{manifest['generator']}`. The distribution manifest",
        f"  is `karume.json` (`{manifest['format']}`).",
    ]


def _stored_dtypes(manifest: Mapping[str, Any]) -> list[str]:
    """The storages of the `model` role, in the order {@link EMBEDDINGGEMMA2_STORAGES} lists them.

    MUST: a manifest that stores the text tower in a storage the card does not describe, or
    without the checkpoint's own, is not drawn — the card would state a storage the distribution
    does not carry.
    """
    dtypes = {
        dtype
        for model in manifest["models"].values()
        for dtype in model["weights"][EMBEDDINGGEMMA2_MODEL_ROLE]
    }
    if EMBEDDINGGEMMA2_CHECKPOINT_DTYPE not in dtypes or not dtypes <= set(
        EMBEDDINGGEMMA2_STORAGES
    ):
        raise ValueError(
            f"the text tower's weights are stored as {sorted(dtypes)} — this card describes the"
            f" checkpoint's {EMBEDDINGGEMMA2_CHECKPOINT_DTYPE} values, optionally beside"
            " "
            + " / ".join(
                dtype
                for dtype in EMBEDDINGGEMMA2_STORAGES
                if dtype != EMBEDDINGGEMMA2_CHECKPOINT_DTYPE
            )
        )
    return [dtype for dtype in EMBEDDINGGEMMA2_STORAGES if dtype in dtypes]


def _storage_lines(manifest: Mapping[str, Any]) -> list[str]:
    """The attribution's opening: the original checkpoint is not shipped, and what each storage of
    the weights holds.

    MUST: a manifest whose default preset does not ship the checkpoint's own values is not drawn —
    the card carries no `base_model_relation` on that ground ({@link _metadata}).
    """
    dtypes = _stored_dtypes(manifest)
    for name, model in manifest["models"].items():
        default = model["quants"][model["defaultQuant"]]["weights"][EMBEDDINGGEMMA2_MODEL_ROLE]
        if default != EMBEDDINGGEMMA2_CHECKPOINT_DTYPE:
            raise ValueError(
                f"model '{name}': the default quant '{model['defaultQuant']}' stores the text"
                f" tower as {default} — this card states that the default download is the"
                f" checkpoint's {EMBEDDINGGEMMA2_CHECKPOINT_DTYPE} values"
            )
    return [
        "Converted from the upstream checkpoint — the original checkpoint files are not",
        "distributed here. The weights of the text tower ship in the storage each quant names:",
        "",
        *(f"- `{dtype}`: {EMBEDDINGGEMMA2_STORAGES[dtype][1]}" for dtype in dtypes),
    ]


def _precision_lines(manifest: Mapping[str, Any]) -> list[str]:
    """Upstream's statement on precision, and what this repository's presets compute in.

    MUST: a preset with a `session` of its own is not drawn — every preset here is the reference
    one (f32 compute, `session` empty — ADR 0104), and a session option could change that, making
    the sentence false.
    """
    for name, model in manifest["models"].items():
        for quant_name, quant in model["quants"].items():
            if quant["session"]:
                raise ValueError(
                    f"model '{name}', quant '{quant_name}' declares a session"
                    f" {quant['session']} — this card states that every preset computes in f32"
                )
    stored = " or ".join(EMBEDDINGGEMMA2_STORAGES[dtype][0] for dtype in _stored_dtypes(manifest))
    return [
        f"- **Precision**: {EMBEDDINGGEMMA2_PRECISION_STATEMENT}. Every preset of this repository",
        f"  computes in f32 over the stored {stored} weights.",
    ]


def _base_weights(manifest: Mapping[str, Any]) -> list[str]:
    """The attribution section. Use (capabilities, limitations, safety, evaluation) is **left to
    the upstream card**."""
    lines = [
        "## Base weights and attribution",
        "",
        *_storage_lines(manifest),
        "",
        # A paragraph between the two lists: CommonMark joins two adjacent lists of one marker.
        "The upstream model, its licence and what was changed:",
        "",
    ]
    for name in manifest["models"]:
        repo = EMBEDDINGGEMMA2_UPSTREAM[name]
        lines.append(
            f"- **`{name}`**: [{repo}](https://huggingface.co/{repo}), licensed **Apache 2.0**"
            f" ([full text]({EMBEDDINGGEMMA2_LICENSE_TEXT_LINK}); a verbatim copy is in"
            " `LICENSE.md`)."
        )
    upstream = EMBEDDINGGEMMA2_UPSTREAM[manifest["defaultModel"]]
    weights_change = (
        [
            "  checkpoint's bf16 values unchanged in `bf16`, and an int8-quantized copy (`i8`) was",
            "  added.",
        ]
        if EMBEDDINGGEMMA2_I8_DTYPE in _stored_dtypes(manifest)
        else ["  checkpoint's bf16 values unchanged."]
    )
    lines += [
        "- **Changes made here** (also listed in `NOTICE.md`, per Apache 2.0 §4(b)): the text",
        "  tower was extracted and re-expressed, with the mean pooling and the L2 normalisation,",
        "  as one graph in the Karume container format; the sliding-window attention mask is a",
        "  band declared on the attention, applied from the token positions; the weights keep the",
        *weights_change,
        "  No retraining and no fine-tuning.",
        *_precision_lines(manifest),
        "- **What the model can and cannot do** — capabilities, limitations, intended use,",
        "  evaluations and safety guidance — is documented on the upstream model card, and this",
        f"  repository does not restate it: read [{upstream}](https://huggingface.co/{upstream})",
        "  before using it.",
    ]
    return lines


def _usage(manifest: Mapping[str, Any], repo: str) -> list[str]:
    """The usage example: the minimal working form (`fromPretrained` → `embed`), with the options
    one would commonly touch as comments (the selectable values derived from the manifest).

    NOTE: `fromAssets` is not shown (ruling of 2026-08-29 — the same for every family).
    """
    model_name = manifest["defaultModel"]
    model = default_model(manifest)
    config = model["pipelineConfig"]
    quant = model["defaultQuant"]
    model_names = " / ".join(sorted(manifest["models"]))
    quant_names = " / ".join(sorted(model["quants"]))
    dimensions = config["dimensions"]
    return [
        "## Usage",
        "",
        "```ts",
        'import { Pipeline } from "jsr:@karume/models/embeddinggemma";',
        "",
        *from_pretrained(
            "Pipeline",
            repo,
            [
                f'  // model: "{model_name}", // default — available: {model_names}',
                f'  // quant: "{quant}", // default — available: {quant_names}',
            ],
            disposable="await using",
        ),
        "",
        'const query = await pipeline.embed("how do I reset my password", { prompt: "query" });',
        "const document = await pipeline.embed(",
        '  "Open Settings, choose Account and select Reset password.",',
        '  { prompt: "document" },',
        ");",
        "",
        "// Unit vectors: their dot product is the cosine similarity.",
        "const score = query.reduce((sum, value, index) => sum + value * document[index], 0);",
        "",
        "// A shorter Matryoshka vector (queries and documents must share a dimension):",
        f'// await pipeline.embed(text, {{ prompt: "query", dimension: {dimensions[-1]} }});'
        f" // available: {_spell(dimensions)}",
        "```",
        "",
        "`embed()` takes one text per call and returns an f32 unit vector. It keeps one GPU",
        "session alive for the lifetime of the pipeline, so embedding many texts uploads the",
        "weights once;",
        "concurrent calls are queued rather than run side by side. A prompt name the repository",
        "does not declare, a dimension it does not list or a text over the token limit is refused.",
        "Weights are fetched once and cached (verified against `karume.json`'s `size` / `sha256`).",
    ]


def _prompts_and_limits(model: Mapping[str, Any]) -> list[str]:
    """The prompts (grouped by prefix, in the order upstream declares them), the dimensions and the
    input limits of one model."""
    config = model["pipelineConfig"]
    by_prefix: dict[str, list[str]] = {}
    for name, prefix in config["prompts"].items():
        by_prefix.setdefault(prefix, []).append(name)
    placeholders = ", ".join(
        f"{kind} {token}" for kind, token in config["placeholderTokens"].items()
    )
    lines = [
        "### Prompts, dimensions and limits",
        "",
        "Read from the checkpoint's own `config_sentence_transformers.json` and `config.json` and",
        "the exported graph, and checked against each other when this repository was assembled.",
        "",
        "`prompt` names one of these; its prefix is prepended to the text as is (no prompt, no",
        "prefix). Pick it by the role a text plays — `query` for a search query and `document`",
        "for what it searches, for instance.",
        "",
        "| Prefix | Prompt names |",
        "| ------ | ------------ |",
    ]
    for prefix, names in by_prefix.items():
        # A table cell escapes the pipe, inside a code span too (GitHub Flavored Markdown).
        cell = prefix.replace("|", "\\|")
        lines.append(f"| `{cell}` | {', '.join(f'`{name}`' for name in names)} |")
    lines += [
        "",
        f"- **dimensions**: {_spell(config['dimensions'])} (the first is the full vector)",
        f"- **tokens**: at most {config['maxTokens']:,} per text, `<bos>` and `<eos>` included",
        f"- **refused tokens**: the multimodal placeholders ({placeholders})",
    ]
    return lines


def _i8_screening(model: Mapping[str, Any]) -> list[str]:
    """The measured distance of the `i8` preset's weights from the original ones — only for a
    model with a preset that stores the text tower as int8 (none, no section)."""
    if not any(
        quant["weights"][EMBEDDINGGEMMA2_MODEL_ROLE] == EMBEDDINGGEMMA2_I8_DTYPE
        for quant in model["quants"].values()
    ):
        return []
    return [
        "### How far the int8 weights are from the original ones",
        "",
        *EMBEDDINGGEMMA2_I8_SCREENING,
    ]


def render_embeddinggemma2_model_card(
    manifest: Mapping[str, Any], repo: str, host_assets: Mapping[FileRefKey, int] = {}
) -> str:
    """Build the body of the EmbeddingGemma 2 distribution's `README.md` (a pure function,
    trailing newline)."""
    require_pipeline(manifest, EMBEDDINGGEMMA2_SUPPORTED_PIPELINE)
    metadata = _metadata(manifest)
    return render(
        (
            frontmatter(metadata),
            introduction(EMBEDDINGGEMMA2_TITLE, hf_upstreams(metadata.base_model)),
            _overview(manifest),
            [""],
            _base_weights(manifest),
            [""],
            models(manifest),
            [""],
            _usage(manifest, repo),
            *model_sections(
                manifest,
                (partial(quants, host_assets=host_assets), _i8_screening, _prompts_and_limits),
            ),
        )
    )
