"""Script that writes the text tower of EmbeddingGemma 2 (sentence embeddings) to an IR v2 container
+ golden io.

The artifacts go under `outputs/series/` and are kept out of commits by the repository root's
`.gitignore` (the f32 weights are about 1.1 GB).

    uv run --with 'transformers==5.23.0' python -m embeddinggemma2.export
    uv run --with 'transformers==5.23.0' python -m embeddinggemma2.export --dtype bf16
    uv run --with 'transformers==5.23.0' python -m embeddinggemma2.export --dtype i8

transformers is **pinned to 5.23.0** (a change of the modeling code changes the graph's form, and
EmbeddingGemma 2 first ships in 5.19.0). It is not added to pyproject.toml / uv.lock but added per
run with `--with`; {@link assert_transformers_version} refuses any other release, so the synced
environment's own release never wins silently.

## Source

The pinned snapshot of `google/embeddinggemma-2` ({@link embeddinggemma2.sources.PIN}); every file
the script reads is checked against the pin before it is read. `model.safetensors` holds three
towers; only its 413 `language_model.*` tensors are read, renamed to the keys of
`EmbeddingGemma2TextModel` and loaded strictly into that class alone ({@link text_state}) — the
vision and audio towers are never built. The weights are bf16 and are upcast to f32 (exact).

## What goes into the graph

The text tower and the two SentenceTransformer stages after it, as one graph:

    EmbeddingGemma2TextModel (24 layers → final RMSNorm → per-token Linear 512→768)
    → Pooling(masked mean, prompt included) → Normalize(L2)

The output is one unit vector `[1, 768]`. Adding the prompt prefix and tokenizing (`<bos>` … `<eos>`
from the post-processor of `tokenizer.json`) happen on the host. The order of operations is
upstream's: each token is projected to 768 inside the text model, then pooled. The layout this
hard-codes is checked against the upstream declarations ({@link assert_sentence_transformer_layout}
— sentence-transformers 6's module types, mean pooling with the prompt included, no Dense).

## Soft tokens from a tower

The graph inputs are `input_ids` `[1, T]` i32, `pool_mask` `[1, T]` f32, `soft_embeds` `[1, T, 512]`
f32 and `soft_mask` `[1, T, 1]` bool ({@link GRAPH_INPUTS}) — the names and the mask form of the
Gemma 4 12B product graph (`gemma4/export_unified.py`). The rows the text tower runs on are
`where(soft_mask, soft_embeds, embed_tokens(input_ids))` ({@link merged_embeddings}): a soft row
takes a tower's output (`embed_vision` of `embeddinggemma2/export_vision.py` for an image) as it
is, after `embed_tokens` applied its `sqrt(512)` to the text rows — upstream's composite model
overwrites the scaled lookup with unscaled soft tokens the same way (`masked_scatter` in
`EmbeddingGemma2Model.forward`). The per-layer embedding is projected from the merged rows
(`EmbeddingGemma2TextModel.forward` computes it from `inputs_embeds`), so nothing else changes;
positions stay `arange(T)` and the band applies to soft rows as to text rows. `soft_mask` is
`[1, T, 1]`: `where` broadcasts it over the 512 columns, so the host binds one word per row.

What sits under a soft row's id does not matter (the lookup is overwritten); the composite feeds
the pad id there, a host may feed the placeholder id itself. With an all-false mask and zero rows
({@link no_soft_inputs}) `where` selects the text operand exactly, so the graph computes the text
forward bit for bit and the text goldens keep their values: {@link window_evidence} checks the
wrapper against the `input_ids` path with `torch.equal` on every long case, and
{@link assert_soft_merge} checks the form (one merge, its text operand from `input_ids` alone and
read by nothing else, the soft inputs read by nothing else).

Nothing upstream is reimplemented: the RMSNorm (`x · (mean(x²) + eps)^-0.5 · weight` — the weight
as is, not `1 + weight` as in the first EmbeddingGemma), the projection-only PLE, the three residual
sub-blocks of each layer and the per-layer `layer_scalar` multiply (none of the 24 values is 1, so
it stays a real multiply) all run through `EmbeddingGemma2TextModel.forward` unchanged.

## Attention: bidirectional, GQA in its true form, the window as a declared band

Every layer is bidirectional (`is_causal = False`). The sliding layers see `|q − kv| <= 512`
(inclusive, symmetric — upstream's `sliding_window_bidirectional_overlay`), the four full layers
(5 / 11 / 17 / 23) see everything.

`EmbeddingGemma2TextModel.forward` hands `attention_mask` to the layers as is when it is a dict, so
the wrapper builds `{"full_attention": None, "sliding_attention": band}` itself
({@link layer_masks}) and upstream's masking utilities are bypassed. The band's leaves are only
`arange` and `full` with T at extent positions, so the export traces it as **one Tmax×Tmax f32
constant + `sym_prefix_slice`** (ADR 0010) shared by the 20 sliding layers — the form
{@link assert_ir_form} checks. The full layers carry **no mask** in the graph (upstream under
torch.export would materialise an all-zero one; the Irodori recipe's precedent of dropping it). No
attention mask is a graph input.

The traced band then goes through the core's band surgery ({@link declare_band} —
`karume.band.to_band_form`, ADR 0155): each sliding layer's attention drops the mask input and
declares `attrs.band` = the config's `sliding_window` (512), so the score kernel checks the window
from positions, and the Tmax×Tmax constant is pruned — at the upstream context it is
`[1, 1, 8192, 8192]` f32 = 256 MiB, eight times the container's 32 MiB block, and it never reaches
the container. The surgery does not read the mask; that the declared band is the mask the model
applies is this script's claim, backed by {@link assert_ir_form} (the traced constant is the
inclusive band of the config's window), {@link assert_band_matches_upstream} (that band is
upstream's own mask function) and {@link window_evidence}. {@link assert_band_form} checks the
result.

MUST: the maximum T is {@link SYM_MAX} = 8,192, the upstream model's context. After the surgery the
bound is carried by the folded RoPE tables (`[1, 1, 8192, 256]` on the sliding layers and
`[1, 1, 8192, 512]` on the full ones, 8 / 16 MiB, cut down by `sym_prefix_slice`): a T beyond it
fails loudly at the prefix slice (`sym_prefix_slice` refuses a prefix longer than its input), never
approximated. The scheme stays single-sequence (one text per run, no padding, no batch).

Grouped-query attention stays in its true form: {@link bidirectional_gqa_attention} is registered
at transformers' `AttentionInterface` and calls SDPA with `enable_gqa=True`, without `repeat_kv`,
so the IR's attention is `q[1,4,T,256] / k,v[1,2,T,256]` on the sliding layers and
`q[1,4,T,512] / k,v[1,1,T,512]` on the full layers. Upstream passes `sliding_window=513` to the
attention implementation for flash attention only; the hook accepts and ignores it (the mask
dictionary is the window's one source of truth). The Gemma 4 recipe's hook is the model for it but
cannot be imported as is: it refuses a missing mask (right for a causal decoder, wrong here, where
the full layers have none).

RoPE (default type, sliding θ 1e4 over 256 dims, full θ 1e6 over 512 dims) is upstream's own; its
`*_inv_freq` buffers are demoted with the core's suffix match (`karume.rope.assert_rope_lifted`) so
the cos / sin tables fold to constants.

## Checks before anything is published

- {@link assert_ir_form} — the traced form: 24 fused attention nodes in layer order with each
  layer's q / k / v shapes, a mask on exactly the 20 sliding layers and none on the full ones, one
  band constant equal to the inclusive band, the four inputs and the one `[1, 768]` output. Each is
  a property that breaks while the numbers still agree.
- {@link assert_soft_merge} — the soft inputs' dtypes and shapes and their one merge: a soft row
  that passed through `embed_scale`, a per-layer embedding projected from the unmerged lookup or a
  mask left unread keep every text golden.
- {@link assert_band_matches_upstream} — the band equals upstream's own mask function at Tmax.
- {@link assert_band_form} — the form that is written: no attention takes a mask, exactly the
  sliding layers declare the band of the window and the full ones none, the traced band constant
  is gone, and the bound of T is still readable from the folded constants left (at Tmax).
- {@link window_evidence} — on every case whose T cuts the window (T >= 514), the wrapper's output
  differs from the same model with every position visible by far more than float noise (the window
  is really exercised), and equals the upstream model's own forward (attention_mask of ones,
  upstream's masking utilities, upstream's SDPA path) to f32 noise; the wrapper fed no soft row
  equals the `input_ids` path bit for bit.
- {@link _sanity} — unit norms, and a near pair beating a far pair in cosine.
- {@link round_weights_to_i8} (`--dtype i8`) — the rounding reached exactly the linears and the
  token table; {@link assert_storage_form} — the stored form is the storage asked for, and under i8
  its weight slots are exactly the rounded weights.

## Numerics

The goldens come from torch on CPU in f32 with the bf16 weights upcast. The embedding scale is then
`sqrt(512)` as an f32 (22.6274); upstream running in bf16 rounds it to 22.625 — the opt-in parity
script ({@link embeddinggemma2.parity}) measures that gap and the agreement with
SentenceTransformer.

## Storage: f32, the checkpoint's own bf16 values, or per-row int8

`--dtype` picks the storage of the weights, one series per storage (ADR 0019 — sharing a directory
would put one storage's goldens on the other's assets):

- `f32` (the default, {@link TEXT_SERIES}) — every weight widened to f32, the series the e2e
  check reads;
- `bf16` (`<series>-bf16`) — what the distribution's original-weights preset ships (ADR 0131):
  every weight in a weight slot (the embedding and the linears) is stored as bf16, the rest (the
  norms, the per-layer scalars, the folded constants) stays f32. The checkpoint is BF16, so nothing
  is rounded: the loaded f32 values are the checkpoint's bits, the writer stores a weight as bf16
  only when the round trip is exact and fails otherwise, and the goldens are the f32 series' own
  (the same values go through the same code).
- `i8` (`<series>-i8`) — the distribution's int8 preset: the same weights (the token table and
  every linear — 219 tensors on the real model) are rounded to per-row symmetric int8
  (`karume.quantize.fake_quant_int8`, one F32 scale per output row, per vocabulary row for the
  table) **before** anything is traced or measured ({@link round_weights_to_i8}), so the window
  evidence, the sanity check and the goldens all run on the rounded model: the series' goldens are
  the quantized model's own outputs, and the e2e tolerance measures implementation error only (ADR
  0006). The rest stays f32 as in bf16. The writer stores each rounded weight with the scale the
  rounding used and checks that `q · scale` gives the rounded values back bit for bit. The graph,
  the band declaration, the cases and the bound are the other series' own.

{@link assert_storage_form} checks the stored form before anything is published.

## Multimodal placeholder ids

The composite model replaces the image / video / audio placeholder ids (258880 / 258884 / 258881)
with the pad id before the embedding lookup; the text tower alone does not. A placeholder row is
only right when the host marks it in `soft_mask` and fills it from a tower ("Soft tokens from a
tower"); a placeholder id left as a text row is not upstream's composite. The text goldens hold
none ({@link build_cases} refuses them); the inputs holding an image are the reference vectors of
`embeddinggemma2/multimodal_reference.py`.

## Output layout

    outputs/series/embeddinggemma-2-text/model-NNNNN-of-NNNNN.krm  weights, constants, descriptors
    outputs/series/embeddinggemma-2-text/io.<case>.safetensors     inputs + torch CPU expectation
    outputs/series/embeddinggemma-2-text-bf16/…                    the same with `--dtype bf16`
    outputs/series/embeddinggemma-2-text-i8/…                      `--dtype i8` (its own goldens)

The io tensor keys are `input.<graph input name>` / `output.<position>` ({@link INPUT_PREFIX} /
{@link OUTPUT_PREFIX}). `pool_mask` is all ones (every token is pooled, the prompt's included);
`soft_embeds` and `soft_mask` are zeros and false (no golden text holds a soft row).

Every run also writes the golden texts to a fixture tracked in git,
{@link GOLDEN_CASE_FIXTURE} ({@link write_golden_case_fixture}): the case names, the prompt names
and the bodies of {@link GOLDEN_CASES} with their lengths, so the TypeScript pipeline's e2e embeds
the same texts through the distribution without spelling the long documents a second time.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Collection, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from torch import nn
from torch.export import Dim

from _shared.paths import REPO_ROOT, SERIES_ROOT
from embeddinggemma2 import sources
from embeddinggemma2.tokenizer import REQUIRED_TRANSFORMERS
from karume.artifacts import staged_publication
from karume.band import BandAttentionSpec, to_band_form
from karume.container import Provenance, container_parts
from karume.convert import PRESERVED_OP_PREFIXES_WITH_ATTENTION, normalize_boundary_tensor
from karume.emit import eligible_compressed_initializers
from karume.ir import IrGraph, IrNode
from karume.ops import ATTENTION_OP, SYM_PREFIX_SLICE_OP
from karume.pipeline import export_module, publish_model
from karume.quantize import QUANT_MODULE_TYPES, Int8Report, fake_quant_int8, iter_quant_targets
from karume.rope import assert_rope_lifted
from karume.shapes import declared_shape

#: The series this script writes (assets-layout: `outputs/series/<series>/`).
TEXT_SERIES = "embeddinggemma-2-text"

#: The storage dtypes of the weights `--dtype` takes (module doc, "Storage").
WEIGHT_DTYPES = ("f32", "bf16", "i8")

#: The default location of the artifacts, one series per storage dtype. MUST: one directory per
#: dtype (ADR 0019 — sharing one puts one storage's goldens on the other's assets).
DEFAULT_OUT_DIRS: Mapping[str, Path] = {
    "f32": SERIES_ROOT / TEXT_SERIES,
    "bf16": SERIES_ROOT / f"{TEXT_SERIES}-bf16",
    "i8": SERIES_ROOT / f"{TEXT_SERIES}-i8",
}

#: The module name of the token table inside the text tower (the one `nn.Embedding` the i8 series
#: rounds besides the linears).
TABLE_MODULE = "embed_tokens"

#: The license identifier baked into the container (container-v1 §2.3) — the upstream model card's
#: front matter at the pinned revision (`THIRD_PARTY_NOTICES.md`).
LICENSE = "apache-2.0"

#: The representative path of the output container inside a series (the text graph's and the two
#: towers').
MODEL_FILE = "model.krm"

#: The container's graph name = **the component name of the distribution** (= the key in `weights`
#: of `karume.json` — container-v1 §2.1). The text graph takes the convention of a single-component
#: series (`model`); it does not match the series directory name (`embeddinggemma-2-text`).
GRAPH_NAME = "model"

#: The golden io files (`io.<case>.safetensors`) and their tensor keys (`input.<graph input name>` /
#: `output.<position>`) — the convention the text graph and the two towers share.
IO_PREFIX = "io."
IO_SUFFIX = ".safetensors"
INPUT_PREFIX = "input."
OUTPUT_PREFIX = "output."

#: The weights file name of the upstream checkpoint (HF's spelling — not the output container).
CHECKPOINT_FILE = "model.safetensors"

#: The key prefix of the text tower in the checkpoint (the composite model's `language_model`).
CHECKPOINT_TEXT_PREFIX = "language_model."

#: The snapshot files the script reads (each is checked against the pin before it is read).
READ_FILES: tuple[str, ...] = (
    CHECKPOINT_FILE,
    "config.json",
    "tokenizer.json",
    "modules.json",
    "1_Pooling/config.json",
    "sentence_bert_config.json",
    "config_sentence_transformers.json",
)

#: `modules.json` as `(path, type)` — exactly the stages the wrapper folds into one graph.
SENTENCE_TRANSFORMER_MODULES: tuple[tuple[str, str], ...] = (
    ("", "sentence_transformers.base.modules.transformer.Transformer"),
    ("1_Pooling", "sentence_transformers.sentence_transformer.modules.pooling.Pooling"),
    ("2_Normalize", "sentence_transformers.base.modules.normalize.Normalize"),
)
#: Where the Pooling config lives.
POOLING_DIR = "1_Pooling"
#: The whole Pooling config the wrapper hard-codes (masked mean, prompt tokens included). Compared
#: as a whole, so an added option fails too.
POOLING_CONFIG: Mapping[str, Any] = {
    "embedding_dimension": 768,
    "pooling_mode": "mean",
    "include_prompt": True,
}
#: How the Transformer stage calls the model for text (`sentence_bert_config.json`): the forward,
#: and its `last_hidden_state` (the per-token 768-dim projections) as the token embeddings.
TEXT_MODALITY: Mapping[str, str] = {"method": "forward", "method_output_name": "last_hidden_state"}

#: Lower bound of the masked-mean denominator (same value as sentence-transformers' Pooling).
POOL_EPS = 1e-9
#: Lower bound of the L2 normalisation denominator (same value as sentence-transformers' Normalize).
NORM_EPS = 1e-12

#: Upper bound of the symbolic dimension T — the folding's evaluation point and the most tokens one
#: text may encode to. It is the upstream model's context window (the model card at the pinned
#: revision: "Context Window | 8,192 tokens"); no configuration file of the snapshot states it
#: (`text_config.max_position_embeddings` is 262,144, the RoPE's own range), so it is this recipe's
#: constant. The traced band constant at this bound (256 MiB) is pruned by {@link declare_band}.
SYM_MAX = 8192

#: The graph inputs that carry soft tokens from a tower (module doc, "Soft tokens from a tower") —
#: the Gemma 4 12B product graph's spellings.
SOFT_EMBEDS = "soft_embeds"
SOFT_MASK = "soft_mask"

#: The graph input names (the wrapper's forward argument names), in order.
GRAPH_INPUTS = ("input_ids", "pool_mask", SOFT_EMBEDS, SOFT_MASK)

#: The IR op of the soft-row merge.
WHERE_OP = "where"

#: The layer types of `config.layer_types` (= the mask dictionary's keys).
FULL_ATTENTION = "full_attention"
SLIDING_ATTENTION = "sliding_attention"

#: The blocking value of an additive mask (the value `normalize._additive_attn_mask` gives a bool
#: mask).
NEG_INF = float("-inf")

#: The name registered with `AttentionInterface`. It contains none of `sdpa` / `flash_attention` /
#: `flex_attention` (a partial name match branches into backend-specific dispatch checks).
ATTENTION_NAME = "karume_bidirectional_gqa"

#: The attention implementation of upstream's own forward in {@link window_evidence}.
UPSTREAM_ATTENTION = "sdpa"

#: The bound of "equal to f32 noise" between the wrapper and upstream's own forward: the max abs
#: difference of the unit-vector output, and the max abs difference of the per-token hidden states
#: relative to their largest magnitude (they reach about 3,700, so an absolute bound means nothing
#: there). Measured: 0 for both, on every case. Moving the window by one position (511 or 513)
#: moves the output by 2.5e-4 to 5.6e-4 — over two orders of magnitude above this bound.
UPSTREAM_TOLERANCE = 1e-6
#: The floor of "differs by more than float noise" between the band and every position visible
#: (max abs difference of the unit-vector output). Measured: 1.09e-2 / 2.91e-2 / 3.89e-2 /
#: 3.95e-2 / 5.09e-2 / 4.98e-2 at T 609 / 1,508 / 2,048 / 3,062 / 5,935 / 8,192 — ten times above
#: the floor at the shortest.
WINDOW_FLOOR = 1e-3

#: The prompt kinds of the golden cases (keys of `prompts` in `config_sentence_transformers.json`,
#: whose values are used verbatim).
QUERY = "query"
DOCUMENT = "document"

#: A passage of about 300 tokens, the start of the long cases.
LONG_PASSAGE = (
    "A municipal archive that decides to digitise its holdings soon learns that scanning is the "
    "easy part. The scanners run day and night, but every page still needs a record that says "
    "what it is, where it came from and who may read it. Old ledgers mix several hands on one "
    "page, maps were folded and refolded until the creases cut through the street names, and "
    "photographs arrive in envelopes whose labels describe a different picture. Cataloguers "
    "therefore work in two passes: a quick one that gives every item a stable identifier and a "
    "date range, and a slow one that adds the descriptions people actually search for. "
    "Search is where the investment pays off or fails. A historian looking for flood damage in the "
    "eighteen nineties rarely types the words the clerk used at the time; she asks about water, "
    "cellars and repairs, while the ledger speaks of assessments, relief payments and the parish "
    "fund. Keyword indexes miss most of those matches. An embedding model that maps the question "
    "and the catalogue entry into the same space finds them, because it has learned that the two "
    "descriptions are about the same event. "
    "The archive does not need the model to be clever, only consistent: the same entry must land "
    "in the same place every time it is encoded, on every machine that serves the catalogue, so "
    "that a result found last week can be found again today. The pages below are the archive's "
    "own log of the figures it was asked about most often."
)

#: The topics the numbered records of the long cases cycle through.
_RECORD_TOPICS = (
    "river levels",
    "bridge repairs",
    "school enrolment",
    "harbour traffic",
    "orchard yields",
    "library loans",
    "tram ridership",
    "rainfall totals",
    "market prices",
    "museum visits",
    "ferry delays",
    "solar output",
    "bakery sales",
    "glacier retreat",
    "bird counts",
)


def numbered_records(count: int) -> str:
    """`count` numbered records (about 17 tokens each), every one different from the others."""
    return " ".join(
        f"Record {n}: {_RECORD_TOPICS[n % len(_RECORD_TOPICS)]} changed by {(n * 37) % 97} percent"
        f" in district {chr(65 + n % 26)}."
        for n in range(1, count + 1)
    )


def long_document(records: int, filler: int = 0) -> str:
    """{@link LONG_PASSAGE} followed by `records` numbered records and `filler` times ` and` (one
    token each — what brings a case to an exact length)."""
    return f"{LONG_PASSAGE} {numbered_records(records)}" + " and" * filler


#: The fixed golden texts (`(case name, prompt kind | None, body)`). Short cases with and without a
#: prompt, then the long ones: T 609 / 1,508 (past the first window cut at T = 514) and 2,048 (the
#: bound of the series before the band was declared), then 3,062 and 5,935 (past the 2,048 of the
#: earlier bound; at 5,935 one head's `T × T` f32 scores are over 128 MiB, the WebGPU default
#: binding) and exactly {@link SYM_MAX} (the upstream context, the far corner of every folded
#: constant). The lengths are pinned by {@link CASE_LENGTHS}.
GOLDEN_CASES: tuple[tuple[str, str | None, str], ...] = (
    ("query-en", QUERY, "What is the capital of France?"),
    ("document-en", DOCUMENT, "Paris is the capital and most populous city of France."),
    ("query-ja", QUERY, "フランスの首都はどこですか。"),
    ("bare", None, "The quick brown fox jumps over the lazy dog."),
    ("document-609", DOCUMENT, long_document(19)),
    ("document-1508", DOCUMENT, long_document(72)),
    ("document-2048", DOCUMENT, long_document(103, filler=10)),
    ("document-3062", DOCUMENT, long_document(160)),
    ("document-5935", DOCUMENT, long_document(320)),
    ("document-8192", DOCUMENT, long_document(445, filler=10)),
)

#: T (`<bos>` and `<eos>` included) of each golden case at the pinned tokenizer. A drift fails
#: {@link build_cases} instead of moving the window evidence or the full-range case silently.
CASE_LENGTHS: Mapping[str, int] = {
    "query-en": 16,
    "document-en": 19,
    "query-ja": 16,
    "bare": 12,
    "document-609": 609,
    "document-1508": 1508,
    "document-2048": 2048,
    "document-3062": 3062,
    "document-5935": 5935,
    "document-8192": 8192,
}

#: The golden texts as the TypeScript pipeline's e2e reads them (tracked in git — read by
#: `packages/e2e/tests/e2e_embeddinggemma2_pipeline_test.ts`).
GOLDEN_CASE_FIXTURE = (
    REPO_ROOT
    / "packages"
    / "models"
    / "tests"
    / "fixtures"
    / "embeddinggemma2"
    / "golden-cases.json"
)

#: The cosine pairs the sanity check compares (near pair, far pair).
NEAR_PAIR = ("query-en", "document-en")
FAR_PAIR = ("query-en", "bare")


class EmbeddingGemma2ExportError(ValueError):
    """An input of the script is not what the graph is built for (fails before writing)."""


def assert_transformers_version(version: str) -> None:
    """The transformers that is imported is {@link REQUIRED_TRANSFORMERS}."""
    if version != REQUIRED_TRANSFORMERS:
        raise EmbeddingGemma2ExportError(
            f"transformers {version} is imported, the script needs {REQUIRED_TRANSFORMERS}: run"
            f" `uv run --with 'transformers=={REQUIRED_TRANSFORMERS}' python -m"
            " embeddinggemma2.export`"
        )


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def assert_sentence_transformer_layout(directory: Path) -> None:
    """The upstream SentenceTransformer layout is exactly what the wrapper hard-codes.

    MUST: read from the upstream declarations and compared — the goldens are taken from the same
    wrapper, so the TS-side comparison would stay green if upstream added a Dense, switched the
    pooling or stopped including the prompt (`pool_mask` all ones rests on `include_prompt`).
    """
    modules = _read_json(directory / "modules.json")
    layout = tuple((module.get("path"), module.get("type")) for module in modules)
    if layout != SENTENCE_TRANSFORMER_MODULES:
        raise EmbeddingGemma2ExportError(
            f"the modules.json layout {layout} is not the wrapper's {SENTENCE_TRANSFORMER_MODULES}"
        )
    pooling = _read_json(directory / POOLING_DIR / "config.json")
    if pooling != POOLING_CONFIG:
        raise EmbeddingGemma2ExportError(
            f"{POOLING_DIR}/config.json is {pooling}, the wrapper hard-codes {dict(POOLING_CONFIG)}"
        )
    text = (
        _read_json(directory / "sentence_bert_config.json").get("modality_config", {}).get("text")
    )
    if text != TEXT_MODALITY:
        raise EmbeddingGemma2ExportError(
            f"sentence_bert_config.json calls the model for text as {text},"
            f" the wrapper takes {dict(TEXT_MODALITY)}"
        )


def bidirectional_band_mask(length: int, window: int) -> torch.Tensor:
    """The additive band mask `[1, 1, length, length]`: 0 where `|q − kv| <= window`, −inf
    elsewhere.

    Equivalent to upstream's `sliding_window_bidirectional_mask_function(window)` (inclusive,
    symmetric — {@link assert_band_matches_upstream} checks it at Tmax). It first masks something
    at `length = window + 2`.

    MUST: the leaves are only `arange` and `full`, `length` appears only as an arange length (an
    extent position), and the test is `sub` + `abs` + `le.Scalar` — every op of it is in
    `convert.FOLDABLE_OPS`, so the band folds into a Tmax constant; a comparison outside that set
    would make the whole export fail.

    NOTE: during export `length` is a `torch.SymInt`.
    """
    rows = torch.arange(length).unsqueeze(-1)
    cols = torch.arange(length).unsqueeze(0)
    keep = torch.full((), 0.0, dtype=torch.float32)
    drop = torch.full((), NEG_INF, dtype=torch.float32)
    return torch.where(torch.abs(rows - cols) <= window, keep, drop).unsqueeze(0).unsqueeze(0)


def layer_masks(length: int, window: int) -> dict[str, torch.Tensor | None]:
    """The mask dictionary `EmbeddingGemma2TextModel.forward` passes to the layers as is.

    The full layers get `None` — with a bidirectional module, the hook reads it as "every position
    visible", and no mask enters the graph.
    """
    return {FULL_ATTENTION: None, SLIDING_ATTENTION: bidirectional_band_mask(length, window)}


def all_visible_masks() -> dict[str, torch.Tensor | None]:
    """The mask dictionary with every position visible on every layer (the window evidence's
    counterpart)."""
    return {FULL_ATTENTION: None, SLIDING_ATTENTION: None}


def bidirectional_gqa_attention(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    dropout: float = 0.0,
    scaling: float | None = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, None]:
    """An attention implementation for a bidirectional encoder that calls SDPA with
    `enable_gqa=True`, without `repeat_kv` (keeps the GQA / MQA form).

    The calling convention transformers' `AttentionInterface` requires (`(module, q, k, v,
    attention_mask, dropout=…, scaling=…, **kwargs)` → `(attn_output[B,T,H,D], attn_weights)`).

    MUST: the module is bidirectional (`is_causal is False`); anything else fails — `None` means
    "every position visible" only there (for a causal module it would silently drop causality).
    MUST: kwargs that change the numeric meaning (`is_causal` / `position_bias` / `softcap`) and a
    non-zero dropout are refused.
    NOTE: `sliding_window` is accepted and ignored — upstream passes it for flash attention only,
    and the window's one source of truth is the mask dictionary ({@link layer_masks}).
    """
    if getattr(module, "is_causal", None) is not False:
        raise ValueError(
            f"{type(module).__name__}: the bidirectional attention hook was called for a module"
            f" whose is_causal is {getattr(module, 'is_causal', None)!r}"
        )
    if float(dropout) != 0.0:
        raise ValueError(f"attention with dropout={dropout} is not supported (inference only)")
    unsupported = sorted(
        name for name in ("is_causal", "position_bias", "softcap") if kwargs.get(name) is not None
    )
    if unsupported:
        raise ValueError(
            f"attention with kwargs {unsupported} is not supported (the numeric path changes)"
        )
    output = torch.nn.functional.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attention_mask,
        dropout_p=0.0,
        scale=scaling,
        is_causal=False,
        enable_gqa=True,
    )
    return output.transpose(1, 2).contiguous(), None


def register_attention() -> None:
    """Register {@link bidirectional_gqa_attention} with transformers' attention registry
    (idempotent)."""
    from transformers import AttentionInterface

    AttentionInterface.register(ATTENTION_NAME, bidirectional_gqa_attention)


@dataclass(frozen=True)
class AttentionLayout:
    """One layer's attention form as the config declares it (what {@link assert_ir_form}
    expects)."""

    layer_type: str
    heads: int
    kv_heads: int
    head_dim: int


def attention_layouts(config: Any) -> tuple[AttentionLayout, ...]:
    """Each layer's attention form, in layer order, from the text config (`per_layer_config`
    resolves the wider full layers)."""
    layer_types = list(config.layer_types)
    if len(layer_types) != int(config.num_hidden_layers):
        raise EmbeddingGemma2ExportError(
            f"config.layer_types has {len(layer_types)} entries for"
            f" {config.num_hidden_layers} layers"
        )
    layouts: list[AttentionLayout] = []
    for index, layer_type in enumerate(layer_types):
        if layer_type not in (FULL_ATTENTION, SLIDING_ATTENTION):
            raise EmbeddingGemma2ExportError(f"layer {index}: unknown layer type {layer_type!r}")
        layer = config.per_layer_config[index]
        layouts.append(
            AttentionLayout(
                layer_type=layer_type,
                heads=int(config.num_attention_heads),
                kv_heads=int(layer.num_key_value_heads),
                head_dim=int(layer.head_dim),
            )
        )
    return tuple(layouts)


def text_state(model_file: Path) -> dict[str, torch.Tensor]:
    """The text tower's tensors of the composite checkpoint, renamed to `EmbeddingGemma2TextModel`'s
    keys and upcast to f32 (exact for bf16).

    Only `language_model.*` is read; the vision / audio towers and their embedders are not.
    """
    state: dict[str, torch.Tensor] = {}
    with safe_open(str(model_file), framework="pt") as handle:
        for key in handle.keys():  # noqa: SIM118 — safe_open is not a mapping
            if key.startswith(CHECKPOINT_TEXT_PREFIX):
                tensor = handle.get_tensor(key)
                state[key[len(CHECKPOINT_TEXT_PREFIX) :]] = tensor.to(torch.float32, copy=True)
    if not state:
        raise EmbeddingGemma2ExportError(f"{model_file} has no '{CHECKPOINT_TEXT_PREFIX}*' tensor")
    return state


def load_text_config(directory: Path) -> tuple[Any, Any]:
    """The composite config and its text part (`EmbeddingGemma2Config` /
    `EmbeddingGemma2TextConfig`)."""
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(directory)
    return config, config.text_config


def load_text_model(directory: Path) -> tuple[nn.Module, Any]:
    """The text tower with the real weights in f32 on the hook, its RoPE buffers demoted, and the
    composite config (for the placeholder ids).

    MUST: the load is strict — a key the class does not have, or a slot the checkpoint does not
    fill, fails (upstream renaming a key would otherwise leave a random tensor in the graph).
    """
    import transformers
    from transformers.models.embedding_gemma2.modeling_embedding_gemma2 import (
        EmbeddingGemma2TextModel,
    )

    assert_transformers_version(transformers.__version__)
    register_attention()
    composite, config = load_text_config(directory)
    if int(config.embedding_dim) != POOLING_CONFIG["embedding_dimension"]:
        raise EmbeddingGemma2ExportError(
            f"text_config.embedding_dim {config.embedding_dim} is not the pooling's"
            f" {POOLING_CONFIG['embedding_dimension']}"
        )
    config._attn_implementation = ATTENTION_NAME
    model = EmbeddingGemma2TextModel(config)
    model.load_state_dict(text_state(directory / CHECKPOINT_FILE), strict=True)
    model.eval()
    dtypes = {tensor.dtype for tensor in model.state_dict().values()}
    if dtypes != {torch.float32}:
        raise EmbeddingGemma2ExportError(f"the loaded text tower holds {dtypes}, not f32 alone")
    # Left as a buffer, inv_freq is not a leaf of constant folding, and sin / cos stay in the IR.
    assert_rope_lifted(model, "embeddinggemma2")
    return model, composite


def _masked_mean(hidden: torch.Tensor, pool_mask: torch.Tensor) -> torch.Tensor:
    """Masked mean of `[1,T,H]` over the token axis (same as `sentence_transformers`' Pooling).

    MUST: do not use `keepdim=True` (the IR's reduce has no keepdim). The denominator, the `[1]` of
    `sum(dim=1)`, is brought back to `[1,1]` with `unsqueeze` before the broadcast division
    (`unsqueeze` is lowered to `reshape` by normalisation).
    """
    total = torch.sum(hidden * pool_mask.unsqueeze(-1), dim=1)
    count = torch.sum(pool_mask, dim=1).unsqueeze(-1)
    return total / torch.clamp(count, min=POOL_EPS)


def _l2_normalize(x: torch.Tensor) -> torch.Tensor:
    """L2-normalise `[1,H]` over the last dimension (same as `sentence_transformers`' Normalize).

    MUST: write it as `x * x` + `sum(dim=-1)` + `sqrt`, not `x ** 2` or `linalg.vector_norm`
    (`pow` / `linalg_vector_norm` are not in the IR vocabulary). The order `clamp_min → division` is
    kept from the original implementation — replacing it with `+eps` loses the property of returning
    0 for a zero input.
    """
    norm = torch.sqrt(torch.sum(x * x, dim=-1)).clamp(min=NORM_EPS).unsqueeze(-1)
    return x / norm


def embed(
    model: nn.Module, input_ids: torch.Tensor, pool_mask: torch.Tensor, masks: Any
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the text tower with the given `attention_mask` (a mask dictionary, or a padding mask for
    upstream's masking utilities) and pool + normalise its per-token output; returns
    `(embedding [B, 768], hidden [B, T, 768])`."""
    hidden = model(input_ids=input_ids, attention_mask=masks).last_hidden_state
    return _l2_normalize(_masked_mean(hidden, pool_mask)), hidden


def merged_embeddings(
    model: nn.Module, input_ids: torch.Tensor, soft_embeds: torch.Tensor, soft_mask: torch.Tensor
) -> torch.Tensor:
    """The rows the text tower runs on: a soft row's tower output where `soft_mask` is set, the
    scaled token lookup elsewhere (module doc, "Soft tokens from a tower").

    MUST: the soft rows are not scaled — upstream overwrites the scaled lookup with them.
    """
    return torch.where(soft_mask, soft_embeds, model.embed_tokens(input_ids))


def no_soft_inputs(input_ids: torch.Tensor, hidden_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """`(soft_embeds [1, T, H], soft_mask [1, T, 1])` of an input without soft rows: zeros and all
    false — what the host binds for a text."""
    length = int(input_ids.shape[1])
    return (
        torch.zeros(1, length, hidden_size, dtype=torch.float32),
        torch.zeros(1, length, 1, dtype=torch.bool),
    )


def soft_inputs(
    input_ids: torch.Tensor, hidden_size: int, blocks: Sequence[tuple[int, torch.Tensor]]
) -> tuple[torch.Tensor, torch.Tensor]:
    """`(soft_embeds, soft_mask)` with each block `(start, rows [n, H])` placed at the positions
    `[start, start + n)` and {@link no_soft_inputs} elsewhere.

    MUST: blocks do not overlap and stay inside `[0, T)` — a soft row written twice or past the end
    fails loudly (upstream's `masked_scatter` count check).
    """
    soft_embeds, soft_mask = no_soft_inputs(input_ids, hidden_size)
    length = int(input_ids.shape[1])
    for start, rows in blocks:
        stop = start + int(rows.shape[0])
        if rows.dim() != 2 or int(rows.shape[1]) != hidden_size or not 0 <= start < stop <= length:
            raise EmbeddingGemma2ExportError(
                f"a soft block of shape {list(rows.shape)} at {start} does not fit [1, {length},"
                f" {hidden_size}]"
            )
        if bool(soft_mask[0, start:stop].any()):
            raise EmbeddingGemma2ExportError(f"the soft block at {start} overlaps another one")
        soft_embeds[0, start:stop] = rows
        soft_mask[0, start:stop] = True
    return soft_embeds, soft_mask


class EmbeddingWrapper(nn.Module):
    """Export wrapper: the text tower over the merged rows ({@link merged_embeddings}) with the
    wrapper-built masks, then masked mean and L2 normalisation.

    MUST: `attention_mask` is always the dictionary of {@link layer_masks}. Omitted, transformers
    builds the masks from the padding mask and the export no longer folds them.
    """

    def __init__(self, model: nn.Module, window: int) -> None:
        super().__init__()
        self.model = model
        self.window = window

    def forward(
        self,
        input_ids: torch.Tensor,
        pool_mask: torch.Tensor,
        soft_embeds: torch.Tensor,
        soft_mask: torch.Tensor,
    ) -> torch.Tensor:
        masks = layer_masks(input_ids.shape[1], self.window)
        rows = merged_embeddings(self.model, input_ids, soft_embeds, soft_mask)
        hidden = self.model(inputs_embeds=rows, attention_mask=masks).last_hidden_state
        return _l2_normalize(_masked_mean(hidden, pool_mask))

    def text(self, input_ids: torch.Tensor, pool_mask: torch.Tensor) -> torch.Tensor:
        """The forward of an input without soft rows (the host's text call)."""
        hidden_size = int(self.model.config.hidden_size)
        return self(input_ids, pool_mask, *no_soft_inputs(input_ids, hidden_size))


def trace(
    wrapper: EmbeddingWrapper, example_ids: torch.Tensor, example_mask: torch.Tensor, sym_max: int
) -> tuple[IrGraph, dict[str, torch.Tensor]]:
    """Export the wrapper with T symbolic in `[2, sym_max]` on every input (the example has no soft
    row — the graph does not depend on the values)."""
    seq = Dim("T", min=2, max=sym_max)
    soft_embeds, soft_mask = no_soft_inputs(example_ids, int(wrapper.model.config.hidden_size))
    return export_module(
        wrapper,
        (example_ids, example_mask, soft_embeds, soft_mask),
        dynamic_shapes=({1: seq}, {1: seq}, {1: seq}, {1: seq}),
        preserved=PRESERVED_OP_PREFIXES_WITH_ATTENTION,
    )


def load_prompts(directory: Path) -> dict[str, str]:
    """`prompts` of `config_sentence_transformers.json` (the source of truth for the prefixes)."""
    return _read_json(directory / "config_sentence_transformers.json")["prompts"]


def case_texts(directory: Path) -> tuple[tuple[str, str], ...]:
    """`(name, text)` of the golden cases, the prompt prefix taken verbatim from the snapshot."""
    prompts = load_prompts(directory)
    return tuple(
        (name, body if kind is None else prompts[kind] + body) for name, kind, body in GOLDEN_CASES
    )


def build_cases(
    directory: Path, composite: Any, sym_max: int
) -> tuple[tuple[str, torch.Tensor, torch.Tensor], ...]:
    """`(name, input_ids, pool_mask)` of the golden cases.

    Tokenization reads `tokenizer.json` with `tokenizers`; its post-processor adds `<bos>` …
    `<eos>`, which is checked here. Each case's T must be {@link CASE_LENGTHS}' and fit
    `[2, sym_max]`, and no case may hold a multimodal placeholder id (the module docstring).
    """
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(directory / "tokenizer.json"))
    text_config = composite.text_config
    placeholders = {
        int(composite.image_token_id),
        int(composite.video_token_id),
        int(composite.audio_token_id),
    }
    cases: list[tuple[str, torch.Tensor, torch.Tensor]] = []
    for name, text in case_texts(directory):
        ids = tokenizer.encode(text).ids
        if ids[0] != int(text_config.bos_token_id) or ids[-1] != int(text_config.eos_token_id):
            raise EmbeddingGemma2ExportError(f"{name}: the ids do not run <bos> … <eos>")
        found = sorted(placeholders.intersection(ids))
        if found:
            raise EmbeddingGemma2ExportError(
                f"{name}: holds the multimodal placeholder ids {found}"
            )
        if len(ids) != CASE_LENGTHS[name]:
            raise EmbeddingGemma2ExportError(
                f"{name}: T={len(ids)}, the case is pinned at {CASE_LENGTHS[name]}"
            )
        if not 2 <= len(ids) <= sym_max:
            raise EmbeddingGemma2ExportError(
                f"{name}: T={len(ids)} is outside the symbolic dimension's range [2, {sym_max}]"
            )
        tensor = torch.tensor([ids], dtype=torch.int64)
        cases.append((name, tensor, torch.ones_like(tensor, dtype=torch.float32)))
    return tuple(cases)


def golden_case_fixture() -> dict[str, Any]:
    """The golden texts as the fixture holds them: per case its name, its prompt name (the key of
    `prompts` in `config_sentence_transformers.json` — absent for a case without a prompt), its
    body and its token length, in {@link GOLDEN_CASES}' order.

    The prefix itself is not written: the pipeline takes it from the distribution's manifest by the
    prompt name, which is what the e2e checks.
    """
    return {
        "series": TEXT_SERIES,
        "cases": [
            {
                "name": name,
                **({} if kind is None else {"prompt": kind}),
                "text": body,
                "tokens": CASE_LENGTHS[name],
            }
            for name, kind, body in GOLDEN_CASES
        ],
    }


def write_golden_case_fixture(path: Path = GOLDEN_CASE_FIXTURE) -> Path:
    """Write {@link golden_case_fixture} to `path` (2-space JSON, UTF-8, trailing newline)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(golden_case_fixture(), indent=2, ensure_ascii=False)
    path.write_text(payload + "\n", encoding="utf-8")
    return path


def assert_case_coverage(
    cases: Sequence[tuple[str, torch.Tensor, torch.Tensor]], sym_max: int, window: int
) -> None:
    """At least one case cuts the window (T >= window + 2) and one runs at exactly `sym_max`.

    MUST: with every T <= window + 1 the band is all zeros and the sliding layers' window is never
    evaluated; without a case at `sym_max` the far corner of the folded constants (the RoPE
    tables) is never read.
    """
    lengths = [int(ids.shape[1]) for _, ids, _ in cases]
    if not any(length >= window + 2 for length in lengths):
        raise EmbeddingGemma2ExportError(
            f"no case reaches T={window + 2}, where the window first masks (lengths {lengths})"
        )
    if sym_max not in lengths:
        raise EmbeddingGemma2ExportError(f"no case runs at T={sym_max} (lengths {lengths})")


def reference_band(sym_max: int, window: int) -> torch.Tensor:
    """The band the folded constant must hold, written from the rule itself (`|i − j| <= window`
    → 0, else −inf) as `[1, 1, sym_max, sym_max]` f32."""
    index = torch.arange(sym_max)
    visible = (index.unsqueeze(-1) - index.unsqueeze(0)).abs() <= window
    zero = torch.zeros((), dtype=torch.float32)
    blocked = torch.full((), NEG_INF, dtype=torch.float32)
    return torch.where(visible, zero, blocked).reshape(1, 1, sym_max, sym_max)


def assert_band_matches_upstream(sym_max: int, window: int) -> None:
    """{@link reference_band} equals upstream's own bidirectional sliding mask function at Tmax
    (so the rule this script states is upstream's, not a reading of it)."""
    from transformers.masking_utils import sliding_window_bidirectional_mask_function

    index = torch.arange(sym_max)
    allowed = sliding_window_bidirectional_mask_function(window)(
        0, 0, index.unsqueeze(-1), index.unsqueeze(0)
    )
    expected = reference_band(sym_max, window)[0, 0] == 0
    if not torch.equal(torch.as_tensor(allowed).expand(sym_max, sym_max), expected):
        raise EmbeddingGemma2ExportError(
            "the stated band |i - j| <= window disagrees with upstream's mask function"
        )


def assert_ir_form(
    graph: IrGraph,
    tensors: Mapping[str, torch.Tensor],
    layouts: Sequence[AttentionLayout],
    sym_max: int,
    window: int,
    embedding_dim: int,
) -> dict[str, Any]:
    """Check the traced form (before {@link declare_band}): the fused attention in the true GQA form
    on every layer, the band on exactly the sliding layers as one folded constant, and the four
    inputs / one output.

    Each property breaks silently with the numbers still matching the goldens:

    - SDPA falling to the decomposed path leaves fewer attention nodes than layers
    - `repeat_kv` materialised turns the k / v head axis into the query's
    - a mask on a full layer (upstream's all-zero one) costs a Tmax² constant and an addition
    - the band reaching the wrong layers, or another window, agrees with the goldens at T <= 513
    - a mask left as a graph input means the host builds T² every call

    The constant being the inclusive band of `window` is also what ties the band
    {@link declare_band} declares to the mask the model applies.
    """
    names = [spec.name for spec in graph.inputs]
    if names != list(GRAPH_INPUTS):
        raise AssertionError(
            f"graph inputs are {names}, not {list(GRAPH_INPUTS)} (a mask may have stayed an input,"
            " or the soft inputs are missing)"
        )
    if len(graph.outputs) != 1:
        raise AssertionError(f"{len(graph.outputs)} IR outputs (an embedding has 1)")
    output_shape = declared_shape(graph, graph.outputs[0])
    if output_shape != [1, embedding_dim]:
        raise AssertionError(f"the output is {output_shape}, not [1, {embedding_dim}]")

    attentions = [node for node in graph.nodes if node.op == ATTENTION_OP]
    if len(attentions) != len(layouts):
        raise AssertionError(
            f"{len(attentions)} attention nodes for {len(layouts)} layers"
            " (SDPA may have fallen to the decomposed path)"
        )
    symbol = graph.symbols[0] if graph.symbols else None
    producer = {out: node for node in graph.nodes for out in node.outs}
    bands: set[str] = set()
    for index, (node, layout) in enumerate(zip(attentions, layouts, strict=True)):
        where = f"attention[{index}] ({layout.layer_type})"
        query, key, value = (declared_shape(graph, name) for name in node.ins[:3])
        expected_q = [1, layout.heads, symbol, layout.head_dim]
        expected_kv = [1, layout.kv_heads, symbol, layout.head_dim]
        if query != expected_q or key != expected_kv or value != expected_kv:
            raise AssertionError(
                f"{where}: q / k / v are {query} / {key} / {value}, not {expected_q} /"
                f" {expected_kv} (k / v widened to the query's heads means repeat_kv)"
            )
        if layout.layer_type == FULL_ATTENTION:
            if len(node.ins) != 3:
                raise AssertionError(f"{where}: {len(node.ins)} ins (a full layer has no mask)")
            continue
        if len(node.ins) != 4:
            raise AssertionError(f"{where}: {len(node.ins)} ins (a sliding layer has the band)")
        source = producer.get(node.ins[3])
        if source is None or source.op != SYM_PREFIX_SLICE_OP:
            raise AssertionError(
                f"{where}: the mask comes from {'no node' if source is None else source.op}, not"
                f" the {SYM_PREFIX_SLICE_OP} of a folded constant"
            )
        bands.add(source.ins[0])
    if len(bands) != 1:
        raise AssertionError(
            f"the sliding layers take {len(bands)} mask constants: {sorted(bands)}"
        )
    (band_name,) = bands
    initializer = graph.initializers.get(band_name)
    if initializer is None or initializer.tensor is None:
        raise AssertionError(f"the band's source '{band_name}' is not an initializer of its own")
    band = tensors[initializer.tensor]
    if tuple(band.shape) != (1, 1, sym_max, sym_max) or band.dtype != torch.float32:
        raise AssertionError(
            f"the band constant is {band.dtype} {tuple(band.shape)}, not f32"
            f" [1, 1, {sym_max}, {sym_max}]"
        )
    if not torch.equal(band, reference_band(sym_max, window)):
        raise AssertionError(f"the band constant is not the inclusive |i - j| <= {window} band")
    return {
        "attention_nodes": len(attentions),
        "sliding_layers": sum(layout.layer_type == SLIDING_ATTENTION for layout in layouts),
        "full_layers": [
            index for index, layout in enumerate(layouts) if layout.layer_type == FULL_ATTENTION
        ],
        "band_constant": band_name,
        "band_visible_per_middle_row": int((band[0, 0, sym_max // 2] == 0).sum()),
    }


def assert_soft_merge(graph: IrGraph, hidden_size: int) -> dict[str, Any]:
    """The soft rows' merge (module doc, "Soft tokens from a tower"): `soft_embeds` f32
    `[1, T, H]` and `soft_mask` bool `[1, T, 1]`, read by exactly one
    `where(soft_mask, soft_embeds, text)` and by nothing else, whose `text` operand derives from
    `input_ids` alone, and every value derived from `input_ids` read only on the way to that merge.

    MUST: checked in the graph — each of these breaks while every text golden still holds: soft
    rows that pass through `embed_scale`, a mask left unread, and a per-layer embedding (or a layer)
    that reads the lookup before the merge, so a soft row would reach one consumer and not another.
    """
    symbol = graph.symbols[0] if graph.symbols else None
    expected = {
        SOFT_EMBEDS: ("f32", [1, symbol, hidden_size]),
        SOFT_MASK: ("bool", [1, symbol, 1]),
    }
    declared = {spec.name: (spec.dtype, list(spec.shape)) for spec in graph.inputs}
    for name, form in expected.items():
        if declared.get(name) != form:
            raise AssertionError(f"'{name}' is {declared.get(name)} — not {form[0]} {form[1]}")
    merges = [node for node in graph.nodes if node.op == WHERE_OP]
    if len(merges) != 1 or merges[0].ins[:2] != [SOFT_MASK, SOFT_EMBEDS]:
        raise AssertionError(
            f"`{WHERE_OP}` nodes are {[node.ins for node in merges]} — not one"
            f" `{WHERE_OP}({SOFT_MASK}, {SOFT_EMBEDS}, text)`: the soft rows must reach it unscaled"
        )
    (merge,) = merges
    readers = sorted(
        {name for node in graph.nodes if node is not merge for name in node.ins if name in expected}
    )
    if readers:
        raise AssertionError(f"{readers} are read besides the merge")
    producers = {out: node for node in graph.nodes for out in node.outs}
    text = merge.ins[2]
    # The values the text operand derives from (walked back through the producers).
    chain: set[str] = set()
    frontier = [text]
    while frontier:
        value = frontier.pop()
        if value in chain:
            continue
        chain.add(value)
        node = producers.get(value)
        if node is not None:
            frontier.extend(node.ins)
    if GRAPH_INPUTS[0] not in chain or chain & {SOFT_EMBEDS, SOFT_MASK, GRAPH_INPUTS[1]}:
        raise AssertionError(
            f"the merge's text operand does not come from '{GRAPH_INPUTS[0]}' alone"
        )
    # The values on the way from input_ids to the merge (derived from input_ids, inside the chain).
    derived = {GRAPH_INPUTS[0]}
    for node in graph.nodes:
        if set(node.ins) & derived and node.outs[0] in chain:
            derived.update(node.outs)
    leaks = sorted(
        f"{name} → {node.op}"
        for node in graph.nodes
        if node is not merge and node.outs[0] not in chain
        for name in node.ins
        if name in derived
    )
    if leaks:
        raise AssertionError(
            f"values derived from '{GRAPH_INPUTS[0]}' are read outside the merge: {leaks}"
        )
    return {"merge": merge.outs[0], "text_operand": text, "text_chain": sorted(derived)}


def _attentions(graph: IrGraph) -> list[IrNode]:
    return [node for node in graph.nodes if node.op == ATTENTION_OP]


def declare_band(
    graph: IrGraph, layouts: Sequence[AttentionLayout], window: int
) -> tuple[IrGraph, list[str]]:
    """The band surgery (`karume.band.to_band_form`): every sliding layer's attention declares the
    band `window` in place of its mask, and the traced band constant is pruned. Returns the new
    graph and the initializers the surgery pruned.

    The layers are named by the attention nodes in layer order (the order {@link assert_ir_form}
    checked against `layouts`); the surgery never reads the mask (module doc, "Attention").
    """
    attentions = _attentions(graph)
    if len(attentions) != len(layouts):
        raise AssertionError(f"{len(attentions)} attention nodes for {len(layouts)} layers")
    specs = [
        BandAttentionSpec(output=node.outs[0], band=window)
        for node, layout in zip(attentions, layouts, strict=True)
        if layout.layer_type == SLIDING_ATTENTION
    ]
    declared = to_band_form(graph, specs)
    pruned = sorted(set(graph.initializers) - set(declared.initializers))
    return declared, pruned


def assert_band_form(
    graph: IrGraph,
    layouts: Sequence[AttentionLayout],
    sym_max: int,
    window: int,
    pruned: Sequence[str],
    band_constant: str,
) -> dict[str, Any]:
    """Check the form that is written (after {@link declare_band}).

    Each property breaks while the numbers can still agree with the goldens:

    - an attention that still takes a mask holds a Tmax² constant the container cannot carry
    - a band on a full layer, a sliding layer without one, or another width agrees with the
      goldens at T <= 513
    - the surgery pruning anything but the traced band constant drops a value something reads
    - without a `sym_prefix_slice` of T at Tmax left, the bound of T is no longer in the graph
      (the distribution's `maxTokens` and the reader's check read it there)
    - a slice with a coefficient or an offset bounds T at another value than its extent; the
      distribution refuses such a slice, so this check refuses it too (the readers of the bound
      agree)
    """
    attentions = _attentions(graph)
    if len(attentions) != len(layouts):
        raise AssertionError(f"{len(attentions)} attention nodes for {len(layouts)} layers")
    for index, (node, layout) in enumerate(zip(attentions, layouts, strict=True)):
        where = f"attention[{index}] ({layout.layer_type})"
        if len(node.ins) != 3:
            raise AssertionError(f"{where}: {len(node.ins)} ins (no attention takes a mask)")
        expected = window if layout.layer_type == SLIDING_ATTENTION else None
        if node.attrs.get("band") != expected:
            raise AssertionError(
                f"{where}: attrs.band is {node.attrs.get('band')!r}, not {expected!r}"
            )
    if list(pruned) != [band_constant]:
        raise AssertionError(
            f"the surgery pruned {list(pruned)}, not the traced band constant '{band_constant}'"
        )
    bounds: set[int] = set()
    for node in graph.nodes:
        if node.op != SYM_PREFIX_SLICE_OP or node.attrs.get("sym") != graph.symbols[0]:
            continue
        shape = declared_shape(graph, node.ins[0])
        for cut in node.attrs["slices"]:
            if cut.get("coeff") != 1 or cut.get("offset") != 0:
                raise AssertionError(
                    f"a sym_prefix_slice of {node.ins[0]} cuts {cut} — the bound of T is read"
                    " only from a slice with coeff 1 and offset 0"
                )
            bounds.add(int(shape[cut["dim"]]))
    if bounds != {sym_max}:
        raise AssertionError(
            f"the folded constants left bound T at {sorted(bounds)}, not exactly {sym_max}"
        )
    return {
        "band_layers": [
            index for index, node in enumerate(attentions) if node.attrs.get("band") is not None
        ],
        "band": window,
        "pruned": list(pruned),
        "bound": sym_max,
    }


@contextmanager
def upstream_attention(model: nn.Module) -> Iterator[None]:
    """Run `model` on upstream's own attention implementation inside the block (restored after)."""
    config = model.config
    previous = config._attn_implementation
    config._attn_implementation = UPSTREAM_ATTENTION
    try:
        yield
    finally:
        config._attn_implementation = previous


def _max_abs(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left - right).abs().max())


def window_evidence(
    wrapper: EmbeddingWrapper, cases: Sequence[tuple[str, torch.Tensor, torch.Tensor]]
) -> dict[str, dict[str, float]]:
    """On every case whose T cuts the window: the band really changes the output, and the wrapper
    equals upstream's own forward.

    - `band_vs_all_visible` — the wrapper's output against the same model with every position
      visible on every layer; MUST exceed {@link WINDOW_FLOOR} (a graph that lost the window would
      pass the goldens only if the window did nothing).
    - `upstream` / `upstream_hidden_rel` — the wrapper's output (max abs) and per-token hidden
      states (max abs relative to their largest magnitude) against upstream's forward with
      `attention_mask` of ones (upstream's masking utilities build the masks, upstream's SDPA
      integration runs); MUST stay within {@link UPSTREAM_TOLERANCE}.

    The measured values are at the constants above.
    """
    model = wrapper.model
    record: dict[str, dict[str, float]] = {}
    for name, ids, pool_mask in cases:
        length = int(ids.shape[1])
        if length < wrapper.window + 2:
            continue
        with torch.no_grad():
            banded, hidden = embed(model, ids, pool_mask, layer_masks(length, wrapper.window))
            visible, _ = embed(model, ids, pool_mask, all_visible_masks())
            with upstream_attention(model):
                upstream, upstream_hidden = embed(model, ids, pool_mask, torch.ones_like(ids))
            exported = wrapper.text(ids, pool_mask)
        measured = {
            "band_vs_all_visible": _max_abs(banded, visible),
            "upstream": _max_abs(banded, upstream),
            "upstream_hidden_rel": _max_abs(hidden, upstream_hidden)
            / float(upstream_hidden.abs().max()),
        }
        # MUST: equal, not close — no soft row means the merge selects the scaled lookup exactly,
        # which is what keeps the text goldens' values (module doc, "Soft tokens from a tower").
        if not torch.equal(exported, banded):
            raise AssertionError(
                f"{name}: the wrapper's forward without soft rows is not the input_ids band path"
                " bit for bit"
            )
        if measured["band_vs_all_visible"] <= WINDOW_FLOOR:
            raise AssertionError(
                f"{name}: the band moves the output by {measured['band_vs_all_visible']:.3e}"
                f" <= {WINDOW_FLOOR} — the window is not exercised"
            )
        if max(measured["upstream"], measured["upstream_hidden_rel"]) > UPSTREAM_TOLERANCE:
            raise AssertionError(
                f"{name}: the wrapper differs from upstream's own forward by {measured}"
                f" (> {UPSTREAM_TOLERANCE})"
            )
        record[name] = measured
    if not record:
        raise AssertionError("no case cuts the window — the evidence is vacuous")
    return record


def write_io(
    wrapper: EmbeddingWrapper,
    graph: IrGraph,
    cases: Sequence[tuple[str, torch.Tensor, torch.Tensor]],
    out_dir: Path,
) -> tuple[list[str], dict[str, torch.Tensor]]:
    """Write each case's four inputs (no soft row — {@link no_soft_inputs}) and the torch CPU
    expected output to `io.<case>.safetensors`; returns the file names and the outputs.

    MUST: one tensor per graph input, lowered to the IR's semantic dtype at the boundary (i64 →
    i32, bool → u32 — ADR 0009), so the series e2e binds the golden's inputs as they are.
    """
    if len(graph.outputs) != 1:
        raise AssertionError(f"{len(graph.outputs)} IR outputs (an embedding has 1)")
    hidden_size = int(wrapper.model.config.hidden_size)
    written: list[str] = []
    embeddings: dict[str, torch.Tensor] = {}
    for name, ids, pool_mask in cases:
        soft_embeds, soft_mask = no_soft_inputs(ids, hidden_size)
        with torch.no_grad():
            output = wrapper(ids, pool_mask, soft_embeds, soft_mask)
        args = {
            GRAPH_INPUTS[0]: ids,
            GRAPH_INPUTS[1]: pool_mask,
            SOFT_EMBEDS: soft_embeds,
            SOFT_MASK: soft_mask,
        }
        tensors = {
            f"{INPUT_PREFIX}{declared.name}": normalize_boundary_tensor(
                args[declared.name], f"{name}: input '{declared.name}'"
            )
            for declared in graph.inputs
        }
        tensors[f"{OUTPUT_PREFIX}0"] = normalize_boundary_tensor(
            output.detach().contiguous(), f"{name}: output 0"
        )
        path = out_dir / f"{IO_PREFIX}{name}{IO_SUFFIX}"
        save_file(tensors, str(path))
        written.append(path.name)
        embeddings[name] = output.detach()
    return written, embeddings


def _sanity(embeddings: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    """The outputs are unit vectors and the near pair beats the far pair in cosine.

    MUST: fail when the order is reversed — the norm alone cannot detect a broken embedding whose
    normalisation still works (mixed-up layers, pooling on the wrong axis).
    """
    vectors = {name: output.reshape(-1) for name, output in embeddings.items()}
    norms = {name: float(vector.norm()) for name, vector in vectors.items()}
    off = [name for name, norm in norms.items() if abs(norm - 1.0) > 1e-5]
    if off:
        raise AssertionError(f"L2 norm is not 1 for the cases: {[(n, norms[n]) for n in off]}")

    def cosine(pair: tuple[str, str]) -> float:
        return float(torch.dot(vectors[pair[0]], vectors[pair[1]]))

    near, far = cosine(NEAR_PAIR), cosine(FAR_PAIR)
    if near <= far:
        raise AssertionError(
            f"the cosine order contradicts the meaning: {NEAR_PAIR}={near:.4f}"
            f" <= {FAR_PAIR}={far:.4f}"
        )
    return {
        "l2_norms": {name: round(norm, 7) for name, norm in norms.items()},
        "cosine": {
            f"{NEAR_PAIR[0]}×{NEAR_PAIR[1]}": round(near, 4),
            f"{FAR_PAIR[0]}×{FAR_PAIR[1]}": round(far, 4),
            "query-en×query-ja": round(cosine(("query-en", "query-ja")), 4),
        },
    }


def round_weights_to_i8(wrapper: nn.Module) -> Int8Report:
    """Round, in place, the weights the bf16 series stores as bf16 — every `nn.Linear` weight and
    the token table — to per-row symmetric int8, and return the scale ledger.

    The ledger is keyed by the wrapper's FQNs (`model.<module>.weight`), the exported tensor keys
    the writer looks each scale up by.

    MUST: called before the reference, the window evidence and the goldens are taken (module doc,
    "Storage") — the goldens are then the quantized model's outputs. MUST: the rounding reaches
    exactly those weights, each once: a weight left out ships its f32 value under the int8 preset's
    name, and a weight of another module type rounded changes a value every other series keeps.
    The target set is decided before anything is rounded, so a refusal leaves the weights as they
    were.
    """
    tables = sorted(
        f"{name}.weight"
        for name, module in wrapper.named_modules()
        if isinstance(module, nn.Embedding)
    )
    if tables != [f"model.{TABLE_MODULE}.weight"]:
        raise EmbeddingGemma2ExportError(
            f"the embedding tables are {tables}, not the token table alone"
        )
    expected = {
        f"{name}.weight"
        for name, module in wrapper.named_modules()
        if isinstance(module, nn.Linear | nn.Embedding)
    }
    quantizable = {fqn for fqn, _weight, _axis in iter_quant_targets(wrapper, QUANT_MODULE_TYPES)}
    if quantizable != expected:
        missing = sorted(expected - quantizable)[:3]
        extra = sorted(quantizable - expected)[:3]
        raise EmbeddingGemma2ExportError(
            f"the int8 rounding would reach {len(quantizable)} weights, not the {len(expected)}"
            f" linears and the token table (missing {missing} / extra {extra})"
        )
    report = fake_quant_int8(wrapper, lambda name: f"{name}.weight" in expected)
    if set(report.scales) != expected:
        missing = sorted(expected - set(report.scales))[:3]
        extra = sorted(set(report.scales) - expected)[:3]
        raise EmbeddingGemma2ExportError(
            f"the int8 rounding reached {len(report.scales)} weights, not the {len(expected)}"
            f" linears and the token table (missing {missing} / extra {extra})"
        )
    return report


def assert_storage_form(
    graph: IrGraph, dtype: str, rounded: Collection[str] = ()
) -> dict[str, int]:
    """Check the storage the writer committed (the graph `publish_model` returns) and count it.

    `f32`: every initializer is f32 (or i32). `bf16` / `i8`: every initializer in a weight slot is
    stored as that dtype and every other one f32 / i32 — a weight slot left f32 would ship two
    (four) times its bytes without a word, and a weight in another compressed storage would not
    hold the values the series' goldens were taken with.

    `rounded` is the tensor keys the int8 rounding reached ({@link round_weights_to_i8}); `i8`
    needs it and the other storages take none. MUST: under `i8` the weight slots are exactly those
    keys — a rounded weight outside a weight slot would ship its rounded values as f32, so the
    preset would hold rounding it does not store.
    """
    if dtype not in WEIGHT_DTYPES:
        raise EmbeddingGemma2ExportError(f"storage {dtype!r} is not one of {WEIGHT_DTYPES}")
    if (dtype == "i8") != bool(rounded):
        raise EmbeddingGemma2ExportError(
            f"the {dtype} storage form takes {'the' if dtype == 'i8' else 'no'} rounded weights"
            f" ({len(rounded)} given)"
        )
    eligible = eligible_compressed_initializers(graph)
    counts: dict[str, int] = {}
    wrong: list[str] = []
    for name, initializer in graph.initializers.items():
        stored = initializer.storage.dtype
        counts[stored] = counts.get(stored, 0) + 1
        expected = {dtype} if dtype != "f32" and name in eligible else {"f32", "i32"}
        if stored not in expected:
            wrong.append(f"{name}: {stored} (expected {' / '.join(sorted(expected))})")
    if wrong:
        raise AssertionError(
            f"the {dtype} storage form does not hold for {len(wrong)} initializers: {wrong[:4]}"
        )
    if dtype != "f32" and counts.get(dtype, 0) == 0:
        raise AssertionError(f"no initializer is stored as {dtype} (the graph has no weight slot)")
    if dtype == "i8":
        slots = {graph.initializers[name].tensor for name in eligible}
        if slots != set(rounded):
            unrounded = sorted(slots - set(rounded))[:3]
            outside = sorted(set(rounded) - slots)[:3]
            raise AssertionError(
                f"the i8 weight slots are not the {len(rounded)} rounded weights (in a slot but"
                f" not rounded {unrounded} / rounded but outside a slot {outside})"
            )
    return dict(sorted(counts.items()))


def export_series(
    directory: Path, out_dir: Path, *, dtype: str = "f32", sym_max: int = SYM_MAX
) -> dict[str, Any]:
    """Write the IR container and the golden io, and return a summary.

    MUST: the artifacts are written to a staging directory and put in place only after every check
    (the traced form, the band against upstream, the window evidence, the band form, the sanity)
    passes — the goldens are taken from the same wrapper, so a broken run placed at the final
    location would turn the TS-side comparison green ({@link karume.artifacts.staged_publication}).
    """
    if dtype not in WEIGHT_DTYPES:
        raise EmbeddingGemma2ExportError(f"storage {dtype!r} is not one of {WEIGHT_DTYPES}")
    sources.assert_pinned(directory, READ_FILES)
    assert_sentence_transformer_layout(directory)
    model, composite = load_text_model(directory)
    config = model.config
    window = int(config.sliding_window)
    wrapper = EmbeddingWrapper(model, window).eval()
    # MUST: i8 rounds before anything below reads the weights (the window evidence, the trace, the
    # goldens — module doc, "Storage").
    rounding = round_weights_to_i8(wrapper) if dtype == "i8" else None
    cases = build_cases(directory, composite, sym_max)
    assert_case_coverage(cases, sym_max, window)
    assert_band_matches_upstream(sym_max, window)
    evidence = window_evidence(wrapper, cases)
    out_dir.parent.mkdir(parents=True, exist_ok=True)

    # The example input is the longest case (at the bound — far from the 0/1 specialisation). min=2
    # avoids the 0/1 specialisation; max is the folding's evaluation point itself (ADR 0010).
    _, example_ids, example_mask = max(cases, key=lambda case: case[1].shape[1])
    graph, tensors = trace(wrapper, example_ids, example_mask, sym_max)
    layouts = attention_layouts(config)
    # MUST: the form is checked before anything is written (the numbers do not catch it).
    form = assert_ir_form(graph, tensors, layouts, sym_max, window, int(config.embedding_dim))
    merge = assert_soft_merge(graph, int(config.hidden_size))
    graph, pruned = declare_band(graph, layouts, window)
    band_form = assert_band_form(graph, layouts, sym_max, window, pruned, form["band_constant"])
    # The tensors of the initializers the surgery pruned are not stored (the contract of the
    # core's golden rewrite and of the states surgery's recipes).
    declared = {initializer.tensor for initializer in graph.initializers.values()}
    tensors = {key: value for key, value in tensors.items() if key in declared}
    with staged_publication(out_dir) as staged:
        # The writer creates the directory (the primitive only hands over a path).
        staged.mkdir()
        graph = publish_model(
            staged / MODEL_FILE,
            graph,
            tensors,
            provenance=Provenance(license=LICENSE, upstream_revision=sources.PIN.revision),
            # The graph name is the component name (the first recipe's convention for a
            # single-component series), not the directory name.
            graph_name=GRAPH_NAME,
            # bf16: nothing to round first — the weights are the checkpoint's bf16 values, and
            # the writer fails on one that bf16 cannot hold. i8: the weights were rounded above,
            # and each is stored with the scale its rounding used (module doc, "Storage").
            weight_dtype=dtype,
            weight_scales=None if rounding is None else rounding.scales,
        )
        # MUST: checked before publishing (on failure the staging directory is removed).
        storage = assert_storage_form(
            graph, dtype, () if rounding is None else tuple(rounding.scales)
        )
        written, embeddings = write_io(wrapper, graph, cases, staged)
        # MUST: evaluated before publishing (on failure the staging directory is removed).
        sanity = _sanity(embeddings)
    return {
        "dir": str(out_dir),
        "nodes": len(graph.nodes),
        "outputs": len(graph.outputs),
        "initializers": len(graph.initializers),
        "parts": len(container_parts(out_dir / MODEL_FILE)),
        "model_bytes": sum(p.stat().st_size for p in container_parts(out_dir / MODEL_FILE)),
        "ops": sorted(graph.required_ops),
        "symbols": list(graph.symbols),
        "storage": storage,
        **(
            {}
            if rounding is None
            else {"rounded": {"tensors": rounding.modules, "elements": rounding.elements}}
        ),
        "form": form,
        "soft_merge": merge,
        "band_form": band_form,
        "window_evidence": evidence,
        "io": written,
        "case_lengths": {name: int(ids.shape[1]) for name, ids, _ in cases},
        "sanity": sanity,
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--snapshot-dir", type=Path, default=sources.SNAPSHOT_DIR)
    parser.add_argument(
        "--dtype",
        choices=WEIGHT_DTYPES,
        default="f32",
        help="storage dtype of the weights (bf16 = the checkpoint's own values in every weight"
        " slot; i8 = the same weights rounded to per-row int8 before the goldens are taken — the"
        " distribution's two storages)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output directory (default: the series of the --dtype —"
        f" {' / '.join(str(path) for path in DEFAULT_OUT_DIRS.values())})",
    )
    args = parser.parse_args(argv)
    out_dir = args.out if args.out is not None else DEFAULT_OUT_DIRS[args.dtype]
    summary = export_series(args.snapshot_dir, out_dir, dtype=args.dtype)
    write_golden_case_fixture()
    print(
        json.dumps(
            {"snapshot_dir": str(args.snapshot_dir), "dtype": args.dtype, **summary},
            indent=1,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
