"""Script that writes the real-weight EmbeddingGemma-300m (sentence embeddings) to an IR v2
container + golden io.

It plays the same role as `deberta/export.py` (numerical agreement on real weights and real token
sequences) for a SentenceTransformer-style model with a **single vector output**. The artifacts go
under `outputs/series/` and are kept out of commits by the repository root's `.gitignore` (the
weights are about 1.2 GB).

    uv run --with 'transformers==5.14.1' python -m embeddinggemma.export

transformers is **pinned to 5.14.1** (for the same reason as `deberta/export.py` — a change in the
modeling code changes the graph's shape). It is not added to pyproject.toml / uv.lock but added
temporarily with `--with`.

## What goes into the graph

Of the 5 SentenceTransformer modules (modules.json), **all of stages 0–4** go into one graph:

    Transformer(Gemma3TextModel) → Pooling(masked mean) → Dense(768→3072) → Dense(3072→768)
    → Normalize(L2)

The output is one unit vector `[1, 768]`. Adding the prompt prefix and tokenizing happen on the
host (using `prompts` of `config_sentence_transformers.json` verbatim).

MUST: call `Gemma3TextModel` **without passing `attention_mask`**. The config has
`use_bidirectional_attention=true`, so without a mask the two kinds of band mask (sliding window
and full) are folded as **constants independent of T** (the first T are cut out with
`sym_prefix_slice` — ADR 0010), and no mask input remains in the IR. To pass a sequence that
contains padding, the caller packs the sequence and shortens T (this form is exactly equivalent to
torch eager).

MUST: **preserve** SDPA (`PRESERVED_OP_PREFIXES_WITH_ATTENTION` — the fused attention of ADR 0023).
This is a script-local setting and does not go into the default decomposition table (ADR 0023
decision 5). The bool band mask Gemma3 passes is lowered by `normalize._additive_attn_mask` to an
additive f32 `[1,1,T,T]`, and constant folding turns it into a Tmax constant + `sym_prefix_slice`.
**No path** down to the decomposed route is left — if preservation fails, the whole export failing
is the correct outcome (it does not silently switch to another numerical path). The resulting IR
form (the attention count, the source of the band masks, the 2 inputs) is checked by
{@link assert_ir_form} before publishing.

MUST: avoid writing anything that falls outside the vocabulary. `torch.mean(dim=)` /
`keepdim=True` / `x**2` / `linalg_vector_norm` each produce an op the IR vocabulary does not have
(`mean.dim` / reduce with keepdim / `pow` / `linalg_vector_norm`), so write with **mul / sum(dim) /
reshape / sqrt / div / clamp_min** only (the implementations of `_masked_mean` and `_l2_normalize`
have that form).

NOTE: `pool_mask` is taken as all ones (since the model is called without a mask, mixing in zeros
would produce a form that exists only on the eager side: "the model sees the token but the pooling
drops it"). It stays an input to leave the runtime an escape hatch for when the host cannot pack the
sequence.

## Output layout

    outputs/series/embeddinggemma-300m/model-NNNNN-of-NNNNN.krm weights / constants + 2 descriptors
    outputs/series/embeddinggemma-300m/io.<case>.safetensors    inputs + expected torch CPU outputs

The io tensor key convention is the same as for tiny golden / DeBERTa (`input.<graph input name>` /
`output.<position>`).

## batch variant (`--batch`)

The default is `--batch 1` (exactly the earlier behaviour described at the top of this section).
`--batch N` (N>1) fixes batch as a **static dimension** of torch.export and writes only a single
golden case (`io.batchN.safetensors`) with `query-en` copied into N rows — an asset to settle the
hypothesis that the GPU time of linear is bound by skinny-M (M=T) by making M = batch×T large (give
the location explicitly with `--out`).

NOTE: N>1 currently fails loudly in the core's conversion stage (`karume/convert.py`)
("EmbeddingGemma's batch>1 export does not pass the conversion stage" in `docs/known-issues.md`).
The flag is kept for the day the conversion side is generalised.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file
from torch import nn
from torch.export import Dim

from _shared.paths import INPUTS_ROOT, SERIES_ROOT
from _shared.upstream import snapshot_revision
from karume.artifacts import staged_publication
from karume.container import Provenance, container_parts
from karume.convert import PRESERVED_OP_PREFIXES_WITH_ATTENTION, normalize_boundary_tensor
from karume.ir import IrGraph
from karume.ops import ATTENTION_OP, SYM_PREFIX_SLICE_OP
from karume.pipeline import export_to_file
from karume.rope import assert_rope_lifted
from karume.shapes import declared_shape

#: Where the official weights live (the target of `hf download google/embeddinggemma-300m`).
DEFAULT_MODEL_DIR = INPUTS_ROOT / "embeddinggemma" / "google-300m"

#: The default location of the artifacts. The storage dtype is f32 only (f16 / i8 are for another
#: series to decide).
DEFAULT_OUT_DIR = SERIES_ROOT / "embeddinggemma-300m"

MODEL_FILE = "model.krm"

#: The container's graph name = **the component name of the distribution** (= the key in `weights`
#: of `karume.json` — container-v1 §2.1). This recipe does not assemble a distribution (there is no
#: `distribution.py`), but the graph name is the spelling the runtime looks up with
#: `prepareContainer(opened, <weights key>)`, so it takes **the convention of a single-component
#: series** (`model`) so that the container in place can be used as is on the day one is assembled.
#: It does not match the series directory name (`embeddinggemma-300m`).
GRAPH_NAME = "model"

#: The license identifier baked into the container (container-v1 §2.3). This recipe does not
#: assemble a distribution (neither a card nor `distribution.py`), so its source is the upstream
#: model card's declaration itself, the value `THIRD_PARTY_NOTICES.md` records as "not confirmed
#: for the revision used".
LICENSE = "gemma"


def upstream_provenance(model_dir: Path) -> Provenance:
    """The provenance baked into the container. The revision is derived from the download record of
    `--model-dir`.

    Not hard-coded ({@link _shared.upstream.snapshot_revision}) — with a constant, the container
    would claim the same revision even when `--model-dir` points at another snapshot. The series
    migration (`migrate_series`) calls the same function.
    """
    return Provenance(license=LICENSE, upstream_revision=snapshot_revision(model_dir))


#: The weights file name of the **upstream checkpoint** (HF's spelling — not the output container).
CHECKPOINT_FILE = "model.safetensors"
IO_PREFIX = "io."
IO_SUFFIX = ".safetensors"
INPUT_PREFIX = "input."
OUTPUT_PREFIX = "output."

#: The two trailing SentenceTransformer stages (2_Dense / 3_Dense of modules.json).
DENSE_DIRS = ("2_Dense", "3_Dense")
#: The only tensor key in a Dense safetensors (one `nn.Linear` without bias).
DENSE_WEIGHT_KEY = "linear.weight"
#: The Dense activation (sentence-transformers' spelling). The wrapper folds a Dense as a plain
#: linear.
DENSE_ACTIVATION = "torch.nn.modules.linear.Identity"
#: The type column of `modules.json` — exactly the 5 stages the wrapper ({@link EmbeddingWrapper})
#: folds into one.
SENTENCE_TRANSFORMER_MODULES = (
    "sentence_transformers.models.Transformer",
    "sentence_transformers.models.Pooling",
    "sentence_transformers.models.Dense",
    "sentence_transformers.models.Dense",
    "sentence_transformers.models.Normalize",
)
#: Where the Pooling config lives (1_Pooling of modules.json).
POOLING_DIR = "1_Pooling"
#: The only pooling-mode field that may be true (the wrapper hard-codes masked mean).
POOLING_MODE = "pooling_mode_mean_tokens"

#: Upper bound of the symbolic dimension T. Equal to sliding_window (512) — it is the folding's
#: evaluation point itself, so raising it grows the band mask constants by Tmax² (ADR 0010).
SYM_MAX = 512

#: The graph input names (the wrapper's forward argument names). {@link assert_ir_form} checks for
#: "these 2 only" — a band mask left as an input instead of being folded adds a third.
GRAPH_INPUTS = ("input_ids", "pool_mask")

#: Lower bound of the masked-mean denominator (same value as sentence-transformers' Pooling).
POOL_EPS = 1e-9
#: Lower bound of the L2 normalisation denominator (same value as sentence-transformers' Normalize).
NORM_EPS = 1e-12

#: A long text of about 300 tokens (the body of the `long-document` case). Shows that the prefix
#: slice of the Tmax folding works at the real length, at a T far enough from the short cases.
LONG_PASSAGE = (
    "Retrieval-augmented generation pipelines split a corpus into passages, embed each passage "
    "once, and keep the resulting vectors in an approximate nearest neighbour index. "
    "At query time the same encoder turns the user question into a vector, the index returns "
    "the closest passages, and a language model reads them before answering. "
    "The quality of such a system depends less on the generator than on the encoder: if the "
    "passage that actually answers the question is not in the top results, no amount of "
    "prompting will recover it. "
    "Running the encoder in the browser changes the trade-offs. There is no server to batch "
    "requests against, the weights have to be downloaded once and cached, and the compute "
    "budget is whatever the local GPU can spare between frames. "
    "A 300M parameter encoder in float32 is roughly 1.2 gigabytes, which is too much to fetch "
    "on every visit but perfectly reasonable to keep in a persistent cache. "
    "Quantising the weights to eight bits cuts that by a factor of four and, for retrieval, "
    "the loss in ranking quality is usually smaller than the loss from a mediocre chunking "
    "strategy. "
    "The interesting engineering problem is therefore not accuracy but scheduling: how to keep "
    "the embedding work off the main thread, how to overlap the download with the first few "
    "queries, and how to avoid re-encoding passages that have not changed since the last visit. "
    "Measured on a mid-range discrete GPU a single 512-token batch takes a few milliseconds "
    "once the pipelines are warm, so the practical ceiling is set by memory bandwidth rather "
    "than by arithmetic."
)

#: The fixed golden texts (`(case name, prompt kind | None, body)`). The prompt strings themselves
#: are taken **verbatim** from `prompts` of `config_sentence_transformers.json` (no copy is kept).
#: The order includes a near pair (query-en × document-en) and a far pair (query-en × bare) so that
#: generation can check that the cosine order agrees with the meaning.
GOLDEN_CASES: tuple[tuple[str, str | None, str], ...] = (
    ("query-en", "query", "what is the capital of France?"),
    ("document-en", "document", "Paris is the capital and most populous city of France."),
    ("query-ja", "query", "フランスの首都はどこですか。"),
    ("bare", None, "The quick brown fox jumps over the lazy dog."),
    ("long-document", "document", LONG_PASSAGE),
)

#: The cosine pairs the sanity record checks (near pair, far pair). If the near pair does not beat
#: the far pair, "the numbers move but it is broken as an embedding" — generation fails loudly.
NEAR_PAIR = ("query-en", "document-en")
FAR_PAIR = ("query-en", "bare")


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


class EmbeddingWrapper(nn.Module):
    """Export wrapper that folds the 5 SentenceTransformer stages into one forward.

    MUST: do not pass `attention_mask` (the MUST of the module docstring). Passing it makes the
    additive bias depend on T, and the folding of the band masks (`sym_prefix_slice`) no longer
    holds.
    """

    def __init__(self, model: nn.Module, dense2: nn.Linear, dense3: nn.Linear) -> None:
        super().__init__()
        self.model = model
        self.dense2 = dense2
        self.dense3 = dense3

    def forward(self, input_ids: torch.Tensor, pool_mask: torch.Tensor) -> torch.Tensor:
        hidden = self.model(input_ids=input_ids, use_cache=False).last_hidden_state
        pooled = _masked_mean(hidden, pool_mask)
        return _l2_normalize(self.dense3(self.dense2(pooled)))


def load_dense(model_dir: Path, name: str) -> nn.Linear:
    """Build a bias-free linear from the in/out of `<name>/config.json` and
    `<name>/model.safetensors`.

    MUST: the shape is read from the config and the weight from the safetensors — inferring from
    only one of them lets "it loaded, but it is a different layer" pass silently when upstream
    changes the Dense dimensions.
    """
    config = json.loads((model_dir / name / "config.json").read_text(encoding="utf-8"))
    if config["bias"]:
        raise ValueError(
            f"{name}: a Dense with bias is unexpected (the official release has bias=false)"
        )
    # Also fail when the field is missing (sentence-transformers' Dense applies Tanh when omitted).
    if config.get("activation_function") != DENSE_ACTIVATION:
        raise ValueError(
            f"{name}: activation_function {config.get('activation_function')!r} is unexpected"
            f" (the wrapper folds it as {DENSE_ACTIVATION})"
        )
    dense = nn.Linear(config["in_features"], config["out_features"], bias=False)
    weight = load_file(str(model_dir / name / CHECKPOINT_FILE))[DENSE_WEIGHT_KEY]
    if tuple(weight.shape) != tuple(dense.weight.shape):
        raise ValueError(
            f"{name}: weight shape {tuple(weight.shape)} disagrees with"
            f" {tuple(dense.weight.shape)} given by the config's in/out"
        )
    dense.weight = nn.Parameter(weight.to(torch.float32))
    dense.eval()
    return dense


def assert_sentence_transformer_layout(model_dir: Path) -> None:
    """Check that the upstream 5-stage layout and pooling mode match what the wrapper hard-codes.

    MUST: the layout is read from the upstream declarations (`modules.json` /
    `1_Pooling/config.json`) and compared — the golden is taken from the same wrapper, so the
    TS-side comparison would stay green even if upstream switched Pooling to CLS or dropped
    Normalize.
    `include_prompt` is checked too because it pairs with the premise that `pool_mask` is taken as
    all ones (the prompt prefix is included in the mean as well).
    """
    modules = json.loads((model_dir / "modules.json").read_text(encoding="utf-8"))
    types = tuple(module["type"] for module in modules)
    if types != SENTENCE_TRANSFORMER_MODULES:
        raise ValueError(
            f"the modules.json layout {types} is not the wrapper's 5 stages"
            f" {SENTENCE_TRANSFORMER_MODULES}"
        )
    pooling = json.loads((model_dir / POOLING_DIR / "config.json").read_text(encoding="utf-8"))
    modes = sorted(
        key for key, value in pooling.items() if key.startswith("pooling_mode_") and value
    )
    if modes != [POOLING_MODE]:
        raise ValueError(f"{POOLING_DIR}: pooling mode {modes} is not {POOLING_MODE} alone")
    if pooling.get("include_prompt") is not True:
        raise ValueError(
            f"{POOLING_DIR}: include_prompt {pooling.get('include_prompt')!r} is unexpected"
            " (it does not fit the premise that pool_mask is taken as all ones)"
        )


def load_wrapper(model_dir: Path) -> EmbeddingWrapper:
    """Load the body + the 2 Dense stages and return an exportable wrapper with the RoPE buffers
    demoted."""
    # Check the layout before loading the body (do not fail after reading hundreds of MB). The
    # transformers import also comes after it — the layout rejection comes first whether or not the
    # dependency is installed (CI has no transformers).
    assert_sentence_transformer_layout(model_dir)
    from transformers import Gemma3TextModel

    model = Gemma3TextModel.from_pretrained(
        model_dir, dtype=torch.float32, attn_implementation="sdpa"
    )
    model.eval()
    # With inv_freq left as a buffer it is not a leaf of constant folding, and sin / cos stay in
    # the IR.
    assert_rope_lifted(model, "embeddinggemma")
    dense2, dense3 = (load_dense(model_dir, name) for name in DENSE_DIRS)
    return EmbeddingWrapper(model, dense2, dense3).eval()


def load_prompts(model_dir: Path) -> dict[str, str]:
    """`prompts` of `config_sentence_transformers.json` (the source of truth for the prefixes)."""
    path = model_dir / "config_sentence_transformers.json"
    return json.loads(path.read_text(encoding="utf-8"))["prompts"]


def build_cases(
    model_dir: Path, sym_max: int
) -> tuple[tuple[str, torch.Tensor, torch.Tensor], ...]:
    """`(name, input_ids, pool_mask)` of the 5 golden cases.

    Tokenization reads the official `tokenizer.json` directly with `tokenizers` (bos / eos are added
    by the post_processor of tokenizer.json — consistent with `add_bos_token` / `add_eos_token` of
    `tokenizer_config.json` both being true).
    """
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    prompts = load_prompts(model_dir)
    cases: list[tuple[str, torch.Tensor, torch.Tensor]] = []
    for name, prompt_key, body in GOLDEN_CASES:
        text = body if prompt_key is None else prompts[prompt_key] + body
        ids = torch.tensor([tokenizer.encode(text).ids], dtype=torch.int64)
        length = int(ids.shape[1])
        if not 2 <= length <= sym_max:
            raise ValueError(
                f"{name}: T={length} is outside the symbolic dimension range [2, {sym_max}]"
            )
        # Since the model is called without a mask, no token may be dropped by the pooling (module
        # docstring).
        cases.append((name, ids, torch.ones_like(ids, dtype=torch.float32)))
    return tuple(cases)


def build_batch_case(
    model_dir: Path, sym_max: int, batch: int
) -> tuple[str, torch.Tensor, torch.Tensor]:
    """A single case with `query-en` copied into `batch` rows (an asset that makes M = batch×T
    large).

    For settling the hypothesis that the GPU time of linear is bound by skinny-M (M=T=4–318, short
    of occupancy). Every row is the same text, so the torch reference output should be the same for
    every row too — `_sanity_batch` checks that the rows agree. The case is looked up in
    GOLDEN_CASES by name (not relying on a fixed index).
    """
    name, prompt_key, body = next(case for case in GOLDEN_CASES if case[0] == "query-en")
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    prompts = load_prompts(model_dir)
    text = prompts[prompt_key] + body
    ids = torch.tensor([tokenizer.encode(text).ids], dtype=torch.int64)
    length = int(ids.shape[1])
    if not 2 <= length <= sym_max:
        raise ValueError(
            f"{name}: T={length} is outside the symbolic dimension range [2, {sym_max}]"
        )
    batched_ids = ids.expand(batch, -1).contiguous()
    batched_mask = torch.ones_like(batched_ids, dtype=torch.float32)
    return f"batch{batch}", batched_ids, batched_mask


def assert_ir_form(graph: IrGraph, config: Any, sym_max: int) -> dict[str, Any]:
    """Check that the IR is "the form with SDPA preserved" and that "the band masks are
    T-independent constants".

    Both are properties that **break silently while the numbers still agree**, and the golden io
    comparison does not catch them:

    - if SDPA falls to the decomposed path, there are fewer attention ops than layers and the
      runtime cannot use the fused attention (a slow asset — the MUST of the module docstring)
    - a band mask left as a graph input becomes the form "the host builds T² and passes it every
      time" (ADR 0010)

    The mask constants are separate bands per layer type (sliding window / full), so their count is
    not constrained.
    """
    layers = int(config.num_hidden_layers)

    names = [spec.name for spec in graph.inputs]
    if names != list(GRAPH_INPUTS):
        raise AssertionError(
            f"graph inputs are {names} — not the 2 inputs {list(GRAPH_INPUTS)}"
            " (a band mask may have been left as an input instead of being folded)"
        )

    producer = {out: node for node in graph.nodes for out in node.outs}
    attentions = [node for node in graph.nodes if node.op == ATTENTION_OP]
    if len(attentions) != layers:
        raise AssertionError(
            f"{len(attentions)} attention nodes, which does not match the layer count {layers}"
            " (SDPA may have fallen to the decomposed path)"
        )

    mask_constants: set[str] = set()
    for index, node in enumerate(attentions):
        where = f"attention[{index}]"
        if len(node.ins) != 4:
            raise AssertionError(f"{where}: {len(node.ins)} ins (expected 4: q / k / v / mask)")
        source = producer.get(node.ins[3])
        if source is None or source.op != SYM_PREFIX_SLICE_OP:
            found = (
                "not a node output (a graph input or an initializer wired directly)"
                if source is None
                else source.op
            )
            raise AssertionError(
                f"{where}: the mask is supplied by {found} — not by the {SYM_PREFIX_SLICE_OP} of a"
                " Tmax constant"
            )
        constant = source.ins[0]
        if constant not in graph.initializers:
            raise AssertionError(f"{where}: the mask source '{constant}' is not an initializer")
        shape = declared_shape(graph, constant)
        if shape != [1, 1, sym_max, sym_max]:
            raise AssertionError(
                f"{where}: the mask constant's shape {shape} differs from the Tmax shape"
                f" [1, 1, {sym_max}, {sym_max}]"
            )
        mask_constants.add(constant)
    return {"attention_nodes": len(attentions), "mask_constants": sorted(mask_constants)}


def _write_io(
    wrapper: nn.Module,
    graph: IrGraph,
    cases: Sequence[tuple[str, torch.Tensor, torch.Tensor]],
    out_dir: Path,
) -> tuple[list[str], dict[str, torch.Tensor]]:
    """Write each case's inputs and torch CPU expected output to `io.<case>.safetensors`.

    The second return value is the expected outputs for the sanity record (passed in their `[B, H]`
    shape — the cross-row comparison of batch>1 needs the shape. For batch=1, `_sanity` applies
    `reshape(-1)` and handles them as before).
    """
    written: list[str] = []
    embeddings: dict[str, torch.Tensor] = {}
    if len(graph.outputs) != 1:
        raise AssertionError(f"{len(graph.outputs)} IR outputs (an embedding has 1)")
    for name, ids, pool_mask in cases:
        with torch.no_grad():
            output = wrapper(ids, pool_mask)
        args = {"input_ids": ids, "pool_mask": pool_mask}
        # MUST: io is also lowered to the actual representation of the IR's semantic dtype
        # (i64 → i32). If it does not match what the runtime receives, the Deno-side E2E cannot read
        # the golden (the boundary normalisation of ADR 0009).
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


def _sanity(embeddings: dict[str, torch.Tensor]) -> dict[str, Any]:
    """Check that the outputs are unit vectors and that the cosine of the semantically near pair
    beats the far pair (batch=1 only).

    MUST: fail when the order is reversed. The norm alone cannot detect "the normalisation works but
    the embedding is broken" (mixed-up layers, pooling on the wrong axis) — do not make the sanity
    check vacuous.
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


#: Tolerance of the cross-row comparison (the rows of a batch are the same input, so in theory they
#: agree exactly — only floating-point rounding is allowed).
BATCH_ROW_ATOL = 1e-4


def _sanity_batch(output: torch.Tensor) -> dict[str, Any]:
    """Sanity of the batch variant (batch>1 only): every row is a unit vector and agrees with the
    row it was copied from.

    MUST: do not miss a disagreement between rows — every row is a copy of the same input, so a
    disagreement means "the weights are right but the handling of the batch axis is broken" (a
    non-vacuous check for the batch variant, the counterpart of the cosine-order check of
    `_sanity`).
    """
    norms = output.norm(dim=-1)
    off = (norms - 1.0).abs() > 1e-5
    if bool(off.any()):
        raise AssertionError(f"L2 norm is not 1 for some rows: {norms[off].tolist()}")

    max_row_diff = float((output - output[0]).abs().max())
    if max_row_diff > BATCH_ROW_ATOL:
        raise AssertionError(
            f"the rows of the output disagree (max abs diff {max_row_diff} > {BATCH_ROW_ATOL})"
            " — the handling of the batch axis is broken"
        )
    return {
        "rows": int(output.shape[0]),
        "l2_norms_min_max": (round(float(norms.min()), 7), round(float(norms.max()), 7)),
        "row_max_abs_diff": round(max_row_diff, 9),
    }


def export_series(
    model_dir: Path, out_dir: Path, *, sym_max: int = SYM_MAX, batch: int = 1
) -> dict[str, Any]:
    """Write the IR container and the golden io, and return a summary.

    `batch` is a **static dimension** of torch.export (default 1 = all 5 cases, as before). T stays
    a dynamic dimension as before. When `batch > 1`, only a single case with `query-en` copied into
    `batch` rows becomes the golden (an asset for testing the occupancy-shortage hypothesis of
    linear).

    MUST: the artifacts are written to a staging directory and put in place only after passing
    **every check** (sanity). Placing them at the final location before the checks would let a
    failed run leave "an asset that can pass the acceptance check" — the io golden is taken from
    the same broken wrapper, so the two agree and the TS-side comparison **turns green** (the
    spelling of "when it may be published" is {@link _shared.decode_series._publish}; the discipline
    of swapping in and cleaning up is the core primitive
    {@link karume.artifacts.staged_publication}).
    """
    # Read the provenance before the weights (a checkpoint without a download record fails before
    # its weights are read).
    provenance = upstream_provenance(model_dir)
    wrapper = load_wrapper(model_dir)
    cases = (
        build_cases(model_dir, sym_max)
        if batch == 1
        else (build_batch_case(model_dir, sym_max, batch),)
    )
    out_dir.parent.mkdir(parents=True, exist_ok=True)

    # The example input is the longest case (the closer T is to the upper bound, the farther from
    # 0/1 specialisation). min=2 avoids 0/1 specialisation; max is the evaluation point of the Tmax
    # folding itself (ADR 0010 — not managed twice through a separate knob).
    _, example_ids, example_mask = max(cases, key=lambda case: case[1].shape[1])
    seq = Dim("T", min=2, max=sym_max)
    with staged_publication(out_dir) as staged:
        # The writer creates the directory (the primitive does not create it — it only hands over
        # a path).
        staged.mkdir()
        graph = export_to_file(
            wrapper,
            (example_ids, example_mask),
            staged / MODEL_FILE,
            provenance=provenance,
            # The graph name is **the component name** ({@link GRAPH_NAME}) — not derived from the
            # directory name.
            graph_name=GRAPH_NAME,
            dynamic_shapes=({1: seq}, {1: seq}),
            preserved=PRESERVED_OP_PREFIXES_WITH_ATTENTION,
        )
        # MUST: check the form before publishing (SDPA preservation and band mask constants — the
        # numerical comparison does not catch them).
        form = assert_ir_form(graph, wrapper.model.config, sym_max)
        written, embeddings = _write_io(wrapper, graph, cases, staged)
        # MUST: evaluate before publishing (the only non-vacuous check of this series — on failure
        # the whole staging directory is removed).
        sanity = (
            _sanity(embeddings) if batch == 1 else _sanity_batch(next(iter(embeddings.values())))
        )
    return {
        "dir": str(out_dir),
        "nodes": len(graph.nodes),
        "outputs": len(graph.outputs),
        "initializers": len(graph.initializers),
        "model_bytes": sum(p.stat().st_size for p in container_parts(out_dir / MODEL_FILE)),
        "ops": sorted(graph.required_ops),
        "symbols": list(graph.symbols),
        "form": form,
        "io": written,
        "case_lengths": {name: int(ids.shape[1]) for name, ids, _ in cases},
        "batch": batch,
        "sanity": sanity,
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--sym-max", type=int, default=SYM_MAX)
    parser.add_argument(
        "--batch",
        type=int,
        default=1,
        help="the static batch dimension of torch.export (default 1 = as before). Above 1, only a"
        " single golden case with query-en copied into batch rows is written.",
    )
    args = parser.parse_args(argv)
    if args.batch < 1:
        raise SystemExit(f"--batch must be at least 1 (got {args.batch})")
    summary = export_series(args.model_dir, args.out, sym_max=args.sym_max, batch=args.batch)
    print(json.dumps({"model_dir": str(args.model_dir), **summary}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
