"""Screen quantized weight forms for the text tower of EmbeddingGemma 2 on torch CPU (stage A of its
quant presets).

**No GPU code** — the weights are rounded in place with the exporter's own fake-quant functions,
so they are the values a series would ship, and the embeddings are taken from the export's own
wrapper ({@link embeddinggemma2.export.EmbeddingWrapper}). The reference is the same wrapper with
the checkpoint's own weights (bf16 values upcast to f32, run in f32 — the goldens' arithmetic).

    uv run --offline --with 'transformers==5.23.0' python -m embeddinggemma2.measure_quant

## Configurations

| name                 | linear weights (218) | token table (1) | activations                     |
| -------------------- | -------------------- | --------------- | ------------------------------- |
| `i8-all`             | i8 per row           | i8 per row      | f32                             |
| `i8-linear`          | i8 per row           | untouched       | f32                             |
| `i4-linear+i8-table` | i4 group 32 (RTN)    | i8 per row      | f32                             |
| `i4-all`             | i4 group 32 (RTN)    | i4 group 32     | f32                             |
| `i8-all+a8`          | i8 per row           | i8 per row      | every linear input per-token i8 |

The weight set is the one the bf16 series stores as bf16 (the token table and every linear: 219
tensors, 270,925,824 elements — counted from the model and checked by {@link assert_rounded}). i8 is
`karume.quantize.fake_quant_int8` (symmetric, one F32 scale per output row / vocabulary row), i4 is
`karume.quantize.fake_quant_int4` (symmetric, one F32 scale per 32 elements along the in axis). The
a8 simulation is `karume.act_quant.attach_act_quant` — the torch mirror of the runtime's
`quantize_rows` (per-token symmetric int8 of each linear's input, scale `amax · f32(1/127)`), the
same simulation the umT5 a8 investigation ran on CPU; the count it attaches must be every linear.

MUST: configurations do not stack — every rounding is in place, so each starts from the pristine
weights ({@link restore}), and a final restore must give the reference back bit for bit.

## Texts

The 10 golden cases of the export (with their prompts; T 12 to 8,192) and the 48 texts of the
calibration corpus ({@link embeddinggemma2.calib_texts.CALIB_TEXTS}) with the prompt
kind each one carries there (`query` / `document` / none), prefixed with EmbeddingGemma 2's own
prompts of the same names. The corpus interleaves 20 query → document pairs (a query followed by
its document — {@link query_document_pairs}) and 8 bare texts.

## Metrics (each at the widths 768 / 512 / 256 / 128 — the leading slice, re-normalised)

1. cosine of each text's embedding against the reference: min and mean over all texts, the short
   ones (T < {@link LONG_FROM}) and the long ones;
2. retrieval over the 20 pairs (20 queries × 20 documents by cosine): how many queries rank their
   own document first (the reference too), how many top-1 documents agree with the reference's, and
   the rank inversion rate — the share of (query, document i, document j) triples whose order
   differs from the reference's;
3. the largest absolute change of any entry of the text × text cosine matrix;
4. the projected container payload ({@link projected_bits} plus the f32 remainder).

All numbers go to `<out>/report.json`; the tables go to stdout, progress to stderr.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import torch
from torch import nn

from _shared.paths import BENCH_ROOT
from embeddinggemma2 import export, sources
from embeddinggemma2.calib_texts import CALIB_TEXTS
from karume.act_quant import attach_act_quant, detach_act_quant
from karume.quantize import (
    DEFAULT_GROUP_SIZE,
    QUANT_MODULE_TYPES,
    fake_quant_int4,
    fake_quant_int8,
    iter_quant_targets,
)

#: Measurement artifacts are kept apart from the series (assets-layout: `outputs/bench/<model>/`).
DEFAULT_OUT = BENCH_ROOT / "embeddinggemma-2" / f"{date.today().isoformat()}_quant-screen"
REPORT_FILE = "report.json"

#: The name of the reference row (the checkpoint's own weights — the denominator of every metric).
REFERENCE = "reference"

#: The i4 group length (the exporter's default — ADR 0069 addendum 1).
GROUP_SIZE = DEFAULT_GROUP_SIZE

#: The Matryoshka widths the upstream model card lists (the distribution's `dimensions`).
WIDTHS: tuple[int, ...] = (768, 512, 256, 128)

#: A text is long from this many tokens on (the bound of the series before the band was declared).
LONG_FROM = 2048

#: The storage forms a weight group can take here. `bf16` = left as the checkpoint's values.
FORMS = ("bf16", "i8", "i4")

BITS_PER_BYTE = 8
MIB = 1024 * 1024
#: The scale of i8 (per row) and of i4 (per group) is F32 (`docs/container-v1.md` §6).
SCALE_BITS = 32
F32_BITS = 32
PAYLOAD_BITS: Mapping[str, int] = {"bf16": 16, "i8": 8, "i4": 4}


@dataclass(frozen=True)
class Config:
    """One configuration: the form of the linear weights, of the token table, and whether every
    linear input is quantized per token."""

    name: str
    linear: str
    table: str
    act_quant: bool = False


CONFIGS: tuple[Config, ...] = (
    Config("i8-all", linear="i8", table="i8"),
    Config("i8-linear", linear="i8", table="bf16"),
    Config("i4-linear+i8-table", linear="i4", table="i8"),
    Config("i4-all", linear="i4", table="i4"),
    Config("i8-all+a8", linear="i8", table="i8", act_quant=True),
)

#: The storage the distribution ships today (the size table's anchor; not a measured config —
#: its values are the reference's).
SHIPPED = Config("bf16 (shipped)", linear="bf16", table="bf16")


@dataclass(frozen=True)
class Text:
    name: str
    prompt: str | None
    body: str


# ---- Texts ----------------------------------------------------------------------------


def query_document_pairs(entries: Sequence[tuple[str | None, str]]) -> tuple[tuple[int, int], ...]:
    """`(query index, document index)` of every query that is directly followed by a document.

    MUST: every query is followed by a document and every document follows a query — otherwise the
    pairing the retrieval metric rests on is not what the corpus means, and it fails loudly.
    """
    pairs: list[tuple[int, int]] = []
    for index, (kind, _body) in enumerate(entries):
        following = entries[index + 1][0] if index + 1 < len(entries) else None
        previous = entries[index - 1][0] if index > 0 else None
        if kind == export.QUERY:
            if following != export.DOCUMENT:
                raise ValueError(f"entry {index}: a query not followed by a document")
            pairs.append((index, index + 1))
        elif kind == export.DOCUMENT and previous != export.QUERY:
            raise ValueError(f"entry {index}: a document that does not follow a query")
    return tuple(pairs)


def calib_name(index: int, kind: str | None) -> str:
    return f"calib-{index:02d}-{kind or 'bare'}"


def corpus_texts() -> tuple[Text, ...]:
    """The golden cases, then the calibration corpus (the order of every matrix row)."""
    golden = tuple(Text(name, kind, body) for name, kind, body in export.GOLDEN_CASES)
    calib = tuple(
        Text(calib_name(index, kind), kind, body) for index, (kind, body) in enumerate(CALIB_TEXTS)
    )
    return golden + calib


def retrieval_pairs() -> tuple[tuple[int, int], ...]:
    """The 20 pairs as row indices of {@link corpus_texts}."""
    offset = len(export.GOLDEN_CASES)
    return tuple((offset + q, offset + d) for q, d in query_document_pairs(CALIB_TEXTS))


def tokenize(directory: Path, composite: Any) -> tuple[torch.Tensor, ...]:
    """`[1, T]` ids of every text. The golden cases go through the export's own `build_cases`
    (pinned lengths); the corpus through the same tokenizer and prompts, `<bos>` … `<eos>` checked
    and no multimodal placeholder allowed."""
    from tokenizers import Tokenizer

    golden = tuple(
        ids for _name, ids, _pool in export.build_cases(directory, composite, export.SYM_MAX)
    )
    tokenizer = Tokenizer.from_file(str(directory / "tokenizer.json"))
    prompts = export.load_prompts(directory)
    text_config = composite.text_config
    placeholders = {
        int(composite.image_token_id),
        int(composite.video_token_id),
        int(composite.audio_token_id),
    }
    calib: list[torch.Tensor] = []
    for index, (kind, body) in enumerate(CALIB_TEXTS):
        ids = tokenizer.encode(body if kind is None else prompts[kind] + body).ids
        if ids[0] != int(text_config.bos_token_id) or ids[-1] != int(text_config.eos_token_id):
            raise AssertionError(f"{calib_name(index, kind)}: the ids do not run <bos> … <eos>")
        if placeholders.intersection(ids):
            raise AssertionError(f"{calib_name(index, kind)}: holds a multimodal placeholder id")
        calib.append(torch.tensor([ids], dtype=torch.int64))
    return golden + tuple(calib)


# ---- Metrics (pure) -------------------------------------------------------------------


def matryoshka(vectors: torch.Tensor, width: int) -> torch.Tensor:
    """The leading `width` entries of each row, re-normalised to unit length (in f64)."""
    if not 0 < width <= int(vectors.shape[-1]):
        raise ValueError(f"width {width} is outside 1..{int(vectors.shape[-1])}")
    sliced = vectors[..., :width].to(torch.float64)
    return sliced / sliced.norm(dim=-1, keepdim=True)


def row_cosines(values: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Cosine of each row of two unit-row matrices."""
    return (values * reference).sum(dim=-1)


def cosine_summary(cosines: torch.Tensor, lengths: Sequence[int]) -> dict[str, Any]:
    """min / mean of the cosines over all texts, the short ones and the long ones."""
    lengths_tensor = torch.tensor(list(lengths))
    groups = {
        "all": torch.ones_like(lengths_tensor, dtype=torch.bool),
        "short": lengths_tensor < LONG_FROM,
        "long": lengths_tensor >= LONG_FROM,
    }
    summary: dict[str, Any] = {}
    for label, keep in groups.items():
        chosen = cosines[keep]
        if chosen.numel() == 0:
            raise ValueError(f"no {label} text (lengths {list(lengths)})")
        summary[label] = {
            "texts": int(chosen.numel()),
            "min": float(chosen.min()),
            "mean": float(chosen.mean()),
        }
    return summary


def own_top1(similarity: torch.Tensor) -> int:
    """How many queries (rows) rank their own document (the same index) first."""
    own = torch.arange(int(similarity.shape[0]))
    return int((similarity.argmax(dim=1) == own).sum())


def top1_agreement(similarity: torch.Tensor, reference: torch.Tensor) -> int:
    """How many queries put the same document first as the reference does."""
    return int((similarity.argmax(dim=1) == reference.argmax(dim=1)).sum())


def rank_inversion_rate(similarity: torch.Tensor, reference: torch.Tensor) -> float:
    """The share of (query, document i < j) triples whose order differs from the reference's."""
    documents = int(similarity.shape[1])
    upper = torch.triu(torch.ones(documents, documents, dtype=torch.bool), diagonal=1)
    order = torch.sign(similarity.unsqueeze(2) - similarity.unsqueeze(1))[:, upper]
    reference_order = torch.sign(reference.unsqueeze(2) - reference.unsqueeze(1))[:, upper]
    return float((order != reference_order).to(torch.float64).mean())


def measure_width(
    vectors: torch.Tensor,
    reference: torch.Tensor,
    lengths: Sequence[int],
    pairs: Sequence[tuple[int, int]],
    width: int,
) -> dict[str, Any]:
    """Every metric of one configuration at one width (`vectors` / `reference`: `[texts, 768]`)."""
    values = matryoshka(vectors, width)
    base = matryoshka(reference, width)
    queries = [q for q, _d in pairs]
    documents = [d for _q, d in pairs]
    similarity = values[queries] @ values[documents].T
    reference_similarity = base[queries] @ base[documents].T
    cosines = row_cosines(values, base)
    return {
        "cosine": cosine_summary(cosines, lengths),
        "textCosines": [float(value) for value in cosines],
        "ownTop1": own_top1(similarity),
        "top1Agreement": top1_agreement(similarity, reference_similarity),
        "rankInversionRate": rank_inversion_rate(similarity, reference_similarity),
        "cosineMatrixMaxAbsChange": float((values @ values.T - base @ base.T).abs().max()),
    }


# ---- Size projection (pure) -----------------------------------------------------------


@dataclass(frozen=True)
class GroupStats:
    """A weight group's elements and rows (rows = per-row scales of i8)."""

    tensors: int
    elements: int
    rows: int


def projected_bits(form: str, stats: GroupStats, group_size: int = GROUP_SIZE) -> int:
    """Payload bits of a weight group in one form: `16·N` (bf16), `8·N + 32·rows` (i8) or
    `4·N + 32·N/g` (i4)."""
    if form not in FORMS:
        raise ValueError(f"unknown form {form!r}")
    bits = PAYLOAD_BITS[form] * stats.elements
    if form == "i8":
        bits += SCALE_BITS * stats.rows
    if form == "i4":
        if stats.elements % group_size:
            raise ValueError(f"{stats.elements} elements do not divide into groups of {group_size}")
        bits += SCALE_BITS * (stats.elements // group_size)
    return bits


def formula(form: str, stats: GroupStats, group_size: int = GROUP_SIZE) -> str:
    n = f"{stats.elements:,}"
    if form == "bf16":
        return f"16·{n}"
    if form == "i8":
        return f"8·{n} + 32·{stats.rows:,}"
    return f"4·{n} + 32·{stats.elements // group_size:,}"


def size_projection(
    config: Config, linear: GroupStats, table: GroupStats, remainder_elements: int
) -> dict[str, Any]:
    """The projected payload of a configuration (weight groups + the f32 remainder), with its
    arithmetic."""
    linear_bits = projected_bits(config.linear, linear)
    table_bits = projected_bits(config.table, table)
    remainder_bits = F32_BITS * remainder_elements
    total = linear_bits + table_bits + remainder_bits
    return {
        "linearBytes": linear_bits // BITS_PER_BYTE,
        "tableBytes": table_bits // BITS_PER_BYTE,
        "remainderBytes": remainder_bits // BITS_PER_BYTE,
        "bytes": total // BITS_PER_BYTE,
        "mib": total / BITS_PER_BYTE / MIB,
        "arithmetic": (
            f"linear {formula(config.linear, linear)} + table {formula(config.table, table)}"
            f" + remainder 32·{remainder_elements:,} bits"
        ),
    }


# ---- The model side -------------------------------------------------------------------


def group_stats(targets: Sequence[tuple[str, torch.Tensor, int]]) -> GroupStats:
    return GroupStats(
        tensors=len(targets),
        elements=sum(int(weight.numel()) for _fqn, weight, _axis in targets),
        rows=sum(int(weight.shape[axis]) for _fqn, weight, axis in targets),
    )


def rope_table_elements(config: Any, sym_max: int) -> int:
    """Elements of the folded RoPE tables: cos and sin `[sym_max, head_dim]` per layer type (the
    export's module docstring — `[1, 1, 8192, 256]` and `[1, 1, 8192, 512]`)."""
    dims = {
        layer_type: int(layer.head_dim)
        for layer_type, layer in zip(config.layer_types, config.per_layer_config, strict=True)
    }
    return sum(2 * sym_max * dim for dim in dims.values())


def apply_weights(model: nn.Module, config: Config, linear_modules: frozenset[str]) -> list[str]:
    """Round the weight groups of `config` in place; returns the FQNs rounded (the scale ledgers'
    keys)."""
    rounded: list[str] = []
    groups: tuple[tuple[str, Callable[[str], bool], tuple[type[nn.Module], ...]], ...] = (
        (config.linear, lambda name: name in linear_modules, (nn.Linear,)),
        (config.table, lambda name: name == export.TABLE_MODULE, (nn.Embedding,)),
    )
    for form, include, op_types in groups:
        if form == "i8":
            rounded += list(fake_quant_int8(model, include).scales)
        elif form == "i4":
            report = fake_quant_int4(model, GROUP_SIZE, include, op_types)
            rounded += list(report.scales)
    return rounded


def expected_rounded(config: Config, linear: Sequence[str], table: Sequence[str]) -> set[str]:
    expected: set[str] = set()
    if config.linear != "bf16":
        expected |= set(linear)
    if config.table != "bf16":
        expected |= set(table)
    return expected


def assert_rounded(rounded: Sequence[str], expected: set[str], name: str) -> None:
    """MUST: the rounding reached exactly the intended weights, each once — a weight left out comes
    out on the good side of every metric without a trace in the numbers."""
    if len(rounded) != len(set(rounded)) or set(rounded) != expected:
        missing = sorted(expected - set(rounded))[:3]
        extra = sorted(set(rounded) - expected)[:3]
        raise AssertionError(
            f"[{name}] rounded {len(rounded)} weights, expected {len(expected)}"
            f" (missing {missing} / extra {extra} / duplicates {len(rounded) - len(set(rounded))})"
        )


def assert_linears_called(model: nn.Module, ids: torch.Tensor, window: int) -> None:
    """Every `nn.Linear` is called as a module in a forward — the a8 hooks see only module calls, so
    a linear reached through its weight alone would stay unquantized silently."""
    calls: dict[str, int] = {}
    handles = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            calls[name] = 0

            def hook(_module: nn.Module, _args: tuple[Any, ...], name: str = name) -> None:
                calls[name] += 1

            handles.append(module.register_forward_pre_hook(hook))
    try:
        with torch.no_grad():
            model(input_ids=ids, attention_mask=export.layer_masks(int(ids.shape[1]), window))
    finally:
        for handle in handles:
            handle.remove()
    silent = sorted(name for name, count in calls.items() if count == 0)
    if silent:
        raise AssertionError(f"{len(silent)} linears are not called as modules: {silent[:3]}")


def embed_all(
    wrapper: export.EmbeddingWrapper, inputs: Sequence[torch.Tensor], texts: Sequence[Text]
) -> tuple[torch.Tensor, dict[str, float]]:
    """`[texts, 768]` embeddings, and the seconds each long text took."""
    rows: list[torch.Tensor] = []
    seconds: dict[str, float] = {}
    for text, ids in zip(texts, inputs, strict=True):
        started = time.perf_counter()
        with torch.no_grad():
            rows.append(wrapper.text(ids, torch.ones_like(ids, dtype=torch.float32)).reshape(-1))
        if int(ids.shape[1]) >= LONG_FROM:
            seconds[text.name] = round(time.perf_counter() - started, 2)
    return torch.stack(rows).detach().clone(), seconds


def restore(weights: Mapping[str, torch.Tensor], pristine: Mapping[str, torch.Tensor]) -> None:
    with torch.no_grad():
        for fqn, weight in weights.items():
            weight.copy_(pristine[fqn])


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def run(directory: Path) -> dict[str, Any]:
    sources.assert_pinned(directory, export.READ_FILES)
    export.assert_sentence_transformer_layout(directory)
    model, composite = export.load_text_model(directory)
    window = int(model.config.sliding_window)
    wrapper = export.EmbeddingWrapper(model, window).eval()
    texts = corpus_texts()
    inputs = tokenize(directory, composite)
    lengths = [int(ids.shape[1]) for ids in inputs]
    pairs = retrieval_pairs()
    _log(f"[texts] {len(texts)} texts, {len(pairs)} pairs, T {min(lengths)}..{max(lengths)}")

    linear_targets = list(iter_quant_targets(model, (nn.Linear,)))
    table_targets = list(iter_quant_targets(model, (nn.Embedding,)))
    if [fqn for fqn, _w, _a in table_targets] != [f"{export.TABLE_MODULE}.weight"]:
        raise AssertionError(f"the embedding targets are {[t[0] for t in table_targets]}")
    every = {fqn for fqn, _w, _a in iter_quant_targets(model, QUANT_MODULE_TYPES)}
    linear_fqns = [fqn for fqn, _w, _a in linear_targets]
    table_fqns = [fqn for fqn, _w, _a in table_targets]
    if every != set(linear_fqns) | set(table_fqns):
        raise AssertionError("weight-slot targets other than the linears and the token table")
    linear_modules = frozenset(fqn.removesuffix(".weight") for fqn in linear_fqns)
    linear_stats, table_stats = group_stats(linear_targets), group_stats(table_targets)
    remainder_tensors = {
        key: int(value.numel()) for key, value in model.state_dict().items() if key not in every
    }
    rope = rope_table_elements(model.config, export.SYM_MAX)
    remainder = sum(remainder_tensors.values()) + rope

    weights = {fqn: weight for fqn, weight, _axis in iter_quant_targets(model, QUANT_MODULE_TYPES)}
    pristine = {fqn: weight.detach().clone() for fqn, weight in weights.items()}
    assert_linears_called(model, inputs[0], window)

    started = time.perf_counter()
    reference, reference_seconds = embed_all(wrapper, inputs, texts)
    golden_names = [name for name, _kind, _body in export.GOLDEN_CASES]
    sanity = export._sanity({name: reference[i] for i, name in enumerate(golden_names)})
    configs: dict[str, dict[str, Any]] = {
        REFERENCE: {
            "linear": "checkpoint",
            "table": "checkpoint",
            "actQuant": False,
            "elapsed": round(time.perf_counter() - started, 1),
            "longTextSeconds": reference_seconds,
            "widths": {
                str(width): measure_width(reference, reference, lengths, pairs, width)
                for width in WIDTHS
            },
        }
    }
    _log(f"[{REFERENCE}] {configs[REFERENCE]['elapsed']}s {reference_seconds}")

    outputs: dict[str, torch.Tensor] = {}
    gates: list[str] = []
    for config in CONFIGS:
        started = time.perf_counter()
        restore(weights, pristine)
        rounded = apply_weights(model, config, linear_modules)
        assert_rounded(rounded, expected_rounded(config, linear_fqns, table_fqns), config.name)
        handles: list[object] = []
        attached = 0
        if config.act_quant:
            handles, attached = attach_act_quant(model)
            if attached != len(linear_fqns):
                detach_act_quant(handles)
                raise AssertionError(f"a8 attached to {attached} of {len(linear_fqns)} linears")
        try:
            vectors, seconds = embed_all(wrapper, inputs, texts)
        finally:
            detach_act_quant(handles)
        outputs[config.name] = vectors
        if torch.equal(vectors, reference):
            gates.append(f"{config.name}: bit-identical to the reference (nothing was rounded)")
        configs[config.name] = {
            "linear": config.linear,
            "table": config.table,
            "actQuant": config.act_quant,
            "rounded": len(rounded),
            "actQuantAttached": attached,
            "elapsed": round(time.perf_counter() - started, 1),
            "longTextSeconds": seconds,
            "size": size_projection(config, linear_stats, table_stats, remainder),
            "widths": {
                str(width): measure_width(vectors, reference, lengths, pairs, width)
                for width in WIDTHS
            },
        }
        full = configs[config.name]["widths"]["768"]
        _log(
            f"[{config.name}] {configs[config.name]['elapsed']}s cos min"
            f" {full['cosine']['all']['min']:.6f} own top-1 {full['ownTop1']}/{len(pairs)}"
        )

    # The table's rounding, and the activation hooks, each reach the output.
    for left, right in (("i8-all", "i8-linear"), ("i4-linear+i8-table", "i4-all")):
        if torch.equal(outputs[left], outputs[right]):
            gates.append(f"{left} and {right} give identical embeddings")
    if torch.equal(outputs["i8-all"], outputs["i8-all+a8"]):
        gates.append("i8-all+a8 equals i8-all (the activation hooks did nothing)")
    restore(weights, pristine)
    again, _ = embed_all(wrapper, inputs[:1], texts[:1])
    if not torch.equal(again[0], reference[0]):
        gates.append("the restored weights do not give the reference back bit for bit")

    return {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "script": "tools/export-recipes/embeddinggemma2/measure_quant.py",
        "torch": torch.__version__,
        "threads": torch.get_num_threads(),
        "snapshot": str(directory),
        "revision": sources.PIN.revision,
        "groupSize": GROUP_SIZE,
        "widths": list(WIDTHS),
        "longFrom": LONG_FROM,
        "texts": [
            {"name": text.name, "prompt": text.prompt, "tokens": length}
            for text, length in zip(texts, lengths, strict=True)
        ],
        "pairs": [[texts[q].name, texts[d].name] for q, d in pairs],
        "referenceSanity": sanity,
        "weights": {
            "linear": vars(linear_stats),
            "table": vars(table_stats),
            "remainder": {
                "stateDictElements": sum(remainder_tensors.values()),
                "stateDictTensors": len(remainder_tensors),
                "ropeTableElements": rope,
                "elements": remainder,
            },
        },
        "shippedSize": size_projection(SHIPPED, linear_stats, table_stats, remainder),
        "configs": configs,
        "gates": gates,
    }


# ---- Tables ---------------------------------------------------------------------------


def _markdown(header: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = [f"| {' | '.join(header)} |", f"| {' | '.join('---' for _ in header)} |"]
    lines += [f"| {' | '.join(row)} |" for row in rows]
    return "\n".join(lines)


def quality_table(report: Mapping[str, Any], width: int) -> str:
    pairs = len(report["pairs"])
    header = [
        "config",
        "cos min",
        "cos mean",
        "short min",
        "short mean",
        "long min",
        "long mean",
        "own top-1",
        "top-1 agree",
        "inversion",
        "max Δ cos matrix",
    ]
    rows: list[list[str]] = []
    for name, entry in report["configs"].items():
        metrics = entry["widths"][str(width)]
        cosine = metrics["cosine"]
        rows.append(
            [
                name,
                f"{cosine['all']['min']:.6f}",
                f"{cosine['all']['mean']:.6f}",
                f"{cosine['short']['min']:.6f}",
                f"{cosine['short']['mean']:.6f}",
                f"{cosine['long']['min']:.6f}",
                f"{cosine['long']['mean']:.6f}",
                f"{metrics['ownTop1']}/{pairs}",
                f"{metrics['top1Agreement']}/{pairs}",
                f"{metrics['rankInversionRate'] * 100:.2f}%",
                f"{metrics['cosineMatrixMaxAbsChange']:.4e}",
            ]
        )
    return _markdown(header, rows)


def size_table(report: Mapping[str, Any]) -> str:
    header = ["config", "bytes", "MiB", "arithmetic (bits)"]
    entries = [(SHIPPED.name, report["shippedSize"])]
    entries += [
        (name, entry["size"]) for name, entry in report["configs"].items() if "size" in entry
    ]
    rows = [
        [name, f"{size['bytes']:,}", f"{size['mib']:.1f}", size["arithmetic"]]
        for name, size in entries
    ]
    return _markdown(header, rows)


def time_table(report: Mapping[str, Any]) -> str:
    header = ["config", "wall s", "long texts (s)"]
    rows = [
        [
            name,
            f"{entry['elapsed']:.1f}",
            " / ".join(f"{text} {sec}" for text, sec in entry["longTextSeconds"].items()),
        ]
        for name, entry in report["configs"].items()
    ]
    return _markdown(header, rows)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--snapshot-dir", type=Path, default=sources.SNAPSHOT_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    report = run(args.snapshot_dir)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / REPORT_FILE).write_text(
        json.dumps(report, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    for width in WIDTHS:
        print(f"Width {width} (against the reference; short = T < {LONG_FROM}):")
        print(quality_table(report, width))
        print()
    print("Projected container payload (weight groups + f32 remainder):")
    print(size_table(report))
    print()
    print("Wall time per configuration (apply + every text):")
    print(time_table(report))
    print()
    print(f"report → {args.out / REPORT_FILE}")
    if report["gates"]:
        raise AssertionError("verification gates are red: " + " / ".join(report["gates"]))
    print("gates: all green")


if __name__ == "__main__":
    main()
