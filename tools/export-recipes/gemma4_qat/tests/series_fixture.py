"""**The valid minimal series** the QAT distribution recipe takes as input (a fixed-storage
container + PLE + assets).

The QAT version of `gemma4/tests/product_fixture.py`. The difference is the fixed quantization
itself — mixed storage (the i2 shared head / i4 / i8), the SRQ around the quantized linears, the
packed PLE (which `gemma4_qat.ple` builds into the container's asset), and `reference.json`.

MUST: never spell the container bytes, the IR rules or the PLE index by hand (the same MUST as
`product_fixture`) — with a copy of the rules, the day the rules move only the fixture goes stale
and "the tests are green, only the real thing fails". Writing goes down the same single path as
the real thing (`karume.emit.stored_model` → `karume.publish.publish_container`,
`gemma4_qat.ple.build_ple`).

MUST: use **numbers different from the real ones** (vocabulary 9, layers 3, dim 32, hidden 16,
position limit 8192) — anything with baked-in dimensions fails.

It uses upstream's fixed-quantization module (`transformers.integrations.gemma_quant`), so in an
environment that has not synced the dependency group `gemma4-qat`, every test importing this
module is SKIPped.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import pytest
import torch
from torch import nn

pytest.importorskip("transformers")
from transformers.integrations.gemma_quant import QuantizedEmbedding

from gemma4.distribution import (
    GEMMA4_ROPE_LAYER_TYPES,
    GEMMA4_ROPE_PARTS,
    gemma4_rope_input_name,
)
from gemma4.rope import FULL_ATTENTION, SLIDING_ATTENTION
from gemma4.tests.product_fixture import tokenizer_asset
from gemma4_qat.config import (
    MAX_CHUNK_LENGTH,
    MAX_SELECTED_ROWS,
    REFERENCE_SCHEMA,
    SOURCES,
)
from gemma4_qat.ple import build_ple
from karume.container import AssetInput, Provenance
from karume.emit import FixedQuantizedWeight, stored_model
from karume.ir import (
    IrGraph,
    IrInitializer,
    IrInput,
    IrNode,
    IrState,
    IrStorage,
    IrValue,
)
from karume.publish import publish_container

#: The fixture's provenance (no distribution is built without `--license` — container-v1 §12).
FIXTURE_PROVENANCE = Provenance(license="apache-2.0", writer="karume-fixture")

#: The synthetic dimensions (the real ones: vocabulary 262144, layers 35/42, dim 256, hidden
#: 1536/2560). `HIDDEN` is a multiple of 16 because of the I2 storage's alignment requirement
#: (`karume.emit._plan_fixed_weights`).
VOCAB = 9
LAYERS = 3
DIM = 32
HIDDEN = 16

#: The intermediate width (a number different from `HIDDEN` — a mix-up fails on shape) and the
#: I4 group length.
INNER = 32
GROUP_SIZE = 16

#: The `text_config` of upstream `config.json` (only the fields the distribution recipe reads).
#: The position limit is larger than the default capacity (4096) and differs from the real one
#: (131072).
MAX_POSITION = 8192
SLIDING_HEAD_DIM = 4
FULL_HEAD_DIM = 8

TEXT_CONFIG: Mapping[str, Any] = {
    "max_position_embeddings": MAX_POSITION,
    "hidden_size": HIDDEN,
    "num_attention_heads": 4,
    "head_dim": SLIDING_HEAD_DIM,
    "global_head_dim": FULL_HEAD_DIM,
    "layer_types": [SLIDING_ATTENTION, FULL_ATTENTION],
    "rope_parameters": {
        SLIDING_ATTENTION: {"rope_type": "default", "rope_theta": 100.0},
        FULL_ATTENTION: {
            "rope_type": "proportional",
            "rope_theta": 1000.0,
            "partial_rotary_factor": 0.5,
        },
    },
}

ROPE_HEAD_DIMS: Mapping[str, int] = {
    SLIDING_ATTENTION: SLIDING_HEAD_DIM,
    FULL_ATTENTION: FULL_HEAD_DIM,
}

#: The recommendations of upstream `generation_config.json` (**values different from the real
#: ones**).
GENERATION_CONFIG: Mapping[str, Any] = {
    "do_sample": True,
    "eos_token_id": [1, 2],
    "temperature": 0.7,
    "top_k": 8,
    "top_p": 0.5,
}

#: Symbols (`M` = chunk rows, `R` = exit rows, `C` = capacity of the full slots).
SEQ_SYMBOL = "M"
ROW_SYMBOL = "R"
CAPACITY_SYMBOL = "C"
SYM_MAX = 4

#: The only linear left unquantized (the one the structural check allows by name).
PROJECTION_TENSOR = "model.model.per_layer_model_projection.weight"

#: The fixed SRQ scale (the op contract requires an exact f32 value).
SRQ_SCALE = 0.125


def packed_embedding(bits: int, rows: int = VOCAB, width: int = LAYERS * DIM) -> QuantizedEmbedding:
    """A packed embedding in upstream's format (the material of PLE and the token embedding)."""
    factor = 8 // bits
    module = QuantizedEmbedding(rows, width, torch.float32, embed_scale=3.5, num_bits=bits)
    module.embedding_quantized = nn.Parameter(
        (torch.arange(rows * width // factor) * 37 % 256)
        .to(torch.uint8)
        .reshape(rows, width // factor),
        requires_grad=False,
    )
    module.embedding_scale = nn.Parameter(
        torch.arange(1, rows * (width // DIM) + 1, dtype=torch.float32).reshape(rows, width // DIM)
        / 19,
        requires_grad=False,
    )
    return module


def _packed(rows: int, width: int, bits: int) -> torch.Tensor:
    factor = 8 // bits
    dtype = torch.int8 if bits == 8 else torch.uint8
    values = torch.arange(rows * width // factor) * 29 % 128
    return values.to(dtype).reshape(rows, width // factor).contiguous()


def _row_scale(rows: int, groups: int = 1) -> torch.Tensor:
    return (
        torch.arange(1, rows * groups + 1, dtype=torch.float32).reshape(rows, groups) / 8
    ).contiguous()


def _ramp(*shape: int) -> torch.Tensor:
    total = 1
    for dim in shape:
        total *= dim
    return torch.arange(total, dtype=torch.float32).reshape(*shape) / total


def qat_container(
    *,
    vocab: int = VOCAB,
    layers: int = LAYERS,
    dim: int = DIM,
    hidden_size: int = HIDDEN,
    head_dims: Mapping[str, int] | None = None,
    mlp_bits: int = 8,
    assets: Mapping[str, AssetInput] | None = None,
) -> list[bytes]:
    """The part bytes of one QAT product graph (in reading order — part 0 first).

    The shape is a scaled-down real one: the token embedding is i2, and the head linear uses the
    same initializer as its weight (the shared head, without SRQ on either side). There is also
    one quantized linear each of i4 / i8, both with SRQ before and after. The only unquantized
    linear is the one allowed projection. PLE ships as an **asset** of the container (ADR 0109
    decision 4).

    `mlp_bits` is the fault-injection point of the mixed storage (`4` removes i8 from the binding
    table).
    """
    widths = {**ROPE_HEAD_DIMS, **dict(head_dims or {})}
    initializers: dict[str, IrInitializer] = {}
    values: dict[str, IrValue] = {}
    tensors: dict[str, torch.Tensor] = {}
    fixed: dict[str, FixedQuantizedWeight] = {}
    nodes: list[IrNode] = []

    def declare(name: str, tensor: torch.Tensor, dtype: str = "f32", key: str | None = None) -> str:
        tensor_key = key if key is not None else f"gemma4qat.{name}"
        initializers[name] = IrInitializer(tensor=tensor_key, storage=IrStorage(dtype=dtype))
        values[name] = IrValue(dtype="i32" if dtype == "i32" else "f32", shape=list(tensor.shape))
        tensors[tensor_key] = tensor
        return tensor_key

    def declare_fixed(name: str, rows: int, width: int, bits: int, groups: int = 1) -> None:
        """One fixed-storage weight (the logical shape is a meta f32; only the packed side holds
        values)."""
        key = declare(name, torch.empty(rows, width, dtype=torch.float32, device="meta"))
        fixed[key] = FixedQuantizedWeight(
            f"i{bits}", _packed(rows, width, bits), _row_scale(rows, groups)
        )

    def quantize(source: str, out: str) -> str:
        values[out] = IrValue(dtype="f32", shape=list(values[source].shape))
        nodes.append(
            IrNode(op="static_quantize", ins=[source], outs=[out], attrs={"scale": SRQ_SCALE})
        )
        return out

    def linear(source: str, weight: str, out: str, width: int) -> str:
        bias = f"{out}_bias"
        declare(bias, _ramp(width))
        values[out] = IrValue(dtype="f32", shape=[*values[source].shape[:-1], width])
        nodes.append(IrNode(op="linear", ins=[source, weight, bias], outs=[out], attrs={}))
        return out

    # ① token embedding (I2) — the shared head uses the same initializer as its weight.
    declare_fixed("embed", vocab, hidden_size, 2)
    embedded = "embedded"
    values[embedded] = IrValue(dtype="f32", shape=[1, SEQ_SYMBOL, hidden_size])
    nodes.append(
        IrNode(
            op="embedding",
            ins=["embed", "input_ids"],
            outs=[embedded],
            attrs={"padding_idx": -1},
        )
    )

    # ② the one allowed unquantized linear (the per-layer projection).
    declare("projection", _ramp(INNER, hidden_size), key=PROJECTION_TENSOR)
    projected = linear(embedded, "projection", "projected", INNER)

    # ③ two quantized linears (I4 / I8) — both with SRQ before and after.
    declare_fixed("attention", INNER, INNER, 4, groups=INNER // GROUP_SIZE)
    declare_fixed("mlp", hidden_size, INNER, mlp_bits)
    attended = linear(quantize(projected, "projected_q"), "attention", "attended", INNER)
    projected_out = linear(quantize(attended, "attended_q"), "mlp", "mlp_out", hidden_size)
    quantize(projected_out, "mlp_q")

    # ④ the full slots (the only place the capacity symbol `C` appears) and the effect ops that
    # touch them.
    const = declare("baked", torch.zeros(1, 1, SYM_MAX, 1, dtype=torch.int32), dtype="i32")
    del const
    values["prefix"] = IrValue(dtype="i32", shape=[1, 1, SEQ_SYMBOL, 1])
    nodes.append(
        IrNode(
            op="sym_prefix_slice",
            ins=["baked"],
            outs=["prefix"],
            attrs={"sym": SEQ_SYMBOL, "slices": [{"dim": 2, "coeff": 1, "offset": 0}]},
        )
    )
    values["kv"] = IrValue(dtype="f32", shape=[1, 1, SEQ_SYMBOL, 1])
    nodes.append(IrNode(op="cast", ins=["prefix"], outs=["kv"], attrs={"to": "f32"}))
    for slot in ("l0.k", "l0.v"):
        nodes.append(
            IrNode(op="state_append", ins=["kv"], outs=[], attrs={}, states={"slot": slot})
        )

    # ⑤ two exits: the logits and hidden of the selected rows. The shared head makes the logits
    # (no SRQ).
    declare("seed", _ramp(1, 1, 1))
    hidden = "hidden"
    values[hidden] = IrValue(dtype="f32", shape=[1, ROW_SYMBOL, hidden_size])
    nodes.append(IrNode(op="expand", ins=["seed"], outs=[hidden], attrs={}))
    logits = linear(hidden, "embed", "logits", vocab)

    graph = IrGraph(
        symbols=[CAPACITY_SYMBOL, SEQ_SYMBOL, ROW_SYMBOL],
        inputs=[
            IrInput(name="input_ids", dtype="i32", shape=[1, SEQ_SYMBOL]),
            *(
                IrInput(
                    name=gemma4_rope_input_name(layer_type, part),
                    dtype="f32",
                    shape=[1, SEQ_SYMBOL, widths[layer_type]],
                )
                for layer_type in GEMMA4_ROPE_LAYER_TYPES
                for part in GEMMA4_ROPE_PARTS
            ),
            IrInput(name="per_layer_inputs", dtype="f32", shape=[1, SEQ_SYMBOL, layers, dim]),
            IrInput(name="last_row", dtype="i32", shape=[ROW_SYMBOL]),
        ],
        outputs=[logits, hidden],
        initializers=initializers,
        values=values,
        states={
            name: IrState(dtype="f32", shape=[1, 1, CAPACITY_SYMBOL, 1])
            for name in ("l0.k", "l0.v")
        },
        nodes=nodes,
    )
    stored = stored_model(graph, tensors, fixed_weights=fixed)
    with TemporaryDirectory() as staging:
        result = publish_container(
            Path(staging) / "model.krm",
            stored.graph,
            stored.tensors,
            stored.bindings,
            graph_name="model",
            provenance=FIXTURE_PROVENANCE,
            assets={} if assets is None else assets,
        )
        return [path.read_bytes() for path in result.parts]


def pinned_checkpoint(model: str) -> dict[str, Any]:
    """The checkpoint fingerprint `export.py` records for an export from the pinned commits
    (`gemma4_qat.config.SOURCES` — the weights' and the tokenizer's; the byte counts are not
    compared, so they are fake)."""
    pins = SOURCES[model].pins
    return {
        "dir": pins.label,
        "files": {
            name: {"bytes": 1, "sha256": sha256}
            for pin in pins.by_role().values()
            for name, sha256 in pin.sha256.items()
        },
        "sources": {
            role: {"repo": pin.repo, "revision": pin.revision, "files": list(pin.sha256)}
            for role, pin in pins.by_role().items()
        },
    }


def reference_record(
    model: str,
    *,
    fixed_weights: int = 3,
    storage_counts: Mapping[str, int] | None = None,
    ple_blocks: int = 1,
    **overrides: Any,
) -> dict[str, Any]:
    """The contents of `reference.json` (by default the counts {@link qat_container} really
    produces)."""
    record = {
        "schema": REFERENCE_SCHEMA,
        "family": "gemma4-qat",
        "model": model,
        "checkpoint": pinned_checkpoint(model),
        "maxChunkLength": MAX_CHUNK_LENGTH,
        "maxSelectedRows": MAX_SELECTED_ROWS,
        "fixedWeights": fixed_weights,
        "storageCounts": dict(
            storage_counts if storage_counts is not None else {"i2": 1, "i4": 1, "i8": 1}
        ),
        "weightFiles": 1,
        "pleBlocks": ple_blocks,
        "upstreamUnused": {"kvCacheScales": 6, "vision": 0, "audio": 0},
    }
    record.update(overrides)
    return record


def write_series(
    source: Path,
    model: str,
    *,
    container_kwargs: Mapping[str, Any] | None = None,
    ple_bits: int | None = None,
    ple_rows: int = VOCAB,
    reference: Mapping[str, Any] | None = None,
    tokenizer: Mapping[str, Any] | None = None,
    text_config: Mapping[str, Any] | None = None,
    generation_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write one series and return the contents of the `reference.json` used.

    PLE is an asset of the container, so the container is always built by {@link qat_container}
    — the axes a caller wants to vary (`mlp_bits` / `hidden_size` …) go in `container_kwargs`.
    """
    from container_series import write_component  # the recipe-shared helper conftest provides

    source.mkdir(parents=True, exist_ok=True)
    bits = ple_bits if ple_bits is not None else (4 if model == "e2b" else 2)
    ple = build_ple(packed_embedding(bits, rows=ple_rows), LAYERS, DIM, source)
    write_component(
        source / "model.krm",
        qat_container(assets=ple.assets, **dict(container_kwargs or {})),
    )
    # The same order as the real thing — the temporary files are deleted once the container is
    # in place (nothing left in the series).
    ple.discard()
    (source / "tokenizer.json").write_text(
        json.dumps(dict(tokenizer if tokenizer is not None else tokenizer_asset(vocab=VOCAB))),
        encoding="utf-8",
    )
    (source / "config.json").write_text(
        json.dumps(
            {
                "model_type": "gemma4",
                "text_config": dict(text_config if text_config is not None else TEXT_CONFIG),
            }
        ),
        encoding="utf-8",
    )
    (source / "generation_config.json").write_text(
        json.dumps(dict(generation_config if generation_config is not None else GENERATION_CONFIG)),
        encoding="utf-8",
    )
    record = dict(
        reference if reference is not None else reference_record(model, ple_blocks=ple.blocks)
    )
    (source / "reference.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record
