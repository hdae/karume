"""**The minimal valid product series** that the distribution recipe's tests use as input
(container + PLE assets + assets).

The shared `ir_fixtures.ir_container` (`tools/exporter/tests/`) cannot hold state slots, so it
cannot build the form the gemma4 gates read (the capacity symbol `C` of the full slots appears
**only in states** = exactly one symbol not determined by the input shapes). This module holds only
that difference, and the writing goes through the shared single path
(`karume.emit.stored_model` → `karume.publish.publish_container`).

MUST: neither the container's bytes nor the IR rules are spelled by hand (the same MUST as
`ir_fixtures`) — holding a copy of the rules would leave only the fixture stale on the day the
rules move, giving "tests green, only the real thing fails".

MUST: use **numbers different from the real ones** (vocabulary 6, layers 2, dim 3, hidden 5,
position limit 37, headDim 4/8) — a baked-in dimension then fails.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
import torch

from gemma4.distribution import (
    GEMMA4_DEFAULT_MODEL,
    GEMMA4_DRAFT_STEPS,
    GEMMA4_DRAFTER_ROLE,
    GEMMA4_DRAFTER_SUFFIX,
    GEMMA4_POSITION_X,
    GEMMA4_POSITION_Y,
    GEMMA4_ROLE,
    GEMMA4_ROPE_LAYER_TYPES,
    GEMMA4_ROPE_PARTS,
    GEMMA4_SOFT_EMBEDS,
    GEMMA4_SOFT_MASK,
    GEMMA4_VISION_PIXELS,
    GEMMA4_VISION_ROLE,
    GEMMA4_VISION_SYMBOL,
    gemma4_rope_input_name,
    gemma4_series_name,
)
from gemma4.rope import FULL_ATTENTION, SLIDING_ATTENTION
from karume.container import BLOCK_MAX_BYTES, AssetInput, Provenance
from karume.emit import stored_model
from karume.ir import (
    IrGraph,
    IrInitializer,
    IrInput,
    IrNode,
    IrShared,
    IrState,
    IrStorage,
    IrValue,
)
from karume.ple import PLE_SCALE_BYTES, ple_assets, ple_row_bytes
from karume.publish import publish_container
from karume.quantize import (
    channel_scale,
    dequantize_int4,
    group_scale,
    quantize_to_int4,
    quantize_to_int8,
)

#: Provenance of the fixture (no distribution is built without `--license` — container-v1 §12).
FIXTURE_PROVENANCE = Provenance(license="apache-2.0", writer="karume-fixture")

#: Synthetic dimensions (the real ones are 262144 / 35 / 256 / 2048). `HIDDEN` is the width of the
#: hidden exit and is **a number different from both VOCAB and DIM** (a combination that mixes up
#: logits and hidden fails on the width).
#: MUST: `HIDDEN` equals `hidden_size` of {@link TEXT_CONFIG} — the distribution recipe compares
#: the upstream declaration with the graph's widths, so if they disagree inside the fixture no
#: valid combination can be built.
VOCAB = 6
LAYERS = 2
#: One PLE row must be **a multiple of 4**, or blocks cannot be cut at multiples of a row
#: (container-v1 §4.1), so `LAYERS * DIM` is chosen divisible by 4 (the real one is 35 × 256).
DIM = 10
HIDDEN = 16

#: The stand-in processor's patch and pooling sizes (the real ones are 16 and 3) and the row width
#: of the vision embedder they give.
VISION_PATCH_SIZE = 2
VISION_POOLING = 2
VISION_PIXEL_WIDTH = (VISION_PATCH_SIZE * VISION_POOLING) ** 2 * 3

#: Minimal form holding only the fields of upstream `config.json`'s `text_config` that the
#: distribution recipe reads. **Numbers different from the real ones** (position limit 131072 → 37,
#: head_dim 256/512 → 4/8, theta also different).
MAX_POSITION = 37
SLIDING_HEAD_DIM = 4
FULL_HEAD_DIM = 8
SLIDING_THETA = 100.0
FULL_THETA = 1000.0
PARTIAL_ROTARY_FACTOR = 0.5

TEXT_CONFIG: Mapping[str, Any] = {
    "max_position_embeddings": MAX_POSITION,
    "hidden_size": HIDDEN,
    # The per-layer embedding width (the real E2B / E4B config declares it — 256; 12B declares 0).
    "hidden_size_per_layer_input": DIM,
    "num_attention_heads": 4,
    "head_dim": SLIDING_HEAD_DIM,
    "global_head_dim": FULL_HEAD_DIM,
    "layer_types": [SLIDING_ATTENTION, FULL_ATTENTION],
    "rope_parameters": {
        SLIDING_ATTENTION: {"rope_type": "default", "rope_theta": SLIDING_THETA},
        FULL_ATTENTION: {
            "rope_type": "proportional",
            "rope_theta": FULL_THETA,
            "partial_rotary_factor": PARTIAL_ROTARY_FACTOR,
        },
    },
}

#: Layer type → width of the RoPE derived inputs (the gates assume it matches the values derived
#: from {@link TEXT_CONFIG}).
ROPE_HEAD_DIMS: Mapping[str, int] = {
    SLIDING_ATTENTION: SLIDING_HEAD_DIM,
    FULL_ATTENTION: FULL_HEAD_DIM,
}

#: Length of the baked constant of `sym_prefix_slice` (the upper bound of the chunk symbol `M`), and
#: the capacity of the full slots.
SYM_MAX = 4
CAPACITY_SYMBOL = "C"
SEQ_SYMBOL = "M"

#: Row-count symbol of the exit (bound by `last_row[R]` — the spelling of
#: `gemma4.export_product.ROW_SYMBOL`).
ROW_SYMBOL = "R"

#: The i4 group length and the shape of the linear weight (the same reason as `ir_fixtures` — the
#: smallest shape whose row length is divisible by group_size; ADR 0069 decision 2).
GROUP_SIZE = 16
_IN = 32
_OUT = 4

#: Initializer name of the retired "bake the tables" form (used by the residue gate — the current
#: assets have none).
BAKED_ROPE_TABLE = "model.model.rotary_emb.full_attention_cos_table"

#: The regular line's storage: the stand-in weights and the default PLE assets (schema 5) of
#: {@link product_container} — the checkpoint's bf16 values.
STORAGE = "bf16"
#: Spelling of the PLE (the source of truth is `karume.ple` /
#: `packages/models/src/gemma/ple-index.ts`).
PLE_STORAGE = STORAGE
PLE_EMBED_SCALE = 2.0

#: The lender's main-table tensor key (the spelling `declare` builds in {@link product_container} —
#: the regular line's bf16 embedding). The borrower's shared initializer names it — if they
#: disagree, the distribution's gate rejects it.
LENDER_SHARED_TENSOR = f"gemma4.embed_{STORAGE}"

#: Format identifier of the compiled tokenizer asset (`_shared/gemma_tokenizer.py`).
TOKENIZER_FORMAT = "karume-gemma-tokenizer/1"

#: The upstream `tokenizer.json` stand-in {@link write_series} places in the checkpoint directory;
#: the asset of {@link tokenizer_asset} records its SHA-256 as its source by default (the
#: distribution's gate compares the SHA-256 only — `assert_gemma4_tokenizer_source`).
UPSTREAM_TOKENIZER = b'{"model": "upstream tokenizer.json stand-in"}'

#: Recommendations of upstream `generation_config.json` (**values different from the real ones** — a
#: hand copy then fails).
GENERATION_CONFIG: Mapping[str, Any] = {
    "do_sample": True,
    "eos_token_id": [1, 2],
    "temperature": 0.7,
    "top_k": 8,
    "top_p": 0.5,
}


def _ramp(*shape: int) -> torch.Tensor:
    total = 1
    for dim in shape:
        total *= dim
    return torch.arange(total, dtype=torch.float32).reshape(*shape) / total


def _publish(
    graph: IrGraph,
    tensors: Mapping[str, torch.Tensor],
    storage: str,
    scales: Mapping[str, torch.Tensor],
    overrides: Mapping[str, str],
    *,
    assets: Mapping[str, AssetInput] = {},
    block_bytes: int = BLOCK_MAX_BYTES,
    named: str = GEMMA4_ROLE,
    provenance: Provenance = FIXTURE_PROVENANCE,
) -> list[bytes]:
    """Write, read back and verify, and return the bytes of each part in index order (the same
    single path as the real thing).

    `named` is the container's graph name = **the component name** (MUST be the key of the
    manifest's weights; container-v1 §2.1). The lender is `model`, the borrower `drafter`, and the
    assembly gate checks that they match.
    """
    stored = stored_model(
        graph,
        tensors,
        weight_dtype=storage,
        weight_scales=scales,
        weight_dtype_overrides=overrides,
    )
    with TemporaryDirectory() as staging:
        result = publish_container(
            Path(staging) / "model.krm",
            stored.graph,
            stored.tensors,
            stored.bindings,
            graph_name=named,
            provenance=provenance,
            assets=assets,
            block_bytes=block_bytes,
        )
        return [path.read_bytes() for path in result.parts]


def ple_container_assets(
    *,
    tokens: int = VOCAB,
    layers: int = LAYERS,
    dim: int = DIM,
    embed_scale: float = PLE_EMBED_SCALE,
    block_bytes: int = BLOCK_MAX_BYTES,
    storage: str = PLE_STORAGE,
    group_size: int | None = None,
) -> dict[str, AssetInput]:
    """PLE as the container's assets — by default the regular line's schema 5 (a bf16 `values`
    block sequence, no scales).

    Both how it is cut and the index's spelling are owned by the core {@link karume.ple.ple_assets}
    — spelling a copy would leave only the fixture stale on the day the rules move. Lowering
    `block_bytes` splits the data into several blocks (the setup that exercises the comparison of
    the index with the asset declarations over several blocks). An integer `storage` writes schema 3
    (one scale per row — `"i8"`), or with `group_size` schema 4 (one scale per group — the q4_0 QAT
    line's form); the bytes are the sizes the writer expects (`karume.ple.ple_row_bytes`), their
    values arbitrary.
    """
    row_bytes = ple_row_bytes(storage, layers, dim, group_size)
    values = np.arange(tokens * row_bytes["values"], dtype=np.uint8)
    value_bytes = memoryview(values).cast("B")
    # A float storage (`bf16` — schema 5) has no scales table.
    read_scales = None
    if "scales" in row_bytes:
        groups = row_bytes["scales"] // PLE_SCALE_BYTES
        scale_bytes = memoryview(np.full(tokens * groups, 0.5, dtype=np.float32)).cast("B")
        read_scales = lambda begin, end: scale_bytes[begin:end]  # noqa: E731
    return ple_assets(
        storage=storage,
        tokens=tokens,
        layers=layers,
        dim=dim,
        embed_scale=embed_scale,
        read_values=lambda begin, end: value_bytes[begin:end],
        read_scales=read_scales,
        block_bytes=block_bytes,
        group_size=group_size,
    )


def product_container(
    *,
    vocab: int = VOCAB,
    layers: int = LAYERS,
    dim: int = DIM,
    hidden_size: int = HIDDEN,
    head_dims: Mapping[str, int] | None = None,
    baked_rope: bool = False,
    free_symbol: bool = True,
    swap_outputs: bool = False,
    assets: Mapping[str, AssetInput] | None = None,
    ple: bool = True,
    provenance: Provenance = FIXTURE_PROVENANCE,
    storage: str = STORAGE,
    embedding_storage: str | None = None,
    soft_mask_width: int = 1,
) -> list[bytes]:
    """The part bytes of one product graph (in read order — part 0 first).

    PLE ships as **assets** of the container (ADR 0109 decision 4) — without `assets`, the
    defaults of {@link ple_container_assets} are placed.

    `head_dims` overrides the widths of the RoPE derived inputs (for the gate that builds a
    generation disagreeing with the declaration). `baked_rope` mixes in one initializer of the
    retired "bake the tables" form (the residue gate). With `free_symbol` false the capacity symbol
    is removed from the states, leaving only `M` / `R`, the two the input shapes bind (the gate on
    how the symbols split — the side that checks there is exactly one free symbol).
    `ple=False` is the graph of a model without per-layer embeddings (12B): no
    `per_layer_inputs` input, the image rows' two inputs instead (`soft_mask` of width
    `soft_mask_width`, for the gate on their form), and no PLE assets unless `assets` names some.
    `provenance` is the
    container's (a pinned model's names its upstream revision). `storage` is the stand-in
    linear's storage and `embedding_storage` the embedding stand-in's (default: `storage`):
    `"bf16"` (the regular line — the checkpoint's values, no int8 / int4), `"i4"` (the q4_0 QAT
    line — with an int4 embedding too the container holds no int8 tensor) or `"i8"`. The
    embedding's key is `gemma4.embed_<its storage>` — the table a drafter borrows
    ({@link LENDER_SHARED_TENSOR} for the regular line).
    `swap_outputs` swaps only the order of the two exits (they have the same shape down to the row
    axis, so this combination passes everything but the width comparison — the order gate).
    """
    widths = {**ROPE_HEAD_DIMS, **dict(head_dims or {})}
    initializers: dict[str, IrInitializer] = {}
    values: dict[str, IrValue] = {}
    tensors: dict[str, torch.Tensor] = {}
    scales: dict[str, torch.Tensor] = {}
    overrides: dict[str, str] = {}
    nodes: list[IrNode] = []

    def declare(name: str, tensor: torch.Tensor, dtype: str = "f32") -> str:
        key = f"gemma4.{name}"
        initializers[name] = IrInitializer(tensor=key, storage=IrStorage(dtype=dtype))
        # Only symbol-dependent constants are i32 (the same as `ir_fixtures` — the rest is
        # semantic f32).
        values[name] = IrValue(dtype="i32" if dtype == "i32" else "f32", shape=list(tensor.shape))
        tensors[key] = tensor
        return key

    # ① The execution skeleton (one linear) — a linear stand-in and an embedding stand-in, each in
    #    its storage:
    #    the header holds the same set of storages as the real container of the line.
    activation = "x"
    bias = "bias"
    declare(activation, _ramp(1, _IN))
    declare(bias, _ramp(_OUT))
    embedding = storage if embedding_storage is None else embedding_storage
    for name, slot_storage in ((f"linear_{storage}", storage), (f"embed_{embedding}", embedding)):
        weight = _ramp(_OUT, _IN)
        if slot_storage == "bf16":
            # bf16-exact values (the round trip the writer requires of a bf16 weight).
            weight = weight.to(torch.bfloat16).to(torch.float32)
            overrides[declare(name, weight)] = "bf16"
        elif slot_storage == "i4":
            scale = group_scale(weight, GROUP_SIZE)
            weight = dequantize_int4(quantize_to_int4(weight, scale), scale)
            key = declare(name, weight)
            scales[key] = scale
            overrides[key] = "i4"
        elif slot_storage == "i8":
            scale = channel_scale(weight, 0)
            weight = quantize_to_int8(weight, scale).to(torch.float32) * scale
            key = declare(name, weight)
            scales[key] = scale
            overrides[key] = "i8"
        else:
            raise ValueError(f"no stand-in storage '{slot_storage}' (bf16 / i4 / i8)")
        out = f"h_{name}"
        values[out] = IrValue(dtype="f32", shape=[1, _OUT])
        nodes.append(IrNode(op="linear", ins=[activation, name, bias], outs=[out], attrs={}))

    # ② Residue of the retired form (not included by default — the control for the gate that checks
    #    "there is none").
    if baked_rope:
        declare(BAKED_ROPE_TABLE, _ramp(2, 2))

    # ③ The symbolic dimensions: `M` is bound by the input shapes, `C` appears only in states.
    const = "baked"
    key = declare(const, torch.zeros(1, 1, SYM_MAX, 1, dtype=torch.int32), dtype="i32")
    del key
    prefix = "prefix"
    values[prefix] = IrValue(dtype="i32", shape=[1, 1, SEQ_SYMBOL, 1])
    nodes.append(
        IrNode(
            op="sym_prefix_slice",
            ins=[const],
            outs=[prefix],
            attrs={"sym": SEQ_SYMBOL, "slices": [{"dim": 2, "coeff": 1, "offset": 0}]},
        )
    )

    # ④ The full slots (the only place the capacity symbol `C` appears) and one effect op touching
    #    them. The IR rejects a slot that is only declared (`verify._check_state_slots` —
    #    referential integrity), so a single-purpose `state_append` (one input, no outputs) stands
    #    in for the real attention.
    kv = "kv"
    values[kv] = IrValue(dtype="f32", shape=[1, 1, SEQ_SYMBOL, 1])
    nodes.append(IrNode(op="cast", ins=[prefix], outs=[kv], attrs={"to": "f32"}))
    for slot in ("l0.k", "l0.v"):
        nodes.append(IrNode(op="state_append", ins=[kv], outs=[], attrs={}, states={"slot": slot}))

    # ⑤ The exits are the logits and hidden of the selected rows (`[1, R, V]` / `[1, R, H]` —
    #    ADR 0083 decision 6 + the groundwork of speculative verify). The order is fixed as
    #    logits → hidden.
    seed = "logits_seed"
    declare(seed, _ramp(1, 1, 1))
    logits = "logits"
    values[logits] = IrValue(dtype="f32", shape=[1, ROW_SYMBOL, vocab])
    nodes.append(IrNode(op="expand", ins=[seed], outs=[logits], attrs={}))
    hidden = "hidden"
    values[hidden] = IrValue(dtype="f32", shape=[1, ROW_SYMBOL, hidden_size])
    nodes.append(IrNode(op="expand", ins=[seed], outs=[hidden], attrs={}))

    # With `free_symbol` false the capacity becomes a concrete number and `M` is the only symbol.
    capacity_dim: str | int = CAPACITY_SYMBOL if free_symbol else SYM_MAX
    graph = IrGraph(
        symbols=(
            [CAPACITY_SYMBOL, SEQ_SYMBOL, ROW_SYMBOL] if free_symbol else [SEQ_SYMBOL, ROW_SYMBOL]
        ),
        inputs=[
            IrInput(name="input_ids", dtype="i32", shape=[1, SEQ_SYMBOL]),
            *(
                []
                if ple
                else [
                    IrInput(
                        name=GEMMA4_SOFT_EMBEDS, dtype="f32", shape=[1, SEQ_SYMBOL, hidden_size]
                    ),
                    IrInput(
                        name=GEMMA4_SOFT_MASK, dtype="bool", shape=[1, SEQ_SYMBOL, soft_mask_width]
                    ),
                ]
            ),
            *(
                IrInput(
                    name=gemma4_rope_input_name(layer_type, part),
                    dtype="f32",
                    shape=[1, SEQ_SYMBOL, widths[layer_type]],
                )
                for layer_type in GEMMA4_ROPE_LAYER_TYPES
                for part in GEMMA4_ROPE_PARTS
            ),
            *(
                [IrInput(name="per_layer_inputs", dtype="f32", shape=[1, SEQ_SYMBOL, layers, dim])]
                if ple
                else []
            ),
            IrInput(name="last_row", dtype="i32", shape=[ROW_SYMBOL]),
        ],
        outputs=[hidden, logits] if swap_outputs else [logits, hidden],
        initializers=initializers,
        values=values,
        # Held as a k / v pair — the borrower's (drafter's) readonly attention requires exactly
        # `{k, v}`, so a lender that declares only k cannot form a valid combination.
        states={
            name: IrState(dtype="f32", shape=[1, 1, capacity_dim, 1]) for name in ("l0.k", "l0.v")
        },
        nodes=nodes,
    )
    if assets is None:
        assets = ple_container_assets(tokens=vocab, layers=layers, dim=dim) if ple else {}
    return _publish(
        graph,
        tensors,
        storage,
        scales,
        overrides,
        assets=assets,
        provenance=provenance,
    )


def drafter_container(
    *,
    vocab: int = VOCAB,
    hidden_size: int = HIDDEN,
    head_dims: Mapping[str, int] | None = None,
    steps: int = GEMMA4_DRAFT_STEPS,
    external: bool = True,
    shared: bool = True,
    capacity_symbol: str | None = CAPACITY_SYMBOL,
    storage: str = "i8",
    head_storage: str = "i8",
    shared_tensor: str = LENDER_SHARED_TENSOR,
    shared_storage: str = STORAGE,
    provenance: Provenance = FIXTURE_PROVENANCE,
) -> list[bytes]:
    """The part bytes of one **borrower** graph (ADR 0096 step 2 — an asset that cannot run on its
    own).

    `head_storage` / `shared_tensor` / `shared_storage` / `provenance` build the q4_0 QAT line's
    drafter: its head int4 too, its shared declaration naming the lender's int4 table
    (`"gemma4.embed_i4"`, declared `"i4"`), and the container naming its drafter's revision.

    Shaped to mesh with the lender ({@link product_container}): it declares the same slot names
    `l0.k` / `l0.v` as **external**, its shared initializer names the lender's main-table tensor
    key, and its only symbol is the capacity. `external` / `shared` / `capacity_symbol` are the
    fault-injection inputs of the distribution's gate
    ({@link gemma4.distribution.assert_gemma4_drafter_graph}).

    The real drafter is **a single i8 storage, linears included** (measured in
    `gemma4/export_drafter.py` — dropping to i4 lowers the acceptance rate by 10 to 30 percent), so
    the default `storage` is `"i8"`. `"i4"` is the fault-injection input of the storage gate
    (`GEMMA4_STORAGE_FORBIDDEN`) and builds **an asset whose linears dropped to i4 while only the
    output head stays i8** — the case a presence check (I8 is there) lets through.
    """
    widths = {**ROPE_HEAD_DIMS, **dict(head_dims or {})}
    initializers: dict[str, IrInitializer] = {}
    values: dict[str, IrValue] = {}
    tensors: dict[str, torch.Tensor] = {}
    scales: dict[str, torch.Tensor] = {}
    overrides: dict[str, str] = {}
    nodes: list[IrNode] = []

    def declare(name: str, tensor: torch.Tensor, dtype: str = "f32") -> str:
        key = f"drafter.{name}"
        initializers[name] = IrInitializer(tensor=key, storage=IrStorage(dtype=dtype))
        values[name] = IrValue(dtype="f32", shape=list(tensor.shape))
        tensors[key] = tensor
        return key

    # ① The real storage form (linears in `storage`, the output head always i8) — the header's set
    #    of dtypes is the storage gate's input as it is.
    declare("x", _ramp(1, _IN))
    declare("bias", _ramp(_OUT))
    slots = ((f"linear_{storage}", storage), (f"head_{head_storage}", head_storage))
    for name, slot_storage in slots:
        weight = _ramp(_OUT, _IN)
        if slot_storage == "i4":
            scale = group_scale(weight, GROUP_SIZE)
            weight = dequantize_int4(quantize_to_int4(weight, scale), scale)
            scales[declare(name, weight)] = scale
        else:
            scale = channel_scale(weight, 0)
            weight = quantize_to_int8(weight, scale).to(torch.float32) * scale
            key = declare(name, weight)
            scales[key] = scale
            overrides[key] = "i8"
        out = f"h_{name}"
        values[out] = IrValue(dtype="f32", shape=[1, _OUT])
        nodes.append(IrNode(op="linear", ins=["x", name, "bias"], outs=[out], attrs={}))

    # ② Shared initializer (no bytes) — it points at the lender container's tensor key.
    embedded = "embedded"
    if shared:
        initializers["target_embed"] = IrInitializer(
            shared=IrShared(tensor=shared_tensor), storage=IrStorage(dtype=shared_storage)
        )
        values["target_embed"] = IrValue(dtype="f32", shape=[vocab, hidden_size])
        values[embedded] = IrValue(dtype="f32", shape=[1, 1, hidden_size])
        nodes.append(
            IrNode(
                op="embedding",
                ins=["target_embed", "token"],
                outs=[embedded],
                attrs={"padding_idx": -1},
            )
        )

    # ③ readonly attention (one q; reads k / v of the external slots).
    seed = "q_seed"
    declare(seed, _ramp(1, 1, 1, 1))
    query = "q"
    values[query] = IrValue(dtype="f32", shape=[1, 1, 1, 1])
    nodes.append(IrNode(op="expand", ins=[seed], outs=[query], attrs={}))
    attended = "attended"
    values[attended] = IrValue(dtype="f32", shape=[1, 1, 1, 1])
    nodes.append(
        IrNode(
            op="attention",
            ins=[query],
            outs=[attended],
            attrs={"scale": 1.0, "readonly": True},
            states={"k": "l0.k", "v": "l0.v"},
        )
    )

    # ④ The exits are k drafts (the real one has 3 argmax tokens — only the count concerns the
    #    distribution's gate).
    outputs: list[str] = []
    for index in range(steps):
        name = f"token_{index}"
        values[name] = IrValue(dtype="f32", shape=[1, 1, 1])
        nodes.append(IrNode(op="reshape", ins=[attended], outs=[name], attrs={}))
        outputs.append(name)

    capacity_dim: str | int = capacity_symbol if capacity_symbol is not None else SYM_MAX
    graph = IrGraph(
        symbols=[capacity_symbol] if capacity_symbol is not None else [],
        inputs=[
            IrInput(name="token", dtype="i32", shape=[1, 1]),
            IrInput(name="hidden", dtype="f32", shape=[1, hidden_size]),
            *(
                IrInput(
                    name=gemma4_rope_input_name(layer_type, part),
                    dtype="f32",
                    shape=[1, 1, widths[layer_type]],
                )
                for layer_type in GEMMA4_ROPE_LAYER_TYPES
                for part in GEMMA4_ROPE_PARTS
            ),
        ],
        outputs=outputs,
        initializers=initializers,
        values=values,
        states={
            name: IrState(dtype="f32", shape=[1, 1, capacity_dim, 1], external=external)
            for name in ("l0.k", "l0.v")
        },
        nodes=nodes,
    )
    return _publish(
        graph, tensors, storage, scales, overrides, named=GEMMA4_DRAFTER_ROLE, provenance=provenance
    )


def vision_container(
    *,
    hidden_size: int = HIDDEN,
    pixel_width: int = VISION_PIXEL_WIDTH,
    storage: str = "bf16",
    provenance: Provenance = FIXTURE_PROVENANCE,
) -> list[bytes]:
    """The part bytes of one vision embedder graph (`gemma4.export_vision`'s inputs and output, one
    weight in `storage` — the distribution's gates read nothing else).

    `hidden_size` / `pixel_width` build a graph of another model, `storage` one that was rounded.
    """
    initializers: dict[str, IrInitializer] = {}
    values: dict[str, IrValue] = {}
    tensors: dict[str, torch.Tensor] = {}
    scales: dict[str, torch.Tensor] = {}
    overrides: dict[str, str] = {}

    def declare(name: str, tensor: torch.Tensor) -> str:
        key = f"vision.{name}"
        initializers[name] = IrInitializer(tensor=key, storage=IrStorage(dtype="f32"))
        values[name] = IrValue(dtype="f32", shape=list(tensor.shape))
        tensors[key] = tensor
        return key

    declare("x", _ramp(1, _IN))
    declare("bias", _ramp(_OUT))
    weight = _ramp(_OUT, _IN)
    if storage == "bf16":
        overrides[declare("weight", weight.to(torch.bfloat16).to(torch.float32))] = "bf16"
    elif storage == "i4":
        scale = group_scale(weight, GROUP_SIZE)
        key = declare("weight", dequantize_int4(quantize_to_int4(weight, scale), scale))
        scales[key] = scale
        overrides[key] = "i4"
    else:
        raise ValueError(f"no stand-in storage '{storage}' (bf16 / i4)")
    values["h"] = IrValue(dtype="f32", shape=[1, _OUT])
    declare("seed", _ramp(1, 1, 1))
    values["embeds"] = IrValue(dtype="f32", shape=[1, GEMMA4_VISION_SYMBOL, hidden_size])
    graph = IrGraph(
        symbols=[GEMMA4_VISION_SYMBOL],
        inputs=[
            IrInput(
                name=GEMMA4_VISION_PIXELS,
                dtype="f32",
                shape=[1, GEMMA4_VISION_SYMBOL, pixel_width],
            ),
            IrInput(name=GEMMA4_POSITION_X, dtype="i32", shape=[1, GEMMA4_VISION_SYMBOL]),
            IrInput(name=GEMMA4_POSITION_Y, dtype="i32", shape=[1, GEMMA4_VISION_SYMBOL]),
        ],
        outputs=["embeds"],
        initializers=initializers,
        values=values,
        states={},
        nodes=[
            IrNode(op="linear", ins=["x", "weight", "bias"], outs=["h"], attrs={}),
            IrNode(op="expand", ins=["seed"], outs=["embeds"], attrs={}),
        ],
    )
    return _publish(
        graph, tensors, storage, scales, overrides, named=GEMMA4_VISION_ROLE, provenance=provenance
    )


def tokenizer_asset(
    *,
    vocab: int = VOCAB,
    format_id: str = TOKENIZER_FORMAT,
    source: bytes | None = UPSTREAM_TOKENIZER,
) -> dict[str, Any]:
    """Minimal form of the compiled tokenizer asset, holding only the fields the gates read.

    `source` is the bytes of the `tokenizer.json` the asset says it was compiled from (its
    SHA-256 goes into `source.sha256`, as the compile script writes it); `None` = no recorded
    source."""
    asset: dict[str, Any] = {"format": format_id, "vocab": [f"t{index}" for index in range(vocab)]}
    if source is not None:
        asset["source"] = {"sha256": hashlib.sha256(source).hexdigest()}
    return asset


def write_series(
    product: Path,
    tokenizer_dir: Path,
    model_dir: Path,
    *,
    container: Sequence[bytes] | None = None,
    drafter: Path | None = None,
    drafter_bytes: Sequence[bytes] | None = None,
    tokenizer: Mapping[str, Any] | None = None,
    generation_config: Mapping[str, Any] | None = None,
    text_config: Mapping[str, Any] | None = None,
) -> None:
    """Writes the series and the checkpoint directory (its `tokenizer.json` is
    {@link UPSTREAM_TOKENIZER}, the source the default tokenizer asset records).

    The product series also holds companions that **do not go into the distribution**
    (`ple.probe.safetensors` / a greedy golden / `reference.json`) — evidence that nothing outside
    the output path table gets mixed in. PLE is an **asset** of the product container, so it does
    not appear as a separate file of the series (ADR 0109 decision 4).
    """
    from container_series import write_component  # recipe-shared helper set up by conftest

    parts = list(container if container is not None else product_container())
    write_component(product / "model.krm", parts)
    # The borrower (drafter) series goes next to product by default (so that not one existing call
    # needs rewriting — the series name is built from the one place the distribution recipe owns).
    borrower = (
        drafter
        if drafter is not None
        else product.parent / gemma4_series_name(GEMMA4_DEFAULT_MODEL, GEMMA4_DRAFTER_SUFFIX)
    )
    write_component(
        borrower / "model.krm",
        list(drafter_bytes if drafter_bytes is not None else drafter_container()),
    )
    # Companions that do not go into the distribution (golden and provenance record) — evidence
    # that they are not in the output path table.
    (borrower / "drafter-golden.short-en.safetensors").write_bytes(b"not distributed")
    (borrower / "reference.json").write_text("{}\n", encoding="utf-8")
    (product / "ple.probe.safetensors").write_bytes(b"not distributed")
    (product / "greedy.capital-en.safetensors").write_bytes(b"not distributed")
    (product / "reference.json").write_text("{}\n", encoding="utf-8")

    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    (tokenizer_dir / "tokenizer.json").write_text(
        json.dumps(dict(tokenizer if tokenizer is not None else tokenizer_asset())),
        encoding="utf-8",
    )
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "tokenizer.json").write_bytes(UPSTREAM_TOKENIZER)
    (model_dir / "generation_config.json").write_text(
        json.dumps(dict(generation_config if generation_config is not None else GENERATION_CONFIG)),
        encoding="utf-8",
    )
    # Upstream `config.json` is a multimodal container whose text part is the `text_config` section
    # (the same shape as the real one).
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "model_type": "gemma4",
                "text_config": dict(text_config if text_config is not None else TEXT_CONFIG),
            }
        ),
        encoding="utf-8",
    )
