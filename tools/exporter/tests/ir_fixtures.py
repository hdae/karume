"""Synthesis of **valid minimal containers** (`krm`) that the assembly-layer tests use as input.

What it returns is not one file but **a sequence of parts** (part 0 = the 2 documents, part 1 =
the const region, part 2 onwards = weights) — the official HF distribution is the split form only
(container-v1 §8), so a single-form fixture would not have the shape of the real thing
(`container.parts` of `karume/5` MUST have 2 or more elements).

The `karume.dist` assembly checks its input containers against every rule of the declarations
({@link karume.dist.assert_weight_components_verified}), so a fixture plugged into the weights
field must be **a real container**. A hand-spelled fake reaches neither the descriptor's structural
checks nor the merge.

It lives in this one place because its consumers span 2 trees — the core side's
`tools/exporter/tests/test_dist.py` and the repository-only recipes'
`tools/export-recipes/<family>/tests/test_distribution.py` (for the latter,
`tools/export-recipes/conftest.py` sets up sys.path). Dependencies still point one way, recipe →
core (ADR 0065 decision 3), and the wheel's contents (`src/karume/`) do not grow by a single byte.

MUST: bytes are not spelled by hand — writing goes through the single path
`karume.emit.stored_model` → `karume.publish.publish_container` (read-back verification
included). Holding a copy of the rules would leave the fixtures stale on the day the rules move:
"tests green, only the real thing fails". The only exception is the fault-injection case of
`ir_container(symbols=…)` — a symbol without a binding point violates the IR acceptance rules,
so the publication check (`publish_container`) rejects it. Only this case is written through the
same writer's low-level entry (`karume.container.write_model_container`), without read-back
verification.

The same arguments give **the same bytes** (no randomness) — the bytes differ per role through
`mark` (the tensor-key prefix), so they can be kept as module constants.

NOTE: every graph input is declared with f32 semantics. The checks read only the names and shapes
(and the count), and the inputs are **declaration-only fields** that no node consumes, so there is
no reason to declare i32 index inputs separately.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from karume.container import (
    BLOCK_MAX_BYTES,
    DEFAULT_PART_BYTES,
    HEADER_BYTES,
    AssetInput,
    Provenance,
    canonical_json,
    read_header,
    write_header,
    write_model_container,
)
from karume.dims import parse_dim
from karume.emit import stored_model
from karume.ir import IrGraph, IrInitializer, IrInput, IrNode, IrStorage, IrValue
from karume.publish import publish_container
from karume.quantize import (
    channel_scale,
    dequantize_int4,
    group_scale,
    quantize_to_int4,
    quantize_to_int8,
)

#: The fixtures' provenance (no distribution is made without `--license` — container-v1 §12).
FIXTURE_PROVENANCE = Provenance(license="apache-2.0", writer="karume-fixture")

#: The shape of the linear weights and the i4 group length. A row length of 32 is 2 groups of 16
#: (the minimal shape that satisfies "the row length is divisible by group_size" of i4 storage —
#: ADR 0069 decision 2).
GROUP_SIZE = 16
_IN = 32
_OUT = 4

#: The tensor of one part of a split fixture (f32, 16 elements = 64 bytes — numbered parts:
#: container-v1 §8).
_FILL_ELEMENTS = 16

#: The prefix of the names the synthesis adds by itself. Recipe checks count layer numbers from
#: initializer names (SBV2), so it is spelled so as not to mix with the names the caller passes.
_OWN = "karume_fixture_"

Shape = Sequence[int | str]


def _ramp(*shape: int) -> torch.Tensor:
    """A tensor of small deterministic values (no randomness — the same spec MUST give the same
    bytes).

    The range is ±3 because then **rounding is the identity** for f16, i8 and i4 alike, and the
    quantization scale is not 0 (= no division by 0).
    """
    count = 1
    for dim in shape:
        count *= dim
    return (torch.arange(count, dtype=torch.float32) % 7 - 3).reshape(shape)


def _symbols(shapes: Sequence[Shape]) -> list[str]:
    """The symbols that appear in the declared shapes (derived forms such as `2T` / `S+24` fold to
    the bare symbol).

    Symbols with a binding point (a dimension position of an input shape) are **derived from the
    shapes they appear in** — having the caller declare them in a separate field would let the
    shapes and the declaration move independently (`ir_container(symbols=…)` is the fault-injection
    case that deliberately declares symbols without a binding point; only it uses the separate
    field).
    """
    found = {parse_dim(dim).sym for shape in shapes for dim in shape if isinstance(dim, str)}
    return sorted(found)


def _write(
    graph: IrGraph,
    tensors: dict[str, torch.Tensor],
    *,
    mark: str,
    named: str | None,
    storage: str,
    scales: dict[str, torch.Tensor],
    overrides: dict[str, str],
    assets: Mapping[str, AssetInput] = {},
    part_bytes: int = DEFAULT_PART_BYTES,
    block_bytes: int = BLOCK_MAX_BYTES,
    verified: bool = True,
) -> list[bytes]:
    """Write, read back and verify, and return the bytes of each part in index order.

    `verified=False` is the fault-injection case (the exception in the module doc) — it only
    writes, without the publication check.
    """
    stored = stored_model(
        graph,
        tensors,
        weight_dtype=storage,
        weight_scales=scales,
        weight_dtype_overrides=overrides,
    )
    with TemporaryDirectory() as staging:
        args = (Path(staging) / "model.krm", stored.graph, stored.tensors, stored.bindings)
        options = {
            "graph_name": named if named is not None else graph_name(mark),
            "provenance": FIXTURE_PROVENANCE,
            "assets": assets,
            "part_bytes": part_bytes,
            "block_bytes": block_bytes,
        }
        parts = (
            publish_container(*args, **options).parts
            if verified
            else write_model_container(*args, **options)
        )
        return [path.read_bytes() for path in parts]


def ir_container(
    *,
    mark: str = "fixture",
    named: str | None = None,
    storage: str = "f32",
    inputs: Sequence[tuple[str, Shape]] = (),
    outputs: Sequence[Shape] = ([1],),
    output_names: Sequence[str] | None = None,
    weights: Sequence[str] = ("weight",),
    baked: tuple[str, int] | None = None,
    symbols: Sequence[str] = (),
    assets: Mapping[str, AssetInput] = {},
    part_bytes: int = DEFAULT_PART_BYTES,
    block_bytes: int = BLOCK_MAX_BYTES,
) -> list[bytes]:
    """The part bytes of one valid container (in index order — part 0 first).

    The result **always has 3 or more elements**: part 0 (header + the 2 documents), part 1 (the
    const region), part 2 onwards (weights). The synthetic assets are small, so the weights fit in
    one part and the result has 3 elements. The caller builds the numbered file names with
    `karume.container.numbered_name` and writes them out.

    `storage` is the series' storage form (`f32` / `f16` / `bf16` / `i8` / `i4`). As in the real
    thing, **only eligible weight slots** get compressed storage; biases and graph constants (the
    i8 / i4 scales too) stay f32, so the binding table's codec set has the same shape as a real
    distribution's assets. `i4` is **mixed** (i4-eligible weights i4, the rest i8, everything
    else f32) — real i4 series have this shape, and a single-codec container cannot reproduce "a
    mix-up between compressed slots".

    `inputs` (names and shapes) and `outputs` (the declared shape of each output) are fields that
    family-specific checks read. The inputs are declaration-only fields that no node consumes,
    and the outputs are small constants passed through `expand`, so even a declaration such as
    2048×2048 stays a few dozen actual bytes. `output_names` gives the outputs' value names
    (default `out_<position>`) — a field whose spelling the checks of families that bind outputs
    **by name** read (the boundary that maps into resident tensors — irodori's `dit_context`).

    `weights` are the initializer names that become linear weights (the spelling the checks that
    count layers read). `baked` is `(symbol name, baked upper bound)` and adds one baked
    `sym_prefix_slice` constant (the symbol must be bound at a dimension position of `inputs`).

    `assets` are the assets bundled into the container (asset name →
    {@link karume.container.AssetInput}) — the field for "bytes that are not weights but are
    distributed in the same container", such as the PLE index or `rope_base`; they go after the
    weights' parts (ADR 0109 decision 4).

    `symbols` are symbols **only added** to the declaration (on top of those derived from the
    `inputs` / `outputs` shapes). This is the fault-injection case for testing the check that
    "graphs with symbolic dimensions are not distributed"; declaring a symbol without a binding
    point is a shape the real thing never has — a symbol with a binding point is simply spelled in
    an `inputs` shape. The container then violates the IR acceptance rules, so only a run that uses
    this field writes without the publication check (the exception in the module doc — families
    that reach their check with a symbol that has a binding point build it that way).

    `named` is the container's graph name = **the component name** (the weights key of
    `karume.json`). The default is the spelling derived from `mark`, and a fixture plugged into the
    weights field MUST **be named with that key** — the assembly checks that the graph name equals
    the key ({@link karume.dist.assert_weight_components_verified}), so the `mark`-derived spelling
    (`siglip2-vision` and the like) would be an input the real thing never has.

    `part_bytes` / `block_bytes` inject the sizes (the `publish_container` options) — they make the
    small synthetic weights split into **several parts** / **a piece sequence**; the defaults are
    256 MiB / 32 MiB, as in the real thing. The number of parts is decided by the actual byte
    count, so the caller does not assume a count and **observes the actual result**.
    """
    graph, tensors, scales, overrides = _spec(
        mark, storage, inputs, outputs, weights, baked, symbols, output_names
    )
    return _write(
        graph,
        tensors,
        mark=mark,
        named=named,
        storage=storage,
        scales=scales,
        overrides=overrides,
        assets=assets,
        part_bytes=part_bytes,
        block_bytes=block_bytes,
        verified=not symbols,
    )


def graph_name(mark: str) -> str:
    """Build the container's graph name from `mark` (characters outside the vocabulary
    `[A-Za-z0-9._-]` fold to `-`).

    `mark` is the tensor-key prefix, and the tests use it to vary "which model claims which bytes"
    — spellings outside the graph-name vocabulary stay usable too.
    """
    return re.sub(r"[^A-Za-z0-9._-]", "-", mark)[:64]


def fill_spec(count: int, *, mark: str) -> tuple[IrGraph, dict[str, torch.Tensor]]:
    """A graph that only holds `count` f32 tensors of the same size (material for exercising the
    split)."""
    initializers = {}
    values = {}
    tensors = {}
    names = [f"{_OWN}fill{index}" for index in range(count)]
    for name in names:
        initializers[name] = IrInitializer(tensor=f"{mark}.{name}", storage=IrStorage(dtype="f32"))
        values[name] = IrValue(dtype="f32", shape=[_FILL_ELEMENTS])
        tensors[f"{mark}.{name}"] = _ramp(_FILL_ELEMENTS)
    return IrGraph(initializers=initializers, values=values, outputs=names[:1]), tensors


def ir_parts(count: int, *, mark: str, named: str | None = None) -> list[bytes]:
    """A valid container that becomes a sequence of `count` parts (in index order — part 0 first).

    The number of parts is decided by the actual byte count, so `count - 2` tensors of the same
    size are lined up and the injected part length is set to one of them, giving **one weight part
    = one tensor**. `count` is the total including part 0 (the 2 documents) and part 1 (the const
    region — 0 bytes, since there are no consts), so it is 3 or more.
    """
    if count < 3:
        raise ValueError(
            f"part count {count} must be 3 or more"
            " (including part 0 <2 documents> + part 1 <const>)"
        )
    graph, tensors = fill_spec(count - 2, mark=mark)
    return _write(
        graph,
        tensors,
        mark=mark,
        named=named,
        storage="f32",
        scales={},
        overrides={},
        part_bytes=_FILL_ELEMENTS * 4,
    )


def fixture_spec(
    mark: str = "fixture",
    storage: str = "f32",
    inputs: Sequence[tuple[str, Shape]] = (),
    outputs: Sequence[Shape] = ([1],),
    weights: Sequence[str] = ("weight",),
    baked: tuple[str, int] | None = None,
    symbols: Sequence[str] = (),
) -> tuple[IrGraph, dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, str]]:
    """**The same material** as {@link ir_container} (graph, stored tensors, scale ledger,
    per-tensor overrides).

    The entry through which the old-distribution synthesis (`legacy_writer`) writes the same
    material into another container — splitting the material in two would break the byte identity
    of "the container migrated from the old form" and "the container written directly" through a
    difference in material.
    """
    return _spec(mark, storage, inputs, outputs, weights, baked, symbols)


def _spec(
    mark: str,
    storage: str,
    inputs: Sequence[tuple[str, Shape]],
    outputs: Sequence[Shape],
    weights: Sequence[str],
    baked: tuple[str, int] | None,
    symbols: Sequence[str],
    output_names: Sequence[str] | None = None,
) -> tuple[IrGraph, dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, str]]:
    """Build the graph, the stored tensors, the scale ledger and the per-tensor storage dtype
    overrides from the spec."""
    if output_names is not None and len(output_names) != len(outputs):
        raise ValueError(f"{len(outputs)} outputs but {len(output_names)} names")
    initializers: dict[str, IrInitializer] = {}
    values: dict[str, IrValue] = {}
    nodes: list[IrNode] = []
    tensors: dict[str, torch.Tensor] = {}
    scales: dict[str, torch.Tensor] = {}
    overrides: dict[str, str] = {}

    def declare(name: str, tensor: torch.Tensor, dtype: str = "f32") -> str:
        key = f"{mark}.{name}"
        initializers[name] = IrInitializer(tensor=key, storage=IrStorage(dtype=dtype))
        values[name] = IrValue(dtype=dtype if dtype == "i32" else "f32", shape=list(tensor.shape))
        tensors[key] = tensor
        return key

    activation = f"{_OWN}x"
    bias = f"{_OWN}bias"
    declare(activation, _ramp(1, _IN))
    # The bias is always f32 (never quantized), so it can carry a `mark`-dependent value — unless
    # the weight parts are **different bytes per mark**, the tests that observe the shared folding
    # would see "the same part for every model" (a difference in tensor keys shows only in part
    # 0's descriptor).
    declare(bias, _ramp(_OUT) + float(sum(mark.encode()) % 97))
    # The const region (part 1) is not left empty — real containers hold constants, and a 0-byte
    # part 1 would create "a slot that is byte-identical for every model" and blur the observation
    # of the shared folding. The values depend on `mark` (marks of the same length still give
    # different bytes = a subject for the sharing decision).
    const_key = f"const.{mark}"
    const_name = f"{_OWN}const"
    initializers[const_name] = IrInitializer(tensor=const_key, storage=IrStorage(dtype="f32"))
    values[const_name] = IrValue(dtype="f32", shape=[_OUT])
    tensors[const_key] = _ramp(_OUT) + float(sum(mark.encode()) % 97)
    # An i4 series is mixed (weights not eligible for i4 stay i8) — one such counterpart is held
    # here.
    rest = f"{_OWN}rest"
    for index, name in enumerate([*weights, *([rest] if storage == "i4" else [])]):
        weight = _ramp(_OUT, _IN)
        dtype = "i8" if name == rest else storage
        scale: torch.Tensor | None = None
        if dtype == "f16":
            weight = weight.to(torch.float16).to(torch.float32)
        elif dtype == "i8":
            scale = channel_scale(weight, 0)
            weight = quantize_to_int8(weight, scale).to(torch.float32) * scale
        elif dtype == "i4":
            scale = group_scale(weight, GROUP_SIZE)
            weight = dequantize_int4(quantize_to_int4(weight, scale), scale)
        key = declare(name, weight)
        if scale is not None:
            scales[key] = scale
        if name == rest:
            overrides[key] = "i8"
        out = f"{_OWN}h{index}"
        values[out] = IrValue(dtype="f32", shape=[1, _OUT])
        nodes.append(IrNode(op="linear", ins=[activation, name, bias], outs=[out], attrs={}))
        shifted = f"{_OWN}c{index}"
        values[shifted] = IrValue(dtype="f32", shape=[1, _OUT])
        nodes.append(IrNode(op="add", ins=[out, const_name], outs=[shifted], attrs={}))

    # The part **size** also varies with `mark` (marks of different lengths make parts of
    # different sizes). A tool for observing the pre-filter of the shared folding (a size mismatch
    # drops out without hashing); one is placed in the const region (part 1) and one in the weights
    # (part 2 onwards).
    #
    # MUST: the padding is **not added to the graph outputs** — family checks detect a mix-up of
    # distributions by "how many outputs there are" (BiRefNet's multi-scale supervision, SigLIP2's
    # all-layer outputs), so if the material added 2 outputs, the `outputs` expectation would be
    # decided by the material. Unconsumed values remain as declarations.
    for region, key, span in (
        # The step is 16 elements = 64 bytes (the unit of a block's start alignment) — any finer
        # and the length difference is absorbed by the alignment padding, so the part size does
        # not move.
        ("const", f"const.{mark}.pad", 16 * (1 + len(mark) % 11)),
        ("weight", f"{mark}.{_OWN}pad", 16 * (1 + len(mark) % 13)),
    ):
        name = f"{_OWN}{region}_pad"
        initializers[name] = IrInitializer(tensor=key, storage=IrStorage(dtype="f32"))
        values[name] = IrValue(dtype="f32", shape=[span])
        tensors[key] = _ramp(span)
        doubled = f"{_OWN}{region}_pad_out"
        values[doubled] = IrValue(dtype="f32", shape=[span])
        nodes.append(IrNode(op="add", ins=[name, name], outs=[doubled], attrs={}))

    declared: list[Shape] = [shape for _, shape in inputs]
    graph_outputs: list[str] = []
    for index, shape in enumerate(outputs):
        seed = f"{_OWN}seed{index}"
        declare(seed, _ramp(*([1] * len(shape))))
        out = f"out_{index}" if output_names is None else output_names[index]
        values[out] = IrValue(dtype="f32", shape=list(shape))
        nodes.append(IrNode(op="expand", ins=[seed], outs=[out], attrs={}))
        graph_outputs.append(out)
        declared.append(shape)

    if baked is not None:
        symbol, sym_max = baked
        const = f"{_OWN}baked"
        declare(const, torch.zeros(1, 1, sym_max, 1, dtype=torch.int32), dtype="i32")
        prefix = f"{_OWN}prefix"
        prefix_shape: list[int | str] = [1, 1, symbol, 1]
        values[prefix] = IrValue(dtype="i32", shape=prefix_shape)
        nodes.append(
            IrNode(
                op="sym_prefix_slice",
                ins=[const],
                outs=[prefix],
                attrs={"sym": symbol, "slices": [{"dim": 2, "coeff": 1, "offset": 0}]},
            )
        )
        declared.append(prefix_shape)

    graph = IrGraph(
        symbols=sorted({*_symbols(declared), *symbols}),
        inputs=[IrInput(name=name, dtype="f32", shape=list(shape)) for name, shape in inputs],
        outputs=graph_outputs,
        initializers=initializers,
        values=values,
        nodes=nodes,
    )
    return graph, tensors, scales, overrides


#: The op **outside the vocabulary** injected for fault injection (a spelling not in
#: `karume.ops.OP_CONTRACTS`).
#:
#: MUST: spelled with **the same byte length** (6 characters) as the `linear` it replaces, and so
#: that its code-point position in `requires.ops` / `capabilities.ops` does not change either — if
#: part 0's length moved, the failure would be "the part length differs from the declaration"
#: instead of the IR acceptance rules, and the subject would be something else.
UNKNOWN_OP = "nosuch"

#: The op being replaced (every synthetic graph has at least one).
_REPLACED_OP = "linear"


def with_an_unknown_op(parts: Sequence[bytes]) -> list[bytes]:
    """Return the part sequence with `linear` in part 0's graph document rewritten to **an op
    outside the vocabulary**.

    The structural checks (`capabilities.ops` equals the union of `graphs[].requires.ops`, the
    block index, the codec ledger) stay satisfied, so only the IR acceptance rules
    ({@link karume.verify.assert_ir_accepted}) can reject it — the subject for the hole where "a
    container declaring an op outside the vocabulary" lands in a distribution without being caught
    by the assembly or by `karume verify`.
    """
    part0, *rest = parts
    header = read_header(part0[:HEADER_BYTES])
    begin = HEADER_BYTES
    document = json.loads(part0[begin : begin + header.graph_length])
    tail = part0[begin + header.graph_length :]
    for graph in document["graphs"].values():
        for node in graph["nodes"]:
            if node["op"] == _REPLACED_OP:
                node["op"] = UNKNOWN_OP
        graph["requires"]["ops"] = sorted({node["op"] for node in graph["nodes"]})
    document["capabilities"]["ops"] = sorted(
        {op for graph in document["graphs"].values() for op in graph["requires"]["ops"]}
    )
    rewritten = canonical_json(document).encode("utf-8")
    if len(rewritten) != header.graph_length:
        raise AssertionError(
            f"the rewrite moved the graph document length {header.graph_length} →"
            f" {len(rewritten)} (the subject would turn into 'a part length mismatch')"
        )
    head = write_header(kind="model", graph_length=len(rewritten), model_length=header.model_length)
    return [head + rewritten + tail, *rest]
