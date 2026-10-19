"""The single path export → normalize → convert → storage conversion → publish.

When each export script (one script per model) spells out the order of the steps itself, missed
normalizations and skipped checks scatter across the scripts. The order is decided here, in one
place.

The distribution is a **container** (`krm` — container-v1), and the three write steps (write →
read back and verify → swap into place) belong to `karume.publish`. All this module adds is "under
which name, with which provenance and with which bundled assets it is put in place".
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from karume.container import (
    BLOCK_MAX_BYTES,
    DEFAULT_PART_BYTES,
    GRAPH_NAME_PATTERN,
    AssetInput,
    ContainerFormatError,
    Provenance,
)
from karume.convert import PRESERVED_OP_PREFIXES, convert, curated_decompositions
from karume.emit import FixedQuantizedWeight, stored_model
from karume.ir import IrGraph
from karume.normalize import normalize_graph
from karume.publish import publish_container
from karume.verify import assert_op_contracts, assert_runtime_support

#: Extension of a container (model container) (container-v1 §1 — the kind is carried by the magic,
#: but the names are kept apart too).
MODEL_SUFFIX = ".krm"


def export_module(
    module: torch.nn.Module,
    args: tuple[Any, ...],
    *,
    dynamic_shapes: Any = None,
    symbol_names: Sequence[str] = ("T",),
    preserved: Sequence[str] = PRESERVED_OP_PREFIXES,
) -> tuple[IrGraph, dict[str, torch.Tensor]]:
    """Convert an nn.Module to an IR graph + stored tensors.

    `preserved` is the set of prefixes of high-level ops that stop decomposition (12 ops by
    default). It can be replaced **per target** for fused attention (ADR 0023): only targets that
    pass `PRESERVED_OP_PREFIXES_WITH_ATTENTION` keep SDPA — widening the table globally would make
    graphs with a masked SDPA impossible to export through the fail-loudly of `_h_attention`.
    """
    ep = torch.export.export(module, args, dynamic_shapes=dynamic_shapes, strict=False)
    decomposed = ep.run_decompositions(curated_decompositions(preserved))
    normalize_graph(decomposed)
    return convert(decomposed, symbol_names=symbol_names)


def _assert_graph_name(graph_name: str) -> str:
    """Reject a container graph name outside the vocabulary (`[A-Za-z0-9._-]{1,64}`).

    MUST: no default (treated like `provenance`) — the caller writes into **the staging directory**
    (`<component>.staging/`), so defaulting to the parent directory name would make the graph name
    of every series `<component>.staging`. The vocabulary includes `.`, so this would not even fail
    loudly; it would first surface as "the container has no graph" when the runtime looks the
    component name up. Only the caller knows the component name (= the key of weights in
    `karume.json`), so the caller names it (container-v1 §2.1 — it MUST be spelled like the
    component name the migration CLI uses).
    """
    if GRAPH_NAME_PATTERN.fullmatch(graph_name) is None:
        raise ContainerFormatError(
            f"Graph name '{graph_name}' is outside the container vocabulary"
            f" ({GRAPH_NAME_PATTERN.pattern})"
        )
    return graph_name


def _assert_model_path(path: str | Path) -> Path:
    """Reject an output path that does not end in the container extension (`.krm`).

    MUST: another extension is not silently read as `.krm` — `krm` is the only distribution form
    (container-v1 §8), and a caller that passes another extension means to write a retired form.
    """
    final = Path(path)
    if final.suffix != MODEL_SUFFIX:
        raise ValueError(
            f"The extension of output path '{final}' is not {MODEL_SUFFIX}"
            " (krm is the only distribution form)"
        )
    return final


def publish_model(
    path: str | Path,
    graph: IrGraph,
    tensors: dict[str, torch.Tensor],
    *,
    provenance: Provenance,
    graph_name: str,
    weight_dtype: str = "f32",
    weight_scales: Mapping[str, torch.Tensor] | None = None,
    weight_dtype_overrides: Mapping[str, str] | None = None,
    fixed_weights: Mapping[str, FixedQuantizedWeight] | None = None,
    assets: Mapping[str, AssetInput] = {},
    single: bool = False,
    _part_bytes: int = DEFAULT_PART_BYTES,
    _block_bytes: int = BLOCK_MAX_BYTES,
) -> IrGraph:
    """Storage conversion → the three publish steps (the entry point that takes a converted graph).

    MUST: every distribution goes through **this one path**. When scripts that insert a rewrite
    (decode variants) spelled out the same steps themselves, changes on the writing side silently
    broke the copies.

    `provenance` is **required** (no default value that misstates the provenance — no distribution
    that dropped `license`). `graph_name` is **required** too, and MUST be spelled as **the
    distribution's component name = the key of weights in `karume.json`** (container-v1 §2.1) —
    the runtime looks the graph up by name with `prepareContainer(opened,
    <weights key>)`, and the migration CLI writes the same key. **It may differ from the name of
    the target directory** (for families that put the container directly under the series, the
    directory name is a series name like `siglip2-so400m-patch14-384`; for irodori it is a hyphen
    spelling like `caption-proj` — the keys are `vision` / `caption_proj` respectively), so **the
    caller names the component with a constant** (not derived from the directory). The check that
    keeps the staging name (`<component>.staging`) from being picked up is
    {@link _assert_graph_name}.
    `assets` (asset name → {@link karume.container.AssetInput}) is the field for "bytes that are
    not weights but ship in the same container", such as the PLE index or `rope_base`; they go into
    a dedicated part **after** the weight parts (ADR 0109 decision 4).

    `path` is the representative path ending in `.krm` (another extension is a `ValueError` —
    {@link _assert_model_path}). In the default split form the files are numbered
    `<stem>-NNNNN-of-NNNNN.krm` from part 0, and `path` itself is not written (the official HF
    distribution is split form only — container-v1 §8). `single=True` is the single form for local
    use.

    Returns **the graph with the storage declarations committed** (the view with the `storage`
    actually written). The `graph` passed in is not changed by a single byte. "Can what was written
    be read" is already settled by the read-back verification of `karume.publish` (payload /
    assets / reconciliation of the two documents).

    MUST: the IR acceptance rules (runtime support + op contracts) are applied **before** writing
    — scripts insert graph rewrites such as `to_states_form`, so the checks of the conversion step
    alone cannot stop a container that "can be written but the runtime cannot read". They are
    applied to the graph `stored_model` committed (the view with the storage actually written), and
    a failed run puts **not a single byte** in place.
    """
    final = _assert_model_path(path)
    stored = stored_model(
        graph,
        tensors,
        weight_dtype=weight_dtype,
        weight_scales=weight_scales,
        weight_dtype_overrides=weight_dtype_overrides,
        fixed_weights=fixed_weights,
    )
    assert_runtime_support(stored.graph)
    assert_op_contracts(stored.graph)
    publish_container(
        final,
        stored.graph,
        stored.tensors,
        stored.bindings,
        graph_name=_assert_graph_name(graph_name),
        provenance=provenance,
        assets=assets,
        single=single,
        part_bytes=_part_bytes,
        block_bytes=_block_bytes,
    )
    return stored.graph


def export_to_file(
    module: torch.nn.Module,
    args: tuple[Any, ...],
    path: str | Path,
    *,
    provenance: Provenance,
    graph_name: str,
    dynamic_shapes: Any = None,
    symbol_names: Sequence[str] = ("T",),
    weight_dtype: str = "f32",
    weight_scales: Mapping[str, torch.Tensor] | None = None,
    weight_dtype_overrides: Mapping[str, str] | None = None,
    preserved: Sequence[str] = PRESERVED_OP_PREFIXES,
    assets: Mapping[str, AssetInput] = {},
) -> IrGraph:
    """Convert, put in place as `krm`, and return the graph with the storage declarations
    committed (the single path from export to publish).

    When `weight_dtype` is `"f16"` / `"bf16"` / `"i8"` / `"i4"`, only eligible weight slots get the
    compressed storage (ADR 0018 / 0131 / 0019 / 0069). The caller MUST **apply the rounding
    (fake-quant) before taking the reference and the goldens** — a missed application fails loudly
    in `emit.stored_model`. `weight_scales` is the i8 / i4 scale ledger (the return value of
    `quantize.fake_quant_int8` / `fake_quant_int4` — for mixed storage, merge both and pass them).
    `weight_dtype_overrides` (tensor key → dtype) is an explicit per-tensor choice that takes
    precedence over the default (mixed storage — the line is drawn in `emit._plan_weight_dtype`).

    `provenance` and `graph_name` (= the distribution's component name = the key of weights in
    `karume.json`) are **required** — meaning and reasons in {@link publish_model}. Everything from
    writing on (atomicity, part splitting, clean-up) belongs there too.
    """
    # Reject the output path's extension before export (do not refuse after a heavy export).
    _assert_model_path(path)
    graph, tensors = export_module(
        module,
        args,
        dynamic_shapes=dynamic_shapes,
        symbol_names=symbol_names,
        preserved=preserved,
    )
    return publish_model(
        path,
        graph,
        tensors,
        provenance=provenance,
        weight_dtype=weight_dtype,
        weight_scales=weight_scales,
        weight_dtype_overrides=weight_dtype_overrides,
        assets=assets,
        graph_name=graph_name,
    )
