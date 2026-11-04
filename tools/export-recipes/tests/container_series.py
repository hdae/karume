"""Glue that lets the recipe tests treat a part list (what `ir_fixtures` returns) as a series /
distribution.

The distribution is always split (container-v1 §8), so `ir_fixtures.ir_container` returns not the
bytes of one file but a **part list** (part 0 = the 2 documents, part 1 = the const region, part 2
onward = weights). Placing it into a series and matching it against what assembly produced both
need the same single spelling that derives the numbered paths from the representative path — that
is the numbering convention of `karume.container`, so all this module holds is a thin call into it.

It lives in this one place because the `tests/test_distribution.py` of 7 families take the same 3
steps (write into the series / list the expected paths / read what was placed). Copying the
numbering spelling into each family would leave only the tests stale on the day the split rule
changes. `tools/export-recipes/conftest.py` puts this directory on sys.path (the same route as
`ir_fixtures`).

MUST: the representative path of a valid component itself (`model.krm`) is **not written** — a
single file next to numbered parts leaves "which one to distribute" undetermined, so
`karume.container.container_parts` fails loudly.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from karume.container import (
    Provenance,
    container_parts,
    numbered_name,
    numbered_path,
    sequence_siblings,
)
from karume.dist import WeightFiles
from karume.emit import FixedQuantizedWeight, stored_model
from karume.ir import IrGraph, IrInitializer, IrInput, IrNode, IrStorage, IrValue
from karume.publish import publish_container

#: The number of parts the return value of `ir_fixtures.ir_container` splits into (2 documents +
#: the const region + 1 weight part). The synthetic assets are a few KB, so the weights fit in one
#: part, but splitting is **always on** (container-v1 §8), so it is never a single file. If this
#: drifts, the match of the expected paths ({@link placed_paths}) fails directly.
CONTAINER_PARTS = 3


def write_component(path: Path, payload: bytes | Sequence[bytes]) -> None:
    """Place the component the representative path points at into the series (creating the parent
    directories).

    There are 2 forms, and which one is passed is exactly the distinction between "a real
    distribution form" and "a fake asset meant to fail a check":

    - `Sequence[bytes]` (what `ir_fixtures` returns) is written as **numbered parts**. In reading
      order, the first is part 0 (header + 2 documents).
    - `bytes` is written as one file at the representative path **itself**. This is the slot for a
      fake asset that checks the stop before the file is opened as a container (the plan check),
      so there is no reason to number it.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(payload, bytes):
        path.write_bytes(payload)
        return
    total = len(payload)
    for index, part in enumerate(payload, start=1):
        numbered_path(path, index, total).write_bytes(part)


def replace_component(path: Path, payload: bytes | Sequence[bytes]) -> None:
    """**Replace** a component already in place with other contents (removing the files of the
    previous form before writing).

    The slot for testing a check by swapping just one slot of the series for a mismatched asset.
    `karume.container.container_parts` fails loudly on a single file next to numbered parts, so
    unless the files of the previous form (split / single) are cleared first, the test **fails for
    another reason before it reaches the check it means to exercise**.
    """
    for stale in sequence_siblings(path):
        stale.unlink()
    write_component(path, payload)


def read_component(path: Path) -> list[bytes]:
    """Return the files of the component the representative path points at as part bytes in
    reading order.

    The inverse of `write_component`; comparable as is with what `ir_fixtures` returns
    (`list[bytes]`).
    """
    return [part.read_bytes() for part in container_parts(path)]


def i2_container(*, named: str) -> list[bytes]:
    """The part list of the smallest valid container that holds one weight as `int2-off` (layout
    `i2`).

    `ir_fixtures.ir_container` can only write `karume.emit.WEIGHT_DTYPES` (f32 / f16 / bf16 / i8 /
    i4), but the vocabulary the storage check looks at is every layout of the codec table, and only
    QAT series (fixed packed values — `FixedQuantizedWeight`) have i2. This is the slot for testing,
    **with a real binding table**, that the forbidden table fails such a series root mistaken for a
    plain f32 slot (the scale companion is f32, so "contains f32" is also true and the requirement
    check passes — the same shape as a real compressed series).
    """
    graph = IrGraph(
        symbols=[],
        inputs=[IrInput(name="ids", dtype="i32", shape=[2])],
        outputs=["y"],
        initializers={"w": IrInitializer(tensor="w", storage=IrStorage(dtype="f32"))},
        values={"w": IrValue(dtype="f32", shape=[2, 16]), "y": IrValue(dtype="f32", shape=[2, 16])},
        nodes=[IrNode(op="embedding", ins=["w", "ids"], outs=["y"], attrs={"padding_idx": -1})],
    )
    fixed = {
        "w": FixedQuantizedWeight(
            dtype="i2",
            packed=torch.zeros((2, 4), dtype=torch.uint8),
            scale=torch.ones((2, 1), dtype=torch.float32),
        )
    }
    tensors = {"w": torch.empty(2, 16, dtype=torch.float32, device="meta")}
    stored = stored_model(graph, tensors, fixed_weights=fixed)
    with TemporaryDirectory() as staging:
        result = publish_container(
            Path(staging) / "model.krm",
            stored.graph,
            stored.tensors,
            stored.bindings,
            graph_name=named,
            provenance=Provenance(license="mit"),
        )
        return [path.read_bytes() for path in result.parts]


def part_paths(rel_path: str, total: int = CONTAINER_PARTS) -> list[str]:
    """Representative relative path → the numbered paths that appear in the distribution."""
    return [numbered_name(rel_path, index, total) for index in range(1, total + 1)]


def placed_paths(
    output_paths: Mapping[str, str],
    weights: Mapping[str, Mapping[str, WeightFiles]],
    totals: Mapping[str, int] = {},
) -> list[str]:
    """Output path table → the relative paths that appear in the distribution (**only the weights
    slots** are expanded into numbered parts).

    Which slots get expanded is derived the same way as `karume.dist.expand_weight_parts` — the set
    of role names the weights declaration points at; the assets slots (single-file references) stay
    as they are. Listing by hand on the recipe side "which roles get split" would leave only the
    expected values stale on the day a slot is added.

    `totals` overrides the part count per role name (the default is {@link CONTAINER_PARTS}) — a
    container that bundles assets gains dedicated parts for them, so a family whose slots differ in
    part count declares it here.
    """
    expanded = {files.file for labels in weights.values() for files in labels.values()}
    return [
        rel
        for role, path in output_paths.items()
        for rel in (
            part_paths(path, totals.get(role, CONTAINER_PARTS)) if role in expanded else [path]
        )
    ]
