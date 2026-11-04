"""Read a written container (`krm`) back **as stored** (the recipe-side reader; ADR 0065
decision 2).

Scripts and distribution recipes cross-check their own output (typical cases: byte equality with
the upstream checkpoint ⟨vowel_detector⟩, reading weights back from an i4 series ⟨irodori⟩, the
QAT storage audit ⟨gemma4_qat⟩ — find every caller with rg). All of them need "tensor key →
declared storage form and raw payload", the shape before it is turned back into semantic f32.

It only folds the supply plan that core (`karume.verify.verify_container`) has already merged, so
it holds no copy of the rules. `ReadContainer.block` checks a block's sha256 on every read
(container-v1 §7), so a payload that went through here matches its declaration.

MUST: read **the distribution form itself** (the part list). Opening a single representative path
would, for the split form, read part 0 (only the two documents) and find "no tensors at all".
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from karume.container import CONST_KEY_PREFIX, Provenance, codec_entry
from karume.dist import component_parts
from karume.verify import VerifiedContainer, verify_container


class ContainerReadError(ValueError):
    """What the caller wants to read is not in the container (the asset is absent, or a `krg`
    was passed).

    The repository's convention is an `Error` subclass (`DistError` / `PleError` /
    `ContainerFormatError`). A `KeyError`'s `repr` wraps the message in double quotes, which makes
    the diagnostic hard to read, and the caller's `except` cannot tell it from "a failed dict
    lookup".
    """


#: The readers' entry (a representative path, or a container opened by {@link open_container}).
Source = Path | VerifiedContainer


def open_container(path: Path) -> VerifiedContainer:
    """Open one component and run it through the merge (the entry for **reusing the opened
    result**).

    MUST: a caller that reads the same container many times opens it here once — passing the
    representative path to the readers below makes `verify_container` redo the descriptor parse
    and the merge on every call (the PLE acceptance check reads per block, so it repeated this
    once per block).
    """
    return verify_container(component_parts(path))


def _opened(source: Source) -> VerifiedContainer:
    """Open a representative path; return an opened container as is."""
    return source if isinstance(source, VerifiedContainer) else open_container(source)


@dataclass(frozen=True)
class StoredTensor:
    """The stored form of one initializer (declaration + concatenated payload)."""

    #: The codec's layout (`f32` / `f16` / `bf16` / `i32` / `i8` / `i4` / `i2`).
    layout: str
    #: The declared concrete shape (the logical shape — the element count even for packed
    #: storage).
    shape: tuple[int, ...]
    #: The raw stored bytes (pieces already concatenated in row order).
    payload: bytes
    #: The raw bytes of the companion scale (f32; quantized codecs only).
    scale: bytes | None = None
    #: The group length of a quantized codec (per-channel: the row length itself).
    group_size: int | None = None
    #: Whether this is a hoisted constant (a tensor key with the `const.` prefix).
    constant: bool = False


def read_stored(source: Source) -> dict[str, StoredTensor]:
    """Turn the initializers of one container into a table keyed by **tensor key**.

    In IR v2 an initializer's name is the tensor key itself (docs/ir-v2.md), so the returned keys
    are the keys the writer passed to `tensors`. Shared declarations (slots pointing at a lender)
    have no supply and do not appear — the data is listed once, under the lender's key.
    """
    verified = _opened(source)
    read = verified.read
    stored: dict[str, StoredTensor] = {}
    for bound in verified.graphs.values():
        values = bound.declaration["values"]
        for name, supply in bound.supplies.items():
            payload = b"".join(
                read.block(block.id)[: block.payload_bytes] for block in supply.blocks
            )
            scale = (
                None
                if supply.scale is None
                else read.block(supply.scale.id)[: supply.scale.payload_bytes]
            )
            stored[name] = StoredTensor(
                layout=codec_entry(supply.encoding.codec).layout,
                shape=tuple(values[name]["shape"]),
                payload=payload,
                scale=scale,
                group_size=supply.encoding.group_size,
                constant=name.startswith(CONST_KEY_PREFIX),
            )
    return stored


def weight_tensors(stored: Mapping[str, StoredTensor]) -> dict[str, StoredTensor]:
    """The entries without hoisted constants (= only the tensor keys that map to upstream
    weights)."""
    return {key: value for key, value in stored.items() if not value.constant}


def read_layouts(source: Source) -> dict[str, str]:
    """Tensor key → storage layout (**declarations only** — not one byte of data is read).

    For comparisons that need no data, such as "is the lender's main table resident as i8". The
    IR v2 graph description has no `storage` (the source of truth is the binding table's codec —
    docs/ir-v2.md), so the merge result is the only way to learn the storage.
    """
    verified = _opened(source)
    return {
        name: codec_entry(supply.encoding.codec).layout
        for bound in verified.graphs.values()
        for name, supply in bound.supplies.items()
    }


def read_asset_declarations(source: Source) -> dict[str, tuple[str, int]]:
    """Asset name → `(role, logical length)` (**declarations only** — not one byte of payload is
    read).

    For checking, without reading data, that the blocks an index names really are in the
    container (PLE's `values` / `scales` total several GB, so the assembly check cannot read
    them).
    """
    model = _opened(source).read.model
    if model is None:
        raise ContainerReadError(f"{source}: a krg has no assets")
    return {name: (record.role, record.length) for name, record in model.assets.items()}


def read_asset(source: Source, name: str) -> bytes:
    """The payload of one asset of the container (cut to the declared logical length — trailing
    padding is stripped).

    Assets are "byte strings that are not weights but ship in the same container" (the PLE index,
    `rope_base` — ADR 0109 decision 4), and the declared `length` comes first. The rule exists so
    that consumers never strip trailing 0x00 by guessing, so readers go through this one function
    too.
    """
    read = _opened(source).read
    model = read.model
    if model is None:
        raise ContainerReadError(f"{source}: a krg has no assets")
    record = model.assets.get(name)
    if record is None:
        raise ContainerReadError(f"{source}: no asset '{name}' (declared: {sorted(model.assets)})")
    return read.block(record.block)[: record.length]


def read_provenance(source: Source) -> Provenance:
    """The provenance the container claims (`provenance` — container-v1 §2.3; reads
    **declarations only**).

    A `krg` does not carry provenance (container-v1 §9), so passing one fails.
    """
    model = _opened(source).read.model
    if model is None:
        raise ContainerReadError(f"{source}: a krg has no provenance")
    return model.provenance
