"""The three steps that put a distribution (`krm`) in place — write → read back and verify →
swap into place.

Every distribution goes through **this one path** (both the export path `karume.pipeline` and the
migration CLI `karume.migrate`). When callers spelled out the three steps themselves, a change on
the writing side (a different number of parts) silently broke the copies that "verify the
temporary path as is". The atomicity discipline and the clean-up live here too.

MUST: right after writing, the file is **read back** and verified — the check that keeps a file
that "was written but cannot be read" out of the distribution (the fail-loudly discipline of
ADR 0005). There are five verifications; 1–3 reconcile the re-read container with the material
the caller passed, 4 and 5 are additional conditions on the declaration:

1. **Payloads match**: the payload bytes of initializers and companion scales match the passed
   data by sha256. What is compared is **the payload part**, not the whole block — the trailing
   padding is bytes the writer adds and has no counterpart on the passing side (container-v1
   §4.1 / invariant 5 of §12).
2. **Assets match**: role, logical length, payload, and trailing padding of 0x00.
3. **The two documents**: byte length and sha256 of the graph descriptor / model descriptor (the
   same fact as the value `container.descriptor` of manifest `karume/5` declares — ADR 0109
   decision 3).
4. **`ternary` codes**: every 2-bit code of the payload is in `{1, 2, 3}` (container-v1 §6.3 —
   the same check as `karume verify`. The TS reader fails at load time, so fail before putting it
   in place).
5. **The IR acceptance rules**: every declared graph passes the whole-graph structural checks
   (duplicate outputs, SSA, forward references, symbol binding) and the op vocabulary and op
   contracts (the same `verify.assert_ir_accepted` as `karume verify` / `karume dist`). Partial
   checks per caller alone would let an IR with graph rewrites in the "every node is right but
   the whole is broken" shape be published.

MUST: to make the check effective, writing and verification run on **a temporary file in the same
directory** (`.partial`), and only once they pass is it put under its final name with
`os.replace` (same directory, so the replacement is atomic). A run that fails in the middle of the
swap **leaves nothing** either — a half-published container would stop the next run with "the
previous run's artifacts are still there" (container-v1 §12).

MUST: once in place, **the leftovers of the previous output** (what a run with a different number
of parts left behind) are deleted. The swap is atomic only per file, so without the deletion "a
container mixing the previous and the current form" would remain, and resolving
`container_parts` would stop with a fail-loudly.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Buffer, Mapping
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from karume.container import (
    BLOCK_MAX_BYTES,
    DEFAULT_PART_BYTES,
    AssetInput,
    DocumentRef,
    Encoding,
    Provenance,
    ReadContainer,
    container_paths,
    read_container,
    sequence_siblings,
    write_model_container,
)
from karume.ir import IrGraph
from karume.verify import (
    BoundGraph,
    ContainerError,
    _assert_ternary_payloads,
    assert_ir_accepted,
    bind_graphs,
)


class PublishError(ValueError):
    """A precondition for putting in place was broken (the re-read container disagrees with the
    material passed)."""


@dataclass(frozen=True)
class PublishResult:
    """The result of putting one component in place."""

    #: The `krm` parts put in place (from part 0; one for the single form).
    parts: tuple[Path, ...]
    #: The `krg` written alongside (`None` when not written).
    graph: Path | None
    #: Number of initializers a supply plan was built for.
    initializers: int
    #: Number of payloads compared by sha256 with the passed data (data + companion scales).
    payloads: int
    #: Number of assets compared.
    assets: int
    #: `(byte length, sha256)` of the two documents (graph descriptor / model descriptor).
    descriptor: tuple[DocumentRef, DocumentRef]


def _digest(raw: Buffer) -> str:
    return hashlib.sha256(memoryview(raw).cast("B")).hexdigest()


def _assert_payloads_match(
    read_back: ReadContainer,
    bound: BoundGraph,
    bindings: Mapping[str, Encoding],
    tensors: Mapping[str, Buffer],
) -> int:
    """Compare the payloads of the written container with the passed data (verification 1 of the
    module doc).

    The bytes are re-read through {@link karume.container.ReadContainer.block}, so the sha256 of
    each block is verified at the same time.
    """
    checked = 0
    for name, supply in sorted(bound.supplies.items()):
        digest = hashlib.sha256()
        for block in supply.blocks:
            digest.update(read_back.block(block.id)[: block.payload_bytes])
        _assert_digest(f"initializer '{name}'", digest.hexdigest(), tensors, name)
        checked += 1
        scale_key = bindings[name].scale_key
        # MUST: a mismatch where only one side is `None` is not let through — it is exactly the
        # shape of the passed bindings and the re-read container's binding table drifting apart,
        # and letting it through would let the counterpart of "declared i4 / data i8" (a slot that
        # should have a scale put in place without one) slip past the check.
        if (supply.scale is None) != (scale_key is None):
            raise PublishError(
                f"initializer '{name}': the passed binding"
                f" {'has' if scale_key is not None else 'has no'} scale, but the written"
                f" container's supply {'has' if supply.scale is not None else 'has no'} scale"
            )
        if supply.scale is not None and scale_key is not None:
            raw = read_back.block(supply.scale.id)[: supply.scale.payload_bytes]
            _assert_digest(
                f"initializer '{name}', scale '{scale_key}'",
                hashlib.sha256(raw).hexdigest(),
                tensors,
                scale_key,
            )
            checked += 1
    return checked


def _assert_digest(where: str, actual: str, tensors: Mapping[str, Buffer], key: str) -> None:
    expected = _digest(tensors[key])
    if actual != expected:
        raise PublishError(
            f"{where}: the payload sha256 differs from the passed data"
            f" (passed {expected} / written {actual})"
        )


def _assert_assets_match(read_back: ReadContainer, assets: Mapping[str, AssetInput]) -> int:
    """Compare the assets of the written container with the passed payloads (verification 2 of
    the module doc).

    What is compared is **the payload part**, and the trailing padding is checked to be 0x00 (per
    §4.1 the padding is bytes the writer adds, so it has no counterpart on the passing side).
    """
    model = read_back.model
    if model is None:  # pragma: no cover - present, a krm was just read back
        raise PublishError("A krg cannot carry assets")
    for name in sorted(assets):
        asset = assets[name]
        binding = model.assets.get(name)
        if binding is None:
            raise PublishError(f"Asset '{name}' is not in the written container's declaration")
        if binding.role != asset.role:
            raise PublishError(f"Asset '{name}': role is '{binding.role}' (passed '{asset.role}')")
        if binding.length != asset.length:
            raise PublishError(
                f"Asset '{name}': declared logical length is {binding.length}"
                f" (passed {asset.length})"
            )
        raw = memoryview(read_back.block(binding.block))
        payload = asset.payload
        expected = memoryview(payload() if callable(payload) else payload).cast("B")
        if expected.nbytes != asset.length:
            raise PublishError(
                f"Asset '{name}': the re-fetched data is {expected.nbytes} bytes"
                f" (declared {asset.length}) — it MUST return the same bytes on every fetch"
            )
        if raw[: expected.nbytes] != expected:
            raise PublishError(f"Asset '{name}': payload differs from the passed bytes")
        if bytes(raw[expected.nbytes :]) != b"\x00" * (raw.nbytes - expected.nbytes):
            raise PublishError(f"Asset '{name}': trailing padding is not 0x00")
    return len(assets)


def publish_container(
    final: Path,
    graph: IrGraph,
    tensors: Mapping[str, Buffer],
    bindings: Mapping[str, Encoding],
    *,
    graph_name: str,
    provenance: Provenance,
    assets: Mapping[str, AssetInput] = {},
    single: bool = False,
    graph_path: Path | None = None,
    part_bytes: int = DEFAULT_PART_BYTES,
    block_bytes: int = BLOCK_MAX_BYTES,
) -> PublishResult:
    """Write and verify a `krm` and put it in place at `final` (single form) / its numbered
    sequence (split form).

    `tensors` is a **tensor key → raw bytes** mapping; each tensor is fetched and released one at
    a time (with a lazy `Mapping`, the whole set never sits in memory). Tensors are **fetched more
    than once** — split form: one write + one read-back verification; single form: two passes by
    the writer (one to take the sha256, one to write) + one read-back verification — so the same
    key MUST return the same bytes every time.

    `part_bytes` / `block_bytes` inject sizes (the option that lets tests hit part crossings and
    piece splitting with small assets). The defaults are container-v1 §4.2's default (256 MiB) for
    `part_bytes` and the block limit of §4.1 / §10 (32 MiB) for `block_bytes`.

    MUST: the caller puts the result into **the staging directory** (inside
    `karume.artifacts.staged_publication`). A run that fails in the middle of the swap closes by
    "deleting the parts it was putting in place and leaving nothing", so when `final` is a
    committed location (where the previous artifacts are) **the previous last-known-good is
    deleted** — a caller that wants an atomic swap there writes into the staging directory and
    uses the side that swaps the whole directory.
    """
    staged = final.with_name(f"{final.stem}.{uuid4().hex}.partial{final.suffix}")
    replaced: list[Path] = []
    published: list[Path] = []
    staged_graph: Path | None = None
    try:
        written = write_model_container(
            staged,
            graph,
            tensors,
            bindings,
            graph_name=graph_name,
            provenance=provenance,
            assets=assets,
            part_bytes=part_bytes,
            block_bytes=block_bytes,
            single=single,
        )
        read_back = read_container(written)
        # MUST: the whole-graph structural checks (duplicate outputs, SSA, forward references,
        # symbol binding) and the IR acceptance rules are applied once to the re-read declaration.
        # Every path — export, graph rewrites, migration — goes through the same one at the
        # publish boundary — partial checks per caller alone could publish a container the
        # runtime's parser rejects.
        try:
            assert_ir_accepted(read_back)
        except ContainerError as cause:
            raise PublishError(
                f"The re-read container does not pass the IR acceptance rules: {cause}"
            ) from cause
        bound = bind_graphs(read_back.graph, read_back.model)[graph_name]
        payloads = _assert_payloads_match(read_back, bound, bindings, tensors)
        # Additional condition on the declaration (verification 4 of the module doc) — goes
        # through the same function as `karume verify`.
        _assert_ternary_payloads(read_back, {graph_name: bound})
        checked = _assert_assets_match(read_back, assets)
        documents = (
            DocumentRef(
                len(read_back.graph_descriptor_bytes),
                hashlib.sha256(read_back.graph_descriptor_bytes).hexdigest(),
            ),
            DocumentRef(
                len(read_back.model_descriptor_bytes),
                hashlib.sha256(read_back.model_descriptor_bytes).hexdigest(),
            ),
        )
        published = [final] if single else list(container_paths(final, len(written)))
        # MUST: the `krg` is extracted **before** the swap (the reader points at the parts under
        # the temporary path). It goes through the same steps as `krm` (temporary name →
        # `os.replace`) — writing straight to the final name would leave **a truncated `krg`**
        # under the final name in a run that fails midway (if the process is killed, the
        # clean-up on the exception path does not run either).
        if graph_path is not None:
            staged_graph = graph_path.with_name(f"{graph_path.name}.{uuid4().hex}.partial")
            staged_graph.write_bytes(read_back.extract_graph())
            os.replace(staged_graph, graph_path)
            # Once in place, the temporary name is gone (the clean-up only covers "the final names
            # this call wrote").
            staged_graph = None
            replaced.append(graph_path)
        for staged_part, target in zip(written, published, strict=True):
            os.replace(staged_part, target)
            replaced.append(target)
    except BaseException:
        # A write that failed midway has no return value, so the leftovers are found from **the
        # name pattern** of the temporary path.
        for leftover in sequence_siblings(staged):
            leftover.unlink(missing_ok=True)
        if staged_graph is not None:
            staged_graph.unlink(missing_ok=True)
        # MUST: only the files **this call put in place** (`replaced`) are deleted — deleting
        # `graph_path` unconditionally would take **the previous `krg`** down with it in a run
        # that failed before the `krg` was extracted.
        for target in replaced:
            target.unlink(missing_ok=True)
        raise
    for stale in sequence_siblings(final):
        if stale not in published:
            stale.unlink()
    return PublishResult(
        tuple(published), graph_path, len(bound.supplies), payloads, checked, documents
    )
