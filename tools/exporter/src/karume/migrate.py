"""Old distribution (safetensors shard sequences) → container format (`krm` / `krg`)
(container-v1 §12).

    # One component (the step 1 surface)
    karume migrate ../../models/karume-depth-anything-v2/small/depth/model.f32.safetensors \
        --out /tmp/migrated --license apache-2.0
    # Whole repository (the step 2 surface — writes up to the `karume/5` karume.json; ADR 0109
    # decision 8)
    karume migrate --manifest ../../models/karume-irodori/karume.json \
        --out /tmp/karume-irodori --license apache-2.0

**Reading the old format lives only in this layer** (§12) — the new readers read only `karume/5`
and the new container, and dual reading is not implemented. What this moves is only **the
container and the shape of the declarations**:

- **Not one byte of the raw payload changes**. Tail padding is **baked anew** as §4.1 says, so
  what is compared is not the whole block but the payload of each initializer (sha256 of the old
  shard's bytes = sha256 of the payload part of the new block).
- **The IR is re-serialized from v1 to v2** (verbatim identity is not kept). Renaming and canonical
  serialization: {@link karume.container.ir_v2_document}.
- **Codecs are mapped to the ledger** (`i8 → int8-sym` / `i4 → int4-sym-g` / `i2 → int2-off`).
  Nothing is mapped to `ternary` — even when the value range is a subset, the claim "this is
  ternary" is the quantizer's to make.
- **`rowAxis` is taken from the consuming op** ({@link karume.container.weight_channel_axes} —
  only `conv_transpose1d` is 1). The old keepdim-shaped scale is confirmed against the actual
  declaration to be **exactly the shape of that axis** before writing (matching byte counts alone
  would let a per-column scale be declared as per-channel).

MUST: put in place **only after passing** the self-check (re-read what was written and compare
sha256 per initializer) — writing goes to a temporary path (`.partial`), and only a run whose check
passed is moved to the final name with `os.replace`. **Old inputs are only read**, never deleted
or rewritten.

Whole-repository mode (`--manifest`) adds four things (ADR 0109 decisions 3 / 4 / 8):

- Shard sequences are built **from the old manifest's declarations** (no reconstruction from a
  representative path — this resolves sequences that cross directories). A sequence that several
  models point at is converted only once.
- The old `extras` (only `rope_base` exists) and the **PLE sidecar** are folded into container
  assets. PLE becomes an index (`ple_index` — schema 3) and block sequences of `values` /
  `scales`, and disappears from the manifest's `assets`.
- Sequences of cross-repository references (`repo` / `revision` on the FileRef) are not
  converted; `container` is copied from the `karume/5` of the **converted directory** that
  `--cross-repo` points at, and `repo` / `revision` are added.
- Ordinary files that were not converted (README / LICENSE / tokenizer / …) are copied as is, and
  the `karume/5` `karume.json` is written.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Any

from karume.artifacts import staged_publication
from karume.container import (
    BLOCK_MAX_BYTES,
    DEFAULT_PART_BYTES,
    GRAPH_NAME_PATTERN,
    PART_LENGTH_CHOICES,
    SHA256_HEX_PATTERN,
    AssetInput,
    ContainerFormatError,
    DocumentRef,
    Encoding,
    Provenance,
    base_path,
    codec_entry,
    concrete_shape,
    container_bindings,
    sequence_siblings,
)
from karume.dist import (
    REPO_RE,
    REVISION_RE,
    DistError,
    assert_manifest_limits,
    assert_rel_path,
    file_ref,
    generator_tag,
    manifest_text,
    sha256_file,
)
from karume.ir import IrGraph
from karume.legacy import (
    IR_METADATA_KEY,
    LEGACY_SCALE_DTYPE,
    LEGACY_STORED_DTYPE,
    SourceTensor,
    _read_metadata,
    payload_chunks,
    read_component,
    resolve_shards,
    safetensors_header,
)
from karume.ple import PLE_INDEX_ASSET, PLE_PACK_FACTOR, PleError, ple_assets, ple_row_bytes
from karume.publish import PublishError, PublishResult, publish_container
from karume.verify import assert_reader_layout, parse_ir_graph

#: Extensions of the new container (§1 — the kind is carried by the magic, but the file names are
#: kept apart too).
MODEL_SUFFIX = ".krm"
GRAPH_SUFFIX = ".krg"

#: Format identifier of the old manifest that is read (no dual reading — ADR 0109 decision 1).
LEGACY_MANIFEST_FORMAT = "karume/4"

#: Format identifier of the manifest that is written (ADR 0109 decision 1).
MANIFEST_FORMAT = "karume/5"

#: File name of the manifest (ADR 0041 decision 1 — a fixed name at the repository root).
MANIFEST_FILE = "karume.json"

#: Old `extras` name → container `(asset name, role)`. Only `rope_base` exists (ADR 0109
#: decision 4); an unknown name fails loudly instead of being **silently dropped**.
EXTRA_ASSETS: Mapping[str, tuple[str, str]] = {"rope_base": ("rope_base", "rope-base")}

#: Owner of the PLE sidecar (pipeline name → name of the component it folds into — ADR 0109
#: decision 4 / 0085).
PLE_OWNER: Mapping[str, str] = {"gemma4": "model", "gemma4-qat": "model"}

#: Metadata key of the old PLE shard (its consistency with the index is cross-checked here).
PLE_METADATA_KEY = "karume_ple"

#: Unit of the CLI's `--part-bytes` (MiB — the choice set is exact integer multiples of MiB).
_MIB = 1024 * 1024


class MigrateError(ValueError):
    """A migration precondition broke (storage with no mapping target, declaration and file
    disagree, leftovers in the output directory)."""


def _assert_part_length(part_bytes: int) -> None:
    """The part length MUST be one of **the writer's choice set** (container-v1 §4.2 / ADR 0108).

    The writer ({@link karume.container.write_model_container}) enforces only the ceiling, so the
    set check lives here — silently passing an odd value under the ceiling (300 MiB, say) would
    let a rebuild meant to compare part lengths mix an out-of-set container into the distribution.
    """
    if part_bytes not in PART_LENGTH_CHOICES:
        choices = ", ".join(str(choice // _MIB) for choice in PART_LENGTH_CHOICES)
        raise MigrateError(
            f"Part length {part_bytes} bytes is outside the writer's choice set {{{choices}}} MiB"
            " — container-v1 §4.2 / ADR 0108"
        )


#: Migration result of one component (the part sequence put in place, the 2 documents, the number
#: of tensors compared). The 3 steps of publication are shared with `karume.publish`, so the
#: result type is the same too.
MigrationResult = PublishResult


class _SourcePayloads(Mapping[str, bytes]):
    """Lazy mapping tensor key → raw bytes (reads **only the one tensor looked up**).

    The writer ({@link karume.container.write_model_container}) looks up the bytes one tensor at
    a time and releases them one at a time, so keeping this entry lazy keeps the whole from
    landing in memory (even for a series of tens of GB the peak is "the largest single tensor").
    """

    def __init__(self, sources: Mapping[str, SourceTensor]) -> None:
        self._sources = sources

    def __getitem__(self, key: str) -> bytes:
        return b"".join(payload_chunks(self._sources[key]))

    def __iter__(self) -> Iterator[str]:
        return iter(self._sources)

    def __len__(self) -> int:
        return len(self._sources)


def _concrete_shape(graph: IrGraph, name: str, where: str) -> list[int]:
    """Declared shape of an initializer (a translation layer over
    {@link karume.container.concrete_shape}).

    The body is **the same single path** as the writer's; this adds only a translation of the
    exception vocabulary (the same shape as {@link _bindings}) — callers of the migration CLI
    catch only `MigrateError`.
    """
    try:
        return concrete_shape(graph, name, where)
    except ContainerFormatError as cause:
        raise MigrateError(str(cause)) from cause


def _bindings(graph: IrGraph) -> dict[str, Encoding]:
    """Old IR `storage` → container bindings.

    MUST: the derivation goes through **the same single path** as the writer's
    ({@link karume.container.container_bindings}). Spelling it separately on the migration side
    would give different bindings from the same asset for "a container migrated from the old
    form" and "a container written directly". This adds only a translation of the exception
    vocabulary.
    """
    try:
        return container_bindings(graph)
    except ContainerFormatError as cause:
        raise MigrateError(str(cause)) from cause


def _expected_scale_shape(shape: Sequence[int], encoding: Encoding) -> list[int]:
    """The scale shape the old distribution should have (the shape old IR v1 accepted — the
    keepdim shape was retired in IR v2, and only this migration CLI knows it)."""
    assert encoding.group_size is not None and encoding.row_axis is not None
    if codec_entry(encoding.codec).grouping == "group":
        # The group shape is exactly rank 2 (rows = leading dimension, last dimension = row length
        # / group length).
        return [shape[0], math.prod(shape[1:]) // encoding.group_size]
    # Per-channel is the keepdim broadcast shape (only the `rowAxis` axis is extended).
    return [dim if axis == encoding.row_axis else 1 for axis, dim in enumerate(shape)]


def _assert_scale_layouts(
    graph: IrGraph, bindings: Mapping[str, Encoding], sources: Mapping[str, SourceTensor]
) -> None:
    """Check that the **shape** of the old scale matches the new declaration (`rowAxis` /
    `groupSize`).

    MUST: matching byte counts is not enough (the writer looks no further) — for a square weight a
    `[1, N]` per-column scale passes as a per-channel scale with `rowAxis: 0`, and the channel
    values are silently swapped. Compare down to the file's **declared shape** before writing.
    """
    for name, initializer in graph.initializers.items():
        if initializer.is_shared or initializer.tensor is None:
            continue
        encoding = bindings[initializer.tensor]
        if encoding.scale_key is None:
            continue
        where = f"initializer '{name}'"
        source = sources.get(encoding.scale_key)
        if source is None:
            raise MigrateError(f"{where}: scale tensor '{encoding.scale_key}' is not in the file")
        if source.entry.dtype != LEGACY_SCALE_DTYPE:
            raise MigrateError(
                f"{where}: scale '{encoding.scale_key}' has dtype {source.entry.dtype}, which"
                f" differs from the old writer's scale spelling {LEGACY_SCALE_DTYPE}"
            )
        shape = _concrete_shape(graph, name, where)
        expected = _expected_scale_shape(shape, encoding)
        if list(source.entry.shape) != expected:
            raise MigrateError(
                f"{where}: scale '{encoding.scale_key}' has shape {list(source.entry.shape)},"
                f" which differs from the shape for codec '{encoding.codec}', rowAxis"
                f" {encoding.row_axis}, groupSize {encoding.group_size}: {expected}"
            )


def _assert_stored_declarations(graph: IrGraph, sources: Mapping[str, SourceTensor]) -> None:
    """Check that the dtype / shape of the old tensors match the IR declaration (the old writer's
    rules).

    MUST: matching byte lengths is not enough (the writer's payload check looks no further) — a
    different type of the same byte length, or a different shape with only the same element count,
    would pass ({@link karume.legacy.LEGACY_STORED_DTYPE}). The scale's dtype and shape are checked
    by {@link _assert_scale_layouts}.
    """
    for name, initializer in graph.initializers.items():
        if initializer.is_shared or initializer.tensor is None:
            continue
        where = f"initializer '{name}'"
        entry = sources[initializer.tensor].entry
        storage = initializer.storage.dtype
        expected = LEGACY_STORED_DTYPE.get(storage)
        if expected is None:
            raise MigrateError(
                f"{where}: storage '{storage}' is a dtype the old writer did not write"
                f" (only {' / '.join(LEGACY_STORED_DTYPE)}) — cannot be read as the old"
                " distribution"
            )
        if entry.dtype != expected:
            raise MigrateError(
                f"{where}: old tensor '{initializer.tensor}' has dtype {entry.dtype}, which"
                f" differs from the old writer's spelling for storage '{storage}': {expected}"
            )
        shape = _concrete_shape(graph, name, where)
        if list(entry.shape) != shape:
            raise MigrateError(
                f"{where}: old tensor '{initializer.tensor}' has shape {list(entry.shape)}, which"
                f" differs from the declared shape {shape} (the old writer writes the logical"
                " shape as is)"
            )


def _assert_tensor_cover(
    bindings: Mapping[str, Encoding], sources: Mapping[str, SourceTensor]
) -> None:
    """Confirm, **by enumerating every entry**, that every tensor of the old container moves.

    MUST: surplus is rejected too (not silently left behind) — the binding table is derived from
    the declarations, so bytes not in the declarations have no place in the new container. The old
    distribution may be the one in error; either way, do not produce "bytes went missing in what
    was meant to be a migration".
    """
    referenced = set(bindings) | {
        encoding.scale_key for encoding in bindings.values() if encoding.scale_key is not None
    }
    missing = sorted(referenced - set(sources))
    surplus = sorted(set(sources) - referenced)
    if missing or surplus:
        raise MigrateError(
            "The old container's tensors and the declarations do not match:"
            f" missing [{', '.join(missing)}] / surplus [{', '.join(surplus)}]"
        )


def _assert_no_leftovers(final: Path, graph_path: Path | None) -> None:
    """No artifact of a previous run remains in the output directory (**stops without deleting**
    — which one to distribute would be undetermined).

    MUST: only migration has this check (re-running an export may swap out the previous output,
    but migration holds only under "old inputs are only read, the output is built from empty", so
    leftovers are not silently deleted).
    """
    existing = [str(path) for path in sequence_siblings(final)]
    if graph_path is not None and graph_path.is_file():
        existing.append(str(graph_path))
    if existing:
        raise MigrateError(
            "Artifacts of a previous run remain in the output directory (delete them and retry):"
            f" {', '.join(existing)}"
        )


def migrate_component(
    path: str | Path,
    out_dir: str | Path,
    *,
    provenance: Provenance,
    graph_name: str | None = None,
    assets: Mapping[str, AssetInput] = {},
    single: bool = False,
    write_graph: bool = False,
    part_bytes: int = DEFAULT_PART_BYTES,
    _block_bytes: int = BLOCK_MAX_BYTES,
) -> MigrationResult:
    """Migrate one component (old single form / old shard sequence) to `krm`.

    `path` may be the representative path or a local file (`...-00001-of-00002.safetensors`)
    ({@link karume.container.base_path} folds it). `graph_name` defaults to **the parent directory
    name** (the distribution's component name); pass it explicitly when that falls outside the
    container vocabulary (`[A-Za-z0-9._-]{1,64}`).

    `assets` (asset name → {@link karume.container.AssetInput}) is the field that folds byte
    sequences **scattered next to** the old shard sequence (PLE sidecar, `rope_base`) into the
    same container. Whole-repository mode builds them itself from the old manifest's declarations
    ({@link _convert_unit}), so this is used by callers that migrate a location without a manifest
    (a series directory) — how to fold (which file is which asset name) can only be decided by
    the side that knows the family, so core only receives it (ADR 0065).

    `part_bytes` is the part length (one of the writer's choice set
    {@link karume.container.PART_LENGTH_CHOICES} — outside the set is `MigrateError`).
    `_block_bytes` is a size injection **touched only by tests** (to exercise piece splitting with
    small synthetic assets) — not a public knob.
    """
    _assert_part_length(part_bytes)
    source = base_path(Path(path))
    name = graph_name if graph_name is not None else source.parent.name
    if GRAPH_NAME_PATTERN.fullmatch(name) is None:
        raise MigrateError(
            f"Graph name '{name}' falls outside the container vocabulary"
            f" ({GRAPH_NAME_PATTERN.pattern}) — pass it explicitly with `--graph-name`"
        )
    final = Path(out_dir) / f"{source.stem}{MODEL_SUFFIX}"
    return _migrate_shards(
        resolve_shards(source),
        final,
        provenance=provenance,
        graph_name=name,
        assets=assets,
        single=single,
        graph_path=final.with_suffix(GRAPH_SUFFIX) if write_graph else None,
        part_bytes=part_bytes,
        block_bytes=_block_bytes,
    )


def _migrate_shards(
    shard_paths: Sequence[Path],
    final: Path,
    *,
    provenance: Provenance,
    graph_name: str,
    assets: Mapping[str, AssetInput] = {},
    single: bool = False,
    graph_path: Path | None = None,
    part_bytes: int = DEFAULT_PART_BYTES,
    block_bytes: int = BLOCK_MAX_BYTES,
) -> MigrationResult:
    """Migrate an **explicit shard sequence** to one `krm` (the path shared by per-component mode
    and whole-repository mode).

    `shard_paths` is in read order (the graph shard first). The caller builds the sequence;
    reconstruction from a representative path ({@link karume.legacy.resolve_shards}) or from the
    old manifest's declarations is done before reaching here.

    MUST: the 3 steps write → re-read and verify → swap go through **the same single path** as the
    export path ({@link karume.publish.publish_container}). What this adds is only reading the old
    form and checking the assumptions that exist only in the old form (scale shape, missing /
    surplus tensors, an empty output directory).
    """
    metadata, stored = read_component(shard_paths)
    graph = parse_ir_graph(metadata[IR_METADATA_KEY])
    bindings = _bindings(graph)
    _assert_tensor_cover(bindings, stored)
    _assert_stored_declarations(graph, stored)
    _assert_scale_layouts(graph, bindings, stored)
    _assert_no_leftovers(final, graph_path)
    try:
        return publish_container(
            final,
            graph,
            _SourcePayloads(stored),
            bindings,
            graph_name=graph_name,
            provenance=provenance,
            assets=assets,
            single=single,
            graph_path=graph_path,
            part_bytes=part_bytes,
            block_bytes=block_bytes,
        )
    except PublishError as cause:
        raise MigrateError(str(cause)) from cause


# ---- Reading the old manifest (`karume/4`) -----------------------------------
#
# Only this layer reads the old format (§12), so every entry is checked here, **separately** from
# the hub's parse. Unknown keys fail loudly — silently dropping them would give "a field vanished
# in what was meant to be a migration".


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise MigrateError(message)


def _object(value: Any, where: str) -> dict[str, Any]:
    _require(isinstance(value, dict), f"{where} is not an object")
    return value


def _keys(
    obj: Mapping[str, Any], required: Sequence[str], optional: Sequence[str], where: str
) -> None:
    for key in required:
        _require(key in obj, f"{where}.{key} is missing")
    known = {*required, *optional}
    for key in obj:
        _require(key in known, f"{where}: unknown key '{key}'")


def _text(value: Any, where: str) -> str:
    _require(isinstance(value, str) and value != "", f"{where} is not a non-empty string")
    return value


def _rel_path(value: Any, where: str) -> str:
    """Path of a FileRef (the same allow-list as the hub's `assertPath` —
    {@link karume.dist.assert_rel_path}).

    MUST: do not pass it on "a non-empty string" alone. The destination joins the old path's parent
    directory onto the output directory as is ({@link _output_location}), so passing `../` or an
    absolute path would write the new container outside the output directory and put a path the
    hub rejects into the manifest.
    """
    try:
        return assert_rel_path(_text(value, where), where)
    except DistError as cause:
        raise MigrateError(str(cause)) from cause


def _sha256(value: Any, where: str) -> str:
    _require(
        isinstance(value, str) and SHA256_HEX_PATTERN.fullmatch(value) is not None,
        f"{where} is not a sha256 (64 lowercase hex characters): {value!r}",
    )
    return value


def _count(value: Any, where: str) -> int:
    _require(
        isinstance(value, int) and not isinstance(value, bool) and value >= 0,
        f"{where} is not a non-negative integer: {value!r}",
    )
    return value


@dataclass(frozen=True)
class FileRef:
    """The triple of ADR 0038 §2 (+ the cross-repository `repo` / `revision`)."""

    path: str
    size: int
    sha256: str
    repo: str | None = None
    revision: str | None = None

    @property
    def cross(self) -> tuple[str, str] | None:
        """`(repo, revision)` for a cross-repository reference, `None` for a same-repository
        reference."""
        return None if self.repo is None else (self.repo, self.revision or "")

    def to_document(self) -> dict[str, Any]:
        document: dict[str, Any] = {"path": self.path, "size": self.size, "sha256": self.sha256}
        if self.repo is not None:
            document["repo"] = self.repo
        if self.revision is not None:
            document["revision"] = self.revision
        return document


def _file_ref(value: Any, where: str) -> FileRef:
    obj = _object(value, where)
    _keys(obj, ["path", "size", "sha256"], ["repo", "revision"], where)
    ref = FileRef(
        _rel_path(obj["path"], f"{where}.path"),
        _count(obj["size"], f"{where}.size"),
        _sha256(obj["sha256"], f"{where}.sha256"),
        None if "repo" not in obj else _text(obj["repo"], f"{where}.repo"),
        None if "revision" not in obj else _text(obj["revision"], f"{where}.revision"),
    )
    _require(
        (ref.repo is None) == (ref.revision is None),
        f"{where}: a cross-repository reference is written as the pair `repo` and `revision`"
        " (one alone does not determine the target)",
    )
    return ref


@dataclass(frozen=True)
class WeightEntry:
    """The files of one (component, dtype) (`{shards, extras?}` of `karume/4`)."""

    shards: tuple[FileRef, ...]
    extras: tuple[tuple[str, FileRef], ...]

    @property
    def refs(self) -> tuple[FileRef, ...]:
        return (*self.shards, *(ref for _, ref in self.extras))


#: Spelling of `pipeline`, `<name>/<major>` (the same accepted form as the hub's `parsePipeline` —
#: the actual value is a string, not a `{name, major}` object). MUST: match with `.fullmatch` —
#: Python's `$` also matches just before a trailing newline, so `.match` would accept
#: `'name/1\n'`, which the hub rejects.
PIPELINE_RE = re.compile(r"^([A-Za-z0-9_-]+)/(\d+)$")


@dataclass(frozen=True)
class LegacyModel:
    #: The `<name>/<major>` string as is (copied to karume/5 with this spelling too).
    pipeline: str
    weights: Mapping[str, Mapping[str, WeightEntry]]
    assets: Mapping[str, FileRef]
    quants: Mapping[str, Any]
    default_quant: str
    pipeline_config: Mapping[str, Any]


@dataclass(frozen=True)
class LegacyManifest:
    default_model: str
    models: Mapping[str, LegacyModel]


def _read_json(path: Path, where: str) -> Any:
    _require(path.is_file(), f"{where} does not exist: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as cause:
        raise MigrateError(f"{where} cannot be read as JSON: {path}") from cause


def read_legacy_manifest(path: Path) -> LegacyManifest:
    """Read an old `karume/4` `karume.json` (a different major fails loudly — no dual reading)."""
    where = MANIFEST_FILE
    root = _object(_read_json(path, where), where)
    # MUST: put the version check before the field checks (for a manifest of another major,
    # saying "that version is not read" rather than "the fields differ" is decisive for whoever
    # fixes it).
    _require(
        root.get("format") == LEGACY_MANIFEST_FORMAT,
        f"{where}.format is {root.get('format')!r} — only"
        f" '{LEGACY_MANIFEST_FORMAT}' can be migrated (ADR 0109 decision 1)",
    )
    _keys(root, ["format", "generator", "defaultModel", "models"], [], where)
    models: dict[str, LegacyModel] = {}
    for name, raw in _object(root["models"], f"{where}.models").items():
        models[name] = _legacy_model(raw, f"{where}.models['{name}']")
    _require(len(models) > 0, f"{where}.models is empty")
    default_model = _text(root["defaultModel"], f"{where}.defaultModel")
    _require(default_model in models, f"{where}.defaultModel '{default_model}' is not in models")
    return LegacyManifest(default_model, models)


def _legacy_model(value: Any, where: str) -> LegacyModel:
    obj = _object(value, where)
    _keys(
        obj,
        ["pipeline", "weights", "assets", "quants", "defaultQuant", "pipelineConfig"],
        [],
        where,
    )
    pipeline = _text(obj["pipeline"], f"{where}.pipeline")
    _require(
        PIPELINE_RE.fullmatch(pipeline) is not None,
        f"{where}.pipeline does not have the form '<name>/<major>': {pipeline!r}",
    )
    weights: dict[str, dict[str, WeightEntry]] = {}
    for component, labels in _object(obj["weights"], f"{where}.weights").items():
        entries: dict[str, WeightEntry] = {}
        for dtype, raw in _object(labels, f"{where}.weights['{component}']").items():
            entries[dtype] = _weight_entry(raw, f"{where}.weights['{component}']['{dtype}']")
        _require(len(entries) > 0, f"{where}.weights['{component}'] is empty")
        weights[component] = entries
    assets = {
        name: _file_ref(raw, f"{where}.assets['{name}']")
        for name, raw in _object(obj["assets"], f"{where}.assets").items()
    }
    quants = _object(obj["quants"], f"{where}.quants")
    default_quant = _text(obj["defaultQuant"], f"{where}.defaultQuant")
    _require(default_quant in quants, f"{where}.defaultQuant '{default_quant}' is not in quants")
    return LegacyModel(
        pipeline,
        weights,
        assets,
        quants,
        default_quant,
        _object(obj["pipelineConfig"], f"{where}.pipelineConfig"),
    )


def _weight_entry(value: Any, where: str) -> WeightEntry:
    obj = _object(value, where)
    _keys(obj, ["shards"], ["extras"], where)
    raw_shards = obj["shards"]
    _require(
        isinstance(raw_shards, list) and raw_shards, f"{where}.shards is not a non-empty array"
    )
    extras = _object(obj.get("extras", {}), f"{where}.extras")
    return WeightEntry(
        tuple(_file_ref(raw, f"{where}.shards[{i}]") for i, raw in enumerate(raw_shards)),
        tuple(
            (name, _file_ref(extras[name], f"{where}.extras['{name}']")) for name in sorted(extras)
        ),
    )


# ---- Whole-repository mode -------------------------------------------------------


@dataclass(frozen=True)
class CrossRepo:
    """One `--cross-repo <owner/name>=<converted directory>@<40-digit revision>`."""

    repo: str
    directory: Path
    revision: str


def parse_cross_repo(spec: str) -> CrossRepo:
    """Read one `--cross-repo` word (a different form fails loudly — no guessing)."""
    shape = "<owner/name>=<converted directory>@<40-digit revision>"
    repo, sep, rest = spec.partition("=")
    _require(bool(sep) and bool(rest), f"--cross-repo '{spec}' does not have the form {shape}")
    directory, at, revision = rest.rpartition("@")
    _require(bool(at) and bool(directory), f"--cross-repo '{spec}' does not have the form {shape}")
    _require(
        REPO_RE.fullmatch(repo) is not None, f"--cross-repo '{spec}': repo is not <owner>/<name>"
    )
    _require(
        REVISION_RE.fullmatch(revision) is not None,
        f"--cross-repo '{spec}': revision is not 40 lowercase hex digits (a pin is a full commit"
        " sha)",
    )
    return CrossRepo(repo, Path(directory), revision)


@dataclass(frozen=True)
class Container:
    """The `container` field of manifest `karume/5` (ADR 0109 decision 3)."""

    #: Expected values of the 2 documents (graph descriptor / model descriptor).
    descriptor: tuple[DocumentRef, DocumentRef]
    #: FileRefs of all parts including part 0 (in index order).
    parts: tuple[FileRef, ...]

    def to_document(self) -> dict[str, Any]:
        graph, model = self.descriptor
        return {
            "descriptor": {"graph": graph.to_document(), "model": model.to_document()},
            "parts": [ref.to_document() for ref in self.parts],
        }


@dataclass(frozen=True)
class _Seat:
    """One slot in the manifest (model, component, dtype label)."""

    model: str
    component: str
    dtype: str

    def __str__(self) -> str:
        return f"{self.model}.weights.{self.component}.{self.dtype}"


@dataclass(frozen=True)
class _Unit:
    """The unit converted **only once**. Slots pointing at the same content (sbv2's shared
    components) are folded here."""

    component: str
    shards: tuple[FileRef, ...]
    extras: tuple[tuple[str, FileRef], ...]
    #: Only slots that fold PLE have it ({@link _PleFold.refs} — the key that keeps slots with
    #: different folding out of the same container).
    ple: tuple[tuple[str, FileRef], ...]


@dataclass(frozen=True)
class _PleFold:
    """Where one model's PLE is folded (component name, the assets moved into the container, the
    old index that was read).

    This carries the old index read **only once** here (deriving the same fact twice drifts when a
    check is added to only one of them).
    """

    component: str
    refs: tuple[tuple[str, FileRef], ...]
    index: Mapping[str, Any]

    @property
    def asset_names(self) -> frozenset[str]:
        return frozenset(name for name, _ in self.refs)

    @property
    def paths(self) -> frozenset[str]:
        return frozenset(ref.path for _, ref in self.refs)


@dataclass(frozen=True)
class ConvertedContainer:
    """Report for one converted container (one entry even when several slots point at it)."""

    rel_paths: tuple[str, ...]
    initializers: int
    payloads: int
    assets: int


@dataclass(frozen=True)
class RepositoryResult:
    """Result of whole-repository mode."""

    manifest: Path
    converted: tuple[ConvertedContainer, ...]
    #: Number of slots copied from cross-repository references (not converted).
    crossed: int
    #: Number of ordinary files copied as is.
    copied: int


def migrate_repository(
    manifest_path: str | Path,
    out_dir: str | Path,
    *,
    provenance: Provenance,
    cross_repos: Sequence[CrossRepo] = (),
    part_bytes: int = DEFAULT_PART_BYTES,
    _block_bytes: int = BLOCK_MAX_BYTES,
) -> RepositoryResult:
    """Migrate a whole old `karume/4` repository to `karume/5` + `krm` (ADR 0109 decision 8).

    The old repository is **only read**, and every artifact goes under `out_dir`. `part_bytes`
    (the writer's choice set — outside the set is `MigrateError`) and `_block_bytes` (a size
    injection touched only by tests) are as in per-component mode.

    MUST: only the **split form** is written (there is no single-form option) — `container.parts`
    of `karume/5` MUST have 2 or more elements, part 0 + part 1, and the single form cannot produce
    that shape (ADR 0109 decision 3).

    MUST: build the whole repository in a **staging path**, and put it in place at `out_dir` only
    when every conversion, the copy of ordinary files and the manifest checks have passed
    ({@link karume.artifacts.staged_publication}). Placing components one by one in the final
    output directory would leave only the earlier components' containers behind on a run that
    fails midway, and re-running the same command would stop at "artifacts of a previous run
    remain". That the output directory is empty is checked once, at the start.
    """
    _assert_part_length(part_bytes)
    source = Path(manifest_path)
    repo = source.parent.resolve()
    out = Path(out_dir).resolve()
    _require(
        repo != out and repo not in out.parents and out not in repo.parents,
        f"Output directory {out} is nested with the input repository {repo} (old inputs MUST only"
        " be read)",
    )
    _require(
        not out.exists() or (out.is_dir() and not any(out.iterdir())),
        f"Artifacts of a previous run remain in the output directory {out} (delete them and retry"
        " — migration builds from an empty output directory)",
    )
    legacy = read_legacy_manifest(source)
    crosses = {cross.repo: cross for cross in cross_repos}
    _require(
        len(crosses) == len(cross_repos),
        "--cross-repo names the same repo twice (which one to use is undetermined)",
    )
    seat_crosses = {seat: _entry_cross(entry, str(seat)) for seat, entry in _seats(legacy)}
    _assert_cross_repos_declared(seat_crosses, crosses)
    _assert_declared_files(repo, legacy)
    folds = {
        name: _plan_ple(repo, model, f"models['{name}']") for name, model in legacy.models.items()
    }

    with staged_publication(out) as staging:
        staging.mkdir(parents=True)
        containers: dict[_Seat, Container] = {}
        converted: dict[_Unit, Container] = {}
        reports: list[ConvertedContainer] = []
        folded: set[str] = set()
        crossed = 0
        for seat, entry in _seats(legacy):
            fold = folds[seat.model]
            cross = seat_crosses[seat]
            if cross is not None:
                containers[seat] = _crossed_container(crosses[cross[0]], entry, seat)
                crossed += 1
                continue
            folding = fold if fold is not None and fold.component == seat.component else None
            if folding is not None:
                folded.add(seat.model)
            unit = _Unit(
                seat.component, entry.shards, entry.extras, () if folding is None else folding.refs
            )
            found = converted.get(unit)
            if found is None:
                found, report = _convert_unit(
                    unit,
                    repo,
                    staging,
                    str(seat),
                    provenance=provenance,
                    fold=folding,
                    part_bytes=part_bytes,
                    block_bytes=_block_bytes,
                )
                converted[unit] = found
                reports.append(report)
            containers[seat] = found
        for name, fold in folds.items():
            # MUST: removing PLE from assets with no fold target at all would silently drop it
            # from the distribution (this happens when the owning component is entirely a
            # cross-repository reference).
            _require(
                fold is None or name in folded,
                f"models['{name}']: PLE owner component '{fold.component if fold else ''}' has"
                " no slot converted in this repository (the fold target is undetermined)",
            )

        skip = {MANIFEST_FILE}
        for unit in converted:
            skip.update(ref.path for ref in unit.shards)
            # MUST: extras that became container assets are not copied either (no undeclared
            # bytes are left in the distribution).
            skip.update(ref.path for _, ref in unit.extras)
        for fold in folds.values():
            if fold is not None:
                skip.update(fold.paths)
        copied = _copy_plain_files(repo, staging, skip)

        manifest = _manifest_document(legacy, containers, folds)
        assert_manifest_limits(manifest)
        # The staging path is built from empty, so no file collides with the old karume.json
        # (which is excluded from the copy).
        (staging / MANIFEST_FILE).write_text(manifest_text(manifest), encoding="utf-8")
    return RepositoryResult(out / MANIFEST_FILE, tuple(reports), crossed, copied)


def _assert_declared_files(repo: Path, legacy: LegacyManifest) -> None:
    """Check the same-repository files the old manifest declares against the actual size (stat)
    and sha256.

    MUST: check all of them **before** converting and copying. The migration's "identical raw
    bytes" is a comparison with the local files, so if the local mirror disagrees with the old
    revision (partly updated, broken files), its contents become a `karume/5` laundered with new
    sha256 values, and the `assets` FileRefs copied verbatim claim a sha256 that differs from the
    file. Cross-repository references have no local bytes, so they are not checked (the spelling
    of the side being copied is checked when it is read).
    """
    checked: set[FileRef] = set()
    for model_name, model in legacy.models.items():
        entries = [entry for labels in model.weights.values() for entry in labels.values()]
        refs = [*(ref for entry in entries for ref in entry.refs), *model.assets.values()]
        for ref in refs:
            if ref.cross is not None or ref in checked:
                continue
            checked.add(ref)
            where = f"models['{model_name}']: '{ref.path}' declared by the old manifest"
            path = repo / ref.path
            # The path grammar was checked while reading — what remains is a symlink pointing
            # outside the input repository.
            _require(
                path.resolve().is_relative_to(repo),
                f"{where} points outside the input repository through a symlink (old inputs are"
                f" read only inside {repo})",
            )
            _require(path.is_file(), f"{where} is not present locally")
            size = path.stat().st_size
            _require(size == ref.size, f"{where} has length {size}, not the declared {ref.size}")
            actual = sha256_file(path)
            _require(
                actual == ref.sha256,
                f"{where} has a sha256 that differs from the declaration (declared {ref.sha256} /"
                f" actual {actual}) — the local mirror disagrees with the old revision",
            )


def _seats(legacy: LegacyManifest) -> Iterator[tuple[_Seat, WeightEntry]]:
    """All slots in the manifest in declaration order (slot traversal goes only through this one
    path)."""
    for model_name, model in legacy.models.items():
        for component, labels in model.weights.items():
            for dtype, entry in labels.items():
                yield _Seat(model_name, component, dtype), entry


def _entry_cross(entry: WeightEntry, where: str) -> tuple[str, str] | None:
    """The cross-repository reference of one slot (MUST be per container — a mix fails loudly;
    ADR 0109 decision 3)."""
    crosses = {ref.cross for ref in entry.refs}
    if len(crosses) != 1:
        shown = ", ".join(
            sorted("this repository" if c is None else f"{c[0]}@{c[1]}" for c in crosses)
        )
        raise MigrateError(
            f"{where}: cross-repository references are mixed (MUST be per container): {shown}"
        )
    return crosses.pop()


def _assert_cross_repos_declared(
    seat_crosses: Mapping[_Seat, tuple[str, str] | None], crosses: Mapping[str, CrossRepo]
) -> None:
    """When `--cross-repo` is short, stop **listing every referenced repo and its count**."""
    missing: Counter[str] = Counter()
    for cross in seat_crosses.values():
        if cross is not None and cross[0] not in crosses:
            missing[cross[0]] += 1
    if missing:
        shown = " / ".join(f"{repo} ({count} slots)" for repo, count in sorted(missing.items()))
        raise MigrateError(
            f"Repos of cross-repository references are not passed with --cross-repo: {shown}"
            " — convert that repo first in the same mode and pass it with"
            " `--cross-repo <owner/name>=<output directory>@<40-digit revision>`"
        )


def _crossed_container(cross: CrossRepo, entry: WeightEntry, seat: _Seat) -> Container:
    """Copy the `container` of the same **container** from the `karume/5` of a converted
    directory.

    The lookup key is not the slot name (model, component, dtype) but **the location of part 0
    derived from the old shard sequence** — the output path rule is the same in both repositories
    ({@link _output_location}), while nothing guarantees that slot names match between lender and
    borrower (a borrower borrowing from a model of a different name does exist).
    """
    where = f"{cross.repo}'s {MANIFEST_FILE}"
    root = _object(_read_json(cross.directory / MANIFEST_FILE, where), where)
    _require(
        root.get("format") == MANIFEST_FORMAT,
        f"{where}.format is '{root.get('format')}' — only '{MANIFEST_FORMAT}' can be copied",
    )
    directory, stem = _output_location(entry.shards, str(seat))
    wanted = f"{directory}/{stem}{MODEL_SUFFIX}"
    found: tuple[str, Mapping[str, Any]] | None = None
    candidates: set[str] = set()
    for label, raw in _crossed_entries(root, where):
        container = _object(raw, label)
        head = container.get("parts")
        _require(isinstance(head, list) and head, f"{label}.parts is not a non-empty array")
        key = base_path(Path(_file_ref(head[0], f"{label}.parts[0]").path)).as_posix()
        candidates.add(key)
        if key == wanted and found is None:
            found = (label, container)
    if found is None:
        raise MigrateError(
            f"{where}: the container to copy, '{wanted}', does not exist"
            " (copying a cross-repository reference looks it up by the location of part 0"
            " derived from the old shard sequence)"
            f" — the target has {', '.join(sorted(candidates)) if candidates else '(none)'}"
        )
    return _crossed_parse(found[1], found[0], cross)


def _crossed_entries(root: Mapping[str, Any], where: str) -> Iterator[tuple[str, Any]]:
    """The `container` field of every (model, component, dtype) of the target `karume/5` (with the
    spelling of its position)."""
    for model_name, raw_model in _object(root.get("models"), f"{where}.models").items():
        model_where = f"{where}: {model_name}"
        weights = _object(_object(raw_model, model_where).get("weights"), f"{model_where}.weights")
        for component, labels in weights.items():
            for dtype, raw_entry in _object(labels, f"{model_where}.weights.{component}").items():
                label = f"{model_where}.weights.{component}.{dtype}"
                obj = _object(raw_entry, label)
                _keys(obj, ["container"], [], label)
                yield f"{label}.container", obj["container"]


def _crossed_parse(container: Mapping[str, Any], where: str, cross: CrossRepo) -> Container:
    """Read one `container` field to copy (a missing field or an unknown key fails loudly)."""
    _keys(container, ["descriptor", "parts"], [], where)
    descriptor = _object(container["descriptor"], f"{where}.descriptor")
    _keys(descriptor, ["graph", "model"], [], f"{where}.descriptor")
    documents = []
    for key in ("graph", "model"):
        document = _object(descriptor[key], f"{where}.descriptor.{key}")
        _keys(document, ["length", "sha256"], [], f"{where}.descriptor.{key}")
        documents.append(
            DocumentRef(
                _count(document["length"], f"{where}.descriptor.{key}.length"),
                _sha256(document["sha256"], f"{where}.descriptor.{key}.sha256"),
            )
        )
    parts: list[FileRef] = []
    for index, raw in enumerate(container["parts"]):
        ref = _file_ref(raw, f"{where}.parts[{index}]")
        _require(
            ref.cross is None,
            f"{where}.parts[{index}] already has a cross-repository reference (a two-step copy"
            " cannot be followed to its target)",
        )
        parts.append(replace(ref, repo=cross.repo, revision=cross.revision))
    return Container((documents[0], documents[1]), tuple(parts))


def _convert_unit(
    unit: _Unit,
    repo: Path,
    out: Path,
    where: str,
    *,
    provenance: Provenance,
    fold: _PleFold | None,
    part_bytes: int,
    block_bytes: int,
) -> tuple[Container, ConvertedContainer]:
    """Convert one unit (the shard sequence of component × dtype) to `krm` and build the
    manifest's `container`."""
    if GRAPH_NAME_PATTERN.fullmatch(unit.component) is None:
        raise MigrateError(
            f"{where}: component name '{unit.component}' falls outside the container's graph name"
            f" vocabulary ({GRAPH_NAME_PATTERN.pattern})"
        )
    directory, stem = _output_location(unit.shards, where)
    # The order of the assets is the physical layout order as is (PLE is in token order — for
    # the scan-type sources of §4.2).
    assets = _ple_assets(repo, fold, block_bytes, where)
    assets.update(_extra_assets(repo, unit.extras, where))
    result = _migrate_shards(
        [repo / ref.path for ref in unit.shards],
        out / directory / f"{stem}{MODEL_SUFFIX}",
        provenance=provenance,
        graph_name=unit.component,
        assets=assets,
        part_bytes=part_bytes,
        block_bytes=block_bytes,
    )
    parts = tuple(_published_ref(out, path) for path in result.parts)
    return (
        Container(result.descriptor, parts),
        ConvertedContainer(
            tuple(ref.path for ref in parts), result.initializers, result.payloads, result.assets
        ),
    )


def _output_location(shards: Sequence[FileRef], where: str) -> tuple[str, str]:
    """`(relative path of the output directory, stem)` — the location is **the parent of the
    weight shards** (no guessing)."""
    stems = set()
    for ref in shards:
        name = PurePosixPath(ref.path)
        _require(
            name.suffix == ".safetensors",
            f"{where}: the extension of old shard '{ref.path}' is not .safetensors",
        )
        stems.add(base_path(Path(name.name)).stem)
    _require(
        len(stems) == 1,
        f"{where}: the file name stems of the shard sequence differ: {sorted(stems)}",
    )
    # The second and later entries of the sequence are weight shards. A sequence of only the graph
    # shard goes in the parent of the first.
    weights = shards[1:] or shards[:1]
    directories = {PurePosixPath(ref.path).parent.as_posix() for ref in weights}
    _require(
        len(directories) == 1,
        f"{where}: the parent directories of the weight shards differ ({sorted(directories)})"
        " — the location is not guessed",
    )
    return directories.pop(), stems.pop()


def _published_ref(out: Path, path: Path) -> FileRef:
    """FileRef of one part put in place (a part of length 0 is listed too, with `size: 0` — ADR
    0109 decision 3)."""
    return FileRef(**file_ref(out, path.relative_to(out).as_posix(), sha256_file(path)))


def _extra_assets(
    repo: Path, extras: Sequence[tuple[str, FileRef]], where: str
) -> dict[str, AssetInput]:
    """Old `extras` → container assets (only `rope_base` exists — an unknown name fails
    loudly)."""
    assets: dict[str, AssetInput] = {}
    for name, ref in extras:
        mapped = EXTRA_ASSETS.get(name)
        if mapped is None:
            raise MigrateError(
                f"{where}: extras '{name}' has no mapping target"
                f" (only {' / '.join(sorted(EXTRA_ASSETS))} can move to container assets)"
            )
        _require(
            ref.cross is None,
            f"{where}: extras '{name}' is a cross-repository reference (cannot be folded into"
            " the container)",
        )
        asset_name, role = mapped
        # The logical length is the old declaration's `size` (if it disagrees with the file, the
        # writer rejects it — no guessing).
        assets[asset_name] = AssetInput(role, ref.size, _file_reader(repo / ref.path))
    return assets


def _file_reader(path: Path) -> Callable[[], bytes]:
    """A lazy call that reads one whole file (not opened until it is called)."""

    def read() -> bytes:
        return path.read_bytes()

    return read


def _copy_plain_files(repo: Path, out: Path, skip: frozenset[str] | set[str]) -> int:
    """Copy the ordinary files that were not converted as is (directories starting with `.` are
    not entered)."""
    copied = 0
    for source in sorted(repo.rglob("*")):
        rel = source.relative_to(repo)
        if any(part.startswith(".") for part in rel.parts[:-1]):
            continue
        if not source.is_file() or rel.as_posix() in skip:
            continue
        target = out / rel
        _require(
            not target.exists(),
            f"{rel.as_posix()} already exists in the output directory (delete it and retry)",
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        copied += 1
    return copied


def _manifest_document(
    legacy: LegacyManifest,
    containers: Mapping[_Seat, Container],
    folds: Mapping[str, _PleFold | None],
) -> dict[str, Any]:
    """The `karume/5` `karume.json` (old values as is; only weights are replaced by
    `container`)."""
    models: dict[str, Any] = {}
    for name, model in legacy.models.items():
        fold = folds[name]
        dropped = frozenset() if fold is None else fold.asset_names
        models[name] = {
            "pipeline": model.pipeline,
            "weights": {
                component: {
                    dtype: {"container": containers[_Seat(name, component, dtype)].to_document()}
                    for dtype in labels
                }
                for component, labels in model.weights.items()
            },
            "assets": {
                asset: ref.to_document()
                for asset, ref in model.assets.items()
                if asset not in dropped
            },
            "quants": dict(model.quants),
            "defaultQuant": model.default_quant,
            "pipelineConfig": dict(model.pipeline_config),
        }
    return {
        "format": MANIFEST_FORMAT,
        "generator": generator_tag(),
        "defaultModel": legacy.default_model,
        "models": models,
    }


# ---- PLE sidecar → container assets (index schema 3) -------------------------------


def _plan_ple(repo: Path, model: LegacyModel, where: str) -> _PleFold | None:
    """Decide where PLE is folded (`None` when there is no index).

    MUST: a pipeline that has `ple_index` but is not in the owner table fails loudly — do not
    produce "vanished from assets" while it is undetermined which component's container it folds
    into.
    """
    index_ref = model.assets.get(PLE_INDEX_ASSET)
    if index_ref is None:
        return None
    pipeline = model.pipeline.split("/", 1)[0]
    component = PLE_OWNER.get(pipeline)
    if component is None:
        raise MigrateError(
            f"{where}: assets has '{PLE_INDEX_ASSET}' but pipeline '{pipeline}' is not a PLE"
            f" owner (table: {', '.join(f'{k} → {v}' for k, v in sorted(PLE_OWNER.items()))})"
        )
    _require(
        component in model.weights,
        f"{where}: PLE owner component '{component}' is not in weights",
    )
    _require(
        index_ref.cross is None,
        f"{where}: '{PLE_INDEX_ASSET}' is a cross-repository reference (cannot be folded into the"
        " container)",
    )
    index = read_ple_index(repo / index_ref.path, f"{where}: {PLE_INDEX_ASSET}")
    refs: list[tuple[str, FileRef]] = [(PLE_INDEX_ASSET, index_ref)]
    for entry in index["shards"]:
        name = str(entry["file"])
        ref = model.assets.get(name)
        if ref is None:
            raise MigrateError(
                f"{where}: PLE shard '{name}' that the index points at is not in assets"
            )
        _require(
            ref.cross is None,
            f"{where}: PLE shard '{name}' is a cross-repository reference (cannot be folded into"
            " the container)",
        )
        refs.append((name, ref))
    return _PleFold(component, tuple(refs), index)


def read_ple_index(path: Path, where: str) -> dict[str, Any]:
    """Read an old PLE index (`ple.json`).

    The accepted set mirrors `packages/models/src/gemma/ple-index.ts` — schema 1 (I8, no `storage`
    field; the real assets of karume-gemma4) and schema 2 (packed `i2` / `i4`, ADR 0097). The
    returned `storage` is normalized to `"i8"` even for schema 1 (the new index schema 3 always
    spells out the storage).
    """
    raw_root = _object(_read_json(path, where), where)
    # MUST: put the version check before the field checks (the same reason as
    # {@link read_legacy_manifest} — for another version, saying "that version cannot be folded"
    # rather than "a field is missing" is decisive for whoever fixes it).
    schema = raw_root.get("schema")
    _require(
        schema in (1, 2),
        f"{where}.schema is {schema!r} — only schema 1 (i8) and 2 (packed) can be folded into the"
        " container",
    )
    if schema == 1:
        _keys(raw_root, ["schema", "tokens", "layers", "dim", "embedScale", "shards"], [], where)
        root: dict[str, Any] = {**raw_root, "storage": "i8"}
    else:
        _keys(
            raw_root,
            ["schema", "storage", "tokens", "layers", "dim", "embedScale", "shards"],
            [],
            where,
        )
        root = dict(raw_root)
    _require(
        root["storage"] in PLE_PACK_FACTOR,
        f"{where}.storage is {root['storage']!r}"
        f" (must be one of {' / '.join(sorted(PLE_PACK_FACTOR))})",
    )
    for key in ("tokens", "layers", "dim"):
        _require(_count(root[key], f"{where}.{key}") > 0, f"{where}.{key} is not 1 or more")
    scale = root["embedScale"]
    _require(
        isinstance(scale, int | float) and not isinstance(scale, bool) and scale > 0,
        f"{where}.embedScale is not a positive number: {scale!r}",
    )
    shards = root["shards"]
    _require(isinstance(shards, list) and shards, f"{where}.shards is not a non-empty array")
    cursor = 0
    for index, raw in enumerate(shards):
        entry = _object(raw, f"{where}.shards[{index}]")
        _keys(entry, ["file", "start", "stop"], [], f"{where}.shards[{index}]")
        _text(entry["file"], f"{where}.shards[{index}].file")
        start = _count(entry["start"], f"{where}.shards[{index}].start")
        stop = _count(entry["stop"], f"{where}.shards[{index}].stop")
        _require(
            start == cursor and stop > start,
            f"{where}.shards[{index}]: range [{start}, {stop}) is not a non-empty range"
            f" continuing from {cursor}",
        )
        cursor = stop
    _require(
        cursor == root["tokens"], f"{where}: the shard total of {cursor} rows differs from tokens"
    )
    return root


@dataclass(frozen=True)
class _StoredEntry:
    """One tensor of the old sidecar (the facts decided by the header alone)."""

    dtype: str
    shape: tuple[int, ...]
    begin: int
    end: int

    @property
    def nbytes(self) -> int:
        return self.end - self.begin


def _read_sidecar_header(path: Path) -> tuple[dict[str, str], dict[str, _StoredEntry], int]:
    """Read **only the header** of the old sidecar's safetensors (a PLE shard runs to several
    hundred MB).

    MUST: go through the same checks as reading old shards ({@link karume.legacy.read_component})
    — check the layout (known dtype, matching declared length, no gaps, alignment) first, and take
    `__metadata__` **verbatim** as a string → string map (coercing types turns a broken sidecar
    into an exception somewhere far away).
    """
    assert_reader_layout(path, allow_legacy_dtypes=True)
    header = safetensors_header(path)
    with path.open("rb") as handle:
        data_start = 8 + int.from_bytes(handle.read(8), "little")
    metadata = _read_metadata(path, header)
    entries: dict[str, _StoredEntry] = {}
    for name, spec in header.items():
        if name == "__metadata__":
            continue
        begin, end = spec["data_offsets"]
        entries[name] = _StoredEntry(spec["dtype"], tuple(spec["shape"]), begin, end)
    return metadata, entries, data_start


def _read_segments(segments: Sequence[tuple[Path, int, int]], begin: int, end: int) -> bytes:
    """Read `[begin, end)` of a concatenated sequence of ranges (PLE rows cross shard
    boundaries)."""
    out = bytearray()
    cursor = 0
    for path, offset, length in segments:
        stop = cursor + length
        if stop > begin and cursor < end:
            lower, upper = max(begin, cursor), min(end, stop)
            with path.open("rb") as handle:
                handle.seek(offset + lower - cursor)
                chunk = handle.read(upper - lower)
            if len(chunk) != upper - lower:
                raise MigrateError(f"{path}: PLE range [{lower}, {upper}) ran out partway")
            out += chunk
        cursor = stop
    if len(out) != end - begin:
        raise MigrateError(
            f"PLE range [{begin}, {end}) does not fit in the shard sequence length {cursor}"
        )
    return bytes(out)


def _segment_reader(
    segments: tuple[tuple[Path, int, int], ...],
) -> Callable[[int, int], bytes]:
    """A "read a byte range" call over a concatenated sequence of ranges (what
    {@link ple_assets} takes)."""

    def read(begin: int, end: int) -> bytes:
        return _read_segments(segments, begin, end)

    return read


def ple_sidecar_assets(
    index_path: Path, *, where: str | None = None, block_bytes: int = BLOCK_MAX_BYTES
) -> dict[str, AssetInput]:
    """Fold an old PLE sidecar (`ple.json` + the shard sequence **in the same directory**) into
    container assets.

    The entry for callers that migrate a location without a manifest (an export series
    directory). The folding is **the same single path** as whole-repository mode
    ({@link _fold_ple}); the only difference is whether the shard locations come from the old
    manifest's `assets` or from the file names the index states.

    NOTE: on this path, **two of the fold-target checks, order and count, are trivially true** (the
    sequence is built from the index itself) — they only have effect on the side that takes
    locations from the old manifest. Checking "a `ple-*.safetensors` not in the index exists on
    disk" is therefore **the caller's responsibility**; this passes it.
    """
    at = str(index_path) if where is None else where
    index = read_ple_index(index_path, at)
    directory = index_path.parent
    shards = [(str(entry["file"]), directory / str(entry["file"])) for entry in index["shards"]]
    return _fold_ple(shards, index, block_bytes, at)


def _ple_assets(
    repo: Path, fold: _PleFold | None, block_bytes: int, where: str
) -> dict[str, AssetInput]:
    """Fold the PLE sidecar the old manifest declared into container assets (locations are the
    `assets` FileRefs)."""
    if fold is None:
        return {}
    # The first entry is the index itself (replaced by the new index at the fold target), so drop
    # it.
    shards = [(name, repo / ref.path) for name, ref in fold.refs[1:]]
    return _fold_ple(shards, fold.index, block_bytes, where)


def _fold_ple(
    shards: Sequence[tuple[str, Path]],
    index: Mapping[str, Any],
    block_bytes: int,
    where: str,
) -> dict[str, AssetInput]:
    """PLE shard sequence → container assets (the assembly itself is the single path of
    {@link ple_assets}).

    This owns only **reading the old sidecar** — it cross-checks the index against the files (row
    counts, byte lengths, shard order, `karume_ple` metadata) and hands over the sequence of ranges
    concatenated in token order as the reader. The old shard boundaries carry no meaning (all
    shards are concatenated in token order and cut anew).
    """
    tokens, layers, dim = index["tokens"], index["layers"], index["dim"]
    try:
        row_bytes = ple_row_bytes(index["storage"], layers, dim)
    except PleError as cause:
        raise MigrateError(f"{where}: {cause}") from cause
    segments: dict[str, list[tuple[Path, int, int]]] = {"values": [], "scales": []}
    # The old writer's declaration of one row (dtype, and the shape without the leading dimension
    # — `values` is the logical shape).
    row_declarations: dict[str, tuple[str, tuple[int, ...]]] = {
        "values": (LEGACY_STORED_DTYPE[index["storage"]], (layers, dim)),
        "scales": (LEGACY_SCALE_DTYPE, (layers,)),
    }
    _require(
        len(shards) == len(index["shards"]),
        f"{where}: {len(shards)} PLE shards (the index has {len(index['shards'])})",
    )
    for position, (name, path) in enumerate(shards):
        declared = index["shards"][position]
        _require(
            name == declared["file"],
            f"{where}: the PLE shard order differs from the index (position {position} is"
            f" '{declared['file']}')",
        )
        rows = declared["stop"] - declared["start"]
        metadata, entries, data_start = _read_sidecar_header(path)
        _assert_ple_metadata(metadata, index, declared, f"{where}: '{name}'")
        _require(
            set(entries) == set(row_bytes),
            f"{where}: '{name}': the tensors are {sorted(entries)}"
            f" (MUST be the two {' / '.join(sorted(row_bytes))})",
        )
        for key, expected in row_bytes.items():
            entry = entries[key]
            _require(
                entry.shape[:1] == (rows,),
                f"{where}: '{name}'.{key}: leading dimension {entry.shape[:1]} differs from the"
                f" index's {rows} rows",
            )
            # Leading dimension and byte length alone let through a different shape with the same
            # element count or a different type of the same byte length — derive the old
            # writer's spelling (dtype and the whole shape) from the index and compare.
            stored_dtype, stored_shape = row_declarations[key]
            _require(
                entry.dtype == stored_dtype,
                f"{where}: '{name}'.{key}: dtype {entry.dtype} differs from the old writer's"
                f" spelling {stored_dtype}",
            )
            _require(
                entry.shape == (rows, *stored_shape),
                f"{where}: '{name}'.{key}: shape {list(entry.shape)} differs from"
                f" {[rows, *stored_shape]}",
            )
            _require(
                entry.nbytes == rows * expected,
                f"{where}: '{name}'.{key}: {entry.nbytes} bytes differs from"
                f" {rows} rows × {expected} bytes",
            )
            segments[key].append((path, data_start + entry.begin, entry.nbytes))
    try:
        return ple_assets(
            storage=index["storage"],
            tokens=tokens,
            layers=layers,
            dim=dim,
            embed_scale=index["embedScale"],
            read_values=_segment_reader(tuple(segments["values"])),
            read_scales=_segment_reader(tuple(segments["scales"])),
            block_bytes=block_bytes,
        )
    except PleError as cause:
        raise MigrateError(f"{where}: {cause}") from cause


def _assert_ple_metadata(
    metadata: Mapping[str, str],
    index: Mapping[str, Any],
    declared: Mapping[str, Any],
    where: str,
) -> None:
    """The shard's `karume_ple` does not disagree with the index (a disagreement is the seed of
    silently wrong values)."""
    raw = metadata.get(PLE_METADATA_KEY)
    if raw is None:
        raise MigrateError(f"{where}: __metadata__.{PLE_METADATA_KEY} is missing")
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as cause:
        raise MigrateError(f"{where}: {PLE_METADATA_KEY} cannot be read as JSON") from cause
    found = _object(parsed, f"{where}.{PLE_METADATA_KEY}")
    # Schema 1 shard metadata has no `storage` (the index side is already normalized to i8).
    keys = ("schema", "tokens", "layers", "dim") + (("storage",) if index["schema"] == 2 else ())
    expected = {key: index[key] for key in keys}
    expected.update({key: declared[key] for key in ("start", "stop")})
    for key, value in expected.items():
        _require(
            found.get(key) == value,
            f"{where}.{PLE_METADATA_KEY}.{key} is {found.get(key)!r} — the index has {value!r}",
        )


# ---- CLI ------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Migrate the old distribution (safetensors) to the container format"
        " (krm / krg); inputs are only read"
    )
    parser.add_argument(
        "models",
        type=Path,
        nargs="*",
        help="representative path of a component to migrate (a `...-00001-of-00002` file may"
        " be passed too)",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="whole-repository mode: the old karume/4 karume.json (cannot be combined with"
        " positional arguments)",
    )
    parser.add_argument(
        "--cross-repo",
        action="append",
        default=[],
        metavar="OWNER/NAME=DIR@REVISION",
        help="source to copy cross-repository references from (a directory converted earlier"
        " in the same mode and its new 40-digit revision)",
    )
    parser.add_argument("--out", type=Path, required=True, help="output directory")
    parser.add_argument(
        "--license",
        required=True,
        help="provenance.license (a license identifier — the text is not included)",
    )
    parser.add_argument("--notice", default=None, help="provenance.notice (a reference to NOTICE)")
    parser.add_argument(
        "--upstream-revision", default=None, help="provenance.upstreamRevision (upstream revision)"
    )
    parser.add_argument(
        "--writer",
        default=None,
        help="provenance.writer (default: not written — it goes into the container only when"
        " the caller passes it explicitly)",
    )
    parser.add_argument(
        "--graph-name",
        default=None,
        help="graph name (default: the parent directory name — cannot be passed with several"
        " components)",
    )
    parser.add_argument(
        "--single",
        action="store_true",
        help="write a single-form krm (default: split form; per-component mode only)",
    )
    parser.add_argument("--graph", action="store_true", help="also write a krg (graph container)")
    choices = [choice // _MIB for choice in PART_LENGTH_CHOICES]
    parser.add_argument(
        "--part-bytes",
        type=int,
        choices=choices,
        default=DEFAULT_PART_BYTES // _MIB,
        help=f"part length — the writer's choice set {{{','.join(map(str, choices))}}} MiB",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Migrate the given components (or a whole repository).

    MUST: stop where it fails (do not migrate the rest and collect at the end) — the exception
    spells out the broken invariant, so raising it as is carries the most information.

    MUST: `provenance.writer` is **not written by default**. Filling in the generator tag by
    default would make migrated containers and containers a recipe exports directly (no
    `PROVENANCE` spells `writer`) differ in bytes forever, giving "rebuilding the mirror moves only
    the model descriptor". The one place that holds the tool version is the `generator` field of
    `karume.json`, and there is no reason to put a copy on the container side (the same reason the
    goldens do not write `writer` — {@link karume.goldens.GOLDEN_PROVENANCE}).
    """
    args = build_parser().parse_args(argv)
    provenance = Provenance(
        license=args.license,
        notice=args.notice,
        upstream_revision=args.upstream_revision,
        writer=args.writer,
    )
    if args.manifest is not None:
        _run_repository(args, provenance)
        return
    if not args.models:
        raise MigrateError(
            "Either the representative path of a component to migrate or --manifest is required"
        )
    if args.cross_repo:
        raise MigrateError("--cross-repo can only be used in whole-repository mode (--manifest)")
    if args.graph_name is not None and len(args.models) > 1:
        raise MigrateError(
            f"--graph-name can only be passed for one component ({len(args.models)} given)"
        )
    for path in args.models:
        result = migrate_component(
            path,
            args.out,
            provenance=provenance,
            graph_name=args.graph_name,
            single=args.single,
            write_graph=args.graph,
            part_bytes=args.part_bytes * _MIB,
        )
        graph = f" + {result.graph.name}" if result.graph is not None else ""
        print(
            f"{path}: parts {len(result.parts)} ({result.parts[0].name} …){graph}"
            f" initializers={result.initializers} payloads={result.payloads}"
        )


def _run_repository(args: argparse.Namespace, provenance: Provenance) -> None:
    """Entry of whole-repository mode (every option that cannot be combined is rejected here)."""
    if args.models:
        raise MigrateError(
            f"--manifest cannot be combined with positional arguments ({len(args.models)}"
            " positional arguments) — whole-repository mode builds the shard sequences from the"
            " manifest's declarations"
        )
    for name, value in (("--graph-name", args.graph_name), ("--graph", args.graph or None)):
        if value is not None:
            raise MigrateError(
                f"{name} is an option of per-component mode (in whole-repository mode the graph"
                " name is the manifest's weights key, and a shared krg is not placed in karume/5"
                " — ADR 0109 decision 5)"
            )
    if args.single:
        raise MigrateError(
            "--single cannot be used in whole-repository mode (container.parts of karume/5 MUST"
            " have 2 or more elements, part 0 + part 1, and the official HF distribution is split"
            " form only — ADR 0109 decision 3 / container-v1 §8)"
        )
    result = migrate_repository(
        args.manifest,
        args.out,
        provenance=provenance,
        cross_repos=[parse_cross_repo(spec) for spec in args.cross_repo],
        part_bytes=args.part_bytes * _MIB,
    )
    for converted in result.converted:
        print(
            f"{converted.rel_paths[0]}: parts {len(converted.rel_paths)}"
            f" initializers={converted.initializers} payloads={converted.payloads}"
            f" assets={converted.assets}"
        )
    print(
        f"{result.manifest}: containers={len(result.converted)}"
        f" crossed={result.crossed} copied={result.copied}"
    )


if __name__ == "__main__":
    main()
