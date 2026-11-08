"""Assembly of the distribution directory — series directories → one repository form that can be
uploaded to HF as is.

The sources of truth are ADR 0109 (`docs/decisions/0109-manifest-v5-container.md`) for the shape
of manifest `karume/5`, and ADR 0041 (`docs/decisions/0041-manifest-v2.md`) for the layout inside
the repository. This module produces the files laid out in the shape of 0041 decision 2, the
`karume.json` (`karume/5`) that declares them, and the model card `README.md` derived mechanically
from the manifest (ADR 0037 §3's "a form that can be uploaded as an HF repository as is").

**The layout inside the repository is uniformly "per-model subtrees + `shared/` + `karume.json` /
`README.md` at the root"** (ADR 0041 decision 9). A single-model repository follows the same rule
and goes under `<model name>/…` — making single-model repositories flat as an exception would move
every existing path the moment a second model is added.

**Per-pipeline dispatch** (`--pipeline`) and **a per-model axis** (`--model`). What is shared is
only the layer that "decides the shared entries / places / takes sha256 / reconciles declarations
with files"; which files are laid out under which names and what is declared belongs to each
pipeline's table ({@link Pipeline}).

MUST: **the accepted set is passed in by the caller** (`pipelines` of {@link main}) — core's
{@link PIPELINES} is "the pipelines that can be built with the core wheel alone", not all of them.
The repository-only recipes (`tools/export-recipes/<family>/`) are outside the wheel, so the
repository's dist driver (`tools/export-recipes/dist.py`) merges them into core's table and passes
the result. Core holds no default pipeline for the same reason — since the table changes with the
caller, baking a default would create "building a pipeline that was not passed, by default" (ADR
0065 decision 2).

MUST: **the default locations are passed in by the caller too** (`default_out_dir` /
`default_series` of {@link main}) — the repository's `models/` and `outputs/series/` are repo
topology, not knowledge of a generic exporter (ADR 0065 Consequences — the spellings live in one
place, `tools/export-recipes/_shared/paths.py`). When they are not passed, core alone requires
`--out` / `--series` and fails without them (it does not silently build somewhere relative to the
cwd).

MUST: **the manifest is not hand-written but derived from the assets** (ADR 0038 Context, kept in
v2). `size` / `sha256` are taken by streaming from the actual files after assembly — both not
reading several GB whole and making the failure mode "the table disagrees with the files"
structurally impossible depend on this.

MUST: the `io.*.safetensors` scattered in the series (E2E input / output fixtures) are **not part
of the distribution**. Only files in the per-pipeline output path table go into the output; nothing
outside the table silently mixes in.

MUST: the checks (full verification of the weights containers / storage / byte identity of the
rope base table / row counts of the style and speaker tables) run **before placement**, **for every
model** — if one fails, not a single file of a half-built distribution is left. Assembly has two
steps, "plan (building {@link ModelPlan} = all checks and reads) → materialise
({@link assemble_family})", and the first writes not a single byte.

MUST: **input containers are not trusted as "verified in the past"** — the components weights
point at are fully checked by {@link assert_weight_components_verified} with
`karume.verify.verify_container` (it reads only the two documents). Assembly itself reads only the
storage and part of the IR ({@link ir_graph}), so skipping this would let a series with a broken
descriptor / binding table slip past the family-specific checks into the distribution, failing
only in the user's `createSession`.

Placement is always **an independent copy** (no hard links — the ruling of 2026-08-09, ADR 0041
addendum). The series writer overwrites existing files by truncating them, so a distribution
sharing links would silently change content on a re-export of the series, and the manifest's
sha256 would disagree with the files. The distribution is emitted as a self-contained snapshot
independent of the series.

Core alone, with an empty accepted set, cannot build anything, so usage is spelled out on the
repository's dist driver side (the module doc of `tools/export-recipes/dist.py`):

    uv run python dist.py --pipeline siglip2               # at the export-recipes root
    uv run python dist.py --pipeline siglip2 --model so400m
    uv run python dist.py --pipeline birefnet --model lucida

`--card-profile` chooses the model card's **attribution** (provenance, licence, citation) and is
required for pipelines with two or more choices ({@link resolve_card_renderer}). It does not
affect the assembled assets — the same weights carry a different attribution depending on which
family they are distributed as.

`--repo` explicitly overrides the HF repository ID spelled in the card's Usage example; the
default is the pipeline's declaration ({@link resolve_repo}). It is not derived from the `--out`
directory name.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import re
import shutil
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from functools import partial
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
from safetensors.numpy import save

from karume.artifacts import ArtifactSwapError, staged_publication

# The limits (part count, part length ceiling, descriptor length) are imported — so that the
# writing side and the declaring side (the manifest check) see the same spelling.
# {@link verify_dist} is a check that can be applied to any local distribution, so a check
# unaware of the limits would split the accepted set in two.
from karume.container import (
    HEADER_BYTES,
    MAX_DESCRIPTOR_BYTES,
    MAX_PARTS,
    PART_MAX_BYTES,
    ContainerFormatError,
    DocumentRef,
    codec_entry,
    container_parts,
    numbered_name,
    read_descriptor_refs,
)
from karume.limits import LimitsError, max_state_slot_bytes, required_limits
from karume.modelcard import HF_OWNER, FileRefKey, file_ref_key
from karume.verify import ContainerError, VerifiedContainer, assert_ir_accepted, verify_container

# ---- ① Shared part: location spellings, shared entries, placement, hashes, reconciliation -------

#: File name of the manifest (ADR 0041 decision 1 — a fixed name at the repository root).
MANIFEST_FILENAME = "karume.json"

#: Format identifier of the manifest (ADR 0109 decision 1 — the hub reads only one form).
#: `karume/5` is the form in which the dtype entries of weights became **the container's entry
#: point** (`{container: {descriptor, parts}}`); both the shard list of `karume/4`
#: (`{shards, extras?}`) and the `extras` field were retired (the one kind of extras content,
#: `rope_base`, moved into the container's assets — ADR 0109 decision 4).
#:
#: MUST: old majors are not read (no dual reading is implemented — ADR 0109 decision 1). Instead
#: of quietly staying readable, the break is declared through the major.
MANIFEST_FORMAT = "karume/5"

#: File name of the model card (ADR 0037 §3 — the fixed name whose frontmatter HF reads).
MODEL_CARD_FILENAME = "README.md"

#: The place where a file several models hold with **the same relative path and the same sha256**
#: is placed once (ADR 0041 decision 5). There is no dedicated indirection; sharing happens just by
#: each model's manifest entry writing the same path.
SHARED_DIRNAME = "shared"

#: **Meta files** of the distribution (a description of the distribution itself, not assets the
#: manifest declares). The undeclared-file check exempts only these two — the exemptions are held
#: as relative paths rather than names so that a same-named file slipped into a subdirectory is
#: not overlooked. **Their presence is not required** ({@link verify_dist} runs **before** the
#: model card is written, so it must pass without them).
META_PATHS = frozenset({MANIFEST_FILENAME, MODEL_CARD_FILENAME})

#: Location of the modification notice (at the distribution repository's root). The container's
#: `provenance.notice` points at this **path fragment**; the text is not in the container
#: (container-v1 §2.3).
NOTICE_FILENAME = "NOTICE.md"

#: Location of the upstream licence text (same as above).
LICENSE_FILENAME = "LICENSE.md"

#: The **legal text** entries that may be placed at the distribution repository's root
#: ({@link Pipeline.root_files}). Files the upstream weight licence requires as a condition of
#: redistribution — only two: a copy of the licence text itself and a Notice stating attribution /
#: modification. The names are restricted to a set to show **by a check** that this is not "a way
#: to bring arbitrary files to the root" (a type cannot say whether something is legal text). Like
#: {@link META_PATHS}, **their presence is not required** (only pipelines that need them pass
#: them).
LEGAL_PATHS = frozenset({LICENSE_FILENAME, NOTICE_FILENAME})

#: Size limits (ADR 0041 decision 7). The hub rejects with the same values, so **the writing side
#: fails first** (rather than the user finding out only after distribution).
MAX_MODELS = 32
MAX_WEIGHTS = 32
MAX_ASSETS = 32
MAX_QUANTS = 32
MAX_PIPELINE_CONFIG_BYTES = 256 * 1024
MAX_MANIFEST_BYTES = 1024 * 1024

#: Accepted vocabulary of the dtype labels of weights = the runtime's **storage dtype vocabulary**.
#:
#: NOTE: the source of truth is the decode-path vocabulary of the codec ledger (the `CodecLayout`
#: type in TS `packages/runtime/src/format/container/codecs.ts`, `layout` of Python
#: {@link karume.container.CodecEntry}), but on the TS side the layout values exist only as runtime
#: values on each entry of the exported `CODEC_LEDGER`; there is neither a constant listing the
#: layout vocabulary nor a source-of-truth file for mechanical reconciliation like the op contract
#: table — this is **a hand-made copy**. However, the same package has the IR verifier's copy
#: ({@link karume.verify.STORAGE_DTYPES}), and a test checks that this one contains it
#: (`TestDtypeLabelVocabulary` in `tests/test_dist.py`) — the failure mode of only one side going
#: stale is mechanised by that one test.
#:
#: It is restricted because the card's body states that the labels are "the storage dtype
#: vocabulary" ({@link karume.modelcard.quants}). The source of truth for storage is the
#: descriptor's `encoding.codec`, and labels are a vocabulary for selection and display (ADR 0071
#: decision 3) — the label's accepted set pins that line.
STORAGE_DTYPE_LABELS = frozenset({"f32", "f16", "bf16", "i8", "i4", "i4f16", "i2", "i32"})

#: Character limits of a quant's display fields (ADR 0075 decision 1). `label` is a short display
#: name for a choice, `description` a one-line explanation; both optional. The hub checks bounds
#: with the same values (the manifest is external input — ADR 0075 decision 2), so like the other
#: limits **the writing side fails first**.
MAX_QUANT_LABEL_CHARS = 64
MAX_QUANT_DESCRIPTION_CHARS = 200

#: Display fields and their limits (the one place {@link assert_quant_presentation} reads).
QUANT_PRESENTATION_LIMITS: Mapping[str, int] = {
    "label": MAX_QUANT_LABEL_CHARS,
    "description": MAX_QUANT_DESCRIPTION_CHARS,
}

# The spelling regexes below MUST be applied with `.fullmatch` — Python's `$` also matches just
# before a trailing newline, so `.match` would accept `'owner/name\n'`. The hub (JS `$`) rejects
# it, so the accepted set would drift from the hub's, giving a manifest that "assembles green but
# the hub cannot open".

#: Spelling of the revision of a cross-repository component reference (the optional `repo` /
#: `revision` fields of ADR 0038 §7) — HF's **full commit sha** (40 lowercase hex digits). Branch
#: names and tags move their target, so they are not a pin ({@link ExternalComponents}).
REVISION_RE = re.compile(r"^[0-9a-f]{40}$")

#: Likewise, the spelling of the target repository (`<owner>/<name>` — each segment has the same
#: allow-list as the hub's `REPO_SEGMENT_RE`). MUST: segments with a leading dot (including `.` /
#: `..`) are rejected — the hub does not percent-encode the `/` of a repo and therefore rejects
#: traversal; letting it through would give a manifest the hub cannot open.
REPO_RE = re.compile(r"^[A-Za-z0-9_-][A-Za-z0-9._-]*/[A-Za-z0-9_-][A-Za-z0-9._-]*$")

#: Fields of the file reference of a cross-repository reference (the 3-field set + 2 source
#: fields).
EXTERNAL_REF_KEYS = frozenset({"repo", "revision", "path", "size", "sha256"})

#: Allowed characters of a model name. A model name is both a manifest key and **a directory name
#: inside the repository**, so it is restricted to the same set as the hub's path segment check
#: (ADR 0041 decision 6).
MODEL_NAME_RE = re.compile(r"^[A-Za-z0-9_-][A-Za-z0-9._-]*$")

#: One segment of a relative path inside a model subtree (the mirror of the hub's `SEGMENT_RE` —
#: ADR 0041 decision 6).
PATH_SEGMENT_RE = re.compile(r"^[A-Za-z0-9._-]+$")

#: Read unit for sha256. Its only requirement is not reading several GB whole; the value itself is
#: a plain I/O unit.
_CHUNK_BYTES = 1 << 20


class DistError(ValueError):
    """An assembly precondition was broken (missing assets, rope base table mismatch, manifest
    disagreeing with the files)."""


def component_parts(path: Path) -> tuple[Path, ...]:
    """A component's representative path → the existing parts (without numbering, the single
    element of the representative path itself).

    A thin layer that only translates failures of the numbering convention (container-v1 §8) into
    assembly vocabulary. This is the one entry point on the assembly side; the storage check, the
    IR read and the plan expansion all see the same list.
    """
    try:
        return container_parts(path)
    except ContainerFormatError as cause:
        raise DistError(str(cause)) from cause


def assert_component_present(path: Path) -> None:
    """Fail unless the component's files (its parts) exist.

    The spelling lives in one place so that the diagnostic for absence keeps naming "the
    representative path" — if the file name in the diagnostic changed with splitting, whoever
    rebuilds the series would not know what to look for.
    """
    if not all(part.is_file() for part in component_parts(path)):
        raise DistError(f"Assembly input is missing: {path}")


def _container_of(path: Path) -> VerifiedContainer:
    """Open a component and take it through the merge (a thin translation of
    {@link karume.verify.verify_container}).

    NOTE: **no** per-path memo — series directories are mutable places rewritten by swaps (module
    doc), so a process-lifetime copy would silently return "other files at the same path". The
    path that opened a container twice within one demand computation was folded into passing the
    opened result along ({@link component_demand_bytes}).
    """
    assert_component_present(path)
    try:
        return verify_container(component_parts(path))
    except (ContainerError, ContainerFormatError) as cause:
        raise DistError(f"{path}: cannot be read as a container: {cause}") from cause


def storage_dtypes(path: Path) -> set[str]:
    """**The storage vocabulary** a component actually uses (codec decode paths — the union over
    the binding table).

    The vocabulary is the codec ledger's `layout` (`f32` / `f16` / `bf16` / `i32` / `i8` / `i4` /
    `i4f16` / `i2`); the safetensors-dialect dtype names (`F16` / `I4`) were retired. The union is
    taken because mixed assets put several codecs in one container (in an `i4` series the scales
    are f32 and the weights i4).

    MUST: the source of truth is **the binding table** (the descriptor's `encoding.codec`), not
    the dtype label (ADR 0071 decision 3's "a label does not claim the storage" — labels are a
    vocabulary for selection and display).
    """
    verified = _container_of(path)
    found: set[str] = set()
    for bound in verified.graphs.values():
        for supply in bound.supplies.values():
            entry = codec_entry(supply.encoding.codec)
            found.add(entry.layout)
            if supply.scale is not None and entry.scale_dtype is not None:
                # A companion scale's dtype is the codec's (f32, or the f16 steps of
                # `int4-sym-g-f16` — container-v1 §6.1) — this is the basis for mixed assets
                # "containing f32", so it is added explicitly from the binding table.
                found.add(entry.scale_dtype)
    return found


def assert_storage(role: str, path: Path, requirements: Mapping[str, str]) -> None:
    """Check that the storage a role requires exists in the binding table (unrelated roles pass
    through).

    The requirement table is an argument because role names collide across pipelines (Anima's
    `text_encoder` requires f16, SBV2's `text_encoder` requires i8). Mixing them into one table
    would silently apply one requirement to the other.
    """
    required = requirements.get(role)
    if required is None:
        return
    found = storage_dtypes(path)
    if required not in found:
        raise DistError(
            f"{role}: storage of {path} has no {required} (actual: {sorted(found)})."
            " Check the --dtype the series was built with (an f16 series needs the fake-quant of"
            " --dtype f16)"
        )


def assert_storage_absent(role: str, path: Path, forbidden: Mapping[str, tuple[str, ...]]) -> None:
    """Check that storage a role **must not have** is absent from the binding table (unrelated
    roles pass through).

    The presence check of {@link assert_storage} alone **lets through the mix-up of plugging a
    compressed series' asset into an f32 entry** — the container of a compressed series holds its
    ineligible weights (biases / norms / graph constants, quantization scales too) as f32, so
    "contains f32" is true for f16 / i8 assets as well. Only by bracketing the one-way presence
    check from both sides is "series × storage" determined uniquely as a set (the detection limit
    of ADR 0027 / 0029 — **a mixed-up series root cannot in principle be detected by the numeric
    checks**, so this is the only detector).

    MUST: the prohibition is held **as a set per role** (with just one, when there are two or more
    compressed series, the one not named would silently pass through). The reverse (an f32 asset in
    a compressed entry) fails in {@link assert_storage} through the missing requirement, so the
    prohibition table is needed only for the plain entries.
    """
    banned = forbidden.get(role)
    if not banned:
        return
    found = storage_dtypes(path)
    intruders = [dtype for dtype in banned if dtype in found]
    if intruders:
        raise DistError(
            f"{role}: storage of {path} has {' / '.join(intruders)}"
            f" (actual: {sorted(found)})."
            f" An asset of a compressed series is mixed into an entry that should point at the"
            f" plain f32 series"
            " (a mixed-up series root — the numeric checks cannot detect it, so it fails here)"
        )


def sha256_file(path: Path) -> str:
    """Take a file's sha256 (64 lowercase hex digits) by streaming."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def place_file(source: Path, dest: Path) -> None:
    """Place `source` at `dest` as **an independent copy**.

    MUST: no hard links — the series writer (`open("wb")` of `emit`) overwrites existing files by
    truncating them, so a distribution placed by link would be silently rewritten by a re-export of
    the series, and the manifest's sha256 would disagree with the files (verify_dist does not take
    the sha256 again, so it stays silent).
    """
    if not source.is_file():
        raise DistError(f"Assembly input is missing: {source}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, dest)


@dataclass(frozen=True)
class Artifact:
    """One file going into the distribution. `rel_path` is a relative path **inside the model
    subtree**.

    There are only two origins: **placement** from the series (`source`) and a **generated file**
    the assembly produces (`payload` — tables moved from `.npy` / ckpt and the like). Exactly one
    of the two is held. A generated file is held as bytes so that "its content is fixed before
    placement" can be handled with the same discipline as the sharing decision
    ({@link assemble_family}). Generated files can only go into `assets` entries; roles weights
    point at are limited to placement ({@link ModelPlan.__post_init__}).
    """

    rel_path: str
    source: Path | None = None
    payload: bytes | None = None

    def __post_init__(self) -> None:
        if (self.source is None) == (self.payload is None):
            raise DistError(
                f"{self.rel_path}: an Artifact must hold exactly one of source / payload"
            )


def materialize(artifact: Artifact, dest: Path) -> None:
    """Create the content of an `Artifact` at `dest` (placement or write)."""
    if artifact.source is not None:
        place_file(artifact.source, dest)
        return
    assert artifact.payload is not None  # invariant of __post_init__
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(artifact.payload)


def table_payload(key: str, table: np.ndarray) -> bytes:
    """Bytes of a safetensors file with a single tensor (the distribution form of a table moved
    from `.npy` / ckpt)."""
    return save({key: table})


def ir_graph(path: Path, verified: VerifiedContainer | None = None) -> Mapping[str, Any]:
    """Read **the IR v2 graph JSON** from a container's graph descriptor (reads only part 0).

    A `karume/5` container has one graph per container (ADR 0109 decision 2), so the only value of
    `graphs` is returned. IR v2 differs from v1 in 4 points (docs/ir-v2.md); the two that matter to
    callers are that **initializer names are renamed to tensor keys** and that `storage` moved out
    to the binding table.

    Passing `verified` uses that container as is (it is not reopened) — the option for callers
    that use both the merge result and the graph document within one demand computation
    ({@link component_demand_bytes}) so they do not open it twice.
    """
    graphs = (verified if verified is not None else _container_of(path)).read.graph.graphs
    if len(graphs) != 1:
        raise DistError(f"{path}: has {len(graphs)} graphs (`karume/5` is one graph per container)")
    return next(iter(graphs.values()))


def graph_inputs(graph: Mapping[str, Any], path: Path) -> dict[str, list[Any]]:
    """`{name: shape}` of the graph inputs (checking the order is the caller's job — this is a
    lookup table)."""
    inputs = graph.get("inputs")
    if not isinstance(inputs, list):
        raise DistError(f"{path}: IR metadata has no inputs")
    return {
        item["name"]: item["shape"]
        for item in inputs
        if isinstance(item, dict) and isinstance(item.get("shape"), list)
    }


def preprocessor_int(raw: Mapping[str, Any], key: str, where: str) -> int:
    """Check and read an integer field (a dimension) of a preprocessing config.

    Shared by the recipes that read HF's `preprocessor_config.json` (SigLIP2 / Depth Anything V2)
    — the value range matches each TS-side `config.ts`, and adding copies that move independently
    would let one of them loosen. A reader with not a single piece of model-specific knowledge, so
    it lives in core (no private names across the boundary).
    """
    value = raw.get(key)
    # bool is a subclass of int. Letting `"height": true` through as 1 would loosen the dimension
    # reconciliation.
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise DistError(f"{where}: {key} is not a positive integer ({value!r})")
    return value


def preprocessor_channels(
    raw: Mapping[str, Any], key: str, where: str, *, channels: int, positive: bool
) -> list[float]:
    """Check and read `image_mean` / `image_std` (the same value range as the TS-side
    `config.ts`).

    0 is not allowed in `std` because division by 0 would produce `pixel_values` of `±Infinity`
    without raising. The channel count is **a constant held by the calling recipe** (only the
    reader is shared; each recipe's declaration stays in its own place).

    Finiteness is checked separately because this value goes into `karume.json` through
    `pipelineConfig`. The upstream config is read with Python's `json.loads`, which **accepts
    `NaN` / `Infinity` by default** — letting them through would bake literals that are not
    standard JSON into the manifest, and the browser's `JSON.parse` would fail (the counterpart of
    {@link manifest_text}). `NaN` also slips past the positivity check, since `<= 0` is false.
    """
    value = raw.get(key)
    if not isinstance(value, list) or len(value) != channels:
        raise DistError(f"{where}: {key} is not an array of length {channels} ({value!r})")
    channels_out: list[float] = []
    for entry in value:
        if not isinstance(entry, int | float) or isinstance(entry, bool):
            raise DistError(f"{where}: {key} has a non-numeric element ({value!r})")
        if not math.isfinite(entry):
            raise DistError(f"{where}: {key} has a non-finite element ({value!r})")
        if positive and entry <= 0:
            raise DistError(f"{where}: {key} has a non-positive element ({value!r})")
        channels_out.append(float(entry))
    return channels_out


@dataclass(frozen=True)
class WeightFiles:
    """One dtype of weights (`{container}` of `karume/5`). The content is **a role name**.

    The actual path belongs to {@link Artifact} — folding into `shared/` moves the path to
    `shared/…`, so if the declaring side held the path directly, the two places would move
    independently.

    There is only one field because a container corresponds to **one component role** (ADR 0109
    decision 2). The `extras` of `karume/4` were retired, and their one kind of content
    (`rope_base`) moved into the container's assets (ADR 0109 decision 4). This points at the
    container's **one representative role**; how many parts are actually declared is resolved from
    the files at assembly time by {@link expand_weight_parts}.
    """

    file: str


@dataclass(frozen=True)
class ModelPlan:
    """The assembly plan for one model — **every check and read done**, not a single byte written
    yet. Family assembly starts placement only after receiving N of these.
    """

    name: str
    #: Pipeline contract (`anima/1` — made per model by ADR 0041 decision 2).
    pipeline: str
    #: Role name → one file going into the distribution.
    artifacts: Mapping[str, Artifact]
    #: weights name → dtype label → files (pointed at by role name).
    weights: Mapping[str, Mapping[str, WeightFiles]]
    #: assets name → role name (unconditional files that do not depend on the quant choice).
    assets: Mapping[str, str]
    quants: Mapping[str, Any]
    default_quant: str
    pipeline_config: Mapping[str, Any]

    def __post_init__(self) -> None:
        """Roles pointed at by weights entries MUST be **placements from the series** (`source`)
        only.

        A generated file (`payload`) is not a container, so if it went into weights it would be
        declared as `parts: [1 element]` without the part resolution of
        {@link expand_weight_parts}, the demand of {@link bake_required_limits} or the full
        verification of {@link assert_weight_components_verified} (a misspelling that passes the
        role name of a table / sidecar to weights would pass every check green and fail only in
        the user's `createSession`). It fails at the plan's entry point.
        """
        for labels in self.weights.values():
            for files in labels.values():
                # A missing role itself fails in the expansion, which spells out the role name
                # ({@link expand_weight_parts}).
                artifact = self.artifacts.get(files.file)
                if artifact is not None and artifact.source is None:
                    raise DistError(
                        f"{self.name}.weights: role '{files.file}' points at a generated file"
                        f" (payload) ({artifact.rel_path}) — weights take only placements from"
                        " the series (source)"
                    )


class PartitionedPlan(NamedTuple):
    """The output of {@link expand_weight_parts} — the plan with parts expanded, and the table
    that assigns them."""

    plan: ModelPlan
    #: Representative role name of weights → **ordered** part role names (from part 0; one element
    #: for the single form).
    parts: Mapping[str, tuple[str, ...]]


#: Spelling of the part role names the expansion creates (`<representative role>#<1-based
#: number>`). Role names are internal keys that do not appear in the manifest, so `#` is only a
#: separator that cannot collide with "role names a recipe wrote".
PART_ROLE_SEPARATOR = "#"


def expand_weight_parts(plan: ModelPlan) -> PartitionedPlan:
    """When a weights role points at **a container in split form**, expand it into per-part
    roles.

    MUST: the number of parts is resolved **from the files** ({@link component_parts}) — how many
    parts there are depends on the bytes written, so it cannot be written in a pipeline's table or
    a recipe's constant. Having a table hold it would allow a silent wrong declaration with
    matching shape and type: "a re-export changed the count but the declaration is the previous
    one".

    The single form (files without numbering) passes through as a one-element list — `karume/5`'s
    `container.parts` MUST have two or more elements (ADR 0109 decision 3), so
    {@link _assert_manifest_shape} fails it by name. For an expanded role, the representative role
    is **removed from artifacts** and an `Artifact` is made per part (the relative path is the
    representative path with the same numbering rule applied — the series-side file name and the
    distribution's file name come from the same single spelling).
    """
    roles = sorted({files.file for labels in plan.weights.values() for files in labels.values()})
    # A spelling that points at **the same role** from an entry other than weights (assets) would
    # leave just the declaring side stranded once the expansion removes the role name. There are
    # only a few, so they are collected and failed by name.
    elsewhere = set(plan.assets.values())
    artifacts = dict(plan.artifacts)
    parts: dict[str, tuple[str, ...]] = {}
    for role in roles:
        artifact = plan.artifacts.get(role)
        if artifact is None:
            raise DistError(
                f"{plan.name}: weights point at role '{role}', which is not in artifacts"
                f" (roles: {sorted(plan.artifacts)})"
            )
        assert artifact.source is not None  # invariant of ModelPlan.__post_init__
        sources = component_parts(artifact.source)
        if len(sources) == 1:
            parts[role] = (role,)
            continue
        if role in elsewhere:
            raise DistError(
                f"{plan.name}: assets also point at role '{role}' of a split-form container"
                " (that entry is a single file reference, so it cannot declare several parts)"
            )
        del artifacts[role]
        members: list[str] = []
        total = len(sources)
        for index, source in enumerate(sources, start=1):
            member = f"{role}{PART_ROLE_SEPARATOR}{index}"
            if member in artifacts:
                raise DistError(
                    f"{plan.name}: part role name '{member}' collides with an existing role"
                    f" (role names containing '{PART_ROLE_SEPARATOR}' cannot be used)"
                )
            artifacts[member] = Artifact(
                numbered_name(artifact.rel_path, index, total), source=source
            )
            members.append(member)
        parts[role] = tuple(members)
    return PartitionedPlan(replace(plan, artifacts=artifacts), parts)


def generator_tag() -> str:
    """The value written into `generator` (ADR 0041 decision 2 — for matching bug reports, no
    execution semantics)."""
    return f"karume/{importlib.metadata.version('karume')}"


def assert_model_name(name: str) -> str:
    """Check and return a model name (a manifest key and a directory name in the repository)."""
    if not MODEL_NAME_RE.fullmatch(name):
        raise DistError(
            f"Model name '{name}' does not match the allowed characters [A-Za-z0-9._-] (no"
            " leading dot) — a model name is also a directory name in the repository, so it is"
            " restricted to the same set as the hub's path check"
        )
    if name == SHARED_DIRNAME:
        raise DistError(
            f"Model name '{SHARED_DIRNAME}' cannot be used (it collides with the shared files'"
            " location)"
        )
    return name


def assert_rel_path(rel_path: str, where: str) -> str:
    """Check and return that `Artifact.rel_path` stays inside the model subtree.

    Checked with the same **allow-list** as the hub's `assertPath` (`packages/hub/src/manifest.ts`).
    It is not a deny-list (rejecting `..`) for the same reason as in the hub — the fetch layer
    passes dots through even when it percent-encodes segments, so a gap in the list becomes
    traversal as is.

    The harm on the assembly side is a step worse than in the fetch layer: placement writes to
    `out_dir / f"{model name}/{rel_path}"`, so a shape like `A/../../victim` **truncates an
    existing file outside staging**. Moreover, the manifest-side root check
    ({@link _assert_manifest_shape}) looks only at the first segment, and the undeclared-file check
    ({@link verify_dist}) looks only inside staging, so **not a single check after writing fires**.
    """
    for segment in rel_path.split("/"):
        if not segment:
            raise DistError(
                f"{where}: rel_path '{rel_path}' has an empty segment"
                " (leading / trailing / consecutive slashes)"
            )
        if segment.startswith("."):
            raise DistError(
                f"{where}: rel_path '{rel_path}' has a segment with a leading dot '{segment}'"
            )
        if not PATH_SEGMENT_RE.fullmatch(segment):
            raise DistError(
                f"{where}: rel_path '{rel_path}' has a segment '{segment}' that does not match"
                " the allowed characters [A-Za-z0-9._-]"
            )
    return rel_path


def assert_plan_paths(plans: Sequence[ModelPlan]) -> None:
    """Fail on model names and `rel_path`s that do not fit, **before placement**.

    Only the name and path checks were outside the planning step, so a programmer error in the
    repository (a recipe's constant) could turn straight into a write outside staging
    ({@link assert_rel_path}).

    MUST: **one relative path can hold only one entry, within a model too** — {@link _plan_shared}
    enforces the same MUST **across** models, yet **within** a model it passed through. When two
    roles claim the same path, the later role overwrites the earlier one's file, and the manifest
    declares both roles with **the later digest**. Files and table agree, so {@link verify_dist}
    stays silent, and a distribution whose earlier role's session reads another graph ships as is.
    """
    for plan in plans:
        assert_model_name(plan.name)
        seats: dict[str, str] = {}
        for role, artifact in plan.artifacts.items():
            assert_rel_path(artifact.rel_path, f"{plan.name}.{role}")
            claimed = seats.setdefault(artifact.rel_path, role)
            if claimed != role:
                raise DistError(
                    f"{plan.name}: relative path '{artifact.rel_path}' is claimed by both role"
                    f" '{claimed}' and '{role}' (one relative path can hold only one entry —"
                    " the later one overwrites, and the manifest declares both roles with the"
                    " later digest)"
                )


def complete_quant_weights(
    weights: Mapping[str, Mapping[str, WeightFiles]],
    quants: Mapping[str, Any],
) -> dict[str, Any]:
    """Fill a quant's `weights` mapping into **a total mapping** (the hub's acceptance requirement
    — ADR 0041 decision 3).

    weights with only one dtype label cannot be chosen by a quant, so the table does not have to
    write them and they are filled in mechanically here (making the table write them would
    duplicate "an entry without a choice" once per quant and require a bulk update whenever a
    dtype is added = duplicated derivable state). **Weights with two or more labels fail unless
    the table names one** — picking a default on its own would silently distribute another storage
    form.
    """
    completed: dict[str, Any] = {}
    for quant_name, quant in quants.items():
        declared = quant["weights"]
        for name in declared:
            if name not in weights:
                raise DistError(f"quant '{quant_name}': unknown weights '{name}'")
            if declared[name] not in weights[name]:
                raise DistError(
                    f"quant '{quant_name}': weights '{name}' has no dtype '{declared[name]}'"
                    f" (available: {sorted(weights[name])})"
                )
        # MUST: the order is **the declaration order of weights** (not the order the table
        # wrote) — if only the filled entries piled up at the end, the same role would be ordered
        # differently in the manifest's weights section and quant section.
        mapping: dict[str, str] = {}
        for name, labels in weights.items():
            if name in declared:
                mapping[name] = declared[name]
                continue
            if len(labels) != 1:
                raise DistError(
                    f"quant '{quant_name}': weights '{name}' has several dtypes {sorted(labels)}"
                    " but none is specified — the quant table must name one"
                )
            mapping[name] = next(iter(labels))
        completed[quant_name] = {**quant, "weights": mapping}
    return completed


#: Spelling of a quant's optional field (ADR 0038 §7 — read by the hub's `parseRequiredLimits`).
REQUIRED_LIMITS_KEY = "requiredLimits"


def component_demand_bytes(
    source: Path, pipeline_config: Mapping[str, Any], memo: dict[Path, int]
) -> int:
    """**The largest single resident buffer in bytes** of one component (weight tensor / state
    slot).

    Reads **only the two documents of part 0** (not a single byte of weight payload). `memo` is a
    memo per representative path — so that the same descriptor is not read again when several
    quants choose the same component (a shared text_encoder and the like).

    MUST: the demand comes from **the merged supply plan** (`bind_graphs`). A weight split into
    pieces goes back into one parent buffer on the GPU (rule ④ of container-v1 §5), so taking the
    maximum per piece would bake an understated `requiredLimits`, the costliest shape: "the
    declaration is met, yet `createSession` fails". companion scales are separate buffers and are
    counted separately. Initializers in the const region (part 1) are resident too, so they are in
    the same set.
    """
    remembered = memo.get(source)
    if remembered is not None:
        return remembered
    verified = _container_of(source)
    demand = 0
    for bound in verified.graphs.values():
        for supply in bound.supplies.values():
            demand = max(demand, sum(block.payload_bytes for block in supply.blocks))
            if supply.scale is not None:
                demand = max(demand, supply.scale.payload_bytes)
    try:
        demand = max(
            demand,
            max_state_slot_bytes(ir_graph(source, verified), pipeline_config, str(source)),
        )
    except LimitsError as cause:
        raise DistError(str(cause)) from cause
    memo[source] = demand
    return demand


def bake_required_limits(plan: ModelPlan) -> ModelPlan:
    """Return the plan with each quant's `requiredLimits` (ADR 0038 §7) derived **from the files**
    and baked in.

    The demand is the maximum over the components the quant chooses of ① the stored payload of the
    largest tensor and ② the largest state slot (dimensions with the capacity made numeric through
    `pipelineConfig`); only limits above the guaranteed defaults become fields (within the
    defaults there is no field at all — the source of truth for the rules and reasons is
    {@link karume.limits}).

    MUST: **fail loudly when the plan already writes the same field** — also giving a recipe's
    quant table a derivable value would allow "only the table claims a stale lower bound" whenever
    the storage form or the capacity changes (duplicated state where declaration and files move
    independently). Readers trust the declaration and refuse before fetching, so a stale lower
    bound gives a distribution that "meets the declaration yet does not run".
    """
    memo: dict[Path, int] = {}
    quants: dict[str, Any] = {}
    for quant_name, quant in plan.quants.items():
        where = f"{plan.name}.quants.{quant_name}"
        if REQUIRED_LIMITS_KEY in quant:
            raise DistError(
                f"{where}.{REQUIRED_LIMITS_KEY} is written by the plan — assembly derives this"
                " field from the files that go into the distribution (tensor dimensions and state"
                " slots); it is not a field for the table"
            )
        demand = 0
        for name, label in quant["weights"].items():
            labels = plan.weights.get(name)
            files = labels.get(label) if labels is not None else None
            if files is None:
                raise DistError(f"{where}.weights: '{name}' has no dtype '{label}'")
            artifact = plan.artifacts.get(files.file)
            if artifact is None:
                raise DistError(f"{where}.weights: role '{files.file}' is not in artifacts")
            # assets are not looked at — whether a table / tokenizer / sidecar becomes a resident
            # GPU buffer is the family's business and is not decided by the container's
            # declaration.
            assert artifact.source is not None  # invariant of ModelPlan.__post_init__
            demand = max(
                demand, component_demand_bytes(artifact.source, plan.pipeline_config, memo)
            )
        limits = required_limits(demand)
        quants[quant_name] = {**quant, **({REQUIRED_LIMITS_KEY: limits} if limits else {})}
    return replace(plan, quants=quants)


def file_ref(out_dir: Path, rel_path: str, sha256: str) -> dict[str, Any]:
    """The 3-field set `{path, size, sha256}` of ADR 0041 decision 2 (size is taken from the placed
    file)."""
    return {
        "path": rel_path,
        "size": (out_dir / rel_path).stat().st_size,
        "sha256": sha256,
    }


class _SharedSeat(NamedTuple):
    """An entry placed **once** at `shared/<rel_path>` — the representative file, the source's
    sha256, and the entries that point at it."""

    rel_path: str
    artifact: Artifact
    digest: str
    #: `(model name, role name)` pairs that point at this entry. Each model's manifest entry writes
    #: the same path.
    members: tuple[tuple[str, str], ...]


def _source_digest(artifact: Artifact, memo: dict[Path, str]) -> str:
    """The sha256 of an `Artifact`'s **source** (the same actual file is read only once).

    This is the only read for settling the sharing decision **before placement**, so the memo key
    is the resolved actual path — several models pointing at the same file under different
    spellings fold into one read. A generated file (`payload`) already has its content in memory,
    so no I/O is needed.
    """
    if artifact.source is None:
        assert artifact.payload is not None  # invariant of __post_init__
        return hashlib.sha256(artifact.payload).hexdigest()
    resolved = artifact.source.resolve()
    digest = memo.get(resolved)
    if digest is None:
        digest = memo[resolved] = sha256_file(resolved)
    return digest


def _artifact_size(artifact: Artifact) -> int:
    """The byte size of an `Artifact`'s content (**the content is not read** — one stat of the
    source).

    A size only for the pre-filter of the sharing decision ({@link _plan_shared}). Unlike sha256
    it only works in the direction "if they differ, the content certainly differs", so a match is
    no ground for folding.
    """
    if artifact.source is None:
        assert artifact.payload is not None  # invariant of __post_init__
        return len(artifact.payload)
    return artifact.source.stat().st_size


def _plan_shared(plans: Sequence[ModelPlan]) -> list[_SharedSeat]:
    """Decide, **before placement**, the entries where two or more models hold a file with the
    same relative path and the same sha256.

    MUST: the match condition is **both the relative path inside the model subtree and the
    sha256** (ADR 0041 decision 5's "sharing by matching paths" holds only after confirming the
    content is the same). Gathering different content into one entry would silently make one of
    the models read other weights.

    MUST: **one relative path can hold only one entry** (`shared/<relative path>` is determined by
    the path alone) — when two or more groups could fold into the same path, none of them is
    folded (each stays an independent copy in its model's subtree). Folding would make the later
    group overwrite the earlier group's file, and the earlier group's models would ship with a
    manifest pointing at **other content** ({@link verify_dist} does not take the sha256 again, so
    it stays silent).

    MUST: the decision is made **per entry** (`(model name, role name)`), not per relative path —
    when "a foldable group" and "a lone different content" share a relative path, only the group is
    folded and the lone one stays in its own subtree.

    Sources are hashed only for entries whose **(relative path, size) is used by two or more
    models** — a relative path used by one model only cannot be folded, and a different size means
    a different hash for certain, so there is no reason to read either (sha256 is taken from the
    placed file). The size is used only as **a ground for excluding**, never for folding: entries
    of the same size are still told apart by hash as before, so the set of folded entries is exactly
    the same with or without the pre-filter (even when "a foldable group" and "a lone one of
    another size" share a relative path, only the lone one drops out).
    """
    sizes: dict[tuple[str, str], int] = {}
    users: dict[tuple[str, int], set[str]] = {}
    for plan in plans:
        for role, artifact in plan.artifacts.items():
            size = sizes[(plan.name, role)] = _artifact_size(artifact)
            users.setdefault((artifact.rel_path, size), set()).add(plan.name)
    memo: dict[Path, str] = {}
    groups: dict[tuple[str, str], list[tuple[str, str]]] = {}
    samples: dict[tuple[str, str], Artifact] = {}
    for plan in plans:
        for role, artifact in plan.artifacts.items():
            if len(users[(artifact.rel_path, sizes[(plan.name, role)])]) < 2:
                continue
            key = (artifact.rel_path, _source_digest(artifact, memo))
            groups.setdefault(key, []).append((plan.name, role))
            samples.setdefault(key, artifact)
    foldable = [key for key, members in groups.items() if len({model for model, _ in members}) >= 2]
    seats = Counter(rel_path for rel_path, _ in foldable)
    shared: list[_SharedSeat] = []
    for key in foldable:
        rel_path, digest = key
        if seats[rel_path] > 1:
            continue
        shared.append(_SharedSeat(rel_path, samples[key], digest, tuple(groups[key])))
    return shared


def container_asset_bytes(source: Path) -> int:
    """**The sum of the logical lengths** of the assets a container declares (ADR 0109 decision 4
    — where the PLE / `rope_base` went).

    Reads only the descriptor (not a single byte of payload). The model card's count of "assets
    read on the host" uses this — the assets moved inside the parts, so counting only the
    manifest's `assets` (= the separate file entries) would silently drop the tables that moved into
    the container from the breakdown.
    """
    model = _container_of(source).read.model
    return 0 if model is None else sum(record.length for record in model.assets.values())


def _model_entry(
    plan: ModelPlan,
    refs: Mapping[str, dict[str, Any]],
    parts: Mapping[str, tuple[str, ...]],
    descriptors: Mapping[str, tuple[DocumentRef, DocumentRef]],
    host_assets: dict[FileRefKey, int],
) -> dict:
    """The manifest entry of one model (ADR 0109 decision 3). `refs` is role name → 3-field set.

    `parts` is the assignment table {@link expand_weight_parts} resolved from the files
    (representative role → ordered part roles), and `descriptors` are the expectations of the two
    documents taken from part 0 of **the placed files**.

    `host_assets` (part 0 identity {@link karume.modelcard.file_ref_key} →
    {@link container_asset_bytes}) is **an output parameter** and does not go into the manifest (a
    derived value only the model card's breakdown note reads — the declaration belongs to the
    container).

    MUST: the asset count opens **only the entries quants choose**. Opening dtype entries no quant
    chooses would add a second detection point to a hole whose only detector is the input
    container check ({@link assert_weight_components_verified}), and that check's fault injection
    (removing the check puts a broken container in place) would become tautological.
    """
    selected = {
        (name, label)
        for quant in plan.quants.values()
        for name, label in quant.get("weights", {}).items()
    }
    weights: dict[str, Any] = {}
    for name, labels in plan.weights.items():
        entry: dict[str, Any] = {}
        for label, files in labels.items():
            graph_ref, model_ref = descriptors[files.file]
            # `parts` is an ordered list, **with part 0 first** (the file with the header + the two
            # documents). The order MUST be by index — the index is the part's id, so touching the
            # list breaks the identifiers (container-v1 §8 / ADR 0109 decision 3).
            members = parts[files.file]
            placed = [refs[role] for role in members]
            entry[label] = {
                "container": {
                    "descriptor": {
                        "graph": graph_ref.to_document(),
                        "model": model_ref.to_document(),
                    },
                    "parts": placed,
                }
            }
            if (name, label) in selected:
                # The asset declaration is taken from **the source container** (that it is the
                # same bytes as the placed file is checked separately by {@link verify_dist} and
                # the cross-repository reconciliation).
                head = plan.artifacts[members[0]].source
                assert head is not None  # invariant of ModelPlan.__post_init__
                host_assets[file_ref_key(placed[0])] = container_asset_bytes(head)
        weights[name] = entry
    return {
        "pipeline": plan.pipeline,
        "weights": weights,
        "assets": {name: refs[role] for name, role in plan.assets.items()},
        "quants": dict(plan.quants),
        "defaultQuant": plan.default_quant,
        "pipelineConfig": dict(plan.pipeline_config),
    }


def assert_quant_presentation(where: str, quant: Mapping[str, Any]) -> None:
    """Fail on the shape and limits of a quant's display fields (`label` / `description` of ADR
    0075 decision 1).

    The hub's manifest parser defines the accepted set (ruling of 2026-10-11): the field is absent,
    or **a string within the limit** (the length is in code points — Python's `len` counts the same
    way as the hub). The hub accepts empty strings, whitespace-only values and values containing
    newlines, so they are accepted here too (the meaning is not interpreted — decision 2).
    MUST: `None` is rejected instead of being read as "not written" — the hub rejects JSON `null`,
    so reading it that way would give "a manifest that was written but cannot be read". The shared
    table `packages/hub/tests/fixtures/quant-presentation.json` binds the two sides together.
    """
    for key, limit in QUANT_PRESENTATION_LIMITS.items():
        if key not in quant:
            continue
        value = quant[key]
        if not isinstance(value, str):
            raise DistError(f"{where}.{key} is not a string ({value!r})")
        if len(value) > limit:
            raise DistError(f"{where}.{key} has {len(value)} characters, over the limit of {limit}")


def _assert_model_scale(
    name: str,
    *,
    weights: int,
    assets: int,
    quants: Mapping[str, Any],
    pipeline_config: Mapping[str, Any],
) -> None:
    """The size limits of one model (ADR 0041 decision 7) and a quant's display fields (ADR 0075
    decision 1).

    The content of `quants` is taken, not just the counts, to check the display field limits. The
    same rules are applied from both the plan ({@link ModelPlan}) and the manifest, so the limits
    are spelled in one place.
    """
    for key, count, limit in (
        ("weights", weights, MAX_WEIGHTS),
        ("assets", assets, MAX_ASSETS),
        ("quants", len(quants), MAX_QUANTS),
    ):
        if count > limit:
            raise DistError(f"{name}.{key} has {count} entries, over the limit of {limit}")
    for quant_name, quant in quants.items():
        assert_quant_presentation(f"{name}.quants.{quant_name}", quant)
    config_bytes = len(json.dumps(dict(pipeline_config), ensure_ascii=False).encode("utf-8"))
    if config_bytes > MAX_PIPELINE_CONFIG_BYTES:
        raise DistError(
            f"{name}.pipelineConfig is {config_bytes} bytes, over the limit of"
            f" {MAX_PIPELINE_CONFIG_BYTES}"
        )


def assert_plan_limits(plans: Sequence[ModelPlan]) -> None:
    """Fail on the size limits decided by the plans alone (ADR 0041 decision 7), **before
    placement**.

    `MAX_MANIFEST_BYTES` is not here — the spelling of the whole manifest cannot be measured until
    the `{path, size, sha256}` of the placed files are known, so it stays in
    {@link assert_manifest_limits}.
    """
    if len(plans) > MAX_MODELS:
        raise DistError(f"models has {len(plans)} entries, over the limit of {MAX_MODELS}")
    for plan in plans:
        _assert_model_scale(
            plan.name,
            weights=len(plan.weights),
            assets=len(plan.assets),
            quants=plan.quants,
            pipeline_config=plan.pipeline_config,
        )


def assert_plan_sources(plans: Sequence[ModelPlan]) -> None:
    """Fail, **before placement**, unless every input file the plans point at exists.

    The storage dtype requirement table ({@link assert_storage}) passes roles without a
    requirement through, so only missing "files that require no dtype", such as a tokenizer,
    slipped past the planning step and surfaced **in the middle of** placement. An
    {@link Artifact} whose source is a generated file (`payload`) already holds the content to
    write and is not covered.
    """
    for plan in plans:
        for role, artifact in plan.artifacts.items():
            if artifact.source is not None and not artifact.source.is_file():
                raise DistError(f"{plan.name}.{role}: assembly input is missing: {artifact.source}")


def weight_components(partitioned: Sequence[PartitionedPlan]) -> list[tuple[tuple[Path, ...], str]]:
    """Collect the parts (in index order) of the **local** containers weights point at, and the
    weights key of each entry.

    Shared components (entries where several models point at the same series) are deduplicated by
    (parts, key) — verifying the same bytes under the same name twice does not change the result.
    The key is **included** in the folding because one container sitting in entries with two
    different keys is exactly the defect the graph-name check of {@link
    assert_weight_components_verified} catches (one container holds only one graph, so both
    cannot hold). Roles of generated files (`payload`) do not appear in weights
    ({@link ModelPlan.__post_init__} rejects them at the entry point), so every role is collected
    here.

    NOTE: only weights are looked at. `assets` (table safetensors such as tokenizers and style
    tables) are not containers, so the container check does not apply to them.
    """
    components: list[tuple[tuple[Path, ...], str]] = []
    seen: set[tuple[tuple[Path, ...], str]] = set()
    for item in partitioned:
        seats = sorted(
            {
                (files.file, component)
                for component, labels in item.plan.weights.items()
                for files in labels.values()
            }
        )
        for role, component in seats:
            sources: list[Path] = []
            for member in item.parts[role]:
                source = item.plan.artifacts[member].source
                assert source is not None  # invariant of ModelPlan.__post_init__
                sources.append(source)
            entry = (tuple(sources), component)
            if entry in seen:
                continue
            seen.add(entry)
            components.append(entry)
    return components


def assert_weight_components_verified(partitioned: Sequence[PartitionedPlan]) -> dict[Path, str]:
    """Verify every weights container against **all the rules the declarations decide** (before
    placement).

    Returns, for part k (k ≥ 1) of the split form, source path → the `parts[k-1].sha256` the
    descriptor declares — used for the comparison with the sha256 of the placed files
    ({@link _materialize_family}).

    MUST: assembly does not trust input containers as "verified in the past". Series directories
    are mutable places rewritten by swaps (module doc), so the operational mistake of **pointing
    `--series` at a series built by an old exporter** can really happen. Assembly itself only
    looks at the storage and part of the IR ({@link ir_graph}), so a broken descriptor / binding
    table would slip past the family-specific checks into the distribution and fail only in the
    user's `createSession`.

    What applies is the reader's structural checks (the two documents, the block index, the codec
    ledger), the merge (`bind_graphs`) and **the IR acceptance rules** (op vocabulary / runtime
    support / op contracts — {@link karume.verify.assert_ir_accepted}). All of them look **only at
    declarations**, so not a single byte of the several GB of payload is read (container-v1 §7's
    three separate hash roles). The content of the parts is not trusted on the grounds that
    publish passed it — the sha256 of the files placed (taken anyway for the manifest) is compared
    with the descriptor's `parts[].sha256` returned here (no additional I/O).

    MUST: the IR acceptance rules are applied here — with structural checks alone, a container
    declaring "ops outside the vocabulary / attrs the runtime does not support / shapes that break
    a contract" would sit in the distribution and fail only in the user's `createSession` (the
    assembly-side half of the purpose stated in `karume.verify`'s module doc).

    MUST: **the container's graph name == the weights key of its entry** is checked here too. The
    runtime looks it up by name with `prepareContainer(opened, <weights key>)` (container-v1
    §2.1), so a container whose spelling diverges would sit there along with the manifest and
    first become "the container has no graph" in the user's `createSession`. The writer-side
    spelling checks (each recipe's constant against the AST) look only at the source, so **this is
    the only check that compares files with declarations**.
    """
    declared: dict[Path, str] = {}
    for parts, component in weight_components(partitioned):
        try:
            verified = verify_container(parts)
            assert_ir_accepted(verified.read)
        except (ContainerError, ContainerFormatError) as cause:
            raise DistError(
                f"{parts[0]}: assembly input does not satisfy the container rules: {cause}"
            ) from cause
        graphs = set(verified.read.graph.graphs)
        if graphs != {component}:
            raise DistError(
                f"{parts[0]}: the container's graph names {sorted(graphs)} differ from the"
                f" weights key '{component}' — the runtime looks the graph up by this key"
                " (container-v1 §2.1), so distributing it as is would fail in createSession"
            )
        model = verified.read.model
        # In the single form a part is only a range of the file, so it cannot be compared with
        # the file's sha256 (in weights entries {@link _assert_manifest_shape} rejects the single
        # form itself). A `krg` declares no parts (in weights entries placement's
        # {@link _descriptor_refs} rejects it).
        if model is not None and len(parts) > 1:
            for part in model.parts:
                declared[parts[part.index]] = part.sha256
    return declared


def assert_root_files(root_files: Mapping[str, str]) -> None:
    """Fail unless the names written at the distribution repository's root fit the legal text
    entries ({@link LEGAL_PATHS}).

    MUST: unknown names fail loudly — the root is the only place the manifest does not declare, so
    letting names through here would loosen the net of "no undeclared files" ({@link verify_dist})
    by one name each time. The meaning of the entries (files with redistribution conditions the
    upstream licence requires) cannot be stated in a type, so the accepted set is spelled out as a
    check.
    """
    unknown = sorted(set(root_files) - LEGAL_PATHS)
    if unknown:
        raise DistError(
            f"Only {sorted(LEGAL_PATHS)} can be written at the distribution repository's root"
            " (entries for legal text only): " + ", ".join(unknown)
        )


#: The sha256 that zero-length content (a part 1 with an empty const region) declares. There is
#: only one such content, so the value is unique (ADR 0109 decision 3 — equal to the hub's
#: `EMPTY_SHA256`).
EMPTY_SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


def assert_container_limits(where: str, container: Mapping[str, Any]) -> None:
    """Fail unless one container entry fits the limits and rules of `karume/5` (ADR 0109
    decision 6).

    MUST: the limits are taken from **the container-side constants** (`karume.container`) — the
    hub side (`packages/hub/src/manifest.ts`) holds the same values, so no third copy is added.
    The constant reconciliation test of `tests/test_dist.py` checks them mechanically.

    Only **what closes over the declarations alone** is checked (counts, ceilings, the `size: 0`
    rule, part 0's length, uniformity of cross-repository references, descriptor length).
    Consistency of descriptor and parts, the block index and the codec ledger belong to
    `verify_container` and are not placed here (no second check point).

    MUST: missing fields fail with `DistError` too (no raw `KeyError` / `TypeError` from plain
    indexing) — {@link verify_dist} is a check that receives **a `karume.json` from outside** (any
    local distribution), so an exception outside assembly's vocabulary would make "a broken
    manifest" and "a bug in the check" indistinguishable.
    """
    documents: list[int] = []
    descriptor = _dist_object(container.get("descriptor"), f"{where}.descriptor")
    for key in ("graph", "model"):
        document = _dist_object(descriptor.get(key), f"{where}.descriptor.{key}")
        length = document.get("length")
        if not isinstance(length, int) or isinstance(length, bool) or length <= 0:
            raise DistError(
                f"{where}.descriptor.{key}.length is not a positive integer ({length!r})"
            )
        if length > MAX_DESCRIPTOR_BYTES:
            raise DistError(
                f"{where}.descriptor.{key}.length {length} is over the limit of"
                f" {MAX_DESCRIPTOR_BYTES}"
            )
        documents.append(length)
    parts = container.get("parts")
    if not isinstance(parts, list) or len(parts) < 2:
        found = f"{len(parts)} entries" if isinstance(parts, list) else repr(parts)
        raise DistError(
            f"{where}.parts is not an array of two or more entries (actual: {found})"
            " — part 0 (descriptor) + part 1 (const) are required (container-v1 §8)"
        )
    if len(parts) > MAX_PARTS:
        raise DistError(f"{where}.parts has {len(parts)} entries, over the limit of {MAX_PARTS}")
    refs = [_dist_part_ref(part, f"{where}.parts[{index}]") for index, part in enumerate(parts)]
    part0 = HEADER_BYTES + documents[0] + documents[1]
    if refs[0]["size"] != part0:
        raise DistError(
            f"{where}.parts[0]: size {refs[0]['size']} of part 0 differs from"
            f" header {HEADER_BYTES} + 2 documents {documents[0]} + {documents[1]} = {part0}"
        )
    head = (refs[0].get("repo"), refs[0].get("revision"))
    for index, ref in enumerate(refs):
        if ref["size"] > PART_MAX_BYTES:
            raise DistError(
                f"{where}.parts[{index}]: part '{ref['path']}' is {ref['size']} bytes,"
                f" over the limit of {PART_MAX_BYTES} (the part length ceiling — container-v1 §10)"
            )
        # Length 0 is only the shape "part 1 with an empty const region". Part 0 has already
        # been checked to be exactly "header + 2 documents" by this point (the 2 documents are
        # positive integers, so it cannot be 0), so the only rule left is the sha256 of the
        # empty string.
        if ref["size"] == 0 and ref["sha256"] != EMPTY_SHA256:
            raise DistError(
                f"{where}.parts[{index}]: the sha256 of size 0 is not the empty string's value"
                f" {EMPTY_SHA256}"
            )
        # Cross-repository references are **per container** (ADR 0109 decision 3) — one-sided or
        # mixed references would fetch the remaining parts from the session's repo, where they
        # are not, failing only in the middle of the fetch.
        if (ref.get("repo"), ref.get("revision")) != head:
            raise DistError(
                f"{where}.parts[{index}]: cross-repository references are mixed within the"
                f" container (part 0 is {head[0] or 'this repo'} / this part is"
                f" {ref.get('repo') or 'this repo'})"
            )


def _dist_object(value: Any, where: str) -> Mapping[str, Any]:
    """Take one section of the manifest as an object (translates missing fields into assembly
    vocabulary)."""
    if not isinstance(value, dict):
        raise DistError(f"{where} is not an object (actual: {value!r})")
    return value


def _dist_part_ref(value: Any, where: str) -> Mapping[str, Any]:
    """Fail unless one part's 3-field set (`path` / `size` / `sha256`) is complete."""
    ref = _dist_object(value, where)
    if not isinstance(ref.get("path"), str) or not ref["path"]:
        raise DistError(f"{where}.path is not a non-empty string (actual: {ref.get('path')!r})")
    size = ref.get("size")
    if not isinstance(size, int) or isinstance(size, bool) or size < 0:
        raise DistError(f"{where}.size is not a non-negative integer (actual: {size!r})")
    if not isinstance(ref.get("sha256"), str) or not ref["sha256"]:
        raise DistError(f"{where}.sha256 is not a non-empty string (actual: {ref.get('sha256')!r})")
    return ref


def assert_manifest_limits(manifest: Mapping[str, Any]) -> None:
    """Fail on the size limits (ADR 0041 decision 7 + ADR 0109 decision 6) first, on the writing
    side."""
    models = manifest["models"]
    if len(models) > MAX_MODELS:
        raise DistError(f"models has {len(models)} entries, over the limit of {MAX_MODELS}")
    for name, model in models.items():
        _assert_model_scale(
            name,
            weights=len(model["weights"]),
            assets=len(model["assets"]),
            quants=model["quants"],
            pipeline_config=model["pipelineConfig"],
        )
        for component, labels in model["weights"].items():
            for label, entry in labels.items():
                assert_container_limits(
                    f"{name}.weights.{component}.{label}.container", entry["container"]
                )
    total = len(manifest_text(manifest).encode("utf-8"))
    if total > MAX_MANIFEST_BYTES:
        raise DistError(f"manifest is {total} bytes, over the limit of {MAX_MANIFEST_BYTES}")


def manifest_text(manifest: Mapping[str, Any]) -> str:
    """The spelling written into `karume.json` (with a trailing newline — the limit check measures
    this spelling too).

    allow_nan=False is required (same text as `karume.ir.IrGraph.to_json`) — NaN / Infinity are
    not standard JSON literals, and the browser's `JSON.parse` (the hub's reader) fails on them.
    Python's `json.dumps` spells them by default and `json.loads` reads them back by default, so
    not a single check on the writing side, {@link verify_dist} included, would fire — to match
    the accepted set to the runtime side, writing fails at this point.
    """
    return json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n"


def _descriptor_refs(part0: Path) -> tuple[DocumentRef, DocumentRef]:
    """Part 0 file → the expectations of the two documents (translates read failures into
    assembly vocabulary)."""
    try:
        return read_descriptor_refs(part0)
    except ContainerFormatError as cause:
        raise DistError(
            f"{part0}: cannot take the expectations of the two documents: {cause}"
        ) from cause


def _materialize_family(
    partitioned: Sequence[PartitionedPlan],
    out_dir: Path,
    default_model: str,
    external: Mapping[str, Mapping[str, Any]],
    declared_parts: Mapping[Path, str],
) -> tuple[dict[str, Any], dict[FileRefKey, int]]:
    """Lay the checked plans out in `out_dir`, write `karume.json`, and return the manifest.

    The second return value is a derived value for the model card (part 0 identity
    {@link karume.modelcard.file_ref_key} → {@link container_asset_bytes}) — it does not go into
    the manifest (the source of truth is the container's asset declaration).

    ① decide the shared entries ({@link _plan_shared} — looks only at the sources' sha256) → ②
    place the shared entries once under `shared/` and the rest in each model's subtree → ③ the
    manifest from the files.

    Roles listed in `external` (role name → cross-repository reference) are **not placed in this
    repository** — the point of a reference is "not uploading the same bytes to two
    repositories", so placing the file and also writing a reference would gain nothing.

    MUST: the sharing decision is settled **before placement** — copying to every model and then
    folding back costs ~3MS of logical I/O per shared file (size S, M models): "M copies + M hashes
    + 1 move". Deciding first and then placing needs only one copy and one hash of S (most of a
    distribution is shared assets).

    The target is the staging directory the caller ({@link assemble_family}) prepares, assumed to
    be **built from empty** (the distribution target is not updated directly, so a failure midway
    only means discarding it).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    plans = [item.plan for item in partitioned]
    placed: dict[tuple[str, str], str] = {}
    digests: dict[str, str] = {}
    folded: dict[tuple[str, str], str] = {}
    # MUST: the shared entries are decided on **the set without the cross-repository roles** — if
    # a role not placed in this repository took a `shared/` entry, one file no model declares
    # would be left (several models borrowing the same bytes = exactly what happens with ADR
    # 0087's extra repository). It sees the same set as the placement loop below, which skips
    # `external`.
    shareable = [
        replace(
            plan,
            artifacts={role: a for role, a in plan.artifacts.items() if role not in external},
        )
        for plan in plans
    ]
    for seat in _plan_shared(shareable):
        target = f"{SHARED_DIRNAME}/{seat.rel_path}"
        materialize(seat.artifact, out_dir / target)
        # sha256 is taken from **the placed file** (structurally removes the failure mode of
        # table and files disagreeing).
        digests[target] = sha256_file(out_dir / target)
        # MUST: only shared entries are also compared with the source's sha256 — only one copy
        # is placed, so a broken copy would make **every model point at the same broken file**.
        # Declaration and file still agree, so {@link verify_dist} stays silent (it does not take
        # the sha256 again).
        if digests[target] != seat.digest:
            raise DistError(
                f"{target}: the sha256 of the placed file disagrees with the source"
                f" (source {seat.digest} / placed {digests[target]})"
            )
        for member in seat.members:
            folded[member] = target
    for plan in plans:
        for role, artifact in plan.artifacts.items():
            if role in external:
                continue
            target = folded.get((plan.name, role))
            if target is not None:
                placed[(plan.name, role)] = target
                continue
            rel_path = f"{plan.name}/{artifact.rel_path}"
            materialize(artifact, out_dir / rel_path)
            placed[(plan.name, role)] = rel_path
            digests[rel_path] = sha256_file(out_dir / rel_path)
    # MUST: compare the placed part files with the descriptor's `parts[k-1].sha256` (container-v1
    # §7's "for reconciliation at publication and repacking"). The manifest's sha256 is taken from
    # the placed files, so a series part broken at the same length would be put in place with
    # declaration and file agreeing, and fetching through HF does not apply the block sha256 (a
    # verified source) — there would be no check to notice it after publication. Cross-repository
    # entries are not placed in this repository and are not covered (identity with the target is
    # checked by {@link external_refs}).
    for item in partitioned:
        for members in item.parts.values():
            for index, member in enumerate(members):
                source = item.plan.artifacts[member].source
                expected = declared_parts.get(source) if source is not None else None
                if member in external or expected is None:
                    continue
                target = placed[(item.plan.name, member)]
                if digests[target] != expected:
                    raise DistError(
                        f"{target}: the sha256 of part {index} disagrees with the descriptor's"
                        f" declaration (declared {expected} / placed {digests[target]})"
                    )

    models: dict[str, Any] = {}
    host_assets: dict[FileRefKey, int] = {}
    for item in partitioned:
        plan = item.plan
        refs = {}
        for role in plan.artifacts:
            reference = external.get(role)
            if reference is not None:
                refs[role] = dict(reference)
                continue
            rel_path = placed[(plan.name, role)]
            refs[role] = file_ref(out_dir, rel_path, digests[rel_path])
        # The descriptor expectations are taken from **the placed files** too (structurally
        # removes the failure mode of table and files disagreeing — the module doc's MUST).
        # Cross-repository entries are not placed in this repository, so they are taken from the
        # local source instead — {@link external_refs} has confirmed "the target is identical to
        # the bytes this assembly would have built", so it is the same fact.
        descriptors: dict[str, tuple[DocumentRef, DocumentRef]] = {}
        for role, members in item.parts.items():
            head = members[0]
            if head in external:
                source = plan.artifacts[head].source
                assert source is not None  # invariant of ModelPlan.__post_init__
                descriptors[role] = _descriptor_refs(source)
            else:
                descriptors[role] = _descriptor_refs(out_dir / placed[(plan.name, head)])
        models[plan.name] = _model_entry(plan, refs, item.parts, descriptors, host_assets)
    manifest = {
        "format": MANIFEST_FORMAT,
        "generator": generator_tag(),
        "defaultModel": default_model,
        "models": models,
    }
    assert_manifest_limits(manifest)
    (out_dir / MANIFEST_FILENAME).write_text(manifest_text(manifest), encoding="utf-8")
    return manifest, host_assets


def assemble_family(
    plans: Sequence[ModelPlan],
    out_dir: Path,
    default_model: str,
    render_card: Callable[..., str] | None = None,
    root_files: Mapping[str, str] | None = None,
    external: Sequence[ExternalComponents] = (),
    check_references: ReferenceCheck | None = None,
) -> dict[str, Any]:
    """Assemble the planned models into one repository and **replace `out_dir` wholesale** with
    that form.

    The publishing discipline (build in staging → put in place once it passes → on failure delete
    only staging → clean up what an earlier interrupted run left behind) is delegated to the core
    primitive ({@link karume.artifacts.staged_publication}) — this function only owns "what is
    built inside the staging directory and which checks it passes" (shared-entry decision →
    placement → `karume.json` → {@link verify_dist} → `README.md`). The existing `out_dir` is not
    touched by a single byte until the final rename, so a failure midway leaves the previous
    distribution as is (the failure mode of the in-place update days, "old manifest + a tree mixing
    old and new", is structurally gone).

    MUST: the replacement is **wholesale** and keeps nothing of `out_dir`'s original content —
    rebuilding only `A` into an output built from `A` + `B` removes `B` (it is not found as an
    undeclared file after copying everything; it simply does not remain). `README.md` is drawn in
    staging too, so the distribution put in place comes complete with its card (`render_card`
    receives **the verified** manifest).

    `root_files` (file name → text) are legal texts written in UTF-8 **at the repository root**
    ({@link LEGAL_PATHS}) — files the upstream weight licence requires as a condition of
    redistribution, not assets the manifest declares (they belong to no model; they apply to the
    distribution repository itself). Like `karume.json` / `README.md`, they are on the exempt side
    of {@link verify_dist}'s undeclared-file check.

    Passing `external` ({@link ExternalComponents} — one per referenced target, any number)
    declares only the named roles as **references to a pinned revision of another repository**,
    not placed in this repository (ADR 0038 §7). An assembly without it is fully self-contained and
    does not change by a single byte. A role named by two targets fails — one role points at one
    place.

    `check_references` ({@link ReferenceCheck}) is the caller's confirmation that each target's
    revision really holds the referenced bytes; it runs once per target, after the references are
    resolved and before the first byte is written.

    MUST: the checks over all of `plans` are all settled **before the first byte is written** (not
    even staging is created) — a failure midway would mean laying out several GB again. Resolving
    cross-repository references (reading the referenced manifest, hashing the files) and the
    caller's confirmation are placed here for the same reason.
    """
    if not plans:
        raise DistError("There is not a single model to assemble")
    # MUST: `requiredLimits` is derived **from the files** and baked by assembly
    # ({@link bake_required_limits}) — the declaration means "the device limit needed to keep these
    # bytes resident", so there is no source of truth other than the files that go into the
    # distribution. It comes before the expansion because the derivation's entry point is **the
    # representative path** (what `component_parts` / `ir_graph` take) — the representative path
    # cannot be recovered from the expanded part roles.
    plans = [bake_required_limits(plan) for plan in plans]
    # MUST: the expansion comes **before every check** (before writing a single byte) — both the
    # relative paths and the sources of the expanded roles must go through the later checks (path
    # containment, input existence, shared folding).
    partitioned = [expand_weight_parts(plan) for plan in plans]
    plans = [item.plan for item in partitioned]
    names = [plan.name for plan in plans]
    if len(set(names)) != len(names):
        raise DistError(f"Duplicate model names: {names}")
    if default_model not in names:
        raise DistError(
            f"Default model '{default_model}' is not among the models to assemble {names}"
        )
    assert_plan_paths(plans)
    assert_plan_limits(plans)
    assert_plan_sources(plans)
    assert_root_files(root_files or {})
    # Full verification of the input containers comes **after the existence checks and before
    # writing a single byte** (it reads only the headers, so placing it here does not mean
    # rereading several GB).
    declared_parts = assert_weight_components_verified(partitioned)
    # Cross-repository references **may apply to several models**. MUST: fail unless the files
    # of the named roles are byte-identical to the target in every model (the reconciliation of
    # {@link external_refs} per plan + identical references across all plans) — "the same role
    # name pointing at different bytes" is what the ambiguity actually is, so that is checked
    # directly instead of using the number of models as a proxy (a mismatch fails loudly, naming
    # the role and both models).
    #
    # Split-form containers can be cross-repository references too — `container.parts` is an
    # array whose **every element** goes through the usual FileRef check (ADR 0038 §7 / ADR 0109
    # decision 3), so one reference can point at one part. This point is reached after the
    # expansion ({@link expand_weight_parts}) split the representative role into part roles, so
    # {@link external_refs} matches the target's files part by part.
    references: dict[str, dict[str, Any]] = {}
    if external:
        assert_distinct_reference_roles(external)
        assert_out_dir_is_not_referenced(out_dir, external)
        resolved = [
            (item.plan.name, [external_refs(components, item) for components in external])
            for item in partitioned
        ]
        first_model, first_targets = resolved[0]
        # The roles of different targets are disjoint (checked above), so merging cannot hide one.
        references = {role: ref for refs in first_targets for role, ref in refs.items()}
        for model_name, targets in resolved[1:]:
            refs = {role: ref for target in targets for role, ref in target.items()}
            differing = sorted(
                role
                for role in set(references) | set(refs)
                if references.get(role) != refs.get(role)
            )
            if differing:
                raise DistError(
                    f"Cross-repository reference targets disagree between models: {differing}"
                    f" ('{first_model}' and '{model_name}')"
                    " — the same role name pointing at different bytes cannot be declared with"
                    " one reference"
                )
        # Every model resolved to the same references, so the first model's are the whole set.
        if check_references is not None:
            for components, refs in zip(external, first_targets, strict=True):
                check_references(components, list(refs.values()))

    try:
        with staged_publication(out_dir) as staging:
            manifest, host_assets = _materialize_family(
                partitioned, staging, default_model, references, declared_parts
            )
            # Legal texts are placed **before** verification — unless their place on the exempt
            # side is exercised through {@link verify_dist} on every assembly, a run where the
            # exemption broke would be noticed only after it was put in place.
            for name, text in (root_files or {}).items():
                (staging / name).write_text(text, encoding="utf-8")
            verify_dist(staging)
            if render_card is not None:
                card = render_card(manifest, host_assets=host_assets)
                (staging / MODEL_CARD_FILENAME).write_text(card, encoding="utf-8")
    except ArtifactSwapError as error:
        # A swap failure is not mistaken for an assembly failure (the underlying I/O failure
        # stays in the chain).
        raise DistError(str(error)) from error
    return manifest


# ---- ② Reconciling declarations with files ------------------------------------------


def _declared_refs(manifest: Mapping[str, Any]) -> Iterator[tuple[str, Mapping[str, Any]]]:
    """Yield every file the manifest references as `(location, 3-field set)` (duplicates are
    yielded as they are).

    MUST: missing fields and wrong types are translated into `DistError` by {@link _dist_object}
    / {@link _dist_part_ref} — {@link external_refs} passes the referenced distribution's manifest
    here without a shape check, so a broken referenced manifest must not leak a raw `KeyError` /
    `TypeError` (the same reason as the MUST of {@link assert_container_limits}).
    """
    models = _dist_object(manifest.get("models"), "models")
    for model_name, model_value in models.items():
        model = _dist_object(model_value, f"models.{model_name}")
        weights = _dist_object(model.get("weights"), f"models.{model_name}.weights")
        for name, labels_value in weights.items():
            labels = _dist_object(labels_value, f"models.{model_name}.weights.{name}")
            for label, entry_value in labels.items():
                entry_where = f"models.{model_name}.weights.{name}.{label}"
                entry = _dist_object(entry_value, entry_where)
                where = f"{entry_where}.container"
                container = _dist_object(entry.get("container"), where)
                parts = container.get("parts")
                if not isinstance(parts, list):
                    raise DistError(f"{where}.parts is not an array (actual: {parts!r})")
                for index, ref in enumerate(parts):
                    yield f"{where}.parts[{index}]", _dist_part_ref(ref, f"{where}.parts[{index}]")
        assets = _dist_object(model.get("assets"), f"models.{model_name}.assets")
        for name, ref in assets.items():
            where = f"models.{model_name}.assets.{name}"
            yield where, _dist_part_ref(ref, where)


def is_external_ref(ref: Mapping[str, Any]) -> bool:
    """Whether a file reference points at another repository (the form with the `repo` /
    `revision` fields of ADR 0038 §7).

    It is **another form sitting in the same place** as this repository's 3-field set, so the
    discriminator is the presence of the field ({@link ExternalComponents}). Every layer that
    compares with local files excludes it by this test.
    """
    return "repo" in ref


def _assert_manifest_shape(manifest: Mapping[str, Any]) -> None:
    """Structural consistency of `karume/5` (the writing side also checks whether it is a shape
    the hub's parser accepts).

    Only **whether this distribution is closed on its own** is checked here — the targets of
    `defaultModel` / `defaultQuant`, quants' total mapping of weights, the container entry points
    of weights, and the layout (ADR 0041 decision 9). Copying every hub check would only create a
    second source of truth, so only the rules that "break first when assembly breaks" are copied
    (the limits and part rules are in {@link assert_container_limits}).
    """
    if manifest.get("format") != MANIFEST_FORMAT:
        raise DistError(f"format is not '{MANIFEST_FORMAT}': {manifest.get('format')!r}")
    if not manifest.get("generator"):
        raise DistError("generator is missing / empty")
    models = manifest.get("models")
    if not isinstance(models, dict) or not models:
        raise DistError("models is not a non-empty object")
    if manifest.get("defaultModel") not in models:
        raise DistError(
            f"defaultModel '{manifest.get('defaultModel')}' is not in models {sorted(models)}"
        )
    for model_name, model_value in models.items():
        model = _dist_object(model_value, f"models.{model_name}")
        weights = _dist_object(model.get("weights"), f"{model_name}.weights")
        quants = _dist_object(model.get("quants"), f"{model_name}.quants")
        for name, labels_value in weights.items():
            labels = _dist_object(labels_value, f"{model_name}.weights.{name}")
            for label, entry_value in labels.items():
                entry = _dist_object(entry_value, f"{model_name}.weights.{name}.{label}")
                # MUST: a dtype label is inside the storage dtype vocabulary
                # ({@link STORAGE_DTYPE_LABELS}).
                if label not in STORAGE_DTYPE_LABELS:
                    raise DistError(
                        f"{model_name}.weights.{name}: dtype label '{label}' is not in the storage"
                        f" dtype vocabulary {sorted(STORAGE_DTYPE_LABELS)}"
                    )
                # MUST: a dtype entry is exactly one container entry point (ADR 0109 decision 3).
                # A manifest still holding `karume/4`'s `{shards, extras?}` fails here too — a
                # distribution with only the format identifier rewritten and old-form content
                # would pass on the writing side although the hub cannot read it.
                if set(entry) != {"container"}:
                    raise DistError(
                        f"{model_name}.weights.{name}.{label}: fields are not ['container']"
                        f" (actual: {sorted(entry)})"
                    )
                container = entry["container"]
                if not isinstance(container, dict) or set(container) != {"descriptor", "parts"}:
                    found = sorted(container) if isinstance(container, dict) else repr(container)
                    raise DistError(
                        f"{model_name}.weights.{name}.{label}.container: fields are not"
                        f" ['descriptor', 'parts'] (actual: {found})"
                    )
                descriptor = container["descriptor"]
                if not isinstance(descriptor, dict) or set(descriptor) != {"graph", "model"}:
                    found = sorted(descriptor) if isinstance(descriptor, dict) else repr(descriptor)
                    raise DistError(
                        f"{model_name}.weights.{name}.{label}.container.descriptor: fields are not"
                        f" ['graph', 'model'] (actual: {found})"
                    )
                assert_container_limits(f"{model_name}.weights.{name}.{label}.container", container)
        default_quant = model.get("defaultQuant")
        if not isinstance(default_quant, str) or default_quant not in quants:
            raise DistError(
                f"{model_name}.defaultQuant '{default_quant}' is not in quants {sorted(quants)}"
            )
        for quant_name, quant_value in quants.items():
            where = f"{model_name}.quants.{quant_name}"
            quant_weights = _dist_object(
                _dist_object(quant_value, where).get("weights"), f"{where}.weights"
            )
            if set(quant_weights) != set(weights):
                raise DistError(
                    f"{where}.weights is not a total mapping of weights"
                    f" (declared {sorted(quant_weights)} / weights {sorted(weights)})"
                )
            for name, label in quant_weights.items():
                if not isinstance(label, str) or label not in weights[name]:
                    raise DistError(
                        f"{where}.weights: '{name}' has no dtype '{label}'"
                        f" (available: {sorted(weights[name])})"
                    )
    allowed_roots = set(models) | {SHARED_DIRNAME}
    for where, ref in _declared_refs(manifest):
        if is_external_ref(ref):
            # The path of a cross-repository reference is **in the target repository's layout**,
            # so it does not fit this repository's root set (only the shape is checked).
            assert_external_ref(where, ref)
            continue
        root = ref["path"].split("/")[0]
        if root not in allowed_roots:
            raise DistError(
                f"{where}: path '{ref['path']}' does not start with any of {sorted(allowed_roots)}"
                " — the layout is per-model subtrees + shared/ (ADR 0041 decision 9)"
            )


def _declared_sizes(manifest: Mapping[str, Any]) -> dict[str, int]:
    """`{path: size}` of every file **in this repository** the manifest references. Duplicate
    paths must agree on the 3-field set.

    A duplicate reference to the same path is exactly sharing between models (ADR 0041 decision 5)
    and is legal, but if `{size, sha256}` disagree the fetch layer's cache oscillates — the same
    rule as the hub's fails here too.

    Cross-repository references ({@link is_external_ref}) are **excluded** — they are files that
    are correctly absent locally, so neither the existence check nor the undeclared-file check
    applies to them (their path space is the target repository's too, a namespace separate from
    the paths here).
    """
    sizes: dict[str, int] = {}
    seen: dict[str, Mapping[str, Any]] = {}
    for where, ref in _declared_refs(manifest):
        if is_external_ref(ref):
            continue
        previous = seen.get(ref["path"])
        if previous is not None and (
            previous["size"] != ref["size"] or previous["sha256"] != ref["sha256"]
        ):
            raise DistError(
                f"{where}: size / sha256 of duplicate path '{ref['path']}' disagree"
                f" ({previous['size']} / {previous['sha256']} and {ref['size']} / {ref['sha256']})"
            )
        seen[ref["path"]] = ref
        sizes[ref["path"]] = ref["size"]
    return sizes


def verify_dist(out_dir: Path) -> dict[str, int]:
    """Reconcile `karume.json` with the files (structural consistency, existence, matching sizes,
    no undeclared files).

    sha256 was taken from the actual files at assembly time and is not taken again (rehashing
    several GB produces no new fact here). Only "whether the table is closed on its own and covers
    the files" is checked.

    The only exemptions from the undeclared-file check are the two of {@link META_PATHS}
    (`karume.json` and `README.md`) and the two of {@link LEGAL_PATHS} (`LICENSE.md` /
    `NOTICE.md`) — the former describe the distribution itself, the latter are legal texts the
    upstream licence requires as a condition of redistribution, and neither is an asset the
    manifest declares. Everything else fails loudly as before (leftovers of a previous assembly
    or stray `io.*` files are not shown to later stages). Both exemptions are held as relative
    paths, not names, so a same-named file slipped into a subdirectory fails as before.

    Files of cross-repository references ({@link ExternalComponents}) are **correctly absent from
    this repository**, so they are not subject to the existence check ({@link _declared_sizes}
    excludes them). Only their shape is checked, by {@link _assert_manifest_shape} through
    {@link assert_external_ref}.
    """
    manifest = json.loads((out_dir / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    _assert_manifest_shape(manifest)
    declared = _declared_sizes(manifest)
    for rel_path, size in sorted(declared.items()):
        path = out_dir / rel_path
        if not path.is_file():
            raise DistError(f"A file the manifest references is missing: {rel_path}")
        actual = path.stat().st_size
        if actual != size:
            raise DistError(
                f"{rel_path}: size differs from the manifest (declared {size} / actual {actual})"
            )
    exempt = META_PATHS | LEGAL_PATHS
    present = {
        relative
        for path in out_dir.rglob("*")
        if path.is_file() and (relative := str(path.relative_to(out_dir))) not in exempt
    }
    extra = sorted(present - set(declared))
    if extra:
        raise DistError(f"Files the manifest does not declare are mixed in: {', '.join(extra)}")
    return declared


# ---- ②' Cross-repository component references (entries pointing at a pinned revision) ----


@dataclass(frozen=True)
class CrossRepoReference:
    """A recipe's declaration that the named roles of the model being built are byte-identical to
    model `model` of the distribution repository `repo_name`, and are **referenced there instead
    of stored** (the optional `repo` / `revision` fields of ADR 0038 §7 — addendum 2026-10-28).

    It holds only what a recipe knows: which repository, which model and which roles. Where the
    referenced distribution lives locally and which commit it is pinned at belong to the caller
    ({@link main}'s `referenced_dist` hook and `--ref-revision`) — the repository's `models/`
    layout and the Hub are not knowledge of the core (ADR 0065).
    """

    #: Name of the referenced distribution repository **without `HF_OWNER`** — the same spelling
    #: as {@link Pipeline.repo_name} (`--ref-revision NAME=SHA` names it this way too).
    repo_name: str
    #: Model name inside the referenced distribution.
    model: str
    #: Role names (artifact keys) replaced by references. A split-form container's role expands
    #: into one reference per part ({@link external_refs}).
    roles: tuple[str, ...]

    def __post_init__(self) -> None:
        if not REPO_RE.fullmatch(self.repo):
            raise DistError(
                f"Cross-repository reference repository name '{self.repo_name}' does not spell"
                f" a repository ('{self.repo}' is not of the form '<owner>/<name>')"
            )
        assert_model_name(self.model)
        if not self.roles:
            raise DistError(
                f"The cross-repository reference to '{self.repo_name}' names not a single role"
            )
        repeated = sorted(role for role, count in Counter(self.roles).items() if count > 1)
        if repeated:
            raise DistError(
                f"The cross-repository reference to '{self.repo_name}' names roles twice:"
                f" {repeated}"
            )

    @property
    def repo(self) -> str:
        """The referenced HF repository (`<owner>/<name>` — the owner is {@link HF_OWNER}, the
        same rule as {@link resolve_repo})."""
        return f"{HF_OWNER}/{self.repo_name}"


@dataclass(frozen=True)
class ExternalComponents:
    """One **resolved** cross-repository reference: the named roles point at a pinned revision of
    another repository instead of being stored in this repository (the optional `repo` /
    `revision` fields of ADR 0038 §7). {@link main} builds it from a recipe's
    {@link CrossRepoReference}, the caller's local mirror and the revision.

    It has exactly one use: "not uploading the same bytes twice to two repositories" — for
    example, the text encoder of the fine-tune repository is byte-identical to the one in the
    official repository, so a reference saves re-uploading several GB. **Without it the
    distribution is fully self-contained** (the default assembly does not change by a single
    byte).

    MUST: `revision` is **a 40-digit commit sha** ({@link REVISION_RE}). Allowing branch names or
    tags would let the target move later, and the `size` / `sha256` declared here would disagree
    with the files — "pointing at immutable bytes" is the very condition for a reference to hold,
    so spellings that are not a pin are not accepted.

    MUST: `size` / `sha256` are taken from **the actual files of the local referenced dist**
    ({@link external_refs}) — copying the values of the referenced `karume.json` would bring back
    the failure mode of table and files disagreeing, across repositories (voiding the reason the
    manifest was made a derived artifact).

    `dist` / `model` are the referenced distribution (already assembled) and the model name in it.
    The target's paths depend on each repository's layout (per-model subtree / `shared/`), so they
    cannot be derived from the relative paths here — the paths the referenced `karume.json`
    declares are looked up.
    """

    #: Target HF repository (`<owner>/<name>`).
    repo: str
    #: Target commit sha (40 hex digits).
    revision: str
    #: The referenced distribution assembled locally (the source of `size` / `sha256`).
    dist: Path
    #: Model name inside the referenced dist (which model's entry the same relative path is
    #: taken from).
    model: str
    #: Role names replaced by references. A role pointing at a split-form container expands into
    #: one reference per part (each element of the manifest's `container.parts` array)
    #: ({@link external_refs}).
    roles: tuple[str, ...]

    def __post_init__(self) -> None:
        if not REPO_RE.fullmatch(self.repo):
            raise DistError(
                f"Cross-repository reference repo '{self.repo}' is not of the form '<owner>/<name>'"
            )
        if not REVISION_RE.fullmatch(self.revision):
            raise DistError(
                f"Cross-repository reference revision '{self.revision}' is not a 40-digit"
                " commit sha — branch names and tags are not a pin (if the target moves,"
                " declaration and files disagree)"
            )
        assert_model_name(self.model)
        if not self.roles:
            raise DistError("There is not a single role to reference across repositories")


#: The caller's confirmation that a target's revision really holds the bytes a build references
#: (taken by {@link assemble_family}): one call per {@link ExternalComponents} with the file
#: references `{repo, revision, path, size, sha256}` resolved for it, before the first byte is
#: written; it raises {@link DistError} to stop the build. The core holds only the shape —
#: confirming a revision needs the Hub, and the core does no network access (the repository's
#: driver confirms against the HF API — `tools/export-recipes/dist.py`).
ReferenceCheck = Callable[[ExternalComponents, Sequence[Mapping[str, Any]]], None]


def assert_distinct_reference_roles(external: Sequence[ExternalComponents]) -> None:
    """Fail when one role is named by two cross-repository references.

    MUST: a role points at one place — merging two claims would keep whichever came last and
    silently drop the other target's check.
    """
    claimed: dict[str, str] = {}
    for components in external:
        for role in components.roles:
            other = claimed.get(role)
            if other is not None:
                raise DistError(
                    f"Role '{role}' is named by two cross-repository references ({other} and"
                    f" {components.repo}) — one role can point at one place only"
                )
            claimed[role] = components.repo


def assert_out_dir_is_not_referenced(out_dir: Path, external: Sequence[ExternalComponents]) -> None:
    """Fail when the output location is the local distribution a reference is reconciled against.

    MUST: compared as real paths, before a byte is written — the wholesale replacement of
    `out_dir` ({@link assemble_family}) would delete the referenced mirror after reading it, leaving
    a distribution whose borrowed files exist nowhere locally. The check on repository names
    ({@link assert_not_self_referencing}) cannot see it: the card's repository comes from the
    pipeline, not from `--out`.
    """
    target = out_dir.resolve()
    for components in external:
        if components.dist.resolve() == target:
            raise DistError(
                f"The output location {out_dir} is the local distribution of {components.repo},"
                " which this build references — building into it would replace the files it"
                " borrows; choose another --out"
            )


def assert_external_ref(where: str, ref: Mapping[str, Any]) -> None:
    """The shape of one cross-repository reference (set of fields, repo spelling, 40-hex-digit
    revision, path containment)."""
    if set(ref) != set(EXTERNAL_REF_KEYS):
        raise DistError(
            f"{where}: cross-repository reference fields are not {sorted(EXTERNAL_REF_KEYS)}"
            f" (actual: {sorted(ref)})"
        )
    if not REPO_RE.fullmatch(str(ref["repo"])):
        raise DistError(f"{where}: repo '{ref['repo']}' is not of the form '<owner>/<name>'")
    if not REVISION_RE.fullmatch(str(ref["revision"])):
        raise DistError(
            f"{where}: revision '{ref['revision']}' is not a 40-digit commit sha"
            " — a reference holds only if it points at immutable bytes"
        )
    assert_rel_path(str(ref["path"]), where)


def _external_ref(
    components: ExternalComponents,
    declared: set[str],
    role: str,
    artifact: Artifact,
    memo: dict[Path, str],
    *,
    part_seat: bool,
) -> dict[str, Any]:
    """One cross-repository file reference (one element of {@link external_refs}).

    `part_seat` is "whether this entry can hold a list of parts" — true only for dtype entries of
    weights, false for assets (entries that can hold only a single file reference). If the target
    is in split form for a false entry, it fails loudly: silently pointing at part 0 alone would
    produce a distribution whose remaining bytes cannot be fetched from anywhere.
    """
    seats = [
        f"{components.model}/{artifact.rel_path}",
        f"{SHARED_DIRNAME}/{artifact.rel_path}",
    ]
    found = [seat for seat in seats if seat in declared]
    if not found:
        if not part_seat and any(
            len(component_parts(components.dist / seat)) > 1 for seat in seats
        ):
            raise DistError(
                f"A cross-repository reference cannot apply to a split-form container: ['{role}']"
                " (the role is split into several parts, so one reference cannot point at it)"
                " — an assets entry can hold only a single file reference"
            )
        raise DistError(
            f"The file of role '{role}' is not in the referenced dist {components.dist}"
            f" ({MANIFEST_FILENAME} declares neither of {seats})"
            " — what is not in the target cannot be referenced"
        )
    source = components.dist / found[0]
    if not source.is_file():
        raise DistError(f"The file the referenced dist declares is missing: {source}")
    digest = sha256_file(source)
    local = _source_digest(artifact, memo)
    if digest != local:
        raise DistError(
            f"The target of role '{role}' differs from the bytes this assembly builds: {source}"
            f" is {digest}, the source of {artifact.rel_path} is {local}"
            ' — a reference with different content would distribute "another model\'s weights"'
            " as our own"
        )
    return {
        "repo": components.repo,
        "revision": components.revision,
        "path": found[0],
        "size": source.stat().st_size,
        "sha256": digest,
    }


def external_refs(
    components: ExternalComponents, partitioned: PartitionedPlan
) -> dict[str, dict[str, Any]]:
    """Role name → cross-repository file reference `{repo, revision, path, size, sha256}`.

    The target path is looked up from **the paths the referenced dist's `karume.json` declares**
    (per-model subtree or `shared/` is a fact decided by that assembly, not derivable from here).
    A role not in the declaration fails loudly — what is not in the target cannot be referenced.

    **A split-form container gives one reference per part role** (each element of the manifest's
    `container.parts` array becomes a reference — ADR 0038 §7 / ADR 0109 decision 3). `repo` /
    `revision` MUST be the same for every element (cross-repository references are per
    container), while `path` / `size` / `sha256` differ per part. The order is exactly the index
    order {@link expand_weight_parts} resolved **from the files**, so the convention first = part 0
    holds for references too. Part file names spell out the number and the total
    (`-NNNNN-of-NNNNN`), so if the target's part count differs from ours it always fails as "not in
    the referenced dist" (a count mismatch is never resolved silently).

    MUST: the reconciliation of declarations with files is not broken across repositories either.
    `size` / `sha256` are taken from the local actual files, and it is further confirmed that they
    **match the bytes this assembly would have built** — a mismatching reference would
    "distribute another model's weights as our own", staying silent with shape and manifest both
    correct. For split-form roles this reconciliation applies to **every element of the part
    list**.

    NOTE: references of the referenced dist (multi-level) are not followed. {@link _declared_sizes}
    excludes cross-repository references, so an entry where the referenced dist points at yet
    another repository fails here as "not in the declaration".
    """
    manifest_path = components.dist / MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise DistError(
            f"The referenced dist of a cross-repository reference has no {MANIFEST_FILENAME}:"
            f" {manifest_path}"
        )
    declared = set(_declared_sizes(json.loads(manifest_path.read_text(encoding="utf-8"))))
    plan = partitioned.plan
    memo: dict[Path, str] = {}
    refs: dict[str, dict[str, Any]] = {}
    for role in components.roles:
        # weights roles are in the assignment table (in split form they are split into part
        # roles and the representative role is gone from artifacts). A role not in it is an
        # assets entry, so a single file reference is taken for the representative role as is.
        for member in partitioned.parts.get(role, (role,)):
            artifact = plan.artifacts.get(member)
            if artifact is None:
                raise DistError(
                    f"A cross-repository reference points at an unknown role: '{role}'"
                    f" (this model's roles: {sorted(plan.artifacts)})"
                )
            refs[member] = _external_ref(
                components,
                declared,
                member,
                artifact,
                memo,
                part_seat=role in partitioned.parts,
            )
    return refs


#: The hook that names the local distribution of a referenced repository (repository name without
#: `HF_OWNER` → its assembled directory; taken by {@link main}). Only the caller knows where the
#: mirrors live (the repository's `models/<name>/`), so the core holds only the shape.
ReferencedDist = Callable[[str], Path]


def parse_reference_revisions(values: Sequence[str] | None) -> dict[str, str]:
    """`--ref-revision NAME=SHA` (repeatable) → {repository name: commit sha}.

    The sha's spelling is checked where it is used ({@link ExternalComponents} — 40 lowercase hex
    digits). A name given twice fails rather than letting the last one win silently.
    """
    revisions: dict[str, str] = {}
    for value in values or ():
        name, separator, revision = value.partition("=")
        if not separator or not name or not revision:
            raise DistError(
                f"--ref-revision '{value}' is not of the form NAME=SHA (NAME = the referenced"
                " repository's name without the owner, as the pipeline declares it)"
            )
        if name in revisions:
            raise DistError(f"--ref-revision names '{name}' twice")
        revisions[name] = revision
    return revisions


def external_components(
    references: Sequence[CrossRepoReference],
    revisions: Mapping[str, str],
    referenced_dist: ReferencedDist | None,
) -> tuple[ExternalComponents, ...]:
    """Resolve a build's declared references with a revision per repository and the caller's
    local mirrors.

    MUST: every declared repository gets a revision, and every revision names a declared
    repository. A missing one fails instead of quietly assembling a self-contained distribution
    from a build meant to reference (noticed only after uploading several GB again); an extra one
    fails because it means the build is not the one the caller had in mind.
    """
    declared = {reference.repo_name for reference in references}
    unknown = sorted(set(revisions) - declared)
    if unknown:
        raise DistError(
            f"--ref-revision names repositories this build does not reference: {unknown}"
            f" (declared: {sorted(declared) if declared else 'none'})"
        )
    missing = sorted(declared - set(revisions))
    if missing:
        raise DistError(
            f"This build references {missing} across repositories but has no pinned revision"
            " for them — pass --ref-revision NAME=SHA (the referenced repository's commit)"
        )
    if not references:
        return ()
    if referenced_dist is None:
        raise DistError(
            "This build references other repositories, but the caller passed no location for"
            " their local distributions (the repository's dist driver passes it)"
        )
    return tuple(
        ExternalComponents(
            repo=reference.repo,
            revision=revisions[reference.repo_name],
            dist=referenced_dist(reference.repo_name),
            model=reference.model,
            roles=reference.roles,
        )
        for reference in references
    )


# ---- ③ Per-pipeline dispatch and the CLI ------------------------------------------


#: The model card renderer (one body from the manifest, the repository ID and the count of
#: "assets read on the host").
#:
#: `host_assets` (part 0 identity {@link karume.modelcard.file_ref_key} → the sum of the logical
#: lengths of the assets the container declares) is **a fact of the container** that cannot be
#: derived from the manifest, so assembly takes it and passes it in (with ADR 0109 decision 4 the
#: PLE / `rope_base` moved into the container and the separate file entries disappeared).
CardRenderer = Callable[[Mapping[str, Any], str, Mapping[FileRefKey, int]], str]


def self_contained(_model: str) -> tuple[CrossRepoReference, ...]:
    """The default of {@link Pipeline.references}: the model's distribution references nothing."""
    return ()


@dataclass(frozen=True)
class Pipeline:
    """One row of the dispatch table, holding only what differs per pipeline.

    The shared part (placement, shared folding, sha256, reconciling declarations with files)
    belongs to the generic functions above, so only these are listed here: "the default model
    name / the distribution repository name a model belongs to / the procedure that builds a plan
    from a series and a model name / the model card renderer", plus the legal texts and the
    cross-repository references a distribution declares.
    """

    default_model: str
    #: Model name → the name of the distribution repository holding that model (without
    #: `HF_OWNER`). The source of truth for the repo spelled in the card's Usage example
    #: ({@link resolve_repo}); the caller's default output location is spelled from it too.
    #: It holds as the name of one repository only when every model built returns the same name.
    repo_name: Callable[[str], str]
    plan: Callable[[Path, str], ModelPlan]
    #: The model card's **attribution profiles** (name → renderer). Even when the template is
    #: pipeline-specific, attribution (provenance, licence, citation) is a separate fact per voice
    #: family, so the mapping is not one to one.
    card_profiles: Mapping[str, CardRenderer]
    #: Legal texts placed **at the root** of the distribution repository (file name → content). The
    #: only accepted names are {@link LEGAL_PATHS}. Passed only by pipelines whose upstream weight
    #: licence requires a copy of the licence text or an Attribution Notice as a condition of
    #: redistribution (empty by default = nothing is placed).
    root_files: Mapping[str, str] = field(default_factory=dict)
    #: Model name → the cross-repository references that model's distribution declares
    #: ({@link CrossRepoReference}; empty = self-contained, the default). The roles are a recipe
    #: fact (which components are byte-identical to another repository's), so they live here and
    #: not in a release command; the revision is supplied per build ({@link main}). A build of a
    #: model that declares references fails without a revision for each of them — it never falls
    #: back to a self-contained distribution silently.
    references: Callable[[str], tuple[CrossRepoReference, ...]] = self_contained


def selected_models(pipeline: Pipeline, models: Sequence[str] | None) -> list[str]:
    """The models a build assembles: the `--model` values in order, else the pipeline's default
    (the first one is `defaultModel`). One function, so the caller and {@link main} read the same
    list."""
    return list(models) if models else [pipeline.default_model]


def declared_references(
    pipeline: Pipeline, models: Sequence[str]
) -> tuple[CrossRepoReference, ...]:
    """The cross-repository references a build declares ({@link Pipeline.references} of its
    models).

    MUST: every model of one build declares the same references — assembly applies each reference
    to every model ({@link assemble_family}), so a family build whose models declare different
    ones would either borrow for a model that declared itself self-contained or store what another
    model declared borrowed. It fails naming both models instead.
    """
    first = pipeline.references(models[0])
    for model in models[1:]:
        if pipeline.references(model) != first:
            raise DistError(
                f"Models '{models[0]}' and '{model}' declare different cross-repository"
                " references — one build applies one set of references to every model, so build"
                " them into separate repositories"
            )
    return first


def assert_not_self_referencing(references: Sequence[CrossRepoReference], repo: str) -> None:
    """Fail when a declared reference points at the repository being built.

    MUST: checked at build time — the hub rejects such a manifest only when it is loaded from the
    HF repository itself (a local mirror and the distribution gate parse it without the owning
    repository), so it would otherwise first fail after upload.
    """
    own = sorted(reference.repo_name for reference in references if reference.repo == repo)
    if own:
        raise DistError(
            f"A cross-repository reference points at the repository being built ({repo}):"
            f" {own} — a repository cannot borrow its own files"
        )


def resolve_card_renderer(pipeline: Pipeline, profile: str | None) -> CardRenderer:
    """Resolve `--card-profile` to a renderer.

    MUST: with two or more choices, omitting it **fails listing the choices** — silently picking a
    default would draw the previous family's attribution onto a new family's repository as is.
    Only the attribution would be wrong while tables and usage stay right, so nobody could notice
    until after distribution. A pipeline with only one choice has nothing to choose, so omission
    passes (the moment a second one appears, this rule starts requiring it explicitly on its own).
    """
    choices = ", ".join(sorted(pipeline.card_profiles))
    if profile is None:
        if len(pipeline.card_profiles) != 1:
            raise DistError(
                f"--card-profile is required (choices: {choices}) — the model card's attribution"
                " is a different fact per family, so no default is picked silently"
            )
        return next(iter(pipeline.card_profiles.values()))
    renderer = pipeline.card_profiles.get(profile)
    if renderer is None:
        raise DistError(f"Attribution profile '{profile}' does not exist (choices: {choices})")
    return renderer


def resolve_repo(pipeline: Pipeline, models: Sequence[str], repo: str | None) -> str:
    """Decide the HF repository ID (`<owner>/<name>`) spelled in the model card's Usage example.

    The source of truth is the pipeline's declaration ({@link Pipeline.repo_name}); `--repo`
    overrides it explicitly.

    MUST: not derived from the output directory name — the location is not a fact of the
    repository, so the card of a distribution built in a directory with another name (staging)
    would be published still pointing at a repository that does not exist.
    MUST: when the declarations of the models built do not agree on one name (a pipeline whose
    repository name includes the model name, bundling several models into one repository), it is
    not guessed; it fails asking for `--repo` — the name of a bundled repository is a naming
    decision and does not follow from the list of model names.
    """
    if repo is not None:
        if not REPO_RE.fullmatch(repo):
            raise DistError(f"--repo '{repo}' is not of the form '<owner>/<name>'")
        return repo
    names = sorted({pipeline.repo_name(model) for model in models})
    if len(names) != 1:
        raise DistError(
            f"The repository names models {' / '.join(models)} declare do not agree"
            f" ({' / '.join(names)}) — name the bundled repository explicitly with"
            " --repo OWNER/NAME"
        )
    return f"{HF_OWNER}/{names[0]}"


#: Pipelines that can be built with the core wheel alone (not all of them — the module doc's
#: MUST).
#:
#: NOTE: **empty**. Every per-family recipe moved out of the wheel
#: (`tools/export-recipes/<family>/`) (ADR 0065 stages 3 and 4, complete), so the only source of
#: truth for the accepted set is the dictionary of the repository's dist driver
#: (`tools/export-recipes/dist.py`). The constant stays to show in the type that core is "the
#: side that receives the table", not "a side without a table" — calling {@link main} with the
#: empty mapping fails whatever `--pipeline` is given (core alone building nothing is the correct
#: behaviour).
PIPELINES: Mapping[str, Pipeline] = {}


#: The hook that builds the output location when `--out` is omitted (taken by {@link main}). Only
#: the caller, which knows the repo topology, can spell the repository's `models/<repo name>/`, so
#: core holds only the shape.
DefaultOutDir = Callable[[Pipeline, Sequence[str]], Path]


def build_parser(
    pipelines: Mapping[str, Pipeline] = PIPELINES,
    default_pipeline: str | None = None,
    *,
    default_series: Path | None = None,
    has_default_out_dir: bool = False,
) -> argparse.ArgumentParser:
    """Take the accepted set of `--pipeline` **as an argument** (the driver passes the table with
    the recipes added).

    A call without `default_pipeline` (= core alone) makes `--pipeline` required ({@link main}
    fails listing the choices). The defaults of `--series` / `--out` are treated the same way:
    when no location is passed, the help says so and {@link main} fails.
    """
    parser = argparse.ArgumentParser(
        description="Assemble a distribution directory (a form that can be uploaded to HF)"
    )
    default_note = "required" if default_pipeline is None else f"default: {default_pipeline}"
    parser.add_argument(
        "--pipeline",
        choices=sorted(pipelines),
        default=default_pipeline,
        help=f"pipeline to assemble ({default_note})",
    )
    parser.add_argument(
        "--model",
        action="append",
        dest="models",
        metavar="NAME",
        help="model name to assemble (repeat to build them into one repository = family"
        " assembly; the first one is defaultModel; default: "
        + " / ".join(f"{name}={spec.default_model}" for name, spec in pipelines.items())
        + ")",
    )
    parser.add_argument(
        "--card-profile",
        dest="card_profile",
        metavar="NAME",
        default=None,
        help="attribution profile of the model card (provenance, licence, citation); required"
        " for pipelines with two or more choices: "
        + " / ".join(
            f"{name}={'|'.join(sorted(spec.card_profiles))}" for name, spec in pipelines.items()
        ),
    )
    parser.add_argument(
        "--series",
        type=Path,
        default=default_series,
        help="parent of the series directories ("
        + (f"default: {default_series}" if default_series is not None else "required")
        + ")",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output location = one directory per HF repository ("
        + (
            "omitted: the caller's default; required when building several models"
            if has_default_out_dir
            else "required"
        )
        + ")",
    )
    parser.add_argument(
        "--repo",
        metavar="OWNER/NAME",
        default=None,
        help="HF repository ID spelled in the model card's Usage example (default: the repository"
        " name the pipeline declares; required when the declarations of the models built do not"
        " agree). Independent of the --out directory name: the card points at this name even"
        " when built into a directory with another name. When overriding, give --out explicitly"
        " too",
    )
    # Cross-repository references (ADR 0038 §7, addendum 2026-10-28): which roles are borrowed
    # from which repository is the pipeline's declaration ({@link Pipeline.references}); the
    # build supplies only the commit each referenced repository is pinned at.
    parser.add_argument(
        "--ref-revision",
        action="append",
        dest="ref_revisions",
        metavar="NAME=SHA",
        default=None,
        help="commit sha (40 hex digits — branch names and tags are not a pin) at which the"
        " referenced repository NAME (its name without the owner, as the pipeline declares it)"
        " is pinned; repeat once per referenced repository. Required for every repository the"
        " models built reference",
    )
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    pipelines: Mapping[str, Pipeline] = PIPELINES,
    default_pipeline: str | None = None,
    default_out_dir: DefaultOutDir | None = None,
    default_series: Path | None = None,
    referenced_dist: ReferencedDist | None = None,
    check_references: ReferenceCheck | None = None,
) -> None:
    """Assemble one distribution from the parsed command line.

    `referenced_dist` names the local distribution of each repository the pipeline's models
    reference ({@link Pipeline.references}); `check_references` confirms that each pinned
    revision holds the referenced bytes ({@link assemble_family}). Both come from the caller —
    the repository's driver knows where the mirrors live and reads the Hub; the core does neither.
    """
    args = build_parser(
        pipelines,
        default_pipeline,
        default_series=default_series,
        has_default_out_dir=default_out_dir is not None,
    ).parse_args(argv)
    if args.pipeline is None:
        # MUST: no default is baked into core (the module doc's MUST) — the table changes with
        # the caller.
        raise DistError(
            f"--pipeline is required (choices: {', '.join(sorted(pipelines))}) — "
            "the table including the repository-only recipes is in tools/export-recipes/dist.py"
        )
    if args.series is None:
        # MUST: no repo topology is baked into core (the module doc's MUST).
        raise DistError(
            "--series is required (parent of the series directories) — the caller passed no default"
        )
    pipeline = pipelines[args.pipeline]
    models = selected_models(pipeline, args.models)
    if args.out is not None:
        out_dir = args.out
    elif args.repo is not None:
        # The default output location is spelled from the pipeline's declaration, so overriding
        # only the name would build "a card pointing at another repository, in a directory named
        # after one repository".
        raise DistError(
            "When overriding with --repo, give --out explicitly too (the default output location"
            " is the declared name)"
        )
    elif default_out_dir is not None:
        out_dir = default_out_dir(pipeline, models)
    else:
        raise DistError(
            "--out is required (output location of the distribution) — the caller passed no default"
        )
    # The attribution profile and the card's repository ID are resolved **before assembly** — a
    # wrong / missing option must not fail at the last file after laying out several GB. The
    # cross-repository references (declaration, own repository, revisions) are settled here for
    # the same reason.
    render_card = resolve_card_renderer(pipeline, args.card_profile)
    repo = resolve_repo(pipeline, models, args.repo)
    references = declared_references(pipeline, models)
    assert_not_self_referencing(references, repo)
    external = external_components(
        references, parse_reference_revisions(args.ref_revisions), referenced_dist
    )
    # MUST: the plans of every model (= checks and reads) are settled **before** placement — even
    # when the second model fails, a distribution holding only the first model is not shown to
    # later stages.
    plans = [pipeline.plan(args.series, model) for model in models]
    # The model card is drawn inside assembly (staging), from **the verified manifest** (an order
    # in which a description never appears while table and files disagree; it is put in place in
    # one go, card included).
    manifest = assemble_family(
        plans,
        out_dir,
        models[0],
        render_card=partial(render_card, repo=repo),
        root_files=pipeline.root_files,
        external=external,
        check_references=check_references,
    )
    verified = verify_dist(out_dir)
    for rel_path, size in sorted(verified.items()):
        print(f"{size:>12}  {rel_path}")
    for rel_path in sorted(META_PATHS | LEGAL_PATHS):
        meta = out_dir / rel_path
        if meta.is_file():
            print(f"{meta.stat().st_size:>12}  {rel_path}")
    # Cross-repository references have no local files, so they do not appear in `verify_dist`'s
    # listing — so that what was left to other repositories can be read from the assembly output,
    # one line per target is printed (several dtype entries point at the same entry, so they are
    # made unique by the same identity as the card — the pair of source and path).
    borrowed = {
        file_ref_key(ref): ref["size"]
        for _, ref in _declared_refs(manifest)
        if is_external_ref(ref)
    }
    for ref_repo, revision, rel_path in sorted(borrowed):
        size = borrowed[(ref_repo, revision, rel_path)]
        print(f"{size:>12}  {ref_repo}@{revision[:12]} {rel_path}")
    listing = ", ".join(
        f"{name}({model['defaultQuant']})" for name, model in manifest["models"].items()
    )
    print(
        f"[dist] {out_dir} — {manifest['generator']} /"
        f" models {listing} / default {manifest['defaultModel']}"
    )


if __name__ == "__main__":
    main()
