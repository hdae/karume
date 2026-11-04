"""Moves the old shard lists of the series outputs (`outputs/series/`) into containers (`krm`) — the
recipe-side driver.

    uv run python -m migrate_series                          # every referenced series
    uv run python -m migrate_series --only deberta --dry-run
    uv run python -m migrate_series --series-dir /path/to/series

The series mode is **not added** to core's `karume migrate` because only a family's
`distribution.py` knows "which directory is which component name" (ADR 0065 — core does not know
the families). The series layout differs per family, and **a directory name is not a component
name**:

- one component directly under the series (siglip2 / birefnet / depth-anything / vowel-detector /
  gemma4 / gemma4-qat / minicpm5 / embeddinggemma) — the directory name is the series name.
- components side by side in subdirectories (anima / sbv2 / irodori / dacvae / deberta) — some
  spellings do not match (`caption-proj` → `caption_proj`, `decoder` → `codec_decoder`, deberta's
  `full-24layer` → the `text_encoder` role SBV2 consumes).

**The old inputs are only read**, and the `krm` goes into the same directory (old shards and
sidecars are not deleted — backups are made separately per directory). Left untouched are the
goldens (`io.*` / `io-i8a8.*` / `greedy.*` / `drafter-golden.*` / `case.*` / `t-embed.*` /
`pipeline.*` / `trim.*`), `ple.probe.safetensors`, `*.json` / `*.txt` / `*.md` and the asset
directories (`text/` / `tokenizer/` / `pipeline/` / `host/`).

**There is one exception**: a QAT series' `reference.json` is rewritten
({@link rewrite_qat_reference}) — the PLE count the record states is the old sidecar's shard count,
a number that exists nowhere once folded. The migration rebuilds what the record points to, so the
record follows.

Only 2 kinds of sidecar (bytes that are not weights but ship in the same container) exist:

- `rope_base.safetensors` (anima's `-dyn` series) → the asset `rope_base`
  (the target of `karume.migrate.EXTRA_ASSETS`).
- `ple.json` + `ple-NNNNN-of-NNNNN.safetensors` (the product series of gemma4 / gemma4-qat) → the
  index `ple_index` + `ple.values.<k>` / `ple.scales.<k>` (`karume.migrate.ple_sidecar_assets` —
  the same single path as the whole-repository mode).

MUST: stop as soon as one component fails (do not migrate the rest and report at the end). A failed
series **is left partly placed**, so delete it by hand and run again (deleting is not the
migration CLI's job — the migration requires "the output directory is built from empty", and the
next run stops with `MigrateError: Artifacts of a previous run remain in the output directory`).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from _shared.paths import INPUTS_ROOT, SERIES_ROOT
from anima import export as anima_export
from anima.distribution import ANIMA_WEIGHTS, ROPE_BASE_ASSET
from birefnet import export as birefnet_export
from birefnet.distribution import BIREFNET_ROLE
from deberta import export as deberta_export
from depth_anything import export as depth_anything_export
from depth_anything.distribution import DEPTH_ANYTHING_ROLE
from embeddinggemma import export as embeddinggemma_export
from gemma4 import export as gemma4_export
from gemma4.distribution import GEMMA4_DRAFTER_ROLE, GEMMA4_DRAFTER_SUFFIX, GEMMA4_ROLE
from gemma4_qat import export as gemma4_qat_export
from gemma4_qat.config import REFERENCE_SCHEMA
from gemma4_qat.distribution import REFERENCE_BLOCKS_FIELD, REFERENCE_SHARDS_FIELD
from irodori import export as irodori_export
from irodori.dacvae import export as dacvae_export
from irodori.distribution import IRODORI_CODEC_ROLES, IRODORI_SERIES_ROLES
from karume.container import BLOCK_MAX_BYTES, AssetInput, Provenance, sequence_siblings
from karume.migrate import (
    EXTRA_ASSETS,
    MigrateError,
    migrate_component,
    ple_sidecar_assets,
    read_ple_index,
)
from karume.ple import PLE_INDEX_ASSET
from minicpm5 import export as minicpm5_export
from sbv2 import export as sbv2_export
from sbv2.distribution import SBV2_FRONT_ROLE, SBV2_SERIES_PREFIX, SBV2_VOICE_ROLE
from siglip2 import export as siglip2_export
from siglip2.distribution import SIGLIP2_ROLE
from vowel_detector import export as vowel_detector_export
from vowel_detector.distribution import VOWEL_DETECTOR_GRAPH_ROLE

#: The representative name of an old shard list (`model` in every family — the stem of the
#: writer's `MODEL_FILE`).
SHARD_STEM = "model"

#: The extension of the old shards and that of the container placed.
SHARD_SUFFIX = ".safetensors"
MODEL_SUFFIX = ".krm"

#: The file names of the sidecars folded into container assets (1:1 with core's targets).
ROPE_BASE_FILE = f"{ROPE_BASE_ASSET}{SHARD_SUFFIX}"
PLE_INDEX_FILE = "ple.json"

#: Prefixes of the `.safetensors` files that may sit in any directory of a series (goldens and
#: references for acceptance). **A `.safetensors` not in this table fails loudly** — the check
#: that keeps the day a new sidecar appears from silently producing "a container that lost an
#: asset while we thought it was migrated" (the migration runs once, so no later chance to notice
#: the loss comes).
#:
#: The last 4 are goldens **outside the component directories**, all read by TS-side tests:
#: `case.*` (irodori's pipeline / dacvae's host references), `t-embed.*`
#: (`irodori.pipeline_ref.T_EMBED_FILE`), `pipeline.*` (per-stage references of anima / irodori),
#: `trim.*` (`irodori.dacvae.host.TRIM_FILE`).
GOLDEN_PREFIXES: tuple[str, ...] = (
    "io.",
    "io-i8a8.",
    "greedy.",
    "drafter-golden.",
    "ple.probe",
    "case.",
    "t-embed.",
    "pipeline.",
    "trim.",
)

#: The QAT family's name (one constant so that the {@link FAMILIES} entry and the lookup in
#: {@link rewrite_qat_reference} do not split the spelling).
QAT_FAMILY = "gemma4-qat"

#: The acceptance record a QAT series holds (the only file the migration rewrites, raising its
#: schema).
QAT_REFERENCE_FILE = "reference.json"

#: The PLE count stated by a record of the old sidecar generation → the block count of the folded
#: assets. The spelling's source of truth is **the reader** (the matching in
#: `gemma4_qat.distribution`) — holding a copy would, the day only one side moves, leave "a field
#: rewritten that the distribution side does not look at".
QAT_SHARDS_FIELD = REFERENCE_SHARDS_FIELD
QAT_BLOCKS_FIELD = REFERENCE_BLOCKS_FIELD

#: Suffixes of series that are not converted (records of experiments and of rejections). They have
#: **no reader**, so they are not migrated — they enter neither the distribution nor the TS-side
#: acceptance, and their profiles were removed from `tools/llm-baseline/data.py` on 2026-09-23
#: (llm-speed became distribution-form only). The measurements themselves remain as prose in
#: `docs/research/`.
UNREFERENCED_SUFFIXES: tuple[str, ...] = ("-probe", "-rejected")


class SeriesMigrationError(ValueError):
    """A premise of the series migration is broken (a series that cannot be classified, a component
    not in the table, a sidecar with nowhere to fold into).

    The repository's convention is a dedicated `Error` subclass (`DistError` /
    `Sbv2FamilyError`) — with a plain `ValueError`, the caller's `except` cannot tell this check
    from a generic "bad argument".
    """


@dataclass(frozen=True)
class Component:
    """One component to migrate (one old shard list + the sidecars folded into the same
    container)."""

    #: The directory holding the old shard list (the `krm` is placed here too).
    directory: Path
    #: The container's graph name = the component name of the distributed form = the key of
    #: `karume.json`'s weights (container-v1 §2.1).
    graph_name: str

    @property
    def representative(self) -> Path:
        """The representative path of the old shard list (`karume.legacy.resolve_shards` resolves it
        to the numbered files)."""
        return self.directory / f"{SHARD_STEM}{SHARD_SUFFIX}"

    @property
    def container(self) -> Path:
        """The representative path of the placed `krm` (the actual files are a numbered part
        list)."""
        return self.directory / f"{SHARD_STEM}{MODEL_SUFFIX}"


@dataclass(frozen=True)
class Family:
    """The series layout of one family (series name matching, component names, provenance).

    An empty `roles` means **one component directly under the series**, whose component name
    {@link flat_role} derives from the series name. Otherwise it is a table of **subdirectory name →
    component name**, and a directory not in the table fails loudly.
    """

    name: str
    #: Series name → whether it belongs to this family (the spellings' source of truth is each
    #: family's distribution / export constants).
    matches: Callable[[str], bool]
    #: Series name → provenance (the `provenance` written into the container; the same value the
    #: recipe's export passes).
    provenance: Callable[[str], Provenance]
    #: Subdirectory name → component name.
    roles: Mapping[str, str] = field(default_factory=dict)
    #: "Series name → component name" for a family with one component directly under the series.
    flat_role: Callable[[str], str] | None = None


def input_checkpoint(root: Path, series: str) -> Path:
    """Series name → where the real weights that produced the series live (the reverse lookup of the
    default output path of the scripts with one component directly under the series).

    The siglip2 / birefnet / depth-anything / minicpm5 scripts build the default output path from
    the directory name of `--model-dir` (lowercased, `_` → `-`, possibly followed by a suffix such
    as `-<resolution>`). The container's provenance (license and revision) is derived from that
    checkpoint itself (each script's `upstream_provenance`), so the migration looks up the same
    place.

    MUST: fail loudly unless it is unique — never produce a container that claims another
    checkpoint's provenance.
    """
    candidates = (
        sorted(entry for entry in root.iterdir() if entry.is_dir()) if root.is_dir() else []
    )
    found = [
        entry
        for entry in candidates
        if series == (stem := entry.name.lower().replace("_", "-")) or series.startswith(f"{stem}-")
    ]
    if len(found) != 1:
        raise SeriesMigrationError(
            f"the checkpoint that series '{series}' was baked from is not unique under {root}"
            f" (candidates: {[entry.name for entry in found]}) — cannot name the source"
        )
    return found[0]


def sbv2_model(series: str) -> str:
    """sbv2 series name → speaker name (`sbv2-F1-i8` → `F1`).

    The provenance (license and upstream version) differs per voice family and can only be looked
    up from the speaker name, so it is stripped here (the reverse of
    `sbv2.export.default_out_root`).
    """
    stem = series.removeprefix(f"{SBV2_SERIES_PREFIX}-")
    for dtype in ("f16", "i8", "i4"):
        stem = stem.removesuffix(f"-{dtype}")
    return stem


#: deberta's variant directories → component name (all three produce SBV2's `text_encoder` role).
DEBERTA_ROLES: Mapping[str, str] = {
    variant.name: deberta_export.GRAPH_NAME for variant in deberta_export.VARIANTS.values()
}

#: sbv2's target directories → component name. The 2 roles in the distribution are constants of
#: `distribution.py`; the 3 golden-only ones (`dp` / `flow` / `dec`) have no component name, so
#: they use the target name itself.
SBV2_ROLES: Mapping[str, str] = {
    sbv2_export.TARGET_FRONT: SBV2_FRONT_ROLE,
    sbv2_export.TARGET_VOICE: SBV2_VOICE_ROLE,
    sbv2_export.TARGET_DP: sbv2_export.TARGET_DP,
    sbv2_export.TARGET_FLOW: sbv2_export.TARGET_FLOW,
    sbv2_export.TARGET_DEC: sbv2_export.TARGET_DEC,
}

#: The family table. The matching order matters only for gemma4-qat / gemma4 (the former shares
#: the latter's prefix), so **qat comes first**.
FAMILIES: tuple[Family, ...] = (
    Family(
        "anima",
        # `anima-f16` (the shared text path + VAE), `<model>-f16` (its own text_conditioner),
        # `<model>-<storage>-dyn` (transformer). The tokenizer / pipeline series have no old shard
        # list.
        lambda name: name.startswith("anima-"),
        lambda _: anima_export.PROVENANCE,
        roles={role: role for role in ANIMA_WEIGHTS},
    ),
    Family(
        "birefnet",
        # The series name is `birefnet_series_name(checkpoint, model, dtype)` = the upstream
        # repository name lowercased + resolution (plus a storage dtype suffix for non-f32).
        lambda name: name.startswith(("birefnet-", "lucida-")),
        lambda name: birefnet_export.upstream_provenance(
            input_checkpoint(birefnet_export.MODELS_ROOT, name)
        ),
        flat_role=lambda _: BIREFNET_ROLE,
    ),
    Family(
        "depth-anything",
        lambda name: name.startswith("depth-anything-"),
        lambda name: depth_anything_export.upstream_provenance(
            input_checkpoint(depth_anything_export.MODELS_ROOT, name)
        ),
        flat_role=lambda _: DEPTH_ANYTHING_ROLE,
    ),
    Family(
        "siglip2",
        lambda name: name.startswith("siglip2-"),
        lambda name: siglip2_export.upstream_provenance(
            input_checkpoint(siglip2_export.MODELS_ROOT, name)
        ),
        flat_role=lambda _: SIGLIP2_ROLE,
    ),
    Family(
        "vowel-detector",
        lambda name: name.startswith("vowel-detector-"),
        lambda _: vowel_detector_export.PROVENANCE,
        flat_role=lambda _: VOWEL_DETECTOR_GRAPH_ROLE,
    ),
    Family(
        "deberta",
        # `deberta{,-i8,-i4}` (the spellings of `deberta.export.DEFAULT_OUT_ROOTS`).
        lambda name: name == "deberta" or name.startswith("deberta-"),
        lambda _: deberta_export.PROVENANCE,
        roles=DEBERTA_ROLES,
    ),
    Family(
        "sbv2",
        lambda name: name.startswith(f"{SBV2_SERIES_PREFIX}-"),
        lambda name: sbv2_export.sbv2_provenance(
            INPUTS_ROOT / SBV2_SERIES_PREFIX / sbv2_model(name)
        ),
        roles=SBV2_ROLES,
    ),
    Family(
        "irodori",
        lambda name: name.startswith("irodori-"),
        lambda _: irodori_export.PROVENANCE,
        roles=IRODORI_SERIES_ROLES,
    ),
    Family(
        "dacvae",
        # The codec is a separate repository with separate weights (`IRODORI_CODEC_NAME` + a dtype
        # suffix).
        lambda name: name.startswith("dacvae-"),
        lambda _: dacvae_export.PROVENANCE,
        roles=IRODORI_CODEC_ROLES,
    ),
    Family(
        QAT_FAMILY,
        lambda name: name.startswith(f"{QAT_FAMILY}-"),
        lambda _: gemma4_qat_export.PROVENANCE,
        flat_role=lambda _: GEMMA4_ROLE,
    ),
    Family(
        "gemma4",
        lambda name: name.startswith("gemma4-"),
        lambda _: gemma4_export.PROVENANCE,
        # Only the MTP drafter has another role (series suffix = the default output path of
        # `gemma4.export_drafter`).
        flat_role=lambda name: (
            GEMMA4_DRAFTER_ROLE if name.endswith(f"-{GEMMA4_DRAFTER_SUFFIX}") else GEMMA4_ROLE
        ),
    ),
    Family(
        "minicpm5",
        lambda name: name.startswith("minicpm5-"),
        lambda name: minicpm5_export.upstream_provenance(
            input_checkpoint(minicpm5_export.DEFAULT_MODEL_DIR.parent, name)
        ),
        flat_role=lambda _: minicpm5_export.GRAPH_NAME,
    ),
    Family(
        "embeddinggemma",
        lambda name: name.startswith("embeddinggemma-"),
        # The series name does not name the checkpoint (the script's default output path is a fixed
        # spelling that does not follow `--model-dir`), so derive it from the default checkpoint
        # that produces that default output.
        lambda _: embeddinggemma_export.upstream_provenance(
            embeddinggemma_export.DEFAULT_MODEL_DIR
        ),
        flat_role=lambda _: embeddinggemma_export.GRAPH_NAME,
    ),
)


@dataclass(frozen=True)
class SeriesPlan:
    """The migration plan of one series (this stage writes not a single byte)."""

    name: str
    family: str
    provenance: Provenance
    components: tuple[Component, ...]


def classify(name: str) -> Family:
    """Series name → family (fails loudly when not in the table — the component name cannot be
    decided)."""
    for family in FAMILIES:
        if family.matches(name):
            return family
    raise SeriesMigrationError(
        f"series '{name}' matches no family's spelling"
        f" (known: {', '.join(entry.name for entry in FAMILIES)}) — the component name cannot be"
        " decided"
    )


def is_unreferenced(name: str) -> bool:
    """Whether the series is one that is not converted ({@link UNREFERENCED_SUFFIXES})."""
    return name.endswith(UNREFERENCED_SUFFIXES)


def shard_paths(directory: Path) -> tuple[Path, ...]:
    """The old shard files present in the directory (empty when there are none)."""
    return sequence_siblings(directory / f"{SHARD_STEM}{SHARD_SUFFIX}")


def component_directories(series: Path) -> list[Path]:
    """The directories holding an old shard list (the series directory and one level below)."""
    candidates = [series, *sorted(entry for entry in series.iterdir() if entry.is_dir())]
    return [directory for directory in candidates if shard_paths(directory)]


def series_directories(series: Path) -> list[Path]:
    """**Every directory** of the series (from the series directory down, unlimited depth).

    The range the leftover check ({@link _assert_nothing_unaccounted}) covers. Looking only at the
    component directories would leave outside the check the sidecar-only subdirectories
    (`<series>/pipeline/` / `<series>/host/`) and whole series without an old shard list
    (`anima-pipeline*`).
    """
    return [series, *sorted(entry for entry in series.rglob("*") if entry.is_dir())]


def plan_series(series: Path) -> SeriesPlan:
    """Builds the migration plan of one series (every component-name assignment failure and every
    left-behind sidecar fails here)."""
    name = series.name
    family = classify(name)
    components = [
        Component(directory, _graph_name(family, name, directory, series))
        for directory in component_directories(series)
    ]
    # MUST: apply the check to **every directory of the series** (even a series with 0 components
    # goes through it) — the migration runs once, so no later chance comes to notice a
    # `.safetensors` left behind with nowhere to fold into.
    for directory in series_directories(series):
        _assert_nothing_unaccounted(directory, series)
    return SeriesPlan(name, family.name, family.provenance(name), tuple(components))


def _graph_name(family: Family, name: str, directory: Path, series: Path) -> str:
    """Component directory → component name (= the container's graph name)."""
    if directory == series:
        if family.flat_role is None:
            raise SeriesMigrationError(
                f"{name}: an old shard sequence sits directly under the series, but family"
                f" '{family.name}' puts its components in subdirectories (no component name)"
            )
        return family.flat_role(name)
    role = family.roles.get(directory.name)
    if role is None:
        raise SeriesMigrationError(
            f"{name}: component directory '{directory.name}' is not in family '{family.name}''s"
            f" table (known: {', '.join(sorted(family.roles))})"
        )
    return role


def _assert_nothing_unaccounted(directory: Path, series: Path) -> None:
    """The directory's `.safetensors` files are exhausted by "old shards / goldens / sidecars to
    fold".

    MUST: stop on an unknown `.safetensors` — the migration runs once, so silently leaving a new
    kind of sidecar behind lets "a container one asset short" through to the distributed form
    (found only at run time).
    """
    accounted = {path.name for path in shard_paths(directory)}
    accounted.add(ROPE_BASE_FILE)
    accounted.update(ple_shard_names(directory))
    where = directory.relative_to(series.parent)
    for path in sorted(directory.glob(f"*{SHARD_SUFFIX}")):
        if path.name in accounted or path.name.startswith(GOLDEN_PREFIXES):
            continue
        raise SeriesMigrationError(
            f"{where}: '{path.name}' has nowhere to fold into"
            f" (neither an old shard, nor a golden <{' / '.join(GOLDEN_PREFIXES)}>, nor a sidecar)"
        )


def ple_shard_names(directory: Path) -> tuple[str, ...]:
    """Takes the PLE sidecar's shard names **from the index's declaration** (empty without an
    index).

    MUST: count from the declaration (not a glob of the files) — what is folded is the list
    {@link ple_sidecar_assets} takes from the index, so counting by glob would let "files not in
    the index" (leftovers of a regeneration with another version, for example) **pass the check
    without being folded**. A disagreement between the index and the files fails on the folding
    side.
    """
    index_path = directory / PLE_INDEX_FILE
    if not index_path.is_file():
        return ()
    index = read_ple_index(index_path, str(index_path))
    return tuple(str(entry["file"]) for entry in index["shards"])


def component_assets(
    component: Component, *, block_bytes: int = BLOCK_MAX_BYTES
) -> dict[str, AssetInput]:
    """The sidecars sitting with a component, as container assets (only 2 kinds exist: PLE and
    `rope_base`).

    The order becomes the physical layout order (PLE is in token order — the scanning source of
    container-v1 §4.2), so the PLE, which needs range reads, comes first.

    `block_bytes` is the step of the assets' block split and MUST be **the same value as the
    container side's step** (the caller passes one value from {@link migrate_series} to both) —
    if they were decided separately, a run with an injected size would place only the assets at
    the default step.
    """
    assets: dict[str, AssetInput] = {}
    index = component.directory / PLE_INDEX_FILE
    if index.is_file():
        assets.update(ple_sidecar_assets(index, block_bytes=block_bytes))
    rope_base = component.directory / ROPE_BASE_FILE
    if rope_base.is_file():
        payload = rope_base.read_bytes()
        assets[ROPE_BASE_ASSET] = AssetInput(
            EXTRA_ASSETS[ROPE_BASE_ASSET][1], len(payload), payload
        )
    return assets


def migrate_series(plan: SeriesPlan, *, block_bytes: int = BLOCK_MAX_BYTES) -> None:
    """Migrates component by component as planned (the self-check is `migrate_component`'s)."""
    for component in plan.components:
        assets = component_assets(component, block_bytes=block_bytes)
        result = migrate_component(
            component.representative,
            component.directory,
            provenance=plan.provenance,
            graph_name=component.graph_name,
            assets=assets,
            _block_bytes=block_bytes,
        )
        print(
            f"done {component.directory} graph={component.graph_name}"
            f" parts={len(result.parts)} initializers={result.initializers}"
            f" payloads={result.payloads} assets={result.assets}"
            f" bytes={total_bytes(result.parts)}",
            flush=True,
        )
        if plan.family == QAT_FAMILY:
            rewrite_qat_reference(component.directory, assets)


def rewrite_qat_reference(directory: Path, assets: Mapping[str, AssetInput]) -> None:
    """Rewrites a QAT series' `reference.json` to schema 3 (`pleBlocks`).

    **The only exception** to "left untouched". The `pleShards` an old record states is the
    sidecar's shard count, a number that exists nowhere once folded (the block cut ignores shard
    boundaries). The migration rebuilds what the record points to, so the record MUST follow —
    otherwise a series remains that only fails the distribution side's matching
    (`gemma4_qat.distribution.qat_plan`) with "field missing".

    The other fields (`fixedWeights` / `storageCounts` / `checkpoint` …) **stay as they are**: the
    migration changes no value by a single bit, so the numbers matched against the files stay
    correct.
    """
    path = directory / QAT_REFERENCE_FILE
    if not path.is_file():
        raise SeriesMigrationError(f"{directory}: the QAT series has no {QAT_REFERENCE_FILE}")
    record = json.loads(path.read_text(encoding="utf-8"))
    if QAT_SHARDS_FIELD not in record or QAT_BLOCKS_FIELD in record:
        raise SeriesMigrationError(
            f"{path}: not a record from before folding (only the form with `{QAT_SHARDS_FIELD}` and"
            f" without `{QAT_BLOCKS_FIELD}` is upgraded) — schema"
            f" {record.get('schema')} / fields {sorted(record)}"
        )
    if PLE_INDEX_ASSET not in assets:
        raise SeriesMigrationError(
            f"{directory}: {QAT_REFERENCE_FILE} is present although no PLE was folded"
        )
    index = json.loads(bytes(assets[PLE_INDEX_ASSET].payload))
    blocks = len(index["values"]["blocks"])
    # Keep the field order (`pleBlocks` takes the place of `pleShards`) — it is the same order as
    # the record a re-export writes, so the two records can be compared verbatim.
    rewritten = {
        (QAT_BLOCKS_FIELD if key == QAT_SHARDS_FIELD else key): (
            blocks if key == QAT_SHARDS_FIELD else value
        )
        for key, value in record.items()
    }
    rewritten["schema"] = REFERENCE_SCHEMA
    path.write_text(json.dumps(rewritten, indent=2) + "\n", encoding="utf-8")
    print(f"rewrote {path} schema={REFERENCE_SCHEMA} {QAT_BLOCKS_FIELD}={blocks}", flush=True)


def total_bytes(paths: Sequence[Path]) -> int:
    return sum(path.stat().st_size for path in paths)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Moves the old shard lists of the series outputs to krm (the old inputs are"
        " only read, the krm goes next to them)"
    )
    parser.add_argument(
        "--series-dir", type=Path, default=SERIES_ROOT, help="parent directory of the series"
    )
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        metavar="SERIES",
        help="migrate only this series (repeatable)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="print the plan only and write not a single byte"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Migrates every referenced series (stops where one fails)."""
    args = build_parser().parse_args(argv)
    root: Path = args.series_dir
    if not root.is_dir():
        raise SeriesMigrationError(f"no parent directory of the series: {root}")
    names = sorted(entry.name for entry in root.iterdir() if entry.is_dir())
    wanted = set(args.only)
    if wanted - set(names):
        raise SeriesMigrationError(
            f"series named by --only not in {root}: {', '.join(sorted(wanted - set(names)))}"
        )
    plans: list[SeriesPlan] = []
    for name in names:
        if wanted and name not in wanted:
            continue
        if is_unreferenced(name):
            print(f"skip {name} (record of an experiment / rejection)")
            continue
        plan = plan_series(root / name)
        if not plan.components:
            print(f"skip {name} (no old shard list)")
            continue
        plans.append(plan)
    for plan in plans:
        print(f"plan {plan.name} family={plan.family} provenance={plan.provenance.to_document()}")
        for component in plan.components:
            source = shard_paths(component.directory)
            print(
                f"  {component.directory.relative_to(root)} graph={component.graph_name}"
                f" shards={len(source)} bytes={total_bytes(source)}"
            )
    if args.dry_run:
        return
    for plan in plans:
        migrate_series(plan)


if __name__ == "__main__":
    try:
        main()
    except (SeriesMigrationError, MigrateError) as error:
        print(f"migrate_series: {error}", file=sys.stderr)
        raise SystemExit(1) from error
