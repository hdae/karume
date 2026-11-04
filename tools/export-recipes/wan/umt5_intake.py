"""Take in a third-party umT5-XXL-compatible encoder (one safetensors file of an HF repository)
(ADR 0122 decisions 5 / 6 — step b; the umT5 counterpart of ADR 0088's Civitai intake).

    uv run --group wan --inexact python -m wan.umt5_intake \\
        --repo <owner/name> --revision <40-digit commit> --file <name>.safetensors --name <name>

Its responsibility is the single step **up to fetching, checking and recording**. Conversion to a
container belongs to `wan.umt5_export` (`--intake`), and assembling the local experimental mirror
to `dist.py --pipeline umt5 --intake` (`wan.umt5_distribution`).

1. Read the HF API values (the LFS sha256 at the pinned revision, the license and base from
   `cardData`).
2. Place `config.json`. If the upstream has one, check its configuration fields
   ({@link CONFIG_FIELDS} — the fields that affect the graph and the relative-position bucket
   configuration) against the config of the original's pin, and fail loudly if they differ
   (because TS fixes the bucket configuration). If it has none, copy the config of the original's
   pin and record "configuration assumed (copy of the original)" (the check is then empty —
   `relative_attention_max_distance` cannot be derived from the weight shapes). The sha256 of the
   placed bytes is recorded, and {@link load_intake} checks it on every read (a rewrite after the
   intake is refused as a configuration not checked against the original).
3. Fetch the one named file at the pinned revision and check its sha256 against the API value
   (`.part` → renamed on a match). Only safetensors is accepted (only the row of the original's pin
   opens a pickle `.bin` — decision 2). The fetched file is placed in `inputs/umt5/<name>/` **with
   its name and content as is** (no F32 copy is made —
   the sha256 and file name stay identical to upstream and can be checked against the API values
   again later). If the intake directory (`<name>` under `--out`) is under `models/`, it fails
   before calling the API (comparing real paths — {@link assert_outside_distribution_root}).
4. Read the header and check the dtype (every tensor the same BF16 or F32 — FP8 or a mix is
   refused), the key set (the same form as Wan's `text_encoder` — the tied alias pair counts as
   one, and with both present the rule of decision 2 applies) and the shapes
   (`wan.umt5_export.inspect_intake_checkpoint`). If the intake directory holds anything outside
   the allowed files (the allow-list of {@link assert_only_intake_files}), it fails before the
   record is written.
5. Write the provenance record {@link INTAKE_FILE} (machine-owned — people do not add to it).

**It does not judge licenses** (ADR 0088 decision 5). It only copies the declared license, and when
there is no declaration or the value does not identify the redistribution terms (`unknown`, empty,
`other` without `license_name` and so on — {@link UNIDENTIFIED_LICENSES}), it records the
undeclared mark {@link UNDECLARED_LICENSE}. A container or an experimental mirror is built from
undeclared weights only when the explicit argument {@link ALLOW_UNDECLARED_LICENSE_FLAG} is given
at the checks of the writer and of the assembly (decision 6).
What the person or agent doing the intake verified goes into `license-review.md` in the same
directory (this command does not touch it — mixing people's notes into the machine record would
lose one of them on every re-fetch).

MUST: torch / transformers / huggingface_hub are not imported at module level — the distribution
recipe (`wan.umt5_distribution` — `import dist` does not load torch) takes the record reader from
here. Heavy dependencies are imported inside the functions that use them
(`tests/test_optional_group_imports.py`).

MUST: third-party repository names and commits are never written to tracked files (ADR 0122
constraint) — the proper names appear only in the runtime arguments and in the git-untracked
`inputs/umt5/<name>/intake.json`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from _shared.paths import DIST_ROOT, INPUTS_ROOT
from karume.dist import REPO_RE, DistError, assert_model_name
from wan.sources import UMT5_CONFIG, UMT5_SOURCES

#: Parent of the intake directories (`inputs/umt5/<name>/` — the input-material location of
#: docs/assets-layout.md).
INTAKE_ROOT = INPUTS_ROOT / "umt5"

#: The machine-owned provenance record (people's verification notes go to `license-review.md`).
INTAKE_FILE = "intake.json"

#: Spelling of the record format (bump the version when the shape changes — the reader refuses an
#: unknown version loudly). `/2` added `config.sha256`. Records of older versions are refused, not
#: migrated (an unreleased tool — redoing the intake with the same arguments rewrites the record in
#: the current version).
INTAKE_FORMAT = "karume-umt5-intake/2"

#: The verification notes of the person or agent doing the intake (this command does not write it
#: — module doc).
LICENSE_REVIEW_FILE = "license-review.md"

#: The undeclared-license mark (SPDX `NOASSERTION` — ADR 0122 decision 6, resolving the open point
#: "spelling of the undeclared mark"). The record's `license`, the container's `provenance.license`
#: and the frontmatter of the experimental mirror's card use the same spelling. The pre-publication
#: check (`tools/release/container_license.ts`) refuses the same spelling.
UNDECLARED_LICENSE = "NOASSERTION"

#: Values that do not identify the redistribution terms ({@link license_unidentified} — compared
#: ignoring surrounding whitespace and case). HF's `license: unknown` is an official choice but
#: states no terms, so it is treated like no declaration. The pre-publication check
#: (`UNIDENTIFIED_LICENSES` in `tools/release/container_license.ts`) refuses the same set.
#:
#: NOTE: `other` is not included — the containers of an existing distribution (anima) declare
#: `other` and are published with the terms written in the NOTICE. In an intake, {@link
#: _card_license} turns `other` without `license_name` into the undeclared mark, and
#: {@link load_intake} refuses a bare `other` in the record (both separate checks from this set).
UNIDENTIFIED_LICENSES = frozenset({"", "noassertion", "unknown", "none"})

#: HF's `license: other` (the identifier is in `license_name` — a bare `other` names no terms).
_OTHER_LICENSE = "other"

#: The explicit permission to build a container and an experimental mirror that carry the
#: undeclared mark (the writer and the dist driver accept the same spelling).
ALLOW_UNDECLARED_LICENSE_FLAG = "--allow-undeclared-license"

#: Accepted source dtypes (the safetensors header spelling). The writer widens BF16 to F32 when
#: reading (lossless). FP8 and the like are derivatives that lost information, so they are not
#: accepted.
INTAKE_DTYPES: tuple[str, ...] = ("BF16", "F32")

#: The configuration fields checked (those that affect the graph and the relative-position bucket
#: configuration). The bucket configuration is not in the graph descriptor (TS fixes `numBuckets`
#: 32 and `maxDistance` 128), so a difference could change only the output while the container
#: stays byte-identical — the intake checks them against the original's config.
CONFIG_FIELDS: tuple[str, ...] = (
    "d_model",
    "d_kv",
    "d_ff",
    "num_layers",
    "num_heads",
    "vocab_size",
    "feed_forward_proj",
    "dense_act_fn",
    "is_gated_act",
    "layer_norm_epsilon",
    "relative_attention_num_buckets",
    "relative_attention_max_distance",
)

#: Where the config came from (`upstream` = the upstream's config, placed after checking it
#: against the original / `base-copy` = the upstream has none, so the config of the original's pin
#: was copied — the configuration is assumed).
CONFIG_SOURCES: tuple[str, ...] = ("upstream", "base-copy")

#: The original's row used as the baseline of the configuration check and the copy (a key of
#: `wan.sources.UMT5_SOURCES` — spelled like the umT5 distribution's model name
#: `wan.umt5_distribution.UMT5_DEFAULT_MODEL`; that module imports this one, so it cannot be
#: imported from here — a test checks they match).
BASE_UPSTREAM = "xxl"

#: An HF commit SHA (40 lowercase hex digits). Branch names and tags are not pins.
_COMMIT_SHA = re.compile(r"[0-9a-f]{40}")
_SHA256 = re.compile(r"[0-9a-f]{64}")

#: The read / write chunk (the sha256 is taken while copying an 11 GB-class safetensors).
_CHUNK_BYTES = 8 << 20

#: The record's fields (no more and no fewer — an unknown field fails loudly).
_RECORD_KEYS = frozenset(
    {
        "format",
        "name",
        "repo",
        "revision",
        "file",
        "dtype",
        "license",
        "base_model",
        "config",
        "fetched_at",
    }
)


class Umt5IntakeError(RuntimeError):
    """The intake's input, an upstream value or an intake record differs from what is assumed (ADR
    0122 decision 5)."""


# ---------------------------------------------------------------------------
# The provenance record
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IntakeFile:
    """One intake safetensors file (the upstream file name, byte size and LFS sha256)."""

    name: str
    size: int
    sha256: str


@dataclass(frozen=True)
class ConfigOrigin:
    """Where `config.json` came from ({@link CONFIG_SOURCES}), the original's pin it was checked
    against / copied from, and the sha256 of the placed bytes ({@link load_intake} checks it on
    every read)."""

    source: str
    base_repo: str
    base_revision: str
    sha256: str


@dataclass(frozen=True)
class Umt5Intake:
    """The contents of `intake.json` ({@link load_intake} checks and builds it).

    `directory` is where the record was read from (not written into the record — where the intake
    file and the config are).
    """

    directory: Path
    name: str
    repo: str
    revision: str
    file: IntakeFile
    #: The source dtype ({@link INTAKE_DTYPES}) — the writer accepts only this dtype.
    dtype: str
    #: The declared license identifier, or {@link UNDECLARED_LICENSE} without a declaration.
    license: str
    #: The declared base (`None` without a declaration).
    base_model: tuple[str, ...] | None
    config: ConfigOrigin
    fetched_at: str

    @property
    def undeclared(self) -> bool:
        """Whether the upstream declares no license (the undeclared mark was recorded).

        {@link load_intake} narrows unidentified values to the mark's spelling only, but this uses
        the same predicate too (even a value built without a record never treats an unidentified
        value as declared)."""
        return license_unidentified(self.license)

    @property
    def config_assumed(self) -> bool:
        """Whether the configuration is assumed (the upstream has no config, and the config of the
        original's pin was copied)."""
        return self.config.source == "base-copy"

    @property
    def checkpoint(self) -> Path:
        return self.directory / self.file.name

    def to_document(self) -> dict[str, Any]:
        return {
            "format": INTAKE_FORMAT,
            "name": self.name,
            "repo": self.repo,
            "revision": self.revision,
            "file": {"name": self.file.name, "size": self.file.size, "sha256": self.file.sha256},
            "dtype": self.dtype,
            "license": self.license,
            "base_model": None if self.base_model is None else list(self.base_model),
            "config": {
                "source": self.config.source,
                "base": {"repo": self.config.base_repo, "revision": self.config.base_revision},
                "sha256": self.config.sha256,
            },
            "fetched_at": self.fetched_at,
        }


def license_unidentified(value: str) -> bool:
    """Whether `value` does not identify redistribution terms ({@link UNIDENTIFIED_LICENSES})."""
    return value.strip().lower() in UNIDENTIFIED_LICENSES


def assert_outside_distribution_root(path: Path, *, root: Path) -> None:
    """Fail if `path` is under `models/` (`root`) (files that come from an intake — ADR 0122
    decision 5).

    The one check shared by the intake (`--out`) and the assembly of the experimental mirror
    (`--out` of `dist.py --intake`).

    MUST: compare real paths (with symlinks and `..` resolved) — comparing spellings lets a
    symlink to `models/` or a relative path containing `..` slip through. `models/` is where the
    publication script uploads `models/<repo>` as is, and files from an intake have no publication
    path, so they (neither the raw third-party safetensors nor a container written from it) have no
    reason to be there (whether or not a license is declared).
    """
    resolved = path.resolve()
    real_root = root.resolve()
    if resolved.is_relative_to(real_root):
        raise Umt5IntakeError(
            f"{path} (real path {resolved}) is under {real_root} — files that come from an"
            " intake are not placed in models/"
        )


def assert_only_intake_files(directory: Path, file: str) -> None:
    """Check that directly under the intake directory `directory` there is nothing outside the
    allowed files (the file `file` the record names, `config.json`, {@link INTAKE_FILE},
    {@link LICENSE_REVIEW_FILE}) (subdirectories also count as outside).

    MUST: fail loudly if there is — the reader opens only the record's file, but with weights
    outside the record in the same directory both people and tools can mistake which one was
    checked (with an index, the moment it reaches a reader that follows `weight_map`, an unchecked
    file is silently read).
    MUST: closed by an allow-list (names compared exactly — case-sensitive) — an open list of
    weight extensions would let uppercase extensions, formats not on the list (`.pkl`, `.npz` and
    so on) and a half-written `.part` slip through.
    MUST: symlinks are refused even with an allowed name — `is_file()` follows symlinks, so a
    symlink with an allowed name could put weights outside the record in the same directory.
    """
    allowed = {file, UMT5_CONFIG, INTAKE_FILE, LICENSE_REVIEW_FILE}
    foreign = sorted(
        path.name + ("/" if path.is_dir() else "")
        for path in directory.iterdir()
        if path.name not in allowed or path.is_symlink() or not path.is_file()
    )
    if foreign:
        raise Umt5IntakeError(
            f"{directory} holds files outside the record: {foreign} — only {sorted(allowed)} may"
            " be there (so that a weight file outside the record is never mistaken for it,"
            " nothing else is placed in the intake directory, whatever its kind)"
        )


def _assert_config_pinned(directory: Path, sha256: str, where: str) -> None:
    """Whether the intake directory's `config.json` matches the record's sha256 (the bytes placed
    at intake time).

    MUST: fail loudly when it differs or is missing — the writer and the reference re-read the
    graph and bucket configuration from this file, so a rewrite after the intake (e.g.
    `relative_attention_max_distance`) would produce a container and goldens with a configuration
    never checked against the original.
    """
    path = directory / UMT5_CONFIG
    if not path.is_file():
        raise Umt5IntakeError(f"{where}: {path} is missing — redo the intake")
    actual = _sha256(path)
    if actual != sha256:
        raise Umt5IntakeError(
            f"{where}: {path}: the sha256 {actual} differs from the record's {sha256} — the"
            " config was rewritten after the intake (only the configuration at intake time,"
            " checked against the original, is accepted; to change it, redo the intake)"
        )


def assert_intake_name(name: str) -> str:
    """Check the intake's name (it becomes the series name `umt5-xxl-<name>-i8-dyn` and the model
    name of the experimental mirror).

    It is restricted to the accepted set of distribution model names (ADR 0077) and never overlaps
    the names of the original's rows (the keys of `wan.sources.UMT5_SOURCES`) — an overlap would
    make the distribution check treat it as the original's row and swap what the provenance is
    checked against.
    """
    try:
        assert_model_name(name)
    except DistError as cause:
        raise Umt5IntakeError(str(cause)) from cause
    if name in UMT5_SOURCES:
        raise Umt5IntakeError(
            f"the name '{name}' is the same as the original's row (wan.sources.UMT5_SOURCES) —"
            " give the intake another name"
        )
    return name


def _assert_file_name(name: str) -> str:
    """The file name to take in (one safetensors at the repository root — no separators, `..` or
    pickles)."""
    if not name or name in {".", ".."} or "/" in name or "\\" in name or ":" in name:
        raise Umt5IntakeError(f"the file name {name!r} is not a basename at the repository root")
    if not name.endswith(".safetensors"):
        raise Umt5IntakeError(
            f"the file {name!r} is not safetensors — an intake accepts safetensors only (only"
            " the row of the original's pin opens a pickle .bin — ADR 0122 decision 2)"
        )
    return name


def _require(mapping: Mapping[str, Any], key: str, kind: type, where: str) -> Any:
    value = mapping.get(key)
    if not isinstance(value, kind) or (isinstance(value, bool) and kind is not bool):
        raise Umt5IntakeError(f"{where}: '{key}' is not {kind.__name__} ({value!r})")
    return value


def load_intake(directory: Path) -> Umt5Intake:
    """Read the record from an intake directory, check its shape and return it.

    MUST: an unknown format, missing or extra fields, or spellings out of range fail loudly — the
    writer accepts only the record's `dtype`, and the container and distribution checks take the
    record's `license` / `revision` as the source of truth for provenance, so silently reading a
    broken record would produce a container with false provenance.
    """
    path = directory / INTAKE_FILE
    if not path.is_file():
        raise Umt5IntakeError(
            f"{path} is missing — take the intake first with `python -m wan.umt5_intake`"
            " (weights without a record are not read)"
        )
    document = json.loads(path.read_text(encoding="utf-8"))
    where = str(path)
    if not isinstance(document, dict):
        raise Umt5IntakeError(f"{where}: not a JSON object")
    if document.get("format") != INTAKE_FORMAT:
        raise Umt5IntakeError(
            f"{where}: format {document.get('format')!r} is not {INTAKE_FORMAT} (records of older"
            " versions are not migrated — redo `python -m wan.umt5_intake` with the same"
            " arguments)"
        )
    if set(document) != _RECORD_KEYS:
        raise Umt5IntakeError(
            f"{where}: the fields differ (missing {sorted(_RECORD_KEYS - set(document))} /"
            f" extra {sorted(set(document) - _RECORD_KEYS)})"
        )
    name = assert_intake_name(_require(document, "name", str, where))
    repo = _require(document, "repo", str, where)
    if REPO_RE.fullmatch(repo) is None:
        raise Umt5IntakeError(f"{where}: repo {repo!r} is not of the form '<owner>/<name>'")
    revision = _require(document, "revision", str, where)
    if _COMMIT_SHA.fullmatch(revision) is None:
        raise Umt5IntakeError(f"{where}: revision {revision!r} is not a 40-digit commit SHA")
    file = _require(document, "file", dict, where)
    if set(file) != {"name", "size", "sha256"}:
        raise Umt5IntakeError(
            f"{where}: the fields of file are not name / size / sha256 ({sorted(file)})"
        )
    sha256 = _require(file, "sha256", str, f"{where} file")
    if _SHA256.fullmatch(sha256) is None:
        raise Umt5IntakeError(f"{where}: file.sha256 {sha256!r} is not 64 lowercase hex digits")
    size = _require(file, "size", int, f"{where} file")
    if size <= 0:
        raise Umt5IntakeError(f"{where}: file.size {size} is not positive")
    dtype = _require(document, "dtype", str, where)
    if dtype not in INTAKE_DTYPES:
        raise Umt5IntakeError(f"{where}: dtype {dtype!r} is outside {list(INTAKE_DTYPES)}")
    license_id = _require(document, "license", str, where)
    if not license_id.strip():
        raise Umt5IntakeError(f"{where}: license is empty ({UNDECLARED_LICENSE} when undeclared)")
    if license_unidentified(license_id) and license_id != UNDECLARED_LICENSE:
        # The intake writes unidentified values as the mark's spelling. Any other spelling is a
        # hand-edited record, and written into the container as is, the pre-publication check
        # could miss it over the spelling difference (narrowed to one form here).
        raise Umt5IntakeError(
            f"{where}: license {license_id!r} is a value that does not identify the"
            f" redistribution terms — the undeclared mark is accepted only as {UNDECLARED_LICENSE}"
        )
    if license_id.strip().lower() == _OTHER_LICENSE:
        raise Umt5IntakeError(
            f"{where}: license {license_id!r} is a value that does not identify the"
            " redistribution terms — the intake records the identifier from license_name"
            f" ({UNDECLARED_LICENSE} when there is none)"
        )
    base_model = document["base_model"]
    if base_model is not None and (
        not isinstance(base_model, list)
        or not base_model
        or not all(isinstance(item, str) and item for item in base_model)
    ):
        raise Umt5IntakeError(
            f"{where}: base_model is neither null nor a list of non-empty strings"
        )
    config = _require(document, "config", dict, where)
    base = _require(config, "base", dict, f"{where} config")
    source = _require(config, "source", str, f"{where} config")
    if (
        set(config) != {"source", "base", "sha256"}
        or set(base) != {"repo", "revision"}
        or source not in CONFIG_SOURCES
    ):
        raise Umt5IntakeError(
            f"{where}: config is not of the form {{source: {list(CONFIG_SOURCES)},"
            " base: {repo, revision}, sha256}"
        )
    config_sha256 = _require(config, "sha256", str, f"{where} config")
    if _SHA256.fullmatch(config_sha256) is None:
        raise Umt5IntakeError(
            f"{where}: config.sha256 {config_sha256!r} is not 64 lowercase hex digits"
        )
    _assert_config_pinned(directory, config_sha256, where)
    return Umt5Intake(
        directory=directory,
        name=name,
        repo=repo,
        revision=revision,
        file=IntakeFile(
            name=_assert_file_name(_require(file, "name", str, where)), size=size, sha256=sha256
        ),
        dtype=dtype,
        license=license_id,
        base_model=None if base_model is None else tuple(base_model),
        config=ConfigOrigin(
            source=source,
            base_repo=_require(base, "repo", str, f"{where} config.base"),
            base_revision=_require(base, "revision", str, f"{where} config.base"),
            sha256=config_sha256,
        ),
        fetched_at=_require(document, "fetched_at", str, where),
    )


def assert_license_intent(intake: Umt5Intake, *, allowed: bool) -> None:
    """For an intake with the undeclared mark, require the explicit permission
    ({@link ALLOW_UNDECLARED_LICENSE_FLAG}).

    MUST: without it, fail before a single byte is written (ADR 0122 decision 6). The only thing
    judged is the machine-determined fact that "the upstream declares no license"; the license
    terms are not interpreted.
    """
    if intake.undeclared and not allowed:
        raise Umt5IntakeError(
            f"the upstream of intake '{intake.name}' declares no license ({UNDECLARED_LICENSE}) —"
            f" to build it for local experiments only, pass {ALLOW_UNDECLARED_LICENSE_FLAG}"
            " explicitly (ADR 0122 decision 6)"
        )


# ---------------------------------------------------------------------------
# HF API (replaceable — tests pass mock responses)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HubFile:
    """One file in a revision (byte size and LFS sha256 — `None` when it is not LFS)."""

    size: int
    sha256: str | None


@dataclass(frozen=True)
class HubRevision:
    """The HF API values the intake uses (from `model_info(revision=…, files_metadata=True)`)."""

    #: The commit the API returned (checked to be the same as the requested revision).
    sha: str
    files: Mapping[str, HubFile]
    #: `license` of `cardData` (`license_name` for `other`). `None` when there is no declaration or
    #: the value does not identify the redistribution terms ({@link license_unidentified}).
    license: str | None
    #: `base_model` of `cardData` (`None` without a declaration).
    base_model: tuple[str, ...] | None


class Hub(Protocol):
    def revision(self, repo: str, revision: str) -> HubRevision: ...

    def download(self, repo: str, revision: str, filename: str) -> Path: ...


def _card_license(license_id: Any, license_name: Any, repo: str) -> str | None:
    """`license` / `license_name` of `cardData` → identifier.

    No declaration or a value that does not identify the redistribution terms (`unknown`, empty,
    `other` without `license_name` and so on — {@link license_unidentified}) gives `None` (the
    record gets the undeclared mark). A value that is not a string is a broken shape and fails
    loudly.
    """
    if license_id is None:
        return None
    if not isinstance(license_id, str):
        raise Umt5IntakeError(f"{repo}: cardData license {license_id!r} is not a string")
    if license_id.strip().lower() != _OTHER_LICENSE:
        return None if license_unidentified(license_id) else license_id
    if license_name is None:
        return None
    if not isinstance(license_name, str):
        raise Umt5IntakeError(f"{repo}: cardData license_name {license_name!r} is not a string")
    return None if license_unidentified(license_name) else license_name


def _card_base_model(value: Any, repo: str) -> tuple[str, ...] | None:
    if value is None:
        return None
    values = [value] if isinstance(value, str) else value
    if (
        not isinstance(values, list)
        or not values
        or not all(isinstance(v, str) and v for v in values)
    ):
        raise Umt5IntakeError(
            f"{repo}: cardData base_model {value!r} is neither a string nor a list of strings"
        )
    return tuple(values)


class HfHub:
    """The default implementation, reading through huggingface_hub (fetches go through HF's
    default cache — nothing is downloaded again once fetched)."""

    def revision(self, repo: str, revision: str) -> HubRevision:
        from huggingface_hub import HfApi

        info = HfApi().model_info(repo, revision=revision, files_metadata=True)
        card = info.card_data
        files = {
            sibling.rfilename: HubFile(
                size=int(sibling.size or 0),
                sha256=None if sibling.lfs is None else sibling.lfs.sha256,
            )
            for sibling in info.siblings or ()
        }
        return HubRevision(
            sha=str(info.sha),
            files=files,
            license=_card_license(
                getattr(card, "license", None), getattr(card, "license_name", None), repo
            ),
            base_model=_card_base_model(getattr(card, "base_model", None), repo),
        )

    def download(self, repo: str, revision: str, filename: str) -> Path:
        from huggingface_hub import hf_hub_download

        return Path(hf_hub_download(repo, filename, revision=revision))


# ---------------------------------------------------------------------------
# Intake
# ---------------------------------------------------------------------------


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _place_checkpoint(source: Path, dest: Path, expected: HubFile) -> None:
    """Copy the fetched file to `dest` and put it in place (nothing happens if it is already there
    with the same sha256 — safe to re-run).

    It writes `<dest>.part` and renames only once the sha256 and byte size match the API values —
    a truncated file left under the final name would be read as "already taken in" by the next
    run.
    """
    assert expected.sha256 is not None
    if dest.exists():
        actual = _sha256(dest)
        if actual != expected.sha256:
            raise Umt5IntakeError(
                f"a different file is in place under the same name: {dest} — API"
                f" {expected.sha256} / measured {actual} (the upstream replaced it, or it is a"
                " different file placed by hand; check before deleting it)"
            )
        return
    part = dest.with_name(f"{dest.name}.part")
    digest = hashlib.sha256()
    written = 0
    with source.open("rb") as reader, part.open("wb") as writer:
        while chunk := reader.read(_CHUNK_BYTES):
            writer.write(chunk)
            digest.update(chunk)
            written += len(chunk)
    actual = digest.hexdigest()
    if actual != expected.sha256 or written != expected.size:
        raise Umt5IntakeError(
            f"the fetched file differs from the API values: sha256 API {expected.sha256} /"
            f" measured {actual}, bytes API {expected.size} / measured {written} ({part} is kept"
            " — check before deleting it)"
        )
    part.rename(dest)


def _place_config(
    hub: Hub, info: HubRevision, repo: str, revision: str, dest: Path
) -> ConfigOrigin:
    """Place `config.json` and return where it came from (step 2 of the module doc).

    The recorded sha256 is taken from the very bytes that were checked (copied) — re-reading after
    placing would make the record endorse a rewrite in between.
    """
    base = UMT5_SOURCES[BASE_UPSTREAM].source
    base_bytes = hub.download(base.repo, base.revision, UMT5_CONFIG).read_bytes()
    if UMT5_CONFIG not in info.files:
        source, placed = "base-copy", base_bytes
    else:
        source, placed = "upstream", hub.download(repo, revision, UMT5_CONFIG).read_bytes()
        upstream = json.loads(placed.decode("utf-8"))
        reference = json.loads(base_bytes.decode("utf-8"))
        differing = {
            key: (upstream.get(key), reference.get(key))
            for key in CONFIG_FIELDS
            if upstream.get(key) != reference.get(key)
        }
        if differing:
            raise Umt5IntakeError(
                f"{repo}@{revision}: {UMT5_CONFIG} differs from the original"
                f" {base.repo}@{base.revision} in the configuration fields (upstream, original):"
                f" {differing} — a umT5 with a different graph or bucket configuration is not a"
                " compatible encoder"
            )
    (dest / UMT5_CONFIG).write_bytes(placed)
    return ConfigOrigin(
        source=source,
        base_repo=base.repo,
        base_revision=base.revision,
        sha256=hashlib.sha256(placed).hexdigest(),
    )


def intake(
    repo: str,
    revision: str,
    file: str,
    name: str,
    *,
    hub: Hub,
    out_root: Path = INTAKE_ROOT,
) -> Umt5Intake:
    """Take one upstream safetensors into `<out_root>/<name>/`, write the record and return it."""
    if REPO_RE.fullmatch(repo) is None:
        raise Umt5IntakeError(f"--repo {repo!r} is not of the form '<owner>/<name>'")
    if _COMMIT_SHA.fullmatch(revision) is None:
        raise Umt5IntakeError(
            f"--revision {revision!r} is not a 40-digit commit SHA (branch names and tags are not"
            " pins)"
        )
    _assert_file_name(file)
    assert_intake_name(name)
    destination = out_root / name
    # Fails before calling the API and before placing a single byte.
    assert_outside_distribution_root(destination, root=DIST_ROOT)
    info = hub.revision(repo, revision)
    if info.sha != revision:
        raise Umt5IntakeError(
            f"the API commit {info.sha} differs from the requested revision {revision}"
        )
    entry = info.files.get(file)
    if entry is None:
        raise Umt5IntakeError(f"{repo}@{revision} has no {file!r} (present: {sorted(info.files)})")
    if entry.sha256 is None:
        raise Umt5IntakeError(
            f"{repo}@{revision}: {file!r} is not LFS — its sha256 cannot be taken from the API"
        )

    record_path = destination / INTAKE_FILE
    if record_path.is_file():
        previous = json.loads(record_path.read_text(encoding="utf-8"))
        if (
            previous.get("repo"),
            previous.get("revision"),
            previous.get("file", {}).get("name"),
        ) != (
            repo,
            revision,
            file,
        ):
            raise Umt5IntakeError(
                f"{destination} is the directory of another intake"
                f" ({previous.get('repo')}@{previous.get('revision')})"
                " — give another name, or check before deleting it"
            )
    destination.mkdir(parents=True, exist_ok=True)
    config = _place_config(hub, info, repo, revision, destination)
    _place_checkpoint(hub.download(repo, revision, file), destination / file, entry)

    from wan.umt5_export import inspect_intake_checkpoint

    dtype = inspect_intake_checkpoint(destination, file, entry.sha256)
    record = Umt5Intake(
        directory=destination,
        name=name,
        repo=repo,
        revision=revision,
        file=IntakeFile(name=file, size=entry.size, sha256=entry.sha256),
        dtype=dtype,
        license=(
            UNDECLARED_LICENSE
            if info.license is None or license_unidentified(info.license)
            else info.license
        ),
        base_model=info.base_model,
        config=config,
        fetched_at=datetime.now(UTC).isoformat(),
    )
    record_path.write_text(
        json.dumps(record.to_document(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    # The written record must pass the reader's checks (the writer and the reader never disagree
    # on the shape).
    return load_intake(destination)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--repo", required=True, help="the upstream HF repository (<owner>/<name>)")
    parser.add_argument("--revision", required=True, help="the commit SHA to pin (40 digits)")
    parser.add_argument(
        "--file", required=True, help="the safetensors to take in (one at the repository root)"
    )
    parser.add_argument(
        "--name",
        required=True,
        help="the intake's name (inputs/umt5/<name>/, series umt5-xxl-<name>-i8-dyn)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=INTAKE_ROOT,
        help="parent of the intake directory (nothing under models/ is accepted)",
    )
    args = parser.parse_args(argv)
    record = intake(args.repo, args.revision, args.file, args.name, hub=HfHub(), out_root=args.out)
    print(json.dumps(record.to_document(), ensure_ascii=False, indent=1))
    print(f"[umt5-intake] {record.directory / INTAKE_FILE}", flush=True)
    if record.undeclared:
        print(
            f"[umt5-intake] the upstream declares no license ({UNDECLARED_LICENSE}) — the writer"
            f" and the assembly build it for local experiments only when"
            f" {ALLOW_UNDECLARED_LICENSE_FLAG} is given explicitly. Record what you verified in"
            f" {record.directory / LICENSE_REVIEW_FILE}",
            flush=True,
        )
    print(
        "[umt5-intake] next: uv run --group wan --inexact python -m wan.umt5_export write"
        f" --intake {record.directory}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
