"""The repository's dist driver — combines the core assembly engine with the recipe-side pipelines.

The {@link karume.dist.PIPELINES} of `karume.dist` are "the pipelines that can be built with the
core wheel alone", not all of them (ADR 0065 decision 2 — per-model recipes live outside the
wheel). Now that every family has moved, the core's table is **empty**, so this dictionary alone
is the source of truth of the accepted set.

    uv run python dist.py                                  # default = anima (5 variants, ADR 0087)
    uv run python dist.py --pipeline anima-extra \\
        --model anima-wai-v1.0 --model anima-copycat-20260610 \\
        --out ../../models/karume-anima-extra \\
        --resolve-refs hf                                  # fine-tunes (reference karume-anima)
    uv run python dist.py --pipeline irodori               # v4-small (self-contained)
    uv run python dist.py --pipeline irodori --model v4.1-small \\
        --resolve-refs hf                                  # references karume-irodori-v4-small
    uv run python dist.py --pipeline sbv2-fn               # the FN line (HF publication on hold)
    uv run python dist.py --pipeline sbv2 \\
        --model F1 --model F2 --out ../../models/karume-sbv2-jvnv
    uv run python dist.py --pipeline siglip2 \\
        --model base --model so400m --out ../../models/karume-siglip2
    uv run python dist.py --pipeline lucida                # a BiRefNet_HR derivative (own repo)
    uv run python dist.py --pipeline umt5                  # umT5-XXL encoder (referenced by Wan)
    uv run python dist.py --pipeline wan --allow-placeholder-ref
                                                           # Wan2.1 T2V 1.3B (not published)
    uv run python dist.py --pipeline wan-ti2v --resolve-refs hf
                                                           # Wan2.2 TI2V 5B (references umT5)
    uv run python dist.py --pipeline moss-image            # MOSS-Image-SFT

The default locations (`--series` / `--out`) are passed from here too — the repository's
`outputs/series/` and `models/` are repository topology, and the core holds no spellings (ADR 0065
Consequences; the same MUST in `karume.dist`).

## Cross-repository references (ADR 0038 §7, addendum 2026-10-28)

Which roles of a model are borrowed from another distribution repository is the pipeline's
declaration (`karume.dist.Pipeline.references`). This driver supplies what the core does not know:
the local mirror of each referenced repository (`DIST_ROOT/<repository name>` — the bytes a
reference is reconciled against) and the commit it is pinned at, from **exactly one** revision
source per build:

- `--resolve-refs hf` reads the referenced repository's main from the HF API and, before a byte is
  written, requires every borrowed file to be in that commit with the bytes reconciled locally
  (the LFS oid = the local sha256; a file stored without LFS: its git blob id) —
  {@link confirm_on_hub}. A stale SHA, a referenced mirror rebuilt but not uploaded, or a renamed
  path then fails at build time instead of at a user's fetch. This is the only network access of
  distribution assembly, and it lives in this driver (the core wheel has none);
- `--ref-revision NAME=SHA` (the core's option) pins it explicitly and offline — nothing is
  checked against the Hub;
- {@link ALLOW_PLACEHOLDER_FLAG} pins every reference at the placeholder {@link
  PLACEHOLDER_REVISION} (40 zeros — ADR 0119 addendum "design of step 10d" E), for a development
  mirror while the referenced repository is unpublished (a local source resolves the reference
  through an explicit `crossRepo` mapping and does not look at the revision). The placeholder has
  the shape of a real sha, so the hub's parse, the core's assembly and `verify_dist` all accept it
  — published, it would point at a commit that does not exist. It is therefore only reachable
  through this explicit flag; `--ref-revision NAME=<40 zeros>` is refused before a byte is
  written, and the publication rebuild (docs/release-runbook.md §0) never passes the flag.

A model that declares references and gets no source fails before a byte is written — there is no
silent self-contained build.

## Local experimental mirrors from an intake (ADR 0122 decisions 5 / 6)

A series written from the intake of a third-party umT5-compatible encoder (`wan.umt5_intake`) can
only be built with `--pipeline umt5` plus `--intake`. The repository name, the root files and the
card are derived from the intake record `intake.json` (`wan.umt5_distribution.intake_pipeline`),
and the default output is under {@link LOCAL_DIST_ROOT}.
**Whether or not a license is declared**, an output under `models/` (the place of distributions
that are uploaded as they are) fails before a byte is written
({@link assert_outside_distribution_root} — comparing real paths). An intake without a declared
license needs the explicit {@link ALLOW_UNDECLARED_LICENSE_FLAG} (it fails at planning time):

    uv run python dist.py --pipeline umt5 --intake ../../inputs/umt5/<name> \
        --allow-undeclared-license        # undeclared only — out is outputs/misc/local-dist/<name>

NOTE: the `**CORE_PIPELINES` expansion stays — the core is still "the side that receives a table"
rather than "the side that has no table", and a pipeline buildable with the core wheel alone would
silently join if one appears in the future.
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import re
import sys
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

from _shared.paths import DIST_ROOT, MISC_ROOT, SERIES_ROOT
from anima import distribution as anima_distribution
from birefnet import distribution as birefnet_distribution
from depth_anything import distribution as depth_anything_distribution
from gemma4 import distribution as gemma4_distribution
from gemma4_qat import distribution as gemma4_qat_distribution
from irodori import distribution as irodori_distribution
from karume.dist import PIPELINES as CORE_PIPELINES
from karume.dist import (
    REVISION_RE,
    CrossRepoReference,
    DistError,
    ExternalComponents,
    Pipeline,
    ReferenceCheck,
    build_parser,
    declared_references,
    parse_reference_revisions,
    selected_models,
)
from karume.dist import main as dist_main
from moss import distribution as moss_distribution
from sbv2 import distribution as sbv2_distribution
from siglip2 import distribution as siglip2_distribution
from vowel_detector import distribution as vowel_detector_distribution
from wan import distribution as wan_distribution
from wan import umt5_distribution
from wan.umt5_intake import ALLOW_UNDECLARED_LICENSE_FLAG, Umt5IntakeError, load_intake
from wan.umt5_intake import assert_outside_distribution_root as intake_outside_distribution_root

#: The whole accepted set. The order is also the order of `--help`, so the default comes first.
PIPELINES: Mapping[str, Pipeline] = {
    "anima": anima_distribution.OFFICIAL_PIPELINE,
    "anima-extra": anima_distribution.EXTRA_PIPELINE,
    "sbv2": sbv2_distribution.PIPELINE,
    # The voice families have different licenses (JVNV = CC BY-SA 4.0, FN = Booth's distribution
    # terms), so they are separate entries.
    "sbv2-fn": sbv2_distribution.FN_PIPELINE,
    "irodori": irodori_distribution.PIPELINE,
    "siglip2": siglip2_distribution.PIPELINE,
    "birefnet": birefnet_distribution.PIPELINE,
    # The derivative is its own repository (ADR 0092 decision 1) — the copyright notice at the
    # repository root differs, so the Pipeline is a separate entry too.
    "lucida": birefnet_distribution.LUCIDA_PIPELINE,
    "depth-anything": depth_anything_distribution.PIPELINE,
    "vowel-detector": vowel_detector_distribution.PIPELINE,
    "gemma4": gemma4_distribution.PIPELINE,
    # The q4_0 QAT line of the `gemma4/1` contract (`karume-gemma4-qat` — E2B / E4B / 12B with
    # their QAT drafters) and the mobile QAT family (`karume-gemma4-qat-mobile` — contract
    # `gemma4-qat/1`).
    "gemma4-qat": gemma4_distribution.QAT_Q4_0_PIPELINE,
    "gemma4-qat-mobile": gemma4_qat_distribution.PIPELINE,
    "wan": wan_distribution.PIPELINE,
    # Wan's generations are separate repositories (ADR 0092 decision 2 — `karume-wan2.2`), so the
    # Pipeline is a separate entry too.
    "wan-ti2v": wan_distribution.TI2V_PIPELINE,
    # What Wan's text_encoder references (ADR 0119 addendum D — a components-only repository; no
    # TS family reads it).
    "umt5": umt5_distribution.PIPELINE,
    # MOSS-Image-SFT (ADR 0138 addendum of 2026-11-01 — self-contained: the Qwen3.5 text tower is
    # the checkpoint's own, stored in this repository).
    "moss-image": moss_distribution.PIPELINE,
    **CORE_PIPELINES,
}

#: Keeps the UX of the old `karume dist` (no arguments) on the driver side.
DEFAULT_PIPELINE = "anima"

#: The placeholder commit sha a development mirror is pinned at while the referenced repository is
#: unpublished (ADR 0119 addendum E — 40 zeros).
PLACEHOLDER_REVISION = "0" * 40

#: The explicit statement that a build is a development mirror pinned at the placeholder (module
#: doc, "Cross-repository references").
ALLOW_PLACEHOLDER_FLAG = "--allow-placeholder-ref"

#: Resolve the references' revisions from the Hub and confirm the borrowed files there (module doc).
RESOLVE_REFS_FLAG = "--resolve-refs"

#: The core's explicit revision option (spelled here only for the messages).
REF_REVISION_FLAG = "--ref-revision"

#: The Hugging Face Hub {@link RESOLVE_REFS_FLAG} reads (the HF API under `/api/`).
HF_ENDPOINT = "https://huggingface.co"

#: Seconds one HF API request may take before the build fails.
HF_TIMEOUT_SECONDS = 60

#: One GET of the HF API: URL → (the decoded JSON body, the URL of the next page or `None` — the
#: tree listing is paginated through the `Link: <…>; rel="next"` header). Injected so that the
#: tests read no network.
HfGet = Callable[[str], tuple[Any, str | None]]

_NEXT_PAGE_RE = re.compile(r'<([^>]+)>\s*;\s*rel="next"')


def hf_get(url: str) -> tuple[Any, str | None]:
    """The real {@link HfGet} (`urllib`). A network, HTTP or JSON failure fails the build naming
    the URL and the offline alternative."""
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=HF_TIMEOUT_SECONDS) as response:
            body = json.load(response)
            link = response.headers.get("Link")
    # `http.client.HTTPException` (e.g. `IncompleteRead` — the connection dropped while the body
    # was read) is not an `OSError`; `ValueError` covers the JSON decode.
    except (OSError, ValueError, http.client.HTTPException) as cause:
        raise DistError(
            f"{RESOLVE_REFS_FLAG} hf could not read {url}: {cause} — it needs the Hugging Face"
            f" API; offline, pin the revision with {REF_REVISION_FLAG} NAME=SHA"
        ) from cause
    match = _NEXT_PAGE_RE.search(link or "")
    return body, match.group(1) if match else None


@dataclass(frozen=True)
class HubCommit:
    """The main of one referenced repository as the HF API reported it: its commit and the files of
    that commit (path → tree entry; `oid` = the git blob id, `lfs.oid` = the sha256 of an LFS
    file)."""

    repo: str
    revision: str
    files: Mapping[str, Mapping[str, Any]]


def read_hub_main(repo: str, get: HfGet) -> HubCommit:
    """Read `repo`'s main: the commit sha (`/api/models/<repo>/revision/main`), then the files of
    **that sha** (`/api/models/<repo>/tree/<sha>?recursive=true`, every page) — so the listing
    cannot drift from the revision written into the manifest."""
    body, _ = get(f"{HF_ENDPOINT}/api/models/{repo}/revision/main")
    revision = body.get("sha") if isinstance(body, dict) else None
    if not isinstance(revision, str) or not REVISION_RE.fullmatch(revision):
        raise DistError(
            f"{RESOLVE_REFS_FLAG} hf: the HF API gave no commit sha for the main of {repo}"
            f" (got {revision!r})"
        )
    files: dict[str, Mapping[str, Any]] = {}
    url: str | None = f"{HF_ENDPOINT}/api/models/{repo}/tree/{revision}?recursive=true"
    while url is not None:
        page, url = get(url)
        if not isinstance(page, list):
            raise DistError(
                f"{RESOLVE_REFS_FLAG} hf: the HF API's tree of {repo}@{revision} is not a list"
            )
        for entry in page:
            if (
                isinstance(entry, dict)
                and entry.get("type") == "file"
                and isinstance(entry.get("path"), str)
            ):
                files[entry["path"]] = entry
    return HubCommit(repo=repo, revision=revision, files=files)


def git_blob_id(path: Path) -> str:
    """The git object id of a file stored without LFS (sha1 over `blob <size>\\0` + the bytes) —
    what the HF tree lists as its `oid`."""
    digest = hashlib.sha1(f"blob {path.stat().st_size}\0".encode(), usedforsecurity=False)
    with path.open("rb") as stream:
        while chunk := stream.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def confirm_on_hub(
    commits: Mapping[str, HubCommit],
    components: ExternalComponents,
    refs: Sequence[Mapping[str, Any]],
) -> None:
    """`karume.dist.ReferenceCheck` against the Hub: every file the build borrows from
    `components.repo` is in the pinned commit, with the bytes the core reconciled in the local
    mirror (an LFS file: `lfs.oid` = its sha256; a file stored without LFS: its git blob id, taken
    from the local mirror's file).

    MUST: every borrowed file is checked, not a sample — a referenced mirror rebuilt locally and
    not uploaded differs in some parts only, and a renamed path is missing for that one file.
    """
    commit = commits[components.repo]
    problems: list[str] = []
    for ref in refs:
        path = str(ref["path"])
        entry = commit.files.get(path)
        if entry is None:
            problems.append(f"{path}: not in the commit")
            continue
        lfs = entry.get("lfs")
        if isinstance(lfs, dict):
            if lfs.get("oid") != ref["sha256"]:
                problems.append(f"{path}: LFS oid {lfs.get('oid')} ≠ local sha256 {ref['sha256']}")
            continue
        local = git_blob_id(components.dist / path)
        if entry.get("oid") != local:
            problems.append(f"{path}: git blob {entry.get('oid')} ≠ local {local}")
    if problems:
        listing = "\n  ".join(problems)
        raise DistError(
            f"{components.repo}@{commit.revision} does not hold the bytes this build references"
            f" ({len(problems)} of {len(refs)} files):\n  {listing}\n— upload the referenced"
            " repository first, or rebuild against the mirror its main was uploaded from"
            " (docs/release-runbook.md §0)"
        )


def reference_pins(
    references: Sequence[CrossRepoReference],
    explicit: Mapping[str, str],
    *,
    resolve: str | None,
    allow_placeholder: bool,
    get: HfGet,
) -> tuple[dict[str, str], ReferenceCheck | None]:
    """The revision source of a build's cross-repository references (module doc).

    Returns the revisions this driver adds to the core's arguments (`--ref-revision NAME=SHA` —
    empty when the caller pinned them explicitly, the core reading those itself) and the check the
    core runs before writing ({@link confirm_on_hub} for `--resolve-refs hf`).

    MUST: exactly one source per build, settled before a byte is written. Combining two would let
    one silently override the other; none, for a build that declares references, would assemble a
    self-contained distribution nobody asked for; a flag for a build that references nothing means
    the build is not the one the caller had in mind. The placeholder is reachable only through its
    explicit flag (module doc).
    """
    sources = [
        flag
        for flag, given in (
            (REF_REVISION_FLAG, bool(explicit)),
            (RESOLVE_REFS_FLAG, resolve is not None),
            (ALLOW_PLACEHOLDER_FLAG, allow_placeholder),
        )
        if given
    ]
    if len(sources) > 1:
        raise DistError(
            f"{' and '.join(sources)} exclude each other — a build takes the revisions of its"
            " cross-repository references from one source"
        )
    # After the exclusivity check, so that its advice (use the flag instead) is never given to a
    # caller who already passed the flag.
    placeholder = sorted(
        name for name, revision in explicit.items() if revision == PLACEHOLDER_REVISION
    )
    if placeholder:
        raise DistError(
            f"{REF_REVISION_FLAG} pins {placeholder} at the placeholder ({PLACEHOLDER_REVISION})"
            f" — a published distribution is built with the real commit ({RESOLVE_REFS_FLAG} hf,"
            f" or the SHA of the referenced main); a development mirror says so with"
            f" {ALLOW_PLACEHOLDER_FLAG} instead (docs/release-runbook.md §0)"
        )
    repos = {reference.repo_name: reference.repo for reference in references}
    if not repos:
        if sources and sources[0] != REF_REVISION_FLAG:
            raise DistError(
                f"{sources[0]} was given, but the models built declare no cross-repository"
                " reference"
            )
        # An explicit revision for an undeclared repository fails in the core (one place).
        return {}, None
    if not sources:
        raise DistError(
            f"The models built reference {sorted(repos)} across repositories — pin the revision"
            f" with {RESOLVE_REFS_FLAG} hf (the referenced main, checked file by file on the"
            f" Hub), {REF_REVISION_FLAG} NAME=SHA, or {ALLOW_PLACEHOLDER_FLAG} for a development"
            " mirror"
        )
    if explicit:
        return {}, None
    if allow_placeholder:
        return {name: PLACEHOLDER_REVISION for name in repos}, None
    commits = {repo: read_hub_main(repo, get) for repo in sorted(set(repos.values()))}
    return (
        {name: commits[repo].revision for name, repo in repos.items()},
        partial(confirm_on_hub, commits),
    )


#: Parent of the default location of local experimental mirrors from an intake
#: (`<LOCAL_DIST_ROOT>/<name>/` — outside `models/`; ADR 0122 decision 5).
LOCAL_DIST_ROOT = MISC_ROOT / "local-dist"

#: The argument that builds a local experimental mirror from an intake (`inputs/umt5/<name>/`)
#: (`--pipeline umt5` only).
INTAKE_FLAG = "--intake"

#: The pipeline that accepts an intake (a key of `dist.PIPELINES`).
INTAKE_PIPELINE = "umt5"


def assert_outside_distribution_root(out_dir: Path) -> None:
    """Fail if the output is under `models/` ({@link DIST_ROOT}) (a model from an intake — ADR
    0122 decision 5).

    The decision is the same single function as the intake's (`--out` of `wan.umt5_intake`)
    (`wan.umt5_intake.assert_outside_distribution_root` — comparing real paths).
    """
    try:
        intake_outside_distribution_root(out_dir, root=DIST_ROOT)
    except Umt5IntakeError as cause:
        raise DistError(
            f"output: {cause} (the default for a local experimental mirror is"
            f" {LOCAL_DIST_ROOT}/<name>)"
        ) from cause


def intake_pipelines(
    args: argparse.Namespace,
) -> tuple[Mapping[str, Pipeline], Callable[[Pipeline, Sequence[str]], Path]]:
    """The pipeline table and the default output of an `--intake` run (both checked before a
    byte is written)."""
    if args.pipeline != INTAKE_PIPELINE:
        raise DistError(f"{INTAKE_FLAG} applies only to --pipeline {INTAKE_PIPELINE}")
    try:
        intake = load_intake(args.intake)
    except Umt5IntakeError as cause:
        raise DistError(str(cause)) from cause
    if args.models is not None and args.models != [intake.name]:
        raise DistError(
            f"--model {args.models} differs from the intake name {intake.name!r} — a model from"
            " an intake is the only one in its repository (never mixed with the base"
            " karume-umt5-xxl)"
        )
    out_dir = args.out if args.out is not None else LOCAL_DIST_ROOT / intake.name
    assert_outside_distribution_root(out_dir)
    pipeline = umt5_distribution.intake_pipeline(
        intake, allow_undeclared_license=args.allow_undeclared_license
    )
    pipelines = {**PIPELINES, INTAKE_PIPELINE: pipeline}
    return pipelines, lambda _pipeline, _models: LOCAL_DIST_ROOT / intake.name


#: The driver-only options that take a value (removed with it before the core reads the arguments).
_DRIVER_VALUE_FLAGS = (INTAKE_FLAG, RESOLVE_REFS_FLAG)


def core_arguments(arguments: Sequence[str]) -> list[str]:
    """The arguments passed to the core: without the ones only the driver reads
    ({@link ALLOW_PLACEHOLDER_FLAG}, {@link ALLOW_UNDECLARED_LICENSE_FLAG}, and
    {@link INTAKE_FLAG} / {@link RESOLVE_REFS_FLAG} with their values)."""
    remaining: list[str] = []
    skip_value = False
    for argument in arguments:
        if skip_value:
            skip_value = False
            continue
        if argument in (ALLOW_PLACEHOLDER_FLAG, ALLOW_UNDECLARED_LICENSE_FLAG):
            continue
        if argument in _DRIVER_VALUE_FLAGS:
            skip_value = True
            continue
        if argument.startswith(tuple(f"{flag}=" for flag in _DRIVER_VALUE_FLAGS)):
            continue
        remaining.append(argument)
    return remaining


def default_out_dir(pipeline: Pipeline, models: Sequence[str]) -> Path:
    """The output when `--out` is omitted (`models/<repository name>/` = one directory, one HF
    repository).

    The repository name comes from the Pipeline's declaration (`Pipeline.repo_name` — the
    family's repository name; sbv2 declares `karume-sbv2-jvnv` too). With several models it
    fails and asks for an explicit `--out` — this function reads the declaration of one model
    only and does not check that the declarations of the listed models agree.
    """
    if len(models) != 1:
        raise DistError(
            f"the repository name cannot be derived when building {len(models)} models — give"
            " the output with --out"
        )
    return DIST_ROOT / pipeline.repo_name(models[0])


def build_driver_parser() -> argparse.ArgumentParser:
    """The core's parser plus the driver-only options ({@link ALLOW_PLACEHOLDER_FLAG},
    {@link RESOLVE_REFS_FLAG} and the intake options) — one parser, so the driver and the core
    read the same arguments.

    Abbreviations are not accepted (`allow_abbrev`). The driver-only options are removed by exact
    spelling before the core reads the arguments, so an abbreviation (`--allow-placeholder` and the
    like) accepted here only would reach the core as an unknown argument — they are spelled out,
    and both read the arguments one way.
    """
    parser = build_parser(
        PIPELINES, DEFAULT_PIPELINE, default_series=SERIES_ROOT, has_default_out_dir=True
    )
    parser.allow_abbrev = False
    parser.add_argument(
        ALLOW_PLACEHOLDER_FLAG,
        dest="allow_placeholder_ref",
        action="store_true",
        help=f"pin every cross-repository reference at the placeholder ({PLACEHOLDER_REVISION})"
        " — development mirrors only, while the referenced repository is unpublished (never for a"
        " distribution that is uploaded)",
    )
    parser.add_argument(
        RESOLVE_REFS_FLAG,
        dest="resolve_refs",
        choices=("hf",),
        default=None,
        help="hf: pin every cross-repository reference at the referenced repository's main on"
        f" {HF_ENDPOINT}, and require every borrowed file to be in that commit with the local"
        " bytes (read-only network access)",
    )
    parser.add_argument(
        INTAKE_FLAG,
        dest="intake",
        type=Path,
        default=None,
        help=f"--pipeline {INTAKE_PIPELINE}: build a local experimental mirror from an intake"
        f" (inputs/umt5/<name>/) (default output {LOCAL_DIST_ROOT}/<name> — it cannot be written"
        " under models/)",
    )
    parser.add_argument(
        ALLOW_UNDECLARED_LICENSE_FLAG,
        dest="allow_undeclared_license",
        action="store_true",
        help=f"{INTAKE_FLAG}: allow building an experimental mirror from an intake without a"
        " declared license (ADR 0122 decision 6)",
    )
    return parser


def referenced_dist(name: str) -> Path:
    """The local mirror of the referenced repository `name` (`models/<name>/` — the same place
    {@link default_out_dir} builds it into; read at call time like that one)."""
    return DIST_ROOT / name


def main(argv: Sequence[str] | None = None, *, get: HfGet = hf_get) -> None:
    """Assemble one distribution. `get` is the HF API access of {@link RESOLVE_REFS_FLAG}
    (injected by the tests only — they read no network)."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    args = build_driver_parser().parse_args(arguments)
    pipelines: Mapping[str, Pipeline] = PIPELINES
    out_dir_hook: Callable[[Pipeline, Sequence[str]], Path] = default_out_dir
    if args.intake is not None:
        pipelines, out_dir_hook = intake_pipelines(args)
    elif args.allow_undeclared_license:
        raise DistError(f"{ALLOW_UNDECLARED_LICENSE_FLAG} applies only together with {INTAKE_FLAG}")
    # Settled before a byte is written (the core's assembly replaces out_dir wholesale through
    # staging) and before the plans read several GB of series.
    pipeline = pipelines[args.pipeline]
    revisions, check = reference_pins(
        declared_references(pipeline, selected_models(pipeline, args.models)),
        parse_reference_revisions(args.ref_revisions),
        resolve=args.resolve_refs,
        allow_placeholder=args.allow_placeholder_ref,
        get=get,
    )
    dist_main(
        [
            *core_arguments(arguments),
            *(f"{REF_REVISION_FLAG}={name}={revision}" for name, revision in revisions.items()),
        ],
        pipelines=pipelines,
        default_pipeline=DEFAULT_PIPELINE,
        default_out_dir=out_dir_hook,
        default_series=SERIES_ROOT,
        referenced_dist=referenced_dist,
        check_references=check,
    )


if __name__ == "__main__":
    main()
