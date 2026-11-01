"""リポジトリ用の dist ドライバ — core の組み立てエンジンに recipe 側 pipeline を合成する。

`karume.dist` が持つ {@link karume.dist.PIPELINES} は「core wheel だけで組める pipeline」で
あって全量ではない（ADR 0065 決定 2 — モデル別 recipe は wheel の外）。family の移行が全部
終わった今、core の表は**空**なので、受理集合の正本はこの辞書だけになった。

    uv run python dist.py                                  # 既定 = anima（公式 5 変種 — ADR 0087）
    uv run python dist.py --pipeline anima-extra \\
        --model anima-wai-v1.0 --model anima-copycat-20260610 \\
        --out ../../models/karume-anima-extra \\
        --resolve-refs hf                                  # fine-tunes (reference karume-anima)
    uv run python dist.py --pipeline irodori               # v4-small (self-contained)
    uv run python dist.py --pipeline irodori --model v4.1-small \\
        --resolve-refs hf                                  # references karume-irodori-v4-small
    uv run python dist.py --pipeline sbv2-fn               # FN 系（HF 公開は保留）
    uv run python dist.py --pipeline sbv2 \\
        --model F1 --model F2 --out ../../models/karume-sbv2-jvnv
    uv run python dist.py --pipeline siglip2 \\
        --model base --model so400m --out ../../models/karume-siglip2
    uv run python dist.py --pipeline lucida                # BiRefNet_HR の派生（別リポ）
    uv run python dist.py --pipeline umt5                  # umT5-XXL encoder（Wan の参照先）
    uv run python dist.py --pipeline wan --allow-placeholder-ref
                                                           # Wan2.1 T2V 1.3B (not published)
    uv run python dist.py --pipeline wan-ti2v --resolve-refs hf
                                                           # Wan2.2 TI2V 5B (references umT5)
    uv run python dist.py --pipeline moss-image            # MOSS-Image-SFT (not published)

置き場の既定（`--series` / `--out`）もここが渡す — リポの `outputs/series/` と `models/` は
repo topology で、core は綴りを持たない（ADR 0065 Consequences・`karume.dist` の同 MUST）。

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

## 取り込み由来の手元の実験用ミラー（ADR 0122 決定 5 / 6）

第三者の umT5 互換 encoder の取り込み（`wan.umt5_intake`）から書いた系列は、`--intake` を付けた
`--pipeline umt5` でだけ組める。リポ名・ルートのファイル・カードは取り込みの記録 `intake.json` から
導き（`wan.umt5_distribution.intake_pipeline`）、既定の出力先は {@link LOCAL_DIST_ROOT} の下。
**ライセンスの宣言の有無によらず**、出力先が `models/`（そのまま上げられる配布形の席）の下なら
1 バイトも書く前に落とす（{@link assert_outside_distribution_root} — 実 path どうしで比べる）。
ライセンス未宣言の取り込みは {@link ALLOW_UNDECLARED_LICENSE_FLAG} の明示が要る（計画の時点で
落とす）:

    uv run python dist.py --pipeline umt5 --intake ../../inputs/umt5/<名前> \
        --allow-undeclared-license        # 未宣言のときだけ — out は outputs/misc/local-dist/<名前>

NOTE: `**CORE_PIPELINES` の展開は残す — core が「表を受け取る側」であって「表を持たない側」で
はないことは変わっておらず、core wheel だけで組める pipeline が将来生えたら黙って合流する。
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

#: 受理集合の全量。並びは `--help` の並びでもあるので、既定を先頭に置く。
PIPELINES: Mapping[str, Pipeline] = {
    "anima": anima_distribution.OFFICIAL_PIPELINE,
    "anima-extra": anima_distribution.EXTRA_PIPELINE,
    "sbv2": sbv2_distribution.PIPELINE,
    # 声のファミリーはライセンスが違う（JVNV = CC BY-SA 4.0・FN = Booth の頒布条件）ので別席。
    "sbv2-fn": sbv2_distribution.FN_PIPELINE,
    "irodori": irodori_distribution.PIPELINE,
    "siglip2": siglip2_distribution.PIPELINE,
    "birefnet": birefnet_distribution.PIPELINE,
    # 派生は別リポ（ADR 0092 決定 1）— リポ直下の著作権表示が違うので Pipeline も別席。
    "lucida": birefnet_distribution.LUCIDA_PIPELINE,
    "depth-anything": depth_anything_distribution.PIPELINE,
    "vowel-detector": vowel_detector_distribution.PIPELINE,
    "gemma4": gemma4_distribution.PIPELINE,
    "gemma4-qat": gemma4_qat_distribution.PIPELINE,
    "wan": wan_distribution.PIPELINE,
    # Wan の世代は別リポ（ADR 0092 決定 2 — `karume-wan2.2`）なので Pipeline も別席。
    "wan-ti2v": wan_distribution.TI2V_PIPELINE,
    # Wan の text_encoder の参照先（ADR 0119 追記 D — 部品だけの別リポ・読む TS の家族は無い）。
    "umt5": umt5_distribution.PIPELINE,
    # MOSS-Image-SFT (ADR 0138 addendum of 2026-11-01 — self-contained: the Qwen3.5 text tower is
    # the checkpoint's own, stored in this repository).
    "moss-image": moss_distribution.PIPELINE,
    **CORE_PIPELINES,
}

#: 旧 `karume dist`（引数なし）の UX をドライバ側で維持する。
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


#: 取り込み由来の手元の実験用ミラーの既定の置き場の親（`<LOCAL_DIST_ROOT>/<名前>/` — `models/`
#: の外。ADR 0122 決定 5）。
LOCAL_DIST_ROOT = MISC_ROOT / "local-dist"

#: 取り込み（`inputs/umt5/<名前>/`）から手元の実験用ミラーを組む引数（`--pipeline umt5` だけ）。
INTAKE_FLAG = "--intake"

#: 取り込みを受ける pipeline（`dist.PIPELINES` のキー）。
INTAKE_PIPELINE = "umt5"


def assert_outside_distribution_root(out_dir: Path) -> None:
    """出力先が `models/`（{@link DIST_ROOT}）の下なら落とす（取り込み由来のモデル — ADR 0122
    決定 5）。

    判定は取り込み（`wan.umt5_intake` の `--out`）と同じ 1 本
    （`wan.umt5_intake.assert_outside_distribution_root` — 実 path どうしで比べる）。
    """
    try:
        intake_outside_distribution_root(out_dir, root=DIST_ROOT)
    except Umt5IntakeError as cause:
        raise DistError(
            f"出力先: {cause}（手元の実験用ミラーの既定は {LOCAL_DIST_ROOT}/<名前>）"
        ) from cause


def intake_pipelines(
    args: argparse.Namespace,
) -> tuple[Mapping[str, Pipeline], Callable[[Pipeline, Sequence[str]], Path]]:
    """`--intake` の回の pipeline の表と既定の出力先（どちらも 1 バイトも書く前に検査する）。"""
    if args.pipeline != INTAKE_PIPELINE:
        raise DistError(f"{INTAKE_FLAG} は --pipeline {INTAKE_PIPELINE} にだけ効く")
    try:
        intake = load_intake(args.intake)
    except Umt5IntakeError as cause:
        raise DistError(str(cause)) from cause
    if args.models is not None and args.models != [intake.name]:
        raise DistError(
            f"--model {args.models} が取り込みの名前 {intake.name!r} と違う — 取り込み由来の"
            "モデルは 1 リポに 1 つだけ（本家の karume-umt5-xxl と混ぜない）"
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
    """`--out` 省略時の出力先（`models/<リポ名>/` = 1 ディレクトリ 1 HF リポ）。

    リポ名は Pipeline の宣言（`Pipeline.repo_name` — ファミリーのリポ名。sbv2 も
    `karume-sbv2-jvnv` を宣言する）から引く。複数モデルでは `--out` の明示を求めて落とす —
    この関数は 1 モデルの宣言だけを引き、並べたモデルの宣言が揃うかの突合を持たない。
    """
    if len(models) != 1:
        raise DistError(
            f"モデルを {len(models)} 個組む場合はリポ名を導出できない — --out で出力先を指定する"
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
        help=f"--pipeline {INTAKE_PIPELINE}: 取り込み（inputs/umt5/<名前>/）から手元の実験用"
        f"ミラーを組む（既定の出力先 {LOCAL_DIST_ROOT}/<名前> — models/ の下へは書けない）",
    )
    parser.add_argument(
        ALLOW_UNDECLARED_LICENSE_FLAG,
        dest="allow_undeclared_license",
        action="store_true",
        help=f"{INTAKE_FLAG}: ライセンス未宣言の取り込みから実験用ミラーを組むことを許す（ADR 0122"
        " 決定 6）",
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
        raise DistError(f"{ALLOW_UNDECLARED_LICENSE_FLAG} は {INTAKE_FLAG} と組むときだけ効く")
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
