"""Read the provenance of an upstream checkpoint **from the local files** (the values written into
the container's `provenance`, and the check against them).

The container's `provenance` (container-v1 §2.3) is there so that the source can be traced even
when a single `krm` is taken away as one file. Writing a constant into it would make the container
claim the same source even when `--model-dir` points at another checkpoint (Base weights claiming
Small's `apache-2.0` — repository-wide review E-RC6-1 of 2026-09-24). So the values are derived
from the files in `--model-dir`:

- **revision** is the first line of the record `hf download <repo> --local-dir <dir>` leaves at
  `<dir>/.cache/huggingface/download/<file>.metadata` (the commit SHA — the second line is the
  etag, the third the fetch time).
- **license identifier** is `license:` in the front matter of the `README.md` fetched with it
  (for `other`, `license_name:` — the spelling HF uses for a custom license).

MUST: when the record is missing or unreadable, **fail loudly** ({@link UpstreamProvenanceError}).
Both "leave it out since it is unknown" and "claim a default" would falsify the container's
provenance.

The assembly side checks the written provenance against the attribution table
({@link assert_upstream_provenance}) — the one function that closes the distribution check on
"the provenance of the container's contents" rather than on "a table of model names".
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from pathlib import Path

from _shared.container_read import read_provenance
from karume.container import Provenance
from karume.dist import DistError
from karume.modelcard import HF_OWNER, CardMetadata, Upstream

#: Where `hf download --local-dir` puts its fetch records (relative to `<model_dir>`).
SNAPSHOT_RECORD_DIR = Path(".cache") / "huggingface" / "download"

#: The file whose record gives the revision by default (one that nearly every HF checkpoint has).
SNAPSHOT_RECORD_FILE = "config.json"

#: An HF commit SHA (40 lower-case hex digits).
_COMMIT_SHA = re.compile(r"[0-9a-f]{40}")

#: One top-level front-matter key line (`license: apache-2.0` / `license: "mit"`).
_FRONT_MATTER_KEY = re.compile(r"^(?P<key>[A-Za-z_][A-Za-z0-9_]*):\s*(?P<value>.*?)\s*$")


class UpstreamProvenanceError(ValueError):
    """The provenance to write into the container (revision / license identifier) cannot be
    determined.

    The repository's convention is an `Error` subclass (`DistError` / `ContainerReadError`). With
    a bare `ValueError`, the caller's `except` could not tell this **provenance check** from "a
    bad argument" in general.
    """


def hub_cache() -> Path:
    """The Hugging Face hub cache, as huggingface_hub itself resolves it
    (`huggingface_hub.constants.HF_HUB_CACHE` — read once, when huggingface_hub is imported).

    MUST: imported here, not at module scope — the default sync of this workspace does not
    install huggingface_hub (it comes with transformers, which the gemma4 recipes run with, and
    with the wan / moss / sbv2 groups), and every recipe imports this module.
    """
    from huggingface_hub import constants

    return Path(constants.HF_HUB_CACHE)


def hub_snapshot_dir(repo: str, revision: str) -> Path:
    """The local hub cache's snapshot of `repo` at commit `revision`.

    A path only (nothing is read or fetched): the readers of the files fail loudly when the
    snapshot is not there. MUST: `revision` is a commit SHA — a branch or a tag is not a pin, and
    the cache names a snapshot directory by the commit it resolved.
    """
    if _COMMIT_SHA.fullmatch(revision) is None:
        raise UpstreamProvenanceError(
            f"{repo}: revision {revision!r} is not a commit SHA (40 lower-case hex digits)"
        )
    return hub_cache() / f"models--{repo.replace('/', '--')}" / "snapshots" / revision


def snapshot_revision(model_dir: Path, file: str = SNAPSHOT_RECORD_FILE) -> str:
    """Read the commit SHA of the local checkpoint from the `hf download --local-dir` record."""
    record = model_dir / SNAPSHOT_RECORD_DIR / f"{file}.metadata"
    if not record.is_file():
        raise UpstreamProvenanceError(
            f"{record} is missing — only a checkpoint fetched with `hf download <repo>"
            f" --local-dir {model_dir}` can claim an upstream revision"
        )
    lines = record.read_text(encoding="utf-8").splitlines()
    revision = lines[0].strip() if lines else ""
    if _COMMIT_SHA.fullmatch(revision) is None:
        raise UpstreamProvenanceError(
            f"{record}: the first line {revision!r} is not a commit SHA (40 hex digits)"
        )
    return revision


def _front_matter(readme: Path) -> dict[str, str]:
    """Pick up only the **top-level scalars** of a `README.md`'s front matter (nesting and lists
    are not read)."""
    lines = readme.read_text(encoding="utf-8").splitlines()
    if not lines or lines[0].strip() != "---":
        raise UpstreamProvenanceError(f"{readme} does not start with front matter (a '---' line)")
    fields: dict[str, str] = {}
    for line in lines[1:]:
        if line.strip() == "---":
            return fields
        matched = _FRONT_MATTER_KEY.match(line)
        if matched is not None and matched["value"]:
            fields[matched["key"]] = matched["value"].strip("\"'")
    raise UpstreamProvenanceError(f"{readme}: the front matter is not closed (no second '---')")


def snapshot_license(model_dir: Path) -> str:
    """Read the license identifier from the front matter of the local checkpoint's `README.md`.

    `license: other` is HF's spelling for "a custom license without an identifier", and the
    identifier itself is in `license_name:` (the `license` of container-v1 §2.3 is the identifier's
    slot — with `other`, the license cannot be identified from a container taken away).
    """
    readme = model_dir / "README.md"
    if not readme.is_file():
        raise UpstreamProvenanceError(
            f"{readme} is missing — the upstream license identifier cannot be read"
        )
    fields = _front_matter(readme)
    license_id = fields.get("license")
    if license_id is None:
        raise UpstreamProvenanceError(f"{readme}: the front matter has no 'license'")
    return _license_identifier(license_id, fields.get("license_name"), str(readme))


def _license_identifier(license_id: str, license_name: str | None, where: str) -> str:
    """HF's `license` / `license_name` pair → the license identifier written into the
    container."""
    if license_id != "other":
        return license_id
    if license_name is None:
        raise UpstreamProvenanceError(
            f"{where} has 'license: other' and no 'license_name' — no identifier can be determined"
        )
    return license_name


def card_license_identifier(metadata: CardMetadata, where: str) -> str:
    """Take the license identifier to write into the container from the card's front matter
    (values confirmed upstream by hand).

    For families with no local upstream README (anima, read directly from the HF cache; the sbv2
    FN models from Booth). `other` is not written for the same reason as in
    {@link snapshot_license} — `other` is HF's spelling for "a custom license without an
    identifier", and the license cannot be identified from a container taken away
    (repository-wide review W-RC3-6 of 2026-09-24).
    """
    return _license_identifier(metadata.license, metadata.license_name, where)


def snapshot_provenance(model_dir: Path, *, notice: str | None) -> Provenance:
    """Build the provenance to write into the container from the files in `--model-dir`
    (license identifier + revision)."""
    return Provenance(
        license=snapshot_license(model_dir),
        notice=notice,
        upstream_revision=snapshot_revision(model_dir),
    )


def assert_upstream_provenance(container: Path, *, license: str, revision: str | None) -> None:
    """Check the provenance a candidate container claims against the attribution table (and the
    fetched revision).

    `revision` is the expected upstream revision — derived from the local checkpoint's fetch
    record (siglip2, depth_anything) or fixed in the attribution table (wan, birefnet). With
    `None`, only "it claims a revision" is checked.

    MUST: close the check on the container's provenance. Closing it on the table of model names
    alone would let a container built from another checkpoint, placed at the series path, pass
    every storage, I/O-shape and preprocessing comparison, giving a distribution whose weights,
    under a different license, claim the attribution table's license (repository-wide review
    E-RC6-1 of 2026-09-24).
    """
    provenance = read_provenance(container)
    if provenance.license != license:
        raise DistError(
            f"{container}: provenance.license is {provenance.license!r}, not the attribution"
            f" table's {license!r} — a container built from a checkpoint the table does not list"
            " is not distributed"
        )
    if provenance.upstream_revision is None:
        raise DistError(
            f"{container} has no provenance.upstreamRevision — a container that cannot claim"
            " an upstream revision is not distributed (rebuild it with the script)"
        )
    if revision is not None and provenance.upstream_revision != revision:
        raise DistError(
            f"{container}: provenance.upstreamRevision is {provenance.upstream_revision}, the"
            f" expected revision is {revision} — a container built from a different revision"
        )


#: HF account (the owner part of a `base_model` repository ID) → who the cards name as its
#: publisher in the house non-endorsement sentence ({@link karume.modelcard.introduction}). Spelled
#: as this repository already names them where it does — in a licence's notice, a copyright line
#: or a card's attribution section; an account with no such name is spelled as the account itself.
#:
#: MUST: an account missing here fails the card ({@link hf_upstreams}) — adding an upstream means
#: stating who published it, never a placeholder.
UPSTREAM_AUTHORS: Mapping[str, str] = {
    # anima — the licensor in the CircleStone licence's attribution notice.
    "circlestone-labs": "CircleStone Labs LLC",
    # birefnet / lucida — the copyright lines of the bundled MIT licence.
    "ZhengPeng7": "ZhengPeng",
    "egeorcun": "egeorcun",
    "depth-anything": "depth-anything",
    # gemma4 / gemma4_qat / siglip2 / umt5 — Gemma's licensor and the publisher of the others.
    "google": "Google",
    # irodori — the copyright lines of the bundled MIT licence.
    "Aratako": "Aratako",
    "sbintuitions": "SB Intuitions",
    # moss — the card's attribution section and the Qwen licence's copyright line.
    "OpenMOSS-Team": "OpenMOSS",
    "Qwen": "Alibaba Cloud",
    "black-forest-labs": "Black Forest Labs",
    # sbv2 — the voices' and the text encoder's publishers.
    "litagin": "litagin",
    "ku-nlp": "ku-nlp",
    "rufflet17": "rufflet17",
    # wan — the card's attribution section ("released by Wan-AI").
    "Wan-AI": "Wan-AI",
}


def hf_upstreams(repos: Sequence[str]) -> tuple[Upstream, ...]:
    """The house sentence's upstreams for HF repositories (a card's `base_model`, typically): each
    repository ID with its account's publisher from {@link UPSTREAM_AUTHORS}.

    MUST: a repository of our own account (`karume.modelcard.HF_OWNER`) is refused — a conversion
    of our own release is not unofficial, and the sentence would disclaim ourselves.
    """
    upstreams: list[Upstream] = []
    for repo in repos:
        owner, _, name = repo.partition("/")
        if not owner or not name:
            raise ValueError(f"{repo!r} is not an HF repository ID (<owner>/<name>)")
        if owner == HF_OWNER:
            raise ValueError(f"{repo} is our own release — it is not an unofficial conversion")
        author = UPSTREAM_AUTHORS.get(owner)
        if author is None:
            raise ValueError(
                f"{repo}: the publisher of HF account '{owner}' is not in UPSTREAM_AUTHORS"
                " — a card that cannot say whose work it converts is not drawn"
            )
        upstreams.append(Upstream(repo, author))
    return tuple(upstreams)
