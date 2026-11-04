"""Imports a civitai checkpoint by AIR / URL (the weights + machine provenance into `inputs/`).

    uv run python -m anima.civitai --air "urn:air:anima:checkpoint:civitai:2544636@2983680"
    uv run python -m anima.civitai --url \
        "https://civitai.com/models/2544636/wai-anima?modelVersionId=2983680"
    uv run python -m anima.civitai --url "https://civitai.com/models/2544636"  # lists versions

Its responsibility is one step: **fetch, verify and record**. Rebuilding into the diffusers layout
belongs to {@link anima.single_file}, and the attribution in the distribution card to
{@link anima.card}.

**It does not judge the licence** — it only copies the API's 4 permission fields and the
description body (HTML) into `civitai.json`; whether the import is allowed is decided by the person
importing, after reading the original (ruling of 2026-09-01 — there is no confirmation flag). The
record of a human review goes into `license-review.md` in the same directory, which this command
never touches (mixing human additions into the machine-written `civitai.json` would lose one side
on every re-fetch).

MUST: do not import torch / diffusers / huggingface_hub (the same reason as the NOTE of
`anima.single_file` — it must work in an environment where the dependency group is not synced).
This only downloads and records one weight file; the standard library is enough.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

from _shared.paths import INPUTS_ROOT

#: Endpoint for metadata (readable without authentication).
API_ROOT = "https://civitai.com/api/v1"

#: Where the download token goes (issued under API Keys at https://civitai.com/user/account).
TOKEN_ENV = "CIVITAI_API_TOKEN"

#: Parent of the import destination (`inputs/anima/civitai-<versionId>/` — docs/assets-layout.md).
DEFAULT_OUT = INPUTS_ROOT / "anima"

#: The import record owned by the machine. Human review records go to `license-review.md`, not
#: here.
PROVENANCE_FILE = "civitai.json"

MIB = 1024 * 1024

#: Read / write chunk size and the progress interval (no silent wait on a 4GB-class checkpoint).
CHUNK_BYTES = 8 * MIB
PROGRESS_BYTES = 256 * MIB

#: Syntax of an AIR (civitai's asset URN). `ecosystem` / `type` are **not validated** (only recorded
#: — when upstream extends the vocabulary, a check here must not be what blocks the import).
AIR_PATTERN = re.compile(
    r"urn:air:(?P<ecosystem>[^:]+):(?P<type>[^:]+):(?P<source>[^:@]+):(?P<model>\d+)"
    r"(?:@(?P<version>\d+))?"
)

#: Path of a model page (`/models/<id>` + an optional slug). The version is `modelVersionId` in the
#: query.
MODEL_PATH_PATTERN = re.compile(r"/models/(?P<model>\d+)(?:/[^/]*)?/?")

#: The accepted set of distribution names (ADR 0077 — a **verbatim copy** of `MODEL_NAME_RE` in
#: `karume.dist`).
#:
#: MUST: it is a copy because of the MUST above (do not import torch) — importing `karume.dist`
#: pulls in torch through `karume.convert` via `karume/__init__.py` (measured 2026-09-05), so the
#: source of truth cannot be called directly. When the source of truth adds a rule, fix this on the
#: same day. Like the source of truth, match with `.fullmatch` (Python's `$` also matches right
#: before a trailing newline).
MODEL_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_-][A-Za-z0-9._-]*$")

#: Reserved name not allowed as a distribution name (`SHARED_DIRNAME` of `karume.dist` — it
#: collides with the shared-files directory).
RESERVED_MODEL_NAME = "shared"

#: Family name dropped from derived names (`WAI-ANIMA` → `wai` — not doubled with the `anima-`
#: prefix).
FAMILY_TOKEN = "anima"


@dataclass(frozen=True)
class Air:
    """The contents of an AIR (`urn:air:{ecosystem}:{type}:civitai:{model_id}@{version_id}`)."""

    ecosystem: str
    type: str
    model_id: int
    version_id: int

    @property
    def urn(self) -> str:
        """The URN rebuilt locally (used to compare against the value the API returns)."""
        return f"urn:air:{self.ecosystem}:{self.type}:civitai:{self.model_id}@{self.version_id}"


@dataclass(frozen=True)
class Target:
    """What a model page points to. A URL may omit the version (then the version list is shown)."""

    model_id: int
    version_id: int | None


def parse_air(text: str) -> Air:
    """Parses an AIR. Rejects sources other than civitai and AIRs without a version."""
    match = AIR_PATTERN.fullmatch(text.strip())
    if match is None:
        raise SystemExit(
            f"Cannot be read as an AIR: {text}"
            " (expected: urn:air:<ecosystem>:<type>:civitai:<modelId>@<versionId>)"
        )
    source = match["source"]
    if source != "civitai":
        raise SystemExit(f"A source other than civitai is not supported: {source} ({text})")
    if match["version"] is None:
        # Without a version we would pick "the latest" on our own, but upstream distributes lines
        # with no order between them at the same time (ADR 0077). A person decides which to import.
        raise SystemExit(f"Specify the version too: {text}@<versionId>")
    return Air(
        ecosystem=match["ecosystem"],
        type=match["type"],
        model_id=int(match["model"]),
        version_id=int(match["version"]),
    )


def parse_url(text: str) -> Target:
    """Parses a model page URL (the version with `?modelVersionId=`, else the model only)."""
    parts = urlsplit(text.strip())
    if parts.netloc.removeprefix("www.") != "civitai.com":
        raise SystemExit(f"Not a civitai.com URL: {text}")
    match = MODEL_PATH_PATTERN.fullmatch(parts.path)
    if match is None:
        raise SystemExit(
            f"Cannot be read as a model page URL: {text} (expected: /models/<modelId>)"
        )
    values = parse_qs(parts.query).get("modelVersionId", [])
    if not values:
        return Target(model_id=int(match["model"]), version_id=None)
    if not values[0].isdigit():
        raise SystemExit(f"modelVersionId is not a number: {values[0]} ({text})")
    return Target(model_id=int(match["model"]), version_id=int(values[0]))


def _slug(text: str) -> str:
    """Collapses characters outside the accepted set into `-` (no runs, no separators at ends)."""
    slug = re.sub(r"[^a-z0-9._-]+", "-", text.lower())
    return re.sub(r"-+", "-", slug).strip("-._")


def _assert_name(name: str) -> str:
    """Finally checks the name is valid as a distribution name (the accepted set of ADR 0077)."""
    if MODEL_NAME_PATTERN.fullmatch(name) is None:
        raise SystemExit(
            f"The distribution name has characters that are not allowed: {name}"
            " (accepted: A-Za-z0-9._-, no leading dot)"
        )
    if name == RESERVED_MODEL_NAME:
        raise SystemExit(
            f"'{RESERVED_MODEL_NAME}' cannot be a distribution name (it collides with the"
            " shared-files directory)"
        )
    return name


def derive_model_name(model_name: str, version_name: str) -> str:
    """Derives the distribution name from upstream's naming.

    Example: `WAI-ANIMA` + `v1.0(base 1.0)` → `anima-wai-v1.0`.

    The mechanical normalisation only folds the spelling into the accepted set; it does not bring
    version names to a common format (ADR 0088 — date names and semver-like names are mixed, but
    unifying them here would make them impossible to match against the origin page; the verbatim
    names are kept in `civitai.json`). It drops only **the parenthesised remarks** (`(base 1.0)` is
    a note on the upstream page and is not needed to identify the version) and the family name on
    the model-name side.
    """
    version = _slug(re.sub(r"\([^)]*\)", "", version_name))
    tokens = [token for token in _slug(model_name).split("-") if token and token != FAMILY_TOKEN]
    model = "-".join(tokens)
    if not model or not version:
        raise SystemExit(
            f"Cannot derive a name: model name {model_name!r} / version name {version_name!r}"
        )
    return _assert_name(f"{FAMILY_TOKEN}-{model}-{version}")


def _assert_basename(name: str) -> str:
    """Confines the API's file name to one level below the destination (checked before joining).

    MUST: reject names with separators, `..` and absolute paths — joining a `Path` climbs to the
    parent on `..` and drops everything before it on an absolute path, so an anomaly in upstream
    metadata would turn straight into a write outside `civitai-<versionId>/` (only the provenance
    record would stay in the destination, and the weights would land outside).

    `:` is rejected too — on Windows `D:x.safetensors` is a drive-relative name that drops the
    joined prefix without containing a single separator (`PureWindowsPath("C:/intake") / "D:x"` →
    `D:x`).
    """
    if not name or name in {".", ".."} or "/" in name or "\\" in name or ":" in name:
        raise SystemExit(
            f"The upstream file name is not a basename: {name!r}"
            " (it could point outside the destination — check the upstream metadata)"
        )
    return name


def _masked(url: str) -> str:
    """For display (drops the token — do not spread it over the terminal and logs)."""
    return re.sub(r"([?&]token=)[^&]*", r"\1***", url)


#: The User-Agent sent. civitai explicitly answers urllib's default `Python-urllib/…` with 403
#: (measured 2026-09-01 — the same URL gives 200 with curl and with this UA). The name lets the
#: origin be traced.
USER_AGENT = "karume-export-recipes/anima.civitai"


def _open(url: str) -> Any:
    """Issues a GET. A failure stops with the status and target (never returns empty silently)."""
    request = urllib.request.Request(url, headers={"Accept": "*/*", "User-Agent": USER_AGENT})
    try:
        return urllib.request.urlopen(request)
    except urllib.error.HTTPError as error:
        raise SystemExit(f"HTTP {error.code}: {_masked(url)}") from error
    except urllib.error.URLError as error:
        raise SystemExit(f"Unreachable: {_masked(url)} ({error.reason})") from error


def _get_json(url: str) -> dict[str, Any]:
    with _open(url) as response:
        payload = response.read()
    return json.loads(payload)


def fetch_version(version_id: int) -> dict[str, Any]:
    """Version metadata (air / files / hashes / usageControl / the version description)."""
    return _get_json(f"{API_ROOT}/model-versions/{version_id}")


def fetch_model(model_id: int) -> dict[str, Any]:
    """Model metadata (name / type / the 4 permission fields / description / version list)."""
    return _get_json(f"{API_ROOT}/models/{model_id}")


def select_file(files: list[dict[str, Any]]) -> dict[str, Any]:
    """Selects the one main file. A bundled Text Encoder / VAE is skipped (the base's is shared).

    Candidates are limited to `type == "Model"` (ADR 0088 addendum) — choosing on the primary flag
    alone would make the VAE or the Text Encoder the main file in a version where upstream marked
    one of them primary.
    """
    models = [entry for entry in files if entry.get("type") == "Model"]
    primary = [entry for entry in models if entry.get("primary")]
    if len(primary) == 1:
        return primary[0]
    if not primary and len(models) == 1:
        return models[0]
    listing = [(entry.get("name"), entry.get("type")) for entry in files]
    raise SystemExit(f"Cannot identify the main file: {listing}")


def _require_token() -> str:
    token = os.environ.get(TOKEN_ENV, "").strip()
    if not token:
        raise SystemExit(
            f"Downloading needs a civitai API token — set it in env {TOKEN_ENV}"
            " (API Keys at https://civitai.com/user/account)"
        )
    return token


def _with_token(url: str, token: str) -> str:
    """The token is passed **in the query**.

    MUST: not as an `Authorization` header — the download endpoint redirects with 307 to a signed
    S3 URL, and a header carried over to the redirect target can collide with the signature and
    fail with 400. In the query, civitai resolves it on its side.
    """
    separator = "&" if "?" in url else "?"
    return f"{url}{separator}token={token}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, dest: Path, expected_sha256: str) -> None:
    """Downloads the main file into place. A no-op if it exists with the same sha (safe to re-run).

    It writes `<dest>.part` and renames it **only after the sha matches the API's value** — a file
    cut off halfway that stayed under the finished name would be read as "already fetched" by the
    next run.
    """
    expected = expected_sha256.casefold()
    if dest.exists():
        actual = _sha256(dest)
        if actual == expected:
            print(f"[civitai] already fetched (sha256 matches): {dest}", flush=True)
            return
        raise SystemExit(
            f"A different file is in place under the same name: {dest} — API {expected} /"
            f" measured {actual}"
            " (upstream replaced it, or another file was put there by hand; check before deleting)"
        )

    token = _require_token()
    part = dest.with_name(f"{dest.name}.part")
    source = _with_token(url, token)
    print(f"[civitai] GET {_masked(source)}", flush=True)
    digest = hashlib.sha256()
    written = 0
    reported = 0
    with _open(source) as response, part.open("wb") as handle:
        while chunk := response.read(CHUNK_BYTES):
            handle.write(chunk)
            digest.update(chunk)
            written += len(chunk)
            if written - reported >= PROGRESS_BYTES:
                reported = written
                print(f"[civitai] {dest.name}: {written // MIB} MiB", flush=True)

    actual = digest.hexdigest()
    if actual != expected:
        raise SystemExit(
            f"sha256 differs from the API's value: API {expected} / measured {actual}"
            f" ({part} is kept — check its contents before deleting)"
        )
    part.rename(dest)
    print(f"[civitai] {dest} ({written // MIB} MiB, sha256 matches)", flush=True)


def _resolve_air(version: dict[str, Any], requested: Air | None) -> str | None:
    """Takes the AIR the API issued (on a mismatch with the local URN, warns and uses the API's)."""
    served = version.get("air")
    if served is None:
        return requested.urn if requested is not None else None
    if requested is not None and served != requested.urn:
        print(
            f"[civitai] warning: the AIR differs from the requested one — requested"
            f" {requested.urn} / API {served}",
            flush=True,
        )
    return str(served)


def build_record(
    model: dict[str, Any],
    version: dict[str, Any],
    file: dict[str, Any],
    derived_name: str,
    air: str | None,
) -> dict[str, Any]:
    """The contents of `civitai.json` (the permissions are copied as they are, **not judged**)."""
    return {
        "air": air,
        "model_id": model["id"],
        "version_id": version["id"],
        "model_name": model["name"],
        "version_name": version["name"],
        "derived_name": derived_name,
        "base_model": version.get("baseModel"),
        "file": {
            "name": file["name"],
            "size_kb": file.get("sizeKB"),
            "sha256": file["hashes"]["SHA256"].casefold(),
            "download_url": file["downloadUrl"],
        },
        "permissions": {
            "allowNoCredit": model.get("allowNoCredit"),
            "allowCommercialUse": model.get("allowCommercialUse"),
            "allowDerivatives": model.get("allowDerivatives"),
            "allowDifferentLicense": model.get("allowDifferentLicense"),
            "usageControl": version.get("usageControl"),
        },
        "descriptions": {
            "model": model.get("description"),
            "version": version.get("description"),
        },
        "fetched_at": datetime.now(UTC).isoformat(),
    }


def print_versions(model_id: int) -> None:
    """The list for choosing a version (guide mode for a URL without a version — no download)."""
    model = fetch_model(model_id)
    print(f"[civitai] versions of {model['name']} ({model.get('type')}):", flush=True)
    for version in model.get("modelVersions", []):
        files = version.get("files", [])
        primary = next((entry.get("name") for entry in files if entry.get("primary")), "-")
        print(f"  {version['id']}  {version['name']}  {primary}", flush=True)
    print(
        f"[civitai] pick a version and run again: --url '…/models/{model_id}?modelVersionId=<id>'"
        " or --air 'urn:air:…@<id>'",
        flush=True,
    )


def fetch_checkpoint(
    model_id: int,
    version_id: int,
    out: Path = DEFAULT_OUT,
    name: str | None = None,
    requested_air: Air | None = None,
) -> Path:
    """Imports one version into `<out>/civitai-<versionId>/`; returns the path of the main file."""
    version = fetch_version(version_id)
    served_model_id = int(version["modelId"])
    if served_model_id != model_id:
        # The URL's `/models/<id>` can disagree with the version (a URL with another model's
        # version id pasted in). The version's `modelId` is authoritative — so the permission
        # fields are fetched from the right place.
        print(
            f"[civitai] warning: the modelId differs from the requested one — requested"
            f" {model_id} / API {served_model_id}",
            flush=True,
        )
    model = fetch_model(served_model_id)

    derived_name = _assert_name(name) if name else derive_model_name(model["name"], version["name"])
    file = select_file(version.get("files", []))
    others = [entry for entry in version.get("files", []) if entry is not file]
    if others:
        listing = ", ".join(f"{entry.get('name')}（{entry.get('type')}）" for entry in others)
        print(f"[civitai] bundled files not fetched: {listing}", flush=True)
    hashes = file.get("hashes", {})
    if "SHA256" not in hashes:
        raise SystemExit(f"The API has no sha256: {file.get('name')} (cannot verify)")

    # The first level of the destination is an upstream value too — without the int conversion a
    # deviant id becomes a path element pointing outside `out` (`mkdir(parents=True)` creates the
    # parents too, so the weights go out with it). As with modelId above, a non-integer fails
    # loudly.
    destination = out / f"civitai-{int(version['id'])}"
    destination.mkdir(parents=True, exist_ok=True)
    checkpoint = destination / _assert_basename(file["name"])
    download(file["downloadUrl"], checkpoint, hashes["SHA256"])

    record = build_record(model, version, file, derived_name, _resolve_air(version, requested_air))
    (destination / PROVENANCE_FILE).write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[civitai] {destination / PROVENANCE_FILE}", flush=True)
    print(
        "[civitai] next: uv run --group anima python -m anima.single_file "
        f"--checkpoint {checkpoint} --out ../../outputs/misc/anima-diffusers/{derived_name}",
        flush=True,
    )
    return checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--air", help="urn:air:<ecosystem>:<type>:civitai:<modelId>@<versionId>")
    source.add_argument("--url", help="model page URL (the version is ?modelVersionId=<id>)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="parent of the destination")
    parser.add_argument(
        "--name", help="override the distribution name (derived from upstream's naming by default)"
    )
    args = parser.parse_args()

    if args.air:
        air = parse_air(args.air)
        fetch_checkpoint(air.model_id, air.version_id, args.out, args.name, requested_air=air)
        return

    target = parse_url(args.url)
    if target.version_id is None:
        print_versions(target.model_id)
        return
    fetch_checkpoint(target.model_id, target.version_id, args.out, args.name)


if __name__ == "__main__":
    main()
