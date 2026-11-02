"""Where the recipe tests read real assets from (ADR 0139).

- **Distribution mirrors** (`models/<repo>/`, `DIST_ROOT`): a file that a distribution carries
  (a tokenizer, Wan's text embeddings, a container) is read from the mirror, located through the
  repository's `karume.json` (decision 1) — never by spelling the file name here.
- **Goldens** (`outputs/test-assets/goldens/<series>/`, `GOLDENS_ROOT`): recipe outputs that no
  distribution carries, fetched by `node tools/test-assets/fetch.ts` (decision 2). The tests build
  those paths themselves (`GOLDENS_ROOT / <series> / <file>`).
- **Series** (`outputs/series/`): implementation-time checks that need the exporter's own output
  are opt-in with `KARUME_SERIES=1` (decision 6) — the same switch as the TS lanes
  (`packages/models/tests/helpers/series-checks.ts`).

An absent mirror or file SKIPs explicitly, naming the path. A mirror that is present but disagrees
with its own manifest (a part of another size, an asset of another sha256) fails loudly: it is a
broken copy, not a missing one.

`tools/export-recipes/conftest.py` puts this directory on `sys.path` (as for `container_series`).
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from _shared.paths import DIST_ROOT
from karume.container import base_path, container_parts
from karume.dist import MANIFEST_FILENAME

#: Implementation-time checks over `outputs/series/` run only with `KARUME_SERIES=1`.
SERIES_CHECKS: bool = os.environ.get("KARUME_SERIES") == "1"

#: The skip reason of an implementation-time check in the default lane.
SERIES_CHECKS_OPT_IN = "implementation-time series check, opt-in KARUME_SERIES=1"

#: Marks a test (or a class) that reads `outputs/series/`. MUST be the decorator closest to the
#: test, so that without `KARUME_SERIES=1` this reason is the one reported (pytest reports the
#: first matching `skipif`, and the innermost decorator is evaluated first).
requires_series = pytest.mark.skipif(not SERIES_CHECKS, reason=SERIES_CHECKS_OPT_IN)

#: The manifest major these tests read (the TS side's `MIRROR_FORMAT`).
MIRROR_FORMAT = "karume/5"

#: The owner of the published distributions: a cross-repo reference `hdae/<name>` is
#: `models/<name>`.
MIRROR_OWNER = "hdae"


def _mirror_model(repo: str, model: str) -> Mapping[str, Any]:
    manifest_path = DIST_ROOT / repo / MANIFEST_FILENAME
    if not manifest_path.is_file():
        pytest.skip(f"the distribution mirror is absent: {manifest_path} (mirror models/{repo}/)")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != MIRROR_FORMAT:
        pytest.skip(
            f"{manifest_path} is not {MIRROR_FORMAT} (format {manifest.get('format')!r}) — mirror"
            f" models/{repo}/ again"
        )
    models = manifest["models"]
    if model not in models:
        pytest.fail(f"{manifest_path} declares no model {model!r} (declared: {sorted(models)})")
    return models[model]


def _file_path(repo: str, ref: Mapping[str, Any]) -> Path:
    """A manifest file reference → its path in the mirrors (a cross-repo reference resolves to the
    sibling mirror)."""
    cross = ref.get("repo")
    if cross is None:
        return DIST_ROOT / repo / ref["path"]
    owner, _, name = str(cross).partition("/")
    if owner != MIRROR_OWNER or not name:
        pytest.fail(f"models/{repo}/ references {cross!r}, which is not a {MIRROR_OWNER}/<name>")
    return DIST_ROOT / name / ref["path"]


def mirror_asset(repo: str, model: str, role: str) -> Path:
    """The declared asset `role` of `model` in the mirror `models/<repo>/`, checked against the
    manifest's size and sha256."""
    assets = _mirror_model(repo, model).get("assets", {})
    if role not in assets:
        pytest.fail(f"models/{repo}/ declares no asset {role!r} for {model!r}")
    ref = assets[role]
    path = _file_path(repo, ref)
    if not path.is_file():
        pytest.skip(f"the mirror's asset is absent: {path} (mirror models/{repo}/)")
    data = path.read_bytes()
    if len(data) != ref["size"] or hashlib.sha256(data).hexdigest() != ref["sha256"]:
        pytest.fail(f"{path} is not the file models/{repo}/{MANIFEST_FILENAME} declares")
    return path


def mirror_container(repo: str, model: str, key: str, label: str) -> Path:
    """The representative path of the container of `key` at storage label `label` in the mirror,
    after checking that its parts are exactly the manifest's (present, of the declared sizes, one
    numbered sequence).

    The representative path is what the recipe readers take (`_shared.container_read`); folding the
    manifest's parts into it and back checks that the readers open the parts the manifest names.
    """
    weights = _mirror_model(repo, model)["weights"]
    if label not in weights.get(key, {}):
        pytest.fail(f"models/{repo}/ declares no container {key!r} at {label!r} for {model!r}")
    refs = weights[key][label]["container"]["parts"]
    parts = tuple(_file_path(repo, ref) for ref in refs)
    for part, ref in zip(parts, refs, strict=True):
        if not part.is_file():
            pytest.skip(f"the mirror's container part is absent: {part} (mirror models/{repo}/)")
        if part.stat().st_size != ref["size"]:
            pytest.fail(f"{part} is not the size models/{repo}/{MANIFEST_FILENAME} declares")
    representative = base_path(parts[0])
    if container_parts(representative) != parts:
        pytest.fail(f"{representative}: the parts on disk are not the manifest's parts")
    return representative
