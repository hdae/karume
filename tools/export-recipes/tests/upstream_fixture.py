"""Glue that reproduces the provenance of an upstream checkpoint (`_shared.upstream`) in tests.

Two things are synthesized:

- **The download record of the local checkpoint** ({@link write_snapshot}) — the
  `.cache/huggingface/download/config.json.metadata` that `hf download --local-dir` leaves, and the
  front matter of `README.md`. The source of the provenance a script writes into the container and
  of the revision assembly matches.
- **The provenance a fixture container claims** ({@link stamp_fixture_provenance}) —
  `ir_fixtures.ir_container` takes no provenance argument and writes the module constant
  `FIXTURE_PROVENANCE` (a core test helper, not edited from the recipe side). The assembly tests
  of the families that have a provenance check replace that constant for the duration of the test
  to make a container "in the shape a script writes".

`tools/export-recipes/conftest.py` puts this directory on sys.path (the same route as
`container_series`).
"""

from __future__ import annotations

from pathlib import Path

import ir_fixtures
import pytest

from _shared.upstream import SNAPSHOT_RECORD_DIR, SNAPSHOT_RECORD_FILE
from karume.container import Provenance

#: A synthetic commit SHA (a spelling that matches no real revision).
FIXTURE_REVISION = "0123456789abcdef0123456789abcdef01234567"

#: Another commit SHA, used for fault injection of a mix-up.
OTHER_REVISION = "fedcba9876543210fedcba9876543210fedcba98"


def write_snapshot(
    model_dir: Path, *, license: str, revision: str = FIXTURE_REVISION, extra: str = ""
) -> None:
    """Synthesize the download record of `hf download --local-dir` and the README front matter.

    `extra` is lines added to the front matter (`license_name: …` and the like).
    """
    record = model_dir / SNAPSHOT_RECORD_DIR / f"{SNAPSHOT_RECORD_FILE}.metadata"
    record.parent.mkdir(parents=True, exist_ok=True)
    record.write_text(f'{revision}\n"etag-fixture"\n1786862475.0\n', encoding="utf-8")
    (model_dir / "README.md").write_text(
        f"---\nlicense: {license}\n{extra}tags:\n- fixture\n---\n\n# fixture\n", encoding="utf-8"
    )


def stamp_fixture_provenance(monkeypatch: pytest.MonkeyPatch, provenance: Provenance) -> None:
    """Replace the provenance `ir_fixtures.ir_container` writes, for the duration of this test."""
    monkeypatch.setattr(ir_fixtures, "FIXTURE_PROVENANCE", provenance)
