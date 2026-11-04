"""Puts the test helpers shared by the recipe tests (`ir_fixtures` / `container_series`) on the
path.

`ir_fixtures` (synthesis of valid minimal containers — `tools/exporter/tests/ir_fixtures.py`) is
used by the tests of **both** core and the recipes. It is kept in one place so that there are not
two copies of the container spellings and the IR rules (if one went stale we would be back to
"the tests are green, only the real thing fails"). The dependency direction stays recipe → core
only (ADR 0065 decision 3), and not one byte of it goes into the wheel (`src/karume/`).

`container_series` (`tools/export-recipes/tests/container_series.py`) is recipe-only glue that
shares across seven families the three moves of treating a part list as a series / distribution
(write / list the expected paths / read). pytest resolves test modules of the same directory on
its own, but not from under the family subpackages (`<family>/tests/` has an `__init__.py`), so the
path is added here explicitly.

MUST: append **at the end** (not `insert(0, …)`) — never let core's test directory resolve the
recipe-side names (`dist` / `_shared` / the family packages) first.
"""

from __future__ import annotations

import sys
from pathlib import Path

#: `tools/export-recipes/conftest.py` → `tools/` → core's test directory.
CORE_TESTS = Path(__file__).resolve().parent.parent / "exporter" / "tests"

#: Where the recipe-side shared test helpers (`container_series`) live.
RECIPE_TESTS = Path(__file__).resolve().parent / "tests"

for helpers in (CORE_TESTS, RECIPE_TESTS):
    if str(helpers) not in sys.path:
        sys.path.append(str(helpers))
