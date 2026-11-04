"""Locations inside the repository — the one place that separates "distributions" from
"generated output".

DECIDED: `models/` holds **only distributions that can be uploaded to HF as they are** (one
directory = one HF repository — ADR 0037 §3). The series the exporter writes (IR + io fixtures)
are intermediate output needed neither to run nor to distribute a model, so they go under
`outputs/`.

Where series live is **knowledge shared by the writers (scripts) and the reader (the dist
driver)**, so only this one place holds the spellings.

NOTE: it is repository topology itself, so it lives on the recipe side rather than in core (PyPI
`karume`) (ADR 0065 Consequences) — core's API keeps taking a Path, and the spellings are
injected from here.
"""

from __future__ import annotations

from pathlib import Path

#: The repository root (_shared/paths.py → _shared → tools/export-recipes → tools → repo).
REPO_ROOT = Path(__file__).resolve().parents[3]

#: Parent of the distributions (`<DIST_ROOT>/<distribution name>/` becomes one HF repository).
DIST_ROOT = REPO_ROOT / "models"

#: Parent of the real weights' **input material** (the ckpt and config are placed by hand in
#: `<INPUTS_ROOT>/<family>/<name>/`). It is not generated output, so not `outputs/`, and not a
#: distribution, so not `models/`.
INPUTS_ROOT = REPO_ROOT / "inputs"

#: Parent of the exporter's generated output. **Never under `models/`** (the DECIDED above).
OUTPUTS_ROOT = REPO_ROOT / "outputs"

#: Parent of the exporter's series output (IR + io fixtures). Generated output that is not a
#: series (demo assets and the like) goes directly under `OUTPUTS_ROOT`, never mixed in here.
SERIES_ROOT = OUTPUTS_ROOT / "series"

#: The goldens of the local test-asset root `outputs/test-assets/` (a cache of the pinned private
#: dataset, fetched by `node tools/test-assets/fetch.ts` — ADR 0139 decision 2):
#: `<GOLDENS_ROOT>/<series>/<path>` holds the recipe outputs of a series that no distribution
#: carries, at the relative paths they have under `SERIES_ROOT / <series>`.
GOLDENS_ROOT = OUTPUTS_ROOT / "test-assets" / "goldens"

#: Parent of benchmarks, evaluations and dumps (`<BENCH_ROOT>/<model>/<date>_<purpose>/`). **Safe
#: to delete**: everything left here can be recreated by rerunning the scripts.
BENCH_ROOT = OUTPUTS_ROOT / "bench"

#: Parent of verification results and artifacts
#: (`<VERIFY_ROOT>/<environment key>/<date>_<series>/`). The environment key is built by
#: `packages/runtime/tests/helpers/environment.ts` (one directory per GPU). **Safe to delete**:
#: everything left here can be recreated by rerunning the tests.
VERIFY_ROOT = OUTPUTS_ROOT / "verify"

#: Parent of host assets (`<MISC_ROOT>/<name>/`). Deleting them means fetching or emitting them
#: again, so they are kept apart from `BENCH_ROOT` (demo assets, dictionaries, frozen copies of
#: real corpora).
MISC_ROOT = OUTPUTS_ROOT / "misc"

#: Parent of the default output of the `examples/` scripts (`<EXAMPLES_ROOT>/<model>/`).
EXAMPLES_ROOT = OUTPUTS_ROOT / "examples"
