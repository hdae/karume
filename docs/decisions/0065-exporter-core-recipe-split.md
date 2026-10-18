# 0065: Split the exporter into a generic core (PyPI) and per-model recipes (repository only)

- Status: accepted (2026-08-15 — the maintainer's ruling: option A of an external structure and
  license audit, following the maintainer's policy of 2026-08-14); all stages implemented; decision 6
  withdrawn 2026-10-18; decision 7's release gate superseded by the per-publication checks of
  [release-runbook](../release-runbook.md) and [0122](0122-umt5-upstream-and-compatible-encoders.md)
  decision 6 (2026-10-04)
- Date: 2026-08-15
- Related: [0037](0037-karume-monorepo.md) (parent decision for PyPI `karume` — this ADR turns the
  "exporter CLI" into a generic exporter / IR tooling) / [0052](0052-exporter-transactional-artifacts.md)
  (transactional dist contract — stays in the engine) / [0063](0063-safetensors-physical-layout.md)
  (safetensors layout rules — a core responsibility)

## Context

- On 2026-08-15 the public API (`convert` / `normalize` / `emit` / `verify` / `pipeline`) was
  generic, and the conversion core did not import `patch_*` (checked by scanning every import).
  **The physical boundary of the wheel, however, was the whole `karume/` directory**: upstream code like
  `patch_anima.py` (much of it declared as verbatim ports), `anima_text.py`, `dist.py` (3,557 lines)
  and `modelcard.py` with family knowledge baked in, and `cli.py` with a roster of ten model scripts
  all shipped in one Python distribution.
- **The API boundary and the distribution / license boundary did not match**: a `pip install karume`
  user received family knowledge torch.export → IR does not need, with its provenance obligations.
- Observed backflow: `dist.py` imported modelcard family metadata and renderers directly and lazily
  imported `style_bert_vits2`; cross-family utilities (`ROPE_BUFFER_NAMES` / `assert_rope_lifted`)
  lived in `patch_anima` and were used by EmbeddingGemma / Irodori; `conftest.py` injected an Irodori
  checkout into `sys.path`.

## Decision

1. **PyPI `karume` = the generic exporter core only.** It keeps IR / ops / dims / extents / shapes /
   convert / normalize / quantize / act_quant / emit / verify / pipeline / goldens, the **generic dist
   engine** (artifact placement, manifest, hashes, staging / swap (ADR 0052), generic checks) and the
   **generic model-card renderer**. Only model-independent CLI commands stay (today `dist` / `migrate`
   / `verify`); family commands and the script roster are removed.
2. **Per-model recipes live in `tools/export-recipes/<family>/`** (outside the wheel, repository
   only): export scripts, `patch_*`, reference pipelines, demos, quant measurements, the **dist
   recipe** (source layout, model names, quant variants, repository names, card choice — today
   `<family>/distribution.py`, driven by `tools/export-recipes/dist.py`), **card templates and
   metadata**, family tests and `THIRD_PARTY_NOTICES.md` (provenance sits with the code, per family).
   Helpers shared by several families that cannot move into core go to `tools/export-recipes/_shared/`.
3. **Dependencies point one way, recipe → core** (core MUST NOT import a recipe). This is a **machine
   check** — the boundary test `tools/exporter/tests/test_architecture_boundary.py` and the wheel
   content inspection in CI (`.github/workflows/ci.yml`, step "Inspect distribution"): tests guard the
   boundary, not convention.
4. **Dependency management.** The recipes form a private uv project
   (`tools/export-recipes/pyproject.toml`) that holds the family dependency groups (anima / sbv2 /
   siglip2 …). Both projects are members of one uv workspace (`tools/pyproject.toml`, one lock and one
   venv), through which the recipes depend on core.
5. **`src/` layout.** Core lives in `tools/exporter/src/karume/` (prevents accidental imports from the
   working tree and makes the distribution content visible).
6. Withdrawn 2026-10-18: "no compatibility shims (unreleased — breaking changes by default); old
   import paths and CLI commands are removed". It governed the one-time reorganisation, which is
   complete; it is not a compatibility policy for the published package (which ships `karume migrate`).
7. **License.** The goal is a state in which one can show mechanically that the core wheel contains
   no copy of an upstream model implementation; `pyproject.toml` declares `license` (MIT). The
   license compatibility check per upstream revision (a human review) is a release gate (superseded by
   docs/release-runbook.md and ADR 0122 decision 6 on 2026-10-04: the check runs at every publication —
   the per-family provenance item of the runbook checklist and the container license check before
   upload). This reorganisation prepares the check and does not contain it.

## Stages

Eight stages, each keeping `main` green: boundary test first → cross-family utilities out of
`patch_anima` (now `karume/rope.py`) → dist / modelcard split into engine and recipe → family
sources and tests moved → `src/` layout and CLI clean-up → packaging and provenance → CI split → docs
sync. All eight are complete.

## Alternatives considered

- **B: each recipe as its own PyPI distribution** — more version matrix and release duty. A can be
  promoted to B once recipes are offered as a stable external API (too much today).
- **C: status quo + extras (`karume[anima]`)** — keeps heterogeneous upstream licenses in one wheel and
  grows the core's release surface with every model. Least fit for publishing a generic exporter.

## Consequences / non-goals

- **Future option (record only — the maintainer's intent, 2026-08-15):** `export-recipes` may move to
  a **separate repository** (the repository form of option B). Not a goal today; recorded so that
  in-repository coupling (`REPO_ROOT` in `_shared/paths.py`, shared runtime conformance fixtures, the
  goldens output path) does not grow deeper.
- The three JSR packages (runtime / hub / models) and the root `models/` (distribution mirror) are
  **out of scope** (the boundary is already orthogonal); so was `examples/`, removed on 2026-10-16
  (`bb35fe2a`).
- Moving the IR conformance fixtures (`packages/runtime/tests/fixtures/`) to a neutral location is
  **deferred** (not needed for the split; revisit a neutral `spec/` directory when demand appears).
- Repository topology (`paths.py`) belongs to the recipe side (`tools/export-recipes/_shared/paths.py`);
  the core API keeps taking `Path` arguments.
- The exporter README was rewritten for the new layout after the move (only minimal edits at G1).

## Addendum (2026-10-11) — model family names and the default distribution owner kept in core

Two exceptions to decision 1 (no model-specific knowledge in core) are allowed; the comments of the
boundary test follow this scope.

1. **`karume.migrate.PLE_OWNER`** (pipeline name `gemma4` / `gemma4-qat` → the component that absorbs
   the PLE sidecar), only for the one-time migration of the old format (`karume/4`), whose manifest
   has no field for the PLE owner (ADR 0108 addendum 3, item 6). New distributions do not use it.
2. **`karume.modelcard.HF_OWNER`** (default HF account): used in the model card's usage example and
   the default repository name of `karume dist`; `--repo` overrides it, so it is no correctness defect.
