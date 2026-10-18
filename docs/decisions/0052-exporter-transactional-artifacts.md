# 0052: Transactional publication of exporter artifacts (temp → verify → replace / staging → swap)

- Status: accepted (2026-08-13); decision 1 superseded by the addendum of 2026-09-24
- Date: 2026-08-13
- Related: [0005](0005-verification.md) (fail loudly, verification) /
  [0041](0041-manifest-v2.md) (dist assembly and verification). Raised by external review
  EXPORTER-004 / EXPORTER-002 / B2-stale (`.claude/reviews/2026-08-13_chatgpt-reviews/TRIAGE.md`,
  untracked).

## Context

`export_to_file` truncated the final path, wrote it, and only **then** ran `verify_model`: a crash
or a failed check lost the good artifact and left an invalid file at the final path.
`karume dist` also updated an existing destination in place (unlink → copy, `karume.json` last), so
an I/O failure could leave "old manifest + mixed tree". Rebuilding A only into an output that held
A+B left B's files behind, caught only by `verify_dist` as undeclared files after GB of copying.

## Decision

1. **Single export**: `export_to_file` writes a temporary file in the same directory
   (`<final>.<uuid>.partial`) with `write_model` → `verify_model(temp)` → `os.replace(temp, final)`;
   on failure the temp file is dropped and an existing final is left untouched. `emit.write_model`
   stays the lower layer that writes to the given path. (Superseded by the addendum of 2026-09-24.)
2. **dist assembly**: `assemble_family` (`tools/exporter/src/karume/dist.py:1389`) builds everything
   in a staging directory (`<output>.staging`, same parent = same file system, so rename is atomic):
   placement → shared-weight folding → `karume.json` → `verify_dist` → `README.md`. Only then two
   renames swap it in (existing → `.old`, staging → final, then `.old` removed). A failure (Ctrl-C
   included) removes only the staging directory; the existing distribution is unchanged. Leftovers
   of an interrupted run are cleared before rebuilding — except that a `.old` with no final is the
   last-known-good and is moved back to final (`_reclaim_superseded`,
   `tools/exporter/src/karume/artifacts.py:54-65`).
3. **A rebuild replaces the whole output**: nothing from the old `out_dir` is carried over — an
   A+B → A rebuild drops B. The model card is also drawn inside staging from the verified manifest,
   so a swapped-in distribution always has a matching card.

## Consequences

- Peak disk use of dist is **about twice** the distribution until the swap (`docs/limitations.md`,
  by-design trade-off).
- `<name>.staging` / `<name>.old` appear temporarily next to the output (`models/` is untracked).
  The now-unreachable `dest.unlink` guards in `place_file` were removed on 2026-08-16.

## Addendum (2026-09-24) — single-export atomicity moved to the workspace

The current `export_to_file` alone **does not** give decision 1's guarantee. `export_to_file` →
`publish_model` → `publish_container` (`tools/exporter/src/karume/{pipeline,publish}.py`) writes
under a temporary name (`{stem}.{uuid}.partial{suffix}`, `publish.py:207`), checks it by reading
back, then moves the `krg` and `krm` parts into place one `os.replace` at a time. A failure part-way
removes what this call placed, so if `final` held the previous artifact, **the previous
last-known-good is lost** (the MUST in the `publish_container` docstring).

The unit of atomicity is now the directory: the workspace `karume.artifacts.staged_publication`
(`artifacts.py:102`) opens decision 2's staging → swap to single-file writers (`final` is placed
only when the `with` block exits cleanly), and an AST check
(`tools/export-recipes/tests/test_staged_publication.py`) keeps every recipe writing inside it.
The move landed in `45e75120` (one publication primitive) and `8086ed37` (writers switched to
`krm`). Recipe writers MUST NOT call `export_to_file` outside the workspace. Decisions 2 and 3
are unchanged.
