# @karume/geometry-profiles

The generator of the runtime's embedded geometry profile tables
(`packages/runtime/src/kernels/geometry-profiles/<id>.ts`, registered in `BUILTIN_GEOMETRY_PROFILES`). A table is a
static answer for one adapter: which tile geometry each op family uses. The runtime only selects a table by the
adapter's match and measures nothing at run time; a table changes only when this command is run on new sweep records.

The command derives with `deriveGeometryProfile` of `@karume/runtime/tune` and writes the table file. It always works
as "the registered table as base + new records for some op families": the families the given records cover are
derived from them, and every other family is carried over from the registered table unchanged. It never re-derives a
whole table from all its historical records, and it never lifts an old record format. A new adapter's first table has
no base.

## The command

From the repository root:

```sh
node tools/geometry-profiles/generate.ts <profile id> --records=<path>[,<path>…] \
  [--attention-bench=<path>[,…]] [--vendor=<v> [--architecture=<a> [--description=<d>]]] \
  [--min-speedup=<x>] [--dry-run]
```

- A registered id takes the registered table as its base and the table's own adapter match; the match options are then
  an error. An id that is not registered needs `--vendor` (with `--architecture`, and `--description` as needed) and
  has no base.
- `--attention-bench` adds attention-bench records, which decide the online attention's key axis; without them that
  field is carried over from the base.
- `--min-speedup` is the adoption threshold (default ×1.05, the runtime's default).
- Without `--dry-run` it writes the table file and prints what it wrote and the verdict lines. With `--dry-run` it
  writes nothing and prints the derived profile as JSON, the verdict lines and, for a registered id, every field that
  differs from the registered table (the provenance fields listed apart).
- Exit code 0 on success, 2 for a command line it cannot act on, 1 for anything else (the derivation's message on
  stderr, unchanged).

Regenerate a family of an existing table from a new sweep (here the `i8a8-linear` cases of the full set, taken with the
workbench on the RTX 5070 Ti):

```sh
node tools/geometry-profiles/generate.ts nvidia-blackwell --dry-run \
  --records=outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/geometry-sweep_<time>.json
```

A new adapter's first table:

```sh
node tools/geometry-profiles/generate.ts intel-xe2 --vendor=intel --architecture=xe-2lpg \
  --records=outputs/workbench/browser-intel-arc-b570/geometry-sweep_<time>.json
```

## What a record may be

- A workbench record (`karume-workbench/1`, written by `apps/workbench`): the `geometry-sweep` entry's record for
  `--records`, the `attention-bench` entry's record for `--attention-bench`. It is unwrapped to its `data`. A record
  without `data` (a run that failed) is an error that names the file and the outcome.
- A bare runtime record (`karume-geometry-sweep/*` or `karume-attention-bench/*`, written before the workbench
  existed), passed as it is.
- Each path is repository-relative or absolute and must lie inside the repository: the table's provenance cites it
  repository-relative, with the sha256 of the file's bytes as stored (for a workbench record, of the envelope file).

Whether a record is usable — its format version (only `karume-geometry-sweep/3` and `karume-attention-bench/2` are
accepted), its adapter, its case sets, a sweep that covers a family only in part while a base exists — is the
derivation's decision; its errors are printed as they are.

## What the header says

The header of a written table is generated from the inputs alone (no date, no machine name, no absolute path: the same
inputs give the same bytes):

- what the table is, the adapter match, and how to change a value;
- "Derived in this generation": the op families derived now, and per record its path, sha256 and date and how it was
  made (for a workbench record: browser or Node, the backend and where it came from, whether timestamps were quantised,
  the checkout commit and whether it was dirty; for a bare record: what the record carries);
- "Carried over": the families taken from the registered table and the last commit that touched the table file, so
  their adoption lines are one `git show <commit>:<path>` away (absent for a first table);
- "Adoption": the adoption rule with the threshold used, then the derivation's verdict lines.

The headers of the three tables registered before this tool existed (`apple-metal-3`, `nvidia-ampere`,
`nvidia-blackwell`) are hand-edited prose; they stay until a table is regenerated, which replaces its header with the
generated one. The rest of each file is already what this command writes (the tests check it).

## Registering a new table

The command writes a new table's file but does not register it, since registration puts the table into automatic
selection for every adapter it matches. It prints the two edits, with line numbers:

1. `packages/runtime/src/kernels/geometry-profiles/index.ts`: import the table's constant and add it to the array of
   `BUILTIN_GEOMETRY_PROFILES`.
2. `packages/runtime/tests/geometry_profile_test.ts`: import it and add it to the list that test compares
   `BUILTIN_GEOMETRY_PROFILES` with.

## Tests

`vp test tools/geometry-profiles` (the core lane; no GPU). The regeneration case needs the records the registered
tables cite under `outputs/` (outside git) and is skipped, naming the missing files, on a machine without them.
