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

A table is measured (matched by the adapter's description, from that device's records), or predicted (matched by vendor
and architecture). `--shared` derives a shared table: one predicted table from the records of two or more devices of
one vendor and architecture, a geometry taken only where it passes the adoption rule on every device
([the spec](../../docs/spec/geometry-profiles.md#shared-tables)).

GEMV sweep records (`--gemv-records`, the workbench's `gemv-sweep` entry) derive the table's GEMV choices of linear
(`linearGemv`) with `deriveLinearGemvChoices` of `@karume/runtime/tune`, onto the table the sweep records give (or
onto the registered table when only GEMV records are given). Without them, a table that carries the field keeps it.
`--gemv-plan` writes, instead, the workbench batch that measures what those records lack.

Records of the geometry sweep's `grid` set (`--grid-records`, the workbench's `geometry-sweep` entry with
`candidateSet=grid`) derive the grid-fill rule of the GEMM (`gemmGrid`) with `deriveGemmGrid`, onto the table the
other options give (or onto the registered table when only grid records are given). Without them, a table that
carries the field keeps it ([the spec](../../docs/spec/geometry-profiles.md#deriving-the-grid-fill-rule)).

## The command

From the repository root:

```sh
node tools/geometry-profiles/generate.ts <profile id> [--records=<path>[,<path>…]] \
  [--attention-bench=<path>[,…]] [--gemv-records=<path>[,<path>…] [--gemv-margin=<x>]] \
  [--grid-records=<path>[,<path>…]] \
  [--vendor=<v> [--architecture=<a> [--description=<d>]]] [--min-speedup=<x>] [--dry-run]
node tools/geometry-profiles/generate.ts <profile id> --shared --records=<path>,<path>[,…] \
  [--vendor=<v> --architecture=<a>] [--min-speedup=<x>] [--dry-run]
node tools/geometry-profiles/generate.ts <profile id> --gemv-plan [--gemv-records=<path>[,<path>…]] [--dry-run]
```

- A registered id takes the registered table as its base and the table's own adapter match; the match options are then
  an error. An id that is not registered needs `--vendor` (with `--architecture`, and `--description` as needed) and
  has no base.
- `--attention-bench` adds attention-bench records, which decide the online attention's key axis; without them that
  field is carried over from the base.
- `--gemv-records` adds GEMV sweep records of the table's adapter, which decide `linearGemv` group by group (one
  record's case at one row count, current when its rows' and its case's fingerprints are the current runtime's) by two
  rules ([the spec](../../docs/spec/geometry-profiles.md#deriving-the-gemv-choices)): a shape a steady record measured
  at M = 1 by the rule of one steady record — a candidate's gain over the default's choice in the same group clears the
  margin × the two rows' gaps in every steady record, its row is not slower at M = 4 and 8 in the same record, and the
  table's choice stays unless beaten so —; any other by the rule of two records — ×1.10 or more over the default's
  choice in every group that is evidence, read against that kernel's fastest time over the records, which a second
  record repeats within ×1.10, from at least two records, its check held, and a parallel row the same evidence at
  M = 4 and 8 without a regression. `--gemv-margin` sets the first rule's margin (default ×1.065; a finite number of
  at least 1). Below ×1.03 (`GEMV_LOWEST_MARGIN`, the lowest one whose rows the sweep records) the command warns on
  stderr and in the verdict lines; the verdict lines end with the decisions that change at ×1.03. Groups whose case is not current
  are reported in the verdict lines, not refused, and so are the current groups without a usable reference. The field
  is written only when a choice is adopted. A registered id can take `--gemv-records` without `--records` (its tile geometry stays); a first
  table needs `--records`, and `--attention-bench` needs `--records`.
- `--shared` takes records of two or more devices of the vendor and architecture, told apart by their descriptions:
  every record must carry a non-empty description (a browser record has one only when the browser reports it — in
  Chrome, with the WebGPU developer-features flag, `chrome://flags/#enable-webgpu-developer-features`), the records
  of one description must agree on the device id too, and at least two descriptions are needed. The records are
  grouped by description and each group is read as one adapter's; a geometry is
  adopted only when, on every device, its output is identical and it is at least the threshold faster than that
  device's default on every case of the field the device's records hold, and among those the one with the largest
  geometric mean of the per-device geometric means is taken. A field some device has no case of keeps the default. A
  first shared table needs `--vendor` and `--architecture` and takes no `--description`; a registered one is
  regenerated onto records of its own adapters. A shared table carries tile geometries only: `--attention-bench`,
  `--gemv-records` and `--gemv-plan` are usage errors beside it. Its verdict lines give each device's ratios.
- `--grid-records` adds grid records of the table's adapter: a pair of integers of a fixed grid qualifies when, on
  every grid case, its ladder tile wrote the default's words, is timed against the table's tile (by a steady record
  alone, else beside a second one) and is not slower than it beyond `1 ÷ (×1.065 × the two rows' gaps)` in any such
  reading, and when the census-weighted GEMM time under it is at most `1 ÷ --min-speedup` of the table's; the lowest
  weighted time is taken, and none qualifying leaves the table without the field. A registered id can take
  `--grid-records` alone; `--shared` and `--gemv-plan` refuse it.
- `--min-speedup` is the tile geometry's adoption threshold (default ×1.05, the runtime's default) and the grid-fill
  rule's threshold on the weighted total; the GEMV choices keep their own rules (`--gemv-margin`).
- Without `--dry-run` it writes the table file and prints what it wrote and the verdict lines. With `--dry-run` it
  writes nothing and prints the derived profile as JSON, the verdict lines and, for a registered id, every field that
  differs from the registered table (the provenance fields listed apart). With GEMV records the verdict lines end with
  the GEMV decisions shape by shape (the default's choice, the derived one, the gains per record, or why it was kept)
  and a line that says whether the table would carry the field; `linearGemv` is compared as each table runs with it.
- `--gemv-plan` derives nothing. From the GEMV records given (none: nothing at hand) it prints, per case and row
  count of the `full` set, how many candidates the derivation would still miss by the rule it takes for each decision
  ([the spec](../../docs/spec/geometry-profiles.md#planning-the-gemv-measurements)), and
  writes `inputs/workbench/batches/gemv-plan-<profile id>.json` (outside git; the workbench lists it as an untracked
  batch) for those of the rule of two records: `gemv-sweep` steps of the `full` set narrowed to the cases, row
  counts and candidates that miss the same, run twice, each step at most 19 groups (cases × row counts) so that its
  record has no class and stays with the rule of two records it was planned for. What only a steady record fills (a
  decision of the rule of one steady record: a steady record timed the case at M = 1) is listed apart and is in no
  step — a step's record has no class —: a steady run of the `full` or the `explore` set fills it. With no record at hand the batch
  is the whole `full` set, twice; with nothing for the batch no batch is written. `--dry-run` prints the batch instead. The derivation's other options beside it are usage errors.
- Exit code 0 on success, 2 for a command line it cannot act on, 1 for anything else (the derivation's message on
  stderr, unchanged).

Regenerate a family of an existing table from a new sweep (here the `i8a8-linear` cases of the full set, taken with the
workbench on the RTX 5070 Ti):

```sh
node tools/geometry-profiles/generate.ts nvidia-blackwell --dry-run \
  --records=outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/geometry-sweep_<time>.json
```

Say what the GEMV sweeps at hand lack, and write the batch that measures it (run it from the workbench's menu, or
under Node with `flock /tmp/karume-gpu.lock node apps/workbench/run.ts --batch=inputs/workbench/batches/gemv-plan-nvidia-ampere.json`):

```sh
node tools/geometry-profiles/generate.ts nvidia-ampere --gemv-plan \
  --gemv-records=outputs/workbench/node-nvidia-geforce-rtx-3080-ti/gemv-sweep_<time 1>.json,outputs/workbench/node-nvidia-geforce-rtx-3080-ti/gemv-sweep_<time 2>.json
```

Derive the GEMV choices of a registered table from GEMV sweeps, its tile geometry unchanged:

```sh
node tools/geometry-profiles/generate.ts nvidia-ampere --dry-run \
  --gemv-records=outputs/workbench/node-nvidia-geforce-rtx-3080-ti/gemv-sweep_<time 1>.json,outputs/workbench/node-nvidia-geforce-rtx-3080-ti/gemv-sweep_<time 2>.json
```

A new adapter's first table:

```sh
node tools/geometry-profiles/generate.ts intel-xe2 --vendor=intel --architecture=xe-2lpg \
  --records=outputs/workbench/browser-intel-arc-b570/geometry-sweep_<time>.json
```

A shared table regenerated from two devices' records (`apple-metal-3-shared`; a registered id takes no match — its
first table also took `--vendor=apple --architecture=metal-3`):

```sh
node tools/geometry-profiles/generate.ts apple-metal-3-shared --shared \
  --records=outputs/workbench/browser-apple-m5/geometry-sweep_<time>.json,outputs/workbench/browser-apple-m2/geometry-sweep_<time>.json
```

## What a record may be

- A workbench record (`karume-workbench/1`, written by `apps/workbench`): the `geometry-sweep` entry's record for
  `--records`, the `attention-bench` entry's record for `--attention-bench`, the `gemv-sweep` entry's record for
  `--gemv-records`. It is unwrapped to its `data`. A record
  without `data` (a run that failed) is an error that names the file and the outcome.
- A bare runtime record (`karume-geometry-sweep/*` or `karume-attention-bench/*`, written before the workbench
  existed, or `karume-gemv-sweep/*`), passed as it is.
- Each path is repository-relative or absolute and must lie inside the repository: the table's provenance cites it
  repository-relative, with the sha256 of the file's bytes as stored (for a workbench record, of the envelope file).

Whether a record is usable — its format version (only `karume-geometry-sweep/4`, `karume-attention-bench/2` and
`karume-gemv-sweep/4` are accepted), its adapter, its case sets, its fingerprints, a sweep that covers a family only in
part while a base exists — is the derivation's decision; its errors are printed as they are. Of a GEMV record, the
groups that are not of the current runtime are reported, not refused.

## What the header says

The header of a written table is generated from the inputs alone (no date, no machine name, no absolute path: the same
inputs give the same bytes):

- what the table is, the adapter match, and how to change a value;
- "Derived in this generation": the op families derived now, and per record its path, sha256 and date and how it was
  made (for a workbench record: browser or Node, the backend and where it came from, whether timestamps were quantised,
  the checkout commit and whether it was dirty; for a bare record: what the record carries);
- "Carried over": the families taken from the registered table and the last commit that touched the table file, so
  their adoption lines are one `git show <commit>:<path>` away (absent for a first table);
- for a shared table: the devices it predicts from and that it is matched without a description, the records listed
  under the device each is of, and the shared adoption rule;
- "Adoption": the adoption rule with the threshold used (left out when no sweep record was given), the GEMV choices'
  rule when GEMV records were given, then the derivations' verdict lines. The GEMV choices are named among the things
  derived or carried over like an op family.

Every registered table's header is the generated one, and the rest of each file is what this command writes (the
tests check it). The tile geometry of `apple-metal-3` was written by a tool that no longer exists; its header's
"Carried over" line names the commit whose file holds that tool's adoption lines.

## Registering a new table

The command writes a new table's file but does not register it, since registration puts the table into automatic
selection for every adapter it matches. It prints the two edits, with line numbers:

1. `packages/runtime/src/kernels/geometry-profiles/index.ts`: import the table's constant and add it to the array of
   `BUILTIN_GEOMETRY_PROFILES`.
2. `packages/runtime/tests/geometry_profile_test.ts`: import it and add it to the list that test compares
   `BUILTIN_GEOMETRY_PROFILES` with.

## Tests

`vp test tools/geometry-profiles` (the core lane; no GPU). The regeneration case needs the records the registered
tables cite under `outputs/` (outside git) and is skipped, naming the missing files, on a machine without them; the
records cited today are `karume-geometry-sweep/3`, so it checks that the derivation refuses them by their format.
The plan's case reads the batch it writes with the workbench's own batch reader.
