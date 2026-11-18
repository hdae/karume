# Geometry profile tables

How the per-adapter geometry profile tables under `packages/runtime/src/kernels/geometry-profiles/` are made now. A
table is a static answer for one adapter: which tile geometry each op family uses. The runtime selects a table by the
adapter's match and measures nothing at run time.

## Where a table comes from

- A table changes only through the generator, `node tools/geometry-profiles/generate.ts`, run on records of the
  workbench's `geometry-sweep` entry (and, for the online attention's key axis, of its `attention-bench` entry —
  [workbench.md](workbench.md#the-entries)). The table files are not edited by hand.
- The generator derives with `deriveGeometryProfile` of `@karume/runtime/tune`. Whether a record is usable — its
  format version, its adapter, its case sets, a sweep that was cut short — is that function's decision, and its
  refusal is the generator's error.
- The generator always works as **the registered table as base plus new records**: the op families the given records
  cover are derived from them, and every other family is carried over from the registered table unchanged. With a
  base, a covered family is derived from every case of its current case set, so a sweep of only some of a family's
  ops is refused.
- A new adapter's first table has no base and needs its adapter match on the command line.
- The GEMV choices of linear come from GEMV sweep records ([below](#deriving-the-gemv-choices)), given next to the
  sweep records or alone for a registered table. Without them, a base table's `linearGemv` is carried as it is,
  also through a regeneration that derives only the tile geometry.
- A whole table is never re-derived from all the records it ever came from, and a record of an older format is never
  rewritten to pass. The records cited by a family of an older format stay cited until that family is swept again.
- A sweep record's row the quick look screened out (`screened`, [workbench.md](workbench.md#the-entries)) has no
  ratio and no output match: it is no observation, so a geometry screened out in a case that no record timed it in is
  not adopted for that case's class (its verdict says so).

## The skeleton form and the counter decode

- A geometry of the f32 GEMM can carry two fields beside its tile: the skeleton form (`form`: `"tb-un"` or
  `"tas-tb-un"`) in the row-count rules of linear / matmul / bmm (`gemmRows`) and in `conv2d`, and the counter decode
  (`decode: "counter"`) in `conv2d` only. The fused attention's fields take neither. The output words are the same
  with and without them; they are a choice of speed, and the pipeline key names them.
- The sweep's candidate sets `quick+` and `full` measure them. A linear / matmul / bmm case measures every tile of the
  set and then the same tile with each form (three geometries per tile). A conv2d case measures every tile, the tile
  with each form, with the decode, and with the decode and each form (six per tile). `quick` measures the tiles alone.
- A case measures the fields only when its kernel takes them: the vec4 kernel of f32 compute, by the predicate the
  pipeline keys use. Every linear / matmul / bmm and conv2d case of the sweep's table runs that kernel. The fused
  attention measures the tiles alone: its kernels keep the original form, so a field there would time one kernel twice.
- The derivation reads a tile with a field as a geometry of its own, under the same rule as any other: identical
  output and the threshold on every case of its class, the largest geometric mean. An adopted one is written into the
  rule with its fields. A class with a case that did not measure it does not adopt it.
- A record's row that carries a field its op does not take (a form on a fused-attention row, the decode outside conv2d)
  or a value the field does not have is refused, naming the row.
- `quick+` lists a registered table's rule that carries a field by its tile; the cases then measure that tile with
  every field, the table's own geometry among them.
- The two sets grow by these candidates: `quick+` from 821 to 1,517 rows over the sweep's 78 cases, `full` from 6,269
  to 10,040. The case sets, the default kernels and their fingerprints do not change, so a registered table still
  matches.
- No registered table carries a field. A table takes one when its family is derived from a sweep that measured it.

## A record

- A workbench record (`karume-workbench/1`), unwrapped to its `data`, or a bare runtime record
  (`karume-geometry-sweep/*`, `karume-attention-bench/*`) written before the workbench existed, or a bare GEMV sweep
  record (`karume-gemv-sweep/*`) as `runGemvSweep` returns it. A workbench record of another entry than the option
  takes and a workbench record without `data` are errors that name the file.
- A record path lies inside the repository. The table's provenance cites it repository-relative, with the sha256 of
  the file's bytes as stored (for a workbench record, of the whole file).
- The records live outside git (`outputs/`).

## The file

- From its `import type` line on, a table file is the profile object as a JavaScript literal passed through the
  repository's formatter. An unchanged profile gives the same bytes.
- The header of a generated table says: the adapter match; the op families derived in this generation and, per record,
  its path, sha256 and date and how it was made (browser or Node, the backend and its source, whether timestamps were
  quantised, the checkout); the families carried over and the last commit that touched the table file, where their
  adoption lines are; the adoption rule with the threshold used; the verdict lines of the derivation. It holds no
  date of generation and no absolute path: the same inputs give the same bytes.
- When the table carries the GEMV choices, the header says where they come from. When this generation derived them,
  they are among the things derived, their records are listed with the others, their rule follows the tile
  geometry's (which is left out when no sweep record was given) and their decisions follow the verdict lines; when
  they were carried, they are among the things carried over.
- The headers of the three tables registered before the generator existed (`apple-metal-3`, `nvidia-ampere`,
  `nvidia-blackwell`) are older prose. Each is replaced when its table is regenerated.

## The GEMV choices of linear

- A profile can also hold the GEMV choices of linear (`linearGemv`): the variant of the sequential kernel and its
  per-shape exceptions, the thread target that decides the row-block height, and the rows of the shapes that take the
  parallel GEMV with their lane count and their packed and staged flags.
- What measures them on a host's GPU is the GEMV sweep: `runGemvSweep` of `@karume/runtime/tune`, run by the
  workbench's `gemv-sweep` entry ([workbench.md](workbench.md#the-entries)). Its cases are the shapes of the default
  profile's GEMV rows and the linears of regular Gemma 4 E2B and E4B in bf16; per case and row count it measures the
  baseline (what a Session at the level `exact` runs with
  the default profile) against the sequential variants and row-block heights, whose words must equal the baseline's,
  and against the parallel GEMV at each lane count, plain, staged and subgroup, held to the band of `linear`, after a
  quick look that times in full only the references and the kernels near enough to matter; at M = 4 and 8 it skips the
  reorder candidates that the case's M = 1 group of the same run leaves unadoptable by the rule below
  ([workbench.md](workbench.md#how-the-gemv-sweep-times-a-row-count)). Its record is `karume-gemv-sweep/4`
  ([workbench.md](workbench.md#the-gemv-sweeps-record)): each row carries the fingerprint of its kernel and each case
  its own.
- A case of a float storage (bf16, f16, f32) is measured at 1, 4 and 8 rows. Such a weight has no row-block form, so
  past one row its baseline is the GEMM under f32 compute with the default profile's geometry (the candidate `gemm`),
  it has no identical-class candidate there, and its reorder candidates are the parallel GEMV's plain and staged
  forms (it has no subgroup form). Its buffers hold no scale.
- No registered table carries them. A table without them runs with the default profile's, the values measured on the
  RTX 3080 Ti: the Session fills them in once, where it selects its profile, so the linear selection, the fusion pass
  and the recipe builder all read the same choices. The memory estimate, which has no adapter, reads the default
  profile's.
- A table carries the field only as a whole: the default's choices with the derived ones in place. Its provenance has
  no entry for the field (the profile's type holds none); the records it came from are in the table's header.
- `SessionOptions.linearGemvRowsThreadTarget`, when given, replaces the profile's thread target.
- The printer writes the field of a table that carries it, and the saved-table reader (`parseGeometryProfileJson`)
  reads it.

## Deriving the GEMV choices

`deriveLinearGemvChoices` of `@karume/runtime/tune` derives `linearGemv` for a table from GEMV sweep records. It
measures nothing and changes no default value.

- **Inputs**: any number of `karume-gemv-sweep/4` records of one adapter — the table's match selects it and, when the
  table has provenance, its four adapter fields are the table's —, with GPU timestamps (`ns` or `deno-raw-tick`)
  that are not quantised. A record of an older format (`karume-gemv-sweep/3`, `/2`, `/1`) is refused with a message that
  names the format, and nothing converts it; so are a wall-clock or quantised record, a record of another adapter, the
  same record twice and no record at all.
- **The unit of evidence is a group**: one record's measurement of one case at one row count. A row is *current* when
  the current runtime has its candidate at that case and row count and builds the same kernel for it (the row's
  `kernel` fingerprint). A group is current when its case is the current runtime's (the record's case fingerprint, and
  the row count in the case table), the default profile's choices its summary names at `exact` and `reorder` are the
  current ones, and the rows of those choices — the baseline, and the choice at `reorder` where the sweep measures it
  — are current and timed. A group is evidence for a candidate when it is current, steady, and the candidate's row is
  current and timed (a row the quick look screened out, or one the sweep skipped because the same record's M = 1 group
  left the candidate unadoptable, is never evidence; a skipped row that is a baseline, has a time or is also screened is
  refused). So a change to one kernel, or a case
  added, leaves the other rows' evidence standing, and records that each hold part of the cases add up.
- **Adoption**: a candidate replaces the default profile's choice for a shape only when, at M = 1, groups of at least
  two different records are evidence for it, it is at least ×1.10 faster than that choice in every one, and its check
  held in every current group, steady or not (identical class: its words equal the baseline's; reorder class: inside
  the band); a row that failed in a current group rules it out too. Among several, the one whose smallest gain is
  largest. A candidate's time is the row's `perDispatch` (the minimum of its rounds ÷ `reps`). The identical class
  gains against the baseline, the reorder class against the default's choice at `reorder`.
- **Steadiness**: a group whose baseline spread (`baselineSpread`) exceeds 1.25, or that has none, is unsteady and no
  evidence. The bound comes from the two `quick` records of the RTX 3080 Ti of 2026-11-17: below 1.25 the spread did
  not predict a less repeatable gain.
- **The identical class** writes the sequential variant's per-shape exceptions (`sequential.shapes`) from M = 1. Its
  key has no group length: the cases that share a key must reach the same decision, or the key keeps the default. An
  adopted variant replaces the key's exception or adds one; when it is the general variant, the key's exception goes.
  It changes no number of any Session.
- **The reorder class** writes the parallel rows (`parallelShapes`): the lanes and the staged flag of the plain and
  staged candidates at M = 1. A parallel row serves M = 1 to 8, so at each other row count of that range the case
  table has (M = 4 and 8: the `full` set) groups of at least two records must be evidence for the candidate, and it
  may be slower than the default's choice in none of them — for a float-stored shape without a row that choice is
  the GEMM. The adopted row replaces the shape's row in place, or is added when the shape had none. A row
  whose default takes packed activations stays: the packed kernels are not measured. A subgroup candidate that meets
  the rule is named in the decision's notes and writes nothing: the subgroup form is a Session option, not a row. It
  changes the numbers of Sessions at `reorder`, inside the band.
- **Not derived**: the row-block thread target (one number for every shape, while the `full` and `row-blocks` sets
  measure a row-block height per case and row count, and no rule turns those into it) and the general sequential
  variant (it also sets the row blocks' columns and read-ahead).
- **Output**: the field (absent when nothing is adopted), the default profile's choices, one decision per sequential
  key, per parallel row and for the thread target (adopted or kept, the default's choice and the derived one, the gains
  per group that is evidence, why it was kept), the groups that are not current (record, case, row count, and whether
  the case, the default's choices or a reference's row moved), the lines that state them (per record its groups —
  current and steady, unsteady, not current by cause — and its rows of current groups that are not current), the
  rule's two numbers and the records used (path, sha256, date, candidate set, user agent and those counts; the adapter
  and the case table).

## Planning the GEMV measurements

`planGemvMeasurements` of `@karume/runtime/tune` reads the same records (the same refusals) and says what they lack
for the derivation to be complete under the current runtime: per case and row count of the `full` set, the
candidates the derivation reads there — at M = 1 every candidate, at M = 4 and 8 the reorder class, on the records'
gates (subgroups, grid limit; records whose gates differ are refused) — that fewer than two records hold a current,
steady group with a current row of. A row the quick look screened out counts as looked at (measured again, it would be
screened again), and so does a row the sweep skipped at M = 4 or 8 (a run that measures the case's M = 1 again would
skip it again); a row that failed does not. The default's choice at `reorder` is no candidate of the list: every
measurement of a group takes it.

The generator writes the plan as a workbench batch ([workbench.md](workbench.md#batches)),
`inputs/workbench/batches/gemv-plan-<profile id>.json`: per list of missing candidates one `gemv-sweep` step of the
`full` set, its `cases` the cases that miss that list at the same row counts, `rows` those row counts, `candidates`
the list; the steps run twice (one record is not evidence). With no record at hand the batch is the whole
measurement: the `full` set, twice. With nothing missing no batch is written.

## The command

`node tools/geometry-profiles/generate.ts <profile id> [--records=<path>[,…]] [--attention-bench=<path>[,…]]
[--gemv-records=<path>[,…]] [--vendor=<v> [--architecture=<a> [--description=<d>]]] [--min-speedup=<x>]
[--dry-run]`

`node tools/geometry-profiles/generate.ts <profile id> --gemv-plan [--gemv-records=<path>[,…]] [--dry-run]`

- A registered id uses its table as base and its table's match; giving a match is an error. An id that is not
  registered needs `--vendor` and `--records`.
- At least one of `--records` and `--gemv-records` is given. With `--gemv-records` alone, a registered table keeps its
  tile geometry and takes the derived GEMV choices; `--attention-bench` needs `--records`. `--min-speedup` is the tile
  geometry's threshold only.
- `--gemv-plan` derives nothing: it prints what the GEMV records lack ([above](#planning-the-gemv-measurements)) and
  writes its batch (with `--dry-run`, prints it); `--records`, `--attention-bench`, a match and `--min-speedup` beside
  it are usage errors. A registered id's table checks the records' adapter as the derivation does.
- `--dry-run` writes nothing and prints the derived profile, the verdict lines (with the GEMV decisions shape by shape
  and whether the table would carry the field) and every field that differs from the registered table; with GEMV
  records, `linearGemv` is compared as each table runs with it (the default's choices where a table has no field).
- A new table is written but not registered: the command prints the two edits registration needs (the array of
  `BUILTIN_GEOMETRY_PROFILES` and the list `geometry_profile_test.ts` compares it with).
- Exit code 0 on success, 2 for a command line it cannot act on, 1 for anything else.

## What is checked

- Every registered table's text from the `import type` line on equals what the generator writes for that profile
  (core lane). A hand edit of a table's object fails this.
- On a machine that has the records: for `nvidia-ampere` and `nvidia-blackwell`, the records their `i8a8-linear`
  family cites, with the registered table as base, derive an object equal to the registered table, provenance
  included — when they are of the format the derivation reads. Those records are `karume-geometry-sweep/3` today, so
  the case checks that the derivation refuses them by their format, until the family is swept again. Without the
  records the case is skipped and says which files are missing.
- `apple-metal-3` has no such case: every record it cites has a format the derivation no longer accepts.
- The skeleton form and the counter decode as candidates (core lane): what each case measures in each set, that no
  case measures one kernel under two names, the derivation adopting a tile with a field and refusing a field where no
  kernel takes it, and a table that carries the fields written by the generator, read back, and matching its adapter
  by the fingerprints of the kernels it runs.
- The default profile's GEMV choices equal, row for row, the values the GEMV kernel module held before they moved into
  the profile, and a profile with other choices changes the selection and the pipeline key (core lane).
- The GEMV derivation on hand-built records: the adoption rule, the steadiness bound, the kept cases, groups as the
  unit of evidence (a stale, screened or skipped row is no evidence while its record's other groups are; records that each hold
  half of the cases derive what a complete pair derives; groups that are not current are reported) and the refusals
  (core lane); the generator's dry run, its write path and the carry-over of an existing `linearGemv`.
- The plan of GEMV measurements: exactly the missing candidates, and a batch the workbench's own batch reader accepts
  (core lane).
