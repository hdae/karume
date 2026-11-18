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
- `nvidia-blackwell` carries them: a form in its four `gemmRows` rules above 64 rows, and a form with the decode in
  both `conv2d` fields. `nvidia-ampere` and `apple-metal-3` carry none; a table takes one when its family is derived
  from a sweep that measured it.

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
- The header of `apple-metal-3`, registered before the generator existed, is older prose. It is replaced when the
  table is regenerated.

## The GEMV choices of linear

- A profile can also hold the GEMV choices of linear (`linearGemv`): the variant of the sequential kernel and its
  per-shape exceptions, the thread target that decides the row-block height, and the rows of the shapes that take the
  parallel GEMV with their lane count, their packed and staged flags and the row block they take at M = 2 to 8
  ([below](#the-row-block-of-a-parallel-row)).
- What measures them on a host's GPU is the GEMV sweep: `runGemvSweep` of `@karume/runtime/tune`, run by the
  workbench's `gemv-sweep` entry ([workbench.md](workbench.md#the-entries)). Its cases are the shapes of the default
  profile's GEMV rows, the other integer-storage linears a decode token or a draft step of the Gemma 4 distributions
  runs at one row (the int2 and int4 output heads, the MTP drafters) and the linears of regular Gemma 4 E2B and E4B
  in bf16, so that a table can be derived for a shape the default profile gives no row; per case and row count it
  measures the baseline (what a Session at the level `exact` runs with
  the default profile) against the sequential variants and row-block heights, whose words must equal the baseline's,
  and against the parallel GEMV at each lane count, plain, staged and subgroup — and at M = 4 and 8 its row-blocked
  form at each height of at most M rows the kernel exists at —, held to the band of `linear`, after a
  quick look that times in full only the references and the kernels near enough to matter; at M = 4 and 8 it skips the
  reorder candidates that the case's M = 1 group of the same run leaves unadoptable by the rule below
  ([workbench.md](workbench.md#how-the-gemv-sweep-times-a-row-count)). Its record is `karume-gemv-sweep/4`
  ([workbench.md](workbench.md#the-gemv-sweeps-record)): each row carries the fingerprint of its kernel and each case
  its own.
- A case of a float storage (bf16, f16, f32) is measured at 1, 4 and 8 rows. Such a weight has no row-block form, so
  past one row its baseline is the GEMM under f32 compute (the candidate `gemm`),
  it has no identical-class candidate there, and its reorder candidates are the parallel GEMV's plain and staged
  forms (it has no subgroup form and no row-blocked form). Its buffers hold no scale.
- The GEMM's tile geometry is not a GEMV choice: a Session takes it from the profile its device resolves to. The
  sweep builds that GEMM with the profile its own device resolves to (the embedded table of the adapter, the default
  profile where none matches, an injected table when the device was acquired with one) and writes the profile's id in
  the record (`geometryProfile`). Every other kernel of the sweep reads nothing of the profile.
- `nvidia-ampere` carries them; `nvidia-blackwell` and `apple-metal-3` do not. A table without them runs with the
  default profile's: the Session fills them in once, where it selects its profile, so the linear selection, the
  fusion pass and the recipe builder all read the same choices. The memory estimate, which has no adapter, reads the
  default profile's; `nvidia-ampere`'s differ from them in per-shape exceptions of the sequential variant alone,
  which no size and no fusion reads.
- The default profile's sequential variant is 32 columns per workgroup and 4 weight words read ahead (`c32u4`). Its
  per-shape exceptions are 23 rows:
  - the int8 output head `262144 × 1536`: `c16u4`. The row predates the derivation
    ([ADR 0082](../decisions/0082-linear-gemv-decode.md), addendum 10); the derivation did not adopt it, and keeps
    it where records hold its case. No shipped model has a linear of this shape and storage;
  - 22 bf16 linears of regular Gemma 4 E2B and E4B, which read 8 or 16 words ahead: `c16u8` on 11 shapes, `c16u16`
    on 10 and `c32u16` on `1536 × 256`. A bf16 word holds 8 elements where an int4 word holds 32, so 4 words ahead
    keep a quarter of the bytes in flight. The derivation adopts them from GEMV sweep records of the RTX 3080 Ti
    ([perf-ledger](../perf-ledger.md) K-112), and the RTX 5070 Ti in Chrome reads each of them faster than `c32u4`
    as well.
- A row is added to the default profile only where no measured device is slower with it: every adapter whose table
  has no GEMV choices of its own runs it. A row that one device gains from and another loses with is a row of the
  first one's table. The devices measured are the RTX 3080 Ti and the RTX 5070 Ti; no Apple device is, and
  `apple-metal-3` runs the default's rows. The 22 bf16 rows meet this. The int8 head's row, which predates it, does
  not: the RTX 5070 Ti reads `c32u4` ×1.04 faster than the row's `c16u4`.
- `nvidia-ampere`'s field is the default profile's choices with four more per-shape exceptions, what the derivation
  adopts from the RTX 3080 Ti's records beyond the default's rows: the int4 output heads of the Gemma 4 q4_0 QAT E2B
  and E4B models, `262144 × 1536` and `262144 × 2560` (`c16u2`), bf16 `10240 × 2560` (`c16u8`) and the MTP drafter
  shape int4 `256 × 2048` (`c16u16`) — each slower than `c32u4` on the RTX 5070 Ti. The derivation leaves the two
  bf16 output heads on the general variant: no candidate gained in every record.
- An exception names a storage and a shape, has no group length, and applies at one row: a row block takes the
  general variant. Its words are the general variant's, so a Session runs it at `exact` too.
- A table carries the field only as a whole: the default's choices with the derived ones in place. Its provenance has
  no entry for the field (the profile's type holds none); the records it came from are in the table's header. So a
  table holds the default's parallel rows, thread target and general variant as they were when it was generated, and
  a change of those in the default profile regenerates the table (a core-lane test compares `nvidia-ampere`'s with
  the default's).
- `SessionOptions.linearGemvRowsThreadTarget`, when given, replaces the profile's thread target.
- The printer writes the field of a table that carries it, and the saved-table reader (`parseGeometryProfileJson`)
  reads it.

### The row block of a parallel row

- A parallel row can carry a row block (`rowBlock`: 2, 4 or 8). A linear of that shape at the level `reorder` with
  M = 2 to 8 rows — a speculative verification, a short prefill — then runs the row-blocked parallel GEMV: a thread
  holds one lane of one output column for R rows, reads each weight word and decodes it once, and multiply-adds it with
  every row's activation, which the workgroup stages per row in workgroup memory. R is the largest of 2, 4 and 8 that
  is at most M and at most the row block: M = 8 under a row block of 4 runs two y tiles of four rows, and M = 3 runs
  two tiles of two, the second holding one row.
- Without a row block every row count runs the row's kernel of M = 1, one workgroup row per row, which reads and
  decodes the weights once per row.
- The row-blocked kernel writes the words of the parallel GEMV at the same lanes: each row's sum takes the same words
  in the same order through the same tree. A verify row therefore equals the decode row of the same input, as before.
- The row block does not depend on the row's staged flag, which is the form at M = 1: the row-blocked kernel always
  stages. A row block over activations read from storage is not built (on the reference device it measured ×0.4 to
  ×1.2 of the kernel without one).
- The kernel exists where the staged activations of its rows and its partial sums fit the workgroup storage WebGPU
  core guarantees (16,384 B), so no device limit decides it: i4 up to 8 rows at 2, 4 and 8 lanes, 4 at 16 and 2 at
  32; i8 up to 8 rows through 16 lanes and 4 at 32; i2 up to 8 rows at 2 and 4 lanes, 4 at 8, 2 at 16 and none at 32.
  A table whose row names a row block the kernel does not exist at, or sets one on a row that takes packed
  activations, is refused.
- A row of a float storage (bf16, f16, f32) carries no row block: the row-blocked kernel exists for i2, i4 and i8
  storage. Such a row runs its kernel of M = 1 at every row count of 1 to 8, one workgroup row per row. A table that
  sets a row block on a float-stored row is refused, the GEMV sweep has no row-blocked candidate for a float case,
  and a derived float row has none.
- M = 1, the level `exact`, row counts above 8, the subgroup form, a row that takes packed activations and the
  linear → static_quantize fusion run what they ran without the field.
- The default profile's rows carry the row block the derivation adopts from two GEMV sweep records of the RTX
  3080 Ti at M = 4 and 8 ([perf-ledger](../perf-ledger.md) K-108): rows of the Gemma 4 q4_0 QAT shapes. A row those
  records did not support — below ×1.10 in a group, or a baseline that was unsteady in one record — and the rows that
  were not measured (the mobile QAT shapes, the i8 output heads) carry none.

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
  — are current and timed. The current kernel of a GEMM baseline is the one the table the records derive for builds:
  a group whose GEMM was measured with another tile geometry (under another profile) is not current, and the reason
  names the record's profile and the table. A record that does not name its profile is read by the same comparison of
  kernel fingerprints. A group is evidence for a candidate when it is current, steady, and the candidate's row is
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
  It changes no number of any Session. A float-stored shape has a sequential key like any other: from records of the
  bf16 cases a derived table may carry `sequential.shapes` rows for bf16, and an `exact` Session on that adapter then
  runs another sequential variant (columns per workgroup, words read ahead) that gives the same words.
- **The reorder class** writes the parallel rows (`parallelShapes`): the lanes and the staged flag of the plain and
  staged candidates at M = 1, and the row block. A parallel row serves M = 1 to 8, and at each other row count of
  that range the case table has (M = 4 and 8: the `full` set) it runs its own kernel or, with a row block, the
  row-blocked kernel of its lanes (`staged-l<lanes>r<rows>`). At each of those row counts groups of at least two
  records must be evidence for the kernel the derived row runs there, and it may be slower than the default's choice
  in none of them — for a float-stored shape without a row that choice is the GEMM; a candidate whose row does not
  meet this is not adopted. The adopted row replaces the shape's row in place, or is added when the shape had none.
  - **The row block** of a row is chosen by one rule, against a reference row of the same lanes: the default's row
    where M = 1 keeps its lanes and staged flag, the candidate without a row block where M = 1 adopts one. The row
    blocks of one lane count write the same words, so the choice among them is speed alone. A row block — or none —
    replaces the one the reference row has only when, at the other row counts where the two run different kernels,
    it is slower than the reference in no group that is evidence (groups of at least two records at each such row
    count, its check held in every current group) and at least ×1.10 faster in every group of one such row count or
    more. A row count at which both run the same kernel — M = 4 under a row block of 4 and of 8 — neither refuses
    nor counts. Among several that qualify, the one whose smallest gain over the row counts where it differs from
    the reference is largest. So an adopted candidate gets a row block only when one clears ×1.10 somewhere and
    regresses nowhere, and a kept row moves between row blocks (4 to 8 on M = 8 alone) or drops its own by the same
    test. The default's own kernel of M = 1 is measured at M = 4 and 8 for this, so a device where the default's
    row block loses can drop it.
  - A row whose default takes packed activations stays: the packed kernels are not measured. A subgroup candidate
    that meets the rule is named in the decision's notes and writes nothing: the subgroup form is a Session option,
    not a row.
  - The class changes the numbers of Sessions at `reorder`, inside the band, where it changes a row's lanes; a
    change of the staged flag or of the row block alone changes none.
- **Not derived**: the row-block thread target (one number for every shape, while the `full` and `row-blocks` sets
  measure a row-block height per case and row count, and no rule turns those into it) and the general sequential
  variant (it also sets the row blocks' columns and read-ahead).
- **Output**: the field (absent when nothing is adopted), the default profile's choices, one decision per sequential
  key, per parallel row and for the thread target (adopted or kept, the default's choice and the derived one, the gains
  per group that is evidence, why it was kept; for a parallel row also the row block of the default's row and of the
  derived one, and the derived row's gains at its other row counts), the groups that are not current (record, case, row count, and whether
  the case, the default's choices or a reference's row moved), the lines that state them (per record its groups —
  current and steady, unsteady, not current by cause — and its rows of current groups that are not current), the
  rule's two numbers and the records used (path, sha256, date, candidate set, user agent and those counts; the adapter
  and the case table).

## Planning the GEMV measurements

`planGemvMeasurements` of `@karume/runtime/tune` reads the same records (the same refusals) and says what they lack
for the derivation to be complete under the current runtime (the GEMM baselines read against the table given, or,
without one, against the embedded table the records' adapter selects): per case and row count of the `full` set, the
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
- The default profile's GEMV choices equal, row for row, a pinned copy — the values the GEMV kernel module held before
  they moved into the profile and the rows added since —; `nvidia-ampere`'s equal the default's with its four pinned
  exceptions after them, and no other registered table carries the field; a profile with other choices changes the
  selection and the pipeline key (core lane).
- Every per-shape exception of the sequential variant that can run — the default profile's and those of every
  registered table that carries GEMV choices: on the development machine's GPU its kernel writes the words the
  general variant writes, on the exception's own shape, on one small shape that leaves the last workgroup partial
  and a remainder after the read-ahead loop, and on one of a single word (or group) per row, which never enters the
  loop; an int4 exception at group lengths of 32 and of 64; with a host reference on a spread of columns (GPU tests).
  The rows are read from the profiles, so a row added later, or a table that starts to carry the field, is checked
  too. On another backend the evidence is that device's GEMV sweep record: its rows of the identical class are
  compared with the baseline word for word on the case's own shape.
- The row-blocked parallel GEMV: every output word equal to the parallel GEMV's at the same lanes for every storage,
  lane count, row block and row count, on the development machine's GPU and, through the workbench's `kernel-check`
  entry, on a browser's backend; a Session takes it exactly where its table's row carries a row block (GPU tests and
  core lane).
- The GEMV derivation on hand-built records: the adoption rule, the row block of a parallel row, the steadiness bound,
  the kept cases, groups as the unit of evidence (a stale, screened or skipped row is no evidence while its record's other groups are; records that each hold
  half of the cases derive what a complete pair derives; groups that are not current are reported) and the refusals
  (core lane); the generator's dry run, its write path and the carry-over of an existing `linearGemv`.
- The plan of GEMV measurements: exactly the missing candidates, and a batch the workbench's own batch reader accepts
  (core lane).
