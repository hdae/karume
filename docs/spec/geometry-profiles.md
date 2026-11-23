# Geometry profile tables

How the per-adapter geometry profile tables under `packages/runtime/src/kernels/geometry-profiles/` are made now. A
table is a static answer for one adapter, or, for a shared table, for the adapters of one vendor and architecture:
which tile geometry each op family uses. The runtime selects a table by the adapter's match and measures nothing at run
time.

## The three levels

- The runtime selects one table per adapter (`selectGeometryProfile`) by exact string equality with the adapter's
  vendor, architecture and description, in this order: (1) a table whose match names all three; (2) a table whose
  match names the vendor and the architecture and no description; (3) a table whose match names the vendor only; (4)
  the default profile. An empty description is no description, so it never reaches (1). Chrome without its
  developer-features flag reports an empty description (and device): a browser user reaches (2) to (4) only.
- The levels are named **measured**, **predicted** and **default**. A measured table is matched by the adapter's
  description and derived from that device's own records. A predicted table is matched by vendor and architecture,
  so the adapter runs geometries measured on other devices of its architecture: a shared table's
  ([below](#shared-tables)), or one device's table matched without its description — `nvidia-ampere` and
  `nvidia-blackwell` are each derived from one device's records and matched so (under Node on Dawn the description
  is the driver's version, and Chrome without its flag reports none). No table is matched by vendor alone. The
  default is the default profile: no table matches.
- Nothing is matched by similarity: a description no table names gets the predicted table of its vendor and
  architecture, never the table of the nearest measured device. The sha256 reference rows are kept per environment
  key, and each must say which table that machine ran with.

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
- Every registered table's header is the generator's. The tile geometry of `apple-metal-3` was written by a tool
  that no longer exists; its header names the commit whose file holds that tool's adoption lines.

## Shared tables

A shared table is one table derived from the sweep records of two or more devices of one vendor and architecture and
matched without a description: the predicted table of [the three levels](#the-three-levels).

- **Asked for explicitly**: `deriveGeometryProfile` with `shared: true`. It needs a vendor and an architecture, takes no description and no injection-only target, and needs the records of at least
  two adapters distinct by the four fields; every record must be of that vendor and architecture. Anything else is
  refused, naming what. Without the option nothing of this applies: records whose four fields differ are refused
  ([above](#where-a-table-comes-from)).
- **Each device is read as one adapter.** The records are grouped by their adapter's four fields. Inside one group
  every rule of a measured table applies as it does there: the group's records combine, the material and rounding
  checks drop the same observations, a row the quick look screened out is no observation, the sweep's default row
  must be the current runtime's default. No timing of one device is compared with another's, and none is pooled:
  every ratio is a geometry against its own device's default on that device.
- **The rule**, per field (a row-count bucket of `gemmRows` and of `i8a8.linear`, each other field): a geometry is
  eligible only when, on every device, it passes the measured table's rule on that device's cases of the field —
  identical output and at least the threshold (×1.05 by default) on every one of them. A geometry a device did not
  measure is not measured there. Among the eligible ones the table takes the one with the largest geometric mean of
  the per-device geometric means — devices weigh equally, whatever number of cases each measured —, ties by name. With
  none eligible the field keeps the default. A field some device has no case of keeps the default, and the verdict
  names that device.
- **Tile geometries only.** A shared table carries no `attentionOnline` and no `linearGemv`: attention-bench
  records are refused beside it (`deriveLinearGemvChoices` and `planGemvMeasurements` refuse a shared table),
  so a Session on its adapters runs the default profile's GEMV choices and the key-axis form the attention's `N`
  gives.
- **Provenance**: `provenance.adapters`, each device's four fields in the order its first record was given, in place
  of `provenance.adapter`. A family's entry cites every record with a row of it, of every device.
  `geometryProfileMismatch` compares the vendor and the architecture of each of the table's adapters with the
  adapter's — not the device or the description, which the table predicts for —, then every family's keys as for a
  measured table; a measured table is compared on all four fields as before. A list of fewer than two adapters is a
  mismatch, and `parseGeometryProfileJson` and the injection check refuse it together with a provenance that has
  both forms.
- **Regenerating**: a registered shared table is the base of a shared derivation only, onto records of the same
  adapters (as a set); each device's records cover a family they have a row of completely. A measured table is not
  the base of a shared derivation, nor a shared one of a measured derivation.
- **The verdict lines** give, per field, each device's number of cases; for an adopted geometry its geometric mean
  over the devices and each device's geometric mean, smallest and largest ratio; for every geometry not taken each
  device's ratios or the reason it is not eligible there.

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
- Every registered table carries them: `nvidia-ampere`, `nvidia-blackwell` and `apple-metal-3`. A table without them
  (an injected one, a first table derived without GEMV records, a shared table) runs with the default profile's, and
  so does an adapter no table matches: the Session fills them in once, where it selects its profile, so the linear
  selection, the fusion pass and the recipe builder all read the same choices.
- The memory estimate has no adapter and reads the default profile's. Of the choices, the fusion pass reads which
  shapes have a parallel row (the linear → `static_quantize` fusion) and a row's packed-activation flag (the packed
  pairing); no size reads the sequential variants, a row's lanes, its staged flag or its row block. No registered
  table removes a parallel row of the default, moves a packed flag or adds a packed row. Each adds rows on shapes the
  default has none for — `nvidia-ampere` 11, `nvidia-blackwell` 20 and `apple-metal-3` 13 —, so on its adapter the
  Session's fusion plan differs from the estimate's where a linear of one of those shapes feeds a `static_quantize`
  with linear at `reorder` and at most 8 rows. A bf16 linear is never fused with a following `static_quantize`, and
  the integer shapes `nvidia-ampere` and `apple-metal-3` add are among those `nvidia-blackwell` adds. No graph of a
  Gemma 4 distribution has such a linear: their plans and estimates are the same under each table
  ([limitations](../limitations.md), the section on `workspaceBytes`).
- The default profile's sequential variant is 32 columns per workgroup and 4 weight words read ahead (`c32u4`). Its
  per-shape exceptions are 22 bf16 linears of regular Gemma 4 E2B and E4B, which read 8 or 16 words ahead: `c16u8`
  on 11 shapes, `c16u16` on 10 and `c32u16` on `1536 × 256`. A bf16 word holds 8 elements where an int4 word holds
  32, so 4 words ahead keep a quarter of the bytes in flight. The derivation adopts them from GEMV sweep records of
  the RTX 3080 Ti ([perf-ledger](../perf-ledger.md) K-112), and the RTX 5070 Ti in Chrome reads each of them faster
  than `c32u4` as well. No integer-stored shape has a row there.
- The default profile has 88 parallel rows: 71 of integer-stored linears of the Gemma 4 distributions (int4, int8
  and int2) and 17 bf16 linears of regular Gemma 4 E2B (9) and E4B (8), 5 of those at 8 lanes, 7 at 16 and 5 at 32,
  two of them staged, none with a row block. The bf16 rows come from GEMV sweep records with the full candidate set
  of three devices — the RTX 3080 Ti (Node, Vulkan), the RTX 5070 Ti (Chrome, D3D12) and an Apple M2 (Chrome,
  Metal) —, each device read conservatively: its fastest baseline over its records against the row's slowest time.
  On none of the three is a row's kernel slower than the default's choice without it (the sequential GEMV at one
  row, the GEMM at 4 and 8) at any of 1, 4 and 8 rows; the smallest gain over the devices and row counts is ×1.09
  to ×3.37 ([perf-ledger](../perf-ledger.md) K-110). The other bf16 linears have no row in the default:
  `8960 × 1536`, `10752 × 2560` and `10240 × 2560` are slower than the GEMM at 8 rows on the RTX 3080 Ti (×0.69 to
  ×0.89), and when the default's rows were derived `1536 × 2048`, `12288 × 1536`, `2560 × 256` and the two output
  heads had no candidate measured at all three row counts on all three devices. The devices' tables carry rows of
  their own on some of these: `2560 × 256` on all three, at 8 lanes on `nvidia-ampere` and `nvidia-blackwell` and 4
  on `apple-metal-3`, and `1536 × 2048` on `nvidia-ampere` and `nvidia-blackwell` (below).
- A row is added to the default profile only where no measured device is slower with it: an adapter whose table has
  no GEMV choices of its own runs it. A row that one device gains from and another loses with is a row of the first
  one's table. The devices measured are the RTX 3080 Ti, the RTX 5070 Ti and an Apple M2, and each has a table with
  GEMV choices of its own, so the default's choices run on adapters no table matches (an Apple adapter whose
  description is empty among them: `apple-metal-3` matches `Apple M2`). The 17 bf16 parallel rows were adopted over
  the records of all three devices. The 22 bf16 exceptions were adopted over those of the RTX 3080 Ti and the RTX
  5070 Ti; the Apple M2's records, taken since, time the general variant against the default's exception on 13 of
  their shapes and read it ×1.01 to ×1.26 faster on 11 of them (×0.94 to ×0.99 on `2048 × 2560` and ×0.84 to ×0.87
  on `1536 × 2048`).
- `nvidia-ampere`'s field is the default profile's choices with what the derivation adopts from 24 GEMV sweep
  records of the RTX 3080 Ti (Node, Vulkan) of 2026-11-20 and 21 in place — 50 sequential decisions and 49 parallel
  ones adopted:
  - The default's 22 exceptions, in the default's order and with the default's variant: no variant reaches the ×1.10
    that replaces one.
  - 50 per-shape exceptions after them, on int4 (37), int8 (11) and bf16 (2) shapes: `c16u16` on 27, `c32u16` on
    14, `c16u4` on the int8 output heads and the bf16 output head of regular E2B, `262144 × 1536`, `c16u2` and
    `c16u1` on two int4 output heads each, `c16u8` on the 12B drafter's int4 output head `262144 × 1024` and `c32u8`
    on bf16 `10240 × 2560` (the smallest gains of the 50 ×1.10 to ×1.88). The bf16 output head of regular E4B,
    `262144 × 2560`, has none: against the general variant's repeated fastest time no candidate reaches ×1.10
    (`c16u4` ×0.97 to ×1.00). Of the 50, 32
    are on shapes `nvidia-blackwell` has an exception for too, 6 of them with the same variant.
  - 49 parallel rows apart from the default's. 38 of the default's 88 rows are changed — 37 integer-stored and bf16
    `512 × 1536`, at 32 lanes in place of 16 (×1.20 at one row, not slower at 4 and 8): other lanes on 14, the staged
    flag set on 14, another row block on 36 (34 without one take one — 2 on 22, 4 on 9, 8 on 3 — and two go from 2
    to 4). 11 rows are on shapes the default has none for: int4 4 and int8 5 (shapes of the MTP drafters), and bf16 2
    — `1536 × 2048` at 32 lanes (×3.54 faster than the sequential GEMV at one row, ×6.10 to ×6.12 than the GEMM at 4
    rows and ×3.41 to ×3.42 at 8) and `2560 × 256` at 8 (×1.10 to ×1.11, ×2.73 to ×2.74 and ×1.64), plain, in two
    records at each row count. Of the 49, 27 gain at one row (×1.10 to ×7.11) and 22 change only the row block.
  - With linear at `reorder`, a Session on that adapter adds in another order than one on another adapter on the 14
    rows with other lanes and on the 11 added ones, inside the band of `linear`. On the other 24 changed rows it
    writes the words of the default's row: the staged form and the row-blocked form give the parallel GEMV's words
    at the same lanes.
- `nvidia-blackwell`'s field is the default profile's choices with what the derivation adopts from 84 GEMV sweep
  records of the RTX 5070 Ti in Chrome (D3D12) in place:
  - The default's 22 exceptions, in the default's order; two of them with the variant the records adopt over the
    default's: bf16 `1536 × 256` runs `c16u8` in place of `c32u16` (×1.19 to ×1.21), and bf16 `4096 × 2560`
    `c16u16` in place of `c16u8` (×1.11 to ×1.12). On the other 20 no variant reaches the ×1.10 that replaces an
    exception.
  - 41 per-shape exceptions after them, on int4 (32) and int8 (9) shapes, which read 8 or 16 words ahead: `c16u8` on
    16 shapes, `c32u8` on 15, `c16u16` on 9 and `c32u16` on one.
  - 62 parallel rows apart from the default's. 41 of the default's 71 integer-stored rows are changed: other lanes
    on 22, the staged flag set on 31, another row block on 6 (three lose theirs, two go from 4 to 8 and one from 2 to
    4). One of its 17 bf16 rows is changed, `2560 × 4096`: 32 lanes, staged, in place of 16 (×1.22 to ×1.23 at one
    row, not slower at 4 and 8); on the other 16 no candidate reaches ×1.10. 20 rows are on shapes the default has
    none for: int4 8 and int8 5 (shapes of the MTP
    drafters), the int2 output heads `262144 × 1536` and `262144 × 2560`, and bf16 5 — `8960 × 1536`,
    `1536 × 2048`, `12288 × 1536`, `10240 × 2560` and `2560 × 256`. Of these 5, 3 have 8 lanes, one 16 and one 32,
    two of them staged; each is ×2.06 to ×4.1 faster than the default's choice at one row (the sequential GEMV),
    ×1.92 to ×4.1 at 4 rows and ×1.04 to ×2.20 at 8 (the GEMM), in four records or more at each row count.
  - With linear at `reorder`, a Session on that adapter adds in another order than one on another adapter on the 23
    rows with other lanes and on the 20 added ones, inside the band of `linear`. On the other 19 changed rows it
    writes the words of the default's row: the staged form and the row-blocked form give the parallel GEMV's words
    at the same lanes.
- `apple-metal-3`'s field is the default profile's choices with what the derivation adopts from 14 GEMV sweep
  records of an Apple M2 in Chrome (Metal) in place — 80 sequential decisions and 65 parallel ones adopted:
  - The default's 22 exceptions, in the default's order; 15 of them with the variant the records adopt over the
    default's, of 32 to 256 columns per workgroup (`c128u8` on 5, `c128u16` on 4, `c256u8` on 4, `c64u2` on
    `1536 × 256` and `c32u2` on `2560 × 256`, ×1.10 to ×1.54). On the other 7 no variant reaches ×1.10.
  - 65 per-shape exceptions after them, on int4 (55) and int8 (10) shapes, all of 128 or 256 columns per workgroup:
    `c128u4` on 48, `c256u4` on 14 and `c128u8` on 3 (the smallest gains of the 80 adopted exceptions ×1.10 to
    ×2.54).
  - 65 parallel rows apart from the default's. 52 of the default's 88 rows are changed — 48 integer-stored and 4
    bf16 (`256 × 1536` and `256 × 2560` from 32 lanes to 16, `512 × 1536` from 16 to 8, `1536 × 256` from 8 to 4):
    other lanes on 35, the staged flag cleared on 16, another row block on 42 (39 without one take one — 2 on 23, 4
    on 14, 8 on 2 — and three go from 2 to 4). 13 rows are on shapes the default has none for: int4 6 and int8 5
    (shapes of the MTP drafters), and bf16 2 — `8960 × 1536` (8 lanes) and `2560 × 256` (4 lanes), each ×1.15 to
    ×2.20 faster than the sequential GEMV at one row, ×4.16 to ×6.32 than the GEMM at 4 rows and ×2.68 to ×3.51 at
    8, in four records at each row count. Of the 65, 48 gain at one row (×1.10 to ×11.2) and 17 change only the row
    block.
  - With linear at `reorder`, a Session on that adapter adds in another order than one on another adapter on the 35
    rows with other lanes and on the 13 added ones, inside the band of `linear`. On the other 17 changed rows it
    writes the words of the default's row.
  - An exception is adopted only where its words equal the baseline's in every record, on Metal
    ([limitations](../limitations.md), the section on the linear GEMV family).
- A parallel row is reached only by a Session that runs linear at `reorder`. Of regular (bf16) Gemma 4's presets,
  `bf16-fast` declares that level and `bf16`, the default, does not, so a Session of `bf16-fast` runs the bf16 rows
  and one of `bf16` does not. `bf16-fast` is in the recipe and the local mirror, not yet in the published repository.
- An exception names a storage and a shape, has no group length, and applies at one row: a row block takes the
  general variant. Its words are the general variant's, so a Session runs it at `exact` too.
- A table carries the field only as a whole: the default's choices with the derived ones in place. Its provenance has
  no entry for the field (the profile's type holds none); the records it came from are in the table's header. So a
  table holds the default's parallel rows, thread target and general variant as they were when it was generated, and
  a change of those in the default profile regenerates the table (a core-lane test compares each such table with the
  default's: `nvidia-ampere`'s row for row, `nvidia-blackwell`'s and `apple-metal-3`'s by what they inherit and by
  the counts of what they change and add).
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

- **Inputs**: any number of `karume-gemv-sweep/4` records of one adapter — they agree on its four fields, the table's
  match selects it and, when the table has provenance, its vendor, architecture and device are the table's (not its
  description: under Node on Dawn that is the driver's version, and a table takes records made after a driver
  update) —, with GPU timestamps (`ns` or `deno-raw-tick`)
  that are not quantised. A record of an older format (`karume-gemv-sweep/3`, `/2`, `/1`) is refused with a message that
  names the format, and nothing converts it; so are a wall-clock or quantised record, a record of another adapter, the
  same record twice and no record at all.
- **The unit of evidence is a group**: one record's measurement of one case at one row count. A row is *current* when
  the current runtime has its candidate at that case and row count and builds the same kernel for it (the row's
  `kernel` fingerprint). A group is current when its case is the current runtime's (the record's case fingerprint, and
  the row count in the case table). Which choices its summary names as the default's is not read for it: a record made
  under older default choices stands wherever it timed the kernels the current default runs. So a change to one
  kernel, a case added or a default choice moved leaves the other rows' evidence standing, and records that each hold
  part of the cases add up.
- **The reference**: a candidate is read against the kernel the current default profile runs for its class at the
  case and row count — the identical class against the choice at `exact` (the baseline), the reorder class against
  the choice at `reorder`. A current group is usable for a class when the reference's row in it is present, has no
  error, is timed and is current; for the identical class, where the reference is not the record's own baseline, its
  row must also have its words equal to that baseline's (then candidate, record baseline and reference write the same
  words). The current kernel of a GEMM reference is the one the table the records derive for builds: a GEMM measured
  with another tile geometry (under another profile) is no reference, and the reason names the record's profile and
  the table. A record that does not name its profile is read by the same comparison of kernel fingerprints. A record's
  baseline that is not the current reference is a candidate like any other, whose words are its own.
- **The reference's time**: at a case and row count, the fastest `perDispatch` of the reference kernel over every
  current group of the given records in which its row is current, timed and without error. It is a time only when at
  least two different records hold a time within ×1.10 of it (`GEMV_REFERENCE_REPEAT`); otherwise no group at that
  case and row count is evidence for the class, and the reason says the fastest time is not repeated. On a device
  whose reference kernel runs in two timing modes, a bound on the spread of one record's rounds discards groups whose
  fastest round is the device's fastest (182 of 265 such groups on the RTX 3080 Ti) and admits groups that stayed slow throughout, which overstate a gain (bf16
  262144 × 1536 at one row on the RTX 3080 Ti: ×2.78 against a record's baseline that stayed slow, ×1.67 against the
  device's fastest). Against the repeated fastest time a slow record can only lower a candidate's gain. The bound
  comes from the records of four devices: of the baseline kernels timed in two or more records, 210 of 222 repeat
  their fastest time within ×1.10 on the RTX 3080 Ti (36 records of 2026-11-18/19), 344 of 345 on the Apple M2, 256
  of 258 on the Apple M5 and 367 of 367 on the RTX 5070 Ti. The summaries' `baselineSpread` is not read.
- **Evidence**: a group is evidence for a candidate when it is current, usable for the candidate's class, the
  reference's time at its case and row count is repeated, and the candidate's row is current and timed (a row the
  quick look screened out, or one the sweep skipped because the same record's M = 1 group left the candidate
  unadoptable, is never evidence; a skipped row that is a baseline, has a time or is also screened is refused). The
  candidate's gain in it is the reference's time ÷ the row's `perDispatch` (the minimum of its rounds ÷ `reps`).
- **Adoption**: a candidate replaces the default profile's choice for a shape only when, at M = 1, groups of at least
  two different records are evidence for it, it is at least ×1.10 faster than that choice in every one, and its check
  held in every current group, evidence or not (identical class: its words equal the record's baseline's; reorder
  class: inside the band); a row that failed in a current group rules it out too. Among several, the one whose
  smallest gain is largest.
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
  derived one, and the derived row's gains at its other row counts), the groups whose case is not current (record,
  case, row count and how the case differs), the lines that state them (per record its groups — current and not
  current —, the current groups without a usable reference at `exact` and at `reorder` (a choice the sweep does not
  measure, a row with packed activations, is not counted) and its rows of current groups that are not current), the
  rule's two numbers (the threshold ×1.10 and the repeat factor ×1.10) and the records used (path, sha256, date,
  candidate set, user agent and those counts; the adapter and the case table).

## Planning the GEMV measurements

`planGemvMeasurements` of `@karume/runtime/tune` reads the same records (the same refusals) and says what they lack
for the derivation to be complete under the current runtime (the GEMM baselines read against the table given, or,
without one, against the embedded table the records' adapter selects): per case and row count of the `full` set, the
candidates the derivation reads there — at M = 1 every candidate, at M = 4 and 8 the reorder class, on the records'
gates (subgroups, grid limit; records whose gates differ are refused) — that fewer than two records could be evidence
for: a current group, usable for the candidate's class, whose reference's time is repeated, with a current row of the
candidate. A row the quick look screened out counts as looked at (measured again, it would be screened again), and so
does a row the sweep skipped at M = 4 or 8 (a run that measures the case's M = 1 again would skip it again); a row
that failed does not. A record made under older default choices counts where it timed the current reference. The
default's choice at `reorder` is no candidate of the list, and neither is the baseline: every measurement of a group
takes both (a narrowed `gemv-sweep` step measures the references whatever candidates it names). So where a
reference's fastest time is not repeated, every candidate read against it at that case and row count is missing, and
the step that measures them measures the reference again. A candidate whose reference is a choice the sweep does not
measure (a row with packed activations) is not listed: no measurement makes it evidence.

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
- `apple-metal-3` has no such case: every record its tile geometry's provenance cites has a format the derivation no
  longer accepts.
- The shared derivation on hand-built records (core lane): a geometry faster on one device and slower on another is
  not taken; among the geometries eligible on every device the largest geometric mean of the per-device geometric
  means is taken (devices weigh equally, not cases); one slow case on one device rules a geometry out; a field one
  device has no case of keeps the default and names the device; the provenance's adapters and the family entries;
  the refusals (one adapter, a description, no architecture, an injection-only target, attention-bench records) and
  the measured derivation still refusing records of two adapters; a shared table as base; matching, the JSON reader
  and the injection check of the several-adapter form; `deriveLinearGemvChoices` refusing a shared table.
- The skeleton form and the counter decode as candidates (core lane): what each case measures in each set, that no
  case measures one kernel under two names, the derivation adopting a tile with a field and refusing a field where no
  kernel takes it, and a table that carries the fields written by the generator, read back, and matching its adapter
  by the fingerprints of the kernels it runs.
- The default profile's GEMV choices equal, row for row, a pinned copy — the values the GEMV kernel module held before
  they moved into the profile, without its one per-shape exception, and the rows added since —; every registered
  table carries the field; `nvidia-ampere`'s equal, row for row, the default's with its 38 pinned changed parallel
  rows in place and its 50 pinned exceptions and 11 pinned parallel rows after them; `nvidia-blackwell`'s and
  `apple-metal-3`'s hold the default's general variant and thread target, the default's 22 exceptions in the
  default's order — each with the default's variant except the ones the case names with the table's (two and 15) —,
  the counted exceptions of their own after them (41 and 65), and every parallel row of the default in the default's
  order, a counted number of them changed (42 and 52), with the counted rows added (20 and 13); no registered table
  removes a parallel row of the default, moves a packed flag or adds a
  packed row; a profile with other choices changes the selection and the pipeline key (core lane).
- Every per-shape exception of the sequential variant that can run — the default profile's and those of every
  registered table that carries GEMV choices: on the development machine's GPU its kernel writes the words the
  general variant writes, on the exception's own shape, on one small shape that leaves the last workgroup partial
  and a remainder after the read-ahead loop, and on one of a single word (or group) per row, which never enters the
  loop; an int4 exception at group lengths of 32 and of 64; with a host reference on a spread of columns (GPU tests).
  The rows are read from the profiles, so a row added later, or a table that starts to carry the field, is checked
  too. On another backend the evidence is that device's GEMV sweep record: its rows of the identical class are
  compared with the baseline word for word on the case's own shape.
- A table's parallel rows select among the kernels of the parallel GEMV, which the GPU tests of that family check
  per storage and lane count on shapes of their own (the band against the f64 reference, and the staged and
  row-blocked forms word for word against the plain one at the same lanes). No test runs a table's rows on their
  own shapes: that evidence is the device's GEMV sweep records, where every row of the reorder class is held to the
  band of `linear` against an f64 oracle on the case's shape, at 1, 4 and 8 rows.
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
