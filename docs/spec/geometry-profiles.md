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
- The Apple tables: `apple-metal-3` is measured, matched by `Apple M2`; `apple-m5` is measured, matched by
  `Apple M5`; `apple-metal-3-shared` is predicted, matched by `apple` / `metal-3`, for every such adapter no
  table names by its description — the M2 and the M5 in Chrome without the flag, and every Apple device of that
  architecture not measured (an adapter that reports `Apple M3` among them).
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
  ops is refused (a shared table keeps the default for such a family instead — [below](#shared-tables)).
- A new adapter's first table has no base and needs its adapter match on the command line.
- A shared table is derived only when the command asks for one (`--shared`, [below](#shared-tables)). Without it the
  records must agree on all four adapter fields (vendor, architecture, device, description): records of two devices
  are refused, whatever the match.
- The GEMV choices of linear come from GEMV sweep records ([below](#deriving-the-gemv-choices)), given next to the
  sweep records or alone for a registered table. Without them, a base table's `linearGemv` is carried as it is,
  also through a regeneration that derives only the tile geometry.
- A whole table is never re-derived from all the records it ever came from, and a record of an older format is never
  rewritten to pass. The records cited by a family of an older format stay cited until that family is swept again.
- A sweep record's row the quick look screened out (`screened`, [workbench.md](workbench.md#the-entries)) has no
  ratio and no output match: it is no observation, so a geometry screened out in a case that no record timed it in is
  not adopted for that case's class (its verdict says so).
- A record of the `explore` set ([workbench.md](workbench.md#the-geometry-sweeps-explore-set)) is read by the same
  rules, as a `full` record that holds part of the candidates: its case-set versions and default fingerprints are
  `full`'s, and its candidate fingerprint covers the rows that are not skipped. A skipped row (`skipped`: an earlier
  case of the class ruled the geometry out in that sweep) is no observation, as a screened row is — neither a failure
  nor a mismatch —, so the geometry is not adopted for the class unless another record timed it in that case; the
  verdict says `not measured in <case> (the explore set dropped it for the class)`. The table's provenance names the
  set (`candidateSet: "explore"`). A skipped default row, a row both skipped and screened, and a candidate set that is
  neither a candidate set nor `explore` are refused, naming the record.

## The skeleton form and the counter decode

- A geometry of the f32 GEMM can carry two fields beside its tile: the skeleton form (`form`: `"tb-un"` or
  `"tas-tb-un"`) in the row-count rules of linear / matmul / bmm (`gemmRows`) and in `conv2d`, and the counter decode
  (`decode: "counter"`) in `conv2d` only. The fused attention's fields take neither. The output words are the same
  with and without them; they are a choice of speed, and the pipeline key names them.
- The sweep's candidate sets `quick+` and `full` measure them (and `explore`, which measures `full`'s candidates). A
  linear / matmul / bmm case measures every tile of the set and then the same tile with each form (three geometries per
  tile). A conv2d case measures every tile, the tile with each form, with the decode, and with the decode and each form
  (six per tile). `quick` measures the tiles alone.
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
- Over the sweep's 78 cases `quick+` forms 1,656 rows (896 with the tiles alone) and `full` 10,040 (6,269 with the
  tiles alone), counted as the sweep forms them: per case the default's row and every candidate of the set by
  another name. The fields change no case set, default kernel or fingerprint, so a registered table matches whether
  or not its family was swept with them.
- `nvidia-blackwell` carries them: a form in its four `gemmRows` rules above 64 rows, and a form with the decode in
  both `conv2d` fields. `apple-m5` carries a form in its five `gemmRows` rules from 17 to 512 rows and a form with the
  decode in both `conv2d` fields; `apple-metal-3-shared` a form in its rule of 33 to 64 rows and a form with the
  decode in both `conv2d` fields. `nvidia-ampere` and `apple-metal-3` carry none; a table takes one when its family is
  derived from a sweep that measured it.
- `quick+` and the evaluation set `k88` take every registered table's geometries, so registering a table can grow
  them: 152 of the 1,656 rows of `quick+` come from `apple-m5` and `apple-metal-3-shared` (1,504 without them), and
  3 of the 213 rows `k88` forms over its own 19 cases (210 without them).

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
- A shared table's header also says from which devices it predicts (each by its description and device id) and that it
  is matched without a description, lists the records under the device each is of, states the shared rule, and its
  verdict lines give each device's ratios ([below](#shared-tables)).
- Every registered table's header is the generator's. The tile geometry of `apple-metal-3` was written by a tool
  that no longer exists; its header names the commit whose file holds that tool's adoption lines.

## Shared tables

A shared table is one table derived from the sweep records of two or more devices of one vendor and architecture and
matched without a description: the predicted table of [the three levels](#the-three-levels).

- **Asked for explicitly**: `deriveGeometryProfile` with `shared: true`, the generator with `--shared`. It needs a
  vendor and an architecture, takes no description and no injection-only target, and needs the records of at least
  two devices; every record must be of that vendor and architecture. Anything else is refused, naming what. Without
  the option nothing of this applies: records whose four fields differ are refused
  ([above](#where-a-table-comes-from)).
- **A device is told apart by its description.** Every record must carry a non-empty description: a record without
  one is refused, since it cannot be told apart from another device's (the same machine in Chrome reports its
  description and device with the developer-features flag and neither without it, so by the four fields it would
  be two adapters). The records are grouped by description, and the records of one description must agree on all
  four fields: one that names another device id is refused, as not of one adapter. At least two descriptions are
  needed.
- **Each device is read as one adapter.** Inside one group every rule of a measured table applies as it does there:
  the group's records combine, the material and rounding
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
  records, GEMV sweep records and `--gemv-plan` are refused beside it (`deriveLinearGemvChoices` and
  `planGemvMeasurements` refuse a shared table), so a Session on its adapters runs the default profile's GEMV
  choices and the key-axis form the attention's `N` gives.
- **Provenance**: `provenance.adapters`, each device's four fields in the order its first record was given, in place
  of `provenance.adapter`. A family's entry cites every record with a row of it, of every device.
  `geometryProfileMismatch` compares the vendor and the architecture of each of the table's adapters with the
  adapter's — not the device or the description, which the table predicts for —, then every family's keys as for a
  measured table; a measured table is compared on all four fields as before. A list that names fewer than two
  devices — fewer than two adapters, an adapter without a description, or two adapters of one description — is a
  mismatch, and `parseGeometryProfileJson` and the injection check refuse it, as they refuse a provenance that has
  both forms.
- **Regenerating**: a registered shared table is the base of a shared derivation only, onto records of the same
  adapters (as a set). A family the records cover but one device has no row of keeps the default, so it is derived
  only where the base holds the default for it too, and refused where the base holds another geometry (it would
  silently go back to the default). A measured table is not the base of a shared derivation, nor a shared one of a
  measured derivation.
- **A family one device covers only in part** keeps the default on a base: when a device has a row of the family and
  some case of the family's current case set has no usable row of the default geometry on that device — no row at all,
  or only failed default rows (a case some record of the device has a default row of that did not fail is covered) —
  every field of the family keeps the default, and each field's verdict line names the device and the cases, marking
  each `no row` or `the default's row failed`. The generated header says so beside the adoption rule. A family every
  device covers in full is derived as above, and a candidate's failure in a case a device did measure counts against
  it as anywhere. The protection is the measured table's refusal of such records (no choice goes into a shape some
  device did not measure), kept as a recorded default: a device can fail to give the case at all — the Apple M2's
  default kernel of an `i8a8-conv3d` case exceeds Metal's command-buffer time limit. The rule is provisional until the
  sweep survives such a device loss (ADR 0156 addendum 2026-11-24). Without a base, a shared derivation judges such a
  family on the cases at hand, as a measured one does.
- **The verdict lines** give, per field, each device's number of cases; for an adopted geometry its geometric mean
  over the devices and each device's geometric mean, smallest and largest ratio; for every geometry not taken each
  device's ratios or the reason it is not eligible there.
- `apple-metal-3-shared` is derived from the Apple M5's full sweep of 2026-11-23 and five Apple M2 records: three of the
  same day that time the geometries the M5 adopts, and the M2's full sweep in two parts, both stopped by a device lost
  to Metal's command-buffer time limit (2026-11-23: 39 of 78 cases; 2026-11-24: 35 of 37). It takes six
  fields: `gemmRows` 17 to 32 rows `reg32x32r2x4w8` (M5 ×1.260, M2 ×1.309) and 33 to 64 rows `reg64x32r4x4w8` (×1.532,
  ×1.485), the `conv2d` fields `igemm64x64:wg16x8:tas-tb-un:dec` (×1.368, ×1.622) and `igemm32x64:wg16x4:dec` (×1.434,
  ×1.865), and `i8a8.linear` up to 256 rows `tile16x64r4x4w16x4k32` (×2.188, ×2.275) and 257 to 512 rows
  `tile64x64r8x4w16x8k32` (×1.072, ×1.116). In every other field but one no geometry passes on both devices — the fused
  attention and the i8a8 attention among them. `i8a8.conv3d` keeps the default because the M2 covers `i8a8-conv3d`
  only in part: the default row of `i8a8-conv3d-cin1024-cout512-k333-in6x128x128` failed (the device was lost) and
  `i8a8-conv3d-cin512-cout256-k333-in6x256x256` has no row.

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
  quick look that times in full only the references (the device table's choices among them) and the kernels near
  enough to matter; at M = 4 and 8 it skips the reorder candidates that the case's M = 1 group of the same run leaves
  unadoptable at the lowest margin the rule below is meant to be re-decided at (×1.03)
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
- `nvidia-ampere`, `nvidia-blackwell`, `apple-metal-3` and `apple-m5` carry them; `apple-metal-3-shared` (a shared
  table carries tile geometries only) does not. A table without them
  (an injected one, a first table derived without GEMV records, a shared table) runs with the default profile's, and
  so does an adapter no table matches: the Session fills them in once, where it selects its profile, so the linear
  selection, the fusion pass and the recipe builder all read the same choices.
- The memory estimate has no adapter and reads the default profile's. Of the choices, the fusion pass reads which
  shapes have a parallel row (the linear → `static_quantize` fusion) and a row's packed-activation flag (the packed
  pairing); no size reads the sequential variants, a row's lanes, its staged flag or its row block. No registered
  table removes a parallel row of the default, moves a packed flag or adds a packed row. Each adds rows on shapes the
  default has none for — `nvidia-ampere` 11, `nvidia-blackwell` 20, `apple-metal-3` 13 and `apple-m5` 17 —, so on
  its adapter the Session's fusion plan differs from the estimate's where a linear of one of those shapes feeds a
  `static_quantize` with linear at `reorder` and at most 8 rows. A bf16 linear is never fused with a following
  `static_quantize`, and the integer shapes `nvidia-ampere`, `apple-metal-3` and `apple-m5` add are among those
  `nvidia-blackwell` adds, except `apple-m5`'s int4 `262144 × 2560`: the output head of Gemma 4 E4B q4_0 QAT, which
  no `static_quantize` follows (its recipe inserts none around the head). No graph of a Gemma 4 distribution has such
  a linear: their plans and estimates are the same under each table ([limitations](../limitations.md), the section
  on `workspaceBytes`).
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
  one's table. The default's rows come from the records of the RTX 3080 Ti, the RTX 5070 Ti and an Apple M2; those
  three and the Apple M5 each have a table with GEMV choices of its own, so the default's choices run on adapters no
  table matches and on those whose table carries none (`apple-metal-3-shared` — every Apple adapter whose
  description is empty among them: `apple-metal-3` matches `Apple M2` and `apple-m5` `Apple M5`). The 17 bf16
  parallel rows were adopted over
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
- `apple-m5`'s field is the default profile's choices with what the derivation adopts from two GEMV sweep records of
  the Apple M5 in Chrome (Metal) of 2026-11-23, each of the `full` set — 43 sequential decisions and 66 parallel
  ones adopted:
  - The default's 22 exceptions, in the default's order and with the default's variant: no variant reaches the ×1.10
    that replaces one.
  - 43 per-shape exceptions after them, on int4 (39) and int8 (4) shapes: `c32u2` on 15, `c16u2` on 11, `c64u2`,
    `c256u2` and `c16u4` on 4 each, `c16u8` and `c32u8` on 2 each and `c128u8` on one (the smallest gains of the 43
    ×1.10 to ×1.77).
  - 66 parallel rows apart from the default's. 49 of the default's 88 rows are changed — int4 42, int8 6 and int2 1,
    no bf16 row —, each with another row block (37 without one take one — 2 on 8, 4 on 29 — and 12 go from 2 to 4
    or, on one, to 8), 18 of them with other lanes, 4 with the staged flag set and 3 with it cleared. 17 rows are on
    shapes the default has none for: int4 6 and int8 5 (shapes of the MTP drafters, and the int4 output head of E4B
    QAT), and bf16 6 — `8960 × 1536`, `12288 × 1536`, `10752 × 2560` and `10240 × 2560` at 32 lanes, staged, and the
    output heads `262144 × 1536` (16 lanes, staged) and `262144 × 2560` (16 lanes) —, each ×1.24 to ×1.33 faster than
    the sequential GEMV at one row, ×2.00 to ×2.43 than the GEMM at 4 rows and ×1.09 to ×1.51 at 8, in both records
    at each row count. Of the 66, 35 gain at one row (×1.11 to ×10.2) and 31 change only the row block.
  - With linear at `reorder`, a Session on that adapter adds in another order than one on another adapter on the 18
    rows with other lanes and on the 17 added ones, inside the band of `linear`. On the other 31 changed rows it
    writes the words of the default's row.
  - The records' verdict is a failure: 1,466 of the 3,524 sequential-variant rows they compare with the baseline at
    one row write other words than the baseline on this GPU — int2, int4 and int8 cases at 1, 2, 8 and 16 words read
    ahead; none at 4 and no bf16 row. An exception is adopted only where its words equal the baseline's in every
    record, so none of those is.
- A parallel row is reached only by a Session that runs linear at `reorder`. Of regular (bf16) Gemma 4's presets,
  `bf16-fast` declares that level and `bf16`, the default, does not, so a Session of `bf16-fast` runs the bf16 rows
  and one of `bf16` does not. `bf16-fast` is in the recipe and the local mirror, not yet in the published repository.
- An exception names a storage and a shape, has no group length, and applies at one row: a row block takes the
  general variant. Its words are the general variant's, so a Session runs it at `exact` too.
- A table carries the field only as a whole: the default's choices with the derived ones in place. Its provenance has
  no entry for the field (the profile's type holds none); the records it came from are in the table's header. So a
  table holds the default's parallel rows, thread target and general variant as they were when it was generated, and
  a change of those in the default profile regenerates the table (a core-lane test compares each such table with the
  default's: `nvidia-ampere`'s row for row, `nvidia-blackwell`'s, `apple-metal-3`'s and `apple-m5`'s by what they
  inherit and by the counts of what they change and add).
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
- **Two rules** (the maintainer, 2026-11-28). A decision — a sequential key, a parallel row — one of whose cases a
  *steady* record measured at M = 1 with a usable reference for the class is decided by **the rule of one steady
  record**, over the steady records; any other decision by **the rule of two records** (below), over every record. A
  failure or a failed check in a current group of any record rules a candidate out under either. The numbers' basis is
  [the research note](../research/2026-11-28-gemv-adoption-rule.md) (the GEMV sweep records of 2026-11-18 to 27 of
  the RTX 5070 Ti, the Apple M2, the Apple M5 — also as `apple-metal-3` without its description — and the RTX 3080 Ti).
- **The class of a record**, from its summaries (`gemvSteadiness`, the same function the sweep states its record's
  class with — the derivation does not read that statement): *steady* when at least 20 of its groups have a baseline
  spread and at most 20 % of those have `baselineSpread` above ×1.05; *unsteady* when more do; *unclassified* — read as
  unsteady — with fewer than 20 such groups. Every group counts, at every row count. Of the records of 20 groups or
  more, the 35 of three devices (RTX 5070 Ti 13, Apple M2 12, Apple M5 10) read 0 to 15.2 % and the 36 of the RTX
  3080 Ti 31.8 to 84.6 %; no record lies between, so 20 % is a point inside an empty band. Below 20 groups the test
  fails both ways: six RTX 3080 Ti records of 12 or 13 groups read 0 to 16.7 % (and the rule of one record adopted 2 of
  74 candidates there that other records read slower), four records of steady devices with 2 to 4 groups read 25 to
  50 %. A record whose timestamps are Deno's raw ticks is read by the rule of two records whatever its class: the
  length of its passes in ms is not known.
- **The rule of one steady record** reads, in each steady record's current group, a kernel's time (`perDispatch`), its
  *gap* — the second-smallest of its positive rounds ÷ the smallest — and, for a reference, its re-timing
  (`retimedPerDispatch` ÷ `perDispatch`). A kernel whose pass (its smallest round) is shorter than 20 ms, or that has
  fewer than two positive rounds, has no time there (3 and 5 such rows in the data, 3 of them wrong; between 20 ms and
  the 80 ms target the gap term widens the bar by itself). A *reading* of a kernel against another in one group is the
  other's time ÷ its own (the gain), both gaps, the other's re-timing and the *break-even margin*, gain ÷ (gap × other's
  gap): the largest margin at which the gain clears its bar. A reading is evidence at a margin unless the other kernel's
  re-timing, or its inverse, is above margin × the two gaps — a reference whose time moved within the group by more
  than the bar a gain must clear can account for such a gain by itself (no record of today repeats a kernel within a
  group, so this reads only records taken from 2026-11-28 on; a record without the re-timing has nothing to test).
  - *At M = 1* a candidate qualifies when its check held, at least one steady record's reading of it against its class's
    reference (the default's choice, its row in the same group — its time need not repeat in a second record) is
    evidence, and every such reading's gain is at least the margin × its gap × the reference's gap. The margin is a
    parameter of the derivation (`DeriveLinearGemvOptions.margin`), ×1.065 by default (`GEMV_MARGIN`): the least
    margin that covers 99 % of the gains' differences between records of one device is ×1.060, ×1.066 and ×1.058 on
    the three steady devices (M = 1, table candidates, 3,633, 4,414 and 8,338 rows), and without the gap term it
    would be ×1.063, ×1.083 and ×1.080. Against a reference that is the default's choice, 0 of 1,481 adoptions is
    slower than ×1/1.02 by the other records (0 of 1,051 by records of another day), and 15 of 1,369 clear winners
    are refused where the rule of two records refuses 135 of 1,108. ×1.065 bounds the error between records of one
    session; it is no bound across sessions: across days it is exceeded in 0.7 % (Apple M5), 1.8 and 3.8 % (RTX 5070
    Ti) and 2.5 % (Apple M2) of the rows, because single kernels move between sessions by more than ×1.065 in 0.6 to
    0.9 % of the cases with nothing to see inside a record. The reference's re-timing is evidence of its own: in
    today's data only the reference alone moved by more than ×1.065 in 7 of 2,932 groups, all at M = 1.
  - *At the other row counts of a parallel row* (M = 4 and 8) the kernel the derived row runs there must have a time
    in every steady record whose reading at M = 1 was evidence — the same record —, and in every reading there that is
    evidence it is not slower than the default's choice in the same group: a gain of at least 1, no margin either way.
    That bar passes 3 of 1,299 rows that other records read slower than ×1/1.02 and refuses 3 of 10,684 they read at
    least ×1.02 faster; 1 ÷ (margin × gaps) would pass 37 to 51 % of the slower rows, margin × gaps refuse 5.5 to 7.6 %
    of the faster ones.
  - *The row block* of a candidate's row (or of a kept row: below) replaces its reference row's — the candidate without
    one, or the kept row with its own — when, at the other row counts where the two run different kernels, each has a
    reading that is evidence, every such ratio of the two kernels in the same group is at least 1, and at one row count
    or more every one is at least the margin × the two kernels' gaps (the reference's time cancels in the ratio, so the
    noise is the two gaps). Among several, the one whose smallest ratio is largest.
  - *Several candidates*: among those that qualify, the one whose smallest gain at M = 1 is largest. Two records of one
    device pick different winners in 5.2, 20.3 and 13.0 % of the pairs (6.9, 16.7 and 9.9 % under the rule of two
    records), mostly near ties.
  - *The table's choice*: where the table the records derive for holds another choice than the default's for the
    shape (a sequential exception, a parallel row, or no row where the default has one), that choice stays unless a
    candidate that qualifies — or the default's choice itself — beats it by the bar of M = 1 in the same group, in
    every steady record that times both, the largest gain over the default's choice then taken; a table's choice that
    no steady record timed at M = 1 stays as it is, its row block too; one that failed or whose check failed in a
    current group is no longer held, and the default's choice is the one to beat. With the table's choice kept so, 0 of
    544, 2 of 321 and 0 of 439 choices move between records of one device, and the time behind the best by further
    records falls from 0.12 / 0.88 / 0.66 % to 0.05 / 0.24 / 0.23 %. A kept row may still move its row block by the
    rule above, against its own. The sweep times the device table's choices for this
    ([workbench.md](workbench.md#how-the-gemv-sweep-times-a-row-count)); a record taken before it times them only where
    they are candidates of its set.
  - *Several steady records*: a decision holds in every steady record that times it — every reading that is evidence
    clears its bar, at M = 1 and at the other row counts —, and one steady record is enough. A sequential key's cases
    must reach the same choice, or the table's stays. (The replays of the adjudication measured the same form — a
    candidate in every record read, `later_pairs.ts` — and found no harm beyond the rule of two records'.)
  - *Not in the rule*: a pair gate on the quick look's time against the final time, which on the RTX 3080 Ti made one
    unsteady record as safe as two, was measured on one unsteady device only.
- **Re-deciding**: a decision of the rule of one steady record carries its evidence (`evidence`: per case, every
  candidate a steady record timed at M = 1 with its readings against the default's choice and against the table's,
  its check, and under each row block its row's kernels at the other row counts with their readings), and the margin
  enters only where `decideOneRecord` reads them. `redecideLinearGemvChoices(derivation, margin)` decides every such
  decision again from the output alone, by the same function, builds the field again from the table's choices the
  derivation started from (`tableChoices`), and lists the decisions whose derived choice or row block changes; the
  decisions of the rule of two records stay. At the derivation's own margin it gives the derivation back.
- **The rule of two records** — **the reference's time**: at a case and row count, the fastest `perDispatch` of the reference kernel over every
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
- **Evidence** (the rule of two records): a group is evidence for a candidate when it is current, usable for the candidate's class, the
  reference's time at its case and row count is repeated, and the candidate's row is current and timed (a row the
  quick look screened out, or one the sweep skipped because the same record's M = 1 group left the candidate
  unadoptable, is never evidence; a skipped row that is a baseline, has a time or is also screened is refused). The
  candidate's gain in it is the reference's time ÷ the row's `perDispatch` (the minimum of its rounds ÷ `reps`).
- **A record of the `explore` set** ([workbench.md](workbench.md#the-explore-set)) is read by these rules alone, as
  any record that holds part of the candidates: its rows — every candidate of a sentinel at M = 1, the predicted
  candidates of another case, what M = 1 carries at M = 4 and 8 — are evidence as the rules say, and a candidate
  without a row in it has none. It measures no row-block height.
- **Adoption** (the rule of two records): a candidate replaces the default profile's choice for a shape only when, at M = 1, groups of at least
  two different records are evidence for it, it is at least ×1.10 faster than that choice in every one, and its check
  held in every current group, evidence or not (identical class: its words equal the record's baseline's; reorder
  class: inside the band); a row that failed in a current group rules it out too. Among several, the one whose
  smallest gain is largest.
- **The identical class** writes the sequential variant's per-shape exceptions (`sequential.shapes`) from M = 1. Its
  key has no group length: the cases that share a key must reach the same decision, or the key keeps the default
  (under the rule of one steady record: the table's choice). An
  adopted variant replaces the key's exception or adds one; when it is the general variant, the key's exception goes.
  It changes no number of any Session. A float-stored shape has a sequential key like any other: from records of the
  bf16 cases a derived table may carry `sequential.shapes` rows for bf16, and an `exact` Session on that adapter then
  runs another sequential variant (columns per workgroup, words read ahead) that gives the same words.
- **The reorder class** writes the parallel rows (`parallelShapes`): the lanes and the staged flag of the plain and
  staged candidates at M = 1, and the row block. A parallel row serves M = 1 to 8, and at each other row count of
  that range the case table has (M = 4 and 8: the `full` set) it runs its own kernel or, with a row block, the
  row-blocked kernel of its lanes (`staged-l<lanes>r<rows>`). Under the rule of two records, at each of those row
  counts groups of at least two records must be evidence for the kernel the derived row runs there, and it may be
  slower than the default's choice in none of them — for a float-stored shape without a row that choice is the GEMM;
  a candidate whose row does not meet this is not adopted (the rule of one steady record: above). The adopted row
  replaces the shape's row in place, or is added when the shape had none; a derived choice of no row removes the
  table's row.
  - **The row block** under the rule of two records is chosen by one rule, against a reference row of the same lanes: the default's row
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
  measure a row-block height per case and row count — the `quick` and `explore` sets measure none —, and no rule
  turns those into it) and the general sequential
  variant (it also sets the row blocks' columns and read-ahead).
- **The field** is the table's choices (the default's where the table carries none) with every decision's derived
  choice in place: so a shape no record measured keeps the table's choice, and under the rule of two records a kept
  decision writes the default's choice, as before. The general variant and the thread target are the default's. The
  field is absent when it equals the default's choices.
- **Output**: the field (absent when it is the default's choices), the default profile's choices, the table's choices
  it started from (`tableChoices`), one decision per sequential key, per parallel row and for the thread target (which
  rule decided it, `one-record` or `two-record`; adopted or kept, the default's choice, under the rule of one steady
  record the table's, and the derived one, the gains per group that is evidence, why it was kept or won; for a
  parallel row also the row blocks of the default's, the table's and the derived row, and the derived row's gains at
  its other row counts; under the rule of one steady record its `evidence`, above), the groups whose case is not
  current (record, case, row count and how the case differs), the lines that state them (per record its groups —
  current and not current —, the current groups without a usable reference at `exact` and at `reorder` (a choice the
  sweep does not measure, a row with packed activations, is not counted), its rows of current groups that are not
  current, and its class with the share and the groups it was read from and which rule reads it; per decision under
  the rule of one steady record the adopted or nearest candidate's break-even margins), the rule's numbers (`margin`,
  the class's 20 groups, 20 % and ×1.05, the 20 ms pass and 5 rounds, then the rule of two records' threshold ×1.10
  and repeat factor ×1.10) and the records used (path, sha256, date, candidate set, user agent, those counts, the class
  and whether the rule of one steady record reads it; the adapter and the case table).

## Planning the GEMV measurements

`planGemvMeasurements` of `@karume/runtime/tune` reads the same records (the same refusals) and says what they lack
for the derivation to be complete under the current runtime (the GEMM baselines read against the table given, or,
without one, against the embedded table the records' adapter selects): per case and row count of the `full` set, the
candidates the derivation reads there — at M = 1 every candidate, at M = 4 and 8 the reorder class, on the records'
gates (subgroups, grid limit; records whose gates differ are refused) — that no steady record and fewer than two
records could be evidence for: a current group, usable for the candidate's class — in a steady record (the rule of one
steady record) by its own reference row, otherwise whose reference's time is repeated —, with a current row of the
candidate. A row the quick look screened out counts as looked at (measured again, it would be screened again), and so
does a row the sweep skipped at M = 4 or 8 (a run that measures the case's M = 1 again would skip it again); a row
that failed does not. Of an `explore` record only the rows it timed count: that set carries fewer candidates from
M = 1 than a run of the `full` set, which would measure a row it skipped, and screens at ×1.10 where the `full` set
screens at ×1.25, so the `full` set may time a row it screened. The plan lists what the `full` set lacks, so
`explore` records do not complete it: a candidate an explore record did not time is missing there as in any record
without its row. A record made under older default choices counts where it timed the current reference. The
default's choice at `reorder` is no candidate of the list, and neither is the baseline: every measurement of a group
takes both (a narrowed `gemv-sweep` step measures the references whatever candidates it names). So where a
reference's fastest time is not repeated, every candidate read against it at that case and row count is missing, and
the step that measures them measures the reference again. A candidate whose reference is a choice the sweep does not
measure (a row with packed activations) is not listed: no measurement makes it evidence.

The generator writes the plan as a workbench batch ([workbench.md](workbench.md#batches)),
`inputs/workbench/batches/gemv-plan-<profile id>.json`: per list of missing candidates one `gemv-sweep` step of the
`full` set, its `cases` the cases that miss that list at the same row counts, `rows` those row counts, `candidates`
the list; the steps run twice: a step narrowed to a few cases makes a record of a few groups, and one of fewer than
20 has no class, so it is evidence only beside a second, even on a steady device. With no record at hand the batch
is the whole measurement: the `full` set, twice (one such record of a steady device would be enough). With nothing
missing no batch is written.

## The command

`node tools/geometry-profiles/generate.ts <profile id> [--records=<path>[,…]] [--attention-bench=<path>[,…]]
[--gemv-records=<path>[,…]] [--vendor=<v> [--architecture=<a> [--description=<d>]]] [--min-speedup=<x>]
[--dry-run]`

`node tools/geometry-profiles/generate.ts <profile id> --shared --records=<path>,<path>[,…]
[--vendor=<v> --architecture=<a>] [--min-speedup=<x>] [--dry-run]`

`node tools/geometry-profiles/generate.ts <profile id> --gemv-plan [--gemv-records=<path>[,…]] [--dry-run]`

- A registered id uses its table as base and its table's match; giving a match is an error. An id that is not
  registered needs `--vendor` and `--records`.
- At least one of `--records` and `--gemv-records` is given. With `--gemv-records` alone, a registered table keeps its
  tile geometry and takes the derived GEMV choices; `--attention-bench` needs `--records`. `--min-speedup` is the tile
  geometry's threshold only.
- `--shared` derives a shared table ([above](#shared-tables)): a first one needs `--vendor` and `--architecture`, a
  registered one is regenerated on its table as base. `--description`, `--attention-bench`, `--gemv-records` and
  `--gemv-plan` beside it are usage errors, and so is `--shared` without `--records`.
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
- On a machine that has the records: for `apple-m5` and `apple-metal-3-shared`, every record any of their families
  cites, with the registered table as base (a shared derivation for the shared one), and the GEMV sweep records the
  header of `apple-m5` cites, from which its `linearGemv` is derived again rather than carried, derive an object
  equal to the registered table, provenance included.
- The shared derivation on hand-built records (core lane): a geometry faster on one device and slower on another is
  not taken; among the geometries eligible on every device the largest geometric mean of the per-device geometric
  means is taken (devices weigh equally, not cases); one slow case on one device rules a geometry out; a field one
  device has no case of keeps the default and names the device; the provenance's adapters and the family entries;
  the refusals (the records of one description only, a record without a description, records of one description
  with two device ids, a description, no architecture, an injection-only target, attention-bench records) and the
  measured derivation still refusing records of two adapters; a shared table as base; on a base, a family one device
  covers only in part (a case without a row, or whose default row failed) keeping the default in every field with the
  device and the cases named, a family covered in full derived as without a base, a failed default row covered by
  another record of the device with the candidate's failure there still counted, the same records without a base
  judged on the cases at hand, and the measured derivation still refusing them; the generator's dry run and header
  naming them; matching, the JSON reader
  and the injection check of the several-adapter form, each refusing a list of fewer than two adapters, one that
  names one description twice and one with an adapter without a description; `deriveLinearGemvChoices` refusing a
  shared table; the generator's `--shared` refusals and a shared table written, its header and the module it exports. The Apple tables
  of the built-in list are selected as [the three levels](#the-three-levels) say (`Apple M2`, `Apple M5`, an empty, absent or
  unmeasured description, another architecture).
- The skeleton form and the counter decode as candidates (core lane): what each case measures in each set, that no
  case measures one kernel under two names, the derivation adopting a tile with a field and refusing a field where no
  kernel takes it, and a table that carries the fields written by the generator, read back, and matching its adapter
  by the fingerprints of the kernels it runs.
- The default profile's GEMV choices equal, row for row, a pinned copy — the values the GEMV kernel module held before
  they moved into the profile, without its one per-shape exception, and the rows added since —; the registered
  tables that carry the field are `apple-metal-3`, `nvidia-ampere`, `nvidia-blackwell` and `apple-m5`, and one
  without it runs the default's; `nvidia-ampere`'s equal, row for row, the default's with its 38 pinned changed
  parallel rows in place and its 50 pinned exceptions and 11 pinned parallel rows after them; `nvidia-blackwell`'s,
  `apple-metal-3`'s and `apple-m5`'s hold the default's general variant and thread target, the default's 22
  exceptions in the default's order — each with the default's variant except the ones the case names with the
  table's (two, 15 and none) —, the counted exceptions of their own after them (41, 65 and 43), and every parallel
  row of the default in the default's order, a counted number of them changed (42, 52 and 49), with the counted rows
  added (20, 13 and 17); no registered table
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
- The rule of one steady record on hand-built steady records (core lane), a fault per point: the class (19 groups, a
  share of 25 %), the reference of the same group, the bar of margin × two gaps and the evidence the decision carries,
  a failed check, the row's kernel at M = 4 and 8 (×0.99, no time in the same record, a gain of 1 enough), the row
  block (×1.05 refused), the largest gain, the table's choice kept unless beaten (×1.04 over it), not moved by a record
  that did not time it, given up for the default's where that beats it, kept on a shape no record measured, an
  unsteady record alone, a pass of 10 ms, a second steady record below the bar, a reference re-timed ×0.9 its rounds'
  minimum, and the re-decision under ×1.03 equal to a derivation at ×1.03 (and the derivation itself at ×1.065); the
  generator deriving from one steady record and stating both rules in its header.
- The plan of GEMV measurements: exactly the missing candidates, one steady record enough, and a batch the workbench's
  own batch reader accepts (core lane).
