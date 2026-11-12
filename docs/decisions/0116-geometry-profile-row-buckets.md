# 0116: Refine the geometry-profile row buckets to seven (the default table keeps three)

- Status: accepted (2026-10-01 — the maintainer's ruling; it implements the bucket refinement
  approved on 2026-09-29 in ADR 0115 addendum 9); implemented 2026-10-01 (`c358832b`). The location
  of the boundary constant in decision 1 is superseded by [0117](0117-app-geometry-tuning.md)
  decision 1 (2026-10-01: moved to `packages/runtime/src/tune/`). Re-graded 2026-10-18: decision
  3's "the default table keeps three buckets" to SHOULD; decision 5's one-time re-run on named
  machines withdrawn.
- Date: 2026-10-01
- Related: [0115](0115-geometry-profiles.md) (geometry profiles; decisions 1 / 4 / 5, addendum
  decisions 1 / 3 / 8, addenda 8 / 9) / [0022](0022-gemm-register-blocking.md) (GEMM skeleton;
  decision 3's MUST "the reduction order is a numerical contract"; the addendum's MUST "no run-time
  autotuning") / [0106](0106-device-keyed-references.md) (sha256 reference values per environment
  key) / perf-ledger K-70, K-71. The read-only investigation (file:line citations, the numeric
  simulation and its script, open questions 1–9) is
  `.claude/reviews/2026-10-01_rows-buckets-recon/SUMMARY.md` (not tracked by git). Measurements:
  [research K-70](../research/2026-09-27-k70-metal-per-op.md) §13–§15.
- Implementation: `packages/runtime/src/tune/cases.ts` (the boundary constant
  `PROFILE_GEMM_ROWS_BOUNDS` and the sweep cases) and `packages/runtime/src/tune/derive.ts` (the
  generator).

## Context

The `gemmRows` field of a geometry profile (ADR 0115) holds one GEMM geometry per bucket of row
count M; linear, matmul and bmm all read it. The generator took its boundaries from the runtime's
default table `GEMM_ROWS_BUCKETS` (64 / 512 / ∞) and stopped on anything else (ADR 0115 decision 4).

Sweeps on the M2 and the RTX 5070 Ti showed that **the best geometry within a bucket differs by M**
(ADR 0115 addendum decision 3, addendum 8):

- M2, ≤ 64: M 16 → `reg16x32r2x4w8` ×1.21, M 32 → `reg32x64r4x4w16` ×1.75, M 64 →
  `reg64x64r4x4w16` ×2.25.
- The RTX single-case rules (≤ 64 `reg32x32r2x4w8` ×1.38, 65–512 `reg128x128r8x8w16` ×1.48) lost
  badly at other M of the same bucket (M 16 ×0.66, M 128 ×0.41).
- The generator requires ×1.05 on every case of a field, so no single geometry qualifies for a
  whole bucket. As a result ≤ 64 and 65–512 had returned to the default on both machines.

On 2026-09-29 the maintainer approved tables whose boundaries are the sweep cases' M (16 / 32 / 64 /
128 / 256 / 512 / ∞), per profile, with the default table kept at three buckets (ADR 0115 addendum
9). The runtime lookup (`gemmRowsGeometry`, a linear scan; the structural check of
`assertGemmRowsRules` — non-empty, strictly ascending, ending in Infinity, divisible) has no lower
or upper limit on the number of buckets, so the change is confined to the generator and to ADR 0115
decision 4.

A re-implementation of the generator rules with arbitrary boundaries (it matched the real generator
on every three-bucket input) predicted adoptions in most of the new buckets on M2, M5 and RTX; the
recon summary holds the table, and research K-70 §15 compares it with the result.

### Buckets with linear cases only

On 2026-10-01 the cases were uneven: M 16 / 32 / 128 / 256 had one linear case each, and only M 64
/ 512 had linear, matmul and bmm. In a linear-only bucket "×1.05 on every case" shrinks to "best on
one linear case", and the safety net of ADR 0115 addendum decision 1 (a geometry that loses on
matmul / bmm is not adopted) does not work, although the table also serves matmul and bmm. Where
all ops were measured (M 64 / 512 on M5, M2 and RTX), the linear-only winner was slower than the
default on some matmul / bmm case in 5 of 6 (machine, M) pairs, each time on an N 64 bmm shape (M 64
N 64 K 128, M 512 N 64 K 64, M 512 N 64 K 512).

## Decision

### 1. Boundaries: the fixed set `[16, 32, 64, 128, 256, 512, ∞]`

- The `gemmRows` boundaries of a profile (the `maxRows` column) are held in one place, the constant
  `PROFILE_GEMM_ROWS_BOUNDS = [16, 32, 64, 128, 256, 512, Infinity]`, next to the sweep cases; the
  generator imports it. (Superseded by ADR 0117 decision 1 on 2026-10-01 for the location: the
  constant moved from `tools/geometry-sweep/cases.ts` to `packages/runtime/src/tune/cases.ts:135`.)
- The boundaries other than the last MUST equal the set of M (≤ 512) of the linear / matmul / bmm
  sweep cases; `packages/runtime/tests/tune_cases_test.ts` checks it. Reason: the upper end of every
  bucket is always measured. Adding cases that change the boundaries moves the constant and the
  cases in the same change.
- Boundaries are not derived from the sweep input, nor taken as an argument (alternatives B / C).

### 2. Seven buckets for every generated profile; no merging, no compaction

- The generator writes seven rules for every profile. Adjacent buckets with the same verdict are not
  merged, and the output is not compacted.
- Reason: merging (option (b)) or compacting the output (option (b')) picks the same geometry per M
  as (a); only the row count of the table and the numbers in the verdict lines differ (recon §3.4).
  A fixed count keeps fields and rules one to one, so the verdict lines of the generated file and
  the per-profile test can be written "one per bucket".
- A table a user injects (`acquireGpu({ geometryProfile })`) may have a different number of buckets
  as long as it passes the runtime's structural check (ADR 0115 decision 1). The fixed seven applies
  to tables the generator writes.

### 3. The default table keeps three buckets; profile boundaries MUST refine the default

- The default table SHOULD stay at three buckets: `GEMM_ROWS_BUCKETS` (64 / 512 / ∞),
  `DEFAULT_GEOMETRY_PROFILE`, the codegen snapshots and the default checks of
  `geometry_profile_test.ts` do not change, and ADR 0115 decision 5 (the default profile is
  byte-identical to the table before it) still holds (re-graded 2026-10-18 to SHOULD: the reason is
  the cost of re-running the sha256 checks on every machine that the ADR 0022 addendum requires for
  a default change, not correctness; the default may change through that procedure).
- The set of profile boundaries MUST be a refinement that contains every default boundary (64 /
  512). The generator stops on any other set (fail loudly). Two reasons:
  - the sweep's default row is built from the three-bucket default table (`gemmGeometryForRows(m)`);
    a bucket crossing a default boundary would mix two bases for its ratios;
  - the per-bucket fallback would not be unique.
- The per-bucket fallback (the value written when a bucket has no case or no adoption, and the
  value the sweep's default row is compared against) is **the geometry of the default bucket that
  covers the range**: ≤ 16, 17–32, 33–64 → the default ≤ 64 geometry; 65–128, 129–256, 257–512 →
  the default 65–512 geometry; > 512 → the default > 512 geometry.
- The runtime lookup (`gemmRowsGeometry`) and structural check (`assertGeometryProfile`) already
  accept any number of buckets and do not change.

### 4. matmul / bmm cases are added to the linear-only buckets

Buckets M 16 / 32 / 128 / 256 get matmul and bmm cases besides linear: 12 cases (matmul 4 + bmm 8),
which takes the sweep from 45 to 57 cases.

| op     | Added cases                                                                                              | Shape source                                                                                                                                        |
| ------ | -------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| matmul | `matmul-m16-n3072-k1024`, `matmul-m32-n3072-k1024`, `matmul-m128-n2048-k1024`, `matmul-m256-n2048-k1024` | counterpart of the linear case with the same M / N / K (`mirrorOf` names it, as for the existing matmul cases)                                       |
| bmm    | `bmm-b16-m{M}-n64-k64` and `bmm-b16-m{M}-n64-k128` for M ∈ {16, 32, 128, 256}                           | `n64-k64` from text_conditioner `[16,512,64] × [16,64,64]`, `n64-k128` from text_encoder `[16,64,128] × [16,128,64]`, with M replaced |

- Why the two N 64 shapes: they were the ones that stopped the linear-only winners at M 64 / 512
  (Context).
- The derived cases have M values absent from the census. Their `source` says "census M … replaced
  by …" (as for the linear M 16 / 32 / 128 / 256 cases), and `censusCount` carries the count of the
  row they replace.
- Every bucket M ≤ 512 now has at least one linear, matmul and bmm case. The > 512 bucket stays at
  linear 6 + matmul 1 (no bmm in the census).

### 5. A class with new cases is measured with full

- A class that contains new cases (here `gemmRows`: linear / matmul / bmm) is measured with
  **full** (the whole grid) before a table is built, not with quick+ alone. Reason: the generator
  only considers geometries measured on every case of a field; measuring new cases with quick+
  alone drops every geometry that exists only in the full grid from that field.
- Withdrawn 2026-10-18: the one-time re-run on M2, M5 and RTX 5070 Ti and the regeneration of the
  registered `apple-metal-3` / `nvidia-blackwell` with the seven-bucket generator (done 2026-10-01,
  `c358832b`).

### 6. Verdict lines name buckets by range

- Verdict lines and rejected-geometry lines in the generated file name a bucket by its range:
  `gemmRows ≤ 16`, `gemmRows 17–32`, `gemmRows 33–64`, `gemmRows 65–128`, `gemmRows 129–256`,
  `gemmRows 257–512`, `gemmRows > 512`.
- No index (`gemmRows[1]` and the like): the three-bucket `gemmRows[1]` (65–512) and the second of
  seven (17–32) would share a spelling for different ranges and could be confused with old comments
  and with ADR 0115.
- The TS table stays a `maxRows` array (`GemmRowsRule[]`); the type does not change.

### 7. The per-profile real-GPU test has one case per rule

- The per-profile check of `packages/runtime/tests/gpu_geometry_profile_test.ts` (each registered
  table matches the default bit for bit as Uint32) has one case at a representative M for every
  `gemmRows` rule. With the earlier three cases (linear M 40 / 300 / 1024), four of the seven
  buckets' adopted geometries would escape the real-GPU identity check.
- The representative M is derived from the table's rules, so it follows table changes
  (`gpu_geometry_profile_test.ts:400`).

## Alternatives not taken

- **Deriving boundaries from the sweep (open question 1, B)**: the M values (≤ 512) present in the
  input become boundaries. Rejected: quick and full, old and new sweeps have different case sets, so
  the bucket layout would depend on the input combination. Following new cases automatically is
  covered by the single test of decision 1.
- **Boundaries as a CLI argument (C, `--rows-bounds 16,32,…`)**: an unrequested setting that
  lengthens the regeneration command and lets wrong boundaries through.
- **Merging adjacent buckets (b) / compacting the output only (b') (open question 2)**: the per-M
  choice is identical to (a), only the representation changes (recon §3.4). (b') can produce a rule
  that crosses the default boundary 64 (on the M5, 33–64 and 65–128 share `reg64x32r4x4w8`), which
  conflicts with the refinement check of decision 3.
- **Deciding linear-only buckets from the one linear case (open question 4, Q)**, with a note in the
  generated file: linear-only winners lose on matmul / bmm in 5 of 6 pairs (Context). It would put
  back into some buckets the extrapolation that ADR 0115 addendum decision 1 avoided, with two
  disciplines in one table.
- **No linear-only buckets (R, boundaries stay 64 / 512)**: the negation of the approved refinement.
- **Seven buckets in the default table too**: against the approved scope ("the default table keeps
  three buckets"). The default applies to every machine, so changing it means re-running the sha256
  checks on all of them (ADR 0022 addendum).
- **Bucket midpoints (M 24 / 48 / 96 / 192 / 384) as validation-only cases (open question 9)**: cases
  that check output identity and regressions but not adoption. They need a new mechanism to
  distinguish case roles; not taken here (see Open points).

## Consequences

- **The winner measured at a bucket's upper end applies to the whole bucket.** The 17–32 bucket
  applies the M = 32 winner to M 17–31. The three-bucket default has the same structure (≤ 64 was
  decided at M 64), but with fewer points per bucket the extrapolation is more visible (a geometry
  with tile edge 32 at M 17 leaves about half the rows idle — estimate).
- **Generated files grow**: 10 → 14 fields, with verdict and rejected-geometry lines for four more.
  The `quick+` candidate set grows with the adopted geometries of the registered tables.
- **Reference sha rows (ADR 0106) are not expected to change.** Geometry changes only the work
  assignment, not the K reduction order (ADR 0022 decision 3 MUST), so f32 / f16 output bits do not
  depend on geometry — an empirical claim (ADR 0022 addendum). The sweep's output-match check covers
  only the swept shapes × that machine × the adopted geometries, so the self A/B on M2 and RTX
  (PNG sha identical, 2026-10-01 / 02, research K-70 §15) confirmed it end to end. No tolerance is
  introduced.
- **The effect size was not measured in advance (estimate).** On anima the refinement moves the
  text_encoder linear / bmm (M 64) and the cross-attention k / v projections and conditioner bmm (M
  512). On the M2 the linear share of the 1.9 s text stage is 0.13 s (ADR 0115 addendum decision 4),
  so the effect on wall time is expected to be small.

## Implementation and acceptance (2026-10-01 to 2026-10-02)

Done in five steps: boundaries and the 12 cases; full re-runs on M2, M5 and RTX 5070 Ti (57 cases, no
failure, no mismatch); the seven-bucket generator (`8fbcef74`, fault injection red in three ways);
regeneration of both tables and one GPU case per rule (`c358832b`, `d1294daa`); unchanged codegen
snapshots and self A/B on M2 / RTX with identical PNG sha (research K-70 §15).

## Open points

- **Which families reach the M 16–256 buckets.** f16 / f32-stored linear uses GEMV only at M = 1, so
  M 2–64 reads `gemmRows`; quantized storage (i2 / i4 / i8) at M ≤ 64 uses the GEMV family and does
  not. Which family paths (for example LLM prefill chunks) fall into these buckets has not been
  investigated.
- **Selection bias of one-case buckets.** Each bucket M ≤ 512 has cases of all three ops, but only
  at its upper end. With tens to over a hundred candidates, picking the maximum biases ratios
  upward (general argument; its size in this sweep is not measured). Adoptions just above ×1.05 have
  no rule yet; validation cases at bucket midpoints (alternatives) are the candidate remedy.
- **A table for the M5.** The M5 has no registered table (`apple-metal-3` matches only when
  `description` is `"Apple M2"`); registering one is outside this ADR.

## Addendum (2026-10-01) — seven-bucket tables against the simulation

Of the 12 linear-only buckets (4 × 3 machines), the simulation predicted 9 adoptions; with the new
matmul / bmm cases only 3 remained, and the three borderline adoptions (M5 ≤ 16 ×1.065, M2 129–256
×1.054, RTX 17–32 ×1.072) all returned to the default — the motivation of decision 4 in practice.
Buckets that already had all three ops matched the simulation. Details: research K-70 §15, `c358832b`.

## Addendum (2026-11-12) — `i8a8.linear` has row-count buckets of its own

ADR [0115](0115-geometry-profiles.md) addendum decision 14 splits the i8a8 linear's field by the row count M with
bounds of its own, `I8A8_LINEAR_ROWS_BOUNDS = [256, 512, ∞]`, and takes this ADR's rules for them: the rule form and
order of `gemmRows` (one guard checks both), every bound but the last a measured case (decision 1 — M 256 joined
the sweep for it), the generator writing the fixed set for every table (decision 2), a bucket without a case keeping
the default (here the default profile's one rule, so any bounds refine it — decision 3), verdict lines named by range
(decision 6) and one real-GPU case per rule (decision 7). The bounds enter the `i8a8-linear` family's case-set
version only.
