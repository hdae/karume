# 0157: The GEMM tile of linear, matmul and bmm may be chosen from the dispatch grid's workgroup count by a rule a table carries

- Status: accepted (2026-11-28 — the maintainer accepted it and confirmed decision 8 for `nvidia-ampere`: the table
  takes the field from two grid records of the RTX 3080 Ti, and this machine's corpus census rows are rewritten,
  gated by the full `vp test` with every output row unchanged; proposed the same day, when the maintainer asked for
  the item and took the recon's recommended options)
- Date: 2026-11-28
- Related: [0022](0022-gemm-register-blocking.md) (the tile geometry changes the work split only) /
  [0115](0115-geometry-profiles.md) (geometry profiles, a static table per adapter, no run-time autotuning) /
  [0116](0116-geometry-profile-row-buckets.md) (the seven row-count buckets) / [0117](0117-app-geometry-tuning.md)
  (the sweep, the records and the derivation) / [0147](0147-numerics-and-kernel-adoption.md) (plan
  transformations and the level `exact`) — the rules as built are in
  [spec/geometry-profiles.md](../spec/geometry-profiles.md); the measurements are in
  [research 2026-11-28](../research/2026-11-28-gemm-grid-fill.md)

## Context

A geometry profile picks the tile of the GEMM of linear, matmul and bmm from the row count M alone: the first
`gemmRows` rule whose `maxRows` holds M (ADR 0115 decision 1, refined to seven buckets by ADR 0116). N, K and the batch
are not in the key. The number of workgroups a tile gives a dispatch, `ceil(N / tileN) · ceil(M / tileM) · batch`,
therefore varies by 10 to 100 times inside one bucket: Gemma 4's 768-row prefill chunk runs its k / v projections
(N 256) and its MLP gate / up (N 12,288) with the same 128 × 128 tile, which gives the first 12 workgroups on an 80-SM
GPU and the second 576. ADR 0116 met the consequence as cases that block an adoption: a tile that is faster on the
bucket's linears loses on an N 64 bmm of the same M, and the every-case rule picks one tile for both.

The recon of 2026-11-28 measured every one of the sweep's 54 tiles on 125 shape classes of the default presets'
GEMM-path linears and bmms on the RTX 3080 Ti (Node, Dawn / Vulkan), against the tile `nvidia-ampere` gives each class
today, in interleaved rounds:

- The tile a Session runs today is the fastest of the measured ones on 13 of the 125 classes.
- What predicts the fastest tile is the grid's workgroup count W. Read pairwise down a ladder of five tiles (128 × 128,
  64 × 64 / 256 threads, 64 × 32 / 128, 32 × 32 / 128, 16 × 16 / 64): 64 × 64 takes ×0.33 to ×0.55 of 128 × 128's time
  where 128 × 128's grid has at most 40 workgroups, ×0.7 to ×0.9 at 96 to 256, parity near 288 and ×1.02 to ×1.14 at
  384 or more. The exceptions are bmm with K 32 to 72 and shapes that pad M or N by 2 to 4 times.
- A rule with two device integers and two shape constants — keep 128 × 128 when its grid has at least 256 workgroups,
  K ≥ 256 and its padding (tile area over output area) is at most 1.5; else take the first smaller tile whose grid has
  at least 48 workgroups at that padding bound; else the last tile — takes ×0.880 of today's census-weighted GEMM time,
  against ×0.877 for the fastest tile of every class. Its geometric-mean regret against that fastest tile is 1.027 to
  1.029, its worst 1.36 (a 13 µs siglip2 bmm), and one class is more than 3 % slower than today.
- Leaving each of the 17 families out of the fit picks (256, 48) again for 16 of them ((320, 48) for Gemma 4 E2B in
  bf16), and every held-out score stays within 0.005 of the full fit's.
- All 6,750 candidate rows (125 classes × 54 tiles) wrote the same output words as the reference tile, over f32, f16,
  bf16, i8, i4 and i2 storage, scalar and vec4, linear and bmm.

## Decision

1. **An optional table field, the grid-fill rule** (`gemmGrid`): a ladder of tiles, largest first; the least workgroup
   count and the least K at which the largest tile is kept; the least workgroup count of every smaller tile; and the
   padding bound. The selection walks the ladder: the largest tile when its three conditions hold, else the first
   smaller tile whose grid is full enough within the padding bound, else the last. The workgroup counts are the
   device's; the K floor and the padding bound are shape constants of the fit, written into the field with the ladder
   so that the table states the whole rule it runs.
2. **The field replaces the M buckets for linear, matmul and bmm in a table that carries it.** It is not a refinement
   that only steps down from the bucket's tile. A table without it, and the default profile, select by `gemmRows`
   exactly as before: every pipeline key and every byte of WGSL is unchanged.
3. **The default profile takes no integers.** Its three buckets stay (ADR 0116 decision 3); only a table derived from
   its device's own grid records carries the field.
4. **A pure function of the static shape and the table.** M (of one matrix for bmm), N, K and the batch are known when
   a Session derives its dispatches. Nothing is measured at run time (ADR 0115, the ADR 0022 addendum), and the tile
   stays in the pipeline key.
5. **Scope**: the GEMM of linear (every weight storage it reads, under f32 and f16 compute) and its split-K partial,
   matmul and bmm. The i8a8 paths, the fused attention's fixed fields, conv1d / conv2d / conv3d, and the states-shaped
   and row-block attention that read the default bucket table directly are outside it.
6. **The contract class is unchanged.** A tile changes which thread computes which output, never the order of a K
   reduction, so a tile the rule picks is admissible at the level `exact` because, and only where, its words equal the
   reference GEMM's — ADR 0147's plan-transformation row D, a choice between bit-identical forms. The evidence per
   device is the grid sweep's word comparison of every ladder row against the default row, and on the development
   machine a GPU test of the rule's tile against the bucket's over every storage.
7. **The integers come from grid sweep records** (a candidate set `grid` of the geometry sweep, about thirty
   census-derived cases spanning the grid fill, timing only the ladder in interleaved rounds). The derivation tries a
   fixed grid of integer pairs, takes the ideas of the GEMV rule of one steady record where they carry over — a
   record's steadiness class, a per-reading bar built from the rows' own gaps and a margin instead of a fixed ×1.05 per
   case — and keeps the derivation threshold ×1.05 where it still applies: on the census-weighted GEMM time of the
   case set over today's table. A table with no qualifying pair carries no field.
8. **The embedded tables are not regenerated by this decision.** `nvidia-ampere` takes the field only after the
   maintainer confirms its regeneration from fresh records and the rewrite of this machine's corpus census rows (the
   kernel keys move by construction; no output row may). Other devices need their own grid records.

## Why

- **Why the grid's workgroup count.** It is the variable the measurements order by: the fastest tile moves with W
  across shapes of one M, and a thread-count target (W × threads) fit the data worse than W, because the ladder's tiles
  have 64 to 256 threads (inference: occupancy on this GPU is bounded per workgroup). M alone keys on one factor of W.
- **Why replace and not refine.** Refining — start at the bucket's tile and only step down — took ×0.909 to ×0.932 of
  today's weighted time against the rule's ×0.880, because today's buckets are also too small in places: umT5's
  M 512 × N 10,240 runs 64 × 32 today and takes ×0.79 of that with 128 × 128.
- **Why two integers per device.** One threshold for every step (W ≥ 192) took ×0.894 and left 16 classes more than
  3 % slower than today; the largest tile's saturation point and the smaller tiles' differ. Two integers capture about
  97 % of the gain of choosing the fastest tile per class, and the leave-one-family-out fits move by at most one grid
  step.
- **Why the default takes none.** The integers are an 80-SM GPU's saturation points. On a GPU with a tenth of the
  cores they would shrink tiles too far (the same caveat as the GEMV row target), and a default change re-runs every
  machine's sha256 checks.
- **Why the field states its constants.** The K floor and the padding bound were fit on the same shapes as the
  integers. Holding them in the table means a later change of the code's constants does not silently change what a
  registered table runs.

## Alternatives not taken

- **M × N bucket classes** (for example the seven M buckets × three N buckets): N cannot see the batch (the same M and
  N of lucida's bmm range over W from 48 to 21,600) or the short-K effect of depth-anything's and so400m's bmm, and the
  every-case rule inside a class still lets the worst case veto. About 40 to 60 more sweep cases against the full
  candidate set, about 30 minutes more per device at the benchmark page's priors (inference), and 21 rules a table.
- **Per-shape rows like the GEMV choices** (`{ op, n, k, batch, M bucket } → tile`): M is dynamic (prefill chunks, the
  last partial chunk, symbolic T and S), so a row must still bucket M, and the rows grow with the census by an order of
  magnitude; a new model gains nothing until it is swept. The GEMV rows work because M = 1 is fixed; here two integers
  reach ×0.880 against ×0.877.
- **Doing nothing**: the M buckets keep 112 of the 125 measured classes off their fastest tile, at a census-weighted
  ×1/0.877 = ×1.14 of the GEMM time that was measured to be available, with identical words.

## What was not measured

- **One device.** Every ratio is the RTX 3080 Ti's, on Vulkan under Node. No other GPU has a grid record; the
  integers are not portable (decision 3).
- **Kernel-level ratios.** The recon timed repeated identical dispatches, so small weights stay warm in the 6 MB L2.
  What a model gains end to end is an inference until the Session A/B of the backlog (depth-anything, siglip2 so400m,
  lucida, Gemma 4 prefill) is taken; the item is killed if that A/B shows no gain.
- **Identity is observed, not guaranteed.** It holds by construction in the WGSL, and below it depends on the shader
  compiler (FMA contraction; Metal compiles with relaxed floating-point math — known-issues). It was observed on
  Vulkan / NVIDIA for every tile and storage the recon timed; on Metal and D3D12 only the tiles the existing tables'
  sweeps timed. Browsers have no standing GEMM-geometry kernel check outside a sweep.
- **The fit is scored on the shapes it was fit on.** The K floor 256 and the padding bound 1.5 are facts of those
  125 classes on one GPU. Several M values are assumptions (EmbeddingGemma 2 at T = 256 and 64, sbv2 at T = 128,
  umT5 at L = 512, Irodori at S = 185), and the Gemma chunk is 768 rows; a partial last chunk is not measured.
- **The skeleton forms (K-88) were not measured** on the ladder; `nvidia-ampere` carries none. Ladders for the tables
  that carry forms must carry and sweep them.

## Consequences

- With no table carrying the field, no Session's kernels change. The selection function, the field's checks and its
  JSON reader exist, and a table can be injected with it (`acquireGpu({ geometryProfile })`).
- A table that carries the field runs more distinct tiles per Session: more pipelines to compile at first use (on
  `nvidia-ampere`, 32 × 32 / 128 is new per op and storage).
- The table-side matching keys (`geometryProfileMismatch`) cover the M buckets, not the field, as they do not cover the
  GEMV choices: the field's evidence is checked when it is derived, against the current runtime's case set and kernels.
- The corpus census rows of the one environment with sha256 rows move when `nvidia-ampere` takes the field (the tile
  is in the kernel key); an output row that moves would falsify decision 6 and is never rewritten.
- The split-K item (perf-ledger K-115) is the lever on long-K, mid-N projections where the rule gains nothing (Gemma's
  down / o at ×0.95 to ×0.98); the two are measured together at the end-to-end A/B if both land.
