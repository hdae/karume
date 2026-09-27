# GEMM tile geometry sweep

A micro-benchmark for perf-ledger K-70: why Anima's DiT runs 10–30× slower on Apple M2 (Chrome)
than on the Arc B570. Per-op profiling showed that bandwidth-bound kernels are slower by about the
nominal ratio (3.2–4×), while the large-tile GEMMs are not: the f32 `reg128x128` tile is ~10×
slower, the int8 `tile128x64` tile ~29×, and the medium `reg64x32` tile ~4.4×. The prime suspect is
the tile geometry. This tool measures **the same shape with only the geometry changed**, so the two
machines can be compared by ratios.

For every case (a kernel and a shape taken from Anima's op census) it runs the production kernel
generator with an explicit geometry for each candidate, times one dispatch, and checks that the
output is bit-identical to the one produced by the default geometry.

## Running

With Deno (any machine with WebGPU; run from the repository root):

```sh
deno run -A tools/geometry-sweep/main.ts --quick --op linear --op i8a8-linear
```

Flags:

- `--op <family>` (repeatable; default: all) — `linear`, `i8a8-linear`, `attention`,
  `i8a8-attention`, `conv2d`.
- `--case <id>` (repeatable) — only these cases (ids are listed in `cases.ts` and printed in the
  table).
- `--quick` — a small candidate set (the defaults plus 4–5 geometries). Without it, the full grid
  runs (54 f32 geometries, 48 int8 geometries, 21 conv2d geometries per case), which takes a long
  time; on a slow GPU, prefer `--quick` or narrow the run with `--op` / `--case`.
- `--rounds N` (default 5) — timed passes per geometry.
- `--out <file.json>` (default
  `outputs/bench/karume/<date>_geometry-sweep/geometry-sweep-<adapter>-<time>.json`).

The CLI prints one table per case: the geometry, the time per dispatch, the speed relative to the
default geometry, TFLOPS, the dispatches per pass, and whether the output matched the default. The
last line of each case is the default geometry measured again (`既定の再測定 ×N.NN`, the repeat
divided by the first measurement); outside 0.9–1.1 it is flagged in red as a hint to re-run that
case. Failed rows and mismatched outputs are also red. The CLI exits with status 1 when any row
failed (`error`); a mismatched output does not change the exit status.

With Chrome, see [browser/README.md](browser/README.md):

```sh
deno task bench:geometry-browser
```

## What is measured

The cases (`cases.ts`) are copied from the Anima op census
(`outputs/bench/karume-anima/2026-09-04_op-census/summary.json`, 1024px, S = 4096); the 512px rows
replace M = 4096 with 1024.

| family           | kernel                                | shapes                                                                                                              |
| ---------------- | ------------------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| `linear`         | f32 GEMM, f16 weights                 | DiT M ∈ {1024, 4096} × (N, K) ∈ {(2048, 2048), (8192, 2048), (2048, 8192)}; M 512 K 1024 N 2048; M 64 K 1024 N 3072 |
| `i8a8-linear`    | int8 × int8 dot product               | the DiT shapes above                                                                                                |
| `attention`      | fused attention QK and PV, f32 scores | B·H 16, D 128: self M = N ∈ {1024, 4096}; cross M ∈ {1024, 4096}, N 512                                             |
| `i8a8-attention` | int8 QK and PV, f16 scores (s16)      | the attention shapes above                                                                                          |
| `conv2d`         | implicit GEMM, f16 weights            | the top 3 VAE layers (Cout 96 at 512², 192 at 256², 384 at 128²)                                                    |

The candidates (`geometries.ts`) are grids filtered by the runtime's own geometry checks: f32
`regM ∈ {1,2,4,8}`, `regN ∈ {4,8}`, `wgX, wgY ∈ {4,8,16}` with at most 256 threads and tile sides
up to 128; int8 `regM, regN ∈ {4,8}`, `wgX ∈ {8,16}`, `wgY ∈ {4,8,16}`, `tileK ∈ {16,32}`; conv2d
uses the f32 grid restricted to n-tiles of 64 and 128. The geometry that the production code picks
for the case always runs first and is marked as the default. Geometry names use the same spelling
as the pipeline keys of each op, without the `v4` marker (`reg128x128r8x8w16` for f32 linear and
attention, `tile128x64r8x8w8x16k16` for int8, `igemm64x128:wg16x8` for conv2d).

Inputs are deterministic pseudo-random data; PV cases first run QK and the row statistics once
with the default geometry to get realistic scores.

Timing follows `tools/opbench/bench.ts`: one compute pass holds the same dispatch repeated until
the pass takes about 80 ms, the pass is timed with `timestampWrites`, and the representative value
is the minimum over the rounds. Before the timed rounds of every geometry, passes are repeated
until they add up to at least `WARMUP_NS` (500 ms) and `WARMUP_MIN_RUNS` (3) passes, capped at 64
passes; the sum uses GPU time when the unit is `ns` and the wall clock otherwise (Deno raw ticks
are not nanoseconds). A round whose timestamp difference is negative is kept as 0 in `rounds` but
is not a candidate for the minimum; if every round is negative, the row fails. The repetition
count is estimated from the wall clock. Without `timestamp-query`, the same procedure uses the
wall clock of each pass. After the last geometry of a case, the default geometry is measured once
more with the same procedure, to show whether the machine drifted during the case.

## Output (`karume-geometry-sweep/2`)

- `userAgent` (`navigator.userAgent`, or `{ deno }` for the CLI), `adapter` (vendor,
  architecture, device, description), `checkout`, `checkoutDirty`, `bundleSha256` (page only).
- `gpuTiming: { feature, unit, quantized }` — `unit` is `ns` (Chrome), `deno-raw-tick` (Deno
  returns raw GPU ticks; 1 tick = 52.08 ns on the Arc B570), or `wall` (no `timestamp-query`).
  `quantized` is true when every timed pass is a multiple of 100 µs (Chrome without the WebGPU
  developer features flag).
- `dp4a` — whether the int8 kernels used `dot4I8Packed` (the numbers are identical either way).
- `settings` — `quick`, `ops`, `cases`, `rounds`, `targetPassMs`, `maxReps`, `warmupNs`,
  `warmupMinRuns`, and `wallTimingNote` (the wall-clock caveat below, in words).
- `cases[]`, one per case: `caseId` and `defaultRepeat: { perDispatch, driftRatio }` (the default
  geometry measured again at the end of the case; `driftRatio` = repeat / first, outside 0.9–1.1
  suggests re-running the case), or `defaultRepeatError` when the repeat failed. A case whose
  first default measurement failed has neither.
- `rows[]`, one per case and geometry: `caseId`, `op`, `shape`, `censusCount`, `geometry`,
  `geometryParams`, `isDefault`, `key` (pipeline key), `workgroups`, `reps` (dispatches per pass),
  `rounds` (each timed pass, in `unit`), `wallRounds` (each pass's wall clock, ns),
  `clampedNegativeSamples` (rounds with a negative timestamp difference, excluded from the
  minimum), `perDispatch` (`min(rounds) / reps` over the non-negative rounds, in `unit`),
  `wallNsPerDispatch` (always ns), `tflops` (TOPS for int8; computed from `perDispatch` when the
  unit is ns, from `wallNsPerDispatch` otherwise), `speedupVsDefault` (default time / this time;
  above 1 is faster), `outputSha256`, `identicalToDefault`, and `error` for a geometry that
  failed (the sweep continues).
- Wall-clock values include the submit-to-completion floor (about 11 ms per pass on Deno). This
  applies to the unit `wall`, to `wallNsPerDispatch`, and to `tflops` under `deno-raw-tick`
  (which is computed from the wall clock): TFLOPS derived from the wall clock reads low, and ratios
  between geometries computed from it shrink toward 1. `speedupVsDefault` under `deno-raw-tick`
  uses the GPU ticks and is not affected.
- `outputSha256` is a two-level digest: the SHA-256 of the concatenated SHA-256 of each 64 MiB
  chunk of the output buffer. Equal outputs give equal digests, which is all the comparison needs.

## Caveats

- Compare **ratios**, not absolute times, across machines. Absolute numbers depend on the machine,
  the runtime, and the time unit.
- A geometry whose output does not match the default (`identicalToDefault: false`, shown in red)
  is not a candidate, however fast it is: for f32 the reduction order must not depend on the
  geometry (ADR 0022), and int8 dot products are exact.
- In Chrome, enable `chrome://flags/#enable-webgpu-developer-features` before measuring; otherwise
  timestamps are rounded to 100 µs.
- The int8 attention kernels use the same `dot4I8Packed` choice as int8 linear (the WGSL language
  feature). The production session decides attention with a correctness canary instead; the sweep
  does not run it, so on a device where the canary rejects `dot4I8Packed` the sweep measures a
  different variant from production.
- The f16-compute variants (`:c16` in the pipeline keys) are not among the candidates: every case
  runs the f32-compute kernel.
- The tool imports runtime internals (`RUNTIME_INTERNAL`: the error-scope lock, the device-lost
  race, and the pipeline cache) to measure with the production pipeline cache and scope
  discipline. A change to those internals can break the tool without breaking the public API.
- f32 self attention at M = N = 4096 has a 1 GiB score buffer S (16 · 4096 · 4096 · 4 B). On a
  device whose `maxStorageBufferBindingSize` is smaller, those rows fail; the production row-block
  variants (`:rwa` / `:rwc`) that handle this case are not measured.
