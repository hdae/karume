# 0117: Sweep geometries inside the user's application, store the profile and inject it into the next `acquireGpu` (`@karume/runtime/tune`)

- Status: accepted (2026-10-01 — the maintainer's ruling: option a for public form, measurement
  check and storage key; recommended options for decisions 2 / 3 / 4 / 6); implemented (2026-10-03);
  partly superseded by [0127](0127-node-dawn-toolchain.md) decision 5 (2026-10-16: decision 1's
  thin shells `tools/geometry-sweep` and `tools/gpu-lab` are not ported, so the regeneration decision
  4 requires has no tool in the repository). Triage 2026-10-18: decisions 3 and 4 kept at their grade.
- Date: 2026-10-01
- Related: [0115](0115-geometry-profiles.md) (geometry profiles — decisions 1 / 4 / 8, addendum
  decisions 6 / 7 / 8, addenda 7 / 9) / [0116](0116-geometry-profile-row-buckets.md) (seven row
  buckets, per-bucket fallback) / [0022](0022-gemm-register-blocking.md) (the addendum's MUSTs "no
  run-time autotuning" and "a change of the default geometry goes together with re-running the
  sha256 checks") / [0008](0008-public-api.md) (thin public surface; a second surface is added
  through `exports`; tools may import `src/` directly) / [0106](0106-device-keyed-references.md)
  (sha256 reference values per environment key) / perf-ledger K-71.
  Code: injection in `packages/runtime/src/gpu/acquire.ts`, the profile type in
  `packages/runtime/src/kernels/geometry-profile.ts`, the sweep core in `packages/runtime/src/tune/`.

## Context

The goal: a user's application sweeps on its own device, builds a table (a geometry profile),
stores it, and from then on injects it into `acquireGpu` as a fixed value. Devices the developers do
not own (for example an 8 GB-class Android phone) otherwise run only with the default geometry.

What existed on 2026-10-01: the injection entry point `acquireGpu({ geometryProfile })` (ADR 0115
addendum decision 6), whose structural check `assertGeometryProfile` (id, shape of `match`, rule
lists, divisibility) runs before the device is created and whose table is copied and fixed for the
device's lifetime, with `match`, `provenance` and the building version not compared; the public
`BUILTIN_GEOMETRY_PROFILES` and `GeometryProfile` (addendum decision 7); a JSON serializer writing the
trailing `maxRows` (Infinity) as `1e999`; and the proof of concept in `tools/gpu-lab` (ADR 0115
addenda 7 / 9), which swept on a dedicated device and kept the last table in localStorage.

Four design points (backlog K-71):

1. The sweep core and the generator imported runtime internals from `src/`: not publishable on JSR.
2. The generator accepted only unquantized timestamp-query records, so Chrome without developer
   flags (timestamps rounded to 100 µs) could not build a table.
3. Injection compared neither `match` nor a version; no storage key told whether a table applies.
4. The sweep's duration was not recorded (`date` was the end time).

Measurements behind the decisions (2026-09-29, history): the sweep imported 64 internal values and 2
internal types, while the public surface held only the table's shape and the way to get a GPU.
Simulating 100 µs rounding on the full sweeps of three machines (M5, M2, RTX 5070 Ti; 200 trials
each) changed no field's adoption, nor did choosing by wall clock alone. A sweep took 7–9 minutes
(quick+) to 42–61 minutes (full), about half of it idle runs. Without flags Chrome reports an empty
adapter `device` and `description`, and Apple M2 and M5 share `apple` / `metal-3`. The subpath adds
36–71 KB (gzip 11–20 KB) to bundles that import it, nothing to others. Full tables: this file at
`cb88cb33`.

## Decision

### 1. Public form: the subpath `./tune` of `@karume/runtime`

- The sweep core moves to `packages/runtime/src/tune/`: the Deno-independent parts of harness /
  cases / geometries / derive / report, the opbench calibration constants (now
  `src/tune/measurement.ts`), and `looksQuantized` with `CHROME_TIMESTAMP_QUANTUM_NS` (now
  `src/tune/report.ts`).
- `exports` becomes `{ ".": "./mod.ts", "./tune": "./tune.ts" }` (`packages/runtime/package.json`
  and `deno.json`). The entry `packages/runtime/tune.ts` is a deliberately designed thin surface like
  `mod.ts` (ADR 0008).
- The internal codegen vocabulary the sweep uses is imported within the package and MUST NOT be
  exported from `./tune` or `.`. Reason: it would become semver surface, against ADR 0008 addendum
  2026-09-03 (no internal APIs on the public surface for tools) and the alternative ADR 0115
  rejected.
- Why this form: **the sweep and the kernels always share a version** — within one package version
  the binding tables, params and WGSL the sweep builds match the production recipes, whereas a
  separate package could have tune@x call the generators of runtime@y (a swap between bindings of the
  same role is invisible even to `assertBindingRoles`); **no `./internal` subpath is needed**; the
  CLI, gpu-lab and user applications share one core.
- `tools/geometry-sweep` (`main.ts` / `profile.ts`) and `tools/gpu-lab` become thin shells over
  `./tune`. What only the tools use (rendering the generated TS `renderProfileSource`, `--check`,
  `--min-speedup`) stays in the tools, which may import `src/tune/` directly (ADR 0008 addendum
  2026-09-03). (Superseded by ADR 0127 on 2026-10-16: both tools were removed in `bb35fe2a`; the
  structure — `./tune` as the core, tools as thin shells — applies unchanged to the browser
  applications that will replace them. Comments in `src/tune/` (`derive.ts:5-7`, `sweep.ts:6`,
  `report.ts:4`) and `tests/tune_profile_test.ts:6` still name the removed tools.)
- Modules in `src/tune/` MUST be free of side effects (cross-cutting invariant): nothing is computed
  at import time (the case-set version is a function — decision 4). The scope (pure constant
  construction is allowed) is stated in the addendum of 2026-10-02 (independent review).
- The main surface (`mod.ts`) gains no values; only the type of `AcquireGpuOptions.geometryProfile`
  (decision 6) and the shape of `GeometryProfile.provenance` (decision 4) change.

### 2. The `./tune` surface: high-level entry points only

| Entry point                                                       | Role                                                                                                                                                         |
| ----------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `runGeometrySweep(options)`                                       | Acquires a dedicated device, sweeps, destroys it; returns the record (`karume-geometry-sweep/2`). Options include the adapter request, the candidate set (`quick` / `quick+` / `full`), an op filter, an `AbortSignal` and progress |
| `deriveGeometryProfile(reports, options)`                         | Builds a table from one or more records (rules: ADR 0115 decision 4, ADR 0116, decision 3 here); returns the table and the per-field verdicts                |
| `geometryProfileJson(profile)` / `parseGeometryProfileJson(text)` | Serialization and the parse at the `unknown` boundary (decision 7)                                                                                           |
| `geometryProfileMismatch(profile, adapterInfo)`                   | Whether a stored table may be applied (decision 5)                                                                                                           |
| `sweepCaseSetId()`                                                | The case-set version (decision 4)                                                                                                                            |
| Types                                                             | The record, options, candidate set and the parse error `GeometryProfileParseError`                                                                           |

- The names are fixed here; argument and return details were settled in steps 2 / 4 (the
  public-surface snapshot check shows the diff).
- `runGeometrySweep` takes no `GpuContext`; it acquires and destroys its own device, so "sweep and
  inference share a device" cannot be expressed in the types (decision 9's MUST kept by structure).
- timestamp-query is requested when the adapter lists it; otherwise the sweep measures by wall clock,
  and the generator rejects that record (decision 3).
- The cases come from Anima's op census (`src/tune/cases.ts` header), so applications of other
  families also get their fields decided by Anima's shapes; the `./tune` doc states this.

### 3. Measurement check: an upper bound on rounding error per observation (amends ADR 0115 decision 4's input check)

The blanket rejection on the quantized flag is dropped. Each observation (one row of one sweep) gets
an upper bound on the error rounding adds to its ratio.

- **Quantization unit q**: 100 µs (Chrome's step) when `gpuTiming.quantized` is true, else 0
  (unquantized `ns` and `deno-raw-tick`).
- **Row bound** e(r) = q ÷ min (the non-negative measured rounds of row r). `perDispatch` = that min ÷
  reps, and the integer reps carries no error. The rounding error of a difference of two timestamps
  is under one step, so e(r) bounds the relative error of `perDispatch` (to first order).
- **Ratio bound** E(r) = e(r) + e(d), d being the default row of the same sweep and case: the sum
  bounds the relative error of `speedupVsDefault`, a ratio of two `perDispatch` (to first order).
- **Threshold**: E(r) ≤ 1% (`ROUNDING_ERROR_LIMIT`, `src/tune/derive.ts:98`) — one constant, neither a
  CLI argument nor an application option. E is a first-order approximation, not a strict bound
  (addendum 2026-10-02, independent review). Basis: on an 80 ms pass e ≈ 0.13% and E ≈ 0.25%; on the
  three full sweeps of 2026-09-29 the default rows' e was at most 0.15%, and only one observation (an
  RTX row whose pass was 3.1 ms because the reps estimate missed; e alone 3.2%) exceeded 1%; 1% is one
  fifth of the ×1.05 adoption margin and one tenth of the re-measurement ratio's range (0.9–1.1 —
  ADR 0115 addendum decision 8).
- **Observations above the bound** (the style of addendum decision 8):
  - The (case, geometry) observation leaves that sweep's ratio data. Output mismatches and failures
    still count (the correctness check does not depend on rounding).
  - Another sweep with the same (case, geometry) decides instead; with none, the geometry counts as
    not measured and is no candidate for that field (the safe side).
  - A large e on the default row pushes every observation of the case over the bound, so the whole
    case drops out, as with an out-of-range re-measurement ratio.
  - Dropped observations and their E are listed in the verdicts (the generated file's header comment
    and the return of `deriveGeometryProfile`). There is no switch to disable the exclusion.
- `gpuTiming.quantized` stays in the record (decided by the writer's `roundsLookQuantized`,
  unchanged). The reader checks that it is a boolean and uses it to choose q.
- **Wall-clock-only records** (`gpuTiming.unit` = `wall`) and records without `gpuTiming` are still
  rejected (fail loudly): wall-clock error has no known constant bound — the submit-to-completion
  floor moves per pass (median 0.65–1.25 ms, up to 13.5 ms on the three Chrome machines; about 11 ms
  on Deno — `WALL_TIMING_NOTE`, `src/tune/report.ts:34`); the floor is readable only next to GPU
  time, so the record alone gives no per-row bound; and wall clock gave the same verdicts on three
  machines only because the floor was small against an 80 ms pass — the mobile floor is unknown.
- The quantization unit rests on the premise that Chrome without flags rounds timestamps to 100 µs.
  It did not hold for Chrome 154 on macOS (addendum 2026-10-02); Windows and Android without flags
  are not observed. The rule stays as decided, as a safeguard for rounded records (kept 2026-10-18).
- The other rules of ADR 0115 decision 4 (merging, matching the default row, candidates, ×1.05,
  keep-the-default, exclusion by re-measurement ratio) are unchanged.

### 4. Structured `provenance`

```ts
readonly provenance?: {
  readonly sweep: string; // paths of the sweep records (in input order, joined with ", ")
  readonly sha256: string; // their sha256
  readonly date: string; // their sweep dates
  readonly candidateSet: string; // their candidate set (quick / quick+ / full)
  readonly userAgent?: readonly string[]; // browsers of the sweeps (addendum 2026-10-02)
  readonly adapter: { readonly vendor: string; readonly architecture: string;
                      readonly device: string; readonly description: string };
  readonly kernels: string; // the kernel fingerprint
  readonly caseSet: string; // the case-set version
};
```

- `adapter` holds the four `GPUAdapterInfo` fields as they are (empty strings included) instead of a
  joined string. The generator requires all input records to agree on the four fields (the earlier
  three-field check plus `device`). A display label is derived from them when rendering.
- `candidateSet` is the record's `settings.candidateSet`; old records without it (before quick+) read
  `quick` / `full` from `settings.quick`.
- Only the generator `buildGeometryProfile` (`src/tune/derive.ts:1193`) writes it; the generated TS
  and the injected JSON both come from it. `provenance` is a field of the public type, so its shape
  is public surface.
- The runtime (`assertGeometryProfile`, injection) does not compare `provenance` (addendum decision
  6); it checks only the field types, so that a value that passed is a well-typed `GeometryProfile`
  (`src/kernels/geometry-profile.ts:241-245`, `:344`).

#### A kernel fingerprint instead of the runtime version

- **Definition**: for each field of the table and each shape of the sweep cases in it, the pipeline
  key, params, WGSL and workgroups of the table's geometry and of the default geometry (the field's
  fallback — ADR 0116 decision 3) are concatenated in a fixed order and hashed with a synchronous
  non-cryptographic hash (64 bits or more), as hex. No GPU, no clock; by codegen determinism the same
  runtime always gives the same value. It is derived through the sweep's case plan (`casePlan`,
  `src/tune/harness.ts:606`), so measured and fingerprinted kernels are never built by separate
  paths. It covers only the fixed compute variants of the sweep cases (addendum 2026-10-02,
  independent review).
- Reasons: **it derives from the source of truth** — the runtime has no version constant, and a
  version would need a copy in `src` (with a sync check) or the application writing its own
  dependency version (silent mistakes); **it invalidates at the right time** — a version discards
  tables at every release, even one not touching GEMM, forcing a 7–61 minute re-sweep, while the
  fingerprint changes only with the kernels the table or the default points to; **it works on
  unreleased checkouts**, where kernels change under the same version string.
- Costs and gaps: each match runs codegen and hashes (no GPU; measured at step 4: median 12–14 ms for
  the two built-in tables on Deno, Ryzen 5 5600, about 26 ms on the first call). Changes outside key,
  params, WGSL and workgroups (for example the order of recipe bindings) do not show; the runtime's
  own checks (codegen snapshots, GPU tests with the default geometry) cover them. The hash is
  synchronous to keep matching and the callback (decision 6) synchronous (`crypto.subtle` is async
  only); only accidental collisions are considered, not forgery (the table is the user's own data).
- The two built-in tables carry the fingerprint too, and a unit test pins "`provenance.kernels` = the
  fingerprint derived by the current runtime" (`packages/runtime/tests/tune_fingerprint_test.ts:406-420`).
  A change to the GEMM kernels goes together with regenerating the two tables in the same change
  (geometry values unchanged): the machine reports that the kernels changed, and the per-profile GPU
  test (`gpu_geometry_profile_test.ts`, ADR 0116 decision 7) re-checks output identity with the new
  kernels — the discipline of ADR 0022's "a change of the default goes together with re-running the
  checks". (Kept 2026-10-18: the rule keeps a stale table from surviving silently. Since ADR 0127
  decision 5 removed `tools/geometry-sweep` (`bb35fe2a`), nothing in the repository renders the
  built-in tables, which call themselves "generated — do not edit by hand"
  (`src/kernels/geometry-profiles/apple-metal-3.ts:2`); a GEMM kernel change turns this test red with
  only a hand edit to fix it. The regeneration path has to be restored before work on the GEMM
  kernels.)

#### Case-set version

- `sweepCaseSetId()` hashes, like the fingerprint, canonical JSON of `SWEEP_CASES`
  (`src/tune/cases.ts:350`) and `PROFILE_GEMM_ROWS_BOUNDS` (`src/tune/cases.ts:135`); only the fields
  that decide the dispatch go in (addendum 2026-10-02, step 4). No hand-bumped version number (no way
  to add a case and forget the bump). A function, not a constant, so nothing runs at import time.

### 5. The pure matching function `geometryProfileMismatch`

- Form: `geometryProfileMismatch(profile: GeometryProfile, adapterInfo: GPUAdapterInfo): string |
  undefined` (parameter narrowed to the four fields — addendum 2026-10-02, step 4). It returns the
  first mismatch as a message, or `undefined`, and never throws. As with `sessionOptionsViolation`
  (`mod.ts:171`), the caller decides what to do (run with the default, re-sweep, tell the user).
- Compared, all by exact string equality: (1) `provenance` is present (a table without it is
  hand-written — "nothing to match against"); (2) the four adapter fields (`vendor` / `architecture` /
  `device` / `description`, empty strings compared as values); (3) the kernel fingerprint
  (re-derived from the table by the current runtime vs `provenance.kernels`); (4) the case-set
  version (`sweepCaseSetId()` vs `provenance.caseSet`).
- Not compared: `candidateSet` (whether quick+ suffices is the application's call), `match`
  (injection ignores it — addendum decision 6), browser and driver versions (no information).
- It lives in `./tune` with the fingerprint and the case-set version. An application that only
  matches at start-up also imports `./tune`; the sweep core is expected to drop out by tree-shaking
  (inference — not measured; see Open points).
- **Limit**: without flags Chrome reports empty `device` and `description`, so Apple M2 and M5 have
  the same four fields (ADR 0115 addendum 9 / addendum decision 7) and a table carried between them
  is not detected. localStorage stays on the device and browser profile, so only an application that
  carries tables itself hits this (inference: Chrome sync does not carry localStorage — unconfirmed).

### 6. Callback form of the injection entry point

- `AcquireGpuOptions.geometryProfile` widens to
  `GeometryProfile | ((adapterInfo: GPUAdapterInfo) => GeometryProfile | undefined)`.
- It is called once per `acquireGpu`, after the adapter is obtained and before the device is created
  (`acquire.ts:628-631`), with the value of `readAdapterInfo` (`acquire.ts:387-388`, missing info
  normalized to empty values) — the same value that becomes `GpuContext.adapterInfo`.
- `undefined` means the automatic selection used without the option (one of the built-in tables). A
  returned table goes through the same check and copy as a table passed directly; a broken table
  raises `GpuFeatureError` before the device is created. An exception from the callback propagates
  as is, and no device is created.
- Reason: the table is chosen with the information of the adapter the runtime actually selected.
  Otherwise the application calls the raw `navigator.gpu.requestAdapter()` itself (the spec does not
  guarantee the same adapter — the NOTE at `acquire.ts:770`) or acquires once without injection.
- **The callback MUST be synchronous.** Reason: an adapter may expire within seconds to minutes, and
  `requestDevice` on an expired adapter does not throw but returns a device lost from birth
  (`acquire.ts:765-767`). No window of waiting on I/O while holding an adapter is opened: loading a
  stored table (IndexedDB and other asynchronous storage) finishes before `acquireGpu`, and the
  callback is a pure function (parse, match, choose).
- The callback measures nothing (there is no device yet), so it does not touch ADR 0022's MUST. The
  runtime does not match at injection (addendum decision 6); the application calls
  `geometryProfileMismatch` inside the callback. `match` is ignored too (A/B with another machine's
  table remains possible).

### 7. Parse at the `unknown` boundary: `parseGeometryProfileJson`

- Form: `parseGeometryProfileJson(text: string): GeometryProfile`. Failures raise
  `GeometryProfileParseError` (public in `./tune`) with a message naming the field path.
- Checks: the text is JSON; the types of all fields (`id` a string, geometries integer fields only,
  `provenance` as in decision 4); **unknown fields are rejected** at every level (table, `match`,
  geometries, `provenance`); the trailing `maxRows` of `gemmRows` is Infinity (restored from
  `1e999`), and a trailing `null` is named as "Infinity missing (plain `JSON.stringify` turns
  Infinity into null — write with `geometryProfileJson`)"; finally `assertGeometryProfile` (the same
  single check as the runtime's).
- Why reject unknown fields: a table is injected and affects execution, and reading a table of a
  different shape version lets a misread field through silently. Discarding an unreadable table and
  sweeping again is the application's correct move (decision 10). The sweep record (decision 8) is
  the opposite: an append-only log whose readers use only the fields they read.
- `geometryProfileJson(profile)` is the serializer (`src/tune/derive.ts:1369`); a unit test pins the
  round trip. In the application's path the parse handles a missing field in stored JSON (otherwise
  a non-public `TypeError`) and a trailing `null` (otherwise a message that does not name the missing
  Infinity); the runtime's injection check (which takes typed values) is unchanged.

### 8. Fields added to the sweep record `karume-geometry-sweep/2`

| Field               | Content                                                              | Reason                                                                                   |
| ------------------- | -------------------------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| `startedAt`         | Start time of the sweep (ISO 8601)                                   | `date` is the end time; the duration could not be read                                   |
| `aborted`           | Whether the sweep ended by abort (boolean)                           | An aborted sweep and an op-filtered sweep could not be told apart (both just lack cases) |
| `cases[].elapsedMs` | Wall clock per case (ms)                                             | Which cases take the time                                                                |
| `caseSet`           | The case-set version (decision 4)                                    | Which case set was measured                                                              |
| `defaultKernels`    | Kernel fingerprint of the default table (`DEFAULT_GEOMETRY_PROFILE`) | Which runtime's kernels the ratios rest on                                               |

- The format stays /2 and the fields are optional. The reader `parseSweepReport`
  (`src/tune/derive.ts:376`) checks only the fields it reads and ignores unknown ones (as when
  `candidateSet` was added).
- When `caseSet` / `defaultKernels` are present, the generator compares them with the current
  runtime and stops on a difference (fail loudly), for the reason of matching the default row (ADR
  0115 decision 4): no table from a record whose baseline is another runtime. Records without them
  are accepted as before. The candidate-side `candidateKernels` follows the same style (addendum
  2026-10-02, independent review).
- `aborted` is not used by the generator: a case cut off midway has no re-measurement of the default,
  so addendum decision 8's rule drops it from the ratio data.
- The only writer is `runGeometrySweep`.

### 9. Invariants and the division of responsibility (consistency with ADR 0022's MUST)

ADR 0022's addendum (`0022-gemm-register-blocking.md:88-89`): "MUST: a change of the default
geometry goes together with re-running the sha256 checks. MUST NOT: geometry that changes at run time
(autotuning) for f32 / f16."

- **Nothing is measured on the runtime's acquire / Session path (MUST).** Only `runGeometrySweep`,
  which the user calls explicitly, measures. Since the runtime package now contains measuring code,
  the subject of the invariant is restated from "the runtime" to "the acquire / Session path".
- **A sweep MUST run on a dedicated device** (acquired and destroyed by `runGeometrySweep` — decision
  2). **It MUST NOT run alongside inference**: that pollutes the measurement, and both compete for the
  GPU. The runtime cannot see how another device is used, so this is a contract with the application,
  stated in the `./tune` doc.
- A table does not change during the device's lifetime (addendum decision 6; the copy in
  `acquireGpu`); a new table takes effect from the next `acquireGpu`, with no way to replace it on
  the same device. The same table and shape always give the same geometry, key and WGSL (ADR 0115
  decision 8); the application decides the table once at start-up, so the meaning of a key holds.

| Who                            | Guarantees                                                                                                                                                                                                        |
| ------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| runtime                        | The injected table does not change during the device's lifetime; nothing is measured on the acquire / Session path; structural check at injection (`assertGeometryProfile`)                                     |
| `./tune` (sweep and generator) | An adopted geometry matched the default's output in every case of its class and was ×1.05 or faster (ADR 0115 decision 4); the evidence covers only the sweep's case set (Anima's shapes) × that device × the adopted geometries |
| user application               | Decides with `geometryProfileMismatch` whether to keep a table; does not sweep alongside inference; records which table ran (output reproducibility)                                                             |

- The end-to-end sha256 checks (PNG / WAV reference values — ADR 0106) cover the built-in tables,
  not tables built in an application; for those, the only evidence of output identity is the sweep's
  output check (within addendum decision 6's "the runtime does not guarantee").
- An application that automates "default → sweep in the background → re-acquire with the new table"
  behaves like run-time autotuning at the application level. The runtime's invariants (fixed per
  device, no measuring on the acquire / Session path) still hold; recording when and with which table
  it ran is the application's responsibility, and the `./tune` doc says so.

### 10. Table lifetime and when to re-sweep

- **Condition for continued use**: `geometryProfileMismatch` returns `undefined` (adapter fields,
  kernel fingerprint and case-set version match).
- **When to re-sweep**: on a mismatch (a runtime update changed the kernels or the case set, another
  adapter was chosen, a flag change altered `device` / `description`) and on the user's explicit
  request. While mismatched, the callback can return `undefined` and run with the default.
- **Undetectable staleness**: browser, driver or OS updates can change performance while matching
  succeeds (no information), so a table can become slower silently. Example: M2's default
  `reg128x128r8x8w16` for linear M 1024 went 19.68 → 13.07 ms (×0.66) from Chrome 153 to 154, the
  adopted `reg128x32r8x4w8` 11.38 → 11.00 ms, and matching succeeded ([research
  K-70](../research/2026-09-27-k70-metal-per-op.md) §15). Output identity still holds (a geometry only
  changes the work split; the fingerprint proves the kernels unchanged); the user re-sweeps.
- Quick+ or full, and when to sweep (first launch, idle time, a settings screen), is the
  application's decision; the durations under Context are the guide. Storage (localStorage,
  IndexedDB, a file) is outside the runtime: store the `geometryProfileJson` string and restore it
  with `parseGeometryProfileJson`.

### Alternatives not taken

- **A separate package `@karume/tune` (A)**: a separate JSR package imports only through the other's
  `exports`, so the runtime would need a subpath of internal vocabulary (55 values used by the
  harness and the three `RUNTIME_INTERNAL` primitives), making codegen vocabulary semver surface
  (against ADR 0008 and 0115); the two versions could also drift.
- **Not publishing (C) / a static gpu-lab page whose JSON users paste into the application (C')**:
  every invariant holds, but not "inside the application"; devices the developers do not own stay on
  the default.
- **The runtime matches at injection**: removes the A/B use of another machine's table that addendum
  decision 6 kept, and a throw from the runtime leaves applications no choice of response. Matching
  is a pure function the application may call.
- **The runtime version as storage key** (decision 4): there is no version constant; it discards
  tables on releases that do not touch GEMM and stays the same on unreleased checkouts whose kernels
  changed.
- **Only removing the quantized-flag rejection, without a bound**: verdicts did not change on three
  machines, but outliers such as the RTX row (3.1 ms pass, about 3% error) would stay in the data;
  short passes appear whenever the reps estimate misses, on any device. The bound drops only those.
- **Accepting wall-clock-only records** (decision 3: no bound from the record), **an asynchronous
  callback** (decision 6: a wait while holding an adapter), **`runGeometrySweep` taking the caller's
  `GpuContext`** (decision 2: the types would allow a device running inference).

## Consequences

- **Public surface**: `@karume/runtime` gains a second surface `./tune`; the main surface gains no
  values; `AcquireGpuOptions.geometryProfile` widens and `GeometryProfile.provenance` changes
  (`adapter` from a string to four fields). The public-surface snapshot check takes its entries from
  `exports` in `package.json` (`packages/runtime/tests/helpers/public-surface.ts:97-110`), so the
  fixture gains the key `"./tune"`.
- **Published package**: it includes the sweep core and calibration constants (`src/tune/`, 4,694
  lines on 2026-10-18). Bundles of applications that do not import `./tune` do not change.
- **The record format and the case set carry semver obligations**: changing the case set changes its
  version, and every stored table then mismatches (re-sweep). Adding a case is decided with that
  cost included.
- **Chrome without flags can build tables.**
- **A GEMM kernel change goes together with regenerating the two built-in tables** (decision 4 and
  its 2026-10-18 note).
- **Small devices** (inference): the four f32 self-attention cases with M = N = 4096 need bindings of
  512 MiB to 1 GiB each; with a small `maxStorageBufferBindingSize` they become failed rows and the
  attention fields stay at the default (the safe side).

## Acceptance

The seven steps ran from 2026-10-01 to 2026-10-03 (`cb88cb33`, `51724541`, `d242008f` for the
move). Step 7 ran sweep → build → store → reload → match → inject by callback on Chrome 154 without
flags on M2 and M5, with PNG sha256 identical to the flagged run with the built-in table ([research
2026-10-02-app-tuning-flagless-m2](../research/2026-10-02-app-tuning-flagless-m2.md) §8). Full
verify: 3,544 passed / 1 failed (the B570 OOM check in `docs/known-issues.md`, unrelated).

## Open points

- **Tuning the 1% threshold on measurements.** Its basis is three full sweeps (2026-09-29) and a
  rounding simulation reusing the same measurements; in records re-measured without flags, thermal
  and jitter differences should exceed rounding (the re-measurement ratio check handles those).
  Chrome's rounding scheme is not confirmed: the "under one step" bound holds for truncation to a
  grid and for rounding to the nearest, not for a scheme that adds jitter.
- Outside the runtime / this ADR: **where an application stores its table** and how storage failures
  (quota, private mode) are handled; **registering an M5 table** (as in ADR 0116).
- **Size and tree-shaking of `./tune` from JSR**: only a `deno bundle` measurement exists; how users'
  bundlers drop it, and whether the sweep core drops out when only matching is imported, are not
  measured.
- **Mobile (8 GB-class Android)**: the floor, whether 1 GiB bindings work, and device loss under
  memory pressure (the Pixel entries in `docs/known-issues.md`) are not observed; all three measured
  machines are desktop class.
- **Wall-clock records**: devices without timestamp-query still cannot build tables. Recording the
  floor per row to derive a bound is considered once such devices appear.
- **Per-row stage times** (compile / probe / warmup / rounds / digest) are not recorded; idle runs
  take about half the duration, so they matter for shortening it.
- **Widening the variants the fingerprint covers** (addendum 2026-10-02, independent review): it
  covers only the sweep cases' fixed variants (f32 compute, no mask, f16 weight storage and so on); a
  change to only the f16 compute path, masked attention, i4 and similar paths does not show. A
  candidate remedy: fingerprinted but unmeasured variant cases (a match then costs more).

## Addendum (2026-10-02) — step 3 results and deviations from decision 3

- **"Non-negative rounds" reads as "positive rounds".** Negative rounds stay in the record as 0
  (`clampedNegativeSamples`) and cannot be told from rounds rounded to 0. Excluding 0 from the min
  changes no verdict: a row with `perDispatch` = 0 never had a ratio to the default.
- **A case whose default row has no bound (no `rounds`, or no positive round) drops out of the ratio
  data as a whole** (the safe side). With q = 0 rounds are not read, so old records pass as before.
- **Records with `quantized: true` and a unit other than `ns` are rejected** (a check decision 3 did
  not have): the writer judges quantization only for `ns`, and a 100 µs step on raw ticks would give
  a different bound.
- **A case-level exclusion row carries e(d)**, not each observation's E; since E ≥ e(d) > 1%, that is
  enough.

## Addendum (2026-10-02) — a browser-version reference field in the table's `provenance` (the maintainer's ruling)

[research K-70](../research/2026-09-27-k70-metal-per-op.md) §15 showed M2's default geometry
`reg128x128r8x8w16` shrinking to ×0.66 between Chrome 153 and 154, which cut the table's gain from
×1.78 to ×1.20 (the first case of decision 10's undetectable staleness). Decision 5's "browser and
driver versions are not compared" stays (version strings do not map one to one to performance, and
a mismatch would discard tables needlessly). Instead the table gets `provenance.userAgent` (optional,
copied from the input records' `userAgent`, not used for matching), added **in the same change as
step 6 (record fields)**, so that an application can tell the user the sweeping browser differs from
the current one and suggest a re-sweep (the decision is the application's).

## Addendum (2026-10-02) — step 4 results and deviations from decisions 2 / 4 / 5

- **CHANGELOG: Added, not Breaking**: `GeometryProfile` and `provenance` arrived after 0.13.0
  (`84d369d9`) and were unreleased.
- **`geometryProfileKernelsId` is not exported** (decision 2); the matcher re-derives it. `./tune`
  gained `parseGeometryProfileJson` / `GeometryProfileParseError` / `geometryProfileMismatch` /
  `sweepCaseSetId`.
- **The case-set version is built only from the fields that decide a case's dispatch** (id, op,
  m / n / k / batch; attention stage / batchHeads / d / scale / score; conv2d channelsIn /
  channelsOut / height / width; plus the bucket bounds). The descriptive `source`, `censusCount` and
  `mirrorOf` are left out, so a fix to a description does not invalidate stored tables. The types
  enforce the classification (a new field is a type error until classified), and a case with an
  unclassified field throws.
- **The matcher's parameter type is `Pick<GPUAdapterInfo, the four fields>`**, so both the real object
  and `GpuContext.adapterInfo` can be passed.
- **The fingerprint's dispatch-count limit is fixed at 65535** (the default
  `maxComputeWorkgroupsPerDimension` WebGPU guarantees everywhere, so the fingerprint does not depend
  on the device). i8a8 cases include both dp4a variants (the adapter information does not decide
  which WGSL runs). Concatenation follows the case order, not the field order (same set of
  dispatches). The stage before ③PV (①QK with the default geometry and the row statistics) is not
  included, by definition.
- **A table whose fingerprint cannot be derived** (it fails the codegen check, or a dispatch count
  exceeds the limit) raises `KernelsIdError` (internal). The generator throws naming field, geometry
  and case; the matcher returns the message without throwing.
- **Sweep records require `adapter.device` and `settings`** (for the four-field check and
  `candidateSet`).

## Addendum (2026-10-02) — steps 5 / 6 results and deviations from decisions 6 / 8 and the userAgent addendum

- **A callback returning a Promise is rejected, and a real Promise gets a no-op catch** (never
  awaited; a custom thenable's `then` is never called). Without it, Deno ends the process on the
  unhandled rejection and the intended `GpuFeatureError` never arrives.
- **`provenance.userAgent` is an array of strings** (order of appearance, duplicates removed; an empty
  array is not written; if any record lacks `userAgent` the field is not written; a record's
  `{ deno }` becomes `Deno/<version>`). The addendum said "joined", but Chrome's UA contains `", "`,
  so joined values could not be split. Not used for matching.
- **The new record fields are required in the TS type** (the writer's output); the reader accepts
  their absence at the `unknown` boundary. `aborted` is true only when an abort left out a geometry, a
  re-measurement of the default or a later case.
- **`EMPTY_ADAPTER_INFO` (the empty value for a missing `adapter.info`) is frozen**, because it
  reaches the user's callback.
- Step 7 ran through a gpu-lab setting (removed by ADR 0127 decision 5) that used the callback
  pattern the `./tune` module doc describes (parse, match, inject or `undefined`).

## Addendum (2026-10-02) — findings of the independent review (Codex)

Design concerns and findings L2 / M7 of the independent review (Codex), as the orchestrator ruled.

- **What the fingerprint covers (decision 4 made explicit)**: only dispatches built from the shapes
  of `SWEEP_CASES` × these fixed variants (`profileKernelsId` in `fingerprint.ts` → `casePlan` in
  `harness.ts`):

  | op             | Variants in the fingerprint                                                      |
  | -------------- | -------------------------------------------------------------------------------- |
  | linear         | f16 weight storage, f32 compute (`linearKey("f16", v4, "f32", m, …)`)            |
  | matmul         | f32                                                                              |
  | bmm            | f32, no row window                                                               |
  | attention      | f32 compute, S stored as f32, no mask, no GQA, no row window (stages ①QK / ③PV) |
  | i8a8-linear    | i8 weights (per-channel), both dp4a variants                                     |
  | i8a8-attention | S stored as f16 (s16), both dp4a variants                                       |
  | conv2d         | f16 weight storage, 3×3, stride 1, padding 1, dilation 1, groups 1, batch 1      |

  - Only the one of v4 / scalar that the case's shape selects goes in; the limit of 65535 and the
    exclusion of the stage before ③PV are as in the step 4 addendum.
  - **Changes that do not show**: those altering only paths outside these variants — the f16 compute
    branch, f16 storage of S in f32 attention, attention with mask / GQA / row window, the bmm row
    window, linear and conv2d with f32 / i8 / i4 / i2 weights, i4 i8a8 linear, the v4 / scalar side
    no case shape selects. The table applies to those paths too, yet matching still succeeds.
  - Matching (decision 5) guarantees only that the table's and the default's geometries build the
    same kernels for these variants; this ADR and the `./tune` doc state the scope. Outside it the
    runtime's own checks (codegen snapshots, GPU tests with the default geometry) remain responsible.
    Fingerprint-only variant cases are listed under Open points.
- **The candidate fingerprint `candidateKernels` (M7 — one row added to decision 8's table)**:

  | Field              | Content                                                                         | Reason                                                                         |
  | ------------------ | ------------------------------------------------------------------------------- | ------------------------------------------------------------------------------ |
  | `candidateKernels` | Fingerprint of the dispatches of the candidates the sweep measured              | Tells apart a runtime that changed only the candidates' WGSL, not the default |

  - Derived in the style of `defaultKernels`: per case × candidate × dp4a variant, the one-line JSON
    of key, params, WGSL and workgroups, joined with newlines and hashed with FNV-1a 64 bit; the case
    plan is the sweep's.
  - The writer is `runGeometrySweep`. When the field is present the generator compares it with the
    current runtime and stops on a difference (fail loudly); records without it are accepted (as for
    `caseSet` / `defaultKernels`). A fault injection pins it: with one candidate's WGSL changed and
    the default unchanged, generating from an old record stops.
  - Reason: comparing only `caseSet` and `defaultKernels` lets an old record pass on a runtime that
    changed only the candidates' WGSL; the generator would decide from stale measurements and stamp
    the new kernels' fingerprint into `provenance.kernels`, undetectable by later matching.
  - **Derivation (ruling at implementation, 2026-10-02)**: not from the current candidate set for the
    record's `candidateSet`, as the review text said, but from **the record's rows (the (case,
    geometry) pairs without `error`) built with the record's dp4a variant**. The quick+ set includes
    the registered tables' geometries, so deriving from the current set would reject quick+ records
    with unchanged kernels after a table is registered; deriving from rows also matches sweeps
    narrowed by op or case and aborted sweeps. Writer and reader share one function.
- **The scope of "nothing is computed at import time" (L2 — decision 1 made explicit)**: prohibited
  are side effects (GPU, I/O, global mutable state, registration) and precomputing values every
  match needs (values that run the codegen, such as `sweepCaseSetId()` and the fingerprint, are
  functions). **Pure constant construction** such as `SWEEP_CASES` (immutable data built with `map` /
  `flatMap` / `filter`) is allowed: no GPU or I/O, always the same result, and the tree-shaking
  condition (cross-cutting invariant) holds.
- **The 1% bound is a first-order approximation (note on decision 3)**: E = e(r) + e(d) approximates
  the ratio's relative error to first order. Example: both observations at 20 ms with q = 0.1 ms give
  E = 1.00%, accepted (exactly 1% is accepted — step 3), yet true values of 19.9001 ms and 20.0999 ms
  give a relative error of about 1.004%. Decision 3's "bound" is read in this sense. A strict
  guarantee would need interval division; it is not taken (1% is one fifth of the ×1.05 margin, and
  the difference is about 0.5% of the 1%).
- **Fixes for findings in steps 5 / 6**:
  - M1 (a record with a default `perDispatch` of 0 was rejected whole): the writer writes ratios
    (`speedupVsDefault`, the re-measurement ratio) only when numerator and denominator are both
    positive; such a case drops out by the reader's rule (no bound for the default → case dropped).
  - M6 (an exception in progress notification became a measurement failure, notified twice):
    notification moves outside the measurement's try/catch; the callback's exception propagates.
  - M3 (a Promise from another realm left an unhandled rejection): a native Promise's rejection is
    handled with a realm-independent check, keeping the contract of never calling an arbitrary
    thenable's `then` (decision 6, step 5 addendum).

## Addendum (2026-10-02) — Chrome 154 without flags (macOS, M2) did not round timestamps

None of 2,605 rounds in the first step 7 record was a multiple of 100 µs. Decision 3's check stays
unchanged as a safeguard for rounded records ([research
2026-10-02-app-tuning-flagless-m2](../research/2026-10-02-app-tuning-flagless-m2.md) §3).

## Addendum (2026-10-03) — acceptance complete

Steps 1–7 all green (see Acceptance); the remaining items stay under Open points.
