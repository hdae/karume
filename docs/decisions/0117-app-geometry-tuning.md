# 0117: Sweep geometries inside the user's application, store the profile and inject it into the next `acquireGpu` (`@karume/runtime/tune`)

- Status: accepted (2026-10-01 — the maintainer's ruling: option a for public form, measurement
  check and storage key; recommended options for decisions 2 / 3 / 4 / 6); implemented (2026-10-03);
  partly superseded by [0127](0127-node-dawn-toolchain.md) decision 5 (2026-10-16: decision 1's
  thin shells `tools/geometry-sweep` and `tools/gpu-lab` are not ported, so the regeneration decision
  4 requires has no tool in the repository). Triage 2026-10-18: decisions 3 and 4 kept at their grade;
  decision 8's record gains a backend field by [0147](0147-numerics-and-kernel-adoption.md) decision 9 (2026-11-10)
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

(Amended 2026-11-12: `sweepCaseSetId()` is replaced by `sweepCaseSetIds()` and `sweepDefaultKernelsIds()`, one value per op family, and `runGeometrySweep` returns `karume-geometry-sweep/3` — addendum "the matching keys per op family".)

| Entry point                                                       | Role                                                                                                                                                         |
| ----------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `runGeometrySweep(options)`                                       | Acquires a dedicated device, sweeps, destroys it; returns the record (`karume-geometry-sweep/2`). Options include the adapter request, the candidate set (`quick` / `quick+` / `full`), an op filter, an `AbortSignal` and progress |
| `deriveGeometryProfile(reports, options)`                         | Builds a table from one or more records (rules: ADR 0115 decision 4, ADR 0116, decision 3 here); returns the table and the per-field verdicts                |
| `geometryProfileJson(profile)` / `parseGeometryProfileJson(text)` | Serialization and the parse at the `unknown` boundary (decision 7)                                                                                           |
| `geometryProfileMismatch(profile, adapterInfo)`                   | Whether a stored table may be applied (decision 5)                                                                                                           |
| `sweepCaseSetId()`                                                | The case-set version (decision 4)                                                                                                                            |
| `geometrySweepCases()`                                            | The cases a sweep can be narrowed to, as id and op (addendum 2026-10-28)                                                                                     |
| `sweepCandidateKernelsId(rows, dp4a)`                             | The candidate fingerprint of a record merged from per-case sweeps (addendum 2026-10-28)                                                                      |
| Types                                                             | The record, options, candidate set and the parse error `GeometryProfileParseError`                                                                           |

- The names are fixed here; argument and return details were settled in steps 2 / 4 (the
  public-surface snapshot check shows the diff).
- `runGeometrySweep` takes no `GpuContext`; it acquires and destroys its own device, so "sweep and
  inference share a device" cannot be expressed in the types (decision 9's MUST kept by structure).
- timestamp-query is requested when the adapter lists it; otherwise the sweep measures by wall clock,
  and the generator rejects that record (decision 3).
- The cases come from Anima's op census (`src/tune/cases.ts` header), so applications of other
  families also get their fields decided by Anima's shapes; the `./tune` doc states this. (Amended by the
  addendum of 2026-10-28: the table also holds the Wan2.2 VAE decoder's conv3d w8a8 cases, which decide
  `i8a8.conv3d`.)

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

(Amended 2026-11-12: provenance and both keys are kept per op family — addendum "the matching keys per op family".)

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
  key, params, WGSL (its token text — comments and whitespace removed, addendum 2026-10-19) and
  workgroups of the table's geometry and of the default geometry (the field's fallback — ADR 0116
  decision 3) are concatenated in a fixed order and hashed with a synchronous non-cryptographic hash
  (64 bits or more), as hex. No GPU, no clock; by codegen determinism the same
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
  kernels.) (Updated 2026-11-11: the records now come from the playground's `/geometry-sweep` page and
  `deriveGeometryProfile` derives the table, but the step that writes the TS module is still an untracked script
  under `outputs/diag/`, so the tracked regeneration path is still missing.)

#### Case-set version

- `sweepCaseSetId()` hashes, like the fingerprint, canonical JSON of `SWEEP_CASES`
  (`src/tune/cases.ts:350`) and `PROFILE_GEMM_ROWS_BOUNDS` (`src/tune/cases.ts:135`); only the fields
  that decide the dispatch go in (addendum 2026-10-02, step 4). No hand-bumped version number (no way
  to add a case and forget the bump). A function, not a constant, so nothing runs at import time.

### 5. The pure matching function `geometryProfileMismatch`

(Amended 2026-11-12: it compares per op family — addendum "the matching keys per op family".)

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

(Amended 2026-11-12: the record is `karume-geometry-sweep/3`, with `caseSets` and `defaultKernels` per op family, and the generator reads `/3` only — addendum "the matching keys per op family".)

| Field               | Content                                                              | Reason                                                                                   |
| ------------------- | -------------------------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| `startedAt`         | Start time of the sweep (ISO 8601)                                   | `date` is the end time; the duration could not be read                                   |
| `aborted`           | Whether the sweep ended by abort (boolean)                           | An aborted sweep and an op-filtered sweep could not be told apart (both just lack cases) |
| `cases[].elapsedMs` | Wall clock per case (ms)                                             | Which cases take the time                                                                |
| `caseSet`           | The case-set version (decision 4)                                    | Which case set was measured                                                              |
| `defaultKernels`    | Kernel fingerprint of the default table (`DEFAULT_GEOMETRY_PROFILE`) | Which runtime's kernels the ratios rest on                                               |

- A backend field is added to the record by ADR [0147](0147-numerics-and-kernel-adoption.md) decision 9
  (amended 2026-11-10; WebGPU exposes no backend). How it is filled — inference from the user agent, which
  is speculation-grade, entry by the person running the sweep, or both — is an implementation-time item of
  that decision.
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
| `./tune` (sweep and generator) | An adopted geometry matched the default's output in every case of its class and was ×1.05 or faster (ADR 0115 decision 4); the evidence covers only the sweep's case set (Anima's shapes, and the Wan2.2 VAE's conv3d cases since the addendum of 2026-10-28) × that device × the adopted geometries |
| user application               | Decides with `geometryProfileMismatch` whether to keep a table; does not sweep alongside inference; records which table ran (output reproducibility)                                                             |

- The end-to-end sha256 checks (PNG / WAV reference values — ADR 0106) cover the built-in tables,
  not tables built in an application; for those, the only evidence of output identity is the sweep's
  output check (within addendum decision 6's "the runtime does not guarantee").
- An application that automates "default → sweep in the background → re-acquire with the new table"
  behaves like run-time autotuning at the application level. The runtime's invariants (fixed per
  device, no measuring on the acquire / Session path) still hold; recording when and with which table
  it ran is the application's responsibility, and the `./tune` doc says so.

### 10. Table lifetime and when to re-sweep

(Amended 2026-11-12: a runtime change invalidates only the entries of the op families it touches, and a sweep of those families derived with the stored table as `base` restores it — addendum "the matching keys per op family".)

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

## Addendum (2026-10-19) — the fingerprint hashes the WGSL without its comments (amends decision 4)

- **What changed**: the kernel fingerprint hashes each dispatch's WGSL as its token text — `//` and
  nested `/* … */` comments removed, runs of whitespace collapsed to one space (`wgslTokenText` in
  `src/tune/fingerprint.ts`) — instead of the generated bytes. Pipeline key, params and workgroups
  are hashed as before.
- **Why**: the fingerprint exists to tell whether the kernels a table was measured with are the
  kernels now running. Two shaders with the same token text compile to the same program, so a
  comment edit or a re-wrapped line changed no kernel yet moved the fingerprint, and every such edit
  forced the two embedded tables to be re-baked by hand (the generator is gone since ADR 0127
  decision 5). The English translation of the WGSL comments (ADR 0130 step 2-9d) made this concrete:
  no kernel changed, both tables mismatched. `linear-i8a8.ts` even carried a MUST NOT against
  editing its WGSL comments for this reason alone; that rule is withdrawn.
- **Consequences**: the two embedded tables' `provenance.kernels` are re-derived once
  (`apple-metal-3` → `fc81e4c33996e068`, `nvidia-blackwell` → `27067d014c2f3b30`; geometry data
  unchanged). Tables saved by applications under the old definition mismatch once on the kernel
  fingerprint and need one re-sweep (0.15 is a breaking release in any case). A change to any WGSL
  token still moves the fingerprint, with the same re-generation discipline as before.
- **Checks**: `tune_fingerprint_test.ts` pins that comment-only and whitespace-only WGSL edits keep
  the value, that one added token changes it, and that nested block comments are stripped.

## Addendum (2026-10-28) — two `./tune` entries for applications that sweep case by case (amends decision 2)

- **What changed**: `./tune` gains `geometrySweepCases()` (with its element type `GeometrySweepCase`)
  and `sweepCandidateKernelsId(rows, dp4a)`.
  - `geometrySweepCases()` returns every case of the sweep's table as `{ id, op }`, in the table's
    order, as a new array of new objects on every call. The ids are what
    `GeometrySweepOptions.cases` already takes. The table itself (`SWEEP_CASES`, with its shapes,
    `source`, `censusCount` and `mirrorOf`) stays internal, so a change to a case's descriptive
    fields is no public change, and nothing a caller does to the result reaches the case set or the
    fingerprints.
  - `sweepCandidateKernelsId` is the function the writer and the reader of `candidateKernels`
    already share (M7 addendum): from a record's rows and its `dp4a`, without a GPU.
- **Why**: the playground's geometry sweep page (the successor of the removed gpu-lab sweep tab, ADR
  0127 decision 5) calls `runGeometrySweep` once per case with idle time between cases, so that heat
  does not move the default geometry's re-measurement. It must list the cases to offer them, and it
  merges the per-case records into one. Each per-case record fingerprints only its own rows, so the
  merged record's `candidateKernels` must be recomputed over all the rows, or
  `deriveGeometryProfile` rejects it. The ampere sweep of 2026-10-23 merged the same way through an
  internal import ([research 2026-10-23-ampere-geometry-sweep](../research/2026-10-23-ampere-geometry-sweep.md)),
  which an application cannot do.
- **Why not a runtime merge function**: a `mergeGeometrySweepReports` would keep decision 2's
  "high-level entry points only", but what a merge means is the application's choice: which cases
  were planned, whether a cancel between cases counts as `aborted`, and how `gpuTiming.quantized`
  combines (the playground takes it as true when any per-case record has it — the larger rounding
  bound — where one call would judge all the rows together). Both entries are pure, GPU-free
  functions of public types, of the same rank as `sweepCaseSetId()`.
- **The cases are no longer Anima's alone** (amends the last point of decision 2): since the
  addendum of 2026-10-26 to [0133](0133-conv3d-w8a8.md) (conv3d geometry) the table also holds the
  conv3d w8a8 cases of Wan2.2 TI2V-5B's VAE decoder, which decide `i8a8.conv3d`. The `./tune`
  module doc says so.

## Addendum (2026-11-11) — a wall-clock session A/B and the public identity comparison (amends decision 2)

The maintainer approved the design of the three extension mechanisms on 2026-11-11 (fusion rules as a Session input,
fusion rules from outside the runtime — [0148](0148-external-fusion-rules.md) — and a way to compare them in `./tune`).
This addendum records its two rulings on `./tune`.

- **What changed**: `./tune` gains `runSessionAb(options)`, `sessionArm(options)`, `sessionAbStatistics(record)`,
  `judgeSessionAb(record, options)` and their types (the arm, its run, its report, the environment, the record
  `karume-session-ab/1` and its parts, the comparison, the verdict and the judge's options), and `wordIdentity`
  with `NanPolicy`, `WordIdentity` and `WordMismatch`.
- **Ruling 1: the comparison of fusions is its own function and its own record, and a fusion is accepted or dropped
  on the wall clock.** A fusion's effect is measured as the end-to-end time of untimed runs of two Session
  configurations (or two pipelines), not as per-dispatch GPU time, which is reported only as a breakdown. Why: on
  2026-11-10 the per-dispatch times of the `gelu_tanh` → `mul` prototype on Gemma 4 26B A4B promised a gain that the
  untimed forward did not show ([research 2026-11-10](../research/2026-11-10-gemma4-26b-fast.md) §8.1; perf-ledger
  K-105); the whole-forward A/B was the reading that held. The A/B shares nothing with the geometry sweep — no case
  list, no fingerprint, no record format: a fusion case added to the sweep's case table would move the case-set id
  and every geometry profile's fingerprint (decision 4), so every table on every GPU would need a new sweep.
  - **The protocol**: an untimed warm-up round, then in every round the arms in order and in reverse order (A B B A),
    so each arm is visited twice per round and a drift linear within the round cancels in each arm's round mean, for
    any number of arms. A cost of switching from one arm to another is balanced only for two arms: with three or more,
    a middle arm follows another arm twice per round and the outer arms once. The wall clock (injectable,
    `performance.now` by default) spans the arm's run alone, its readback included; opening and closing an arm are
    never timed, also when the arms are opened and closed around every visit (for models whose two Sessions do not fit
    in memory together). There a visit is open, one untimed run, the timed run, close: the first run of a fresh
    Session derives its plan, which is not what the A/B measures. An arm's counts (fusion hits, dispatches) are read by
    its optional `report` after the clock stopped (`sessionArm` reads the Session's diagnostics there, so nothing but
    the Session's run is timed), and every run must report its arm's first run's counts: a plan that changes between
    runs makes the A/B meaningless, so the runner refuses it. Every run's outputs, the untimed runs' included, are
    compared with a copy of the base arm's first outputs, taken when they are captured; an output of another array
    type fails whole, and a base whose first run returns no word is refused (an identity over nothing proves nothing).
    A timed span the clock reads as not finite or negative is refused in the runner, in the judge's words.
  - **The statistics**: per round, the mean of each arm's two visits; the median of the paired differences; the gain
    over the base's median round time; the noise floor (the larger of the two arms' medians over rounds of
    |visit 1 − visit 2|), conservative on purpose — it does not shrink with the number of rounds: more rounds make
    the medians steadier, they do not let a gain smaller than the jitter of one visit through; and the
    order-statistic interval of the median of the paired differences (the k-th smallest and k-th largest, with the
    largest k that keeps 90% coverage; the smallest and the largest with 3 or 4 rounds, at 75% and 87.5%), which
    catches an arm whose speed moves from round to round while its two visits of a round agree. A spread of the
    differences around their own median was tried first and dropped: it is 0 whenever half the rounds tie, so a
    3-of-5 split still adopted and the verdict changed with the parity of the round count. **The verdict** is
    `"inconclusive"` inside the noise floor or when the interval holds 0, otherwise `"adopt"` at a gain of at least the
    caller's `minGain`, otherwise `"reject"`; identity with the base is required by default (a class E rule — ADR
    0147 decision 2), and an arm that differs is `"reject"` whatever its times. The threshold is the caller's. A
    verdict holds for the record's device, backend and work only; `"inconclusive"` does not say the difference is zero.
  - **The record** carries what ADR 0147 decision 9 asks of per-GPU evidence: the adapter's four fields, the user
    agent, the backend's name as the caller states it (WebGPU does not expose it, so the runner does not infer it),
    and a date the caller passes (the runner reads no clock for dates; the adapter is copied field by field, so a
    browser's `GPUAdapterInfo`, whose fields are accessors, can be passed as it is); per arm its fusion counts,
    dispatch count, every raw time and its worst run's identity. The record holds no identity flag: the judge reads
    identity from the worst run's failures, and refuses a malformed identity block by name.
  - **Arms**: `sessionArm` builds an arm from a prepared model, the GpuContext, Session options and feeds through the
    public API; a caller with a pipeline (an LLM decode) passes its own `open` / `run` / `close` value, with an
    optional `report`, and the runner never looks inside it.
  - The rule of decision 9 applies: never run it while other work uses the same GPU (the times would measure the
    contention; the runtime cannot see other work).
- **Ruling 2: the part of the fused-vs-unfused check that the public API can express lives in `./tune`.** That part
  is the u32 identity comparison with a NaN policy, `wordIdentity`, which the A/B and `checkFusionTwin` of
  `packages/kernel-checks` both use (one implementation). Why: the models package's tests may not import
  `packages/kernel-checks` (the test import rules of [0129](0129-test-layout-and-lanes.md), addendum of 2026-10-29), and a rule written outside the runtime
  needs the same comparison. This changes the earlier design proposal (its question 4), which had kept the whole twin harness in
  `packages/kernel-checks`. The built-in checks that drive runtime internals stay there; nothing else moves.
- **Why public**: an application or a package that passes its own fusion rule (ADR 0148) has to measure and check it
  on its users' backends; the hand-made script of 2026-11-10 imported runtime internals to do it. The entries are
  high-level and of the rank decision 2 asks for: the runner, the arm builder and two pure functions of a record.
- **Checks**: `packages/runtime/tests/tune_session_ab_test.ts` (synthetic arms and a fake clock: the three verdicts
  with hand-checkable numbers, the noise floor and the interval of the median each deciding a verdict (rounds that
  tie, one and two outlier rounds of eight, the same verdict from 3 to 8 rounds), the boundaries |median delta| =
  floor and gain = `minGain`, the visit order for two and three arms, open, close, the report, the untimed run and the
  reading of the outputs outside the timed spans in both opening modes, counts that change between runs refused,
  closing on a failing run and on failing closes, the identity failures (another array type, a base that rewrites one
  array, the untimed run), the refusals of the runner and of the judge, a record judged again after a JSON round
  trip) and `packages/runtime/tests/gpu_session_ab_test.ts` (two `sessionArm`s on the GPU, without and with an
  external test rule, the adapter passed as the `GPUAdapterInfo` itself: identical outputs, the rule's hit and one
  dispatch fewer in the record; no verdict on real times). Fault injections (each turned the first test red): the
  reverse half of the visit order dropped, the open timed inside the span, the identity comparison skipped for the
  non-base arm, the noise floor forced to 0, the interval test removed, the interval fixed at the smallest and largest
  difference, the counts read inside the span,
  the comparison of every run's counts removed, the untimed run of `openPerVisit` removed.
- **Not in this addendum**: the playground's page that runs the A/B on real models, and running it on real weights.
  Both were built later the same day ([0148](0148-external-fusion-rules.md) addendum of 2026-11-11): the `/session-ab`
  page, whose arms are two loads of one Gemma 4 pipeline with and without a candidate rule and whose output is the
  generated token ids, and the opt-in Node run on 26B. Neither changes the runner, the record or the judge.

## Addendum (2026-11-12) — attention-bench records as the derivation's second input (amends decisions 2, 3, 4 and 5)

The maintainer approved adopting the online attention's tiled key axis (perf-ledger K-106) for Ampere on 2026-11-12,
through the derivation and never by a hand edit of a generated table. A geometry profile now has one field that is a
kernel variant rather than a geometry, `attentionOnline` (`{ keyWords: "tiled" }` or absent — ADR
[0132](0132-online-int8-attention.md) addendum 2026-11-12), and this addendum records how the derivation decides it.

- **What changed**: `deriveGeometryProfile` takes an optional second input, `DeriveGeometryProfileOptions.attentionBench`
  — attention-bench records (`karume-attention-bench/2` from `runAttentionRouteBench` with `onlineTiled` and
  `quantization`) as `AttentionBenchInput` (the record, a path, the SHA-256 of its bytes). They decide `attentionOnline`;
  without them the table has neither the field nor its provenance, and every derivation is unchanged. `./tune` gains
  `ATTENTION_BENCH_SHAPES`, the bench's default shapes and the set the records must cover: Wan2.2 TI2V-5B's three
  attention shapes and MOSS-Image's 1024² attention (`moss-image-1024`, S = 4,159).
- **The rule** (amends decision 3 for this field): adopted only when every shape of every record is identical (the
  record's identity block: 0 differing O words and 0 differing scale words — both producers on the same f32 Vᵀ, both
  kernels on their outputs) and every shape of `ATTENTION_BENCH_SHAPES` gains at least `minSpeedup` (×1.05 by default,
  the geometry fields' threshold). The gain of one record at one shape is (N's kernel per node + N's producer) ÷ (tiled
  kernel per node + tiled producer), each term the arm's minimum pass ÷ its repetitions, the kernel's scaled from the
  measured window to the node's query rows; the records of one shape combine by the geometric mean of their gains.
  Otherwise the field is absent (the form N gives).
  - **Why minimums** and not the record's median of the paired rounds' ratios: a stall only adds time. In the RTX
    3080 Ti record of 2026-11-11 at the cross-attention shape (`outputs/diag/attn-tiled-2026-11-11/run1/`), N's
    kernel ran at about 6× its time in 5 of its 10 passes, so the median of the per-round ratios is 0.243 (a void
    ×4.1 gain) while the ratio of minimums is 0.860. The minimum per dispatch is decision 3's statistic for the sweep.
    A band filter like ADR [0115](0115-geometry-profiles.md) addendum decision 8's would drop clean data here: the
    passes of a stall-free arm in those records spread up to ×1.15 around its minimum. The minimum assumes one clean
    pass per arm; a record whose arm had none overstates the gain (the RTX 3080 Ti record of 2026-11-12 at the
    cross-attention shape, run 1 — ADR 0132 addendum 2026-11-12).
  - **Why kernel + producer**: it is what a Session pays per node, and the tiled producer is slower than N's (×1.26
    per dispatch at 121f self-attention), so a kernel-only gain could hide a node that loses.
  - **Material check**: an observation whose kernel or producer pair failed, or whose rounding-error bound
    E = q ÷ min(A) + q ÷ min(B) on either pair exceeds `ROUNDING_ERROR_LIMIT` (1%), is dropped (q = Chrome's 100 µs
    when every pass of the record is a multiple of it, else 0 — the bench record carries no quantization flag); a
    shape left without an observation keeps the default. An observation that is not identical keeps the default
    whatever the gains.
- **Refused records** (fail loudly — measure again with the current runtime): another format, or
  `settings.onlineTiled` / `settings.quantization` not both true; the wall clock (`timing.unit` must be `ns` or
  `deno-raw-tick`); another adapter than the sweep records' four fields; one record given twice; a shape outside
  `ATTENTION_BENCH_SHAPES` or with other dimensions; a kernel pair whose keys are not the current runtime's
  `attentionOnlineI8a8Key` of N's form and of `tiled` (the record's dp4a variant, the online route's reduction form,
  `bhsd`), or a producer pair whose keys are not `attentionValueQuantization`'s; a measured window other than the first
  one production runs; and records that together leave a shape of the set unmeasured. Coverage is checked over all the
  records given, because the Node bench writes one record per shape. The playground's `/attention-bench` file holds the
  record under `bench`: the application passes that value (the runtime does not read the page's wrapper).
- **What is written** (amends decision 4): `attentionOnline` exactly when adopted, and `provenance.attentionBench`
  (`records`, `sha256` and `date` joined with `", "` in the order given, and `shapes`, the id of the bench's shape set:
  FNV-1a 64 of the canonical JSON of `ATTENTION_BENCH_SHAPES`) whenever records are given. The verdict lines gain the
  field's line, with each shape's combined gain and each record's. The kernel fingerprint gains, for a table carrying
  the field only, the online attention's kernels the field chooses between: per bench shape, for N's form and `tiled`,
  the Vᵀ producer and the first window's kernel in both dp4a variants × both reduction forms (`bhsd`). A table without
  the field hashes exactly the lines it hashed before, so the three built-in tables keep their `provenance.kernels`.
- **Matching** (amends decision 5): when a table has `provenance.attentionBench`, `geometryProfileMismatch` also
  compares its `shapes` with the current runtime's shape-set id, after the case-set version.
- **Reading and guarding**: `parseGeometryProfileJson` reads `attentionOnline` (`keyWords` must be `"tiled"`) and
  `provenance.attentionBench` (four strings), unknown fields refused as everywhere; `assertGeometryProfile` checks both
  shapes; the frozen copies of `BUILTIN_GEOMETRY_PROFILES` copy both.
- **The bench**: the record format is unchanged. The `online` row times the route's form on the device and reports it
  (`keyWords`); the candidate pair's A arm and the quantization line's `quantize_vt` stay N's form, so a record
  measured on a table that chose `tiled` is still a derivation input. `WAN22_ATTENTION_SHAPES` stays exported; the
  bench's default is now `ATTENTION_BENCH_SHAPES`.
- **First use**: `nvidia-ampere`, derived from its two sweep records and eight Node records of 2026-11-12
  (`outputs/diag/attn-tiled-2026-11-12/`), keeps N's form — `moss-image-1024` gains ×1.046 (×1.044 and ×1.048), under
  ×1.05; Wan2.2's shapes gain ×1.072 / ×1.119 / ×1.687 — and every other field is identical to the checked-in table, so
  it is not regenerated. The numbers are in ADR 0132 addendum 2026-11-12.
- **Checks**: `packages/runtime/tests/tune_attention_profile_test.ts` on synthetic records whose keys the current
  runtime builds (`tests/helpers/attention-bench-records.ts`): adoption and what is written, the fingerprint covering
  the field (and a table matching the current runtime), a stall burst in three of five rounds still read at the
  minimums' gain, records combined by geometric mean (×1.25 with ×0.97 adopts, ×1.10 with ×0.98 does not), one shape
  under ×1.05, one shape not identical, a failed pair and a quantised record over the rounding bound, per-shape records
  covering the set together, and every refusal above; `tune_profile_json_test.ts` (both fields read back and refused
  when malformed); `tune_fingerprint_test.ts` (the built-in tables' fingerprints unchanged).

## Addendum (2026-11-12) — the matching keys per op family (amends decisions 4, 5, 8 and 10)

The maintainer approved this design on 2026-11-12 (the backlog item "Scope the geometry sweep's `caseSet` per op family",
filed 2026-10-27 after the ruling of 2026-10-26).

- **Why**: the case-set version was one hash over every case of `SWEEP_CASES`, and the kernel fingerprint one hash
  over the kernels of every case. Adding one op family's cases moved both for every table: every embedded and every
  saved table mismatched and every sweep record was refused, also for the families whose cases had not changed. And a
  table whose records had no case of a class still carried the whole case set (`apple-metal-3` and its `i8a8.conv3d` —
  ADR [0115](0115-geometry-profiles.md) addendum decision 11).
- **The families**, the unit a class's evidence stands on (`GeometryProfileFamily` in
  `src/kernels/geometry-profile.ts`, since kernels do not import tune; the maps from sweep op and from profile field to
  family are one table each in `src/tune/families.ts`, written with `satisfies` so that a new op or field must be
  classified):

  | Family | Sweep ops | Profile fields |
  | --- | --- | --- |
  | `gemm` | `linear`, `matmul`, `bmm` | `gemmRows` (every bucket) |
  | `attention` | `attention` | `attention.qk`, `attention.pv` |
  | `conv2d` | `conv2d` | `conv2d.rows64`, `conv2d.rows32` |
  | `i8a8-linear` | `i8a8-linear` | `i8a8.linear` |
  | `i8a8-attention` | `i8a8-attention` | `i8a8.attentionQk`, `i8a8.attentionPv` |
  | `i8a8-conv3d` | `i8a8-conv3d` | `i8a8.conv3d` |

  The row-count bucket bounds (`PROFILE_GEMM_ROWS_BOUNDS`) enter the `gemm` family's case-set version only. The hashed
  form has a bounds entry per family (absent for the other five), so another family can get bounds without a change
  of format. (Updated 2026-11-12: `i8a8-linear` has bounds of its own, `I8A8_LINEAR_ROWS_BOUNDS` — ADR
  [0115](0115-geometry-profiles.md) addendum decision 14.)
- **Provenance** (amends decision 4): `adapter` (unchanged) and `families`, one entry per family with `sweep`,
  `sha256`, `date`, `candidateSet` and `userAgent` restricted to the records that have a row of the family, `kernels`
  (the fingerprint over the family's cases only) and `caseSet` (the version over the family's cases and bounds). A
  family that no record has a case of has no entry, and its fields are the default. The single `sweep`, `sha256`,
  `date`, `candidateSet`, `userAgent`, `kernels` and `caseSet` are gone; `parseGeometryProfileJson` refuses them as
  unknown fields, and there is no reader for the earlier shape.
- **The online attention's form is not a family**: `attentionOnline` is decided from attention-bench records, not
  sweep cases. The online attention's launch lines, which the single fingerprint appended for a table carrying the
  field (addendum of 2026-11-12 on attention-bench records), are now `provenance.attentionBench.kernels`, their hash
  alone (`attentionBenchKernelsId`), present exactly when the table carries the field. No family fingerprint contains
  them, and a table without the field hashes nothing new.
- **The sweep record is `karume-geometry-sweep/3`** (amends decision 8): `caseSet` becomes `caseSets` and
  `defaultKernels` becomes a map, both with one value per family of the planned cases, each hashed over the candidate
  set's own case table restricted to the family (an evaluation set still never claims the sweep's case set).
  `candidateKernels` is unchanged. The reader accepts `/3` only: for every family the record has a row of,
  `caseSets[family]` and `defaultKernels[family]` must be present and equal the current runtime's, and a missing key
  throws. The leniency for records without the keys goes with `/2`. An application that merges per-case records takes
  the union of their keys, since each record holds the families of its own cases (the playground's page does).
- **Derivation with a base table**: `deriveGeometryProfile(reports, { …, base })`. A family is covered when some
  record has a row of one of its cases, and a covered family is derived from the records as before. A family the
  records do not cover takes `base`'s fields of the family and its provenance entry verbatim, after the entry's
  `kernels` and `caseSet` are checked against the current runtime (a stale entry throws, naming the family: sweep this
  family again). Without a `base` entry the family keeps the default and gets no entry; a `base` whose fields of that
  family are not the default is refused (a table edited by hand). With `base`, `base.provenance` must exist with the
  records' adapter, and a covered family must have every case of its current case set in the records: carrying over
  promises that nothing else moves, so a half-swept family must not silently put its unswept fields back to the
  default. Without attention-bench records, `base`'s attention-bench block and its `attentionOnline` are carried
  likewise after `shapes` (and `kernels`, when present) are checked (stale: measure the attention bench again); a
  `base` carrying `attentionOnline` without the block, or without its `kernels`, is refused. `id` and `match` still
  come from the options, and at least one record is still required. A carried field gets one verdict line naming the
  base table and the sweeps of its entry. Without `base` the coverage rule is unchanged: each field is judged on the
  cases of it the records have, and a field with no case keeps the default (a family with some of its cases swept is
  derived from those cases, and its entry carries the family's whole case-set version).
- **Matching** (amends decision 5): `provenance` is present → the four adapter fields → per family in the fixed order
  (`gemm`, `attention`, `conv2d`, `i8a8-linear`, `i8a8-attention`, `i8a8-conv3d`): with an entry, `kernels`, then
  `caseSet`; without one, the family's fingerprint equals the default table's ("no provenance for family X but its
  geometry is not the default"). Then the online attention's block: a table with `attentionOnline` needs the block
  with `kernels`, `kernels` is compared when present and `shapes` whenever the block exists. The first mismatch is
  returned as a message that names the family; the function still returns `string | undefined`.
- **Public surface**: `./tune` exports `sweepCaseSetIds(set?)` and `sweepDefaultKernelsIds(set?)` (one value per
  family, `GeometryProfileFamilyIds`) in place of `sweepCaseSetId`, whose callers no longer compile; `.` exports
  `GeometryProfileFamily` and `GeometryProfileFamilyProvenance`. The default table's fingerprints are public because
  a record must now carry them for every family it has rows of: records written by `runGeometrySweep` carry them, but
  the function gives the current runtime's values without running a sweep, which a caller that builds or checks a
  record without a GPU (the playground's record tests) needs now that a record without them is refused.
- **The embedded tables** are converted in format only. Before the conversion, each table's single `kernels` and
  `caseSet` equalled the runtime's (the equality `tune_fingerprint_test.ts` pinned), so its per-family keys are those
  of the same runtime. A family has an entry exactly when one of the records the table names has a row of it.
  `nvidia-ampere` and `nvidia-blackwell` have an entry for every family (each of their two records has rows of all
  six); `apple-metal-3` has none for `i8a8-conv3d` (no record has a conv3d case), and its other entries name the
  records with rows of the family (five for `gemm`, four for `attention` and `conv2d`, three for the two i8a8
  families). No field value, geometry, kernel, codegen snapshot or sha row moves.
- **Consequences** (amends decision 10): a table saved before this change no longer parses and a `/2` record is
  refused, so an application sweeps once more. From then on a runtime change that touches one family's kernels,
  cases or bounds invalidates only that family's entries, and a sweep of that family derived with the stored table as
  `base` restores the table.
- **Checks**: `tune_fingerprint_test.ts` (each family's keys move with that family's fields, kernels, cases and bounds
  only; the built-in tables' entries are the current runtime's and a family without one is the default);
  `tune_profile_test.ts` (a record missing a family's key is refused, naming it; a family without rows is not read;
  `/2` is refused); `tune_profile_base_test.ts` (the carry-over copies fields and entry verbatim; a stale base entry,
  a base whose family without an entry is not the default, a half-covered family, a base without provenance or of
  another adapter, and a stale or incomplete attention-bench block are refused; each entry names only the records
  with rows of its family); `tune_mismatch_test.ts` (the message names the family, the order, a family without an
  entry, every built-in table against its own adapter); `tune_attention_profile_test.ts` (`attentionBench.kernels`
  exactly when adopted, no family fingerprint moved by the field, the matching rule); `tune_profile_json_test.ts`
  and `geometry_profile_test.ts` (the shape).
- **Not in this addendum**: a `base` input on the playground's `/geometry-sweep` page, and a structured answer to
  "which families are stale" (the message names the first one).

## Addendum (2026-11-12) — the `geometries` sweep option (amends decisions 2 and 8)

- **What.** `runGeometrySweep({ geometries })`: the candidate geometries to measure, by the name a record's rows carry
  (`geometry`, e.g. `tile64x64r8x4w16x8k64s`), in the order given. The default geometry's row is measured as always,
  first. Omitted (or an empty list, as for `cases`), every candidate of the set is measured, as before. A name that
  is not a candidate of every chosen case, or a name given twice, throws before the GPU is touched; the check and the
  sweep take the candidates through one function (`selectedCandidates` in `src/tune/harness.ts`).
- **Why.** A machine that heats within one case measures that case in chunks with a cool-down between calls. The
  i8a8 linear's `full` grid holds 230 geometries per case (ADR [0115](0115-geometry-profiles.md) addendum decision
  14), and in the K-85 evaluation on the RTX 3080 Ti every case but the guard drifted ×1.14–1.21 over its run
  (perf-ledger K-85), outside the 0.9–1.1 range in which the derivation keeps a case's ratios (ADR 0115 addendum
  decision 8).
- **The record** (amends decision 8): the option is recorded in `settings.geometries` (absent when omitted). A record
  taken with it is an ordinary derivation input: the generator does not read the field, each row stands for itself,
  and a candidate the record did not measure is "not measured" there, as with narrowed `cases` — the chunks of one
  case, given together, are the case's observations.
- **Checks.** `tune_entry_test.ts` (the refusals before the GPU; on a real GPU the default row first, then the named
  geometries in the order given, `settings.geometries` recorded and the record derived); `tune_geometries_test.ts`
  (`selectedCandidates`).
