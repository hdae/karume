# 0021: GPU timing diagnostics (timestamp-query, per-op breakdown)

- Status: accepted (2026-08-03 — the maintainer chose option A with a three-valued option); partly
  superseded by [0032](0032-deserialize-submit-and-optin-timing.md) (decision 1: the default does
  not request the feature, 2026-08-04) and [0028](0028-f16-compute-variants.md) (decision 4: no
  shader-side feature dependency on the default path only, 2026-08-04); marked in place below.
- Date: 2026-08-03
- Related: [0006](0006-quantization.md) (always-on diagnostics) / [0127](0127-node-dawn-toolchain.md)
  (Dawn timestamp quantisation)

## Context

Performance work needs the real GPU time per op; wall-clock timing per stage mixes chunks and ops
and cannot rank optimisation targets. The submit layer encodes one chunk as one compute pass
(`packages/runtime/src/gpu/submit.ts`), and WebGPU places timestamps only at pass boundaries
(`timestampWrites`). Per-dispatch timing inside a pass is a native wgpu extension, not standard.

## Decision

1. **The opt-in is an option of `acquireGpu`**, because a feature can only be requested when the
   device is created. `gpuTiming?: boolean` has three values:
   - `undefined` (default): originally "auto" — request `timestamp-query` when the adapter has it.
     **(superseded by ADR 0032 on 2026-08-04: the default does not request the feature.** Timing
     costs one pass per dispatch, measured at 370–375 ms per step, so the default path's wall clock
     depended on the adapter. Same rule as `shaderF16`.)
   - `true`: required; without the feature `acquireGpu` fails loudly (`GpuFeatureError`).
   - `false`: not requested; same behaviour as `undefined`, stated explicitly for control runs.
2. Only when timing is on, each dispatch is encoded as its own pass. Pass begin / end timestamps go
   into a per-chunk query set and return through `resolveQuerySet` and a readback. The query set
   holds 2 × the chunk's dispatch count, capped at 2,048 (`MAX_TIMESTAMP_QUERIES` — twice the
   default `maxChunkSize` of 1,024, inside the WebGPU limit of 4,096). Dispatches in one pass are
   already ordered, so splitting passes does not change execution semantics.
3. **Aggregation is per pipeline key** (total time, dispatch count, workgroup count). The key
   already names the op and its variant (`:wi8` and so on), so no new naming is introduced. The
   table is `Session.diagnostics().lastRunTiming`, with the lifetime of the other last-run fields.
4. "No required features" reads as "**by default** nothing is requested": with `false` or without
   the feature, `requiredFeatures` is empty, and shaders depend on no feature such as `enable f16`.
   **(narrowed by ADR 0028 on 2026-08-04: on the default path only.** The opt-in f16 compute
   variants, which require `acquireGpu({ shaderF16: true })` and a runtime canary, emit
   `enable f16`; the default f32 WGSL is unchanged, which the snapshots check.)

## Verification

- On a real GPU: the summed dispatch count matches `SubmitStats.dispatchCount`, every entry is
  ≥ 0, the key set equals the pipelines that ran, and each value behaves as specified without the
  feature. The same graph with timing on and off gives identical output bytes.

## Consequences

- Timing is not free (370–375 ms per step, independent of resolution); a breakdown is taken in a
  separate run with `gpuTiming: true`.
- Implementations may quantise timestamps. Totals over many dispatches are robust; a single
  dispatch's value is indicative only. Dawn (ADR 0127) quantises to 65,536 ns by default;
  measurements that need finer steps MUST disable it (`KARUME_DAWN_FLAGS`), and the test harness
  disables it by default (ADR 0127 Consequences and Open points).
- Rejected: B — per-chunk timing (ops mixed); C — finer CPU timing (`onSubmittedWorkDone` reports
  cumulative completion, a known trap of ADR 0004).

## Addendum (2026-09-25) — timestamp unit on Deno

Deno's WebGPU returned raw ticks, not nanoseconds, so totals were off by `timestampPeriod` on GPUs
where it is not 1 (Intel Arc B570: 52.0833 ns). Details: `docs/known-issues.md`, Intel Arc B570.

## Addendum (2026-09-26) — the pipeline-key list no longer needs timing

`SessionDiagnostics.lastRunPipelines` gives the per-key dispatch count of the last run with or
without timing, on `run` and on the resident path's `enqueue` (ADR 0054), derived on read from the
run's recipe list — so a fallback (ADR 0058 decision 3) is visible where timing cannot be used.
The counts are planned values (a zero-work states-shaped dispatch is counted but not issued), so
they can exceed the timing breakdown's `dispatchCount`.
