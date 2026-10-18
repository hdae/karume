# 0032: Non-serialising submit scheduling (no per-submit `onSubmittedWorkDone`) and explicit opt-in GPU timing

- Status: accepted (2026-08-04); partly superseded by [0054](0054-resident-loop-and-fence.md)
  (decision 1: a normal run completes on one `mapAsync` fence, 2026-08-13) and amended by
  [0123](0123-submit-inflight-bound.md) (a completion marker every 64 submits, the one exception to
  decision 1, 2026-10-07); decision 4's demo flag went with the examples
  ([0127](0127-node-dawn-toolchain.md) decision 5) and its benchmark rule was re-graded to SHOULD on
  2026-10-18
- Date: 2026-08-04
- Related: [0004](0004-execution-model.md) (adaptive control — this ADR revises its window ④) /
  [0021](0021-gpu-timing-diagnostics.md) (GPU timing diagnostics — this ADR revises decision 1,
  "`undefined` = auto", the re-ruling that ADR announced if problems appeared) /
  [0030](0030-attention-a8-execution.md) and [0031](0031-attention-score-f16-storage.md) (their
  "non-GPU fixed cost" is explained by this Context) / recon:
  [2026-08-04-host-overhead-recon.md](../research/2026-08-04-host-overhead-recon.md) (attribution
  measurements, isolation experiments, options 1–3)

## Context

The attribution recon of the "non-GPU fixed cost" of 1024 px turbo (695–704 ms per step) split it
into three parts:

1. **Timing overhead, 370–375 ms per step, independent of resolution.** The auto default of
   `gpuTiming` made demos and benchmarks always run with timing on, which encodes one pass per
   dispatch (ADR 0021 decision 2). The cost of `encoder.finish()` grew linearly with the number of
   passes × the number of buffers ever created on the device (measured on wgpu; the mechanism inside
   wgpu is a guess), so with throw-away params buffers it **grew with the step number** (non-GPU
   470 → 706 ms from step 2 to 10; flat with timing off, so not thermal).
2. **True host encoding, about 302 ms per step** (breakdown in recon §3.1).
3. **The structural cause: a per-submit `queue.onSubmittedWorkDone()` (fire-and-forget) blocked the
   host on Deno / wgpu** in its synchronous part until all submitted work had finished, serialising
   CPU encoding and GPU execution (wall = GPU + host; in a bare WebGPU isolation experiment, calling
   it gave wall = gpu + host and not calling it gave wall = max(gpu, host) — recon §4.1). Browser
   implementations do not block there.

errorScope, GC and the await chain were cleared.

## Decision

1. **`onSubmittedWorkDone` MUST NOT be called per submit** (`packages/runtime/src/gpu/submit.ts`).
   It is called once, in `flush()`, where the wait is needed anyway for flush-before-destroy; CPU
   encoding then overlaps GPU execution. Rationale: on some implementations its synchronous part
   blocks the host until all submitted work is done (measured on Deno 2.9.6 — ADR 0123 decision 5);
   on Dawn, the development implementation since ADR 0127, it has not been measured, and not calling
   it costs nothing. (Revised 2026-08-13 by ADR 0054 decision 6: a normal run — `gpuTiming` off and
   at least one graph output — completes on the readback `mapAsync` as its only fence; the flush's
   `onSubmittedWorkDone` and the re-flush after `arena.destroy` were removed. Amended 2026-10-07 by
   ADR 0123 decision 1: between steps a run places one completion marker when 64 submits have
   accumulated since the previous marker; it is not a fence.)
2. **The observation window of the adaptive control is one flush.** Window = the first submit of
   the window → resolution of the flush's `onSubmittedWorkDone`; estimate = measured time ÷ total
   workgroups of the window. ADR 0004 invariants ①–③ hold (a measurement of 0 **or no work** carries
   no information and does not update — no work would give 0 ÷ 0 → Infinity and pin the size at
   `minChunkSize`; the decide-before-adding rule is unchanged; the unit is the workgroup). The
   estimate includes host encoding time, so it is larger than the GPU time: chunks shrink, which is
   the TDR-safe side. `minChunkSize` / `maxChunkSize` bound it. (Current form: a normal run closes
   the window right after its single `mapAsync` fence, and a `beginBatch` scope at
   `BatchScope.finish`, both through `closeMeasurementWindowAfterFence` —
   `packages/runtime/src/gpu/submit.ts:33-49`. The window still includes host time.)
3. **The default of `gpuTiming` (`undefined`) is "do not request"** (revising ADR 0021 decision 1;
   the same rule as `shaderF16`). `true` = required (absent feature fails loudly) is unchanged;
   `false` is the explicit form of the default. Rationale: with auto-detection the cost of timing
   (one pass per dispatch) lands on the default path depending on the adapter, and "slower because
   measured" becomes the performance baseline. Diagnostics are paid for explicitly.
4. **Benchmarks: plain timing runs and breakdown runs SHOULD be separate runs, each preceded by a
   cool-down** (re-graded 2026-10-18 to SHOULD: the size of the timing overhead depends on the
   implementation — measured on wgpu, not on Dawn or Chrome). The breakdown comes from an explicit
   `gpuTiming: true` device (`Session.diagnostics().lastRunTiming`). Withdrawn 2026-10-18: the demo
   flag `--gpu-timing`; the Deno CLI examples were not ported (ADR 0127 decision 5).
5. `SubmitStats.pendingMeasurements` was removed on 2026-08-04 (the fire-and-forget count it
   reported no longer exists; a breaking change without a shim before release).

## Measurements (2026-08-04, after the change, production form, seed 42, 10-step mean)

1024 px wall per step 2,131–2,299 → 1,502–1,512 ms (−29 to −35%); 512 px 971.6 → 431–438 ms
(−55%). The old values are the old default (timing on, old scheduler). The sha256 checks of both
series were unchanged. Full record: commit `99a3c649`, this file.

## Detection limits

1. **A regression of this ADR (per-submit `onSubmittedWorkDone` returning) cannot be detected
   numerically** — outputs and statistics do not change by one bit; it is only slower. The detector
   is a wiring check that counts the calls (`FakeGpu.calls` in
   `packages/runtime/tests/gpu_submit_test.ts`; MUST in its doc).
2. **A key check that reads the timing table (`lastRunTiming` entries) passes vacuously when timing
   is off** — fault injection F7 broke the expected keys and stayed green under a plain
   `acquireGpu()`. Since 2026-09-26 the census reads the timing-independent
   `SessionDiagnostics.lastRunPipelines` (ADR 0110 decision 5,
   `packages/runtime/tests/helpers/pipeline-census.ts`).
3. **The cost of `encoder.finish()` grows with the number of buffers ever created and does not drop
   with `destroy`** (wgpu micro-measurement: +5 µs per pass per 1,000 buffers) — the reason a params
   cache also stops the growth of the timing overhead.
4. With timing off by default, the timing path (one pass per dispatch) lost its incidental coverage;
   only tests with an explicit `gpuTiming: true` reach it. New kernels need that path in mind.

## Consequences

- **Wall time approaches max(GPU, host).** Extra dispatches (for example the +224 per step of
  attention a8) no longer hit the wall clock directly, which relaxes the "conversion dispatches make
  non-GPU time worse" constraint of the f16 intermediate design (§3.4 ③). For the host-bound side
  (512 px and below, SBV2, DeBERTa) the recon's option 2 (params / bind group caching) was later
  implemented: the content-addressed params cache and the prepared execution plan of
  [ADR 0042](0042-prepared-execution-plan.md) (see also [limitations](../limitations.md)).
- **Adaptive control now updates once per run** (before, whenever the queue drained). The first run
  runs at `initialChunkSize` throughout (conservative, TDR-safe).
- The "non-GPU fixed cost" of ADR 0030 / 0031 (~350 / 695 ms per step) included the timing overhead
  and depended on the step number.
- The per-workgroup estimate (`#msPerWorkgroup` of `SubmitScheduler`) is now an upper bound that
  includes host time. Using GPU timestamps as the
  estimation source when available is parked in the [backlog](../backlog.md).
