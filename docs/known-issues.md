# Known issues — open bugs

> Placement rule: open bugs (intended design constraints go to [limitations.md](limitations.md)).
> When a bug is resolved, delete its section and leave no pointer to the fixing commit (git holds the history).

## A fixed revision on the pseudo-HF server reuses a stale manifest

The HTTP connectivity test server `packages/test-support/src/local-dist-server.ts` fixes the repo and the revision.
The manifest cache key is the resolve URL, so when a previously used port is assigned again, a manifest of a different
generation from the current directory can be hit. An entry pointing at the old one-output Gemma graph remained, and in a
reproduction experiment that updated only the content under the same URL and revision, the old manifest was returned.
[Investigation and reproduction steps](research/2026-09-13-rms-subgroup-reduction.md#%E5%85%A8%E4%BD%93%E6%A4%9C%E8%A8%BC%E3%81%AE%E5%88%87%E3%82%8A%E5%88%86%E3%81%91).

A candidate cause of the event in the full verify of 2026-09-13 where the graph was rejected as one-output while a
standalone run succeeded. The port at the time of the failure was not saved, so the causal link to this event is
unconfirmed. Do not equate it with the GPU flakes.
The fix candidate is to tie the test server's revision to the distribution content, so that generations are distinguished
even when a port is reused.
Ordinary local CLI and browser comparisons do not use this pseudo-HF server.

## Full `deno task verify` runs rarely flake under GPU VRAM pressure

On machines where the models are large relative to VRAM, feeding GB-class models in succession **rarely produces
`GpuOutOfMemoryError` / `GpuDeviceLostError`**. The failing test differs every time (not a defect specific to one test),
and **re-running the failed file alone is green**. It appears in the same form not only in full runs but also **when lanes
are run back to back without a gap** (on some machines the release is delayed until the next device poll — the
"Intel Arc B570" section below).

Operational workaround = **leave a few minutes between lanes**, or **re-run the failed file alone to confirm**
(green means a flake). However, in an environment that panics instead of raising an exception on device lost
(Deno 2.9.6 — same section), the whole process disappears, so the run itself stops there before any standalone re-run.

## On Metal (Apple GPU), the GPU output of attention i8a8 is 1 ULP off the TS reference (+ 4 conv1d/conv2d parity tests + 2 gru_scan parity tests + 1 linear GEMV u32 check)

On real **Apple M2** hardware (first seen on Deno 2.9.4; re-verified on 2.9.6 on 2026-08-29; the section's targets
inventoried in the full verify of 2026-08-31; **re-measured on 2026-09-02 at HEAD `2b096c4` after memory management
phase A on Deno 2.9.4 = 12 tests in total reproduce with the same signature, 0 new**; **re-measured on 2026-09-03 at
HEAD `b7e32c1` after the review-fix phase on macOS 26 = the same 12 reproduce with the same signature. The full verify is
1856 passed / 13 failed / 139 ignored, and the 13th is not a target of this section but of the `--diagnostics` section
below**), 4 attention i8a8 tests
(3 in `gpu_attention_pv_i8a8_test.ts` + 1 in `gpu_attention_pv_i8a8_test.ts`) +
**2 conv1d parity + 2 conv2d parity tests**
(the old text listed conv2d only — the 2 conv1d tests were missing from the record; same symptom) +
**2 decomposition parity tests of gru_scan / gru_scan_reverse** (below) + 1 GEMV u32 check + 1 OOM check
(next section) are red (Linux / Vulkan are all green).
**The canary hardware verification of 2026-08-29 updated the understanding of the mechanism**:

- **The dp4a and emulation variants are bit-identical on M2 too** (both canary arms equal; the PV mutual-agreement
  test green; 0% mismatches over 62,088 elements of the qP integer stage). The old text "since integer arithmetic has no
  rounding difference, QK / PV actually produce different values" (an inference of an inter-variant mismatch)
  **was wrong** — the failing comparison was **GPU vs the TS reference**, and the old observation was most likely of
  the same form (withdrawn 2026-08-29).
- What actually happens is that **the output of the f32 epilogue (applying the scale) shared by both variants is exactly
  1 ULP off the TS reference** (1.9e-6 near QK 28.4, 9.5e-7 near 12.0, 1.5e-8 near PV 0.2). The integer stage matches
  exactly. Hypothesis (unconfirmed): a difference in FMA contraction from naga → MSL changes the rounding of the
  multiplication chain.
- The only real harm is that **the atol=0 cross-platform parity check does not hold on Metal**. The quality impact is
  negligible at 1 ULP (as the track record of the Mac generating correct images shows). Browser execution goes through
  Dawn / Tint, not naga, so the same symptom is not guaranteed (unverified).
- Variant selection is handled by **decision rule v2** of the run-time canary (ADR
  [0058](decisions/0058-numerics-opt-in-contract.md) addendum) — "both arms bit-identical, inside the band (rtol 1e-5)
  against the reference" picks dp4a and continues with one warning (a8 runs). Only outside the band is `GpuFeatureError`.
- **The fixed input of canary ① QK was moved onto the f16 grid** (2026-08-30, RC1-1): by restricting the multiplier to
  powers of 2 and keeping the significant digits of `acc` within the f16 mantissa, all 8,192 elements of the known answer
  are exact f16 values, and **no rounding occurs even once** in the epilogue. Hence ① QK is expected to match the known
  answer exactly both on a healthy device and on M2, and **this 1-ULP difference will from now on be observed only on the
  ③ PV side** (③ PV retains rounding because the GPU produces `qP = round(127·exp(S−m))`). **Re-confirmed on real M2
  hardware (2026-09-01)**: all 16 canary tests of the time green — the plain decision picks dp4a, and the branch and the
  exact-match flag are consistent; the fault-injection tests behave as expected (the current count is 17, and the one
  added later has not been measured on M2).
  The axis reduce parity tests (2 in `gpu_reduce_axis_parity_test.ts`) are also green on M2.
- **The 4 conv1d / conv2d parity tests** (bit agreement implicit GEMM ↔ direct kernel; the golden tolerance judgement
  is green) remain without an identified cause — most likely the same kind of epilogue rounding difference, but unverified.
- **The 2 decomposition parity tests of gru_scan / gru_scan_reverse are red on M2 (measured 2026-08-31)**:
  in `gpu_gru_scan_parity_test.ts`, T4 N1 H128 has 158/512 and 71/512 elements disagreeing by 1–64 ULP
  (no NaN; the cases with small H are green; Linux / Vulkan all green). **Not a regression of the tanh_stable change
  (gru_scan v2)** — the same signature (same cases, same counts, same bit patterns) was measured red at `b35cf5c`, the
  HEAD before the change = pre-existing, and a demonstration that v2 is in-band bit-invariant on Metal too. Suspected
  mechanism = the same class as the head of this section: MSL contraction spans the reduction / update expressions on the
  fused side, and **the rounding barrier of the workgroup memory round trip does not act as a barrier on Metal** (the
  form that the doc of src/kernels/gru-scan.ts anticipated — "the rounding barrier is not a guarantee of the WGSL
  specification"). It surfaces when H=128 = the reduction length grows.
  The real harm, as elsewhere, is only that "the cross-path atol=0 check does not hold on Metal" — the actual quality is
  ensured by the vowel-detector golden (a tolerance check) being green on M2. Root-cause candidate = strengthening the
  expression form of the rounding barrier
  (a bitcast round trip was measured to be ineffective, so another means — grouped with the root-cause candidate below;
  needs a real-M2 loop; not started).
- **The exact u32 check of linear's GEMV family (M=1 × i4 — ADR [0082](decisions/0082-linear-gemv-decode.md)) is red
  by 1 ULP on M2 (measured 2026-09-01)**: the check-key test of `gpu_linear_gemv_test.ts` is
  green; the u32 comparison against the default path fails at column 1 of the divisible shape k128 n64 g32 with
  `0x414b3249` vs `0x414b3248` (12.699776… vs 12.699775… = 1 ULP). Linux / Vulkan are green. Suspected mechanism = the
  same class as the head of this section (a difference in FMA contraction from naga → MSL changes the rounding of the
  multiplication chain — unconfirmed). **Withdrawn (2026-09-01)**: the old text
  "chat e2e completes golden on M2 = behaviour and quality are sound" **had been written without the backing of an
  actual run** (what had been measured on M2 was only the canary / reduce parity / skinny / gemv checks — discovered
  through a remark by the user). At the time chat e2e could not run to completion on M2 because of gemma4 prefill NaN
  (independent of gemv; Metal fast-math overflow of `gelu_tanh` — resolved by `tanh_stable`; interactive runs confirmed
  on M2). The proposition that the margin check absorbs 1 ULP **was measured to hold on M2 (2026-09-02)**: after the NaN
  fix, `e2e_gemma4_chat_test.ts` (character-level agreement with the temperature-0 greedy golden — 6 checks) is all
  green on M2 (ADR [0082](decisions/0082-linear-gemv-decode.md) addendum 3 — re-establishing ground 1 that addendum 2
  had put on hold). The known real harm is, as with attention i8a8,
  that "the cross-path atol=0 check does not hold on Metal". **Isolated (2026-09-01)**: the cross-bucket u32 check of
  `gpu_gemm_skinny_test.ts` is green on M2 — **specific to GEMV** (the general-rule hypothesis rejected). **Ruling =
  keep the default path** (ADR [0082](decisions/0082-linear-gemv-decode.md) addendum 1 — suspected mechanism: GEMM
  stores the dequantization in a shared tile before reading it = a rounding barrier exists / GEMV inlines it in one
  expression = MSL contraction can span it). Root-cause candidate = searching for an expression form that puts an
  explicit rounding point in GEMV (needs a real-M2 loop; grouped with the root-cause candidate below; not started).
  **The i8 variant (K-16, 2026-09-06, `5ddd186`) has not been measured on M2** — the same single inline expression, so
  it is expected in the same band (1 ULP). On Linux / Vulkan the u32 check is green for all 6 i8 shapes. **The tiled
  paths ①ₜ / ③ₜ of states-shaped attention (K-13, 2026-09-06) have not been measured on M2 either** — they use the GEMM
  skeleton (through a shared tile = a rounding barrier exists), so the mechanism differs from i4 GEMV, and u32 agreement
  with ① / ③ is expected to hold as long as the same text runs on the same backend.
  **The row-block variant (K-21, 2026-09-07, `5701262`, 1 ≤ M ≤ 64) has not been measured on M2 either** — the same
  single inline expression (the dequantization is put in a `let` and shared across rows), so it is expected in the same
  band (1 ULP). At the same time, the comparison partner of the u32 check moved from M=2 (geometry M16N16) to M=65
  (geometry M64N32, outside the check), so the signature above (`0x414b3249` vs `0x414b3248`) may change when re-taken
  on M2. Linux / Vulkan are green for the 12 shapes of M=1 + the 13 row-block shapes.
  **Behaviour has been confirmed on M2 (user run on 2026-09-07 — the latest main including the K-21 row-block variant +
  the budgeted backing of H-15)**. Red / green of the exact u32 check has not been reported (still expected within the
  1-ULP band above).
- **The sha256 comparisons (anima PNG 9 / sbv2 WAV 6 / irodori WAV 2 (no-ref / voice-clone)) are explicitly SKIPped on
  Metal + 3 reference guards are red**:
  the reference-value fixtures have no row for Metal's environment key, so the 17 cases are SKIPped without comparison,
  and the per-family reference guards (anima / sbv2 / irodori) go red with "there is not a single row for this
  environment" (as specified by ADR [0106](decisions/0106-device-keyed-references.md) and the limitations section
  "The sha256 reference check is for the reference environment only"). Not counted among this section's targets. The rows
  can be created by running the same lane with `KARUME_REFERENCE=write`, but that operation takes the run's measurement as
  correct, so soundness is confirmed separately before creating them. Measured sha values agree across different paths
  (the base CFG with fromPretrained vs the split fromAssets form, 512 euler vs fromPretrained-512, 1024 vs
  onEvent-1024) = deterministic on Metal, and the load path does not change the output bytes. The user has confirmed by
  eye / ear that the outputs are sound.

Deno 2.9.5 / 2.9.6 contain no Metal / naga / wgpu updates (denoland/deno#36257 = mapped range
only). Root-cause candidate = an explicit `fma()` was found effective as a means of fixing the rounding on the WGSL side
(ADR [0105](decisions/0105-packed-static-quantize-activations.md) addendum 4 — the bitcast round trip + XOR 0 rounding
barrier was measured ineffective on M2 and withdrawn). It is applied to the parallel GEMV family and the subgroup32
variant (2026-09-25; the M2 re-acceptance is in
[the 0101 addendum](decisions/0101-linear-gemv-subgroup.md)); the sequential GEMV, row block, conv, gru and attention
epilogue do not have it yet. Making the TS reference FMA-tolerant has not been started. Records =
[research/2026-08-06-metal-silent-miscompute.md](research/2026-08-06-metal-silent-miscompute.md)
(snapshot) and
[research/2026-08-29-chatgpt-review-verification.md](research/2026-08-29-chatgpt-review-verification.md)
(M2 re-verification).

## On Metal the out-of-memory errorScope stays silent — the fail-loudly check does not fire (an independent bug)

On real **M2 (24GB / maxBufferSize 14.3GB)** hardware, the test in `gpu_generation_context_test.ts`
"state allocation failure fails loudly via the out-of-memory errorScope" is
red with **Expected function to reject** (2026-09-01) = **the allocation of a 64GiB state slot silently
succeeds** (Metal's lazy allocation — wgpu's Metal backend has no total budget, and `newBufferWithLength:` does not
return nil even when physical memory is exceeded). The buffer layer itself is sound as measured on M2
(probe measurements during the Metal NaN investigation — single buffers of 64–640MiB and a cumulative 835 buffers of a
real profile all matched. The probes have served their purpose and were deleted on 2026-08-31 — restore them from git
history).

**Partly resolved (2026-09-01, ADR 0089 = memory management phase A)**: for both weights and state, **exceeding the
absolute limit for a single buffer now fails deterministically through an explicit check before allocation**
(`assertWeightsWithinLimits` / the two limits on the state side — the fix candidate "explicit size check" is done).
What remains is **the physical overrun by the sum of several buffers**, which cannot be checked in advance because
WebGPU does not expose free capacity (ADR 0070 decision 5), and stays errorScope-silent on Metal — moved as a by-design
constraint to the limitations section "GPU memory pre-checks go up to \"the device's absolute limits\"". Making
`requiredLimits` effective at load time (rejection before download) was wired in wave 2
(`335ad7a` — ADR 0089 addendum 2026-09-01).

**Re-measurement on M2 after phase A (2026-09-02, Deno 2.9.4)**: this check remains red as predicted (1GiB × 64
buffers are each within the limit, and the total overrun is outside the check). The pre-download check passed without
false rejection for the distribution mirrors of gemma4 (declared 384MiB) / anima (declared 311,164,928B) at the M2
adapter values (maxBufferSize 14,302,248,960 / maxStorageBufferBindingSize 4,294,967,292), and generation ran to
completion.

## On Metal, `--diagnostics` (`gpuTiming: true`) takes the device down — the isolation experiment has not been started (needs real M2 hardware)

On real **Apple M2 / macOS 26 / Deno 2.9.x** hardware, running `examples/gemma4` (removed in 0.15) with
`--diagnostics` loses the device on the first turn and fails. The mechanism and the established facts are written in
the [limitations](limitations.md) section "On Metal (Apple GPU), GPU-side timestamp measurement is not practical"
(`createQuerySet` fails → `DeviceError::Unexpected` → device lost; it does not enter the errorScope).
**What is unconfirmed is the resource axis**: it has not been isolated which dominates, ① the number of query sets alive
at the same time in the first run (one chunk is fixed at 16 dispatches, so about 100 for gemma4 prefill ≈1,500
dispatches) or ② the amount that stays until GC because Deno's `GPUQuerySet.destroy()` is a no-op. The two are not
exclusive. The hardware was **confirmed to be macOS 26 (Metal 4) (2026-09-03)**, so it also falls within the range of a
separate unfixed problem where timestamps are all zero even when allocation succeeds
([wgpu#9414](https://github.com/gfx-rs/wgpu/issues/9414)) — even with the fix candidate below, op-level breakdowns may
not become readable on this machine.

**The full verify also has 1 red in the same form (2026-09-03, measured on M2 / macOS 26)**: even in
`deno task verify`, which does not pass `--diagnostics`, the census check at
`packages/models/tests/e2e_gemma4_pretrained_test.ts:208` ("gemma4 distribution: the pipeline default runs with ③'
parallel reduction…") runs the real-weight gemma4 prefill with `acquireGpu({ gpuTiming:
true })`, so both the parallel and the sequential steps are red
(the **13th** of full verify 1856 passed / 13 failed / 139 ignored — a different class from the 12 counted by the
1 ULP section above). On Metal too, it **does not fall under the `ignore` condition**: the decision is only the
enumeration of `adapter.features` (`TIMESTAMP_QUERY_AVAILABLE` in `packages/models/tests/helpers/gpu.ts`), and Metal
declares `timestamp-query`. The understanding "Metal does not advertise it, so the timing tests are skipped" is wrong.
The observed error, verbatim:

```
GpuDeviceLostError: flush 中に device が失われた（再構築が必要） — reason: unknown / device was lost
```

(The stack is `SubmitScheduler.flush` (`submit.ts:532`) → `RunArena.destroy` (`arena.ts:189`) →
`GpuContext.onLost` (`context.ts:351`).) **What was carried in `reason` was only Metal's generic wording "device was
lost"; the true-cause string of `createQuerySet` did not arrive** — the limitations statement "the true-cause string
put in by the backend reaches the caller for the first time" does not hold on this path. Therefore "about 100 query sets
alive at the same time is the cause" **remains a hypothesis** (no verbatim backing yet). As circumstantial evidence,
`packages/runtime/tests/gpu_timing_test.ts` with a single query set is green on the same M2 = the failure depends on the
count and is not unconditional.
The "all-zero timestamps" of wgpu#9414 has not been observed in this configuration.
**2026-09-26**: the census was made independent of timing (its source moved to `lastRunPipelines`, and
`gpuTiming: true` and the `TIMESTAMP_QUERY_AVAILABLE` decision were removed from the census test), so this red that
depends on the query set count (the 13th) is expected to disappear — awaiting confirmation on M2.

Isolation experiment (needs the hardware; run only one turn):

1. An A/B that raises `SubmitPolicy.initialChunkSize` from 16 to 256 to cut the count to about 1/16. If it no longer
   fails, the number alive at the same time dominates (①).
2. Force GC per run with `deno --v8-flags=--expose-gc`. If it no longer fails, the retention dominates (②).
3. ~~Confirm the hardware's macOS version~~ **Confirmed (2026-09-03) = macOS 26 (Metal 4)**. As above, it falls within
   the range of wgpu#9414, so the investment decision on the fix is made on that premise.

Fix candidate: **keep only one query set and reuse it** (capacity fixed at the per-set limit; the same idea as Dawn's
counter sample buffer pool). Since execution order within the same queue is guaranteed, there is expected to be no need
to drop to a form that waits on the host per chunk (`resolveBuffer` / `readBuffer` is needed per chunk, but there
`destroy()` works even on Deno). The opposite direction (smaller steps) is not taken, since the total number of samples
does not change and only the count grows.

**Current state**: neither the isolation experiment (① ②) nor the fix candidate above **has been started** (the
experiment needs real M2 hardware — backlog now section "Isolation experiment for Metal `--diagnostics`"). Since the
hardware is confirmed to be macOS 26 and timestamps may be all zero due to wgpu#9414 even if reusing the query set makes
allocation succeed, the experiment comes before the change (ruling of 2026-09-03). For the release decision, **it is
accepted that 1 census check of verify stays red on M2** (Linux / Vulkan green; the equivalence of parallel and
sequential is ensured by `packages/e2e/tests/e2e_gemma4_reduce_parity_test.ts`, which does not require timing).

## Intel Arc B570 (Linux / Vulkan ANV / xe): the OOM check is red + Deno's timestamp unit and the device-lost panic (measured 2026-09-20, recorded only)

Of the results of running the full verify after swapping the development machine's GPU from RTX 3080 Ti to
**Intel Arc B570** (BMG G21, VRAM 9.93 GiB, Mesa 25.0.7, Linux xe driver, Deno 2.9.6), these are the ones kept **as
records only** while unresolved (ruling 2026-09-20). The mismatches of 16 sha256 reference checks are the by-design case
of the limitations section "The sha256 reference check is for the reference environment only", and **were resolved by
changing the reference values to rows per environment key** (ADR
[0106](decisions/0106-device-keyed-references.md) — the B570 row `deno-intel-graphics-bmg-g21` has been created, so
the checks on this machine again point only at "a regression on this machine"). The BiRefNet 2048² device lost is
explicitly SKIPped by the held row of this machine's environment key (`HELD_SERIES` in `e2e_birefnet_test.ts` — the
full run does not stop at this family; limitations section "BiRefNet"),
`sin` of golden `activations` was resolved by moving the tolerance toward the WGSL specification band
(`OUTPUT_TOLERANCE` in `e2e_golden_test.ts`), and the `createResident` limit check was a hole in a test premise
(`maxBufferSize` being a multiple of 4) and is fixed.

- **The OOM check (`gpu_generation_context_test.ts` "state allocation failure fails loudly via the out-of-memory
  errorScope") is red**: the 64 GiB state allocation fails with `GpuOutOfMemoryError` as expected, but the
  survivor (1 GiB) right after it also fails with `not enough memory left`. **Not a leak on the karume side** —
  reproduced with a probe of plain WebGPU only (OOM at 1 GiB × 9 → a 1 GiB allocation right after `destroy()` of all of
  them is OOM → a 1 GiB allocation after `onSubmittedWorkDone` + 200 ms succeeds). On Intel / wgpu, the release of
  `destroy()` is delayed until the next device poll. The test is not changed (taking the survivor after a poll would
  weaken the meaning of the check, "detect an implementation that failed to return the release"). A different kind from
  Metal's "errorScope silence" (here it is the side that fails).
  **It is red only inside a full run and green when run alone** (re-confirmed in the full verify of 2026-09-27; red 10
  times out of 12). Whether it turns red depends on the VRAM headroom at that moment.
  **2026-10-01**: red 4 / 4 times both in `deno task test:core` and in a run of this test file alone (no `--filter`,
  26 tests) (both at HEAD `feb6d8fe` and at `7c2c22e`, where the full verify was green — not a code regression). Green
  when only 1 test is selected with `--filter`.
  The same delayed release also appears as **a flake when lanes are run back to back without a gap** (2026-09-21:
  running the sbv2 lane right after the anima lane (9 minutes), the 12 tests from the runtime golden `i8 / flow / p512`
  on are OOM — the later ones fail in `requestDevice` itself with `Not enough memory left`. 20 seconds is sometimes not
  enough; running alone after a gap of a few minutes, all 198 tests are green).
  **The same form appears without crossing lanes** (measured 2026-09-22): even the tiny golden (the 32 models of
  `e2e_golden_test.ts`), which repeats acquiring and destroying a device more than 30 times within one process,
  sometimes has its last cases go red with `Not enough memory left` from `requestDevice` or `GpuOutOfMemoryError` on
  weight upload, and running the same file again in succession increases the number of reds (2 on the 1st run, 7 on the
  2nd right after). Standalone runs (`--filter`) are green, and it recovers after a few minutes. **Up to here is
  observation**.
  **Inference (isolation not finished)**: the number of reds is the amount of release delay accumulated by that run,
  not the size of a regression. However, at the time this observation was taken, the tests had paths that did not guard
  the span from `acquireGpu` to `gpu.destroy()` with try/finally (if Session construction throws, destruction is never
  reached, and the rest of that run proceeds holding a device that is never destroyed), which was mixed in as **a
  separate path that increases the count within the same run**. The destruction leak itself is closed (`e2e_golden` /
  `e2e_sbv2` / `e2e_gemma4` / `e2e_minicpm5`), so the next time the same form appears, the pattern of counts is measured
  again to narrow the cause to one.
  The operational workaround is canonical in the section above on full runs that rarely flake.
- **Deno does not convert timestamp-query values to ns** (ext/webgpu returns wgpu's raw ticks as is; the WebGPU
  specification says ns). The B570's Vulkan `timestampPeriod` is 52.0833 ns, so the breakdowns of
  `lastRunTiming` / `--diagnostics` on Deno are **×52 too small** (BiRefNet 1024² GPU total raw 64.8 ms
  × 52.08 ≒ 3,375 ms, matching the wall-clock measurement window of 3,452 ms). On RTX the period is 1 ns, so it did not
  surface.
  **2026-10-06, measured on RTX 3080 Ti (NVIDIA driver 615.71)**: 1 tick = 1 ns (in passes of 133–153 ms, raw ticks
  matched wall clock − the fixed cost of submit (about 11 ms) within 1%; with both disposable and reused query sets) → a
  row was added to `TIMESTAMP_UNIT_NS`. Probe =
  `tools/diag/timestamp-period-probe.ts` (removed in 0.15).
  Chrome (Dawn) converts. karume cannot correct it (the WebGPU API does not expose the period), so
  when reading the GPU breakdown on Deno, beware of GPUs with period ≠ 1. No issue has been filed with Deno yet.
- **Deno 2.9.6 panics on device lost instead of raising an exception** (`device_poll(...).unwrap()` of ext/webgpu
  — buffer.rs:247 (mapAsync) / queue.rs:109 (onSubmittedWorkDone) crash on `Err(Device(Lost))`).
  The whole process disappears before reaching karume's `GpuDeviceLostError` path, so when a device lost occurs during a
  full verify run, the run stops there (an environment where the symptom of the section above on full runs that rarely
  flake is "the process disappears" rather than "one red test"). The completion marker of the in-flight submit bound
  (ADR 0123 — an `onSubmittedWorkDone` every 64 submits between fences — placed only by run) also goes through this path,
  so in a long run the loss surfaces in the synchronous part of the marker and panics.
- **Destroying a device and creating a new one reduces the total that can be allocated on the next device** (observed
  only, 2026-09-26; cause not investigated). In a probe of plain WebGPU, repeating "fill the device up, destroy it,
  wait 500 ms" reduced the total allocatable on the next device as 9 → 8 → 7 → 6 GiB
  ([research 2026-09-26](research/2026-09-26-anima-residency-bench.md)). It may be the same lineage as the flake
  "`requestDevice` itself fails with `Not enough memory left`" above (speculation).

## EmbeddingGemma batch>1 export does not pass the conversion stage

`python -m embeddinggemma.export --batch N` (N>1, on the tools/export-recipes side — the launch form follows the
recipe separation of ADR 0065) fails loudly in core's `karume/convert.py`
(B=1 succeeds as before). The mechanism has 2 stages:

1. transformers (the 5.14 line) `masking_utils.find_packed_sequence_indices` enters the packed-sequence branch during
   tracing because of the mismatch between `position_ids [1,T]` inside `Gemma3TextModel` and `batch_size=N`, and
   `aten.eq.Tensor` / `aten.index.Tensor` / `aten.ne.Scalar` survive into the IR (at B=1 this mismatch does not occur,
   and it is absorbed by the existing Tmax constant + `sym_prefix_slice` folding).
2. Removing this branch with a monkeypatch then hits, for the band mask's `aten.bitwise_or.Tensor`, the constraint that
   **a bool constant cannot be an IR v1 initializer** (f32 / i32 only). Moreover, the same
   patch **triggers the same error even at B=1** — the existence of the packed-sequence branch itself is a premise of the
   current band-mask constant-folding pattern, and even a patch meant to be eager-equivalent is not
   safe (measured A/B, 2026-08-11).

The root fix is a generalisation on the convert/normalize side (turning bool constants into f32/i32, or extending
initializer dtypes + supporting mask folding for batch>1), which needs a design decision on the core conversion
foundation. The `--batch` flag itself is kept in a form that can be used as is once the generalisation lands.

## On Pixel (8GB-class Android Chrome), loading anima turbo i4 fails — true cause not identified

The error text of the hardware report "BodyStreamBuffer was aborted" is a fixed text that Chrome substitutes for the
reason of a collateral abort, not the true cause (the diagnostic bug where hub surfaced the collateral side was fixed on
2026-08-25 — restoring the true cause + byte budget + serialising verification. **Released in 0.7.0**).
The leading hypothesis is memory pressure (even turbo i4 is ~2.56GiB resident at completion + verification temporaries;
with i8 a forced browser termination was reported), but it looks identical to a connection drop or an app-side abort, so
it cannot be confirmed until `err.cause` is observed on the hardware with the fixed version. Reducing residency itself
came with the shard-split distribution of 0.7.0 (the R1 integration phase), and is currently replaced by block-level
fetching of containers and Session construction (the browser source is seek-type —
ADR [0108](decisions/0108-container-format.md) addendum 5). anima's HF distribution has been re-uploaded as
`karume/5` (the 0.13.0 re-upload, 2026-09-24). Remaining task = re-observation on the hardware only (the Pixel item of
"User hardware" in the [backlog](backlog.md) now section).

## On Pixel (Mali-G715, Android Chrome 154), Anima's per-stage GPU time exceeds the wall clock — suspected broken timestamp reads (observed 2026-10-01, recorded only)

In the user's Pixel 10a record (adapter `arm` / `valhall` / `Mali-G715`, gpu-lab's Anima tab, 512²,
`residency: transformer`)
`outputs/bench-browser/anima-residency-browser-f16-2026-10-01T10-43-00.425Z.json`, the per-stage `gpu.totalNs` far
exceeds the wall clock:
text_encoder 186.7 s vs 49.8 s (3.7×), transformer 94,533 s vs 234 s (400×), vae_decoder 1,295 s vs 20 s (64×).
In the breakdown, a single strided / elementwise dispatch shows 4–5 seconds (impossible). On the same device the
geometry sweep (repeating 1 pass = 1 dispatch;
`…geometry-sweep-browser-2026-10-01T09-37-40.824Z.json`) has GPU time matching the wall clock (e.g. 89.8 ms vs 92.6 ms),
so the sweep path is correct, and it breaks only on Anima's diagnostic path, which gathers many timestamp queries into
one run. `clampedNegativeSamples` is 0.
**This device's per-stage GPU time (`lastRunTiming` / gpu-lab's GPU column) cannot be trusted; look at the wall clock
only.**
Inference (isolation not finished): Mali timestamp overflow, or a resolve error on the Dawn / driver side when the
number of queries is large. Isolation needs a minimal reproduction on the same device that varies the query count (does
not occur on M2 / RTX / B570).

## hub: narrowing weights to `drafter` only in `evictCachedAssets` still deletes the shared assets (tokenizer)

`evictCachedAssets(loaded, { model, quant, weights: ["drafter"] })` can be written with the intent "keep the main model,
delete only the drafter", but **only the main weights remain, and assets such as the tokenizer are deleted** —
that (model, quant) is no longer "stock that can be used as is" (the assets are downloaded again on the next start).
Derived by reading the code on 2026-09-09 (reproduction on hardware not observed).

The mechanism has 2 stages:

1. `resolveSelection` (`packages/hub/src/resolve.ts`) **expands all assets regardless of the narrowing of
   weights** (comment "assets are always complete"). The reference set of the narrowed selection = the drafter's weights
   files + all assets, and the assets enter the deletion candidates.
2. The reference accounting of `packages/hub/src/inventory.ts` "**does not count the same (model, quant) as the target
   on the protecting side**" (another subset of the same label does not become a candidate — even if passed to
   `protect`, the target itself is ignored). Hence they are not protected by "the unnarrowed main selection uses the
   assets", and assets survive only in a mirror where another model / quant is complete in stock.

The result is silent, and `EvictedAssets.alsoEvicted` names **only other (model, quant)** pairs (that the same
selection fell to partial stock is not listed). The only way to confirm is to query `listCachedAssets` again without
narrowing.

Operational workaround = **to delete only the drafter, do not use evict; delete the parts of the drafter's container by
hand** (`protect` cannot work around it). Fix proposal = keep the shared assets by default when evicting partial weights
(either remove the assets from the reference set of the "narrowed selection", or count the rest of the same label as the
target on the protecting side). Both change the definition of the granularity of "deletion per selection", so a design
ruling is needed — filed only here.

## Intel Arc B570: the "GPU time per submit ≤ 1 s" check at Wan DiT full size exceeded once after a long continuous load (2026-10-03, recorded only)

Running `deno task test:models:wan` for the acceptance of ADR [0118](decisions/0118-wan21-video-generation.md) step 8,
the timing mode at S = 32,760 produced **a maximum GPU time per submit of 1,268.6 ms** (check 1,000 ms) and went red
(`e2e_wan_dit_test.ts`). The one exceeding was 1 of 28,410 submits, containing a single linear dispatch
(`linear:v2:f32:reg128x128r8x8w16v4:wf16`). The same dispatch took 192 ms in the other 13 forwards of the same run, the
maximum of the 255 unbacked submits was 458.6 ms, the same as the previous run, and only that forward went from 68 s to
80 s of GPU time. Just before (after about 10 hours of continuous GPU load),
sysfs showed `freq0/throttle/reasons = thermal`. Re-running the same test alone gave 459.3 ms (7 passed), and the
records every 5 seconds during the re-run (332 records) showed act_freq 2,683–2,750 MHz, max temperature 68 ℃, and only
`pl` (power limit) as the throttle reason.

- Attribution (inference): not a problem of the plan (how chunks are cut) but a transient stall on the GPU side (clock
  drop from heat or power, context switch). If it were a plan problem the unbacked maximum would move, and it did not.
- Operational workaround = for a run where the timing check failed, look at the throttling reason and temperature with
  `tools/diag/gpu-busy-xe.zsh` (`throttle=` field added on 2026-10-03), and re-run alone after cooling (green means a
  flake). When running lanes in succession, avoid doing so right after heavy GPU jobs (a 50-step full run, mass
  production for visual checks).
- Whether to change the check's statistic (the unbacked maximum + the window mean after backing, etc.) is a design
  ruling — filed only here.

## Component replacement: replacing a component with a larger one makes `fromPretrained` succeed and the first `generate()` fail (ADR 0122 decision 7, item 5; filed only)

When the replacement needs a larger buffer than the original, **`fromPretrained` finishes fetching several GiB of weights
and succeeds, and the first `generate()` (for Wan, the construction of the text stage's Session) fails** (speculation;
no actual case yet). When a component is replaced through `components` of `fromPretrained`, the replacement preset's
`session` / `gpuFeatures` / `requiredLimits` are not used, and **the pre-check uses the values of the original
manifest's preset**, so it passes the check before fetching weights (the `requiredLimits` comparison). Construction
creates no Session at all (`packages/models/src/wan/family.ts:493`), and the text stage creates a Session per generate
(inside `encodeWanPrompts` — `packages/models/src/wan/text-stage.ts:369`), so the failure comes after construction.
This is a state that lacks the check of "consistency between the quant preset and the execution settings" that ADR
[0108](decisions/0108-container-format.md) decision 19 requires of admission, and it is not by design.

- Mechanism: the replaced component takes only the container from the replacement's manifest
  (`packages/models/src/hub/components.ts:471-478` — of the return of `resolveSelection` only `containers[key]` is
  used). Wan's family check reads the preset of the original manifest (`packages/models/src/wan/family.ts:407-414`),
  and makes the pre-fetch check with its `requiredLimits` (same file `:320-324`).
- Current impact: the recipe writer (`umt5_export` — including `--intake`) writes the vocabulary embedding as i8, so a
  replacement with a compatible umT5 made by the recipe does not trigger it (if the graph descriptor is the same, the
  buffer sizes are the same too). However, the shared check of `umt5Contract`
  (`umt5GraphContract` — `packages/models/src/umt5/graph-contract.ts:129-151`) only requires "storage is i8 or f32,
  with at least one i8", so a component that holds the vocabulary embedding in
  f32 (the embedding buffer about 4×) passes the check and can fail in this way.
- Fix candidates (ruling needed): check the replacement preset's `requiredLimits` / `gpuFeatures` together with the
  original preset's (take the larger), or reject in admission a replacement whose preset differs.

## Deno: `GPUDevice.destroy()` does not return VRAM, and the remainder accumulates over a sequence of tests that create and discard devices — the practical preset at S = 32,760 OOMs in the Wan lane (2026-10-04 to 05; cause established; a test-side symptomatic fix is in; the root fix is not done)

In `deno task test:models:wan`, the step of `e2e_wan_dit_test.ts` "practical preset `f16+dit8-a8-attn8-s16`, normal
mode, S = 32,760" fails with
**`GpuOutOfMemoryError: run encoding and readback: not enough memory left`**. In the lane it failed 2 times out of 2
(385 passed, 1 failed / 394 passed, 1 failed); running the same file alone with `--filter 実用席` is green 2 times out
of 2. The failing step only records time and allocation, and all sha row and band comparisons pass.

- **Cause (established from the upstream source and measurements)**: on the development machine before the hardware
  swap (B570), with Deno 2.9.6 (wgpu-core 29), `GPUDevice.destroy()` only marks the device invalid and releases
  nothing. The device and its resources (blocks the allocator keeps, the zero buffer, the driver pools, query sets)
  are returned when V8's GC collects the device's wrapper. The tests repeat `acquireGpu` → `settleReleases` →
  `destroy()` per test, so destroyed devices remain until GC and keep eating the VRAM limit of the same process (about
  9.6 GiB on the B570). `settleReleases` returns only explicitly destroyed buffers and has no effect on per-device
  remains. That they remain after `gc()` within a file is because the runner keeps the function passed to a step ("Why
  it remains" under "What remains" below).
- **Measurements** (`outputs/bench/2026-10-05_dead-device/`, B570):
  - A probe that creates and discards a device 5 times in plain WebGPU: per destroyed device that did nothing, one DRM
    client remains and keeps holding vram0 0.192 GiB and gtt 64 MiB (1.15 GiB / 384 MiB after 5 times). One `gc()` call
    does not reduce it, and **everything returns on the second** (the clients go back to the single fd of the physical
    device). Waiting (settle) before the destruction makes no difference.
  - A run of the 3 tests of `e2e_wan_dit_test.ts` (full-size normal mode → practical-preset timing mode →
    practical-preset normal mode), recorded per DRM client from outside: while S = 32,760 of the practical-preset normal
    mode was running, **2 destroyed devices kept holding 0.64 GiB and 1.54 GiB**
    (peak total 8.34 GiB = 0.64 + 1.54 + its own 6.16). The allocation at the start of this step was 2.37 GiB, more than
    in a standalone run (1.73 GiB). In the lane, more devices had been discarded before this, and it exceeds the limit.
  - A query set in timing mode is about 4.1 KiB each (41 MiB for 10,000). `GPUQuerySet.destroy()` does nothing on Deno,
    and it is returned by GC.
- **Symptomatic fix applied** (user ruling 2026-10-05 — a symptomatic fix is acceptable; deep investigation and an
  upstream report are postponed): the tests' GPU acquisition entry point
  `acquireTestGpu` (`packages/models/tests/helpers/gpu.ts`) awaits `collectDestroyedDevices`
  (`packages/models/tests/helpers/collect-destroyed-devices.ts`) before `acquireGpu`. It does `gc()` → 100 ms → `gc()`
  → 100 ms, with the count and the waits matched to the probe measurement above (everything returned within the 100 ms
  after the second). Since `gc` is needed, the Vitest configuration passes `--expose-gc` to its workers (`execArgv` in
  `vite.config.ts`; `packages/runtime/tests/verify_lanes_test.ts` fails if it does not). Only `acquireGpu` of the e2e
  tests of the Wan lanes (`models:wan` / `models:wan-ti2v`) goes through it.
  Whether collection happened is not checked.
- **What remains**:
  - Root fix: the upstream (Deno) report and a check before acquisition that the DRM clients of destroyed devices have
    gone (naming and failing if any remain) are postponed. Test-side candidate = the relay step (worked on the B570 in
    experiment E4 below; proposal on hold — ruling below). The GC timing is the behaviour of the V8 / cppgc
    implementation, not a guarantee of a specification.
  - A run started without `--expose-gc` (outside the Vitest configuration) has no `gc`, and the mitigation does
    nothing.
  - Tests outside the Wan lanes and devices that the pipeline takes internally (loads that do not pass `gpu`) do not go
    through the entry point (the GC before the next acquisition through the entry point collects what was discarded
    until then together).
  - Product side: the same remains appear in usage that recreates a device after `GpuContext.destroy()` (recovery from
    device lost, etc.). No remedy.
  - The re-run of the lane with the mitigation (2026-10-05, `outputs/bench/karume-wan2.1/2026-10-05_stage3-gpu-rerun/`)
    was 395 passed, 0 failed. However,
    **the headroom is only about 0.15 GiB**: the allocation at the start of the step that used to fail was 3.48 GiB, the
    peak 9.44 GiB (limit about 9.6 GiB). In the external record, the mitigation collected only the devices of the
    previous file and one destroyed device within the same file. The 3 devices from the timing-mode device on remained
    after `gc()` (1.62 + 0.64 + 1.03 = 3.29 GiB at the peak) and disappeared together at the end of the file.
  - Why it remains (established with a plain-WebGPU probe, 2026-10-05): the `deno test` runner keeps the function
    passed to `t.step` until the file ends. If that function holds the test scope's `gpu`, the destroyed device's
    wrapper is not collected however many times `gc()` is called, and the VRAM remains until the end of the file.
    Probe = `outputs/diag/step-retention-probe_test.ts`, log =
    `outputs/bench/karume-wan2.2/2026-10-05_stage6-close/step-retention-probe2.log` (A: no step → the DRM client
    disappears after 2 `gc()` calls / B: the step closure holds the device → remains even after 4 / E: relay step (only
    the relay function is passed to the runner, and the body is released at the end of the step) → collected). This is
    why the current symptomatic fix (2 `gc()` calls before acquisition) does not work for devices held by the function
    passed to a step.
  - **It recurred even with the mitigation** (2026-10-05, the final state of ADR 0121 stages 5 / 6 `128b511e`,
    `outputs/bench/karume-wan2.2/2026-10-05_stage6-close/`):
    `test:models:wan` was 403 passed, 1 failed, failing at the same step with the same exception. On the same frozen
    copy, a standalone re-run with `--filter 実用席` was 2 passed, 0 failed
    (S = 32,760 normal mode allocated 5.46 GiB). The thinness of the headroom was re-confirmed. A re-run of the lane on
    the same frozen copy was also 403 passed, 1 failed, 78 min 59 s
    (same step, same exception — 2 out of 2). Running only `e2e_wan_dit_test.ts` gave 12 passed, 1 failed (same step);
    on the B570, splitting by file is not enough.
  - **Experiments** (B570, frozen copy, `oom-e2/` / `oom-e4/` in the same location): in E2 (3 reduced tests + the
    current symptomatic fix), 2 destroyed devices (0.64 / 1.54 GiB) remained until the end of the file. In E4 (an
    experimental version of `e2e_wan_dit_test.ts` with its 8 `t.step` sites replaced by relay steps), the whole file was
    13 passed, 0 failed, 36 min 17 s. The VRAM at the start of the step that used to fail was 0.192 GiB (2.38–4.15 GiB
    before the remedy), the peak 5.883 GiB (limit 9.375 GiB), and the destroyed devices disappeared every time before
    the next acquisition.
- **User ruling (2026-10-05, the B570 lane)**: no remedy to make the lane pass on the B570 environment. The relay-step
  proposal (helper + replacing 72 sites in the 10 Wan e2e files + a check that fails direct writes) has not been approved
  (on hold — not rejected). The lane is to be run on the RTX 3080 Ti after the development-machine swap, and the sha rows
  of the new GPU are created with `KARUME_REFERENCE=write` on the frozen copy `128b511e` and then compared at HEAD (the
  B570 rows are kept — ruling 2026-10-06, ADR 0121 addendum "the development machine changed"). The swap was completed on
  2026-10-06 (RTX 3080 Ti 12,288 MiB,
  environment key `deno-nvidia-geforce-rtx-3080-ti`). **On 2026-10-06 it was run on the RTX 3080 Ti and passed without
  the remedy** (frozen copy `128b511e`: 2.1 = 401 passed, 2.2 = 40 passed (the only red was a missing row in the
  conversion table of the timing check); in the full verify at HEAD `b744a9fd`, both Wan lanes all green — ADR 0121
  addendum "RTX 3080 Ti lanes and full verify").
  How destroyed devices remain on the NVIDIA driver has not been measured.
- **Operational workaround** (Deno era; the practical-preset normal-mode case it re-runs was dropped on 2026-10-17 by `466b7196`): for a run where only this step still fails with OOM after the mitigation, re-run
  `e2e_wan_dit_test.ts` alone with the test-name filter `-t 実用席`. When running outside the Vitest configuration,
  pass `--expose-gc` (the configuration does it through `execArgv` in `vite.config.ts`; without it the mitigation does
  not work). On machines after the swap, first run the lane without the remedy and look at the result. The NVIDIA driver
  does not emit `drm-total-*` in DRM fdinfo, so the tests' VRAM observation (`helpers/drm-usage.ts`) becomes
  `n/a` (an observation, not a check — it does not turn red). VRAM is watched from outside with `nvidia-smi`.
- **Existing items the same mechanism could explain (speculation, unverified)**: "destroying a device and creating a
  new one reduces the total that can be allocated on the next device" and
  "the OOM at the end of the tiny golden that piles up acquisitions and destructions" in the B570 section above.
- The investigation record is `.claude/reviews/2026-10-05_wan-lane-oom-investigation.json` (not tracked by git), and the
  experiment procedure is `outputs/diag/dead-device-README.md`.

## RTX 3080 Ti (Deno): the duration of some e2e tests splits into two peaks (observed 2026-10-07, cause not investigated)

Found in the before/after speed comparison of ADR 0123. Values (sha256, bands) do not change; only the duration jumps
2–3× depending on the run.

- The decode of the tall 9 frames (`accept`) of the Wan2.1 VAE tile e2e (`packages/e2e/tests/e2e_wan_vae_tiles_test.ts`)
  splits into around 12 s and 20–37 s. The wide 33 frames (`band`) do not split.
- The prefill of 598 tokens (`context-en`) of the Gemma 4 sequence e2e
  (`packages/e2e/tests/e2e_gemma4_sequence_test.ts`) splits into around 8 s and the 13 s range (34 s vs 55 s for the
  whole test).
- Both also appear with the code before ADR 0123 (`b3fa0d34`) (VAE 1 in 4 runs, Gemma 1 in 9). Unrelated to ADR 0123's
  completion marker (the VAE path does not place markers with enqueue; in the whole Gemma test there is 1 marker and 0
  waits).
- Working hypothesis (speculation, unconfirmed): fluctuation in the estimate of the submit time-budget split (measured
  window ÷ workgroup count) changes how chunks are cut from run to run.
  Confounding with thermal limiting (this GPU has weak cooling and runs around 93 ℃) is also possible.
- Record: `.claude/reviews/2026-10-07_submit-inflight/ab-vae-tiles.log`, `ab-gemma.log` (not tracked by git).

## Adapter matching of reference values: the 4 families whose pipeline acquires the GPU internally do not match the GPU of the actual run (only on multi-GPU machines; filed only)

sha256 reference values are held as rows per environment key (adapter name) (ADR
[0106](decisions/0106-device-keyed-references.md)). The e2e tests of gemma4 / anima / sbv2 / irodori use
`assertRunningAdapter` (`packages/runtime/tests/helpers/environment.ts`) for matching, because the pipeline acquires the
device internally. It acquires a separate GPU once for the check, matches it, and caches the result within the process.
The acquisition of the actual run is not matched.

- There is no guarantee that two `requestAdapter` calls return the same physical adapter (the [limitations.md](limitations.md)
  section "The GPU fit check before download"). On a multi-GPU machine, if only the actual run picks a different adapter,
  that adapter's output is compared with the row of the matched key (a false mismatch), or writing with
  `KARUME_REFERENCE` can record another adapter's output into that row. It does not happen on single-GPU machines.
- Of the e2e tests with sha256 rows, those that acquire the GPU themselves (the 4 Wan pipeline / swap files, the
  runtime-side real-asset e2e, the real-asset e2e of depth_anything / birefnet / vowel_detector in `packages/e2e`) match
  the actual GPU itself right after acquisition (`verifyRunningGpu` / `assertAdapterMatchesEnvironment`). The Wan e2e
  tests without sha256 rows (the dit / umt5 / ab / vae files) match it too.
- Fix candidates: inject the GPU from the test (it must be acquired with the same features the pipeline requests
  itself), or make the adapter actually used by the pipeline readable (a change to the public surface).
- **User ruling (2026-10-11): filed, not fixed** (the remainder of Codex deep review R8-003). Reasons: it happens only on
  multi-GPU machines. Injection risks moving the sha rows if it diverges from the pipeline's internal acquisition. An
  adapter-reading entry point would add public surface for this problem alone.

## `gpu_full_write_test.ts`: the linear i8a8 case never puts its poison into `xq` / `xs` (filed 2026-10-27; tests only)

The full-write case for linear i8a8's temporaries (`packages/runtime/tests/gpu_full_write_test.ts:594-663`) sizes its
poison buffer to the temporary under test (`poisonCount` = m·k/4 for `xq`, m for `xs` — `:602-605`); its comment
(`:563-564`) reasons in size classes, which the liveness packer (ADR [0093](decisions/0093-transient-liveness-packing.md),
least region growth since its addendum of 2026-10-26) does not place by. The packer takes slots largest first: the node
output `y` (80 B) opens region 0; the poison (32 B or 16 B) shares `y`'s bytes there (their lifetimes do not overlap, so
the growth is 0); `xq` (32 B) and `xs` (16 B) are kept out of `y`'s region (the GEMM reads them and writes `y`) and open
fresh regions. Replaying the case's allocation program through `planTransients` gives, in both cases, the poison and `y`
at region 0 offset 0, `xq` and `xs` alone in regions 1 and 2, and one shared slot.

- So the GEMM's full write of `y` covers the poison, and the temporaries the case names start zeroed; `reuseCount >= 1`
  (`:650-651`) passes on the poison sharing `y`. A word the quantiser left unwritten would still read as 0 (an int8 0 or
  a zero scale), which the atol = 0 match catches unless the reference has a 0 there too, but the poisoned-temporary check
  the case describes does not happen.
- The conv3d case below it had the same gap and was rebuilt in `21ec9fa8` (a poison larger than every slot, `y` smaller
  than `xq`, `reuseCount` asserted to be exactly 2); the same construction fits here.
- Impact: tests only (the runtime is unchanged).

## The memory estimate does not count conv3d w8a8's in-node temporaries, nor list them as unaccounted (filed 2026-10-27; estimate callers on conv8 Sessions)

With `convCompute: "a8"` an eligible conv3d node allocates two temporaries between its two dispatches
(`buildConv3dI8a8`, `packages/runtime/src/runtime/recipe-builders/conv.ts:569-570`): `xq` = P·Cin bytes (the per-pixel
int8 words) and `xs` = 4·P bytes (the per-pixel scales), P = T·H·W of the conv input, alive from the quantiser to the
GEMM. `EstimateOptions` has no `convCompute` (`packages/runtime/src/runtime/estimate.ts:302`), the estimate counts
neither, and its `UNACCOUNTED` list (`estimate.ts:456-462`) names linear i8a8's quantized intermediates but not
conv3d's — against that list's own MUST (list every item known to be not included). ADR
[0133](decisions/0133-conv3d-w8a8.md)'s addendum of 2026-10-26 (channel-major quantiser) notes the gap as pre-existing.

- Sizes on the Wan2.2 TI2V-5B decoder (every conv3d runs the w8a8 key — `e2e_wan_ti2v_vae_a8_test.ts`; conv input
  shapes from the local containers' graph descriptors, `outputs/diag/wave-b-2026-10-26/v6/census.json`, not tracked): at
  tile 16 the largest pair is `xq` 24 MiB + `xs` 192 KiB on the first chunk (Cin 512, P 49,152 = 3×128×128) and
  48 MiB + 384 KiB on the next chunk (Cin 512, P 98,304 = 6×128×128); at tile 32 it is 96 MiB + 768 KiB (first, Cin 512,
  P 196,608) and 192 MiB + 1.5 MiB (next, Cin 512, P 393,216). Arithmetic from the shapes, not measured.
- The pair lives inside one node, so the live peak misses at most the largest pair; how the region total moves depends
  on the placement. No family calls the estimate on the Wan decoder today (`packages/models/src`); a caller of
  `PreparedModel.estimate` / `estimateGraphMemory` on a conv8 preset gets the under-count without a note.
- Impact: callers of `PreparedModel.estimate` / `estimateGraphMemory` on Sessions with `convCompute: "a8"`; no shipped family pipeline today (execution is unaffected).
- Fix candidates: `EstimateOptions.convCompute`, counting the pair under the same eligibility as execution
  (`conv3dI8a8Eligible`, `packages/runtime/src/kernels/conv3d-i8a8.ts:114`), or at least an `UNACCOUNTED` item.

## `/gemma4-bench` split mode fails intermittently (filed 2026-11-03; playground bench split mode only)

On the maintainer's M2 (Chrome 158), `/gemma4-bench` arms with "Split dispatches" on failed — once with "Integer GEMV"
on, once on a plain QAT arm — with `setBindGroup: The provided value cannot be converted to a sequence`, raised in the bench's GPU tap
(`apps/playground/src/models/gemma4-bench/tap.ts`). It was not reproduced in isolation and has not been investigated;
the same plain arm and a byte-plane arm ran with the split in later runs
([research 2026-11-03](research/2026-11-03-qat-integer-gemv-m2.md) §8, §12).

- Impact: the playground bench's split mode only (a run can fail before its per-key attribution is complete).

## The bidirectional block of a generation run: four edge cases outside the shipped path (filed 2026-11-08; recorded only)

A review of `GenerationRun.bidirectionalRows` (ADR [0144](decisions/0144-gemma4-12b-vision.md) decision 3) found four
cases where the block's behavior differs from its contract. Gemma 4 12B's image chat cannot reach any of them: it runs
the block on its own immediate chunk of a context the Session made, on sliding slots, far below 2³² rows.

- **A context from outside the type.** `assertBidirectionalRows` (`packages/runtime/src/runtime/executor.ts:428`) reads
  `run.context[RUNTIME_INTERNAL]` before the context check that runs later in issue. A run with a block on an object that
  is not a `GenerationContext` rejects with a `TypeError`, not the `ExecutionError` the same run without a block gets.
  No state is touched.
- **The CPU reference on a full slot.** `referenceStateAttention`
  (`packages/runtime/src/reference/state-attention.ts:194`) applies the block whatever the window, so with window 0 (a
  full slot) it ORs the block in. The runtime keeps full layers causal and the kernel builder refuses a block without a
  window. The reference describes a behavior the GPU path refuses; no test passes a block to a full slot.
- **The u32 bound of the block words.** `#writeLengths` (`packages/runtime/src/runtime/generation-context.ts:1531-1533`)
  writes `pastLength + start` and `pastLength + end` into the lengths uniform without a bound check. When
  `pastLength + end` is 2³², the upper word wraps to 0 and the block becomes empty, so the run is silently causal.
- **A second slot backing for block runs.** The prepared key gains `|bidir`
  (`packages/runtime/src/runtime/executor.ts:1699`), and the slot backings are keyed by it, so an image chunk and a text
  chunk of the same 768-row signature hold two backings. Under a backing budget sized for one, alternating chunks could
  evict and rebuild one per chunk. Performance and memory only; not measured.

- Impact: callers of the runtime's `GenerationRun.bidirectionalRows` outside the shipped 12B pipeline, and the CPU
  reference's tests; the shipped image chat is unaffected.
- Fix candidates: the context check before the block rule; the reference refusing a block on window 0 as the kernel
  builder does; a u32 bound in the block rule; measure the backing split before changing the key.
