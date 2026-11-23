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
are run back to back without a gap** (on some machines the release by `destroy()` is delayed until the next device
poll — measured on an Intel Arc B570 under Linux xe / wgpu).

Operational workaround = **leave a few minutes between lanes**, or **re-run the failed file alone to confirm**
(green means a flake). However, in an environment that panics instead of raising an exception on device lost
(Deno 2.9.6 — its ext/webgpu unwraps the result of the device poll), the whole process disappears, so the run itself stops there before any standalone re-run.

## On Metal (Apple GPU), the GPU output of attention i8a8 is 1 ULP off the TS reference (+ 4 conv1d/conv2d parity tests + 2 gru_scan parity tests + 1 linear GEMV u32 check)

On real **Apple M2** hardware (first seen on Deno 2.9.4; re-verified on 2.9.6 on 2026-08-29; the section's targets
inventoried in the full verify of 2026-08-31; **re-measured on 2026-09-02 at HEAD `2b096c4` after memory management
phase A on Deno 2.9.4 = 12 tests in total reproduce with the same signature, 0 new**; **re-measured on 2026-09-03 at
HEAD `b7e32c1` after the review-fix phase on macOS 26 = the same 12 reproduce with the same signature. The full verify is
1856 passed / 13 failed / 139 ignored, and the 13th is not a target of this section but of the `--diagnostics` section
below**), 4 attention i8a8 tests
(all in `gpu_attention_pv_i8a8_test.ts` today; three of them were in `gpu_attention_i8a8_test.ts` until it was merged there on 2026-10-17) +
**2 conv1d parity + 2 conv2d parity tests**
(the old text listed conv2d only — the 2 conv1d tests were missing from the record; same symptom) +
**2 decomposition parity tests of gru_scan / gru_scan_reverse** (below) + 1 GEMV u32 check + 1 OOM check
(next section) are red (Linux / Vulkan are all green).

Every measurement in this section was taken under Deno (wgpu, naga → MSL). Since 0.15 the test lanes run under
Node + Dawn (Tint), and no lane has been run on an Apple machine under Node yet, so whether these tests are red there
is not known. What Chrome on Metal (Dawn / Tint) does with the kernel rewrites is the next section.

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
  [0147](decisions/0147-numerics-and-kernel-adoption.md) decision 6) — "both arms bit-identical, inside the band (rtol 1e-5)
  against the reference" picks dp4a and continues with one warning (a8 runs). Only outside the band is `GpuFeatureError` (ADR 0147 decision 8
  turns that into a warning and f32 for that attention; the code change is filed in the backlog).
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
  The observed harm, as elsewhere, is only that "the cross-path atol=0 check does not hold on Metal" (1–64 ULP, no
  NaN). No tolerance check on real weights covers the op on M2: no distributed model uses `gru_scan` (the one that did
  was removed before 0.24.0). Root-cause candidate = strengthening the
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
- **The sha256 comparisons SKIP on Metal and the reference guards are red there** — by design: the fixtures have no
  row for a Metal environment key (the limitations section "The sha256 reference check is for the reference
  environment only"). Not counted among this section's targets. Measured on the M2 under Deno (2026-09): the sha values
  agree across load paths (deterministic on Metal), and the user confirmed by eye / ear that the outputs are sound.

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

## In Chrome on Metal, kernel rewrites that are word-identical on Vulkan and D3D12 differ by a few ULP (measured 2026-11-11, recorded and accepted)

The kernel checks (run as the Playground's `/kernel-check` page; now the workbench's `kernel-check` entry) on **Apple M2 and Apple M5 (the M5's adapter reports no name), Chrome 157, Metal** FAILs the
checks below; on Chrome / D3D12 (RTX 5070 Ti) and on Node + Dawn / Vulkan (RTX 3080 Ti) the same cases are
word-identical (D3D12 keeps its one known FAIL, the K-95 classes of the states-attention check). Each check compares
two WGSL texts of the same arithmetic as u32 words; every one that FAILs has a sum over many terms in it, and the
differences are 1–2 ULP of finite values. The order-only mutants of the states-attention check are all detected
on Metal too, so the harness is not blind there.

| Check (part) | Compared | Failing cases on Metal |
| --- | --- | --- |
| `state-attention-rewrites` (`state-qk-fold`) | folded ①′ vs ①′ | 252 of 252, the F = 8 classes the product folds included |
| `state-attention-rewrites` (`state-pv-unroll`, `state-stats-pv-unroll`) | the unrolled forms (perf-ledger K-91) vs their v1 WGSL | 532 of 546 and 476 of 490 |
| `state-kv16` (`state-kv16-readers`) | f16 full-slot readers vs their f32 forms over a rounded slot | 31 of 246, the read-only reader's O words only |

Identical on Metal: every fusion twin part, the prefetch form of `rms_norm` (below), the subgroup GEMV and RMS
reductions, the integer and byte-plane GEMVs, the row-blocked parallel GEMV, the f16 score store, the bidirectional
block and the one-row form of the expert linear.

The prefetch form of `rms_norm` and `fusion-rms-norm-rope` failed here until 2026-11-13: with x and the weight in
registers before `inv` existed, Metal computed `x · inv · weight` as `(x · weight) · inv` (recomputing both orders in
f32 reproduces the M2 record exactly). Since the key's v3 rounds `x · inv` through the XOR barrier (ADR
[0017](decisions/0017-rms-norm-conv2d-clamp-min.md) addendum 3) both pass on the M2 and the M5
(`outputs/bench-browser/kernel-check_2026-11-13T21-22-18.json` / `…T20-59-22.json`, untracked).

The sequential GEMV's candidates in the workbench's `gemv-sweep` entry (Apple M2, the 14 records of 2026-11-18 and
2026-11-19 that the table `apple-metal-3` is derived from): a candidate of the identical class is one that writes the
words of the record's baseline, which every one does on Vulkan and D3D12. On the M2, 423 of the 2,768 variant rows
compared at one row differ from their baseline — every one a variant of 32 columns or more whose unroll is not 4
(`c32`, `c64`, `c128` and `c256` with `u1`, `u2`, `u8` or `u16`), every one over an integer weight (int4, int8, int2);
no `c16` variant, no `u4` variant and no variant over a bf16 weight differs. So do the sequential row blocks of height
2 and 4 on one shape, int4 g32 `2816 × 2112` at 4 and 8 rows, against height 1 (8 rows of two records). The sweep's
verdict of such a record is `fail` (8 of the 14), and the derivation refuses a candidate that is not identical to its
baseline, so no table holds one: none of the 87 sequential exceptions of `apple-metal-3` and none of the default
profile's 22 differs in those records. What it costs is the verdict of a sweep on Metal and the candidates the
derivation can choose from there, not an output. On the Apple M5 (eight records of 2026-11-19 under the default
profile) more variants differ: 1,165 of the 2,749 rows compared at one row, every one over an integer weight and with
an unroll that is not 4, 238 of them 16-column variants, which never differ on the M2
([research](research/2026-11-19-gemv-tables-across-devices.md) §7).

- **Impact**: what does not hold on Metal is the statement "this rewrite changed no word", not the output. The
  product runs one form of each kernel on every device, so no two paths of one device disagree. Gemma 4 26B
  (`i4-fast`, f16 KV) generated the same token ids on the M5 as on the RTX 5070 Ti over three turns (temperature 1,
  seed 0). Affected in the last bits on Metal: the decode attention of the Gemma 4 lines.
  The grouped form of `expert_linear` (a run of 2 to 768 tokens over weight rows of 64 words or more) takes the
  words that its 4-word unrolled loop leaves with one loop per row for this reason: with every row's word in one
  loop body, the M2 rounded the rows after the first of a walk unlike the one-row form (55 of 72 cases at each of
  the three shapes with words left, first difference 4 ULP; none at the three shapes without), and with a loop per
  row all 432 cases are identical (record `outputs/workbench/browser-apple-m2/kernel-check_2026-11-19T09-39-19.json`,
  untracked, checkout `184fd433`). No shipped model was affected: Gemma 4 26B A4B's grouped projection, gate_up,
  has 88 words and leaves none. The kernel as it is now, its pooled grouped form included, is identical on the M2
  in all 216 cases of the check (`kernel-check_2026-11-19T09-55-15.json` in the same folder, checkout `b2a66718`).
- **Mechanism (unconfirmed)**: Dawn compiles Metal shaders with relaxed floating-point math where the OS allows it,
  so the rounding of a sum or a product chain depends on the surrounding code. The same class as the sections above.
  Confirmed for the fixed `rms_norm` prefetch form (a product chain reassociated, above). Analysis of the table's
  rows (2026-11-13, M2 record `kernel-check_2026-11-13T21-22-18.json`, untracked):
  - `state-qk-fold`, the ①′ side: Metal computes the term `(q · scale) · (k · scale)` as
    `fma((scale · scale) · q, k, acc)`. Recomputed in f32 this reproduces the ①′ word at the first failure of every
    one of the 36 failing decode cases (M = Q = 1). The folded side matched no simple grouping or accumulator split
    tried, so its form is unknown.
  - `state-pv-unroll` and `state-stats-pv-unroll`: cases where each lane takes one column (none reaches the batched
    loop) fail too, so the difference is in the term `exp(s − amax) · inv · v`, not in the order of the sum. v1
    computes `p` before the branch that loads `v`; the unrolled forms write the whole product in one expression.
    Not reproduced word for word (`exp` is not correctly rounded).
  - `state-kv16-readers`: the f16 and f32 read-only readers differ only in how `v` is read, with `p` and the
    product in one expression in both; the sequential and parallel readers, which compute `p` before the branch,
    are identical.

  Inferred, not measured: a rounding barrier on the factors (as in the `rms_norm` fix) would fix the grouping in
  each of these forms.
- **Ruling (2026-11-11, provisional)**: recorded and accepted — bit identity across forms was never guaranteed on
  Metal. A release reads the Metal page against this table (release runbook). What to do about Metal beyond that is
  undecided. 2026-11-13: the state-attention rows stay recorded; a barrier spike is filed under later in the backlog.

## On Metal, the VAE fusion rule `models/vae_rms_silu` differs from the unfused plan (measured 2026-11-17; recorded by the ruling of 2026-11-12)

The Wan (2.1 and 2.2) and anima pipelines pass the fusion rule `models/vae_rms_silu` by default to their VAE decoder
Sessions on every backend (ADR [0148](decisions/0148-external-fusion-rules.md) addendum of 2026-11-12 on it). Its bit
identity with the unfused plan rests on a rounding barrier: each value the unfused plan stores passes an XOR with a mask
that is 0 at run time, which keeps the compiler from contracting a multiply-add across it. That form is measured to hold
on Vulkan and D3D12 on NVIDIA (the rule's u32 check, `packages/e2e/tests/e2e_vae_rms_silu_rule_test.ts`, and the
Chrome / D3D12 runs of [perf-ledger](perf-ledger.md) V-10). On Metal the XOR form is not settled: under Deno
(naga → MSL) the bitcast round trip with an XOR of 0 was ineffective on the M2
(ADR [0105](decisions/0105-packed-static-quantize-activations.md) addendum 4), while in Chrome (Dawn / Tint) the XOR
barrier of `rms_norm`'s prefetch form holds on the M2 and the M5 (the section above).

- **Measured (2026-11-17, the maintainer's M2 / Chrome 157 / metal-3, runtime 0.22.0)**: the workbench's `session-ab`
  entry on the anima VAE decode found 749,031 of the forward's 786,432 output words different from the unfused plan (the
  first, word 0 of `clamp_30`, by 6 in the last place; the record keeps only the first differing word), while the
  unfused plan repeated its own words over 20 runs. The rule is 10% faster there (median −231 ms of 2,313 ms). So
  anima's images on Metal are not the unfused plan's words; Wan's VAE passes the same rule and was not run there. The
  entry's verdict on Metal is `fail`, because it requires identity.
- **Ruling (2026-11-12)**: the rule runs on Metal too, and a difference there is recorded rather than gated, as for the
  Chrome-on-Metal kernel rewrites just above; no option turns it off. The maintainer saw the measurement above and
  kept the ruling for now (2026-11-17).

## On Metal the out-of-memory errorScope stays silent — the fail-loudly check does not fire (an independent bug)

On real **M2 (24GB / maxBufferSize 14.3GB)** hardware under Deno (wgpu), the test in `gpu_generation_context_test.ts`
"a failed state allocation is fail loudly through an out-of-memory errorScope (real GPU)" is
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

## On Metal under Deno (wgpu), `gpuTiming: true` takes the device down — the isolation experiment has not been started (needs real M2 hardware)

On real **Apple M2 / macOS 26 / Deno 2.9.x** hardware, a run with `gpuTiming: true` (then `examples/gemma4
--diagnostics`, removed in 0.15) loses the device on the first turn and fails. In Chrome on the same M2 (Dawn) the
timestamps work (the workbench's decode bench reads GPU time per token there). The mechanism and the established facts
are written in the [limitations](limitations.md) section "On Metal under Deno (wgpu), GPU-side timestamp measurement is
not practical"
(`createQuerySet` fails → `DeviceError::Unexpected` → device lost; it does not enter the errorScope).
**What is unconfirmed is the resource axis**: it has not been isolated which dominates, ① the number of query sets alive
at the same time in the first run (one chunk is fixed at 16 dispatches, so about 100 for gemma4 prefill ≈1,500
dispatches) or ② the amount that stays until GC because Deno's `GPUQuerySet.destroy()` is a no-op. The two are not
exclusive. The hardware was **confirmed to be macOS 26 (Metal 4) (2026-09-03)**, so it also falls within the range of a
separate unfixed problem where timestamps are all zero even when allocation succeeds
([wgpu#9414](https://github.com/gfx-rs/wgpu/issues/9414)) — even with the fix candidate below, op-level breakdowns may
not become readable on this machine.

**The full verify also has 1 red in the same form (2026-09-03, measured on M2 / macOS 26)**: even in
`deno task verify`, which does not pass `--diagnostics`, the census check of
`packages/e2e/tests/e2e_gemma4_pretrained_test.ts` (then under `packages/models/tests/`; "gemma4 distribution: the
pipeline default runs with ③' parallel reduction…") ran the real-weight gemma4 prefill with `acquireGpu({ gpuTiming:
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
experiment needs real M2 hardware — backlog `later` item "Isolation experiment for `gpuTiming: true` on Metal through
Deno's wgpu"). Since the hardware is confirmed to be macOS 26 and timestamps may be all zero due to wgpu#9414 even if
reusing the query set makes allocation succeed, the experiment comes before the change (ruling of 2026-09-03). For the
release decision, **it is accepted that 1 census check of verify stays red on M2** (Linux / Vulkan green; the
equivalence of parallel and sequential is ensured by `packages/e2e/tests/e2e_gemma4_reduce_parity_test.ts`, which does
not require timing).

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

## Component replacement: replacing a component with a larger one makes `fromPretrained` succeed and the first `generate()` fail (ADR 0122 decision 7, item 5; filed only)

When the replacement needs a larger buffer than the original, **`fromPretrained` finishes fetching several GiB of weights
and succeeds, and the first `generate()` (for Wan, the construction of the text stage's Session) fails** (speculation;
no actual case yet). When a component is replaced through `components` of `fromPretrained`, the replacement preset's
`session` / `gpuFeatures` / `requiredLimits` are not used, and **the pre-check uses the values of the original
manifest's preset**, so it passes the check before fetching weights (the `requiredLimits` comparison). Construction
creates no Session at all, and the text stage creates a Session per generate
(inside `encodeWanPrompts` — `packages/models/src/wan/text-stage.ts:393`), so the failure comes after construction.
This is a state that lacks the check of "consistency between the quant preset and the execution settings" that ADR
[0108](decisions/0108-container-format.md) decision 19 requires of admission, and it is not by design.

- Mechanism: the replaced component takes only the container from the replacement's manifest
  (`packages/models/src/hub/components.ts:565-585` — of the return of `resolveSelection` only `containers[key]` is
  used). Wan's family check reads the preset of the original manifest (`admitWan`, `packages/models/src/wan/family.ts:481`),
  and makes the pre-fetch check with its `requiredLimits` (called before the weight parts are fetched, same file
  `:413-421`).
- Current impact: the recipe writer (`umt5_export` — including `--intake`) writes the vocabulary embedding as i8, so a
  replacement with a compatible umT5 made by the recipe does not trigger it (if the graph descriptor is the same, the
  buffer sizes are the same too). However, the shared check of `umt5Contract`
  (`umt5GraphContract` — `packages/models/src/umt5/graph-contract.ts:134-155`) only requires "storage is i8 or f32,
  with at least one i8", so a component that holds the vocabulary embedding in
  f32 (the embedding buffer about 4×) passes the check and can fail in this way.
- Fix candidates (ruling needed): check the replacement preset's `requiredLimits` / `gpuFeatures` together with the
  original preset's (take the larger), or reject in admission a replacement whose preset differs.

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
- Of the e2e tests with sha256 rows, those that acquire the GPU themselves (the Wan pipeline / swap files, MOSS-Image's
  pipeline and generate files, the Qwen3.5 encoder, the Gemma 4 26B cut model, the runtime-side real-asset e2e, the
  real-asset e2e of depth_anything / birefnet in `packages/e2e`) match the actual GPU itself right after
  acquisition (`verifyRunningGpu` / `assertAdapterMatchesEnvironment`). The Wan e2e tests without sha256 rows (the dit /
  umt5 / ab / vae files) match it too.
- Fix candidates: inject the GPU from the test (it must be acquired with the same features the pipeline requests
  itself), or make the adapter actually used by the pipeline readable (a change to the public surface).
- **User ruling (2026-10-11): filed, not fixed** (the remainder of Codex deep review R8-003). Reasons: it happens only on
  multi-GPU machines. Injection risks moving the sha rows if it diverges from the pipeline's internal acquisition. An
  adapter-reading entry point would add public surface for this problem alone.

## `gpu_full_write_test.ts`: the linear i8a8 case never puts its poison into `xq` / `xs` (filed 2026-10-27; tests only)

The full-write case for linear i8a8's temporaries (`packages/runtime/tests/gpu_full_write_test.ts:673-791`) sizes its
poison buffer to the temporary under test (`poisonCount` = m·k/4 for `xq`, m for `xs` — `:724-726`); its comment
(`:683-685`, `:718-720`) reasons in size classes, which the liveness packer (ADR [0093](decisions/0093-transient-liveness-packing.md),
least region growth since its addendum of 2026-10-26) does not place by. The packer takes slots largest first: the node
output `y` (80 B) opens region 0; the poison (32 B or 16 B) shares `y`'s bytes there (their lifetimes do not overlap, so
the growth is 0); `xq` (32 B) and `xs` (16 B) are kept out of `y`'s region (the GEMM reads them and writes `y`) and open
fresh regions. Replaying the case's allocation program through `planTransients` gives, in both cases, the poison and `y`
at region 0 offset 0, `xq` and `xs` alone in regions 1 and 2, and one shared slot.

- So the GEMM's full write of `y` covers the poison, and the temporaries the case names start zeroed; `reuseCount >= 1`
  (`:774-775`) passes on the poison sharing `y`. A word the quantiser left unwritten would still read as 0 (an int8 0 or
  a zero scale), which the atol = 0 match catches unless the reference has a 0 there too, but the poisoned-temporary check
  the case describes does not happen.
- The conv3d case below it had the same gap and was rebuilt in `21ec9fa8` (a poison larger than every slot, `y` smaller
  than `xq`, `reuseCount` asserted to be exactly 2); the same construction fits here.
- Impact: tests only (the runtime is unchanged).

## The link check accepts absolute filesystem paths (filed 2026-11-19; tooling only)

`tools/lang/check-doc-links.ts` resolves a Markdown link target against the linking file's directory with `resolve`,
which leaves an absolute path unchanged, and then checks only that the path exists. A link written as an absolute path
of the development machine therefore passes there and fails on every other checkout, CI included: three such links in
a research snapshot turned CI red on 2026-11-19 (fixed in the snapshot the same day; the check was not changed).

- Impact: the core lane's link test (`tools/lang/tests/repository_links_test.ts`) passes on the machine that wrote the
  link and fails elsewhere; no runtime code.
- Fix candidate (not approved): report a target that starts with `/` (or a drive letter) as broken, with the reason
  "absolute path".

## The memory estimate does not count conv3d w8a8's in-node temporaries (filed 2026-10-27; listed as unaccounted since 2026-11-11; estimate callers on conv8 Sessions)

With `convCompute: "a8"` an eligible conv3d node allocates two temporaries between its two dispatches
(`buildConv3dI8a8`, `packages/runtime/src/runtime/recipe-builders/conv.ts:622-623`): `xq` = P·Cin bytes (the per-pixel
int8 words) and `xs` = 4·P bytes (the per-pixel scales), P = T·H·W of the conv input, alive from the quantiser to the
GEMM. `EstimateOptions` has no `convCompute` (`packages/runtime/src/runtime/estimate.ts:360`) and the estimate counts
neither. Its `UNACCOUNTED` list (`estimate.ts:625-631`) names them since 2026-11-11, when linear i8a8's pair became
counted (ADR [0093](decisions/0093-transient-liveness-packing.md) addendum 2026-11-11). ADR
[0133](decisions/0133-conv3d-w8a8.md)'s addendum of 2026-10-26 (channel-major quantiser) notes the gap as pre-existing.

- Sizes on the Wan2.2 TI2V-5B decoder (every conv3d runs the w8a8 key — `e2e_wan_ti2v_vae_a8_test.ts`; conv input
  shapes from the local containers' graph descriptors, `outputs/diag/wave-b-2026-10-26/v6/census.json`, not tracked): at
  tile 16 the largest pair is `xq` 24 MiB + `xs` 192 KiB on the first chunk (Cin 512, P 49,152 = 3×128×128) and
  48 MiB + 384 KiB on the next chunk (Cin 512, P 98,304 = 6×128×128); at tile 32 it is 96 MiB + 768 KiB (first, Cin 512,
  P 196,608) and 192 MiB + 1.5 MiB (next, Cin 512, P 393,216). Arithmetic from the shapes, not measured.
- The pair lives inside one node, so the live peak misses at most the largest pair; how the region total moves depends
  on the placement. No family calls the estimate on the Wan decoder today (`packages/models/src`); a caller of
  `PreparedModel.estimate` / `estimateGraphMemory` on a conv8 preset gets the under-count, named in `unaccounted`.
- Impact: callers of `PreparedModel.estimate` / `estimateGraphMemory` on Sessions with `convCompute: "a8"`; no shipped family pipeline today (execution is unaffected).
- Fix candidates: `EstimateOptions.convCompute`, counting the pair under the same eligibility as execution
  (`conv3dI8a8Eligible`, `packages/runtime/src/kernels/conv3d-i8a8.ts:114`) from a bytes function shared with
  `buildConv3dI8a8`, as linear i8a8 does.

## Gemma 4 26B with the expert cache: a second pipeline on a device acquired with the subgroup features runs out of memory on a 12 GB GPU (filed 2026-11-16; cause not investigated)

On the RTX 3080 Ti (12 GB; Node + Dawn / Vulkan), with the device acquired by `acquireGpu({ subgroups: "available" })`,
building `gemma4-qat` / `26b` at `i4` with `expertCache: { slots: 64, hostBytes: all }`, disposing that pipeline (its
dispose awaited) and building it again on the same device fails in the second build: `GpuOutOfMemoryError: expert pool
allocation` (`VK_ERROR_OUT_OF_DEVICE_MEMORY`). On a device acquired by a plain `acquireGpu()` the same sequence passes —
repeated over the 18 builds of a session A/B.

- **Reproduce**: `flock /tmp/karume-gpu.lock node apps/workbench/run.ts gemma4-session-ab --model=gemma4-qat/26b
  --preset=i4 --expertCacheSlots=64 --hostTier=all` fails within half a minute (the base probe builds and runs, the rule
  probe's build fails); with `--subgroups=false` it passes.
- **Blast radius**: an application that builds this model twice on one device with the subgroup features on a GPU of
  this size (the Playground's chat page acquires its device that way; a pipeline that acquires its own device takes the
  features its manifest asks for). One build passes. Not seen on the RTX 5070 Ti (16 GB).
- **Not known**: whether the first pipeline's memory is not back when its dispose resolves, or the second build needs
  more than the first; what the subgroup features change in this model's plan or allocations.

## Under Node + Dawn on the RTX 3080 Ti, the NVIDIA driver sometimes prints `NVVM compilation failed: 3` at the end of a process (filed 2026-11-18; not reproduced, cause unknown)

At or after the end of a Node process that used WebGPU through Dawn (Vulkan), stderr has shown

```text
NVVM compilation failed: 3
Warning: NVIDIA: CreatePipeline: failed to compile internal representation
```

sometimes followed by `Segmentation fault (core dumped)`. Every result of those runs was written, and where the run
checked its outputs, checked, before the lines appeared.

- **Sightings**: seven, all in investigation runs of 2026-10-24 to 2026-10-31 (records under `outputs/diag/`, local,
  untracked):

  | # | Run | Kernels | Where | What |
  | --- | --- | --- | --- | --- |
  | 1, 2 | `perf-recon-2026-10-24/matrix-kernels-other`, round 1 | f32 GEMM skeleton variants (the spike of perf-ledger K-88) | "at the end of two long multi-pipeline Node runs" | the first line |
  | 3 | the same lens, the verifier's run of 6 variants × 5 shapes (30 pipelines), 5 rounds | the same | at the very end, after every pipeline was used and compared; not in two re-runs | both lines |
  | 4 | `perf-recon-2026-10-24/attn-compliant`, set `factor` | online attention variants | after the last result line of the second of two processes | both lines, then a segmentation fault |
  | 5 | the same, set `hog` | online attention variants | after the results of the first of two processes | both lines |
  | 6 | `perf-recon-2026-10-24/attn-d3d12` | online attention variants | at exit, after the results were written | both lines, then a segmentation fault |
  | 7 | `qat-0151-2026-10-31/llm-decode`, 32 processes | the runtime's own kernels (a Gemma 4 decode) | among the last two lines of one process | the first line (only two lines per process were kept) |

- **In common**: Node 24 with Dawn (npm `webgpu` 0.6.2) on Vulkan, the RTX 3080 Ti, the NVIDIA driver 615.78.08
  (recorded for sighting 7; the driver installed today); a harness that ended with `process.exit(0)` while its device
  was alive; the lines at or after the last result. Four of the seven ran no K-88 code, and one ran only the runtime's
  own kernels. No sighting in Chrome on either development GPU, in the test lanes, or in a run that destroys its device
  before it exits (`runGeometrySweep`, the workbench's entries).
- **Where the lines come from**: the first string is in the driver's SPIR-V compiler (`libnvidia-glvkspirv.so`), the
  second in `libnvidia-glcore.so` (`strings`). They report a pipeline compile that failed inside the driver, at a point
  where the application creates no pipeline. What the code 3 stands for is not known.
- **A/B of 2026-11-18** (`outputs/diag/decode-optimization-2026-11-18/gemm-forms/nvvm/`, local, untracked): fresh
  processes that build 48 f32 GEMM pipelines through the runtime's pipeline cache (linear, matmul and conv2d on the
  tiles with the largest register blocks), run each, read the last output back and end. Four arms, interleaved: the
  kernels without a K-88 field or with the skeleton forms and the counter decode, and `process.exit(0)` at once or
  after `device.destroy()` and `device.lost`. Every process had an empty driver shader cache of its own
  (`__GL_SHADER_DISK_CACHE_PATH`), so each compiled its pipelines as a first run does. 16 processes per arm: no line
  and no non-zero exit status in any of the 64. Two workbench sweeps of the same
  day built 134 pipelines with a K-88 field each, over five device lifetimes, without a line. So the message was **not
  reproduced** and stays unattributed: nothing ties it to the K-88 kernels (32 processes with them and an immediate
  exit gave none: below about 9 % per process at 95 %), and the exit path is neither shown nor ruled out as its cause.
- **Impact**: none known on results. A harness that exits without destroying its device can print the lines and, as
  in two of the seven sightings, end with a segmentation fault, which turns its exit status non-zero after its
  results are complete.
- **How the A/B is run**: from the repository root, with nothing else on the GPU,
  `flock /tmp/karume-gpu.lock zsh outputs/diag/decode-optimization-2026-11-18/gemm-forms/nvvm/run-nvvm.zsh <label> <rounds> cold [<load seconds>]`.
  One round is one process per arm, the arms' order rotated per round. `nvvm-ab.ts` beside it starts the processes
  (`nvvm-process.ts`, or `nvvm-process-load.ts` when the load seconds are given) and writes one JSON line per process
  to `<label>.jsonl`: the arm, the exit status or signal, the count of each of the two lines on stderr, and the files
  the driver left in the process's cache directory; the last lines of `<label>.stdout.txt` are the counts per arm.
  The run above was `cold-1 16 cold` (64 processes, 4 minutes 45 seconds).
- **Not run** (the GPU was shared by six pieces of work that day; each needs it alone):
  1. The same A/B nearer the sightings' harnesses: a device with `timestamp-query`, the sweep's shapes
     `linear-m1024-n2048-k2048`, `matmul-m512-n2048-k1024` and `conv2d-c192-256x256`, every pipeline kept alive and
     all of them run in turn for the load seconds before the process ends —
     `… run-nvvm.zsh cold-load-1 7 cold 10` (28 processes, about 8 minutes), then more rounds if it shows a line.
  2. The harnesses that showed the lines most often: sightings 4 and 5 had them in one of two processes each. Port
     `perf-recon-2026-10-24/attn-compliant/run-bench.ts` to the current checkout (it imports a worktree that no
     longer exists), and run its `factor` and `hog` sets 20 times per arm with the two exit paths (about an hour).
  Reading either: lines only in the arms that exit at once attribute the message to the exit path, and a harness that
  destroys its device before it exits is then free of it; lines in the arms that destroy the device first make it a
  property of the kernels compiled, and then the arms with and without the K-88 fields say whether those kernels
  matter.

## The decode bench's split mode fails intermittently (filed 2026-11-03; the `gemma4-bench` entry's split arms only)

On the maintainer's M2 (Chrome 158), arms of the decode bench (then the Playground's `/gemma4-bench` page, now the
workbench's `gemma4-bench` entry) with the split on failed — once with "Integer GEMV"
on, once on a plain QAT arm — with `setBindGroup: The provided value cannot be converted to a sequence`, raised in the bench's GPU tap
(`apps/workbench/src/gemma4-bench/tap.ts`). It was not reproduced in isolation and has not been investigated;
the same plain arm and a byte-plane arm ran with the split in later runs
([research 2026-11-03](research/2026-11-03-qat-integer-gemv-m2.md) §8, §12).

- Impact: the decode bench's split mode only (a run can fail before its per-key attribution is complete).

## The bidirectional block of a generation run: four edge cases outside the shipped path (filed 2026-11-08; recorded only)

A review of `GenerationRun.bidirectionalRows` (ADR [0144](decisions/0144-gemma4-12b-vision.md) decision 3) found four
cases where the block's behavior differs from its contract. Gemma 4 12B's image chat cannot reach any of them: it runs
the block on its own immediate chunk of a context the Session made, on sliding slots, far below 2³² rows.

- **A context from outside the type.** `assertBidirectionalRows` (`packages/runtime/src/runtime/executor.ts:439`) reads
  `run.context[RUNTIME_INTERNAL]` before the context check that runs later in issue. A run with a block on an object that
  is not a `GenerationContext` rejects with a `TypeError`, not the `ExecutionError` the same run without a block gets.
  No state is touched.
- **The CPU reference on a full slot.** `referenceStateAttention`
  (`packages/runtime/src/reference/state-attention.ts:192`) applies the block whatever the window, so with window 0 (a
  full slot) it ORs the block in. The runtime keeps full layers causal and the kernel builder refuses a block without a
  window. The reference describes a behavior the GPU path refuses; no test passes a block to a full slot.
- **The u32 bound of the block words.** `#writeLengths` (`packages/runtime/src/runtime/generation-context.ts:1539-1542`)
  writes `pastLength + start` and `pastLength + end` into the lengths uniform without a bound check. When
  `pastLength + end` is 2³², the upper word wraps to 0 and the block becomes empty, so the run is silently causal.
- **A second slot backing for block runs.** The prepared key gains `|bidir`
  (`packages/runtime/src/runtime/executor.ts:1786`), and the slot backings are keyed by it, so an image chunk and a text
  chunk of the same 768-row signature hold two backings. Under a backing budget sized for one, alternating chunks could
  evict and rebuild one per chunk. Performance and memory only; not measured.

- Impact: callers of the runtime's `GenerationRun.bidirectionalRows` outside the shipped 12B pipeline, and the CPU
  reference's tests; the shipped image chat is unaffected.
- Fix candidates: the context check before the block rule; the reference refusing a block on window 0 as the kernel
  builder does; a u32 bound in the block rule; measure the backing split before changing the key.

## Wan2.2: the VAE decode of a 5-frame clip takes several times longer than that of a longer clip (filed 2026-11-15; cause not investigated)

At 1280×704 with the f16 VAE decoder (the presets `f16+dit8` and `f16+dit8-a8-attn8-s16` / `-online`), the `vae_decoder`
stage of a 2-step run took, on the RTX 3080 Ti (Node + Dawn, geometry `nvidia-ampere`):

| Frames | Chunk runs per tile | `vae_decoder` stage | Per chunk run (28 tiles) |
| --- | --- | --- | --- |
| 5 | first + 1 next | 227–246 s (4 runs, one of them alone in its process) | about 4.1 s |
| 9 | first + 2 next | 45 s | about 0.54 s |
| 17 | first + 4 next | 81–91 s | about 0.63 s |

The chunk graphs and their input shapes are the same at every length (one latent frame per run), so the work per run
should not depend on the clip's length. In the 5-frame runs the executor reports a largest window mean of about 1.3 s for
`vae_decoder_first` and 1.2–1.5 s for `vae_decoder_next`, with 660 and 720 windows over budget; at 9 frames it reports
105 ms and 48 ms with 54 and 0 over budget. The int8 decoder on int8 activations (`i8-a8-attn8-online-conv8`) is not
affected: its whole 5-frame case took 39 s (20 s at `vaeTile: 32`). Not checked: other sizes, Wan2.1, Chrome, other GPUs.

- Impact: a 5-frame clip (the shortest the descriptor accepts) decodes in about 2.6 times the time of a 17-frame clip.
  The outputs are finite and deterministic (the same sha256 in two runs). The default lane's short seed cases run 9 frames
  for this reason (`SHORT_FRAMES` in `packages/e2e/tests/e2e_wan_ti2v_pipeline_test.ts`).
- Records: `outputs/diag/restructure-2026-11-15/` (`wan-ti2v-write.log`, `wan-5f-alone.log`, `wan-9f-alone.log`; local,
  untracked).

## On the RTX 3080 Ti development machine, two slowdowns distort measurements (filed 2026-11-19; the machine, not a product bug)

- **The slow mode** ([research 2026-10-25](research/2026-10-25-speed-recon-3080ti.md) §9.3): tiled GEMM-class kernels
  run ×15–42 slower for whole submits, in episodes of 5–8 submits (about 14 s) that recur under sustained gap-free load
  and once after a wake from idle; other kernels in the same submits are unaffected. The memory controller reads
  96–100% busy while clocks and power stay at P0, and no thermal or power-brake counter fires. It attaches to the
  pipeline object: a fresh pipeline from the same WGSL runs at normal speed. Not seen on the RTX 5070 Ti. The mechanism
  is open; the display was detached from the card on 2026-11-19 to see whether it stops (no result yet). The sequential
  GEMV and the GEMM also run in two timing modes on this card ([research 2026-11-18](research/2026-11-18-float-parallel-gemv.md)
  §6), which spreads the rounds of a GEMV sweep's baseline — in the turbo mode too (216 of 690 groups of two `full`
  records of 2026-11-20 beyond ×1.25), so it is not the thermal throttling below; the derivation reads a reference at its fastest time
  over the records for that reason (perf-ledger K-113). The periodic stretches of EmbeddingGemma 2 at 2,048 tokens
  ([below](#embeddinggemma-2-on-the-rtx-3080-ti-under-node--dawn-a-major-gc-stalls-some-long-embeds-on-the-host-and-2048-token-embeds-have-periodic-stretches-about-11-slower-on-the-gpu-filed-2026-11-22-the-host-cause-found-2026-11-22-the-stretches-cause-not-investigated))
  show the same signature in its attention kernels.
- **Clock changes under load** (thermal throttling identified 2026-11-17; mitigated 2026-11-20 by switching the card
  from its quiet mode to its turbo mode): in the quiet mode the card ran at 77–96 °C under long or shared load with its
  thermal slowdown flag set, and a long run after a quiet spell was slower than the same run warm. In the turbo mode
  the same GEMV sweep (three cases, 10 rounds) reads the flag in 0.7% and 2.6% of the samples of two runs, against
  79%, and peaks at 78 °C, against 95 °C; a gap-free geometry sweep of 15 minutes at about 300 W levels at 79–81 °C
  with the flag in 0–6% of a minute's samples. What is left is the power cap: at full load its flag is set in about
  nine samples of ten and the SM clock still moves between 1,305 and 1,920 MHz. Its effect on timings is not measured.
- Impact: wall-clock and per-dispatch numbers on this machine, and the sweep records the device tables are derived
  from. Outputs are not affected. Compare arms interleaved in one process, and read a median beside a minimum. Records
  made before 2026-11-20 were taken in the quiet mode.

## EmbeddingGemma 2 on the RTX 3080 Ti under Node + Dawn: a major GC stalls some long embeds on the host, and 2,048-token embeds have periodic stretches about 11× slower on the GPU (filed 2026-11-22; the host cause found 2026-11-22; the stretches' cause not investigated)

Seen in the workbench's `embeddinggemma-preset-ab` (the `bf16` and `i8` presets of the local mirror resident on one
device, A B B A rounds after a warm-up round) on the RTX 3080 Ti (Node + Dawn / Vulkan, geometry `nvidia-ampere`),
checkout `cd475469` with the uncommitted `i8` work, and investigated on 2026-11-22 at `943ad5c9` with a standalone
script under plain Node and under the Vite module runner the workbench runs in. Filed first as a slowdown of the `bf16`
preset itself (every other `bf16` embed at 8,192 tokens about 1.8× slower); the investigation showed it is not one.

- **8,192 tokens: one visit position per round, stalled on the host.** In the four workbench records one visit of
  each A B B A round took 1,514 to 1,816 ms against 930 to 986 ms for the others, and that visit was a `bf16` one in
  all four (the round's first visit in `…T12-08-37.json`, its second in the swapped `…T12-12-10.json`, its fourth in
  `…T13-47-24.json` and `…T13-57-58.json`). What the investigation observed:
  - While a readback's `mapAsync` is pending, the Node binding of Dawn (npm `webgpu` 0.6.2) polls through
    `setImmediate`, each poll a new native callback: 335 to 361 thousand polls per 8,192-token embed (the runs of e08
    and e09 that held no major GC), and about 260 MiB of short-lived objects per embed under those callbacks (a heap
    sampling of one embed, a01: 217 and 43 of 320 MiB).
  - Every few embeds a major GC stops the main thread for 322 to 400 ms (e08, w02), and in most stalled runs another
    213 to 223 ms (e08) or 236 to 293 ms (w02) pass after the readback resolves before the next embed begins.
  - In the workbench process the major GC came about every 4.6 s (w01 `--trace-gc`: the heap at 1,040 to 1,126 MB,
    back to 269 MB after), which is one A B B A round at this length, so it landed on the same visit every round.
    Under plain Node the heap stayed smaller (383 to 421 MB), the major GC came every 1.5 to 1.7 s with pauses of 97
    to 139 ms (e03), and a run was slow only when the GC met the end of its GPU wait.
- **Evidence** (`outputs/diag/eg2-bf16-alternation-2026-11-22/logs/`; wall time of one `embed`, the first two runs
  of each log left out; the runs of the slow mode, 11.5 to 21.4 s, also left out):

  | Run | How | Slow runs | Other runs |
  | --- | --- | --- | --- |
  | e01 | Plain Node, `bf16` loaded alone | 6 of 18, every third, 1,181–1,211 ms | 932–943 ms |
  | e02 | Plain Node, `i8` loaded alone | 6 of 18, every third, 1,114–1,167 ms | 925–937 ms |
  | e08 | Vite module runner, both resident, A B B A (`bf16` first) | 10 of 38, every fourth run (a `bf16` visit), 1,162–1,381 ms | 923–974 ms |
  | e15 | As e08, B A A B (`i8` first) | 5 of 20, every fourth run (now an `i8` visit), 1,100–1,710 ms | 925–974 ms |
  | e09 | As e08, a forced full GC between visits, outside the timed span | none of 38 | 926–977 ms |
  | e12 | As e08, with GPU timestamps (runs 2 to 17) | 3 (1,110–1,218 ms), GPU time 922–929 ms | 942–984 ms, GPU time 914–931 ms |

  GPU time does not separate the presets either: in e12 the `bf16` runs took 917 to 931 ms of it, the `i8` runs 914
  to 927 ms.
- **2,048 tokens: a different thing, on the GPU.** Outside the stretches an embed takes 114 to 125 ms. From round 11
  on, a stretch of eight consecutive embeds recurs every 6 rounds (every 12 runs of each arm); in it an embed takes
  1,357 to 1,375 ms, about 11 times as long, and its first embed and every third one after it 1,528 to 2,152 ms. With
  GPU timestamps (e14, 138 runs) the GPU time itself grows in a stretch, 1,363 to 1,368 ms against 113 to 119 ms, and
  only in the attention kernels (median per run: the banded score kernel about 15 → 880 ms, the full-attention score
  kernel 5.6 → 329 to 330 ms, the PV kernel 16 → 80 ms), while the linears keep 59 to 62 ms. That is the signature of
  the slow mode in [the section above](#on-the-rtx-3080-ti-development-machine-two-slowdowns-distort-measurements-filed-2026-11-19-the-machine-not-a-product-bug).
  Only the extra-slow runs inside a stretch are the GC stall above (e14: 2,169 and 2,065 ms of wall time on 1,366 and
  1,368 ms of GPU time). An earlier, irregular stretch near the start of each run (embeds of about 300 ms in one run,
  about 440 ms in the other) was a GPU wait too (a `mapAsync` wait of 435 to 438 ms, w03) and was not looked into
  further. Seen in both runs of 31 rounds and in w03.
- Records: `outputs/workbench/node-nvidia-geforce-rtx-3080-ti/embeddinggemma-preset-ab_2026-11-22T12-08-37.json` and
  `…T12-12-10.json` (8,192 tokens, the two arm orders), `…T11-57-57.json` and `…T12-05-37.json` (2,048 tokens); the
  investigation's workbench runs `…T13-47-24.json`, `…T13-57-58.json` (8,192) and `…T14-11-18.json` (2,048); the
  `nvidia-smi` samples in `outputs/diag/eg2-preset-ab-2026-11-22/`; the investigation's scripts and logs (e01 to e15,
  w01 to w03, a01) in `outputs/diag/eg2-bf16-alternation-2026-11-22/` (all local, untracked); the reading in
  [research 2026-11-22](research/2026-11-22-embeddinggemma2-quant-screen.md) (its section on the speed of the `i8`
  preset).
- **Not verified**: the binding's source was not read — its polling is described from the observed behaviour (every
  callback native, a new function each time, the count growing with the GPU wait); the reason for the stall after the
  GC (in it `getMappedRange()` to `unmap()` took 113 to 195 ms in the stalled runs and 0.0 ms in the others, e11);
  why the stalled position was a `bf16` visit in all four workbench records (e15 shows that the position decides, not
  the preset); whether a browser shows it (it has no such polling loop: expected not, untested). Not tried: another
  GPU. The cause of the slow mode: not investigated.
- **Impact**: Node + Dawn runs and measurements only — the wall time of long embeds, and the medians of long-text
  measurements under Node. The runtime's GPU work and the outputs are not affected: each preset's vector was
  bit-identical over every run of every record. At 8,192 tokens the two presets take the same GPU time; the 1.37 to
  1.39 times of the `bf16` mean over `i8`'s in the first two records is the stall's position, not the preset.
- Fix candidates: an upstream report to the binding (one allocation per poll; not decided); recording the number of
  major GCs and their pause per visit in the workbench A/B record — built 2026-11-22: `karume-embeddinggemma-preset-ab/2`
  holds, per arm and timed visit, the count of major GCs that began inside the visit's `embed` and their summed pause,
  and the same for the other kinds together (or that the host cannot observe GC), and its summary says per arm how many
  timed visits held a major GC and the range of their pauses ([the spec](spec/workbench.md#garbage-collections-of-the-host));
  and, as a measurement option, not a default, a forced GC between visits outside the timed span (it removes the
  pattern, e09, but hides a cost a Node caller pays; not decided).
