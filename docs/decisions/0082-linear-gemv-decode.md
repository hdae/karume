# 0082: GEMV kernel family for linear (decode M=1 × i4 weights, reference path, bit-identical)

- Status: accepted (2026-08-31; perf-ledger K-11 — the spike measured mechanism, gain and bit
  identity before the implementation, as in ADR 0060); partly superseded by
  [0098](0098-linear-gemv-parallel.md) (decision 5, 2026-09-13: K-parallel reduction adopted as an
  opt-in); the storage clause of decision 4 opened by addenda 4, 6 and 7 and by
  [0131](0131-bf16-weight-execution.md) (bf16); re-graded 2026-10-18 to SHOULD: the `v4` clause of
  decision 4 and the 256-element cap of addendum 5.
- Scope: runtime (`src/kernels/linear-gemv.ts`, `buildLinear` in
  `src/runtime/recipe-builders/linear.ts`). IR, exporter, distributed assets and public API
  unchanged (no re-export).
- Related: ADR [0022](0022-gemm-register-blocking.md) (decision 3, the numerics contract of the
  reduction order — this ADR **extends** its reach to a second kernel family, it does not withdraw
  it) / [0069](0069-packed-w4-storage.md) (i4 storage, group scale, nibble order) /
  [0058](0058-numerics-opt-in-contract.md) (numerics opt-in contract — this ADR **adds no option**,
  see decision 3) / [0021](0021-gpu-timing-diagnostics.md) (per-key GPU time, the measuring point).
- Demand: [research/2026-08-30-gemma4-decode-wallclock.md](../research/2026-08-30-gemma4-decode-wallclock.md)
  (decode ≈85 ms/token, 85% of it 73.3 ms in 276 `linear:…:wi4g32` dispatches; §7 holds this ADR's
  measurements).

## Context

Gemma 4 E2B decode took ≈85 ms per token, over 80% of it in `linear` at M=1. The K-11 entry read
"effective ≈16 GB/s, under 2% of peak bandwidth, so bandwidth is used badly", but **the spike showed
it was not bandwidth starvation**:

- Time is **proportional to k and almost independent of n**; even shapes whose weights fit in L2 take
  about the same time per dispatch (a bandwidth bound would scale with n).
- The cause is the double `workgroupBarrier()` per K tile of 16: **the latency of weight reads is
  exposed serially, once per tile** (measured `k / 16 × ≈1.3 µs`). The M=1 bucket geometry `M16N16`
  (64 threads) also has **only 4 threads writing output**; 15 of the 16 rows of the shared A tile are
  zero-filled dead weight.

The GEMM skeleton (shared tiles + barriers) does not fit M=1 structurally, and no geometry sweep
reaches it. This is the point where a second kernel family is added instead of a new geometry.

## Decision

### 1. `linear` gets a second kernel family, `linear_gemv` (reference path, no opt-in option)

`src/kernels/linear-gemv.ts` holds a GEMV with **one thread per output column**, without shared
memory or barriers. Binding numbers, the uniform (`{m,n,k}`, 3 words) and the output are the same as
the GEMM path; only two element types (weights as `vec4<u32>`, 16-byte reads; scalar `f32` output)
and the work assignment change. `unroll` weight words are issued ahead to create memory parallelism,
and `n` independent reductions hide the latency.

The generation parameters (`LinearGemvVariant` = `cols` columns per workgroup × `unroll` words read
ahead) have **one selection point, `defaultLinearGemvVariant()`** (default `c32 u4`; addendum 10 adds
one shape). As for the i8a8 and f32 geometries, there is no run-time autotuning, and the variant is
part of the pipeline key.

### 2. Selection has two stages (the reach of gemm-geometry's "single selection point" MUST)

`src/kernels/gemm-geometry.ts` says "`defaultGemmGeometry` is the single selection point" and
"`gemmGeometryForRows` is pure"; their reach is **inside the GEMM skeleton** (which geometry cuts the
tiles). With this ADR the selection for linear has two stages:

1. **Family selection** — in one place, `buildLinear`; a predicate over plan-time shapes and Session
   options only.
2. **Geometry selection inside the skeleton** — `gemmGeometryForRows` (only after the GEMM family is
   chosen).

MUST: both stages are **pure functions of plan-time shapes**, and the result is always in the
pipeline key (family `linear_gemv`, variant `c32u4`, group `g32`). ADR 0022's "no run-time
autotuning" and "same key → byte-identical WGSL" are not relaxed by one character.
MUST: adding a family does not add a stage — family selection stays in one place, `buildLinear`.

### 3. Numerics: the reference path (no ADR 0058 opt-in option)

ADR 0058 options are for optimizations that **change numerics**. This family does not, so it adds no
option (the same judgement as ADR 0060 decision 4). Bit identity rests on the same freedom ADR 0022
decision 3 grants geometries: **only the work assignment changes**.

- Each output element is reduced sequentially in ascending k (ascending words × ascending elements
  within a word).
- The multiply-add is written `acc = acc + a * b`.
- Weights are restored as `f32(i32(u) − 8) * scale`, a per-component f32 multiply (same as
  `dequant4`).
- The bias is added once, after the reduction.

MUST: read-ahead (`unroll`) only **reads** words together; the multiply-adds stay in ascending word
order. Mixing multiply-adds across words breaks the contract.
NOTE: f32 reduction has no theoretical order independence, so bit identity is, as in gemm-geometry,
a **measured claim** (Verification). The ADR 0058 addendum of 2026-08-29 ("automatic selection that
claims bit identity needs a machine check on that device") is met by the u32 check in
`packages/runtime/tests/gpu_linear_gemv_test.ts`; since the family is chosen by shape alone, no
run-time canary is needed. Apple M2: see addenda 1 to 3.

### 4. Eligibility stays inside the measured range

`buildLinear` picks the GEMV family when `m === 1` × `i4` weights × `f32` compute × `v4` ×
`groupSize % 32 === 0` × `k % 32 === 0`. The first four are the swept combination; the last two are
structural (one weight word of 32 elements does not cross a group; the 16-byte alignment of the
`vec4<u32>` binding — `k % 32` also follows from ADR 0069 decision 2, restated as a binding
requirement).

The eligibility SHOULD NOT be widened to shapes that are not `v4` until they are measured, by the
same discipline as `gemmGeometryForRows` ("do not fill points outside the sweep by interpolation").
(Re-graded 2026-10-18 to SHOULD: the clause keeps the eligibility inside the measured range; a sweep
plus the u32 check can open it.) The clause also closed i8 / f16 storage (superseded for storage by
addendum 4 on 2026-09-06 for i8, addenda 6 and 7 on 2026-09-11 for f16 and f32, and ADR 0131 on
2026-10-17 for bf16).
NOTE: `v4` (`n % 4 == 0`) is **not** a GEMV requirement (the output is written as scalars); it stays
only to keep to the measured range, and removing it is a matter for a future sweep.

### 5. split-K is not adopted (kept as a record)

(Superseded by ADR 0098 on 2026-09-13: K-parallel reduction, `SessionOptions.linearGemvReduce:
"parallel"`, is an opt-in option, and the gemma4 default preset `i4-fast` declares it —
`tools/export-recipes/gemma4/distribution.py`, ADR 0128; ADR [0101](0101-linear-gemv-subgroup.md)
adds a subgroup variant.)

split-K (several threads share one output column and sum partial results) was implemented and
measured. **It loses bit identity (a different reduction order needs an ADR 0058 option) for only
×1.40 over the bit-identical version**, beyond which the fence floor ≈11 ms (perf-ledger H-2)
dominates and the wall clock flattens (RTX 3080 Ti, Deno). With nothing gained for breaking the
contract, it was **not adopted** and not kept in the code.

## Verification (measured on RTX 3080 Ti / Deno 2.9.6 / Linux, 2026-08-30 to 31)

The u32 checks now run in the Vitest GPU lane on Node + Dawn (ADR [0129](0129-test-layout-and-lanes.md)).

- **Gain**: the kernel alone, census-weighted, 62.38 ms → 7.38 ms (**×8.45**, far over the ×1.3 kill
  criterion). Real graph: decode GPU time 86 → 20 ms; decode wall 84.2 → 32.5 ms/token (**×2.59** —
  the rest moves to the fence floor and non-linear ops).
- **Bit identity (spike)**: 16 shapes alone (12 census shapes + 4 edge shapes, `n=100 k=1568`,
  `n=1500 k=1536`, `n=36 k=288`, `n=4 k=32`) agree **as u32 on every element**. On the real graph the
  three margins of greedy parity match exactly.
- **Bit identity (permanent check)**: `gpu_linear_gemv_test.ts` compares GEMV output with the leading
  rows of the GEMM path **as exact u32** (`gpu_gemm_skinny_test.ts`'s cross-bucket check extended
  across families; the comparison side runs at M = 65 since addendum 5, M=2 before), plus allclose
  against a CPU reference (no tautology if both sides ran GEMV). The shapes cover **edges production
  never hits**: `n % 32 != 0`, `units % 4 != 0` (read-ahead remainder), `units < 4` (no read-ahead).
- **Fault injection**: ① a group-scale shift off by one fails the u32 check; ② "upper nibble first"
  in-word order (**same products, different addition order**) fails only the u32 check, by **1 ULP**
  (`0xc083c922` vs `0xc083c921`), while allclose passes — the check means something only as exact u32.
- **Codegen determinism**: `fixtures/wgsl/linear_gemv_g32.wgsl` and `_g64.wgsl` (with g32 the scale
  index `(unit · 32) >> shift` collapses to an identity, so the baked shift is visible only with g64 —
  **the pair is the condition**). The same enumeration detects any byte change in `linear_wi4*`.
- **Eligibility itself**: on a real GPU, `linear_gemv:` appears in the pipeline key only for the
  eligible shapes, and shapes that drop **exactly one** condition (M = 65 since addendum 5, group 16,
  n = 33) keep the `linear:` key.

## Consequences

- Decode is now bound by the fence floor (perf-ledger H-2, ≈11 ms) and non-linear ops. The revival
  condition of the LLM lenses L-7 / L-12 (dispatch diet / prefill chunkLength; not perf-ledger's L-7)
  is met; re-evaluating them is out of scope.
- **Prefill (M=32) was out of scope** (GEMM at 5.2 ms/token, a different mechanism); addendum 5 later
  extended the family to M ≤ 64. **Metal**: dynamic `vec4` indexing is avoided; see addenda 1 to 3.
- Diagnostics and the census count this family by the **`linear_gemv:` prefix**; a
  `startsWith("linear:")` filter does not catch it.

## Addendum 1 (2026-09-01) — Metal: the bit-identity claim breaks by 1 ULP on M2; the default stays

On an M2 the exact u32 check fails by 1 ULP (`0x414b3249` vs `0x414b3248`, k128 n64 g32; the key
check is green; Linux / Vulkan are green). The cross-bucket u32 check of `gpu_gemm_skinny_test.ts` is
green on M2, so the break is **specific to GEMV**. Suspected mechanism (unconfirmed): the GEMM path
stores the i4 dequantization in the shared B tile before the MACs read it (workgroup memory acts as a
rounding barrier), while GEMV writes `x × (f32(q−8) × ws)` as one expression that MSL contraction may
round differently (the same class as the 1-ULP epilogue issue in the Metal section of
`docs/known-issues.md`).

Ruling (2026-09-01): **the reference path keeps GEMV**, on three grounds: (1) the product contract
holds — the margin check (≥2.5e-2) is far above 1 ULP, and the chat e2e was golden-identical on the
M2 with the decode speed-up in effect there; (2) the u32 check exists to detect kernel regressions
and keeps doing so on Linux / Vulkan, with the M2 red treated as a known red; (3) disabling GEMV on
Metal would cost ×8 in decode, out of proportion. Decision 3's claim is refined to "**holds on Vulkan / Linux; Metal is within 1 ULP**". The
root-cause candidate (an explicit rounding point in GEMV's dequantization) is not started.

## Addendum 2 (2026-09-01) — ground 1 of addendum 1 withdrawn

Ground 1 (chat e2e golden on M2) had not been run (gemma4 prefill then gave NaN on Metal, a separate
bug); the ruling stood on grounds 2 and 3.

## Addendum 3 (2026-09-02) — the claim re-established on M2

After the NaN fix (`tanh_stable`) the gemma4 chat e2e (temperature 0, equal to the Linux golden)
passed on M2 with the u32 check still 1 ULP red (HEAD `2b096c4`): the output does not move. Grounds
1–3 hold.

## Addendum 4 (2026-09-06) — eligibility widened to i8 storage: lm_head (M=1) joins GEMV (perf-ledger K-16)

Decision 4 closed i8 × f32 × M=1 for lack of measurements; it was measured and opened (`5ddd186`).

- Target: gemma4 decode lm_head (i8 [262,144 × 1,536], per-output-channel scale), 24% of decode GPU
  time (5.3 ms/token) on the GEMM skeleton at M=1. Same mechanism as decision 1, independent of the
  storage dtype.
- Implementation: a storage axis (one word = 32 i4 or 16 i8 elements; `linearGemvUnit(storage)` is the
  single derivation point). i8 restores as `f32(q) * scale` with the per-channel scale outside the
  reduction (ADR 0019); order and form per decision 3 — **bit-identical to the GEMM path** (u32 check
  on 6 i8 shapes + exact gemma4 golden). i4 WGSL is byte-unchanged.
- Eligibility: `m === 1 × i8 × f32 × v4 × k % 16 === 0` (no group condition). `n % 4 != 0` and
  `k % 16 != 0` stay on the GEMM path per decision 4.
- Measured ([research 2026-09-06](../research/2026-09-06-gemv-i8-k16.md)): kernel 5.317 → 1.056 ms
  (×5.03); decode GPU 22.8 / 21.8 → 17.5 / 17.0 ms/token (−21 to −24%); wall −3.2 to −4.8%. `c32 u4`
  is best.
- The remaining i4 down-projection and ①QK (K-14) reduce k sequentially in one thread and cannot be
  split without changing the order, so they need an opt-in option design (done in ADR 0098).

## Addendum 5 (2026-09-07) — eligibility widened to 1 ≤ M ≤ 64: row-block variant (perf-ledger K-21)

Decision 4 closed M ≥ 2 for lack of measurements; it was measured and opened (`5701262`).

- Demand: M ≥ 2 linear fell to the GEMM skeleton (M16N16 for M ≤ 64) and paid an **M-independent fixed
  cost of 65–67 ms** over gemma4 E2B's 276 dispatches (short chat prefill, speculative verify —
  [research 2026-09-07 §7.2](../research/2026-09-07-codex-perf-review-followup.md)). Same mechanism.
- Implementation ([research 2026-09-07-gemv-rows-k21](../research/2026-09-07-gemv-rows-k21.md)): a
  **row-block variant** `linearGemvRowsWgsl` (one thread = one column × `rows` rows; weight word and
  scale read once and multiplied with each row's x; `ceil(m / rows)` y tiles). M=1 keeps its variant
  (decode WGSL byte-unchanged, guarded by the fixtures).
- Selection stays two pure stages over plan-time shapes (decision 2's MUST unchanged): family in
  `buildLinear` (1 ≤ m ≤ `LINEAR_GEMV_MAX_ROWS` = 64, other conditions as decision 4 / addendum 4);
  variant by `defaultLinearGemvRowsVariant(storage, m, n)` (`r<rows>` in the key). `rows` starts at
  the largest power of two ≤ m under the cap (i4 8 rows / i8 16 rows) and halves until
  `n · ceil(m / rows)` threads reach 16384. That target is the saturation point of the reference
  device (RTX 3080 Ti), not a portable optimum (`docs/limitations.md`;
  `SessionOptions.linearGemvRowsThreadTarget` overrides it).
- Numerics: reference path (decision 3 extended to row blocks). Per-element reduction order, product
  pairing and bias order equal the M=1 variant and the GEMM path; the dequantized f32 is held in a
  `let` shared across rows (the same single multiply as rounding point). Bit identity remains a
  measured claim: a spike over 6,422 combinations and the permanent `gpu_linear_gemv_test.ts` (6 i4 /
  5 i8 row-block shapes; comparison side M = 65), gemma4 greedy parity and chat golden, and the other
  families' sha256 regression checks all pass. Fault injection (nibble order swapped on the row-block
  side only) fails only the u32 check, by 1 ULP.
- **The cap of 256 elements per word is a compile-cost decision**: generated text grows with the row
  count and naga's parse and validation cost grows super-linearly with text (i4 16 rows = 202 KB /
  733 ms; i8 32 rows = 165 KB / 315 ms). Lowering the run-time optimum (512) to 256 and hoisting the
  dequantization cut the five shaders of gemma4's first turn from ≈1.3 s to ≈150 ms (cost:
  census-weighted M=32 +4%, lm_head +1.6 ms/run). Folding the row loop into a WGSL `for` was 2–7×
  slower (private arrays spill to local memory). The value 256 SHOULD be kept as a provisional value
  until it is re-swept on Dawn / Tint (re-graded 2026-10-18 to SHOULD: it was measured with naga, which
  only the secondary Deno path uses now; Tint parse cost was not measured). Changing it MUST come with
  a re-measured sweep of run time and compile time (`ROWS_ELEMENTS_PER_WORD_CAP`,
  `src/kernels/linear-gemv.ts`). The codegen check bounds the generated text (below 12,000 characters
  for every default height since addendum 9, `tests/linear_gemv_rows_test.ts`).
- Gain (census-weighted, same submit): M=2 ×7.0 / M=8 ×6.0 / M=32 ×3.1 / M=64 ×2.6 (i4); M=32
  ×1.7–2.2 (i8). Over the ×1.3 kill criterion everywhere. Real-graph numbers: research §8 /
  perf-ledger K-21.
- Open: row blocks unmeasured on Metal (expected within 1 ULP — `docs/known-issues.md`); M > 64 stays
  on the GEMM skeleton (M64N32 and up).

## Addendum 6 (2026-09-11) — f16 storage at M=1 joins the GEMV family (K-25)

- Added as a storage variant after the maintainer approved the f16 GEMV work. Public API, IR,
  distribution format and stored weights unchanged. One weight word is the same `vec4<u32>` carrying 8
  elements; the low and high halves of `unpack2x16float` are multiplied and added in ascending K. No
  scale binding.
- Eligibility: `m === 1`, f16 storage, f32 compute, `v4`, `k > 0`, `k % 8 === 0`. f16 row blocks are
  rejected at generation; K=0 stays on the GEMM path (zero-size dimensions, `docs/limitations.md`).
- The variant stays `c32u4`; `c32u16` wins the weighted kernel figure but loses on the vocabulary
  head, and Qwen's overall result depends on input. No per-shape option.
- Reduction order, restoration and bias follow decision 3; i4 / i8 WGSL byte-unchanged. Added: f16
  snapshot, eligibility tests, u32 equality with the GEMM path's leading row, independent CPU
  reference; reversing only the multiply-add order fails the u32 check.
- Accepted on RTX 3080 Ti / Deno (kernel, Qwen3-0.6B f16) and Chrome (Qwen); numbers in
  [research 2026-09-10](../research/2026-09-10-codex-mtp-optimization.md) (f16 GEMV section). Apple M2
  not verified; RTX bit equality and gains are not claimed for other backends.

## Addendum 7 (2026-09-11) — f32 storage at M=1 joins the GEMV family (K-26)

- Prototyped after f16 with existing f32 assets of Qwen3-0.6B / MiniCPM5-2B: all logits bit-equal to
  the GEMM path, faster generation (research 2026-09-10, f32 GEMV section).
- One weight word is `vec4<f32>` (4 elements); ascending K, per-component multiply-add and final bias
  kept; no scale binding. f16 / i4 / i8 WGSL byte-unchanged.
- Eligibility: M=1, f32 compute, K>0, K%4=0, N%4=0. Row blocks and f16 compute stay separate.
  Variant `c32u4`. No public API, IR or asset format added.
- Permanent checks: 4-element words and read-ahead edges, N edges, each eligibility condition, u32
  equality with the GEMM path's leading row, CPU reference. The f32 control in the f16 test now
  expects the new f32 GEMV key (scope widened; tolerances unchanged). M2 not verified.

## Addendum 8 (2026-09-11) — smaller INT2 row-block shaders (K-30)

- To shorten QAT's first preparation, only the INT2 row block moves the 64-element-word multiply-add
  into a WGSL function, keeping per-row ascending K, static component indices, the dequantization and
  multiply-add expressions and the final bias. Row count, variant selection, eligibility, public API,
  IR and stored assets unchanged.
- Applying the same function to M=1 was rejected (slightly slower after warm-up on Deno, slower first
  run on Chrome). M=1, other storage types and GEMM WGSL and keys are unchanged; only the changed INT2
  row block uses the key `linear_gemv:v2:...:wi2`.
- The default r4 WGSL shrinks from 76,161 to 9,169 characters (no dynamic row loop or array). The
  in-function dequantization runs per row, so shorter source does not imply faster GPU execution.
  Accepted by u32 comparison on 54 real-weight conditions, comparison with M=1 on long K and edge rows,
  snapshots and full verification.
- Measured (research 2026-09-10, INT2 row-block section): QAT E2B's first response ≈21% shorter on
  Deno, 15–16% on Chrome; warm kernels unchanged (not a warm-speed gain). M2 and E4B not measured.

## Addendum 9 (2026-09-11) — the function form for INT4 and INT8 row blocks too

- The same first-parse reduction is applied to INT4 and INT8 row blocks: a WGSL function generated
  from the M=1 in-word multiply-add receives per-row accumulators, keeping ascending K, the f32
  dequantization multiply, static indices and the final bias. The expression layout differs from the
  old source (which shared the dequantization across rows), so numerical equality is checked by u32
  equality with the GEMM path and the CPU reference, not by source comparison.
- The changed row-block keys move to v2; M=1 and the INT2 row block keep WGSL and keys. Public API,
  IR, assets, variant selection and numerics contract unchanged.
- At the default maximum height INT4 shrinks 72,592 → 7,835 characters, INT8 70,318 → 10,212. A check
  keeps every default height under 12,000 characters; K=4096 and partial-tile numerical tests added.
- Measured (research 2026-09-10, INT4 / INT8 row-block section): shorter first runs on Deno (normal
  and QAT, empty and reused cache); Chrome QAT slightly worse on the reversed empty-cache comparison,
  better on reuse. Not a warm-speed change. M2 not accepted.

## Addendum 10 (2026-09-12) — large-vocabulary INT8 at M=1 selects c16

- `defaultLinearGemvVariant` takes an optional shape and returns `c16u4` only for INT8, N=262144,
  K=1536; the shape is passed only when building the M=1 recipe. Codegen without a shape and row
  blocks keep `c32u4`.
- Applied to the vocabulary output of standard Gemma 4 E2B, based on u32 equality of every output
  with real weights and Chrome ABBA / BAAB comparisons (4 pairs each)
  ([follow-up](../research/2026-09-12-chrome-gemv-followup.md)).
- WGSL, addition order, dequantization and bias unchanged; c16 uses the existing key format. Public
  API, IR, storage format and resident memory unchanged. Not widened to unmeasured shapes or storage
  types. Existing GPU, CPU-reference and Gemma checks kept. M2 not accepted.
