# 0141: Byte-plane GEMV for the QAT lm_head — i2 weights × f32 activations, an exact decode, class R (`linearGemvPlanes`)

- Status: proposed (2026-11-03 — implemented and measured on the maintainer's Apple M2 under Chrome; the GPU tests on
  the development machine's RTX 3080 Ti are pending — that GPU was busy; opt-in, declared by no preset; adopting it is
  the maintainer's decision)
- Date: 2026-11-03
- Related: ADR [0098](0098-linear-gemv-parallel.md) (the parallel GEMV — slower here; the measured-shape rule) /
  [0110](0110-practical-tier-numerics-contract.md) (contract classes — this kernel is R; §3 the four-point check; §6
  in-device determinism; §7 sha rows; §8 quality verdict) / [0058](0058-numerics-opt-in-contract.md) (one named option per
  numerics change) / [0111](0111-session-options-composition.md) (option composition, family acceptance tables) /
  [0140](0140-integer-gemv.md) (the integer GEMV — the byte-plane decode it shares, the decode-bound M2 GEMV) /
  [0105](0105-packed-static-quantize-activations.md) (packed int8 activations — never read here) /
  [0103](0103-linear-static-quantize-fusion.md) (linear→SRQ fusion — never taken here) / perf-ledger K-52 / research
  [2026-11-03-qat-integer-gemv-m2](../research/2026-11-03-qat-integer-gemv-m2.md) §10 (every number below).

## Context

QAT Gemma 4 E2B's lm_head is an i2 linear of 262144 × 1536 (one scale per output channel) that runs once per decode
token on the sequential GEMV (`linear_gemv:v1:f32:c32u4:wi2`). In the M2 split attribution it took 2.917 ms per token at
35 GB/s (research §4). It has no SRQ before or after it (perf-ledger K-52), so its activation is f32: neither the packed
activations of ADR 0105 nor the integer GEMV of ADR 0140 apply. K-52 had rejected parallelising it on the RTX 3080 Ti
(lanes 2 to 32 all slower than the sequential GEMV — [research 2026-09-12](../research/2026-09-12-chrome-gemv-parallel.md)).

A probe on the M2 (research §10) found the sequential GEMV bound first by its access pattern, then by the decode:

- The sequential GEMV gives every thread one column, so a SIMD group's 32 lanes read 32 rows 384 B apart. Without the
  decode that layout still stays at 46–53 GB/s; with it, 35 GB/s.
- A layout where `lanes` threads share a column and a lane group holds `rows` columns reaches 87 GB/s without the decode
  (L16R4) — the read roofline. From there the decode instructions bound the kernel.
- The parallel GEMV (ADR 0098) is slower than the sequential GEMV at every lane count it can dispatch here.

The probe also found that **the sequential GEMV is not order-stable on the M2**: its unroll 2 and 8 variants differ in
u32 from unroll 4, because Metal's fast-math reassociates or contracts `acc + x · d`. Its words are therefore no fixed
reference to stay bit-identical with.

## Decision

### 1. Layout and work assignment

A 128-thread workgroup is `128 / lanes` lane groups; a lane group holds `rows` consecutive output channels. The `lanes`
threads of a group share each channel's K and walk its u32 words (16 i2 codes each) as `lane, lane + lanes, …`, so a
lane group reads a row contiguously; each lane reads an activation entry (four `vec4<f32>`, 16 elements) once for all
`rows` channels. The lane partials meet in a workgroup tree, and lane `j < rows` of a group stores channel `j`. The grid
is `(workgroups(n), m)`, M = 1..8 as for the parallel GEMV (ADR 0098 decision 2). Channels past `n` read the last
channel's words (in range) and are never stored. The key is `linear_gemv_planes:v1:wi2:l<L>r<R>`; the bindings keep the
linear's numbering (1 x / 2 w / 3 bias / 4 out / 5 the per-channel scale).

### 2. The decode: byte planes and an exact magic-number conversion

A word `w` is split into four byte planes, `(w >> 2p) & 0x03030303` (plane `p` holds elements `4b + p` in its bytes
`b`). `unpack4x8unorm` turns a plane into `u / 255`, and

```
fma(255, u / 255, 12582912) − 12582914        (12582912 = 1.5 · 2^23)
```

gives the signed weight value `u − 2` exactly. Near 1.5 · 2^23 consecutive f32 values are 1 apart, so adding the
constant rounds `255 · (u / 255)` (within 1e-6 of `u`) to the integer `u` — **whether or not the multiply is fused with
the add**: fused, the exact product is rounded once; unfused, the rounded product is still within 0.5 of `u` before the
add rounds. The subtraction of two integers below 2^24 is exact. So the decode needs no rounding barrier and gives the
same words on a compiler that contracts and on one that does not. The four plane dot products with the regrouped
activations (component `p` of each activation quad) are added to the lane's accumulator.

- NOTE: a compiler that folds the two constants (`fma(a, b, c) − d` → `fma(a, b, c − d)`) would decode code 2 as about
  1.2e-7 instead of 0 — the inexact `255 · unorm − 2` of the alternatives. The M2's Metal does not (the kernel is
  u32-equal to a `round` decode — research §10); on another backend the class R check (decision 4) bounds the result
  either way.

### 3. Contract class R — the addition order and the scale applied once

```
out[n] = (Σ_k q[n,k] · x[k]) · w_scale[n] + bias[n]        q = stored code − 2 ∈ {−2, −1, 0, 1}
```

Because `q` is in −2..1, every product `q · x` is exact in f32. The kernel's roundings are the additions (each lane's
running sum, the dot products, the four planes, the workgroup tree), one multiply by the channel scale and the bias add.
The reference op multiplies per element by `q · scale` and adds in ascending k. The kernel differs from it in two
points:

- the addition order — what class R of ADR 0110 admits;
- **the scale applied once after the sum**, `(Σ q · x) · s` in place of `Σ (q · s) · x` — a distributive regrouping. This
  is the one point beyond the addition order, and class R's definition ("only the addition order differs") does not
  name it. It stays inside the reference op band because the products `q · x` are exact: the kernel's sum carries only
  addition roundings, and the one multiply by the scale adds a rounding at most as large as the reference's per-element
  roundings of `(q · s) · x`. The claim checked is the reference op band itself (`GEMM_TOLERANCE`, atol 2e-4 + rtol
  4e-6) against an f64 evaluation — not a wider band, so the kernel is not class Q.

Why not E (ADR 0110 decision 2 asks why a stronger class cannot be built): the activations are f32, so the sum is not
exact and depends on the order of its additions, and WGSL leaves to the implementation the order inside a `dot` and
whether `dot` / `fma` fuse. A TS mirror would have to pin words the specification does not fix.

In-device determinism (ADR 0110 §6) holds by construction: no atomics, no arrival-order merge, a fixed-shape tree; within
one compiled pipeline the order is fixed, so two dispatches of the same input are bit-identical (point 3 of the check).

### 4. The four-point check (ADR 0110 §3)

`checkPlanesGemv` (`packages/kernel-checks/src/gemv-planes.ts`, shared by the GPU test and the playground's
`/kernel-check` page) runs 240 cases: the 15 (lanes, rows) variants × K = 16 / 80 / 1536 with random codes and
activations in [−2, 2), plus a saturated fixture at K = 1536 (every code 0 = value −2 against activations of one sign, so
no term cancels), each at 37 and 1029 columns (multiples of no workgroup's column count) and M = 1 / 8.

1. The f64 band: every output within `GEMM_TOLERANCE` of an f64 evaluation of `(Σ_k q · x) · scale + bias`.
2. The A/B band: where the sequential GEMV runs (M = 1, K = 1536), every output within twice that band of the sequential
   GEMV's output on the same input (the theoretical bound of ADR 0110 §4).
3. Structure: a second dispatch writes the same words; nothing past the M × n outputs is written; a NaN at one element
   of a row reaches every output of that row.
4. Fault injection: `packages/runtime/tests/gpu_linear_gemv_planes_test.ts` runs the check on two WGSL mutants and
   asserts that the f64 band fails — the decode off by a constant (on every case) and two planes' activations swapped
   (on every mixed case; the saturated fixture holds one weight value, so the swap cannot change it). The page does not
   run the mutants.

The same GPU test runs a Session at the measured shape: with the option on, the byte-plane key runs and the sequential
GEMV's does not, every output within the f64 band; with it off, the sequential GEMV runs and no byte-plane key does
(the census, ADR 0058 decision 4 ③).

### 5. Measured shapes only

A linear is eligible only at a shape in `PLANES_GEMV_SHAPES` (src/kernels/linear-gemv-planes.ts), with its measured
(lanes, rows) — ADR 0098 decision 4's rule:

| Storage | Linear | n × k | Scale | Lanes × rows | Sequential GEMV → byte-plane, single kernel (M2, µs, median / min) |
| --- | --- | --- | --- | --- | --- |
| i2 | lm_head, QAT E2B | 262144 × 1536 | per channel | 8 × 4 | 2987 / 2425 → 2393 / 1845 |

The workgroup count at L8R4 is n / 64 = 4096; L32R1 would need 65536 workgroups, past the 65535 limit.

### 6. The opt-in option `linearGemvPlanes`

`SessionOptions.linearGemvPlanes?: boolean`, default false — with the default no byte of any path changes (ADR 0058).
It needs no other option. The hub's `SessionSpec` accepts the key (`SESSION_KEYS`); the Gemma acceptance table
(`packages/models/src/gemma/session-options.ts`) lists it and no other family's table accepts it; `GemmaPipeline`
passes it to target and drafter in the order explicit → the quant's `session` → runtime default (ADR 0111). No preset
declares it. The playground's `/gemma4-bench` has a per-arm "Planes GEMV" switch that passes it explicitly.

### 7. One eligibility function; packed activations and the linear→SRQ rule

`linearPlanesGemvVariant` (src/runtime/linear-route.ts) is the only function that decides whether a linear is a
byte-plane GEMV linear and with which variant: the option is on, the weight is i2-resident, M is 1..8 and
`planesGemvVariantFor` names a measured shape. Three places read it:

- The packed pairing (`readsPackedActivation` in fusion.ts) returns "not packed" for such a linear, checked before the
  integer GEMV's and the parallel GEMV's tables, so the option decides the linear's route on its own: its kernel binds
  f32 quads only.
- The linear→SRQ rule (`fusion-rules/linear-static-quantize.ts`) does not take the window of such a linear: taking it
  would build the parallel GEMV in its place and leave the option silently without effect. The decision reads only the
  option and the linear, never the packed values, so both scans of the fusion plan grab the same windows. No measured
  parallel GEMV row is a byte-plane shape today, so this rejects no window the rule would otherwise take.
- The recipe builder (`buildLinear` → `buildLinearPlanesGemv`) takes the branch in place of the i2 GEMV family (the
  M = 1 sequential GEMV or the row-block variant), and fails loudly if such a linear is declared to read packed
  activations (f32 quads would read int8 words — a silent wrong value otherwise).

MUST: none of them decides on its own. The lm_head is in no parallel GEMV, packed or integer GEMV table, so the two
opt-in GEMVs never compete for a linear.

## Results (2026-11-03, Apple M2, Chrome)

- Kernel check: 240 / 240 cases pass (the f64 band, the A/B band against the sequential GEMV, two dispatches identical,
  nothing written past the outputs, NaN propagation).
- Fault injection: the decode off by a constant turned the f64 band red on 240 / 240 cases; two planes' activations
  swapped turned it red on all 180 mixed cases.
- Single kernel: the runtime kernel (`planesGemvWgsl`, L8R4) is u32-equal to the probe's L8R4 magic-decode kernel and
  timed 2393 / 1845 µs (median / min) against 2987 / 2425 µs for the sequential c32u4 in the same run; its largest error against f64 was 0.17% of the GEMM band.
  The probe's sweep (21 rounds) had 2383 / 1837 against 2973 / 2420.
- End to end (QAT E2B, greedy, tap off, 3 rounds; arms: integer GEMV only / integer GEMV + byte-plane GEMV):
  - No measurable change: 22.822 → 22.835 ms/token (Story, ×1.001) and 22.781 → 22.766 (Explainer, ×0.999), warm-chat
    medians; outputs identical across rounds within each arm.
  - In the model the kernel does save GPU time: split attribution (one round, without the integer GEMV — the split mode
    fails intermittently, see known issues) put the lm_head at 3.197 / 3.163 ms per token on the sequential GEMV and
    2.498 ms on the byte-plane GEMV (−0.68 ms).
  - Reading (inferred, not measured): with the integer GEMV on, the decode's wall time on the M2 is no longer bound by
    GPU time but by the host side of a token (about 1,036 dispatches recorded and submitted, the readback); the i4
    shapes' shortfall against their single-kernel estimate (ADR 0140) fits the same floor. A GPU-time saving shows up
    end to end only once the host side shrinks (run-ahead, fewer dispatches).

## Consequences

- One more Session option, one more kernel family and key prefix, and one more manifest key. The reference path and
  every existing preset are unchanged; with the option off the lm_head keeps the sequential GEMV.
- With the option, QAT E2B's output may change relative to the sequential GEMV (class R): a greedy run may decode other
  tokens. A preset that declares it needs the quality verdict of ADR 0110 §8 first, and its sha rows are practical-preset
  rows (ADR 0110 §7).
- Only the QAT graphs have an i2 linear at the measured shape; regular Gemma 4 is not affected.
- A reader whose hub vocabulary lacks the key refuses a manifest that declares it (the closed `session` vocabulary), so
  a distribution can declare it only for a release that has the key.
- The probe module that produced the layout and decode comparisons was a measurement aid and is not in the repository;
  its numbers live in the research record.

## Alternatives considered

- **The parallel GEMV (ADR 0098)** — rejected: slower than the sequential GEMV on the M2 at every lane count (l2 2924,
  l4 3060, l8 3374, l16 5544 µs against 2887), and l32 exceeds the 65535-workgroup limit at n = 262144.
- **Other exact decodes** (`round(255 · unorm) − 2`, shift / and / convert, `unpack4xU8`, bytes) — not taken: all
  about 2.8 ms median / 2.3 ms min on the lanes × rows layout (L8R4 and L8R8 alike), no faster than the sequential GEMV. Folding the offset
  out of the loop (`− 2 · Σ x`) did not help either.
- **`255 · unorm − 2` without a round** — rejected: the fastest spelling (L8R8 2276 / 1729 µs), but not exact on the M2:
  Metal fuses it into an `fma`, which decodes code 2 as 1.18e-7 (3145 of 4099 words differ from the exact decode). The
  error would no longer be the addition order's, so the kernel would be class Q, which ADR 0110 admits only with an
  error model.
- **WebML-style accumulation of `unorm` values with the offset folded** (sum `(u / 255) · x`, scale by 255 and subtract
  `2 · Σ x` at the end) — rejected: no faster, and the per-element products are no longer exact, so it is class Q.
- **A pair-table decode** (16-entry tables of element pairs in workgroup memory; the products exact since values are
  −2..1) — rejected: 2.3–5.4 ms, bound by workgroup memory.
- **Keeping bit identity with the sequential GEMV** — not pursued: that kernel is not order-stable on the M2 (its unroll
  variants differ in u32 under Metal's fast-math), so its words are no fixed reference; class R's f64 band is the
  contract instead.

## Open points

- The GPU tests on the RTX 3080 Ti (`packages/runtime/tests/gpu_linear_gemv_planes_test.ts`: the shared check, the two
  mutants and the Session census) — what keeps this ADR proposed. Speed on the RTX is not measured either: K-52 found
  lm_head parallelisation slower there, and the shape table is the M2's.
- QAT E4B's lm_head shape was not measured, so it is not eligible.
- The magic decode might also replace the integer GEMV's per-step `round(255 · s)` (ADR 0140 decision 3) — not measured.
- Declaring it in a preset — needs the quality verdict of ADR 0110 §8 and the end-to-end result; not ruled.
