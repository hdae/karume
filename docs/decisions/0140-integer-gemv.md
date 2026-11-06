# 0140: Integer GEMV for the QAT decode linears — exact integer sums per group (`integerGemv`)

- Status: accepted (2026-11-06 — the maintainer; proposed 2026-11-03 — implemented and measured on the maintainer's Apple M2
  under Chrome; its GPU tests pass on the development machine's RTX 3080 Ti; opt-in, declared by no preset)
- Date: 2026-11-03
- Related: ADR [0105](0105-packed-static-quantize-activations.md) (packed int8 activations — the only input form this
  kernel reads) / [0098](0098-linear-gemv-parallel.md) (the parallel GEMV it replaces on its shapes; the measured-shape
  rule) / [0103](0103-linear-static-quantize-fusion.md) (linear→SRQ fusion — the SRQ epilogue) /
  [0110](0110-practical-tier-numerics-contract.md) (contract classes — this kernel is E; §6 in-device determinism; §7
  sha rows; §8 quality verdict) / [0058](0058-numerics-opt-in-contract.md) (one named option per numerics change) /
  [0111](0111-session-options-composition.md) (option composition, family acceptance tables) /
  [0097](0097-gemma4-qat-integration.md) (QAT, fixed SRQ) / [0025](0025-w8a8-linear-execution.md),
  [0076](0076-w4a8-linear-execution.md) (the integer dot-product paths) / perf-ledger K-45 / research
  [2026-11-03-qat-integer-gemv-m2](../research/2026-11-03-qat-integer-gemv-m2.md) (every number below).

## Context

On the maintainer's M2 (Chrome), Karume's QAT Gemma 4 E2B (`i4-fast`, greedy) decoded at 36.7 / 38.1 tok/s while the
WebML Community "Gemma 4 WebGPU kernels" Space decoded the same checkpoint at 46.1–50.8 tok/s (research §3). A split
attribution of Karume's decode put the linears at 80.7% of GPU time (24.8 ms per token; research §4), and the M2 GEMV
turned out to be bound by the per-element weight decode, not by memory: the same loop without the decode streams at 79
GB/s, with the shift-mask decode at 26–37 GB/s, against a read roofline of about 83 GB/s (research §5). On the RTX
3080 Ti the bound had been the activation loads (K-45 step 0, [research 2026-09-19](../research/2026-09-19-qat-speed-recon.md)
§14), which ADR 0105's packed activations addressed; packed activations cut loads, not decode work.

QAT E2B's decode linears already receive their activation as the int8 codes of a fixed SRQ (ADR 0097), packed four per
word under `packedStaticQuantize` (ADR 0105), and their weights are integer codes (i2 / i4 / i8). The product of the two
code vectors is an integer, so a GEMV can sum it exactly and apply the scales once — the form K-45 was filed for. The
WebML kernels show two techniques that cut the decode work: an `unpack4x8unorm` decode of byte planes and row blocking.
Their numerics (`code / 255` floats without re-rounding, a last-arriver merge with float atomics) are not admissible
here (Alternatives).

## Decision

### 1. Exact integer sums per group; the scales in a fixed epilogue

For output `n`, with `q` the signed weight value (stored code − offset) and `c` the activation code in [−128, 127]:

```
S[n,g] = Σ_{k ∈ g} q[n,k] · c[k]                                   (exact)
out[n] = ((Σ_g round(S[n,g] · w_scale[n,g])) · x_scale) + bias[n]   (then the optional SRQ)
```

A group is the whole row for i2 (offset 2, one scale per output channel) and i8 (symmetric, one scale per channel), and
one weight group for i4 (offset 8, one scale per channel and group). The epilogue rounds `S` to f32, each group's
product with its scale, each partial sum over the groups in **ascending group order**, the `x_scale` multiply and the
bias add — each once, as separate IEEE operations, never a fused multiply-add. The output SRQ, when the linear→SRQ
fusion takes the window (ADR 0103), runs as the kernel's epilogue.

### 2. Exactness bounds: lane sums in f32 at or below 2^24, then an i32 tree

A lane accumulates its share of one group in f32, which holds every integer up to 2^24 exactly; the lane partials meet
in an i32 workgroup tree. Both are exact in any order, so `S` does not depend on the work assignment.

- MUST: the lane bound (`integerGemvLaneBound` — steps per lane × 16 × the largest weight factor × 128) stays at or
  below 2^24 for every generated shape, and every group sum fits i32; the params builder refuses a shape past either.
- MUST: `S` reaches f32 as `f32(S >> 12) · 4096 + f32(S & 0xfff)`. Both parts and the product are exact, so the one
  add is the only rounding and is correctly rounded, as the mirror's `Math.fround`. A bare `f32(S)` would leave the
  rounding of a sum past 2^24 (i8 at K = 1536) to the implementation.

### 3. The decode: byte planes and a per-step round (i2 / i4); `unpack4xI8` (i8)

One step covers 16 elements: one activation entry (`vec4<u32>`, 16 codes) and the matching weight unit (i2 `u32`, i4
`vec2<u32>`, i8 `vec4<u32>`). For i2 / i4 the unit is split into four byte planes (masks `0x03030303` / `0x0f0f0f0f`
after shifts), each turned into floats by one `unpack4x8unorm` (code / 255); the four plane dot products with the
regrouped activation planes are summed and turned back into the exact integer by `round(255 · s)`.

- MUST: the rounding is per step (16 elements). There `|s|` stays at about 121 (i4) or 24 (i2), so the f32 error of the
  dot products and of `unpack4x8unorm` stays below 0.05 after the scaling by 255, far inside the 0.5 that `round`
  absorbs, whatever the device's unpack accuracy. Rounding a longer sum would let the error grow with K.
- i8: `vec4<f32>(unpack4xI8(word))` is already the exact integers and each dot product (≤ 2^16) is exact.
- The zero point is folded: a lane accumulates `Σ u·c` per row and group (u = the stored code) and one `Σ c` per group
  shared by the rows; the epilogue forms `Σ u·c − offset · Σ c` in i32.

### 4. Row blocking

A 128-thread workgroup is `128 / lanes` lane groups; a lane group owns `rows` consecutive output channels, and each lane
reads an activation entry once and adds its dot product with each of the `rows` weight units. Decision 2 is what makes
this free of an addition-order contract: any (lanes, rows) gives the same `S`. The (lanes, rows) pair, the storage and
the i4 group shape are in the pipeline key (`linear_gemv_integer:v2:<wi2 | wi4g<G>x<groups> | wi8>:l<L>r<R>`, with
`:static-quantize:v2` for the SRQ epilogue).

### 5. Contract class E — the TS mirror, the rounding barrier

The kernel is mirror class E (ADR 0110 decision 2): every output word equals the CPU reference
(`src/reference/linear-gemv-integer.ts`), which adds plain JS integers in ascending k per group and applies each
epilogue step as a `Math.fround` of one IEEE operation. MUST: the reference copies none of the kernel's lane / row /
plane / group-loop structure, so the work assignment and the decode stay under verification.

- The epilogue's roundings are separated by a run-time rounding barrier, `bitcast<f32>(bitcast<u32>(v) ^
  rounding_mask)` with a uniform `rounding_mask` that is always 0, so no compiler contracts two of them into a
  multiply-add or regroups the products (Metal contracts per expression form — ADR 0105 addendum 4). ADR 0105 fixed its
  divergence with an explicit `fma()`; here the mirror rounds the multiply and the add separately, so the barrier is the
  spelling. That it holds is measured per backend (the M2: 692 / 692 — research §7), not a guarantee of the WGSL spec.
- MUST: the group sum starts with the first group's term. `rounded(0.0 + rounded(t))` simplifies to two adjacent XORs
  that cancel, and Metal then reassociated `(S · w_scale) · x_scale` into `S · (w_scale · x_scale)` — 1-ulp mismatches
  on every f32-output case on the M2 (research §9).
- In-device determinism (ADR 0110 §6) holds by construction: no atomics, no arrival-order merge, a fixed-shape tree.
- The check (`packages/kernel-checks/src/gemv-integer.ts`, shared by the GPU test and the playground's `/kernel-check`
  page) runs 692 cases: i2 / i4 / i8, every (lanes, rows) on small K and the corners on the widths in use, M = 1 / 8,
  with and without the SRQ epilogue, saturated fixtures at each storage's largest K (lane sums closest to 2^24; the i8
  one past 2^24 and 3 mod 4, so a truncating conversion rounds the tie the wrong way), and rows past M left unwritten.

### 6. Measured shapes only

A linear is eligible only at a shape in `INTEGER_GEMV_SHAPES` (src/kernels/linear-gemv-integer.ts), each with its
measured (lanes, rows) — the parallel GEMV's rule (ADR 0098 decision 4), held strictly here because the kernel lost to
the parallel GEMV on five of the twelve measured shapes. M = 1..8 as for the parallel GEMV, so the speculative verify rows run
the decode kernel (ADR 0098 decision 2).

| Storage | Linear (QAT E2B) | n × k | Group | Lanes × rows | Parallel GEMV → integer, single kernel (M2, µs) |
| --- | --- | --- | --- | --- | --- |
| i2 | gate / up, the 20 KV-shared layers | 12288 × 1536 | row | 8 × 4 | 165 → 131 (median, stage 1) |
| i2 | down, the 20 KV-shared layers | 1536 × 12288 | row | 16 × 4 | 292 → 102 (median, stage 1) |
| i4 | o, sliding layers | 1536 × 2048 | 2048 × 1 | 16 × 4 | 46.2 → 24.7 |
| i4 | o, full layers | 1536 × 4096 | 4096 × 1 | 16 × 2 | 82.7 → 45.9 |
| i4 | down, the own-KV layers | 1536 × 6144 | 2048 × 3 | 8 × 2 | 168.5 → 73.4 |

The K = 1536 i4 shapes (q / k / v / gate / up) and the i8 PLE shapes are not eligible (Alternatives).

### 7. The opt-in option `integerGemv`, which requires `packedStaticQuantize`

`SessionOptions.integerGemv?: boolean`, default false — with the default, no byte of any path changes (ADR 0058). It
reads only packed int8 activations, so `integerGemv: true` without `packedStaticQuantize: true` (itself limited to
`linearGemvReduce: "parallel"` / `linearCompute: "f32"`) is rejected at construction (`sessionOptionsViolation`) rather
than accepted as a no-op. Under the option an eligible linear counts as a packed consumer, so its producer SRQ is packed
even where the parallel GEMV's table does not pack that shape (the i2 gate / up). The hub's `SessionSpec` accepts the
key; the Gemma acceptance table (`packages/models/src/gemma/session-options.ts`) lists it; `GemmaPipeline` passes it to
target and drafter in the order explicit → the quant's `session` → runtime default (ADR 0111). No preset declares it.

### 8. One eligibility function

`linearIntegerGemvVariant` (src/runtime/linear-route.ts) is the only function that decides whether a linear is an
integer GEMV linear and with which variant: the option is on, the weight is i2- / i4- / i8-resident, M is 1..8 and
`integerGemvVariantFor` names a measured shape. The packed pairing (`readsPackedActivation` in fusion.ts), the
linear→SRQ rule (`fusion-rules/linear-static-quantize.ts`) and the recipe builder (`buildLinear` → `buildLinearIntegerGemv`)
call it. MUST: none of them decides on its own — the kernel binds the activation as `vec4<u32>` codes, so a linear
judged eligible by one side and not the other would receive f32 words: a silent wrong value. The kernel runs only when
the activation does arrive packed; otherwise the linear keeps its path, and a linear declared packed that lands on
neither the packed parallel GEMV nor the integer GEMV fails loudly in the recipe builder. Whether the integer kernel
takes a linear→SRQ window enters no acceptance condition of the rule, only the dispatch it builds.

## Results (2026-11-03, Apple M2, Chrome 158, QAT E2B `i4-fast`, greedy, tap off)

- Kernel check: 692 / 692 cases equal the mirror. Fault injection: i2 no per-step round, no zero point and swapped
  planes; i4 no round, even / odd swap, offset 7, group scale and reversed group order; i8 lane swap and a truncating
  integer-to-f32 conversion — all caught. A bare `f32(S)` is not caught on the M2, which converts round-to-nearest-even
  (research §6–§7).
- End to end, i2 + i4: 27.767 → 22.754 ms/token (Story, ×0.820) and 27.835 → 22.768 ms/token (Explainer, ×0.818);
  outputs identical across rounds within each arm. By the A / B / B / A attribution the i4 shapes contribute 0.93 /
  1.68 ms of the gap, the i2 shapes the rest (mean deltas: i2 alone 4.143 / 3.263 ms, i2 + i4 5.075 / 4.939 ms —
  research §8). The i2-only stage measured 38.4 → 43.8 and 38.0 → 44.4 tok/s (research §6).
- The generated text differs from the option off (stage 1: an English generation diverged at its 111th character; a
  Japanese 160-token one was identical), as the different rounding sequence implies.

## Consequences

- One more Session option, one more kernel family and key prefix, and one more manifest key. The reference path and
  every existing preset are unchanged; the playground's `/gemma4-bench` passes the option explicitly for its "Integer
  GEMV" arm.
- With the option, QAT E2B's output changes relative to the parallel GEMV. A preset that declares it needs the quality
  verdict of ADR 0110 §8 first, and its sha rows are practical-preset rows (ADR 0110 §7).
- Only QAT E2B has linears at the measured shapes. Regular Gemma 4 (i4 with groups of 32) and QAT E4B (other widths)
  are not affected beyond what their packed activations change.
- A reader whose hub vocabulary lacks the key refuses a manifest that declares it (the closed `session` vocabulary —
  ADR 0098, adoption section), so a distribution can declare it only for a release that has the key.

## Alternatives considered

- **`code / 255` floats without re-rounding** (the WebML kernels' decode) — rejected: an approximation whose error grows
  with K, contract class Q, which ADR 0110 admits only with all four points and an error model. The per-step `round(255
  · s)` (decision 3) keeps the same decode instructions and makes the result exact, class E.
- **Last-arriver workgroup merge and float atomics** (the WebML kernels' K split) — rejected: ADR 0110 §6 forbids both.
  The i32 tree inside one workgroup is enough at these widths.
- **GPU argmax run-ahead** (the WebML kernels) — out of scope: it serves greedy decoding only and is not a GEMV change.
- **The i4 shapes at K = 1536 and the i8 PLE shapes** — measured slower or not faster on the M2 (q 2048 27.1 → 29.7 µs,
  k/v 256 9.2 → 10.7, gate / up 6144 78.1 → 84.1, PLE gate 8.6 → 8.8, PLE projection 7.4 → 8.0 — research §7). k/v 512
  (15.7 → 13.7) and q 4096 (87.0 / 47.8 → 54.0, the parallel cell's median and min ×1.8 apart) were the exceptions,
  but **partial adoption is impossible there**: q / k / v read one SRQ per layer, as gate / up do, and the producer is
  packed only when every consumer reads packed (fusion.ts condition 4). The parallel GEMV does not read packed
  activations at K = 1536 (packing those shapes was measured not to help — ADR 0105 addendum 1), so k/v 512 alone on the
  integer GEMV would leave the producer f32 and the linear on its old path.
- **The `dot4I8Packed` route of K-45's plan** (route A: unpack each weight element, subtract the zero point, repack as
  signed bytes, then `dot4I8Packed`; route B: an unsigned dot product with the activations biased by 0x80 and two
  correction terms) — not taken. On the M2 the GEMV is bound by the per-element weight decode (Context); route A adds
  per-element work to exactly that part, and route B adds per-element activation work and epilogue terms, while
  `unpack4x8unorm` turns a whole byte plane into floats in one instruction and the zero point leaves the loop through
  `Σ c`. The float dot product with a per-step round is exact by decision 3, so the integer instruction buys no
  numerics either. The two routes were not measured on the M2. For i8 both sides are signed and `dot4I8Packed` would
  apply directly; the i8 shapes were not adopted (above), so that variant was not pursued.

## Open points

- The sha256 rows of ADR 0110 §7 are not recorded.
- Speed on backends other than the M2's Metal (the RTX under Dawn, D3D12) — not measured; the shapes and geometries are
  the M2's.
- The i4 shapes' end-to-end gain (0.93 / 1.68 ms) is below the single-kernel estimate (2.3 ms); not investigated.
- Declaring it in a preset (`i4-fast` of QAT E2B) — needs the quality verdict of ADR 0110 §8; not ruled.
