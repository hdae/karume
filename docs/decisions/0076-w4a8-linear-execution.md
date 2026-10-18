# 0076: w4a8 — integer dot-product path for i4-resident weights

- Status: accepted (2026-08-21, the maintainer's ruling; implemented and measured the same day);
  decision 6 (anima's i4 presets do not use the path) re-graded to SHOULD NOT on 2026-10-18
- Date: 2026-08-21
- Related: [0025](0025-w8a8-linear-execution.md) (w8a8, the parent; the predicate is its 2026-08-21
  addendum) / [0069](0069-packed-w4-storage.md) (i4 storage; decision 5's scale application revised
  by its addendum 8) / [0058](0058-numerics-opt-in-contract.md) (numerics opt-ins; storage is a
  separate axis) / [0074](0074-quant-seat-naming.md) (value renamed to `"a8"` in 0.5.0) / perf-ledger
  Q-8 / [research/2026-08-21-anima-i4-seat-speed.md](../research/2026-08-21-anima-i4-seat-speed.md)

## Context

The predicate of `linearCompute: "i8a8"` required i8-resident weights (`weightStorage === "i8"`).
An i4-resident linear fell outside it and went, without failing, to the ordinary f32 compute path.

The measurements of 2026-08-21 showed the cost for anima's i4 preset: for −21.2% download and
−22.6% VRAM, the DiT ran 2.0× slower than the default (1,646 vs 822 ms/step, medians of two runs,
research §3). The in-kernel i4 dequantization is almost free (+2% between f32 computes); only the
dp4a gain was lost. For families with an i8a8 default (anima `w8a8-s16`, irodori `w8a8` — preset
names of that date), choosing i4 halved the speed.

## Decision

### 1. Add i4 to the predicate (no new value)

`linearCompute === "a8" && (weightStorage === "i8" || weightStorage === "i4") && k > 0 &&
k % 4 === 0` (the value was `"i8a8"` until 0.5.0, ADR 0074). The option means "quantize activations
to i8 and compute with an integer dot product"; storage is a separate axis (i8-resident → w8a8,
i4-resident → w4a8). A value such as `"i4a8"` is rejected: the asset defines the storage, so the
manifest would hold it twice, and a mixed asset could accelerate only one of its parts (anima's i4
series held 453 i4 tensors and 1 i8 tensor).

### 2. The numeric contract differs from w8a8 (partial reduction per group)

    accf = 0
    for gi in 0..k/g-1:
        acci = Σ_{i∈group gi} xq[row,i]·wq[col,i]      # exact in i32 (|acci| ≤ g·127·8)
        accf = fma(f32(acci), wscale[col, gi], accf)   # flush exactly at the group boundary
    out = fma(accf, xs[row], bias[col])

Rounding happens in `k/g + 1` fma operations. The i32 → f32 conversion `f32(acci)` is not counted:
it is lossless while `|acci| ≤ 2^24` (shipped `g` is 16 to 64, so `|acci| ≤ g·1016 ≤ 65,024`).
For `g > 16,513` the conversion adds rounding; the limit `LINEAR_W4A8_MAX_GROUP` (= 2,113,664) only
guards i32 wrap-around and lets that range through. The atol=0 claim holds for the shipped `g`.

The w8a8 rule that `xs·wscale` MUST be folded into one f32 first does not hold, because `wscale`
changes per group; `xs` goes to the final fma. Counted the same way, w8a8 rounds twice. This
asymmetry is stated in the public API docstring and in both references.

### 3. Flush exactly at the group boundary, not at the tile boundary

The rounding count then does not depend on the geometry, so "tile geometry is outside the numeric
contract" (ADR 0025) holds for w4a8 too, and the CPU reference stays a plain four-level loop. A
geometry whose K tile crosses a group boundary fails with a **CodegenError** (it would mix two
groups' weights into one i32, a silently wrong value).

### 4. The overflow limit is on the group length, not on k

For i8 the whole reduction is one i32, so the limit is on **k** (`LINEAR_I8A8_MAX_K`). For i4 the
flush is per group, so the limit is on the **group length** (`LINEAR_W4A8_MAX_GROUP`). The k limit
would be an unfounded over-restriction for i4.

### 5. Nibbles expand to i8 lanes as integers

`packages/runtime/src/kernels/weight-storage.ts` gains an integer-lane expansion
(`i4IntegerLanesWgsl`), keeping the MUST on nibble order in one file (the group flush applies the
scale). It is separate from `dequant4` (f32 expansion); the five fused kernels generate byte-identical
WGSL.

### 6. anima's i4 presets do not declare `linearCompute` (visual ruling of 2026-08-21)

On the real GPU (1024², 8 steps, seed 42), `w4-a8-s16` with `linearCompute` declared was judged by
the maintainer "broken detail, rough lines"; without it, "keeps high quality". Speed recovers
(1,640 → 955 ms/step), but the preset exists for download size and VRAM, not speed (the default
`w8a8-s16` is faster at 823 ms/step). Therefore anima's i4 presets SHOULD NOT declare
`linearCompute` (re-graded 2026-10-18 to SHOULD: the ruling rests on one image of `anima-turbo`,
retired by ADR 0087, rendered before the calibration capture order was fixed (research §9); anima's
current base models ship no i4 preset (perf-ledger Q-5); a new visual A/B with at least four seeds
can reopen it). The runtime implementation is kept for future demand for low VRAM with speed.

Current fact (2026-10-18): no shipped preset uses the w4a8 path. The i4 presets of irodori
(`i8+dit4`, `tools/export-recipes/irodori/distribution.py:338-342`) and SBV2 (`i4`) declare an empty
`session`; the path runs only when a caller builds `SessionOptions` by hand.

## Consequences

- **BREAKING**: `"a8"` on i4-resident weights changes the output bits (f32 compute → integer dot
  product). No public manifest declared it; hand-built `SessionOptions` are affected. The four
  existing i8 variants generate byte-identical output (checked against the fixtures).
- Measured (RTX 3080 Ti, 1024², 8 steps): **955 ms/step**, +16% over the default 823 (research §6);
  the −21.2% download stays. **VRAM is not measured**: the activation quantization adds temporaries
  (`xq` / `xs`) (research §8-8).
- The remaining +16%: ① nibble expansion in the inner loop, ② `k/g` f32 flushes, ③ 16 i32 plus 16
  f32 accumulators, twice the register pressure of i8. A dedicated geometry (`linear_w4a8` in
  `I8a8GemmOp`, `packages/runtime/src/kernels/i8a8-geometry.ts`, with a lower `regM`) is not tried.
- Two numeric contracts share one `linearCompute` value (i8 = exact over the full k; i4 = partial
  reduction per group). Mixing them up voids the atol=0 claim, so the `:wi4g32` suffix of the
  pipeline key and the diagnostics identify the path.

## Addendum (2026-08-31) — numerics review G1-1: WGSL `fma` is not guaranteed to round once

The counts of §2 treat fma as one rounding, but WGSL only says `fma(x,y,z)` inherits the accuracy of
`x*y+z` (§15.7.4.1; §17.5.32 Note: a non-fused expansion conforms). The atol=0 match with the CPU
reference is a property of fused implementations; a non-fused one can differ by 1 ULP and conform
(the Metal case in `docs/known-issues.md`). The atol=0 check stays as a per-device conformance check
and is not loosened; the contract text was corrected instead (also in `kernels/linear-i8a8.ts`,
`reference/i8a8.ts` and `runtime/session-types.ts` under `packages/runtime/src/`).

## Addendum (2026-09-25) — where the storage is defined after the container format

Since ADR [0108](0108-container-format.md), "the asset defines the storage" (decision 1) means the
container descriptor's `encoding.codec`; `weightStorage` points at the layout derived from it (codec
registry — 0108 decisions 12 / 13, [container-v1](../container-v1.md) §6). The manifest still does
not hold it twice; `karume/5` carries the same field (ADR [0109](0109-manifest-v5-container.md)
decision 2).
