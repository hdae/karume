# 0025: w8a8 linear execution (per-token i8 activations × integer dot product, opt-in)

- Status: accepted (2026-08-03; the quality was accepted by the maintainer's visual review — "a
  different image, but it follows the prompt, so the result is correct"); partly superseded: the
  knob value was renamed `"i8a8"` → `"a8"` by [0074](0074-quant-seat-naming.md) (0.5.0,
  2026-08-25); eligibility was extended to i4 weights by [0076](0076-w4a8-linear-execution.md)
  (2026-08-21, addendum); the rationale of decision 1's default moved to
  [0058](0058-numerics-opt-in-contract.md) decision 2 and [0128](0128-default-preset-original-weights.md);
  decision 6's E2E detector was replaced by [0110](0110-practical-tier-numerics-contract.md)
  decision 5 and [0129](0129-test-layout-and-lanes.md) (addendum 2026-10-17 (2)).
- Date: 2026-08-03
- Scope: `SessionOptions.linearCompute: "a8"` (`packages/runtime/src/kernels/quantize-rows.ts`,
  `linear-i8a8.ts`, `packages/runtime/src/reference/i8a8.ts`,
  `packages/runtime/src/runtime/recipe-builders/linear.ts`). This record writes the value as
  `"a8"`; the decisions were made under the old spelling `"i8a8"`.
- Related: [0019](0019-i8-weight-execution.md) (i8 weights) / [0020](0020-nan-propagation-bitwise.md)
  (NaN propagation) / [0022](0022-gemm-register-blocking.md) (GEMM skeleton) /
  [0002](0002-web-standard-api-only.md)

## Context

After the GEMM replacement (ADR 0022), linear was still 69.8% (1024 px, i8 weights) / 84.2%
(512 px, f16 weights) of DiT GPU time. The prototype's "DP4a 4.73×" was against the old 16×16
kernel; against the current kernel the net estimate was 1.543×, plus the removed i8 dequantisation.
Design: [research/2026-08-03-dp4a-w8a8-design.md](../research/2026-08-03-dp4a-w8a8-design.md).

Quality gate (Q0, measured before implementation on torch CPU with
`tools/export-recipes/anima/measure_quant.py`): w8a8 adds per-step noise 15.6× that of w8 and
changes composition and colours (a trajectory branch of the diffusion) without noise, banding or
blur. The maintainer accepted it visually, as for w8 in ADR 0019. Keeping some layers in f32 does
not pay; SmoothQuant-style smoothing is a separate task, since it breaks byte reuse of the w8
assets.

## Decision

1. **An opt-in execution knob; no new asset series.** Existing w8 assets are used byte for byte;
   `SessionOptions.linearCompute: "a8"` switches only the execution form. Eligible: a linear with
   resident i8 weights, `k % 4 == 0` and `k ≤ 2^17`; anything else stays on the usual path.
   Rejected: separate asset series (option B — duplicate bytes); an IR op (option C — quantisation
   is not part of graph semantics, ADR 0006).
   The default is `"f32"` (MUST), because auto-applying would silently change the sha256 checks.
   **(superseded by ADR 0058 decision 2 on 2026-08-14 and ADR 0128 on 2026-10-17: the runtime's
   omitted value is still `"f32"` — a MUST, now because the runtime default is the reference path
   (ADR 0058 decision 2). A distribution's default preset may declare `a8` (ADR 0128; anima,
   irodori and wan do), so "w8a8 is never used by default" does not hold for distributions.)**
2. **`quantize_rows`** (the first of the node's 2 dispatches; one row per workgroup of 256,
   grid-stride along the row): `s = max(amax·(1/127), tiny)`, `q = clamp(round(x/s), ±127)`,
   `pack4xI8`. **Closed at ±127** (as ADR 0019; −128 unused). NaN in the absmax propagates through
   a bitwise check (ADR 0020; plain `max` swallows NaN).
   **MUST: the scale is built by multiplying with 1/127, not dividing by 127** (`INV_127`). WGSL
   f32 division may be off by 2.5 ULP; on the development machine it differed from IEEE in 55,605
   of 200,000 samples by 1 ULP (multiply, add and `fma`: 0). `x/s` has a data-dependent divisor
   and cannot be replaced, so **elements near a rounding boundary may move by ±1 step**
   (`docs/limitations.md`); the atol = 0 tests measure the boundary margin and use it as a check.
3. **i8a8 GEMM.** As decided, the tile geometry was ADR 0022's 64×64 / 16×16 / 4×4 per thread /
   K tile 16 elements (4 packs); it is now the i8a8 geometry of ADR 0022's 2026-08-10 addendum
   (`packages/runtime/src/kernels/i8a8-geometry.ts`). Shared tiles are laid out `[pack][row]` /
   `[pack][col]` (bank conflicts 8-way → 2-way). The reduction is exact integer addition in
   `vec4<i32>` (order-independent). Dequantisation is
   **`out = fma(f32(acc), xs[row]·wscale[col], bias[col])`**: the product `xs·wscale` is formed
   first, and `fma` is explicit (plain `a*b + c` may or may not be fused by the driver; the
   development machine fused it). Codegen fixes both. **Overflow gate `k ≤ 131,072` (2^17)** fails
   loudly (`LINEAR_I8A8_MAX_K` in `linear-i8a8.ts`; derived from `|acc| ≤ k·127·128` and the i32
   limit — 16× the DiT's largest k = 8,192). The key is `linear:v4:i8a8:<geometry>:dp4a` or
   `:dp4aEmu` (`v5` for i4 weights; `linear-i8a8.ts`).
4. **Relation to ADR 0019 ("scale per element").** ADR 0019 rejected `(Σ x·q)·s` because f32
   reduction is order-dependent and GPU and CPU would lose bit identity. The w8a8 reduction is
   integer and order-independent, so `(Σ)·s` lets GPU and the CPU reference agree at atol = 0 — a
   stronger contract. ADR 0019 stays canonical for w8 (f32 compute).
5. **The dp4a and emulated variants are numerically identical.** `dot4I8Packed` (WGSL language
   extension `packed_4x8_integer_dot_product`) and `dot(unpack4xI8, unpack4xI8)` (core WGSL)
   return the same integer (intermediate at most 4·127·128 = 65,024 since weights may hold −128;
   corrected 2026-10-10 from 4·127² = 64,516). The extension affects speed only — no fail-loud and
   no per-environment tolerance (consistent with ADR 0002). For linear the variant is selected from
   the `wgslLanguageFeatures` enumeration (`linearI8a8Dot`, set at Session build); this is used only
   to choose between numerically identical variants. The test-only knob `I8A8_DOT` (unique symbol,
   `packages/runtime/src/runtime/session-types.ts`, not exported from `mod.ts`) forces each variant
   so both run against atol = 0.
   Limits on this rule today: the enumeration is valid for linear only. ADR 0058 (addendum
   2026-08-29, general rule 3) allows an auto-selection that claims bit identity only when the
   identity is machine-checked on the device — attention uses a runtime canary because the claim
   failed there on Apple M2. A device without dp4a gets one warning per device, not a fallback (ADR
   0128 decision 5, `packages/runtime/src/runtime/a8-emulation-warning.ts`). (Kept at the
   2026-10-18 triage.)
6. **E2E cannot be a numeric parity net.** Activation quantisation is discontinuous: an upstream
   difference of order 1e-5 makes rounding jump ±1 step and saturates within a few layers, so GPU
   and a torch mirror become different samples of one distribution. A band of "5–10× the measured
   error" is vacuous. As decided, the E2E detectors were (1) a tight tolerance on the text stage,
   (2) a discriminating band at step 1 that fails if the path silently drops to f32, and (3) a
   check of the pipeline keys that ran. **(superseded by ADR 0110 decision 5 on 2026-09-26 and ADR
   0129 addendum 2026-10-17 (2): the E2E check is an A/B against the same machine's reference path
   (census, difference floor, collapse ceiling — `packages/e2e/tests/e2e_anima_ab_test.ts`), and
   comparison with the upstream torch mirror is opt-in. The old test file was deleted on
   2026-08-15. ADR 0110 decision 4 generalises the ban on "5–10× the measured" bands.)**
   **The real numeric contract is `packages/runtime/tests/gpu_i8a8_test.ts` at atol = 0** (GPU vs
   the TS reference, dp4a vs emulation). The Python numeric spec is
   `tools/exporter/src/karume/act_quant.py`; the mirror pipeline is
   `tools/export-recipes/anima/pipeline_ref.py --dtype i8 --act-quant`.

## Detection limits and constraints

- Non-finite values: NaN propagates over the whole row through the row scale (same granularity as
  f32). **The sign of Inf does not match the f32 path** (`docs/limitations.md`). When the row scale
  is non-finite, the contents of `xq` are outside the contract.
- WGSL `round` rounds half to even (as torch; JS `Math.round` does not); the TS reference uses
  `roundTiesToEven` only.
- Clamping to ±128 is structurally unreachable (`s = amax/127` gives `|x/s| ≤ 127`); the detector
  is the scale's divisor, not the clamp.
- SBV2 is not a target for w8a8 (conv1d dominates all five targets; linear is about 0 GFLOP).

## Consequences

- Measured on 2026-08-03 (512 px, one DiT step): linear 816.1 → 251.0 ms (3.25×), DiT step
  947.2 → 420.8 ms (2.25×) — above the 1.543× estimate because the i8 dequantisation of the
  baseline disappears too. Whole demo 512 px 23.9 → 15.6 s, 1024 px 68.3 → 36.5 s.
- Images differ from f32 / w8 execution (a trajectory branch), so this path is outside the
  reference-path sha256 checks and its outputs are named separately.
- Possible follow-ups: SmoothQuant-style smoothing (exporter change), w8a16, integer dot products
  for conv1d / conv2d.

## Addendum (2026-08-21) — eligibility extended to i4 weights by ADR 0076

ADR 0076 (w4a8) widened decision 1's eligibility from resident i8 to resident integer weights
(i8 / i4). The predicate is `linearCompute === "a8" && (weightStorage === "i8" || weightStorage ===
"i4") && k > 0 && k % 4 === 0` (`packages/runtime/src/runtime/recipe-builders/linear.ts`).

- The k gate (`k ≤ 2^17`) applies to the i8 variant only. The i4 variant flushes per group, so
  only one group's sum sits in i32; its gate is the group length (`LINEAR_W4A8_MAX_GROUP`, ADR 0076
  decision 4).
- The numeric contracts differ (i8: exact over full k; i4: partial reduction per group). The
  pipeline key's `:wi4g32` suffix tells the two apart.
- Declaring `"a8"` on an i4 asset is therefore no longer a no-op.
