# 0103: Optional fusion of parallel GEMV with static requantization

- Status: accepted (2026-09-15, within the maintainer's request to continue optimization; M2 short
  generation accepted); partly superseded by [0104](0104-gemma-fast-quant.md) (2026-09-15: the QAT
  `i4-fast` presets declare `fuseLinearStaticQuantize: true` — "defaults unchanged" and the M2
  condition) and [0111](0111-session-options-composition.md) (2026-09-26: composition of explicit
  options, quant declarations and runtime defaults); on 2026-10-18 the Chrome page selection was
  withdrawn
- Date: 2026-09-15
- Related: [0097](0097-gemma4-qat-integration.md), [0098](0098-linear-gemv-parallel.md),
  [0096](0096-speculative-decoding.md), [0040](0040-fusion-pass.md)
- Evidence: [re-check of held candidates](../research/2026-09-15-held-combinations.md),
  [acceptance of the product implementation](../research/2026-09-15-linear-static-quantize-fusion.md)

## Problem and choice

Static mobile QAT E2B runs a static requantization to a fixed scale (SRQ) right after each matrix
product. A spike merging both into one dispatch on the current parallel GEMV gained about 5.5%; the
earlier rejection on the old sequential GEMV does not carry over (2026-09-15). Broader fusion, weight
transposition and a changed submit limit are not adopted together; only this fusion is added.

## Option and eligibility

Add `SessionOptions.fuseLinearStaticQuantize?: boolean`, default false. True is accepted only with
`linearGemvReduce: "parallel"` and `linearCompute: "f32"`; another combination or a non-boolean is
rejected at Session construction, and the parallel setting is not added implicitly. The shared
regular / QAT Gemma pipeline options pass the setting to the target and drafter Sessions.

Fused are adjacent `linear → static_quantize` pairs with all inputs and outputs f32, equal output
shapes, a private linear output with one consumer and three distinct input bindings; limited to
resident INT2 / 4 / 8 weights, the shapes the existing `linearGemvParallelLanes` accepts, and
physical M = 1..8. That is ADR 0098's table, which since 2026-09-26 includes unmeasured E4B rows that
QAT E4B `i4-fast` fuses on (ADR 0104 addendum 2026-09-26). Other shapes and public or shared
intermediates stay unfused. `lastRunFusions.linearStaticQuantize` and a separate key show it.

The defaults of models, quants and CLI, the manifest, IR, weights and published source pins do not
change. (Superseded by ADR 0104 on 2026-09-15: the option is in the manifest `session` vocabulary
(`packages/hub/src/manifest.ts`) and the QAT `i4-fast` presets declare it.) False returns to the
unfused path. Regular Gemma has no matching SRQ in its graph, so nothing is fused there.

## Arithmetic and code generation

The fused kernel shares the existing `parallel` WGSL generator and switches only the output part.
The cyclic K distribution, in-word dequantization and multiply-accumulate, workgroup summation tree
and bias order stay; keys and WGSL of the parallel, sequential and subgroup variants do not change.

The result with bias is reinterpreted as u32, XORed with a uniform 0 and passed to the SRQ integer
table; this fourth word is not replaced by a constant expression. Scale 0 keeps the identity
operation; saturation, NaN classes and signed zero come from the same table as the standalone SRQ,
built by the shared `staticQuantizeParams`. Only inside the fusion is the upper-bound search at the
128 boundary unrolled to a fixed count; the standalone SRQ's binary search, key, WGSL and reference
goldens stay.

The fused key includes storage, group length and lane count; shape and scale are uniforms, and the
same key generates the same WGSL. M = 1 / 4 / 8 share the arithmetic; a tile count above the device
limit is rejected. u32 identity on real devices is an acceptance result, not a specified guarantee
of floating-point rounding on every GPU.

## Ownership and acceptance

The pure fusion function receives only a description of resident storage, no GPU objects. At recipe
derivation the weight scale is borrowed from the existing resident buffer; no GPU transposition,
weight copy or separate scale owner. The existing fusion mechanism derives the total use count of
external inputs, and the scale lives as long as the Session / `sharedWeights` lease. The fused params
are the linear's 4 words and the SRQ's 260, through the existing Session params cache.

Checks: application and non-application; every boundary and special value against the original
`parallel` + SRQ; scale 0; execution of the rounding XOR; row-0 identity for M = 1 / 4 / 8; borrowing
and release of the shared scale; the fusion count of the real distribution; generation and speed;
the MTP reference path. QAT MTP itself is not implemented or guaranteed by this change.

Promotion to a speed-bundled default waited for the M2 check, and default adoption was to go through
the `quant.session` allow table and the precedence of explicit options (superseded by ADR 0104 on
2026-09-15 and ADR 0111 on 2026-09-26: both are met). Withdrawn 2026-10-18: the Chrome page
selection (not ported — ADR 0127 decision 5).

## Addendum (2026-09-15) — after the M2 check

The maintainer's 40 generations on M2 matched with a small gain; the fusion stays as an optional
speed-up and joined the fast QAT E2B preset (ADR 0104). The speed samples overlap; no fixed gain is
guaranteed everywhere. Data: [M2 adoption record](../research/2026-09-15-m2-linear-srq-adoption.md).
