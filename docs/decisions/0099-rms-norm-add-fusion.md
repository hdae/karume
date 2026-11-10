# 0099: Optional fusion of RMS normalization with the following add

- Status: accepted (2026-09-13; opt-in in the runtime); partly superseded by
  [0104](0104-gemma-fast-quant.md) (2026-09-15: the Gemma `i4-fast` presets, E2B's default, declare
  `fuseRmsNormAdd: true` — decision 4's "defaults unchanged") and by
  [0128](0128-default-preset-original-weights.md) decision 1 (2026-10-17: the M2 condition on
  promotion); on 2026-10-18 decision 1's width limit was re-graded to SHOULD and decision 5 withdrawn
- Date: 2026-09-13
- Related: [0017](0017-rms-norm-conv2d-clamp-min.md), [0058](0058-numerics-opt-in-contract.md),
  [0068](0068-decode-exit-multi-output.md), [0098](0098-linear-gemv-parallel.md)

## Context

Regular and QAT Gemma E2B contain 106 adjacent `rms_norm → add` pairs with a private intermediate;
the normalized value is the first add argument in 1 and the second in 105, so both orders and the
fusion count on the real distribution need checks. Chaining them directly lets the compiler fuse the
last multiply with the add, which changes rounding. Keeping the rounding through workgroup memory
cancelled the gain. Inserting an integer operation matched the reference (output u32 on Deno / RTX,
a fixed generated sequence on Chrome) with no extra memory. Fusion alone did not speed up regular
E2B; with a submit limit of 768 it did (2026-09-13). Dispatch counts and rejected candidates: the
[research record](../research/2026-09-13-rms-norm-add-fusion.md).

## Decision

1. Add `SessionOptions.fuseRmsNormAdd?: boolean`, default false; a wrong type is rejected at
   Session construction. Only adjacent f32 RMS → add pairs with the same shape, whose intermediate
   has one consumer and is not a graph output, are accepted. The last dimension SHOULD be limited to
   the widths the unit check covers — today 256 / 1536 / 2560
   (`packages/runtime/src/runtime/fusion-rules/rms-norm-add.ts`) — and a width is added together
   with the same width in the unit u32 check (re-graded 2026-10-18 to SHOULD: the limit comes from
   test coverage, not from the arithmetic). Broadcast, duplicate bindings, non-adjacent forms and
   other widths run on the existing path. `lastRunFusions.rmsNormAdd` reports the applications.
2. The kernel shares the reduction and grid-stride of the 256-thread RMS, with two keys that keep the
   left / right order of the add. The normalized value is reinterpreted as u32, XORed with a uniform
   0 (the fourth word of `rmsNormParams`) and turned back into f32 before the add; this is not
   simplified into a constant-expression bitcast round trip. A check that changes the mask confirms
   that the integer operation runs.
3. This is not a specified guarantee of bit identity on every GPU: WGSL allows
   [reassociation and fusion](https://www.w3.org/TR/WGSL/#reassociation-and-fusion) and leeway for
   special values ([bitcast](https://www.w3.org/TR/WGSL/#bitcast-builtin)). The default reference
   path, the original key and WGSL, and the goldens stay. A mismatch with the reference does not
   weaken the tolerance; cause, quality and speed are examined as a separate execution setting.
4. `Gemma4Pipeline` and `Gemma4QatPipeline` accept `fuseRmsNormAdd` and the existing runtime option
   `submitPolicy` and pass them to the target and the drafter. Fusion does not change the submit
   limit implicitly; the two settings are independent. The defaults of the runtime, models, CLI and
   quant definitions, the hub storage format, the exporter and the published revisions do not
   change. (Superseded by ADR 0104 on 2026-09-15: `fuseRmsNormAdd` is in the manifest `session`
   vocabulary (`packages/hub/src/manifest.ts`) and the `i4-fast` presets declare it.)
5. Withdrawn 2026-10-18: the three settings on the Chrome comparison page (the Deno-era measurement
   tools were not ported — ADR 0127 decision 5).

## Acceptance

Checked on 2026-09-13: byte-identical original 128 / 256-thread RMS keys and WGSL; matcher accepts
both orders, rejects shared, public, broadcast, non-adjacent and unchecked-width forms, and fuses all
106 pairs; u32 identity of finite values with NaN classes, signed zero, cancellation, forced
grid-stride, input lifetime and first rows for M = 1 / 4 / 8; real-weight generation, speculative
verify row 0 and decode rows in u32. Speed is measured on the normal GPU greedy output path, not the
diagnostic path. Promotion to a default waited for the maintainer's M2 check (superseded by ADR 0128
decision 1 on 2026-10-17: the default preset is chosen per model by a recorded procedure).

## Addendum (2026-09-13) — after the M2 check

The maintainer's 120 generations on M2 matched the reference with a small gain (user M2 results in
[research 2026-09-13](../research/2026-09-13-rms-subgroup-reduction.md)). The fused quant choice and
the submit limit (a host policy) stay separate: `submitPolicy` is not added to the quant storage
vocabulary (reaffirmed by ADR 0104). ADR 0104 declared the fusion in the default preset on 2026-09-15.

## Addendum (2026-10-31) — the fused kernel is the prefetch form (v2, perf-ledger K-99)

The fused kernel's output loop now reads the x, weight and residual words it loaded before the
reduction and runs a constant `dim / 256` trips; the reduction, the normalised value and the XOR
rounding barrier are the same text, so it writes v1's words (checked as u32 on the GPU against the
v1 WGSL kept in `packages/runtime/tests/fixtures/wgsl-baseline/`). The key carries the trip count
(`rms_norm_add:v2:rms_norm:v2:f32:lastdim:wg256:trip<n>:<order>:xor-round`), and the generator
accepts only the widths of ADR [0017](0017-rms-norm-conv2d-clamp-min.md) addendum 2 — the rule's
measured widths 256 / 1536 / 2560 are all among them. The subgroup32 form is unchanged.

## Addendum (2026-11-06) — width 3840 for 12B

The rule's widths are now 256 / 1536 / 2560 / 3840. 3840 is among the widths the generator accepts (ADR
[0017](0017-rms-norm-conv2d-clamp-min.md) addendum 2), and it is covered by the same u32 check against the unfused pair
(`packages/runtime/tests/gpu_rms_norm_add_fusion_test.ts`). It is part of 12B's `i4-fast`
([research 2026-11-06](../research/2026-11-06-gemma4-12b-fast.md)); its own gain was not separated from the other rows.

## Addendum (2026-11-10) — width 2816 for 26B

The rule's widths are now 256 / 1536 / 2560 / 2816 / 3840. 2816 (11 × 256) is among the widths the generator accepts,
and it is covered by the same u32 check against the unfused pair. On 26B it fuses 90 pairs per decode forward, about
0.5 ms of GPU time on the RTX 3080 Ti ([research 2026-11-10](../research/2026-11-10-gemma4-26b-fast.md) §5). It is part
of 26B's `i4-fast`.
