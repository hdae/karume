# 0098: Opt-in K-parallel summation for quantized GEMV

- Status: accepted (2026-09-13 — opt-in in the runtime, declared by quant presets); partly superseded
  by [0104](0104-gemma-fast-quant.md) / [0111](0111-session-options-composition.md) /
  [0109](0109-manifest-v5-container.md) (session fields, manifest; 2026-09-22 to 26) and
  [0128](0128-default-preset-original-weights.md) decision 1 (E4B default, 2026-10-17); decision 4
  re-graded 2026-10-18 to SHOULD; decision 5 withdrawn 2026-10-18
- Related: [0058](0058-numerics-opt-in-contract.md), [0082](0082-linear-gemv-decode.md),
  [0096](0096-speculative-decoding.md), [0097](0097-gemma4-qat-integration.md).

## Context

Sequential GEMV computes the K direction of one output on one thread. On Chrome long K waited long;
spreading the 16-byte packed words over lanes helped but slowed some shapes, so shapes are selected by
sweeps and whole-model A/B. The summation order changes, and for QAT the difference reaches the tokens
through the re-quantization boundaries. The default MUST NOT change silently.

## Decision

1. Add `SessionOptions.linearGemvReduce`: `"sequential" | "parallel"`, default `"sequential"`. The
   Gemma regular / QAT pipelines pass it to the target and the drafter Session. The family code default
   stays unchanged (distribution presets declare it — see the adoption section).
2. `parallel` is limited to f32 compute, INT2 / INT4 / INT8 storage, the selection table's shapes and
   physical M = 1..8. f16 / a8 compute is not implemented and is refused at construction. f16 / f32
   storage and other shapes use the previous path, told apart by the diagnostics key; M > 8 uses the
   prefill path. Switching for M=1 only is not taken: it would break the match with MTP verify row 0.
3. A 128-thread workgroup spreads the K packed words of one output round-robin over 2 / 4 / 16 / 32
   lanes. Each lane keeps the in-word order and the f32 rounding of dequantization; a fixed-order tree
   adds the partial sums and the bias is added once. No subgroups, no cross-workgroup sync, no extra f16
   rounding; every thread passes the barrier. The tolerance band and bit identity with the reference
   path are kept distinct. M=1 / 4 / 8 of one shape share shader and summation order.
4. The geometry SHOULD come from a static selection table of measured shapes; an unmeasured row MAY be
   added when the table marks it so. No automatic tuning from the GPU name or a run-time clock.
   (Re-graded 2026-10-18 to SHOULD: measurement serves speed only — correctness is covered for every
   shape by the FP64 band and the M=1 / 4 / 8 u32 match; 22 unmeasured E4B rows already run, see the
   2026-09-26 addendum.) The key holds `linear_gemv_parallel`, storage, group and lane count; WGSL of one
   key is byte-identical. Dispatch limits are refused by the existing checks. No extra weight or
   intermediate buffers; shared memory is 512 bytes per workgroup.
5. Withdrawn 2026-10-18: "selectable from the CLI and the Chrome comparison page" — those tools were
   removed by ADR [0127](0127-node-dawn-toolchain.md) decision 5 (2026-10-16). The public API only gains
   an optional field; IR, assets and existing goldens are unchanged.

## Acceptance (history)

Stages 1 to 4 (through the maintainer's M2 re-test) were completed by 2026-09-13:
[acceptance record](../research/2026-09-12-chrome-gemv-parallel.md). Quality is not judged by speed alone.

## Adoption in the quant definition (2026-09-13)

Adopted on the maintainer's M2 / Chrome measurements (regular E2B ≈ 25 %, QAT E2B ≈ 18 to 21 % faster,
TTFT about unchanged), within the delegated speed / quality judgement; not a broad quality guarantee.
Numbers: [the M2 adoption decision](../research/2026-09-13-m2-gemv-adoption.md).

- The regular / QAT E2B recipes add `i4-gemvpar` (the `i4` weights plus only
  `session: { linearGemvReduce: "parallel" }`) and made it `defaultQuant`; `i4` and its reference golden
  are unchanged. (The E2B default later moved to `i4-fast`, ADR 0104; E4B defaults: the 0104 addendum
  of 2026-09-26 for QAT, ADR 0128 decision 1 for regular E4B.)
- The hub's `SessionSpec` gains `linearGemvReduce`; unknown values are refused, an empty session is
  unchanged, an older reader refuses the new key instead of ignoring it (manifest now `karume/5`).
- `fromPretrained` passes the quant's declaration to target and drafter: explicit option →
  `quant.session` → runtime reference default (`fromAssets`: explicit only); never by model name,
  layer count or GPU name.
- A quant declaring a session field Gemma does not accept is refused before download (acceptance
  table: `packages/models/src/gemma/session-options.ts`, ADR 0111).
- Local distributions and published pins are not rewritten; a rebuilt distribution carries the new
  default without re-quantization. `i4` and explicit `"sequential"` remain for reference.

## Addendum (2026-09-20) — the parallel family spells multiply-add with explicit `fma()`

Two kernels of one formula (f32 and packed activations) differed in u32 on M2: Metal contracts to fma
differently per expression shape. The parallel family's multiply-add (f32 / packed × linear→SRQ fusion
off / on) is fixed to `acc = fma(x, d, acc)`. RTX / Vulkan numbers do not change (540 combinations, 0
mismatches); on Metal the parallel output changes and the variants align. Sequential GEMV and row blocks
keep their spelling (subgroup32 aligned on 2026-09-25, ADR 0101 addendum). Details: ADR
[0105](0105-packed-static-quantize-activations.md) addendum 4 (2026-09-20).

## Addendum (2026-09-26) — E4B rows take the lanes of the matching E2B rows (unmeasured)

The selection table (`PARALLEL_SHAPES`, `packages/runtime/src/kernels/linear-gemv.ts`) had no E4B row,
so `parallel` and what builds on it (ADR 0103, 0105) fell back on every E4B linear.

- **As an exception to decision 4, unmeasured rows are added**: regular E4B 12 (i4 g32 11 + i8 lm_head)
  and QAT E4B 10 (i4 g512 / g2048 / g4096 8 + i8 2). Lanes and `packedActivations` are copied from the
  E2B row with **the same role** (sliding / full included); each row says so.
- Shapes: regular E4B from the upstream config, QAT E4B from the mirror's container descriptor.
  Not added: QAT E4B's i2 lm_head (no E2B row), f32 per_layer_model_projection (perf-ledger K-51).
- Checks: GPU tests gain the E4B reduction length × group pairs (FP64 band, M=1 / 4 / 8 u32 match);
  the planned key of each QAT E4B linear is pinned by
  `packages/e2e/tests/e2e_gemma4_qat_parallel_gemv_plan_test.ts` (explicit SKIP without the mirror).
- Speed adoption is open (per-key census and GPU time decide lanes or removal; the flag-bench named
  for it went with the Deno tools, ADR 0127 decision 5).

## Addendum (2026-11-06) — the 12B rows, measured as whole-model decode

12B had no row, so `parallel` fell back on every 12B linear. The rows are in `PARALLEL_SHAPES`
(`packages/runtime/src/kernels/linear-gemv.ts`), int4 group 32 throughout:

| Role | Shape (n × k) | Lanes |
| ---- | ------------- | ----- |
| expanding: gate / up, full q, sliding q, sliding k / v | 15360 / 8192 / 4096 / 2048 × 3840 | 4 |
| full k | 512 × 3840 | 32 |
| reducing: o, down | 3840 × 4096 / 8192 / 15360 | 32, staged activations |
| output head | 262144 × 3840 | 16 |

- **Unlike the E4B copies, these are measured**, as whole-model decode at P ≈ 1K on the RTX 3080 Ti
  ([research 2026-11-06](../research/2026-11-06-gemma4-12b-fast.md) §2). The rows that widen take 4 lanes where E2B's take
  more: 4 beat 16 and 32 by 3.4–3.8 ms/token. On the reducing rows staged 32 tied staged 16 and beat unstaged 32 by about
  4 ms/token.
- **The output head is the first int4 `[262144, H]` row.** Lanes 16 and 4 tie, and 16 is kept; lanes 32 does not build on
  this device. E2B and E4B still have no such row (perf-ledger K-101).
- Generated tokens were identical in every arm of the sweep at this prompt. The rows change rounding, so that is an
  observation about these prompts, not a guarantee; the quant's description says so (ADR 0104).

## Addendum (2026-11-10) — the 26B rows, measured per kernel as whole-model decode

26B A4B (the mixture of experts, ADR [0145](0145-gemma4-26b-moe-resident.md)) had no row either, so `parallel` fell back
on each of its 236 dense linears, about half of a decode forward. Its rows are in `PARALLEL_SHAPES`, int4 group 32
throughout:

| Role | Shape (n × k) | Lanes |
| ---- | ------------- | ----- |
| expanding: full q, sliding q, sliding k / v, gate / up | 8192 / 4096 / 2048 / 2112 × 2816 | 4 |
| full k, router | 1024 / 128 × 2816 | 32 |
| reducing: o, down | 2816 × 4096 / 8192 / 2112 | 32, staged activations |
| output head | 262144 × 2816 | 4 |

- **Measured** on the RTX 3080 Ti as the per-kernel GPU time of whole-model decode
  ([research 2026-11-10](../research/2026-11-10-gemma4-26b-fast.md) §4). On the expanding rows 4, 16 and 32 tie and 2
  and 8 lose, so they take 12B's 4. The reducing rows tie between staged 32 and staged 16 and lose unstaged. The head
  takes 4, which beat 8 and 16 by a quarter of its time; 32 does not build, as on 12B.
- The dense linears go from 12.4 to 5.6 ms per decode forward, and the forward from 24.6–25.7 to 17.5–17.9 ms.
- **The routed experts are not on this table.** `expert_linear` keeps its sequential order under every option (its own
  numeric contract, ADR 0145 decision 4), so the exact expert cache's reference does not move
  (ADR [0146](0146-exact-expert-cache.md)).
- Generated tokens were identical in every arm on the three prompts measured; as for 12B, that is an observation, not a
  guarantee.
