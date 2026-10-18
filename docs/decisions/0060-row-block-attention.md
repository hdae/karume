# 0060: Row-block execution of decomposed attention (K-5 option B; default path; bit-identical)

- Status: accepted (2026-08-14; the maintainer's ruling "B first — check bit identity first", after
  a spike); in force — [0124](0124-attention-row-block-budget.md) added a user budget on 2026-10-11
  without changing the default
- Date: 2026-08-14
- Related: [0058](0058-numerics-opt-in-contract.md) / [0022](0022-gemm-register-blocking.md) /
  [0023](0023-fused-attention.md) / [0044](0044-runtime-attention-mask.md) / perf-ledger K-5

## Context

Irodori DiT's decomposed attention (bmm QKᵀ → add mask → safe_softmax → bmm PV) materialises
`scores[1,20,750,2269]` = 136,140,000 B at S=750, so allocation and binding fail on a device with
the WebGPU core default `maxStorageBufferBindingSize` of 128 MiB
([dit-export-recon](../research/2026-08-11-dit-export-recon.md)). Scope: runtime only (`fusion.ts`,
`fusion-rules/row-block-attention.ts`, `executor.ts`, `kernels/bmm.ts`, `kernels/gemm.ts`,
`gpu/acquire.ts`); IR, exporter and assets do not change. Option B (this ADR) solves portability on
the default path with bit identity; option A (online attention) was to follow as an opt-in option.

## Decision

1. **A `FusedStep` becomes "a dispatch sequence plus step-local temporaries"** (an extension of the
   fusion mechanism). The single-dispatch rules stay single-dispatch. Temporary lifetimes are
   declared by dispatch-boundary indices and the executor replays them into `StepRecipeBuilder`;
   the release bookkeeping still meets in one place (`executeStepRecipe`) and use counts still come
   from one list (`FusedStep.ins`). Workgroup counts are a discriminated union (grid-stride, which
   may shrink / tiled, the GEMM family, which fails loudly above the limit).
2. **Fusion rule `rowBlockAttention`.** The exporter's run of nine consecutive nodes (bmm → reshape
   → add → safe_softmax → identity expand/reshape ×2 → bmm; every edge and shape matched exactly)
   is replaced by the same four dispatches per query row block. A matched window is **always**
   fused. The block count n comes from the pure function `planRowBlocks` at Session build (inputs:
   granted limit, H, C and resolved S only — no run-time autotuning, ADR 0022): the smallest even
   split whose block fits the binding limit. **With n = 1 the result is identical to the plain four
   dispatches** (same keys and params, zero extra cost). A shape where even one row does not fit
   fails loudly (falling back to the plain sequence would make an allocation failure look like "the
   fusion merely did not apply").
3. **Row-window variants of bmm** (`:rwa` = the QK side reads q with full-M stride plus a row
   offset; `:rwc` = the PV side writes the output the same way). Only this one bit enters the key;
   the offset and full M are bmm-only uniform values (the shared `gemmParams` and other ops'
   snapshots do not change). The code differs only in two lines of base arithmetic in
   `batchPrologue`; **the K reduction order and multiply-accumulate text of an output element do not
   change by one character**, which makes row blocks bit-identical to one block (numeric contract of
   `gemm-geometry.ts`). add (mask `[1,1,1,C]`) and safe_softmax stay within a row and reuse the
   existing kernels.
4. **Numerics: the default path.** Bit identity means no opt-in option under ADR 0058 is needed
   (options are for numerics-changing optimisations — option A only). Checks: ① existing WAV / PNG /
   golden digests unchanged; ② forced-split parity (the internal `ROW_BLOCK_SPLIT` forces 2, 3 and
   uneven splits and must match one block in Uint32; the output is also checked to be non-constant
   so the check cannot pass vacuously).
5. **Portability check.** The internal `LIMIT_CAPS` (it can only **lower** `requiredLimits` in
   `acquireGpu`, never raise them) creates a device with `maxStorageBufferBindingSize` = 128 MiB. A
   synthetic graph with scores above 128 MiB passes with row blocks and, with the window broken,
   fails with `GpuValidationError` on a real GPU. Running on a lowered device is the only way to
   confirm behaviour on a core-default machine (enumeration is no evidence).
6. **Observation.** `lastRunFusions.rowBlockAttention` (Irodori DiT = 12, in the fusion-count check
   `packages/e2e/tests/e2e_irodori_fusion_counts_test.ts`). `identityExpand` 48 → 24 is a move —
   the identity expands inside the window are absorbed — not a regression. gpuTiming attribution
   keys split into `…:rwa` / `…:rwc` **only when n ≥ 2** (n = 1 keeps the old keys).

## Acceptance (2026-08-14, RTX 3080 Ti / Vulkan)

Full verify green with WAV / PNG / golden digests unchanged; forced-split parity and the 128 MiB
portability check green; n = 1 non-regression within noise, S=750 two blocks 46.8 ms vs one block
48.9 ms, intermediate peak 267.0 → 137.2 MiB.

## Remaining and connected items

- Option A was rejected after a spike on 2026-08-14 (×1.19 < kill 1.3; inverted at S=170) —
  perf-ledger K-5b, [research](../research/2026-08-14-attention-chain-fusion-spike.md).
- Out of scope: moving this window to the IR op `attention` (ADR 0023) needs the mask contract
  (`[1,1,M,N]` vs `[1,1,1,N]`) and safe_softmax's empty-row contract (ADR 0044). It would avoid
  materialising P (`exp(S−m)·inv`) and save dispatches, but not S, which fused attention also
  allocates (`kernels/attention.ts` header); dropping S needs online attention (K-5b; corrected
  2026-10-10).
- The operand order of an omitted `FusedDispatch.operands` is a doc-level MUST only
  (`fusion-rule.ts`); typing it was not adopted, weighed against four rules then (seven today).
- Addendum (2026-10-11): the block count also honours a user budget
  (`SessionOptions.attentionRowBlockMaxBytes`, per row-block temporary); the limit is the smaller of
  the two. Without a budget this ADR's default holds — [ADR 0124](0124-attention-row-block-budget.md).
