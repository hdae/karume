/**
 * The checks the entry runs: the shared implementations of `@karume/kernel-checks` (the same cases
 * the runtime's node GPU tests run), with the entry's titles and what each needs of the device.
 */

import {
  ATTENTION_ONLINE_TILED_CASES,
  checkAttentionOnlineTiled,
  checkExpertLinear,
  checkFusionTwins,
  checkGemvParallelRows,
  checkGemvSubgroup,
  checkIntegerGemv,
  type CheckOptions,
  checkPlanesGemv,
  checkRmsNormPrefetch,
  checkRmsSubgroup,
  checkStateAttentionRewrites,
  checkScoreF16Store,
  checkStateBidirectional,
  checkStateKv16,
  GEMV_INTEGER_CASES,
  GEMV_PARALLEL_ROWS_CASES,
  GEMV_PLANES_CASES,
  GEMV_SUBGROUP_CASES,
  RMS_NORM_PREFETCH_CASES,
  RMS_SUBGROUP_CASES,
  STATE_ATTENTION_CASES,
  STATE_ATTENTION_MUTANTS,
  SCORE_F16_CASES,
  STATE_BIDIRECTIONAL_CASES,
  STATE_KV16_CASES,
  EXPERT_LINEAR_CASES,
  FUSION_TWIN_CASES,
} from "@karume/kernel-checks";
import type { GpuContext } from "@karume/runtime";

import type { CheckReport, KernelCheckId } from "./verdict.ts";

export type KernelCheck = {
  readonly id: KernelCheckId;
  readonly title: string;
  /** Needs `subgroups` + `subgroup-size-control` + WGSL `subgroup_id` on the device. */
  readonly needsSubgroups: boolean;
  readonly run: (gpu: GpuContext, options: CheckOptions) => Promise<CheckReport>;
};

export const KERNEL_CHECKS: readonly KernelCheck[] = [
  // The parallel GEMV's subgroup32 reduction against the plain parallel reduction: u32 identity,
  // plus the FP64 band (ADR 0101).
  {
    id: "gemv-subgroup32",
    title: "GEMV subgroup32",
    needsSubgroups: true,
    run: async (gpu, options) => ({
      parts: [await checkGemvSubgroup(gpu, options)],
      expectedCases: GEMV_SUBGROUP_CASES,
    }),
  },
  // rms_norm's subgroup32 form through real Sessions: within the CPU reference's rms_norm
  // tolerance, row 0 identical at every M, fused rms_norm + add identical to unfused (ADR 0100).
  {
    id: "rms-subgroup32",
    title: "RMS subgroup32",
    needsSubgroups: true,
    run: async (gpu, options) => ({
      parts: await checkRmsSubgroup(gpu, options),
      expectedCases: RMS_SUBGROUP_CASES,
    }),
  },
  // The prefetch form of the 256-wide rms_norm and of both RMS → add orders against their
  // runtime-trip v1 WGSL: u32 identity, NaN by payload (perf-ledger K-99).
  {
    id: "rms-norm-prefetch",
    title: "rms_norm: prefetch form vs runtime-trip form",
    needsSubgroups: false,
    run: async (gpu, options) => ({
      parts: await checkRmsNormPrefetch(gpu, options),
      expectedCases: RMS_NORM_PREFETCH_CASES,
    }),
  },
  // The folded ①′, the unrolled ③′ and the fused stats + PV against their v1 WGSL: u32 identity,
  // plus order-only mutants that must mismatch.
  {
    id: "state-attention-rewrites",
    title: "States attention rewrites (K-55 / K-91)",
    needsSubgroups: false,
    run: async (gpu, options) => ({
      ...(await checkStateAttentionRewrites(gpu, options)),
      expectedCases: STATE_ATTENTION_CASES,
      expectedMutants: STATE_ATTENTION_MUTANTS,
    }),
  },
  // The integer GEMV against its CPU reference, with and without the SRQ epilogue: u32 identity,
  // plus saturated fixtures.
  {
    id: "gemv-integer",
    title: "Integer GEMV",
    needsSubgroups: false,
    run: async (gpu, options) => ({
      parts: [await checkIntegerGemv(gpu, options)],
      expectedCases: GEMV_INTEGER_CASES,
    }),
  },
  // The byte-plane GEMV against an FP64 reference (within the GEMM band) and the sequential GEMV
  // (within twice that band); two runs identical, nothing written past the outputs (ADR 0147
  // decision 2).
  {
    id: "gemv-planes",
    title: "Byte-plane GEMV",
    needsSubgroups: false,
    run: async (gpu, options) => ({
      parts: [await checkPlanesGemv(gpu, options)],
      expectedCases: GEMV_PLANES_CASES,
    }),
  },
  // The row-blocked parallel GEMV (a profile row's row block, M = 2..8) against the parallel GEMV
  // at the same lanes: u32 identity, plus the FP64 band (perf-ledger K-108).
  {
    id: "gemv-parallel-rows",
    title: "Parallel GEMV row blocks",
    needsSubgroups: false,
    run: async (gpu, options) => ({
      parts: [await checkGemvParallelRows(gpu, options)],
      expectedCases: GEMV_PARALLEL_ROWS_CASES,
    }),
  },
  // The f16 append against host rounding, and every reader of an f16 slot against its f32 kernel
  // over the slot rounded on the host: u32 identity (ADR 0143).
  {
    id: "state-kv16",
    title: "f16 full state slots",
    needsSubgroups: false,
    run: async (gpu, options) => ({
      parts: await checkStateKv16(gpu, options),
      expectedCases: STATE_KV16_CASES,
    }),
  },
  // The f16 store of the attention scores against host rounding: u32 identity (pack2x16float may
  // round to nearest or toward zero; the record names which).
  {
    id: "score-f16-store",
    title: "f16 attention scores (S)",
    needsSubgroups: false,
    run: async (gpu, options) => ({
      parts: [await checkScoreF16Store(gpu, options)],
      expectedCases: SCORE_F16_CASES,
    }),
  },
  // The sliding ①ₜ with GenerationRun.bidirectionalRows against the same kernel keeping every score
  // and a host mirror of the predicate: u32 identity (ADR 0144).
  {
    id: "state-bidirectional",
    title: "Bidirectional block (Gemma 4 12B images)",
    needsSubgroups: false,
    run: async (gpu, options) => ({
      parts: [await checkStateBidirectional(gpu, options)],
      expectedCases: STATE_BIDIRECTIONAL_CASES,
    }),
  },
  // expert_linear with f16 steps at the 26B shapes, every pick against linear's i4 GEMV on that
  // expert with the steps widened: u32 identity (ADR 0145).
  {
    id: "expert-linear",
    title: "Expert linear (Gemma 4 26B A4B)",
    needsSubgroups: false,
    run: async (gpu, options) => ({
      parts: [await checkExpertLinear(gpu, options)],
      expectedCases: EXPERT_LINEAR_CASES,
    }),
  },
  // Each built-in fusion against the same graph run node by node: u32 identity, plus the fusion
  // counters of both sides (ADR 0147 decision 9); a part the device cannot run reads as not run.
  {
    id: "fusion-twins",
    title: "Fusion rules: fused vs unfused",
    needsSubgroups: false,
    run: async (gpu, options) => ({
      parts: await checkFusionTwins(gpu, options),
      expectedCases: FUSION_TWIN_CASES,
    }),
  },
  // A measure-first candidate (perf-ledger K-106), not a production route: the tiled quantize_rows
  // against the host permutation of today's Vᵀ producer bytes, and the online kernel reading the
  // tiled Vᵀ against today's kernel, every O word identical.
  {
    id: "attention-online-tiled",
    title: "Online attention: tiled Vᵀ key axis (candidate)",
    needsSubgroups: false,
    run: async (gpu, options) => ({
      parts: await checkAttentionOnlineTiled(gpu, options),
      expectedCases: ATTENTION_ONLINE_TILED_CASES,
    }),
  },
];
