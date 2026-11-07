/**
 * The checks the page runs: the shared implementations of `@karume/kernel-checks` (the same cases
 * the runtime's node GPU tests run), with the page's titles and what each needs of the device.
 */

import {
  checkGemvSubgroup,
  checkIntegerGemv,
  type CheckOptions,
  checkPlanesGemv,
  checkRmsSubgroup,
  checkStateAttentionRewrites,
  checkScoreF16Store,
  checkStateKv16,
  GEMV_INTEGER_CASES,
  GEMV_PLANES_CASES,
  GEMV_SUBGROUP_CASES,
  RMS_SUBGROUP_CASES,
  STATE_ATTENTION_CASES,
  STATE_ATTENTION_MUTANTS,
  SCORE_F16_CASES,
  STATE_KV16_CASES,
} from "@karume/kernel-checks";
import type { GpuContext } from "@karume/runtime";

import type { CheckReport, KernelCheckId } from "./verdict";

export type KernelCheck = {
  readonly id: KernelCheckId;
  readonly title: string;
  readonly summary: string;
  /** Needs `subgroups` + `subgroup-size-control` + WGSL `subgroup_id` on the device. */
  readonly needsSubgroups: boolean;
  readonly run: (gpu: GpuContext, options: CheckOptions) => Promise<CheckReport>;
};

export const KERNEL_CHECKS: readonly KernelCheck[] = [
  {
    id: "gemv-subgroup32",
    title: "GEMV subgroup32",
    summary:
      'linearGemvReduce: "parallel-subgroup32" against the parallel reduction, u32 identity (i2 / i4 / i8, 5 lane counts, M = 1 / 4 / 8), plus the FP64 band (ADR 0101).',
    needsSubgroups: true,
    run: async (gpu, options) => ({
      parts: [await checkGemvSubgroup(gpu, options)],
      expectedCases: GEMV_SUBGROUP_CASES,
    }),
  },
  {
    id: "rms-subgroup32",
    title: "RMS subgroup32",
    summary:
      'rmsNormReduce: "subgroup32" through real Sessions: the rms_norm tolerance of the CPU reference, row 0 identical at every M, fused rms_norm + add identical to unfused, grid-stride (ADR 0100).',
    needsSubgroups: true,
    run: async (gpu, options) => ({
      parts: await checkRmsSubgroup(gpu, options),
      expectedCases: RMS_SUBGROUP_CASES,
    }),
  },
  {
    id: "state-attention-rewrites",
    title: "States attention rewrites (K-55 / K-91)",
    summary:
      "The folded ①′ against ①′ (S words) and the unrolled ③′ and fused stats + PV against their v1 WGSL (O words), u32 identity, plus order-only mutants that must mismatch (on one case, and on every case of their shape where their code path runs).",
    needsSubgroups: false,
    run: async (gpu, options) => ({
      ...(await checkStateAttentionRewrites(gpu, options)),
      expectedCases: STATE_ATTENTION_CASES,
      expectedMutants: STATE_ATTENTION_MUTANTS,
    }),
  },
  {
    id: "gemv-integer",
    title: "Integer GEMV",
    summary:
      "The integer GEMV against its CPU reference, u32 identity (i2 / i4 / i8, up to 15 variants per form, M = 1 / 8, with and without the SRQ epilogue), plus saturated fixtures.",
    needsSubgroups: false,
    run: async (gpu, options) => ({
      parts: [await checkIntegerGemv(gpu, options)],
      expectedCases: GEMV_INTEGER_CASES,
    }),
  },
  {
    id: "gemv-planes",
    title: "Byte-plane GEMV",
    summary:
      "The byte-plane GEMV (linearGemvPlanes) within the GEMM band of an FP64 reference and within twice that band of the sequential GEMV (15 variants, K = 16 / 80 / 1536, 37 and 1029 columns, M = 1 / 8, plus a saturated fixture); two runs identical, nothing written past the outputs, a NaN reaches its whole row (ADR 0110 class R).",
    needsSubgroups: false,
    run: async (gpu, options) => ({
      parts: [await checkPlanesGemv(gpu, options)],
      expectedCases: GEMV_PLANES_CASES,
    }),
  },
  {
    id: "state-kv16",
    title: "f16 full state slots",
    summary:
      "The f16 append against host rounding (word layout included), and every reader of an f16 slot against its f32 kernel over the same slot rounded on the host, u32 identity (ADR 0143).",
    needsSubgroups: false,
    run: async (gpu, options) => ({
      parts: await checkStateKv16(gpu, options),
      expectedCases: STATE_KV16_CASES,
    }),
  },
  {
    id: "score-f16-store",
    title: "f16 attention scores (S)",
    summary:
      'The store of attentionScoreStorage: "f16" against host rounding (u32 identity): whether pack2x16float rounds like the host on this backend.',
    needsSubgroups: false,
    run: async (gpu, options) => ({
      parts: [await checkScoreF16Store(gpu, options)],
      expectedCases: SCORE_F16_CASES,
    }),
  },
];
