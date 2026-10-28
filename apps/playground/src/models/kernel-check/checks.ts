/**
 * The checks the page runs: the shared implementations of `@karume/kernel-checks` (the same cases
 * the runtime's node GPU tests run), with the page's titles and what each needs of the device.
 */

import {
  checkGemvSubgroup,
  type CheckOptions,
  checkRmsSubgroup,
  checkStateAttentionRewrites,
  GEMV_SUBGROUP_CASES,
  RMS_SUBGROUP_CASES,
  STATE_ATTENTION_CASES,
  STATE_ATTENTION_MUTANTS,
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
];
