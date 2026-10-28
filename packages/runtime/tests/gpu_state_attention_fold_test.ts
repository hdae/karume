// Word-for-word identity gates of two bit-identical rewrites of the states-form decode attention
// kernels (ADR 0067 addendum of 2026-10-27):
//   - the folded ①' (F = r query heads of one kv plane per workgroup — perf-ledger K-55) against
//     ①' (F = 1, still in the product for every other plan). F = 8 is selected in the product;
//     F = 4 (E4B) is not — it differs on D3D12 — and stays here as the check perf-ledger K-95 needs
//   - the unrolled ③' and fused stats + PV (`attention_state_pv:v2` / `attention_state_stats_pv:v2`
//     — perf-ledger K-91) against their v1 WGSL, kept test-only in fixtures/wgsl-baseline/ (the
//     generators no longer emit it)
//
// The contract is bit identity, not a band: every word of S / O is compared as u32, including the
// poison left in the words a kernel must not write, NaN and −inf. The identity rests on the backend
// compiling the same expression the same way, which WGSL does not guarantee — this file is the check
// to run on each backend. Order-only mutants (a reassociated product, two terms swapped) must
// mismatch on the same harness, or a zero-mismatch result proves nothing.
// The cases, inputs, mutants and the word comparison live in @karume/kernel-checks, one
// implementation with the playground's /kernel-check page (the same check on a browser's backend:
// D3D12, Metal); this file asserts the counts it returns.
// NOTE: F = 8 matched on Dawn / Vulkan / RTX 3080 Ti and on Chrome / D3D12 / RTX 5070 Ti; F = 4
// matched on Vulkan and differed on D3D12 (2026-10-27 — perf-ledger K-95), so on D3D12 this test
// fails on the E4B cases until K-95 lands.
// NOTE: on Dawn / Vulkan / NVIDIA (2026-10-27) NaN payloads come out canonical and no output word is
// subnormal, consistent with flush-to-zero on this driver (observed by the independent check; not
// isolated), so the comparison cannot tell payloads or subnormal orderings apart there.

import { describe, it } from "vite-plus/test";
import { assert, assertEquals } from "@std/assert";
import {
  type CheckPart,
  checkStateAttentionMutants,
  checkStatePvUnroll,
  checkStateQkFold,
  checkStateStatsPvUnroll,
  STATE_ATTENTION_CASES,
  STATE_ATTENTION_MUTANTS,
} from "@karume/kernel-checks";
import { acquireGpu } from "../src/gpu/device.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";
import { assertCheckPart } from "./helpers/kernel-check-assert.ts";

/** The census requirement of a part (the word classes its reference outputs met). */
const censusOf = (part: CheckPart): string => {
  const census = part.requirements.find(({ id }) => id === "census");
  assert(census !== undefined, `${part.id}: no census requirement`);
  return census.detail;
};

describe.skipIf(!GPU_AVAILABLE)(
  "states attention kernels rewritten bit-identically (real GPU)",
  () => {
    it("the folded ①' writes the same S words as ①' (E2B / E4B / MiniCPM5 shapes, B = 2)", async () => {
      const gpu = await acquireGpu();
      try {
        // E2B and MiniCPM5 are the shapes the selection folds (at decode M = 1); E4B is excluded
        // (it differs on D3D12 — perf-ledger K-95) and stays as the check K-95 needs. The identity
        // is checked on every row shape, folded by the selection or not; the part's requirements
        // hold the selection and the S word classes the grid met (finite, −inf, NaN, poison)
        const part = await checkStateQkFold(gpu);
        assertEquals(part.cases.length, STATE_ATTENTION_CASES["state-qk-fold"]);
        assertCheckPart(part);
        console.log(
          `folded ①' vs ①': ${part.cases.length} cases, 0 mismatching words, S census ${censusOf(part)}`,
        );
      } finally {
        gpu.destroy();
      }
    });

    it("the unrolled ③' writes the same O words as the v1 ③' (full / sliding, GQA and not)", async () => {
      const gpu = await acquireGpu();
      try {
        // Requirements: exact zeros (pad rows, empty rows), NaN and finite outputs all occurred, and
        // no poison is left (③' writes every row of O)
        const part = await checkStatePvUnroll(gpu);
        assertEquals(part.cases.length, STATE_ATTENTION_CASES["state-pv-unroll"]);
        assertCheckPart(part);
        console.log(
          `unrolled ③' vs v1: ${part.cases.length} cases, 0 mismatching words, O census ${censusOf(part)}`,
        );
      } finally {
        gpu.destroy();
      }
    });

    it("the unrolled fused stats + PV writes the same O words as the v1 fused kernel", async () => {
      const gpu = await acquireGpu();
      try {
        // Sliding shapes, and full shapes whose column cap fits the fused kernel (≤ 1024)
        const part = await checkStateStatsPvUnroll(gpu);
        assertEquals(part.cases.length, STATE_ATTENTION_CASES["state-stats-pv-unroll"]);
        assertCheckPart(part);
        console.log(
          `unrolled fused vs v1: ${part.cases.length} cases, 0 mismatching words, O census ${censusOf(part)}`,
        );
      } finally {
        gpu.destroy();
      }
    });

    it("order-only mutants of the rewritten kernels mismatch (the word gate is not vacuous)", async () => {
      const gpu = await acquireGpu();
      try {
        const mutants = await checkStateAttentionMutants(gpu);
        assertEquals(mutants.length, STATE_ATTENTION_MUTANTS);
        for (const { label, kernel, differingWords, words } of mutants) {
          const [against, output] = kernel === "qk-fold" ? ["①'", "S"] : ["v1", "O"];
          assert(differingWords > 0, `${label}: the mutant matched ${against} word for word`);
          console.log(`mutant '${label}': ${differingWords} of ${words} ${output} words differ`);
        }
      } finally {
        gpu.destroy();
      }
    });
  },
);
