// The band of plain attention (`attrs.band` — ADR 0155) on the real GPU.
//
// The evidence of class E is the shared check of `@karume/kernel-checks` (the same cases the
// workbench's kernel-check entry runs): every O word of the band node equals the same node fed the
// band as an additive 0 / −inf mask, over the scalar and vec4 ①QK, GQA and not, T off the tile,
// bands below, at and far above T and 0, several row blocks, f16 score storage and M ≠ N both ways
// (M > N with rows that see no column). This file adds
// what is no twin on the workbench's device or no twin at all:
//
// - the fault injection: the band side declaring band ± 1 must mismatch on every case where that
//   changes which columns are visible (a check that also matched an off-by-one band would see
//   nothing);
// - the f16 compute cases (`attentionCompute: "f16"` needs a device acquired with shader-f16);
// - the band node against the CPU reference (the semantics, independently of the mask plan);
// - the refusal of band × a8 (the three a8 dispatches and the online kernel alike).

import { describe, it } from "vite-plus/test";
import { assert, assertEquals, assertRejects } from "@std/assert";
import {
  ATTENTION_BAND_CASES,
  ATTENTION_BAND_GRID,
  type AttentionBandCase,
  attentionBandAttrs,
  attentionBandFeeds,
  attentionBandGraph,
  attentionBandKeys,
  checkAttentionBand,
  runAttentionBandCase,
} from "@karume/kernel-checks";
import { acquireGpu } from "../src/gpu/device.ts";
import { compareTensors, formatAllclose } from "../src/reference/allclose.ts";
import { opTolerance } from "../src/reference/op-tolerance.ts";
import { referenceAttention } from "../src/reference/ops.ts";
import { createSessionFromContainer, type SessionOptions } from "../src/runtime/executor.ts";
import { ExecutionError } from "../src/runtime/plan.ts";
import { GPU_AVAILABLE, SHADER_F16_AVAILABLE } from "./helpers/gpu.ts";
import { assertCheckCase, assertCheckPart } from "./helpers/kernel-check-assert.ts";
import { GRAPH_NAME, openGraphModel } from "./helpers/model-fixture.ts";

/**
 * Whether declaring `band + offset` changes which columns the case's rows see: every distance up to
 * max(M, N) − 1 occurs, so a band beyond it hides nothing more.
 */
const changesVisibility = (testCase: AttentionBandCase, offset: number): boolean => {
  const widest = Math.max(testCase.tokens, attentionBandKeys(testCase)) - 1;
  const seen = (band: number): number => Math.min(band, widest);
  return seen(testCase.band + offset) !== seen(testCase.band);
};

/** The f16 compute cases (scalar and vec4, GQA, several row blocks). */
const C16_CASES: readonly AttentionBandCase[] = [
  { ...ATTENTION_BAND_GRID[0], name: "c16 scalar H2 T37 D12 band 5", c16: true },
  {
    ...ATTENTION_BAND_GRID[8],
    name: "c16 vec4 GQA H6:2 T136 D16 band 9 / 3 row blocks",
    c16: true,
  },
];

describe.skipIf(!GPU_AVAILABLE)("the attention band (real GPU)", () => {
  it("writes the words of the additive band mask on every case of the shared check", async () => {
    const gpu = await acquireGpu();
    try {
      const part = await checkAttentionBand(gpu);
      assertEquals(part.cases.length, ATTENTION_BAND_CASES["attention-band"]);
      assertCheckPart(part);
    } finally {
      gpu.destroy();
    }
  });

  it("the comparison can fail: a band off by one mismatches wherever it changes the visible columns", async () => {
    const gpu = await acquireGpu();
    try {
      let faulted = 0;
      for (const testCase of ATTENTION_BAND_GRID) {
        for (const offset of [1, -1]) {
          if (testCase.band + offset < 0 || !changesVisibility(testCase, offset)) continue;
          faulted += 1;
          const result = await runAttentionBandCase(gpu, testCase, offset);
          const output = result.criteria.find((criterion) => criterion.name === "output o");
          assert(output !== undefined, `${testCase.name}: no output criterion`);
          assert(
            output.failures > 0,
            `${testCase.name} band ${offset > 0 ? "+" : ""}${offset}: no word differed`,
          );
        }
      }
      // The eleven grid cases with 0 < band < max(M, N) − 1 take both faults, band 0 takes +1, and
      // the two that hide nothing (band ≥ T − 1 either way) take none.
      assertEquals(faulted, 11 * 2 + 1, "faults injected");
    } finally {
      gpu.destroy();
    }
  });

  it("agrees with the CPU reference within attention's op tolerance on every banded case", async () => {
    const gpu = await acquireGpu();
    try {
      for (const testCase of ATTENTION_BAND_GRID) {
        const session = await createSessionFromContainer(
          gpu,
          await openGraphModel(attentionBandGraph(testCase, testCase.band)),
          GRAPH_NAME,
        );
        try {
          const { q, k, v } = attentionBandFeeds(testCase);
          const out = (await session.run({ q, k, v }))["o"];
          const expected = referenceAttention(q, k, v, attentionBandAttrs(testCase, testCase.band));
          const report = compareTensors(out, expected, opTolerance("attention"));
          assert(report.pass, `${testCase.name}: ${formatAllclose(report)}`);
        } finally {
          await session.dispose();
        }
      }
    } finally {
      gpu.destroy();
    }
  });

  it("refuses band × a8, on the three a8 dispatches and on the online kernel alike", async () => {
    const gpu = await acquireGpu({ subgroups: "available" });
    try {
      // D = 64: eligible for i8a8 in both stages and for the online kernel.
      const testCase: AttentionBandCase = {
        name: "a8 H2 T40 D64 band 7",
        batch: 1,
        heads: 2,
        kvHeads: 2,
        tokens: 40,
        depth: 64,
        band: 7,
        blocks: 1,
      };
      const { q, k, v } = attentionBandFeeds(testCase);
      for (const options of [
        { attentionCompute: "a8" },
        { attentionCompute: "a8", attentionScoreStorage: "none" },
      ] satisfies SessionOptions[]) {
        await assertRejects(
          async () => {
            const session = await createSessionFromContainer(
              gpu,
              await openGraphModel(attentionBandGraph(testCase, testCase.band)),
              GRAPH_NAME,
              options,
            );
            try {
              await session.run({ q, k, v });
            } finally {
              await session.dispose();
            }
          },
          ExecutionError,
          "attention with attrs.band cannot be combined with attentionCompute 'a8'",
        );
      }
    } finally {
      gpu.destroy();
    }
  });
});

describe.skipIf(!GPU_AVAILABLE || !SHADER_F16_AVAILABLE)(
  "the attention band under f16 compute (real GPU with shader-f16)",
  () => {
    it("writes the words of the additive band mask, both sides rounding S to f16 after the epilogue", async () => {
      const gpu = await acquireGpu({ shaderF16: true });
      try {
        for (const testCase of C16_CASES)
          assertCheckCase(await runAttentionBandCase(gpu, testCase));
      } finally {
        gpu.destroy();
      }
    });
  },
);
