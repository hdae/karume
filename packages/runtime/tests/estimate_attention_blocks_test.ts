/**
 * Two rules of the estimate's attention accounting (no GPU).
 *
 * 1. **The states form copies execution's allocation program.** `buildStateAttention` allocates the
 *    score S and the row statistics both before ① QK and releases both after ③ PV; the estimate's
 *    program for the same node must place to the same total. The expected side here is built from
 *    the shared block plan (`planStateAttention`) in execution's order and placed by the same
 *    packer, over a family of shapes (full / sliding, one block / several blocks, prefill /
 *    decode). NOTE: with today's packer the order inside a block (stats allocated before or after
 *    ① QK) gives the same intervals, so this pins the totals, not that order.
 * 2. **Decomposed row-block attention needs the binding limit when its count depends on it**
 *    (ADR 0124 decision 7, the rule fused attention already follows): without
 *    `maxStorageBufferBindingSize` the estimate may count only shapes whose row-block count is the
 *    same on every device.
 */

import { assertEquals, assertThrows } from "@std/assert";
import { describe, it } from "vite-plus/test";
import { numel } from "../src/ops.ts";
import { ExecutionError } from "../src/runtime/plan.ts";
import { planStateAttention } from "../src/runtime/state-attention-plan.ts";
import {
  CORE_TRANSIENT_LIMITS,
  planTransients,
  type TransientDispatchSpec,
  type TransientTempSpec,
} from "../src/runtime/transient-plan.ts";
import {
  bothScenarios,
  decomposedAttentionGraph,
  openGraph,
  stateAttentionGraph,
} from "./helpers/estimate-graphs.ts";

// ---------------------------------------------------------------------------
// 1. states-form attention: estimate == execution's allocation program
// ---------------------------------------------------------------------------

/** The shape of `stateAttentionGraph`: B = 1, Hkv = 2, D = 8. */
const DEPTH = 8;

type StateCase = {
  readonly capacity: number;
  readonly chunkLength: number;
  readonly heads: number;
  readonly window?: number;
  readonly limit: number;
};

/**
 * The workspace of one scenario of `stateAttentionGraph` as execution plans it: the output `o`,
 * then per row block S and the row statistics allocated before ① QK (`allocTemp` twice, in that
 * order), the three dispatches with execution's bindings, and both released after ③ PV. The two
 * `state_append` steps write only state slots (no transient slot).
 */
const executionWorkspace = (spec: StateCase, rows: number): number => {
  const { blocks } = planStateAttention(
    { batchHeads: spec.heads, chunkRows: rows, capacity: spec.capacity, window: spec.window },
    spec.limit,
  );
  const temps: TransientTempSpec[] = [];
  const dispatches: TransientDispatchSpec[] = [];
  const output = { kind: "value", name: "o" } as const;
  for (const block of blocks) {
    const base = dispatches.length;
    const scores = { kind: "temp", id: temps.length } as const;
    temps.push({ byteLength: block.scoreBytes, allocBefore: base, releaseAfter: base + 2 });
    const stats = { kind: "temp", id: temps.length } as const;
    temps.push({ byteLength: block.statsBytes, allocBefore: base, releaseAfter: base + 2 });
    const q = { kind: "value", name: "q" } as const;
    const k = { kind: "value", name: "k" } as const;
    const v = { kind: "value", name: "v" } as const;
    dispatches.push({ reads: [q, k], writes: [scores] });
    dispatches.push({ reads: [scores], writes: [stats] });
    dispatches.push({ reads: [scores, stats, v], writes: [output] });
  }
  const append = (name: string) => ({
    outputs: [],
    temps: [],
    dispatches: [{ reads: [{ kind: "value", name } as const], writes: [] }],
    releases: [name],
  });
  return planTransients(
    [
      {
        outputs: [
          {
            name: "o",
            kind: "alloc",
            byteLength: numel([1, spec.heads, rows, DEPTH]) * 4,
            uses: 0,
            pinned: true,
          },
        ],
        temps,
        dispatches,
        releases: ["q", "k", "v"],
      },
      append("k"),
      append("v"),
    ],
    {
      maxBufferSize: Math.max(CORE_TRANSIENT_LIMITS.maxBufferSize, spec.limit),
      maxStorageBufferBindingSize: spec.limit,
      offsetAlignment: CORE_TRANSIENT_LIMITS.offsetAlignment,
    },
  ).totalBytes;
};

const WIDE_LIMIT = 1 << 20;

const STATE_CASES: readonly StateCase[] = [
  // One block (the limit never binds).
  { capacity: 16, chunkLength: 4, heads: 4, limit: WIDE_LIMIT },
  { capacity: 64, chunkLength: 4, heads: 4, limit: WIDE_LIMIT },
  { capacity: 16, chunkLength: 4, heads: 4, window: 6, limit: WIDE_LIMIT },
  // Several blocks: one S row is H·colCap·4 = 16·64·4 = 4096B, so a 4096B limit gives one row per
  // block (4 blocks at prefill); 8192B gives two.
  { capacity: 64, chunkLength: 4, heads: 16, limit: 4096 },
  { capacity: 64, chunkLength: 4, heads: 16, limit: 8192 },
  // Sliding with several blocks: colCap = W − 1 + M = 9, one row 32·9·4 = 1152B, so 2 + 2 rows
  // under 4096B (the k / v slots, 2·64·8·4 = 4096B, and the output o, 32·4·8·4 = 4096B, still bind).
  { capacity: 64, chunkLength: 4, heads: 32, window: 6, limit: 4096 },
];

describe("states-form attention: the estimate places execution's allocation program", () => {
  for (const spec of STATE_CASES) {
    const label = `C ${spec.capacity} · M ${spec.chunkLength} · H ${spec.heads}${
      spec.window === undefined ? "" : ` · W ${spec.window}`
    } · limit ${spec.limit}B`;
    it(`${label}: prefill and decode workspace equal execution's plan`, () => {
      const { prefill, decode } = bothScenarios(
        openGraph(stateAttentionGraph(spec.window, spec.heads)).estimate({
          generation: { chunkLength: spec.chunkLength, bindings: { C: spec.capacity } },
          maxStorageBufferBindingSize: spec.limit,
        }),
      );
      assertEquals(prefill.workspaceBytes, executionWorkspace(spec, spec.chunkLength));
      assertEquals(decode.workspaceBytes, executionWorkspace(spec, 1));
    });
  }
});

// ---------------------------------------------------------------------------
// 2. decomposed row-block attention without the binding limit
// ---------------------------------------------------------------------------

const CORE = CORE_TRANSIENT_LIMITS.maxStorageBufferBindingSize;

/** S = 4·2048·4200·4 = 137,625,600B > the core default 128 MiB: two row blocks under it. */
const SPLIT_UNDER_CORE = { heads: 4, queries: 2048, keys: 4200, headDim: 8 } as const;
/** S = 4·1024·4096·4 = 64 MiB: one block under the core default, so one block on every device. */
const ONE_BLOCK_UNDER_CORE = { heads: 4, queries: 1024, keys: 4096, headDim: 8 } as const;

describe("decomposed row-block attention: the estimate needs the binding limit when the count depends on it", () => {
  it("fails loudly without the limit when S splits under the core default", () => {
    assertThrows(
      () => openGraph(decomposedAttentionGraph(SPLIT_UNDER_CORE)).estimate(),
      ExecutionError,
      "options.maxStorageBufferBindingSize is needed",
    );
  });

  it("fails loudly with a budget above the core default (the limit still decides the count)", () => {
    assertThrows(
      () =>
        openGraph(decomposedAttentionGraph(SPLIT_UNDER_CORE)).estimate({
          attentionRowBlockMaxBytes: CORE + 1,
        }),
      ExecutionError,
      "options.maxStorageBufferBindingSize is needed",
    );
  });

  it("counts the same shape once the limit is passed", () => {
    const { scenarios } = openGraph(decomposedAttentionGraph(SPLIT_UNDER_CORE)).estimate({
      maxStorageBufferBindingSize: CORE,
    });
    assertEquals(scenarios.length, 1);
  });

  it("counts without the limit when the budget is at most the core default", () => {
    const { scenarios } = openGraph(decomposedAttentionGraph(SPLIT_UNDER_CORE)).estimate({
      attentionRowBlockMaxBytes: CORE,
    });
    assertEquals(scenarios.length, 1);
  });

  it("counts without the limit when S fits in one block under the core default", () => {
    const report = openGraph(decomposedAttentionGraph(ONE_BLOCK_UNDER_CORE)).estimate();
    const withLimit = openGraph(decomposedAttentionGraph(ONE_BLOCK_UNDER_CORE)).estimate({
      maxStorageBufferBindingSize: CORE,
    });
    // The same number as a device granted exactly the core default.
    assertEquals(report.scenarios[0].workspaceBytes, withLimit.scenarios[0].workspaceBytes);
  });
});
