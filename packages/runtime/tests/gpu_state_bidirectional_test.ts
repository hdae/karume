// The bidirectional block of `GenerationRun.bidirectionalRows` (ADR 0144 decision 3): the kernel's
// words (the check in @karume/kernel-checks, shared with /kernel-check), and the run through a real
// Session against the CPU reference, with the plans it selects and the runs it refuses.

import { describe, it } from "vite-plus/test";
import { assert, assertEquals, assertRejects } from "@std/assert";
import {
  checkStateBidirectional,
  STATE_BIDIRECTIONAL_CASES,
  stateBidirectionalMutants,
} from "@karume/kernel-checks";
import { acquireGpu, type GpuContext } from "../src/gpu/device.ts";
import { stateQkTiledKey } from "../src/kernels/state-attention.ts";
import { createSessionFromContainer, type Session, type Tensor } from "../src/runtime/executor.ts";
import { ExecutionError } from "../src/runtime/plan.ts";
import { compareTensors, formatAllclose } from "../src/reference/allclose.ts";
import { referenceStateAppend, referenceStateAttention } from "../src/reference/state-attention.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";
import { assertCheckPart } from "./helpers/kernel-check-assert.ts";
import { type DeclarationJson, GRAPH_NAME, openGraphModel } from "./helpers/model-fixture.ts";
import { requireCensus } from "./helpers/pipeline-census.ts";
import { halfScale, seeded } from "./helpers/state-dispatch.ts";

const HEADS = 2;
const DEPTH = 4;
const WINDOW = 32;
const CAPACITY = 40;
const ROWS = 16;

/** One sliding layer, one attention and its two appends. */
const graph = (): DeclarationJson => ({
  format: "karume-ir",
  version: 2,
  requires: { ops: ["attention", "state_append"] },
  symbols: ["M"],
  inputs: [
    { name: "q", dtype: "f32", shape: [1, HEADS, "M", DEPTH] },
    { name: "k", dtype: "f32", shape: [1, HEADS, "M", DEPTH] },
    { name: "v", dtype: "f32", shape: [1, HEADS, "M", DEPTH] },
  ],
  outputs: ["o"],
  initializers: {},
  values: { o: { dtype: "f32", shape: [1, HEADS, "M", DEPTH] } },
  states: {
    kslot: { dtype: "f32", shape: [1, HEADS, CAPACITY, DEPTH] },
    vslot: { dtype: "f32", shape: [1, HEADS, CAPACITY, DEPTH] },
  },
  nodes: [
    {
      op: "attention",
      ins: ["q", "k", "v"],
      outs: ["o"],
      attrs: { scale: halfScale(DEPTH), window: WINDOW },
      states: { k: "kslot", v: "vslot" },
    },
    {
      op: "state_append",
      ins: ["k"],
      outs: [],
      attrs: { window: WINDOW },
      states: { slot: "kslot" },
    },
    {
      op: "state_append",
      ins: ["v"],
      outs: [],
      attrs: { window: WINDOW },
      states: { slot: "vslot" },
    },
  ],
});

const valuesOf = (rows: number, salt: number): Float32Array<ArrayBuffer> =>
  seeded(HEADS * rows * DEPTH, (i) => (((i * 7 + salt) % 23) - 11) * 0.17);

const tensor = (rows: number, data: Float32Array<ArrayBuffer>): Tensor => ({
  dtype: "f32",
  shape: [1, HEADS, rows, DEPTH],
  data,
});

const withSession = async (work: (gpu: GpuContext, session: Session) => Promise<void>) => {
  const gpu = await acquireGpu();
  try {
    const session = await createSessionFromContainer(
      gpu,
      await openGraphModel(graph()),
      GRAPH_NAME,
    );
    try {
      await work(gpu, session);
    } finally {
      await session.dispose();
    }
  } finally {
    gpu.destroy();
  }
};

describe.skipIf(!GPU_AVAILABLE)("the bidirectional block (real GPU)", () => {
  it("writes the words of the predicate on every case", async () => {
    const gpu = await acquireGpu();
    try {
      const part = await checkStateBidirectional(gpu);
      assertEquals(part.cases.length, STATE_BIDIRECTIONAL_CASES["state-bidirectional"]);
      assertCheckPart(part);
    } finally {
      gpu.destroy();
    }
  });

  it("the comparison can fail: each predicate fault mismatches", async () => {
    const gpu = await acquireGpu();
    try {
      for (const { label, failures } of await stateBidirectionalMutants(gpu)) {
        assert(failures > 0, `${label}: no word differed`);
      }
    } finally {
      gpu.destroy();
    }
  });

  it("runs through a Session like the CPU reference, on the variant only for that run", async () => {
    await withSession(async (_gpu, session) => {
      const context = await session.createGenerationContext({ chunkLength: ROWS });
      try {
        const zeros = (): Float32Array<ArrayBuffer> => new Float32Array(HEADS * CAPACITY * DEPTH);
        let slotK = zeros();
        let slotV = zeros();
        let past = 0;
        const step = async (salt: number, block?: { start: number; end: number }) => {
          const q = valuesOf(ROWS, salt);
          const k = valuesOf(ROWS, salt + 1);
          const v = valuesOf(ROWS, salt + 2);
          const outputs = await session.run(
            { q: tensor(ROWS, q), k: tensor(ROWS, k), v: tensor(ROWS, v) },
            {},
            {
              context,
              queryLength: ROWS,
              ...(block === undefined ? {} : { bidirectionalRows: block }),
            },
          );
          const common = {
            chunkRows: ROWS,
            depth: DEPTH,
            capacity: CAPACITY,
            window: WINDOW,
            past,
            query: ROWS,
          };
          const expected = referenceStateAttention({
            ...common,
            batch: 1,
            heads: HEADS,
            kvHeads: HEADS,
            q,
            insK: k,
            insV: v,
            slotK,
            slotV,
            scale: halfScale(DEPTH),
            ...(block === undefined ? {} : { bidirectionalRows: block }),
          });
          const report = compareTensors(
            { dtype: "f32", data: outputs["o"].data },
            { dtype: "f32", data: expected.data },
            { atol: 5e-6, rtol: 0 },
          );
          assertEquals(report.pass, true, formatAllclose(report));
          const append = { ...common, kvPlanes: HEADS };
          slotK = referenceStateAppend({ ...append, x: k, slot: slotK })
            .data as Float32Array<ArrayBuffer>;
          slotV = referenceStateAppend({ ...append, x: v, slot: slotV })
            .data as Float32Array<ArrayBuffer>;
          past += ROWS;
          return requireCensus(session.diagnostics().lastRunPipelines, `salt ${salt}`).map(
            (row) => row.key,
          );
        };
        const variant = stateQkTiledKey(true, false, ROWS, "f32", true);
        const plain = stateQkTiledKey(true, false, ROWS);
        const text = await step(3);
        assert(text.includes(plain) && !text.includes(variant), `text run: ${text.join(" / ")}`);
        const image = await step(11, { start: 2, end: 14 });
        assert(image.includes(variant) && !image.includes(plain), `block: ${image.join(" / ")}`);
        const after = await step(19);
        assert(after.includes(plain) && !after.includes(variant), `after: ${after.join(" / ")}`);
      } finally {
        await context.dispose();
      }
    });
  });

  it("moves the output of a row inside the block, and only inside it", async () => {
    await withSession(async (_gpu, session) => {
      const run = async (block?: { start: number; end: number }) => {
        const context = await session.createGenerationContext({ chunkLength: ROWS });
        try {
          const outputs = await session.run(
            {
              q: tensor(ROWS, valuesOf(ROWS, 3)),
              k: tensor(ROWS, valuesOf(ROWS, 4)),
              v: tensor(ROWS, valuesOf(ROWS, 5)),
            },
            {},
            {
              context,
              queryLength: ROWS,
              ...(block === undefined ? {} : { bidirectionalRows: block }),
            },
          );
          return outputs["o"].data as Float32Array<ArrayBuffer>;
        } finally {
          await context.dispose();
        }
      };
      const causal = await run();
      const block = await run({ start: 4, end: 8 });
      const rowDiffers = (head: number, row: number): boolean => {
        const at = (head * ROWS + row) * DEPTH;
        return causal.subarray(at, at + DEPTH).some((value, d) => value !== block[at + d]);
      };
      for (let row = 0; row < ROWS; row += 1) {
        // The block's last row has nothing after it inside the block.
        const expected = row >= 4 && row < 7;
        assertEquals(rowDiffers(0, row), expected, `row ${row}`);
      }
    });
  });

  it("refuses what is not one block of an immediate run", async () => {
    await withSession(async (_gpu, session) => {
      const context = await session.createGenerationContext({ chunkLength: ROWS });
      const inputs = {
        q: tensor(ROWS, valuesOf(ROWS, 3)),
        k: tensor(ROWS, valuesOf(ROWS, 4)),
        v: tensor(ROWS, valuesOf(ROWS, 5)),
      };
      try {
        for (const [label, generation] of [
          ["empty", { queryLength: ROWS, bidirectionalRows: { start: 4, end: 4 } }],
          ["past the run", { queryLength: 8, bidirectionalRows: { start: 4, end: 9 } }],
          [
            "deferred",
            { queryLength: 4, commit: "deferred", bidirectionalRows: { start: 0, end: 4 } },
          ],
        ] as const) {
          await assertRejects(
            () => session.run(inputs, {}, { context, ...generation }),
            ExecutionError,
            "bidirectionalRows",
            label,
          );
        }
      } finally {
        await context.dispose();
      }
    });
  });

  it("refuses a plan below ①ₜ's 16 rows", async () => {
    await withSession(async (_gpu, session) => {
      const context = await session.createGenerationContext({
        chunkLength: ROWS,
        chunkBuckets: [8],
      });
      try {
        await assertRejects(
          () =>
            session.run(
              {
                q: tensor(8, valuesOf(8, 3)),
                k: tensor(8, valuesOf(8, 4)),
                v: tensor(8, valuesOf(8, 5)),
              },
              {},
              { context, queryLength: 8, bidirectionalRows: { start: 0, end: 8 } },
            ),
          ExecutionError,
          "at least 16 rows",
        );
      } finally {
        await context.dispose();
      }
    });
  });
});
