// Chained deferred runs (`GenerationRun.chain`) on a real GPU: the values.
//
// The invariant: chaining changes only how the runs are batched, never what they compute. Each
// chained run must read the logical lengths written **for it** — the context has one lengths uniform,
// so if a later run's write reached the queue before an earlier run's dispatches, the earlier run
// would attend over one row too many and its output would differ from the same step run on its own.
// The admission rules without a GPU are `runtime_chained_generation_test.ts`.

import { test as vitestTest } from "vite-plus/test";
import { assertEquals, assertRejects, assertStrictEquals, assertThrows } from "@std/assert";
import {
  acquireGpu,
  createSessionFromContainer,
  type ResidentTensor,
  type Tensor,
} from "../mod.ts";
import { ExecutionError } from "../src/runtime/plan.ts";
import {
  type DeclarationJson,
  GRAPH_NAME,
  openGraphModel,
  singleOpDeclaration,
} from "./helpers/model-fixture.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";

const test = (name: string, fn: () => Promise<void>): void =>
  vitestTest.skipIf(!GPU_AVAILABLE)(name, fn);

/** Sliding window 4 over 8 rows of capacity: the slack (rows that can stay uncommitted) is 4. */
const WINDOW = 4;
const CAPACITY = 8;

const graph = (): DeclarationJson => ({
  format: "karume-ir",
  version: 2,
  symbols: ["M"],
  requires: { ops: ["attention", "state_append"] },
  initializers: {},
  inputs: ["q", "k", "v"].map((name) => ({ name, dtype: "f32", shape: [1, 1, "M", 4] })),
  outputs: ["o"],
  values: { o: { dtype: "f32", shape: [1, 1, "M", 4] } },
  states: {
    kslot: { dtype: "f32", shape: [1, 1, CAPACITY, 4] },
    vslot: { dtype: "f32", shape: [1, 1, CAPACITY, 4] },
  },
  nodes: [
    {
      op: "attention",
      ins: ["q", "k", "v"],
      outs: ["o"],
      attrs: { scale: 0.5, window: WINDOW },
      states: { k: "kslot", v: "vslot" },
    },
    ...[
      ["k", "kslot"],
      ["v", "vslot"],
    ].map(([name, slot]) => ({
      op: "state_append",
      ins: [name],
      outs: [],
      attrs: { window: WINDOW },
      states: { slot },
    })),
  ],
});

/** Distinct inputs per step (`phase`), so a run that read another step's lengths shows. */
const inputs = (rows: number, phase: number): Record<string, Tensor> =>
  Object.fromEntries(
    ["q", "k", "v"].map((name, n) => [
      name,
      {
        dtype: "f32",
        shape: [1, 1, rows, 4],
        data: Float32Array.from(
          { length: rows * 4 },
          (_, i) => (((i * 7 + n * 3 + phase * 5) % 17) - 8) * 0.13,
        ),
      },
    ]),
  );

const bits = (data: ArrayBuffer | Float32Array): number[] =>
  Array.from(
    data instanceof Float32Array
      ? new Uint32Array(data.buffer, data.byteOffset, data.length)
      : new Uint32Array(data),
  );

const open = async () => {
  const gpu = await acquireGpu();
  const session = await createSessionFromContainer(gpu, await openGraphModel(graph()), GRAPH_NAME);
  return { gpu, session };
};

test("chain: the deferred runs of one batch compute what the same steps compute one by one", async () => {
  const { gpu, session } = await open();
  const reference = await session.createGenerationContext({ chunkLength: 4 });
  const chained = await session.createGenerationContext({ chunkLength: 4 });
  const sinks: ResidentTensor[] = [];
  try {
    for (let k = 0; k < 4; k += 1) sinks.push(await gpu.createResident(16));
    for (const context of [reference, chained]) {
      await session.run(inputs(4, 0), {}, { context, queryLength: 3 });
    }
    /** One batch of chained single-row runs, one sink per step. */
    const chain = async (phases: readonly number[]): Promise<number[][]> => {
      const batch = await gpu.beginBatch();
      try {
        for (const [k, phase] of phases.entries()) {
          await session.enqueue(inputs(1, phase), {
            batch,
            generation: { context: chained, queryLength: 1, commit: "deferred", chain: true },
            copyOutputs: { o: sinks[k] },
          });
        }
        const read = await batch.finishAndRead(
          Object.fromEntries(phases.map((_, k) => [`o${k}`, sinks[k]])),
        );
        return phases.map((_, k) => bits(read[`o${k}`]));
      } finally {
        await batch.finish();
      }
    };
    const step = async (phase: number): Promise<number[]> =>
      bits(
        (await session.run(inputs(1, phase), {}, { context: reference, queryLength: 1 })).o
          .data as Float32Array,
      );

    // Four steps (the whole slack) in one batch, all committed.
    const first = await chain([1, 2, 3, 4]);
    assertEquals(chained.pendingCommit, { pastLength: 3, queryLength: 4 });
    for (const [k, phase] of [1, 2, 3, 4].entries()) {
      assertEquals(first[k], await step(phase), `step at position ${3 + k}`);
    }
    chained.commit(4);
    assertEquals([chained.pastLength, reference.pastLength], [7, 7]);

    // Three steps, two committed: the next run overwrites the rejected row.
    const second = await chain([5, 6, 7]);
    chained.commit(2);
    assertEquals(second[0], await step(5));
    assertEquals(second[1], await step(6));
    assertEquals([chained.pastLength, reference.pastLength], [9, 9]);
    const after = await session.run(inputs(1, 8), {}, { context: chained, queryLength: 1 });
    assertEquals(bits(after.o.data as Float32Array), await step(8));

    // A chained `Session.run` continues a settled deferred run.
    const verify = await session.run(
      inputs(1, 9),
      {},
      { context: chained, queryLength: 1, commit: "deferred" },
    );
    const ahead = await session.run(
      inputs(1, 10),
      {},
      { context: chained, queryLength: 1, commit: "deferred", chain: true },
    );
    assertEquals(chained.pendingCommit, { pastLength: 10, queryLength: 2 });
    chained.commit(2);
    assertEquals(bits(verify.o.data as Float32Array), await step(9));
    assertEquals(bits(ahead.o.data as Float32Array), await step(10));
    assertEquals([chained.pastLength, reference.pastLength], [12, 12]);
  } finally {
    await reference.dispose();
    await chained.dispose();
    await session.dispose();
    for (const sink of sinks) sink.dispose();
    gpu.destroy();
  }
});

test("chain: refused without commit 'deferred', beyond the slack, and next to an unchained run", async () => {
  const { gpu, session } = await open();
  const context = await session.createGenerationContext({ chunkLength: 4 });
  try {
    await assertRejects(
      () => session.run(inputs(1, 0), {}, { context, queryLength: 1, chain: true }),
      ExecutionError,
      "needs commit 'deferred'",
    );
    await assertRejects(
      () =>
        session.run(
          inputs(1, 0),
          {},
          { context, queryLength: 1, commit: "deferred", chain: "yes" as unknown as boolean },
        ),
      ExecutionError,
      "must be a boolean",
    );
    const batch = await gpu.beginBatch();
    try {
      for (let phase = 0; phase < CAPACITY - WINDOW; phase += 1) {
        await session.enqueue(inputs(1, phase), {
          batch,
          generation: { context, queryLength: 1, commit: "deferred", chain: true },
        });
      }
      await assertRejects(
        () =>
          session.enqueue(inputs(1, 9), {
            batch,
            generation: { context, queryLength: 1, commit: "deferred", chain: true },
          }),
        ExecutionError,
        "slack 4",
      );
      await assertRejects(
        () => session.enqueue(inputs(1, 9), { batch, generation: { context, queryLength: 1 } }),
        ExecutionError,
        "in progress",
      );
      await batch.finish();
    } finally {
      await batch.finish();
    }
    assertEquals(context.pendingCommit, { pastLength: 0, queryLength: 4 });
    context.commit(0);
    assertEquals(context.pastLength, 0);
  } finally {
    await context.dispose();
    await session.dispose();
    gpu.destroy();
  }
});

test("chain: a failed member of the batch poisons a context its chained runs wrote", async () => {
  const { gpu, session } = await open();
  const context = await session.createGenerationContext({ chunkLength: 4 });
  const downstream = await createSessionFromContainer(
    gpu,
    await openGraphModel(singleOpDeclaration("neg", [[4]], [[4]])),
    GRAPH_NAME,
  );
  try {
    const batch = await gpu.beginBatch();
    for (const phase of [0, 1]) {
      await session.enqueue(inputs(1, phase), {
        batch,
        generation: { context, queryLength: 1, commit: "deferred", chain: true },
      });
    }
    const failure = await assertRejects(() =>
      downstream.enqueue({ x0: { dtype: "f32", shape: [1], data: Float32Array.of(1) } }, { batch }),
    );
    assertStrictEquals(await assertRejects(() => batch.finish()), failure);
    assertThrows(() => context.pastLength, ExecutionError, "poisoned");
    assertThrows(() => context.pendingCommit, ExecutionError, "poisoned");
  } finally {
    await context.dispose();
    await session.dispose();
    await downstream.dispose();
    gpu.destroy();
  }
});
