import { test as vitestTest } from "vite-plus/test";
import { assert, assertEquals, assertRejects, assertThrows } from "@std/assert";
import {
  acquireGpu,
  createSessionFromContainer,
  type GpuContext,
  type ResidentTensor,
  type Session,
  type Tensor,
} from "@karume/runtime";
import { createGemmaGreedyOutput } from "../src/gemma/greedy-output.ts";
import {
  type DeclarationJson,
  GRAPH_NAME,
  openModelBytes,
} from "../../runtime/tests/helpers/model-fixture.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";
const test = (name: string, fn: () => Promise<void>): void =>
  vitestTest.skipIf(!GPU_AVAILABLE)(name, fn);
const model = (): DeclarationJson => ({
  format: "karume-ir",
  version: 2,
  symbols: [],
  requires: { ops: ["neg", "reshape", "state_append"] },
  initializers: {},
  inputs: [{ name: "x", dtype: "f32", shape: [1, 1, 4] }],
  outputs: ["y"],
  values: { y: { dtype: "f32", shape: [1, 1, 4] }, z: { dtype: "f32", shape: [1, 1, 1, 4] } },
  states: { slot: { dtype: "f32", shape: [1, 1, 16, 4] } },
  nodes: [
    { op: "neg", ins: ["x"], outs: ["y"], attrs: {} },
    { op: "reshape", ins: ["x"], outs: ["z"], attrs: {} },
    { op: "state_append", ins: ["z"], outs: [], attrs: {}, states: { slot: "slot" } },
  ],
});
/** Writes a graph with 0 weights as `krm`, opens it and makes it the target Session. */
const targetSession = async (gpu: GpuContext): Promise<Session> =>
  await createSessionFromContainer(gpu, await openModelBytes(model(), []), GRAPH_NAME);
const input = (values = [-1, -4, -4, -2]): Record<string, Tensor> => ({
  x: { dtype: "f32", shape: [1, 1, 4], data: Float32Array.from(values) },
});
test("greedy output: a normal run and a batch issued concurrently on separate contexts are serialised", async () => {
  const gpu = await acquireGpu(),
    target = await targetSession(gpu);
  const h = createGemmaGreedyOutput(gpu, target, "y", 4);
  const a = await h.session.createGenerationContext({ chunkLength: 1 }),
    b = await h.session.createGenerationContext({ chunkLength: 1 });
  try {
    for (let i = 0; i < 3; i++) {
      const tasks = [
        () => h.session.run(input(), undefined, { context: a, queryLength: 1 }),
        () => h.greedy(input(), { context: b, queryLength: 1 }),
      ];
      if (i % 2) tasks.reverse();
      const results = await Promise.all(tasks.map((run) => run()));
      const result = results[i % 2 ? 0 : 1];
      assertEquals(result, { value: 4, index: 1 });
      assertEquals([a.pastLength, b.pastLength], [i + 1, i + 1]);
    }
    assert(h.extraBytes >= 16 + 8);
  } finally {
    await a.dispose();
    await b.dispose();
    await h.dispose();
    await target.dispose();
    gpu.destroy();
  }
});
test("greedy output: a failed preparation returns the partial allocations, and the next issue retries", async () => {
  const gpu = await acquireGpu(),
    target = await targetSession(gpu);
  const h = createGemmaGreedyOutput(gpu, target, "y", 4),
    context = await h.session.createGenerationContext({ chunkLength: 1 });
  const create = gpu.createResident.bind(gpu),
    made: ResidentTensor[] = [];
  let calls = 0;
  gpu.createResident = async (...args) => {
    if (++calls === 2) throw Error("resident fault");
    const r = await create(...args);
    made.push(r);
    return r;
  };
  try {
    await assertRejects(
      () => h.greedy(input(), { context, queryLength: 1 }),
      Error,
      "resident fault",
    );
    assertEquals(made.length, 1);
    assert(made[0].disposed);
    assertEquals(context.pastLength, 0);
    gpu.createResident = create;
    assertEquals(await h.greedy(input(), { context, queryLength: 1 }), { value: 4, index: 1 });
  } finally {
    gpu.createResident = create;
    await context.dispose();
    await h.dispose();
    await target.dispose();
    gpu.destroy();
  }
});
test("greedy output: dispose after an un-awaited generation waits for completion and returns every resident", async () => {
  const gpu = await acquireGpu(),
    target = await targetSession(gpu);
  const h = createGemmaGreedyOutput(gpu, target, "y", 4),
    context = await h.session.createGenerationContext({ chunkLength: 1 });
  const create = gpu.createResident.bind(gpu),
    made: ResidentTensor[] = [];
  gpu.createResident = async (...args) => {
    const r = await create(...args);
    made.push(r);
    return r;
  };
  try {
    const pending = h.greedy(input(), { context, queryLength: 1 }),
      disposed = h.dispose();
    assertThrows(() => h.greedy(input(), { context, queryLength: 1 }));
    assertEquals(await pending, { value: 4, index: 1 });
    await disposed;
    assertEquals(made.length, 3);
    assert(made.every((r) => r.disposed));
    assertEquals(context.pastLength, 1);
  } finally {
    gpu.createResident = create;
    await context.dispose();
    await h.dispose();
    await target.dispose();
    gpu.destroy();
  }
});
