import { test as vitestTest } from "vite-plus/test";
import { assertEquals, assertRejects, assertStrictEquals, assertThrows } from "@std/assert";
import { acquireGpu, createSessionFromContainer, type Tensor } from "../mod.ts";
import { RUNTIME_INTERNAL } from "../src/gpu/device.ts";
import {
  type DeclarationJson,
  GRAPH_NAME,
  openGraphModel,
  singleOpDeclaration,
} from "./helpers/model-fixture.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";
import { countFences } from "./helpers/fences.ts";
const graph = (window?: number): DeclarationJson => {
  const attrs = window === undefined ? {} : { window };
  return {
    format: "karume-ir",
    version: 2,
    symbols: ["M"],
    requires: { ops: ["attention", "state_append"] },
    initializers: {},
    inputs: ["q", "k", "v"].map((name) => ({ name, dtype: "f32", shape: [1, 1, "M", 4] })),
    outputs: ["o"],
    values: { o: { dtype: "f32", shape: [1, 1, "M", 4] } },
    states: {
      kslot: { dtype: "f32", shape: [1, 1, window === undefined ? 16 : 8, 4] },
      vslot: { dtype: "f32", shape: [1, 1, window === undefined ? 16 : 8, 4] },
    },
    nodes: [
      {
        op: "attention",
        ins: ["q", "k", "v"],
        outs: ["o"],
        attrs: { scale: 0.5, ...attrs },
        states: { k: "kslot", v: "vslot" },
      },
      ...[
        ["k", "kslot"],
        ["v", "vslot"],
      ].map(([name, slot]) => ({
        op: "state_append",
        ins: [name],
        outs: [],
        attrs,
        states: { slot },
      })),
    ],
  };
};
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
const open = async (window?: number) => {
  const gpu = await acquireGpu(),
    session = await createSessionFromContainer(
      gpu,
      await openGraphModel(graph(window)),
      GRAPH_NAME,
    );
  return { gpu, session };
};
const test = (name: string, fn: () => Promise<void>): void =>
  vitestTest.skipIf(!GPU_AVAILABLE)(name, fn);
for (const window of [undefined, 4]) {
  test(`batch generation: ${window === undefined ? "full" : "sliding"} prefill and decode match a plain run as u32`, async () => {
    const { gpu, session } = await open(window),
      a = await session.createGenerationContext({ chunkLength: 4 }),
      b = await session.createGenerationContext({ chunkLength: 4 });
    const sinks = [await gpu.createResident(64), await gpu.createResident(16)];
    try {
      let past = 0;
      for (let step = 0; step < 7; step++) {
        const rows = step === 0 ? 4 : 1,
          q = step === 0 ? 3 : 1,
          x = inputs(rows, step),
          sink = sinks[step === 0 ? 0 : 1];
        const ref = await session.run(x, {}, { context: a, queryLength: q });
        const batch = await gpu.beginBatch(),
          fences = countFences(gpu);
        try {
          await session.enqueue(x, {
            batch,
            generation: { context: b, queryLength: q },
            copyOutputs: { o: sink },
          });
          assertEquals(b.pastLength, past);
          if (step === 0) {
            await batch.settle();
            assertEquals(b.pastLength, past);
            assertThrows(() => b.rewind(0));
          }
          const result = await batch.finishAndRead({ o: sink });
          const data = ref.o.data;
          assertEquals(
            Array.from(new Uint32Array(result.o)),
            Array.from(new Uint32Array(data.buffer, data.byteOffset, data.length)),
          );
          assertEquals(fences.count(), step === 0 ? 1 : 0);
          past += q;
          assertEquals(b.pastLength, past);
        } finally {
          fences.restore();
          await batch.finish();
        }
      }
      if (window === undefined) {
        a.rewind(2);
        b.rewind(2);
        const x = inputs(1, 15),
          ref = await session.run(x, {}, { context: a, queryLength: 1 });
        const batch = await gpu.beginBatch();
        await session.enqueue(x, {
          batch,
          generation: { context: b, queryLength: 1 },
          copyOutputs: { o: sinks[1] },
        });
        assertEquals(
          Array.from(new Float32Array((await batch.finishAndRead({ o: sinks[1] })).o)),
          Array.from(ref.o.data),
        );
      }
    } finally {
      await a.dispose();
      await b.dispose();
      await session.dispose();
      for (const sink of sinks) sink.dispose();
      gpu.destroy();
    }
  });
}

test("batch generation: keeps the snapshot taken at issue and rejects a concurrent run, rewind or dispose of the same context", async () => {
  const { gpu, session } = await open(),
    a = await session.createGenerationContext({ chunkLength: 4 }),
    b = await session.createGenerationContext({ chunkLength: 4 }),
    sink = await gpu.createResident(16);
  try {
    const generation: { context: typeof a; queryLength: number; commit: "immediate" | "deferred" } =
      { context: a, queryLength: 1, commit: "immediate" };
    const batch = await gpu.beginBatch(),
      pending = session.enqueue(inputs(1, 0), { batch, generation, copyOutputs: { o: sink } });
    generation.context = b;
    generation.queryLength = 99;
    generation.commit = "deferred";
    assertThrows(() => a.rewind(0));
    await assertRejects(() => a.dispose());
    await assertRejects(() => session.dispose());
    await assertRejects(() => session.run(inputs(1, 0), {}, { context: a, queryLength: 1 }));
    await assertRejects(() =>
      session.enqueue(inputs(1, 0), { batch, generation: { context: a, queryLength: 1 } }),
    );
    const read = batch.finishAndRead({ o: sink });
    await Promise.all([pending, read]);
    assertEquals(a.pastLength, 1);
    assertEquals(b.pastLength, 0);
    a.rewind(0);
    const next = await session.run(inputs(1, 1), {}, { context: a, queryLength: 1 });
    assertEquals(next.o.data.length, 4);
  } finally {
    await a.dispose();
    await b.dispose();
    await session.dispose();
    sink.dispose();
    gpu.destroy();
  }
});

test("batch generation: deferred can be committed only after the batch succeeds", async () => {
  const { gpu, session } = await open(4),
    context = await session.createGenerationContext({ chunkLength: 4 }),
    sink = await gpu.createResident(64);
  try {
    const batch = await gpu.beginBatch();
    await session.enqueue(inputs(4, 0), {
      batch,
      generation: { context, queryLength: 3, commit: "deferred" },
      copyOutputs: { o: sink },
    });
    assertThrows(() => context.commit(1));
    assertEquals(context.pastLength, 0);
    await batch.finishAndRead({ o: sink });
    assertEquals(context.pastLength, 0);
    assertThrows(() => context.rewind(0));
    context.commit(2);
    assertEquals(context.pastLength, 2);
    await session.run(inputs(1, 1), {}, { context, queryLength: 1 });
    assertEquals(context.pastLength, 3);
  } finally {
    await context.dispose();
    await session.dispose();
    sink.dispose();
    gpu.destroy();
  }
});

test("batch generation: an input failure before the state write leaves the context reusable", async () => {
  const { gpu, session } = await open(),
    context = await session.createGenerationContext({ chunkLength: 4 }),
    sink = await gpu.createResident(16);
  try {
    const batch = await gpu.beginBatch(),
      bad = inputs(1, 0);
    bad.q = { dtype: "f32", shape: [1], data: Float32Array.of(0) };
    const failure = await assertRejects(() =>
      session.enqueue(bad, { batch, generation: { context, queryLength: 1 } }),
    );
    assertStrictEquals(await assertRejects(() => batch.finishAndRead({ o: sink })), failure);
    assertEquals(context.pastLength, 0);
    context.rewind(0);
    await session.run(inputs(1, 1), {}, { context, queryLength: 1 });
    assertEquals(context.pastLength, 1);
  } finally {
    await context.dispose();
    await session.dispose();
    sink.dispose();
    gpu.destroy();
  }
});

test("batch generation: a later Session's failure poisons the context already written", async () => {
  const { gpu, session } = await open(),
    context = await session.createGenerationContext({ chunkLength: 4 }),
    sink = await gpu.createResident(16);
  const downstream = await createSessionFromContainer(
    gpu,
    await openGraphModel(singleOpDeclaration("neg", [[4]], [[4]])),
    GRAPH_NAME,
  );
  try {
    const batch = await gpu.beginBatch();
    await session.enqueue(inputs(1, 0), {
      batch,
      generation: { context, queryLength: 1 },
      copyOutputs: { o: sink },
    });
    const failure = await assertRejects(() =>
      downstream.enqueue({ x0: { dtype: "f32", shape: [1], data: Float32Array.of(1) } }, { batch }),
    );
    assertStrictEquals(await assertRejects(() => batch.finishAndRead({ o: sink })), failure);
    assertThrows(() => context.pastLength);
    assertThrows(() => context.rewind(0));
  } finally {
    await context.dispose();
    await session.dispose();
    await downstream.dispose();
    sink.dispose();
    gpu.destroy();
  }
});

test("batch generation: disposing another Session that rides along before it settles is rejected and does not poison the context", async () => {
  const { gpu, session } = await open(),
    context = await session.createGenerationContext({ chunkLength: 4 }),
    sink = await gpu.createResident(16);
  const reader = await createSessionFromContainer(
    gpu,
    await openGraphModel(singleOpDeclaration("neg", [[4]], [[4]])),
    GRAPH_NAME,
  );
  try {
    const batch = await gpu.beginBatch();
    await session.enqueue(inputs(1, 0), {
      batch,
      generation: { context, queryLength: 1 },
      copyOutputs: { o: sink },
    });
    const read = reader.enqueueRead(
      { x0: { dtype: "f32", shape: [4], data: Float32Array.of(1, 2, 3, 4) } },
      { batch },
    );
    await read.admitted;
    // Letting it through would make the settling copy take the disposed slot backing as its source; the
    // whole range would then fail validation and poison the context too (ADR 0066: a later Session's
    // failure is a failure of the whole range).
    await assertRejects(() => reader.dispose(), Error, "enqueueRead readback");
    await batch.finishAndRead({ o: sink });
    assertEquals(Array.from((await read.outputs).y.data), [-1, -2, -3, -4]);
    assertEquals(context.pastLength, 1);
    await session.run(inputs(1, 1), {}, { context, queryLength: 1 });
    assertEquals(context.pastLength, 2);
  } finally {
    await reader.dispose();
    await context.dispose();
    await session.dispose();
    sink.dispose();
    gpu.destroy();
  }
});

for (const fault of ["compile", "map"] as const) {
  test(`batch generation: keeps the state and the reservation when ${fault} fails`, async () => {
    const { gpu, session } = await open(),
      context = await session.createGenerationContext({ chunkLength: 4 }),
      sink = await gpu.createResident(16);
    const compile = gpu.device.createComputePipelineAsync.bind(gpu.device),
      create = gpu.device.createBuffer.bind(gpu.device),
      error = new Error(`injected ${fault}`);
    if (fault === "compile") gpu.device.createComputePipelineAsync = () => Promise.reject(error);
    else {
      gpu.device.createBuffer = (desc) => {
        const b = create(desc);
        if (desc.label === "batch-readback") b.mapAsync = () => Promise.reject(error);
        return b;
      };
    }
    try {
      const batch = await gpu.beginBatch();
      const pending = session.enqueue(inputs(1, 0), {
        batch,
        generation: { context, queryLength: 1 },
        copyOutputs: { o: sink },
      });
      if (fault === "compile") assertStrictEquals(await assertRejects(() => pending), error);
      else await pending;
      assertStrictEquals(await assertRejects(() => batch.finishAndRead({ o: sink })), error);
      if (fault === "compile") {
        assertEquals(context.pastLength, 0);
        context.rewind(0);
      } else assertThrows(() => context.pastLength);
      gpu.device.createComputePipelineAsync = compile;
      gpu.device.createBuffer = create;
      if (fault === "compile") {
        await session.run(inputs(1, 1), {}, { context, queryLength: 1 });
        assertEquals(context.pastLength, 1);
      }
    } finally {
      gpu.device.createComputePipelineAsync = compile;
      gpu.device.createBuffer = create;
      await context.dispose();
      await session.dispose();
      sink.dispose();
      gpu.destroy();
    }
  });
}

test("batch generation: when the second commit fails, poisons every context and returns the reservations", async () => {
  const { gpu, session } = await open(),
    a = await session.createGenerationContext({ chunkLength: 4 }),
    b = await session.createGenerationContext({ chunkLength: 4 }),
    sink = await gpu.createResident(16);
  const error = new Error("injected second advance"),
    // oxlint-disable-next-line typescript/unbound-method -- internals are arrow-function properties (no `this`)
    advance = b[RUNTIME_INTERNAL].advance;
  try {
    const batch = await gpu.beginBatch();
    await session.enqueue(inputs(1, 0), { batch, generation: { context: a, queryLength: 1 } });
    await session.enqueue(inputs(1, 1), {
      batch,
      generation: { context: b, queryLength: 1 },
      copyOutputs: { o: sink },
    });
    b[RUNTIME_INTERNAL].advance = () => {
      throw error;
    };
    assertStrictEquals(await assertRejects(() => batch.finishAndRead({ o: sink })), error);
    assertThrows(() => a.pastLength);
    assertThrows(() => b.pastLength);
  } finally {
    b[RUNTIME_INTERNAL].advance = advance;
    await a.dispose();
    await b.dispose();
    await session.dispose();
    sink.dispose();
    gpu.destroy();
  }
});

test("batch generation: rejects an invalid commit before reserving and recovers in the same batch", async () => {
  const { gpu, session } = await open(4),
    context = await session.createGenerationContext({ chunkLength: 4 });
  try {
    const batch = await gpu.beginBatch();
    await assertRejects(() =>
      // oxlint-disable-next-line typescript/unbound-method -- Reflect.apply passes `session` as `this`
      Reflect.apply(session.enqueue, session, [
        inputs(1, 0),
        {
          batch,
          generation: { context, queryLength: 1, commit: "typo" },
        },
      ]),
    );
    await assertRejects(() =>
      session.enqueue(inputs(4, 0), {
        batch,
        generation: { context, queryLength: 5, commit: "deferred" },
      }),
    );
    assertEquals(context.pastLength, 0);
    // sliding cannot rewind even with an empty history. The returned reservation is checked by the valid enqueue right after.
    await session.enqueue(inputs(1, 0), { batch, generation: { context, queryLength: 1 } });
    await batch.finish();
    assertEquals(context.pastLength, 1);
  } finally {
    await context.dispose();
    await session.dispose();
    gpu.destroy();
  }
});

test("batch generation: the borrower holds the lender's reservation to the end too and does not advance the logical length", async () => {
  const { gpu, session } = await open(4);
  const lender = await session.createGenerationContext({ chunkLength: 4 });
  const g = graph(4);
  const borrowerGraph: DeclarationJson = {
    ...g,
    symbols: [],
    requires: { ops: ["attention"] },
    inputs: [{ name: "q", dtype: "f32", shape: [1, 1, 1, 4] }],
    values: { o: { dtype: "f32", shape: [1, 1, 1, 4] } },
    states: Object.fromEntries(
      Object.entries(g.states ?? {}).map(([name, value]) => [name, { ...value, external: true }]),
    ),
    nodes: [
      {
        op: "attention",
        ins: ["q"],
        outs: ["o"],
        attrs: { scale: 0.5, window: 4, readonly: true },
        states: { k: "kslot", v: "vslot" },
      },
    ],
  };
  const reader = await createSessionFromContainer(
    gpu,
    await openGraphModel(borrowerGraph),
    GRAPH_NAME,
  );
  const borrower = await reader.createGenerationContext({ chunkLength: 1, borrow: lender });
  const sink = await gpu.createResident(16);
  try {
    await session.run(inputs(4, 0), {}, { context: lender, queryLength: 4 });
    const x = { q: inputs(1, 1).q };
    const ref = await reader.run(x, {}, { context: borrower, queryLength: 1 });
    const batch = await gpu.beginBatch();
    try {
      const pending = reader.enqueue(x, {
        batch,
        generation: { context: borrower, queryLength: 1 },
        copyOutputs: { o: sink },
      });
      await assertRejects(() => borrower.dispose());
      await assertRejects(() => reader.dispose());
      await assertRejects(() => session.run(inputs(1, 2), {}, { context: lender, queryLength: 1 }));
      await pending;
      await assertRejects(() => borrower.dispose());
      assertThrows(() => lender.rewind(0));
      const out = await batch.finishAndRead({ o: sink });
      assertEquals(new Uint32Array(out.o), new Uint32Array(ref.o.data.buffer));
      assertEquals(lender.pastLength, 4);
      await session.run(inputs(1, 2), {}, { context: lender, queryLength: 1 });
      assertEquals(lender.pastLength, 5);
    } finally {
      await batch.finish();
    }
  } finally {
    await borrower.dispose();
    await reader.dispose();
    await lender.dispose();
    await session.dispose();
    sink.dispose();
    gpu.destroy();
  }
});

test("batch generation: rejects an invalid context and leaves no use reservation of the resident outputs", async () => {
  const { gpu, session } = await open(),
    context = await session.createGenerationContext({ chunkLength: 4 }),
    sink = await gpu.createResident(16),
    batch = await gpu.beginBatch();
  try {
    const generation = { context, queryLength: 1 };
    Reflect.set(generation, "context", undefined);
    await assertRejects(async () => {
      await session.enqueue(inputs(1, 0), { batch, generation, copyOutputs: { o: sink } });
    });
    assertEquals(sink.useReferences, 0);
    await session.enqueue(inputs(1, 1), {
      batch,
      generation: { context, queryLength: 1 },
      copyOutputs: { o: sink },
    });
    await batch.finishAndRead({ o: sink });
    assertEquals(context.pastLength, 1);
  } finally {
    await batch.finish();
    await context.dispose();
    await session.dispose();
    try {
      sink.dispose();
    } finally {
      gpu.destroy();
    }
  }
});

test("batch generation: enqueueRead advances the logical length only at settlement and returns outputs that match run as u32", async () => {
  const { gpu, session } = await open(),
    a = await session.createGenerationContext({ chunkLength: 4 }),
    b = await session.createGenerationContext({ chunkLength: 4 });
  const u32 = (data: Float32Array): number[] =>
    Array.from(new Uint32Array(data.buffer, data.byteOffset, data.length));
  try {
    // Prefill is aligned with plain runs; decode compares enqueueRead (b) with run (a).
    await session.run(inputs(4, 0), {}, { context: a, queryLength: 3 });
    await session.run(inputs(4, 0), {}, { context: b, queryLength: 3 });
    let past = 3;
    for (let step = 1; step < 4; step++) {
      const x = inputs(1, step);
      const ref = await session.run(x, {}, { context: a, queryLength: 1 });
      const batch = await gpu.beginBatch(),
        fences = countFences(gpu);
      try {
        const read = session.enqueueRead(x, { batch, generation: { context: b, queryLength: 1 } });
        await read.admitted;
        assertEquals(b.pastLength, past, "the logical length advanced before settlement");
        await batch.finish();
        past += 1;
        assertEquals(b.pastLength, past);
        const out = await read.outputs;
        assertEquals(u32(out.o.data as Float32Array), u32(ref.o.data as Float32Array));
        assertEquals(fences.count(), 0);
      } finally {
        fences.restore();
        await batch.finish();
      }
    }
  } finally {
    await a.dispose();
    await b.dispose();
    await session.dispose();
    gpu.destroy();
  }
});
