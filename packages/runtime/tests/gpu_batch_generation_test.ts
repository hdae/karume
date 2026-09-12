import { assertEquals, assertRejects, assertStrictEquals, assertThrows } from "@std/assert";
import { acquireGpu, createSession, openModel, type Tensor } from "../mod.ts";
import { RUNTIME_INTERNAL } from "../src/gpu/device.ts";
import type { GraphJson } from "./helpers/format.ts";
import { graphModelBuffer, singleOpGraph } from "./helpers/graph.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";
import { countFences } from "./helpers/fences.ts";
const graph = (window?: number): GraphJson => {
  const attrs = window === undefined ? {} : { window };
  return {
    format: "karume-ir",
    version: 1,
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
      ...[["k", "kslot"], ["v", "vslot"]].map(([name, slot]) => ({
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
    ["q", "k", "v"].map((
      name,
      n,
    ) => [name, {
      dtype: "f32",
      shape: [1, 1, rows, 4],
      data: Float32Array.from(
        { length: rows * 4 },
        (_, i) => ((i * 7 + n * 3 + phase * 5) % 17 - 8) * .13,
      ),
    }]),
  );
const open = async (window?: number) => {
  const gpu = await acquireGpu(),
    session = await createSession(gpu, openModel(graphModelBuffer(graph(window))));
  return { gpu, session };
};
const test = (name: string, fn: () => Promise<void>): void =>
  Deno.test({ name, ignore: !GPU_AVAILABLE, fn });
for (const window of [undefined, 4]) {
  test(`batch generation: ${window === undefined ? "full" : "sliding"}のprefillとdecodeが通常runとu32一致する`, async () => {
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
        const batch = await gpu.beginBatch(), fences = countFences(gpu);
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
        const x = inputs(1, 15), ref = await session.run(x, {}, { context: a, queryLength: 1 });
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

test("batch generation: 発行時snapshotを保ち同じcontextの並行run・rewind・disposeを拒否する", async () => {
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
      session.enqueue(inputs(1, 0), { batch, generation: { context: a, queryLength: 1 } })
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

test("batch generation: deferredはバッチ成功後にだけcommitできる", async () => {
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

test("batch generation: state書き込み前の入力失敗ならcontextを再利用できる", async () => {
  const { gpu, session } = await open(),
    context = await session.createGenerationContext({ chunkLength: 4 }),
    sink = await gpu.createResident(16);
  try {
    const batch = await gpu.beginBatch(), bad = inputs(1, 0);
    bad.q = { dtype: "f32", shape: [1], data: Float32Array.of(0) };
    const failure = await assertRejects(() =>
      session.enqueue(bad, { batch, generation: { context, queryLength: 1 } })
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

test("batch generation: 後続Session失敗は書き込み済みcontextをpoisonする", async () => {
  const { gpu, session } = await open(),
    context = await session.createGenerationContext({ chunkLength: 4 }),
    sink = await gpu.createResident(16);
  const downstream = await createSession(
    gpu,
    openModel(graphModelBuffer(singleOpGraph("neg", [[4]], [[4]]))),
  );
  try {
    const batch = await gpu.beginBatch();
    await session.enqueue(inputs(1, 0), {
      batch,
      generation: { context, queryLength: 1 },
      copyOutputs: { o: sink },
    });
    const failure = await assertRejects(() =>
      downstream.enqueue({ x0: { dtype: "f32", shape: [1], data: Float32Array.of(1) } }, { batch })
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

for (const fault of ["compile", "map"] as const) {
  test(`batch generation: ${fault}失敗時のstateと予約を保つ`, async () => {
    const { gpu, session } = await open(),
      context = await session.createGenerationContext({ chunkLength: 4 }),
      sink = await gpu.createResident(16);
    const compile = gpu.device.createComputePipelineAsync.bind(gpu.device),
      create = gpu.device.createBuffer.bind(gpu.device),
      error = new Error(`injected ${fault}`);
    if (fault === "compile") gpu.device.createComputePipelineAsync = () => Promise.reject(error);
    else {gpu.device.createBuffer = (desc) => {
        const b = create(desc);
        if (desc.label === "batch-readback") b.mapAsync = () => Promise.reject(error);
        return b;
      };}
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

test("batch generation: 2つ目の確定が失敗しても全contextをpoisonして予約を返す", async () => {
  const { gpu, session } = await open(),
    a = await session.createGenerationContext({ chunkLength: 4 }),
    b = await session.createGenerationContext({ chunkLength: 4 }),
    sink = await gpu.createResident(16);
  const error = new Error("injected second advance"), advance = b[RUNTIME_INTERNAL].advance;
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

test("batch generation: 不正commitは予約前に拒否し同じbatchで回復できる", async () => {
  const { gpu, session } = await open(4),
    context = await session.createGenerationContext({ chunkLength: 4 });
  try {
    const batch = await gpu.beginBatch();
    await assertRejects(() =>
      Reflect.apply(session.enqueue, session, [inputs(1, 0), {
        batch,
        generation: { context, queryLength: 1, commit: "typo" },
      }])
    );
    await assertRejects(() =>
      session.enqueue(inputs(4, 0), {
        batch,
        generation: { context, queryLength: 5, commit: "deferred" },
      })
    );
    assertEquals(context.pastLength, 0);
    // slidingは履歴0でもrewind不可。予約の返却は直後の正当なenqueueで検証する。
    await session.enqueue(inputs(1, 0), { batch, generation: { context, queryLength: 1 } });
    await batch.finish();
    assertEquals(context.pastLength, 1);
  } finally {
    await context.dispose();
    await session.dispose();
    gpu.destroy();
  }
});

test("batch generation: 借り手は貸し手の予約も終端まで保ち論理長を進めない", async () => {
  const { gpu, session } = await open(4);
  const lender = await session.createGenerationContext({ chunkLength: 4 });
  const g = graph(4);
  const borrowerGraph: GraphJson = {
    ...g,
    symbols: [],
    requires: { ops: ["attention"] },
    inputs: [{ name: "q", dtype: "f32", shape: [1, 1, 1, 4] }],
    values: { o: { dtype: "f32", shape: [1, 1, 1, 4] } },
    states: Object.fromEntries(
      Object.entries(g.states ?? {}).map(([name, value]) => [name, { ...value, external: true }]),
    ),
    nodes: [{
      op: "attention",
      ins: ["q"],
      outs: ["o"],
      attrs: { scale: .5, window: 4, readonly: true },
      states: { k: "kslot", v: "vslot" },
    }],
  };
  const reader = await createSession(gpu, openModel(graphModelBuffer(borrowerGraph)));
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
    await sink.dispose();
    gpu.destroy();
  }
});

test("batch generation: 不正contextを拒否して常駐出力の使用予約を残さない", async () => {
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
