// Behaviour tests of the fixed-length greedy generation loop (`src/generation/greedy.ts`). Needs neither a GPU nor
// real assets.
//
// What is pinned here is "the shape of one step the host assembles" — how fixed-length chunks are split, the 0 of pad
// rows, how absolute positions advance, the derived-input hook (`derive`), the feedback of the previous step's
// returned token, and the context's lifetime. On a real GPU all of these only show as "the values are a bit off" (the
// pad no-op contract of ADR 0066 decision 4 raises no exception even when it consumes wrong positions or leftovers),
// so an implementation error would collapse into a single mismatch in the acceptance e2e. The Session is taken through
// a narrow interface (`GreedySession`), so the fake is a single plain object.

import { test } from "vite-plus/test";
import { assertEquals, assertRejects, assertThrows } from "@std/assert";
import type {
  GenerationContextSpec,
  RunInputs,
  RunOutputs,
  Session,
  SymbolBindings,
} from "@karume/runtime";
import { generateGreedy, type GreedySession, planPrefillChunks } from "../src/generation/greedy.ts";

type FakeContext = { dispose(): Promise<void> };

/** The record of one run (fine-grained enough that the call list alone shows the whole shape of a step). */
type RunCall = {
  readonly ids: readonly number[];
  readonly idsShape: readonly number[];
  readonly positions: readonly number[];
  readonly positionsShape: readonly number[];
  readonly bindings: SymbolBindings | undefined;
  readonly queryLength: number;
  readonly sameContext: boolean;
};

const IDS = "input_ids";
const POSITIONS = "position_ids";
const TOKEN = "next_token";

/**
 * A derived input that passes the position sequence straight into the graph inputs (the minimal implementation of the
 * `derive` hook).
 *
 * What a real distribution supplies is f32 rows built from the positions (gemma4's RoPE cos / sin — `gemma/rope.ts`),
 * but this unit test looks at **the assembly of the position sequence itself** (the 0 of pad rows, how absolute
 * positions advance, the physical row count), so the positions are passed through as i32 and read from the fake's
 * call record.
 */
const derivePositions = (
  _ids: Int32Array<ArrayBuffer>,
  positions: Int32Array<ArrayBuffer>,
): RunInputs => ({
  [POSITIONS]: { dtype: "i32", shape: [1, positions.length], data: positions },
});

const readRow = (
  inputs: RunInputs,
  name: string,
): { readonly shape: readonly number[]; readonly values: readonly number[] } => {
  if (!Object.hasOwn(inputs, name)) throw new Error(`fake: input '${name}' was not passed`);
  const tensor = inputs[name];
  if (!("data" in tensor)) throw new Error(`fake: input '${name}' is not a host tensor`);
  if (tensor.dtype !== "i32") throw new Error(`fake: input '${name}' is not i32`);
  return { shape: tensor.shape, values: [...tensor.data] };
};

/**
 * The default output: `next_token[0, row, 0] = call * 1000 + row`.
 *
 * Both the call count and the row number are folded into the value, so a single returned value tells "which row of
 * which run was read" (an implementation that mixes up rows — for example, always row 0 — always fails in prefill).
 * A decoy of matching shape (`logits`) sits alongside to catch implementations that look outputs up by shape instead
 * of by name.
 */
const defaultOutputs = (call: number, rows: number): RunOutputs => ({
  logits: { dtype: "f32", shape: [1, rows, 1], data: new Float32Array(rows) },
  [TOKEN]: {
    dtype: "i32",
    shape: [1, rows, 1],
    data: Int32Array.from({ length: rows }, (_, row) => call * 1000 + row),
  },
});

type FakeOptions = {
  /** The outputs run returns (default {@link defaultOutputs}). */
  readonly outputs?: (call: number, rows: number) => RunOutputs;
  /** Makes the run with this call index (0-based) fail. */
  readonly failAt?: number;
};

const fakeSession = (options: FakeOptions = {}) => {
  const calls: RunCall[] = [];
  const specs: GenerationContextSpec[] = [];
  let disposals = 0;
  const context: FakeContext = {
    dispose: () => {
      disposals += 1;
      return Promise.resolve();
    },
  };
  const session: GreedySession<FakeContext> = {
    createGenerationContext: (spec) => {
      specs.push(spec);
      return Promise.resolve(context);
    },
    run: (inputs, bindings, generation) => {
      const ids = readRow(inputs, IDS);
      const positions = readRow(inputs, POSITIONS);
      const call = calls.length;
      calls.push({
        ids: ids.values,
        idsShape: ids.shape,
        positions: positions.values,
        positionsShape: positions.shape,
        bindings,
        queryLength: generation.queryLength,
        sameContext: generation.context === context,
      });
      if (options.failAt === call) return Promise.reject(new Error("run failed"));
      return Promise.resolve((options.outputs ?? defaultOutputs)(call, ids.values.length));
    },
  };
  return { session, calls, specs, disposals: () => disposals };
};

/** The prompt under test (values are 1:1 with positions, so a shift shows directly in the values). */
const promptOf = (length: number, base = 100): number[] =>
  Array.from({ length }, (_, index) => base + index);

test("planPrefillChunks: splits T=87 / L=32 into 3 chunks of 32 + 32 + 23", () => {
  assertEquals(planPrefillChunks(87, 32), [
    { position: 0, queryLength: 32 },
    { position: 32, queryLength: 32 },
    { position: 64, queryLength: 23 },
  ]);
});

test("planPrefillChunks: when T is a multiple of L, no chunk has pad rows", () => {
  assertEquals(planPrefillChunks(32, 32), [{ position: 0, queryLength: 32 }]);
  assertEquals(planPrefillChunks(64, 32), [
    { position: 0, queryLength: 32 },
    { position: 32, queryLength: 32 },
  ]);
});

test("planPrefillChunks: T < L is 1 chunk (the rest are pad rows)", () => {
  assertEquals(planPrefillChunks(1, 32), [{ position: 0, queryLength: 1 }]);
  assertEquals(planPrefillChunks(31, 32), [{ position: 0, queryLength: 31 }]);
});

test("planPrefillChunks: positions neither skip nor overlap, and the valid rows sum to T", () => {
  // The invariants of the split themselves (getting one boundary wrong always breaks one of them).
  for (const [length, chunkLength] of [
    [87, 32],
    [1, 1],
    [5, 7],
    [64, 8],
    [9, 4],
  ]) {
    const chunks = planPrefillChunks(length, chunkLength);
    let expected = 0;
    for (const chunk of chunks) {
      assertEquals(chunk.position, expected);
      assertEquals(chunk.queryLength >= 1 && chunk.queryLength <= chunkLength, true);
      expected += chunk.queryLength;
    }
    assertEquals(expected, length);
  }
});

test("planPrefillChunks: fails unless the length / chunkLength are integers of 1 or more", () => {
  assertThrows(() => planPrefillChunks(0, 32), Error, "prompt length 0");
  assertThrows(() => planPrefillChunks(-1, 32), Error, "prompt length -1");
  assertThrows(() => planPrefillChunks(1.5, 32), Error, "prompt length 1.5");
  assertThrows(() => planPrefillChunks(8, 0), Error, "chunkLength 0");
  assertThrows(() => planPrefillChunks(8, 2.5), Error, "chunkLength 2.5");
});

test("generateGreedy: prefill runs in fixed-length chunks, pad rows are 0, and positions are absolute", async () => {
  const fake = fakeSession();
  const prompt = promptOf(87);
  const generated = await generateGreedy({
    session: fake.session,
    inputIds: IDS,
    derive: derivePositions,
    token: TOKEN,
    chunkLength: 32,
    maxPosition: 512,
    prompt,
    maxNewTokens: 1,
  });

  assertEquals(fake.calls.length, 3);
  assertEquals(
    fake.calls.map((call) => call.queryLength),
    [32, 32, 23],
  );
  // The physical chunk row count is fixed at chunkLength for all 3 (ADR 0066 decision 4).
  assertEquals(
    fake.calls.map((call) => call.idsShape),
    [
      [1, 32],
      [1, 32],
      [1, 32],
    ],
  );
  assertEquals(
    fake.calls.map((call) => call.positionsShape),
    [
      [1, 32],
      [1, 32],
      [1, 32],
    ],
  );

  assertEquals(fake.calls[0].ids, prompt.slice(0, 32));
  assertEquals(fake.calls[1].ids, prompt.slice(32, 64));
  // The last chunk has 23 valid rows, and the remaining 9 are pad = 0 (the value contract of addendum 6).
  assertEquals(fake.calls[2].ids, [...prompt.slice(64, 87), ...new Array(9).fill(0)]);

  assertEquals(
    fake.calls[0].positions,
    Array.from({ length: 32 }, (_, i) => i),
  );
  assertEquals(
    fake.calls[1].positions,
    Array.from({ length: 32 }, (_, i) => 32 + i),
  );
  assertEquals(fake.calls[2].positions, [
    ...Array.from({ length: 23 }, (_, i) => 64 + i),
    ...new Array(9).fill(0),
  ]);

  // The first generated token is read from **the last valid row of the last chunk** (call=2 / row=22).
  assertEquals(generated, [2022]);
});

test("generateGreedy: T exactly equal to chunkLength is 1 chunk with no pad rows", async () => {
  const fake = fakeSession();
  const prompt = promptOf(4);
  const generated = await generateGreedy({
    session: fake.session,
    inputIds: IDS,
    derive: derivePositions,
    token: TOKEN,
    chunkLength: 4,
    maxPosition: 512,
    prompt,
    maxNewTokens: 1,
  });

  assertEquals(fake.calls.length, 1);
  assertEquals(fake.calls[0].queryLength, 4);
  assertEquals(fake.calls[0].ids, prompt);
  assertEquals(fake.calls[0].positions, [0, 1, 2, 3]);
  assertEquals(generated, [3]);
});

test("generateGreedy: T < chunkLength is 1 chunk with the tail filled with pad 0", async () => {
  const fake = fakeSession();
  const generated = await generateGreedy({
    session: fake.session,
    inputIds: IDS,
    derive: derivePositions,
    token: TOKEN,
    chunkLength: 4,
    maxPosition: 512,
    prompt: [7, 8],
    maxNewTokens: 1,
  });

  assertEquals(fake.calls.length, 1);
  assertEquals(fake.calls[0].queryLength, 2);
  assertEquals(fake.calls[0].ids, [7, 8, 0, 0]);
  assertEquals(fake.calls[0].positions, [0, 1, 0, 0]);
  // It reads the last valid row (row=1), not the pad rows (row=2,3).
  assertEquals(generated, [1]);
});

test("generateGreedy: decode is fixed at queryLength=1, and positions advance T, T+1, …", async () => {
  const fake = fakeSession();
  const generated = await generateGreedy({
    session: fake.session,
    inputIds: IDS,
    derive: derivePositions,
    token: TOKEN,
    chunkLength: 4,
    maxPosition: 512,
    prompt: [10, 11, 12],
    maxNewTokens: 3,
  });

  assertEquals(fake.calls.length, 3);
  assertEquals(
    fake.calls.map((call) => call.queryLength),
    [4 - 1, 1, 1],
  );
  // The physical chunk row count of decode is 1 (a PreparedPlan separate from the prefill shape — ADR 0066 decision 4).
  assertEquals(fake.calls[1].idsShape, [1, 1]);
  assertEquals(fake.calls[2].idsShape, [1, 1]);
  assertEquals(fake.calls[1].positions, [3]);
  assertEquals(fake.calls[2].positions, [4]);

  // Feedback: the previous step's returned token becomes the next input as is.
  assertEquals(generated, [2, 1000, 2000]);
  assertEquals(fake.calls[1].ids, [generated[0]]);
  assertEquals(fake.calls[2].ids, [generated[1]]);
});

test("generateGreedy: maxNewTokens=1 runs decode not even once", async () => {
  const fake = fakeSession();
  const generated = await generateGreedy({
    session: fake.session,
    inputIds: IDS,
    derive: derivePositions,
    token: TOKEN,
    chunkLength: 4,
    maxPosition: 512,
    prompt: [10, 11, 12],
    maxNewTokens: 1,
  });

  assertEquals(generated.length, 1);
  assertEquals(fake.calls.length, 1);
});

test("generateGreedy: the capacity symbol is passed only to createGenerationContext (not to run)", async () => {
  // ADR 0066 addendum 7 — the binding point of a states-only symbol is context creation alone. Passing it to run's
  // bindings would trip the "states are not a binding source" check or mix an extra symbol into the plan key.
  const fake = fakeSession();
  await generateGreedy({
    session: fake.session,
    inputIds: IDS,
    derive: derivePositions,
    token: TOKEN,
    chunkLength: 4,
    maxPosition: 512,
    bindings: { C: 8 },
    prompt: [10, 11],
    maxNewTokens: 2,
  });

  assertEquals(fake.specs.length, 1);
  assertEquals(fake.specs[0].chunkLength, 4);
  assertEquals(fake.specs[0].bindings, { C: 8 });
  assertEquals(
    fake.calls.map((call) => call.bindings),
    [undefined, undefined],
  );
  // Every step uses the same context (1 generation = 1 context).
  assertEquals(
    fake.calls.every((call) => call.sameContext),
    true,
  );
});

test("generateGreedy: on normal completion the context is disposed exactly once", async () => {
  const fake = fakeSession();
  await generateGreedy({
    session: fake.session,
    inputIds: IDS,
    derive: derivePositions,
    token: TOKEN,
    chunkLength: 4,
    maxPosition: 512,
    prompt: [10, 11],
    maxNewTokens: 3,
  });
  assertEquals(fake.disposals(), 1);
});

test("generateGreedy: even when run fails the context is disposed, and the exception propagates as is", async () => {
  // A failed context is poisoned and cannot be reused (ADR 0066 addendum 3), so holding on to it would pile up the KV
  // capacity's worth of VRAM per generation. Both the prefill side and the decode side are checked.
  for (const failAt of [0, 1]) {
    const fake = fakeSession({ failAt });
    await assertRejects(
      () =>
        generateGreedy({
          session: fake.session,
          inputIds: IDS,
          derive: derivePositions,
          token: TOKEN,
          chunkLength: 4,
          maxPosition: 512,
          prompt: [10, 11],
          maxNewTokens: 3,
        }),
      Error,
      "run failed",
    );
    assertEquals(fake.calls.length, failAt + 1);
    assertEquals(fake.disposals(), 1);
  }
});

test("generateGreedy: a wrong name / dtype / shape of the token output fails loudly", async () => {
  const cases: readonly (readonly [string, (call: number, rows: number) => RunOutputs, string])[] =
    [
      [
        "name missing",
        (_call, rows) => ({
          logits: { dtype: "f32", shape: [1, rows, 1], data: new Float32Array(rows) },
        }),
        "Graph output 'next_token' is missing",
      ],
      [
        "dtype is f32",
        (_call, rows) => ({
          [TOKEN]: { dtype: "f32", shape: [1, rows, 1], data: new Float32Array(rows) },
        }),
        "is not i32",
      ],
      [
        "rank is 2",
        (_call, rows) => ({
          [TOKEN]: { dtype: "i32", shape: [1, rows], data: new Int32Array(rows) },
        }),
        "not [1,4,1]",
      ],
      [
        "row count differs from the physical chunk row count",
        (_call, rows) => ({
          [TOKEN]: { dtype: "i32", shape: [1, rows + 1, 1], data: new Int32Array(rows + 1) },
        }),
        "not [1,4,1]",
      ],
      [
        "last axis is not 1",
        (_call, rows) => ({
          [TOKEN]: { dtype: "i32", shape: [1, rows, 2], data: new Int32Array(rows * 2) },
        }),
        "not [1,4,1]",
      ],
    ];

  for (const [name, outputs, message] of cases) {
    const fake = fakeSession({ outputs });
    await assertRejects(
      () =>
        generateGreedy({
          session: fake.session,
          inputIds: IDS,
          derive: derivePositions,
          token: TOKEN,
          chunkLength: 4,
          maxPosition: 512,
          prompt: [10, 11],
          maxNewTokens: 2,
        }),
      Error,
      message,
      name,
    );
    // The context is returned even when the check fails.
    assertEquals(fake.disposals(), 1, name);
  }
});

test("generateGreedy: the entry checks fail before the context is created", async () => {
  const base = {
    inputIds: IDS,
    derive: derivePositions,
    token: TOKEN,
    chunkLength: 4,
    maxPosition: 512,
    prompt: [10, 11],
    maxNewTokens: 2,
  };
  const cases: readonly (readonly [Partial<typeof base>, string])[] = [
    [{ prompt: [] }, "prompt length 0"],
    [{ chunkLength: 0 }, "chunkLength 0"],
    [{ chunkLength: 2.5 }, "chunkLength 2.5"],
    [{ maxNewTokens: 0 }, "maxNewTokens 0"],
    [{ maxNewTokens: -1 }, "maxNewTokens -1"],
    [{ maxNewTokens: 1.5 }, "maxNewTokens 1.5"],
    [{ maxPosition: 0 }, "maxPosition 0"],
    [{ maxPosition: 2.5 }, "maxPosition 2.5"],
    [{ prompt: [10, 1.5] }, "prompt[1] 1.5 is not an integer in 0..2147483647"],
    // `Int32Array` wraps without an exception into another valid id (2^32+1 → 1) — the entry point is the only line of detection.
    [{ prompt: [10, 4294967297] }, "prompt[1] 4294967297 is not an integer in 0..2147483647"],
    [{ prompt: [-1, 10] }, "prompt[0] -1 is not an integer in 0..2147483647"],
    // The largest position reached = T + K - 2. maxPosition is an exclusive upper bound, so a shape that exactly reaches it fails.
    [{ maxPosition: 2 }, "reaches the last position 2"],
  ];

  for (const [override, message] of cases) {
    const fake = fakeSession();
    await assertRejects(
      () => generateGreedy({ ...base, ...override, session: fake.session }),
      Error,
      message,
    );
    assertEquals(fake.specs.length, 0, message);
    assertEquals(fake.calls.length, 0, message);
  }
});

test("generateGreedy: maxPosition is an exclusive upper bound (last position + 1 passes exactly)", async () => {
  // With T=2, K=2 the last position is T + K - 2 = 2 (the first decode reaches position 2). A limit of 3 = just legal.
  const fake = fakeSession();
  const generated = await generateGreedy({
    session: fake.session,
    inputIds: IDS,
    derive: derivePositions,
    token: TOKEN,
    chunkLength: 4,
    maxPosition: 3,
    prompt: [10, 11],
    maxNewTokens: 2,
  });

  assertEquals(generated.length, 2);
  assertEquals(fake.calls[1].positions, [2]);
});

test("GreedySession: satisfies the real Session's surface by type (detects spelling drift)", () => {
  // A type-check-only check (does nothing at run time). `GreedySession` **mirrors** the real Session's surface so a fake
  // can be plugged in, so if the spelling or arguments of `run` / `createGenerationContext` change on the runtime side,
  // this one line becomes a compile error (the only detection point for the silent degradation of a drifted mirror with
  // only the unit tests green).
  const asGreedySession = (session: Session): GreedySession => session;
  assertEquals(typeof asGreedySession, "function");
});

test("generateGreedy(derive): ids / positions for the physical row count are passed, and the return value becomes run's input", async () => {
  // Position-dependent graph inputs (gemma4's RoPE cos / sin — ADR 0091 decision 1) enter run **only** through this
  // hook. That the 2 have the same physical row count and that pad rows are 0 in both is visible to no other check (the
  // graph side silently consumes the pad rows' values — ADR 0066 addendum 6).
  const seen: { readonly ids: readonly number[]; readonly positions: readonly number[] }[] = [];
  const fake = fakeSession();
  const generated = await generateGreedy({
    session: fake.session,
    inputIds: IDS,
    token: TOKEN,
    derive: (ids, positions) => {
      seen.push({ ids: [...ids], positions: [...positions] });
      return derivePositions(ids, positions);
    },
    chunkLength: 4,
    maxPosition: 512,
    prompt: [10, 11],
    maxNewTokens: 2,
  });

  assertEquals(generated.length, 2);
  // Prefill is 4 physical rows (2 valid + 2 pad = 0) and decode is 1 row, with positions advancing from T.
  assertEquals(seen, [
    { ids: [10, 11, 0, 0], positions: [0, 1, 0, 0] },
    { ids: [generated[0]], positions: [2] },
  ]);
  // The keys of the return value go onto run's inputs as is (the fake requires `position_ids`).
  assertEquals(
    fake.calls.map((call) => call.positions),
    [[0, 1, 0, 0], [2]],
  );
});

test("generateGreedy: without derive, nothing but input_ids is passed", async () => {
  // Sending an extra input to assets without derived inputs (a 1-input graph) makes the runtime reject it as "an input
  // that is not wired". The key set pins that omitting it has no side effects.
  const keys: (readonly string[])[] = [];
  const session: GreedySession<FakeContext> = {
    createGenerationContext: () => Promise.resolve({ dispose: () => Promise.resolve() }),
    run: (inputs) => {
      keys.push(Object.keys(inputs).sort());
      return Promise.resolve(defaultOutputs(keys.length - 1, readRow(inputs, IDS).values.length));
    },
  };

  await generateGreedy({
    session,
    inputIds: IDS,
    token: TOKEN,
    chunkLength: 4,
    maxPosition: 64,
    prompt: [10, 11],
    maxNewTokens: 2,
  });

  assertEquals(keys, [[IDS], [IDS]]);
});

// ---- token-only output (the default shape of ADR 0068 decision 4 — a single-row output with `lastRow`) ----
// The token-only exit series was removed on 2026-10-17; these cases pin the `lastRow` option of
// `generateGreedy` itself with a fake session.

const LAST_ROW = "last_row";

/** The token-only fake: records `last_row` and returns a `[1,1,1]` token. */
const tokenOnlySession = () => {
  const lastRows: (readonly number[])[] = [];
  const rows: number[] = [];
  let disposals = 0;
  const context: FakeContext = {
    dispose: () => {
      disposals += 1;
      return Promise.resolve();
    },
  };
  let call = 0;
  const session: GreedySession<FakeContext> = {
    createGenerationContext: () => Promise.resolve(context),
    run: (inputs, _bindings, generation) => {
      const row = readRow(inputs, LAST_ROW);
      lastRows.push(row.shape);
      rows.push(row.values[0]);
      void generation;
      const value = 5000 + call;
      call += 1;
      return Promise.resolve({
        [TOKEN]: { dtype: "i32", shape: [1, 1, 1], data: Int32Array.of(value) } as const,
      } as RunOutputs);
    },
  };
  return { session, lastRows, rows, disposals: () => disposals };
};

test("generateGreedy(lastRow): prefill passes the last valid row's index and decode passes 0, as [1] i32", async () => {
  const fake = tokenOnlySession();

  const generated = await generateGreedy({
    session: fake.session,
    inputIds: IDS,
    derive: derivePositions,
    token: TOKEN,
    lastRow: LAST_ROW,
    chunkLength: 4,
    maxPosition: 64,
    prompt: promptOf(6),
    maxNewTokens: 3,
  });

  // 2 prefill chunks (4 / 2 valid rows) → indices 3, 1. 2 decodes → indices 0, 0.
  assertEquals(fake.rows, [3, 1, 0, 0]);
  // The shape is always `[1]` (one row-selection index — a rank-1 input separate from the chunk shape `[1,M]`).
  for (const shape of fake.lastRows) assertEquals(shape, [1]);
  // The start is the last chunk's value (call=1), then each decode's value (reading element 0 of `[1,1,1]`).
  assertEquals(generated, [5001, 5002, 5003]);
  assertEquals(fake.disposals(), 1);
});

test("generateGreedy(lastRow): fails loudly unless the token output is [1,1,1]", async () => {
  const session: GreedySession<FakeContext> = {
    createGenerationContext: () => Promise.resolve({ dispose: () => Promise.resolve() }),
    run: () =>
      // Assets that return the all-row shape `[1,M,1]` despite lastRow (= mistaking it for the logits opt-in shape).
      Promise.resolve({
        [TOKEN]: { dtype: "i32", shape: [1, 4, 1], data: new Int32Array(4) },
      } as RunOutputs),
  };

  await assertRejects(
    () =>
      generateGreedy({
        session,
        inputIds: IDS,
        derive: derivePositions,
        token: TOKEN,
        lastRow: LAST_ROW,
        chunkLength: 4,
        maxPosition: 64,
        prompt: promptOf(4),
        maxNewTokens: 1,
      }),
    Error,
    "not [1,1,1]",
  );
});

test("generateGreedy: without lastRow, no last_row input is passed (the current shape is unaffected)", async () => {
  const seen: boolean[] = [];
  const fake = fakeSession({
    outputs: (call, rows) => defaultOutputs(call, rows),
  });
  const spying: GreedySession<FakeContext> = {
    createGenerationContext: (spec) => fake.session.createGenerationContext(spec),
    run: (inputs, bindings, generation) => {
      seen.push(Object.hasOwn(inputs, LAST_ROW));
      return fake.session.run(inputs, bindings, generation);
    },
  };

  await generateGreedy({
    session: spying,
    inputIds: IDS,
    derive: derivePositions,
    token: TOKEN,
    chunkLength: 4,
    maxPosition: 64,
    prompt: promptOf(6),
    maxNewTokens: 2,
  });

  assertEquals(seen, [false, false, false]);
});
