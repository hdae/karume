// `SessionOptions.fullStateStorage` through real Sessions and real contexts (the kernels alone are
// checked in gpu_state_kv16_test.ts).
//
// The expected value is **exact, not a tolerance**. The contract is one rounding point: an f16 run
// equals the f32 kernels over a slot rounded on the host. So every run has three arms:
//
//   packed   the f16 Session, appending the rows as they are (the kernel rounds them)
//   rounded  an f32 Session, appending `Math.f16round` of the same rows (the host rounds them)
//   plain    an f32 Session, appending the rows as they are (today's numbers)
//
// `packed` MUST equal `rounded` word for word on every run, which a misread word, half or row breaks.
// On the first run there is no past, so `packed` also equals `plain` word for word. Later runs differ
// from `plain`, checked to be by a little, so a run that never reached the slot cannot pass.

import { describe, it } from "vite-plus/test";
import { assert, assertEquals, assertRejects } from "@std/assert";
import { CodegenError } from "../src/codegen/errors.ts";
import { type Session, createSessionFromContainer, type Tensor } from "../src/runtime/executor.ts";
import { acquireGpu, type GpuContext, RUNTIME_INTERNAL } from "../src/gpu/device.ts";
import { stateAppendKey } from "../src/kernels/state-append.ts";
import {
  statePvKey,
  statePvParallelKey,
  statePvParallelReadonlyKey,
  statePvTiledKey,
  stateQkFoldedKey,
  stateQkKey,
  stateQkParallelKey,
  stateQkParallelReadonlyKey,
  stateQkTiledKey,
} from "../src/kernels/state-attention.ts";
import { stateStatsPvKey } from "../src/kernels/state-attention-stats-pv.ts";
import type { GenerationContext } from "../src/runtime/generation-context.ts";
import { ExecutionError } from "../src/runtime/plan.ts";
import type { SessionOptions, StateAttentionReduce } from "../src/runtime/session-types.ts";
import {
  fullStateBorrowerGraph,
  fullStateGraph,
  type FullStateShape,
} from "./helpers/full-state-graph.ts";
import { openGraph } from "./helpers/estimate-graphs.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";
import { type DeclarationJson, openGraphModel } from "./helpers/model-fixture.ts";
import {
  assertPipelineCensus,
  countDispatches,
  requireCensus,
  SEAT_SIGNATURES,
} from "./helpers/pipeline-census.ts";
import { seeded } from "./helpers/state-dispatch.ts";

const QUERY = (i: number): number => (((i * 7) % 23) - 11) * 0.17;
const KEY = (i: number): number => (((i * 11) % 19) - 9) * 0.23;
const VALUE = (i: number): number => (((i * 5) % 17) - 8) * 0.31;

/** Round-off of the slot moves an output by this little; a misread slot moves it far more. */
const ROUNDING_BAND = 1e-2;

const tensor = (shape: readonly number[], data: Float32Array<ArrayBuffer>): Tensor => ({
  dtype: "f32",
  shape,
  data,
});

const rounded = (data: Float32Array<ArrayBuffer>): Float32Array<ArrayBuffer> =>
  data.map((value) => Math.f16round(value));

const words = (tensorData: Tensor["data"]): number[] => {
  const data = tensorData as Float32Array<ArrayBuffer>;
  return [...new Uint32Array(data.buffer, data.byteOffset, data.length)];
};

/** A run's inputs; `round` makes the appended rows what the f16 append would have stored. */
const stepInputs = (
  { heads, kvHeads, depth }: FullStateShape,
  rows: number,
  salt: number,
  round: boolean,
): Record<string, Tensor> => {
  const kvShape = [1, kvHeads, rows, depth];
  const kvCount = kvHeads * rows * depth;
  const k = seeded(kvCount, (i) => KEY(i + salt));
  const v = seeded(kvCount, (i) => VALUE(i + salt));
  return {
    q: tensor(
      [1, heads, rows, depth],
      seeded(heads * rows * depth, (i) => QUERY(i + salt)),
    ),
    k: tensor(kvShape, k),
    v: tensor(kvShape, v),
    append_k: tensor(kvShape, round ? rounded(k) : k),
    append_v: tensor(kvShape, round ? rounded(v) : v),
  };
};

type Arm = { readonly session: Session; readonly context: GenerationContext };

type Arms = { readonly packed: Arm; readonly rounded: Arm; readonly plain: Arm };

const sessionOf = async (
  gpu: GpuContext,
  graph: DeclarationJson,
  options: SessionOptions,
): Promise<Session> =>
  await createSessionFromContainer(gpu, await openGraphModel(graph), "model", options);

/** One Session and context per arm, for `body`; everything is released on the way out. */
const withArms = async (
  shape: FullStateShape,
  rows: number,
  capacity: number,
  reduce: StateAttentionReduce,
  body: (arms: Arms, gpu: GpuContext) => Promise<void>,
): Promise<void> => {
  const gpu = await acquireGpu();
  const graph = fullStateGraph(shape);
  const armOf = async (fullStateStorage: "f32" | "f16"): Promise<Arm> => {
    const session = await sessionOf(gpu, graph, {
      fullStateStorage,
      stateAttentionReduce: reduce,
    });
    return {
      session,
      context: await session.createGenerationContext({
        chunkLength: rows,
        bindings: { C: capacity },
      }),
    };
  };
  const arms = {
    packed: await armOf("f16"),
    rounded: await armOf("f32"),
    plain: await armOf("f32"),
  };
  try {
    await body(arms, gpu);
  } finally {
    for (const { session, context } of Object.values(arms)) {
      await context.dispose();
      await session.dispose();
    }
    gpu.destroy();
  }
};

const runArm = async (
  { session, context }: Arm,
  shape: FullStateShape,
  rows: number,
  query: number,
  salt: number,
  round: boolean,
): Promise<Record<string, Tensor>> =>
  await session.run(
    stepInputs(shape, rows, salt, round),
    {},
    {
      context,
      queryLength: query,
    },
  );

/** Runs one step on all three arms and checks the contract on the outputs the graph has. */
const stepAll = async (
  arms: Arms,
  shape: FullStateShape,
  rows: number,
  query: number,
  salt: number,
  label: string,
): Promise<void> => {
  const past = arms.packed.context.pastLength;
  const packed = await runArm(arms.packed, shape, rows, query, salt, false);
  const host = await runArm(arms.rounded, shape, rows, query, salt, true);
  const plain = await runArm(arms.plain, shape, rows, query, salt, false);
  if (shape.slots !== "sliding") {
    assertEquals(words(packed["o"].data), words(host["o"].data), `${label}: f16 vs host-rounded`);
    if (past === 0) {
      assertEquals(
        words(packed["o"].data),
        words(plain["o"].data),
        `${label}: the first run has no past`,
      );
    } else {
      const a = packed["o"].data as Float32Array<ArrayBuffer>;
      const b = plain["o"].data as Float32Array<ArrayBuffer>;
      const largest = a.reduce((most, value, i) => Math.max(most, Math.abs(value - b[i])), 0);
      assert(largest > 0, `${label}: no difference from f32, so the slot was not read rounded`);
      assert(largest < ROUNDING_BAND, `${label}: moved by ${largest}, more than rounding can`);
    }
  }
  // A sliding slot rounds nothing, so under f16 its output stays today's number. It is compared with
  // `plain` because the graph gives both slot kinds one `append_*`, which `rounded` rounds.
  if (shape.slots !== "full") {
    assertEquals(words(packed["os"].data), words(plain["os"].data), `${label}: the sliding output`);
  }
};

describe.skipIf(!GPU_AVAILABLE)("fullStateStorage: option and bytes (real GPU)", () => {
  it("an invalid spelling is rejected at Session construction", async () => {
    const gpu = await acquireGpu();
    try {
      const shape = { heads: 4, kvHeads: 2, depth: 8, slots: "full" } as const;
      await assertRejects(
        () => sessionOf(gpu, fullStateGraph(shape), { fullStateStorage: "F16" as "f16" }),
        ExecutionError,
        "fullStateStorage",
      );
    } finally {
      gpu.destroy();
    }
  });

  it("a real context's resident bytes equal the estimate, in both storages", async () => {
    const gpu = await acquireGpu();
    try {
      const graph = fullStateGraph({ heads: 4, kvHeads: 2, depth: 8, slots: "both" });
      const generation = { chunkLength: 4, bindings: { C: 64 } };
      const resident = async (fullStateStorage: "f32" | "f16"): Promise<number> => {
        const session = await sessionOf(gpu, graph, { fullStateStorage });
        const context = await session.createGenerationContext(generation);
        try {
          return session.diagnostics().stateBacking.residentBytes;
        } finally {
          await context.dispose();
          await session.dispose();
        }
      };
      for (const fullStateStorage of ["f32", "f16"] as const) {
        assertEquals(
          await resident(fullStateStorage),
          openGraph(graph).estimate({
            generation,
            maxStorageBufferBindingSize: gpu.limits.maxStorageBufferBindingSize,
            fullStateStorage,
          }).resident.stateBytes,
          fullStateStorage,
        );
      }
      assertEquals(
        (await resident("f32")) - (await resident("f16")),
        2 * (1 * 2 * 64 * 8 * 2),
        "f16 saves two bytes per element of the two full slots, and nothing of the sliding ones",
      );
    } finally {
      gpu.destroy();
    }
  });

  it("an odd D fails at createGenerationContext under f16, and only there", async () => {
    const gpu = await acquireGpu();
    try {
      const graph = fullStateGraph({ heads: 4, kvHeads: 2, depth: 7, slots: "full" });
      const generation = { chunkLength: 4, bindings: { C: 16 } };
      const packed = await sessionOf(gpu, graph, { fullStateStorage: "f16" });
      await assertRejects(
        () => packed.createGenerationContext(generation),
        CodegenError,
        "even depth",
      );
      await packed.dispose();
      const plain = await sessionOf(gpu, graph, {});
      await (await plain.createGenerationContext(generation)).dispose();
      await plain.dispose();
    } finally {
      gpu.destroy();
    }
  });
});

/** Every reader form the builders can pick, with the keys that prove it ran. */
const FORMS = [
  {
    name: "① and ③ (sequential, M = 4)",
    shape: { heads: 4, kvHeads: 2, depth: 8, slots: "both" },
    rows: 4,
    capacity: 64,
    reduce: "sequential",
    keys: [statePvKey(false, true, "f16"), stateQkKey(false, true, "f16")],
  },
  {
    name: "①' and ③' (parallel, M = 4)",
    shape: { heads: 4, kvHeads: 2, depth: 8, slots: "both" },
    rows: 4,
    capacity: 64,
    reduce: "parallel",
    keys: [statePvParallelKey(false, true, "f16"), stateQkParallelKey(false, true, "f16")],
  },
  {
    name: "①' and ③' without GQA (parallel, M = 1)",
    shape: { heads: 4, kvHeads: 4, depth: 8, slots: "both" },
    rows: 1,
    capacity: 64,
    reduce: "parallel",
    keys: [statePvParallelKey(false, false, "f16"), stateQkParallelKey(false, false, "f16")],
  },
  {
    name: "①ₜ and ③ₜ (M = 16)",
    shape: { heads: 4, kvHeads: 2, depth: 8, slots: "both" },
    rows: 16,
    capacity: 128,
    reduce: "sequential",
    keys: [statePvTiledKey(false, true, 16, "f16"), stateQkTiledKey(false, true, 16, "f16")],
  },
  {
    name: "the folded ①' (the E2B full-layer class, M = 4)",
    shape: { heads: 8, kvHeads: 1, depth: 512, slots: "full" },
    rows: 4,
    capacity: 32,
    reduce: "parallel",
    keys: [statePvParallelKey(false, true, "f16"), stateQkFoldedKey(8, "f16")],
  },
  {
    name: "parallel-fused: ③' on the f16 slot, the fused kernel still on the sliding one",
    shape: { heads: 4, kvHeads: 2, depth: 8, slots: "both" },
    rows: 4,
    capacity: 64,
    reduce: "parallel-fused",
    keys: [statePvParallelKey(false, true, "f16"), stateStatsPvKey(true, true)],
  },
] as const satisfies readonly {
  readonly name: string;
  readonly shape: FullStateShape;
  readonly rows: number;
  readonly capacity: number;
  readonly reduce: StateAttentionReduce;
  readonly keys: readonly string[];
}[];

describe.skipIf(!GPU_AVAILABLE)("fullStateStorage: a conversation (real GPU)", () => {
  for (const { name, shape, rows, capacity, reduce, keys } of FORMS) {
    it(`${name}: every run equals the host-rounded f32 run word for word`, async () => {
      await withArms(shape, rows, capacity, reduce, async (arms) => {
        const partial = Math.max(1, rows - 1);
        for (const [index, query] of [rows, partial, rows, partial].entries()) {
          await stepAll(arms, shape, rows, query, index * 101, `run ${index}`);
        }
        const census = arms.packed.session.diagnostics().lastRunPipelines;
        assertPipelineCensus(
          census,
          [
            ...keys.map((key) => ({ label: key, match: key, atLeast: 1 })),
            {
              label: "the f16 append",
              match: stateAppendKey(false, "f16"),
              count: 2,
            },
            {
              label: "f16 on a sliding key",
              match: (key: string) => key.includes(":sliding") && key.endsWith(":kv16"),
              count: 0,
            },
          ],
          name,
        );
        assertEquals(
          countDispatches(
            requireCensus(arms.rounded.session.diagnostics().lastRunPipelines, name),
            SEAT_SIGNATURES.fullStateStorage.f16,
          ),
          0,
          "the f32 Session ran an f16 key",
        );
        assert(
          countDispatches(requireCensus(census, name), SEAT_SIGNATURES.fullStateStorage.f16) > 0,
          "the census reports no f16 key",
        );
      });
    });
  }

  it("sliding slots keep today's numbers and keys under f16", async () => {
    const shape = { heads: 4, kvHeads: 2, depth: 8, slots: "sliding" } as const;
    await withArms(shape, 4, 64, "sequential", async (arms) => {
      for (let step = 0; step < 4; step += 1) {
        await stepAll(arms, shape, 4, 4, step * 37, `step ${step}`);
      }
      assertEquals(
        countDispatches(
          requireCensus(arms.packed.session.diagnostics().lastRunPipelines, "sliding"),
          SEAT_SIGNATURES.fullStateStorage.f16,
        ),
        0,
      );
    });
  });

  it("a fork is an f16 copy: it diverges from its source and each stays on the contract", async () => {
    const shape = { heads: 4, kvHeads: 2, depth: 8, slots: "full" } as const;
    await withArms(shape, 4, 64, "parallel", async (arms) => {
      await stepAll(arms, shape, 4, 4, 0, "before the fork");
      await stepAll(arms, shape, 4, 3, 101, "before the fork");
      const packedFork = await arms.packed.context.fork();
      const roundedFork = await arms.rounded.context.fork();
      try {
        const slots = packedFork[RUNTIME_INTERNAL].slots;
        assertEquals(slots.get("kf")?.storage, "f16");
        assertEquals(slots.get("kf")?.byteLength, 1 * 2 * 64 * 8 * 2);
        // Different rows on each side, so a fork that still bound the source's slots would show.
        const onForks = async (salt: number, label: string): Promise<void> => {
          const packed = await runArm(
            { session: arms.packed.session, context: packedFork },
            shape,
            4,
            4,
            salt,
            false,
          );
          const host = await runArm(
            { session: arms.rounded.session, context: roundedFork },
            shape,
            4,
            4,
            salt,
            true,
          );
          assertEquals(words(packed["o"].data), words(host["o"].data), label);
        };
        await onForks(7, "the fork's first run");
        await stepAll(arms, shape, 4, 4, 303, "the source after the fork");
        await onForks(11, "the fork's second run");
      } finally {
        await packedFork.dispose();
        await roundedFork.dispose();
      }
    });
  });

  it("a rewind leaves no trace: the overwritten rows read as if never appended", async () => {
    const shape = { heads: 4, kvHeads: 2, depth: 8, slots: "full" } as const;
    const gpu = await acquireGpu();
    try {
      const session = await sessionOf(gpu, fullStateGraph(shape), { fullStateStorage: "f16" });
      const generation = { chunkLength: 4, bindings: { C: 64 } };
      const rewound = await session.createGenerationContext(generation);
      const straight = await session.createGenerationContext(generation);
      const run = async (
        context: GenerationContext,
        query: number,
        salt: number,
      ): Promise<Tensor> => (await runArm({ session, context }, shape, 4, query, salt, false))["o"];
      try {
        await run(rewound, 4, 0);
        await run(rewound, 2, 11);
        await run(rewound, 4, 22);
        rewound.rewind(5);
        await run(straight, 4, 0);
        await run(straight, 1, 11);
        assertEquals(straight.pastLength, 5);
        const after = await run(rewound, 4, 33);
        assertEquals(words(after.data), words((await run(straight, 4, 33)).data), "after a rewind");
        const later = await run(rewound, 3, 44);
        assertEquals(words(later.data), words((await run(straight, 3, 44)).data), "a later run");
      } finally {
        await rewound.dispose();
        await straight.dispose();
        await session.dispose();
      }
    } finally {
      gpu.destroy();
    }
  });
});

describe.skipIf(!GPU_AVAILABLE)("fullStateStorage: borrowing (real GPU)", () => {
  const BOTH = { heads: 4, kvHeads: 2, depth: 8, slots: "both" } as const;
  const FULL = { ...BOTH, slots: "full" } as const;
  const SLIDING = { ...BOTH, slots: "sliding" } as const;

  const borrowerInput = (salt: number): Record<string, Tensor> => ({
    q: tensor(
      [1, BOTH.heads, 1, BOTH.depth],
      seeded(BOTH.heads * BOTH.depth, (i) => QUERY(i + salt)),
    ),
  });

  it("a borrower with another storage fails loudly, in both directions and on a fork", async () => {
    const gpu = await acquireGpu();
    try {
      const generation = { chunkLength: 4, bindings: { C: 64 } };
      for (const [lenderStorage, borrowerStorage] of [
        ["f16", "f32"],
        ["f32", "f16"],
      ] as const) {
        const lender = await sessionOf(gpu, fullStateGraph(FULL), {
          fullStateStorage: lenderStorage,
        });
        const borrower = await sessionOf(gpu, fullStateBorrowerGraph(FULL), {
          fullStateStorage: borrowerStorage,
        });
        const context = await lender.createGenerationContext(generation);
        const fork = await context.fork();
        try {
          for (const borrow of [context, fork]) {
            await assertRejects(
              () => borrower.createGenerationContext({ chunkLength: 1, borrow }),
              ExecutionError,
              "fullStateStorage",
            );
          }
        } finally {
          await fork.dispose();
          await context.dispose();
          await borrower.dispose();
          await lender.dispose();
        }
      }
    } finally {
      gpu.destroy();
    }
  });

  it("the option alone does not refuse a sliding slot: its storage is the same either way", async () => {
    const gpu = await acquireGpu();
    try {
      const lender = await sessionOf(gpu, fullStateGraph(SLIDING), { fullStateStorage: "f16" });
      const borrower = await sessionOf(gpu, fullStateBorrowerGraph(SLIDING), {});
      const context = await lender.createGenerationContext({
        chunkLength: 4,
        bindings: { C: 64 },
      });
      const borrowed = await borrower.createGenerationContext({ chunkLength: 1, borrow: context });
      await borrowed.dispose();
      await context.dispose();
      await borrower.dispose();
      await lender.dispose();
    } finally {
      gpu.destroy();
    }
  });

  it("a drafter's readonly kernels on an f16 lender equal an f32 pair over the host-rounded slot", async () => {
    const gpu = await acquireGpu();
    try {
      const generation = { chunkLength: 4, bindings: { C: 64 } };
      const armOf = async (fullStateStorage: "f32" | "f16") => {
        const lender = await sessionOf(gpu, fullStateGraph(BOTH), { fullStateStorage });
        const borrower = await sessionOf(gpu, fullStateBorrowerGraph(BOTH), { fullStateStorage });
        const context = await lender.createGenerationContext(generation);
        const borrowed = await borrower.createGenerationContext({
          chunkLength: 1,
          borrow: context,
        });
        return { lender, borrower, context, borrowed };
      };
      const packed = await armOf("f16");
      const host = await armOf("f32");
      try {
        for (let step = 0; step < 3; step += 1) {
          const salt = step * 53;
          await packed.lender.run(
            stepInputs(BOTH, 4, salt, false),
            {},
            {
              context: packed.context,
              queryLength: 4,
            },
          );
          // The sliding slots of the graph take the same `append_*` rows, so the host arm's are
          // rounded too; the comparison below is on the full slot's output `o` only.
          await host.lender.run(
            stepInputs(BOTH, 4, salt, true),
            {},
            {
              context: host.context,
              queryLength: 4,
            },
          );
          const onPacked = await packed.borrower.run(
            borrowerInput(salt),
            {},
            {
              context: packed.borrowed,
              queryLength: 1,
            },
          );
          const onHost = await host.borrower.run(
            borrowerInput(salt),
            {},
            {
              context: host.borrowed,
              queryLength: 1,
            },
          );
          assertEquals(words(onPacked["o"].data), words(onHost["o"].data), `step ${step}`);
          assert(
            (onPacked["o"].data as Float32Array).some((value) => Math.abs(value) > 1e-3),
            `step ${step}: the output is trivial`,
          );
        }
        assertPipelineCensus(
          packed.borrower.diagnostics().lastRunPipelines,
          [
            {
              label: "the f16 readonly ①'",
              match: stateQkParallelReadonlyKey(false, true, "f16"),
              atLeast: 1,
            },
            {
              label: "the f16 readonly ③'",
              match: statePvParallelReadonlyKey(false, true, "f16"),
              atLeast: 1,
            },
            {
              label: "the f32 readonly ①' on the sliding slot",
              match: stateQkParallelReadonlyKey(true, true),
              atLeast: 1,
            },
          ],
          "the drafter",
        );
      } finally {
        for (const arm of [packed, host]) {
          await arm.borrowed.dispose();
          await arm.context.dispose();
          await arm.borrower.dispose();
          await arm.lender.dispose();
        }
      }
    } finally {
      gpu.destroy();
    }
  });
});
