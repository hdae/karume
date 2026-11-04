// The Session lifecycle and fail loudly (real GPU — createSession requires a device).
// The one GPU-independent case (the IR parser's binding check) lives in runtime_executor_test.ts.
// The exception is the "__proto__" pair: the krm-side rejection and the memory-path execution share
// the same two fixtures, so only one half (the krm side) sits here without a GPU.

import { test } from "vite-plus/test";
import { assertEquals, assertRejects, assertStringIncludes } from "@std/assert";
import { DispatchLimitError } from "../src/codegen/errors.ts";
import { ContainerFormatError } from "../src/format/container/header.ts";
import { acquireGpu } from "../src/gpu/device.ts";
import { defaultGemmGeometry, gemmTileN } from "../src/kernels/gemm-geometry.ts";
import { DEFAULT_SUBMIT_POLICY } from "../src/gpu/submit.ts";
import { RuntimeSupportError } from "../src/ops/support.ts";
import { allclose, compareTensors, formatAllclose } from "../src/reference/allclose.ts";
import { applyReferenceOp, applyReferenceOpOutputs, refTensor } from "../src/reference/ops.ts";
import {
  createSessionFromContainer,
  type SessionOptions,
  type Tensor,
} from "../src/runtime/executor.ts";
import { ExecutionError } from "../src/runtime/plan.ts";
import { B, chainGraph, openChainModel, W } from "./helpers/chain-graph.ts";
import {
  type DeclarationJson,
  f32Bytes,
  fill,
  memoryModel,
  openGraphModel,
} from "./helpers/model-fixture.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";

const expectedChain = (x: Tensor): ArrayLike<number> => {
  const h = applyReferenceOp("matmul", [x, refTensor([4, 3], W)]);
  const g = applyReferenceOp("add", [h, refTensor([3], B)]);
  return applyReferenceOp("relu", [g]).data;
};

test.skipIf(!GPU_AVAILABLE)(
  "a chain graph with initializers matches the CPU reference and returns only the graph outputs (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    const session = await createSessionFromContainer(gpu, await openChainModel(), "model");
    try {
      const x = fill([5, 4], (i) => ((i % 11) - 5) * 0.5);
      const outputs = await session.run({ x });
      assertEquals(Object.keys(outputs), ["y"]);
      assertEquals(outputs["y"].shape, [5, 3]);
      const report = allclose(outputs["y"].data, expectedChain(x));
      assertEquals(report.pass, true, formatAllclose(report));

      // The same Session can rebind T to another value
      const x2 = fill([2, 4], (i) => i - 3);
      const outputs2 = await session.run({ x: x2 });
      assertEquals(outputs2["y"].shape, [2, 3]);
      assertEquals(allclose(outputs2["y"].data, expectedChain(x2)).pass, true);
    } finally {
      await session.dispose();
      gpu.destroy();
    }
  },
);

/**
 * Minimal graphs that use "__proto__" as a name, split into input / symbol name (the first) and
 * output name (the second) — the IR declares a name in exactly one place, so both cannot share one
 * graph.
 *
 * MUST: put "__proto__" into `values` with the computed key `["__proto__"]`. A literal
 * `__proto__:` sets [[Prototype]] instead of an own key, so it never reaches JSON.stringify and the
 * test misses what it means to check.
 */
const protoInputGraph = (): DeclarationJson => ({
  format: "karume-ir",
  version: 2,
  requires: { ops: ["relu"] },
  symbols: ["__proto__"],
  inputs: [{ name: "__proto__", dtype: "f32", shape: ["__proto__", 3] }],
  outputs: ["y"],
  initializers: {},
  values: { y: { dtype: "f32", shape: ["__proto__", 3] } },
  nodes: [{ op: "relu", ins: ["__proto__"], outs: ["y"], attrs: {} }],
});

const protoOutputGraph = (): DeclarationJson => ({
  format: "karume-ir",
  version: 2,
  requires: { ops: ["relu"] },
  symbols: ["T"],
  inputs: [{ name: "x", dtype: "f32", shape: ["T", 3] }],
  outputs: ["__proto__"],
  initializers: {},
  values: { ["__proto__"]: { dtype: "f32", shape: ["T", 3] } },
  nodes: [{ op: "relu", ins: ["x"], outs: ["__proto__"], attrs: {} }],
});

// The krm path runs the descriptor document's JSON through `assertJsonSafe`, so a declaration
// where "__proto__" appears as a **key** (= an output name / value name) fails loudly there — on
// the krm side, "not silently dropped" takes the form "rejected", not "the name gets through". An
// input name / symbol name appears only as a JSON value, so krm accepts it too, and whether the
// name gets through is decided by the same path as the memory path.
// It needs no GPU, so it sits outside the ignore.
test("krm rejects a declaration with '__proto__' as a value name instead of silently dropping it", async () => {
  const error = await assertRejects(() => openGraphModel(protoOutputGraph()), ContainerFormatError);
  assertStringIncludes(error.message, "__proto__");
});

test.skipIf(!GPU_AVAILABLE)(
  "'__proto__' as an input name / symbol name / output name is not silently dropped by the in-memory container (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    const x = fill([4, 3], (i) => ((i % 7) - 3) * 0.5);
    const expected = applyReferenceOp("relu", [x]).data;
    try {
      const byInput = await createSessionFromContainer(
        gpu,
        memoryModel(protoInputGraph(), []),
        "model",
      );
      try {
        // The computed key is a MUST (same reason as the fixture above — with a literal, the name
        // never reaches run)
        const outputs = await byInput.run({ ["__proto__"]: x });
        assertEquals(allclose(outputs["y"].data, expected).pass, true);
      } finally {
        await byInput.dispose();
      }

      const byOutput = await createSessionFromContainer(
        gpu,
        memoryModel(protoOutputGraph(), []),
        "model",
      );
      try {
        const outputs = await byOutput.run({ x });
        // With a plain `{}` again, in the browser the Tensor would turn into [[Prototype]] and only
        // this output would vanish. In Deno an own property gets created, so hasOwn / keys cannot
        // detect the regression; pin that the container has a null prototype as well.
        assertEquals(Object.getPrototypeOf(outputs), null);
        assertEquals(Object.hasOwn(outputs, "__proto__"), true);
        assertEquals(Object.keys(outputs), ["__proto__"]);
        assertEquals(allclose(outputs["__proto__"].data, expected).pass, true);
      } finally {
        await byOutput.dispose();
      }
    } finally {
      gpu.destroy();
    }
  },
);

/**
 * A chain that exercises the reshape / identity-expand alias (ADR 0011).
 *
 * - `r` is a graph output that points directly at the real buffer of the intermediate `h` (pinning
 *   works through the alias)
 * - the allocation of `g` must not take the buffer of `h` (the alias `r` is still alive)
 * - `q` is only consumed, as an alias (the consumer count decrements correctly through the alias)
 *
 * If the count is off by even one, the allocation of `g` receives the buffer of `h` from the pool
 * and the contents of `r` silently turn into exp(x) (no exception).
 */
const aliasGraph = (aliasOp: "reshape" | "expand"): DeclarationJson => {
  // An identity expand has no replicated axis, so it is aliased only when the input and output
  // shapes match exactly.
  const aliasShape = aliasOp === "reshape" ? ["2T", 2] : ["T", 4];
  return {
    format: "karume-ir",
    version: 2,
    requires: { ops: ["neg", aliasOp, "exp", "add"] },
    symbols: ["T"],
    inputs: [{ name: "x", dtype: "f32", shape: ["T", 4] }],
    outputs: ["r", "s"],
    initializers: {},
    values: {
      h: { dtype: "f32", shape: ["T", 4] },
      r: { dtype: "f32", shape: aliasShape },
      g: { dtype: "f32", shape: ["T", 4] },
      q: { dtype: "f32", shape: aliasShape },
      s: { dtype: "f32", shape: aliasShape },
    },
    nodes: [
      { op: "neg", ins: ["x"], outs: ["h"], attrs: {} },
      { op: aliasOp, ins: ["h"], outs: ["r"], attrs: {} },
      { op: "exp", ins: ["x"], outs: ["g"], attrs: {} },
      { op: aliasOp, ins: ["g"], outs: ["q"], attrs: {} },
      { op: "add", ins: ["r", "q"], outs: ["s"], attrs: {} },
    ],
  };
};

for (const [aliasOp, expectedShape, identityExpands] of [
  ["reshape", [6, 2], 0],
  ["expand", [3, 4], 2],
] as const) {
  test.skipIf(!GPU_AVAILABLE)(
    `the ${aliasOp} alias needs no copy and a live alias is not handed out to a later allocation (real GPU)`,
    async () => {
      const gpu = await acquireGpu();
      const session = await createSessionFromContainer(
        gpu,
        await openGraphModel(aliasGraph(aliasOp)),
        "model",
      );
      try {
        const x = fill([3, 4], (i) => ((i % 7) - 3) * 0.5);
        const outputs = await session.run({ x });
        assertEquals(Object.keys(outputs).sort(), ["r", "s"]);
        assertEquals(outputs["r"].shape, expectedShape);
        assertEquals(outputs["s"].shape, expectedShape);

        const negated = applyReferenceOp("neg", [x]);
        const exponent = applyReferenceOp("exp", [x]);
        // Even through an alias, the graph output reads back the original values (with reshape,
        // only the shape becomes [6,2])
        assertEquals(
          allclose(outputs["r"].data, negated.data).pass,
          true,
          "readback of the alias output",
        );
        const sum = applyReferenceOp("add", [negated, exponent]);
        assertEquals(
          allclose(outputs["s"].data, sum.data).pass,
          true,
          "the value downstream of the alias",
        );

        // Pin from the recorded stats that no materializing copy was made. The live peak under pool
        // management (the output storage dispatches write) is h / g / s, three buffers = 48×3. If
        // the alias side allocated, r / q would add on top.
        assertEquals(session.diagnostics().lastRun?.peakTransientBytes, 144);
        // Only three dispatches: neg / exp / add (reshape / identity expand both take none)
        assertEquals(session.diagnostics().submit.dispatchCount, 3);
        assertEquals(
          session.diagnostics().lastRunFusions?.identityExpand,
          identityExpands,
          "alias counter",
        );
      } finally {
        await session.dispose();
        gpu.destroy();
      }
    },
  );
}

/**
 * Lifetimes of a multi-output node (ADR 0068 acceptance criterion ④ — `topk` is the first user).
 *
 * The two outputs of `topk` **have different lifetimes**:
 *
 * - `v` (values) is an intermediate consumed once by `neg` → returns to the pool after that step
 * - `i` (indices) is a graph output → **pinned, never returned to the pool**
 * - the allocation of `e = exp(neg(v))` may take the storage of `v` (`v` is consumed) but must not
 *   take the storage of `i`
 *
 * If the per-slot `uses` / `pinned` is off by even one, the allocation of `e` receives the buffer
 * of `i` from the pool, and **the indices' readback silently turns into exp's f32 bit patterns**
 * (no exception — the same silent failure as in the alias test). Conversely, if `v` is not
 * released, `e` takes new storage and the live peak grows by one buffer.
 *
 * The second graph is the shape where **one output is consumed by no one** (an unreachable value
 * with `uses = 0` that is not a graph output either). If the definition's retain is not released,
 * `assertDrained` at the end of the run fails.
 *
 * MUST: run both shapes **twice on the same Session**. The slot backing activates only on a hit run
 * (`Session.#activateBacking`), so with one run each, a missing or swapped slot-1 binding of
 * `bakeBindGroups`, pinned output mapping or backed readback hides behind the arena path's green.
 */
const topkLifetimeGraph = (pinIndices: boolean): DeclarationJson => ({
  format: "karume-ir",
  version: 2,
  requires: { ops: ["topk", "neg", "exp"] },
  symbols: [],
  inputs: [{ name: "x", dtype: "f32", shape: [3, 8] }],
  outputs: pinIndices ? ["e", "i"] : ["e"],
  initializers: {},
  values: {
    v: { dtype: "f32", shape: [3, 2] },
    i: { dtype: "i32", shape: [3, 2] },
    nv: { dtype: "f32", shape: [3, 2] },
    e: { dtype: "f32", shape: [3, 2] },
  },
  nodes: [
    { op: "topk", ins: ["x"], outs: ["v", "i"], attrs: { k: 2 } },
    { op: "neg", ins: ["v"], outs: ["nv"], attrs: {} },
    { op: "exp", ins: ["nv"], outs: ["e"], attrs: {} },
  ],
});

test.skipIf(!GPU_AVAILABLE)(
  "a multi-output node's per-slot uses / pin are correct, and values stay intact when only one output is consumed (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    try {
      const x = fill([3, 8], (i) => ((i % 7) - 3) * 0.5 + (i % 3) * 0.25);
      const expected = applyReferenceOpOutputs("topk", [x], { k: 2 });
      const negated = applyReferenceOp("neg", [expected[0]]);
      const exponent = applyReferenceOp("exp", [negated]);

      // ① The indices as a graph output (pinning works on slot 1 of a multi-output node)
      const session = await createSessionFromContainer(
        gpu,
        await openGraphModel(topkLifetimeGraph(true)),
        "model",
      );
      try {
        const outputs = await session.run({ x });
        assertEquals(Object.keys(outputs).sort(), ["e", "i"]);
        assertEquals(outputs["i"].dtype, "i32");
        // Not one index word is corrupted (if handed out again from the pool, f32 bit patterns
        // would appear)
        assertEquals([...outputs["i"].data], [...expected[1].data], "readback of the indices");
        assertEquals(
          allclose(outputs["e"].data, exponent.data).pass,
          true,
          "downstream of the values",
        );
        // The live peak under pool management is v / i / nv, three buffers (24 bytes each) = 72.
        // `e` reuses the storage of the consumed `v`, so nothing adds on top, and it never takes
        // `i`.
        assertEquals(session.diagnostics().lastRun?.peakTransientBytes, 72);
        // Three dispatches: topk / neg / exp (topk writes both outputs in one dispatch)
        assertEquals(session.diagnostics().submit.dispatchCount, 3);

        // The second run on the same Session = hits the derived plan and enters the **slot
        // backing path** (bind groups are baked at build time, and the readback targets are a
        // mapping fixed at build time). Even if the multi-output slot-1 binding, the pinned output
        // mapping or the backed readback were missing or swapped, the first run on the arena path
        // alone would stay green.
        const again = await session.run({ x });
        assertEquals(
          session.diagnostics().lastRunPrepared?.hit,
          true,
          "run 2 uses the derived plan",
        );
        assertEquals(session.diagnostics().planBacking.buildCount, 1, "the backing was built");
        assertEquals(Object.keys(again).sort(), ["e", "i"]);
        assertEquals(again["i"].dtype, "i32");
        assertEquals(
          [...again["i"].data],
          [...expected[1].data],
          "readback of the indices on the backed path",
        );
        assertEquals(
          allclose(again["e"].data, exponent.data).pass,
          true,
          "the values on the backed path",
        );
        // The queued command sequence is identical to the arena path (3 → 6 cumulative)
        assertEquals(session.diagnostics().submit.dispatchCount, 6);
      } finally {
        await session.dispose();
      }

      // ② The indices consumed by no one (an unreachable value — confirms that the definition's
      // release balances; if not, assertDrained at the end of the run fails)
      const orphan = await createSessionFromContainer(
        gpu,
        await openGraphModel(topkLifetimeGraph(false)),
        "model",
      );
      try {
        const outputs = await orphan.run({ x });
        assertEquals(Object.keys(outputs), ["e"]);
        assertEquals(
          allclose(outputs["e"].data, exponent.data).pass,
          true,
          "the values are unchanged",
        );
        // Indices with zero consumers return to the pool at the end of their step, so `nv` takes
        // that storage (the live peak is v / i, two buffers = 48).
        assertEquals(orphan.diagnostics().lastRun?.peakTransientBytes, 48);

        // The second run of the orphan shape is on the backing path too. If the readback mapping
        // mixed up slots, `e` would take unpinned storage (`Session.#isReadable` allows only pinned
        // slots, so most such cases fail with an exception; the value comparison here catches the
        // ones that get through).
        const again = await orphan.run({ x });
        assertEquals(
          orphan.diagnostics().lastRunPrepared?.hit,
          true,
          "run 2 uses the derived plan",
        );
        assertEquals(orphan.diagnostics().planBacking.buildCount, 1, "the backing was built");
        assertEquals(Object.keys(again), ["e"]);
        assertEquals(
          allclose(again["e"].data, exponent.data).pass,
          true,
          "unchanged on the backed path too",
        );
      } finally {
        await orphan.dispose();
      }
    } finally {
      gpu.destroy();
    }
  },
);

test.skipIf(!GPU_AVAILABLE)(
  "concurrent runs on one Session are serialized and all match the CPU reference (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    const session = await createSessionFromContainer(gpu, await openChainModel(), "model");
    try {
      const batch = [
        fill([5, 4], (i) => ((i % 11) - 5) * 0.5),
        fill([2, 4], (i) => i - 3),
        fill([7, 4], (i) => ((i % 7) - 3) * 0.25),
        fill([1, 4], (i) => i + 1),
      ];
      // Issue them all without await. Without serialization, the per-device LIFO errorScope would
      // interleave across runs, and a run would miss its own validation failure or an unrelated run
      // would fail.
      const outputs = await Promise.all(batch.map((x) => session.run({ x })));
      for (const [index, x] of batch.entries()) {
        assertEquals(outputs[index]["y"].shape, [x.shape[0], 3], `shape of run ${index}`);
        const report = allclose(outputs[index]["y"].data, expectedChain(x));
        assertEquals(report.pass, true, `run ${index}: ${formatAllclose(report)}`);
      }
      assertEquals(session.diagnostics().submit.dispatchCount, 3 * batch.length);
    } finally {
      await session.dispose();
      gpu.destroy();
    }
  },
);

test.skipIf(!GPU_AVAILABLE)(
  "dispose while a run is in flight destroys the weights after the run completes (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    const session = await createSessionFromContainer(gpu, await openChainModel(), "model");
    try {
      const x = fill([6, 4], (i) => ((i % 9) - 4) * 0.5);
      const running = session.run({ x });
      // Dispose without await. Without serialization, the weight buffers would be destroyed under
      // the running run, the submit would fail with its whole command buffer, and wrong values
      // would silently remain.
      const disposing = session.dispose();
      const outputs = await running;
      await disposing;
      assertEquals(allclose(outputs["y"].data, expectedChain(x)).pass, true);
      // dispose takes effect at the call (any later run fails without waiting)
      await assertRejects(() => session.run({ x }), ExecutionError);
    } finally {
      gpu.destroy();
    }
  },
);

test.skipIf(!GPU_AVAILABLE)(
  "capping the chunk to 1 with submitPolicy submits after every dispatch (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    // With the default initialChunkSize=16, a 3-dispatch graph never goes through a mid-chunk
    // submit
    const session = await createSessionFromContainer(gpu, await openChainModel(), "model", {
      submitPolicy: {
        timeBudgetMs: DEFAULT_SUBMIT_POLICY.timeBudgetMs,
        initialChunkSize: 1,
        minChunkSize: 1,
        maxChunkSize: 1,
      },
    });
    try {
      const x = fill([6, 4], (i) => ((i % 7) - 3) * 0.5);
      const outputs = await session.run({ x });
      const submit = session.diagnostics().submit;
      assertEquals(submit.dispatchCount, 3);
      // On the single-fence path (gpuTiming OFF, one output) the readback copy is queued into
      // **the same command sequence** too, and with a chunk cap of 1 it takes a chunk of its own
      // (H-1). So there are four submits: "3 dispatches + 1 copy". The dispatch count stays 3.
      assertEquals(submit.submitCount, 4, "the chunk cap is 1, so every command is submitted");
      const report = allclose(outputs["y"].data, expectedChain(x));
      assertEquals(report.pass, true, formatAllclose(report));
    } finally {
      await session.dispose();
      gpu.destroy();
    }
  },
);

test.skipIf(!GPU_AVAILABLE)(
  "diagnostics report the pipeline count and the latest run's intermediate buffer reuse (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    const session = await createSessionFromContainer(gpu, await openChainModel(), "model");
    try {
      assertEquals(session.diagnostics().lastRun, undefined);
      // The two weights were uploaded by the explicit async stage
      assertEquals(session.diagnostics().weights.allocCount, 2);

      await session.run({ x: fill([8, 4], (i) => i * 0.25) });
      const stats = session.diagnostics();
      assertEquals(stats.pipelineCount, 3);
      assertEquals(stats.submit.dispatchCount, 3);
      // h is released after add is encoded and handed out again for relu's output
      assertEquals((stats.lastRun?.reuseCount ?? 0) >= 1, true);
      assertEquals((stats.lastRun?.peakTransientBytes ?? 0) > 0, true);
    } finally {
      await session.dispose();
      gpu.destroy();
    }
  },
);

/**
 * Builds **a caller from outside the types** (a JS consumer, an old build that still uses a
 * spelling from before a rename). The TS types guard none of these paths, so the accepted-set check
 * can only be exercised here.
 */
const jsCallerOptions = (options: Readonly<Record<string, string>>): SessionOptions =>
  options as unknown as SessionOptions;

test.skipIf(!GPU_AVAILABLE)(
  "spellings outside an execution knob's union fail at Session build with all of them listed (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    const opened = await openChainModel();
    try {
      // `"i8a8"` is the spelling renamed to `"a8"` in 0.5.0 (ADR 0074 decision 3, no
      // compatibility shim). Without the check, every downstream equality test (`=== "a8"`)
      // misses and the run takes the default f32 without the opt-in applied = a silent failure that
      // reads as "measured a8".
      const error = await assertRejects(
        () =>
          createSessionFromContainer(
            gpu,
            opened,
            "model",
            jsCallerOptions({ linearCompute: "i8a8", attentionScoreStorage: "s16" }),
          ),
        ExecutionError,
        "outside the accepted set: 2",
      );
      assertEquals(error.message.includes(`linearCompute: "i8a8"`), true, error.message);
      assertEquals(error.message.includes(`attentionScoreStorage: "s16"`), true, error.message);
      // The message spells out the accepted set itself (the diagnostic alone tells what passes).
      assertEquals(error.message.includes("'f32' / 'a8' / 'f16'"), true, error.message);

      // Control: spellings inside the union still build (proof that the above is not "fails
      // whatever is passed").
      const session = await createSessionFromContainer(gpu, opened, "model", {
        linearCompute: "a8",
        attentionScoreStorage: "f16",
      });
      await session.dispose();
    } finally {
      gpu.destroy();
    }
  },
);

test.skipIf(!GPU_AVAILABLE)(
  "the parallel GEMV reduction rejects an invalid spelling and unimplemented compute precisions at build time",
  async () => {
    const gpu = await acquireGpu();
    const opened = await openChainModel();
    try {
      for (const options of [
        jsCallerOptions({ linearGemvReduce: "paralel" }),
        { linearGemvReduce: "parallel", linearCompute: "f16" } as const,
        { linearGemvReduce: "parallel", linearCompute: "a8" } as const,
        { linearGemvReduce: "parallel-subgroup32", linearCompute: "f16" } as const,
        { linearGemvReduce: "parallel-subgroup32", linearCompute: "a8" } as const,
      ]) {
        await assertRejects(
          () => createSessionFromContainer(gpu, opened, "model", options),
          ExecutionError,
          "linearGemvReduce",
        );
      }
    } finally {
      gpu.destroy();
    }
  },
);

test.skipIf(!GPU_AVAILABLE)(
  "planBackingBudgetBytes accepts only non-negative safe integers (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    const opened = await openChainModel();
    try {
      // The budget is "the cap on the retained set" (ADR 0095 decision 1), and also the amount
      // the estimate (estimate.ts) books. Silently accepting a negative / non-integer / NaN value
      // makes the comparison in `#evictBackingsFor` always false / always true, silently turning
      // into "evict everything every run" or "retain without limit".
      for (const budget of [-1, 1.5, Number.NaN]) {
        await assertRejects(
          () =>
            createSessionFromContainer(gpu, opened, "model", {
              planBackingBudgetBytes: budget,
            }),
          ExecutionError,
          "non-negative safe integer",
        );
      }
      // Control: 0 (the former capacity of 1) and positive integers pass (proof that the three
      // above are not "fails whatever is passed").
      for (const budget of [0, 4096]) {
        const session = await createSessionFromContainer(gpu, opened, "model", {
          planBackingBudgetBytes: budget,
        });
        await session.dispose();
      }
    } finally {
      gpu.destroy();
    }
  },
);

test.skipIf(!GPU_AVAILABLE)(
  "linearGemvRowsThreadTarget accepts only safe integers of at least 1 (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    const opened = await openChainModel();
    try {
      // The target is the threshold: rows is halved until "thread count = n × y tile count"
      // reaches it (src/kernels/linear-gemv.ts). Silently accepting 0 / a negative / NaN makes the
      // comparison always false, and **a different run** — "rows always at the ceiling" — runs
      // looking like the default (the measured condition is misreported).
      for (const target of [0, -1, 1.5, Number.NaN]) {
        await assertRejects(
          () =>
            createSessionFromContainer(gpu, opened, "model", {
              linearGemvRowsThreadTarget: target,
            }),
          ExecutionError,
          "safe integer of at least 1",
        );
      }
      // Control: positive integers pass (proof that the four above are not "fails whatever is
      // passed").
      for (const target of [1, 2048]) {
        const session = await createSessionFromContainer(gpu, opened, "model", {
          linearGemvRowsThreadTarget: target,
        });
        await session.dispose();
      }
    } finally {
      gpu.destroy();
    }
  },
);

test.skipIf(!GPU_AVAILABLE)(
  "diagnostics right after a failed run keep no stats from the previous successful run (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    const session = await createSessionFromContainer(gpu, await openChainModel(), "model");
    try {
      // Control: right after a successful run, all four "latest run" fields are filled.
      await session.run({ x: fill([8, 4], (i) => i * 0.25) });
      const success = session.diagnostics();
      assertEquals(success.lastRun !== undefined, true);
      assertEquals(success.lastRunParams !== undefined, true);
      assertEquals(success.lastRunFusions !== undefined, true);
      assertEquals(success.lastRunPrepared !== undefined, true);

      // A run that passes derivation (up to the fusion decisions) and fails the input validation
      // (an i32 passed for a declared f32).
      await assertRejects(
        () => session.run({ x: { dtype: "i32", shape: [8, 4], data: new Int32Array(32) } }),
        ExecutionError,
        "declared dtype",
      );

      // The four fields reflect the same single run (= the failed run). Only the fusion counts
      // remain, because they are the failed run's own derivation result; the three fields filled
      // only on the success path fall back to undefined — if this split, one snapshot would mix
      // "the failed run's fusion counts" with "the previous successful run's arena / params stats"
      // (the diagnostic most wanted at the moment of isolating a failure).
      const failed = session.diagnostics();
      assertEquals(failed.lastRun, undefined);
      assertEquals(failed.lastRunParams, undefined);
      assertEquals(failed.lastRunPrepared, undefined);
      assertEquals(failed.lastRunFusions !== undefined, true);
    } finally {
      await session.dispose();
      gpu.destroy();
    }
  },
);

/**
 * `d` is an intermediate output that no one reads and that is not a graph output (the IR does not
 * forbid unreachable values). A value with zero consumers that is not released at the node boundary
 * drops out of pool reuse, one more allocation is made, and peakTransientBytes comes out larger
 * than it really is.
 */
const deadValueGraph = (): DeclarationJson => ({
  ...chainGraph(),
  values: {
    w: { dtype: "f32", shape: [4, 3] },
    b: { dtype: "f32", shape: [3] },
    h: { dtype: "f32", shape: ["T", 3] },
    d: { dtype: "f32", shape: ["T", 3] },
    g: { dtype: "f32", shape: ["T", 3] },
    y: { dtype: "f32", shape: ["T", 3] },
  },
  nodes: [
    { op: "matmul", ins: ["x", "w"], outs: ["h"], attrs: {} },
    { op: "relu", ins: ["h"], outs: ["d"], attrs: {} },
    { op: "add", ins: ["h", "b"], outs: ["g"], attrs: {} },
    { op: "relu", ins: ["g"], outs: ["y"], attrs: {} },
  ],
});

test.skipIf(!GPU_AVAILABLE)(
  "an intermediate output with zero consumers also returns to the pool at the node boundary (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    const session = await createSessionFromContainer(
      gpu,
      await openChainModel(deadValueGraph()),
      "model",
    );
    try {
      const x = fill([8, 4], (i) => ((i % 11) - 5) * 0.5);
      const outputs = await session.run({ x });
      // Even with the unreachable d in between, the final result is the same as the chain graph's
      const report = allclose(outputs["y"].data, expectedChain(x));
      assertEquals(report.pass, true, formatAllclose(report));

      // The four intermediates (h/d/g/y) run on two real allocations. If d lingered, reuse would be
      // 1 and the peak three buffers.
      const lastRun = session.diagnostics().lastRun;
      assertEquals(lastRun?.reuseCount, 2);
      assertEquals(lastRun?.peakTransientBytes, 2 * 8 * 3 * 4);
    } finally {
      await session.dispose();
      gpu.destroy();
    }
  },
);

test.skipIf(!GPU_AVAILABLE)(
  "a missing capability (unsupported op / unsupported semantic dtype) fails at Session build (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    try {
      // conv_transpose2d is not even in the vocabulary (the allowlist freeze of ADR 0007) — it is
      // not in the contract table, so if it appears it fails, listed as a missing capability.
      // NOTE: conv2d entered the contract table in ADR 0017 (it showed up in Anima's VAE decoder),
      // so this op takes over as the representative of "stored but not executable".
      const foreign = chainGraph();
      foreign.requires.ops = ["matmul", "add", "conv_transpose2d"];
      foreign.nodes[2] = { op: "conv_transpose2d", ins: ["g"], outs: ["y"], attrs: {} };
      const foreignModel = await openChainModel(foreign);
      await assertRejects(
        () => createSessionFromContainer(gpu, foreignModel, "model"),
        RuntimeSupportError,
      );

      // NOTE: an unsupported **storage** can no longer be built — every codec in the ledger is
      // executable (f16 by ADR 0018, bf16 by ADR 0131). The dtype-side representative is "a
      // semantic dtype the supporting op does not accept" (an i32 matmul weight).
      const unsupportedDtype = chainGraph();
      unsupportedDtype.values["w"] = { dtype: "i32", shape: [4, 3] };
      const i32Model = await openChainModel(unsupportedDtype, [
        {
          graph: "model",
          initializer: "w",
          bytes: new Uint8Array(new ArrayBuffer(48)),
          encoding: { codec: "i32" },
        },
        {
          graph: "model",
          initializer: "b",
          bytes: f32Bytes([...B]),
          encoding: { codec: "f32" },
        },
      ]);
      await assertRejects(
        () => createSessionFromContainer(gpu, i32Model, "model"),
        RuntimeSupportError,
      );
    } finally {
      gpu.destroy();
    }
  },
);

test.skipIf(!GPU_AVAILABLE)(
  "a graph outside the contract (non-empty attrs / a dtype the op does not accept) fails at Session build (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    try {
      // The capability match (op name + dtype + attrs) catches both as the first check
      const withAttrs = chainGraph();
      withAttrs.nodes[2].attrs = { approximate: "tanh" };
      const withAttrsModel = await openChainModel(withAttrs);
      await assertRejects(
        () => createSessionFromContainer(gpu, withAttrsModel, "model"),
        RuntimeSupportError,
        "unimplemented attrs (1): nodes[2] (relu): approximate",
      );

      // Transferring i32 is allowed (ADR 0009), but matmul is f32-only, so the op contract fails it
      const intInput = chainGraph();
      intInput.inputs = [{ name: "x", dtype: "i32", shape: ["T", 4] }];
      const intInputModel = await openChainModel(intInput);
      await assertRejects(
        () => createSessionFromContainer(gpu, intInputModel, "model"),
        RuntimeSupportError,
        "unsupported semantic dtype (1): value 'x': i32",
      );
    } finally {
      gpu.destroy();
    }
  },
);

/**
 * The mask path of an observed graph (the DeBERTa front), shrunk as is:
 * mask outer product mul(i32) → cast to bool → bitwise_not → cast to a weight → f32 weighting.
 * One bool output also exercises the non-f32 readback path.
 */
const maskGraph = (): DeclarationJson => ({
  format: "karume-ir",
  version: 2,
  requires: { ops: ["mul", "cast", "bitwise_not"] },
  symbols: ["T"],
  inputs: [
    { name: "mcol", dtype: "i32", shape: ["T", 1] },
    { name: "mrow", dtype: "i32", shape: [1, "T"] },
    { name: "scores", dtype: "f32", shape: ["T", "T"] },
  ],
  outputs: ["masked", "drop"],
  initializers: {},
  values: {
    pair: { dtype: "i32", shape: ["T", "T"] },
    keep: { dtype: "bool", shape: ["T", "T"] },
    drop: { dtype: "bool", shape: ["T", "T"] },
    weight: { dtype: "f32", shape: ["T", "T"] },
    masked: { dtype: "f32", shape: ["T", "T"] },
  },
  nodes: [
    { op: "mul", ins: ["mcol", "mrow"], outs: ["pair"], attrs: {} },
    { op: "cast", ins: ["pair"], outs: ["keep"], attrs: { to: "bool" } },
    { op: "bitwise_not", ins: ["keep"], outs: ["drop"], attrs: {} },
    { op: "cast", ins: ["keep"], outs: ["weight"], attrs: { to: "f32" } },
    { op: "mul", ins: ["scores", "weight"], outs: ["masked"], attrs: {} },
  ],
});

test.skipIf(!GPU_AVAILABLE)(
  "a chain of i32 input → bool intermediate → non-f32 output matches the CPU reference exactly (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    const session = await createSessionFromContainer(
      gpu,
      await openGraphModel(maskGraph()),
      "model",
    );
    try {
      const mcol = fill([4, 1], (i) => (i === 2 ? 0 : 1), "i32");
      const mrow = fill([1, 4], (i) => (i === 3 ? 0 : 1), "i32");
      const scores = fill([4, 4], (i) => (i % 7) - 3);
      const outputs = await session.run({ mcol, mrow, scores });

      const pair = applyReferenceOp("mul", [mcol, mrow]);
      const keep = applyReferenceOp("cast", [pair], { to: "bool" });
      const drop = applyReferenceOp("bitwise_not", [keep]);
      const weight = applyReferenceOp("cast", [keep], { to: "f32" });
      const masked = applyReferenceOp("mul", [scores, weight]);

      // A bool output comes back as a Uint32Array (u32 0/1) — a shape an f32-only readback cannot
      // pass
      assertEquals(outputs["drop"].dtype, "bool");
      assertEquals(outputs["drop"].data.constructor.name, "Uint32Array");
      assertEquals(compareTensors(outputs["drop"], drop).pass, true);
      assertEquals(outputs["masked"].dtype, "f32");
      assertEquals(compareTensors(outputs["masked"], masked).pass, true);
    } finally {
      await session.dispose();
      gpu.destroy();
    }
  },
);

// MUST: every element type is 4 bytes, so a TypedArray mix-up passes straight through writeBuffer
// and reinterprets the bit patterns (no exception). Matching against the declared dtype is the only
// safeguard.
test.skipIf(!GPU_AVAILABLE)(
  "an input tensor that disagrees with the declared dtype fails before upload (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    const session = await createSessionFromContainer(
      gpu,
      await openGraphModel(maskGraph()),
      "model",
    );
    try {
      const mrow = fill([1, 4], () => 1, "i32");
      const scores = fill([4, 4], () => 1);
      // An f32 Tensor for an input declared i32 (both the discriminant and the array type differ)
      await assertRejects(
        () => session.run({ mcol: fill([4, 1], () => 1), mrow, scores }),
        ExecutionError,
        "declared dtype 'i32'",
      );
      // Only the discriminant matches and the array type is spoofed (TS cannot produce this, so it
      // is built explicitly)
      const spoofed = {
        dtype: "i32",
        shape: [4, 1],
        data: Float32Array.from([1, 1, 1, 1]),
      } as unknown as Tensor;
      await assertRejects(
        () => session.run({ mcol: spoofed, mrow, scores }),
        ExecutionError,
        "Float32Array",
      );
    } finally {
      await session.dispose();
      gpu.destroy();
    }
  },
);

test.skipIf(!GPU_AVAILABLE)(
  "inconsistent run inputs and a run after dispose fail loudly (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    const session = await createSessionFromContainer(gpu, await openChainModel(), "model");
    try {
      // A rank / numeric dimension mismatch against the declared [T,4]
      await assertRejects(() => session.run({ x: fill([5], () => 1) }), ExecutionError);
      await assertRejects(() => session.run({ x: fill([5, 3], () => 1) }), ExecutionError);
      // shape and element count disagree (the shape matches the declaration but data is short)
      await assertRejects(
        () =>
          session.run({ x: { dtype: "f32", shape: [5, 4], data: Float32Array.from([1, 2, 3]) } }),
        ExecutionError,
      );
      // An explicit binding conflicts with the binding from the input shape
      await assertRejects(
        () => session.run({ x: fill([5, 4], () => 1) }, { T: 4 }),
        ExecutionError,
      );
      // A missing input
      await assertRejects(() => session.run({}), ExecutionError);
    } finally {
      await session.dispose();
    }
    await assertRejects(() => session.run({ x: fill([5, 4], () => 1) }), ExecutionError);
    // dispose is idempotent
    await session.dispose();
    gpu.destroy();
  },
);

test.skipIf(!GPU_AVAILABLE)(
  "a mismatch between the declared shape and the computed shape fails without a single dispatch (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    // The IR does not compute shapes, so a declaration error is first found at execution planning
    const wrong = chainGraph();
    wrong.values.h = { dtype: "f32", shape: ["T", 4] };
    wrong.values.g = { dtype: "f32", shape: ["T", 4] };
    wrong.values.y = { dtype: "f32", shape: ["T", 4] };
    const session = await createSessionFromContainer(gpu, await openChainModel(wrong), "model");
    try {
      await assertRejects(() => session.run({ x: fill([3, 4], () => 1) }), ExecutionError);
      assertEquals(session.diagnostics().submit.dispatchCount, 0);
    } finally {
      await session.dispose();
      gpu.destroy();
    }
  },
);

test.skipIf(!GPU_AVAILABLE)(
  "a run that fails in the derivation phase queues no GPU command at all (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    // The second node (matmul) throws for exceeding a limit. Dispatches are issued only in the
    // **recipe execution phase**, and the derivation phase fails before that, so nothing is queued
    // for the first node (relu) either (the two-phase split of src/runtime/recipe.ts — a run known
    // to fail does no GPU work).
    // MUST: the over-limit threshold is set by matmul's **output tile edge**
    // (`ceil(n / tileN) > limit`). Derive the tile edge **from the default geometry**, not a
    // constant (a drift where changing the edge stopped the throw and only assertRejects silently
    // failed happened twice, 16 → 64 → 128).
    const tileN = gemmTileN(defaultGemmGeometry());
    const n = gpu.limits.maxComputeWorkgroupsPerDimension * tileN + tileN;
    const graph: DeclarationJson = {
      format: "karume-ir",
      version: 2,
      requires: { ops: ["relu", "matmul"] },
      symbols: [],
      inputs: [
        { name: "x0", dtype: "f32", shape: [1, 1] },
        { name: "x1", dtype: "f32", shape: [1, n] },
      ],
      outputs: ["y"],
      initializers: {},
      values: {
        t: { dtype: "f32", shape: [1, 1] },
        y: { dtype: "f32", shape: [1, n] },
      },
      nodes: [
        { op: "relu", ins: ["x0"], outs: ["t"], attrs: {} },
        { op: "matmul", ins: ["t", "x1"], outs: ["y"], attrs: {} },
      ],
    };
    const session = await createSessionFromContainer(gpu, await openGraphModel(graph), "model");
    try {
      await assertRejects(
        () => session.run({ x0: fill([1, 1], () => 1), x1: fill([1, n], () => 1) }),
        DispatchLimitError,
      );

      const stats = session.diagnostics().submit;
      assertEquals(stats.dispatchCount, 0, "nothing was queued for the preceding node either");
      // Flushing the leftovers in cleanup (arena.destroy → flush) would submit after the lock is
      // released with no errorScope, and the validation error would be attributed to another
      // concurrent Session's section.
      assertEquals(stats.submitCount, 0, "the leftovers of a failed run are never submitted");
      assertEquals(stats.discardedDispatches, 0, "nothing was queued, so nothing is discarded");
    } finally {
      await session.dispose();
      gpu.destroy();
    }
  },
);

// The exhaustive value sweep (5 knobs × 6 values) runs GPU-independently in
// tests/runtime_execution_knobs_test.ts. Only one case stays here, "the validation is wired into
// the real build path" (proof that `createSession` reaches `assertExecutionKnobs`, not a check of
// the accepted set itself).
test.skipIf(!GPU_AVAILABLE)(
  "a convertible object for an execution setting is not accepted as a string by the real build either",
  async () => {
    const gpu = await acquireGpu();
    try {
      const model = await openChainModel();
      let conversions = 0;
      const object = {
        [Symbol.toPrimitive](): string {
          conversions++;
          return "parallel-fused";
        },
        toJSON(): string {
          conversions++;
          throw new Error("the diagnostic must not call the caller's conversion");
        },
      };
      const options: SessionOptions = {};
      Object.defineProperty(options, "stateAttentionReduce", { value: object });
      await assertRejects(
        async () => {
          const session = await createSessionFromContainer(gpu, model, "model", options);
          // On a regression too, dispose the wrongly built Session in the test before failing.
          await session.dispose();
        },
        ExecutionError,
        "stateAttentionReduce",
      );
      assertEquals(conversions, 0);
    } finally {
      gpu.destroy();
    }
  },
);
