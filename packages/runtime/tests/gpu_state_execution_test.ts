// **Execution integration** checks of the states-form attention + `state_append` (phase D-3 —
// ADR 0066 / 0067).
//
// The kernel-level grid lives in gpu_state_attention_test.ts (direct dispatch). This file looks at
// "does one step of Session.run with a GenerationContext keep matching an oracle that **chains**
// the CPU reference" — that is, the execution wiring itself (recipe / bindings / carrying the
// logical length / advancing):
//
//   ① multi-step parity (prefill → decode ×n, pastLength advancing, the cache from the 2nd step on)
//   ② crossing the sliding ring / ③ GQA (r=2) and the census / ④ whether rewind is allowed /
//   ⑤ the pre-run check of full capacity / ⑥ the 3 rejections of queryLength / ⑦ the context not
//   being part of the plan key / ⑧ rejection of stateless execution
//   ⑨ a failure of a run that submitted state poisons the context
//
// Phase D-4 (separating the baking units, ADR 0066 decision 5) added these 4:
//
//   ⑩ switching A/B (zero re-derivations + zero backing rebuilds + one re-bake per context + zero
//      mix-ups)
//   ⑪ recovery when a backing is retired and rebuilt (re-baked by its generation identifier) —
//      became 3 tests with ADR 0095 (multiple retained, with a budget): budget 0 rebuilds and
//      re-bakes at every switch / a round trip between shapes retained within the budget adds no
//      re-bakes / a shape that was retired (over budget) is dropped with its bundle and re-baked
//      once by the run that comes back to it
//   ⑫ fault injection (mixing up the context-side bundle breaks parity = proof that ⑩ does not
//      miss)
//   ⑬ the backed transition of ① (from the 3rd run on it runs on the slot backing, and the values
//      do not change at the transition)
//
// MUST: the oracle **carries the slots around on the host** (the result of `referenceStateAppend`
// feeds the next step's `referenceStateAttention`). Comparing a single step would not check the
// wiring itself: "the next step reads the rows append wrote as its past".

import { test } from "vite-plus/test";
import { assert, assertEquals, assertRejects, assertThrows } from "@std/assert";
import { compareTensors, formatAllclose, type Tolerance } from "../src/reference/allclose.ts";
import { referenceStateAppend, referenceStateAttention } from "../src/reference/state-attention.ts";
import {
  statePvKey,
  statePvParallelKey,
  statePvTiledKey,
  stateQkFoldedKey,
  stateQkKey,
  stateQkParallelKey,
  stateQkTiledKey,
} from "../src/kernels/state-attention.ts";
import { acquireGpu, type GpuContext, RUNTIME_INTERNAL } from "../src/gpu/device.ts";
import {
  createSessionFromContainer,
  ROW_BLOCK_SPLIT,
  type Session,
  type SessionOptions,
  type Tensor,
} from "../src/runtime/executor.ts";
import type { GenerationContext } from "../src/runtime/generation-context.ts";
import type { PlanBackingStats } from "../src/runtime/session-types.ts";
import type { BakedGroups } from "../src/runtime/recipe.ts";
import { OpContractError } from "../src/ops.ts";
import { ExecutionError } from "../src/runtime/plan.ts";
import { type DeclarationJson, GRAPH_NAME, openGraphModel } from "./helpers/model-fixture.ts";
import { halfScale, seeded } from "./helpers/state-dispatch.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";
import { injectOutOfMemory } from "./helpers/oom-injection.ts";
import { countDispatches, requireCensus, SEAT_SIGNATURES } from "./helpers/pipeline-census.ts";

/**
 * The comparison tolerance. Same grounds as `STATE_TOLERANCE` in gpu_state_attention_test.ts
 * (the reference accumulates in f64, the GPU sequentially in f32, and the difference is
 * proportional to `D` and `live`).
 *
 * MUST: `rtol = 0`. The states form legitimately contains **elements whose correct value is exactly
 * 0** (empty rows, contributions outside the predicate), so a relative term does nothing near 0 and
 * only serves to hide errors in large elements.
 */
const STATE_TOLERANCE: Tolerance = { atol: 5e-6, rtol: 0 };

/** Deterministic input sequences (no random numbers — a failure would not reproduce). */
const QUERY = (i: number): number => (((i * 7) % 23) - 11) * 0.17;
const KEY = (i: number): number => (((i * 11) % 19) - 9) * 0.23;
const VALUE = (i: number): number => (((i * 5) % 17) - 8) * 0.31;

/**
 * The shape of one model (`B = 1` fixed — the batch axis is out of scope per ADR 0066 decision
 * 8).
 */
type StateModel = {
  readonly heads: number;
  readonly kvHeads: number;
  readonly depth: number;
  /**
   * The slot's row capacity. With the symbol `"C"` (a states-only symbol) only
   * `createGenerationContext(bindings)` decides it. The symbol `"M"` **also appears in an input
   * shape**, which gives two binding points (ADR 0066 addendum 7 — at run time it is checked that
   * the two binding points agree).
   */
  readonly capacity: number | "C" | "M";
  /** `W` (absent = the whole context). */
  readonly window?: number;
};

/**
 * The graph of one states-form layer (one attention + two `state_append`s — in the order "all
 * readers → writers", following the issue convention of ADR 0067 decision 5b).
 *
 * `M` (the physical chunk rows) is a **symbol**. Prefill (M=4) and decode (M=1) are different plans
 * with different declared shapes, and one Session / one context spanning both is exactly the
 * execution shape of ADR 0066 decision 4.
 */
const stateGraph = (model: StateModel): DeclarationJson => {
  const { heads, kvHeads, depth, capacity, window } = model;
  const windowAttrs = window === undefined ? {} : { window };
  const append = (name: string, slot: string) => ({
    op: "state_append",
    ins: [name],
    outs: [] as string[],
    attrs: { ...windowAttrs },
    states: { slot },
  });
  return {
    format: "karume-ir",
    version: 2,
    requires: { ops: ["attention", "state_append"] },
    symbols: capacity === "C" ? ["M", "C"] : ["M"],
    inputs: [
      { name: "q", dtype: "f32", shape: [1, heads, "M", depth] },
      { name: "k", dtype: "f32", shape: [1, kvHeads, "M", depth] },
      { name: "v", dtype: "f32", shape: [1, kvHeads, "M", depth] },
    ],
    outputs: ["o"],
    initializers: {},
    values: { o: { dtype: "f32", shape: [1, heads, "M", depth] } },
    states: {
      kslot: { dtype: "f32", shape: [1, kvHeads, capacity, depth] },
      vslot: { dtype: "f32", shape: [1, kvHeads, capacity, depth] },
    },
    nodes: [
      {
        op: "attention",
        ins: ["q", "k", "v"],
        outs: ["o"],
        attrs: { scale: halfScale(depth), ...windowAttrs },
        states: { k: "kslot", v: "vslot" },
      },
      append("k", "kslot"),
      append("v", "vslot"),
    ],
  };
};

const stateSession = async (
  gpu: GpuContext,
  model: StateModel,
  options: SessionOptions = {},
): Promise<Session> =>
  await createSessionFromContainer(
    gpu,
    await openGraphModel(stateGraph(model)),
    GRAPH_NAME,
    options,
  );

/** The inputs of one step (`M` rows — only the first `queryLength` rows are valid). */
type StepInputs = {
  readonly q: Float32Array<ArrayBuffer>;
  readonly k: Float32Array<ArrayBuffer>;
  readonly v: Float32Array<ArrayBuffer>;
};

/**
 * Per-step inputs. Changing `salt` per step makes an implementation that "rereads the previous
 * step's inputs" fail on the values (with identical inputs, a chain error could cancel itself
 * out).
 */
const stepInputs = (model: StateModel, chunkRows: number, salt: number): StepInputs => ({
  q: seeded(model.heads * chunkRows * model.depth, (i) => QUERY(i + salt)),
  k: seeded(model.kvHeads * chunkRows * model.depth, (i) => KEY(i + salt)),
  v: seeded(model.kvHeads * chunkRows * model.depth, (i) => VALUE(i + salt)),
});

const tensor = (shape: readonly number[], data: Float32Array<ArrayBuffer>): Tensor => ({
  dtype: "f32",
  shape,
  data,
});

/** The host-side slot contents and logical length (the mirror of the GPU-side context). */
type Oracle = {
  past: number;
  slotK: Float32Array<ArrayBuffer>;
  slotV: Float32Array<ArrayBuffer>;
};

/** A freshly created context has **zero-initialized slots** (WebGPU's buffer contract). */
const newOracle = (model: StateModel, capacity: number): Oracle => ({
  past: 0,
  slotK: new Float32Array(model.kvHeads * capacity * model.depth),
  slotV: new Float32Array(model.kvHeads * capacity * model.depth),
});

const asF32 = (data: Float32Array | Int32Array | Uint32Array): Float32Array<ArrayBuffer> => {
  if (!(data instanceof Float32Array)) throw new Error("the reference did not return f32");
  return data as Float32Array<ArrayBuffer>;
};

/**
 * Advances the oracle by one step (**read, then write** — exactly the node array order). Returns
 * this step's expected output `[1,H,M,D]`.
 */
const advanceOracle = (
  model: StateModel,
  capacity: number,
  state: Oracle,
  inputs: StepInputs,
  chunkRows: number,
  query: number,
): Float32Array<ArrayBuffer> => {
  const common = {
    chunkRows,
    depth: model.depth,
    capacity,
    window: model.window ?? 0,
    past: state.past,
    query,
  };
  const expected = referenceStateAttention({
    ...common,
    batch: 1,
    heads: model.heads,
    kvHeads: model.kvHeads,
    q: inputs.q,
    insK: inputs.k,
    insV: inputs.v,
    slotK: state.slotK,
    slotV: state.slotV,
    scale: halfScale(model.depth),
  });
  const appendBase = { ...common, kvPlanes: model.kvHeads };
  state.slotK = asF32(referenceStateAppend({ ...appendBase, x: inputs.k, slot: state.slotK }).data);
  state.slotV = asF32(referenceStateAppend({ ...appendBase, x: inputs.v, slot: state.slotV }).data);
  state.past += query;
  return asF32(expected.data);
};

/** One GPU-side step (the 3rd argument of `run` is the generation surface). */
const runStep = async (
  session: Session,
  context: GenerationContext,
  model: StateModel,
  inputs: StepInputs,
  chunkRows: number,
  query: number,
  bindings: Record<string, number> = {},
): Promise<Float32Array<ArrayBuffer>> => {
  const outputs = await session.run(
    {
      q: tensor([1, model.heads, chunkRows, model.depth], inputs.q),
      k: tensor([1, model.kvHeads, chunkRows, model.depth], inputs.k),
      v: tensor([1, model.kvHeads, chunkRows, model.depth], inputs.v),
    },
    bindings,
    { context, queryLength: query },
  );
  const out = outputs["o"];
  assertEquals(out.dtype, "f32");
  return out.data as Float32Array<ArrayBuffer>;
};

/** Runs one step and compares it with the oracle (one check point that combines both sides). */
const assertStep = async (
  session: Session,
  context: GenerationContext,
  model: StateModel,
  capacity: number,
  state: Oracle,
  chunkRows: number,
  query: number,
  salt: number,
  label: string,
  bindings: Record<string, number> = {},
): Promise<void> => {
  const inputs = stepInputs(model, chunkRows, salt);
  const actual = await runStep(session, context, model, inputs, chunkRows, query, bindings);
  const expected = advanceOracle(model, capacity, state, inputs, chunkRows, query);
  // MUST: check every time that the expected values are not trivial (if both sides are all 0 the
  // comparison is a tautology — the states form legitimately produces exact 0 for empty rows, so
  // without this check an implementation that "computes nothing" would pass green).
  assert(
    expected.some((value) => Math.abs(value) > 1e-3),
    `${label}: the expected output is trivial (all ~0)`,
  );
  const report = compareTensors(
    { dtype: "f32", data: actual },
    { dtype: "f32", data: expected },
    STATE_TOLERANCE,
  );
  assertEquals(report.pass, true, `${label}: ${formatAllclose(report)}`);
  assertEquals(context.pastLength, state.past, `${label}: pastLength drifted from the oracle`);
};

const FULL: StateModel = { heads: 4, kvHeads: 4, depth: 4, capacity: 8 };

test.skipIf(!GPU_AVAILABLE)(
  "the prefill → decode chain matches the CPU reference, and from the 2nd step on the recipe hits the cache",
  async () => {
    const gpu = await acquireGpu();
    const session = await stateSession(gpu, FULL);
    const context = await session.createGenerationContext({ chunkLength: 4 });
    const state = newOracle(FULL, 8);
    try {
      // Prefill (M=4 / Q=4). The first run derives the plan.
      await assertStep(session, context, FULL, 8, state, 4, 4, 0, "prefill");
      assertEquals(session.diagnostics().lastRunPrepared?.hit, false, "prefill is a deriving run");
      assertEquals(context.pastLength, 4);

      // Decode (M=1 / Q=1). M changes, so the first one is a different plan = a miss.
      await assertStep(session, context, FULL, 8, state, 1, 1, 11, "decode 1");
      assertEquals(
        session.diagnostics().lastRunPrepared?.hit,
        false,
        "a different M is a different key",
      );
      // The 2nd decode hits **the same key** ("zero recipe re-derivations" of ADR 0066
      // acceptance criterion ③).
      await assertStep(session, context, FULL, 8, state, 1, 1, 23, "decode 2");
      assertEquals(
        session.diagnostics().lastRunPrepared?.hit,
        true,
        "the 2nd run with the same context and the same M did not hit the cache",
      );
      await assertStep(session, context, FULL, 8, state, 1, 1, 37, "decode 3");
      assertEquals(session.diagnostics().lastRunPrepared?.hit, true);
      assertEquals(
        context.pastLength,
        7,
        "the logical length goes 4 → 5 → 6 → 7, advancing only on successful runs",
      );

      // Phase D-4: generation runs also go on the slot backing (the separated baking of ADR 0066
      // decision 5). What goes on it is **only the dispatches that bind Session-owned buffers**;
      // the positions that bind state are baked by the context side. Decode 2 / 3 ran on that
      // backed path, and **the values not changing at the arena → backed transition** is covered
      // by the per-step comparisons above themselves.
      const diagnostics = session.diagnostics();
      assertEquals(
        diagnostics.planBacking.buildCount,
        1,
        "consecutive decodes with the same key rebuild the backing (switch thrashing)",
      );
      assert(
        diagnostics.planBacking.residentBytes > 0,
        "the generation run is not on the backed path (the slot is not resident)",
      );
      assertEquals(
        diagnostics.stateBacking.rebindCount,
        1,
        "consecutive runs with the same (context, backing) re-bake the context-side bind groups",
      );
    } finally {
      await context.dispose();
      await session.dispose();
      gpu.destroy();
    }
  },
);

test.skipIf(!GPU_AVAILABLE)(
  "pad rows (queryLength < M) are not written to the KV and do not appear in the next step's past (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    const session = await stateSession(gpu, FULL);
    const context = await session.createGenerationContext({ chunkLength: 4 });
    const state = newOracle(FULL, 8);
    try {
      // A chunk of M=4 with only Q=2 valid. An implementation that writes the 2 pad rows shifts the
      // next step's past by 2 rows (the oracle writes only Q rows), so the comparisons below fail.
      await assertStep(session, context, FULL, 8, state, 4, 2, 5, "prefill with pad");
      assertEquals(context.pastLength, 2);
      await assertStep(session, context, FULL, 8, state, 4, 3, 41, "2nd with pad");
      assertEquals(context.pastLength, 5);
      await assertStep(session, context, FULL, 8, state, 1, 1, 59, "decode");
      assertEquals(context.pastLength, 6);
    } finally {
      await context.dispose();
      await session.dispose();
      gpu.destroy();
    }
  },
);

const SLIDING: StateModel = { heads: 2, kvHeads: 2, depth: 4, capacity: 4, window: 4 };

test.skipIf(!GPU_AVAILABLE)(
  "sliding matches the CPU reference across the ring's wrap (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    const session = await stateSession(gpu, SLIDING);
    const context = await session.createGenerationContext({ chunkLength: 3 });
    const state = newOracle(SLIDING, 4);
    try {
      // Prefill 3 rows → decode ×5. The capacity-4 ring wraps at past=4 and is overwritten every
      // step after that.
      await assertStep(session, context, SLIDING, 4, state, 3, 3, 3, "prefill");
      for (let step = 0; step < 5; step += 1) {
        await assertStep(
          session,
          context,
          SLIDING,
          4,
          state,
          1,
          1,
          71 + step * 13,
          `decode ${step} (past=${state.past})`,
        );
      }
      assertEquals(
        context.pastLength,
        8,
        "the logical length keeps advancing even with a capacity-4 ring",
      );
    } finally {
      await context.dispose();
      await session.dispose();
      gpu.destroy();
    }
  },
);

// ---------------------------------------------------------------------------
// The verify shape of speculative decoding (sliding slack + deferred commit)
// ---------------------------------------------------------------------------

/**
 * The slack rows (`capacity − window` of a sliding slot). Writing `Q` rows ahead of the logical
 * length and accepting only `a` of them works when `C ≥ W + Q` (the logical position `P+i−C` that a
 * rejected row `P+i` overwrites is always below the lower edge of any reader's window after the
 * commit — the lowest edge is the borrower's = the readonly reader's `P+a−W`). So a slack of 8
 * supports `Q` up to 8.
 */
const VERIFY_SLACK = 8;
const VERIFY_WINDOW = 8;
const VERIFY_QUERY = VERIFY_SLACK;
/**
 * The accepted rows (the other 5 are rejected = written to the physical ring but not part of the
 * logical length).
 */
const VERIFY_COMMIT = 3;
/**
 * The past built up before the verify shape (`P > W` — start with the window's lower-bound
 * predicate in effect).
 */
const VERIFY_PREFILL = 12;

const verifyModel = (capacity: number): StateModel => ({
  heads: 2,
  kvHeads: 2,
  depth: 4,
  capacity,
  window: VERIFY_WINDOW,
});

/** Cuts one row of `[1, planes, M, D]` out as `[1, planes, 1, D]`. */
const takeRow = (
  data: Float32Array<ArrayBuffer>,
  planes: number,
  chunkRows: number,
  depth: number,
  row: number,
): Float32Array<ArrayBuffer> => {
  const out = new Float32Array(planes * depth);
  for (let plane = 0; plane < planes; plane += 1) {
    for (let d = 0; d < depth; d += 1) {
      out[plane * depth + d] = data[(plane * chunkRows + row) * depth + d];
    }
  }
  return out;
};

const bitsOfF32 = (data: Float32Array<ArrayBuffer>): Uint32Array =>
  new Uint32Array(data.buffer, data.byteOffset, data.length);

/**
 * Returns the next step's output for "a context that wrote `Q` rows deferred and committed only
 * `a`".
 *
 * The reference side (`speculate = false`) feeds the same tokens **one row at a time in normal
 * runs** (= plain generation without speculation). If the two are bit-identical, the writes of the
 * 5 rejected rows did not corrupt a single word of the live past KV.
 */
const runVerifyChain = async (
  gpu: GpuContext,
  capacity: number,
  speculate: boolean,
): Promise<Float32Array<ArrayBuffer>> => {
  const model = verifyModel(capacity);
  const session = await stateSession(gpu, model);
  const context = await session.createGenerationContext({
    chunkLength: 16,
    chunkBuckets: [VERIFY_QUERY],
  });
  try {
    // ① The shared prefill (P = 12 > W = 8)
    const prefill = stepInputs(model, 16, 5);
    await runStep(session, context, model, prefill, 16, VERIFY_PREFILL);

    // ② Inputs for the 9 draft rows (both sides use **the same values** — the reference side feeds
    // the first 3 rows one at a time)
    const draft = stepInputs(model, VERIFY_QUERY, 41);
    if (speculate) {
      const outputs = await session.run(
        {
          q: tensor([1, model.heads, VERIFY_QUERY, model.depth], draft.q),
          k: tensor([1, model.kvHeads, VERIFY_QUERY, model.depth], draft.k),
          v: tensor([1, model.kvHeads, VERIFY_QUERY, model.depth], draft.v),
        },
        {},
        { context, queryLength: VERIFY_QUERY, commit: "deferred" },
      );
      assertEquals(outputs["o"].shape, [1, model.heads, VERIFY_QUERY, model.depth]);
      assertEquals(
        context.pendingCommit,
        { pastLength: VERIFY_PREFILL, queryLength: VERIFY_QUERY },
        "the deferred run did not create a pending commit",
      );
      context.commit(VERIFY_COMMIT);
    } else {
      for (let row = 0; row < VERIFY_COMMIT; row += 1) {
        await runStep(
          session,
          context,
          model,
          {
            q: takeRow(draft.q, model.heads, VERIFY_QUERY, model.depth, row),
            k: takeRow(draft.k, model.kvHeads, VERIFY_QUERY, model.depth, row),
            v: takeRow(draft.v, model.kvHeads, VERIFY_QUERY, model.depth, row),
          },
          1,
          1,
        );
      }
    }
    assertEquals(
      context.pastLength,
      VERIFY_PREFILL + VERIFY_COMMIT,
      "the logical length does not match the accepted rows",
    );

    // ③ One step after the acceptance (does the past KV read here contain rejected rows)
    const next = stepInputs(model, 1, 97);
    return await runStep(session, context, model, next, 1, 1);
  } finally {
    await context.dispose();
    await session.dispose();
  }
};

test.skipIf(!GPU_AVAILABLE)(
  "deferred commit + sliding slack: rejected rows do not move a single word of the next step's output (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    try {
      const slack = VERIFY_WINDOW + VERIFY_SLACK;
      const speculative = await runVerifyChain(gpu, slack, true);
      const sequential = await runVerifyChain(gpu, slack, false);
      assertEquals(
        [...bitsOfF32(speculative)],
        [...bitsOfF32(sequential)],
        "speculation with slack + commit is not bit-identical to row-by-row generation",
      );

      // With the slack back at 0 (capacity = window), rejected rows overwrite logical positions
      // inside the live window (silent corruption with no exception and no NaN). That shape fails
      // in the synchronous section of the run's issue — a deferred run's queryLength is limited to
      // the "slack" (the check in `GenerationContext.slidingSlack`). The overwritten values
      // themselves are not reachable from the public surface, so what is checked here is "the
      // guard works" = the issue is rejected.
      const noSlack = await assertRejects(
        () => runVerifyChain(gpu, VERIFY_WINDOW, true),
        ExecutionError,
      );
      assert(noSlack.message.includes("slack 0"), noSlack.message);
    } finally {
      gpu.destroy();
    }
  },
);

/**
 * The borrower (the drafter's role): a graph in which one readonly attention only reads external
 * slots with **the same names** as the lender's (ADR 0096 step 2 §2.1). It has no writer
 * (`state_append`), so advancing and committing the logical length are both the lender's job.
 */
const verifyBorrowerGraph = (capacity: number): DeclarationJson => {
  const model = verifyModel(capacity);
  return {
    format: "karume-ir",
    version: 2,
    requires: { ops: ["attention"] },
    symbols: [],
    inputs: [{ name: "q", dtype: "f32", shape: [1, model.heads, 1, model.depth] }],
    outputs: ["o"],
    initializers: {},
    values: { o: { dtype: "f32", shape: [1, model.heads, 1, model.depth] } },
    states: {
      kslot: { dtype: "f32", shape: [1, model.kvHeads, capacity, model.depth], external: true },
      vslot: { dtype: "f32", shape: [1, model.kvHeads, capacity, model.depth], external: true },
    },
    nodes: [
      {
        op: "attention",
        ins: ["q"],
        outs: ["o"],
        attrs: { scale: halfScale(model.depth), window: VERIFY_WINDOW, readonly: true },
        states: { k: "kslot", v: "vslot" },
      },
    ],
  };
};

/**
 * Returns the output of one read by the **borrower** (a readonly reader) of "a lender that wrote
 * `Q = slack` rows deferred and committed only `m`". The reference side (`speculate = false`) feeds
 * the first `m` rows of the same tokens one row at a time in normal runs (= a run that never wrote
 * a single word of the rejected rows).
 *
 * The only difference from `runVerifyChain` is the kind of reader. A states-form reader's window
 * has its lower edge one position higher because of this step's ins (`P+m−(W−1)`), while the
 * borrower's readonly reader goes down to `P+m−W` (the readonly section of
 * `src/kernels/state-attention.ts`, `column_base = P − min(P, W)`). That one position is why the
 * check was tightened from `Q ≤ slack + 1` to `Q ≤ slack`, and only this shape reaches it.
 */
const runBorrowedVerifyChain = async (
  gpu: GpuContext,
  speculate: boolean,
  commitRows: number,
): Promise<Float32Array<ArrayBuffer>> => {
  const capacity = VERIFY_WINDOW + VERIFY_SLACK;
  const model = verifyModel(capacity);
  const lender = await stateSession(gpu, model);
  const borrower = await createSessionFromContainer(
    gpu,
    await openGraphModel(verifyBorrowerGraph(capacity)),
    GRAPH_NAME,
  );
  const lenderContext = await lender.createGenerationContext({
    chunkLength: 16,
    chunkBuckets: [VERIFY_QUERY],
  });
  const borrowed = await borrower.createGenerationContext({
    chunkLength: 1,
    borrow: lenderContext,
  });
  try {
    // ① The shared prefill (P = 12 > W = 8)
    await runStep(lender, lenderContext, model, stepInputs(model, 16, 5), 16, VERIFY_PREFILL);

    // ② Inputs for the 8 draft rows (both sides use **the same values** — the reference side feeds
    // the first m rows one at a time)
    const draft = stepInputs(model, VERIFY_QUERY, 41);
    if (speculate) {
      await lender.run(
        {
          q: tensor([1, model.heads, VERIFY_QUERY, model.depth], draft.q),
          k: tensor([1, model.kvHeads, VERIFY_QUERY, model.depth], draft.k),
          v: tensor([1, model.kvHeads, VERIFY_QUERY, model.depth], draft.v),
        },
        {},
        { context: lenderContext, queryLength: VERIFY_QUERY, commit: "deferred" },
      );
      lenderContext.commit(commitRows);
    } else {
      for (let row = 0; row < commitRows; row += 1) {
        await runStep(
          lender,
          lenderContext,
          model,
          {
            q: takeRow(draft.q, model.heads, VERIFY_QUERY, model.depth, row),
            k: takeRow(draft.k, model.kvHeads, VERIFY_QUERY, model.depth, row),
            v: takeRow(draft.v, model.kvHeads, VERIFY_QUERY, model.depth, row),
          },
          1,
          1,
        );
      }
    }
    assertEquals(
      lenderContext.pastLength,
      VERIFY_PREFILL + commitRows,
      "the logical length does not match the accepted rows",
    );

    // ③ One borrower run (reads `[P−W, P)` — does it contain rejected rows)
    const outputs = await borrower.run(
      {
        q: tensor(
          [1, model.heads, 1, model.depth],
          seeded(model.heads * model.depth, (i) => QUERY(i + 97)),
        ),
      },
      {},
      { context: borrowed, queryLength: 1 },
    );
    assertEquals(
      borrowed.pastLength,
      VERIFY_PREFILL + commitRows,
      "the P the borrower copied drifted from the lender's",
    );
    return outputs["o"].data as Float32Array<ArrayBuffer>;
  } finally {
    await borrowed.dispose();
    await lenderContext.dispose();
    await borrower.dispose();
    await lender.dispose();
  }
};

test.skipIf(!GPU_AVAILABLE)(
  "deferred commit + the borrower's readonly window: rejected rows do not move a single word of what the drafter reads (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    try {
      // `m = 0` (everything rejected = the exception recovery's `commit(0)`) is the worst case.
      // With C = 16 / W = 8 / P = 12 the physical rows overwritten by rejected rows j = 0..7 are
      // 12,13,14,15,0,1,2,3, and the positions the borrower reads, `[4,12)`, are physical rows
      // 4..11: **exactly one row is free** (loosening `Q` to 9 would make the 9th, logical position
      // 20, overwrite physical row 4 = the lower edge of the borrower's window). This shape being
      // green is the safety of the tightened check `Q ≤ slidingSlack` itself. `m = VERIFY_COMMIT`
      // is the control in which accepted rows are mixed in.
      for (const commitRows of [0, VERIFY_COMMIT]) {
        const speculative = await runBorrowedVerifyChain(gpu, true, commitRows);
        const sequential = await runBorrowedVerifyChain(gpu, false, commitRows);
        // MUST: the expected values are not trivial (if both sides are all 0 the comparison is a
        // tautology).
        assert(
          speculative.some((value) => Math.abs(value) > 1e-3),
          `commit(${commitRows}): the borrower's output is trivial (all ~0)`,
        );
        assertEquals(
          [...bitsOfF32(speculative)],
          [...bitsOfF32(sequential)],
          `commit(${commitRows}): rejected rows corrupted what the borrower reads`,
        );
      }
    } finally {
      gpu.destroy();
    }
  },
);

const GQA: StateModel = { heads: 4, kvHeads: 2, depth: 4, capacity: 8 };

test.skipIf(!GPU_AVAILABLE)(
  "the states form with GQA (r=2) matches the CPU reference (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    const session = await stateSession(gpu, GQA);
    const context = await session.createGenerationContext({ chunkLength: 4 });
    const state = newOracle(GQA, 8);
    try {
      await assertStep(session, context, GQA, 8, state, 4, 4, 2, "prefill");
      await assertStep(session, context, GQA, 8, state, 1, 1, 17, "decode 1");
      await assertStep(session, context, GQA, 8, state, 1, 1, 29, "decode 2");
      assertEquals(context.pastLength, 6);
    } finally {
      await context.dispose();
      await session.dispose();
      gpu.destroy();
    }
  },
);

/**
 * **Census** (ADR 0058 decision 4) — checks that the GQA variant keys **actually ran**.
 *
 * MUST: the source is `lastRunPipelines`, which does not depend on timing. An empty census is an
 * unconditional FAIL (letting empty `entries` pass would leave every case green and unchecked —
 * `requireCensus`).
 */
test.skipIf(!GPU_AVAILABLE)(
  "the states-form attention dispatches run as their variant keys say (census, real GPU)",
  async () => {
    const gpu = await acquireGpu();
    try {
      for (const [label, model, sliding] of [
        ["full r=1", FULL, false],
        ["full r=2", GQA, false],
        ["sliding r=1", SLIDING, true],
      ] as const) {
        const session = await stateSession(gpu, model);
        const context = await session.createGenerationContext({ chunkLength: 2 });
        try {
          await runStep(session, context, model, stepInputs(model, 2, 7), 2, 2);
          const keys = requireCensus(session.diagnostics().lastRunPipelines, label).map(
            (row) => row.key,
          );
          assert(keys.length > 0, `${label}: the breakdown is empty (the key check would miss)`);
          assertEquals(
            keys.includes(stateQkKey(sliding, model.heads !== model.kvHeads)),
            true,
            `${label}: the expected ①QK variant key is missing (${keys.join(" / ")})`,
          );
          // The other side of being a separate family — not a single fused-attention key appears.
          assertEquals(
            keys.filter((key) => key.startsWith("attention_qk")),
            [],
            `${label}: fused-attention keys are mixed in`,
          );
        } finally {
          await context.dispose();
          await session.dispose();
        }
      }
    } finally {
      gpu.destroy();
    }
  },
);

/**
 * **Census** (ADR 0058 decision 4 ③) — with `stateAttentionReduce: "parallel"`, checks that the
 * ①' and ③' keys (`:par`) **actually run in the expected plans** and that not one of their
 * sequential counterparts appears. Conversely, the default produces only ① ③ (closing both
 * directions: the opt-in silently falling back to the default / the default silently moving up to
 * the variant).
 *
 * MUST: check **both** stages (one option switches the two stages together, so checking only one
 * would let "①' dropped out of the wiring and ① is running" through — the values stay inside the
 * band, so the numeric checks would not fire either).
 * MUST: **only ①' has an applicability condition on the option** (`M ≤ 8` = decode and the
 * speculative verify). This table checks "①' + ③'" at M=1 / M=2; the non-applicable case (M=9 →
 * ① + ③') is held by the census of the 3 paths below (this model's full slot has capacity 8 and
 * cannot take M=9). ③' does not depend on M (the applicability condition and the measurements are
 * in `stateQkParallelEligible` of src/kernels/state-attention.ts). To keep the check independent
 * of the condition, the expectation is written per row as a `qkParallel` boolean (importing the
 * decision would let the test be wrong together with the implementation).
 */
test.skipIf(!GPU_AVAILABLE)(
  "the reduction forms of the states-form attention ①QK / ③PV run as the option and the plan's M say (census, real GPU)",
  async () => {
    const gpu = await acquireGpu();
    try {
      for (const [label, model, sliding, reduce, chunkRows, qkParallel, pvParallel] of [
        ["full r=2 parallel decode M=1", GQA, false, "parallel", 1, true, true],
        // M=2 (the speculative verify's shape, M ≤ 8) moves up both ①' and ③'
        ["full r=2 parallel verify M=2", GQA, false, "parallel", 2, true, true],
        ["sliding r=1 parallel decode M=1", SLIDING, true, "parallel", 1, true, true],
        ["sliding r=1 parallel verify M=2", SLIDING, true, "parallel", 2, true, true],
        // With the option at its default, not even M=1 moves up (catches an implementation that
        // looks only at the applicability condition and ignores the option)
        ["full r=2 sequential decode M=1", GQA, false, "sequential", 1, false, false],
        ["full r=2 sequential prefill M=2", GQA, false, "sequential", 2, false, false],
      ] as const) {
        const session = await stateSession(gpu, model, { stateAttentionReduce: reduce });
        // chunkLength is the row's M itself (both the M=1 decode shape and the M=9 prefill shape
        // are inside the same context contract).
        const context = await session.createGenerationContext({
          chunkLength: Math.max(chunkRows, 2),
        });
        try {
          await runStep(
            session,
            context,
            model,
            stepInputs(model, chunkRows, 7),
            chunkRows,
            chunkRows,
          );
          const keys = requireCensus(session.diagnostics().lastRunPipelines, label).map(
            (row) => row.key,
          );
          assert(keys.length > 0, `${label}: the breakdown is empty (the key check would miss)`);
          const gqa = model.heads !== model.kvHeads;
          for (const [stage, expected, other] of [
            [
              "①QK",
              qkParallel ? stateQkParallelKey(sliding, gqa) : stateQkKey(sliding, gqa),
              qkParallel ? stateQkKey(sliding, gqa) : stateQkParallelKey(sliding, gqa),
            ],
            [
              "③PV",
              pvParallel ? statePvParallelKey(sliding, gqa) : statePvKey(sliding, gqa),
              pvParallel ? statePvKey(sliding, gqa) : statePvParallelKey(sliding, gqa),
            ],
          ] as const) {
            assertEquals(
              keys.includes(expected),
              true,
              `${label}: the expected ${stage} key is missing (${keys.join(" / ")})`,
            );
            assertEquals(
              keys.includes(other),
              false,
              `${label}: a ${stage} key the option and the plan do not lead to is mixed in (${keys.join(
                " / ",
              )})`,
            );
          }
        } finally {
          await context.dispose();
          await session.dispose();
        }
      }
    } finally {
      gpu.destroy();
    }
  },
);

/**
 * Shapes for the folded ①' census: the selected classes fold on full layers (r = 8 at D = 512 and
 * D = 128 — at decode M = 1); r = 4 at D = 512 (E4B — excluded, perf-ledger K-95), an unmeasured D,
 * r = 3 / 16 and sliding do not.
 */
const R8: StateModel = { heads: 8, kvHeads: 1, depth: 512, capacity: 32 };
const R4: StateModel = { heads: 8, kvHeads: 2, depth: 512, capacity: 32 };
const R8_D128: StateModel = { heads: 16, kvHeads: 2, depth: 128, capacity: 32 };
const R8_D16: StateModel = { heads: 8, kvHeads: 1, depth: 16, capacity: 32 };
const R3: StateModel = { heads: 6, kvHeads: 2, depth: 16, capacity: 32 };
const R16: StateModel = { heads: 16, kvHeads: 1, depth: 16, capacity: 32 };
const R8_SLIDING: StateModel = { heads: 8, kvHeads: 1, depth: 16, capacity: 12, window: 8 };

/**
 * **census + values** of the folded ①' (perf-ledger K-55) through a Session: a full-attention plan
 * of a selected class (r, D, M) that takes ①' runs the folded key in its place (and no other ①QK
 * key), every other plan keeps its ①QK path, and the outputs still match the CPU reference along a prefill → decode
 * chain. The word-for-word identity with ①' is held by gpu_state_attention_fold_test.ts; this test
 * is about the wiring of the single selection site (key, WGSL, params and the per-step workgroup
 * count of `buildStateAttention`).
 * MUST: the expectation is written per row (importing the selection rule would let the test be
 * wrong together with the implementation).
 */
test.skipIf(!GPU_AVAILABLE)(
  "full-attention plans of the selected classes run the folded ①' and every other plan its own ①QK (census + values, real GPU)",
  async () => {
    const gpu = await acquireGpu();
    try {
      for (const [label, model, reduce, expected] of [
        ["full r=8 D=512 parallel", R8, "parallel", stateQkFoldedKey(8)],
        // E4B's r = 4 was measured faster but differs from ①' on D3D12 (perf-ledger K-95)
        ["full r=4 D=512 parallel", R4, "parallel", stateQkParallelKey(false, true)],
        // The fused option keeps ①' on the QK side, so the fold applies there too
        ["full r=8 D=512 parallel-fused", R8, "parallel-fused", stateQkFoldedKey(8)],
        // The MiniCPM5 class folds at decode M = 1 only (its M = 4 prefill plan keeps ①', and the
        // decode values still follow the reference across the two)
        ["full r=8 D=128 parallel", R8_D128, "parallel", stateQkFoldedKey(8)],
        // An unmeasured head dim keeps ①'
        ["full r=8 D=16 parallel", R8_D16, "parallel", stateQkParallelKey(false, true)],
        ["full r=2 parallel", GQA, "parallel", stateQkParallelKey(false, true)],
        ["full r=3 parallel", R3, "parallel", stateQkParallelKey(false, true)],
        ["full r=16 parallel", R16, "parallel", stateQkParallelKey(false, true)],
        ["sliding r=8 parallel", R8_SLIDING, "parallel", stateQkParallelKey(true, true)],
        // The fold rides on ①' only — the reference path stays ①
        ["full r=8 sequential", R8, "sequential", stateQkKey(false, true)],
      ] as const) {
        const capacity = model.capacity;
        if (typeof capacity !== "number") throw new Error(`${label}: needs a numeric capacity`);
        const session = await stateSession(gpu, model, { stateAttentionReduce: reduce });
        const context = await session.createGenerationContext({ chunkLength: 4 });
        const state = newOracle(model, capacity);
        try {
          await assertStep(session, context, model, capacity, state, 4, 4, 5, `${label} prefill`);
          for (const [step, salt] of [
            ["decode 1", 13],
            ["decode 2", 29],
          ] as const) {
            await assertStep(
              session,
              context,
              model,
              capacity,
              state,
              1,
              1,
              salt,
              `${label} ${step}`,
            );
            const keys = requireCensus(session.diagnostics().lastRunPipelines, label).map(
              (row) => row.key,
            );
            assertEquals(
              keys.filter((key) => key.startsWith("attention_state_qk")),
              [expected],
              `${label} ${step}: the ①QK keys that ran (${keys.join(" / ")})`,
            );
          }
        } finally {
          await context.dispose();
          await session.dispose();
        }
      }
    } finally {
      gpu.destroy();
    }
  },
);

/**
 * A shape that reaches ①ₜ / ③ₜ (the row-tile-sharing paths selected at `M ≥ 16`) (capacity for
 * `M = 16`).
 */
const TILED: StateModel = { heads: 4, kvHeads: 2, depth: 4, capacity: 64 };

const TILED_SLIDING: StateModel = { heads: 2, kvHeads: 2, depth: 4, capacity: 16, window: 16 };

/**
 * **Census** (the tiled-path version of ADR 0058 decision 4 ③ — perf-ledger K-13) — checks that
 * **the 3 paths each** of ①QK and ③PV run as the plan's `M` and the option say.
 *
 * ①ₜ / ③ₜ (sharing row tiles of K / V) are **bit-identical to ① / ③**, so they are **default paths
 * that do not depend on the option**, selected only by plans with `M ≥ 16` (the applicability
 * conditions are `stateQkTiledEligible` / `statePvTiledEligible` in
 * src/kernels/state-attention.ts). So the expectation is "M ≥ 16 → the tiled path (unchanged even
 * with the option at `"parallel"`), M < 16 → as the option says". **Only ①' is further limited to
 * M ≤ 8 within the option**, so the parallel row at `M = 9` **differs per stage**: "①QK
 * sequential, ③PV ③'".
 *
 * MUST: check **all** 3 paths (both that the expected one appears and that neither of the other two
 * appears at all). Checking only one side would let "the tiled path dropped out of the wiring and
 * the reference path is running" through — ①ₜ / ③ₜ do not differ from the reference paths by a
 * single bit, so the numeric checks cannot fire in principle.
 * MUST: write the expectation per row **and per stage** (importing the decision would let the test
 * be wrong together with the implementation).
 */
test.skipIf(!GPU_AVAILABLE)(
  "the 3 paths of the states-form attention ①QK / ③PV run as the plan's M and the option say (census, real GPU)",
  async () => {
    const gpu = await acquireGpu();
    try {
      for (const [label, model, sliding, reduce, chunkRows, qkExpected, pvExpected] of [
        ["full r=2 prefill M=16", TILED, false, "sequential", 16, "tiled", "tiled"],
        // The option does not move the tiled paths (bit-identical, so they are default paths — not
        // something the option switches)
        ["full r=2 parallel prefill M=16", TILED, false, "parallel", 16, "tiled", "tiled"],
        ["sliding r=1 prefill M=16", TILED_SLIDING, true, "sequential", 16, "tiled", "tiled"],
        // At M=2 neither tiled path applies, and both ①' (M ≤ 8) and ③' move up as the option says
        ["full r=2 parallel M=2", TILED, false, "parallel", 2, "parallel", "parallel"],
        // M=9 is outside ①' (beyond M ≤ 8, below the tiled path), so ①QK stays sequential and only
        // ③PV moves up to ③' as the option says (the direct observation point of the stages
        // having different applicability ranges)
        ["full r=2 parallel M=9", TILED, false, "parallel", 9, "sequential", "parallel"],
        // M=1 is not applicable either. Both stages split as the option says
        ["full r=2 decode M=1", TILED, false, "sequential", 1, "sequential", "sequential"],
        ["full r=2 parallel decode M=1", TILED, false, "parallel", 1, "parallel", "parallel"],
      ] as const) {
        const session = await stateSession(gpu, model, { stateAttentionReduce: reduce });
        const context = await session.createGenerationContext({ chunkLength: chunkRows });
        try {
          await runStep(
            session,
            context,
            model,
            stepInputs(model, chunkRows, 7),
            chunkRows,
            chunkRows,
          );
          const keys = requireCensus(session.diagnostics().lastRunPipelines, label).map(
            (row) => row.key,
          );
          assert(keys.length > 0, `${label}: the breakdown is empty (the key check would miss)`);
          const gqa = model.heads !== model.kvHeads;
          const shown = keys.join(" / ");
          for (const [stage, expected, routes] of [
            [
              "①QK",
              qkExpected,
              [
                ["sequential", stateQkKey(sliding, gqa)],
                ["parallel", stateQkParallelKey(sliding, gqa)],
                ["tiled", stateQkTiledKey(sliding, gqa, chunkRows)],
              ],
            ],
            [
              "③PV",
              pvExpected,
              [
                ["sequential", statePvKey(sliding, gqa)],
                ["parallel", statePvParallelKey(sliding, gqa)],
                ["tiled", statePvTiledKey(sliding, gqa, chunkRows)],
              ],
            ],
          ] as const) {
            for (const [route, key] of routes) {
              assertEquals(
                keys.includes(key),
                route === expected,
                `${label}: presence of the ${stage} ${route} path key ${key} differs from the expectation (breakdown of the run: ${shown})`,
              );
            }
          }
        } finally {
          await context.dispose();
          await session.dispose();
        }
      }
    } finally {
      gpu.destroy();
    }
  },
);

test.skipIf(!GPU_AVAILABLE)(
  "rewind works on a context with only full slots and is rejected outright on a context with any sliding slot (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    const full = await stateSession(gpu, FULL);
    const sliding = await stateSession(gpu, SLIDING);
    const context = await full.createGenerationContext({ chunkLength: 4 });
    const slidingContext = await sliding.createGenerationContext({ chunkLength: 2 });
    const state = newOracle(FULL, 8);
    try {
      await assertStep(full, context, FULL, 8, state, 4, 4, 0, "prefill");
      await assertStep(full, context, FULL, 8, state, 1, 1, 101, "decode (discarded)");
      assertEquals(context.pastLength, 5);

      // Rewind: go back to logical position 4 and overwrite the same position with other inputs.
      context.rewind(4);
      state.past = 4;
      assertEquals(context.pastLength, 4);
      await assertStep(full, context, FULL, 8, state, 1, 1, 211, "decode after rewind");
      assertEquals(context.pastLength, 5, "it advances again from the rewound position");

      // A context with even one sliding slot rejects every positional rewind (ADR 0066 addendum 2).
      const rejected = assertThrows(() => slidingContext.rewind(0), ExecutionError);
      assert(rejected.message.includes("sliding"), rejected.message);
      assert(rejected.message.includes("kslot"), rejected.message);
    } finally {
      await context.dispose();
      await slidingContext.dispose();
      await full.dispose();
      await sliding.dispose();
      gpu.destroy();
    }
  },
);

/**
 * **The lease of a run in progress** (phase D-7).
 *
 * A run writes the uniform with the `pastLength` it captured at its start, computes the dispatch
 * counts from it, and on success advances from that value. If `rewind` moved the logical length
 * midway, "the P the GPU read" and "the P the advance was based on" would split, and only the KV's
 * logical positions would drift, with no exception and no NaN. The lease is taken in the
 * **synchronous section** of `Session.run`, so it catches even a sequence that does not await
 * `run()` (taking the lease in the body would open a window of one microtask).
 */
test.skipIf(!GPU_AVAILABLE)(
  "rewind while a generation run is in progress fails loudly (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    const session = await stateSession(gpu, FULL);
    const context = await session.createGenerationContext({ chunkLength: 1 });
    const state = newOracle(FULL, 8);
    try {
      await assertStep(session, context, FULL, 8, state, 1, 1, 0, "decode 1");
      await assertStep(session, context, FULL, 8, state, 1, 1, 17, "decode 2");
      assertEquals(context.pastLength, 2);

      // Fire rewind **without awaiting** (an implementation that takes the lease in the body lets
      // this through).
      const inputs = stepInputs(FULL, 1, 29);
      const running = runStep(session, context, FULL, inputs, 1, 1);
      const rejected = assertThrows(() => context.rewind(0), ExecutionError);
      assert(rejected.message.includes("a generation run is in progress"), rejected.message);

      // The run itself is unharmed (the rejected rewind does not take the run down with it).
      const actual = await running;
      const expected = advanceOracle(FULL, 8, state, inputs, 1, 1);
      const report = compareTensors(
        { dtype: "f32", data: actual },
        { dtype: "f32", data: expected },
        STATE_TOLERANCE,
      );
      assertEquals(report.pass, true, `run under lease: ${formatAllclose(report)}`);
      assertEquals(context.pastLength, 3);

      // Control: a rewind after settlement passes (the lease was really returned = it is not a
      // permanent rejection).
      context.rewind(1);
      assertEquals(context.pastLength, 1);
    } finally {
      await context.dispose();
      await session.dispose();
      gpu.destroy();
    }
  },
);

/**
 * **Two-stage dispose** (phase D-7).
 *
 * Only the first stage (no new admissions) takes effect synchronously; cutting off the internal
 * surface is the second stage, in which the dispose body queued on the Session chain runs. The
 * dispose body runs **after** the preceding runs, so the non-awaited sequence `run();
 * context.dispose();` has the same semantics as the Session's `run(); session.dispose();` (an
 * admitted run completes, and then the buffers are released). An implementation that closes in
 * one stage fails the run here with "disposed".
 */
test.skipIf(!GPU_AVAILABLE)(
  "dispose closes only new admissions synchronously, and an admitted run completes (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    const session = await stateSession(gpu, FULL);
    const context = await session.createGenerationContext({ chunkLength: 1 });
    const state = newOracle(FULL, 8);
    try {
      await assertStep(session, context, FULL, 8, state, 1, 1, 0, "decode 1");
      await assertStep(session, context, FULL, 8, state, 1, 1, 11, "decode 2");
      assertEquals(context.pastLength, 2);

      // dispose **without awaiting**. The admitted run runs to the end with past=2.
      const inputs = stepInputs(FULL, 1, 23);
      const running = runStep(session, context, FULL, inputs, 1, 1);
      const disposing = context.dispose();
      const actual = await running;
      const expected = advanceOracle(FULL, 8, state, inputs, 1, 1);
      const report = compareTensors(
        { dtype: "f32", data: actual },
        { dtype: "f32", data: expected },
        STATE_TOLERANCE,
      );
      assertEquals(report.pass, true, `run alongside dispose: ${formatAllclose(report)}`);
      await disposing;

      // The first stage's effect: the user surface is closed from **the moment dispose is called**.
      assertThrows(() => context.pastLength, ExecutionError, "disposed");
      // A new run is rejected at admission (before it is queued on the chain = decided in the
      // synchronous section).
      const rejected = await assertRejects(
        () => runStep(session, context, FULL, stepInputs(FULL, 1, 41), 1, 1),
        ExecutionError,
      );
      assert(rejected.message.includes("disposed"), rejected.message);
    } finally {
      await session.dispose();
      gpu.destroy();
    }
  },
);

/**
 * **Forced-split parity** of the row blocks (ADR 0067 decision 7 / the same approach as ADR 0060).
 *
 * The default block count is decided statically from the device's
 * `maxStorageBufferBindingSize`, so on a machine with a roomy limit it is always 1 = the
 * multi-block path (row_offset across blocks, per-block temporaries) never runs.
 * `ROW_BLOCK_SPLIT` is the only way to run that path on real hardware, and the values MUST be
 * **bit-identical** (only the width of the intermediate materialization is folded).
 */
test.skipIf(!GPU_AVAILABLE)(
  "forcing a row-block split keeps the output bit-identical (states form, real GPU)",
  async () => {
    const gpu = await acquireGpu();
    const single = await stateSession(gpu, FULL);
    const split = await stateSession(gpu, FULL, { [ROW_BLOCK_SPLIT]: 2 });
    const inputs = stepInputs(FULL, 4, 97);
    try {
      const contexts = await Promise.all([
        single.createGenerationContext({ chunkLength: 4 }),
        split.createGenerationContext({ chunkLength: 4 }),
      ]);
      try {
        const one = await runStep(single, contexts[0], FULL, inputs, 4, 4);
        const many = await runStep(split, contexts[1], FULL, inputs, 4, 4);
        assertEquals(
          Array.from(new Uint32Array(many.buffer, many.byteOffset, many.length)),
          Array.from(new Uint32Array(one.buffer, one.byteOffset, one.length)),
          "the run split into 2 blocks does not match the 1-block run bit for bit",
        );
      } finally {
        for (const context of contexts) await context.dispose();
      }
    } finally {
      await single.dispose();
      await split.dispose();
      gpu.destroy();
    }
  },
);

/**
 * The states form split by the budget `attentionRowBlockMaxBytes` (ADR 0124) also **matches the
 * 1-block run bit for bit**.
 *
 * FULL's prefill (B·H = 4, capacity 8, M = 4) has 4×8×4 = 128 B of scores per row. A 256 B budget
 * gives 2 rows per block, so 2 blocks (without a budget the binding limit has room, so 1 block).
 * The states form cannot be combined with numeric variants (the test below), so f32 alone is
 * enough.
 * MUST: not tautological. Check at the same time that the ①QK dispatch count (= the block count) is
 * 2, unlike the 1 without a budget, and that the intermediate peak is strictly smaller (do not let
 * "the budget never reaches the derivation phase and the same path is fired twice" go green).
 */
test.skipIf(!GPU_AVAILABLE)(
  "the budget-split states form is bit-identical to the 1-block run, with more blocks and a smaller peak (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    // Even if building the 2nd throws, the 1st and the gpu are destroyed (one try per Session).
    try {
      const single = await stateSession(gpu, FULL);
      try {
        const budgeted = await stateSession(gpu, FULL, { attentionRowBlockMaxBytes: 256 });
        try {
          const inputs = stepInputs(FULL, 4, 97);
          const contexts = await Promise.all([
            single.createGenerationContext({ chunkLength: 4 }),
            budgeted.createGenerationContext({ chunkLength: 4 }),
          ]);
          try {
            const one = await runStep(single, contexts[0], FULL, inputs, 4, 4);
            const many = await runStep(budgeted, contexts[1], FULL, inputs, 4, 4);
            assertEquals(
              Array.from(new Uint32Array(many.buffer, many.byteOffset, many.length)),
              Array.from(new Uint32Array(one.buffer, one.byteOffset, one.length)),
              "the run split into 2 blocks by the budget does not match the 1-block run bit for bit",
            );
            assert(
              new Set(one).size > 1,
              "the output is constant (the bit comparison is a tautology)",
            );
            const qkKey = stateQkKey(false, false);
            const blocksOf = (session: Session): number =>
              countDispatches(
                requireCensus(session.diagnostics().lastRunPipelines, "states form"),
                qkKey,
              );
            assertEquals(blocksOf(single), 1, "block count without a budget");
            assertEquals(blocksOf(budgeted), 2, "block count with the budget");
            const peakOf = (session: Session): number =>
              session.diagnostics().lastRun?.peakTransientBytes ?? 0;
            assert(
              peakOf(single) > 0,
              "the intermediate peak is 0 (the observation point would miss)",
            );
            assert(
              peakOf(budgeted) < peakOf(single),
              `the intermediate peak ${peakOf(budgeted)}B did not shrink from ${peakOf(single)}B without a budget`,
            );
          } finally {
            for (const context of contexts) await context.dispose();
          }
        } finally {
          await budgeted.dispose();
        }
      } finally {
        await single.dispose();
      }
    } finally {
      gpu.destroy();
    }
  },
);

/**
 * Numeric variants (`attentionCompute` / `attentionScoreStorage`) × the states form **fail loudly**
 * (ADR 0058 decision 3 — "an unimplemented combination fails loudly instead of degrading").
 *
 * MUST: do not run the f32 path silently. An opt-in that was requested but has no effect would
 * stay with unchanged values and diagnostics, merely "not faster".
 */
test.skipIf(!GPU_AVAILABLE)(
  "the states form cannot be combined with numeric variants other than f32 (fails loudly instead of degrading, real GPU)",
  async () => {
    const gpu = await acquireGpu();
    const cases: readonly (readonly [string, SessionOptions, string])[] = [
      ["attentionCompute a8", { attentionCompute: "a8" }, "attentionCompute"],
      ["attentionScoreStorage f16", { attentionScoreStorage: "f16" }, "attentionScoreStorage"],
    ];
    try {
      for (const [label, options, needle] of cases) {
        const session = await stateSession(gpu, FULL, options);
        const context = await session.createGenerationContext({ chunkLength: 1 });
        try {
          const error = await assertRejects(
            () => runStep(session, context, FULL, stepInputs(FULL, 1, 0), 1, 1),
            ExecutionError,
          );
          assert(error.message.includes(needle), `${label}: ${error.message}`);
          assertEquals(context.pastLength, 0, `${label}: a rejected run does not advance`);
        } finally {
          await context.dispose();
          await session.dispose();
        }
      }
    } finally {
      gpu.destroy();
    }
  },
);

const SMALL: StateModel = { heads: 2, kvHeads: 2, depth: 4, capacity: 6 };

test.skipIf(!GPU_AVAILABLE)(
  "exceeding a full slot's capacity fails before encoding and does not taint the context (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    const session = await stateSession(gpu, SMALL);
    const context = await session.createGenerationContext({ chunkLength: 4 });
    const state = newOracle(SMALL, 6);
    try {
      await assertStep(session, context, SMALL, 6, state, 4, 4, 0, "prefill");
      assertEquals(context.pastLength, 4);

      // 4 + 4 > 6. It fails without queuing a single dispatch.
      const inputs = stepInputs(SMALL, 4, 13);
      const error = await assertRejects(
        () => runStep(session, context, SMALL, inputs, 4, 4),
        ExecutionError,
      );
      assert(error.message.includes("full slot"), error.message);
      // MUST: not tainted (if it were poisoned, reading pastLength would fail).
      assertEquals(
        context.pastLength,
        4,
        "a rejected run changes neither the logical length nor the taint",
      );

      // The next run that fits the capacity passes (the other side of the rejection not breaking
      // the context).
      await assertStep(session, context, SMALL, 6, state, 4, 2, 31, "run that fits");
      assertEquals(context.pastLength, 6);
    } finally {
      await context.dispose();
      await session.dispose();
      gpu.destroy();
    }
  },
);

/**
 * The check of the execution forms (ADR 0066 decision 4 — only the fixed-length chunk and decode).
 *
 * `M ∈ the allowed set` (this context declares no `chunkBuckets`, so `{chunkLength, 1}`) is now
 * enforced at run time (phase D-7), so the upper bound of `queryLength` is **folded into** the
 * single `Q ≤ M`: with `M = chunkLength`, `Q ≤ M` implies `Q ≤ chunkLength`, and with `M = 1` it
 * implies `Q = 1`. So a shape that reaches only `Q > chunkLength` through `Session.run` **cannot be
 * built structurally** (it would need M ∉ the allowed set, which fails first). The
 * `queryLength ≤ chunkLength` check on the `GenerationContext` side stays as a backstop of the
 * internal surface, and the check that drives it directly lives in
 * tests/gpu_generation_context_test.ts (no double bookkeeping).
 */
test.skipIf(!GPU_AVAILABLE)(
  "the only execution forms are the fixed-length chunk and decode, and queryLength is rejected by M / as a positive integer (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    const session = await stateSession(gpu, FULL);
    // With chunkLength=2 only M=2 (the prefill form) and M=1 (the decode form) pass.
    const context = await session.createGenerationContext({ chunkLength: 2 });
    try {
      // ① Q > M (outside the decode form's declared shape — seen by the chunkRows the derivation
      // phase collected)
      const overRows = await assertRejects(
        () => runStep(session, context, FULL, stepInputs(FULL, 1, 0), 1, 2),
        ExecutionError,
      );
      assert(overRows.message.includes("physical chunk row count"), overRows.message);

      // ② Q > M (the prefill form — the chunkLength bound is folded into this one too)
      const overChunk = await assertRejects(
        () => runStep(session, context, FULL, stepInputs(FULL, 2, 0), 2, 3),
        ExecutionError,
      );
      assert(overChunk.message.includes("physical chunk row count"), overChunk.message);

      // ③ Q = 0 ("a run that advances nothing" — the correspondence between progress and the
      // physical contents could no longer be observed)
      const zero = await assertRejects(
        () => runStep(session, context, FULL, stepInputs(FULL, 2, 0), 2, 0),
        ExecutionError,
      );
      assert(zero.message.includes("queryLength"), zero.message);

      // ④ M ∉ the allowed set (this context declares no chunkBuckets, so {1, chunkLength}):
      //    letting any M through would add a plan with a different key per distinct M, pollute the
      //    PreparedPlan LRU, and silently drop the decode hot path into re-derivation (ADR 0066
      //    decision 4, "two PreparedPlans in the steady state" — the addendum on buckets widens it
      //    only "by the declared number").
      for (const chunkRows of [3, 4]) {
        const wrongForm = await assertRejects(
          () => runStep(session, context, FULL, stepInputs(FULL, chunkRows, 0), chunkRows, 1),
          ExecutionError,
        );
        assert(wrongForm.message.includes("fixed chunk contract"), wrongForm.message);
      }

      assertEquals(context.pastLength, 0, "a rejected run advances nothing");

      // Control: the two execution forms pass as is (the other side of the check not "turning
      // everything red").
      const state = newOracle(FULL, 8);
      await assertStep(session, context, FULL, 8, state, 2, 2, 71, "prefill form (M=chunkLength)");
      await assertStep(session, context, FULL, 8, state, 1, 1, 83, "decode form (M=1)");
      assertEquals(context.pastLength, 3);
    } finally {
      await context.dispose();
      await session.dispose();
      gpu.destroy();
    }
  },
);

/**
 * Extra check of the new rejection shapes (M=2 / M=3 with chunkLength=4 — matching neither the
 * decode form's 1).
 */
test.skipIf(!GPU_AVAILABLE)(
  "an M that is neither chunkLength nor 1 fails loudly (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    const session = await stateSession(gpu, FULL);
    const context = await session.createGenerationContext({ chunkLength: 4 });
    try {
      for (const chunkRows of [2, 3]) {
        const error = await assertRejects(
          () => runStep(session, context, FULL, stepInputs(FULL, chunkRows, 0), chunkRows, 1),
          ExecutionError,
        );
        assert(error.message.includes(`physical chunk row count ${chunkRows}`), error.message);
        // The diagnosis lists the allowed set (no buckets are declared, so the prefill form is the
        // one chunkLength).
        assert(error.message.includes("prefill form's {4}"), error.message);
      }
      assertEquals(context.pastLength, 0, "a rejected run advances nothing");
    } finally {
      await context.dispose();
      await session.dispose();
      gpu.destroy();
    }
  },
);

/**
 * The pre-run check (`assertGenerationRun`) comes after the derivation phase, so a miss run has
 * already loaded its plan and params by the time it fails there. That is a host-side failure and
 * the errorScope caught nothing — the plan and params that were built are valid, so they are not
 * removed. Removing them would re-derive and allocate fresh params on every repetition of the same
 * failure, and since they would only be dropped from the cache they would pile up until `dispose`
 * (the ADR 0095 addendum of 2026-10-10, "a GPU-side failure rolls back the params and plan that
 * execution unit added"). The run fails before the errorScope pop, so this check also turns red on
 * the shape that "removes without looking at what the discarded scope holds".
 */
test.skipIf(!GPU_AVAILABLE)(
  "when a miss run fails the pre-run check its plan and params stay, and repeating it allocates no new params (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    const session = await stateSession(gpu, FULL);
    const context = await session.createGenerationContext({ chunkLength: 2 });
    // queryLength 2 on the decode form (M=1) — fails the pre-run check after the derivation for
    // exceeding the "physical chunk row count".
    const failOnce = async (): Promise<void> => {
      const error = await assertRejects(
        () => runStep(session, context, FULL, stepInputs(FULL, 1, 0), 1, 2),
        ExecutionError,
      );
      assert(error.message.includes("physical chunk row count"), error.message);
    };
    try {
      const initial = session.diagnostics().weights.allocCount;
      await failOnce();
      assertEquals(
        session.diagnostics().lastRunPrepared?.hit,
        false,
        "the 1st run did not derive (the check would miss)",
      );
      const derived = session.diagnostics().weights.allocCount;
      assert(derived > initial, "the 1st run did not load params (the check would miss)");

      await failOnce();
      await failOnce();
      assertEquals(
        session.diagnostics().weights.allocCount,
        derived,
        "fresh params are allocated on every repetition of the same host-side failure",
      );
      assertEquals(
        session.diagnostics().lastRunPrepared?.hit,
        true,
        "a host-side failure discards the plan that was built",
      );
      assertEquals(context.pastLength, 0, "a rejected run advances nothing");
    } finally {
      await context.dispose();
      await session.dispose();
      gpu.destroy();
    }
  },
);

/**
 * The counterpart of the check above. Even for a run that fails before the pop, if the discarded
 * errorScope caught a GPU-side failure (a non-fatal OOM while allocating params), the plan and
 * params that run added are removed. Keeping them would make the next runs keep hitting a plan that
 * refers directly to invalid buffers. The body's exception stays the host-side failure; it is not
 * replaced by the caught OOM. The injection replaces only the pop of the out-of-memory scope
 * (`helpers/oom-injection.ts`), so it hits the cleanup pop of the run that failed the pre-run
 * check.
 */
test.skipIf(!GPU_AVAILABLE)(
  "when a miss run fails the pre-run check but the discarded errorScope caught a non-fatal OOM, its plan and params are removed and rebuilt by the next run (injection, real GPU)",
  async () => {
    const gpu = await acquireGpu();
    const session = await stateSession(gpu, FULL);
    const context = await session.createGenerationContext({ chunkLength: 2 });
    const oom = injectOutOfMemory(gpu.device);
    // queryLength 2 on the decode form (M=1) — fails the pre-run check after the derivation for
    // exceeding the "physical chunk row count".
    const failOnce = async (): Promise<void> => {
      const error = await assertRejects(
        () => runStep(session, context, FULL, stepInputs(FULL, 1, 0), 1, 2),
        ExecutionError,
      );
      assert(error.message.includes("physical chunk row count"), error.message);
    };
    try {
      oom.arm();
      await failOnce();
      assertEquals(
        oom.fired(),
        1,
        "the injection did not hit the run's cleanup pop (the check would miss)",
      );
      assertEquals(
        session.diagnostics().lastRunPrepared?.hit,
        false,
        "the 1st run did not derive (the check would miss)",
      );
      const derived = session.diagnostics().weights.allocCount;

      await failOnce();
      assertEquals(
        session.diagnostics().lastRunPrepared?.hit,
        false,
        "the plan of a run whose discarded errorScope caught an OOM remains",
      );
      assert(
        session.diagnostics().weights.allocCount > derived,
        "the params of a run whose discarded errorScope caught an OOM are reused from the cache",
      );
      assertEquals(context.pastLength, 0, "a rejected run advances nothing");
    } finally {
      oom.restore();
      await context.dispose();
      await session.dispose();
      gpu.destroy();
    }
  },
);

/**
 * Lays out `query` valid rows into `[1, planes, rows, depth]` of `rows` physical rows, padded with
 * 0 (the input values of the pad region MUST be 0 — ADR 0066 addendum 6).
 *
 * Needed to pass **the same valid rows** to the bucket form (M=4) and the prefill form (M=8). The
 * flat sequence of `stepInputs` has its row breaks at different places per M, so passing it to
 * both as is would compare different values.
 */
const scatterRows = (
  planes: number,
  rows: number,
  depth: number,
  query: number,
  valid: Float32Array<ArrayBuffer>,
): Float32Array<ArrayBuffer> => {
  const out = new Float32Array(planes * rows * depth);
  for (let plane = 0; plane < planes; plane += 1) {
    const from = plane * query * depth;
    out.set(valid.subarray(from, from + query * depth), plane * rows * depth);
  }
  return out as Float32Array<ArrayBuffer>;
};

/**
 * Takes only the first `query` rows of `[1, planes, rows, depth]` (for comparisons with the pad
 * rows dropped).
 */
const takeRows = (
  planes: number,
  rows: number,
  depth: number,
  query: number,
  data: Float32Array<ArrayBuffer>,
): Float32Array<ArrayBuffer> => {
  const out = new Float32Array(planes * query * depth);
  for (let plane = 0; plane < planes; plane += 1) {
    const from = plane * rows * depth;
    out.set(data.subarray(from, from + query * depth), plane * query * depth);
  }
  return out as Float32Array<ArrayBuffer>;
};

/** Inputs of M rows with the same valid rows (using the `salt` rule of `stepInputs` as is). */
const bucketInputs = (
  model: StateModel,
  chunkRows: number,
  query: number,
  salt: number,
): StepInputs => ({
  q: scatterRows(
    model.heads,
    chunkRows,
    model.depth,
    query,
    seeded(model.heads * query * model.depth, (i) => QUERY(i + salt)),
  ),
  k: scatterRows(
    model.kvHeads,
    chunkRows,
    model.depth,
    query,
    seeded(model.kvHeads * query * model.depth, (i) => KEY(i + salt)),
  ),
  v: scatterRows(
    model.kvHeads,
    chunkRows,
    model.depth,
    query,
    seeded(model.kvHeads * query * model.depth, (i) => VALUE(i + salt)),
  ),
});

/**
 * The extra prefill forms chunkBuckets allows (the ADR 0066 addendum on buckets).
 *
 * Three things are checked: ① the bucket rows pass ② the same row count fails on a context without
 * buckets (the other side of the permission coming from the buckets) ③ **the valid rows' output of
 * the bucket form matches the prefill form (M=chunkLength)** — running a short chunk without pad is
 * a speedup only as long as this match holds, so if it broke the result would be "fast but
 * different values" (with no exception and no warning).
 */
test.skipIf(!GPU_AVAILABLE)(
  "the physical row counts of chunkBuckets pass, and the output of the valid rows matches M=chunkLength (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    const session = await stateSession(gpu, FULL);
    // chunkLength=8 / buckets [4]. The allowed set is {1, 4, 8}.
    const bucketed = await session.createGenerationContext({ chunkLength: 8, chunkBuckets: [4] });
    const plain = await session.createGenerationContext({ chunkLength: 8 });
    try {
      // ② A gap in the set (3) fails.
      const gap = await assertRejects(
        () => runStep(session, bucketed, FULL, stepInputs(FULL, 3, 0), 3, 3),
        ExecutionError,
      );
      assert(gap.message.includes("prefill form's {4, 8}"), gap.message);

      // ② The same M=4 fails on a context that declares no buckets.
      const withoutBuckets = await assertRejects(
        () => runStep(session, plain, FULL, bucketInputs(FULL, 4, 3, 0), 4, 3),
        ExecutionError,
      );
      assert(withoutBuckets.message.includes("prefill form's {8}"), withoutBuckets.message);
      assertEquals(bucketed.pastLength, 0, "a rejected run advances nothing");
      assertEquals(plain.pastLength, 0);

      // ① / ③ Run the same 3 valid rows at M=4 (bucket) and M=8 (prefill form).
      const QUERY_ROWS = 3;
      const bucketState = newOracle(FULL, 8);
      const plainState = newOracle(FULL, 8);
      const bucketIn = bucketInputs(FULL, 4, QUERY_ROWS, 0);
      const plainIn = bucketInputs(FULL, 8, QUERY_ROWS, 0);

      const bucketOut = await runStep(session, bucketed, FULL, bucketIn, 4, QUERY_ROWS);
      const plainOut = await runStep(session, plain, FULL, plainIn, 8, QUERY_ROWS);
      assertEquals(bucketed.pastLength, QUERY_ROWS);
      assertEquals(plain.pastLength, QUERY_ROWS);

      // Both match the CPU reference (looking at only one side would miss "both broke the same
      // way").
      for (const [label, state, inputs, actual, chunkRows] of [
        ["bucket form (M=4)", bucketState, bucketIn, bucketOut, 4],
        ["prefill form (M=8)", plainState, plainIn, plainOut, 8],
      ] as const) {
        const expected = advanceOracle(FULL, 8, state, inputs, chunkRows, QUERY_ROWS);
        const valid = takeRows(FULL.heads, chunkRows, FULL.depth, QUERY_ROWS, expected);
        assert(
          valid.some((value) => Math.abs(value) > 1e-3),
          `${label}: the expected output is trivial (all ~0)`,
        );
        const report = compareTensors(
          { dtype: "f32", data: actual },
          { dtype: "f32", data: expected },
          STATE_TOLERANCE,
        );
        assertEquals(report.pass, true, `${label}: ${formatAllclose(report)}`);
      }

      // ③ Compare the valid rows themselves (pad rows dropped, compared directly).
      const report = compareTensors(
        { dtype: "f32", data: takeRows(FULL.heads, 4, FULL.depth, QUERY_ROWS, bucketOut) },
        { dtype: "f32", data: takeRows(FULL.heads, 8, FULL.depth, QUERY_ROWS, plainOut) },
        STATE_TOLERANCE,
      );
      assertEquals(
        report.pass,
        true,
        `valid rows of the bucket and prefill forms: ${formatAllclose(report)}`,
      );

      // The decode form (M=1) passes as before even with buckets declared.
      await assertStep(session, bucketed, FULL, 8, bucketState, 1, 1, 37, "decode form (M=1)");
    } finally {
      await bucketed.dispose();
      await plain.dispose();
      await session.dispose();
      gpu.destroy();
    }
  },
);

const SYMBOLIC: StateModel = { heads: 2, kvHeads: 2, depth: 4, capacity: "C" };

test.skipIf(!GPU_AVAILABLE)(
  "the plan key is decided by the capacity, not by the context's identity (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    const session = await stateSession(gpu, SYMBOLIC);
    const first = await session.createGenerationContext({ chunkLength: 1, bindings: { C: 8 } });
    const second = await session.createGenerationContext({ chunkLength: 1, bindings: { C: 8 } });
    const wider = await session.createGenerationContext({ chunkLength: 1, bindings: { C: 12 } });
    const inputs = stepInputs(SYMBOLIC, 1, 0);
    try {
      // MUST: the capacity symbol `C` is **not needed** in the run's bindings (the only binding
      // point is createGenerationContext — the division of scope of ADR 0066 addendum 7. Requiring
      // it would mean double bookkeeping with the context).
      await runStep(session, first, SYMBOLIC, inputs, 1, 1);
      assertEquals(
        session.diagnostics().lastRunPrepared?.hit,
        false,
        "the first is a deriving run",
      );

      // MUST: **another context with the same capacity** has the same key (if the context's
      // identity were part of the key, a re-derivation would happen here and the decode hot path
      // would be wiped out for every sequence — ADR 0066 decision 5).
      await runStep(session, second, SYMBOLIC, inputs, 1, 1);
      assertEquals(
        session.diagnostics().lastRunPrepared?.hit,
        true,
        "another context with the same capacity got a different key",
      );
      // A hit run, so the slot backing is built (generation runs are backed too — phase D-4).
      assertEquals(session.diagnostics().planBacking.buildCount, 1);
      assertEquals(session.diagnostics().stateBacking.rebindCount, 1, "second bakes once");

      // A different capacity is a different key (the recipe bakes the capacity into the params and
      // the allocation size of S).
      await runStep(session, wider, SYMBOLIC, inputs, 1, 1);
      assertEquals(
        session.diagnostics().lastRunPrepared?.hit,
        false,
        "a context with a different capacity reuses the same recipe",
      );
      // MUST: a miss run builds no backing (the check that one-off runs do not pay for slot
      // memory), so a context with a different capacity in between leaves the active backing in
      // place = no re-bake either.
      assertEquals(session.diagnostics().planBacking.buildCount, 1, "a miss run built a backing");
      assertEquals(session.diagnostics().stateBacking.rebindCount, 1);

      // Going back hits the first key again (both are in the LRU).
      await runStep(session, first, SYMBOLIC, inputs, 1, 1);
      assertEquals(session.diagnostics().lastRunPrepared?.hit, true);
      // The backing stays the same object (its generation identifier does not move), and the only
      // bake is the one for first.
      assertEquals(session.diagnostics().planBacking.buildCount, 1);
      assertEquals(session.diagnostics().stateBacking.rebindCount, 2, "first bakes once");

      // Writing a states-only symbol into the run's bindings fails loudly (not accepted silently
      // with only the key splitting — a silent degradation in which the same plan is derived twice
      // under different keys).
      const rejected = await assertRejects(
        () => runStep(session, first, SYMBOLIC, inputs, 1, 1, { C: 8 }),
        ExecutionError,
      );
      assert(rejected.message.includes("states-only symbol"), rejected.message);
    } finally {
      for (const context of [first, second, wider]) await context.dispose();
      await session.dispose();
      gpu.destroy();
    }
  },
);

/**
 * The check of **symbols that appear in both places** (the run-time enforcement of "the division of
 * scope" in ADR 0066 addendum 7).
 *
 * Using the same symbol `M` for the slot capacity as for the inputs gives two binding points: the
 * input shape (run — `bindSymbols`) and `spec.bindings` (context). Running with them in
 * disagreement would quietly establish a split such as "allocated capacity 4, plan and dispatch
 * 3" with no exception and no warning, so it fails before encoding.
 */
test.skipIf(!GPU_AVAILABLE)(
  "a symbol that appears in both the states and the inputs fails loudly when its two binding points disagree (real GPU)",
  async () => {
    const shared: StateModel = { heads: 2, kvHeads: 2, depth: 4, capacity: "M" };
    const gpu = await acquireGpu();
    const session = await stateSession(gpu, shared);
    // A context allocated with capacity M=4 (chunkLength is also 4 = the prefill form's M).
    const context = await session.createGenerationContext({ chunkLength: 4, bindings: { M: 4 } });
    try {
      // The split shape: the run's input shape resolves M=3 (the context was allocated with 4).
      const split = await assertRejects(
        () => runStep(session, context, shared, stepInputs(shared, 3, 0), 3, 3),
        ExecutionError,
      );
      assert(split.message.includes("Symbol 'M'"), split.message);
      assert(split.message.includes("4"), split.message);
      assert(split.message.includes("3"), split.message);
      assertEquals(context.pastLength, 0, "a rejected run does not advance");

      // Control: it passes when they agree (the other side of the check not being "always red").
      const state = newOracle(shared, 4);
      await assertStep(session, context, shared, 4, state, 4, 4, 13, "agrees at M=4");
      assertEquals(context.pastLength, 4);
    } finally {
      await context.dispose();
      await session.dispose();
      gpu.destroy();
    }
  },
);

test.skipIf(!GPU_AVAILABLE)(
  "running a state-referencing graph without generation fails loudly (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    const session = await stateSession(gpu, FULL);
    try {
      const inputs = stepInputs(FULL, 1, 0);
      // It fails in the shape computation layer (a run without the slots' resolved shapes cannot
      // get past it).
      const error = await assertRejects(
        () =>
          session.run({
            q: tensor([1, FULL.heads, 1, FULL.depth], inputs.q),
            k: tensor([1, FULL.kvHeads, 1, FULL.depth], inputs.k),
            v: tensor([1, FULL.kvHeads, 1, FULL.depth], inputs.v),
          }),
        OpContractError,
      );
      assert(error.message.includes("GenerationContext"), error.message);
    } finally {
      await session.dispose();
      gpu.destroy();
    }
  },
);

/**
 * The internal surface (drives the logical length's advance directly / peeks at the baked
 * bundles). **Only for fault injection and setting up preconditions**; all regular wiring is
 * checked through `Session.run`.
 */
const internals = (context: GenerationContext) => context[RUNTIME_INTERNAL];

/**
 * The poison wiring (ADR 0066 addendum 3).
 *
 * As a deterministic failure injection, **after advancing the logical length to the u32 limit**,
 * one decode-form run that writes state goes through. Every dispatch succeeds and is submitted, but
 * the `advance` after success fails at the u32 limit, giving the shape "a run that submitted state
 * ends in an exception" — if the snapshot comparison works, the context is poisoned; if not, "a
 * context whose physical ring alone advanced" keeps returning normal-looking values.
 *
 * NOTE: what runs is **the decode form with M=1** (the fixed-chunk contract has
 * `M ∈ {chunkLength, 1}` — the large `chunkLength` is only needed to "jump to the u32 limit with
 * one manual advance").
 * NOTE: this reaches the wiring of the decision, not a GPU-side failure itself (device loss,
 * validation). Injection for that is waiting on what phase C deferred (L8 fake device).
 */
test.skipIf(!GPU_AVAILABLE)(
  "a failure of a run that submitted state poisons the context (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    // Sliding is used to avoid the full-capacity pre-run check (past + query ≤ C) — a ring keeps
    // going at any logical length.
    const session = await stateSession(gpu, SLIDING);
    const context = await session.createGenerationContext({ chunkLength: 0xffffffff });
    try {
      internals(context).advance(0, 0xffffffff);
      assertEquals(context.pastLength, 0xffffffff);

      const inputs = stepInputs(SLIDING, 1, 0);
      const error = await assertRejects(
        () => runStep(session, context, SLIDING, inputs, 1, 1),
        ExecutionError,
      );
      assert(error.message.includes("the u32 limit"), error.message);

      // MUST: every later operation is rejected (reads included — the physical ring has already
      // been written).
      const poisoned = assertThrows(() => context.pastLength, ExecutionError);
      assert(poisoned.message.includes("poisoned"), poisoned.message);
      assert(
        poisoned.message.includes("A run that submitted state-changing dispatches failed"),
        `the real cause is not kept: ${poisoned.message}`,
      );
    } finally {
      await context.dispose();
      await session.dispose();
      gpu.destroy();
    }
  },
);

/**
 * **Switching A/B** (the complete form of ADR 0066 acceptance criterion ③ — decision 5's separation
 * of the baking units).
 *
 * Two contexts of the same Session and the same capacity decode alternately. If the separation
 * works, three things hold at once: ① zero recipe re-derivations (the context is not part of the
 * key) ② zero backing rebuilds (the Session-owned baking does not depend on the context) ③ zero
 * stale reads (the bind groups that bind state are per context).
 *
 * MUST: change the input salt per context — a KV mix-up raises no exception and **shows only in the
 * values**, so with identical inputs a chain error could cancel itself out.
 */
test.skipIf(!GPU_AVAILABLE)(
  "switching contexts alternately causes zero re-derivations and zero backing rebuilds and never mixes up the KV (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    const session = await stateSession(gpu, FULL);
    const a = await session.createGenerationContext({ chunkLength: 1 });
    const b = await session.createGenerationContext({ chunkLength: 1 });
    const stateA = newOracle(FULL, 8);
    const stateB = newOracle(FULL, 8);
    try {
      // The derivation runs only for the first M=1 run. Everything after that hits the same key.
      await assertStep(session, a, FULL, 8, stateA, 1, 1, 0, "A first");
      assertEquals(
        session.diagnostics().lastRunPrepared?.hit,
        false,
        "the first is a deriving run",
      );

      for (let step = 0; step < 3; step += 1) {
        await assertStep(session, b, FULL, 8, stateB, 1, 1, 301 + step * 7, `B decode ${step}`);
        assertEquals(
          session.diagnostics().lastRunPrepared?.hit,
          true,
          `B decode ${step}: switching the context caused a re-derivation (the context is part of the key)`,
        );
        await assertStep(session, a, FULL, 8, stateA, 1, 1, 401 + step * 7, `A decode ${step}`);
        assertEquals(
          session.diagnostics().lastRunPrepared?.hit,
          true,
          `A decode ${step}: switching the context caused a re-derivation`,
        );
      }

      const diagnostics = session.diagnostics();
      assertEquals(
        diagnostics.planBacking.buildCount,
        1,
        "switching contexts rebuilds the slot backing (the full-rebuild thrashing decision 5 avoided)",
      );
      // Re-baking happens **once per context** only. A bundle is valid for a backing object, so
      // switching does not invalidate it (growing at every switch would open the way to rebuilds
      // proportional to the number of runs).
      assertEquals(
        diagnostics.stateBacking.rebindCount,
        2,
        "re-baking the context-side bind groups does not stay at once per context",
      );
      assertEquals(diagnostics.stateBacking.contextCount, 2);
      assertEquals(a.pastLength, 4);
      assertEquals(b.pastLength, 3);
    } finally {
      for (const context of [a, b]) await context.dispose();
      await session.dispose();
      gpu.destroy();
    }
  },
);

/**
 * Recovery after a backing was **retired and rebuilt** (the generation identifier of ADR 0066
 * decision 5).
 *
 * M=4 (prefill form) and M=1 (decode form) are different keys, and in a Session with budget 0 they
 * compete for one slot backing (the former capacity of 1 — ADR 0095 decision 1). The slot / input
 * buffers of a retired backing are `destroy()`ed in the run's cleanup, so a context side that kept
 * an old bundle would issue **dispatches that bind destroyed buffers** (always broken, either in
 * the values or with an exception). That re-baking follows the backing's rebuilds is pinned by both
 * the correctness of the values and the counts.
 */
test.skipIf(!GPU_AVAILABLE)(
  "with budget 0 every switch rebuilds the backing and the context side re-bakes too (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    const session = await stateSession(gpu, FULL, { planBackingBudgetBytes: 0 });
    const context = await session.createGenerationContext({ chunkLength: 4 });
    const state = newOracle(FULL, 8);
    try {
      // Derive each of the two keys (up to here every run is a miss = no backing used).
      await assertStep(session, context, FULL, 8, state, 4, 1, 0, "M=4 first (derive)");
      await assertStep(session, context, FULL, 8, state, 1, 1, 11, "M=1 first (derive)");
      assertEquals(session.diagnostics().planBacking.buildCount, 0, "a miss run built a backing");

      /** `[backing builds, re-bakes]` when each run settles. */
      const builds: [number, number][] = [];
      for (const [rows, salt] of [
        [1, 23],
        [4, 31],
        [1, 43],
        [4, 53],
      ] as const) {
        await assertStep(session, context, FULL, 8, state, rows, 1, salt, `switch to M=${rows}`);
        assertEquals(
          session.diagnostics().lastRunPrepared?.hit,
          true,
          "the recipe is not re-derived",
        );
        const diagnostics = session.diagnostics();
        builds.push([diagnostics.planBacking.buildCount, diagnostics.stateBacking.rebindCount]);
      }
      // With budget 0 every key change retires → rebuilds, so the context side re-bakes on every
      // run too.
      // MUST: the re-bake count follows the backing build count — if it does not, a bundle that
      // binds a retired backing's buffers is being reused.
      assertEquals(
        builds,
        [
          [1, 1],
          [2, 2],
          [3, 3],
          [4, 4],
        ],
        "the context side's re-bakes do not follow the backing rebuilds",
      );
      assertEquals(
        session.diagnostics().planBacking.retainedCount,
        1,
        "more than one retained with budget 0",
      );
      assertEquals(context.pastLength, 6);
    } finally {
      await context.dispose();
      await session.dispose();
      gpu.destroy();
    }
  },
);

/**
 * Going back and forth between **shapes retained** within the budget adds no re-bakes (ADR 0095
 * decision 3).
 *
 * A bundle's lifetime is still the (context, backing object) pair, with one bundle per retained
 * backing. The same round trips as the budget-0 check above run with the default budget, and the
 * test checks that `rebindCount` stops at one initial bake per shape (re-baking at every
 * prefill ⇄ decode switch within one generation turn is what this phase removed).
 */
test.skipIf(!GPU_AVAILABLE)(
  "round trips between 2 shapes retained within the budget add no re-bakes (default budget, real GPU)",
  async () => {
    const gpu = await acquireGpu();
    const session = await stateSession(gpu, FULL);
    const context = await session.createGenerationContext({ chunkLength: 4 });
    const state = newOracle(FULL, 8);
    try {
      await assertStep(session, context, FULL, 8, state, 4, 1, 0, "M=4 first (derive)");
      await assertStep(session, context, FULL, 8, state, 1, 1, 11, "M=1 first (derive)");

      const builds: [number, number][] = [];
      for (const [rows, salt] of [
        [1, 23],
        [4, 31],
        [1, 43],
        [4, 53],
        [1, 61],
        [4, 71],
      ] as const) {
        await assertStep(session, context, FULL, 8, state, rows, 1, salt, `switch to M=${rows}`);
        assertEquals(
          session.diagnostics().lastRunPrepared?.hit,
          true,
          "the recipe is not re-derived",
        );
        const diagnostics = session.diagnostics();
        builds.push([diagnostics.planBacking.buildCount, diagnostics.stateBacking.rebindCount]);
      }
      // Even after 3 round trips, builds stop at one per shape and re-bakes at one per shape. The
      // correctness of the values is held by each step's comparison (`assertStep`), so "binding
      // another shape's buffers without re-baking" fails on the values even if this is green.
      assertEquals(
        builds,
        [
          [1, 1],
          [2, 2],
          [2, 2],
          [2, 2],
          [2, 2],
          [2, 2],
        ],
        "round trips between retained shapes rebuild the backing / re-bake",
      );
      assertEquals(
        session.diagnostics().planBacking.retainedCount,
        2,
        "the 2 shapes are not retained",
      );
      assertEquals(context.pastLength, 8);
    } finally {
      await context.dispose();
      await session.dispose();
      gpu.destroy();
    }
  },
);

/**
 * The measurements of one backing (region = `residentBytes` / owned input buffers =
 * `inputBytes`).
 */
type Measured = { readonly residentBytes: number; readonly inputBytes: number };

/**
 * The amount the budget accounts for (the same formula as `#evictBackingsFor` — both the region
 * and the owned input buffers). Building it from the region alone would put the real accounting
 * above the budget by the input buffers, and a Session meant to be "exactly at budget" would run
 * over budget from the start (= always retiring everything).
 */
const accountedBytes = (measured: readonly Measured[]): number =>
  measured.reduce((total, one) => total + one.residentBytes + one.inputBytes, 0);

/** The diagnostics when the retained set is `retained` (**every field** — no partial match). */
const expectStats = (retained: readonly Measured[], buildCount: number): PlanBackingStats => ({
  residentBytes: retained.reduce((total, one) => total + one.residentBytes, 0),
  inputBytes: retained.reduce((total, one) => total + one.inputBytes, 0),
  retainedCount: retained.length,
  buildCount,
});

/**
 * Measures the slot backing bytes of one shape (M=`rows`) with **a separate Session that runs only
 * that shape** (the budget checks are built from this measurement for the same reason as the helper
 * of the same name in `gpu_plan_backing_test.ts` — with a constant, "exactly at budget" would
 * silently shift its meaning when the slot table's packing changes).
 */
const stateBackingBytes = async (
  gpu: GpuContext,
  model: StateModel,
  spec: Parameters<Session["createGenerationContext"]>[0],
  rows: number,
): Promise<Measured> => {
  const session = await stateSession(gpu, model);
  const context = await session.createGenerationContext(spec);
  try {
    const inputs = stepInputs(model, rows, 0);
    // 1st run = miss (derive) / 2nd run = hit (build the backing).
    await runStep(session, context, model, inputs, rows, 1);
    await runStep(session, context, model, inputs, rows, 1);
    const stats = session.diagnostics().planBacking;
    assertEquals(
      stats.retainedCount,
      1,
      `a Session that ran only the M=${rows} shape does not retain exactly one`,
    );
    assert(stats.residentBytes > 0, `the M=${rows} backing is 0 bytes (the check would miss)`);
    // q / k / v are non-resident `Tensor`s, so their buffers are owned by the backing = counted in
    // the budget.
    assert(
      stats.inputBytes > 0,
      `the owned input buffers of M=${rows} are 0 bytes (the budget does not account for them)`,
    );
    return { residentBytes: stats.residentBytes, inputBytes: stats.inputBytes };
  } finally {
    await context.dispose();
    await session.dispose();
  }
};

/**
 * The shape used by the budget checks (the capacity leaves room for the logical length of 10
 * round-trip steps).
 */
const BUDGET_MODEL: StateModel = { heads: 4, kvHeads: 4, depth: 4, capacity: 32 };
/**
 * The spec that issues the 3 shapes M ∈ {1, 2, 4} from one context (buckets: the ADR 0066
 * addendum on buckets).
 */
const BUDGET_SPEC = { chunkLength: 4, chunkBuckets: [2] } as const;

/**
 * **Retire → rebuild adds exactly one re-bake** (= the bundle is dropped together with the
 * retirement — the MUST of ADR 0095 decision 3).
 *
 * The budget is narrowed to exactly 2 shapes (M=4 + M=1), and a 3rd shape (M=2) retires just the
 * oldest, M=4. Two things are checked:
 * ① the bundle of the retired backing's generation is **gone** from the context (`bakedGroups` is
 * undefined) — not dropping it would keep groups that bind destroyed buffers alive as long as the
 * context refers to them ② the run that comes back adds one re-bake. The bundles of the shapes
 * still retained remain (the counterpart showing ① is not "drop everything").
 */
test.skipIf(!GPU_AVAILABLE)(
  "the bundle of a retired backing is dropped, and the run that comes back re-bakes exactly once (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    const wide = await stateBackingBytes(gpu, BUDGET_MODEL, BUDGET_SPEC, 4);
    const decode = await stateBackingBytes(gpu, BUDGET_MODEL, BUDGET_SPEC, 1);
    const mid = await stateBackingBytes(gpu, BUDGET_MODEL, BUDGET_SPEC, 2);
    // With budget = M=4 + M=1, the condition for M=2 to always fit after retiring the oldest
    // (M=4) is `M=1 + M=2 ≤ M=4 + M=1` ⟺ `M=2 ≤ M=4`. If it broke, two would be retired and the
    // test would no longer be looking at "retiring the single oldest". What is compared is the
    // amount the budget accounts for (region + owned inputs).
    assert(
      accountedBytes([mid]) <= accountedBytes([wide]),
      `the measurement of M=2 ${JSON.stringify(mid)} is not at most M=4's ${JSON.stringify(
        wide,
      )} (the check would miss)`,
    );

    const session = await stateSession(gpu, BUDGET_MODEL, {
      planBackingBudgetBytes: accountedBytes([wide, decode]),
    });
    const context = await session.createGenerationContext(BUDGET_SPEC);
    const state = newOracle(BUDGET_MODEL, BUDGET_MODEL.capacity as number);
    const capacity = BUDGET_MODEL.capacity as number;
    /**
     * The generation identifiers of the baked bundles (collects, in order, the tokens
     * `#generationGroups` hands over).
     */
    const tokens: number[] = [];
    const inner = internals(context);
    // oxlint-disable-next-line typescript/unbound-method -- internals are arrow-function properties (no `this`)
    const setBaked = inner.setBakedGroups;
    inner.setBakedGroups = (token, groups) => {
      tokens.push(token);
      setBaked(token, groups);
    };
    try {
      // Run M=4 and M=1 twice each so both are retained (the first is a derivation = no backing
      // used).
      for (const [rows, salt] of [
        [4, 0],
        [4, 7],
        [1, 13],
        [1, 19],
      ] as const) {
        await assertStep(
          session,
          context,
          BUDGET_MODEL,
          capacity,
          state,
          rows,
          1,
          salt,
          `M=${rows}`,
        );
      }
      assertEquals(
        session.diagnostics().planBacking,
        expectStats([wide, decode], 2),
        "the 2 shapes exactly at budget are not retained",
      );
      assertEquals(tokens.length, 2, "re-bakes do not stay at one per shape");
      const [wideToken, decodeToken] = tokens;

      // Touching M=1 again adds no re-bake (the bundle is alive). The LRU order stays [M=4, M=1].
      await assertStep(
        session,
        context,
        BUDGET_MODEL,
        capacity,
        state,
        1,
        1,
        23,
        "M=1 touched again",
      );
      assertEquals(
        session.diagnostics().stateBacking.rebindCount,
        2,
        "a retained shape was re-baked",
      );

      // The 3rd shape (the M=2 bucket form). The budget is exact, so only the oldest, M=4, is
      // retired.
      for (const salt of [29, 31]) {
        await assertStep(session, context, BUDGET_MODEL, capacity, state, 2, 1, salt, "M=2");
      }
      assertEquals(
        session.diagnostics().planBacking,
        expectStats([decode, mid], 3),
        "what was retired is not just the oldest (M=4)",
      );
      // The 3rd shape itself is baked once (the retained M=1 is not re-baked).
      assertEquals(
        session.diagnostics().stateBacking.rebindCount,
        3,
        "the 3rd shape was not baked exactly once",
      );
      // ① The bundle of the retired generation is gone from the context (not dropping it would
      // extend the buffers' lifetime).
      assertEquals(
        inner.bakedGroups(wideToken),
        undefined,
        "the bundle of the retired backing remains in the context (groups that bind destroyed buffers survive)",
      );
      assert(
        inner.bakedGroups(decodeToken) !== undefined,
        "the bundles of retained backings are dropped too (back to re-baking on every round trip)",
      );

      // ② The run that comes back adds exactly one rebuild + one re-bake.
      await assertStep(session, context, BUDGET_MODEL, capacity, state, 4, 1, 37, "back to M=4");
      assertEquals(
        session.diagnostics().lastRunPrepared?.hit,
        true,
        "the recipe is not re-derived",
      );
      assertEquals(
        session.diagnostics().planBacking.buildCount,
        4,
        "the retired shape was not rebuilt",
      );
      assertEquals(
        session.diagnostics().stateBacking.rebindCount,
        4,
        "coming back did not re-bake",
      );
      assertEquals(tokens.length, 4);
      assert(tokens[3] !== wideToken, "the rebuilt backing claims the same generation identifier");
    } finally {
      inner.setBakedGroups = setBaked;
      await context.dispose();
      await session.dispose();
      gpu.destroy();
    }
  },
);

/**
 * **Fault injection** — the proof that switching A/B (above) does not miss.
 *
 * Context B's baked bundle is replaced with **A's** = it reproduces an implementation that shares
 * the bind groups containing state across contexts (the shape phase D-3's DECIDED named: "keeps
 * running with the previous context's KV bound"). A / B have the same capacity, so the buffer
 * sizes do not differ by a single byte, and **validation passes while only the values quietly
 * change** — if this comparison did not fail, the parity of switching A/B would protect nothing.
 */
test.skipIf(!GPU_AVAILABLE)(
  "mixing up the context-side bundle makes the output read the previous context's KV (fault injection, real GPU)",
  async () => {
    const gpu = await acquireGpu();
    const session = await stateSession(gpu, FULL);
    const a = await session.createGenerationContext({ chunkLength: 1 });
    const b = await session.createGenerationContext({ chunkLength: 1 });
    const stateA = newOracle(FULL, 8);
    const stateB = newOracle(FULL, 8);
    try {
      // Baking starts from hit runs, so warm A up to backed before intercepting its bundle.
      await assertStep(session, a, FULL, 8, stateA, 1, 1, 0, "A first (derive)");
      const internalsA = internals(a);
      // oxlint-disable-next-line typescript/unbound-method -- internals are arrow-function properties (no `this`)
      const setBaked = internalsA.setBakedGroups;
      let stolen: BakedGroups | undefined;
      internalsA.setBakedGroups = (token, groups) => {
        stolen = groups;
        setBaked(token, groups);
      };
      await assertStep(session, a, FULL, 8, stateA, 1, 1, 13, "A backed");
      assert(
        stolen !== undefined,
        "A's context-side bind groups were not baked (the injection would miss)",
      );

      // Injection: B uses A's bundle (A's KV slots and A's logical-length uniform) instead of its
      // own.
      internals(b).bakedGroups = () => stolen;
      const inputs = stepInputs(FULL, 1, 77);
      const actual = await runStep(session, b, FULL, inputs, 1, 1);
      const expected = advanceOracle(FULL, 8, stateB, inputs, 1, 1);
      const report = compareTensors(
        { dtype: "f32", data: actual },
        { dtype: "f32", data: expected },
        STATE_TOLERANCE,
      );
      assertEquals(
        report.pass,
        false,
        "B's run with A's bundle matched B's oracle (the comparison of switching A/B misses)",
      );
    } finally {
      for (const context of [a, b]) await context.dispose();
      await session.dispose();
      gpu.destroy();
    }
  },
);

/**
 * The check that looks at the option `"parallel"` (①''s D-parallel reduction + ③''s KV-parallel
 * reduction) **down to the values through `Session.run`**.
 *
 * The runtime's checks are currently split between "the band of direct dispatch"
 * (tests/gpu_state_attention_parallel_test.ts) and "the key check through the executor" (the census
 * above — it discards what `runStep` returns), and **what lies between them** (whether the recipe
 * correctly builds the workgroup counts, bindings and S / stats allocations for ①' / ③') is covered
 * only by the asset-backed models e2e (explicitly SKIPped in environments without the mirrors).
 * This closes that gap without assets.
 *
 * MUST: bit identity is not required (the reference path and ①' / ③' reduce in a different order —
 * small shapes just happen to show no difference,
 * and making a match the contract would be a different claim). Two things are checked: "matches
 * the oracle within the band" and "stays in the same band across the options".
 */
test.skipIf(!GPU_AVAILABLE)(
  "stateAttentionReduce:'parallel' matches the reference chain through Session.run too (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    const parallel = await stateSession(gpu, GQA, { stateAttentionReduce: "parallel" });
    const sequential = await stateSession(gpu, GQA);
    const parallelContext = await parallel.createGenerationContext({ chunkLength: 4 });
    const sequentialContext = await sequential.createGenerationContext({ chunkLength: 4 });
    const state = newOracle(GQA, 8);
    try {
      // Prefill (M=4 / Q=4) → decode ×3 (M=1 / Q=1). The salt changes per step
      for (const [chunkRows, query, salt, label] of [
        [4, 4, 2, "prefill"],
        [1, 1, 17, "decode 1"],
        [1, 1, 29, "decode 2"],
        [1, 1, 43, "decode 3"],
      ] as const) {
        const inputs = stepInputs(GQA, chunkRows, salt);
        const actual = await runStep(parallel, parallelContext, GQA, inputs, chunkRows, query);
        const baseline = await runStep(
          sequential,
          sequentialContext,
          GQA,
          inputs,
          chunkRows,
          query,
        );
        const expected = advanceOracle(GQA, 8, state, inputs, chunkRows, query);
        // The anti-tautology check: the expected output is not trivial (if both sides are all 0
        // the comparison sees nothing)
        assert(
          expected.some((value) => Math.abs(value) > 1e-3),
          `${label}: the expected output is trivial (all ~0)`,
        );
        // ① ③' matches the reference chain
        const against = compareTensors(
          { dtype: "f32", data: actual },
          { dtype: "f32", data: expected },
          STATE_TOLERANCE,
        );
        assertEquals(against.pass, true, `${label} parallel: ${formatAllclose(against)}`);
        // ② ③' and ③ stay in the same band (the option's difference does not leak into the values)
        const cross = compareTensors(
          { dtype: "f32", data: actual },
          { dtype: "f32", data: baseline },
          STATE_TOLERANCE,
        );
        assertEquals(cross.pass, true, `${label} parallel vs sequential: ${formatAllclose(cross)}`);
        // ③ The logical length's advance does not depend on the option
        assertEquals(parallelContext.pastLength, state.past, `${label}: pastLength of parallel`);
        assertEquals(
          sequentialContext.pastLength,
          state.past,
          `${label}: pastLength of sequential`,
        );
      }
    } finally {
      await parallelContext.dispose();
      await sequentialContext.dispose();
      await parallel.dispose();
      await sequential.dispose();
      gpu.destroy();
    }
  },
);

/**
 * Shapes on both sides of the fusion's applicability (the condition is `chunkRows <= 8` and
 * `colCap <= 1024`).
 */
const FUSION_CASES = [
  [1, 64, undefined, true],
  [8, 39, 31, true],
  [9, 64, undefined, false],
  [1, 1025, undefined, false],
  [8, 1024, undefined, true],
] as const;

/**
 * The first of the checks that look at the option `"parallel-fused"` (fusing ② and ③') through
 * `Session.run` = the values.
 *
 * NOTE: the numeric match and the key check are split into separate tests (the key check is
 * below). The key check's source is the census (`lastRunPipelines`), which does not depend on
 * timing, so both run on a device without timing.
 */
test.skipIf(!GPU_AVAILABLE)(
  "parallel-fused matches parallel as u32 even after state updates (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    try {
      for (const [chunkRows, capacity, window] of FUSION_CASES) {
        const model: StateModel = { heads: 4, kvHeads: 1, depth: 17, capacity, window };
        const sessions = [
          await stateSession(gpu, model, { stateAttentionReduce: "parallel" }),
          await stateSession(gpu, model, { stateAttentionReduce: "parallel-fused" }),
        ];
        const contexts = await Promise.all(
          sessions.map((s) => s.createGenerationContext({ chunkLength: chunkRows })),
        );
        try {
          for (let step = 0; step < (window ? 8 : 4); step++) {
            const input = stepInputs(model, chunkRows, step * 13);
            const before = await runStep(
              sessions[0],
              contexts[0],
              model,
              input,
              chunkRows,
              chunkRows,
            );
            const after = await runStep(
              sessions[1],
              contexts[1],
              model,
              input,
              chunkRows,
              chunkRows,
            );
            assertEquals(new Uint32Array(after.buffer), new Uint32Array(before.buffer));
          }
          assertEquals(contexts[1].pastLength, contexts[0].pastLength);
        } finally {
          for (const c of contexts) await c.dispose();
          for (const s of sessions) await s.dispose();
        }
      }
    } finally {
      gpu.destroy();
    }
  },
);

/**
 * The second = the keys. Applicable shapes show exactly one fused key; non-applicable shapes show
 * the usual ② / ③'.
 *
 * NOTE: whether the fusion applies is decided when the recipe is built, from `chunkRows` and
 * `colCap` alone, so one step is enough for all keys to appear (the advance of the values is
 * checked by the test above).
 */
test.skipIf(!GPU_AVAILABLE)(
  "parallel-fused merges stats/PV only for applicable shapes (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    try {
      for (const [chunkRows, capacity, window, expectedFusion] of FUSION_CASES) {
        const model: StateModel = { heads: 4, kvHeads: 1, depth: 17, capacity, window };
        const session = await stateSession(gpu, model, {
          stateAttentionReduce: "parallel-fused",
        });
        const context = await session.createGenerationContext({ chunkLength: chunkRows });
        try {
          await runStep(
            session,
            context,
            model,
            stepInputs(model, chunkRows, 13),
            chunkRows,
            chunkRows,
          );
          const entries = requireCensus(
            session.diagnostics().lastRunPipelines,
            `M=${chunkRows} cap=${capacity}`,
          );
          assertEquals(
            countDispatches(entries, SEAT_SIGNATURES.stateAttentionReduce["parallel-fused"]) > 0,
            expectedFusion,
          );
          assertEquals(
            entries.some((e) => e.key.startsWith("attention_state_stats:")),
            !expectedFusion,
          );
          assertEquals(
            entries.some((e) => e.key.startsWith("attention_state_pv:")),
            !expectedFusion,
          );
        } finally {
          await context.dispose();
          await session.dispose();
        }
      }
    } finally {
      gpu.destroy();
    }
  },
);
