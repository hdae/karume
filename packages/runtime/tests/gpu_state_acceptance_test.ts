// **Acceptance tests** for states-form execution (phase D-5 — ADR 0066 acceptance criterion ② /
// ADR 0067 acceptance criterion ⑤ and others).
//
// The checks of gpu_state_execution_test.ts (execution wiring) and gpu_state_attention_test.ts
// (kernels alone) both use **the CPU reference built in D-2** (src/reference/state-attention.ts)
// as the oracle. If the reference and the kernels shared the same error, all of them would be
// tautologies, so this file places five groups that close that hole:
//
//   1. **Cross oracle** — builds the same computation with a mechanism trusted since before phase A
//      (one-shot fused attention with the mask as the fourth input) and compares it with the
//      multi-step states form (a verification chain with no dependence on the reference)
//   2. **ADR 0066 acceptance criterion ②** — with and without pad, **the KV slot bytes match
//      exactly** + the outputs match
//   3. **ADR 0067 acceptance criterion ⑤** — empty row → exactly 0 through `Session.run` (a direct
//      check through the executor)
//   4. **Capacity independence** — C=8 and C=4096 are bit-identical at every step (the value-side
//      flip of decision 3's work acceptance condition)
//   5. **Execution** of **KV-sharing layers** (ADR 0067 decision 4 — the 20-layer pattern of
//      Gemma 4 E2B)
//
// MUST: each group carries its own evidence that "the check fires" (non-trivial expected values,
// controls, fault injection). The states form legitimately produces exact zeros, so without that
// evidence an implementation that "computes nothing" would pass green.

import { test } from "vite-plus/test";
import { assert, assertEquals } from "@std/assert";
import { compareTensors, formatAllclose, type Tolerance } from "../src/reference/allclose.ts";
import { referenceStateAppend, referenceStateAttention } from "../src/reference/state-attention.ts";
import { acquireGpu, type GpuContext, RUNTIME_INTERNAL } from "../src/gpu/device.ts";
import { createSessionFromContainer, type Session, type Tensor } from "../src/runtime/executor.ts";
import type { GenerationContext } from "../src/runtime/generation-context.ts";
import {
  type DeclarationJson,
  GRAPH_NAME,
  openGraphModel,
  singleOpDeclaration,
} from "./helpers/model-fixture.ts";
import { halfScale, seeded } from "./helpers/state-dispatch.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";

/**
 * The return type of {@link compareTensors} (allclose.ts does not export the type, so take it from
 * the value).
 */
type Report = ReturnType<typeof compareTensors>;

/** Deterministic input sequences (no random numbers — a failure would not reproduce). */
const QUERY = (i: number): number => (((i * 7) % 23) - 11) * 0.17;
const KEY = (i: number): number => (((i * 11) % 19) - 9) * 0.23;
const VALUE = (i: number): number => (((i * 5) % 17) - 8) * 0.31;

/** One model's shape (`B = 1` fixed — the batch axis is out of scope per ADR 0066 decision 8). */
type StateModel = {
  readonly heads: number;
  readonly kvHeads: number;
  readonly depth: number;
  /**
   * The slot's row capacity. With the symbol `"C"`, `createGenerationContext(bindings)` decides it.
   */
  readonly capacity: number | "C";
  /** `W` (omitted = the whole context). */
  readonly window?: number;
};

/**
 * The states-form graph (`readers` readers + two `state_append`).
 *
 * The node order follows the issue rule of ADR 0067 decision 5b: "**every reader of the slot →
 * the write**". `readers = 2` is the shape of a KV-sharing layer (the owning layer + a layer with
 * no KV of its own); the sharing layer computes no projection of its own and **wires the owning
 * layer's k/v values straight into ins** (decision 4) — which is exactly two attention nodes
 * sharing this graph's `k` / `v`, and the sharing shows up as "the same slot name + the same value
 * name".
 */
const stateGraph = (model: StateModel, readers = 1): DeclarationJson => {
  const { heads, kvHeads, depth, capacity, window } = model;
  const windowAttrs = window === undefined ? {} : { window };
  const names = Array.from({ length: readers }, (_, index) => index);
  return {
    format: "karume-ir",
    version: 2,
    requires: { ops: ["attention", "state_append"] },
    symbols: capacity === "C" ? ["M", "C"] : ["M"],
    inputs: [
      ...names.map((index) => ({
        name: `q${index}`,
        dtype: "f32",
        shape: [1, heads, "M", depth] as (number | string)[],
      })),
      { name: "k", dtype: "f32", shape: [1, kvHeads, "M", depth] },
      { name: "v", dtype: "f32", shape: [1, kvHeads, "M", depth] },
    ],
    outputs: names.map((index) => `o${index}`),
    initializers: {},
    values: Object.fromEntries(
      names.map((index) => [`o${index}`, { dtype: "f32", shape: [1, heads, "M", depth] }]),
    ),
    states: {
      kslot: { dtype: "f32", shape: [1, kvHeads, capacity, depth] },
      vslot: { dtype: "f32", shape: [1, kvHeads, capacity, depth] },
    },
    nodes: [
      ...names.map((index) => ({
        op: "attention",
        ins: [`q${index}`, "k", "v"],
        outs: [`o${index}`],
        attrs: { scale: halfScale(depth), ...windowAttrs },
        states: { k: "kslot", v: "vslot" },
      })),
      {
        op: "state_append",
        ins: ["k"],
        outs: [],
        attrs: { ...windowAttrs },
        states: { slot: "kslot" },
      },
      {
        op: "state_append",
        ins: ["v"],
        outs: [],
        attrs: { ...windowAttrs },
        states: { slot: "vslot" },
      },
    ],
  };
};

const stateSession = async (gpu: GpuContext, model: StateModel, readers = 1): Promise<Session> =>
  await createSessionFromContainer(
    gpu,
    await openGraphModel(stateGraph(model, readers)),
    GRAPH_NAME,
  );

const tensor = (shape: readonly number[], data: Float32Array<ArrayBuffer>): Tensor => ({
  dtype: "f32",
  shape,
  data,
});

/** The inputs of one step (`M` rows — only the first `queryLength` rows are valid). */
type StepInputs = {
  /** One q per reader (KV-sharing layers each have their own q). */
  readonly q: readonly Float32Array<ArrayBuffer>[];
  readonly k: Float32Array<ArrayBuffer>;
  readonly v: Float32Array<ArrayBuffer>;
};

/**
 * Per-step inputs. Changing `salt` per step makes an implementation that "re-reads the previous
 * step's input" fail on the values (handing out the same input could let an error in the chain
 * cancel itself out). The salt is also shifted per reader (if the two readers had the same q, a
 * wiring mistake in the sharing layer would not show in the values).
 */
const stepInputs = (
  model: StateModel,
  chunkRows: number,
  salt: number,
  readers = 1,
): StepInputs => ({
  q: Array.from({ length: readers }, (_, index) =>
    seeded(model.heads * chunkRows * model.depth, (i) => QUERY(i + salt + index * 97)),
  ),
  k: seeded(model.kvHeads * chunkRows * model.depth, (i) => KEY(i + salt)),
  v: seeded(model.kvHeads * chunkRows * model.depth, (i) => VALUE(i + salt)),
});

/**
 * One step on the GPU side (the third argument of `run` is the generation surface). Returns one
 * `[1,H,M,D]` per reader.
 */
const runStep = async (
  session: Session,
  context: GenerationContext,
  model: StateModel,
  inputs: StepInputs,
  chunkRows: number,
  query: number,
): Promise<readonly Float32Array<ArrayBuffer>[]> => {
  const tensors: Record<string, Tensor> = {
    k: tensor([1, model.kvHeads, chunkRows, model.depth], inputs.k),
    v: tensor([1, model.kvHeads, chunkRows, model.depth], inputs.v),
  };
  inputs.q.forEach((data, index) => {
    tensors[`q${index}`] = tensor([1, model.heads, chunkRows, model.depth], data);
  });
  const outputs = await session.run(tensors, {}, { context, queryLength: query });
  return inputs.q.map((_, index) => {
    const out = outputs[`o${index}`];
    assertEquals(out.dtype, "f32");
    return out.data as Float32Array<ArrayBuffer>;
  });
};

/**
 * Extracts the first `rows` rows of `[1,H,M,D]` (the valid rows = the rows for the logical
 * coordinates `[P, P+Q)`).
 */
const takeRows = (
  data: Float32Array<ArrayBuffer>,
  planes: number,
  chunkRows: number,
  depth: number,
  rows: number,
): Float32Array<ArrayBuffer> => {
  const out = new Float32Array(planes * rows * depth);
  for (let plane = 0; plane < planes; plane += 1) {
    for (let row = 0; row < rows; row += 1) {
      for (let d = 0; d < depth; d += 1) {
        out[(plane * rows + row) * depth + d] = data[(plane * chunkRows + row) * depth + d];
      }
    }
  }
  return out;
};

const bitsOf = (data: Float32Array<ArrayBuffer>): Uint32Array =>
  new Uint32Array(data.buffer, data.byteOffset, data.length);

/** Bit-pattern mismatches (the first four only). */
const mismatches = (actual: ArrayLike<number>, expected: ArrayLike<number>): readonly string[] => {
  const found: string[] = [];
  for (let i = 0; i < actual.length && found.length < 4; i += 1) {
    if (actual[i] !== expected[i]) found.push(`[${i}] ${actual[i]} vs ${expected[i]}`);
  }
  return found;
};

const assertBitsEqual = (
  actual: ArrayLike<number>,
  expected: ArrayLike<number>,
  where: string,
): void => {
  assertEquals(actual.length, expected.length, `${where}: the lengths differ`);
  assertEquals(mismatches(actual, expected), [], `${where}: the bit patterns differ`);
};

/**
 * Reads back the **raw bytes** of a state slot (what ADR 0066 acceptance criterion ② compares).
 *
 * The execution path never puts the slot contents into an output, so the only observation point is
 * peeking at the `RUNTIME_INTERNAL` buffer directly with copyBufferToBuffer + mapAsync (the same
 * approach as `readFloats` in helpers/state-dispatch.ts). It returns a **`Uint32Array`** — "the
 * bytes match exactly" cannot be expressed by f32 value comparison (it collapses `-0` and `+0` and
 * NaN payloads).
 *
 * MUST: call it after `run` resolves (`Session.run` settles by reading back the outputs, so by then
 * every dispatch of that run has been submitted).
 */
const readSlotBits = async (
  gpu: GpuContext,
  context: GenerationContext,
  slot: string,
): Promise<Uint32Array> => {
  const backing = context[RUNTIME_INTERNAL].slots.get(slot);
  if (backing === undefined) throw new Error(`no state slot '${slot}'`);
  const staging = gpu.device.createBuffer({
    size: backing.byteLength,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  try {
    const encoder = gpu.device.createCommandEncoder();
    encoder.copyBufferToBuffer(backing.buffer, 0, staging, 0, backing.byteLength);
    gpu.device.queue.submit([encoder.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const copy = staging.getMappedRange().slice(0);
    staging.unmap();
    return new Uint32Array(copy);
  } finally {
    staging.destroy();
  }
};

// ---------------------------------------------------------------------------
// Group 1: the cross oracle (multi-step comparison with one-shot fused attention with a band mask)
// ---------------------------------------------------------------------------

/**
 * The "blocked" value of the band mask.
 *
 * MUST: a **finite** large negative value (no `-inf`). The fused attention's mask is additive, not
 * safe_softmax (ADR 0044 decision 3 / the ATTENTION_OP contract in names.ts), so with `-inf`,
 * depending on the implementation's branches, a single "row whose max is also `-inf`" could fall
 * into `exp(-inf − (-inf)) = NaN`. With a finite value both addition and subtraction stay within
 * f32's finite range, and `exp(-3e38 − max)` is 36 orders of magnitude below f32's exponent floor
 * (≈ −88), so it collapses to **exactly 0**. `-3e38` is inside f32's largest finite value
 * ≈ 3.4028e38, so `S + mask` cannot overflow to `-inf` either.
 *
 * NOTE: every row always contains the causal diagonal (`col = P + row`), so an "all −inf row"
 * cannot occur structurally — this mask does not break the fused attention's contract (an all −inf
 * row violates it).
 */
const MASK_BLOCK = -3e38;

/**
 * The band mask `[1,1,Q,Ntot]` in logical coordinates (the **two-sided** causal + sliding predicate
 * — ADR 0067 decision 4).
 *
 * Row `i` is logical position `P + i`. Allowed: `max(0, P+i−W+1) ≤ col ≤ P+i` (`W = 0` has no
 * lower bound).
 */
const bandMask = (
  past: number,
  query: number,
  total: number,
  window: number,
): Float32Array<ArrayBuffer> => {
  const mask = new Float32Array(query * total);
  for (let row = 0; row < query; row += 1) {
    const limit = past + row;
    const lower = window > 0 ? Math.max(0, limit - window + 1) : 0;
    for (let col = 0; col < total; col += 1) {
      mask[row * total + col] = col >= lower && col <= limit ? 0 : MASK_BLOCK;
    }
  }
  return mask;
};

/**
 * The logical KV history (1 row = `[Hkv·D]`). Kept in **a container separate** from the states
 * side's slots — a plain append-only list with neither capacity nor ring, so an error in the
 * mapping (`col % W`) shows in the cross comparison.
 */
type History = Float32Array[];

const appendHistory = (
  history: History,
  data: Float32Array<ArrayBuffer>,
  model: StateModel,
  chunkRows: number,
  query: number,
): void => {
  const { kvHeads, depth } = model;
  for (let row = 0; row < query; row += 1) {
    const logical = new Float32Array(kvHeads * depth);
    for (let kv = 0; kv < kvHeads; kv += 1) {
      for (let d = 0; d < depth; d += 1) {
        logical[kv * depth + d] = data[(kv * chunkRows + row) * depth + d];
      }
    }
    history.push(logical);
  }
};

/**
 * **Materializes** the history into `[1,H,N,D]` (repeat_kv — the one-shot side of a GQA series is
 * built without divisible broadcasting. The same approach as the acceptance criterion of ADR 0067
 * decision 2, so the one-shot side does not depend on the changes of phase A).
 */
const materializeHistory = (history: History, model: StateModel): Float32Array<ArrayBuffer> => {
  const { heads, kvHeads, depth } = model;
  const repeat = heads / kvHeads;
  const total = history.length;
  const out = new Float32Array(heads * total * depth);
  for (let head = 0; head < heads; head += 1) {
    const kv = Math.floor(head / repeat);
    for (let n = 0; n < total; n += 1) {
      for (let d = 0; d < depth; d += 1) {
        out[(head * total + n) * depth + d] = history[n][kv * depth + d];
      }
    }
  }
  return out;
};

/** The one-shot side's Session (`M` / `N` are symbols, so one Session covers every step). */
const oneShotSession = async (gpu: GpuContext, model: StateModel): Promise<Session> => {
  const { heads, depth } = model;
  const graph = singleOpDeclaration(
    "attention",
    [
      [1, heads, "M", depth],
      [1, heads, "N", depth],
      [1, heads, "N", depth],
      [1, 1, "M", "N"],
    ],
    [[1, heads, "M", depth]],
    { attrs: { scale: halfScale(depth) }, symbols: ["M", "N"] },
  );
  return await createSessionFromContainer(gpu, await openGraphModel(graph), GRAPH_NAME);
};

/**
 * The tolerance of the cross oracle.
 *
 * The two paths are **kernels of different families** (the states form is the three kernels
 * state_qk / state_stats / state_pv — the census separately pins that "no fused attention key
 * appears" / the one-shot side is fused attention), and they differ in how the reduction is split
 * and where they read from (the past in ring physical order + current from ins / the concatenated
 * logical order), so in general nothing guarantees the same rounding sequence. Hence allclose.
 *
 * MUST: `rtol = 0` (the states form legitimately contains elements whose correct value is exactly
 * 0, and a relative term does nothing near 0).
 * MUST: this width must always detect "one out-of-window key mixed in" — demonstrated by fault
 * injection (the graph with window +1). The measured effect is 0.054–2.12, 1.8e4–7e5 times this
 * width.
 *
 * NOTE: the width itself is **only headroom**. On the reference machine every step of all four
 * series (20 steps) measured `maxAbs = 0` (= the two paths are bit-identical); no rounding
 * difference has ever been observed. The initial assumption that "a different reduction order must
 * produce error" was refuted by measurement — the contribution of blocked columns is exactly 0
 * (both `exp(-3e38 − max)` and `0 · v`), and both families agree on the order of the valid columns
 * and on the reduction order along D.
 * MUST: do **not bake** that "bit identity" **into a check** (phase D-7 removed the
 * `worstAbs <= 0` assert and only prints the measured worst value to stdout). How fma is used
 * depends on the driver, and other machines legitimately show rounding differences — making a
 * hardware-dependent bit behavior a portable check would turn a correct implementation red on that
 * machine only (the same lesson as the 2026-08-16 ruling that restricted the byte comparison of the
 * golden io to the reference environment). The verdict rests solely on {@link CROSS_TOLERANCE} +
 * the detection margin of the fault injection.
 */
const CROSS_TOLERANCE: Tolerance = { atol: 3e-6, rtol: 0 };

type CrossSeries = {
  readonly name: string;
  readonly model: StateModel;
  readonly chunkLength: number;
  /** The list of `[M (physical chunk rows), Q (queryLength)]`. */
  readonly steps: readonly (readonly [number, number])[];
};

const CROSS_SERIES: readonly CrossSeries[] = [
  {
    // Logical length 0 → 4 → 5 → 6 → 7 → 9 (the last is a chunk with pad)
    name: "full r=1",
    model: { heads: 4, kvHeads: 4, depth: 4, capacity: 16 },
    chunkLength: 4,
    steps: [
      [4, 4],
      [1, 1],
      [1, 1],
      [1, 1],
      [4, 2],
    ],
  },
  {
    // GQA starting from a prefill with pad (0 → 3 → 4 → 5 → 9)
    name: "full r=2 (GQA)",
    model: { heads: 4, kvHeads: 2, depth: 8, capacity: 16 },
    chunkLength: 4,
    steps: [
      [4, 3],
      [1, 1],
      [1, 1],
      [4, 4],
    ],
  },
  {
    // 0 → 3 → 4 → 5 → 6 → 7 → 9. The W=4 ring wraps at past=4 and evicts at every step after that
    name: "sliding W=4 r=1 (eviction across W)",
    model: { heads: 2, kvHeads: 2, depth: 4, capacity: 4, window: 4 },
    chunkLength: 3,
    steps: [
      [3, 3],
      [1, 1],
      [1, 1],
      [1, 1],
      [1, 1],
      [3, 2],
    ],
  },
  {
    // 0 → 4 → 5 → 6 → 7 → 10. Eviction × GQA × a last chunk with pad
    name: "sliding W=6 r=2 (GQA × eviction)",
    model: { heads: 4, kvHeads: 2, depth: 4, capacity: 6, window: 6 },
    chunkLength: 4,
    steps: [
      [4, 4],
      [1, 1],
      [1, 1],
      [1, 1],
      [4, 3],
    ],
  },
];

/**
 * Runs one series and returns the comparison report of each step.
 *
 * `windowDelta` is the **fault-injection** hook — it rebuilds the graph with a wider window on the
 * states side only (the one-shot side's expected values always use the real window). An
 * implementation with a window one wider is exactly the shape that "silently mixes in one
 * out-of-window key" (the reason ADR 0067 decision 4 made the two-sided predicate a MUST).
 */
const crossSeriesReports = async (
  gpu: GpuContext,
  series: CrossSeries,
  windowDelta = 0,
): Promise<readonly Report[]> => {
  const { model, steps } = series;
  const capacity = model.capacity;
  if (windowDelta !== 0 && (model.window === undefined || typeof capacity !== "number")) {
    throw new Error(
      "fault injection (widening the window) applies only to a sliding series with a concrete capacity",
    );
  }
  const stateModel: StateModel =
    windowDelta === 0
      ? model
      : {
          ...model,
          window: (model.window ?? 0) + windowDelta,
          capacity: (capacity as number) + windowDelta,
        };
  const states = await stateSession(gpu, stateModel);
  const oneShot = await oneShotSession(gpu, model);
  const context = await states.createGenerationContext({ chunkLength: series.chunkLength });
  const kHistory: History = [];
  const vHistory: History = [];
  const reports: Report[] = [];
  let past = 0;
  try {
    for (const [index, [chunkRows, query]] of steps.entries()) {
      const inputs = stepInputs(model, chunkRows, 7 + index * 29);
      const actual = (await runStep(states, context, model, inputs, chunkRows, query))[0];
      appendHistory(kHistory, inputs.k, model, chunkRows, query);
      appendHistory(vHistory, inputs.v, model, chunkRows, query);
      const total = past + query;
      const expected = (
        await oneShot.run({
          x0: tensor(
            [1, model.heads, query, model.depth],
            takeRows(inputs.q[0], model.heads, chunkRows, model.depth, query),
          ),
          x1: tensor([1, model.heads, total, model.depth], materializeHistory(kHistory, model)),
          x2: tensor([1, model.heads, total, model.depth], materializeHistory(vHistory, model)),
          x3: tensor([1, 1, query, total], bandMask(past, query, total, model.window ?? 0)),
        })
      )["y"].data as Float32Array<ArrayBuffer>;
      // MUST: the expected values are not trivial (if both sides were ~0 the comparison would be a
      // tautology).
      assert(
        expected.some((value) => Math.abs(value) > 1e-3),
        `${series.name} step ${index}: the one-shot side's expected output is trivial (all ~0)`,
      );
      reports.push(
        compareTensors(
          {
            dtype: "f32",
            data: takeRows(actual, model.heads, chunkRows, model.depth, query),
          },
          { dtype: "f32", data: expected },
          CROSS_TOLERANCE,
        ),
      );
      past += query;
    }
    assertEquals(
      context.pastLength,
      past,
      `${series.name}: the logical length drifted from the host-side progress`,
    );
  } finally {
    await context.dispose();
    await states.dispose();
    await oneShot.dispose();
  }
  return reports;
};

test.skipIf(!GPU_AVAILABLE)(
  "multi-step states form matches one-shot fused attention with a band mask (cross oracle, real GPU)",
  async () => {
    const gpu = await acquireGpu();
    let worstAbs = 0;
    try {
      for (const series of CROSS_SERIES) {
        const reports = await crossSeriesReports(gpu, series);
        reports.forEach((report, index) => {
          assertEquals(
            report.pass,
            true,
            `${series.name} step ${index}: ${formatAllclose(report)}`,
          );
          worstAbs = Math.max(worstAbs, report.maxAbsError);
        });
      }
    } finally {
      gpu.destroy();
    }
    // The measured worst value is **only printed as a diagnostic** (the verdict is on the
    // CROSS_TOLERANCE side — the doc above). The reference machine prints 0 (both families
    // bit-identical), but fma use depends on the driver, so a non-zero value elsewhere is normal.
    console.log(`[states cross-oracle] maxAbs=${worstAbs}`);
  },
);

test.skipIf(!GPU_AVAILABLE)(
  "a states form with the window widened by one does not match the cross oracle (fault injection, real GPU)",
  async () => {
    const gpu = await acquireGpu();
    try {
      // Rebuild the W=4 series with W=5 (capacity 5 too). The logical length progresses
      // 0 → 3 → 4 → 5 → 6 → 7, so only **step 2 onwards** (the steps where `past + row ≥ W`) take
      // the shape that "sees one extra resident out-of-window key" and deviate from the correct
      // band-mask expectation.
      const reports = await crossSeriesReports(gpu, CROSS_SERIES[2], 1);
      const failing = reports.map((report) => report.pass);
      assertEquals(
        failing,
        [true, true, false, false, false, false],
        `some step does not detect the window +1 implementation (the comparison never fires): ${reports
          .map(formatAllclose)
          .join(" / ")}`,
      );
      // Detection headroom: the effect of one mixed-in key is at least three orders of magnitude
      // above the tolerance (measured 0.054–2.12 — it survives loosening the width by two orders).
      // The first two steps staying green is evidence that "the injection does not turn everything
      // red".
      for (const [index, report] of reports.slice(2).entries()) {
        assert(
          report.maxAbsError > CROSS_TOLERANCE.atol * 1000,
          `step ${
            index + 2
          }: the effect of the mixed-in out-of-window key ${report.maxAbsError} is too close to the tolerance ${CROSS_TOLERANCE.atol}`,
        );
      }
    } finally {
      gpu.destroy();
    }
  },
);

// ---------------------------------------------------------------------------
// Group 2: ADR 0066 acceptance criterion ② — mechanical proof that padding rows are a no-op
// ---------------------------------------------------------------------------

const PAD_CAPACITY = 8;
const PAD_MODEL: StateModel = { heads: 2, kvHeads: 2, depth: 4, capacity: PAD_CAPACITY };
/** The physical chunk row count with pad (`M = 8` — only the first Q rows are valid). */
const PAD_WIDE_ROWS = 8;
/** The physical chunk row count of the prefill shape on the side without pad (`M = Q = 3`). */
const PAD_TIGHT_ROWS = 3;

/**
 * Builds `M` rows of input (the first `query` rows are "the logical content", the rest is pad).
 *
 * With `padFill` `undefined`, pad rows are **zero-filled** (the host-side value contract of
 * ADR 0066 addendum 6). Passing a value puts non-zero garbage into the pad rows — for directly
 * proving that the slot does not read pad rows.
 */
const paddedInputs = (
  model: StateModel,
  chunkRows: number,
  query: number,
  salt: number,
  padFill?: (index: number) => number,
): StepInputs => {
  const build = (planes: number, generator: (i: number) => number): Float32Array<ArrayBuffer> =>
    seeded(planes * chunkRows * model.depth, (i) => {
      const row = Math.floor(i / model.depth) % chunkRows;
      if (row < query) {
        return generator(
          (Math.floor(i / (chunkRows * model.depth)) * query + row) * model.depth +
            (i % model.depth) +
            salt,
        );
      }
      return padFill === undefined ? 0 : padFill(i);
    });
  return {
    q: [build(model.heads, QUERY)],
    k: build(model.kvHeads, KEY),
    v: build(model.kvHeads, VALUE),
  };
};

/** The same logical content without pad (`M = query`). */
const tightInputs = (model: StateModel, query: number, salt: number): StepInputs =>
  paddedInputs(model, query, query, salt);

test.skipIf(!GPU_AVAILABLE)(
  "pad rows appear neither in the KV slot bytes nor in the output (ADR 0066 acceptance criterion ②, real GPU)",
  async () => {
    const gpu = await acquireGpu();
    const session = await stateSession(gpu, PAD_MODEL);
    // Run three contexts on the same Session. With or without pad is a difference in the physical
    // chunk row count M, so under the fixed-chunk contract (`M ∈ {chunkLength, 1}` — ADR 0066
    // decision 4) **chunkLength differs too** (`tight` is the prefill shape M=Q=3 = chunkLength 3).
    // The claim (identical KV bytes) compares contexts on the same Session, so it holds as is.
    const [wide, tight, garbage] = await Promise.all([
      session.createGenerationContext({ chunkLength: PAD_WIDE_ROWS }),
      session.createGenerationContext({ chunkLength: PAD_TIGHT_ROWS }),
      session.createGenerationContext({ chunkLength: PAD_WIDE_ROWS }),
    ]);
    try {
      // Run two steps (the second has a past — the same two points hold after the logical length
      // advances). The first is the prefill shape (tight is M=Q=3) and the second the decode shape
      // (tight is M=Q=1), so this also checks that the same two points hold in both execution
      // shapes.
      for (const [index, query] of [PAD_TIGHT_ROWS, 1].entries()) {
        const salt = 11 + index * 37;
        const wideOut = (
          await runStep(
            session,
            wide,
            PAD_MODEL,
            paddedInputs(PAD_MODEL, PAD_WIDE_ROWS, query, salt),
            PAD_WIDE_ROWS,
            query,
          )
        )[0];
        const tightOut = (
          await runStep(
            session,
            tight,
            PAD_MODEL,
            tightInputs(PAD_MODEL, query, salt),
            query,
            query,
          )
        )[0];
        // NOTE: zero-filling pad rows is **a host-side value contract** (ADR 0066 addendum 6 — it
        // blocks, by value, the NaN that non-finite k/v in pad rows would leak into valid rows
        // through the "add −inf + exp" path). Pad rows are never written into the KV slots in the
        // first place, so this contract has no effect on KV — the `garbage` context below shows
        // it (filling pad rows with non-zero values does not move a single bit of the KV bytes).
        const garbageOut = (
          await runStep(
            session,
            garbage,
            PAD_MODEL,
            paddedInputs(
              PAD_MODEL,
              PAD_WIDE_ROWS,
              query,
              salt,
              (i) => (((i * 13) % 29) - 14) * 0.41,
            ),
            PAD_WIDE_ROWS,
            query,
          )
        )[0];

        const label = `step ${index}（Q=${query}）`;
        // ① the KV slot bytes match exactly (regardless of pad and of its contents)
        for (const slot of ["kslot", "vslot"]) {
          const expected = await readSlotBits(gpu, tight, slot);
          assertBitsEqual(await readSlotBits(gpu, wide, slot), expected, `${label}: ${slot} / pad`);
          assertBitsEqual(
            await readSlotBits(gpu, garbage, slot),
            expected,
            `${label}: ${slot} / garbage in pad`,
          );
          // MUST: the comparison fires — the written rows are non-zero, and from `pastLength` on
          // they stay 0 (`state_append` writes only the first queryLength rows = a direct
          // restatement of acceptance criterion ②. It is a full slot, so logical col = physical row
          // and the boundary shows directly at `pastLength`).
          const past = tight.pastLength;
          const rowFloats = PAD_MODEL.depth;
          for (let plane = 0; plane < PAD_MODEL.kvHeads; plane += 1) {
            const base = plane * PAD_CAPACITY * rowFloats;
            const written = expected.subarray(base, base + past * rowFloats);
            const beyond = expected.subarray(
              base + past * rowFloats,
              base + PAD_CAPACITY * rowFloats,
            );
            assert(
              written.some((bits) => bits !== 0),
              `${label}: ${slot} plane ${plane} has an all-0 written range (never fires)`,
            );
            assertEquals(
              beyond.every((bits) => bits === 0),
              true,
              `${label}: ${slot} plane ${plane} has writes beyond the logical length (rows ${past}..${
                PAD_CAPACITY - 1
              })`,
            );
          }
        }
        // ② the valid output rows [0,Q) match exactly in Uint32
        const valid = (data: Float32Array<ArrayBuffer>, rows: number): Uint32Array =>
          bitsOf(takeRows(data, PAD_MODEL.heads, rows, PAD_MODEL.depth, query));
        const tightBits = valid(tightOut, query);
        assert(
          new Set(tightBits).size > 1,
          `${label}: the output is constant (bit equality is a tautology)`,
        );
        assertBitsEqual(valid(wideOut, PAD_WIDE_ROWS), tightBits, `${label}: output / pad`);
        // Row locality (addendum 6) — garbage in pad rows leaks not a single bit into valid rows.
        assertBitsEqual(
          valid(garbageOut, PAD_WIDE_ROWS),
          tightBits,
          `${label}: output / garbage in pad`,
        );
        assertEquals(
          wide.pastLength,
          tight.pastLength,
          `${label}: the logical length differs with and without pad`,
        );
      }
      assertEquals(tight.pastLength, PAD_TIGHT_ROWS + 1);
    } finally {
      for (const context of [wide, tight, garbage]) await context.dispose();
      await session.dispose();
      gpu.destroy();
    }
  },
);

// ---------------------------------------------------------------------------
// Group 3: ADR 0067 acceptance criterion ⑤ — the direct check of empty row → 0 (through the
// executor)
// ---------------------------------------------------------------------------

const EMPTY_MODEL: StateModel = { heads: 2, kvHeads: 2, depth: 4, capacity: 2, window: 2 };

/**
 * Whether row `row` is **empty** (no existing col satisfies the predicate).
 *
 * MUST: write it in logical coordinates only, without the kernel-side helpers (`stateColumnBase` /
 * `stateLiveColumns`) — even if the implementation gets its column range wrong, this verdict does
 * not follow the error. The existing cols are `[0, P+Q)` (past ∪ this step), and the allowed ones
 * are `max(0, P+row−W+1) ≤ col ≤ P+row`.
 */
const isEmptyRow = (past: number, query: number, window: number, row: number): boolean => {
  const limit = past + row;
  const lower = window > 0 ? Math.max(0, limit - window + 1) : 0;
  return Math.min(limit, past + query - 1) < lower;
};

test.skipIf(!GPU_AVAILABLE)(
  "output rows whose pad rows dropped out of the window are exactly 0 (ADR 0067 acceptance criterion ⑤, through Session.run, real GPU)",
  async () => {
    const gpu = await acquireGpu();
    const session = await stateSession(gpu, EMPTY_MODEL);
    const context = await session.createGenerationContext({ chunkLength: 4 });
    const { heads, depth, window } = EMPTY_MODEL;
    const chunkRows = 4;
    try {
      // Run an M=4 chunk with Q=1 against W=2. For rows 2 / 3, every existing col within the
      // causal upper bound falls outside the window (`col + W ≤ P + row`), so they are
      // legitimately empty.
      for (const [index, past] of [0, 1, 2].entries()) {
        assertEquals(context.pastLength, past, `step ${index}: the assumed logical length drifted`);
        const out = (
          await runStep(
            session,
            context,
            EMPTY_MODEL,
            stepInputs(EMPTY_MODEL, chunkRows, 5 + index * 19),
            chunkRows,
            1,
          )
        )[0];
        const empties = Array.from({ length: chunkRows }, (_, row) => row).filter((row) =>
          isEmptyRow(past, 1, window ?? 0, row),
        );
        // MUST: empty rows exist (with none in this shape, the whole check would never fire).
        assertEquals(empties, [2, 3], `step ${index}: the assumed empty rows no longer hold`);
        // MUST: empty rows are a **subset** of the pad rows (`row ≥ Q`) (Q=1, so pad is 1,2,3).
        // Since phase D-7, "pad row → exactly 0" structurally includes "empty row → exactly 0", so
        // also check that the inclusion still holds (a valid row is never empty thanks to the
        // causal self-reference).
        assertEquals(
          empties.filter((row) => row < 1),
          [],
          `step ${index}: an empty row appeared among the valid rows`,
        );
        for (let head = 0; head < heads; head += 1) {
          // Pad rows (including empty rows) are exactly 0 (ADR 0067 acceptance criterion ⑤ / the
          // value contract of ADR 0066 addendum 6).
          for (let row = 1; row < chunkRows; row += 1) {
            for (let d = 0; d < depth; d += 1) {
              const value = out[(head * chunkRows + row) * depth + d];
              assertEquals(
                Object.is(value, 0),
                true,
                `step ${index}: pad row (head ${head}, row ${row}, d ${d}) is not exactly 0 (${value})`,
              );
            }
          }
          // Control: **valid rows** are non-zero (evidence that "an implementation writing all
          // zeros" does not pass this check).
          const slice = out.subarray(head * chunkRows * depth, (head * chunkRows + 1) * depth);
          assert(
            slice.some((value) => Math.abs(value) > 1e-3),
            `step ${index}: even the valid row (head ${head}, row 0) is ~0`,
          );
        }
        // Not a single NaN in the whole output (direct evidence that empty rows go through the
        // "zero guard" and do not fall into `exp(-inf − (-inf))` — ADR 0067 decision 6).
        assertEquals(
          out.some((value) => Number.isNaN(value)),
          false,
          `step ${index}: NaN in the output`,
        );
      }
    } finally {
      await context.dispose();
      await session.dispose();
      gpu.destroy();
    }
  },
);

// ---------------------------------------------------------------------------
// Group 4: capacity independence (ADR 0066 decision 3 — the value side of the work being determined
// by the logical length alone)
// ---------------------------------------------------------------------------

const SYMBOLIC_MODEL: StateModel = { heads: 2, kvHeads: 2, depth: 4, capacity: "C" };

test.skipIf(!GPU_AVAILABLE)(
  "capacities C=8 and C=4096 give bit-identical output at every step (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    const session = await stateSession(gpu, SYMBOLIC_MODEL);
    const small = await session.createGenerationContext({ chunkLength: 4, bindings: { C: 8 } });
    const large = await session.createGenerationContext({ chunkLength: 4, bindings: { C: 4096 } });
    try {
      // MUST: the capacity **really takes effect** (if the binding were silently ignored, the two
      // would be the same context and bit identity a tautology). The capacity is part of the plan
      // key itself (ADR 0066 decision 3), so if this differs, the two run with different recipes,
      // different S strides and different allocation sizes.
      assertEquals(small[RUNTIME_INTERNAL].slots.get("kslot")?.shape, [1, 2, 8, 4]);
      assertEquals(large[RUNTIME_INTERNAL].slots.get("kslot")?.shape, [1, 2, 4096, 4]);

      // prefill (M=4 / Q=4) → decode ×3. The capacity affects only S's column stride and the
      // allocation size, so the same input at the same logical length must give the same result
      // **down to the rounding sequence**.
      for (const [index, [chunkRows, query]] of (
        [
          [4, 4],
          [1, 1],
          [1, 1],
          [1, 1],
        ] as const
      ).entries()) {
        const inputs = stepInputs(SYMBOLIC_MODEL, chunkRows, 3 + index * 41);
        const a = (await runStep(session, small, SYMBOLIC_MODEL, inputs, chunkRows, query))[0];
        const b = (await runStep(session, large, SYMBOLIC_MODEL, inputs, chunkRows, query))[0];
        const bits = bitsOf(a);
        assert(
          new Set(bits).size > 1,
          `step ${index}: the output is constant (bit equality is a tautology)`,
        );
        assertBitsEqual(bitsOf(b), bits, `step ${index}: the output changed with the capacity`);
      }
      assertEquals(small.pastLength, 7);
      assertEquals(large.pastLength, 7);
    } finally {
      for (const context of [small, large]) await context.dispose();
      await session.dispose();
      gpu.destroy();
    }
  },
);

// ---------------------------------------------------------------------------
// Group 5: execution of KV-sharing layers (ADR 0067 decision 4 — the 20-layer pattern of Gemma 4
// E2B)
// ---------------------------------------------------------------------------

/**
 * The tolerance for comparing the sharing layers (the oracle is D-2's CPU reference chain — this
 * group checks **execution**, and group 1 separately guarantees the reference's independence). The
 * rationale is the same as `STATE_TOLERANCE` in gpu_state_execution_test.ts (the reference
 * accumulates in f64, the GPU accumulates sequentially in f32).
 */
const SHARED_TOLERANCE: Tolerance = { atol: 5e-6, rtol: 0 };

const asF32 = (data: Float32Array | Int32Array | Uint32Array): Float32Array<ArrayBuffer> => {
  if (!(data instanceof Float32Array)) throw new Error("the reference did not return f32");
  return data as Float32Array<ArrayBuffer>;
};

type SharedSeries = {
  readonly name: string;
  readonly model: StateModel;
  readonly capacity: number;
  readonly chunkLength: number;
  readonly steps: readonly (readonly [number, number])[];
  /**
   * The step that checks the difference from the oracle where "append ran before the readers"
   * (omitted = not checked). A difference appears only where the ring wrap overwrites past rows of
   * this step's readers (the only window where the order of ADR 0067 decision 5b affects the
   * values).
   */
  readonly orderProbeStep?: number;
};

const SHARED_SERIES: readonly SharedSeries[] = [
  {
    name: "full",
    model: { heads: 2, kvHeads: 2, depth: 4, capacity: 16 },
    capacity: 16,
    chunkLength: 3,
    steps: [
      [3, 3],
      [3, 3],
      [3, 3],
      [1, 1],
    ],
  },
  {
    // A W=4 / C=4 ring. The append of step 1 (past=3 / Q=3) writes physical 3,0,1, so two of the
    // physical rows 0,1,2 the readers read first get overwritten = where an order violation shows
    // in the values.
    name: "sliding W=4",
    model: { heads: 2, kvHeads: 2, depth: 4, capacity: 4, window: 4 },
    capacity: 4,
    chunkLength: 3,
    steps: [
      [3, 3],
      [3, 3],
      [3, 3],
      [1, 1],
    ],
    orderProbeStep: 1,
  },
];

test.skipIf(!GPU_AVAILABLE)(
  "KV-sharing layers (two readers → one append) execute (ADR 0067 decisions 4 / 5b, real GPU)",
  async () => {
    const gpu = await acquireGpu();
    try {
      for (const series of SHARED_SERIES) {
        const { model, capacity } = series;
        const session = await stateSession(gpu, model, 2);
        const context = await session.createGenerationContext({ chunkLength: series.chunkLength });
        let slotK = new Float32Array(model.kvHeads * capacity * model.depth);
        let slotV = new Float32Array(model.kvHeads * capacity * model.depth);
        let past = 0;
        try {
          for (const [index, [chunkRows, query]] of series.steps.entries()) {
            const inputs = stepInputs(model, chunkRows, 17 + index * 23, 2);
            const outs = await runStep(session, context, model, inputs, chunkRows, query);
            const common = {
              chunkRows,
              depth: model.depth,
              capacity,
              window: model.window ?? 0,
              past,
              query,
            };
            const read = (
              q: Float32Array<ArrayBuffer>,
              k: Float32Array<ArrayBuffer>,
              v: Float32Array<ArrayBuffer>,
            ): Float32Array<ArrayBuffer> =>
              asF32(
                referenceStateAttention({
                  ...common,
                  batch: 1,
                  heads: model.heads,
                  kvHeads: model.kvHeads,
                  q,
                  insK: inputs.k,
                  insV: inputs.v,
                  slotK: k,
                  slotV: v,
                  scale: halfScale(model.depth),
                }).data,
              );

            // MUST: **both readers read the slots from before the append** (the issue rule of
            // decision 5b).
            const expected = inputs.q.map((q) => read(q, slotK, slotV));
            outs.forEach((actual, layer) => {
              const label = `${series.name} step ${index} layer ${layer}`;
              assert(
                expected[layer].some((value) => Math.abs(value) > 1e-3),
                `${label}: the expected output is trivial (all ~0)`,
              );
              const report = compareTensors(
                { dtype: "f32", data: actual },
                { dtype: "f32", data: expected[layer] },
                SHARED_TOLERANCE,
              );
              assertEquals(report.pass, true, `${label}: ${formatAllclose(report)}`);
            });
            // The owning layer and the sharing layer have **the same KV but different q**, so their
            // outputs differ (this catches a wiring mix-up where the second is a copy of the
            // first).
            assertEquals(
              mismatches(bitsOf(outs[0]), bitsOf(outs[1])).length > 0,
              true,
              `${series.name} step ${index}: the two layers' outputs are bit-identical (the sharing layer copies the owner)`,
            );

            // Detection power for the order: no match with the oracle that "ran append before the
            // readers".
            if (series.orderProbeStep === index) {
              const appendBase = { ...common, kvPlanes: model.kvHeads };
              const wrongK = asF32(
                referenceStateAppend({ ...appendBase, x: inputs.k, slot: slotK }).data,
              );
              const wrongV = asF32(
                referenceStateAppend({ ...appendBase, x: inputs.v, slot: slotV }).data,
              );
              outs.forEach((actual, layer) => {
                const report = compareTensors(
                  { dtype: "f32", data: actual },
                  { dtype: "f32", data: read(inputs.q[layer], wrongK, wrongV) },
                  SHARED_TOLERANCE,
                );
                assertEquals(
                  report.pass,
                  false,
                  `${series.name} step ${index} layer ${layer}: also matched the append-first oracle` +
                    " (the probe picks a position where decision 5b's order does not show in the values = never fires)",
                );
              });
            }

            const appendBase = { ...common, kvPlanes: model.kvHeads };
            slotK = asF32(referenceStateAppend({ ...appendBase, x: inputs.k, slot: slotK }).data);
            slotV = asF32(referenceStateAppend({ ...appendBase, x: inputs.v, slot: slotV }).data);
            past += query;
            assertEquals(
              context.pastLength,
              past,
              `${series.name} step ${index}: the logical length drifted (two appends advanced it twice)`,
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
