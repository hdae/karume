// Real-GPU checks for the **readonly variants** of states-form attention (①' ro / ② ro / ③' ro —
// ADR 0096 decision 1, step 2).
//
// The drafter (MTP head) has no K/V of its own and only reads the slots of the lender context. The
// query is one row at logical position P−1 and attends to the columns `[P − min(P, W), P)`
// (`[0, P)` for full). Three checks:
//
// ① within the band of the f64 reference (`referenceStateAttentionReadonly`) (the same atol as the
//    states form)
// ② **equivalence with the states form**: readonly(P) is **bit-identical** to ①'/③' with
//    "past = P−1, Q = 1, ins = the row at position P−1" (it sums the same columns in the same
//    order, so there is no reason for the values to change — if they do, the predicate or the
//    mapping differs)
// ③ structural guarantees: P = 0 is exactly 0 (empty row), determinism, rejected rows (col ≥ P) and
//    out-of-window rows (col < P−W) are **not read** (poison values placed in those slot rows leave
//    the values unchanged), and fault injection (widening live by one column / moving the window
//    start one column earlier / turning the mapping's modulus back to window) turns red

import { test } from "vite-plus/test";
import { assert, assertEquals } from "@std/assert";
import { compareTensors, formatAllclose, type Tolerance } from "../src/reference/allclose.ts";
import {
  referenceStateAttentionReadonly,
  type StateAttentionReadonlyRefInput,
} from "../src/reference/state-attention.ts";
import { stateColumnBaseReadonly, stateSliding } from "../src/kernels/state-attention.ts";
import { acquireGpu } from "../src/gpu/device.ts";
import {
  halfScale,
  runStateAttention,
  runStateAttentionReadonly,
  seeded,
  type StateCase,
  type StateInputs,
  type StateMutation,
  type StatePipelineCache,
  type StateReadonlyInputs,
} from "./helpers/state-dispatch.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";

/**
 * The tolerance against the f64 reference (the same value and rationale as `STATE_TOLERANCE` in
 * gpu_state_attention_test.ts).
 */
const STATE_TOLERANCE: Tolerance = { atol: 5e-6, rtol: 0 };

const QUERY = (i: number): number => Math.sin(i * 0.37 + 0.2) * 0.9;
const KEY = (i: number): number => Math.cos(i * 0.53 + 1.1) * 0.8;
const VALUE = (i: number): number => Math.sin(i * 0.71 + 2.3) * 1.2;
/**
 * The poison value placed outside live (rejected rows, out of window, uninitialized) (reading it
 * always exceeds the band of the reference).
 */
const SLOT_POISON = 2 ** 60;

/** A readonly case with M = Q = 1 fixed (`past` = the lender's P, capacity C, window W). */
const readonlyCase = (
  name: string,
  shape: { heads: number; kvHeads: number; depth: number; capacity: number; window: number },
  past: number,
): StateCase => ({
  name,
  batch: 1,
  heads: shape.heads,
  kvHeads: shape.kvHeads,
  chunkRows: 1,
  depth: shape.depth,
  capacity: shape.capacity,
  window: shape.window,
  past,
  query: 1,
});

/**
 * GQA 4:1, close to the drafter's real shape (sliding is a scaled-down W=8 / C=16 = headroom 8).
 */
const SLIDING = { heads: 4, kvHeads: 1, depth: 16, capacity: 16, window: 8 };
const FULL = { heads: 4, kvHeads: 1, depth: 16, capacity: 24, window: 0 };
const PLAIN = { heads: 2, kvHeads: 2, depth: 8, capacity: 12, window: 0 };

const CASES: readonly StateCase[] = [
  readonlyCase("full P=0 (empty row)", FULL, 0),
  readonlyCase("full P=1", FULL, 1),
  readonlyCase("full P=7", FULL, 7),
  readonlyCase("full P=C (at capacity)", FULL, 24),
  readonlyCase("plain P=5", PLAIN, 5),
  readonlyCase("sliding P=0 (empty row)", SLIDING, 0),
  readonlyCase("sliding P=3 < W", SLIDING, 3),
  readonlyCase("sliding P=8 = W", SLIDING, 8),
  readonlyCase("sliding P=9 (the window starts moving)", SLIDING, 9),
  readonlyCase("sliding P=17 (after one ring wrap)", SLIDING, 17),
  readonlyCase("sliding P=40 (several ring wraps)", SLIDING, 40),
];

/**
 * Inputs: one q row and the lender's slots. Physical rows outside live (rejected rows = rows that
 * logical col ≥ P maps to, out of window, uninitialized) are filled with **poison values** — if
 * readonly reads a single word of them, it exceeds the band of the reference.
 */
const makeInputs = (spec: StateCase): StateReadonlyInputs => {
  const { batch, heads, kvHeads, depth, capacity, window, past } = spec;
  const kvPlanes = batch * kvHeads;
  const q = seeded(batch * heads * depth, QUERY);
  const slotK = seeded(kvPlanes * capacity * depth, (i) => KEY(i + 3));
  const slotV = seeded(kvPlanes * capacity * depth, (i) => VALUE(i + 5));
  const resident = new Set<number>();
  for (let col = stateColumnBaseReadonly(window, past); col < past; col += 1) {
    resident.add(stateSliding(window) ? col % capacity : col);
  }
  for (let plane = 0; plane < kvPlanes; plane += 1) {
    for (let row = 0; row < capacity; row += 1) {
      if (resident.has(row)) continue;
      for (let d = 0; d < depth; d += 1) {
        slotK[(plane * capacity + row) * depth + d] = SLOT_POISON;
        slotV[(plane * capacity + row) * depth + d] = SLOT_POISON;
      }
    }
  }
  return { q, slotK, slotV };
};

const refInput = (
  spec: StateCase,
  inputs: StateReadonlyInputs,
): StateAttentionReadonlyRefInput => ({
  batch: spec.batch,
  heads: spec.heads,
  kvHeads: spec.kvHeads,
  depth: spec.depth,
  capacity: spec.capacity,
  window: spec.window,
  past: spec.past,
  q: inputs.q,
  slotK: inputs.slotK,
  slotV: inputs.slotV,
  scale: halfScale(spec.depth),
});

/**
 * The counterpart of check ②: builds the states form (①'/③') on the same slots with "past = P−1,
 * Q = 1, ins = the row at position P−1". What goes into ins is exactly the contents of the lender
 * slot's physical row `slot_row(P−1)`.
 */
const equivalentStatesRun = (
  spec: StateCase,
  inputs: StateReadonlyInputs,
): {
  readonly spec: StateCase;
  readonly inputs: StateInputs;
} => {
  const { batch, kvHeads, depth, capacity, window, past } = spec;
  const kvPlanes = batch * kvHeads;
  const last = past - 1;
  const physical = stateSliding(window) ? last % capacity : last;
  const insK = new Float32Array(kvPlanes * depth);
  const insV = new Float32Array(kvPlanes * depth);
  for (let plane = 0; plane < kvPlanes; plane += 1) {
    for (let d = 0; d < depth; d += 1) {
      insK[plane * depth + d] = inputs.slotK[(plane * capacity + physical) * depth + d];
      insV[plane * depth + d] = inputs.slotV[(plane * capacity + physical) * depth + d];
    }
  }
  return {
    spec: { ...spec, past: last, query: 1 },
    inputs: { q: inputs.q, insK, insV, slotK: inputs.slotK, slotV: inputs.slotV },
  };
};

const bitsOf = (data: Float32Array<ArrayBuffer>): Uint32Array =>
  new Uint32Array(data.buffer, data.byteOffset, data.length);

const assertBitIdentical = (
  a: Float32Array<ArrayBuffer>,
  b: Float32Array<ArrayBuffer>,
  label: string,
): void => {
  assertEquals(a.length, b.length, `${label}: length`);
  const ab = bitsOf(a);
  const bb = bitsOf(b);
  for (let i = 0; i < ab.length; i += 1) {
    if (ab[i] !== bb[i]) {
      throw new Error(`${label}: element ${i} differs in bits (${a[i]} vs ${b[i]})`);
    }
  }
};

test.skipIf(!GPU_AVAILABLE)(
  "states-form readonly variants: within the reference band, bit-identical to the states form (P−1, Q=1), empty row and determinism (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    const cache: StatePipelineCache = new Map();
    try {
      let worst = 0;
      for (const spec of CASES) {
        const inputs = makeInputs(spec);
        const expected = referenceStateAttentionReadonly(refInput(spec, inputs));
        const got = await runStateAttentionReadonly(gpu.device, spec, inputs, { cache });

        // ① the band of the f64 reference (reading a single poison word always falls outside it)
        const ref = compareTensors({ dtype: "f32", data: got.out }, expected, STATE_TOLERANCE);
        assert(ref.pass, `${spec.name} vs reference: ${formatAllclose(ref)}`);
        worst = Math.max(worst, ref.maxAbsError);

        if (spec.past === 0) {
          // ③ an empty row is exactly 0 (Object.is — lets neither −0 nor NaN through; no poison
          // initialization is left)
          for (let i = 0; i < got.out.length; i += 1) {
            assert(
              Object.is(got.out[i], 0),
              `${spec.name}: empty-row output ${i} is not exactly 0 (${got.out[i]})`,
            );
          }
        } else {
          // ② **bit-identical** to the states form (past = P−1, Q = 1, ins = the row at position
          // P−1)
          const eq = equivalentStatesRun(spec, inputs);
          const states = await runStateAttention(gpu.device, eq.spec, eq.inputs, {
            cache,
            qkReduce: "parallel",
            pvReduce: "parallel",
          });
          assertBitIdentical(got.out, states.out, `${spec.name}: readonly vs states form (①'/③')`);
        }

        // ③ determinism
        const again = await runStateAttentionReadonly(gpu.device, spec, inputs, { cache });
        assertBitIdentical(got.out, again.out, `${spec.name}: determinism`);
      }
      console.log(
        `readonly measured worst: vs reference maxAbs ${worst.toExponential(2)} (band ${STATE_TOLERANCE.atol})`,
      );
    } finally {
      gpu.destroy();
    }
  },
);

/**
 * Fault injection — mutations that break the live expression and the physical-row mapping fail the
 * reference band (proof that the check fires).
 */
const MUTATIONS: readonly { readonly name: string; readonly mutate: StateMutation }[] = [
  {
    // Widen live by one column (reads position P = a rejected row (poison))
    name: "live = min(past, window) → min(past, window + 1u)",
    mutate: (_kernel, wgsl) =>
      wgsl.replaceAll("return min(past, params.window);", "return min(past, params.window + 1u);"),
  },
  {
    // Move the window start one column earlier (reads the out-of-window P−W−1 (poison) and drops
    // P−1)
    name: "column_base = past − min(past, window) → past − min(past, window + 1u)",
    mutate: (_kernel, wgsl) =>
      wgsl.replaceAll(
        "return past - min(past, params.window);",
        "return past - min(past, params.window + 1u);",
      ),
  },
  {
    // Turn the read-side mapping back to modulo window (breaks "read and write use the same
    // expression" — after one ring wrap it points at another physical row)
    name: "slot_row = col % capacity → col % window",
    mutate: (_kernel, wgsl) =>
      wgsl.replaceAll("return col % params.capacity;", "return col % params.window;"),
  },
];

test.skipIf(!GPU_AVAILABLE)(
  "states-form readonly variants: fault injection (shifted live, shifted window start, mapping modulus) fails the reference band (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    const cache: StatePipelineCache = new Map();
    try {
      // Cases where a mutation shows in the values: sliding cases where the ring has wrapped, with
      // poison in the physical row of position P and in the out-of-window rows
      const targets = CASES.filter((spec) => spec.past >= 9 && stateSliding(spec.window));
      assert(targets.length >= 2, "not enough target cases for fault injection");
      for (const mutation of MUTATIONS) {
        let caught = 0;
        for (const spec of targets) {
          const inputs = makeInputs(spec);
          const expected = referenceStateAttentionReadonly(refInput(spec, inputs));
          let changed = 0;
          const mutated = await runStateAttentionReadonly(gpu.device, spec, inputs, {
            cache,
            mutate: (_kernel, wgsl) => {
              const after = mutation.mutate(_kernel, wgsl);
              if (after !== wgsl) changed += 1;
              return after;
            },
          });
          assert(
            changed > 0,
            `fault injection "${mutation.name}" rewrote no kernel (the source text changed)`,
          );
          const cmp = compareTensors(
            { dtype: "f32", data: mutated.out },
            expected,
            STATE_TOLERANCE,
          );
          if (!cmp.pass) caught += 1;
        }
        assert(
          caught > 0,
          `fault injection "${mutation.name}" turned no case red (the check never fires)`,
        );
      }
    } finally {
      gpu.destroy();
    }
  },
);
