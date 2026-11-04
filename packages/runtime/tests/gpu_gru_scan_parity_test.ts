// The **bit-identity check** of gru_scan / gru_scan_reverse (ADR 0056).
//
// For the same inputs, it runs
//
//   ① a reference graph built from existing GPU ops that "copies torch's GRU decomposition
//      literally" (`linear → slice → add → sigmoid / mul → add → tanh / sub → mul → add`, T times)
//   ② one `gru_scan` op node
//
// on **the same real GPU** and compares the outputs as **f32 bit patterns**. The point is to use an
// exact `Uint32Array` match rather than allclose: changes in the rounding sequence that a tolerance
// hides (the hidden-side reduction drifting from the GEMM of `linear` / only the fused side being
// contracted into fma / the update rewritten as `(1−z)·n + z·h`) can only be detected here.
//
// **This op has the same "real A/B oracle" as attention (ADR 0023 decision 2).** With T fixed,
// torch's decomposed graph is itself a complete reference implementation, so the degenerate oracle
// that the new primitive (deform_conv2d) fell back to is not needed.
//
// MUST: do not let it become vacuous. ①② run **separate graphs** in separate Sessions, and every
// case also checks
//   - that the output is not a constant sequence (the values move along time)
//   - that the outputs of `gru_scan` and `gru_scan_reverse` are **different** (the direction
//     matters)
// (an implementation that ignores the scan direction stays green with the first check alone).

import { test } from "vite-plus/test";
import { assert, assertEquals } from "@std/assert";
import { acquireGpu, type GpuContext } from "../src/gpu/device.ts";
import { createSessionFromContainer, type Tensor } from "../src/runtime/executor.ts";
import {
  type DeclarationJson,
  fill,
  type FilledTensor,
  GRAPH_NAME,
  openModelBytes,
  singleOpDeclaration,
} from "./helpers/model-fixture.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";

type NodeJson = DeclarationJson["nodes"][number];
type ValueSpec = DeclarationJson["values"][string];

type Geometry = {
  readonly time: number;
  readonly batch: number;
  readonly hidden: number;
};

/**
 * Reference graph that copies torch's GRU decomposition literally into the existing vocabulary (T
 * is unrolled statically).
 *
 * | torch's decomposed nodes          | here                                        |
 * | --------------------------------- | ------------------------------------------- |
 * | `linear(h, w_hh, b_hh)`           | `linear` (bias is the GEMM epilogue = last) |
 * | `sigmoid(add(gh_r, gi_r))`        | `add` → `sigmoid`                           |
 * | `tanh(add(gi_n, mul(gh_n, r)))`   | `mul` → `add` → `tanh`                      |
 * | `add(mul(sub(h, n), z), n)`       | `sub` → `mul` → `add`                       |
 * | stack each step's h along time    | `reshape` → `cat`(dim 0)                    |
 *
 * MUST: do not reorder the arguments. Addition and multiplication of finite f32 are commutative,
 * but **this sequence is the source of truth for the rounding sequence**, so changing the order
 * changes "what it is meant to match" (ADR 0056 decision 3).
 * MUST: the reverse direction inverts **only the scan order**; the sequence passed to `cat` stays
 * in forward time order (no `flip` in between = the same semantics as the form the op folds).
 */
const decomposedGraph = (geometry: Geometry, reverse: boolean): DeclarationJson => {
  const { time, batch, hidden } = geometry;
  const gates = 3 * hidden;
  const nodes: NodeJson[] = [];
  const values: Record<string, ValueSpec> = {};
  const declare = (name: string, shape: readonly number[]): string => {
    values[name] = { dtype: "f32", shape: [...shape] };
    return name;
  };
  /** Slices 1 gate (`0 = r / 1 = z / 2 = n`) out of the `[N, 3H]` gate bundle. */
  const gate = (source: string, name: string, index: number): string => {
    nodes.push({
      op: "slice",
      ins: [source],
      outs: [declare(name, [batch, hidden])],
      attrs: { dim: 1, start: index * hidden, end: (index + 1) * hidden },
    });
    return name;
  };
  const unary = (op: string, source: string, name: string): string => {
    nodes.push({ op, ins: [source], outs: [declare(name, [batch, hidden])], attrs: {} });
    return name;
  };
  const binary = (op: string, a: string, b: string, name: string): string => {
    nodes.push({ op, ins: [a, b], outs: [declare(name, [batch, hidden])], attrs: {} });
    return name;
  };

  // Scan order (t descending for the reverse direction). The time index written to is always t
  // itself.
  const order = Array.from({ length: time }, (_unused, step) => (reverse ? time - 1 - step : step));
  const stacked: string[] = [];
  let state = "h0";
  for (const t of order) {
    const gh = declare(`gh${t}`, [batch, gates]);
    nodes.push({ op: "linear", ins: [state, "w", "b"], outs: [gh], attrs: {} });
    nodes.push({
      op: "slice",
      ins: ["gi"],
      outs: [declare(`giStep${t}`, [1, batch, gates])],
      attrs: { dim: 0, start: t, end: t + 1 },
    });
    const inputGates = declare(`giFlat${t}`, [batch, gates]);
    nodes.push({ op: "reshape", ins: [`giStep${t}`], outs: [inputGates], attrs: {} });

    const reset = unary(
      "sigmoid",
      binary("add", gate(gh, `ghR${t}`, 0), gate(inputGates, `giR${t}`, 0), `sumR${t}`),
      `r${t}`,
    );
    const update = unary(
      "sigmoid",
      binary("add", gate(gh, `ghZ${t}`, 1), gate(inputGates, `giZ${t}`, 1), `sumZ${t}`),
      `z${t}`,
    );
    const candidate = unary(
      "tanh",
      binary(
        "add",
        gate(inputGates, `giN${t}`, 2),
        binary("mul", gate(gh, `ghN${t}`, 2), reset, `prodN${t}`),
        `sumN${t}`,
      ),
      `n${t}`,
    );
    const next = binary(
      "add",
      binary("mul", binary("sub", state, candidate, `diff${t}`), update, `decay${t}`),
      candidate,
      // MUST: do not collide with the input name `h0` (spelling the t = 0 state `h0` would violate
      // SSA)
      `state${t}`,
    );
    nodes.push({
      op: "reshape",
      ins: [next],
      outs: [declare(`step${t}`, [1, batch, hidden])],
      attrs: {},
    });
    stacked[t] = `step${t}`;
    state = next;
  }
  nodes.push({
    op: "cat",
    ins: stacked,
    outs: [declare("y", [time, batch, hidden])],
    attrs: { dim: 0 },
  });

  return {
    format: "karume-ir",
    version: 2,
    requires: {
      ops: ["linear", "slice", "reshape", "add", "sub", "mul", "sigmoid", "tanh", "cat"],
    },
    symbols: [],
    inputs: [
      { name: "gi", dtype: "f32", shape: [time, batch, gates] },
      { name: "h0", dtype: "f32", shape: [batch, hidden] },
      { name: "w", dtype: "f32", shape: [gates, hidden] },
      { name: "b", dtype: "f32", shape: [gates] },
    ],
    outputs: ["y"],
    initializers: {},
    values,
    nodes,
  };
};

const run = async (
  gpu: GpuContext,
  graph: DeclarationJson,
  inputs: Readonly<Record<string, FilledTensor>>,
): Promise<Tensor> => {
  const session = await createSessionFromContainer(
    gpu,
    await openModelBytes(graph, []),
    GRAPH_NAME,
  );
  try {
    return (await session.run(inputs))["y"];
  } finally {
    await session.dispose();
  }
};

/**
 * f32 bit patterns (a difference of 1 ulp at the end and the difference between `0.0` and `-0.0`
 * are both visible).
 */
const bits = (tensor: Tensor): Uint32Array =>
  new Uint32Array(tensor.data.buffer, tensor.data.byteOffset, tensor.data.length);

// Deterministic sequences (no random numbers — a failure would not reproduce). The range keeps the
// gates from saturating fully: if sigmoid sticks to 0 / 1, rounding differences in the update do
// not show in the values and the check becomes vacuous.
const GATE_INPUT = (i: number): number => (((i * 7) % 23) - 11) * 0.21;
const STATE_INIT = (i: number): number => (((i * 5) % 17) - 8) * 0.13;
const WEIGHT = (i: number): number => (((i * 11) % 29) - 14) * 0.037;
const BIAS = (i: number): number => (((i * 3) % 13) - 6) * 0.11;

const CASES: readonly Geometry[] = [
  // Smallest shape (T = 2 = the minimum where the state is carried at least once)
  { time: 2, batch: 1, hidden: 3 },
  // The vowel-detector hidden width (H = 128) at its real size
  { time: 4, batch: 1, hidden: 128 },
  // H exactly the workgroup width (the limit of 1 lane = 1 hidden unit — the edge lane is
  // effective)
  { time: 3, batch: 1, hidden: 256 },
  // Batch > 1 (checks that the state does not mix between workgroups)
  { time: 5, batch: 3, hidden: 7 },
  // A reduction length that is not a multiple of the GEMM K tile (16) (the zero padding of the edge
  // tile matters)
  { time: 3, batch: 2, hidden: 19 },
];

const inputsFor = (geometry: Geometry): Record<string, FilledTensor> => {
  const { time, batch, hidden } = geometry;
  const gates = 3 * hidden;
  return {
    gi: fill([time, batch, gates], GATE_INPUT),
    h0: fill([batch, hidden], STATE_INIT),
    w: fill([gates, hidden], WEIGHT),
    b: fill([gates], BIAS),
  };
};

/**
 * Saturation-range check (regression detection for gru_scan v2 = the shared tanh_stable). Puts the
 * n-gate inputs from the saturation range (|x| > 9.5) up to the range where exp(2x) overflows f32
 * (|x| > 44.36). A regression back to the plain built-in `tanh` stays green and bit-identical on an
 * IEEE-faithful implementation, but turns the Number.isFinite check below red on an implementation
 * that computes `(exp(2x)−1)/(exp(2x)+1)` (Metal fast-math). The r / z gates keep the usual range
 * (if sigmoid sticks to 0 / 1, rounding differences in the update do not show in the values).
 */
const SATURATED_N_GATE = [12, -15, 50, -50, 120, -120, 9.4, -9.6] as const;

const saturatedInputsFor = (geometry: Geometry): Record<string, FilledTensor> => {
  const { time, batch, hidden } = geometry;
  const gates = 3 * hidden;
  const gi = (i: number): number =>
    i % gates >= 2 * hidden ? SATURATED_N_GATE[i % SATURATED_N_GATE.length] : GATE_INPUT(i);
  return {
    gi: fill([time, batch, gates], gi),
    h0: fill([batch, hidden], STATE_INIT),
    w: fill([gates, hidden], WEIGHT),
    b: fill([gates], BIAS),
  };
};

const scanGraph = (op: string, geometry: Geometry): DeclarationJson => {
  const { time, batch, hidden } = geometry;
  const gates = 3 * hidden;
  return singleOpDeclaration(
    op,
    [[time, batch, gates], [batch, hidden], [gates, hidden], [gates]],
    [[time, batch, hidden]],
  );
};

/** Renames to the input names of a single-node graph (`x0`…). */
const asSingleOpInputs = (
  named: Readonly<Record<string, FilledTensor>>,
): Record<string, FilledTensor> => ({
  x0: named["gi"],
  x1: named["h0"],
  x2: named["w"],
  x3: named["b"],
});

for (const op of ["gru_scan", "gru_scan_reverse"] as const) {
  test.skipIf(!GPU_AVAILABLE)(
    `${op}: the output of 1 node **matches torch's literal decomposition (linear + elementwise ops) bit for bit** (real GPU)`,
    async () => {
      const gpu = await acquireGpu();
      try {
        const parityCases = [
          ...CASES.map((geometry) => ({ geometry, named: inputsFor(geometry), tag: "" })),
          // Saturation case (H = the measured 128 — the candidate gate's argument crosses ±9.5)
          {
            geometry: { time: 3, batch: 2, hidden: 128 },
            named: saturatedInputsFor({ time: 3, batch: 2, hidden: 128 }),
            tag: " saturated",
          },
        ];
        for (const { geometry, named, tag } of parityCases) {
          const label = `${op} T${geometry.time} N${geometry.batch} H${geometry.hidden}${tag}`;
          const fused = await run(gpu, scanGraph(op, geometry), asSingleOpInputs(named));
          const decomposed = await run(
            gpu,
            decomposedGraph(geometry, op === "gru_scan_reverse"),
            named,
          );

          assertEquals(fused.shape, decomposed.shape, label);
          const a = bits(fused);
          const b = bits(decomposed);
          const mismatches: number[] = [];
          for (let i = 0; i < a.length; i += 1) {
            if (a[i] !== b[i]) mismatches.push(i);
          }
          assertEquals(
            mismatches.length,
            0,
            `${label}: ${mismatches.length} / ${a.length} elements differ in bits (first ${mismatches
              .slice(0, 4)
              .map((i) => `#${i} fused=0x${a[i].toString(16)} decomposed=0x${b[i].toString(16)}`)
              .join(" / ")})`,
          );

          // Not vacuous: the output is not a constant sequence (if the decomposed side were all one
          // value, a match would say nothing)
          const values = fused.data as Float32Array;
          assert(values.every(Number.isFinite), `${label}: has non-finite values`);
          assert(
            new Set(values).size > 1,
            `${label}: the output is a constant sequence (the check is vacuous)`,
          );
        }
      } finally {
        gpu.destroy();
      }
    },
  );
}

// Checks that the scan direction really matters. Implementing the reverse direction as "the same
// scan as forward" can keep the parity above green because **both sides share the same error**
// (the reference graph takes its order from the same function).
test.skipIf(!GPU_AVAILABLE)(
  "gru_scan and gru_scan_reverse give different outputs for the same inputs (the scan direction matters)",
  async () => {
    const gpu = await acquireGpu();
    try {
      for (const geometry of CASES) {
        const named = asSingleOpInputs(inputsFor(geometry));
        const forward = await run(gpu, scanGraph("gru_scan", geometry), named);
        const reverse = await run(gpu, scanGraph("gru_scan_reverse", geometry), named);
        const a = bits(forward);
        const b = bits(reverse);
        let same = 0;
        for (let i = 0; i < a.length; i += 1) {
          if (a[i] === b[i]) same += 1;
        }
        assert(
          same < a.length,
          `T${geometry.time} N${geometry.batch} H${geometry.hidden}: the forward and reverse outputs match exactly (the scan direction is ignored)`,
        );
      }
    } finally {
      gpu.destroy();
    }
  },
);
