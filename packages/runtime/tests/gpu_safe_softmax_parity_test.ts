// The **bit-identity check** of safe_softmax (ADR 0044).
//
// For the same input,
//
//   ① a reference equivalent to the "decomposed guard", built from the existing GPU op sequence
//      (`ge_scalar → sum → cast → bitwise_not → masked_fill(softmax(x), empty row, 0)`)
//   ② a single `safe_softmax` op node
//
// run on **the same real GPU**, and their outputs are compared as **f32 bit patterns**. The point
// is exact `Uint32Array` equality rather than allclose: a change in the rounding sequence that a
// tolerance would hide (e.g. the reduction order of ②③ drifting from plain softmax / turning
// `1.0 / Σ` back into a division) is detected only here.
//
// The mapping table of {@link guardedGraph} shows that ① copies torch's guard
// `where(¬any(¬eq(src,−inf)), 0, softmax(src))` op by op.
//
// MUST: do not let this become a tautology. ① and ② run as **separate graphs** in separate
// Sessions (comparing against safe_softmax's CPU reference alone cannot confirm that the GPU side's
// empty-row branch "takes the same path as plain softmax").
// MUST: always run **a case with an all −inf row** (fault injection — without such a row the
// masked_fill of ① never fires, and the test degenerates into a tautology where both merely agree
// as plain softmax).

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

/** The smallest finite f32. "The row max is not −inf" is equivalent to `amax_row >= this value`. */
const NEG_F32_MAX = -3.4028234663852886e38;

/**
 * The reference graph equivalent to the decomposed guard (built from the existing vocabulary only).
 *
 * torch's guard `where(¬any(¬eq(src,−inf)), 0, softmax(src))`, copied op by op:
 *
 * | torch                  | here                                      |
 * | ---------------------- | ----------------------------------------- |
 * | `¬eq(src, −inf)`       | `ge_scalar(src, −F32_MAX)` (bool)         |
 * | `any(…, −1)`           | `sum` (bool → i32) → `cast` to bool       |
 * | `logical_not`          | `bitwise_not`                             |
 * | `where(…, 0, softmax)` | `masked_fill(softmax(src), empty row, 0)` |
 *
 * `ge_scalar(v, −F32_MAX)` is equivalent to `¬eq(v, −inf)` because every finite f32 is at least
 * −F32_MAX (the input is "finite or −inf" — the contract of ADR 0044 decision 3).
 *
 * MUST: do not detect an empty row with the row max (`amax`). The runtime's row reduce has the
 * identity ±F32_MAX (limitations.md), so `amax` of an all −inf row returns −F32_MAX rather than
 * −inf — indistinguishable from masked_fill's fill value. Testing each element and then reducing,
 * as here, is the correct mapping.
 */
const guardedGraph = (rows: number, dim: number): DeclarationJson => ({
  format: "karume-ir",
  version: 2,
  requires: {
    ops: ["softmax", "ge_scalar", "sum", "cast", "bitwise_not", "reshape", "masked_fill"],
  },
  symbols: [],
  inputs: [{ name: "x", dtype: "f32", shape: [rows, dim] }],
  outputs: ["y"],
  initializers: {},
  values: {
    probs: { dtype: "f32", shape: [rows, dim] },
    finiteEl: { dtype: "bool", shape: [rows, dim] },
    count: { dtype: "i32", shape: [rows] },
    anyFinite: { dtype: "bool", shape: [rows] },
    emptyRow: { dtype: "bool", shape: [rows] },
    empty: { dtype: "bool", shape: [rows, 1] },
    y: { dtype: "f32", shape: [rows, dim] },
  },
  nodes: [
    { op: "softmax", ins: ["x"], outs: ["probs"], attrs: { dim: 1 } },
    { op: "ge_scalar", ins: ["x"], outs: ["finiteEl"], attrs: { value: NEG_F32_MAX } },
    { op: "sum", ins: ["finiteEl"], outs: ["count"], attrs: { dim: 1 } },
    { op: "cast", ins: ["count"], outs: ["anyFinite"], attrs: { to: "bool" } },
    { op: "bitwise_not", ins: ["anyFinite"], outs: ["emptyRow"], attrs: {} },
    { op: "reshape", ins: ["emptyRow"], outs: ["empty"], attrs: {} },
    // The mask [rows,1] is read widened to [rows,dim] by right-aligned broadcasting
    { op: "masked_fill", ins: ["probs", "empty"], outs: ["y"], attrs: { value: 0 } },
  ],
});

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
 * The f32 bit patterns (a form where both `0.0` vs `-0.0` and a last-place 1 ulp difference show).
 */
const bits = (tensor: Tensor): Uint32Array =>
  new Uint32Array(tensor.data.buffer, tensor.data.byteOffset, tensor.data.length);

/** A deterministic score sequence (no random numbers — a failure would not reproduce). */
const SCORE = (i: number): number => (((i * 13) % 29) - 14) * 0.37;

type Case = {
  readonly name: string;
  readonly rows: number;
  readonly dim: number;
  /** The set of rows whose elements are all set to −inf. */
  readonly emptyRows: readonly number[];
  /** Which columns within a row are set to −inf individually (the holes of an additive mask). */
  readonly blocked: (row: number, col: number) => boolean;
};

const CASES: readonly Case[] = [
  // No empty row — this is bit-identical to plain softmax (a check that the variant has not
  // degenerated into another implementation)
  { name: "no empty row [5,9]", rows: 5, dim: 9, emptyRows: [], blocked: () => false },
  // The observed shape of a run-time bool mask: some columns are −inf, and one row is all −inf
  {
    name: "one all −inf row [4,7]",
    rows: 4,
    dim: 7,
    emptyRows: [2],
    blocked: (row, col) => (row + col) % 3 === 2,
  },
  // Consecutive empty rows (rows drop out together, as in the uncond batch of CFG)
  {
    name: "two consecutive empty rows [6,5]",
    rows: 6,
    dim: 5,
    emptyRows: [1, 2],
    blocked: (_, col) => col === 0,
  },
  // Row length > workgroup width (256) — exercises both ①'s grid-stride scan and the tree reduction
  {
    name: "row length 300 (wider than a workgroup) [3,300]",
    rows: 3,
    dim: 300,
    emptyRows: [1],
    blocked: (_, col) => col % 5 === 4,
  },
  // Also exercises grid-stride with rows > workgroups (one workgroup reduces several rows)
  {
    name: "130 rows (grid-stride) [130,4]",
    rows: 130,
    dim: 4,
    emptyRows: [0, 65, 129],
    blocked: (row, col) => (row + col) % 4 === 3,
  },
];

test.skipIf(!GPU_AVAILABLE)(
  "the output of a single safe_softmax node is **bit-identical** to the decomposed-guard equivalent (softmax + masked_fill) (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    try {
      for (const testCase of CASES) {
        const { name, rows, dim, emptyRows } = testCase;
        const empty = new Set(emptyRows);
        const x = fill([rows, dim], (i) => {
          const row = Math.floor(i / dim);
          const col = i % dim;
          return empty.has(row) || testCase.blocked(row, col) ? Number.NEGATIVE_INFINITY : SCORE(i);
        });

        const safe = await run(
          gpu,
          singleOpDeclaration("safe_softmax", [[rows, dim]], [[rows, dim]], { attrs: { dim: 1 } }),
          { x0: x },
        );
        const guarded = await run(gpu, guardedGraph(rows, dim), { x });

        assertEquals(safe.shape, guarded.shape, name);
        const a = bits(safe);
        const b = bits(guarded);
        const mismatches: number[] = [];
        for (let i = 0; i < a.length; i += 1) {
          if (a[i] !== b[i]) mismatches.push(i);
        }
        assertEquals(
          mismatches.length,
          0,
          `${name}: ${mismatches.length} elements differ in bits (first: ${mismatches
            .slice(0, 4)
            .map((i) => `#${i} safe=0x${a[i].toString(16)} guarded=0x${b[i].toString(16)}`)
            .join(" / ")})`,
        );

        // Confirm the fault injection: empty rows really are 0 and non-empty rows are not (rules
        // out the state where both merely agree as "plain softmax NaN against NaN").
        const values = safe.data as Float32Array;
        for (let row = 0; row < rows; row += 1) {
          const slice = [...values.slice(row * dim, (row + 1) * dim)];
          assertEquals(
            slice.every(Number.isFinite),
            true,
            `${name}: non-finite value in row ${row}`,
          );
          if (empty.has(row)) {
            assertEquals(slice, Array(dim).fill(0), `${name}: empty row ${row} is not 0`);
          } else {
            assert(
              slice.some((v) => v > 0),
              `${name}: non-empty row ${row} is all 0`,
            );
          }
        }
      }
    } finally {
      gpu.destroy();
    }
  },
);

// On input without empty rows, safe_softmax MUST be **bit-identical to plain softmax** (the direct
// check that changing ①'s identity to −inf has not leaked into finite rows — the parity above goes
// through masked_fill, so equality with plain softmax itself is checked separately).
test.skipIf(!GPU_AVAILABLE)(
  "safe_softmax is bit-identical to plain softmax on input without −inf (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    try {
      for (const [rows, dim] of [
        [4, 9],
        [3, 300],
        [130, 4],
      ] as const) {
        const x = fill([rows, dim], SCORE);
        const safe = await run(
          gpu,
          singleOpDeclaration("safe_softmax", [[rows, dim]], [[rows, dim]], { attrs: { dim: 1 } }),
          { x0: x },
        );
        const plain = await run(
          gpu,
          singleOpDeclaration("softmax", [[rows, dim]], [[rows, dim]], { attrs: { dim: 1 } }),
          { x0: x },
        );
        assertEquals([...bits(safe)], [...bits(plain)], `[${rows},${dim}]`);
      }
    } finally {
      gpu.destroy();
    }
  },
);
