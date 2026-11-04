// E2E written against the public surface (mod.ts) only. The standing test for ADR 0008's
// "mechanically detect drift between the surface and the implementation".
// MUST: do not import src/ directly — written against internals, it cannot detect a feature that
// fell off the public surface. Do not pull in test helpers either (if a helper imports src/, this
// check loses that much of its detection power).

import { readTextFile } from "@karume/test-support/fs";
import { test } from "vite-plus/test";
import { assertAlmostEquals, assertEquals, assertRejects, assertThrows } from "@std/assert";
import {
  acquireGpu,
  type BoundContainer,
  capabilities,
  ContainerFormatError,
  createSessionFromContainer,
  type FusionCounts,
  type GenerationRun,
  type GpuContext,
  openContainer,
  openMemoryContainer,
  parseSafetensors,
  SafetensorsError,
  type Session,
  type SessionDiagnostics,
  type Tensor,
} from "../mod.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";

const W = [0.5, -1.5, 2, 0.25, -0.75, 1];
const B = [0.125, -0.5];

const f32Bytes = (values: readonly number[]): Uint8Array<ArrayBuffer> => {
  const array = Float32Array.from(values);
  return new Uint8Array(array.buffer);
};

/**
 * y[T] = Σ_c (x[T,3] · w[3,2] + b[2]). A 3-node graph with the symbolic dimension T, built with
 * the public surface's **in-memory container** (the surface that supplies bytes in hand as is).
 */
const model = (): BoundContainer =>
  openMemoryContainer({
    graphs: {
      model: {
        format: "karume-ir",
        version: 2,
        requires: { ops: ["add", "matmul", "sum"] },
        symbols: ["T"],
        inputs: [{ name: "x", dtype: "f32", shape: ["T", 3] }],
        outputs: ["y"],
        initializers: { w: { shared: false }, b: { shared: false } },
        values: {
          w: { dtype: "f32", shape: [3, 2] },
          b: { dtype: "f32", shape: [2] },
          h: { dtype: "f32", shape: ["T", 2] },
          g: { dtype: "f32", shape: ["T", 2] },
          y: { dtype: "f32", shape: ["T"] },
        },
        states: {},
        nodes: [
          { op: "matmul", ins: ["x", "w"], outs: ["h"], attrs: {}, states: {} },
          { op: "add", ins: ["h", "b"], outs: ["g"], attrs: {}, states: {} },
          { op: "sum", ins: ["g"], outs: ["y"], attrs: { dim: 1 }, states: {} },
        ],
      },
    },
    tensors: {
      model: {
        w: { bytes: f32Bytes(W), encoding: { codec: "f32" } },
        b: { bytes: f32Bytes(B), encoding: { codec: "f32" } },
      },
    },
  });

/**
 * Computed by hand, rounding to f32 at every node (a reference independent of the implementation).
 */
const expectedRows = (x: readonly number[], rows: number): Float32Array<ArrayBuffer> => {
  const out = new Float32Array(rows);
  for (let row = 0; row < rows; row += 1) {
    let total = 0;
    for (let col = 0; col < 2; col += 1) {
      let dot = 0;
      for (let i = 0; i < 3; i += 1) dot += x[row * 3 + i] * W[i * 2 + col];
      total += Math.fround(Math.fround(dot) + B[col]);
    }
    out[row] = Math.fround(total);
  }
  return out;
};

test.skipIf(!GPU_AVAILABLE)(
  "opens a model through the public surface only and runs a graph with symbolic dimensions (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    // MUST: device destruction also covers a failing `createSession`. If it is missed, the rest of
    // the run carries an undestroyed device and later tests go red with OOM (deferred release in
    // known-issues).
    try {
      const session = await createSessionFromContainer(gpu, model(), "model");
      try {
        const rows = 4;
        const values = Array.from({ length: rows * 3 }, (_, i) => ((i % 9) - 4) * 0.5);
        const x: Tensor = { dtype: "f32", shape: [rows, 3], data: Float32Array.from(values) };

        const outputs = await session.run({ x });
        assertEquals(Object.keys(outputs), ["y"]);
        assertEquals(outputs["y"].shape, [rows]);
        const expected = expectedRows(values, rows);
        for (let row = 0; row < rows; row += 1) {
          assertAlmostEquals(outputs["y"].data[row], expected[row], 1e-5, `row ${row}`);
        }

        // Symbolic dimensions are rebound on every run
        const wide = Array.from({ length: 9 * 3 }, (_, i) => (i % 5) - 2);
        const outputs2 = await session.run(
          {
            x: { dtype: "f32", shape: [9, 3], data: Float32Array.from(wide) },
          },
          { T: 9 },
        );
        assertEquals(outputs2["y"].shape, [9]);
        const expected2 = expectedRows(wide, 9);
        for (let row = 0; row < 9; row += 1) {
          assertAlmostEquals(outputs2["y"].data[row], expected2[row], 1e-5, `wide row ${row}`);
        }

        // MUST: the diagnostics are received **typed**. Even if one of the types
        // `SessionDiagnostics` refers to fell off the mod.ts export list, it would still be
        // structurally assignable, so the line below that writes the name is the mechanical
        // detector of a missing export (it could not be written while `FusionCounts` was private).
        const diagnostics: SessionDiagnostics = session.diagnostics();
        const fusions: FusionCounts | undefined = diagnostics.lastRunFusions;
        // This graph (matmul / add / sum) matches no rule = every counter is 0.
        assertEquals(fusions, {
          silu: 0,
          upsample2x: 0,
          rope: 0,
          rmsNormRope: 0,
          adaln: 0,
          gatedResidual: 0,
          rmsNormAdd: 0,
          linearStaticQuantize: 0,
          geluMulStaticQuantize: 0,
          rowBlockAttention: 0,
          identityExpand: 0,
          packedStaticQuantize: 0,
          linearPrologue: 0,
          attentionPermute: 0,
        });
        assertEquals(diagnostics.pipelineCount, 3);
        // The weight arena owns the params cache (resident in the Session) in addition to the two
        // initializers. These two runs have different T, so params are loaded with different
        // contents, 3 nodes' worth each.
        assertEquals(diagnostics.weights.allocCount, 2 + 3 + 3);
        assertEquals(diagnostics.lastRunParams, { allocCount: 3, reuseCount: 0 });
        assertEquals((diagnostics.lastRun?.allocCount ?? 0) > 0, true);
      } finally {
        await session.dispose();
      }
    } finally {
      gpu.destroy();
    }
  },
);

/**
 * The path of zero-element tensors (shapes containing 0). An element count of 0 produces 0
 * dispatches and an allocation size of 0, and gets through by riding the lower clamps in various
 * places (minimum 4 bytes). This pins that zero-length data comes back with the shape kept, and
 * that later runs on the same Session do not break.
 */
test.skipIf(!GPU_AVAILABLE)(
  "a zero-element tensor (symbolic dimension T=0) comes back with length 0 and its shape kept (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    // MUST: device destruction also covers a failing `createSession`. If it is missed, the rest of
    // the run carries an undestroyed device and later tests go red with OOM (deferred release in
    // known-issues).
    try {
      const session = await createSessionFromContainer(gpu, model(), "model");
      try {
        const empty: Tensor = { dtype: "f32", shape: [0, 3], data: new Float32Array(0) };
        const outputs = await session.run({ x: empty });
        assertEquals(Object.keys(outputs), ["y"]);
        assertEquals(
          outputs["y"].shape,
          [0],
          "even with 0 elements the shape does not degenerate to a scalar",
        );
        assertEquals(outputs["y"].data.length, 0);
        // All 3 nodes are encoded as "zero-element dispatches" (not silently skipped)
        assertEquals(session.diagnostics().submit.dispatchCount, 3);

        // The zero-element run did not break the Session's state (pools, pipelines)
        const values = [1, -2, 0.5, 3, 0.25, -1];
        const next = await session.run({
          x: { dtype: "f32", shape: [2, 3], data: Float32Array.from(values) },
        });
        assertEquals(next["y"].shape, [2]);
        const expected = expectedRows(values, 2);
        for (let row = 0; row < 2; row += 1) {
          assertAlmostEquals(next["y"].data[row], expected[row], 1e-5, `row ${row}`);
        }
      } finally {
        await session.dispose();
      }
    } finally {
      gpu.destroy();
    }
  },
);

test("the runtime's internal serialization primitives do not appear on the public type surface (ADR 0008)", () => {
  // A type-level check. If `withScopeLock` / `raceDeviceLost` were on GpuContext under their plain
  // names, the assignment below would be a type error (the runtime assert is a tautology; the type
  // checker is what detects it). Visible under plain names, `gpu.withScopeLock(() =>
  // session.run(...))` could be written, and run would take the same lock and **self-deadlock with
  // no exception and no diagnostic** (no reentrancy detector can be placed).
  // MUST: negate each name separately. A conditional type with a union on the left is **not a naked
  // type parameter**, so it does not distribute; if only one name leaked, the union as a whole is
  // not assignable to `keyof GpuContext` = it stays in the false branch and passes (a one-sided
  // leak slips through).
  const scopeLockHidden: "withScopeLock" extends keyof GpuContext ? false : true = true;
  const raceDeviceLostHidden: "raceDeviceLost" extends keyof GpuContext ? false : true = true;
  assertEquals([scopeLockHidden, raceDeviceLostHidden], [true, true]);
});

test("the third argument of a generation run can be named with a public type (the call form of ADR 0066 decision 6)", () => {
  // A type-level check. If `GenerationRun` falls off the mod.ts export list, the type check fails
  // (an object literal can still be passed through structural subtyping, so the line below that
  // writes the type appearing in the public method's signature **by name** is the mechanical
  // detector of a missing export).
  const asThirdArgument = (run: GenerationRun): Parameters<Session["run"]>[2] => run;
  assertEquals(typeof asThirdArgument, "function");
});

test("the public surface provides the capability query and the model-parsing error types", async () => {
  const ops = capabilities().ops;
  for (const op of ["matmul", "add", "sum", "gelu", "amax"]) {
    assertEquals(ops.includes(op), true, op);
  }
  assertEquals(capabilities().storage, ["bf16", "f16", "f32", "i2", "i32", "i4", "i8"]);
  // The two types for a container (`krm` / `krg`) parse failure and an attached-asset (safetensors)
  // parse failure.
  await assertRejects(
    () => openContainer({ kind: "bytes", bytes: new Uint8Array(new ArrayBuffer(4)) }),
    ContainerFormatError,
  );
  assertThrows(() => parseSafetensors(new ArrayBuffer(4)), SafetensorsError);
});

test("this E2E imports no implementation other than the public surface (mod.ts)", async () => {
  const source = await readTextFile(new URL(import.meta.url));
  const specifiers = [...source.matchAll(/from "([^"]+)"/g)].map((match) => match[1]);
  assertEquals(
    specifiers.filter((path) => path.includes("/src/")),
    [],
  );
});
