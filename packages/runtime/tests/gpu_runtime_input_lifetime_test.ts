// Checks for the lifetime contract of run / enqueue inputs (a metadata snapshot at issue time +
// borrowed data).
//
// The body of run executes behind `#serialize` (one microtask later or more), so combining the
// non-awaited concurrent issue the JSDoc recommends with "mutation right after issue" would let the
// body see the caller's Record / shape arrays / bindings. To cut that off, the implementation
// **copies only the metadata in the synchronous part of the issue**, so these checks pin four
// points: ① a shape mutation does not reach binding resolution ② replacing a Record member does
// not reach it ③ a bindings mutation does not reach it ④ `Tensor.data` is still **borrowed**
// (= not cloned, as the contract says); and ⑤ enqueue behaves the same way.
//
// Without ①–③, a regression comes back split into either "fails loudly by chance" or "a silent
// wrong value" (in both, run's return value looks normal). ④ is a negative check: it turns red if
// the contract is flipped to cloning.

import { test } from "vite-plus/test";
import { assert, assertEquals } from "@std/assert";
import { acquireGpu, type GpuContext, type ResidentTensor } from "../src/gpu/device.ts";
import {
  createSessionFromContainer,
  type RunInput,
  type Session,
  type Tensor,
} from "../src/runtime/executor.ts";
import { type DeclarationJson, openGraphModel } from "./helpers/model-fixture.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";

const ROWS = 4;
const COLS = 3;
const COUNT = ROWS * COLS;
const BYTES = COUNT * 4;

/**
 * y = x + x. The row count is the symbolic dimension `N` so that the shape where **both the
 * bindings and the input shape feed binding resolution** (= where a mutation after issue would turn
 * into an unrelated "conflicting binding" failure) is actually exercised.
 */
const GRAPH: DeclarationJson = {
  format: "karume-ir",
  version: 2,
  requires: { ops: ["add"] },
  symbols: ["N"],
  inputs: [{ name: "x", dtype: "f32", shape: ["N", COLS] }],
  outputs: ["y"],
  initializers: {},
  values: { y: { dtype: "f32", shape: ["N", COLS] } },
  nodes: [{ op: "add", ins: ["x", "x"], outs: ["y"], attrs: {} }],
};

/**
 * An input whose values change with `phase`. `shape` can be passed in so that **an array the caller
 * still holds** can be mutated after issue (the `readonly` of `Tensor.shape` is only a compile-time
 * matter; the real object is a mutable array the caller owns — that is what the contract must
 * protect).
 */
const input = (phase: number, shape: number[] = [ROWS, COLS]): Tensor => ({
  dtype: "f32",
  shape,
  data: Float32Array.from({ length: COUNT }, (_, i) => (((i + phase * 3) % 9) - 4) * 0.5),
});

/** Reference values (computed by hand, rounding to f32 — independent of the implementation). */
const doubled = (phase: number): Float32Array<ArrayBuffer> => {
  const x = input(phase).data;
  return Float32Array.from({ length: COUNT }, (_, i) => Math.fround(x[i] + x[i]));
};

/**
 * Bit-pattern comparison (does not tolerate a rounding mix-up). A resident tensor reads back as a
 * plain ArrayBuffer.
 */
const bits = (data: Tensor["data"] | ArrayBuffer): readonly number[] =>
  Array.from(
    data instanceof ArrayBuffer
      ? new Uint32Array(data)
      : new Uint32Array(data.buffer, data.byteOffset, data.length),
  );

const openSession = async (gpu: GpuContext): Promise<Session> =>
  await createSessionFromContainer(gpu, await openGraphModel(GRAPH), "model");

test.skipIf(!GPU_AVAILABLE)(
  "a shape mutation right after run does not reach the resolved shape (snapshot at issue, real GPU)",
  async () => {
    const gpu = await acquireGpu();
    const session = await openSession(gpu);
    try {
      const shape = [ROWS, COLS];
      // Issue without await (the form run's JSDoc recommends) → break the shape in the same tick.
      const pending = session.run({ x: input(0, shape) });
      shape[0] = ROWS * 2;

      const outputs = await pending;
      assertEquals(
        bits(outputs["y"].data),
        bits(doubled(0)),
        "did not run with the shape at issue",
      );
      // A check on the resolved shape itself: a mutation that keeps the element count passes the
      // validation, and only the output's **shape** silently drifts (the values look correct).
      assertEquals(
        outputs["y"].shape,
        [ROWS, COLS],
        "the resolved shape moved with a mutation after issue",
      );
    } finally {
      await session.dispose();
      gpu.destroy();
    }
  },
);

test.skipIf(!GPU_AVAILABLE)(
  "replacing an inputs member right after run does not reach execution (snapshot at issue, real GPU)",
  async () => {
    const gpu = await acquireGpu();
    const session = await openSession(gpu);
    try {
      // A check against tautology: if the two phases expected the same values, a mix-up could not
      // be detected.
      assert(
        JSON.stringify(bits(doubled(0))) !== JSON.stringify(bits(doubled(1))),
        "the expected values do not change per phase (the detector would never fire)",
      );
      const inputs: Record<string, RunInput> = { x: input(0) };
      const pending = session.run(inputs);
      inputs["x"] = input(1);

      const outputs = await pending;
      assertEquals(
        bits(outputs["y"].data),
        bits(doubled(0)),
        "the replacement Tensor was executed",
      );
    } finally {
      await session.dispose();
      gpu.destroy();
    }
  },
);

test.skipIf(!GPU_AVAILABLE)(
  "a bindings mutation right after run causes no binding conflict (snapshot at issue, real GPU)",
  async () => {
    const gpu = await acquireGpu();
    const session = await openSession(gpu);
    try {
      // Explicitly bind a symbol that the input shape can also resolve (= a mutation would make the
      // body fail with "conflicting binding for N"). If it was copied at issue, this run stays at
      // N=ROWS to the end.
      const bindings: Record<string, number> = { N: ROWS };
      const pending = session.run({ x: input(0) }, bindings);
      bindings["N"] = ROWS + 1;

      const outputs = await pending;
      assertEquals(bits(outputs["y"].data), bits(doubled(0)));
      assertEquals(
        outputs["y"].shape,
        [ROWS, COLS],
        "the resolved shape moved with a binding after issue",
      );
    } finally {
      await session.dispose();
      gpu.destroy();
    }
  },
);

test.skipIf(!GPU_AVAILABLE)(
  "a data mutation right after run reaches execution (negative check of borrowing, real GPU)",
  async () => {
    const gpu = await acquireGpu();
    const session = await openSession(gpu);
    try {
      const tensor = input(0);
      const pending = session.run({ x: tensor });
      // MUST NOT (the caller's side of the contract): mutation before the returned Promise
      // settles. Here the contract is broken **on purpose** to pin that the runtime does not clone
      // `data`.
      tensor.data.fill(0);

      const outputs = await pending;
      // This is by design (only the metadata is copied and data is borrowed — so that no run pays
      // for a GiB-scale copy). If the contract changed to cloning, this case would turn red = a
      // contract change cannot slip in silently.
      assertEquals(
        bits(outputs["y"].data),
        bits(new Float32Array(COUNT)),
        "data was cloned (the borrowing contract changed — if this is a spec change, update the ADR and this check together)",
      );
    } finally {
      await session.dispose();
      gpu.destroy();
    }
  },
);

test.skipIf(!GPU_AVAILABLE)(
  "enqueue also runs on the snapshot at issue (inputs / shape / copyOutputs, real GPU)",
  async () => {
    const gpu = await acquireGpu();
    const session = await openSession(gpu);
    const sink = await gpu.createResident(BYTES, "sink");
    const decoy = await gpu.createResident(BYTES, "decoy");
    try {
      // Fill the decoy with known values first (if replacing the copy target got through, **this is
      // what gets overwritten**, so detection does not rely on zero initialization).
      decoy.write(input(2).data);
      const shape = [ROWS, COLS];
      const inputs: Record<string, RunInput> = { x: input(0, shape) };
      const copyOutputs: Record<string, ResidentTensor> = { y: sink };
      const batch = await gpu.beginBatch();
      // enqueue is a surface meant for non-awaited issue (finish can be called without waiting on
      // the returned Promise), so a mutation right after issue is easier to hit than with run.
      const pending = session.enqueue(inputs, { batch, copyOutputs });
      inputs["x"] = input(1);
      shape[0] = ROWS * 2;
      copyOutputs["y"] = decoy;

      await pending;
      await batch.finish();
      assertEquals(
        bits(await sink.read()),
        bits(doubled(0)),
        "a replacement after issue reached execution",
      );
      assertEquals(
        bits(await decoy.read()),
        bits(input(2).data),
        "replacing the copy target reached execution",
      );
    } finally {
      await session.dispose();
      sink.dispose();
      decoy.dispose();
      gpu.destroy();
    }
  },
);
