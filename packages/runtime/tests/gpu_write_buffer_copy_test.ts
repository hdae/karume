/**
 * Pins on real hardware that `queue.writeBuffer` copies the bytes **at the time of the call** (the
 * WebGPU spec's content timeline — "Let dataContents be a copy of the bytes held by the buffer
 * source data.").
 *
 * Session build reads and uploads one block at a time and lets go of the bytes in the iteration
 * that uploaded them (it does not hold them until the fence — the fence NOTE in `session-build.ts`,
 * ADR 0070 decision 3 / the 2026-09-24 addendum to ADR 0108 decision 9). The only ground for that
 * being correct is this spec rule; an implementation that read lazily (reading the original bytes
 * at submit / fence time) would give silent wrong values. Here, during the build, the bytes passed
 * to writeBuffer are filled with poison right after it returns, and the test checks that the GPU
 * residents after the fence keep the original values (the output is bit-identical to a build
 * without poison).
 *
 * Synthetic model: linear (i8 + per-channel scale, 2 pieces = 2 parts, f32 bias) → add (f16 weight
 * — not in a weight slot, so it is expanded to f32 on the CPU). One test goes through raw-byte
 * weights, a companion scale, writes at piece offsets, expanded f32 and a fence across parts.
 */

import { assert, assertEquals } from "@std/assert";
import { describe, it } from "vite-plus/test";
import { type MemoryContainerInput, openMemoryContainer } from "../src/format/container/memory.ts";
import { type IrDeclaration, parseIrDeclarationValue } from "../src/format/ir.ts";
import { acquireGpu } from "../src/gpu/device.ts";
import { createSessionFromContainer } from "../src/runtime/executor.ts";
import { quantizeF16 } from "./helpers/f16.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";
import { quantizeI8 } from "./helpers/i8.ts";
import { f32Bytes, fill } from "./helpers/model-fixture.ts";

const M = 4;
const K = 16;
const N = 8;

const declaration = (): IrDeclaration =>
  parseIrDeclarationValue({
    format: "karume-ir",
    version: 2,
    requires: { ops: ["add", "linear"] },
    symbols: [],
    inputs: [{ name: "x", dtype: "f32", shape: [M, K] }],
    outputs: ["y"],
    initializers: { "enc.w": {}, "enc.b": {}, "enc.h": {} },
    values: {
      "enc.w": { dtype: "f32", shape: [N, K] },
      "enc.b": { dtype: "f32", shape: [N] },
      "enc.h": { dtype: "f32", shape: [N] },
      t: { dtype: "f32", shape: [M, N] },
      y: { dtype: "f32", shape: [M, N] },
    },
    nodes: [
      { op: "linear", ins: ["x", "enc.w", "enc.b"], outs: ["t"], attrs: {} },
      { op: "add", ins: ["t", "enc.h"], outs: ["y"], attrs: {} },
    ],
  });

/**
 * Builds the supplies in fresh buffers every time. The poison overwrites the supply bytes
 * themselves, so the reference build and the poisoned build share no buffers. `bias` is used to
 * confirm that the poison reached the supply's buffer.
 */
const supply = (): { readonly input: MemoryContainerInput; readonly bias: Uint8Array } => {
  const w = quantizeI8(fill([N, K], (i) => Math.sin(i * 0.37) * 0.5).data, [N, K], 0);
  const bias = f32Bytes(fill([N], (i) => (i % 3) * 0.05).data);
  const half = (N / 2) * K;
  return {
    bias,
    input: {
      graphs: { main: declaration() },
      tensors: {
        main: {
          "enc.w": {
            pieces: [
              { rows: [0, N / 2], read: () => Promise.resolve(w.bytes.subarray(0, half)) },
              { rows: [N / 2, N], read: () => Promise.resolve(w.bytes.subarray(half, N * K)) },
            ],
            encoding: { codec: "int8-sym", rowAxis: 0, groupSize: K, scale: f32Bytes(w.scale) },
          },
          "enc.b": { bytes: bias, encoding: { codec: "f32" } },
          "enc.h": {
            bytes: quantizeF16(fill([N], (i) => (i % 5) * 0.125 - 0.25).data).bytes,
            encoding: { codec: "f16" },
          },
        },
      },
    },
  };
};

/** Poison that is NaN in f32 (0xffffffff), −1 in i8 and NaN in f16. */
const POISON = 0xff;

/**
 * Wraps `writeBuffer` for the duration of `body` only and fills the passed bytes with poison right
 * after the original call returns. The prototype is wrapped (so it does not matter whether
 * `device.queue` returns the same object every time).
 *
 * MUST: always unwrap after the build (before the run). Poisoning the run's input and params writes
 * too would corrupt the CPU-side copy of the params cache, and the test would break for reasons
 * unrelated to this spec rule.
 */
const withPoisonedWrites = async <T>(
  queue: GPUQueue,
  body: () => Promise<T>,
): Promise<{ readonly result: T; readonly writes: number }> => {
  const prototype = Object.getPrototypeOf(queue) as GPUQueue;
  // oxlint-disable-next-line typescript/unbound-method -- called only as original.apply(this, args), then reassigned
  const original = prototype.writeBuffer;
  let writes = 0;
  prototype.writeBuffer = function (
    this: GPUQueue,
    ...args: Parameters<GPUQueue["writeBuffer"]>
  ): undefined {
    const returned = original.apply(this, args);
    const data = args[2];
    const bytes = ArrayBuffer.isView(data)
      ? new Uint8Array(data.buffer, data.byteOffset, data.byteLength)
      : new Uint8Array(data);
    bytes.fill(POISON);
    writes += 1;
    return returned;
  };
  try {
    return { result: await body(), writes };
  } finally {
    prototype.writeBuffer = original;
  }
};

describe.skipIf(!GPU_AVAILABLE)("writeBuffer copies the bytes at the time of the call", () => {
  it("filling the original bytes with poison right after writeBuffer returns leaves the residents after the fence (the output) at the original values", async () => {
    const gpu = await acquireGpu();
    try {
      const x = fill([M, K], (i) => ((i * 7) % 11) / 11 - 0.5);
      const reference = await createSessionFromContainer(
        gpu,
        openMemoryContainer(supply().input),
        "main",
      );
      let expected: Float32Array<ArrayBuffer>;
      try {
        expected = (await reference.run({ x }))["y"].data as Float32Array<ArrayBuffer>;
      } finally {
        await reference.dispose();
      }
      // Poison (NaN) would make the output non-finite — if the reference were not finite, the
      // comparison would miss.
      assert(expected.every(Number.isFinite), "the reference output has non-finite values");

      const poisoned = supply();
      const { result: session, writes } = await withPoisonedWrites(gpu.device.queue, () =>
        createSessionFromContainer(gpu, openMemoryContainer(poisoned.input), "main"),
      );
      try {
        // The poison is in the very buffer passed to writeBuffer (here the supply's buffer).
        assert(
          poisoned.bias.every((byte) => byte === POISON),
          "the supply's buffer is not filled with poison (writeBuffer got another buffer — the detector would miss)",
        );
        assert(writes >= 4, `the 4 weight blocks did not go through writeBuffer (${writes} calls)`);
        const actual = (await session.run({ x }))["y"].data as Float32Array<ArrayBuffer>;
        assertEquals(new Uint8Array(actual.buffer), new Uint8Array(expected.buffer));
      } finally {
        await session.dispose();
      }
    } finally {
      gpu.destroy();
    }
  });
});
