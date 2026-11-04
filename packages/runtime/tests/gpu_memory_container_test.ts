/**
 * Source A/B — `krm` (`openContainer`) and the in-memory container (`openMemoryContainer`) must produce **the
 * same output bytes** from **the same weight bytes**. If only one of them falls back to CPU decoding / GPU
 * residency, the resident byte counts of the diagnostics differ.
 *
 * Synthetic model: linear (i4 + group scale) → add (f16 weights) → linear (i8 + per-channel scale, bias f32).
 * One model exercises the supply plans of 3 codecs, scale normalization (rank 2) and piece splitting.
 *
 * MUST: the two paths **deliberately split pieces differently** (`krm` lets the writer decide from a 128 B block
 * limit; the memory side states the row ranges explicitly, 4 rows each). Comparing with the same split would put
 * the invariant "splitting does not change a single byte of the GPU-side layout" outside the detector.
 *
 * One more pair: the same `krm` read through a source that copies out blocks and through a source that, like the
 * hub's scan path, returns views into the part's buffer (poisoned outside the buffer's blocks); output and
 * resident byte counts must match. If any reader drops the byteOffset, it reads the head of the buffer or the
 * poison and breaks.
 *
 * External ground truth: the 2 pairs above share everything from the merge layer to Session construction on both
 * sides, so an error there lands equally on both and they still match. So the stored bytes are also decoded back
 * to f32 with the CPU codec decoding (`decodeI4` / `decodeF16` / `decodeI8`), and the output is compared within a
 * tolerance against linear → add → linear traced by the CPU reference (`applyReferenceOp`). The same case also
 * checks that a reference with one decoding mixed up (the i4 scale shifted by one channel) fails on the same
 * output — confirming the tolerance is not loose enough to let a codec decoding error through.
 */

import { assert, assertEquals, assertNotEquals, assertRejects } from "@std/assert";
import type { BoundContainer } from "../src/format/container/bind.ts";
import { ContainerFormatError } from "../src/format/container/header.ts";
import { type MemoryContainerInput, openMemoryContainer } from "../src/format/container/memory.ts";
import { type BlockSource, openContainer } from "../src/format/container/open.ts";
import { decodeF16 } from "../src/format/f16.ts";
import { decodeI4 } from "../src/format/i4.ts";
import { decodeI8 } from "../src/format/i8.ts";
import { type IrDeclaration, parseIrDeclaration } from "../src/format/ir.ts";
import { acquireGpu } from "../src/gpu/device.ts";
import { compareTensors, formatAllclose } from "../src/reference/allclose.ts";
import { applyReferenceOp, type RefTensor, refTensor } from "../src/reference/ops.ts";
import {
  createSessionFromContainer,
  prepareContainer,
  type Tensor,
} from "../src/runtime/executor.ts";
import { GEMM_TOLERANCE } from "./helpers/op-tolerance.ts";
import { f32Bytes, fill, type FilledTensor } from "./helpers/model-fixture.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";
import { quantizeF16 } from "./helpers/f16.ts";
import { quantizeI4 } from "./helpers/i4.ts";
import { quantizeI8 } from "./helpers/i8.ts";
import {
  writeGraphContainer,
  writeModelContainer,
  type WrittenContainer,
} from "./helpers/container-write.ts";
import { describe, it } from "vite-plus/test";

const M = 4;
const K = 32;
const H = 24;
const N = 8;

/** Deterministic weights (no random numbers). */
const weights = () => ({
  w1: fill([H, K], (i) => Math.sin(i * 0.37) * 0.5),
  b1: fill([H], (i) => (i % 3) * 0.05),
  h: fill([H], (i) => (i % 5) * 0.125 - 0.25),
  w2: fill([N, H], (i) => Math.cos(i * 0.91) * 0.75),
  b2: fill([N], (i) => i * 0.01),
});

const declaration = (): IrDeclaration =>
  parseIrDeclaration(
    JSON.stringify({
      format: "karume-ir",
      version: 2,
      requires: { ops: ["add", "linear"] },
      symbols: [],
      inputs: [{ name: "x", dtype: "f32", shape: [M, K] }],
      outputs: ["y"],
      initializers: { "enc.w1": {}, "enc.b1": {}, "enc.h": {}, "enc.w2": {}, "enc.b2": {} },
      values: {
        "enc.w1": { dtype: "f32", shape: [H, K] },
        "enc.b1": { dtype: "f32", shape: [H] },
        "enc.h": { dtype: "f32", shape: [H] },
        "enc.w2": { dtype: "f32", shape: [N, H] },
        "enc.b2": { dtype: "f32", shape: [N] },
        t: { dtype: "f32", shape: [M, H] },
        u: { dtype: "f32", shape: [M, H] },
        y: { dtype: "f32", shape: [M, N] },
      },
      nodes: [
        { op: "linear", ins: ["x", "enc.w1", "enc.b1"], outs: ["t"], attrs: {} },
        { op: "add", ins: ["t", "enc.h"], outs: ["u"], attrs: {} },
        { op: "linear", ins: ["u", "enc.w2", "enc.b2"], outs: ["y"], attrs: {} },
      ],
    }),
  );

type Quantized = {
  readonly w1: ReturnType<typeof quantizeI4>;
  readonly b1: Uint8Array<ArrayBuffer>;
  readonly h16: Uint8Array<ArrayBuffer>;
  readonly w2: ReturnType<typeof quantizeI8>;
  readonly b2: Uint8Array<ArrayBuffer>;
};

const quantize = (): Quantized => {
  const w = weights();
  return {
    w1: quantizeI4(w.w1.data, [H, K], 16),
    b1: f32Bytes(w.b1.data),
    h16: quantizeF16(w.h.data).bytes,
    w2: quantizeI8(w.w2.data, [N, H], 0),
    b2: f32Bytes(w.b2.data),
  };
};

/** Input of the `krm`. The block limit is lowered to 128 B so that the writer splits pieces. */
const containerInput = (q: Quantized) => ({
  graphs: { main: declaration() },
  consts: [],
  weights: [
    {
      graph: "main",
      initializer: "enc.w1",
      bytes: q.w1.bytes,
      encoding: {
        codec: "int4-sym-g" as const,
        groupSize: 16,
        scale: { bytes: f32Bytes(q.w1.scale), dtype: "f32" as const },
      },
    },
    { graph: "main", initializer: "enc.b1", bytes: q.b1, encoding: { codec: "f32" as const } },
    { graph: "main", initializer: "enc.h", bytes: q.h16, encoding: { codec: "f16" as const } },
    {
      graph: "main",
      initializer: "enc.w2",
      bytes: q.w2.bytes,
      encoding: {
        codec: "int8-sym" as const,
        groupSize: H,
        scale: { bytes: f32Bytes(q.w2.scale), dtype: "f32" as const },
      },
    },
    { graph: "main", initializer: "enc.b2", bytes: q.b2, encoding: { codec: "f32" as const } },
  ],
  assets: [],
  provenance: { license: "test" },
});

/** A piece reader that splits the i8 weights into 2 by row range (a 4/4 split, unlike the 5/3 split on the `krm` side). */
const w2Pieces = (q: Quantized, counter: { reads: number }) => {
  const rowBytes = H;
  const ranges: readonly (readonly [number, number])[] = [
    [0, N / 2],
    [N / 2, N],
  ];
  return ranges.map((rows) => ({
    rows,
    read: (): Promise<Uint8Array<ArrayBuffer>> => {
      counter.reads += 1;
      return Promise.resolve(q.w2.bytes.subarray(rows[0] * rowBytes, rows[1] * rowBytes));
    },
  }));
};

/** Input of the in-memory container (the same bytes, passed as they are without writing a container). */
const memoryInput = (q: Quantized, counter: { reads: number }): MemoryContainerInput => ({
  graphs: { main: declaration() },
  tensors: {
    main: {
      "enc.w1": {
        bytes: q.w1.bytes,
        encoding: { codec: "int4-sym-g", groupSize: 16, scale: f32Bytes(q.w1.scale) },
      },
      "enc.b1": { bytes: q.b1, encoding: { codec: "f32" } },
      "enc.h": { bytes: q.h16, encoding: { codec: "f16" } },
      "enc.w2": {
        pieces: w2Pieces(q, counter),
        encoding: { codec: "int8-sym", rowAxis: 0, groupSize: H, scale: f32Bytes(q.w2.scale) },
      },
      "enc.b2": { bytes: q.b2, encoding: { codec: "f32" } },
    },
  },
});

/**
 * CPU reference of the miniature: decodes the same stored bytes that were given to the container back to f32
 * with the production CPU decoding, and traces the graph's 3 nodes with reference ops. It shares no
 * implementation with the merge layer or Session construction, and the resident i4 / i8 slots are also
 * independent of the GPU-side decoding (unpack + scale inside the kernel), so places where both paths are wrong
 * by the same amount break here too (f16 goes through the host's `decodeF16` in the decode slot, so that part
 * alone shares the reference's implementation).
 * `i4Scale` is for fault injection (by default the stored scale itself).
 */
const referenceOutput = (
  q: Quantized,
  x: FilledTensor,
  i4Scale: Float32Array<ArrayBuffer> = q.w1.scale,
): RefTensor => {
  const w = weights();
  const w1 = refTensor([H, K], decodeI4(q.w1.bytes, [H, K], i4Scale, q.w1.scaleShape, 16));
  const h = refTensor([H], decodeF16(q.h16));
  const w2 = refTensor([N, H], decodeI8(q.w2.bytes, [N, H], q.w2.scale, q.w2.scaleShape));
  const t = applyReferenceOp(
    "linear",
    [refTensor(x.shape, x.data), w1, refTensor([H], w.b1.data)],
    {},
    [M, H],
  );
  const u = applyReferenceOp("add", [t, h], {}, [M, H]);
  return applyReferenceOp("linear", [u, w2, refTensor([N], w.b2.data)], {}, [M, N]);
};

/** Shifts the i4 group scale by one output channel (row r reads the scale of row r+1). */
const shiftI4ScaleOneChannel = (q: Quantized): Float32Array<ArrayBuffer> => {
  const groups = q.w1.scaleShape[1];
  const scale = q.w1.scale;
  return Float32Array.from(scale, (_, i) => scale[(i + groups) % scale.length]);
};

const OPTIONS = { partBytes: 1024, blockBytes: 128 } as const;

const POISON = 0xff;

/**
 * A source of the same shape as the hub's scan path (`packages/hub/src/container.ts`): it holds one tight buffer
 * per part, and `read` returns a `subarray` of that buffer (no copy). The parts of the buffer outside blocks
 * (alignment gaps, the tail) are filled with poison — so that a read that dropped the byteOffset does not pass by
 * chance by reading "the neighbouring valid bytes".
 * `verified: true` is the same as the HF source (the path that does not check block digests).
 */
const scanShapedSource = (written: WrittenContainer) => {
  const blocksByPart = new Map<number, { readonly offset: number; readonly length: number }[]>();
  const blocks = [
    ...written.graph.const.blocks.map((block) => ({ ...block, part: 1 })),
    ...written.model.blocks,
  ];
  for (const block of blocks) {
    const list = blocksByPart.get(block.part) ?? [];
    list.push(block);
    blocksByPart.set(block.part, list);
  }
  let poisoned = 0;
  // Part 0 (header + 2 documents) has no blocks, so it is copied as is.
  const vessels = written.parts.map((part, index) => {
    if (index === 0) return part.slice();
    const vessel = new Uint8Array(new ArrayBuffer(part.byteLength)).fill(POISON);
    for (const block of blocksByPart.get(index) ?? []) {
      vessel.set(part.subarray(block.offset, block.offset + block.length), block.offset);
    }
    poisoned += vessel.reduce((count, byte, i) => count + (byte !== part[i] ? 1 : 0), 0);
    return vessel;
  });
  const offsets: number[] = [];
  const source: BlockSource = {
    partCount: vessels.length,
    verified: true,
    partLength: (index) => vessels[index].byteLength,
    read: (part, offset, length) => {
      if (part >= 2) offsets.push(offset);
      return Promise.resolve(vessels[part].subarray(offset, offset + length));
    },
  };
  return { source, poisoned, offsets };
};

/** A source that returns copies of blocks (tight) — the baseline of the comparison. */
const copyingSource = (written: WrittenContainer): BlockSource => ({
  partCount: written.parts.length,
  verified: true,
  partLength: (index) => written.parts[index].byteLength,
  read: (part, offset, length) =>
    Promise.resolve(written.parts[part].slice(offset, offset + length)),
});

describe.skipIf(!GPU_AVAILABLE)("memory container session", () => {
  it("a Session built from the in-memory container matches the krm path in output bytes and resident byte counts", async () => {
    const q = quantize();
    const written = await writeModelContainer(containerInput(q), OPTIONS);
    const gpu = await acquireGpu();
    try {
      const x = fill([M, K], (i) => ((i * 7) % 11) / 11 - 0.5);
      const opened = await openContainer({ kind: "parts", parts: written.parts });
      // With a 128 B block limit the writer splits both i8 (192 B) and i4 (384 B) into pieces.
      assertEquals(opened.graphs.main.supplies.get("enc.w2")?.blocks.length, 2);
      assertEquals(opened.graphs.main.supplies.get("enc.w1")?.blocks.length, 3);
      let expected: Float32Array<ArrayBuffer>;
      let expectedResident: number;
      const fromContainer = await createSessionFromContainer(gpu, opened, "main");
      try {
        expected = (await fromContainer.run({ x }))["y"].data as Float32Array<ArrayBuffer>;
        expectedResident = fromContainer.diagnostics().storage.residentCompressedBytes;
      } finally {
        await fromContainer.dispose();
      }

      const counter = { reads: 0 };
      const memory = openMemoryContainer(memoryInput(q, counter));
      // At open time no piece reader has been called even once (lazy).
      assertEquals(counter.reads, 0);
      assertEquals(memory.graphs.main.supplies.get("enc.w2")?.blocks.length, 2);
      // Precondition (the MUST of the module doc): the two paths split pieces differently. If the writer's
      // granularity changes and the splits become the same, this A/B stops checking the splitting invariant.
      const w2Rows = (bound: BoundContainer) =>
        bound.graphs.main.supplies.get("enc.w2")?.blocks.map((block) => block.rows);
      assertEquals(w2Rows(memory), [
        [0, N / 2],
        [N / 2, N],
      ]);
      assertNotEquals(
        w2Rows(opened),
        w2Rows(memory),
        "krm and the memory side split pieces the same way",
      );
      const session = await createSessionFromContainer(gpu, memory, "main");
      try {
        const actual = (await session.run({ x }))["y"].data as Float32Array<ArrayBuffer>;
        assertEquals(new Uint8Array(actual.buffer), new Uint8Array(expected.buffer));
        assertEquals(session.diagnostics().storage.residentCompressedBytes, expectedResident);
        assert(
          expectedResident > 0,
          "no weight is resident in compressed form (the i4 / f16 / i8 slots are not in effect)",
        );
        // Each piece reader is called exactly once during Session construction (not retained).
        assertEquals(counter.reads, 2);
      } finally {
        await session.dispose();
      }
      // The estimate is the same too (the slot classification does not split by source).
      assertEquals(
        prepareContainer(memory, "main").estimate(),
        prepareContainer(opened, "main").estimate(),
      );
    } finally {
      gpu.destroy();
    }
  });

  it("the outputs of krm and the in-memory container match, within tolerance, the output traced by reference ops over the stored bytes decoded on the CPU", async () => {
    const q = quantize();
    const written = await writeModelContainer(containerInput(q), OPTIONS);
    const x = fill([M, K], (i) => ((i * 7) % 11) / 11 - 0.5);
    const expected = referenceOutput(q, x);
    const miswired = referenceOutput(q, x, shiftI4ScaleOneChannel(q));
    const sources: readonly (readonly [string, BoundContainer])[] = [
      ["krm", await openContainer({ kind: "parts", parts: written.parts })],
      ["in-memory container", openMemoryContainer(memoryInput(q, { reads: 0 }))],
    ];
    const gpu = await acquireGpu();
    try {
      for (const [label, container] of sources) {
        const session = await createSessionFromContainer(gpu, container, "main");
        let output: Tensor;
        try {
          output = (await session.run({ x }))["y"];
        } finally {
          await session.dispose();
        }
        assertEquals(output.shape, expected.shape, label);
        const report = compareTensors(output, expected, GEMM_TOLERANCE);
        assertEquals(report.pass, true, `${label}: ${formatAllclose(report)}`);
        // The same output fails against a reference with one decoding mixed up (the tolerance does not let a codec error through).
        const miswiredReport = compareTensors(output, miswired, GEMM_TOLERANCE);
        assertEquals(
          miswiredReport.pass,
          false,
          `${label}: passed even against the reference with the i4 scale shifted by one channel (${formatAllclose(
            miswiredReport,
          )})`,
        );
      }
    } finally {
      gpu.destroy();
    }
  });

  it("a Session built from a source that returns views into the part buffer matches a source that returns copies in output and resident byte counts", async () => {
    const q = quantize();
    const written = await writeModelContainer(containerInput(q), OPTIONS);
    const viewed = scanShapedSource(written);
    // The poison actually went into the buffer, and blocks are read from the middle of the buffer (64 B aligned, not 0).
    assert(
      viewed.poisoned > 0,
      "the buffer has no gaps outside blocks, so no poison could be placed",
    );
    const gpu = await acquireGpu();
    try {
      const x = fill([M, K], (i) => ((i * 7) % 11) / 11 - 0.5);
      let expected: Float32Array<ArrayBuffer>;
      let expectedResident: number;
      const copied = await openContainer({ kind: "source", source: copyingSource(written) });
      const reference = await createSessionFromContainer(gpu, copied, "main");
      try {
        expected = (await reference.run({ x }))["y"].data as Float32Array<ArrayBuffer>;
        expectedResident = reference.diagnostics().storage.residentCompressedBytes;
      } finally {
        await reference.dispose();
      }

      const opened = await openContainer({ kind: "source", source: viewed.source });
      const session = await createSessionFromContainer(gpu, opened, "main");
      try {
        const actual = (await session.run({ x }))["y"].data as Float32Array<ArrayBuffer>;
        assertEquals(new Uint8Array(actual.buffer), new Uint8Array(expected.buffer));
        assertEquals(session.diagnostics().storage.residentCompressedBytes, expectedResident);
        assert(
          expectedResident > 0,
          "no weight is resident in compressed form (the i4 / f16 / i8 slots are not in effect)",
        );
      } finally {
        await session.dispose();
      }
      assert(
        viewed.offsets.some((offset) => offset !== 0),
        "every weight block is read from the head of a part (the byteOffset ≠ 0 view is not exercised)",
      );
      assert(
        viewed.offsets.every((offset) => offset % 64 === 0),
        "a block start is not 64 B aligned",
      );
    } finally {
      gpu.destroy();
    }
  });

  it("a Session cannot be built from a krg (no weight supply)", async () => {
    const q = quantize();
    const graphOnly = await writeGraphContainer(containerInput(q), OPTIONS);
    const opened = await openContainer({ kind: "bytes", bytes: graphOnly.bytes });
    const gpu = await acquireGpu();
    try {
      await assertRejects(
        () => createSessionFromContainer(gpu, opened, "main"),
        ContainerFormatError,
        "no weight supply",
      );
    } finally {
      gpu.destroy();
    }
  });
});
