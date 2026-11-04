/**
 * The in-memory container (`openMemoryContainer`) — the surface that supplies local bytes without
 * writing a container.
 *
 * This file pins two things:
 * - **The checks this layer owns** (matching the declaration, `pieces` having two or more entries)
 *   and **the checks it delegates to the merge layer** (codec × semantic dtype, piece coverage,
 *   whether a scale is needed) are actually in effect. Since "the in-memory container does not go
 *   through a descriptor", the latter would slip through unchecked if the delegation broke.
 * - **The rules of the synthesized index** (block ids, part assignment, how `readBlock` returns).
 *   The part assignment is exactly the fence boundary of Session construction, so if it breaks the
 *   RAM peak changes silently.
 */

import { assertEquals, assertRejects, assertStrictEquals, assertThrows } from "@std/assert";
import { describe, it } from "vite-plus/test";
import { ContainerFormatError } from "../src/format/container/header.ts";
import { type MemoryTensor, openMemoryContainer } from "../src/format/container/memory.ts";
import { type IrDeclaration, parseIrDeclarationValue } from "../src/format/ir.ts";
import { prepareContainer } from "../src/runtime/executor.ts";
import {
  baseDeclaration,
  baseTensors,
  type DeclarationJson,
  f32Bytes,
  GRAPH_NAME,
  memoryModel,
  memoryTensorOf,
  openModelBytes,
} from "./helpers/model-fixture.ts";

const declarationOf = (graph: DeclarationJson): IrDeclaration => parseIrDeclarationValue(graph);

/**
 * Opens one declaration + one supply set as is (the bare entry point with no mapping in between).
 */
const openWith = (
  graph: DeclarationJson,
  tensors: Readonly<Record<string, MemoryTensor>>,
  graphKey: string = GRAPH_NAME,
) =>
  openMemoryContainer({
    graphs: { [GRAPH_NAME]: declarationOf(graph) },
    tensors: { [graphKey]: tensors },
  });

const baseSupply = (): Record<string, MemoryTensor> =>
  Object.fromEntries(baseTensors().map((tensor) => [tensor.initializer, memoryTensorOf(tensor)]));

// --- Fixture for quantization + piece splitting ------------------------------
// q is a per-channel int8 [4,16] (payload 64 B; the scale is rank 2 [4,1] = 16 B). Splitting it
// into 2 pieces of 2 rows each makes the middle piece exactly 32 B, which cannot be padded.

const ROWS = 4;
const ROW_LENGTH = 16;

const quantizedDeclaration = (): DeclarationJson => ({
  format: "karume-ir",
  version: 2,
  requires: { ops: ["linear"] },
  symbols: [],
  inputs: [{ name: "x", dtype: "f32", shape: [2, ROW_LENGTH] }],
  outputs: ["y"],
  initializers: { q: {}, b: {} },
  values: {
    q: { dtype: "f32", shape: [ROWS, ROW_LENGTH] },
    b: { dtype: "f32", shape: [ROWS] },
    y: { dtype: "f32", shape: [2, ROWS] },
  },
  nodes: [{ op: "linear", ins: ["x", "q", "b"], outs: ["y"], attrs: {} }],
});

const qPayload = (): Uint8Array<ArrayBuffer> =>
  Uint8Array.from({ length: ROWS * ROW_LENGTH }, (_, i) => i % 7);

const qScale = (): Uint8Array<ArrayBuffer> => f32Bytes(new Array(ROWS).fill(0.5));

const biasTensor = (): MemoryTensor => ({
  bytes: f32Bytes(new Array(ROWS).fill(0.25)),
  encoding: { codec: "f32" },
});

/**
 * The supply of q split by row range (tests replace the `rows` column to make a gap / an end
 * mismatch).
 */
const quantizedPieces = (
  rows: readonly (readonly [number, number])[],
  onRead: (index: number) => void = () => {},
): MemoryTensor => {
  const payload = qPayload();
  return {
    encoding: { codec: "int8-sym", rowAxis: 0, groupSize: ROW_LENGTH, scale: qScale() },
    pieces: rows.map((range, index) => ({
      rows: range,
      read: () => {
        onRead(index);
        return Promise.resolve(payload.subarray(range[0] * ROW_LENGTH, range[1] * ROW_LENGTH));
      },
    })),
  };
};

describe("openMemoryContainer: matching the declaration", () => {
  it("enumerates every missing non-shared initializer and fails", () => {
    const error = assertThrows(
      () => openWith(baseDeclaration(), {}),
      ContainerFormatError,
      "missing [w, b]",
    );
    assertEquals(error.message.includes("surplus []"), true, error.message);
  });

  it("rejects a supply for an initializer not in the declaration as surplus", () => {
    assertThrows(
      () => openWith(baseDeclaration(), { ...baseSupply(), z: biasTensor() }),
      ContainerFormatError,
      "surplus [z]",
    );
  });

  it("rejects a supply for an undeclared graph name", () => {
    assertThrows(
      () => openWith(baseDeclaration(), baseSupply(), "other"),
      ContainerFormatError,
      "Supply has an undeclared graph 'other'",
    );
  });

  it("rejects a supply with a single piece (pass bytes for a single piece)", () => {
    assertThrows(
      () =>
        openWith(quantizedDeclaration(), {
          q: quantizedPieces([[0, ROWS]]),
          b: biasTensor(),
        }),
      ContainerFormatError,
      "pieces must have 2 or more entries",
    );
  });
});

describe("openMemoryContainer: matching the supply against the synthesized index", () => {
  it("fails when a whole supply's byteLength differs from the declared payload (too long)", () => {
    assertThrows(
      () =>
        openWith(baseDeclaration(), {
          ...baseSupply(),
          w: { bytes: f32Bytes(new Array(13).fill(0.5)), encoding: { codec: "f32" } },
        }),
      ContainerFormatError,
      "payload 48 bytes + padding",
    );
  });

  it("fails when a whole supply's byteLength differs from the declared payload (too short)", () => {
    assertThrows(
      () =>
        openWith(baseDeclaration(), {
          ...baseSupply(),
          w: { bytes: f32Bytes(new Array(11).fill(0.5)), encoding: { codec: "f32" } },
        }),
      ContainerFormatError,
      "payload 48 bytes + padding",
    );
  });

  it("fails when the scale's byteLength differs from the declaration's group shape", () => {
    assertThrows(
      () =>
        openWith(quantizedDeclaration(), {
          q: {
            bytes: qPayload(),
            encoding: {
              codec: "int8-sym",
              rowAxis: 0,
              groupSize: ROW_LENGTH,
              scale: f32Bytes(new Array(ROWS + 1).fill(0.5)),
            },
          },
          b: biasTensor(),
        }),
      ContainerFormatError,
      "payload 16 bytes + padding",
    );
  });

  it("fails when the scale's byteOffset is not 4-byte aligned", () => {
    // An unaligned view turns into a bare RangeError at the moment `session-build` creates a
    // Float32Array over it, disguised as a message of the transfer layer (a rule the supplier
    // guarantees).
    const backing = new Uint8Array(new ArrayBuffer(ROWS * 4 + 1));
    assertThrows(
      () =>
        openWith(quantizedDeclaration(), {
          q: {
            bytes: qPayload(),
            encoding: {
              codec: "int8-sym",
              rowAxis: 0,
              groupSize: ROW_LENGTH,
              scale: backing.subarray(1),
            },
          },
          b: biasTensor(),
        }),
      ContainerFormatError,
      "scale byteOffset 1 is not 4-byte aligned",
    );
  });

  it("fails when synthesized block ids collide (no silent last-wins overwrite)", () => {
    // An initializer name containing `#scale` takes the same id as another initializer's companion
    // scale.
    const graph = quantizedDeclaration();
    graph.initializers["q#scale"] = {};
    graph.values["q#scale"] = { dtype: "f32", shape: [ROWS] };
    assertThrows(
      () =>
        openWith(graph, {
          q: {
            bytes: qPayload(),
            encoding: {
              codec: "int8-sym",
              rowAxis: 0,
              groupSize: ROW_LENGTH,
              scale: qScale(),
            },
          },
          "q#scale": biasTensor(),
          b: biasTensor(),
        }),
      ContainerFormatError,
      "is bound twice",
    );
  });

  it("readBlock fails when a piece's read() returns bytes of a length other than the declared one", async () => {
    const payload = qPayload();
    const container = openWith(quantizedDeclaration(), {
      q: {
        encoding: { codec: "int8-sym", rowAxis: 0, groupSize: ROW_LENGTH, scale: qScale() },
        pieces: [
          // Piece 1 claims the row range [0,2) (= 32 B) but returns the full 64 B. The merge layer
          // compares derived values with each other, so its check is a tautology, and the
          // truncation in `containerBatches` never reaches the check either.
          { rows: [0, 2], read: () => Promise.resolve(payload) },
          {
            rows: [2, ROWS],
            read: () => Promise.resolve(payload.subarray(2 * ROW_LENGTH)),
          },
        ],
      },
      b: biasTensor(),
    });
    await assertRejects(
      () => container.readBlock(`${GRAPH_NAME}/q#piece1`),
      ContainerFormatError,
      "fetched length 64 differs from the declared 32",
    );
  });
});

describe("openMemoryContainer: checks delegated to the merge layer", () => {
  it("fails when the row ranges of the piece list have a gap", () => {
    assertThrows(
      () =>
        openWith(quantizedDeclaration(), {
          q: quantizedPieces([
            [0, 1],
            [2, ROWS],
          ]),
          b: biasTensor(),
        }),
      ContainerFormatError,
      "does not continue from row 1",
    );
  });

  it("fails when the end of the piece list falls short of the leading dimension of the declared shape", () => {
    assertThrows(
      () =>
        openWith(quantizedDeclaration(), {
          q: quantizedPieces([
            [0, 2],
            [2, 3],
          ]),
          b: biasTensor(),
        }),
      ContainerFormatError,
      "falls short of / exceeds the leading dimension of the declared shape, 4 rows",
    );
  });

  it("fails when the end of the piece list exceeds the leading dimension of the declared shape", () => {
    // Relaxing the end match to `<` would silently let a supply longer than the declaration
    // through.
    assertThrows(
      () =>
        openWith(quantizedDeclaration(), {
          q: quantizedPieces([
            [0, 2],
            [2, ROWS + 1],
          ]),
          b: biasTensor(),
        }),
      ContainerFormatError,
      "ends at row 5, which falls short of / exceeds the leading dimension of the declared shape, 4 rows",
    );
  });

  it("fails when a piece's row range is empty", () => {
    // [2,2) satisfies both continuity (start = previous end) and the end match, so only the
    // empty-range check catches it.
    assertThrows(
      () =>
        openWith(quantizedDeclaration(), {
          q: quantizedPieces([
            [0, 2],
            [2, 2],
            [2, ROWS],
          ]),
          b: biasTensor(),
        }),
      ContainerFormatError,
      "piece[1]: row range [2, 2) is an empty range",
    );
  });

  it("fails the non-negativity check when a piece's row range starts below zero", () => {
    assertThrows(
      () =>
        openWith(quantizedDeclaration(), {
          q: quantizedPieces([
            [-1, 2],
            [2, ROWS],
          ]),
          b: biasTensor(),
        }),
      ContainerFormatError,
      "piece[0]: row range start -1 is not a non-negative safe integer",
    );
  });

  it("fails when a piece's row range is not an integer", () => {
    // A fractional row boundary slips through every coverage check (empty range, continuity, end)
    // and then puts the byteOffset of GPU construction at a position that is not a multiple of 4.
    assertThrows(
      () =>
        openWith(quantizedDeclaration(), {
          q: quantizedPieces([
            [0, 1.5],
            [1.5, ROWS],
          ]),
          b: biasTensor(),
        }),
      ContainerFormatError,
      "is not a non-negative safe integer",
    );
  });

  it("fails when a middle piece's byte length is not a multiple of 4", () => {
    // Splitting an f16 [4,3] (6 bytes per row) after the first row leaves a middle piece of 6 bytes
    // that cannot be padded. Without failing here, the failure is delayed until GPU writeBuffer
    // validation.
    const bytes = new Uint8Array(new ArrayBuffer(24));
    assertThrows(
      () =>
        openWith(baseDeclaration(), {
          ...baseSupply(),
          w: {
            encoding: { codec: "f16" },
            pieces: [
              { rows: [0, 1], read: () => Promise.resolve(bytes.subarray(0, 6)) },
              { rows: [1, 4], read: () => Promise.resolve(bytes.subarray(6)) },
            ],
          },
        }),
      ContainerFormatError,
      "byte count 6 of a middle piece's row range is not a multiple of 4",
    );
  });

  it("fails when the pair of semantic dtype and codec is mismatched", () => {
    const graph = baseDeclaration();
    graph.values["w"] = { dtype: "i32", shape: [4, 3] };
    assertThrows(
      () => openWith(graph, baseSupply()),
      ContainerFormatError,
      "semantic dtype 'i32' cannot be paired with codec 'f32'",
    );
  });

  it("fails when a quantized codec has no scale / groupSize", () => {
    assertThrows(
      () =>
        openWith(quantizedDeclaration(), {
          q: { bytes: qPayload(), encoding: { codec: "int8-sym" } },
          b: biasTensor(),
        }),
      ContainerFormatError,
      "codec 'int8-sym' needs scale and groupSize",
    );
  });

  it("fails when a non-quantized codec carries scale / groupSize / rowAxis", () => {
    assertThrows(
      () =>
        openWith(baseDeclaration(), {
          ...baseSupply(),
          b: { bytes: f32Bytes([1, 2, 3]), encoding: { codec: "f32", scale: qScale() } },
        }),
      ContainerFormatError,
      "codec 'f32' cannot have scale / groupSize / rowAxis",
    );
  });
});

// --- rowAxis per codec ---------------------------------------------------------
// Expansion (`decodeI4` / `decodeI2` and the WGSL) does not take the declared axis and is fixed to
// axis 0, so for an `[N,N]` with both axes the same length the scale-shape match passes. Only the
// merge layer looks at the declared axis.

/**
 * A graph that consumes `q` (f32 semantics) in one op. Rank 2 is linear, rank 3 is
 * conv_transpose1d.
 */
const weightDeclaration = (shape: readonly number[]): DeclarationJson => {
  const common = {
    format: "karume-ir",
    version: 2,
    symbols: [],
    outputs: ["y"],
    initializers: { q: {}, b: {} },
  };
  if (shape.length === 2) {
    const [rows, width] = shape;
    return {
      ...common,
      requires: { ops: ["linear"] },
      inputs: [{ name: "x", dtype: "f32", shape: [2, width] }],
      values: {
        q: { dtype: "f32", shape: [...shape] },
        b: { dtype: "f32", shape: [rows] },
        y: { dtype: "f32", shape: [2, rows] },
      },
      nodes: [{ op: "linear", ins: ["x", "q", "b"], outs: ["y"], attrs: {} }],
    };
  }
  const [channelsIn, channelsOut, kernel] = shape;
  const length = 5;
  return {
    ...common,
    requires: { ops: ["conv_transpose1d"] },
    inputs: [{ name: "x", dtype: "f32", shape: [1, channelsIn, length] }],
    values: {
      q: { dtype: "f32", shape: [...shape] },
      b: { dtype: "f32", shape: [channelsOut] },
      y: { dtype: "f32", shape: [1, channelsOut, length - 1 + kernel] },
    },
    nodes: [
      {
        op: "conv_transpose1d",
        ins: ["x", "q", "b"],
        outs: ["y"],
        attrs: { stride: 1, padding: 0 },
      },
    ],
  };
};

/**
 * A quantized `q` with `rowAxis: 1` and an f32 bias (zero-filled, since its contents are not looked
 * at).
 */
const rowAxisOneSupply = (input: {
  readonly codec: "int8-sym" | "int4-sym-g" | "int2-off";
  readonly payloadBytes: number;
  readonly groupSize: number;
  readonly scaleElements: number;
  readonly biasElements: number;
}): Record<string, MemoryTensor> => ({
  q: {
    bytes: new Uint8Array(input.payloadBytes),
    encoding: {
      codec: input.codec,
      rowAxis: 1,
      groupSize: input.groupSize,
      scale: new Uint8Array(input.scaleElements * 4),
    },
  },
  b: { bytes: new Uint8Array(input.biasElements * 4), encoding: { codec: "f32" } },
});

describe("openMemoryContainer: rowAxis per codec (merge layer)", () => {
  it("int4-sym-g with rowAxis 1 fails even for a shape whose two axes have the same length", () => {
    // [32,32], group 16: the scale is [32,2] whether read as axis 0 or axis 1, so the shape match
    // cannot tell them apart.
    assertThrows(
      () =>
        openWith(
          weightDeclaration([32, 32]),
          rowAxisOneSupply({
            codec: "int4-sym-g",
            payloadBytes: (32 * 32) / 2,
            groupSize: 16,
            scaleElements: 32 * 2,
            biasElements: 32,
          }),
        ),
      ContainerFormatError,
      "rowAxis of codec 'int4-sym-g' must be 0 only (declared 1",
    );
  });

  it("int2-off with rowAxis 1 fails at the merge layer, not at expansion time", () => {
    assertThrows(
      () =>
        openWith(
          weightDeclaration([16, 16]),
          rowAxisOneSupply({
            codec: "int2-off",
            payloadBytes: (16 * 16) / 4,
            groupSize: 16,
            scaleElements: 16,
            biasElements: 16,
          }),
        ),
      ContainerFormatError,
      "rowAxis of codec 'int2-off' must be 0 only (declared 1",
    );
  });

  it("accepts int8-sym with rowAxis 1 (the conv_transpose1d shape)", () => {
    // The rows of [Cin=4, Cout=8, K=3] are axis 1 (8 rows, row length 12).
    const opened = openWith(
      weightDeclaration([4, 8, 3]),
      rowAxisOneSupply({
        codec: "int8-sym",
        payloadBytes: 4 * 8 * 3,
        groupSize: 12,
        scaleElements: 8,
        biasElements: 8,
      }),
    );
    assertEquals(opened.graphs[GRAPH_NAME].supplies.get("q")?.encoding.rowAxis, 1);
  });
});

// --- Quantized weights with zero rows -----------------------------------------
// A weight with no quantized row has no use in a real model and is rejected at the merge layer. If
// the merge layer accepted it, a downstream that owns its own row-length formula (the CPU expansion
// `decodeI4`'s `numel / shape[0]` = 0 / 0) would fail at another layer with another message
// ("row length NaN").

describe("openMemoryContainer: quantized weights with zero rows (merge layer)", () => {
  it("int4-sym-g [0,16] fails at the merge layer, not at CPU expansion time", () => {
    assertThrows(
      () =>
        openWith(weightDeclaration([0, 16]), {
          q: {
            bytes: new Uint8Array(0),
            encoding: { codec: "int4-sym-g", groupSize: 16, scale: new Uint8Array(0) },
          },
          b: { bytes: new Uint8Array(0), encoding: { codec: "f32" } },
        }),
      ContainerFormatError,
      "initializer 'q': quantized weights with 0 rows (shape[0] = 0) are not accepted",
    );
  });

  it("int8-sym with rowAxis 1 also fails for [4,0,3], whose row axis is 0", () => {
    assertThrows(
      () =>
        openWith(
          weightDeclaration([4, 0, 3]),
          rowAxisOneSupply({
            codec: "int8-sym",
            payloadBytes: 0,
            groupSize: 1,
            scaleElements: 0,
            biasElements: 0,
          }),
        ),
      ContainerFormatError,
      "initializer 'q': quantized weights with 0 rows (shape[1] = 0) are not accepted",
    );
  });
});

describe("openMemoryContainer: the synthesized index", () => {
  it("returns a whole tensor's backing as passed (no copy)", async () => {
    const tensors = baseTensors();
    const container = memoryModel(baseDeclaration(), tensors);
    assertStrictEquals(await container.readBlock(`${GRAPH_NAME}/w`), tensors[0].bytes);
    assertStrictEquals(await container.readBlock(`${GRAPH_NAME}/b`), tensors[1].bytes);
  });

  it("re-runs a piece's read() on every readBlock (does not keep it)", async () => {
    const reads: number[] = [];
    const container = openWith(quantizedDeclaration(), {
      q: quantizedPieces(
        [
          [0, 2],
          [2, ROWS],
        ],
        (index) => reads.push(index),
      ),
      b: biasTensor(),
    });
    assertEquals(reads, []);
    await container.readBlock(`${GRAPH_NAME}/q#piece1`);
    await container.readBlock(`${GRAPH_NAME}/q#piece1`);
    await container.readBlock(`${GRAPH_NAME}/q#piece2`);
    assertEquals(reads, [0, 0, 1]);
  });

  it("fails loudly on an unknown block id", async () => {
    const container = memoryModel();
    await assertRejects(
      () => container.readBlock(`${GRAPH_NAME}/missing`),
      ContainerFormatError,
      "Undeclared block",
    );
  });

  it("whole tensors and scales go in part 2; pieces get one part each from the one after 2", () => {
    const container = openWith(quantizedDeclaration(), {
      q: quantizedPieces([
        [0, 2],
        [2, ROWS],
      ]),
      b: biasTensor(),
    });
    const supplies = container.graphs[GRAPH_NAME].supplies;
    const q = supplies.get("q");
    assertEquals(
      q?.blocks.map((block) => [block.id, block.part, block.length]),
      [
        [`${GRAPH_NAME}/q#piece1`, 3, 2 * ROW_LENGTH],
        [`${GRAPH_NAME}/q#piece2`, 4, 2 * ROW_LENGTH],
      ],
    );
    // Rule ③: the companion scale is in the same part as piece 1.
    assertEquals(
      [q?.scale?.id, q?.scale?.part, q?.scale?.length],
      [`${GRAPH_NAME}/q#scale`, 3, ROWS * 4],
    );
    const b = supplies.get("b");
    assertEquals(
      b?.blocks.map((block) => [block.id, block.part, block.length]),
      [[`${GRAPH_NAME}/b`, 2, ROWS * 4]],
    );
  });

  it("a supply of whole tensors only goes entirely into part 2", () => {
    const supplies = memoryModel().graphs[GRAPH_NAME].supplies;
    assertEquals(
      [...supplies].map(([name, supply]) => [name, supply.blocks[0].part]),
      [
        ["w", 2],
        ["b", 2],
      ],
    );
  });

  it("whole supplies move to the next part where the running total exceeds the part length", () => {
    // The part assignment = the cap on staging VRAM during construction (`session-build` submits
    // and waits for completion per part). Putting everything in one part would disable that measure
    // only for models with whole supplies.
    const CHUNK = 96 * 1024 * 1024;
    const names = ["a", "b", "c"] as const;
    const graph: DeclarationJson = {
      format: "karume-ir",
      version: 2,
      requires: { ops: ["add"] },
      symbols: [],
      inputs: [{ name: "x", dtype: "f32", shape: [1] }],
      outputs: ["y"],
      initializers: Object.fromEntries(names.map((name) => [name, {}])),
      values: {
        ...Object.fromEntries(names.map((name) => [name, { dtype: "f32", shape: [CHUNK / 4] }])),
        y: { dtype: "f32", shape: [1] },
      },
      nodes: [{ op: "add", ins: ["x", "x"], outs: ["y"], attrs: {} }],
    };
    const bigTensor = (): MemoryTensor => ({
      bytes: new Uint8Array(new ArrayBuffer(CHUNK)),
      encoding: { codec: "f32" },
    });
    const supplies = openWith(graph, Object.fromEntries(names.map((name) => [name, bigTensor()])))
      .graphs[GRAPH_NAME].supplies;
    // 96 MiB × 2 fits in 256 MiB, so they share a part; the third overflows into the next part.
    assertEquals(
      [...supplies].map(([name, supply]) => [name, supply.blocks[0].part]),
      [
        ["a", 2],
        ["b", 2],
        ["c", 3],
      ],
    );
  });
});

describe("BoundContainer: A/B of the krm path and the memory path", () => {
  it("an OpenedContainer can be passed to prepareContainer as a BoundContainer", async () => {
    const opened = await openModelBytes();
    const prepared = prepareContainer(opened, GRAPH_NAME);
    assertEquals(prepared.graph.initializers["w"], { storage: { codec: "f32" } });
  });

  it("the same input gives the same merged graph and estimate", async () => {
    const fromContainer = prepareContainer(await openModelBytes(), GRAPH_NAME);
    const fromMemory = prepareContainer(memoryModel(), GRAPH_NAME);
    assertEquals(fromMemory.graph, fromContainer.graph);
    assertEquals(
      fromMemory.estimate({ bindings: { T: 2 } }),
      fromContainer.estimate({
        bindings: { T: 2 },
      }),
    );
  });
});

/**
 * Temporarily redefines `Object.prototype.__proto__` as the spec's accessor (assignment replaces
 * the prototype — the browser default), runs `body`, and always restores the original descriptor.
 * Deno's default setter creates an own property, so on bare Deno a defect of the form "assigning
 * to the name `__proto__` replaces the prototype" is not visible.
 */
const withBrowserProtoAccessor = <T>(body: () => T): T => {
  const original = Object.getOwnPropertyDescriptor(Object.prototype, "__proto__");
  Object.defineProperty(Object.prototype, "__proto__", {
    configurable: true,
    enumerable: false,
    get(this: object) {
      return Object.getPrototypeOf(this);
    },
    set(this: object, value: unknown) {
      if (typeof value === "object" || value === null) Object.setPrototypeOf(this, value);
    },
  });
  try {
    return body();
  } finally {
    if (original === undefined) {
      delete (Object.prototype as { __proto__?: unknown }).__proto__;
    } else {
      Object.defineProperty(Object.prototype, "__proto__", original);
    }
  }
};

describe("openMemoryContainer: the name '__proto__'", () => {
  it("keeps graph and initializer names as own properties even under the browser's __proto__ accessor", () => {
    // The declaration's own keys MUST be created with JSON.parse (`"__proto__":` in an object
    // literal sets [[Prototype]]).
    const declaration = declarationOf(
      JSON.parse(`{
      "format": "karume-ir",
      "version": 2,
      "requires": { "ops": ["matmul"] },
      "symbols": [],
      "inputs": [{ "name": "x", "dtype": "f32", "shape": [2, 4] }],
      "outputs": ["y"],
      "initializers": { "__proto__": {} },
      "values": {
        "__proto__": { "dtype": "f32", "shape": [4, 3] },
        "y": { "dtype": "f32", "shape": [2, 3] }
      },
      "nodes": [{ "op": "matmul", "ins": ["x", "__proto__"], "outs": ["y"], "attrs": {} }]
    }`),
    );
    const tensors: Record<string, MemoryTensor> = JSON.parse('{"__proto__":null}');
    tensors["__proto__"] = { bytes: f32Bytes(new Array(12).fill(1)), encoding: { codec: "f32" } };
    const opened = withBrowserProtoAccessor(() =>
      openMemoryContainer({
        graphs: { ["__proto__"]: declaration },
        tensors: { ["__proto__"]: tensors },
      }),
    );
    assertEquals(Object.hasOwn(opened.graphs, "__proto__"), true);
    assertEquals(opened.graphs["__proto__"].supplies.has("__proto__"), true);
  });
});
