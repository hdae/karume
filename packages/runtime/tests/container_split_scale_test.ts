/**
 * Split scales — a quantized binding's scale split on the data's pieces when one block cannot hold
 * it (docs/container-v1.md §5 rule ③, ADR 0108 addendum 15).
 *
 * What is pinned here: the writer's split lands on the data's piece boundaries, the reader hands
 * construction **one contiguous scale** under the joined id (`readBlock` and `containerBatches`), a
 * scale that fits keeps the one-block spelling, and every reader rule rejects its broken form —
 * structure (gap / overlap / one entry / both keys) and placement (part / role / block limit) on the
 * opened container, alignment (count / rows / lengths) in the merge layer on hand-built inputs.
 * The cross-language case (a Python-written split scale) is in `container_fixture_test.ts`.
 */

import { assert, assertEquals, assertRejects, assertThrows } from "@std/assert";
import { describe, it } from "vite-plus/test";
import { bindDeclarations, joinedScaleId } from "../src/format/container/bind.ts";
import { codecEntry } from "../src/format/container/codecs.ts";
import {
  parseGraphDescriptor,
  parseModelDescriptor,
  type WeightSupply,
} from "../src/format/container/descriptor.ts";
import { ContainerFormatError, writeHeader } from "../src/format/container/header.ts";
import { BLOCK_MAX_BYTES, HEADER_BYTES } from "../src/format/container/limits.ts";
import { openContainer } from "../src/format/container/open.ts";
import { type IrDeclaration, parseIrDeclaration } from "../src/format/ir.ts";
import { containerBatches } from "../src/runtime/executor.ts";
import {
  type ModelInput,
  writeModelContainer,
  type WrittenContainer,
} from "./helpers/container-write.ts";

/** Bytes rising by 1 from `seed` (a mix-up between tensors shows as a different seed). */
const patternBytes = (length: number, seed: number): Uint8Array<ArrayBuffer> =>
  Uint8Array.from({ length }, (_, i) => (seed + i) & 0xff);

/** A minimal IR v2 graph consuming each initializer with one node. */
const declaration = (shapes: Readonly<Record<string, readonly number[]>>): IrDeclaration => {
  const names = Object.keys(shapes);
  return parseIrDeclaration(
    JSON.stringify({
      format: "karume-ir",
      version: 2,
      requires: { ops: ["matmul"] },
      symbols: ["T"],
      inputs: [{ name: "x", dtype: "f32", shape: ["T", 4] }],
      outputs: names.map((_, i) => `y${i}`),
      initializers: Object.fromEntries(names.map((name) => [name, {}])),
      values: {
        ...Object.fromEntries(names.map((name) => [name, { dtype: "f32", shape: shapes[name] }])),
        ...Object.fromEntries(names.map((_, i) => [`y${i}`, { dtype: "f32", shape: ["T", 4] }])),
      },
      nodes: names.map((name, i) => ({
        op: "matmul",
        ins: ["x", name],
        outs: [`y${i}`],
        attrs: {},
      })),
    }),
  );
};

// `w`: i4 [16, 64], group 16 — 512 B of data (4 pieces of 4 rows at the 128 B block limit) and
// 256 B of scale (over the limit, so split into 4 pieces of 64 B). `v`: i8 [8, 32] — 256 B of data
// (2 pieces) and 32 B of scale (fits, so one block).
const W_SCALE = patternBytes(16 * (64 / 16) * 4, 31);
const V_SCALE = patternBytes(8 * 4, 41);
const OPTIONS = { partBytes: 1024, blockBytes: 128 } as const;

const model = (): ModelInput => ({
  graphs: { g: declaration({ w: [16, 64], v: [8, 32] }) },
  consts: [],
  weights: [
    {
      graph: "g",
      initializer: "w",
      bytes: patternBytes((16 * 64) / 2, 21),
      encoding: { codec: "int4-sym-g", groupSize: 16, scale: { bytes: W_SCALE, dtype: "f32" } },
    },
    {
      graph: "g",
      initializer: "v",
      bytes: patternBytes(8 * 32, 22),
      encoding: { codec: "int8-sym", groupSize: 32, scale: { bytes: V_SCALE, dtype: "f32" } },
    },
  ],
  assets: [],
  provenance: { license: "apache-2.0", writer: "test" },
});

const mutateJson = (
  bytes: Uint8Array<ArrayBuffer>,
  mutate: (doc: Record<string, unknown>) => void,
): Uint8Array<ArrayBuffer> => {
  const doc = JSON.parse(new TextDecoder().decode(bytes)) as Record<string, unknown>;
  mutate(doc);
  return new TextEncoder().encode(JSON.stringify(doc)) as Uint8Array<ArrayBuffer>;
};

// deno-lint-ignore no-explicit-any
const anyOf = (value: unknown): any => value;

/** The split scale's descriptor entry of `w` (for mutations). */
const wScale = (doc: Record<string, unknown>) => anyOf(doc).binding.g.w.encoding.scale;

/** Re-opens a written container with only its model descriptor rewritten (part 0 rebuilt). */
const openWithModelDescriptor = async (
  written: WrittenContainer,
  mutate: (doc: Record<string, unknown>) => void,
): Promise<void> => {
  const parts = [...written.parts];
  const modelBytes = mutateJson(written.modelDescriptorBytes, mutate);
  const header = writeHeader({
    kind: "model",
    version: 1,
    graphDescriptorLength: written.graphDescriptorBytes.byteLength,
    modelDescriptorLength: modelBytes.byteLength,
  });
  const part0 = new Uint8Array(
    new ArrayBuffer(HEADER_BYTES + written.graphDescriptorBytes.byteLength + modelBytes.byteLength),
  );
  part0.set(header, 0);
  part0.set(written.graphDescriptorBytes, HEADER_BYTES);
  part0.set(modelBytes, HEADER_BYTES + written.graphDescriptorBytes.byteLength);
  parts[0] = part0;
  await openContainer({ kind: "parts", parts });
};

describe("a split scale (writer → reader)", () => {
  it("is split on the data's piece boundaries, all in the part of piece 1", async () => {
    const written = await writeModelContainer(model(), OPTIONS);
    const opened = await openContainer({ kind: "parts", parts: [...written.parts] });
    const w = opened.graphs["g"].supplies.get("w");
    assert(w?.scale?.pieces !== undefined, "the scale of w was not split");

    assertEquals(
      w.blocks.map((block) => block.rows),
      [
        [0, 4],
        [4, 8],
        [8, 12],
        [12, 16],
      ],
    );
    assertEquals(
      w.scale.pieces.map((piece) => piece.rows),
      w.blocks.map((block) => block.rows),
    );
    assertEquals(
      w.scale.pieces.map((piece) => [piece.length, piece.payloadBytes]),
      [
        [64, 64],
        [64, 64],
        [64, 64],
        [64, 64],
      ],
    );
    assertEquals(new Set(w.scale.pieces.map((piece) => piece.part)), new Set([w.blocks[0].part]));
  });

  it("is read whole under the joined id, which names no declared block", async () => {
    const written = await writeModelContainer(model(), OPTIONS);
    for (const opened of [
      await openContainer({ kind: "parts", parts: [...written.parts] }),
      await openContainer({ kind: "bytes", bytes: written.single }),
    ]) {
      const scale = opened.graphs["g"].supplies.get("w")?.scale;
      assert(scale?.pieces !== undefined);

      assertEquals(scale.id, joinedScaleId(scale.pieces.map((piece) => piece.id)));
      assertEquals([scale.payloadBytes, scale.length, scale.rows], [256, 256, [0, 16]]);
      assertEquals(await opened.readBlock(scale.id), W_SCALE);
      assertThrows(() => opened.locate(scale.id), ContainerFormatError, "Undeclared block");
    }
  });

  it("keeps a scale that fits one block in the one-block spelling", async () => {
    const written = await writeModelContainer(model(), OPTIONS);
    const doc = anyOf(JSON.parse(new TextDecoder().decode(written.modelDescriptorBytes)));

    assertEquals(Object.keys(doc.binding.g.v.encoding.scale), ["block", "dtype"]);
    assertEquals(Object.keys(doc.binding.g.w.encoding.scale), ["pieces", "dtype"]);
    const opened = await openContainer({ kind: "parts", parts: [...written.parts] });
    const v = opened.graphs["g"].supplies.get("v");
    assertEquals(v?.blocks.length, 2);
    assertEquals(v?.scale?.pieces, undefined);
    assertEquals(await opened.readBlock(v?.scale?.id ?? ""), V_SCALE);
  });

  it("hands construction the whole scale with piece 1 (containerBatches)", async () => {
    const written = await writeModelContainer(model(), OPTIONS);
    const opened = await openContainer({ kind: "parts", parts: [...written.parts] });
    const items: { name: string; scale?: { bytes: Uint8Array; shape: readonly number[] } }[] = [];
    for await (const batch of containerBatches(opened, "g")) {
      for await (const item of batch.items) items.push(item);
    }
    const w = items.filter((item) => item.name === "w");

    assertEquals(w.length, 4);
    assertEquals(w[0].scale?.bytes, W_SCALE);
    assertEquals(w[0].scale?.shape, [16, 4]);
    assertEquals(
      w.slice(1).map((item) => item.scale),
      [undefined, undefined, undefined],
    );
  });

  it("fails the joined read when one piece's bytes differ from its sha256", async () => {
    const written = await writeModelContainer(model(), OPTIONS);
    const opened = await openContainer({ kind: "parts", parts: [...written.parts] });
    const scale = opened.graphs["g"].supplies.get("w")?.scale;
    const third = scale?.pieces?.[2];
    assert(scale !== undefined && third !== undefined);
    const parts = written.parts.map((part) => part.slice());
    parts[third.part][third.offset] ^= 0xff;
    // The descriptor is unchanged, so the tampered container has the same joined id.
    const tampered = await openContainer({ kind: "parts", parts });

    await assertRejects(
      () => tampered.readBlock(scale.id),
      ContainerFormatError,
      `block '${third.id}' sha256 differs`,
    );
  });
});

describe("the reader rejects a broken split scale (opened container)", () => {
  it("a gap between pieces", async () => {
    const written = await writeModelContainer(model(), OPTIONS);
    await assertRejects(
      () =>
        openWithModelDescriptor(written, (doc) => {
          wScale(doc).pieces[1].rows = [5, 8];
        }),
      ContainerFormatError,
      "scale.pieces[1]: does not continue from row 4",
    );
  });

  it("overlapping pieces", async () => {
    const written = await writeModelContainer(model(), OPTIONS);
    await assertRejects(
      () =>
        openWithModelDescriptor(written, (doc) => {
          wScale(doc).pieces[1].rows = [3, 8];
        }),
      ContainerFormatError,
      "scale.pieces[1]: does not continue from row 4",
    );
  });

  it("rows that do not match the data's pieces", async () => {
    const written = await writeModelContainer(model(), OPTIONS);
    await assertRejects(
      () =>
        openWithModelDescriptor(written, (doc) => {
          wScale(doc).pieces[0].rows = [0, 2];
          wScale(doc).pieces[1].rows = [2, 8];
        }),
      ContainerFormatError,
      "rows [0, 2) differ from the rows [0, 4) of data piece 0",
    );
  });

  it("a piece over the block limit", async () => {
    const written = await writeModelContainer(model(), OPTIONS);
    await assertRejects(
      () =>
        openWithModelDescriptor(written, (doc) => {
          const id = wScale(doc).pieces[3].block;
          const block = anyOf(doc).blocks.find((entry: { id: string }) => entry.id === id);
          block.length = BLOCK_MAX_BYTES + 4;
        }),
      ContainerFormatError,
      `exceeds the block limit ${BLOCK_MAX_BYTES}`,
    );
  });

  it("a piece list of one entry, and both or neither of block / pieces", async () => {
    const written = await writeModelContainer(model(), OPTIONS);
    await assertRejects(
      () =>
        openWithModelDescriptor(written, (doc) => {
          wScale(doc).pieces = [{ block: wScale(doc).pieces[0].block, rows: [0, 16] }];
        }),
      ContainerFormatError,
      "scale.pieces must have 2 or more entries",
    );
    await assertRejects(
      () =>
        openWithModelDescriptor(written, (doc) => {
          wScale(doc).block = wScale(doc).pieces[0].block;
        }),
      ContainerFormatError,
      "write exactly one of 'block' (one block) and 'pieces' (split scale)",
    );
    await assertRejects(
      () =>
        openWithModelDescriptor(written, (doc) => {
          delete wScale(doc).pieces;
        }),
      ContainerFormatError,
      "write exactly one of 'block' (one block) and 'pieces' (split scale)",
    );
  });

  it("a piece in another part than piece 1 of the data (rule ③)", async () => {
    const written = await writeModelContainer(model(), OPTIONS);
    assertThrows(
      () =>
        parseModelDescriptor(
          mutateJson(written.modelDescriptorBytes, (doc) => {
            // Move the last scale piece to a fresh tail of the last part (only the declarations are
            // read here, so lengthening the part is enough to keep the block inside it).
            const id = wScale(doc).pieces[3].block;
            const block = anyOf(doc).blocks.find((entry: { id: string }) => entry.id === id);
            const last = anyOf(doc).parts.at(-1);
            block.part = last.index;
            block.offset = Math.ceil(last.length / 64) * 64;
            last.length = block.offset + block.length;
          }),
        ),
      ContainerFormatError,
      "encoding.scale.pieces[3]: scale is in part 3, the first data block in part 2 (same part MUST)",
    );
  });

  it("a piece whose block is not a scale block", async () => {
    const written = await writeModelContainer(model(), OPTIONS);
    await assertRejects(
      () =>
        openWithModelDescriptor(written, (doc) => {
          const id = wScale(doc).pieces[2].block;
          anyOf(doc).blocks.find((entry: { id: string }) => entry.id === id).role = "weight";
        }),
      ContainerFormatError,
      "has role 'weight' ('scale' is required)",
    );
  });

  it("a split scale on a const", async () => {
    const input: ModelInput = {
      ...model(),
      graphs: { g: declaration({ w: [16, 64], v: [8, 32], "const.k": [4, 8] }) },
      consts: [
        {
          graph: "g",
          initializer: "const.k",
          bytes: patternBytes(4 * 8 * 4, 51),
          encoding: { codec: "f32" },
        },
      ],
    };
    const written = await writeModelContainer(input, OPTIONS);
    assertThrows(
      () =>
        parseGraphDescriptor(
          mutateJson(written.graphDescriptorBytes, (doc) => {
            const entry = anyOf(doc).const.constants[0];
            const id = entry.block;
            entry.encoding = {
              codec: "int8-sym",
              packing: { blockElements: 1, blockBytes: 1, alignBytes: 4 },
              rowAxis: 0,
              groupSize: 8,
              scale: {
                pieces: [
                  { block: id, rows: [0, 2] },
                  { block: id, rows: [2, 4] },
                ],
                dtype: "f32",
              },
            };
          }),
        ),
      ContainerFormatError,
      "a const cannot have a split scale",
    );
  });
});

// The merge layer on hand-built inputs: `w` is i4 [8, 32], group 16 — data pieces [0, 4) / [4, 8) of
// 64 B each, a scale of 8 rows × 2 groups × 4 B = 64 B, so 32 B per scale piece.
const I4 = codecEntry("int4-sym-g");

const bindW = (supply: WeightSupply, lengths: Readonly<Record<string, number>>) =>
  bindDeclarations({
    graphs: { g: declaration({ w: [8, 32] }) },
    constants: [],
    binding: { g: { w: supply } },
    locate: (id, by) => {
      if (!Object.hasOwn(lengths, id)) {
        throw new ContainerFormatError(`${by}: undeclared block '${id}'`);
      }
      return { part: 2, record: { id, offset: 0, length: lengths[id] } };
    },
  })["g"].supplies.get("w");

const splitSupply = (
  scalePieces: readonly { block: string; rows: readonly [number, number] }[],
  data: "pieces" | "block" = "pieces",
): WeightSupply => {
  const encoding = {
    codec: "int4-sym-g" as const,
    packing: I4.packing,
    rowAxis: 0 as const,
    groupSize: 16,
    scale: { pieces: scalePieces, dtype: "f32" as const },
  };
  return data === "block"
    ? { block: "w", encoding }
    : {
        pieces: [
          { block: "w1", rows: [0, 4] },
          { block: "w2", rows: [4, 8] },
        ],
        encoding,
      };
};

const DATA = { w: 128, w1: 64, w2: 64 } as const;

describe("the merge layer's split-scale rules", () => {
  it("accepts one scale piece per data piece over the same rows", () => {
    const supply = bindW(
      splitSupply([
        { block: "s1", rows: [0, 4] },
        { block: "s2", rows: [4, 8] },
      ]),
      { ...DATA, s1: 32, s2: 32 },
    );

    assertEquals(supply?.scale?.id, "s1+s2");
    assertEquals(supply?.scale?.payloadBytes, 64);
    assertEquals(
      supply?.scale?.pieces?.map((piece) => [piece.rows, piece.payloadBytes]),
      [
        [[0, 4], 32],
        [[4, 8], 32],
      ],
    );
  });

  it("rejects a scale piece count that differs from the data's", () => {
    assertThrows(
      () =>
        bindW(
          splitSupply([
            { block: "s1", rows: [0, 2] },
            { block: "s2", rows: [2, 4] },
            { block: "s3", rows: [4, 8] },
          ]),
          { ...DATA, s1: 16, s2: 16, s3: 32 },
        ),
      ContainerFormatError,
      "3 scale pieces for 2 data pieces",
    );
  });

  it("rejects scale rows that are not the data piece's rows", () => {
    assertThrows(
      () =>
        bindW(
          splitSupply([
            { block: "s1", rows: [0, 2] },
            { block: "s2", rows: [2, 8] },
          ]),
          { ...DATA, s1: 16, s2: 48 },
        ),
      ContainerFormatError,
      "rows [0, 2) differ from the rows [0, 4) of data piece 0",
    );
  });

  it("rejects a middle piece with padding, and a last piece longer than its rows + padding", () => {
    const pieces = [
      { block: "s1", rows: [0, 4] },
      { block: "s2", rows: [4, 8] },
    ] as const;
    assertThrows(
      () => bindW(splitSupply(pieces), { ...DATA, s1: 36, s2: 32 }),
      ContainerFormatError,
      "block 's1' length 36 differs from the 32 bytes of its rows",
    );
    assertThrows(
      () => bindW(splitSupply(pieces), { ...DATA, s1: 32, s2: 36 }),
      ContainerFormatError,
      "block 's2' length 36 is not payload 32 bytes + padding",
    );
  });

  it("rejects a split scale beside data in one block", () => {
    assertThrows(
      () =>
        bindW(
          splitSupply(
            [
              { block: "s1", rows: [0, 4] },
              { block: "s2", rows: [4, 8] },
            ],
            "block",
          ),
          { ...DATA, s1: 32, s2: 32 },
        ),
      ContainerFormatError,
      "a split scale needs a piece list",
    );
  });
});
