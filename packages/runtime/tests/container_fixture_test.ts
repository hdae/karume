/**
 * Cross-language fixture — opens with the TS reader a `krm` written by the **Python writer**
 * (`tools/exporter/src/karume/container.py`). docs/container-v1.md §7 / §9, docs/ir-v2.md
 * "Canonical serialization".
 *
 * What this checks is not that "the TS writer and the TS reader agree with each other" but that **a
 * byte stream written by another implementation opens** (`container_format_test.ts` checks the
 * former, and a round trip alone cannot detect a disagreement between the two implementations).
 *
 * The assets are regenerated with one command on the exporter side:
 * `cd tools/exporter && KARUME_FIXTURE=write uv run pytest tests/test_container.py -k fixture`
 */

import { readFile } from "@karume/test-support/fs";
import { assert, assertEquals } from "@std/assert";
import { describe, it } from "vite-plus/test";
import {
  parseGraphDescriptor,
  parseModelDescriptor,
  serializeGraphDescriptor,
  serializeModelDescriptor,
  sha256Hex,
} from "../src/format/container/descriptor.ts";
import { readHeader } from "../src/format/container/header.ts";
import { HEADER_BYTES } from "../src/format/container/limits.ts";
import { openContainer } from "../src/format/container/open.ts";

const FIXTURE = new URL("./fixtures/container/synthetic.krm", import.meta.url);

/** Part count of the split-form fixture (from part 0 — const is listed as a zero-length file). */
const SPLIT_FIXTURE_PARTS = 5;
const splitFixture = (index: number): URL =>
  new URL(
    `./fixtures/container/synthetic-split-${String(index + 1).padStart(5, "0")}-of-${String(
      SPLIT_FIXTURE_PARTS,
    ).padStart(5, "0")}.krm`,
    import.meta.url,
  );

const fixtureBytes = async (): Promise<Uint8Array<ArrayBuffer>> =>
  (await readFile(FIXTURE)) as Uint8Array<ArrayBuffer>;

const splitParts = async (): Promise<Uint8Array<ArrayBuffer>[]> =>
  await Promise.all(
    Array.from(
      { length: SPLIT_FIXTURE_PARTS },
      async (_, i) => (await readFile(splitFixture(i))) as Uint8Array<ArrayBuffer>,
    ),
  );

/**
 * The fixture payload (the same **simple** pattern as `container_fixture.pattern_bytes` on the
 * Python side — bytes that count up by 1 from `seed`). It is a one-line form, not a copy of the
 * generator, so regenerating the fixture leaves no room for a red caused by "the mirror drifted".
 */
const patternBytes = (length: number, seed: number): Uint8Array<ArrayBuffer> =>
  Uint8Array.from({ length }, (_, i) => (seed + i) & 0xff);

describe("container fixture (Python writer)", () => {
  it("opens the single form and every block comes back with its sha256 verified", async () => {
    const bytes = await fixtureBytes();
    const container = await openContainer({ kind: "bytes", bytes });

    assertEquals(container.header.kind, "model");
    assertEquals(container.header.version, 1);
    assertEquals(Object.keys(container.graphs), ["synthetic"]);
    assertEquals(container.model?.provenance, { license: "apache-2.0", writer: "karume" });

    const ids = [
      ...container.graph.const.blocks.map((block) => block.id),
      ...(container.model?.blocks ?? []).map((block) => block.id),
    ];
    assert(ids.length > 0, "no block at all");
    for (const id of ids) {
      // readBlock checks against the declared sha256, so a returned block is already verified.
      const block = await container.readBlock(id);
      assertEquals(await sha256Hex(block), container.locate(id).record.sha256, id);
    }
  });

  it("the merge layer derives the supply plan from the Python binding table (piece list / rowAxis 1 / const / shared)", async () => {
    const bytes = await fixtureBytes();
    const container = await openContainer({ kind: "bytes", bytes });
    const bound = container.graphs["synthetic"];

    // A shared declaration is outside the set matched against the binding table (it has no supply
    // plan).
    assertEquals(bound.declaration.initializers["lm_head.weight"], { shared: true });
    assertEquals(bound.supplies.has("lm_head.weight"), false);

    const enc = bound.supplies.get("enc.weight");
    assertEquals(enc?.origin, "model");
    assertEquals(enc?.encoding.codec, "int4-sym-g");
    assertEquals(enc?.encoding.groupSize, 32);
    assertEquals(enc?.blocks.length, 1);
    assertEquals(enc?.scale?.payloadBytes, 8 * 4);

    // A weight over the block limit (256 B in the fixture) is split in two at a row boundary.
    const big = bound.supplies.get("big.weight");
    assertEquals(
      big?.blocks.map((block) => block.rows),
      [
        [0, 8],
        [8, 16],
      ],
    );
    // Rule 3: the companion scale is in the same part as piece 1.
    assertEquals(big?.scale?.part, big?.blocks[0].part);

    // conv_transpose1d shape (row axis 1) — shape[1] scales.
    const conv = bound.supplies.get("conv.weight");
    assertEquals(conv?.encoding.rowAxis, 1);
    assertEquals(conv?.scale?.payloadBytes, 3 * 4);

    // The writer bakes the trailing padding (42-byte payload → 44-byte block).
    const dec = bound.supplies.get("dec.weight");
    assertEquals(dec?.blocks[0].payloadBytes, 42);
    assertEquals(dec?.blocks[0].length, 44);

    // An initializer supplied from the const region (part 1).
    assertEquals(bound.supplies.get("const.a1b2c3d4e5f60718")?.origin, "const");
    assertEquals(bound.supplies.get("const.b1b2c3d4e5f6071a")?.encoding.codec, "i32");
  });

  it("both documents match the canonical serialization (parse → serialize is byte-identical)", async () => {
    const bytes = await fixtureBytes();
    const header = readHeader(bytes);
    const graphBytes = bytes
      .subarray(HEADER_BYTES, HEADER_BYTES + header.graphDescriptorLength)
      .slice();
    const modelBytes = bytes
      .subarray(
        HEADER_BYTES + header.graphDescriptorLength,
        HEADER_BYTES + header.graphDescriptorLength + header.modelDescriptorLength,
      )
      .slice();

    assertEquals(serializeGraphDescriptor(parseGraphDescriptor(graphBytes)), graphBytes);
    assertEquals(serializeModelDescriptor(parseModelDescriptor(modelBytes)), modelBytes);
  });

  it("krg extraction is a byte copy of the krm and carries the graph descriptor unchanged", async () => {
    const bytes = await fixtureBytes();
    const container = await openContainer({ kind: "bytes", bytes });
    const extracted = await container.extractGraph();

    const header = readHeader(extracted);
    assertEquals(header.kind, "graph");
    assertEquals(header.modelDescriptorLength, 0);
    assertEquals(
      extracted.subarray(HEADER_BYTES, HEADER_BYTES + header.graphDescriptorLength),
      bytes.subarray(HEADER_BYTES, HEADER_BYTES + header.graphDescriptorLength),
    );
    const reopened = await openContainer({ kind: "bytes", bytes: extracted });
    assertEquals(reopened.model, undefined);
    assertEquals(Object.keys(reopened.graphs), ["synthetic"]);
  });
});

describe("split container fixture (Python writer)", () => {
  it("opens the part list and every block comes back with its sha256 verified", async () => {
    const parts = await splitParts();
    const container = await openContainer({ kind: "parts", parts });

    assertEquals(container.header.kind, "model");
    assertEquals(Object.keys(container.graphs), ["synthetic-split"]);
    // Even when const is empty, part 1 is listed as a zero-byte file (§8).
    assertEquals(container.graph.const.blocks, []);
    assertEquals(container.model?.parts[0].length, 0);
    assertEquals(parts[1].byteLength, 0);
    // Two weight parts + one asset part (4 excluding part 0 — the breakdown is in the case below).
    assertEquals(container.model?.parts.length, SPLIT_FIXTURE_PARTS - 1);

    for (const block of container.model?.blocks ?? []) {
      const bytes = await container.readBlock(block.id);
      assertEquals(await sha256Hex(bytes), block.sha256, block.id);
    }
  });

  it("assets declare name / role / logical length, and a block is the payload + 0x00 padding", async () => {
    const parts = await splitParts();
    const container = await openContainer({ kind: "parts", parts });
    const assets = container.model?.assets ?? {};

    assertEquals(
      Object.fromEntries(Object.entries(assets).map(([name, a]) => [name, [a.role, a.length]])),
      { ple_index: ["ple-index", 64], rope_base: ["rope-base", 37] },
    );

    // A 37-byte payload becomes a 40-byte block whose last 3 bytes are 0x00 (§4.1).
    // The logical length is in the declaration, so a consumer does not guess-strip the trailing
    // 0x00 bytes (ADR 0109 decision 4).
    const rope = await container.readBlock(assets["rope_base"].block);
    assertEquals(rope.byteLength, 40);
    assertEquals(rope.subarray(0, assets["rope_base"].length), patternBytes(37, 52));
    assertEquals(rope.subarray(assets["rope_base"].length), new Uint8Array([0, 0, 0]));

    // An asset that needs no padding is exactly its payload.
    const index = await container.readBlock(assets["ple_index"].block);
    assertEquals(index, patternBytes(64, 51));

    // Asset blocks do not share a part with weight blocks (§4.2 — assets read by range get their
    // own part).
    const assetBlocks = new Set(Object.values(assets).map((a) => a.block));
    const blocks = container.model?.blocks ?? [];
    const assetParts = new Set(blocks.filter((b) => assetBlocks.has(b.id)).map((b) => b.part));
    const weightParts = new Set(blocks.filter((b) => !assetBlocks.has(b.id)).map((b) => b.part));
    // Pinned in a form that cannot pass vacuously: 1 asset part and 2 weight parts (3 parts in
    // all).
    assertEquals(assetParts.size, 1);
    assertEquals(weightParts.size, 2);
    assertEquals(assetParts.intersection(weightParts).size, 0);
  });

  it("both documents match the canonical serialization (parse → serialize is byte-identical)", async () => {
    const parts = await splitParts();
    const header = readHeader(parts[0]);
    const graphBytes = parts[0]
      .subarray(HEADER_BYTES, HEADER_BYTES + header.graphDescriptorLength)
      .slice();
    const modelBytes = parts[0]
      .subarray(
        HEADER_BYTES + header.graphDescriptorLength,
        HEADER_BYTES + header.graphDescriptorLength + header.modelDescriptorLength,
      )
      .slice();

    assertEquals(serializeGraphDescriptor(parseGraphDescriptor(graphBytes)), graphBytes);
    assertEquals(serializeModelDescriptor(parseModelDescriptor(modelBytes)), modelBytes);
  });
});

const SPLIT_SCALE_FIXTURE = new URL(
  "./fixtures/container/synthetic-split-scale.krm",
  import.meta.url,
);

/** The 2 documents of a single-form container (sliced out as their own buffers). */
const documentsOf = (
  bytes: Uint8Array<ArrayBuffer>,
): { readonly graph: Uint8Array<ArrayBuffer>; readonly model: Uint8Array<ArrayBuffer> } => {
  const header = readHeader(bytes);
  const graphEnd = HEADER_BYTES + header.graphDescriptorLength;
  return {
    graph: bytes.subarray(HEADER_BYTES, graphEnd).slice(),
    model: bytes.subarray(graphEnd, graphEnd + header.modelDescriptorLength).slice(),
  };
};

describe("split-scale container fixture (Python writer — ADR 0108 addendum 15)", () => {
  it("reads the split scale whole and the scale that fits as one block, every block verified", async () => {
    const bytes = (await readFile(SPLIT_SCALE_FIXTURE)) as Uint8Array<ArrayBuffer>;
    const container = await openContainer({ kind: "bytes", bytes });
    for (const block of container.model?.blocks ?? []) {
      assertEquals(await sha256Hex(await container.readBlock(block.id)), block.sha256, block.id);
    }
    const bound = container.graphs["synthetic-split-scale"];

    // `w.weight`: 512 B of i4 in 4 pieces at the fixture's 128 B limit; its 256 B scale is split on
    // the same rows, all in the part of piece 1 (rule ③).
    const w = bound.supplies.get("w.weight");
    assert(w?.scale?.pieces !== undefined, "the Python writer did not split the scale");
    assertEquals(
      w.scale.pieces.map((piece) => piece.rows),
      w.blocks.map((block) => block.rows),
    );
    assertEquals(w.scale.pieces.length, 4);
    assertEquals(new Set(w.scale.pieces.map((piece) => piece.part)), new Set([w.blocks[0].part]));
    assertEquals(await container.readBlock(w.scale.id), patternBytes(16 * 4 * 4, 62));

    // `v.weight`: piece-split too, but its 32 B scale fits one block.
    const v = bound.supplies.get("v.weight");
    assertEquals(v?.blocks.length, 2);
    assertEquals(v?.scale?.pieces, undefined);
    assertEquals(await container.readBlock(v?.scale?.id ?? ""), patternBytes(8 * 4, 64));
  });

  it("the 2 documents equal their canonical serialization (parse → serialize, byte-identical)", async () => {
    const documents = documentsOf((await readFile(SPLIT_SCALE_FIXTURE)) as Uint8Array<ArrayBuffer>);

    assertEquals(serializeGraphDescriptor(parseGraphDescriptor(documents.graph)), documents.graph);
    assertEquals(serializeModelDescriptor(parseModelDescriptor(documents.model)), documents.model);
  });
});
