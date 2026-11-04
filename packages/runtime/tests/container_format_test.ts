/**
 * The reader of the container format (`krm` / `krg`) — header, the two descriptor documents, block
 * reads, `krg` extraction and the merge layer. Pins the rules of docs/container-v1.md in the form
 * "break exactly one point of a valid case".
 */

import { assertEquals, assertRejects, assertThrows } from "@std/assert";
import { describe, it } from "vite-plus/test";
import { bindDeclarations } from "../src/format/container/bind.ts";
import {
  parseGraphDescriptor,
  parseModelDescriptor,
  sha256Hex,
  validateAgainstGraph,
} from "../src/format/container/descriptor.ts";
import {
  ContainerFormatError,
  derivePartOffsets,
  readHeader,
  writeHeader,
} from "../src/format/container/header.ts";
import {
  BLOCK_START_ALIGN,
  HEADER_BYTES,
  MAX_DESCRIPTOR_BYTES,
  PART_MAX_BYTES,
} from "../src/format/container/limits.ts";
import { type BlockSource, openContainer } from "../src/format/container/open.ts";
import { type IrDeclaration, parseIrDeclaration } from "../src/format/ir.ts";
import {
  type ModelInput,
  writeGraphContainer,
  writeModelContainer,
  type WrittenContainer,
} from "./helpers/container-write.ts";

// ---------------------------------------------------------------------------
// Synthetic model
// ---------------------------------------------------------------------------

/** Deterministic pseudo-random bytes (xorshift32 from seed). */
const bytesOf = (length: number, seed: number): Uint8Array<ArrayBuffer> => {
  const out = new Uint8Array(new ArrayBuffer(length));
  let state = seed >>> 0 || 1;
  for (let i = 0; i < length; i += 1) {
    state ^= state << 13;
    state >>>= 0;
    state ^= state >>> 17;
    state ^= state << 5;
    state >>>= 0;
    out[i] = state & 0xff;
  }
  return out;
};

type InitializerSpec = {
  readonly shape: readonly number[];
  readonly dtype?: "f32" | "i32";
  readonly shared?: true;
};

/** The smallest IR v2 graph that consumes each initializer in one node. */
const declaration = (initializers: Readonly<Record<string, InitializerSpec>>): IrDeclaration => {
  const names = Object.keys(initializers);
  return parseIrDeclaration(
    JSON.stringify({
      format: "karume-ir",
      version: 2,
      requires: { ops: ["matmul"] },
      symbols: ["T"],
      inputs: [{ name: "x", dtype: "f32", shape: ["T", 4] }],
      outputs: names.map((_, i) => `y${i}`),
      initializers: Object.fromEntries(
        names.map((name) => [name, initializers[name].shared ? { shared: true } : {}]),
      ),
      values: {
        ...Object.fromEntries(
          names.map((name) => [
            name,
            { dtype: initializers[name].dtype ?? "f32", shape: initializers[name].shape },
          ]),
        ),
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

const F32 = 4;

const syntheticModel = (): ModelInput => ({
  graphs: {
    text_encoder: declaration({
      "enc.weight": { shape: [16, 32] },
      "proj.weight": { shape: [12, 64] },
      "model.lm_head.weight": { shape: [8, 8], shared: true },
    }),
    front: declaration({
      "big.weight": { shape: [32, 32] },
      "front.bias": { shape: [65] },
      "const.rope": { shape: [64] },
    }),
    voice: declaration({
      "const.window": { shape: [65] },
      "conv.weight": { shape: [4, 3, 8] },
    }),
  },
  consts: [
    {
      graph: "front",
      initializer: "const.rope",
      bytes: bytesOf(64 * F32, 11),
      encoding: { codec: "f32" },
    },
    // A const whose length is not a multiple of 4 (checks that the writer bakes the trailing
    // padding).
    {
      graph: "voice",
      initializer: "const.window",
      bytes: bytesOf(65 * 2, 12),
      encoding: { codec: "f16" },
    },
  ],
  weights: [
    {
      graph: "text_encoder",
      initializer: "enc.weight",
      bytes: bytesOf(16 * 32 * 2, 21),
      encoding: { codec: "f16" },
    },
    {
      // i4 + companion scale (placed in the same part by rule 3). Row length 64 / group 32 = 2
      // groups.
      graph: "text_encoder",
      initializer: "proj.weight",
      bytes: bytesOf((12 * 64) / 2, 22),
      encoding: {
        codec: "int4-sym-g",
        groupSize: 32,
        scale: { bytes: bytesOf(12 * 2 * F32, 23), dtype: "f32" },
      },
    },
    {
      // Exceeds the block limit (512 B in the tests), so it is split into pieces.
      graph: "front",
      initializer: "big.weight",
      bytes: bytesOf(32 * 32, 24),
      encoding: {
        codec: "int8-sym",
        groupSize: 32,
        scale: { bytes: bytesOf(32 * F32, 25), dtype: "f32" },
      },
    },
    // An f16 whose length is not a multiple of 4 (trailing padding).
    {
      graph: "front",
      initializer: "front.bias",
      bytes: bytesOf(65 * 2, 26),
      encoding: { codec: "f16" },
    },
    {
      // conv_transpose1d shape (per-channel scale on rowAxis 1).
      graph: "voice",
      initializer: "conv.weight",
      bytes: bytesOf(4 * 3 * 8, 27),
      encoding: {
        codec: "int8-sym",
        rowAxis: 1,
        groupSize: 32,
        scale: { bytes: bytesOf(3 * F32, 28), dtype: "f32" },
      },
    },
  ],
  assets: [
    { name: "style_vectors", role: "style-vectors", bytes: bytesOf(300, 31) },
    { name: "ple_table", role: "ple-table", bytes: bytesOf(400, 32), dedicatedPart: true },
  ],
  provenance: { license: "apache-2.0", writer: "test" },
});

const OPTIONS = { partBytes: 2048, blockBytes: 512 } as const;

const mutateJson = (
  bytes: Uint8Array<ArrayBuffer>,
  mutate: (doc: Record<string, unknown>) => void,
): Uint8Array<ArrayBuffer> => {
  const doc = JSON.parse(new TextDecoder().decode(bytes)) as Record<string, unknown>;
  mutate(doc);
  return new TextEncoder().encode(JSON.stringify(doc)) as Uint8Array<ArrayBuffer>;
};

const rejectsModel = (
  hint: string,
  bytes: Uint8Array<ArrayBuffer>,
  mutate: (doc: Record<string, unknown>) => void,
  expected?: string,
): void => {
  assertThrows(
    () => parseModelDescriptor(mutateJson(bytes, mutate)),
    ContainerFormatError,
    expected,
    hint,
  );
};

// deno-lint-ignore no-explicit-any
const anyOf = (value: unknown): any => value;

/**
 * Rewrites only the model descriptor of a valid `krm` baked by the writer, rebuilds part 0 and
 * passes it to the reader. A path for hitting the reader's rules (index → merge layer) without
 * going through the writer's checks.
 */
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

// ---------------------------------------------------------------------------
// Header
// ---------------------------------------------------------------------------

describe("container header", () => {
  it("round-trips krm / krg, and only the magic carries the kind", () => {
    const model = readHeader(
      writeHeader({
        kind: "model",
        version: 1,
        graphDescriptorLength: 10,
        modelDescriptorLength: 20,
      }),
    );
    assertEquals(model, {
      kind: "model",
      version: 1,
      graphDescriptorLength: 10,
      modelDescriptorLength: 20,
    });
    const graph = readHeader(
      writeHeader({
        kind: "graph",
        version: 1,
        graphDescriptorLength: 10,
        modelDescriptorLength: 0,
      }),
    );
    assertEquals(graph.kind, "graph");
  });

  it("rejects an unknown magic, an unsupported version, a kind that contradicts the lengths, and exceeded limits", () => {
    const bytes = writeHeader({
      kind: "model",
      version: 1,
      graphDescriptorLength: 10,
      modelDescriptorLength: 20,
    });
    const corrupt = (
      mutate: (view: DataView, copy: Uint8Array<ArrayBuffer>) => void,
    ): Uint8Array<ArrayBuffer> => {
      const copy = bytes.slice();
      mutate(new DataView(copy.buffer), copy);
      return copy;
    };
    assertThrows(
      () => readHeader(corrupt((_, c) => c.set([0x50, 0x4b, 0x03, 0x04], 0))),
      ContainerFormatError,
      "magic",
    );
    assertThrows(
      () => readHeader(corrupt((v) => v.setUint32(4, 2, true))),
      ContainerFormatError,
      "version",
    );
    assertThrows(
      () => readHeader(corrupt((v) => v.setBigUint64(16, 0n, true))),
      ContainerFormatError,
      "krm but",
    );
    assertThrows(
      () => readHeader(corrupt((v) => v.setBigUint64(8, 0n, true))),
      ContainerFormatError,
      "length is 0",
    );
    assertThrows(
      () => readHeader(corrupt((v) => v.setBigUint64(8, 1n << 60n, true))),
      ContainerFormatError,
      "safe integer",
    );
    assertThrows(
      () => readHeader(corrupt((v) => v.setBigUint64(8, BigInt(MAX_DESCRIPTOR_BYTES + 1), true))),
      ContainerFormatError,
      "limit",
    );
    assertThrows(
      () =>
        writeHeader({
          kind: "graph",
          version: 1,
          graphDescriptorLength: 1,
          modelDescriptorLength: 5,
        }),
      ContainerFormatError,
    );
    assertThrows(
      () => readHeader(bytes.subarray(0, HEADER_BYTES - 1)),
      ContainerFormatError,
      "but only",
    );
  });

  it("part offsets of the single form are 64 B aligned, and no padding is inserted for a zero-length part", () => {
    assertEquals(derivePartOffsets(100, [0, 30]), { offsets: [0, 100, 128], totalLength: 158 });
    assertEquals(derivePartOffsets(64, [10, 0]), { offsets: [0, 64, 74], totalLength: 74 });
    assertEquals(derivePartOffsets(1, [1]).offsets[1] % BLOCK_START_ALIGN, 0);
  });
});

// ---------------------------------------------------------------------------
// Write → read
// ---------------------------------------------------------------------------

describe("container round trip", () => {
  it("opening the single form returns every block with its sha256 verified, and the merge layer derives the supply plan", async () => {
    const written = await writeModelContainer(syntheticModel(), OPTIONS);
    const opened = await openContainer({ kind: "bytes", bytes: written.single });
    assertEquals(opened.header.kind, "model");
    assertEquals(Object.keys(opened.graphs).sort(), ["front", "text_encoder", "voice"]);

    const supplies = opened.graphs.front.supplies;
    const big = supplies.get("big.weight");
    if (big === undefined) throw new Error("no supply for big.weight");
    // 32 rows × 32 B = 1024 B > block limit 512 B → split into pieces.
    assertEquals(big.blocks.length > 1, true);
    assertEquals(big.blocks[0].rows[0], 0);
    assertEquals(big.blocks[big.blocks.length - 1].rows[1], 32);
    assertEquals(big.scale?.part, big.blocks[0].part);
    // Reading and concatenating the blocks of the piece list gives back the original payload.
    const joined = new Uint8Array(1024);
    let cursor = 0;
    for (const block of big.blocks) {
      const bytes = await opened.readBlock(block.id);
      joined.set(bytes.subarray(0, block.payloadBytes), cursor);
      cursor += block.payloadBytes;
    }
    assertEquals(joined, bytesOf(32 * 32, 24));

    // An f16 whose length is not a multiple of 4 is declared with its padding, and the payload
    // length is determined by the declared shape.
    const bias = supplies.get("front.bias");
    assertEquals(bias?.blocks[0].length, 132);
    assertEquals(bias?.blocks[0].payloadBytes, 130);
    assertEquals(
      (await opened.readBlock(bias!.blocks[0].id)).subarray(130),
      new Uint8Array([0, 0]),
    );

    // The const region is part 1, and its origin is const.
    const rope = supplies.get("const.rope");
    assertEquals(rope?.origin, "const");
    assertEquals(rope?.blocks[0].part, 1);
    assertEquals(await opened.readBlock(rope!.blocks[0].id), bytesOf(64 * F32, 11));

    // A shared declaration has no supply plan.
    assertEquals(opened.graphs.text_encoder.supplies.has("model.lm_head.weight"), false);
    // An asset in a dedicated part occupies one part by itself.
    const ple = opened.model?.assets["ple_table"];
    const pleBlock = opened.model?.blocks.find((block) => block.id === ple?.block);
    assertEquals(opened.model?.blocks.filter((block) => block.part === pleBlock?.part).length, 1);
    assertEquals(opened.model?.codecs, ["f16", "int4-sym-g", "int8-sym"]);
  });

  it("split form and single form are byte-identical in both descriptors and blocks", async () => {
    const written = await writeModelContainer(syntheticModel(), OPTIONS);
    const single = await openContainer({ kind: "bytes", bytes: written.single });
    const split = await openContainer({ kind: "parts", parts: written.parts });
    assertEquals(split.graph, single.graph);
    assertEquals(split.model, single.model);
    for (const block of [...written.graph.const.blocks, ...written.model.blocks]) {
      assertEquals(await split.readBlock(block.id), await single.readBlock(block.id));
    }
    assertEquals(await split.extractGraph(), await single.extractGraph());
  });

  it("a krg extracted from a krm is byte-identical to a directly written krg and opens as a krg", async () => {
    const input = syntheticModel();
    const written = await writeModelContainer(input, OPTIONS);
    const direct = await writeGraphContainer(input, OPTIONS);
    const opened = await openContainer({ kind: "bytes", bytes: written.single });
    const extracted = await opened.extractGraph();
    assertEquals(extracted, direct.bytes);
    const graphOnly = await openContainer({ kind: "bytes", bytes: extracted });
    assertEquals(graphOnly.header.kind, "graph");
    assertEquals(graphOnly.model, undefined);
    assertEquals(graphOnly.graph, opened.graph);
    // Blocks of the const region are readable from a krg too (the graph descriptor carries the
    // const binding table — §9).
    const rope = graphOnly.graphs.front.supplies.get("const.rope");
    assertEquals(await graphOnly.readBlock(rope!.blocks[0].id), bytesOf(64 * F32, 11));
    // Weights have no supply plan (the requirement is derived from the declared shape).
    assertEquals(graphOnly.graphs.front.supplies.has("big.weight"), false);
  });

  it("krg extraction works even when the const region is empty (no padding for a zero-length part)", async () => {
    const model: ModelInput = {
      ...syntheticModel(),
      consts: [],
      graphs: {
        text_encoder: declaration({ "enc.weight": { shape: [16, 32] } }),
      },
      weights: [
        {
          graph: "text_encoder",
          initializer: "enc.weight",
          bytes: bytesOf(16 * 32 * 2, 21),
          encoding: { codec: "f16" },
        },
      ],
      assets: [],
    };
    const written = await writeModelContainer(model, OPTIONS);
    assertEquals(written.parts[1].byteLength, 0);
    const opened = await openContainer({ kind: "parts", parts: written.parts });
    assertEquals(await opened.extractGraph(), (await writeGraphContainer(model, OPTIONS)).bytes);
  });

  it("the outer expected hash is verified before the two documents are parsed", async () => {
    const written = await writeModelContainer(syntheticModel(), OPTIONS);
    const expect = {
      graph: {
        length: written.graphDescriptorBytes.byteLength,
        sha256: await sha256Hex(written.graphDescriptorBytes),
      },
      model: {
        length: written.modelDescriptorBytes.byteLength,
        sha256: await sha256Hex(written.modelDescriptorBytes),
      },
    };
    await openContainer({ kind: "parts", parts: written.parts }, expect);
    const tampered = written.single.slice();
    tampered[HEADER_BYTES + 5] ^= 0x01;
    await assertRejects(
      () => openContainer({ kind: "bytes", bytes: tampered }, expect),
      ContainerFormatError,
      "graph descriptor sha256",
    );
    await assertRejects(
      () => openContainer({ kind: "bytes", bytes: written.graphContainer }, expect),
      ContainerFormatError,
      "krg but",
    );
  });

  it("passing a krm only the graph's expected value does not open it with an unverified model descriptor", async () => {
    const written = await writeModelContainer(syntheticModel(), OPTIONS);
    const graphOnly = {
      graph: {
        length: written.graphDescriptorBytes.byteLength,
        sha256: await sha256Hex(written.graphDescriptorBytes),
      },
    };
    await assertRejects(
      () => openContainer({ kind: "bytes", bytes: written.single }, graphOnly),
      ContainerFormatError,
      "expected values for the model descriptor",
    );
  });

  it("sha256 catches a one-byte tampering of a block, and krg extraction catches tampering of the const region", async () => {
    const written = await writeModelContainer(syntheticModel(), OPTIONS);
    const parts = written.parts.map((part) => part.slice());
    const weight = written.model.blocks.find((block) => block.role === "weight");
    if (weight === undefined) throw new Error("no weight block");
    parts[weight.part][weight.offset + 3] ^= 0x80;
    const opened = await openContainer({ kind: "parts", parts });
    await assertRejects(
      () => opened.readBlock(weight.id),
      ContainerFormatError,
      "sha256 differs from the declared value",
    );
    parts[1][0] ^= 0x01;
    const again = await openContainer({ kind: "parts", parts });
    await assertRejects(() => again.extractGraph(), ContainerFormatError, "const region sha256");
  });

  it("krg extraction only from a krm — a krg input cannot be checked for the integrity of its const region, so it is not extracted", async () => {
    // A krg has no document carrying the hash of the whole const region (parts[0] of the model
    // descriptor exists only in a krm — §9). If a tampered krg could be extracted, corrupted bytes
    // would go out looking like a legitimate krg.
    const written = await writeModelContainer(syntheticModel(), OPTIONS);
    const tampered = written.graphContainer.slice();
    tampered[tampered.byteLength - 1] ^= 0x01;
    const opened = await openContainer({ kind: "bytes", bytes: tampered });
    assertEquals(opened.header.kind, "graph");
    await assertRejects(() => opened.extractGraph(), ContainerFormatError, "only from a krm");
  });

  it("krg extraction from a krm fails when the source returns a short const region (the length is checked before the hash)", async () => {
    const written = await writeModelContainer(syntheticModel(), OPTIONS);
    const parts = written.parts;
    const source: BlockSource = {
      partCount: parts.length,
      verified: false,
      partLength: (index) => parts[index].byteLength,
      read: (part, offset, length) => {
        const bytes = parts[part].subarray(offset, offset + length);
        return Promise.resolve(part === 1 ? bytes.subarray(0, bytes.byteLength - 4) : bytes);
      },
    };
    const opened = await openContainer({ kind: "source", source });
    await assertRejects(
      () => opened.extractGraph(),
      ContainerFormatError,
      `fetched length ${written.graph.const.length - 4}`,
    );
  });

  it("a source that claims to be verified skips the block sha256, and a source that does not claim it gets the check", async () => {
    // A path where the fetch layer verified the whole file (hub's HF source) claims verified to
    // avoid the double digest when cold and the digest when warm (ADR 0109 decision 7). The
    // observation is "does tampering slip through" — the number of digests itself cannot be
    // counted from outside.
    const written = await writeModelContainer(syntheticModel(), OPTIONS);
    const parts = written.parts.map((part) => part.slice());
    const weight = written.model.blocks.find((block) => block.role === "weight");
    if (weight === undefined) throw new Error("no weight block");
    parts[weight.part][weight.offset + 3] ^= 0x80;
    const sourceOf = (verified: boolean): BlockSource => ({
      partCount: parts.length,
      verified,
      partLength: (index) => parts[index].byteLength,
      read: (part, offset, length) =>
        Promise.resolve(parts[part].subarray(offset, offset + length)),
    });
    const trusted = await openContainer({ kind: "source", source: sourceOf(true) });
    assertEquals(
      await trusted.readBlock(weight.id),
      parts[weight.part].subarray(weight.offset, weight.offset + weight.length),
    );
    const untrusted = await openContainer({ kind: "source", source: sourceOf(false) });
    await assertRejects(
      () => untrusted.readBlock(weight.id),
      ContainerFormatError,
      "sha256 differs from the declared value",
    );
  });

  it("the asset reader returns only the range, and with an unverified source verifies the block once before slicing", async () => {
    const written = await writeModelContainer(syntheticModel(), OPTIONS);
    const parts = written.parts.map((part) => part.slice());
    const ple = written.model.assets["ple_table"];
    const pleBlock = written.model.blocks.find((block) => block.id === ple.block);
    if (pleBlock === undefined) throw new Error("no block for ple_table");
    const reads: number[] = [];
    const sourceOf = (verified: boolean): BlockSource => ({
      partCount: parts.length,
      verified,
      partLength: (index) => parts[index].byteLength,
      read: (part, offset, length) => {
        reads.push(length);
        return Promise.resolve(parts[part].subarray(offset, offset + length));
      },
    });

    // Verified: fetches only the length of the range (does not read the whole 400 B block).
    const trusted = await openContainer({ kind: "source", source: sourceOf(true) });
    const reader = trusted.asset("ple_table");
    assertEquals(reader.role, "ple-table");
    assertEquals(reader.length, 400);
    reads.length = 0;
    assertEquals(await reader.read(16, 8), bytesOf(400, 32).subarray(16, 24));
    assertEquals(reads, [8]);

    // Unverified: fetches the whole block once on first use to verify it, and does not fetch it the
    // second time.
    const untrusted = await openContainer({ kind: "source", source: sourceOf(false) });
    const cached = untrusted.asset("ple_table");
    reads.length = 0;
    assertEquals(await cached.read(0, 4), bytesOf(400, 32).subarray(0, 4));
    assertEquals(await cached.read(396, 4), bytesOf(400, 32).subarray(396, 400));
    assertEquals(reads, [pleBlock.length]);

    // Tampering is caught only with an unverified source (with a verified one, the fetch layer is
    // the side that guarantees it).
    parts[pleBlock.part][pleBlock.offset + 5] ^= 0x40;
    const tampered = await openContainer({ kind: "source", source: sourceOf(false) });
    await assertRejects(
      () => tampered.asset("ple_table").read(0, 4),
      ContainerFormatError,
      "sha256 differs from the declared value",
    );

    // Out of range, undeclared and krg fail loudly.
    await assertRejects(() => reader.read(398, 4), ContainerFormatError, "runs past");
    assertThrows(() => trusted.asset("nope"), ContainerFormatError, "Undeclared asset");
    const graphOnly = await openContainer({ kind: "bytes", bytes: written.graphContainer });
    assertThrows(() => graphOnly.asset("ple_table"), ContainerFormatError, "Undeclared asset");
  });

  it("the asset reader fails when the range is not a safe integer, and also when a verified source returns short", async () => {
    const written = await writeModelContainer(syntheticModel(), OPTIONS);
    const parts = written.parts.map((part) => part.slice());
    // Return correct data up to opening (header and descriptor reads) and make only the reads after
    // opening short.
    let shortBy = 0;
    const sourceOf = (verified: boolean): BlockSource => ({
      partCount: parts.length,
      verified,
      partLength: (index) => parts[index].byteLength,
      read: (part, offset, length) =>
        Promise.resolve(parts[part].subarray(offset, offset + length - shortBy)),
    });

    // NaN slips through every range comparison, and with an unverified source subarray used to
    // silently return 0 bytes.
    for (const verified of [true, false]) {
      const reader = (await openContainer({ kind: "source", source: sourceOf(verified) })).asset(
        "ple_table",
      );
      await assertRejects(
        () => reader.read(Number.NaN, 4),
        ContainerFormatError,
        "is not a safe integer",
      );
      await assertRejects(() => reader.read(1.5, 4), ContainerFormatError, "is not a safe integer");
      await assertRejects(
        () => reader.read(0, Number.NaN),
        ContainerFormatError,
        "is not a safe integer",
      );
    }

    // A verified source takes the path that does not read the whole block, so the returned length
    // is checked against the request.
    const short = await openContainer({ kind: "source", source: sourceOf(true) });
    shortBy = 1;
    await assertRejects(
      () => short.asset("ple_table").read(16, 8),
      ContainerFormatError,
      "fetched length 7 differs from the requested 8",
    );
  });

  it("the part count and lengths of the split form must match the declaration", async () => {
    const written = await writeModelContainer(syntheticModel(), OPTIONS);
    await assertRejects(
      () => openContainer({ kind: "parts", parts: written.parts.slice(0, -1) }),
      ContainerFormatError,
      "are declared",
    );
    const shorter = written.parts.map((part, i) =>
      i === 2 ? part.subarray(0, part.byteLength - 4) : part,
    );
    await assertRejects(
      () => openContainer({ kind: "parts", parts: shorter }),
      ContainerFormatError,
      "part 2 length",
    );
    await assertRejects(
      () =>
        openContainer({
          kind: "bytes",
          bytes: written.single.subarray(0, written.single.byteLength - 1),
        }),
      ContainerFormatError,
      "Single-form length",
    );
  });

  it("if a krg's const.length exceeds the part-length ceiling, it fails at open, before part 1 is read", async () => {
    // The size of a krg's part 1 is known only from the graph descriptor's const.length (§2.1).
    // Without applying the ceiling (§10 — every part, part 0 / 1 included) in the descriptor
    // check, a reader declaring the same length would reach a huge single read.
    const written = await writeModelContainer(syntheticModel(), OPTIONS);
    const huge = PART_MAX_BYTES + 1;
    const graphBytes = mutateJson(written.graphDescriptorBytes, (doc) => {
      anyOf(doc).const.length = huge;
    });
    const header = writeHeader({
      kind: "graph",
      version: 1,
      graphDescriptorLength: graphBytes.byteLength,
      modelDescriptorLength: 0,
    });
    const part0 = new Uint8Array(new ArrayBuffer(HEADER_BYTES + graphBytes.byteLength));
    part0.set(header, 0);
    part0.set(graphBytes, HEADER_BYTES);
    const reads: number[] = [];
    const source: BlockSource = {
      partCount: 2,
      verified: false,
      partLength: (index) => (index === 0 ? part0.byteLength : huge),
      read: (part, offset, length) => {
        reads.push(part);
        return Promise.resolve(part0.slice(offset, offset + length));
      },
    };
    await assertRejects(
      () => openContainer({ kind: "source", source }),
      ContainerFormatError,
      "part length ceiling",
    );
    assertEquals(reads.includes(1), false);
  });
});

// ---------------------------------------------------------------------------
// Declaration checks of the descriptors
// ---------------------------------------------------------------------------

describe("container descriptor", () => {
  it("the two canonically serialized documents return to the same bytes through parse → serialize", async () => {
    const written = await writeModelContainer(syntheticModel(), OPTIONS);
    const { serializeGraphDescriptor, serializeModelDescriptor } =
      await import("../src/format/container/descriptor.ts");
    assertEquals(
      serializeGraphDescriptor(parseGraphDescriptor(written.graphDescriptorBytes)),
      written.graphDescriptorBytes,
    );
    assertEquals(
      serializeModelDescriptor(parseModelDescriptor(written.modelDescriptorBytes)),
      written.modelDescriptorBytes,
    );
  });

  it("rejects a codec missing from the ledger, a packing that disagrees with the ledger, and unknown keys", async () => {
    const { modelDescriptorBytes: bytes } = await writeModelContainer(syntheticModel(), OPTIONS);
    const supply = (doc: Record<string, unknown>) => anyOf(doc).binding.text_encoder["proj.weight"];
    rejectsModel(
      "unknown codec",
      bytes,
      (doc) => {
        supply(doc).encoding.codec = "int3-sym";
      },
      "is not in the ledger",
    );
    rejectsModel(
      "packing mismatch",
      bytes,
      (doc) => {
        supply(doc).encoding.packing.blockElements = 4;
      },
      "a different version of the ledger",
    );
    rejectsModel(
      "unknown key",
      bytes,
      (doc) => {
        supply(doc).encoding.bits = 4;
      },
      "unknown key",
    );
    rejectsModel(
      "asset logical length inconsistent with block length",
      bytes,
      (doc) => {
        anyOf(doc).assets.style_vectors.length = 296;
      },
      "differs from block",
    );
    rejectsModel(
      "groupSize on a non-quantized supply",
      bytes,
      (doc) => {
        anyOf(doc).binding.text_encoder["enc.weight"].encoding.groupSize = 32;
      },
      "is not quantized, so it cannot be written",
    );
    rejectsModel(
      "quantized supply without scale",
      bytes,
      (doc) => {
        delete supply(doc).encoding.scale;
      },
      "is quantized, so it is required",
    );
    rejectsModel(
      "codecs set differs from the binding table",
      bytes,
      (doc) => {
        anyOf(doc).codecs = ["f16"];
      },
      "codec set",
    );
    const { graphDescriptorBytes } = await writeModelContainer(syntheticModel(), OPTIONS);
    assertThrows(
      () =>
        parseGraphDescriptor(
          mutateJson(graphDescriptorBytes, (doc) => {
            anyOf(doc).capabilities.features = ["range-fetch"];
          }),
        ),
      ContainerFormatError,
      "unknown features",
    );
  });

  it("fails when the scale dtype is outside the accepted set (f16 bits are not read as f32)", async () => {
    // The reader reads scale with its declared dtype, so without checking the accepted set (only
    // f32 in the first version) it would read an f16 list as f32, a silent wrong value. This is the
    // only place the value-range check is hit.
    const { modelDescriptorBytes: bytes } = await writeModelContainer(syntheticModel(), OPTIONS);
    rejectsModel(
      "scale.dtype outside the accepted set",
      bytes,
      (doc) => {
        anyOf(doc).binding.text_encoder["proj.weight"].encoding.scale.dtype = "f16";
      },
      "accepted set",
    );
  });

  it("fails when the binding table points at a block that does not exist (the reverse of rule 5 — the target is missing)", async () => {
    // A surplus block (referenced from nowhere) is rejected by rule 5. The reverse form, where the
    // binding side names an id that is not in the index, is blocked here too (it fails before the
    // reader goes to fetch the block).
    const { modelDescriptorBytes: bytes } = await writeModelContainer(syntheticModel(), OPTIONS);
    rejectsModel(
      "data block undeclared",
      bytes,
      (doc) => {
        anyOf(doc).binding.front["front.bias"].block = "absent";
      },
      "is not in the index",
    );
    rejectsModel(
      "scale block undeclared",
      bytes,
      (doc) => {
        anyOf(doc).binding.text_encoder["proj.weight"].encoding.scale.block = "absent";
      },
      "is not in the index",
    );
  });

  it("rejects numbers that overflow to Infinity, nesting that is too deep, and __proto__", async () => {
    const { graphDescriptorBytes: bytes } = await writeModelContainer(syntheticModel(), OPTIONS);
    const text = new TextDecoder().decode(bytes);
    const overflow = new TextEncoder().encode(
      text.replace('"length":', '"length":1e999,"x":'),
    ) as Uint8Array<ArrayBuffer>;
    assertThrows(() => parseGraphDescriptor(overflow), ContainerFormatError);
    const nested = new TextEncoder().encode(
      `{"a":${"[".repeat(70)}1${"]".repeat(70)}}`,
    ) as Uint8Array<ArrayBuffer>;
    assertThrows(() => parseGraphDescriptor(nested), ContainerFormatError, "depth limit");
    const proto = new TextEncoder().encode('{"__proto__":{"format":1}}') as Uint8Array<ArrayBuffer>;
    assertThrows(() => parseGraphDescriptor(proto), ContainerFormatError, "__proto__");
  });

  it("rules 1–5: rejects double binding, a surplus block, a hole in the pieces, a mixed-up role, and scale in another part", async () => {
    const written = await writeModelContainer(syntheticModel(), OPTIONS);
    const bytes = written.modelDescriptorBytes;
    const bigSupply = (doc: Record<string, unknown>) => anyOf(doc).binding.front["big.weight"];
    rejectsModel(
      "double binding (rule 1)",
      bytes,
      (doc) => {
        anyOf(doc).binding.front["front.bias"].block = bigSupply(doc).pieces[0].block;
      },
      "bound twice",
    );
    rejectsModel(
      "surplus block (rule 5)",
      bytes,
      (doc) => {
        delete anyOf(doc).assets["style_vectors"];
      },
      "is not referenced",
    );
    rejectsModel(
      "hole in the pieces (rule 4)",
      bytes,
      (doc) => {
        bigSupply(doc).pieces[1].rows[0] += 1;
      },
      "does not continue from",
    );
    rejectsModel(
      "a single piece",
      bytes,
      (doc) => {
        bigSupply(doc).pieces = [bigSupply(doc).pieces[0]];
      },
      "2 or more",
    );
    rejectsModel(
      "both block and pieces",
      bytes,
      (doc) => {
        bigSupply(doc).block = "w0001";
      },
      "exactly one of",
    );
    rejectsModel(
      "mixed-up role",
      bytes,
      (doc) => {
        const scaleId = anyOf(doc).binding.text_encoder["proj.weight"].encoding.scale.block;
        anyOf(doc).blocks.find((block: { id: string }) => block.id === scaleId).role = "weight";
      },
      "has role",
    );
    rejectsModel(
      "scale in another part (rule 3)",
      bytes,
      (doc) => {
        const scaleId = anyOf(doc).binding.text_encoder["proj.weight"].encoding.scale.block;
        const block = anyOf(doc).blocks.find((block: { id: string }) => block.id === scaleId);
        const other = anyOf(doc).parts.find(
          (part: { index: number }) => part.index >= 2 && part.index !== block.part,
        );
        block.part = other.index;
        block.offset = Math.ceil(other.length / 64) * 64;
        other.length = block.offset + block.length;
      },
      "same part MUST",
    );
    rejectsModel(
      "offset alignment",
      bytes,
      (doc) => {
        anyOf(doc).blocks[0].offset += 4;
      },
      "is not a multiple of",
    );
    rejectsModel(
      "parts index",
      bytes,
      (doc) => {
        anyOf(doc).parts[0].index = 0;
      },
      "index is not 1",
    );
    rejectsModel(
      "block in part 1",
      bytes,
      (doc) => {
        anyOf(doc).blocks[0].part = 1;
      },
      "less than 2",
    );
  });

  it("the key set of the binding table equals exactly the set of initializers that are neither shared nor const (rule 5)", async () => {
    const written = await writeModelContainer(syntheticModel(), OPTIONS);
    const withBinding = (mutate: (doc: Record<string, unknown>) => void): Promise<void> =>
      openWithModelDescriptor(written, mutate);
    await assertRejects(
      () =>
        withBinding((doc) => {
          // Put a shared declaration into the binding table (surplus).
          anyOf(doc).binding.text_encoder["model.lm_head.weight"] =
            anyOf(doc).binding.text_encoder["enc.weight"];
          delete anyOf(doc).binding.text_encoder["enc.weight"];
        }),
      ContainerFormatError,
      "missing [enc.weight] / surplus [model.lm_head.weight]",
    );
  });

  it("a container whose piece parts are not non-decreasing in index order fails at open (rule 4)", async () => {
    // With the block limit and part length at 512 B, the 2 pieces are placed one each in part 2 /
    // part 3. Swapping only where the 2 blocks live (their parts) keeps the index structure
    // (overlaps, part lengths) valid while piece 1 lands in a later part than piece 2. Without the
    // check it would open, and the build would fail on broken internal bookkeeping.
    const written = await writeModelContainer(
      {
        graphs: { g: declaration({ "tall.weight": { shape: [32, 8] } }) },
        consts: [],
        weights: [
          {
            graph: "g",
            initializer: "tall.weight",
            bytes: bytesOf(32 * 8 * F32, 41),
            encoding: { codec: "f32" },
          },
        ],
        assets: [],
        provenance: { license: "apache-2.0", writer: "test" },
      },
      { partBytes: 512, blockBytes: 512 },
    );
    const pieces = (doc: Record<string, unknown>): { block: string }[] =>
      anyOf(doc).binding.g["tall.weight"].pieces;
    const blockOf = (doc: Record<string, unknown>, id: string): { part: number } =>
      anyOf(doc).blocks.find((block: { id: string }) => block.id === id);
    // Precondition: the writer places pieces in ascending part order (a form where the swap means
    // something).
    const original = JSON.parse(new TextDecoder().decode(written.modelDescriptorBytes));
    assertEquals(
      pieces(original).map((piece) => blockOf(original, piece.block).part),
      [2, 3],
    );
    await openWithModelDescriptor(written, () => {});

    await assertRejects(
      () =>
        openWithModelDescriptor(written, (doc) => {
          const [first, second] = pieces(doc).map((piece) => blockOf(doc, piece.block));
          [first.part, second.part] = [second.part, first.part];
        }),
      ContainerFormatError,
      "graph 'g' initializer 'tall.weight' piece[1]: part 2 comes before part 3 of the previous piece",
    );
  });

  it("the test writer's check: an input whose payload length does not match the declared shape fails before writing", async () => {
    const base = syntheticModel();
    await assertRejects(
      () =>
        writeModelContainer(
          {
            ...base,
            weights: base.weights.map((w) =>
              w.initializer === "enc.weight" ? { ...w, bytes: bytesOf(16 * 32 * 2 + 2, 1) } : w,
            ),
          },
          OPTIONS,
        ),
      ContainerFormatError,
      "the declaration determines",
    );
  });

  it("merge layer: checks payload length, piece coverage, rowAxis with pieces, group step and scale length against the declared shape", async () => {
    const base = syntheticModel();
    const rewrite = async (mutate: (input: ModelInput) => ModelInput): Promise<unknown> => {
      const written = await writeModelContainer(mutate(base), OPTIONS);
      return openContainer({ kind: "parts", parts: written.parts });
    };
    // The reader rejects a krm whose block length does not match the payload length of the
    // declared shape (tampering that bypasses the writer's check). It swaps the blocks of
    // front.bias (f16 [65] = 130 B, block 132 B) and conv.weight (int8 [4,3,8] = 96 B) — this
    // avoids double binding (rule 1), and both are in part 3 so the same-part rule for scale
    // (rule 3) holds too. The graphs merge in declaration order (front → voice), so front.bias is
    // hit first.
    const written = await writeModelContainer(base, OPTIONS);
    await assertRejects(
      () =>
        openWithModelDescriptor(written, (doc) => {
          const bias = anyOf(doc).binding.front["front.bias"];
          const conv = anyOf(doc).binding.voice["conv.weight"];
          [bias.block, conv.block] = [conv.block, bias.block];
        }),
      ContainerFormatError,
      "initializer 'front.bias': block 'w0009' length 96 is not payload 130 bytes + padding",
    );
    // Make an initializer with rowAxis 1 split into pieces (block limit smaller than one row).
    await assertRejects(
      () =>
        writeModelContainer(
          {
            ...base,
            graphs: { voice: declaration({ "conv.weight": { shape: [4, 3, 8] } }) },
            weights: base.weights.filter((w) => w.initializer === "conv.weight"),
            consts: [],
            assets: [],
          },
          { partBytes: 2048, blockBytes: 64 },
        ).then((written) => openContainer({ kind: "parts", parts: written.parts })),
      ContainerFormatError,
      "cannot be split into pieces",
    );
    // A per-channel codec whose groupSize differs from the row length.
    await assertRejects(
      () =>
        rewrite((input) => ({
          ...input,
          weights: input.weights.map((w) =>
            w.initializer === "big.weight"
              ? { ...w, encoding: { ...w.encoding, groupSize: 16 } }
              : w,
          ),
        })),
      ContainerFormatError,
      "equal the row length 32",
    );
    // A group codec whose groupSize is not a power of two ≥ 16 / does not divide the row length.
    await assertRejects(
      () =>
        rewrite((input) => ({
          ...input,
          weights: input.weights.map((w) =>
            w.initializer === "proj.weight"
              ? { ...w, encoding: { ...w.encoding, groupSize: 24 } }
              : w,
          ),
        })),
      ContainerFormatError,
      "power of two",
    );
    // The scale length does not match `rows × groups × 4`.
    await assertRejects(
      () =>
        rewrite((input) => ({
          ...input,
          weights: input.weights.map((w) =>
            w.initializer === "proj.weight"
              ? {
                  ...w,
                  encoding: { ...w.encoding, scale: { bytes: bytesOf(12 * F32, 1), dtype: "f32" } },
                }
              : w,
          ),
        })),
      ContainerFormatError,
      "scale",
    );
    // f16 cannot be paired with the semantic type i32.
    await assertRejects(
      () =>
        writeModelContainer(
          {
            ...base,
            graphs: { g: declaration({ idx: { shape: [8], dtype: "i32" } }) },
            consts: [],
            assets: [],
            weights: [
              {
                graph: "g",
                initializer: "idx",
                bytes: bytesOf(16, 3),
                encoding: { codec: "f16" },
              },
            ],
          },
          OPTIONS,
        ).then((written) => openContainer({ kind: "parts", parts: written.parts })),
      ContainerFormatError,
      "cannot be paired with",
    );
  });

  it("merge layer: rowAxis 1 is int8-sym only — even an i4 shape with equal lengths on both axes fails in openContainer", async () => {
    // [32,32] with group 16 gives a [32,2] scale whether read along axis 0 or axis 1, so the length
    // match cannot tell them apart.
    const written = await writeModelContainer(
      {
        graphs: { g: declaration({ "square.weight": { shape: [32, 32] } }) },
        consts: [],
        weights: [
          {
            graph: "g",
            initializer: "square.weight",
            bytes: bytesOf((32 * 32) / 2, 41),
            encoding: {
              codec: "int4-sym-g",
              rowAxis: 1,
              groupSize: 16,
              scale: { bytes: bytesOf(32 * 2 * F32, 42), dtype: "f32" },
            },
          },
        ],
        assets: [],
        provenance: { license: "apache-2.0", writer: "test" },
      },
      OPTIONS,
    );
    await assertRejects(
      () => openContainer({ kind: "parts", parts: written.parts }),
      ContainerFormatError,
      "rowAxis of codec 'int4-sym-g' must be 0 only (declared 1",
    );
    // Control: the i8 conv_transpose1d shape (rowAxis 1) in the same container is accepted.
    const opened = await openContainer({
      kind: "parts",
      parts: (await writeModelContainer(syntheticModel(), OPTIONS)).parts,
    });
    assertEquals(opened.graphs["voice"].supplies.get("conv.weight")?.encoding.rowAxis, 1);
  });

  it("does not resolve Object.prototype names (constructor / toString) from outside the declarations", async () => {
    // IR containers use null prototypes, so an initializer named toString can be declared
    // legitimately. Looking up a plain {} by name would make Object.prototype's functions "exist",
    // turning into a missed entry in the list of missing supplies or a TypeError.
    const input: ModelInput = {
      graphs: { g: declaration({ toString: { shape: [4, 4] }, a: { shape: [4, 4] } }) },
      consts: [],
      weights: ["toString", "a"].map((initializer, i) => ({
        graph: "g",
        initializer,
        bytes: bytesOf(16 * F32, 51 + i),
        encoding: { codec: "f32" },
      })),
      assets: [{ name: "style_vectors", role: "style-vectors", bytes: bytesOf(64, 53) }],
      provenance: { license: "apache-2.0", writer: "test" },
    };
    const written = await writeModelContainer(input, OPTIONS);
    const graph = parseGraphDescriptor(written.graphDescriptorBytes);

    // Remove toString from the binding table (the block is reattached under another name to keep
    // the reference) → it is listed as missing.
    const moved = parseModelDescriptor(
      mutateJson(written.modelDescriptorBytes, (doc) => {
        const supplies = anyOf(doc).binding.g;
        supplies["ghost"] = supplies["toString"];
        delete supplies["toString"];
      }),
    );
    assertThrows(
      () => validateAgainstGraph(moved, graph),
      ContainerFormatError,
      "missing [toString] / surplus [ghost]",
    );

    // The merge layer does not mistake the binding table's toString for Object.prototype's function
    // either.
    assertThrows(
      () =>
        bindDeclarations({
          graphs: { g: input.graphs["g"] },
          constants: [],
          binding: { g: {} },
          locate: () => {
            throw new Error("should fail before fetching a block");
          },
        }),
      ContainerFormatError,
      "no supply in the binding table",
    );

    // A const binding with graph name constructor is an undeclared graph.
    const withConst = await writeModelContainer(syntheticModel(), OPTIONS);
    assertThrows(
      () =>
        parseGraphDescriptor(
          mutateJson(withConst.graphDescriptorBytes, (doc) => {
            anyOf(doc).const.constants[0].graph = "constructor";
          }),
        ),
      ContainerFormatError,
      "undeclared graph 'constructor'",
    );

    // An asset named constructor is an undeclared asset.
    const opened = await openContainer({ kind: "parts", parts: written.parts });
    assertThrows(() => opened.asset("constructor"), ContainerFormatError, "Undeclared asset");
  });
});
