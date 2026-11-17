// The converting block source and the writer on a small synthetic container. The target is written
// by the runtime's test writer (the mirror of the exporter's layout rules) with the exporter's own
// encodings of the fixture tensors (`fixtures/encoders.json`), so "the same weights as the
// distributed preset" holds by construction: a weight tensor split into two pieces, an int8 tensor
// on row axis 1, payloads with tail padding, bf16 and f32 tensors, a const and a dedicated asset
// part. The source tensors are the fixtures' source values. No GPU, no local mirror.

import { describe, it } from "vite-plus/test";
import { assert, assertEquals, assertRejects, assertStringIncludes } from "@std/assert";
import {
  type BlockSource,
  type IrDeclaration,
  openContainer,
  parseIrDeclarationValue,
} from "@karume/runtime";
import { sha256Hex } from "@karume/runtime/container";
import {
  type ContainerSink,
  ConvertError,
  ENCODED_CODECS,
  INVALID_SHA256,
  openConvertingSource,
  readPlanCodecs,
  type SourceTensor,
  type TensorDtype,
  writeContainer,
} from "../mod.ts";
import {
  type EncodingInput,
  type TensorInput,
  writeModelContainer,
  type WrittenContainer,
} from "../../runtime/tests/helpers/container-write.ts";
import { writeHeader } from "../../runtime/src/format/container/header.ts";
import fixtures from "./fixtures/encoders.json" with { type: "json" };

const GRAPH = "transformer";

const hexBytes = (hex: string): Uint8Array<ArrayBuffer> =>
  Uint8Array.from({ length: hex.length / 2 }, (_, i) => parseInt(hex.slice(2 * i, 2 * i + 2), 16));

const fixture = (name: string): (typeof fixtures.cases)[number] => {
  const found = fixtures.cases.find((entry) => entry.name === name);
  if (found === undefined) throw new Error(`fixture: no case ${name}`);
  return found;
};

/** The smallest IR v2 graph that consumes each initializer in one node. */
const declaration = (shapes: Readonly<Record<string, readonly number[]>>): IrDeclaration => {
  const names = Object.keys(shapes);
  return parseIrDeclarationValue({
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
    nodes: names.map((name, i) => ({ op: "matmul", ins: ["x", name], outs: [`y${i}`], attrs: {} })),
  });
};

/** One weight of the synthetic target: its initializer, the fixture it is stored from, its codec. */
type Weight = {
  readonly initializer: string;
  readonly fixture: string;
};

const WEIGHTS: readonly Weight[] = [
  { initializer: "blocks.0.w", fixture: "int8-rows" },
  { initializer: "blocks.0.t", fixture: "int8-axis1" },
  { initializer: "blocks.0.s", fixture: "int8-small" },
  { initializer: "blocks.0.n", fixture: "bf16-from-f32" },
  { initializer: "blocks.0.c", fixture: "bf16-from-bf16" },
  { initializer: "blocks.0.b", fixture: "f32-from-bf16" },
];

const CONST_ROPE = Uint8Array.from({ length: 64 }, (_, i) => (i * 37) & 0xff);
const ROPE_BASE = Uint8Array.from({ length: 30 }, (_, i) => (i * 11 + 3) & 0xff);

const encodingOf = (name: string): EncodingInput => {
  const entry = fixture(name);
  if (entry.codec === "bf16") return { codec: "bf16" };
  if (entry.codec === "f32") return { codec: "f32" };
  const rows = entry.shape[entry.rowAxis];
  return {
    codec: "int8-sym",
    rowAxis: entry.rowAxis === 1 ? 1 : 0,
    groupSize: entry.shape.reduce((count, dim) => count * dim, 1) / rows,
    scale: { bytes: hexBytes(entry.scale), dtype: "f32" },
  };
};

/**
 * The synthetic target: 512-byte parts and a 200-byte block limit, so the [8, 47] int8 weight is
 * two pieces of 4 rows and the weights fill two parts (2 and 3) with gaps between blocks; part 1
 * is the const region and part 4 the asset.
 */
const writeTarget = async (
  extra: readonly TensorInput[] = [],
  extraShapes: Readonly<Record<string, readonly number[]>> = {},
): Promise<WrittenContainer> =>
  writeModelContainer(
    {
      graphs: {
        [GRAPH]: declaration({
          ...Object.fromEntries(
            WEIGHTS.map((weight) => [weight.initializer, fixture(weight.fixture).shape]),
          ),
          ...extraShapes,
          "const.rope": [16],
        }),
      },
      consts: [
        {
          graph: GRAPH,
          initializer: "const.rope",
          bytes: CONST_ROPE.slice(),
          encoding: { codec: "f32" },
        },
      ],
      weights: [
        ...WEIGHTS.map((weight) => ({
          graph: GRAPH,
          initializer: weight.initializer,
          bytes: hexBytes(fixture(weight.fixture).data),
          encoding: encodingOf(weight.fixture),
        })),
        ...extra,
      ],
      assets: [
        { name: "rope_base", role: "rope-base", bytes: ROPE_BASE.slice(), dedicatedPart: true },
      ],
      provenance: { license: "apache-2.0" },
    },
    { partBytes: 512, blockBytes: 200 },
  );

/** A `BlockSource` over the target's parts that records which parts were read. */
const recordingSource = (
  parts: readonly Uint8Array<ArrayBuffer>[],
  verified: boolean,
): { readonly source: BlockSource; readonly partsRead: Set<number> } => {
  const partsRead = new Set<number>();
  return {
    partsRead,
    source: {
      partCount: parts.length,
      verified,
      partLength: (index) => parts[index].byteLength,
      read: (part, offset, length) => {
        partsRead.add(part);
        return Promise.resolve(parts[part].slice(offset, offset + length));
      },
    },
  };
};

const DTYPES: readonly TensorDtype[] = ["F32", "F16", "BF16"];
const dtypeOf = (name: string): TensorDtype => {
  const dtype = DTYPES.find((candidate) => candidate === name);
  if (dtype === undefined) throw new Error(`fixture: unknown source dtype ${name}`);
  return dtype;
};

/** The fixtures' source tensors under their initializer names, each counting its reads. */
const sourceTensors = (): {
  readonly tensors: Map<string, SourceTensor>;
  readonly reads: Map<string, number>;
} => {
  const reads = new Map<string, number>();
  const tensors = new Map<string, SourceTensor>();
  for (const weight of WEIGHTS) {
    const entry = fixture(weight.fixture);
    tensors.set(weight.initializer, {
      dtype: dtypeOf(entry.sourceDtype),
      shape: entry.shape,
      read: () => {
        reads.set(weight.initializer, (reads.get(weight.initializer) ?? 0) + 1);
        return Promise.resolve(hexBytes(entry.source));
      },
    });
  }
  return { tensors, reads };
};

/** A sink that keeps every part it is given (and the order of the writes). */
const collectingSink = (): {
  readonly sink: ContainerSink;
  readonly parts: Uint8Array<ArrayBuffer>[];
  readonly order: number[];
} => {
  const parts: Uint8Array<ArrayBuffer>[] = [];
  const order: number[] = [];
  return {
    parts,
    order,
    sink: {
      write: (part, offset, bytes) => {
        assertEquals(offset, 0);
        parts[part] = bytes.slice();
        order.push(part);
        return Promise.resolve();
      },
    },
  };
};

const opened = async (written: WrittenContainer, verified = false) => {
  const target = recordingSource(written.parts, verified);
  const { tensors, reads } = sourceTensors();
  const converting = await openConvertingSource({ target: target.source, tensors });
  return { target, tensors, reads, converting };
};

describe("the converting source", () => {
  it("serves every weight block, a range inside a block and a range across blocks and gaps as the target's bytes", async () => {
    const written = await writeTarget();
    const { converting } = await opened(written);
    const view = await openContainer({ kind: "source", source: converting.source });
    const model = view.model;
    assert(model !== undefined);
    // Every block read whole (tail padding included) is the target's block.
    for (const block of model.blocks) {
      const bytes = await converting.source.read(block.part, block.offset, block.length);
      assertEquals(
        bytes,
        written.parts[block.part].slice(block.offset, block.offset + block.length),
      );
    }
    // The weight pieces really are two, in part 2 with a gap after each.
    const pieced = view.graphs[GRAPH].supplies.get("blocks.0.w");
    assertEquals(
      pieced?.blocks.map((block) => block.rows),
      [
        [0, 4],
        [4, 8],
      ],
    );
    // A range inside a block, and one from inside piece 1 across the gap, the scale and the next
    // gap into piece 2.
    for (const [part, offset, length] of [
      [2, 10, 50],
      [2, 150, 120],
      [3, 0, written.parts[3].byteLength],
      [2, 0, written.parts[2].byteLength],
    ] as const) {
      assertEquals(
        await converting.source.read(part, offset, length),
        written.parts[part].slice(offset, offset + length),
      );
    }
  });

  it("reads a tensor once for its codes and its scale, and keeps no tensor but the last one asked for", async () => {
    const written = await writeTarget();
    const { converting, reads } = await opened(written);
    const view = await openContainer({ kind: "source", source: converting.source });
    const supply = (name: string) => {
      const found = view.graphs[GRAPH].supplies.get(name);
      if (found?.scale === undefined) throw new Error(`no quantized supply ${name}`);
      return { data: found.blocks[0], scale: found.scale };
    };
    const t = supply("blocks.0.t");
    const s = supply("blocks.0.s");
    const read = (block: { part: number; offset: number; length: number }) =>
      converting.source.read(block.part, block.offset, block.length);
    await read(t.data);
    await read(t.scale);
    assertEquals(reads.get("blocks.0.t"), 1);
    await read(s.data);
    await read(t.scale);
    assertEquals(reads.get("blocks.0.t"), 2);
    assertEquals(reads.get("blocks.0.s"), 1);
  });

  describe("retainEncoded", () => {
    /** Every block of one initializer: its data blocks (pieces), then its scale (or the scale's pieces). */
    const blocksOf = (
      view: Awaited<ReturnType<typeof openContainer>>,
      name: string,
    ): { readonly part: number; readonly offset: number; readonly length: number }[] => {
      const supply = view.graphs[GRAPH].supplies.get(name);
      if (supply === undefined) throw new Error(`no supply ${name}`);
      const scale = supply.scale === undefined ? [] : (supply.scale.pieces ?? [supply.scale]);
      return [...supply.blocks, ...scale];
    };

    /**
     * Reads every block of every weight, initializer by initializer, twice over (in the second pass
     * every initializer but the last is read after a later one), each block twice in a row, and
     * checks the bytes against the target's; returns how often each tensor was read.
     */
    const readTwice = async (retainEncoded: boolean | undefined) => {
      const written = await writeTarget();
      const target = recordingSource(written.parts, false);
      const { tensors, reads } = sourceTensors();
      const converting = await openConvertingSource({
        target: target.source,
        tensors,
        ...(retainEncoded === undefined ? {} : { retainEncoded }),
      });
      const view = await openContainer({ kind: "source", source: converting.source });
      for (let pass = 0; pass < 2; pass += 1) {
        for (const weight of WEIGHTS) {
          for (const block of blocksOf(view, weight.initializer)) {
            for (let again = 0; again < 2; again += 1) {
              assertEquals(
                await converting.source.read(block.part, block.offset, block.length),
                written.parts[block.part].slice(block.offset, block.offset + block.length),
              );
            }
          }
        }
      }
      return Object.fromEntries(
        WEIGHTS.map(({ initializer }) => [initializer, reads.get(initializer)]),
      );
    };

    it("on: reads each tensor once — a block read again, and a block of an earlier initializer read after a later one, come from memory", async () => {
      assertEquals(
        await readTwice(true),
        Object.fromEntries(WEIGHTS.map(({ initializer }) => [initializer, 1])),
      );
    });

    it("off (the default, and false): reads an earlier initializer's tensor again after a later one, once per pass", async () => {
      const twice = Object.fromEntries(WEIGHTS.map(({ initializer }) => [initializer, 2]));
      assertEquals(await readTwice(undefined), twice);
      assertEquals(await readTwice(false), twice);
    });

    it("on: a failed encoding is not kept, so a later read tries again", async () => {
      const written = await writeTarget();
      const target = recordingSource(written.parts, false);
      const { tensors, reads } = sourceTensors();
      const n = tensors.get("blocks.0.n");
      assert(n !== undefined);
      // Fault: the first read of the tensor fails, the next ones succeed.
      let failures = 1;
      tensors.set("blocks.0.n", {
        ...n,
        read: () => (failures-- > 0 ? Promise.reject(new Error("transient")) : n.read()),
      });
      const converting = await openConvertingSource({
        target: target.source,
        tensors,
        retainEncoded: true,
      });
      const view = await openContainer({ kind: "source", source: converting.source });
      const [block] = blocksOf(view, "blocks.0.n");
      const read = () => converting.source.read(block.part, block.offset, block.length);
      await assertRejects(read, Error, "transient");
      const expected = written.parts[block.part].slice(block.offset, block.offset + block.length);
      assertEquals(await read(), expected);
      assertEquals(await read(), expected);
      assertEquals(reads.get("blocks.0.n"), 1);
    });
  });

  it("serves part 0 with the invalid sha256 for converted blocks and their parts, and the true one for the rest", async () => {
    const written = await writeTarget();
    const { converting } = await opened(written);
    const part0 = await converting.source.read(0, 0, converting.source.partLength(0));
    assertEquals(part0.byteLength, written.parts[0].byteLength);
    const view = await openContainer({ kind: "source", source: converting.source });
    const model = view.model;
    assert(model !== undefined);
    const target = written.model;
    for (const [i, block] of model.blocks.entries()) {
      const expected = block.role === "asset" ? target.blocks[i].sha256 : INVALID_SHA256;
      assertEquals([block.id, block.sha256], [target.blocks[i].id, expected]);
    }
    assertEquals(
      model.parts.map((part) => part.sha256),
      target.parts.map((part) =>
        part.index === 2 || part.index === 3 ? INVALID_SHA256 : part.sha256,
      ),
    );
    // The graph descriptor (with the const hashes) is the target's, byte for byte.
    const graphStart = 24;
    assertEquals(
      part0.slice(graphStart, graphStart + written.graphDescriptorBytes.byteLength),
      written.graphDescriptorBytes,
    );
  });

  it("reports every problem of the tensor list together, before any tensor is read", async () => {
    const written = await writeTarget(
      [
        {
          graph: GRAPH,
          initializer: "blocks.0.h",
          bytes: new Uint8Array(8),
          encoding: { codec: "f16" },
        },
      ],
      { "blocks.0.h": [2, 2] },
    );
    const target = recordingSource(written.parts, false);
    const { tensors, reads } = sourceTensors();
    tensors.delete("blocks.0.w");
    const t = tensors.get("blocks.0.t");
    const s = tensors.get("blocks.0.s");
    assert(t !== undefined && s !== undefined);
    tensors.set("blocks.0.t", { ...t, shape: [5, 3, 4] });
    tensors.set("blocks.0.s", { ...s, shape: [3, 6] });
    tensors.set("blocks.0.h", {
      dtype: "F32",
      shape: [2, 2],
      read: () => Promise.reject(new Error("read")),
    });
    const error = await assertRejects(
      () => openConvertingSource({ target: target.source, tensors }),
      ConvertError,
    );
    for (const line of [
      "graph 'transformer' initializer 'blocks.0.w': the tensor list has no tensor 'blocks.0.w'",
      "initializer 'blocks.0.t': tensor 'blocks.0.t' has shape [5, 3, 4] (60 elements), the plan declares [3, 5, 4] (60 elements)",
      "initializer 'blocks.0.s': tensor 'blocks.0.s' has shape [3, 6] (18 elements), the plan declares [3, 5] (15 elements)",
      "initializer 'blocks.0.h': codec 'f16' is not one this package encodes",
    ]) {
      assertStringIncludes(error.message, line);
    }
    assertEquals(reads.size, 0);
    assertEquals([...target.partsRead], [0]);
  });

  it("lists the tensors the plan does not use, and maps initializer names to tensor names", async () => {
    const written = await writeTarget();
    const target = recordingSource(written.parts, false);
    const { tensors } = sourceTensors();
    const renamed = new Map([...tensors].map(([name, tensor]) => [`model.${name}`, tensor]));
    renamed.set("model.extra.weight", {
      dtype: "F32",
      shape: [1],
      read: () => Promise.reject(new Error("read")),
    });
    const converting = await openConvertingSource({
      target: target.source,
      tensors: renamed,
      tensorName: (initializer) => `model.${initializer}`,
    });
    assertEquals(converting.unusedTensors, ["model.extra.weight"]);
  });

  it("refuses a target whose model descriptor is not in canonical form (its hashes could not be replaced in place)", async () => {
    const written = await writeTarget();
    const text = new TextDecoder().decode(written.modelDescriptorBytes);
    const spaced = new TextEncoder().encode(
      text.replace(`"format":"karume-model"`, `"format": "karume-model"`),
    );
    assertEquals(spaced.byteLength, written.modelDescriptorBytes.byteLength + 1);
    const header = writeHeader({
      kind: "model",
      version: 1,
      graphDescriptorLength: written.graphDescriptorBytes.byteLength,
      modelDescriptorLength: spaced.byteLength,
    });
    const part0 = new Uint8Array(
      new ArrayBuffer(
        header.byteLength + written.graphDescriptorBytes.byteLength + spaced.byteLength,
      ),
    );
    part0.set(header, 0);
    part0.set(written.graphDescriptorBytes, header.byteLength);
    part0.set(spaced, header.byteLength + written.graphDescriptorBytes.byteLength);
    const target = recordingSource([part0, ...written.parts.slice(1)], false);
    // The runtime's reader accepts it: only the conversion needs the canonical form.
    await openContainer({ kind: "source", source: target.source });
    const error = await assertRejects(
      () => openConvertingSource({ target: target.source, tensors: sourceTensors().tensors }),
      ConvertError,
    );
    assertStringIncludes(error.message, "not in canonical form");
  });

  it("refuses at read time a value the target codec cannot store, naming the initializer", async () => {
    const written = await writeTarget();
    const target = recordingSource(written.parts, false);
    const { tensors } = sourceTensors();
    tensors.set("blocks.0.n", {
      dtype: "F32",
      shape: [3, 3],
      read: () =>
        Promise.resolve(new Uint8Array(new Float32Array([1, 1.1, 2, 3, 4, 5, 6, 7, 8]).buffer)),
    });
    const converting = await openConvertingSource({ target: target.source, tensors });
    const view = await openContainer({ kind: "source", source: converting.source });
    const error = await assertRejects(
      () => view.readBlock(view.graphs[GRAPH].supplies.get("blocks.0.n")?.blocks[0]?.id ?? ""),
      ConvertError,
    );
    assertStringIncludes(
      error.message,
      "graph 'transformer' initializer 'blocks.0.n' (tensor 'blocks.0.n'): element 1",
    );
  });

  it("names the parts it builds: the weight parts, not part 0, the const part or the asset part", async () => {
    const written = await writeTarget();
    const { converting } = await opened(written);
    assertEquals(converting.convertedParts, [2, 3]);
  });

  it("checks what it passes through from an unverified target against the declared sha256", async () => {
    const written = await writeTarget();
    const tampered = written.parts.map((part) => part.slice());
    tampered[1][5] ^= 0xff;
    const target = recordingSource(tampered, false);
    const { tensors } = sourceTensors();
    const converting = await openConvertingSource({ target: target.source, tensors });
    const error = await assertRejects(() => converting.source.read(1, 0, 16), ConvertError);
    assertStringIncludes(error.message, "part 1: the target's block at 0 has sha256");
    // The asset part is checked the same way and passes.
    assertEquals(await converting.source.read(4, 0, 32), written.parts[4]);
  });
});

describe("the plan read before a conversion (readPlanCodecs)", () => {
  it("gives every initializer the plan stores in its weight parts with its codec, from part 0 alone", async () => {
    const written = await writeTarget(
      [
        {
          graph: GRAPH,
          initializer: "blocks.0.h",
          bytes: new Uint8Array(8),
          encoding: { codec: "f16" },
        },
      ],
      { "blocks.0.h": [2, 2] },
    );
    const target = recordingSource(written.parts, false);
    const codecs = await readPlanCodecs(target.source);
    // The const (const.rope) is not converted, so it is not listed; the f16 initializer is listed
    // with its codec although this package does not encode it (the caller decides on it).
    assertEquals(
      codecs
        .map(({ graph, initializer, codec }) => [graph, initializer, codec])
        .sort(([, left = ""], [, right = ""]) => left.localeCompare(right)),
      [
        [GRAPH, "blocks.0.b", "f32"],
        [GRAPH, "blocks.0.c", "bf16"],
        [GRAPH, "blocks.0.h", "f16"],
        [GRAPH, "blocks.0.n", "bf16"],
        [GRAPH, "blocks.0.s", "int8-sym"],
        [GRAPH, "blocks.0.t", "int8-sym"],
        [GRAPH, "blocks.0.w", "int8-sym"],
      ],
    );
    assertEquals([...target.partsRead], [0]);
    assertEquals(ENCODED_CODECS, ["int8-sym", "bf16", "f32"]);
  });
});

describe("the writer", () => {
  it("writes the target's parts byte for byte, part 0 last, and returns their lengths and sha256", async () => {
    const written = await writeTarget();
    const { converting, target } = await opened(written);
    const { sink, parts, order } = collectingSink();
    const result = await writeContainer(converting.source, sink);
    assertEquals(order, [1, 2, 3, 4, 0]);
    assertEquals(parts, [...written.parts]);
    assertEquals(
      result.parts,
      await Promise.all(
        written.parts.map(async (part) => ({
          length: part.byteLength,
          sha256: await sha256Hex(part),
        })),
      ),
    );
    assertEquals(result.descriptor, {
      graph: {
        length: written.graphDescriptorBytes.byteLength,
        sha256: await sha256Hex(written.graphDescriptorBytes),
      },
      model: {
        length: written.modelDescriptorBytes.byteLength,
        sha256: await sha256Hex(written.modelDescriptorBytes),
      },
    });
    // No weight part of the target was read: only part 0, the const part and the asset part.
    assertEquals(
      [...target.partsRead].sort((a, b) => a - b),
      [0, 1, 4],
    );
  });

  it("writes a container that opens from plain bytes, every hash checked, and binds", async () => {
    const written = await writeTarget();
    const { converting } = await opened(written);
    const { sink, parts } = collectingSink();
    const result = await writeContainer(converting.source, sink);
    const reopened = await openContainer({ kind: "parts", parts }, result.descriptor);
    assertEquals(reopened.source.verified, false);
    const model = reopened.model;
    assert(model !== undefined);
    for (const block of [...model.blocks, ...reopened.graph.const.blocks]) {
      await reopened.readBlock(block.id);
    }
    assertEquals(
      [...reopened.graphs[GRAPH].supplies.keys()].sort(),
      [...WEIGHTS.map((weight) => weight.initializer), "const.rope"].sort(),
    );
  });
});
