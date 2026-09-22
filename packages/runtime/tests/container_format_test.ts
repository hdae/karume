/**
 * コンテナ形式（`krm` / `krg`）の読み手 — ヘッダ・2 文書 descriptor・block 取得・`krg` 抽出・合流層。
 * docs/container-v1.md の規則を「正常系から 1 点だけ壊す」形で固定する。
 */

import { assertEquals, assertRejects, assertThrows } from "@std/assert";
import { describe, it } from "@std/testing/bdd";
import {
  parseGraphDescriptor,
  parseModelDescriptor,
  sha256Hex,
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
} from "../src/format/container/limits.ts";
import { type BlockSource, openContainer } from "../src/format/container/open.ts";
import { type IrDeclaration, parseIrDeclaration } from "../src/format/ir.ts";
import {
  type ModelInput,
  writeGraphContainer,
  writeModelContainer,
} from "./helpers/container-write.ts";

// ---------------------------------------------------------------------------
// 合成モデル
// ---------------------------------------------------------------------------

/** 決定的な疑似乱数バイト列（seed から xorshift32）。 */
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

/** initializer ごとに 1 ノードで消費する最小の IR v2 グラフ。 */
const declaration = (initializers: Readonly<Record<string, InitializerSpec>>): IrDeclaration => {
  const names = Object.keys(initializers);
  return parseIrDeclaration(JSON.stringify({
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
        names.map((
          name,
        ) => [name, { dtype: initializers[name].dtype ?? "f32", shape: initializers[name].shape }]),
      ),
      ...Object.fromEntries(names.map((_, i) => [`y${i}`, { dtype: "f32", shape: ["T", 4] }])),
    },
    nodes: names.map((name, i) => ({ op: "matmul", ins: ["x", name], outs: [`y${i}`], attrs: {} })),
  }));
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
    // 長さが 4 の倍数でない const（末尾の詰め物を書き手が焼くことの確認）。
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
      // i4 + companion scale（規則③で同一 part に置かれる）。行長 64 / group 32 = 2 group。
      graph: "text_encoder",
      initializer: "proj.weight",
      bytes: bytesOf(12 * 64 / 2, 22),
      encoding: {
        codec: "int4-sym-g",
        groupSize: 32,
        scale: { bytes: bytesOf(12 * 2 * F32, 23), dtype: "f32" },
      },
    },
    {
      // block 上限（テストでは 512 B）を超えるので piece 分割される。
      graph: "front",
      initializer: "big.weight",
      bytes: bytesOf(32 * 32, 24),
      encoding: {
        codec: "int8-sym",
        groupSize: 32,
        scale: { bytes: bytesOf(32 * F32, 25), dtype: "f32" },
      },
    },
    // 長さが 4 の倍数でない f16（末尾の詰め物）。
    {
      graph: "front",
      initializer: "front.bias",
      bytes: bytesOf(65 * 2, 26),
      encoding: { codec: "f16" },
    },
    {
      // conv_transpose1d 形（rowAxis 1 の per-channel scale）。
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

// ---------------------------------------------------------------------------
// ヘッダ
// ---------------------------------------------------------------------------

describe("container header", () => {
  it("krm / krg を往復し、種別は magic だけが持つ", () => {
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

  it("未知の magic・未対応の版・種別と長さの矛盾・上限超過を拒否する", () => {
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
      "版",
    );
    assertThrows(
      () => readHeader(corrupt((v) => v.setBigUint64(16, 0n, true))),
      ContainerFormatError,
      "krm なのに",
    );
    assertThrows(
      () => readHeader(corrupt((v) => v.setBigUint64(8, 0n, true))),
      ContainerFormatError,
      "長さが 0",
    );
    assertThrows(
      () => readHeader(corrupt((v) => v.setBigUint64(8, 1n << 60n, true))),
      ContainerFormatError,
      "安全整数",
    );
    assertThrows(
      () => readHeader(corrupt((v) => v.setBigUint64(8, BigInt(MAX_DESCRIPTOR_BYTES + 1), true))),
      ContainerFormatError,
      "上限",
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
      "しかない",
    );
  });

  it("単一形の part offset は 64 B 整列し、長さ 0 の part には詰め物を挟まない", () => {
    assertEquals(derivePartOffsets(100, [0, 30]), { offsets: [0, 100, 128], totalLength: 158 });
    assertEquals(derivePartOffsets(64, [10, 0]), { offsets: [0, 64, 74], totalLength: 74 });
    assertEquals(derivePartOffsets(1, [1]).offsets[1] % BLOCK_START_ALIGN, 0);
  });
});

// ---------------------------------------------------------------------------
// 書く → 読む
// ---------------------------------------------------------------------------

describe("container round trip", () => {
  it("単一形を開くと全 block が sha256 検証つきで戻り、合流層が供給計画を出す", async () => {
    const written = await writeModelContainer(syntheticModel(), OPTIONS);
    const opened = await openContainer({ kind: "bytes", bytes: written.single });
    assertEquals(opened.header.kind, "model");
    assertEquals(Object.keys(opened.graphs).sort(), ["front", "text_encoder", "voice"]);

    const supplies = opened.graphs.front.supplies;
    const big = supplies.get("big.weight");
    if (big === undefined) throw new Error("big.weight の供給が無い");
    // 32 行 × 32 B = 1024 B > block 上限 512 B → piece 分割される。
    assertEquals(big.blocks.length > 1, true);
    assertEquals(big.blocks[0].rows[0], 0);
    assertEquals(big.blocks[big.blocks.length - 1].rows[1], 32);
    assertEquals(big.scale?.part, big.blocks[0].part);
    // piece 列の block を読んで連結すると元の payload が戻る。
    const joined = new Uint8Array(1024);
    let cursor = 0;
    for (const block of big.blocks) {
      const bytes = await opened.readBlock(block.id);
      joined.set(bytes.subarray(0, block.payloadBytes), cursor);
      cursor += block.payloadBytes;
    }
    assertEquals(joined, bytesOf(32 * 32, 24));

    // 長さが 4 の倍数でない f16 は詰め物込みで宣言され、payload 長は宣言 shape から決まる。
    const bias = supplies.get("front.bias");
    assertEquals(bias?.blocks[0].length, 132);
    assertEquals(bias?.blocks[0].payloadBytes, 130);
    assertEquals(
      (await opened.readBlock(bias!.blocks[0].id)).subarray(130),
      new Uint8Array([0, 0]),
    );

    // const 領域は part 1・origin は const。
    const rope = supplies.get("const.rope");
    assertEquals(rope?.origin, "const");
    assertEquals(rope?.blocks[0].part, 1);
    assertEquals(await opened.readBlock(rope!.blocks[0].id), bytesOf(64 * F32, 11));

    // shared 宣言は供給計画を持たない。
    assertEquals(opened.graphs.text_encoder.supplies.has("model.lm_head.weight"), false);
    // 専用 part の資産は単独で 1 part を占める。
    const ple = opened.model?.assets["ple_table"];
    const pleBlock = opened.model?.blocks.find((block) => block.id === ple?.block);
    assertEquals(opened.model?.blocks.filter((block) => block.part === pleBlock?.part).length, 1);
    assertEquals(opened.model?.codecs, ["f16", "int4-sym-g", "int8-sym"]);
  });

  it("分割形と単一形は descriptor も block もバイト単位に同一", async () => {
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

  it("krm から抜いた krg は直接書いた krg とバイト同一で、krg として開ける", async () => {
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
    // krg でも const 領域の block は読める（グラフ記述が const の束縛表を持つ — §9）。
    const rope = graphOnly.graphs.front.supplies.get("const.rope");
    assertEquals(await graphOnly.readBlock(rope!.blocks[0].id), bytesOf(64 * F32, 11));
    // 重みは供給計画を持たない（要求は宣言 shape から導く）。
    assertEquals(graphOnly.graphs.front.supplies.has("big.weight"), false);
  });

  it("const 領域が空でも krg 抽出が成り立つ（長さ 0 の part に詰め物を挟まない）", async () => {
    const model: ModelInput = {
      ...syntheticModel(),
      consts: [],
      graphs: {
        text_encoder: declaration({ "enc.weight": { shape: [16, 32] } }),
      },
      weights: [{
        graph: "text_encoder",
        initializer: "enc.weight",
        bytes: bytesOf(16 * 32 * 2, 21),
        encoding: { codec: "f16" },
      }],
      assets: [],
    };
    const written = await writeModelContainer(model, OPTIONS);
    assertEquals(written.parts[1].byteLength, 0);
    const opened = await openContainer({ kind: "parts", parts: written.parts });
    assertEquals(await opened.extractGraph(), (await writeGraphContainer(model, OPTIONS)).bytes);
  });

  it("外側の期待 hash は 2 文書を parse する前に検証する", async () => {
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
      "グラフ記述の sha256",
    );
    await assertRejects(
      () => openContainer({ kind: "bytes", bytes: written.graphContainer }, expect),
      ContainerFormatError,
      "krg なのに",
    );
  });

  it("block の 1 バイト改ざんは sha256 が、const 領域の改ざんは krg 抽出が捕まえる", async () => {
    const written = await writeModelContainer(syntheticModel(), OPTIONS);
    const parts = written.parts.map((part) => part.slice());
    const weight = written.model.blocks.find((block) => block.role === "weight");
    if (weight === undefined) throw new Error("weight block が無い");
    parts[weight.part][weight.offset + 3] ^= 0x80;
    const opened = await openContainer({ kind: "parts", parts });
    await assertRejects(
      () => opened.readBlock(weight.id),
      ContainerFormatError,
      "sha256 が宣言と違う",
    );
    parts[1][0] ^= 0x01;
    const again = await openContainer({ kind: "parts", parts });
    await assertRejects(() => again.extractGraph(), ContainerFormatError, "const 領域の sha256");
  });

  it("検証済みを名乗る取得元では block の sha256 を掛けず、名乗らない取得元では掛ける", async () => {
    // 取得層がファイル全体を検証した経路（hub の HF 取得元）は cold の 2 重 digest と warm の digest を
    // 避けるために verified を名乗る（ADR 0109 決定 7）。観測は「改ざんが素通りするか」で行う —
    // digest の回数そのものは外から数えられない。
    const written = await writeModelContainer(syntheticModel(), OPTIONS);
    const parts = written.parts.map((part) => part.slice());
    const weight = written.model.blocks.find((block) => block.role === "weight");
    if (weight === undefined) throw new Error("weight block が無い");
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
      "sha256 が宣言と違う",
    );
  });

  it("分割形の part 本数と長さは宣言と一致しなければならない", async () => {
    const written = await writeModelContainer(syntheticModel(), OPTIONS);
    await assertRejects(
      () => openContainer({ kind: "parts", parts: written.parts.slice(0, -1) }),
      ContainerFormatError,
      "本だが宣言は",
    );
    const shorter = written.parts.map((
      part,
      i,
    ) => (i === 2 ? part.subarray(0, part.byteLength - 4) : part));
    await assertRejects(
      () => openContainer({ kind: "parts", parts: shorter }),
      ContainerFormatError,
      "part 2 の長さ",
    );
    await assertRejects(
      () =>
        openContainer({
          kind: "bytes",
          bytes: written.single.subarray(0, written.single.byteLength - 1),
        }),
      ContainerFormatError,
      "単一形の長さ",
    );
  });
});

// ---------------------------------------------------------------------------
// descriptor の宣言検査
// ---------------------------------------------------------------------------

describe("container descriptor", () => {
  it("正準直列化した 2 文書は parse → serialize で同じバイト列に戻る", async () => {
    const written = await writeModelContainer(syntheticModel(), OPTIONS);
    const { serializeGraphDescriptor, serializeModelDescriptor } = await import(
      "../src/format/container/descriptor.ts"
    );
    assertEquals(
      serializeGraphDescriptor(parseGraphDescriptor(written.graphDescriptorBytes)),
      written.graphDescriptorBytes,
    );
    assertEquals(
      serializeModelDescriptor(parseModelDescriptor(written.modelDescriptorBytes)),
      written.modelDescriptorBytes,
    );
  });

  it("台帳に無い codec・台帳と食い違う packing・未知キーを拒否する", async () => {
    const { modelDescriptorBytes: bytes } = await writeModelContainer(syntheticModel(), OPTIONS);
    const supply = (doc: Record<string, unknown>) => anyOf(doc).binding.text_encoder["proj.weight"];
    rejectsModel("未知 codec", bytes, (doc) => {
      supply(doc).encoding.codec = "int3-sym";
    }, "台帳に無い codec");
    rejectsModel("packing 不一致", bytes, (doc) => {
      supply(doc).encoding.packing.blockElements = 4;
    }, "別の版の台帳");
    rejectsModel("未知キー", bytes, (doc) => {
      supply(doc).encoding.bits = 4;
    }, "未知のキー");
    rejectsModel("非量子化に groupSize", bytes, (doc) => {
      anyOf(doc).binding.text_encoder["enc.weight"].encoding.groupSize = 32;
    }, "量子化でないので書けない");
    rejectsModel("量子化に scale 無し", bytes, (doc) => {
      delete supply(doc).encoding.scale;
    }, "量子化なので必須");
    rejectsModel("codecs の集合が束縛表と違う", bytes, (doc) => {
      anyOf(doc).codecs = ["f16"];
    }, "codec 集合");
    const { graphDescriptorBytes } = await writeModelContainer(syntheticModel(), OPTIONS);
    assertThrows(
      () =>
        parseGraphDescriptor(mutateJson(graphDescriptorBytes, (doc) => {
          anyOf(doc).capabilities.features = ["range-fetch"];
        })),
      ContainerFormatError,
      "未知の機能",
    );
  });

  it("溢れて Infinity になる数・深すぎる入れ子・__proto__ を拒否する", async () => {
    const { graphDescriptorBytes: bytes } = await writeModelContainer(syntheticModel(), OPTIONS);
    const text = new TextDecoder().decode(bytes);
    const overflow = new TextEncoder().encode(
      text.replace('"length":', '"length":1e999,"x":'),
    ) as Uint8Array<ArrayBuffer>;
    assertThrows(() => parseGraphDescriptor(overflow), ContainerFormatError);
    const nested = new TextEncoder().encode(
      `{"a":${"[".repeat(70)}1${"]".repeat(70)}}`,
    ) as Uint8Array<ArrayBuffer>;
    assertThrows(() => parseGraphDescriptor(nested), ContainerFormatError, "深さ上限");
    const proto = new TextEncoder().encode('{"__proto__":{"format":1}}') as Uint8Array<ArrayBuffer>;
    assertThrows(() => parseGraphDescriptor(proto), ContainerFormatError, "__proto__");
  });

  it("規則①〜⑤: 二重束縛・余剰 block・piece の穴・role の取り違え・scale の別 part を拒否する", async () => {
    const written = await writeModelContainer(syntheticModel(), OPTIONS);
    const bytes = written.modelDescriptorBytes;
    const bigSupply = (doc: Record<string, unknown>) => anyOf(doc).binding.front["big.weight"];
    rejectsModel("二重束縛（規則①）", bytes, (doc) => {
      anyOf(doc).binding.front["front.bias"].block = bigSupply(doc).pieces[0].block;
    }, "二重に束縛");
    rejectsModel("余剰 block（規則⑤）", bytes, (doc) => {
      delete anyOf(doc).assets["style_vectors"];
    }, "参照されていない");
    rejectsModel("piece の穴（規則④）", bytes, (doc) => {
      bigSupply(doc).pieces[1].rows[0] += 1;
    }, "から続かない");
    rejectsModel("piece 1 本", bytes, (doc) => {
      bigSupply(doc).pieces = [bigSupply(doc).pieces[0]];
    }, "2 本以上");
    rejectsModel("block と pieces の両方", bytes, (doc) => {
      bigSupply(doc).block = "w0001";
    }, "どちらか一方");
    rejectsModel("role の取り違え", bytes, (doc) => {
      const scaleId = anyOf(doc).binding.text_encoder["proj.weight"].encoding.scale.block;
      anyOf(doc).blocks.find((block: { id: string }) => block.id === scaleId).role = "weight";
    }, "role は");
    rejectsModel("scale が別 part（規則③）", bytes, (doc) => {
      const scaleId = anyOf(doc).binding.text_encoder["proj.weight"].encoding.scale.block;
      const block = anyOf(doc).blocks.find((block: { id: string }) => block.id === scaleId);
      const other = anyOf(doc).parts.find((part: { index: number }) =>
        part.index >= 2 && part.index !== block.part
      );
      block.part = other.index;
      block.offset = Math.ceil(other.length / 64) * 64;
      other.length = block.offset + block.length;
    }, "同一 part MUST");
    rejectsModel("offset の整列", bytes, (doc) => {
      anyOf(doc).blocks[0].offset += 4;
    }, "の倍数でない");
    rejectsModel("parts の添字", bytes, (doc) => {
      anyOf(doc).parts[0].index = 0;
    }, "index が 1 でない");
    rejectsModel("part 1 への block", bytes, (doc) => {
      anyOf(doc).blocks[0].part = 1;
    }, "2 未満");
  });

  it("束縛表のキー集合は shared でも const でもない initializer の集合と完全一致する（規則⑤）", async () => {
    const written = await writeModelContainer(syntheticModel(), OPTIONS);
    const withBinding = async (mutate: (doc: Record<string, unknown>) => void): Promise<void> => {
      const parts = [...written.parts];
      const modelBytes = mutateJson(written.modelDescriptorBytes, mutate);
      const header = writeHeader({
        kind: "model",
        version: 1,
        graphDescriptorLength: written.graphDescriptorBytes.byteLength,
        modelDescriptorLength: modelBytes.byteLength,
      });
      const part0 = new Uint8Array(
        new ArrayBuffer(
          HEADER_BYTES + written.graphDescriptorBytes.byteLength + modelBytes.byteLength,
        ),
      );
      part0.set(header, 0);
      part0.set(written.graphDescriptorBytes, HEADER_BYTES);
      part0.set(modelBytes, HEADER_BYTES + written.graphDescriptorBytes.byteLength);
      parts[0] = part0;
      await openContainer({ kind: "parts", parts });
    };
    await assertRejects(
      () =>
        withBinding((doc) => {
          // shared 宣言を束縛表に載せる（余剰）。
          anyOf(doc).binding.text_encoder["model.lm_head.weight"] =
            anyOf(doc).binding.text_encoder["enc.weight"];
          delete anyOf(doc).binding.text_encoder["enc.weight"];
        }),
      ContainerFormatError,
      "不足 [enc.weight] / 余剰 [model.lm_head.weight]",
    );
  });

  it("合流層: payload 長・piece の被覆・rowAxis と piece・group の刻み・scale 長を宣言 shape と突き合わせる", async () => {
    const base = syntheticModel();
    const rewrite = async (mutate: (input: ModelInput) => ModelInput): Promise<unknown> => {
      const written = await writeModelContainer(mutate(base), OPTIONS);
      return openContainer({ kind: "parts", parts: written.parts });
    };
    // payload 長が宣言 shape と合わない → 書き手が先に落とす（宣言から決まる値の突合）。
    await assertRejects(
      () =>
        rewrite((input) => ({
          ...input,
          weights: input.weights.map((
            w,
          ) => (w.initializer === "enc.weight" ? { ...w, bytes: bytesOf(16 * 32 * 2 + 2, 1) } : w)),
        })),
      ContainerFormatError,
      "宣言から決まる",
    );
    // rowAxis 1 の initializer を piece 分割させる（block 上限を 1 行分より小さく）。
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
      "piece 分割できない",
    );
    // per-channel codec の groupSize が行長と違う。
    await assertRejects(
      () =>
        rewrite((input) => ({
          ...input,
          weights: input.weights.map((w) => (w.initializer === "big.weight"
            ? { ...w, encoding: { ...w.encoding, groupSize: 16 } }
            : w)
          ),
        })),
      ContainerFormatError,
      "行長 32 に等しい",
    );
    // group codec の groupSize が 2 冪 ≥ 16 でない / 行長を割らない。
    await assertRejects(
      () =>
        rewrite((input) => ({
          ...input,
          weights: input.weights.map((w) => (w.initializer === "proj.weight"
            ? { ...w, encoding: { ...w.encoding, groupSize: 24 } }
            : w)
          ),
        })),
      ContainerFormatError,
      "2 冪",
    );
    // scale の長さが `rows × groups × 4` と合わない。
    await assertRejects(
      () =>
        rewrite((input) => ({
          ...input,
          weights: input.weights.map((w) => (w.initializer === "proj.weight"
            ? {
              ...w,
              encoding: { ...w.encoding, scale: { bytes: bytesOf(12 * F32, 1), dtype: "f32" } },
            }
            : w)
          ),
        })),
      ContainerFormatError,
      "scale",
    );
    // 意味論 i32 に f16 は組めない。
    await assertRejects(
      () =>
        writeModelContainer({
          ...base,
          graphs: { g: declaration({ "idx": { shape: [8], dtype: "i32" } }) },
          consts: [],
          assets: [],
          weights: [{
            graph: "g",
            initializer: "idx",
            bytes: bytesOf(16, 3),
            encoding: { codec: "f16" },
          }],
        }, OPTIONS).then((written) => openContainer({ kind: "parts", parts: written.parts })),
      ContainerFormatError,
      "組めない",
    );
  });
});
