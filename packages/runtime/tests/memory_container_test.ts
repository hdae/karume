/**
 * メモリ内容器（`openMemoryContainer`）— 手元のバイト列を容器を書かずに供給する面。
 *
 * ここで固定するのは 2 つ:
 * - **この層が自分で持つ検査**（宣言との対応・`pieces` が 2 本以上）と、**合流層へ委ねた検査**
 *   （codec × 意味論 dtype・piece の被覆・scale の要否）が実際に効いていること。後者は
 *   「メモリ内容器は descriptor を通らない」ぶん、委譲が崩れると無検査で素通りする。
 * - **合成した目次の規則**（block id・part 割り・`readBlock` の返し方）。part 割りは Session 構築の
 *   フェンス境界そのものなので、崩れると RAM ピークが静かに変わる。
 */

import { assertEquals, assertRejects, assertStrictEquals, assertThrows } from "@std/assert";
import { describe, it } from "@std/testing/bdd";
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

/** 宣言 1 本 + 供給 1 組をそのまま開く（写像を挟まない素の入口）。 */
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

// --- 量子化 + piece 分割の fixture ------------------------------------------
// q は per-channel int8 の [4,16]（payload 64 B・scale は rank 2 [4,1] = 16 B）。行を 2 本ずつ
// 2 piece に割ると、中間 piece は詰め物不可の 32 B ちょうどになる。

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

/** q を行範囲で割った供給（`rows` の列をテストが差し替えて穴 / 末尾不一致を作る）。 */
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

describe("openMemoryContainer: 宣言との対応", () => {
  it("shared でない initializer の不足を全件列挙して落とす", () => {
    const error = assertThrows(
      () => openWith(baseDeclaration(), {}),
      ContainerFormatError,
      "不足 [w, b]",
    );
    assertEquals(error.message.includes("余剰 []"), true, error.message);
  });

  it("宣言に無い initializer の供給を余剰として落とす", () => {
    assertThrows(
      () => openWith(baseDeclaration(), { ...baseSupply(), z: biasTensor() }),
      ContainerFormatError,
      "余剰 [z]",
    );
  });

  it("未宣言のグラフ名への供給を落とす", () => {
    assertThrows(
      () => openWith(baseDeclaration(), baseSupply(), "other"),
      ContainerFormatError,
      "供給に未宣言のグラフ 'other' がある",
    );
  });

  it("pieces が 1 本の供給を落とす（1 本なら bytes で渡す）", () => {
    assertThrows(
      () =>
        openWith(quantizedDeclaration(), {
          q: quantizedPieces([[0, ROWS]]),
          b: biasTensor(),
        }),
      ContainerFormatError,
      "pieces は 2 本以上",
    );
  });
});

describe("openMemoryContainer: 供給と合成した目次の突合", () => {
  it("丸ごと供給の byteLength が宣言 payload と違えば落ちる（長すぎ）", () => {
    assertThrows(
      () =>
        openWith(baseDeclaration(), {
          ...baseSupply(),
          w: { bytes: f32Bytes(new Array(13).fill(0.5)), encoding: { codec: "f32" } },
        }),
      ContainerFormatError,
      "payload 48 バイト + 詰め物",
    );
  });

  it("丸ごと供給の byteLength が宣言 payload と違えば落ちる（短すぎ）", () => {
    assertThrows(
      () =>
        openWith(baseDeclaration(), {
          ...baseSupply(),
          w: { bytes: f32Bytes(new Array(11).fill(0.5)), encoding: { codec: "f32" } },
        }),
      ContainerFormatError,
      "payload 48 バイト + 詰め物",
    );
  });

  it("scale の byteLength が group 形の宣言と違えば落ちる", () => {
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
      "payload 16 バイト + 詰め物",
    );
  });

  it("scale の byteOffset が 4 バイト整列でなければ落ちる", () => {
    // 整列していない view は `session-build` が Float32Array を張る瞬間に素の RangeError に
    // なり、転送層の文言へ化ける（供給元が保証する側の規則）。
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
      "scale の byteOffset 1 が 4 バイト整列でない",
    );
  });

  it("合成した block id が衝突すれば落ちる（後勝ちの沈黙上書きにしない）", () => {
    // initializer 名に `#scale` を含む形は、別 initializer の companion scale と同じ id を取る。
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
      "が二重に束縛されている",
    );
  });

  it("piece の read() が宣言長と違うバイト列を返せば readBlock が落ちる", async () => {
    const payload = qPayload();
    const container = openWith(quantizedDeclaration(), {
      q: {
        encoding: { codec: "int8-sym", rowAxis: 0, groupSize: ROW_LENGTH, scale: qScale() },
        pieces: [
          // piece 1 は行範囲 [0,2)（= 32 B）を名乗りながら全量 64 B を返す。合流層の突合は
          // 導出値どうしなので恒真で、`containerBatches` の切り詰めも門に届かない。
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
      "取得長 64 が宣言 32 と違う",
    );
  });
});

describe("openMemoryContainer: 合流層へ委ねた検査", () => {
  it("piece 列の行範囲に穴があれば落ちる", () => {
    assertThrows(
      () =>
        openWith(quantizedDeclaration(), {
          q: quantizedPieces([[0, 1], [2, ROWS]]),
          b: biasTensor(),
        }),
      ContainerFormatError,
      "行 1 から続かない",
    );
  });

  it("piece 列の末尾が宣言 shape の先頭次元に届かなければ落ちる", () => {
    assertThrows(
      () =>
        openWith(quantizedDeclaration(), {
          q: quantizedPieces([[0, 2], [2, 3]]),
          b: biasTensor(),
        }),
      ContainerFormatError,
      "宣言 shape の先頭次元 4 行に届かない",
    );
  });

  it("piece 列の末尾が宣言 shape の先頭次元を超えれば落ちる", () => {
    // 末尾の一致を `<` へ緩めると、宣言より長い供給が黙って通る。
    assertThrows(
      () =>
        openWith(quantizedDeclaration(), {
          q: quantizedPieces([[0, 2], [2, ROWS + 1]]),
          b: biasTensor(),
        }),
      ContainerFormatError,
      "末尾 5 行が宣言 shape の先頭次元 4 行に届かない / 超える",
    );
  });

  it("piece の行範囲が空区間なら落ちる", () => {
    // [2,2) は連続性（開始 = 直前の終端）と末尾の一致をどちらも満たすので、空区間の門だけが掴む。
    assertThrows(
      () =>
        openWith(quantizedDeclaration(), {
          q: quantizedPieces([[0, 2], [2, 2], [2, ROWS]]),
          b: biasTensor(),
        }),
      ContainerFormatError,
      "piece[1]: 行範囲 [2, 2) が空区間",
    );
  });

  it("piece の行範囲の開始が負なら非負の門で落ちる", () => {
    assertThrows(
      () =>
        openWith(quantizedDeclaration(), {
          q: quantizedPieces([[-1, 2], [2, ROWS]]),
          b: biasTensor(),
        }),
      ContainerFormatError,
      "piece[0]: 行範囲の 開始 -1 が非負の安全整数でない",
    );
  });

  it("piece の行範囲が整数でなければ落ちる", () => {
    // 小数の行境界は被覆検査（空区間・連続性・末尾）をどれも素通りしたうえで、GPU 構築の
    // byteOffset を 4 の倍数でない位置にする。
    assertThrows(
      () =>
        openWith(quantizedDeclaration(), {
          q: quantizedPieces([[0, 1.5], [1.5, ROWS]]),
          b: biasTensor(),
        }),
      ContainerFormatError,
      "非負の安全整数でない",
    );
  });

  it("中間 piece のバイト長が 4 の倍数でなければ落ちる", () => {
    // f16 の [4,3]（行 6 バイト）を 1 行目で割ると、中間 piece は詰め物を掛けられないまま
    // 6 バイトになる。ここで落とさないと GPU の writeBuffer validation まで失敗が遅れる。
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
      "中間 piece の行範囲のバイト数 6 が 4 の倍数でない",
    );
  });

  it("意味論 dtype と codec の組が交差していれば落ちる", () => {
    const graph = baseDeclaration();
    graph.values["w"] = { dtype: "i32", shape: [4, 3] };
    assertThrows(
      () => openWith(graph, baseSupply()),
      ContainerFormatError,
      "意味論 dtype 'i32' に codec 'f32' は組めない",
    );
  });

  it("量子化 codec に scale / groupSize が無ければ落ちる", () => {
    assertThrows(
      () =>
        openWith(quantizedDeclaration(), {
          q: { bytes: qPayload(), encoding: { codec: "int8-sym" } },
          b: biasTensor(),
        }),
      ContainerFormatError,
      "codec 'int8-sym' は scale と groupSize が要る",
    );
  });

  it("非量子化 codec に scale / groupSize / rowAxis が付いていれば落ちる", () => {
    assertThrows(
      () =>
        openWith(baseDeclaration(), {
          ...baseSupply(),
          b: { bytes: f32Bytes([1, 2, 3]), encoding: { codec: "f32", scale: qScale() } },
        }),
      ContainerFormatError,
      "codec 'f32' は scale / groupSize / rowAxis を持てない",
    );
  });
});

// --- codec 別の rowAxis ------------------------------------------------------
// 展開（`decodeI4` / `decodeI2` と WGSL）は宣言の軸を受け取らず軸 0 固定なので、両軸の長さが
// 等しい `[N,N]` では scale 形の突合が通ってしまう。宣言の軸を見るのは合流層だけ。

/** `q`（f32 意味論）を 1 本の op で消費するグラフ。rank 2 は linear・rank 3 は conv_transpose1d。 */
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
    nodes: [{
      op: "conv_transpose1d",
      ins: ["x", "q", "b"],
      outs: ["y"],
      attrs: { stride: 1, padding: 0 },
    }],
  };
};

/** `rowAxis: 1` の量子化 `q` と f32 の bias（中身は見ないので 0 埋め）。 */
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

describe("openMemoryContainer: codec 別の rowAxis（合流層）", () => {
  it("int4-sym-g の rowAxis 1 は、両軸の長さが等しい形でも落ちる", () => {
    // [32,32]・group 16: 軸 0 と読んでも軸 1 と読んでも scale は [32,2] で、形の突合は区別しない。
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
      "codec 'int4-sym-g' の rowAxis は 0 だけ（宣言は 1",
    );
  });

  it("int2-off の rowAxis 1 は、展開の時点でなく合流層で落ちる", () => {
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
      "codec 'int2-off' の rowAxis は 0 だけ（宣言は 1",
    );
  });

  it("int8-sym の rowAxis 1（conv_transpose1d 形）は受理する", () => {
    // [Cin=4, Cout=8, K=3] の行は軸 1（8 行・行長 12）。
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

describe("openMemoryContainer: 合成した目次", () => {
  it("丸ごとの実体は渡された器をそのまま返す（複製しない）", async () => {
    const tensors = baseTensors();
    const container = memoryModel(baseDeclaration(), tensors);
    assertStrictEquals(await container.readBlock(`${GRAPH_NAME}/w`), tensors[0].bytes);
    assertStrictEquals(await container.readBlock(`${GRAPH_NAME}/b`), tensors[1].bytes);
  });

  it("piece は readBlock のたびに read() を引き直す（保持しない）", async () => {
    const reads: number[] = [];
    const container = openWith(quantizedDeclaration(), {
      q: quantizedPieces([[0, 2], [2, ROWS]], (index) => reads.push(index)),
      b: biasTensor(),
    });
    assertEquals(reads, []);
    await container.readBlock(`${GRAPH_NAME}/q#piece1`);
    await container.readBlock(`${GRAPH_NAME}/q#piece1`);
    await container.readBlock(`${GRAPH_NAME}/q#piece2`);
    assertEquals(reads, [0, 0, 1]);
  });

  it("未知の block id は fail loudly", async () => {
    const container = memoryModel();
    await assertRejects(
      () => container.readBlock(`${GRAPH_NAME}/missing`),
      ContainerFormatError,
      "未宣言の block",
    );
  });

  it("丸ごとの実体と scale は part 2・piece は 2 の次から 1 本 1 part", () => {
    const container = openWith(quantizedDeclaration(), {
      q: quantizedPieces([[0, 2], [2, ROWS]]),
      b: biasTensor(),
    });
    const supplies = container.graphs[GRAPH_NAME].supplies;
    const q = supplies.get("q");
    assertEquals(q?.blocks.map((block) => [block.id, block.part, block.length]), [
      [`${GRAPH_NAME}/q#piece1`, 3, 2 * ROW_LENGTH],
      [`${GRAPH_NAME}/q#piece2`, 4, 2 * ROW_LENGTH],
    ]);
    // 規則③: companion scale は piece 1 と同じ part。
    assertEquals([q?.scale?.id, q?.scale?.part, q?.scale?.length], [
      `${GRAPH_NAME}/q#scale`,
      3,
      ROWS * 4,
    ]);
    const b = supplies.get("b");
    assertEquals(b?.blocks.map((block) => [block.id, block.part, block.length]), [
      [`${GRAPH_NAME}/b`, 2, ROWS * 4],
    ]);
  });

  it("丸ごとだけの供給は全部 part 2 に載る", () => {
    const supplies = memoryModel().graphs[GRAPH_NAME].supplies;
    assertEquals([...supplies].map(([name, supply]) => [name, supply.blocks[0].part]), [
      ["w", 2],
      ["b", 2],
    ]);
  });

  it("丸ごと供給は累積が part 長を超えたところで次の part へ移る", () => {
    // part 割り = 構築時の staging VRAM の上限（`session-build` は part ごとに submit + 完了
    // 待ちを出す）。全部を 1 part に載せると、丸ごと供給のモデルだけその対策が無効になる。
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
        ...Object.fromEntries(
          names.map((name) => [name, { dtype: "f32", shape: [CHUNK / 4] }]),
        ),
        y: { dtype: "f32", shape: [1] },
      },
      nodes: [{ op: "add", ins: ["x", "x"], outs: ["y"], attrs: {} }],
    };
    const bigTensor = (): MemoryTensor => ({
      bytes: new Uint8Array(new ArrayBuffer(CHUNK)),
      encoding: { codec: "f32" },
    });
    const supplies = openWith(
      graph,
      Object.fromEntries(names.map((name) => [name, bigTensor()])),
    ).graphs[GRAPH_NAME].supplies;
    // 96 MiB × 2 は 256 MiB に収まるので同じ part・3 本目で溢れて次の part へ。
    assertEquals([...supplies].map(([name, supply]) => [name, supply.blocks[0].part]), [
      ["a", 2],
      ["b", 2],
      ["c", 3],
    ]);
  });
});

describe("BoundContainer: krm 経路とメモリ経路の A/B", () => {
  it("OpenedContainer は BoundContainer として prepareContainer に渡せる", async () => {
    const opened = await openModelBytes();
    const prepared = prepareContainer(opened, GRAPH_NAME);
    assertEquals(prepared.graph.initializers["w"], { storage: { codec: "f32" } });
  });

  it("同じ入力なら合流後のグラフと見積りが一致する", async () => {
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
