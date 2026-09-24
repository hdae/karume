/**
 * 供給元 A/B — `krm`（`openContainer`）とメモリ内容器（`openMemoryContainer`）が**同じ重みバイト列**
 * から**同じ出力バイト列**を出すこと。片方だけが CPU 展開 / GPU 常駐へ倒れると、診断の常駐
 * バイト数で差が出る。
 *
 * 合成モデル: linear（i4 + group scale）→ add（f16 重み）→ linear（i8 + per-channel scale・bias f32）。
 * 3 codec の供給計画・scale の正規化（rank 2）・piece 分割を 1 本で通す。
 *
 * MUST: 2 経路の piece の割り方を**わざと違える**（`krm` は block 上限 128 B から書き手が決め、
 * メモリ側は行範囲を明示して 4 行ずつ）。同じ割り方で比べると「分割は GPU 側の配置を 1 バイトも
 * 変えない」という不変条件が検出器の外に出てしまう。
 *
 * もう 1 組: 同じ `krm` を、block を写して返す取得元と、hub の scan 経路と同じく part の器の view を
 * 返す取得元（器の block の外は毒）から読み、出力と常駐バイト数が一致すること。読み手のどこかが
 * byteOffset を落とすと、器の先頭や毒を読んで割れる。
 *
 * 外部の正解: 上の 2 組は合流層から Session 構築までを両辺で共有するので、そこの誤りは両辺に
 * 同じだけ乗って一致してしまう。そこで格納バイト列を CPU の codec 展開（`decodeI4` / `decodeF16` /
 * `decodeI8`）で f32 へ戻し、CPU 参照（`applyReferenceOp`）で linear → add → linear を辿った出力とも
 * 許容差で突き合わせる。展開を 1 つ取り違えた参照（i4 の scale を 1 チャネルずらす）が同じ出力で
 * 落ちることも同じケースで見る — 許容差が codec 展開の誤りを通すほど緩くないことの確認。
 */

import { assert, assertEquals, assertRejects } from "@std/assert";
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
import { describe, it } from "@std/testing/bdd";

const M = 4;
const K = 32;
const H = 24;
const N = 8;

/** 決定的な重み（乱数は使わない）。 */
const weights = () => ({
  w1: fill([H, K], (i) => Math.sin(i * 0.37) * 0.5),
  b1: fill([H], (i) => (i % 3) * 0.05),
  h: fill([H], (i) => (i % 5) * 0.125 - 0.25),
  w2: fill([N, H], (i) => Math.cos(i * 0.91) * 0.75),
  b2: fill([N], (i) => i * 0.01),
});

const declaration = (): IrDeclaration =>
  parseIrDeclaration(JSON.stringify({
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
  }));

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

/** `krm` の入力。block 上限を 128 B に下げて書き手に piece 分割させる。 */
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

/** i8 の重みを行範囲で 2 本に割った piece の読み口（`krm` 側の 5/3 分割とは別の 4/4 分割）。 */
const w2Pieces = (q: Quantized, counter: { reads: number }) => {
  const rowBytes = H;
  const ranges: readonly (readonly [number, number])[] = [[0, N / 2], [N / 2, N]];
  return ranges.map((rows) => ({
    rows,
    read: (): Promise<Uint8Array<ArrayBuffer>> => {
      counter.reads += 1;
      return Promise.resolve(q.w2.bytes.subarray(rows[0] * rowBytes, rows[1] * rowBytes));
    },
  }));
};

/** メモリ内容器の入力（同じバイト列を、容器を書かずにそのまま渡す）。 */
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
 * 縮図の CPU 参照: 容器へ渡したのと同じ格納バイト列を本番の CPU 展開で f32 に戻し、参照 op で
 * グラフの 3 ノードを辿る。合流層・Session 構築と実装を共有せず、常駐席の i4 / i8 は GPU 側の
 * 展開（カーネル内の unpack + scale）とも独立なので、2 経路が同じだけ誤る箇所もここで割れる
 * （f16 は展開席でホストの `decodeF16` を通るため、そこだけは参照と同じ実装）。
 * `i4Scale` は fault injection 用（既定は格納した scale そのもの）。
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

/** i4 の group scale を出力チャネル 1 本ぶんずらす（行 r が行 r+1 の scale を読む）。 */
const shiftI4ScaleOneChannel = (q: Quantized): Float32Array<ArrayBuffer> => {
  const groups = q.w1.scaleShape[1];
  const scale = q.w1.scale;
  return Float32Array.from(scale, (_, i) => scale[(i + groups) % scale.length]);
};

const OPTIONS = { partBytes: 1024, blockBytes: 128 } as const;

const POISON = 0xff;

/**
 * hub の scan 経路（`packages/hub/src/container.ts`）と同じ形の取得元: part ごとに tight な器を
 * 1 本持ち、`read` は器の `subarray` を返す（写さない）。器のうち block の外（整列の隙間・末尾）は
 * 毒で埋める — byteOffset を落とした読みが「隣の有効なバイト列」を読んで偶然通るのを防ぐ。
 * `verified: true` は HF 取得元と同じ（block の digest を掛けない経路）。
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
  // part 0（ヘッダ + 2 文書）は block を持たないのでそのまま写す。
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

/** 旧 hub と同じく block を写して返す（tight）取得元 — 比較の基準。 */
const copyingSource = (written: WrittenContainer): BlockSource => ({
  partCount: written.parts.length,
  verified: true,
  partLength: (index) => written.parts[index].byteLength,
  read: (part, offset, length) =>
    Promise.resolve(written.parts[part].slice(offset, offset + length)),
});

describe("memory container session", { ignore: !GPU_AVAILABLE }, () => {
  it("メモリ内容器から作った Session は krm 経路と出力バイト列・常駐バイト数が一致する", async () => {
    const q = quantize();
    const written = await writeModelContainer(containerInput(q), OPTIONS);
    const gpu = await acquireGpu();
    try {
      const x = fill([M, K], (i) => ((i * 7) % 11) / 11 - 0.5);
      const opened = await openContainer({ kind: "parts", parts: written.parts });
      // 書き手は block 上限 128 B で i8（192 B）も i4（384 B）も piece に割っている。
      assertEquals(opened.graphs.main.supplies.get("enc.w2")?.blocks.length, 2);
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
      // 開いた時点では piece の読み口を 1 度も引いていない（lazy）。
      assertEquals(counter.reads, 0);
      assertEquals(memory.graphs.main.supplies.get("enc.w2")?.blocks.length, 2);
      const session = await createSessionFromContainer(gpu, memory, "main");
      try {
        const actual = (await session.run({ x }))["y"].data as Float32Array<ArrayBuffer>;
        assertEquals(new Uint8Array(actual.buffer), new Uint8Array(expected.buffer));
        assertEquals(session.diagnostics().storage.residentCompressedBytes, expectedResident);
        assert(
          expectedResident > 0,
          "圧縮のまま常駐した重みが無い（i4 / f16 / i8 の席が効いていない）",
        );
        // piece の読み口は Session 構築でちょうど 1 本 1 回だけ引かれる（保持しない）。
        assertEquals(counter.reads, 2);
      } finally {
        await session.dispose();
      }
      // 見積りも同じ（席の分類が供給元で割れていない）。
      assertEquals(
        prepareContainer(memory, "main").estimate(),
        prepareContainer(opened, "main").estimate(),
      );
    } finally {
      gpu.destroy();
    }
  });

  it("krm とメモリ内容器の出力は、格納バイト列を CPU で展開して参照 op で辿った出力と許容差内で一致する", async () => {
    const q = quantize();
    const written = await writeModelContainer(containerInput(q), OPTIONS);
    const x = fill([M, K], (i) => ((i * 7) % 11) / 11 - 0.5);
    const expected = referenceOutput(q, x);
    const miswired = referenceOutput(q, x, shiftI4ScaleOneChannel(q));
    const sources: readonly (readonly [string, BoundContainer])[] = [
      ["krm", await openContainer({ kind: "parts", parts: written.parts })],
      ["メモリ内容器", openMemoryContainer(memoryInput(q, { reads: 0 }))],
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
        // 同じ出力が、展開を 1 つ取り違えた参照では落ちる（許容差が codec の誤りを通さない）。
        const miswiredReport = compareTensors(output, miswired, GEMM_TOLERANCE);
        assertEquals(
          miswiredReport.pass,
          false,
          `${label}: i4 の scale を 1 チャネルずらした参照でも通った（${
            formatAllclose(miswiredReport)
          }）`,
        );
      }
    } finally {
      gpu.destroy();
    }
  });

  it("part の器の view を返す取得元から組んだ Session は、写しを返す取得元と出力・常駐バイト数が一致する", async () => {
    const q = quantize();
    const written = await writeModelContainer(containerInput(q), OPTIONS);
    const viewed = scanShapedSource(written);
    // 毒が実際に器へ入り、block が器の中程（64 B 整列・0 以外）から読まれていること。
    assert(viewed.poisoned > 0, "器に block の外の隙間が無く、毒を置けていない");
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
          "圧縮のまま常駐した重みが無い（i4 / f16 / i8 の席が効いていない）",
        );
      } finally {
        await session.dispose();
      }
      assert(
        viewed.offsets.some((offset) => offset !== 0),
        "重みの block が全部 part の先頭から読まれている（byteOffset ≠ 0 の view を通っていない）",
      );
      assert(viewed.offsets.every((offset) => offset % 64 === 0), "block 開始が 64 B 整列でない");
    } finally {
      gpu.destroy();
    }
  });

  it("krg（重みの供給が無い）からは Session を組めない", async () => {
    const q = quantize();
    const graphOnly = await writeGraphContainer(containerInput(q), OPTIONS);
    const opened = await openContainer({ kind: "bytes", bytes: graphOnly.bytes });
    const gpu = await acquireGpu();
    try {
      await assertRejects(
        () => createSessionFromContainer(gpu, opened, "main"),
        ContainerFormatError,
        "重みの供給が無い",
      );
    } finally {
      gpu.destroy();
    }
  });
});
