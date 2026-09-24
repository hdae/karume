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
 */

import { assert, assertEquals, assertRejects } from "@std/assert";
import { ContainerFormatError } from "../src/format/container/header.ts";
import { type MemoryContainerInput, openMemoryContainer } from "../src/format/container/memory.ts";
import { openContainer } from "../src/format/container/open.ts";
import { type IrDeclaration, parseIrDeclaration } from "../src/format/ir.ts";
import { acquireGpu } from "../src/gpu/device.ts";
import { createSessionFromContainer, prepareContainer } from "../src/runtime/executor.ts";
import { f32Bytes, fill } from "./helpers/model-fixture.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";
import { quantizeF16 } from "./helpers/f16.ts";
import { quantizeI4 } from "./helpers/i4.ts";
import { quantizeI8 } from "./helpers/i8.ts";
import { writeGraphContainer, writeModelContainer } from "./helpers/container-write.ts";
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

const OPTIONS = { partBytes: 1024, blockBytes: 128 } as const;

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
