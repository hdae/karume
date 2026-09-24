/**
 * `queue.writeBuffer` は**呼んだ時点で**バイト列を写す（WebGPU 仕様の content timeline —
 * "Let dataContents be a copy of the bytes held by the buffer source data."）ことを実機で固定する。
 *
 * Session 構築は block を 1 本ずつ読んで上げ、上げた反復でバイト列を手放す（フェンスまで握らない
 * — `session-build.ts` のフェンスの NOTE・ADR 0070 決定 3 / ADR 0108 決定 9 の 2026-09-24 追記）。
 * その正しさの根拠はこの仕様だけで、実装が遅延読み（submit / フェンスの時点で元のバイト列を読む）
 * なら黙った誤値になる。ここでは構築のあいだ writeBuffer が戻った直後に渡されたバイト列を毒で
 * 埋め、フェンス後の GPU 常駐が元の値のままであること（出力が毒を入れない構築とビット一致）を見る。
 *
 * 合成モデル: linear（i8 + per-channel scale・2 piece = 2 part・bias f32）→ add（f16 重み — 重み
 * スロット外なので CPU で f32 へ展開される）。生バイトの重み・companion scale・piece のオフセット
 * 書き込み・展開後の f32・part を跨ぐフェンスを 1 本で通す。
 */

import { assert, assertEquals } from "@std/assert";
import { describe, it } from "@std/testing/bdd";
import { type MemoryContainerInput, openMemoryContainer } from "../src/format/container/memory.ts";
import { type IrDeclaration, parseIrDeclarationValue } from "../src/format/ir.ts";
import { acquireGpu } from "../src/gpu/device.ts";
import { createSessionFromContainer } from "../src/runtime/executor.ts";
import { quantizeF16 } from "./helpers/f16.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";
import { quantizeI8 } from "./helpers/i8.ts";
import { f32Bytes, fill } from "./helpers/model-fixture.ts";

const M = 4;
const K = 16;
const N = 8;

const declaration = (): IrDeclaration =>
  parseIrDeclarationValue({
    format: "karume-ir",
    version: 2,
    requires: { ops: ["add", "linear"] },
    symbols: [],
    inputs: [{ name: "x", dtype: "f32", shape: [M, K] }],
    outputs: ["y"],
    initializers: { "enc.w": {}, "enc.b": {}, "enc.h": {} },
    values: {
      "enc.w": { dtype: "f32", shape: [N, K] },
      "enc.b": { dtype: "f32", shape: [N] },
      "enc.h": { dtype: "f32", shape: [N] },
      t: { dtype: "f32", shape: [M, N] },
      y: { dtype: "f32", shape: [M, N] },
    },
    nodes: [
      { op: "linear", ins: ["x", "enc.w", "enc.b"], outs: ["t"], attrs: {} },
      { op: "add", ins: ["t", "enc.h"], outs: ["y"], attrs: {} },
    ],
  });

/**
 * 供給を毎回新しい器で組む。毒は供給のバイト列そのものを書き換えるので、基準と毒入りの 2 つの
 * 構築で器を共有しない。`bias` は毒が供給の器に届いたことの確認に使う。
 */
const supply = (): { readonly input: MemoryContainerInput; readonly bias: Uint8Array } => {
  const w = quantizeI8(fill([N, K], (i) => Math.sin(i * 0.37) * 0.5).data, [N, K], 0);
  const bias = f32Bytes(fill([N], (i) => (i % 3) * 0.05).data);
  const half = (N / 2) * K;
  return {
    bias,
    input: {
      graphs: { main: declaration() },
      tensors: {
        main: {
          "enc.w": {
            pieces: [
              { rows: [0, N / 2], read: () => Promise.resolve(w.bytes.subarray(0, half)) },
              { rows: [N / 2, N], read: () => Promise.resolve(w.bytes.subarray(half, N * K)) },
            ],
            encoding: { codec: "int8-sym", rowAxis: 0, groupSize: K, scale: f32Bytes(w.scale) },
          },
          "enc.b": { bytes: bias, encoding: { codec: "f32" } },
          "enc.h": {
            bytes: quantizeF16(fill([N], (i) => (i % 5) * 0.125 - 0.25).data).bytes,
            encoding: { codec: "f16" },
          },
        },
      },
    },
  };
};

/** f32 の NaN（0xffffffff）・i8 の −1・f16 の NaN になる毒。 */
const POISON = 0xff;

/**
 * `body` のあいだだけ `writeBuffer` を包み、元の呼び出しが戻った直後に渡されたバイト列を毒で
 * 埋める。包むのは prototype（`device.queue` が毎回同じオブジェクトを返すかに依らないため）。
 *
 * MUST: 構築の後（run の前）に必ず外す。run の入力・params の書き込みまで毒で埋めると、params
 * キャッシュの CPU 側の写しが壊れ、この仕様とは無関係の理由で割れる。
 */
const withPoisonedWrites = async <T>(
  queue: GPUQueue,
  body: () => Promise<T>,
): Promise<{ readonly result: T; readonly writes: number }> => {
  const prototype = Object.getPrototypeOf(queue) as GPUQueue;
  const original = prototype.writeBuffer;
  let writes = 0;
  prototype.writeBuffer = function (
    this: GPUQueue,
    ...args: Parameters<GPUQueue["writeBuffer"]>
  ): undefined {
    const returned = original.apply(this, args);
    const data = args[2];
    const bytes = ArrayBuffer.isView(data)
      ? new Uint8Array(data.buffer, data.byteOffset, data.byteLength)
      : new Uint8Array(data);
    bytes.fill(POISON);
    writes += 1;
    return returned;
  };
  try {
    return { result: await body(), writes };
  } finally {
    prototype.writeBuffer = original;
  }
};

describe("writeBuffer は呼んだ時点でバイト列を写す", { ignore: !GPU_AVAILABLE }, () => {
  it("writeBuffer が戻った直後に元のバイト列を毒で埋めても、フェンス後の常駐（出力）は元の値のまま", async () => {
    const gpu = await acquireGpu();
    try {
      const x = fill([M, K], (i) => ((i * 7) % 11) / 11 - 0.5);
      const reference = await createSessionFromContainer(
        gpu,
        openMemoryContainer(supply().input),
        "main",
      );
      let expected: Float32Array<ArrayBuffer>;
      try {
        expected = (await reference.run({ x }))["y"].data as Float32Array<ArrayBuffer>;
      } finally {
        await reference.dispose();
      }
      // 毒（NaN）が載れば出力は非有限になる — 基準が有限でなければ比較が空振りする。
      assert(expected.every(Number.isFinite), "基準の出力に非有限値がある");

      const poisoned = supply();
      const { result: session, writes } = await withPoisonedWrites(
        gpu.device.queue,
        () => createSessionFromContainer(gpu, openMemoryContainer(poisoned.input), "main"),
      );
      try {
        // 毒は writeBuffer に渡った器そのもの（ここでは供給の器）に入っている。
        assert(
          poisoned.bias.every((byte) => byte === POISON),
          "供給の器が毒で埋まっていない（writeBuffer に渡ったのが別の器 — 検出器が空振りする）",
        );
        assert(writes >= 4, `重みの block 4 本が writeBuffer を通っていない（${writes} 回）`);
        const actual = (await session.run({ x }))["y"].data as Float32Array<ArrayBuffer>;
        assertEquals(new Uint8Array(actual.buffer), new Uint8Array(expected.buffer));
      } finally {
        await session.dispose();
      }
    } finally {
      gpu.destroy();
    }
  });
});
