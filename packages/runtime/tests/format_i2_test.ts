import { assertEquals, assertThrows } from "@std/assert";
import { ContainerFormatError } from "../src/format/container/header.ts";
import { decodeI2, I2Error } from "../src/format/i2.ts";
import { prepareContainer } from "../src/runtime/executor.ts";
import { planWeightBuffers, planWeightResidency } from "../src/runtime/weight-residency.ts";
import type { TensorInput } from "./helpers/container-write.ts";
import {
  type DeclarationJson,
  GRAPH_NAME,
  memoryModel,
  openModelBytes,
} from "./helpers/model-fixture.ts";

const graph = (): DeclarationJson => ({
  format: "karume-ir",
  version: 2,
  requires: { ops: ["embedding"] },
  symbols: [],
  inputs: [{ name: "ids", dtype: "i32", shape: [2] }],
  outputs: ["y"],
  initializers: { w: {} },
  values: { w: { dtype: "f32", shape: [2, 16] }, y: { dtype: "f32", shape: [2, 16] } },
  nodes: [{ op: "embedding", ins: ["w", "ids"], outs: ["y"], attrs: { padding_idx: -1 } }],
});
const bytes = Uint8Array.of(0xe4, 0x1b, 0x72, 0x8d, 0x1b, 0xe4, 0x8d, 0x72);
const scales = Float32Array.of(0.375, 1.25);

/**
 * `int2-off` の供給 1 本。per-channel なので `groupSize` は行長（16）で、scale は行ごとに
 * 1 本（`[rows, 1]` ぶんの f32 — v1 の keepdim 形と同じバイト列）。
 */
const i2Weight = (
  scaleBytes: Uint8Array<ArrayBuffer> = new Uint8Array(scales.buffer),
  groupSize = 16,
): TensorInput => ({
  graph: GRAPH_NAME,
  initializer: "w",
  bytes,
  encoding: { codec: "int2-off", groupSize, scale: { bytes: scaleBytes, dtype: "f32" } },
});

Deno.test("INT2 の全符号値・byte内位置と行別scaleを解釈する", async () => {
  const expected = [
    -2,
    -1,
    0,
    1,
    1,
    0,
    -1,
    -2,
    0,
    -2,
    1,
    -1,
    -1,
    1,
    -2,
    0,
    1,
    0,
    -1,
    -2,
    -2,
    -1,
    0,
    1,
    -1,
    1,
    -2,
    0,
    0,
    -2,
    1,
    -1,
  ].map((q, i) => Math.fround(q * scales[Math.floor(i / 16)]));
  assertEquals(decodeI2(bytes, [2, 16], scales, [2, 1]), new Float32Array(expected));
  // 容器（`krm`）を 1 往復させて、読み手の側でも i2 の席が立つことを見る。
  const loaded = prepareContainer(await openModelBytes(graph(), [i2Weight()]), GRAPH_NAME);
  const buffers = planWeightBuffers(planWeightResidency(loaded.graph));
  assertEquals(buffers.map((b) => b.declaredBytes), [8, 8]);
  assertEquals(buffers.map((b) => b.seat), ["i2", "i2"]);
});

Deno.test("INT2 は不正な形・scale・端数バイトを拒否する", () => {
  // 宣言 shape の門は**合流層**（宣言 shape × codec）— IR v2 の宣言は格納を持たないので、
  // 「i2 なのに rank 1 / 行長が 16 の倍数でない」は供給と突き合わせて初めて決まる。
  for (const shape of [[32], [2, 15], [2, 20], [0, 16], [2, 0], [1, 2, 16]]) {
    const g = graph();
    g.values.w.shape = shape;
    assertThrows(
      () => memoryModel(g, [i2Weight()]),
      ContainerFormatError,
      "正の rank 2・行長 16 の倍数",
    );
    assertThrows(() => decodeI2(bytes, shape, scales, [2, 1]), I2Error);
  }
  // per-channel codec の `groupSize` は行長に等しい MUST（v1 の `group_size` 宣言の行き先）。
  assertThrows(
    () => memoryModel(graph(), [i2Weight(new Uint8Array(scales.buffer), 8)]),
    ContainerFormatError,
    "per-channel",
  );
  // scale を持たない `int2-off` は供給の時点で落ちる（v1 の `storage.scale` 欠落の行き先）。
  assertThrows(
    () =>
      memoryModel(graph(), [{
        graph: GRAPH_NAME,
        initializer: "w",
        bytes,
        encoding: { codec: "int2-off" },
      }]),
    ContainerFormatError,
    "scale と groupSize が要る",
  );
  // scale のバイト数は `rows × groups × 4` — 行数と合わない列は落ちる（v1 の scale shape 検査）。
  // 文言はバイト数の突合そのものを名指しする（`"scale"` だけだと「scale の有無」へ門が退行
  // しても緑のままになる — 上の 2 ケースの文言にも `scale` が入っている）。
  assertThrows(
    () => memoryModel(graph(), [i2Weight(new Uint8Array(Float32Array.of(0.375).buffer))]),
    ContainerFormatError,
    "の長さ 4 が payload 8 バイト",
  );
  assertThrows(() => decodeI2(bytes.subarray(1), [2, 16], scales, [2, 1]), I2Error, "ペイロード");
  assertThrows(() => decodeI2(bytes, [2, 16], scales, [1, 2]), I2Error, "scale");
});

Deno.test("INT2 と同じ重みを graph 出力へ出すと f32 展開分を見積もる", () => {
  const g = graph();
  g.outputs.push("w");
  const plan = planWeightResidency(
    prepareContainer(memoryModel(g, [i2Weight()]), GRAPH_NAME).graph,
  );
  assertEquals(plan.get("w"), { seat: "expanded", payloadBytes: 8, expandedBytes: 128 });
});
