import { assertEquals, assertThrows } from "@std/assert";
import { ContainerError, openModel } from "../src/format/container.ts";
import { decodeI2, I2Error } from "../src/format/i2.ts";
import { IrError, parseIrGraph } from "../src/format/ir.ts";
import { parseSafetensors, SafetensorsError } from "../src/format/safetensors.ts";
import { planWeightBuffers, planWeightResidency } from "../src/runtime/weight-residency.ts";
import { buildSafetensors, type GraphJson } from "./helpers/format.ts";

const graph = (): GraphJson => ({
  format: "karume-ir",
  version: 1,
  requires: { ops: ["embedding"] },
  symbols: [],
  inputs: [{ name: "ids", dtype: "i32", shape: [2] }],
  outputs: ["y"],
  initializers: { w: { tensor: "w", storage: { dtype: "i2", scale: "scale" } } },
  values: { w: { dtype: "f32", shape: [2, 16] }, y: { dtype: "f32", shape: [2, 16] } },
  nodes: [{ op: "embedding", ins: ["w", "ids"], outs: ["y"], attrs: { padding_idx: -1 } }],
});
const bytes = Uint8Array.of(0xe4, 0x1b, 0x72, 0x8d, 0x1b, 0xe4, 0x8d, 0x72);
const scales = Float32Array.of(0.375, 1.25);
const model = (g = graph(), scaleShape = [2, 1]): ArrayBuffer =>
  buildSafetensors([
    { name: "scale", dtype: "F32", shape: scaleShape, data: new Uint8Array(scales.buffer) },
    { name: "w", dtype: "I2", shape: [2, 16], data: bytes },
  ], { karume_ir: JSON.stringify(g) });

Deno.test("INT2 の全符号値・byte内位置と行別scaleを解釈する", () => {
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
  const loaded = openModel(model());
  const buffers = planWeightBuffers(planWeightResidency(loaded.graph));
  assertEquals(buffers.map((b) => b.declaredBytes), [8, 8]);
  assertEquals(buffers.map((b) => b.seat), ["i2", "i2"]);
});

Deno.test("INT2 は不正な形・scale・端数バイトを拒否する", () => {
  for (const shape of [[32], [2, 15], [2, 20], [0, 16], [2, 0], [1, 2, 16]]) {
    const g = graph();
    g.values.w.shape = shape;
    assertThrows(() => parseIrGraph(JSON.stringify(g)), IrError);
    assertThrows(() => decodeI2(bytes, shape, scales, [2, 1]), I2Error);
  }
  const grouped = graph();
  grouped.initializers.w.storage.group_size = 16;
  assertThrows(() => parseIrGraph(JSON.stringify(grouped)), IrError, "group_size");
  const missing = graph();
  delete missing.initializers.w.storage.scale;
  assertThrows(() => parseIrGraph(JSON.stringify(missing)), IrError, "scale");
  assertThrows(() => decodeI2(bytes.subarray(1), [2, 16], scales, [2, 1]), I2Error, "ペイロード");
  assertThrows(() => decodeI2(bytes, [2, 16], scales, [1, 2]), I2Error, "scale");
  assertThrows(() => openModel(model(graph(), [1, 2])), ContainerError);
  assertThrows(() =>
    parseSafetensors(buildSafetensors([
      { name: "w", dtype: "I2", shape: [3], data: Uint8Array.of(0) },
    ])), SafetensorsError);
  assertThrows(
    () =>
      parseSafetensors(buildSafetensors([
        { name: "odd", dtype: "I8", shape: [1], data: Uint8Array.of(0) },
        { name: "w", dtype: "I2", shape: [16], data: Uint8Array.of(0, 0, 0, 0) },
      ])),
    SafetensorsError,
    "整列",
  );
});

Deno.test("INT2 と同じ重みを graph 出力へ出すと f32 展開分を見積もる", () => {
  const g = graph();
  g.outputs.push("w");
  const plan = planWeightResidency(openModel(model(g)).graph);
  assertEquals(plan.get("w"), { seat: "expanded", payloadBytes: 8, expandedBytes: 128 });
});
