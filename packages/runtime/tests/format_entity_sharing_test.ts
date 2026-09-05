// 「1 実体 1 initializer」の構築時門（format/container.ts の assertNoScaleKeyCollision）。
//
// scale キーの衝突・共有は既に落ちるのに、**実体キーの共有**（2 本の initializer が同じ
// `initializer.tensor` を指す形）だけが無検査だった。通すと実行層は initializer 名ごとに
// バッファを確保・転送するので同じバイト列が 2 度 GPU へ上がり、i8 / i4 では同じ量子化バイトが
// 2 つの別 scale で逆量子化される（どちらも診断が出ない）。
//
// NOTE: 置き場は `format_shard_test.ts` の scale 共有ケースの隣が本来だが、同ファイルは
// 並走する別レッグの担当なので独立ファイルにした（統合時に畳んでよい）。

import { assertEquals, assertThrows } from "@std/assert";
import { ContainerError, createShardValidator, extractIrGraph } from "../src/format/container.ts";
import { parseSafetensors, type SafetensorsFile } from "../src/format/safetensors.ts";
import { buildSafetensors, f32Bytes, type GraphJson, type TensorSpec } from "./helpers/format.ts";

const weightShard = (tensors: readonly TensorSpec[]): SafetensorsFile =>
  parseSafetensors(buildSafetensors(tensors));

const graphShard = (graph: GraphJson): SafetensorsFile =>
  parseSafetensors(buildSafetensors([], { karume_ir: JSON.stringify(graph) }));

/**
 * 2 本の initializer が `wa` / `wb` として同じ形の重みを持つ最小グラフ。実体キーと storage は
 * ケースごとに差し替える（正しい形を 1 箇所に置き、異常系は 1 点だけ壊す規律）。
 */
const twoInitializerGraph = (
  entity: (slot: "wa" | "wb") => string,
  storage: (slot: "wa" | "wb") => Record<string, unknown>,
): GraphJson => ({
  format: "karume-ir",
  version: 1,
  requires: { ops: ["matmul"] },
  symbols: ["T"],
  inputs: [{ name: "x", dtype: "f32", shape: ["T", 3] }],
  outputs: ["ha", "hb"],
  initializers: {
    wa: { tensor: entity("wa"), storage: storage("wa") },
    wb: { tensor: entity("wb"), storage: storage("wb") },
  },
  values: {
    wa: { dtype: "f32", shape: [3, 4] },
    wb: { dtype: "f32", shape: [3, 4] },
    ha: { dtype: "f32", shape: ["T", 4] },
    hb: { dtype: "f32", shape: ["T", 4] },
  },
  nodes: [
    { op: "matmul", ins: ["x", "wa"], outs: ["ha"], attrs: {} },
    { op: "matmul", ins: ["x", "wb"], outs: ["hb"], attrs: {} },
  ],
});

Deno.test("createShardValidator: 2 本の initializer が同じ実体テンソルを指すグラフを構築時に落とす", () => {
  const graph = extractIrGraph(graphShard(
    twoInitializerGraph(() => "m.w", () => ({ dtype: "f32" })),
  ));

  const error = assertThrows(
    () => createShardValidator(graph),
    ContainerError,
    "1 実体 1 initializer",
  );
  // 帰属が分かる診断 MUST — 名前が出ないと直す側はどちらを改名するか決められない
  assertEquals(error.message.includes("'m.w'"), true, error.message);
  assertEquals(error.message.includes("'wa'"), true, error.message);
  assertEquals(error.message.includes("'wb'"), true, error.message);
});

Deno.test("createShardValidator: scale が別々でも実体だけを共有する i8 の 2 本は落ちる", () => {
  // scale キーは別々なので既存の scale 共有門では捕まらない — この 1 本の存在理由。
  const graph = extractIrGraph(graphShard(
    twoInitializerGraph(
      () => "m.w",
      (slot) => ({ dtype: "i8", scale: `m.${slot}.scale` }),
    ),
  ));

  assertThrows(() => createShardValidator(graph), ContainerError, "1 実体 1 initializer");
});

// 実体共有の検査が常に鳴る退行の裏取り（実体も scale も別々なら従来どおり構築でき、intake も
// 2 件返す）。
Deno.test("createShardValidator: 実体キーが別々の 2 本は従来どおり構築・受理できる", () => {
  const graph = extractIrGraph(graphShard(
    twoInitializerGraph(
      (slot) => `m.${slot}`,
      (slot) => ({ dtype: "i8", scale: `m.${slot}.scale` }),
    ),
  ));
  const validator = createShardValidator(graph);

  const ready = validator.intake(weightShard([
    { name: "m.wa", dtype: "I8", shape: [3, 4], data: new Uint8Array(12) },
    { name: "m.wa.scale", dtype: "F32", shape: [3, 1], data: f32Bytes([1, 1, 1]) },
    { name: "m.wb", dtype: "I8", shape: [3, 4], data: new Uint8Array(12) },
    { name: "m.wb.scale", dtype: "F32", shape: [3, 1], data: f32Bytes([1, 1, 1]) },
  ]));
  assertEquals(ready.map((item) => item.name), ["wa", "wb"]);
  validator.finish();
});
