// golden io の格納規約（tests/helpers/golden-io.ts — ADR 0009 / tools/exporter README）。
//
// `ioTensor` の「格納 dtype が宣言と食い違ったら落とす」MUST は、全 e2e 門で「golden の格納形と
// IR の宣言 dtype がずれた」を捕まえる**唯一の**検出器（要素は全型 4 バイトなので、黙って
// 読み替えるとビット列の再解釈が通ってしまう）。ところがこの 1 行は正常な golden を通す経路
// でしか実行されず、assert を外しても全門が緑のまま通っていた。
//
// GPU も実資産も要らない（helpers/format.ts の `buildSafetensors` で組んだ最小の
// safetensors を `parseSafetensors` で読む）。

import { assertEquals, AssertionError, assertStrictEquals, assertThrows } from "@std/assert";
import { parseSafetensors, type TensorView } from "../src/format/safetensors.ts";
import { IO_ENCODING, ioTensor } from "./helpers/golden-io.ts";
import { buildSafetensors, f32Bytes } from "./helpers/format.ts";

/** 4 バイト整数のバイト列（I32 / U32 の実体）。 */
const i32Bytes = (values: readonly number[]): Uint8Array =>
  new Uint8Array(Int32Array.from(values).buffer);

/** 意味論 3 dtype をそれぞれ 1 本ずつ持つ golden io（正常な形はここ 1 箇所）。 */
const goldenIo = () =>
  parseSafetensors(buildSafetensors([
    { name: "output.0", dtype: "F32", shape: [2], data: f32Bytes([1.5, -2.5]) },
    { name: "output.1", dtype: "I32", shape: [2], data: i32Bytes([7, -9]) },
    { name: "output.2", dtype: "U32", shape: [2], data: i32Bytes([0, 1]) },
  ]));

const viewOf = (
  file: ReturnType<typeof goldenIo>,
  name: string,
): TensorView => {
  const view = file.tensors.get(name);
  if (view === undefined) throw new Error(`${name} が無い`);
  return view;
};

Deno.test("IO_ENCODING は意味論 3 dtype の格納形を全て宣言する（ADR 0009 の写し）", () => {
  assertEquals(IO_ENCODING, { f32: "F32", i32: "I32", bool: "U32" });
});

Deno.test("ioTensor は宣言 dtype どおりの型で読み、バイト列をコピーしない", () => {
  const file = goldenIo();
  const f32 = ioTensor(file, viewOf(file, "output.0"), "f32");
  assertEquals([f32.dtype, f32.shape], ["f32", [2]]);
  assertEquals([...f32.data], [1.5, -2.5]);

  const i32 = ioTensor(file, viewOf(file, "output.1"), "i32");
  assertEquals([i32.dtype, i32.shape], ["i32", [2]]);
  assertEquals([...i32.data], [7, -9]);

  const bool = ioTensor(file, viewOf(file, "output.2"), "bool");
  assertEquals([bool.dtype, bool.shape], ["bool", [2]]);
  assertEquals([...bool.data], [0, 1]);

  // view であること = 元バイト列を書き換えると読めた値も動く（コピーしていない検出器）
  new DataView(file.buffer).setFloat32(viewOf(file, "output.0").byteOffset, 42, true);
  assertStrictEquals(f32.data[0], 42);
});

Deno.test("ioTensor は格納 dtype が宣言と食い違う組を落とす（ビット列の再解釈を通さない）", () => {
  const file = goldenIo();
  // 要素は全型 4 バイトなので、この門が無ければ 3 本とも「数値として読めて」しまう
  const cases: readonly (readonly ["f32" | "i32" | "bool", string])[] = [
    ["f32", "output.1"], // I32 を f32 で
    ["i32", "output.0"], // F32 を i32 で
    ["bool", "output.0"], // F32 を bool で
  ];
  for (const [dtype, name] of cases) {
    const error = assertThrows(
      () => ioTensor(file, viewOf(file, name), dtype),
      AssertionError,
      "の dtype",
    );
    // 診断はテンソル名を名乗る MUST（どの出力がずれたか分からないと直せない）
    assertEquals(error.message.includes(`'${name}'`), true, error.message);
  }
});
