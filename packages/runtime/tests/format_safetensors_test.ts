import { assertEquals, assertStrictEquals, assertThrows } from "@std/assert";
import { parseSafetensors, SafetensorsError, tensorBytes } from "../src/format/safetensors.ts";
import {
  buildSafetensors,
  f32Bytes,
  packSafetensors,
  packSafetensorsRaw,
} from "./helpers/format.ts";

const F32_2 = f32Bytes([1, 2]);

Deno.test("parseSafetensors: 正常系はテンソル表と __metadata__ を取り出す", () => {
  const buffer = buildSafetensors(
    [
      { name: "a", dtype: "F32", shape: [2, 1], data: f32Bytes([1, 2]) },
      { name: "b", dtype: "I8", shape: [3], data: new Uint8Array([7, 8, 9]) },
    ],
    { karume_ir: "{}", extra: "x" },
  );
  const file = parseSafetensors(buffer);

  assertEquals([...file.tensors.keys()].sort(), ["a", "b"]);
  assertEquals(file.metadata.get("karume_ir"), "{}");
  assertEquals(file.metadata.get("extra"), "x");

  const a = file.tensors.get("a");
  assertEquals(a?.dtype, "F32");
  assertEquals(a?.shape, [2, 1]);
  assertEquals(a?.byteLength, 8);
});

Deno.test("parseSafetensors: view はコピーせず元の ArrayBuffer を参照する", () => {
  const buffer = buildSafetensors([{ name: "a", dtype: "F32", shape: [2], data: F32_2 }]);
  const file = parseSafetensors(buffer);
  const view = file.tensors.get("a");
  if (view === undefined) throw new Error("tensor a が無い");

  const bytes = tensorBytes(file, view);
  assertStrictEquals(bytes.buffer, buffer);
  assertEquals(bytes.byteOffset, view.byteOffset);
  assertEquals(new Float32Array(buffer, view.byteOffset, 2), new Float32Array([1, 2]));
});

Deno.test("parseSafetensors: ヘッダ長すら無いファイルを拒否する", () => {
  assertThrows(() => parseSafetensors(new ArrayBuffer(4)), SafetensorsError, "短すぎる");
});

Deno.test("parseSafetensors: ヘッダ長がファイル長を超えるものを拒否する", () => {
  const buffer = new ArrayBuffer(16);
  new DataView(buffer).setBigUint64(0, 4096n, true);
  assertThrows(() => parseSafetensors(buffer), SafetensorsError, "ファイル長");
});

Deno.test("parseSafetensors: 壊れたヘッダ JSON を拒否する", () => {
  assertThrows(
    () => parseSafetensors(packSafetensorsRaw("{not json", new Uint8Array(0))),
    SafetensorsError,
    "ヘッダ JSON",
  );
});

Deno.test("parseSafetensors: ヘッダがオブジェクトでないものを拒否する", () => {
  assertThrows(
    () => parseSafetensors(packSafetensorsRaw("[]", new Uint8Array(0))),
    SafetensorsError,
    "オブジェクトでない",
  );
});

Deno.test("parseSafetensors: 未対応 dtype を拒否する", () => {
  const buffer = packSafetensors(
    { a: { dtype: "F64", shape: [1], data_offsets: [0, 8] } },
    new Uint8Array(8),
  );
  assertThrows(() => parseSafetensors(buffer), SafetensorsError, "未対応の dtype");
});

Deno.test("parseSafetensors: shape と data_offsets のサイズ不一致を拒否する", () => {
  const buffer = packSafetensors(
    { a: { dtype: "F32", shape: [2], data_offsets: [0, 4] } },
    new Uint8Array(4),
  );
  assertThrows(() => parseSafetensors(buffer), SafetensorsError, "サイズ不一致");
});

Deno.test("parseSafetensors: 負の次元・非整数の宣言を拒否する", () => {
  const negative = packSafetensors(
    { a: { dtype: "F32", shape: [-1], data_offsets: [0, 0] } },
    new Uint8Array(0),
  );
  assertThrows(() => parseSafetensors(negative), SafetensorsError, "shape 要素");

  const fractional = packSafetensors(
    { a: { dtype: "F32", shape: [1], data_offsets: [0.5, 4.5] } },
    new Uint8Array(4),
  );
  assertThrows(() => parseSafetensors(fractional), SafetensorsError, "data_offsets");
});

Deno.test("parseSafetensors: データ節の範囲外を拒否する", () => {
  const buffer = packSafetensors(
    { a: { dtype: "F32", shape: [2], data_offsets: [0, 8] } },
    new Uint8Array(4),
  );
  assertThrows(() => parseSafetensors(buffer), SafetensorsError, "範囲外");
});

Deno.test("parseSafetensors: 領域の重複を拒否する", () => {
  const buffer = packSafetensors(
    {
      a: { dtype: "F32", shape: [2], data_offsets: [0, 8] },
      b: { dtype: "F32", shape: [2], data_offsets: [4, 12] },
    },
    new Uint8Array(12),
  );
  assertThrows(() => parseSafetensors(buffer), SafetensorsError, "重複");
});

Deno.test("parseSafetensors: テンソル間の隙間を拒否する", () => {
  const buffer = packSafetensors(
    {
      a: { dtype: "F32", shape: [1], data_offsets: [0, 4] },
      b: { dtype: "F32", shape: [1], data_offsets: [8, 12] },
    },
    new Uint8Array(12),
  );
  assertThrows(() => parseSafetensors(buffer), SafetensorsError, "未使用領域");
});

Deno.test("parseSafetensors: 末尾の未使用領域を拒否する", () => {
  const buffer = packSafetensors(
    { a: { dtype: "F32", shape: [1], data_offsets: [0, 4] } },
    new Uint8Array(8),
  );
  assertThrows(() => parseSafetensors(buffer), SafetensorsError, "末尾に未使用領域");
});

Deno.test("parseSafetensors: 要素サイズに整列しないテンソルを拒否する", () => {
  // I8 1 バイトの直後に F32 を置くとデータ節先頭からの相対が 1 になり view を張れない。
  const buffer = packSafetensors(
    {
      a: { dtype: "I8", shape: [1], data_offsets: [0, 1] },
      b: { dtype: "F32", shape: [1], data_offsets: [1, 5] },
    },
    new Uint8Array(5),
  );
  assertThrows(() => parseSafetensors(buffer), SafetensorsError, "整列していない");
});

Deno.test("parseSafetensors: __metadata__ の非文字列値を拒否する", () => {
  const buffer = packSafetensors({ __metadata__: { k: 1 } }, new Uint8Array(0));
  assertThrows(() => parseSafetensors(buffer), SafetensorsError, "文字列でない");
});

Deno.test("parseSafetensors: data_offsets の逆転を拒否する", () => {
  const buffer = packSafetensors(
    { a: { dtype: "F32", shape: [1], data_offsets: [8, 4] } },
    new Uint8Array(8),
  );
  assertThrows(() => parseSafetensors(buffer), SafetensorsError, "逆転");
});
