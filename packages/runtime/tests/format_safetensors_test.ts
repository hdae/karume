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

// ADR 0069 決定 2: I4 は shape を論理形のまま持ち、バイト数だけが bit 幅から決まる（numel/2）。
Deno.test("parseSafetensors: I4 は論理 shape のまま numel/2 バイトで受理する", () => {
  const buffer = buildSafetensors([
    { name: "w", dtype: "I4", shape: [3, 32], data: new Uint8Array(48) },
  ]);
  const file = parseSafetensors(buffer);
  const view = file.tensors.get("w");
  if (view === undefined) throw new Error("tensor w が無い");

  assertEquals(view.shape, [3, 32]);
  assertEquals(view.byteLength, 48);
  // 4bit の TypedArray は存在しないので view は raw バイトのまま（3 面目 = raw + 論理 numel）
  assertEquals(tensorBytes(file, view).byteLength, 48);
});

// 要素数が奇数だと bit 総量が byte 境界に乗らず、末尾要素が半バイトだけ突き出す。
Deno.test("parseSafetensors: 要素数が奇数の I4 を拒否する", () => {
  const buffer = packSafetensors(
    { w: { dtype: "I4", shape: [3], data_offsets: [0, 2] } },
    new Uint8Array(2),
  );
  assertThrows(() => parseSafetensors(buffer), SafetensorsError, "byte 境界に乗らない");
});

// I4 は要素整列（0.5 バイト）ではなく「テンソル先頭が 4 バイト整列」を要求する（u32 束縛）。
Deno.test("parseSafetensors: I4 のテンソル先頭が 4 バイト整列していない形を拒否する", () => {
  const buffer = packSafetensors(
    {
      a: { dtype: "I8", shape: [2], data_offsets: [0, 2] },
      w: { dtype: "I4", shape: [4], data_offsets: [2, 4] },
    },
    new Uint8Array(4),
  );
  assertThrows(() => parseSafetensors(buffer), SafetensorsError, "整列していない");
});

Deno.test("parseSafetensors: __metadata__ の非文字列値を拒否する", () => {
  const buffer = packSafetensors({ __metadata__: { k: 1 } }, new Uint8Array(0));
  assertThrows(() => parseSafetensors(buffer), SafetensorsError, "文字列でない");
});

/**
 * 残りのガード節（ヘッダ長 / ヘッダの文字コード / 宣言の型 / 要素数）。いずれも下流の検査が
 * 別の文言で落とす見込みが高く沈黙誤値にはならないが、**配布形を組み直す側が読む診断の
 * 帰属**が変わる（「ヘッダの型が違う」が「サイズ不一致」として出ると直す場所を取り違える）。
 */
Deno.test("parseSafetensors: ヘッダ長とヘッダ表の型のガードを個別の文言で落とす", () => {
  // ① ヘッダ長が安全整数を超える（Number へ落とすより前に見る）
  const huge = new ArrayBuffer(16);
  new DataView(huge).setBigUint64(0, 2n ** 60n, true);
  assertThrows(() => parseSafetensors(huge), SafetensorsError, "安全整数を超える");

  // ② ヘッダが UTF-8 として不正（0xff は単独では不正なバイト）
  const invalidUtf8 = new ArrayBuffer(16);
  new DataView(invalidUtf8).setBigUint64(0, 8n, true);
  new Uint8Array(invalidUtf8, 8).fill(0xff);
  assertThrows(() => parseSafetensors(invalidUtf8), SafetensorsError, "UTF-8 として不正");

  // ③ ヘッダ項目がオブジェクトでない
  assertThrows(
    () => parseSafetensors(packSafetensors({ a: 5 }, new Uint8Array(0))),
    SafetensorsError,
    "ヘッダ項目がオブジェクトでない",
  );

  // ④ shape が配列でない
  assertThrows(
    () =>
      parseSafetensors(
        packSafetensors({ a: { dtype: "F32", shape: "x", data_offsets: [0, 4] } }, F32_2),
      ),
    SafetensorsError,
    "shape が配列でない",
  );

  // ⑤ data_offsets が配列でない / 要素数が 2 でない
  assertThrows(
    () =>
      parseSafetensors(
        packSafetensors({ a: { dtype: "F32", shape: [1], data_offsets: 0 } }, new Uint8Array(4)),
      ),
    SafetensorsError,
    "data_offsets が配列でない",
  );
  assertThrows(
    () =>
      parseSafetensors(
        packSafetensors({ a: { dtype: "F32", shape: [1], data_offsets: [0] } }, new Uint8Array(4)),
      ),
    SafetensorsError,
    "data_offsets の要素数が 1",
  );

  // ⑥ 要素数が安全整数を超える（宣言だけで踏める — 実データは要らない）
  assertThrows(
    () =>
      parseSafetensors(
        packSafetensors(
          { a: { dtype: "F32", shape: [2 ** 30, 2 ** 30], data_offsets: [0, 0] } },
          new Uint8Array(0),
        ),
      ),
    SafetensorsError,
    "要素数が安全整数を超える",
  );

  // ⑦ __metadata__ がオブジェクトでない（配列は JSON では object 型だが表ではない）
  assertThrows(
    () => parseSafetensors(packSafetensors({ __metadata__: [] }, new Uint8Array(0))),
    SafetensorsError,
    "__metadata__ がオブジェクトでない",
  );
});

Deno.test("parseSafetensors: data_offsets の逆転を拒否する", () => {
  const buffer = packSafetensors(
    { a: { dtype: "F32", shape: [1], data_offsets: [8, 4] } },
    new Uint8Array(8),
  );
  assertThrows(() => parseSafetensors(buffer), SafetensorsError, "逆転");
});
