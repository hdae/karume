import { assertEquals, assertStrictEquals, assertThrows } from "@std/assert";
import {
  parseSafetensors,
  parseSafetensorsHeader,
  SafetensorsError,
  safetensorsHeaderLength,
  tensorBytes,
} from "../src/format/safetensors.ts";
import {
  buildSafetensors,
  f32Bytes,
  packSafetensors,
  packSafetensorsRaw,
} from "./helpers/safetensors.ts";

const F32_2 = f32Bytes([1, 2]);

Deno.test("parseSafetensors: 正常系はテンソル表と __metadata__ を取り出す", () => {
  const buffer = buildSafetensors(
    [
      { name: "a", dtype: "F32", shape: [2, 1], data: f32Bytes([1, 2]) },
      { name: "b", dtype: "I8", shape: [3], data: new Uint8Array([7, 8, 9]) },
    ],
    { note: "{}", extra: "x" },
  );
  const file = parseSafetensors(buffer);

  assertEquals([...file.tensors.keys()].sort(), ["a", "b"]);
  assertEquals(file.metadata.get("note"), "{}");
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

  // 旧配布形の方言だった packed 4bit / 2bit（公式 safetensors には無い綴り）も同じ門で落ちる
  // MUST — 黙って受理すると、容器（`krm`）へ移したはずの量子化格納が資産経路から戻ってくる。
  for (const dtype of ["I4", "I2"]) {
    assertThrows(
      () =>
        parseSafetensors(
          packSafetensors({ w: { dtype, shape: [8], data_offsets: [0, 4] } }, new Uint8Array(4)),
        ),
      SafetensorsError,
      "未対応の dtype",
    );
  }
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

/**
 * ヘッダだけを解く面（区間読みする呼び手のための部分適用）。データ節を 1 バイトも持たずに
 * 全量解析と同じ表が出ること、prefix / ファイル長の取り違えが黙って通らないことを縛る。
 */
const HEADER_FIXTURE = buildSafetensors(
  [
    { name: "a", dtype: "F32", shape: [2, 1], data: f32Bytes([1, 2]) },
    { name: "b", dtype: "I8", shape: [3], data: new Uint8Array([7, 8, 9]) },
  ],
  { note: "{}", extra: "x" },
);

Deno.test("parseSafetensorsHeader: ヘッダ区間だけで全量解析と同じ表を返す", () => {
  const full = parseSafetensors(HEADER_FIXTURE);

  // 呼び手の 2 段読み: 8 バイト読む → ヘッダ長 N を知る → 8+N バイトを読み直す。
  const headerLength = safetensorsHeaderLength(new Uint8Array(HEADER_FIXTURE, 0, 8));
  const prefix = new Uint8Array(HEADER_FIXTURE, 0, 8 + headerLength);
  const header = parseSafetensorsHeader(prefix, HEADER_FIXTURE.byteLength);

  assertEquals(header.dataStart, 8 + headerLength);
  assertEquals([...header.metadata], [...full.metadata]);
  // byteOffset はファイル先頭からの絶対値のまま — 呼び手はこれをそのまま区間読みへ渡す。
  assertEquals([...header.tensors], [...full.tensors]);

  // 固定長読み（例 64KiB）で余分に読んだ prefix でも同じ表 — 切り出しは 8+N の内側で閉じている。
  assertEquals(
    [
      ...parseSafetensorsHeader(
        new Uint8Array(HEADER_FIXTURE, 0, 8 + headerLength + 2),
        HEADER_FIXTURE.byteLength,
      ).tensors,
    ],
    [...full.tensors],
  );
  // prefix を `ArrayBuffer` のまま渡す成功経路（view 版と同じ表）。
  assertEquals(
    [
      ...parseSafetensorsHeader(
        HEADER_FIXTURE.slice(0, 8 + headerLength),
        HEADER_FIXTURE.byteLength,
      )
        .tensors,
    ],
    [...full.tensors],
  );

  // 器の途中に置いた prefix（view の byteOffset 越し）でも同じ結果になる。
  const vessel = new Uint8Array(new ArrayBuffer(16 + prefix.byteLength));
  vessel.set(prefix, 16);
  const offsetPrefix = vessel.subarray(16);
  assertEquals(safetensorsHeaderLength(offsetPrefix), headerLength);
  assertEquals(
    [...parseSafetensorsHeader(offsetPrefix, HEADER_FIXTURE.byteLength).tensors],
    [...full.tensors],
  );
});

Deno.test("parseSafetensorsHeader: prefix がヘッダ途中で切れていれば必要長を文言に載せて落ちる", () => {
  const need = 8 + safetensorsHeaderLength(HEADER_FIXTURE);
  assertThrows(
    () =>
      parseSafetensorsHeader(
        new Uint8Array(HEADER_FIXTURE, 0, need - 1),
        HEADER_FIXTURE.byteLength,
      ),
    SafetensorsError,
    `先頭 ${need} バイトが必要`,
  );
});

Deno.test("parseSafetensorsHeader: ファイル長の取り違えは全量解析と同じ文言で落ちる", () => {
  // 実長より長い = 覆えていない末尾がある / 短い = 宣言がデータ節をはみ出す。
  assertThrows(
    () => parseSafetensorsHeader(HEADER_FIXTURE, HEADER_FIXTURE.byteLength + 8),
    SafetensorsError,
    "末尾に未使用領域",
  );
  assertThrows(
    () => parseSafetensorsHeader(HEADER_FIXTURE, HEADER_FIXTURE.byteLength - 4),
    SafetensorsError,
    "範囲外",
  );
  assertThrows(
    () => parseSafetensors(HEADER_FIXTURE, HEADER_FIXTURE.byteLength - 4),
    SafetensorsError,
    "範囲外",
  );

  assertThrows(
    () => parseSafetensorsHeader(HEADER_FIXTURE, -1),
    SafetensorsError,
    "非負整数でない",
  );
  assertThrows(() => parseSafetensorsHeader(HEADER_FIXTURE, 4), SafetensorsError, "短すぎる");
});

Deno.test("safetensorsHeaderLength: 8 バイト未満の prefix を拒否する", () => {
  assertThrows(
    () => safetensorsHeaderLength(new ArrayBuffer(7)),
    SafetensorsError,
    "先頭 8 バイトが必要",
  );
  assertThrows(
    () => safetensorsHeaderLength(new Uint8Array(new ArrayBuffer(7))),
    SafetensorsError,
    "先頭 8 バイトが必要",
  );
});

/**
 * **2 引数形**（`parseSafetensors(buffer, byteLength)`）— 供給側が最大 shard 長の buffer を
 * 使い回し、そこへ毎回の shard を先頭から読む形では、buffer の末尾に前回の残骸が居る。
 * 長さの検査が buffer 全体ではなく**渡された長さ**で行われることと、長さの取り違えが黙って
 * 通らないことを縛る。
 */
const PREFIX_FILE = buildSafetensors([
  { name: "a", dtype: "F32", shape: [2], data: f32Bytes([1, 2]) },
]);

Deno.test("parseSafetensors: byteLength を渡せば buffer 末尾の余白（前回の残骸）を無視する", () => {
  const buffer = new ArrayBuffer(PREFIX_FILE.byteLength + 64);
  const bytes = new Uint8Array(buffer);
  bytes.set(new Uint8Array(PREFIX_FILE));
  bytes.fill(0xab, PREFIX_FILE.byteLength); // 前の shard の残骸に見せる
  const file = parseSafetensors(buffer, PREFIX_FILE.byteLength);

  const view = file.tensors.get("a");
  if (view === undefined) throw new Error("tensor a が無い");
  assertEquals(view.shape, [2]);
  assertEquals(tensorBytes(file, view), f32Bytes([1, 2]));

  // 長さを渡さなければ余白は「末尾の未使用領域」として従来どおり落ちる。
  assertThrows(() => parseSafetensors(buffer), SafetensorsError, "未使用領域");
});

Deno.test("parseSafetensors: buffer より長い byteLength・負・非整数は拒否する", () => {
  for (const bad of [PREFIX_FILE.byteLength + 1, -1, 1.5]) {
    assertThrows(() => parseSafetensors(PREFIX_FILE, bad), SafetensorsError, "収まっていない");
  }
});

Deno.test("parseSafetensors: byteLength がヘッダ長やデータ節より短ければ従来の門で落ちる", () => {
  assertThrows(() => parseSafetensors(PREFIX_FILE, 4), SafetensorsError, "短すぎる");
  assertThrows(
    () => parseSafetensors(PREFIX_FILE, PREFIX_FILE.byteLength - 4),
    SafetensorsError,
    "範囲外",
  );
});
