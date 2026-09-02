// parseSafetensors の「ファイル長を別に受ける」面（器の使い回し — 供給側が最大 shard 長の buffer へ
// 毎回の shard を先頭から読むと、buffer の末尾に前回の残りが居る）。長さの検査が buffer 全体ではなく
// 渡された長さで行われることと、長さの取り違えが黙って通らないことを縛る。
import { assertEquals, assertThrows } from "@std/assert";
import { parseSafetensors, SafetensorsError, tensorBytes } from "../src/format/safetensors.ts";
import { buildSafetensors, f32Bytes } from "./helpers/format.ts";

const FILE = buildSafetensors([{ name: "a", dtype: "F32", shape: [2], data: f32Bytes([1, 2]) }]);

Deno.test("parseSafetensors: byteLength を渡せば buffer 末尾の余白（前回の残り）を無視する", () => {
  const vessel = new Uint8Array(new ArrayBuffer(FILE.byteLength + 64));
  vessel.set(new Uint8Array(FILE));
  vessel.fill(0xab, FILE.byteLength); // 前の shard の残骸に見せる
  const file = parseSafetensors(vessel.buffer, FILE.byteLength);
  assertEquals(file.tensors.get("a")?.shape, [2]);
  assertEquals(tensorBytes(file, file.tensors.get("a")!), f32Bytes([1, 2]));
  // 長さを渡さなければ余白は「末尾の未使用領域」として従来どおり落ちる。
  assertThrows(() => parseSafetensors(vessel.buffer), SafetensorsError, "未使用領域");
});

Deno.test("parseSafetensors: buffer より長い byteLength・負・非整数は拒否する", () => {
  for (const bad of [FILE.byteLength + 1, -1, 1.5]) {
    assertThrows(() => parseSafetensors(FILE, bad), SafetensorsError, "収まっていない");
  }
});

Deno.test("parseSafetensors: byteLength がヘッダ長やデータ節より短ければ従来の門で落ちる", () => {
  assertThrows(() => parseSafetensors(FILE, 4), SafetensorsError, "短すぎる");
  assertThrows(() => parseSafetensors(FILE, FILE.byteLength - 4), SafetensorsError, "範囲外");
});
