// hold-vram の保持量の解釈（GPU 不要 — 本体は import.meta.main の内側なので import しても
// GPU を掴まない）。
//
// 「診断の道具でも黙って既定に落ちない」を固定する: 既定は引数を渡さなかったときだけで、
// 綴りが壊れていれば受け取った値ごと落ちる。

import { assertEquals, assertThrows } from "@std/assert";
import { parseMib } from "./hold-vram.ts";

Deno.test("parseMib: 引数なしは既定 4608MiB", () => {
  assertEquals(parseMib(undefined), 4608);
});

Deno.test("parseMib: 正の整数はそのまま受ける", () => {
  assertEquals(parseMib("256"), 256);
});

Deno.test("parseMib: 0 / 小数 / 非数は受け取った値を添えて落ちる（既定に化けない）", () => {
  assertThrows(() => parseMib("0"), Error, "受け取った値: 0");
  assertThrows(() => parseMib("-1"), Error, "受け取った値: -1");
  assertThrows(() => parseMib("4.5"), Error, "受け取った値: 4.5");
  assertThrows(() => parseMib("abc"), Error, "受け取った値: abc");
  assertThrows(() => parseMib(""), Error, "受け取った値: ");
});
