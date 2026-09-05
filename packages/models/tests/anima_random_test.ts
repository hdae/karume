// Anima の決定的乱数（初期 latent を丸ごと決める生成器）の挙動テスト。GPU も資産も要らない。
//
// `src/anima/random.ts` は sbv2 / irodori と**意図的な複製**（`generation/random.ts` の doc）
// なので、他家族のテスト（`sbv2_host_test.ts` / `irodori_host_test.ts`）は anima 側の写しの
// 退行を検出しない。実 GPU の PNG sha256 門（`e2e_anima_test.ts`）は資産の無い機では走らない
// ので、資産不要の門はここが唯一になる。
//
// NOTE: `random.ts:76` の `u1 === 0` 置換（`log(0) = −Inf` 回避）は 1 ドローあたり 2⁻⁵³ で、
// 故障注入の口（差し替え可能な一様乱数源）も無いので**直接は踏めない** — この分岐だけは
// テスト不能として明記する（塞ぐには生成器の面を広げるしかなく、それは値の正本を緩める）。

import { assert, assertEquals, assertThrows } from "@std/assert";
import { assertAcceptableSeed, Randn } from "../src/anima/random.ts";

Deno.test("Randn: 同じ seed は同じ列・違う seed は違う列", () => {
  const first = new Randn(7).normals(64);
  const second = new Randn(7).normals(64);
  const other = new Randn(8).normals(64);
  assertEquals([...first], [...second]);
  assert([...first].some((value, index) => value !== other[index]), "seed を変えても同じ列");
});

Deno.test("Randn: 奇数長でも Box–Muller の対を持ち越さない", () => {
  // 持ち越すと「同じ seed でも呼び出し順で列が変わる」形になり、seed を渡した再現性が崩れる。
  // 奇数長 → 次の呼び出し、が取り直した生成器の同じ 2 回と一致することで固定する。
  const split = new Randn(5);
  const head = split.normals(3);
  const tail = split.normals(2);
  const fresh = new Randn(5);
  assertEquals([...head], [...fresh.normals(3)]);
  assertEquals([...tail], [...fresh.normals(2)]);
});

Deno.test("Randn: scale は同じ列に線形に効く（列そのものは seed で決まる）", () => {
  const plain = new Randn(3).normals(16);
  const scaled = new Randn(3).normals(16, 2);
  for (const [index, value] of plain.entries()) {
    assertEquals(scaled[index], Math.fround(value * 2), `要素 ${index}`);
  }
});

Deno.test("Randn: seed の受理集合は assertAcceptableSeed と同じ（生成器側の入口）", () => {
  // 受理集合は 1 本しか持たない（`generate` の入口と生成器で条件が割れると、片方だけ緩む）。
  for (const seed of [-1, 1.5, Number.MAX_SAFE_INTEGER + 1]) {
    assertThrows(() => assertAcceptableSeed(seed), RangeError, "非負の安全整数でない");
    assertThrows(() => new Randn(seed), RangeError, "非負の安全整数でない");
  }
  assertAcceptableSeed(0);
  assertEquals(new Randn(0).normals(2).length, 2);
});
