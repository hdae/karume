// テストの GPU の取得口の緩和（破棄済みの device の回収を促す — 対症療法）の振る舞い。GPU は使わない。

import { assert, assertEquals } from "@std/assert";
import { describe, it } from "@std/testing/bdd";
import { collectDestroyedDevices, GC_SETTLE_MS } from "./helpers/collect-destroyed-devices.ts";

// タイマの発火時刻の丸め分（setTimeout が要求より早く発火することは無いが、計測側の時計の粒度を許す）。
const CLOCK_SLACK_MS = 2;

/**
 * `start()` の Promise がタイマ（macrotask）を 1 つも待たずに解決するか。待ちの有無を経過時間の閾値で
 * 見ると負荷の高い機で揺れるので、0 ms のタイマとの先着で見る（タイマを待たない解決は microtask で先に
 * 着く）。見張りのタイマを `start()` より先に積むので、相手が 0 ms のタイマを 1 つでも待てば見張りが先に着く。
 */
const resolvesWithoutTimer = async (start: () => Promise<void>): Promise<boolean> => {
  let timer: ReturnType<typeof setTimeout> | undefined;
  const timerFired = new Promise<"timer">((resolve) => {
    timer = setTimeout(() => resolve("timer"), 0);
  });
  try {
    return await Promise.race([start().then(() => "helper" as const), timerFired]) === "helper";
  } finally {
    clearTimeout(timer);
  }
};

describe("collectDestroyedDevices", () => {
  describe("gc がある", () => {
    it("gc をちょうど 2 回呼び、2 回目の後の待ちを終えてから解決する", async () => {
      const calls: number[] = [];
      const start = performance.now();
      await collectDestroyedDevices({ gc: () => calls.push(performance.now()) });
      const resolved = performance.now();
      assertEquals(calls.length, 2);
      const [first, second] = calls;
      assert(first - start < GC_SETTLE_MS, "1 回目の gc を待ちの前に呼んでいない");
      assert(
        second - first >= GC_SETTLE_MS - CLOCK_SLACK_MS,
        `1 回目の後の待ちが短い（${second - first} ms）`,
      );
      assert(
        resolved - second >= GC_SETTLE_MS - CLOCK_SLACK_MS,
        `2 回目の後の待ちを終える前に解決した（${resolved - second} ms）`,
      );
    });
  });

  describe("gc が無い", () => {
    it("何も呼ばず、待たずに解決する", async () => {
      assert(await resolvesWithoutTimer(() => collectDestroyedDevices({})), "gc が無いのに待った");
    });

    it("gc が関数でなければ呼ばず、待たずに解決する", async () => {
      // `--expose-gc` の無い走行で別の何かが `gc` の名を持っていても、呼ぼうとして落ちず、待ちもしない。
      assert(
        await resolvesWithoutTimer(() => collectDestroyedDevices({ gc: "not a function" })),
        "gc が関数でないのに待った",
      );
    });
  });

  describe("既定（globalThis）", () => {
    it("--expose-gc の走行では 2 回分待ってから解決し、無い走行では待たずに解決する", async () => {
      // 走行のフラグで分岐する: deno.json の task（フラグ付き）では前者、フラグ無しの直の走行では後者を見る。
      const host: object = globalThis;
      const exposedGc = "gc" in host ? host.gc : undefined;
      if (typeof exposedGc !== "function") {
        assert(await resolvesWithoutTimer(() => collectDestroyedDevices()), "gc が無いのに待った");
        return;
      }
      const start = performance.now();
      await collectDestroyedDevices();
      const elapsed = performance.now() - start;
      assert(
        elapsed >= 2 * (GC_SETTLE_MS - CLOCK_SLACK_MS),
        `既定の globalThis.gc を使っていない（${elapsed} ms で解決した）`,
      );
    });
  });
});
