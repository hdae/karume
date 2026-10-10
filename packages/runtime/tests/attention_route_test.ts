/**
 * 融合 attention の経路の判定（src/runtime/attention-route.ts）— 席の要求 × 形 → 段ごとの実経路。
 * GPU 不要。条件の綴り（① は `D % 4`・③ は `N % 4`・S の f16 格納は両方）は ADR 0030 決定 5 /
 * score-storage.ts の doc が正本で、ここはその仕様を外から固定する。
 */

import { assertEquals } from "@std/assert";
import { describe, it } from "@std/testing/bdd";
import { attentionRoute, type AttentionSeats } from "../src/runtime/attention-route.ts";

const A8_S16: AttentionSeats = { attentionCompute: "a8", attentionScoreStorage: "f16" };

describe("attentionRoute: a8 + s16 の要求", () => {
  it("D と N がともに 4 の倍数なら、①③ が i8a8・S が f16 格納（宣言どおり）", () => {
    assertEquals(attentionRoute(A8_S16, 128, 4096), { qk: "i8a8", pv: "i8a8", score: "f16" });
  });

  it("N % 4 != 0 は ③PV と S の格納だけが f32 へ落ち、①QK は i8a8 のまま（段ごとに独立）", () => {
    for (const cols of [1089, 1122, 3969]) {
      assertEquals(
        attentionRoute(A8_S16, 128, cols),
        { qk: "i8a8", pv: "dense", score: "f32" },
        `N=${cols}`,
      );
    }
  });

  it("D % 4 != 0 は ①QK と S の格納だけが落ち、③PV は i8a8 のまま", () => {
    assertEquals(attentionRoute(A8_S16, 66, 512), { qk: "dense", pv: "i8a8", score: "f32" });
  });

  it("D・N の小さい格子の全点で、① は D % 4・③ は N % 4・S は両方の条件だけで決まる", () => {
    for (let depth = 1; depth <= 12; depth++) {
      for (let cols = 1; cols <= 12; cols++) {
        const qkOk = depth % 4 === 0;
        const pvOk = cols % 4 === 0;
        assertEquals(attentionRoute(A8_S16, depth, cols), {
          qk: qkOk ? "i8a8" : "dense",
          pv: pvOk ? "i8a8" : "dense",
          score: qkOk && pvOk ? "f16" : "f32",
        }, `D=${depth} N=${cols}`);
      }
    }
  });
});

describe("attentionRoute: a8 / s16 を要求しない席", () => {
  it("f32 / f16 の計算は、適格な形でも ①③ が dense（i8a8 は要求したときだけ）", () => {
    for (const attentionCompute of ["f32", "f16"] as const) {
      assertEquals(
        attentionRoute({ attentionCompute, attentionScoreStorage: "f32" }, 128, 4096),
        { qk: "dense", pv: "dense", score: "f32" },
        attentionCompute,
      );
    }
  });

  it("s16 だけの要求は、①③ が dense のまま S だけが f16 格納になる（2 つの軸は直交）", () => {
    assertEquals(
      attentionRoute({ attentionCompute: "f32", attentionScoreStorage: "f16" }, 128, 512),
      { qk: "dense", pv: "dense", score: "f16" },
    );
  });

  it("a8 だけの要求は、S が f32 格納のまま（N % 4 != 0 でも ③ だけが落ちる）", () => {
    const seats: AttentionSeats = { attentionCompute: "a8", attentionScoreStorage: "f32" };
    assertEquals(attentionRoute(seats, 128, 4096), { qk: "i8a8", pv: "i8a8", score: "f32" });
    assertEquals(attentionRoute(seats, 128, 1089), { qk: "i8a8", pv: "dense", score: "f32" });
  });
});
