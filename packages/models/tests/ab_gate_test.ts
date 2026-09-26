// 自機 A/B 門（ADR 0110 決定 5）の純関数部の門（GPU も資産も要らない — 入力は全て合成）。
//
// 実 GPU の門（e2e_*_ab_test.ts）が赤になる理由は全てここの判定を通るので、ここで固定するのは
// 「門が空振りしない」側の性質である: 同一配列は床で赤・非有限は上限側で赤・帯が未導出なら
// 実測を出して赤・参照席の候補が 0 / 2 個なら throw。

import { assertEquals, assertRejects, assertStringIncludes, assertThrows } from "@std/assert";
import { describe, it } from "@std/testing/bdd";
import { linearKey } from "../../runtime/src/kernels/linear.ts";
import { linearI8a8Key } from "../../runtime/src/kernels/linear-i8a8.ts";
import {
  type AbBand,
  assertBitIdentical,
  captureThenAbort,
  describeBundleCensus,
  judgeAb,
  maxAbs,
  measureAb,
  overrideQuantSession,
  type QuantTable,
  referencePartnerOf,
  relRms,
} from "./helpers/ab-gate.ts";

const I8_DIT = { text_encoder: "f16", transformer: "i8", vae_decoder: "f16" };
const F16_DIT = { text_encoder: "f16", transformer: "f16", vae_decoder: "f16" };

/** anima の配布形と同じ席の並び（同じ重みの参照席・実用席・重みの違う席）。 */
const ANIMA_LIKE: QuantTable = {
  models: {
    turbo: {
      quants: {
        f16: { weights: F16_DIT, session: {} },
        "f16+dit8": { weights: I8_DIT, session: {} },
        "f16+dit8-a8": { weights: I8_DIT, session: { linearCompute: "a8" } },
        "f16+dit8-a8-attn8-s16": {
          weights: I8_DIT,
          session: { linearCompute: "a8", attentionCompute: "a8", attentionScoreStorage: "f16" },
        },
        "f16-c16": {
          weights: F16_DIT,
          session: { linearCompute: "f16", attentionCompute: "f16" },
          gpuFeatures: { shaderF16: true },
        },
      },
    },
  },
};

const BAND: AbBand = { floor: 1e-3, ceiling: 0.1 };

describe("referencePartnerOf: 実用席の参照席を manifest から導く", () => {
  it("同じ weights で session が空の席がちょうど 1 つならそれを返す", () => {
    assertEquals(referencePartnerOf(ANIMA_LIKE, "turbo", "f16+dit8-a8-attn8-s16"), "f16+dit8");
    assertEquals(referencePartnerOf(ANIMA_LIKE, "turbo", "f16+dit8-a8"), "f16+dit8");
  });

  it("参照席の gpuFeatures が実用席の部分集合なら対になる（f16-c16 ↔ f16）", () => {
    assertEquals(referencePartnerOf(ANIMA_LIKE, "turbo", "f16-c16"), "f16");
  });

  it("参照席だけが feature を要る形は対にしない（候補 0 で throw）", () => {
    const table: QuantTable = {
      models: {
        m: {
          quants: {
            ref: { weights: I8_DIT, session: {}, gpuFeatures: { shaderF16: true } },
            fast: { weights: I8_DIT, session: { linearCompute: "a8" } },
          },
        },
      },
    };
    assertThrows(() => referencePartnerOf(table, "m", "fast"), Error, "が 0 個（なし）");
  });

  it("候補が 0 個なら throw する（重みが同じ参照席が配られていない）", () => {
    const table: QuantTable = {
      models: {
        m: {
          quants: {
            f16: { weights: F16_DIT, session: {} },
            "i8-a8": { weights: I8_DIT, session: { linearCompute: "a8" } },
          },
        },
      },
    };
    assertThrows(() => referencePartnerOf(table, "m", "i8-a8"), Error, "が 0 個");
  });

  it("候補が 2 個なら throw する（どちらと比べたかで床の意味が変わる）", () => {
    const table: QuantTable = {
      models: {
        m: {
          quants: {
            i8: { weights: I8_DIT, session: {} },
            "i8-copy": { weights: I8_DIT, session: {} },
            "i8-a8": { weights: I8_DIT, session: { linearCompute: "a8" } },
          },
        },
      },
    };
    assertThrows(() => referencePartnerOf(table, "m", "i8-a8"), Error, "が 2 個（i8 / i8-copy）");
  });

  it("session が空の席を実用席として渡すと throw する", () => {
    assertThrows(() => referencePartnerOf(ANIMA_LIKE, "turbo", "f16+dit8"), Error, "session が空");
  });

  it("無い model / quant は throw する", () => {
    assertThrows(() => referencePartnerOf(ANIMA_LIKE, "base", "f16"), Error, "model 'base' が無い");
    assertThrows(() => referencePartnerOf(ANIMA_LIKE, "turbo", "i4"), Error, "quant 'i4' が無い");
  });
});

describe("relRms / maxAbs", () => {
  it("‖p − r‖₂ / ‖r‖₂ と最大絶対差を返す", () => {
    const reference = new Float32Array([3, 4]);
    const practical = new Float32Array([3, 3]);
    assertEquals(relRms(practical, reference), 1 / 5);
    assertEquals(maxAbs(practical, reference), 1);
  });

  it("NaN を 0 として素通りさせない", () => {
    const reference = new Float32Array([1, 2, 3]);
    const practical = new Float32Array([1, Number.NaN, 3]);
    assertEquals(Number.isNaN(relRms(practical, reference)), true);
    assertEquals(Number.isNaN(maxAbs(practical, reference)), true);
  });

  it("要素数が違えば throw する", () => {
    assertThrows(
      () => relRms(new Float32Array(2), new Float32Array(3)),
      Error,
      "観測点の要素数が違う",
    );
  });
});

describe("judgeAb: A/B の判定", () => {
  const reference = new Float32Array([1, -2, 3, -4]);

  it("同一配列は帯の有無に関係なく床で赤になる（束が Session に届いていない形）", () => {
    const measured = measureAb(new Float32Array(reference), reference);
    assertThrows(() => judgeAb(measured, BAND, "case"), Error, "床の失敗 — 実用席が参照席と 1 bit");
    assertThrows(() => judgeAb(measured, undefined, "case"), Error, "床の失敗");
  });

  it("非有限は上限側の失敗として赤になる", () => {
    const measured = measureAb(new Float32Array([1, Number.NaN, 3, -4]), reference);
    assertThrows(() => judgeAb(measured, BAND, "case"), Error, "上限側の失敗 — 非有限");
    const infinite = measureAb(new Float32Array([1, -2, Number.POSITIVE_INFINITY, -4]), reference);
    assertThrows(() => judgeAb(infinite, BAND, "case"), Error, "上限側の失敗 — 非有限");
  });

  it("帯が未導出なら実測を文面に出して赤で止まる", () => {
    const measured = measureAb(new Float32Array([1, -2, 3, -4.5]), reference);
    const error = assertThrows(() => judgeAb(measured, undefined, "anima 512"), Error);
    assertStringIncludes(error.message, "anima 512: 帯が未導出（実測 relRMS ");
    assertStringIncludes(error.message, measured.relRms.toExponential(4));
    assertStringIncludes(error.message, "maxAbs 5.0000e-1");
  });

  it("帯の中なら緑、床の下・上限の上なら赤", () => {
    const measured = measureAb(new Float32Array([1, -2, 3, -4.1]), reference);
    judgeAb(measured, { floor: measured.relRms / 2, ceiling: measured.relRms * 2 }, "case");
    assertThrows(
      () => judgeAb(measured, { floor: measured.relRms * 2, ceiling: measured.relRms * 4 }, "case"),
      Error,
      "床の失敗 — relRMS",
    );
    assertThrows(
      () => judgeAb(measured, { floor: 0, ceiling: measured.relRms / 2 }, "case"),
      Error,
      "上限の失敗",
    );
  });

  it("壊れた帯の宣言は throw する", () => {
    const measured = measureAb(new Float32Array([1, -2, 3, -4.1]), reference);
    assertThrows(
      () => judgeAb(measured, { floor: 0.2, ceiling: 0.1 }, "case"),
      Error,
      "帯の宣言が壊れている",
    );
  });
});

describe("assertBitIdentical: デバイス内決定性", () => {
  it("ビット同一なら通る", () => {
    assertBitIdentical(new Float32Array([1, 2]), new Float32Array([1, 2]), "twice");
  });

  it("1 要素でも違えば数と位置を出して赤になる（±0 もビットでは別物）", () => {
    assertThrows(
      () => assertBitIdentical(new Float32Array([1, 0, 3]), new Float32Array([1, -0, 3]), "twice"),
      Error,
      "twice: 同じ席・同じ入力の 2 回がビット一致しない",
    );
    const error = assertThrows(
      () => assertBitIdentical(new Float32Array([1, 2, 3, 4]), new Float32Array([1, 5, 3, 6]), "x"),
      Error,
    );
    assertStringIncludes(error.message, "2 / 4 要素が違う。最初は 1: 2 / 5");
  });
});

describe("captureThenAbort: 観測点で生成を打ち切る", () => {
  it("観測点の値を返し、その後の生成は走らない", async () => {
    const seen: number[] = [];
    const value = await captureThenAbort<number>("gen", async (capture) => {
      for (let step = 1; step <= 3; step += 1) {
        await Promise.resolve();
        seen.push(step);
        if (step === 1) capture(step * 10);
      }
    });
    assertEquals(value, 10);
    assertEquals(seen, [1]);
  });

  it("打ち切り以外の例外はそのまま上げる", async () => {
    const cause = new Error("DiT の構築に失敗");
    const thrown = await assertRejects(() =>
      captureThenAbort<number>("gen", () => Promise.reject(cause))
    );
    assertEquals(thrown, cause);
  });

  it("観測点が 1 度も来ないまま終わったら throw する", async () => {
    await assertRejects(
      () => captureThenAbort<number>("gen", () => Promise.resolve()),
      Error,
      "gen: 観測点が 1 度も来ないまま生成が終わった",
    );
  });
});

describe("overrideQuantSession: 故障注入の一時 manifest", () => {
  it("指定した席の session だけを差し替え、元の値は書き換えない", () => {
    const raw = {
      format: "karume/5",
      models: {
        m: {
          defaultQuant: "i8-a8",
          quants: {
            i8: { weights: { dit: "i8" }, session: {} },
            "i8-a8": { weights: { dit: "i8" }, session: { linearCompute: "a8" } },
          },
        },
      },
    };
    const before = structuredClone(raw);
    const injected = overrideQuantSession(raw, "m", "i8-a8", {});
    assertEquals(injected, {
      ...raw,
      models: {
        m: {
          ...raw.models.m,
          quants: { ...raw.models.m.quants, "i8-a8": { weights: { dit: "i8" }, session: {} } },
        },
      },
    });
    assertEquals(raw, before);
    assertThrows(() => overrideQuantSession(raw, "m", "i4", {}), Error, "quants.i4");
  });
});

describe("describeBundleCensus: 束の各ノブの本数", () => {
  it("非参照値の席だけを、その変種の dispatch 本数つきで並べる", () => {
    const rows = [
      { key: linearI8a8Key(true, true), dispatchCount: 3 },
      { key: linearKey("f32", true), dispatchCount: 5 },
    ];
    assertEquals(
      describeBundleCensus(rows, { linearCompute: "a8", attentionCompute: "f32" }),
      "linearCompute=a8 ×3",
    );
    assertEquals(describeBundleCensus(rows, { attentionCompute: "a8" }), "attentionCompute=a8 ×0");
  });
});
