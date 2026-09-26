// manifest の `session`（manifest 所有語彙）→ runtime `SessionOptions` の写像。
// ADR 0038 §3 の綴りの契約そのもので、抜けは**沈黙劣化**（未知キーは runtime が黙って無視する）
// になる。7 家族が同じ 1 本を使うので、門もここ 1 本に集約する（元は anima / sbv2 の
// pipeline テストへ 2 本に割れていて、残り 5 家族は写像を直接叩く門を持っていなかった）。

import { assert, assertEquals, assertThrows } from "@std/assert";
import { describe, it } from "@std/testing/bdd";
import type { SessionSpec } from "@karume/hub";
import { sessionOptionsViolation } from "@karume/runtime";
import { ANIMA_SESSION_POLICY } from "../src/anima/pipeline.ts";
import { BIREFNET_SESSION_POLICY } from "../src/birefnet/pipeline.ts";
import { DEPTH_ANYTHING_SESSION_POLICY } from "../src/depth-anything/pipeline.ts";
import { ModelInputError } from "../src/errors.ts";
import { GEMMA_SESSION_POLICY } from "../src/gemma/session-options.ts";
import { IRODORI_SESSION_POLICY } from "../src/irodori/admission.ts";
import { SBV2_SESSION_POLICY } from "../src/sbv2/pipeline.ts";
import {
  assertSessionOverrides,
  type FamilySessionPolicy,
  resolveSessionOptions,
  type SessionOverrides,
  toSessionOptions,
} from "../src/session/options.ts";
import { SIGLIP2_SESSION_POLICY } from "../src/siglip2/pipeline.ts";
import { VOWEL_DETECTOR_SESSION_POLICY } from "../src/vowel-detector/pipeline.ts";

Deno.test("toSessionOptions: 宣言したキーを 1 つずつ写す（未指定は欄ごと作らない）", () => {
  assertEquals(toSessionOptions({}), {});
  assertEquals(toSessionOptions({ linearGemvReduce: "parallel" }), {
    linearGemvReduce: "parallel",
  });
  assertEquals(toSessionOptions({ linearGemvReduce: "sequential" }), {
    linearGemvReduce: "sequential",
  });
  assertEquals(toSessionOptions({ linearCompute: "a8" }), { linearCompute: "a8" });
  assertEquals(toSessionOptions({ attentionCompute: "f16" }), { attentionCompute: "f16" });
  assertEquals(toSessionOptions({ attentionScoreStorage: "f16" }), {
    attentionScoreStorage: "f16",
  });
  // 配布物の既定 quant（f16+dit8-a8-attn8-s16）の 3 キーが全て通ること。1 キーでも落とすと
  // 「名前だけ s16」の沈黙劣化になる。
  assertEquals(
    toSessionOptions({
      linearCompute: "a8",
      attentionCompute: "a8",
      attentionScoreStorage: "f16",
    }),
    {
      linearCompute: "a8",
      attentionCompute: "a8",
      attentionScoreStorage: "f16",
    },
  );
});

Deno.test("toSessionOptions: manifest 側に無いノブ（submitPolicy）は写さない", () => {
  // `SessionOptions.submitPolicy` は TDR 予算 = **ホスト政策**なので配布者に書かせない
  // （ADR 0038 §3 の理由 ③）。スプレッド素通しに書き換えるとここが素通りしうる。
  const mapped = toSessionOptions({ linearCompute: "a8" }) as Record<string, unknown>;
  assertEquals(Object.hasOwn(mapped, "submitPolicy"), false);
  assertEquals(Object.keys(mapped), ["linearCompute"]);
});

Deno.test("toSessionOptions: SessionSpec の全キーを写す（キー追加の取り残しを検出）", () => {
  // 写像の網羅は `src/session/options.ts` の `WRITERS`（`Required<SessionSpec>` の網羅表）が
  // 型で固定しており、キーが増えれば**コンパイルエラー**になる。ここはその型門が生きている
  // ことを実行時からも見る対（型を緩めた改変は型検査を通ってしまうため）—
  // 全キーを埋めた spec を渡し、出てくるキー集合が入力と一致することを確かめる。
  const full: Required<SessionSpec> = {
    linearCompute: "f16",
    attentionCompute: "f16",
    attentionScoreStorage: "f16",
    linearGemvReduce: "sequential",
    stateAttentionReduce: "parallel-fused",
    fuseRmsNormAdd: false,
    fuseLinearStaticQuantize: false,
    packedStaticQuantize: false,
  };
  const mapped = toSessionOptions(full) as Record<string, unknown>;
  assertEquals(Object.keys(mapped).sort(), Object.keys(full).sort());
});

Deno.test("toSessionOptions: 融合の明示falseとtrueを保持する", () => {
  for (const value of [false, true]) {
    const spec = {
      fuseRmsNormAdd: value,
      fuseLinearStaticQuantize: value,
      packedStaticQuantize: value,
    };
    assertEquals(toSessionOptions(spec), spec);
  }
});

// ---- 合成（明示 > quant 宣言 > runtime 既定）— ADR 0058 追記 2026-09-26 ------------------

/** 8 キーを全て受ける受理表（規則そのものを家族の表から切り離して見るため）。 */
const ALL: FamilySessionPolicy = {
  linearCompute: true,
  attentionCompute: true,
  attentionScoreStorage: true,
  linearGemvReduce: true,
  stateAttentionReduce: true,
  fuseRmsNormAdd: true,
  fuseLinearStaticQuantize: true,
  packedStaticQuantize: true,
};

/** linear の実行形だけを受ける表（irodori / sbv2 と同じ形）。 */
const LINEAR_ONLY: FamilySessionPolicy = {
  ...ALL,
  attentionCompute: false,
  attentionScoreStorage: false,
  linearGemvReduce: false,
  stateAttentionReduce: false,
  fuseRmsNormAdd: false,
  fuseLinearStaticQuantize: false,
  packedStaticQuantize: false,
};

/** 全キーを埋めた spec（キーが増えればこの宣言が型検査で落ちる）。 */
const FULL: Required<SessionSpec> = {
  linearCompute: "f32",
  attentionCompute: "f32",
  attentionScoreStorage: "f32",
  linearGemvReduce: "parallel",
  stateAttentionReduce: "parallel",
  fuseRmsNormAdd: false,
  fuseLinearStaticQuantize: false,
  packedStaticQuantize: false,
};

/** 型の外から来る呼び手（JS の消費者）の明示指定を再現する — TS の型はこの経路を守らない。 */
const overridesWith = (key: string, value: unknown): SessionOverrides => {
  const overrides: SessionOverrides = {};
  Object.defineProperty(overrides, key, { value, enumerable: true });
  return overrides;
};

describe("resolveSessionOptions: キーごとの優先順位", () => {
  it("明示があれば明示・無ければ宣言・どちらも無ければ欄を作らない", () => {
    assertEquals(resolveSessionOptions(ALL, {}, {}, "test"), {});
    assertEquals(resolveSessionOptions(ALL, { linearCompute: "a8" }, {}, "test"), {
      linearCompute: "a8",
    });
    assertEquals(
      resolveSessionOptions(ALL, { linearCompute: "a8", attentionCompute: "a8" }, {
        linearCompute: "f16",
      }, "test"),
      { linearCompute: "f16", attentionCompute: "a8" },
    );
    assertEquals(resolveSessionOptions(ALL, {}, { attentionScoreStorage: "f16" }, "test"), {
      attentionScoreStorage: "f16",
    });
  });

  it("明示の false は宣言の true に勝ち、undefined は「指定なし」として宣言が残る", () => {
    const declared = { linearGemvReduce: "parallel", fuseRmsNormAdd: true } as const;
    assertEquals(resolveSessionOptions(ALL, declared, { fuseRmsNormAdd: false }, "test"), {
      linearGemvReduce: "parallel",
      fuseRmsNormAdd: false,
    });
    assertEquals(
      resolveSessionOptions(ALL, declared, { fuseRmsNormAdd: undefined }, "test"),
      declared,
    );
  });

  it("GPU の能力に依る値（f16 計算・subgroup 変種）はこの層では落とさない", () => {
    assertEquals(resolveSessionOptions(ALL, {}, { linearCompute: "f16" }, "test"), {
      linearCompute: "f16",
    });
    assertEquals(
      resolveSessionOptions(ALL, {}, { linearGemvReduce: "parallel-subgroup32" }, "test"),
      { linearGemvReduce: "parallel-subgroup32" },
    );
  });
});

describe("resolveSessionOptions: 受理できない指定", () => {
  it("不正な明示値は宣言へ戻さず ModelInputError・値の変換も呼ばない", () => {
    let conversions = 0;
    const object = {
      [Symbol.toPrimitive](): string {
        conversions++;
        return "f32";
      },
    };
    const messages: Readonly<Record<keyof SessionSpec, string>> = {
      linearCompute: "linearComputeが不正",
      attentionCompute: "attentionComputeが不正",
      attentionScoreStorage: "attentionScoreStorageが不正",
      linearGemvReduce: "linearGemvReduceが不正",
      stateAttentionReduce: "stateAttentionReduceが不正",
      fuseRmsNormAdd: "fuseRmsNormAddはbooleanでなければならない",
      fuseLinearStaticQuantize: "fuseLinearStaticQuantizeはbooleanでなければならない",
      packedStaticQuantize: "packedStaticQuantizeはbooleanでなければならない",
    };
    for (const [key, message] of Object.entries(messages)) {
      for (const value of [null, 0, "bogus", [], object]) {
        // 宣言は受理される値で埋めておく — `??` で宣言へ戻す退行はここで通ってしまう。
        assertThrows(
          () => resolveSessionOptions(ALL, FULL, overridesWith(key, value), "test"),
          ModelInputError,
          `test: ${message}`,
        );
      }
    }
    assertEquals(conversions, 0);
  });

  it("受理表が拒否するキーの宣言は取得前に素の Error（明示で隠しても同じ）", () => {
    for (const key of ["attentionCompute", "linearGemvReduce", "packedStaticQuantize"] as const) {
      const declared = { [key]: FULL[key] } satisfies SessionSpec;
      const error = assertThrows(
        () => resolveSessionOptions(LINEAR_ONLY, declared, {}, "test"),
        Error,
        `test: session.${key}は未対応`,
      );
      assert(!(error instanceof ModelInputError), "資産の齟齬が入力起因の型で飛んでいる");
    }
    // 明示指定は宣言の不備を隠さない（受けない宣言は、上書きの有無に依らず資産の齟齬）。
    assertThrows(
      () =>
        resolveSessionOptions(LINEAR_ONLY, { attentionCompute: "a8" }, {
          linearCompute: "f32",
        }, "test"),
      Error,
      "session.attentionComputeは未対応",
    );
  });

  it("受理表が拒否するキーの明示は ModelInputError（黙って捨てない）", () => {
    assertThrows(
      () => resolveSessionOptions(LINEAR_ONLY, {}, { attentionCompute: "f16" }, "test"),
      ModelInputError,
      "test: attentionComputeはこの系列では指定できない",
    );
    // undefined は「指定なし」なので受理表に依らず通る。
    assertEquals(
      resolveSessionOptions(LINEAR_ONLY, {}, { attentionCompute: undefined }, "test"),
      {},
    );
  });

  it("組合せ違反は runtime と同じ判定・文言で、出所で送出型を分ける", () => {
    // 宣言だけで成立 = 資産の齟齬（呼び手が指定を直しても直らない）→ 素の Error。
    const declaredOnly = { attentionCompute: "f16", attentionScoreStorage: "f16" } as const;
    const fromAsset = assertThrows(() => resolveSessionOptions(ALL, declaredOnly, {}, "test"));
    assert(fromAsset instanceof Error && !(fromAsset instanceof ModelInputError));
    assertEquals(fromAsset.message, `test: ${sessionOptionsViolation(declaredOnly)}`);
    // 明示が関与して初めて成立 = 入力起因 → ModelInputError（文言は同じ runtime の 1 本）。
    const fromInput = assertThrows(
      () =>
        resolveSessionOptions(ALL, { attentionScoreStorage: "f16" }, {
          attentionCompute: "f16",
        }, "test"),
      ModelInputError,
    );
    assertEquals(fromInput.message, fromAsset.message);
  });

  it("宣言の違反を明示が解消する形は通す", () => {
    // 宣言だけでは runtime が受けない組合せ（並列 GEMV × a8）でも、実効設定が受理されれば通す
    // （受理集合は実効設定 1 本で決める）。
    const declared = { linearCompute: "a8", linearGemvReduce: "parallel" } as const;
    assertThrows(() => resolveSessionOptions(ALL, declared, {}, "test"), Error, "f32 のみ対応");
    assertEquals(resolveSessionOptions(ALL, declared, { linearCompute: "f32" }, "test"), {
      linearCompute: "f32",
      linearGemvReduce: "parallel",
    });
  });
});

describe("assertSessionOverrides", () => {
  it("宣言 {} を相手にした resolveSessionOptions と同じ判定を下す", () => {
    const cases: readonly SessionOverrides[] = [
      {},
      { linearCompute: "f16" },
      { attentionCompute: "f16", attentionScoreStorage: "f16" },
      { fuseLinearStaticQuantize: true },
      overridesWith("linearCompute", null),
    ];
    const verdict = (run: () => void): string | undefined => {
      try {
        run();
        return undefined;
      } catch (error) {
        assert(error instanceof ModelInputError, "明示指定だけの違反は入力起因");
        return error.message;
      }
    };
    for (const overrides of cases) {
      assertEquals(
        verdict(() => assertSessionOverrides(ALL, overrides, "test")),
        verdict(() => resolveSessionOptions(ALL, {}, overrides, "test")),
      );
    }
  });
});

describe("家族の受理表", () => {
  it("8 家族の表が SessionSpec の全キーを持つ（型の網羅の実行時の対）", () => {
    // 表の型は `Required<SessionSpec>` の網羅なのでキーが増えれば型検査で落ちる。ここは型を
    // 緩めた改変（`Partial` 化など）が型検査を通ってしまう場合の対。
    const policies = {
      anima: ANIMA_SESSION_POLICY,
      birefnet: BIREFNET_SESSION_POLICY,
      "depth-anything": DEPTH_ANYTHING_SESSION_POLICY,
      gemma: GEMMA_SESSION_POLICY,
      irodori: IRODORI_SESSION_POLICY,
      sbv2: SBV2_SESSION_POLICY,
      siglip2: SIGLIP2_SESSION_POLICY,
      "vowel-detector": VOWEL_DETECTOR_SESSION_POLICY,
    };
    for (const [family, policy] of Object.entries(policies)) {
      assertEquals(Object.keys(policy).sort(), Object.keys(FULL).sort(), family);
    }
  });
});
