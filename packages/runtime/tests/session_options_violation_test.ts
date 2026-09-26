// `sessionOptionsViolation`（SessionOptions の GPU 非依存の受理条件）の門 — GPU に触れない純関数
// なので、アダプタ無し環境でも回帰を撃てる位置に置く。
//
// Session 構築はこの関数の違反を `ExecutionError` として送出し、models 側の合成（明示 > 宣言 >
// runtime 既定）は同じ関数で重みの取得前に落とす。ここが縛るのは「構築と同じ判定・同じ文言を
// 返す」ことと「GPU の能力に依る条件を混ぜない」ことの 2 点（混ぜると、device を見ずに判定する
// 呼び手が shader-f16 を持つ device でも f16 を拒否する）。

import { assert, assertEquals } from "@std/assert";
import { describe, it } from "@std/testing/bdd";
import { type SessionOptions, sessionOptionsViolation } from "../mod.ts";

/** 型の外から来る呼び手（JS の消費者）の欄を再現する — TS の型はこの経路を守らない。 */
const withValue = (key: string, value: unknown): SessionOptions => {
  const options: SessionOptions = {};
  Object.defineProperty(options, key, { value, enumerable: true });
  return options;
};

describe("sessionOptionsViolation", () => {
  it("既定・受理される組合せでは undefined を返す", () => {
    assertEquals(sessionOptionsViolation({}), undefined);
    assertEquals(
      sessionOptionsViolation({
        linearGemvReduce: "parallel",
        fuseRmsNormAdd: true,
        fuseLinearStaticQuantize: true,
        packedStaticQuantize: true,
      }),
      undefined,
    );
    assertEquals(
      sessionOptionsViolation({
        linearCompute: "a8",
        attentionCompute: "a8",
        attentionScoreStorage: "f16",
      }),
      undefined,
    );
  });

  it("GPU の能力に依る条件（shader-f16 / subgroups）では落とさない", () => {
    // f16 計算・subgroup 変種は device を見ないと決まらない — 構築側の門の担当。
    assertEquals(sessionOptionsViolation({ linearCompute: "f16" }), undefined);
    assertEquals(sessionOptionsViolation({ attentionCompute: "f16" }), undefined);
    assertEquals(sessionOptionsViolation({ linearGemvReduce: "parallel-subgroup32" }), undefined);
    assertEquals(sessionOptionsViolation({ rmsNormReduce: "subgroup32" }), undefined);
  });

  it("組合せ違反は構築と同じ文言で返す", () => {
    const cases: readonly (readonly [SessionOptions, string])[] = [
      [
        { fuseLinearStaticQuantize: true },
        "fuseLinearStaticQuantize は linearGemvReduce: parallel / linearCompute: f32 のみ対応",
      ],
      [
        { linearGemvReduce: "parallel", linearCompute: "a8", fuseLinearStaticQuantize: true },
        "fuseLinearStaticQuantize は linearGemvReduce: parallel / linearCompute: f32 のみ対応",
      ],
      [
        { packedStaticQuantize: true, linearGemvReduce: "sequential" },
        "packedStaticQuantize は linearGemvReduce: parallel / linearCompute: f32 のみ対応",
      ],
      [
        { linearGemvReduce: "parallel", linearCompute: "a8" },
        "linearGemvReduce: parallel は linearCompute: f32 のみ対応",
      ],
    ];
    for (const [options, message] of cases) {
      assertEquals(sessionOptionsViolation(options), message);
    }
    const s16c16 = sessionOptionsViolation({
      attentionCompute: "f16",
      attentionScoreStorage: "f16",
    });
    assert(s16c16?.startsWith("attentionScoreStorage 'f16' と attentionCompute 'f16' は同時に"));
  });

  it("型・綴り・値域の違反を返し、診断で利用者の変換を呼ばない", () => {
    let conversions = 0;
    const object = {
      [Symbol.toPrimitive](): string {
        conversions++;
        return "parallel";
      },
    };
    assertEquals(
      sessionOptionsViolation(withValue("fuseRmsNormAdd", 1)),
      "options.fuseRmsNormAdd はbooleanでなければならない",
    );
    assert(
      sessionOptionsViolation(withValue("linearGemvReduce", object))?.includes(
        "linearGemvReduce: object",
      ),
    );
    assert(
      sessionOptionsViolation(withValue("linearCompute", "i8a8"))?.includes(
        'linearCompute: "i8a8"',
      ),
    );
    assert(
      sessionOptionsViolation({ planBackingBudgetBytes: -1 })?.includes("非負の安全な整数"),
    );
    assert(
      sessionOptionsViolation({ linearGemvRowsThreadTarget: 0 })?.includes("1 以上の安全な整数"),
    );
    assertEquals(conversions, 0);
  });

  it("省略と同じく null も runtime 既定として読む（構築と同じ読み方）", () => {
    // 「欄はあるが値が不正」を既定へ戻さず拒否するのは、明示指定を受ける上位層の責務
    // （@karume/models の resolveSessionOptions）。ここで null を拒否に変えると構築の受理集合が
    // 動く — その変更はこのテストで見える。
    assertEquals(sessionOptionsViolation(withValue("linearCompute", null)), undefined);
  });
});
