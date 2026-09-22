import { assertEquals, assertThrows } from "@std/assert";
import { describe, it } from "@std/testing/bdd";
import type { SessionSpec } from "@karume/hub";
import { ModelInputError } from "../src/errors.ts";
import { resolveGemmaSessionOptions as resolve } from "../src/gemma/session-options.ts";

describe("Gemmaのquant実行設定", () => {
  const fast = {
    linearGemvReduce: "parallel",
    fuseRmsNormAdd: true,
    fuseLinearStaticQuantize: true,
  } as const;
  it("未指定・quant宣言・明示上書きを区別する", () => {
    assertEquals(resolve({}, {}, "test"), {});
    assertEquals(resolve(fast, {}, "test"), fast);
    assertEquals(resolve(fast, { fuseRmsNormAdd: undefined }, "test"), fast);
    assertEquals(resolve(fast, { fuseRmsNormAdd: false }, "test"), {
      ...fast,
      fuseRmsNormAdd: false,
    });
    assertEquals(resolve(fast, { fuseLinearStaticQuantize: false }, "test"), {
      ...fast,
      fuseLinearStaticQuantize: false,
    });
    const reference = {
      linearGemvReduce: "sequential",
      fuseRmsNormAdd: false,
      fuseLinearStaticQuantize: false,
    } as const;
    assertEquals(resolve(fast, reference, "test"), reference);
    assertEquals(resolve({ fuseRmsNormAdd: false }, { fuseRmsNormAdd: true }, "test"), {
      fuseRmsNormAdd: true,
    });
    assertEquals(resolve({}, { linearGemvReduce: "parallel-subgroup32" }, "test"), {
      linearGemvReduce: "parallel-subgroup32",
    });
  });
  it("未対応の宣言を上書きで隠さず、実効設定の不正な組合せを拒否する", () => {
    assertThrows(
      () => resolve({ linearCompute: "f32" }, {}, "test"),
      ModelInputError,
      "session.linearCompute",
    );
    for (const linearGemvReduce of ["sequential", "parallel-subgroup32"] as const) {
      assertThrows(
        () => resolve(fast, { linearGemvReduce }, "test"),
        ModelInputError,
        "parallelが必要",
      );
      assertEquals(
        resolve(fast, {
          linearGemvReduce,
          fuseLinearStaticQuantize: false,
        }, "test"),
        { ...fast, linearGemvReduce, fuseLinearStaticQuantize: false },
      );
    }
    assertThrows(
      () => resolve({}, { fuseLinearStaticQuantize: true }, "test"),
      ModelInputError,
      "parallelが必要",
    );
    assertEquals(
      resolve({ fuseLinearStaticQuantize: true }, {
        linearGemvReduce: "parallel",
      }, "test"),
      { linearGemvReduce: "parallel", fuseLinearStaticQuantize: true },
    );
  });
  it("packedStaticQuantizeはquant宣言から入り、明示falseが勝ち、parallel以外を拒否する", () => {
    // ADR 0105 追記 2（語彙への昇格）— 実配布のi4-fastと同じ4欄が宣言から素通りする。
    const packedFast = { ...fast, packedStaticQuantize: true } as const;
    assertEquals(resolve(packedFast, {}, "test"), packedFast);
    assertEquals(resolve(packedFast, { packedStaticQuantize: false }, "test"), {
      ...packedFast,
      packedStaticQuantize: false,
    });
    assertEquals(resolve(fast, { packedStaticQuantize: true }, "test"), packedFast);
    // parallelを伴わない宣言はquant由来でも拒否する（i4-fastはparallelを宣言する）。
    assertThrows(
      () => resolve({ packedStaticQuantize: true }, {}, "test"),
      ModelInputError,
      "packedStaticQuantizeはlinearGemvReduce: parallelが必要",
    );
    assertThrows(
      () => resolve({}, { packedStaticQuantize: true }, "test"),
      ModelInputError,
      "packedStaticQuantizeはlinearGemvReduce: parallelが必要",
    );
    assertThrows(
      () =>
        resolve(fast, {
          linearGemvReduce: "sequential",
          fuseLinearStaticQuantize: false,
          packedStaticQuantize: true,
        }, "test"),
      ModelInputError,
      "packedStaticQuantizeはlinearGemvReduce: parallelが必要",
    );
  });
  it("SessionSpecの全キーが許可表か拒否パスのどちらかに現れる", () => {
    // hubのSESSION_KEYSとmodelsのWRITERSは`Required<SessionSpec>`の網羅表なので、ノブが
    // 増えれば型検査が落ちる。Gemmaだけは手書きの文字列比較4本（ADR 0104の3欄 +
    // ADR 0105追記2のpacked）なので型検査が落ちず、追随漏れは実行時の拒否でしか
    // 見えない。ここは全キーを埋めた
    // spec（キーが増えればこの宣言が型検査で落ちる）から、1キーずつ通る／落ちるを確かめる。
    const full: Required<SessionSpec> = {
      linearCompute: "f16",
      attentionCompute: "f16",
      attentionScoreStorage: "f16",
      linearGemvReduce: "sequential",
      fuseRmsNormAdd: false,
      fuseLinearStaticQuantize: false,
      packedStaticQuantize: false,
    };
    const accepted = [
      "linearGemvReduce",
      "fuseRmsNormAdd",
      "fuseLinearStaticQuantize",
      "packedStaticQuantize",
    ];
    const rejected: string[] = [];
    for (const key of Object.keys(full) as (keyof typeof full)[]) {
      const spec = { [key]: full[key] } satisfies SessionSpec;
      if (accepted.includes(key)) {
        assertEquals(resolve(spec, {}, "test"), spec);
        continue;
      }
      assertThrows(() => resolve(spec, {}, "test"), ModelInputError, `session.${key}は未対応`);
      rejected.push(key);
    }
    assertEquals([...accepted, ...rejected].sort(), Object.keys(full).sort());
  });
  it("不正な明示値をquantの値で置き換えず、値の文字列変換も呼ばない", () => {
    let conversions = 0;
    const object = {
      toString: () => {
        conversions++;
        return "parallel";
      },
    };
    // 期待メッセージまで縛るのは、型検査ではなく組合せ検査で落ちる「理由のすり替わり」を
    // 通さないため（resolveは4種類の異なる理由でthrowする）。
    const invalid = [
      ["linearGemvReduce", "linearGemvReduceが不正"],
      ["fuseRmsNormAdd", "fuseRmsNormAddはbooleanでなければならない"],
      ["fuseLinearStaticQuantize", "fuseLinearStaticQuantizeはbooleanでなければならない"],
      ["packedStaticQuantize", "packedStaticQuantizeはbooleanでなければならない"],
    ] as const;
    for (const [key, message] of invalid) {
      for (const value of [null, 0, 1, [], object]) {
        const overrides = {};
        // 実呼び出しが渡すoptionsは列挙可能なプロパティしか持たないので、同じ形で検査する。
        Object.defineProperty(overrides, key, { value, enumerable: true });
        assertThrows(() => resolve(fast, overrides, "test"), ModelInputError, message);
      }
    }
    assertEquals(conversions, 0);
  });
});
