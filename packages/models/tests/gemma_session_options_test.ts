import { assertEquals, assertThrows } from "@std/assert";
import { describe, it } from "@std/testing/bdd";
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
      Error,
      "session.linearCompute",
    );
    for (const linearGemvReduce of ["sequential", "parallel-subgroup32"] as const) {
      assertThrows(() => resolve(fast, { linearGemvReduce }, "test"), Error, "parallelが必要");
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
      Error,
      "parallelが必要",
    );
    assertEquals(
      resolve({ fuseLinearStaticQuantize: true }, {
        linearGemvReduce: "parallel",
      }, "test"),
      { linearGemvReduce: "parallel", fuseLinearStaticQuantize: true },
    );
  });
  it("不正な明示値をquantの値で置き換えず、値の文字列変換も呼ばない", () => {
    let conversions = 0;
    const object = {
      toString: () => {
        conversions++;
        return "parallel";
      },
    };
    for (const key of ["linearGemvReduce", "fuseRmsNormAdd", "fuseLinearStaticQuantize"]) {
      for (const value of [null, 0, 1, [], object]) {
        const overrides = {};
        Object.defineProperty(overrides, key, { value });
        assertThrows(() => resolve(fast, overrides, "test"), Error);
      }
    }
    assertEquals(conversions, 0);
  });
});
