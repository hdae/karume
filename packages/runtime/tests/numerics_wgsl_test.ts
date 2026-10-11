/**
 * `f32Literal`（src/codegen/numerics-wgsl.ts）が値を WGSL の浮動小数リテラルとして出すこと
 * （GPU 不要）。整数値を AbstractInt に落とさないのが目的で、指数表記の値に `.0` を足すと
 * `1e+21.0` のような不正な WGSL になる。
 */

import { assertEquals, assertMatch, assertThrows } from "@std/assert";
import { describe, it } from "@std/testing/bdd";
import { CodegenError } from "../src/codegen/errors.ts";
import { f32Literal } from "../src/codegen/numerics-wgsl.ts";

// WGSL の decimal_float_literal のうち接尾辞の無い形（先頭の `-` は単項演算子として許す）。
const WGSL_FLOAT_LITERAL =
  /^-?(?:[0-9]*\.[0-9]+(?:[eE][+-]?[0-9]+)?|[0-9]+\.[0-9]*(?:[eE][+-]?[0-9]+)?|[0-9]+[eE][+-]?[0-9]+)$/;

describe("f32Literal", () => {
  it("整数値は小数リテラルにする（AbstractInt に落とさない）", () => {
    assertEquals(f32Literal(2), "2.0");
    assertEquals(f32Literal(-3), "-3.0");
    assertEquals(f32Literal(0.25), "0.25");
  });

  it("指数表記になる値（1e21・1e-7）も正しい浮動小数リテラルを返す", () => {
    for (const value of [1e21, 1e-7, 1.5e300, -2e25]) {
      assertMatch(f32Literal(value), WGSL_FLOAT_LITERAL, String(value));
    }
  });

  it("非有限値（NaN・±Infinity）は WGSL にリテラルが無いので fail loudly", () => {
    for (const value of [Number.NaN, Number.POSITIVE_INFINITY, Number.NEGATIVE_INFINITY]) {
      assertThrows(() => f32Literal(value), CodegenError, undefined, String(value));
    }
  });
});
