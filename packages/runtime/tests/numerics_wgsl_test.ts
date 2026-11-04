/**
 * `f32Literal` (src/codegen/numerics-wgsl.ts) emits a value as a WGSL floating-point literal (no
 * GPU needed). The aim is not to let an integer value fall to AbstractInt; appending `.0` to a
 * value in exponent notation would give invalid WGSL such as `1e+21.0`.
 */

import { assertEquals, assertMatch, assertThrows } from "@std/assert";
import { describe, it } from "vite-plus/test";
import { CodegenError } from "../src/codegen/errors.ts";
import { f32Literal } from "../src/codegen/numerics-wgsl.ts";

// The suffix-less form of WGSL's decimal_float_literal (a leading `-` is allowed as the unary
// operator).
const WGSL_FLOAT_LITERAL =
  /^-?(?:[0-9]*\.[0-9]+(?:[eE][+-]?[0-9]+)?|[0-9]+\.[0-9]*(?:[eE][+-]?[0-9]+)?|[0-9]+[eE][+-]?[0-9]+)$/;

describe("f32Literal", () => {
  it("turns an integer value into a decimal literal (does not fall to AbstractInt)", () => {
    assertEquals(f32Literal(2), "2.0");
    assertEquals(f32Literal(-3), "-3.0");
    assertEquals(f32Literal(0.25), "0.25");
  });

  it("values in exponent notation (1e21, 1e-7) also match the WGSL floating-point literal grammar", () => {
    // Only the text is checked. 1.5e300 is an input for checking the text of a 3-digit exponent; it
    // is outside the f32 range, so shader generation would not accept it (the range is not
    // something f32Literal checks).
    for (const value of [1e21, 1e-7, 1.5e300, -2e25]) {
      assertMatch(f32Literal(value), WGSL_FLOAT_LITERAL, String(value));
    }
  });

  it("fails loudly on non-finite values (NaN, ±Infinity), since WGSL has no literal for them", () => {
    for (const value of [Number.NaN, Number.POSITIVE_INFINITY, Number.NEGATIVE_INFINITY]) {
      assertThrows(() => f32Literal(value), CodegenError, undefined, String(value));
    }
  });
});
