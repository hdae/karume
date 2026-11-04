/**
 * The value range of the normalisation eps (no GPU needed). The op contract (attrs) and the kernel
 * params both hold the same condition, "finite and rounds to a normal f32 number (2^-126 or
 * above)". If only one side lets a subnormal through, a GPU with FTZ (treating subnormals as 0 —
 * which WGSL allows) runs with eps = 0, and an all-zero row divides by zero. This file pins that
 * both sides accept the same set.
 */

import { assertEquals, assertThrows } from "@std/assert";
import { describe, it } from "vite-plus/test";
import { CodegenError } from "../src/codegen/errors.ts";
import { layerNormParams } from "../src/kernels/layer-norm.ts";
import { rmsNormParams } from "../src/kernels/rms-norm.ts";
import { OpContractError } from "../src/ops/names.ts";
import { LAYER_NORM_ATTRS, RMS_NORM_ATTRS } from "../src/ops/attrs.ts";

const F32_MIN_NORMAL = 2 ** -126;
const F32_MIN_SUBNORMAL = 2 ** -149;
const F32_MAX_SUBNORMAL = 2 ** -126 - 2 ** -149;

const accepts = (check: () => unknown): boolean => {
  try {
    check();
    return true;
  } catch (error) {
    if (error instanceof CodegenError || error instanceof OpContractError) return false;
    throw error;
  }
};

const VALUES: readonly (readonly [string, number])[] = [
  ["smallest subnormal 2^-149", F32_MIN_SUBNORMAL],
  ["largest subnormal", F32_MAX_SUBNORMAL],
  ["1e-50, which rounds to 0 in f32", 1e-50],
  ["smallest normal number 2^-126", F32_MIN_NORMAL],
  ["1e-5", 1e-5],
  ["1e39, which rounds to infinity in f32", 1e39],
];

describe("normalisation params: eps", () => {
  it("rejects a non-zero subnormal (2^-149 and the largest subnormal) for both layer_norm and rms_norm", () => {
    for (const eps of [F32_MIN_SUBNORMAL, F32_MAX_SUBNORMAL]) {
      assertThrows(
        () => layerNormParams(4, 8, eps),
        CodegenError,
        "normal number",
        `layer_norm ${eps}`,
      );
      assertThrows(
        () => rmsNormParams(4, 8, eps),
        CodegenError,
        "normal number",
        `rms_norm ${eps}`,
      );
    }
  });

  it("accepts the smallest normal number 2^-126 (the counterpart showing the check is not too wide)", () => {
    assertEquals(new Float32Array(layerNormParams(4, 8, F32_MIN_NORMAL).buffer)[2], F32_MIN_NORMAL);
    assertEquals(new Float32Array(rmsNormParams(4, 8, F32_MIN_NORMAL).buffer)[2], F32_MIN_NORMAL);
  });

  it("the op contract (attrs) and the params accept the same set", () => {
    for (const [label, eps] of VALUES) {
      const contract = accepts(() => LAYER_NORM_ATTRS.eps(eps, "test"));
      assertEquals(
        accepts(() => layerNormParams(4, 8, eps)),
        contract,
        `layer_norm ${label}`,
      );
      assertEquals(
        accepts(() => RMS_NORM_ATTRS.eps(eps, "test")),
        contract,
        `rms contract ${label}`,
      );
      assertEquals(
        accepts(() => rmsNormParams(4, 8, eps)),
        contract,
        `rms_norm ${label}`,
      );
    }
  });
});
