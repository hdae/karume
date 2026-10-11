/**
 * 正規化の eps の値域（GPU 不要）。op 契約（attrs）とカーネルの params は同じ条件
 * 「有限かつ f32 の正規数（2^-126 以上）に丸まる」を二重に持つ。片側だけが subnormal を通すと、
 * FTZ（subnormal を 0 として扱う — WGSL が許す）の GPU では eps = 0 で走り、全ゼロ行が
 * ゼロ除算になる。ここでは両側の受理集合が一致することを固定する。
 */

import { assertEquals, assertThrows } from "@std/assert";
import { describe, it } from "@std/testing/bdd";
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
  ["最小の subnormal 2^-149", F32_MIN_SUBNORMAL],
  ["最大の subnormal", F32_MAX_SUBNORMAL],
  ["f32 で 0 に丸まる 1e-50", 1e-50],
  ["最小の正規数 2^-126", F32_MIN_NORMAL],
  ["1e-5", 1e-5],
  ["f32 で無限大に丸まる 1e39", 1e39],
];

describe("正規化の params: eps", () => {
  it("非ゼロの subnormal（2^-149 と最大の subnormal）を layer_norm / rms_norm とも拒否する", () => {
    for (const eps of [F32_MIN_SUBNORMAL, F32_MAX_SUBNORMAL]) {
      assertThrows(() => layerNormParams(4, 8, eps), CodegenError, "正規数", `layer_norm ${eps}`);
      assertThrows(() => rmsNormParams(4, 8, eps), CodegenError, "正規数", `rms_norm ${eps}`);
    }
  });

  it("最小の正規数 2^-126 は受理する（門を広げすぎていない対）", () => {
    assertEquals(new Float32Array(layerNormParams(4, 8, F32_MIN_NORMAL).buffer)[2], F32_MIN_NORMAL);
    assertEquals(new Float32Array(rmsNormParams(4, 8, F32_MIN_NORMAL).buffer)[2], F32_MIN_NORMAL);
  });

  it("op 契約（attrs）と params の受理集合が一致する", () => {
    for (const [label, eps] of VALUES) {
      const contract = accepts(() => LAYER_NORM_ATTRS.eps(eps, "test"));
      assertEquals(accepts(() => layerNormParams(4, 8, eps)), contract, `layer_norm ${label}`);
      assertEquals(accepts(() => RMS_NORM_ATTRS.eps(eps, "test")), contract, `rms 契約 ${label}`);
      assertEquals(accepts(() => rmsNormParams(4, 8, eps)), contract, `rms_norm ${label}`);
    }
  });
});
