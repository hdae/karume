import { describe, expect, it } from "vite-plus/test";

import { familyOfKey, type OpFamily } from "../src/models/gemma4-bench/family.ts";

/**
 * Keys as the runtime's builders print them (packages/runtime/src/kernels and codegen, at the
 * presets the Gemma 4 decode runs: i4 g32 / i8 storage, the parallel GEMV, the states attention).
 */
const CASES: readonly (readonly [string, OpFamily])[] = [
  ["attention_state_qk:v1:f32:wg16x16:par:sliding:gqa", "attention-core"],
  ["attention_state_qk:v1:f32:wg16x16:par:gqa", "attention-core"],
  ["attention_state_stats:v2:f32:wg256", "attention-core"],
  ["attention_state_pv:v2:f32:wg16x16:par:gqa", "attention-core"],
  ["attention_state_stats_pv:v2:f32:wg16x16:sliding:gqa", "attention-core"],
  ["state_append:v1:f32:wg256:sliding", "attention-other"],
  ["rope:v1:half:bshd:f32:wg256", "attention-other"],
  ["strided:v1:f32:r4:wg256", "attention-other"],
  ["linear_gemv:v1:f32:c32u4:wi4g32", "linear"],
  ["linear_gemv_parallel:wi4g32:l32", "linear"],
  ["linear_gemv_parallel:wi8:l16", "linear"],
  ["linear_gemv_parallel:wi4g32:l32:static-quantize:v2", "linear"],
  ["linear_gemv_parallel:wi4g32:l32:staged-x", "linear"],
  ["linear_gemv_parallel:wi8:l32:packed-x-i8", "linear"],
  ["linear_gemv_integer:v1:wi2:l8r4", "linear"],
  ["linear_gemv_integer:v1:wi2:l16r4:static-quantize:v2", "linear"],
  ["linear:v2:f32:reg128x128r8x8w16v4:wi4g32", "linear"],
  ["rms_norm:v1:f32:lastdim:wg256", "norm"],
  ["rms_norm_add:v2:rms_norm:v2:f32:lastdim:wg256:trip6:norm-residual:xor-round", "norm"],
  ["rms_norm_rope:v1:pair:bshd:f32:wg256", "norm"],
  ["ew:v3:gelu_tanh:f32>f32:r3:wg256", "elementwise"],
  ["ew:v3:mul:f32>f32:r4:wg256", "elementwise"],
  ["static_quantize:v2:f32:wg128", "static-quantize"],
  ["static_quantize:v2:packed-i8:wg128", "static-quantize"],
  ["gelu_mul_static_quantize:v1:gelu-other:f32:wg128", "static-quantize"],
  ["embedding:v1:f32:i32:wg256:wi8", "embedding"],
  ["gather:v1:f32:i32:lastdim:wg256", "embedding"],
  ["argmax:v2:f32>i32:lastdim:minindex:split-partial:wg256e16", "other"],
];

describe("the op family of a pipeline key", () => {
  it.each(CASES)("puts %s in %s", (key, family) => {
    expect(familyOfKey(key)).toBe(family);
  });

  it("puts an unknown kernel, an empty key and a name that only looks like a family in other", () => {
    expect(familyOfKey("conv2d:v1:f32:direct:wg256")).toBe("other");
    expect(familyOfKey("")).toBe("other");
    expect(familyOfKey("linearize:v1")).toBe("other");
    expect(familyOfKey("toString")).toBe("other");
  });
});
