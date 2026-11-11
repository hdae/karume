import {
  FUSION_TWIN_CASES,
  GEMV_INTEGER_CASES,
  GEMV_PLANES_CASES,
  STATE_KV16_CASES,
} from "@karume/kernel-checks";
import { describe, expect, it } from "vite-plus/test";

import { KERNEL_CHECKS } from "../src/models/kernel-check/checks.ts";
import { KERNEL_CHECK_IDS } from "../src/models/kernel-check/verdict.ts";

describe("the kernel-check page's checks", () => {
  it("has one check per id, in the order the verdicts and the record list them", () => {
    expect(KERNEL_CHECKS.map(({ id }) => id)).toEqual([...KERNEL_CHECK_IDS]);
  });

  it("runs the integer GEMV check on any device, against the case count the check exports", () => {
    const check = KERNEL_CHECKS.find(({ id }) => id === "gemv-integer");
    expect(check?.title).toBe("Integer GEMV");
    expect(check?.needsSubgroups).toBe(false);
    expect(GEMV_INTEGER_CASES).toEqual({ "gemv-integer": 692 });
  });

  it("runs the byte-plane GEMV check on any device, against the case count the check exports", () => {
    const check = KERNEL_CHECKS.find(({ id }) => id === "gemv-planes");
    expect(check?.title).toBe("Byte-plane GEMV");
    expect(check?.needsSubgroups).toBe(false);
    expect(GEMV_PLANES_CASES).toEqual({ "gemv-planes": 240 });
  });

  it("runs the f16 full slot check on any device, against the case counts the check exports", () => {
    const check = KERNEL_CHECKS.find(({ id }) => id === "state-kv16");
    expect(check?.title).toBe("f16 full state slots");
    expect(check?.needsSubgroups).toBe(false);
    expect(Object.keys(STATE_KV16_CASES)).toEqual(["state-kv16-readers", "state-kv16-append"]);
  });

  it("runs the fusion twins on any device, one part per rule in a fixed order", () => {
    const check = KERNEL_CHECKS.find(({ id }) => id === "fusion-twins");
    expect(check?.title).toBe("Fusion rules: fused vs unfused");
    expect(check?.needsSubgroups).toBe(false);
    expect(Object.keys(FUSION_TWIN_CASES)).toEqual([
      "fusion-gated-residual",
      "fusion-silu",
      "fusion-adaln",
      "fusion-rope",
      "fusion-upsample2x",
      "fusion-rms-norm-rope",
      "fusion-linear-prologue",
      "fusion-attention-permute",
      "fusion-attention-permute-sg32",
      "fusion-row-block-attention",
      "fusion-rms-norm-add",
      "fusion-gelu-mul-static-quantize",
      "fusion-linear-static-quantize",
    ]);
  });
});
