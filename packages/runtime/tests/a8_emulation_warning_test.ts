// The warning when int8 activations (a8) are requested on a device without dp4a (ADR 0128 decision
// 5 — a warning, neither a fallback nor an automatic sibling preset). The decision is a pure
// function (src/runtime/a8-emulation-warning.ts); "once per device" is handled by GpuContext's
// `warnOnce`. Both are pinned here without a GPU. That a real GPU emits the warning from Session
// construction (and does not on a device with dp4a) is covered by the "device without dp4a"
// section of tests/gpu_i8a8_test.ts.

import { describe, it } from "vite-plus/test";
import { assert, assertEquals, assertStringIncludes } from "@std/assert";
import { RUNTIME_INTERNAL } from "../src/gpu/device.ts";
import { DP4A_WGSL_FEATURE } from "../src/kernels/linear-i8a8.ts";
import {
  A8_EMULATION_WARNING_KEY,
  a8EmulationWarning,
} from "../src/runtime/a8-emulation-warning.ts";
import { fakeDevice, fakeGpuContext } from "./helpers/fake-gpu.ts";

/** Replaces `console.warn`, runs `body` and always restores it (returns the emitted text as is). */
const captureWarnings = (body: () => void): string[] => {
  const original = console.warn;
  const warnings: string[] = [];
  console.warn = (...args: unknown[]) => {
    warnings.push(args.map((arg) => String(arg)).join(" "));
  };
  try {
    body();
  } finally {
    console.warn = original;
  }
  return warnings;
};

const WITHOUT_DP4A: ReadonlySet<string> = new Set();
const WITH_DP4A: ReadonlySet<string> = new Set([DP4A_WGSL_FEATURE]);

describe("a8EmulationWarning (a8 requested but the device has no dp4a)", () => {
  it("on a device without dp4a that requests a8 only for linear, the warning names linearCompute", () => {
    const message = a8EmulationWarning("a8", "f32", "f32", WITHOUT_DP4A);
    assert(message !== undefined);
    assertStringIncludes(message, "@karume/runtime");
    assertStringIncludes(message, DP4A_WGSL_FEATURE);
    assertStringIncludes(message, 'linearCompute: "a8"');
    assert(
      !message.includes("attentionCompute: "),
      "does not name attention, which was not requested",
    );
    // The text says what makes the warning go away (the two ways out)
    assertStringIncludes(message, "pass quant explicitly");
    assertStringIncludes(message, "'f32'");
  });

  it("requesting both linear and attention names both", () => {
    const message = a8EmulationWarning("a8", "f32", "a8", WITHOUT_DP4A);
    assert(message !== undefined);
    assertStringIncludes(message, 'linearCompute / attentionCompute: "a8"');
  });

  it("warns for a8 on conv alone and names convCompute", () => {
    const message = a8EmulationWarning("f32", "a8", "f32", WITHOUT_DP4A);
    assert(message !== undefined);
    assertStringIncludes(message, 'convCompute: "a8"');
    assert(!message.includes("linearCompute: "), "does not name linear, which was not requested");
  });

  it("warns for a8 on attention alone", () => {
    const message = a8EmulationWarning("f32", "f32", "a8", WITHOUT_DP4A);
    assert(message !== undefined);
    assertStringIncludes(message, 'attentionCompute: "a8"');
  });

  it("does not warn on a device with dp4a even when a8 is requested", () => {
    assertEquals(a8EmulationWarning("a8", "f32", "a8", WITH_DP4A), undefined);
  });

  it("does not warn on a device without dp4a when a8 is not requested (f32 / f16 compute is unaffected)", () => {
    assertEquals(a8EmulationWarning("f32", "f32", "f32", WITHOUT_DP4A), undefined);
    assertEquals(a8EmulationWarning("f16", "f32", "f16", WITHOUT_DP4A), undefined);
  });
});

describe("GpuContext.warnOnce (once per device)", () => {
  it("the same key is emitted only once however many times it is called", () => {
    const gpu = fakeGpuContext(fakeDevice());
    const warnings = captureWarnings(() => {
      gpu[RUNTIME_INTERNAL].warnOnce(A8_EMULATION_WARNING_KEY, "1st");
      gpu[RUNTIME_INTERNAL].warnOnce(A8_EMULATION_WARNING_KEY, "2nd (not emitted)");
      gpu[RUNTIME_INTERNAL].warnOnce(A8_EMULATION_WARNING_KEY, "3rd (not emitted)");
    });
    assertEquals(warnings, ["1st"]);
  });

  it("a different key is emitted as a separate fact", () => {
    const gpu = fakeGpuContext(fakeDevice());
    const warnings = captureWarnings(() => {
      gpu[RUNTIME_INTERNAL].warnOnce("a", "A");
      gpu[RUNTIME_INTERNAL].warnOnce("b", "B");
      gpu[RUNTIME_INTERNAL].warnOnce("a", "A (not emitted)");
    });
    assertEquals(warnings, ["A", "B"]);
  });

  it("a different GpuContext (another device) emits once again", () => {
    const first = fakeGpuContext(fakeDevice());
    const second = fakeGpuContext(fakeDevice());
    const warnings = captureWarnings(() => {
      first[RUNTIME_INTERNAL].warnOnce(A8_EMULATION_WARNING_KEY, "device 1");
      second[RUNTIME_INTERNAL].warnOnce(A8_EMULATION_WARNING_KEY, "device 2");
      first[RUNTIME_INTERNAL].warnOnce(A8_EMULATION_WARNING_KEY, "device 1 (not emitted)");
    });
    assertEquals(warnings, ["device 1", "device 2"]);
  });
});
