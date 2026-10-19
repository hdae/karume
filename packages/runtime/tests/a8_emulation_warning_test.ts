// dp4a の無い device で int8 活性（a8）を要求したときの警告（ADR 0128 決定 5 — フォールバックでも
// 自動の兄弟プリセットでもなく、警告）。判定は純関数（src/runtime/a8-emulation-warning.ts）、
// 「device ごとに 1 回」は GpuContext の `warnOnce` が担う。どちらも GPU 無しで固定する。
// 実 GPU で Session 構築から警告が出ること（と dp4a のある device では出ないこと）は
// tests/gpu_i8a8_test.ts の「dp4a の無い device」の節が持つ。

import { describe, it } from "vite-plus/test";
import { assert, assertEquals, assertStringIncludes } from "@std/assert";
import { RUNTIME_INTERNAL } from "../src/gpu/device.ts";
import { DP4A_WGSL_FEATURE } from "../src/kernels/linear-i8a8.ts";
import {
  A8_EMULATION_WARNING_KEY,
  a8EmulationWarning,
} from "../src/runtime/a8-emulation-warning.ts";
import { fakeDevice, fakeGpuContext } from "./helpers/fake-gpu.ts";

/** `console.warn` を差し替えて `body` を走らせ、必ず元へ戻す（出た文言をそのまま返す）。 */
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

describe("a8EmulationWarning（a8 を要求したが device に dp4a が無い）", () => {
  it("linear の a8 だけを要求した dp4a 無しの device では、linearCompute を名指しして警告する", () => {
    const message = a8EmulationWarning("a8", "f32", WITHOUT_DP4A);
    assert(message !== undefined);
    assertStringIncludes(message, "@karume/runtime");
    assertStringIncludes(message, DP4A_WGSL_FEATURE);
    assertStringIncludes(message, 'linearCompute: "a8"');
    assert(!message.includes("attentionCompute: "), "要求していない attention を名指ししない");
    // 何をすれば消えるか（逃げ先 2 つ）を文言が持つ
    assertStringIncludes(message, "pass quant explicitly");
    assertStringIncludes(message, "'f32'");
  });

  it("linear と attention の両方を要求すれば両方を名指しする", () => {
    const message = a8EmulationWarning("a8", "a8", WITHOUT_DP4A);
    assert(message !== undefined);
    assertStringIncludes(message, 'linearCompute / attentionCompute: "a8"');
  });

  it("attention の a8 だけでも警告する", () => {
    const message = a8EmulationWarning("f32", "a8", WITHOUT_DP4A);
    assert(message !== undefined);
    assertStringIncludes(message, 'attentionCompute: "a8"');
  });

  it("dp4a のある device では a8 を要求しても警告しない", () => {
    assertEquals(a8EmulationWarning("a8", "a8", WITH_DP4A), undefined);
  });

  it("a8 を要求しなければ dp4a の無い device でも警告しない（f32 / f16 の計算は無関係）", () => {
    assertEquals(a8EmulationWarning("f32", "f32", WITHOUT_DP4A), undefined);
    assertEquals(a8EmulationWarning("f16", "f16", WITHOUT_DP4A), undefined);
  });
});

describe("GpuContext.warnOnce（device ごとに 1 回）", () => {
  it("同じキーは何度呼んでも 1 回しか出ない", () => {
    const gpu = fakeGpuContext(fakeDevice());
    const warnings = captureWarnings(() => {
      gpu[RUNTIME_INTERNAL].warnOnce(A8_EMULATION_WARNING_KEY, "1 回目");
      gpu[RUNTIME_INTERNAL].warnOnce(A8_EMULATION_WARNING_KEY, "2 回目（出ない）");
      gpu[RUNTIME_INTERNAL].warnOnce(A8_EMULATION_WARNING_KEY, "3 回目（出ない）");
    });
    assertEquals(warnings, ["1 回目"]);
  });

  it("キーが違えば別の事実として出る", () => {
    const gpu = fakeGpuContext(fakeDevice());
    const warnings = captureWarnings(() => {
      gpu[RUNTIME_INTERNAL].warnOnce("a", "A");
      gpu[RUNTIME_INTERNAL].warnOnce("b", "B");
      gpu[RUNTIME_INTERNAL].warnOnce("a", "A（出ない）");
    });
    assertEquals(warnings, ["A", "B"]);
  });

  it("別の GpuContext（別 device）では改めて 1 回出る", () => {
    const first = fakeGpuContext(fakeDevice());
    const second = fakeGpuContext(fakeDevice());
    const warnings = captureWarnings(() => {
      first[RUNTIME_INTERNAL].warnOnce(A8_EMULATION_WARNING_KEY, "device 1");
      second[RUNTIME_INTERNAL].warnOnce(A8_EMULATION_WARNING_KEY, "device 2");
      first[RUNTIME_INTERNAL].warnOnce(A8_EMULATION_WARNING_KEY, "device 1（出ない）");
    });
    assertEquals(warnings, ["device 1", "device 2"]);
  });
});
