// nvidia-smi の標本化の補助（helpers/nvidia-smi.ts）の読み方の門 — 子 process も GPU も使わない純関数だけを撃つ。
// 縛るのは「読めない行を標本に混ぜない（NaN・0 MiB に化けさせない）」ことと、「標本 0 と読めない行を記録で
// 見分けられる」ことの 2 点（ADR 0124 の段 4b の記録が、古いドライバの機で黙って空の基準を名乗らないため）。

import { assertEquals } from "@std/assert";
import { describe, it } from "@std/testing/bdd";
import { formatNvidiaTimeline, parseNvidiaSample } from "./helpers/nvidia-smi.ts";

describe("parseNvidiaSample", () => {
  it("3 欄の行を使用量・温度・理由のビット列として読む", () => {
    assertEquals(parseNvidiaSample("1234, 56, 0x0000000000000004"), {
      memoryMiB: 1234,
      temperatureC: 56,
      reasons: 4n,
    });
  });

  it("理由が 0x の綴りでない行（[Not Supported]）は理由だけを欠いた標本にする", () => {
    assertEquals(parseNvidiaSample("1234, 56, [Not Supported]"), {
      memoryMiB: 1234,
      temperatureC: 56,
      reasons: undefined,
    });
  });

  it("3 欄に割れない行と、使用量・温度が数でない行はエラー行（undefined）", () => {
    for (
      const line of [
        "Failed to initialize NVML: Driver/library version mismatch",
        "1234, 56",
        "1234, 56, 0x0, 7",
        "[N/A], 56, 0x0",
        ", 56, 0x0",
      ]
    ) {
      assertEquals(parseNvidiaSample(line), undefined, line);
    }
  });
});

describe("formatNvidiaTimeline", () => {
  it("標本 0 の区間は「標本 0」と書き、読めない行は本数と異なる綴りを残す", () => {
    const text = formatNvidiaTimeline({
      phases: new Map([
        ["start", { peakMemoryMiB: 0, maxTemperatureC: 0, eventReasons: "0x0", samples: 0 }],
        ["空き", { peakMemoryMiB: 0, maxTemperatureC: 0, eventReasons: "0x0", samples: 0 }],
      ]),
      errorLines: ["Failed to initialize NVML", "Failed to initialize NVML"],
    });
    assertEquals(text, "[空き] 標本 0（観測なし） / 読めない行 2 本（Failed to initialize NVML）");
  });

  it("nvidia-smi の無い機は観測なしと書く", () => {
    assertEquals(
      formatNvidiaTimeline(undefined),
      "nvidia-smi なし（GPU 全体の VRAM は観測していない）",
    );
  });
});
