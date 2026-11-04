// Checks for how the nvidia-smi sampling helper (helpers/nvidia-smi.ts) parses — exercises only pure functions, with no
// child process and no GPU. Two things are pinned: unreadable lines are never mixed into the samples (never turned into
// NaN or 0 MiB), and the record tells zero samples apart from unreadable lines (so that the record of ADR 0124 step 4b
// does not silently present an empty baseline on a machine with an old driver).

import { assertEquals } from "@std/assert";
import { describe, it } from "vite-plus/test";
import { formatNvidiaTimeline, parseNvidiaSample } from "./helpers/nvidia-smi.ts";

describe("parseNvidiaSample", () => {
  it("reads a three-field line as memory used, temperature and the reasons bit field", () => {
    assertEquals(parseNvidiaSample("1234, 56, 0x0000000000000004"), {
      memoryMiB: 1234,
      temperatureC: 56,
      reasons: 4n,
    });
  });

  it("a line whose reasons are not spelled 0x… ([Not Supported]) becomes a sample without reasons", () => {
    assertEquals(parseNvidiaSample("1234, 56, [Not Supported]"), {
      memoryMiB: 1234,
      temperatureC: 56,
      reasons: undefined,
    });
  });

  it("a line that does not split into three fields, or whose memory or temperature is not a number, is an error line (undefined)", () => {
    for (const line of [
      "Failed to initialize NVML: Driver/library version mismatch",
      "1234, 56",
      "1234, 56, 0x0, 7",
      "[N/A], 56, 0x0",
      ", 56, 0x0",
    ]) {
      assertEquals(parseNvidiaSample(line), undefined, line);
    }
  });
});

describe("formatNvidiaTimeline", () => {
  it("writes a phase with zero samples as zero samples, and records the count and distinct spellings of unreadable lines", () => {
    const text = formatNvidiaTimeline({
      phases: new Map([
        ["start", { peakMemoryMiB: 0, maxTemperatureC: 0, eventReasons: "0x0", samples: 0 }],
        ["空き", { peakMemoryMiB: 0, maxTemperatureC: 0, eventReasons: "0x0", samples: 0 }],
      ]),
      errorLines: ["Failed to initialize NVML", "Failed to initialize NVML"],
    });
    assertEquals(text, "[空き] 標本 0（観測なし） / 読めない行 2 本（Failed to initialize NVML）");
  });

  it("writes 'not observed' on a machine without nvidia-smi", () => {
    assertEquals(
      formatNvidiaTimeline(undefined),
      "nvidia-smi なし（GPU 全体の VRAM は観測していない）",
    );
  });
});
