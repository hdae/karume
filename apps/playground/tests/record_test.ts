import { createHash } from "node:crypto";
import { wanFrameToRgba } from "@karume/models/wan";
import { describe, expect, it } from "vite-plus/test";

import {
  countNonFinite,
  sha256Hex,
  stackRgbaFrames,
  summarizeDiagnostics,
} from "../src/lib/record.ts";

describe("countNonFinite", () => {
  it("counts NaN and both infinities, not finite extremes", () => {
    const data = Float32Array.of(0, Number.NaN, 1, Infinity, -Infinity, 3.4e38, -0);
    expect(countNonFinite(data)).toBe(3);
  });

  it("is 0 for an all-finite buffer", () => {
    expect(countNonFinite(new Float32Array(16))).toBe(0);
  });
});

describe("frames → RGBA → sha256", () => {
  // A [3, frames = 2, height = 1, width = 2] video in [-1, 1] (channel-major, as Wan returns it).
  const video = {
    frames: 2,
    width: 2,
    height: 1,
    // R plane: frame 0 = (-1, 1), frame 1 = (0, 0.5); G plane: …; B plane: …
    data: Float32Array.of(-1, 1, 0, 0.5, 1, -1, -0.5, 0, 0, 0, 1, 1),
  };
  // clamp(x/2 + 0.5) · 255, rounded: -1 → 0, -0.5 → 64, 0 → 128, 0.5 → 191, 1 → 255.
  const expected = Uint8Array.of(
    // frame 0: pixel 0 (R -1, G 1, B 0), pixel 1 (R 1, G -1, B 0)
    0,
    255,
    128,
    255,
    255,
    0,
    128,
    255,
    // frame 1: pixel 0 (R 0, G -0.5, B 1), pixel 1 (R 0.5, G 0, B 1)
    128,
    64,
    255,
    255,
    191,
    128,
    255,
    255,
  );

  const stack = stackRgbaFrames(video.frames, video.width, video.height, (frame) =>
    wanFrameToRgba(video, frame),
  );

  it("lays the frames out one after another", () => {
    expect(Array.from(stack.bytes)).toEqual(Array.from(expected));
    expect(Array.from(stack.frame(1))).toEqual(Array.from(expected.subarray(8)));
  });

  it("hashes the concatenated RGBA bytes", async () => {
    const reference = createHash("sha256").update(expected).digest("hex");
    expect(await sha256Hex(stack.bytes)).toBe(reference);
  });

  it("rejects a frame of the wrong size", () => {
    expect(() => stackRgbaFrames(1, 2, 2, () => new Uint8ClampedArray(4))).toThrow(/expected 16/);
  });
});

describe("summarizeDiagnostics", () => {
  const report = (allocatedBytes: number, submitCount: number) => ({
    weights: { allocatedBytes },
    submit: { submitCount, dispatchCount: submitCount * 10 },
    geometryProfile: "default",
  });

  it("counts runs and keeps the latest figures", () => {
    const first = summarizeDiagnostics(undefined, report(2 ** 30, 3));
    const second = summarizeDiagnostics(first, report(1.5 * 2 ** 30, 7));
    expect(second).toEqual({
      runs: 2,
      weightsAllocatedGiB: 1.5,
      submitCount: 7,
      dispatchCount: 70,
      geometryProfile: "default",
    });
  });
});
