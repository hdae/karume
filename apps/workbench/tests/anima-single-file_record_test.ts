import { describe, expect, it } from "vite-plus/test";

import {
  ANIMA_SINGLE_FILE_FORMAT,
  animaSingleFileData,
  animaSingleFileSummary,
  type ArmRecord,
} from "../src/anima-single-file/record.ts";

const arm = (name: ArmRecord["name"], hashes: readonly string[]): ArmRecord => ({
  name,
  buildSeconds: 12.5,
  generations: hashes.map((pixelSha256, index) => ({ seconds: 20 + index, pixelSha256 })),
});

const fields = (arms: readonly ArmRecord[], retainConverted = false) => ({
  request: {
    model: "anima-v1.0",
    quant: "bf16",
    textConditioner: "file" as const,
    retainConverted,
    compare: arms.length > 1,
    prompt: "p",
    resolution: { width: 512, height: 512 },
    steps: 4,
    seed: 42,
  },
  file: {
    path: "inputs/anima/base.safetensors",
    size: 10,
    read: "directory" as const,
    layout: "single-file",
    placed: { transformer: 567, text_conditioner: 118 },
  },
  distribution: {
    directory: "models/karume-anima",
    manifestSha256: "c".repeat(64),
    read: "directory" as const,
  },
  fromFile: ["transformer", "text_conditioner"],
  arms,
});

describe("the anima-single-file record", () => {
  it("says the arms give the same image when every hash agrees", () => {
    const data = animaSingleFileData(
      fields([
        arm("fromSingleFile", ["a".repeat(64), "a".repeat(64)]),
        arm("fromPretrained", ["a".repeat(64), "a".repeat(64)]),
      ]),
    );
    expect(data.format).toBe(ANIMA_SINGLE_FILE_FORMAT);
    expect(data.equal).toEqual({ all: true, arms: true });
    expect(animaSingleFileSummary(data).at(-1)).toBe(
      "fromSingleFile and fromPretrained give the same image; every image equal: yes",
    );
  });

  it("tells the arms apart from the generations of one arm", () => {
    // The two arms differ while each arm's two images agree (a from-file text conditioner).
    const data = animaSingleFileData(
      fields([
        arm("fromSingleFile", ["a".repeat(64), "a".repeat(64)]),
        arm("fromPretrained", ["b".repeat(64), "b".repeat(64)]),
      ]),
    );
    expect(data.equal).toEqual({ all: false, arms: false });
  });

  it("has no comparison of arms with one arm", () => {
    const data = animaSingleFileData(
      fields([arm("fromSingleFile", ["a".repeat(64), "d".repeat(64)])]),
    );
    expect(data.equal).toEqual({ all: false, arms: null });
    expect(animaSingleFileSummary(data)[0]).toBe(
      "inputs/anima/base.safetensors (single-file) → anima-v1.0 / bf16: from the file transformer, text_conditioner",
    );
  });

  it("keeps retainConverted in the request and says so in the first line", () => {
    const data = animaSingleFileData(
      fields([arm("fromSingleFile", ["a".repeat(64), "a".repeat(64)])], true),
    );
    expect(data.request.retainConverted).toBe(true);
    expect(animaSingleFileSummary(data)[0]).toBe(
      "inputs/anima/base.safetensors (single-file) → anima-v1.0 / bf16: from the file transformer, text_conditioner; the converted weights kept in memory",
    );
  });
});
