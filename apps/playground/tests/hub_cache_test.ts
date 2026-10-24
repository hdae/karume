import { describe, expect, it } from "vite-plus/test";

import { unionInventory } from "../src/lib/hub-cache.ts";

const ref = (path: string, size: number, repo?: string) => ({
  path,
  size,
  sha256: path.length.toString(16).padStart(64, "0"),
  ...(repo === undefined ? {} : { repo, revision: "b".repeat(40) }),
});

describe("unionInventory", () => {
  it("counts a file several presets share once", () => {
    const shared = ref("vae/model.krm", 100);
    expect(
      unionInventory([
        { cached: [shared, ref("dit/i8.krm", 1000)], missing: [] },
        { cached: [shared], missing: [ref("dit/f16.krm", 2000)] },
      ]),
    ).toEqual({ cachedFiles: 2, cachedBytes: 1100, totalFiles: 3, totalBytes: 3100 });
  });

  it("tells a cross-repo file apart from a same-named file of the distribution", () => {
    expect(
      unionInventory([
        { cached: [ref("model.krm", 10)], missing: [ref("model.krm", 20, "owner/other")] },
      ]),
    ).toEqual({ cachedFiles: 1, cachedBytes: 10, totalFiles: 2, totalBytes: 30 });
  });

  it("is empty for no presets", () => {
    expect(unionInventory([])).toEqual({
      cachedFiles: 0,
      cachedBytes: 0,
      totalFiles: 0,
      totalBytes: 0,
    });
  });
});
