import { readdir } from "node:fs/promises";
import { describe, expect, it } from "vite-plus/test";

import { MOUNTS } from "../src/mounts.ts";
import { MODEL_PAGES, PAGE_GROUPS, pageAt, pagesByGroup } from "../src/models/registry.ts";

/** The file routes under `src/routes/` other than the root layout and the index, as paths. */
const routePaths = async (): Promise<string[]> =>
  (await readdir(new URL("../src/routes/", import.meta.url)))
    .filter((file) => file.endsWith(".tsx") && file !== "__root.tsx" && file !== "index.tsx")
    .map((file) => `/${file.slice(0, -".tsx".length)}`)
    .sort();

describe("the page registry", () => {
  it("has exactly one entry per route file", async () => {
    expect(MODEL_PAGES.map(({ path }) => path).sort()).toEqual(await routePaths());
  });

  it("names only mounts the dev server serves", () => {
    for (const page of MODEL_PAGES) {
      for (const mount of page.mounts) expect(Object.hasOwn(MOUNTS, mount)).toBe(true);
    }
  });

  it("lists the groups in sidebar order, leaving none of the pages out", () => {
    const grouped = pagesByGroup();
    expect(grouped.map(({ group }) => group)).toEqual(
      PAGE_GROUPS.filter((group) => MODEL_PAGES.some((page) => page.group === group)),
    );
    expect(grouped.flatMap(({ pages }) => pages)).toHaveLength(MODEL_PAGES.length);
  });

  it("finds a page by its exact path only", () => {
    expect(pageAt("/wan22")?.title).toBe("Wan2.2 TI2V 5B");
    expect(pageAt("/wan22/")).toBeUndefined();
    expect(pageAt("/")).toBeUndefined();
  });
});
