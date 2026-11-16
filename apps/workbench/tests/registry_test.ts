import { readdir } from "node:fs/promises";
import { describe, expect, it } from "vite-plus/test";

import { entriesOf, registeredEntries, RegistryError } from "../src/entries.ts";
import type { Entry } from "../src/entry.ts";

const fakeEntry = (id: string): Entry => ({
  id,
  title: id,
  description: "A test entry.",
  parameters: [],
  run: () => Promise.resolve({ summary: [], data: null }),
});

describe("the registry", () => {
  it("holds one entry per file of src/entries/, each id equal to its file name", async () => {
    const files = (await readdir(new URL("../src/entries/", import.meta.url)))
      .filter((file) => file.endsWith(".ts"))
      .map((file) => file.slice(0, -".ts".length))
      .sort();
    expect(registeredEntries().map(({ id }) => id)).toEqual(files);
  });

  it("has unique ids", () => {
    const ids = registeredEntries().map(({ id }) => id);
    expect(new Set(ids).size).toBe(ids.length);
  });

  it("throws at start on an id that is not the file name, or a file without a default export", () => {
    expect(() => entriesOf({ "./entries/kernel-check.ts": fakeEntry("kernel-checks") })).toThrow(
      RegistryError,
    );
    expect(() => entriesOf({ "./entries/Kernel.ts": fakeEntry("Kernel") })).toThrow(RegistryError);
    expect(() => entriesOf({ "./entries/empty.ts": undefined })).toThrow(RegistryError);
    expect(
      entriesOf({ "./entries/b.ts": fakeEntry("b"), "./entries/a.ts": fakeEntry("a") }),
    ).toEqual([expect.objectContaining({ id: "a" }), expect.objectContaining({ id: "b" })]);
  });
});
