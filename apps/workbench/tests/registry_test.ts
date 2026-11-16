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

const idsOf = (entries: readonly Entry[]): readonly string[] => entries.map(({ id }) => id);

describe("the registry", () => {
  it("holds one entry per file of src/entries/ and src/entries/local/, each id equal to its file name", async () => {
    const namesIn = async (directory: string): Promise<readonly string[]> =>
      (await readdir(new URL(directory, import.meta.url)))
        .filter((file) => file.endsWith(".ts"))
        .map((file) => file.slice(0, -".ts".length));
    const files = [
      ...(await namesIn("../src/entries/")),
      ...(await namesIn("../src/entries/local/")),
    ].sort();
    expect(files).toContain("session-ab");
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

describe("the entries that read a local distribution", () => {
  it("are the files under entries/local/, in file-name order with the rest", () => {
    expect(
      idsOf(
        entriesOf({
          "./entries/c.ts": fakeEntry("c"),
          "./entries/local/b.ts": fakeEntry("b"),
          "./entries/a.ts": fakeEntry("a"),
        }),
      ),
    ).toEqual(["a", "b", "c"]);
  });

  it("throw at start when two files share a name", () => {
    expect(() =>
      entriesOf({ "./entries/a.ts": fakeEntry("a"), "./entries/local/a.ts": fakeEntry("a") }),
    ).toThrow(RegistryError);
  });
});
