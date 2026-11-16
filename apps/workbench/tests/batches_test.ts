import { readdir } from "node:fs/promises";
import { describe, expect, it } from "vite-plus/test";

import { BATCH_FORMAT, listBatches, resolveBatch } from "../src/batch.ts";
import { trackedBatchFiles } from "../src/batches.ts";
import { entriesOf, registeredEntries } from "../src/entries.ts";
import type { Entry } from "../src/entry.ts";

/** The registry of a production build: the same glob without the local entries. */
const PRODUCTION_ENTRIES = entriesOf(
  import.meta.glob<Entry>("../src/entries/*.ts", { eager: true, import: "default" }),
);

describe("the tracked batches", () => {
  it("are the files of apps/workbench/batches/, each id its file name", async () => {
    const files = (await readdir(new URL("../batches/", import.meta.url)))
      .filter((file) => file.endsWith(".json"))
      .map((file) => file.slice(0, -".json".length))
      .sort();
    expect(files).toContain("synthetic-quick");
    expect(trackedBatchFiles().map(({ id }) => id)).toEqual(files);
  });

  it("are each valid against the full registry, so a renamed parameter turns this red", () => {
    const entries = registeredEntries();
    for (const file of trackedBatchFiles()) {
      if (!("text" in file)) throw new Error(`${file.id}: ${file.problem}`);
      const resolution = resolveBatch(file.id, file.text, entries);
      expect(resolution.kind === "problems" ? resolution.problems : [], file.id).toEqual([]);
    }
  });

  it("include synthetic-quick: the three synthetic entries, the sweep at the quick set without cooling", () => {
    const file = trackedBatchFiles().find(({ id }) => id === "synthetic-quick");
    if (file === undefined || !("text" in file)) throw new Error("no synthetic-quick.json");
    const batch = resolveBatch(file.id, file.text, PRODUCTION_ENTRIES);
    if (batch.kind !== "resolved") throw new Error(batch.problems.join("; "));
    expect(batch.steps.map(({ entry, given }) => [entry.id, given])).toEqual([
      ["kernel-check", {}],
      ["geometry-sweep", { candidateSet: "quick", rounds: "1", idleSeconds: "0" }],
      ["attention-bench", {}],
    ]);
  });
});

describe("the production listing", () => {
  it("holds no entry that reads a local distribution", () => {
    const ids = PRODUCTION_ENTRIES.map(({ id }) => id);
    expect(ids).toContain("kernel-check");
    expect(ids).not.toContain("session-ab");
    expect(registeredEntries().map(({ id }) => id)).toContain("session-ab");
  });

  it("leaves out a batch that names a local-distribution entry; the dev server lists it", () => {
    const tracked = [
      ...trackedBatchFiles(),
      {
        id: "needs-a-mirror",
        text: JSON.stringify({
          format: BATCH_FORMAT,
          title: "Needs a mirror",
          steps: [{ id: "kernel-check" }, { id: "session-ab", parameters: { source: "direct" } }],
        }),
      },
    ];
    const production = listBatches({
      tracked,
      untracked: [],
      entries: PRODUCTION_ENTRIES,
      production: true,
    });
    expect(production.map(({ id }) => id)).toEqual(trackedBatchFiles().map(({ id }) => id));
    const development = listBatches({
      tracked,
      untracked: [],
      entries: registeredEntries(),
      production: false,
    });
    expect(development.find(({ id }) => id === "needs-a-mirror")?.kind).toBe("resolved");
  });
});
