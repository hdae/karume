// Opening the mirror, the cross-repo reads and the missing mirror are `openMountDirectory`'s
// (packages/local-distributions/tests/node_test.ts); this adapter adds the facts.

import { createHash } from "node:crypto";
import { mkdir, mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { isDistributionSource } from "@karume/hub";
import { afterAll, beforeAll, describe, expect, it } from "vite-plus/test";

import { nodeDistributions } from "../node-distributions.ts";
import { DISTRIBUTION_MANIFEST } from "./helpers/distribution-manifest.ts";

/** A repository root of its own: the `wan22` mirror under `models/`, at the mount table's directory. */
let root = "";

beforeAll(async () => {
  root = await mkdtemp(join(tmpdir(), "workbench-distributions-"));
  const wan = join(root, "models", "karume-wan2.2");
  await mkdir(wan, { recursive: true });
  await writeFile(join(wan, "karume.json"), DISTRIBUTION_MANIFEST);
});

afterAll(async () => {
  await rm(root, { recursive: true, force: true });
});

describe("nodeDistributions", () => {
  it.each(["cached", "direct"] as const)(
    "reads the mirror's directory when a browser would read it %s, and records what it read",
    async (source) => {
      const opened = await nodeDistributions(root)("wan22", source);
      expect(isDistributionSource(opened.ref)).toBe(true);
      expect(opened.facts).toEqual({
        directory: "models/karume-wan2.2",
        manifestSha256: createHash("sha256").update(DISTRIBUTION_MANIFEST).digest("hex"),
        read: "directory",
      });
    },
  );
});
