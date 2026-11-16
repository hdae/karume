import { createHash } from "node:crypto";
import { mkdir, mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fetchAssets, isDistributionSource, loadManifest, parseManifest } from "@karume/hub";
import { MountUnavailableError } from "@karume/local-distributions/browser";
import { afterAll, beforeAll, describe, expect, it } from "vite-plus/test";

import { nodeDistributions } from "../node-distributions.ts";

const hex = (character: string): string => character.repeat(64);
const EMBEDDINGS = Uint8Array.from({ length: 16 }, (_, index) => index);

/**
 * A minimal `karume/5` manifest of the `wan22` mount whose one asset is a cross-repo reference into
 * `hdae/karume-umt5-xxl` (the shape of Wan's text encoder).
 */
const MANIFEST = JSON.stringify({
  format: "karume/5",
  generator: "test",
  defaultModel: "w",
  models: {
    w: {
      pipeline: "test/1",
      assets: {
        embeddings: {
          path: "text_encoder/embeddings.bin",
          size: EMBEDDINGS.byteLength,
          sha256: hex("c"),
          repo: "hdae/karume-umt5-xxl",
          revision: "b".repeat(40),
        },
      },
      weights: {
        net: {
          f16: {
            container: {
              descriptor: {
                graph: { length: 8, sha256: hex("d") },
                model: { length: 8, sha256: hex("e") },
              },
              parts: [
                { path: "net/model-00001-of-00002.krm", size: 40, sha256: hex("a") },
                { path: "net/model-00002-of-00002.krm", size: 8, sha256: hex("b") },
              ],
            },
          },
        },
      },
      quants: { f16: { weights: { net: "f16" }, session: {} } },
      defaultQuant: "f16",
      pipelineConfig: {},
    },
  },
});

/** A repository root of its own: the two mirrors under `models/`, at the mount table's directories. */
let root = "";

beforeAll(async () => {
  root = await mkdtemp(join(tmpdir(), "workbench-distributions-"));
  const wan = join(root, "models", "karume-wan2.2");
  const umt5 = join(root, "models", "karume-umt5-xxl");
  await mkdir(wan, { recursive: true });
  await mkdir(join(umt5, "text_encoder"), { recursive: true });
  await writeFile(join(wan, "karume.json"), MANIFEST);
  await writeFile(join(umt5, "karume.json"), "{}");
  await writeFile(join(umt5, "text_encoder", "embeddings.bin"), EMBEDDINGS);
});

afterAll(async () => {
  await rm(root, { recursive: true, force: true });
});

describe("nodeDistributions", () => {
  it.each(["cached", "direct"] as const)(
    "reads the mirror's directory when a browser would read it %s, and records what it read",
    async (source) => {
      expect(parseManifest(MANIFEST).defaultModel).toBe("w");
      const opened = await nodeDistributions(root)("wan22", source);
      expect(isDistributionSource(opened.ref)).toBe(true);
      expect(opened.facts).toEqual({
        directory: "models/karume-wan2.2",
        manifestSha256: createHash("sha256").update(MANIFEST).digest("hex"),
        read: "directory",
      });
    },
  );

  it("reads a cross-repo reference from the directory of the mount that holds it", async () => {
    const opened = await nodeDistributions(root)("wan22", "direct");
    const loaded = await loadManifest(opened.ref);
    const embeddings = loaded.manifest.models["w"]?.assets["embeddings"];
    if (embeddings === undefined) throw new Error("the fixture has no asset 'embeddings'");
    expect((await fetchAssets(loaded, { embeddings }))["embeddings"]).toEqual(EMBEDDINGS);
  });

  it("names the mount when its mirror is missing", async () => {
    const open = nodeDistributions(root)("moss-image", "direct");
    await expect(open).rejects.toThrow(MountUnavailableError);
    await expect(open).rejects.toThrow(/moss-image.*models\/karume-moss-image/);
  });
});
