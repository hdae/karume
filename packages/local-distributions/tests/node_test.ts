import { createHash } from "node:crypto";
import { mkdir, mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fetchAssets, loadManifest, parseManifest } from "@karume/hub";
import { afterAll, beforeAll, describe, expect, it } from "vite-plus/test";

import { MountUnavailableError } from "../browser.ts";
import { MOUNTS } from "../mounts.ts";
import { localDistributionIndex, openMountDirectory } from "../node.ts";

const hex = (character: string): string => character.repeat(64);
const VAE = Uint8Array.from({ length: 4 }, (_, index) => index + 1);
const ANIMA_REVISION = "b".repeat(40);

/** A minimal `karume/5` manifest with one container of its own and the asset `vae`. */
const manifestText = (vae: Readonly<Record<string, string>>): string =>
  JSON.stringify({
    format: "karume/5",
    generator: "test",
    defaultModel: "a",
    models: {
      a: {
        pipeline: "test/1",
        assets: { vae: { path: "vae/model.krm", size: VAE.byteLength, sha256: hex("c"), ...vae } },
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

/** `anima-extra` takes its VAE from `hdae/karume-anima`, which the `anima` mount holds. */
const EXTRA = manifestText({ repo: "hdae/karume-anima", revision: ANIMA_REVISION });
/** The `anima` mount lists the VAE as its own file, so the `/hf/` route may alias the revision. */
const ANIMA = manifestText({});

const revisionOf = (text: string): string =>
  createHash("sha256").update(text).digest("hex").slice(0, 40);

/** A repository root of its own, with two mirrors at the mount table's directories. */
let root = "";

beforeAll(async () => {
  root = await mkdtemp(join(tmpdir(), "local-distributions-node-"));
  const extra = join(root, "models", "karume-anima-extra");
  const anima = join(root, "models", "karume-anima");
  await mkdir(extra, { recursive: true });
  await mkdir(join(anima, "vae"), { recursive: true });
  await writeFile(join(extra, "karume.json"), EXTRA);
  await writeFile(join(anima, "karume.json"), ANIMA);
  await writeFile(join(anima, "vae", "model.krm"), VAE);
});

afterAll(async () => {
  await rm(root, { recursive: true, force: true });
});

describe("localDistributionIndex", () => {
  it("answers what /models/index.json answers: every mount, its directory, whether it is there, its /hf/ coordinates", async () => {
    expect(parseManifest(EXTRA).defaultModel).toBe("a");
    const index = await localDistributionIndex(root);
    expect(Object.keys(index)).toEqual(Object.keys(MOUNTS));
    expect(index["anima-extra"]).toEqual({
      directory: "models/karume-anima-extra",
      available: true,
      hf: { repo: "hdae/karume-anima-extra", revision: revisionOf(EXTRA) },
    });
    expect(index["anima"]).toEqual({
      directory: "models/karume-anima",
      available: true,
      hf: { repo: "hdae/karume-anima", revision: revisionOf(ANIMA) },
    });
    expect(index["moss-image"]).toEqual({
      directory: "models/karume-moss-image",
      available: false,
    });
    expect(index["umt5"]).toEqual({ directory: "models/karume-umt5-xxl", available: false });
  });
});

describe("openMountDirectory", () => {
  it("opens the mount's directory with its manifest's bytes and directory", async () => {
    const opened = await openMountDirectory(root, "anima-extra");
    expect(opened.directory).toBe("models/karume-anima-extra");
    expect(new TextDecoder().decode(opened.manifestBytes)).toBe(EXTRA);
    expect(Object.keys((await loadManifest(opened.source)).manifest.models)).toEqual(["a"]);
  });

  it("reads a cross-repo reference from the directory of the mount that holds the repository", async () => {
    const loaded = await loadManifest((await openMountDirectory(root, "anima-extra")).source);
    const vae = loaded.manifest.models["a"]?.assets["vae"];
    if (vae === undefined) throw new Error("the fixture has no asset 'vae'");
    expect((await fetchAssets(loaded, { vae }))["vae"]).toEqual(VAE);
  });

  it("throws a MountUnavailableError naming the mount when its mirror is missing", async () => {
    const open = openMountDirectory(root, "moss-image");
    await expect(open).rejects.toThrow(MountUnavailableError);
    await expect(open).rejects.toThrow(/moss-image.*models\/karume-moss-image/);
  });
});
