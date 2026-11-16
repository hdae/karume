import { createHash } from "node:crypto";
import { fetchAssets, isDistributionSource, loadManifest, parseManifest } from "@karume/hub";
import { MountUnavailableError } from "@karume/local-distributions/browser";
import { afterEach, beforeEach, describe, expect, it, vi } from "vite-plus/test";

import { browserDistributions } from "../src/shell/distributions.ts";

/** The page's base URL (`document.baseURI`); a path shows that every URL is resolved against it. */
const BASE = "http://127.0.0.1:8791/sub/";

const hex = (character: string): string => character.repeat(64);

/**
 * A minimal `karume/5` manifest of `anima-extra` whose one asset is a cross-repo reference into
 * `hdae/karume-anima` (the shape of the Anima fine-tunes).
 */
const MANIFEST = JSON.stringify({
  format: "karume/5",
  generator: "test",
  defaultModel: "a",
  models: {
    a: {
      pipeline: "test/1",
      assets: {
        vae: {
          path: "vae/model.krm",
          size: 4,
          sha256: hex("c"),
          repo: "hdae/karume-anima",
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
                { path: "a/model-00001-of-00002.krm", size: 40, sha256: hex("a") },
                { path: "a/model-00002-of-00002.krm", size: 8, sha256: hex("b") },
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
const MANIFEST_SHA256 = createHash("sha256").update(MANIFEST).digest("hex");

/** What the dev server's `/models/index.json` answers (the plugin's revision rule for `/hf/`). */
const INDEX = {
  "anima-extra": {
    directory: "models/karume-anima-extra",
    available: true,
    hf: { repo: "hdae/karume-anima-extra", revision: MANIFEST_SHA256.slice(0, 40) },
  },
  anima: {
    directory: "models/karume-anima",
    available: true,
    hf: { repo: "hdae/karume-anima", revision: "2".repeat(40) },
  },
  umt5: { directory: "models/karume-umt5-xxl", available: true },
  "moss-image": { directory: "models/karume-moss-image", available: false },
};

let requested: string[] = [];

beforeEach(() => {
  requested = [];
  vi.stubGlobal("fetch", (input: RequestInfo | URL) => {
    const url = typeof input === "string" ? input : input instanceof URL ? input.href : input.url;
    requested.push(url);
    if (url === `${BASE}models/index.json`) return Promise.resolve(Response.json(INDEX));
    if (url.endsWith("/karume.json")) return Promise.resolve(new Response(MANIFEST));
    if (url === `${BASE}models/anima/vae/model.krm`) {
      return Promise.resolve(new Response(new Uint8Array(4)));
    }
    return Promise.resolve(new Response("not found", { status: 404 }));
  });
});

afterEach(() => {
  vi.unstubAllGlobals();
});

/** The error `open` rejects with, which must be a {@link MountUnavailableError}. */
const unavailable = async (open: Promise<unknown>): Promise<MountUnavailableError> => {
  try {
    await open;
  } catch (error) {
    if (error instanceof MountUnavailableError) return error;
    throw error;
  }
  throw new Error("expected a MountUnavailableError");
};

describe("browserDistributions", () => {
  it("hands the cached route's HubRepoRef below the base and records the manifest it read", async () => {
    expect(parseManifest(MANIFEST).defaultModel).toBe("a");
    const opened = await browserDistributions(BASE)("anima-extra", "cached");
    expect(opened.ref).toEqual({
      repo: "hdae/karume-anima-extra",
      revision: MANIFEST_SHA256.slice(0, 40),
      hubUrl: `${BASE}hf`,
    });
    expect(opened.facts).toEqual({
      directory: "models/karume-anima-extra",
      manifestSha256: MANIFEST_SHA256,
      read: "cached",
    });
    expect(requested).toEqual([
      `${BASE}models/index.json`,
      `${BASE}models/anima-extra/karume.json`,
    ]);
  });

  it("reads the direct route from /models/, and a cross-repo reference from the mount that holds it", async () => {
    const opened = await browserDistributions(BASE)("anima-extra", "direct");
    expect(opened.facts).toEqual({
      directory: "models/karume-anima-extra",
      manifestSha256: MANIFEST_SHA256,
      read: "direct",
    });
    expect(isDistributionSource(opened.ref)).toBe(true);
    const loaded = await loadManifest(opened.ref);
    const vae = loaded.manifest.models["a"]?.assets["vae"];
    if (vae === undefined) throw new Error("the fixture has no asset 'vae'");
    await fetchAssets(loaded, { vae });
    expect(requested.slice(2)).toEqual([
      `${BASE}models/anima-extra/karume.json`,
      `${BASE}models/anima/vae/model.krm`,
    ]);
  });

  it("names the mount when its mirror is missing", async () => {
    const error = await unavailable(browserDistributions(BASE)("moss-image", "direct"));
    expect(error.mount).toBe("moss-image");
    expect(error.message).toContain("moss-image");
    expect(error.message).toContain("models/karume-moss-image");
  });

  it("names the mount when it is asked for cached but has no HF-compatible coordinates", async () => {
    const error = await unavailable(browserDistributions(BASE)("umt5", "cached"));
    expect(error.mount).toBe("umt5");
    expect(error.message).toContain("umt5");
  });
});
