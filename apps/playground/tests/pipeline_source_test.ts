import { fetchAssets, isDistributionSource, loadManifest } from "@karume/hub";
import { afterEach, describe, expect, it, vi } from "vite-plus/test";

import { pretrainedRef } from "../src/lib/distributions.ts";

const REF = { repo: "owner/karume-x", revision: "a".repeat(40), hubUrl: "http://127.0.0.1/hf" };

const hex = (character: string): string => character.repeat(64);

/**
 * A minimal `karume/5` manifest of `anima-extra` whose one asset is a cross-repo reference into
 * `hdae/karume-anima` (the shape of the Anima fine-tunes: their shared files live in the base repo).
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

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("pretrainedRef", () => {
  it("hands the cached route's ref to fromPretrained as is", () => {
    expect(pretrainedRef({ kind: "cached", ref: REF, cacheGeneration: 0 }, "anima")).toBe(REF);
  });

  it("reads the direct route from the mount, and a cross-repo reference from the mapped mount", async () => {
    const requested: string[] = [];
    vi.stubGlobal("fetch", (input: RequestInfo | URL) => {
      const url = typeof input === "string" ? input : input instanceof URL ? input.href : input.url;
      requested.push(url);
      return Promise.resolve(
        url.endsWith("/karume.json")
          ? new Response(MANIFEST, { status: 200 })
          : new Response(new Uint8Array(4), { status: 200 }),
      );
    });
    const ref = pretrainedRef({ kind: "direct" }, "anima-extra", {
      "hdae/karume-anima": "anima",
    });
    expect(isDistributionSource(ref)).toBe(true);
    const loaded = await loadManifest(ref);
    const vae = loaded.manifest.models["a"]?.assets["vae"];
    if (vae === undefined) throw new Error("the fixture has no asset 'vae'");
    await fetchAssets(loaded, { vae });
    expect(requested).toEqual(["models/anima-extra/karume.json", "models/anima/vae/model.krm"]);
  });
});
