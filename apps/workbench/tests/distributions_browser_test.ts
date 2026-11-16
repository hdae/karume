// The routes, the cross-repo mapping and the refusals are `openMountRoute`'s
// (packages/local-distributions/tests/open_mount_route_test.ts); this adapter adds the facts.

import { createHash } from "node:crypto";
import { afterEach, beforeEach, describe, expect, it, vi } from "vite-plus/test";

import { browserDistributions } from "../src/shell/distributions.ts";

/** The page's base URL (`document.baseURI`). */
const BASE = "http://127.0.0.1:8791/sub/";

/** A minimal `karume/5` manifest (the direct route parses it; the record hashes its bytes). */
const MANIFEST = JSON.stringify({
  format: "karume/5",
  generator: "test",
  defaultModel: "a",
  models: {
    a: {
      pipeline: "test/1",
      assets: {},
      weights: {
        net: {
          f16: {
            container: {
              descriptor: {
                graph: { length: 8, sha256: "d".repeat(64) },
                model: { length: 8, sha256: "e".repeat(64) },
              },
              parts: [
                { path: "a/model-00001-of-00002.krm", size: 40, sha256: "a".repeat(64) },
                { path: "a/model-00002-of-00002.krm", size: 8, sha256: "b".repeat(64) },
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

const INDEX = {
  "moss-image": {
    directory: "models/karume-moss-image",
    available: true,
    hf: { repo: "hdae/karume-moss-image", revision: MANIFEST_SHA256.slice(0, 40) },
  },
};

beforeEach(() => {
  vi.stubGlobal("fetch", (input: RequestInfo | URL) => {
    const url = typeof input === "string" ? input : input instanceof URL ? input.href : input.url;
    if (url === `${BASE}models/index.json`) return Promise.resolve(Response.json(INDEX));
    if (url === `${BASE}models/moss-image/karume.json`) {
      return Promise.resolve(new Response(MANIFEST));
    }
    return Promise.resolve(new Response("not found", { status: 404 }));
  });
});

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("browserDistributions", () => {
  it.each(["cached", "direct"] as const)(
    "records the mount's directory, the sha256 of the manifest it read and that it read it %s",
    async (source) => {
      const opened = await browserDistributions(BASE)("moss-image", source);
      expect(opened.facts).toEqual({
        directory: "models/karume-moss-image",
        manifestSha256: MANIFEST_SHA256,
        read: source,
      });
    },
  );
});
