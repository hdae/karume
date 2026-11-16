// The routes, the cross-repo mapping and the refusals are `openMountRoute`'s
// (packages/local-distributions/tests/open_mount_route_test.ts); this adapter adds the facts.

import { createHash } from "node:crypto";
import { afterEach, beforeEach, describe, expect, it, vi } from "vite-plus/test";

import { browserDistributions } from "../src/shell/distributions.ts";
import { DISTRIBUTION_MANIFEST } from "./helpers/distribution-manifest.ts";

/** The page's base URL (`document.baseURI`). */
const BASE = "http://127.0.0.1:8791/sub/";

const MANIFEST_SHA256 = createHash("sha256").update(DISTRIBUTION_MANIFEST).digest("hex");

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
      return Promise.resolve(new Response(DISTRIBUTION_MANIFEST));
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
