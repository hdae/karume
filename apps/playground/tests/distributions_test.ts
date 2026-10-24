import { mkdir, mkdtemp, readFile, rm, symlink, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterAll, beforeAll, describe, expect, it } from "vite-plus/test";

import {
  type DistributionMount,
  type DistributionRequest,
  type DistributionResponse,
  handleDistributionRequest,
} from "../vite-plugin-distributions.ts";

const BYTES = Uint8Array.from({ length: 64 }, (_, index) => index);

let dir = "";
let mounts: DistributionMount[] = [];

beforeAll(async () => {
  dir = await mkdtemp(join(tmpdir(), "karume-playground-"));
  const root = join(dir, "dist-a");
  await mkdir(join(root, "sub"), { recursive: true });
  await writeFile(join(root, "karume.json"), "{}");
  await writeFile(join(root, "sub", "blob.krm"), BYTES);
  await writeFile(join(dir, "secret.txt"), "outside");
  await symlink(join(dir, "secret.txt"), join(root, "escape.txt"));
  mounts = [
    { name: "a", root, label: "models/dist-a" },
    { name: "missing", root: join(dir, "nope"), label: "models/nope" },
  ];
});

afterAll(async () => {
  await rm(dir, { recursive: true, force: true });
});

const get = (url: string, headers: Record<string, string> = {}): Promise<DistributionResponse> =>
  handleDistributionRequest({ method: "GET", url, headers } satisfies DistributionRequest, mounts);

const responded = (response: DistributionResponse) => {
  if (response.kind !== "respond") throw new Error(`expected a response, got ${response.kind}`);
  return response;
};

/** The bytes a `file` body would stream. */
const bodyBytes = async (response: DistributionResponse): Promise<Uint8Array> => {
  const { body } = responded(response);
  if (body?.kind !== "file") throw new Error("expected a file body");
  return new Uint8Array((await readFile(body.path)).subarray(body.start, body.end + 1));
};

describe("handleDistributionRequest", () => {
  describe("a whole file", () => {
    it("answers 200 with the length and the bytes", async () => {
      const response = responded(await get("/models/a/sub/blob.krm"));
      expect(response.status).toBe(200);
      expect(response.headers["Content-Length"]).toBe("64");
      expect(response.headers["Accept-Ranges"]).toBe("bytes");
      expect(await bodyBytes(response)).toEqual(BYTES);
    });

    it("ignores the query string", async () => {
      expect(responded(await get("/models/a/karume.json?t=1")).status).toBe(200);
    });

    it("carries CORP so the isolated page may load it", async () => {
      const { headers } = responded(await get("/models/a/karume.json"));
      expect(headers["Cross-Origin-Resource-Policy"]).toBe("same-origin");
      expect(headers["Content-Type"]).toBe("application/json");
    });

    it("answers HEAD without a body", async () => {
      const response = responded(
        await handleDistributionRequest(
          { method: "HEAD", url: "/models/a/sub/blob.krm", headers: {} },
          mounts,
        ),
      );
      expect(response.status).toBe(200);
      expect(response.headers["Content-Length"]).toBe("64");
      expect(response.body).toBeUndefined();
    });

    it("refuses writes", async () => {
      const response = await handleDistributionRequest(
        { method: "PUT", url: "/models/a/karume.json", headers: {} },
        mounts,
      );
      expect(responded(response).status).toBe(405);
    });
  });

  describe("a range", () => {
    it("answers 206 with exactly the requested bytes", async () => {
      const response = responded(await get("/models/a/sub/blob.krm", { range: "bytes=10-19" }));
      expect(response.status).toBe(206);
      expect(response.headers["Content-Range"]).toBe("bytes 10-19/64");
      expect(response.headers["Content-Length"]).toBe("10");
      expect(await bodyBytes(response)).toEqual(BYTES.subarray(10, 20));
    });

    it("serves an open-ended range to the end", async () => {
      const response = responded(await get("/models/a/sub/blob.krm", { range: "bytes=60-" }));
      expect(response.headers["Content-Range"]).toBe("bytes 60-63/64");
      expect(await bodyBytes(response)).toEqual(BYTES.subarray(60));
    });

    it("answers 416 for a range that starts beyond the end", async () => {
      const response = responded(await get("/models/a/sub/blob.krm", { range: "bytes=64-80" }));
      expect(response.status).toBe(416);
      expect(response.headers["Content-Range"]).toBe("bytes */64");
      expect(response.body).toBeUndefined();
    });

    it("answers 416 for a malformed range", async () => {
      const response = responded(await get("/models/a/sub/blob.krm", { range: "bytes=5-2" }));
      expect(response.status).toBe(416);
    });
  });

  describe("paths outside a distribution", () => {
    it.each([
      "/models/a/../a/karume.json",
      "/models/a/sub/%2e%2e/karume.json",
      "/models/a//etc/passwd",
      "/models/a/%2Fetc%2Fpasswd",
      "/models/a/sub%5C..%5Ckarume.json",
    ])("answers 404 for %s", async (url) => {
      expect(responded(await get(url)).status).toBe(404);
    });

    it("answers 404 for a symlink that leaves the directory", async () => {
      expect(responded(await get("/models/a/escape.txt")).status).toBe(404);
    });

    it("answers 404 for an unknown prefix", async () => {
      expect(responded(await get("/models/b/karume.json")).status).toBe(404);
    });

    it("answers 404 naming the directory when a distribution is missing", async () => {
      const response = responded(await get("/models/missing/karume.json"));
      expect(response.status).toBe(404);
      expect(response.body).toEqual({
        kind: "text",
        text: "Distribution directory models/nope is missing",
      });
    });
  });

  describe("the index", () => {
    it("lists every mount with its directory and availability", async () => {
      const response = responded(await get("/models/index.json"));
      expect(response.status).toBe(200);
      if (response.body?.kind !== "text") throw new Error("expected a text body");
      expect(JSON.parse(response.body.text)).toEqual({
        a: { directory: "models/dist-a", available: true },
        missing: { directory: "models/nope", available: false },
      });
    });
  });

  describe("the page and everything else", () => {
    it.each(["/", "/wan22", "/src/main.tsx"])("passes %s to Vite with COOP / COEP", async (url) => {
      const response = await get(url);
      expect(response.kind).toBe("pass");
      expect(response.headers["Cross-Origin-Opener-Policy"]).toBe("same-origin");
      expect(response.headers["Cross-Origin-Embedder-Policy"]).toBe("require-corp");
    });
  });
});
