import { createHash } from "node:crypto";
import { mkdir, mkdtemp, readFile, rm, symlink, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterAll, beforeAll, describe, expect, it } from "vite-plus/test";

import {
  type DistributionMount,
  type DistributionResponse,
  handleDistributionRequest,
  hfServingTable,
} from "../vite-plugin-distributions.ts";

const MAIN_REPO = "owner/karume-dist-main";
const CROSS_REPO = "owner/karume-dist-b";
const CROSS_REVISION = "b".repeat(40);
const ASSET = Uint8Array.from({ length: 64 }, (_, index) => index);
const CROSS_PART = Uint8Array.from({ length: 64 }, (_, index) => 255 - index);

const hex = (character: string): string => character.repeat(64);

/** A minimal `karume/5` manifest; `cross` puts its one container in another repo. */
const manifestText = (cross?: { readonly repo: string; readonly revision: string }): string =>
  JSON.stringify({
    format: "karume/5",
    generator: "test",
    defaultModel: "m",
    models: {
      m: {
        pipeline: "wan-ti2v/1",
        assets: { notes: { path: "sub/blob.krm", size: 64, sha256: hex("c") } },
        weights: {
          text_encoder: {
            i8: {
              container: {
                descriptor: {
                  graph: { length: 8, sha256: hex("d") },
                  model: { length: 8, sha256: hex("e") },
                },
                parts: [
                  { path: "enc/model-00001-of-00002.krm", size: 40, sha256: hex("a"), ...cross },
                  { path: "enc/model-00002-of-00002.krm", size: 64, sha256: hex("f"), ...cross },
                ],
              },
            },
          },
        },
        quants: { q: { weights: { text_encoder: "i8" }, session: {} } },
        defaultQuant: "q",
        pipelineConfig: {},
      },
    },
  });

let dir = "";
let mainRoot = "";
let crossRoot = "";
let mounts: DistributionMount[] = [];

beforeAll(async () => {
  dir = await mkdtemp(join(tmpdir(), "karume-playground-hf-"));
  mainRoot = join(dir, "karume-dist-main");
  crossRoot = join(dir, "karume-dist-b");
  await mkdir(join(mainRoot, "sub"), { recursive: true });
  await mkdir(join(crossRoot, "enc"), { recursive: true });
  await writeFile(
    join(mainRoot, "karume.json"),
    manifestText({ repo: CROSS_REPO, revision: CROSS_REVISION }),
  );
  await writeFile(join(mainRoot, "sub", "blob.krm"), ASSET);
  await writeFile(join(crossRoot, "karume.json"), manifestText());
  await writeFile(join(crossRoot, "enc", "model-00002-of-00002.krm"), CROSS_PART);
  await writeFile(join(dir, "secret.txt"), "outside");
  await symlink(join(dir, "secret.txt"), join(mainRoot, "escape.txt"));
  mounts = [
    { name: "main", root: mainRoot, label: "models/karume-dist-main", hfRepo: MAIN_REPO },
    { name: "b", root: crossRoot, label: "models/karume-dist-b" },
  ];
});

afterAll(async () => {
  await rm(dir, { recursive: true, force: true });
});

const request = (url: string, method = "GET"): Promise<DistributionResponse> =>
  handleDistributionRequest({ method, url, headers: {} }, mounts);

const responded = (response: DistributionResponse) => {
  if (response.kind !== "respond") throw new Error(`expected a response, got ${response.kind}`);
  return response;
};

const jsonBody = (response: DistributionResponse): unknown => {
  const { body } = responded(response);
  if (body?.kind !== "text") throw new Error("expected a text body");
  return JSON.parse(body.text);
};

const fileBytes = async (response: DistributionResponse): Promise<Uint8Array> => {
  const { body } = responded(response);
  if (body?.kind !== "file") throw new Error("expected a file body");
  return new Uint8Array((await readFile(body.path)).subarray(body.start, body.end + 1));
};

/** The revision the route must answer: the first 40 hex digits of sha256(karume.json). */
const expectedRevision = async (): Promise<string> =>
  createHash("sha256")
    .update(await readFile(join(mainRoot, "karume.json")))
    .digest("hex")
    .slice(0, 40);

describe("the HF-compatible route", () => {
  describe("the revision API", () => {
    it("answers the sha256-derived revision of the main repo for any ref", async () => {
      const revision = await expectedRevision();
      for (const ref of ["main", revision]) {
        const response = responded(await request(`/hf/api/models/${MAIN_REPO}/revision/${ref}`));
        expect(response.status).toBe(200);
        expect(response.headers["Content-Type"]).toBe("application/json");
        expect(jsonBody(response)).toEqual({ sha: revision });
      }
    });

    it("answers the declared revision of a cross-repo reference", async () => {
      expect(jsonBody(await request(`/hf/api/models/${CROSS_REPO}/revision/main`))).toEqual({
        sha: CROSS_REVISION,
      });
    });

    it("answers 404 for an unknown repo", async () => {
      expect(responded(await request("/hf/api/models/owner/nope/revision/main")).status).toBe(404);
    });
  });

  describe("the resolve URL", () => {
    it("serves the bytes with their length, CORP and no-store", async () => {
      const response = responded(
        await request(`/hf/${MAIN_REPO}/resolve/${await expectedRevision()}/sub/blob.krm`),
      );
      expect(response.status).toBe(200);
      expect(response.headers["Content-Length"]).toBe("64");
      expect(response.headers["Cross-Origin-Resource-Policy"]).toBe("same-origin");
      expect(response.headers["Cross-Origin-Embedder-Policy"]).toBe("require-corp");
      expect(response.headers["Cache-Control"]).toBe("no-store");
      expect(await fileBytes(response)).toEqual(ASSET);
    });

    it("serves a cross-repo file from the mount named after the repo", async () => {
      const response = responded(
        await request(`/hf/${CROSS_REPO}/resolve/${CROSS_REVISION}/enc/model-00002-of-00002.krm`),
      );
      expect(response.status).toBe(200);
      expect(await fileBytes(response)).toEqual(CROSS_PART);
    });

    it("answers 404 for a revision the repo is not at", async () => {
      const stale = "0".repeat(40);
      expect(
        responded(await request(`/hf/${MAIN_REPO}/resolve/${stale}/sub/blob.krm`)).status,
      ).toBe(404);
      expect(
        responded(await request(`/hf/${CROSS_REPO}/resolve/${stale}/enc/model-00002-of-00002.krm`))
          .status,
      ).toBe(404);
    });

    it("answers 404 for an unknown repo", async () => {
      expect(
        responded(await request(`/hf/owner/nope/resolve/${CROSS_REVISION}/karume.json`)).status,
      ).toBe(404);
    });

    it("answers 404 for a path that is not a route", async () => {
      expect(responded(await request("/hf/owner/karume-dist-main/blob/x")).status).toBe(404);
    });

    it("refuses writes", async () => {
      expect(
        responded(await request(`/hf/api/models/${MAIN_REPO}/revision/main`, "POST")).status,
      ).toBe(405);
    });
  });

  describe("paths outside the repo", () => {
    it.each([
      "../karume-dist-b/karume.json",
      "sub/%2e%2e/%2e%2e/secret.txt",
      "%2Fetc%2Fpasswd",
      "sub//blob.krm",
      "escape.txt",
    ])("answers 404 for %s", async (path) => {
      const url = `/hf/${MAIN_REPO}/resolve/${await expectedRevision()}/${path}`;
      expect(responded(await request(url)).status).toBe(404);
    });
  });

  describe("a re-assembled mirror", () => {
    it("moves the revision, so the old revision answers 404", async () => {
      const before = await expectedRevision();
      const original = await readFile(join(mainRoot, "karume.json"), "utf8");
      await writeFile(join(mainRoot, "karume.json"), `${original}\n`);
      try {
        const after = await expectedRevision();
        expect(after).not.toBe(before);
        expect(jsonBody(await request(`/hf/api/models/${MAIN_REPO}/revision/main`))).toEqual({
          sha: after,
        });
        expect(
          responded(await request(`/hf/${MAIN_REPO}/resolve/${before}/karume.json`)).status,
        ).toBe(404);
        expect(
          responded(await request(`/hf/${MAIN_REPO}/resolve/${after}/karume.json`)).status,
        ).toBe(200);
      } finally {
        await writeFile(join(mainRoot, "karume.json"), original);
      }
    });
  });

  describe("the index", () => {
    it("names the repo and revision of a mount served on the route", async () => {
      expect(jsonBody(await request("/models/index.json"))).toEqual({
        main: {
          directory: "models/karume-dist-main",
          available: true,
          hf: { repo: MAIN_REPO, revision: await expectedRevision() },
        },
        b: { directory: "models/karume-dist-b", available: true },
      });
    });
  });

  describe("the serving table", () => {
    it("fails loudly when one repo would be served at two revisions", async () => {
      const both = mounts.map((mount) =>
        mount.name === "b" ? { ...mount, hfRepo: CROSS_REPO } : mount,
      );
      await expect(hfServingTable(both)).rejects.toThrow(/one directory holds one revision/);
    });

    it("lists a cross-repo reference no mount holds", async () => {
      const mainOnly = mounts.filter(({ name }) => name === "main");
      expect((await hfServingTable(mainOnly)).unserved).toEqual([
        `${CROSS_REPO}@${CROSS_REVISION}`,
      ]);
    });
  });
});

describe("a referenced revision on a mount that opts in to aliases", () => {
  const ALIAS_MAIN = "owner/karume-alias-main";
  const ALIAS_TARGET = "owner/karume-alias-target";
  let aliasMounts: DistributionMount[] = [];
  let targetRoot = "";
  let conflictRoot = "";

  beforeAll(async () => {
    const mainDir = join(dir, "karume-alias-main");
    targetRoot = join(dir, "karume-alias-target");
    conflictRoot = join(dir, "karume-alias-conflict");
    await mkdir(join(mainDir, "sub"), { recursive: true });
    await mkdir(join(targetRoot, "enc"), { recursive: true });
    await mkdir(join(conflictRoot, "enc"), { recursive: true });
    // The main manifest takes both encoder parts from the target repo at CROSS_REVISION; the
    // target's own manifest lists the same parts (same path, size, sha256) as its own files.
    await writeFile(
      join(mainDir, "karume.json"),
      manifestText({ repo: ALIAS_TARGET, revision: CROSS_REVISION }),
    );
    await writeFile(join(targetRoot, "karume.json"), manifestText());
    await writeFile(join(targetRoot, "enc", "model-00002-of-00002.krm"), CROSS_PART);
    // The same paths with a different sha256: aliasing would serve other bytes.
    await writeFile(
      join(conflictRoot, "karume.json"),
      manifestText().replaceAll(hex("f"), hex("9")),
    );
    aliasMounts = [
      { name: "alias-main", root: mainDir, label: "models/karume-alias-main", hfRepo: ALIAS_MAIN },
      {
        name: "alias-target",
        root: targetRoot,
        label: "models/karume-alias-target",
        hfRepo: ALIAS_TARGET,
        aliasReferencedRevisions: true,
      },
    ];
  });

  const aliasRequest = (url: string): Promise<DistributionResponse> =>
    handleDistributionRequest({ method: "GET", url, headers: {} }, aliasMounts);

  const targetRevision = async (): Promise<string> =>
    createHash("sha256")
      .update(await readFile(join(targetRoot, "karume.json")))
      .digest("hex")
      .slice(0, 40);

  it("serves a referenced file at the declared revision", async () => {
    const response = responded(
      await aliasRequest(
        `/hf/${ALIAS_TARGET}/resolve/${CROSS_REVISION}/enc/model-00002-of-00002.krm`,
      ),
    );
    expect(response.status).toBe(200);
    expect(await fileBytes(response)).toEqual(CROSS_PART);
  });

  it("keeps serving the repo at its own revision", async () => {
    const revision = await targetRevision();
    expect(jsonBody(await aliasRequest(`/hf/api/models/${ALIAS_TARGET}/revision/main`))).toEqual({
      sha: revision,
    });
    const response = responded(
      await aliasRequest(`/hf/${ALIAS_TARGET}/resolve/${revision}/karume.json`),
    );
    expect(response.status).toBe(200);
  });

  it("answers the alias revision as itself on the revision API", async () => {
    expect(
      jsonBody(await aliasRequest(`/hf/api/models/${ALIAS_TARGET}/revision/${CROSS_REVISION}`)),
    ).toEqual({ sha: CROSS_REVISION });
  });

  it("answers 404 at the alias revision for a file no reference names", async () => {
    expect(
      responded(await aliasRequest(`/hf/${ALIAS_TARGET}/resolve/${CROSS_REVISION}/karume.json`))
        .status,
    ).toBe(404);
  });

  it("names the alias for the startup log", async () => {
    expect((await hfServingTable(aliasMounts)).aliases).toEqual([
      {
        repo: ALIAS_TARGET,
        revision: CROSS_REVISION,
        servedRevision: await targetRevision(),
        directory: "models/karume-alias-target",
        referencedBy: [ALIAS_MAIN],
        files: 2,
      },
    ]);
  });

  it("fails loudly when a referenced file differs from the mount's manifest", async () => {
    const conflicting = aliasMounts.map((mount) =>
      mount.name === "alias-target"
        ? { ...mount, root: conflictRoot, label: "models/karume-alias-conflict" }
        : mount,
    );
    await expect(hfServingTable(conflicting)).rejects.toThrow(/would serve different bytes/);
  });

  it("fails loudly when a referenced file is not the mount's own", async () => {
    const strayRoot = join(dir, "karume-alias-stray");
    await mkdir(strayRoot, { recursive: true });
    // Its manifest takes the encoder parts from yet another repo: they are not its own files.
    await writeFile(
      join(strayRoot, "karume.json"),
      manifestText({ repo: "owner/elsewhere", revision: "c".repeat(40) }),
    );
    const stray = aliasMounts.map((mount) =>
      mount.name === "alias-target"
        ? { ...mount, root: strayRoot, label: "models/karume-alias-stray" }
        : mount,
    );
    await expect(hfServingTable(stray)).rejects.toThrow(/does not list that file as its own/);
  });
});
