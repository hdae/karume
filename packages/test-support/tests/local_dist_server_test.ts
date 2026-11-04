// Checks for the throwaway fake HF server (needs neither a real GPU nor real assets — only a
// throwaway port on 127.0.0.1 and dummy parts of a few bytes).
//
// The only consumer (`packages/models/tests/e2e_gemma4_pretrained_test.ts`) needs a real GPU + real
// assets, so the path check, the serving table and the startup validation are pinned here. What is
// observed is the status code and a substring of the message.

import { makeTempDir, mkdir, remove, writeFile, writeTextFile } from "@karume/test-support/fs";
import { test } from "vite-plus/test";
import { assertEquals, assertStringIncludes, assertThrows } from "@std/assert";
import { serveLocalDist } from "@karume/test-support/local-dist-server";

const encoder = new TextEncoder();
/** The main repo and the fixed SHA the server reports (spelled as in `local-dist-server.ts`). */
const REPO = "karume-local/dist";
const SHA = "0".repeat(40);
const CROSS_REPO = "someone/shared";
const CROSS_REVISION = "c".repeat(40);
const PART0_PATH = "net/model.f16-00001-of-00002.krm";
const PART1_PATH = "net/model.f16-00002-of-00002.krm";
const CROSS_PATH = "text_encoder/model.safetensors";

/** Container header length (container-v1 §1). Part 0 is exactly "header + the 2 documents". */
const HEADER_BYTES = 24;
/**
 * Lengths of the 2 documents of the synthetic descriptor (their content is only served — the
 * manifest check is on lengths).
 */
const GRAPH_BYTES = 10;
const MODEL_BYTES = 12;

const sha256Hex = async (bytes: Uint8Array<ArrayBuffer>): Promise<string> =>
  [...new Uint8Array(await crypto.subtle.digest("SHA-256", bytes))]
    .map((byte) => byte.toString(16).padStart(2, "0"))
    .join("");

type Json = Record<string, unknown>;

const fileRef = async (path: string, bytes: Uint8Array<ArrayBuffer>): Promise<Json> => ({
  path,
  size: bytes.byteLength,
  sha256: await sha256Hex(bytes),
});

/**
 * Puts the distribution in `<root>/dist` and the cross-repo mirror in `<root>/shared` (the sibling
 * same-name mirror layout).
 */
const buildDist = async (
  root: string,
  options: { readonly cross?: readonly string[]; readonly mirror?: boolean } = {},
): Promise<string> => {
  const dir = `${root}/dist`;
  await mkdir(`${dir}/net`, { recursive: true });
  const part0Bytes = new Uint8Array(new ArrayBuffer(HEADER_BYTES + GRAPH_BYTES + MODEL_BYTES)).fill(
    0x41,
  );
  const part1Bytes = encoder.encode("part-bytes");
  await writeFile(`${dir}/${PART0_PATH}`, part0Bytes);
  await writeFile(`${dir}/${PART1_PATH}`, part1Bytes);
  const assets: Json = {};
  for (const [at, revision] of (options.cross ?? []).entries()) {
    assets[`cross${at}`] = {
      path: CROSS_PATH,
      size: 4,
      sha256: "1".repeat(64),
      repo: CROSS_REPO,
      revision,
    };
  }
  if (options.mirror === true) {
    await mkdir(`${root}/shared/text_encoder`, { recursive: true });
    await writeFile(`${root}/shared/${CROSS_PATH}`, encoder.encode("xxxx"));
  }
  const manifest = {
    format: "karume/5",
    generator: "karume/test",
    defaultModel: "m",
    models: {
      m: {
        pipeline: "anima/1",
        weights: {
          net: {
            f16: {
              container: {
                descriptor: {
                  graph: { length: GRAPH_BYTES, sha256: "1".repeat(64) },
                  model: { length: MODEL_BYTES, sha256: "2".repeat(64) },
                },
                parts: [
                  await fileRef(PART0_PATH, part0Bytes),
                  await fileRef(PART1_PATH, part1Bytes),
                ],
              },
            },
          },
        },
        assets,
        quants: { f16: { weights: { net: "f16" }, session: {} } },
        defaultQuant: "f16",
        pipelineConfig: {},
      },
    },
  };
  await writeTextFile(`${dir}/karume.json`, JSON.stringify(manifest));
  return dir;
};

const withRoot = async (fn: (root: string) => Promise<void>): Promise<void> => {
  const root = await makeTempDir({ prefix: "karume-dist-server-test-" });
  try {
    await fn(root);
  } finally {
    await remove(root, { recursive: true });
  }
};

/** Reads the body to the end (a body left unread trips the test's resource check). */
const get = async (
  url: string,
): Promise<{ status: number; body: string; length: string | null }> => {
  const response = await fetch(url);
  const length = response.headers.get("content-length");
  return { status: response.status, body: await response.text(), length };
};

test("serveLocalDist: revision resolution returns the fixed 40-digit SHA", async () => {
  await withRoot(async (root) => {
    const dir = await buildDist(root);
    await using server = await serveLocalDist(dir);
    const hub = server.source.hubUrl;
    const { status, body } = await get(`${hub}/api/models/${REPO}/revision/main`);
    assertEquals(status, 200);
    assertEquals(JSON.parse(body), { sha: SHA });
  });
});

test("serveLocalDist: resolve returns 200 and content-length matches the real file length", async () => {
  await withRoot(async (root) => {
    const dir = await buildDist(root);
    await using server = await serveLocalDist(dir);
    const hub = server.source.hubUrl;
    const { status, body, length } = await get(`${hub}/${REPO}/resolve/${SHA}/${PART1_PATH}`);
    assertEquals(status, 200);
    assertEquals(body, "part-bytes");
    assertEquals(length, String(encoder.encode("part-bytes").byteLength));
  });
});

test("serveLocalDist: a path that leaves the distribution is 404 (`..` / leading / / empty segment)", async () => {
  await withRoot(async (root) => {
    const dir = await buildDist(root);
    await using server = await serveLocalDist(dir);
    const hub = server.source.hubUrl;
    for (const spelled of [
      "..%2F..%2Fetc%2Fpasswd",
      "%2e%2e%2ffoo",
      "%2Fetc%2Fpasswd",
      "net%2F%2Fmodel.f16-00002-of-00002.krm",
    ]) {
      const { status } = await get(`${hub}/${REPO}/resolve/${SHA}/${spelled}`);
      assertEquals(status, 404, `${spelled} is not 404`);
    }
  });
});

test("serveLocalDist: a revision other than the declared one is 404, and so is an unknown repo", async () => {
  await withRoot(async (root) => {
    const dir = await buildDist(root);
    await using server = await serveLocalDist(dir);
    const hub = server.source.hubUrl;
    assertEquals((await get(`${hub}/${REPO}/resolve/${"d".repeat(40)}/${PART1_PATH}`)).status, 404);
    assertEquals((await get(`${hub}/someone/other/resolve/${SHA}/${PART1_PATH}`)).status, 404);
    assertEquals((await get(`${hub}/api/models/someone/other/revision/main`)).status, 404);
  });
});

test("serveLocalDist: fails at startup when cross-repo references to one repo disagree on the revision", async () => {
  await withRoot(async (root) => {
    const dir = await buildDist(root, {
      cross: [CROSS_REVISION, "e".repeat(40)],
      mirror: true,
    });
    assertThrows(
      () => serveLocalDist(dir),
      Error,
      "the revisions of cross-repo references to one repo disagree",
    );
  });
});

test("serveLocalDist: fails at startup when a cross-repo reference has no mirror (not a 404 at run time)", async () => {
  await withRoot(async (root) => {
    const dir = await buildDist(root, { cross: [CROSS_REVISION] });
    const error = assertThrows(
      () => serveLocalDist(dir),
      Error,
      "no mirror for a cross-repo reference",
    );
    assertStringIncludes(error.message, CROSS_PATH);
  });
});
