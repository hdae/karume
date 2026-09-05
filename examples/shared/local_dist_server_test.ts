// 使い捨ての疑似 HF サーバの門（実 GPU も実資産も要らない — 127.0.0.1 の使い捨てポートと
// 数バイトのダミー shard だけ）。
//
// 唯一の消費者（`packages/models/tests/e2e_gemma4_pretrained_test.ts`）は実 GPU + 実資産が要る
// ので、パス門・配信表・起動時検査はここで縛る。観測はステータスコードと文言の部分一致。

import { assertEquals, assertStringIncludes, assertThrows } from "@std/assert";
import { serveLocalDist } from "./local-dist-server.ts";

const encoder = new TextEncoder();
/** サーバが名乗る主リポと固定 SHA（`local-dist-server.ts` の綴り）。 */
const REPO = "karume-local/dist";
const SHA = "0".repeat(40);
const CROSS_REPO = "someone/shared";
const CROSS_REVISION = "c".repeat(40);
const SHARD_PATH = "net/model.safetensors";
const CROSS_PATH = "text_encoder/model.safetensors";

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

/** `<root>/dist` に配布形を、`<root>/shared` に越境ミラーを置く（隣接同名ミラーの配置）。 */
const buildDist = async (
  root: string,
  options: { readonly cross?: readonly string[]; readonly mirror?: boolean } = {},
): Promise<string> => {
  const dir = `${root}/dist`;
  await Deno.mkdir(`${dir}/net`, { recursive: true });
  const shardBytes = encoder.encode("shard-bytes");
  await Deno.writeFile(`${dir}/${SHARD_PATH}`, shardBytes);
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
    await Deno.mkdir(`${root}/shared/text_encoder`, { recursive: true });
    await Deno.writeFile(`${root}/shared/${CROSS_PATH}`, encoder.encode("xxxx"));
  }
  const manifest = {
    format: "karume/4",
    generator: "karume/test",
    defaultModel: "m",
    models: {
      m: {
        pipeline: "anima/1",
        weights: { net: { f16: { shards: [await fileRef(SHARD_PATH, shardBytes)] } } },
        assets,
        quants: { f16: { weights: { net: "f16" }, session: {} } },
        defaultQuant: "f16",
        pipelineConfig: {},
      },
    },
  };
  await Deno.writeTextFile(`${dir}/karume.json`, JSON.stringify(manifest));
  return dir;
};

const withRoot = async (fn: (root: string) => Promise<void>): Promise<void> => {
  const root = await Deno.makeTempDir({ prefix: "karume-dist-server-test-" });
  try {
    await fn(root);
  } finally {
    await Deno.remove(root, { recursive: true });
  }
};

/** 本文まで読み切る（読み残すとテストのリソース検査に引っかかる）。 */
const get = async (
  url: string,
): Promise<{ status: number; body: string; length: string | null }> => {
  const response = await fetch(url);
  const length = response.headers.get("content-length");
  return { status: response.status, body: await response.text(), length };
};

Deno.test("serveLocalDist: revision 解決は 40 桁の固定 SHA を返す", async () => {
  await withRoot(async (root) => {
    const dir = await buildDist(root);
    await using server = serveLocalDist(dir);
    const hub = server.source.hubUrl;
    const { status, body } = await get(`${hub}/api/models/${REPO}/revision/main`);
    assertEquals(status, 200);
    assertEquals(JSON.parse(body), { sha: SHA });
  });
});

Deno.test("serveLocalDist: resolve は 200 で返し content-length が実ファイル長と一致する", async () => {
  await withRoot(async (root) => {
    const dir = await buildDist(root);
    await using server = serveLocalDist(dir);
    const hub = server.source.hubUrl;
    const { status, body, length } = await get(`${hub}/${REPO}/resolve/${SHA}/${SHARD_PATH}`);
    assertEquals(status, 200);
    assertEquals(body, "shard-bytes");
    assertEquals(length, String(encoder.encode("shard-bytes").byteLength));
  });
});

Deno.test("serveLocalDist: 配布形の外へ出る path は 404（`..` / 先頭 / / 空セグメント）", async () => {
  await withRoot(async (root) => {
    const dir = await buildDist(root);
    await using server = serveLocalDist(dir);
    const hub = server.source.hubUrl;
    for (
      const spelled of [
        "..%2F..%2Fetc%2Fpasswd",
        "%2e%2e%2ffoo",
        "%2Fetc%2Fpasswd",
        "net%2F%2Fmodel.safetensors",
      ]
    ) {
      const { status } = await get(`${hub}/${REPO}/resolve/${SHA}/${spelled}`);
      assertEquals(status, 404, `${spelled} が 404 でない`);
    }
  });
});

Deno.test("serveLocalDist: revision が宣言と違えば 404、未知の repo も 404", async () => {
  await withRoot(async (root) => {
    const dir = await buildDist(root);
    await using server = serveLocalDist(dir);
    const hub = server.source.hubUrl;
    assertEquals(
      (await get(`${hub}/${REPO}/resolve/${"d".repeat(40)}/${SHARD_PATH}`)).status,
      404,
    );
    assertEquals((await get(`${hub}/someone/other/resolve/${SHA}/${SHARD_PATH}`)).status, 404);
    assertEquals((await get(`${hub}/api/models/someone/other/revision/main`)).status, 404);
  });
});

Deno.test("serveLocalDist: 越境参照の revision が 1 リポで割れていたら起動時に落ちる", async () => {
  await withRoot(async (root) => {
    const dir = await buildDist(root, {
      cross: [CROSS_REVISION, "e".repeat(40)],
      mirror: true,
    });
    assertThrows(
      () => serveLocalDist(dir),
      Error,
      "越境参照の revision が 1 リポで割れている",
    );
  });
});

Deno.test("serveLocalDist: 越境参照のミラーが無ければ起動時に落ちる（実行中の 404 にしない）", async () => {
  await withRoot(async (root) => {
    const dir = await buildDist(root, { cross: [CROSS_REVISION] });
    const error = assertThrows(
      () => serveLocalDist(dir),
      Error,
      "越境参照のミラーが無い",
    );
    assertStringIncludes(error.message, CROSS_PATH);
  });
});
