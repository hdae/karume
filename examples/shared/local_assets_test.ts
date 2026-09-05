// ローカル配布形の判定と**全量読み**の門（GPU も実資産も要らない — 合成 manifest と数バイトの
// ダミー shard だけ）。
//
// 全量読みの消費者は `examples/sbv2/dump.ts` 1 本だが、越境参照（`FileRef.repo`）を無視すると
// **同名 path のローカルファイルが別リポのバイト列に化ける**ので、その門をここで縛る。

import { assertEquals, assertRejects } from "@std/assert";
import { isLocalDist, loadLocalAssets, MANIFEST_FILE } from "./local-assets.ts";

const encoder = new TextEncoder();

const sha256Hex = async (bytes: Uint8Array<ArrayBuffer>): Promise<string> =>
  [...new Uint8Array(await crypto.subtle.digest("SHA-256", bytes))]
    .map((byte) => byte.toString(16).padStart(2, "0"))
    .join("");

const withDir = async (fn: (dir: string) => Promise<void>): Promise<void> => {
  const dir = await Deno.makeTempDir({ prefix: "karume-local-assets-test-" });
  try {
    await fn(dir);
  } finally {
    await Deno.remove(dir, { recursive: true });
  }
};

const SHARDS = ["net/model-00001-of-00002.safetensors", "net/model-00002-of-00002.safetensors"];

type Json = Record<string, unknown>;

/**
 * 2 shard の最小配布形を書く。`tokenizer` は **1 本目の shard と同じ path** を指すので、
 * 取得キーが 2 つで実体が 1 つという形（メモ化の観測点）になる。
 * `cross` を真にすると越境参照の asset が 1 本乗る。
 */
const writeDist = async (dir: string, options: { cross?: boolean } = {}): Promise<void> => {
  await Deno.mkdir(`${dir}/net`, { recursive: true });
  const refs: Json[] = [];
  for (const [at, path] of SHARDS.entries()) {
    const bytes = encoder.encode(`shard-${at}`);
    await Deno.writeFile(`${dir}/${path}`, bytes);
    refs.push({ path, size: bytes.byteLength, sha256: await sha256Hex(bytes) });
  }
  const assets: Json = { tokenizer: refs[0] };
  if (options.cross === true) {
    assets.text_encoder = {
      path: "text_encoder/model.safetensors",
      size: 4,
      sha256: "0".repeat(64),
      repo: "someone/shared",
      revision: "c".repeat(40),
    };
  }
  const manifest = {
    format: "karume/4",
    generator: "karume/test",
    defaultModel: "m",
    models: {
      m: {
        pipeline: "anima/1",
        weights: { net: { f16: { shards: refs } } },
        assets,
        quants: { f16: { weights: { net: "f16" }, session: {} } },
        defaultQuant: "f16",
        pipelineConfig: {},
      },
    },
  };
  await Deno.writeTextFile(`${dir}/${MANIFEST_FILE}`, JSON.stringify(manifest));
};

Deno.test("isLocalDist: karume.json の有無でローカル配布形かを決める", async () => {
  await withDir(async (dir) => {
    assertEquals(await isLocalDist(dir), false);
    await writeDist(dir);
    assertEquals(await isLocalDist(dir), true);
  });
});

Deno.test("loadLocalAssets: 取得キーをそのまま並べ、同じ path を指すキーは同一のバイト列を受ける", async () => {
  await withDir(async (dir) => {
    await writeDist(dir);
    const { assets } = await loadLocalAssets(dir);
    assertEquals(Object.keys(assets).sort(), ["net[0]", "net[1]", "tokenizer"]);
    assertEquals(new TextDecoder().decode(assets["net[0]"]), "shard-0");
    // `tokenizer` は 1 本目の shard と同じ path なので、読み返しは 1 回きり（同一参照）。
    assertEquals(assets["tokenizer"] === assets["net[0]"], true);
    assertEquals(assets["net[1]"] === assets["net[0]"], false);
  });
});

Deno.test("loadLocalAssets: 越境参照を含む配布形は取得キーつきで落ちる（黙って別リポの path を開かない）", async () => {
  await withDir(async (dir) => {
    await writeDist(dir, { cross: true });
    const error = await assertRejects(
      () => loadLocalAssets(dir),
      Error,
      "越境参照 'someone/shared' を解けない",
    );
    assertEquals(error.message.includes("text_encoder"), true);
  });
});
