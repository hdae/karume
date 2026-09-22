// ローカル配布形の判定と**全量読み**の門（GPU も実資産も要らない — 合成 manifest と数バイトの
// ダミー part だけ）。
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

const PARTS = ["net/model.f16-00001-of-00002.krm", "net/model.f16-00002-of-00002.krm"];

/** コンテナのヘッダ長（container-v1 §1）。part 0 は「ヘッダ + 2 文書ちょうど」。 */
const HEADER_BYTES = 24;
/** 合成 descriptor の 2 文書の長さ（中身は読まれない — 長さの整合だけが manifest の門）。 */
const GRAPH_BYTES = 10;
const MODEL_BYTES = 12;

type Json = Record<string, unknown>;

/**
 * 2 part の最小配布形を書く。`tokenizer` は **part 0 と同じ path** を指すので、キーが 2 つで
 * 実体が 1 つという形（メモ化の観測点）になる。`cross` を真にすると越境参照の asset が 1 本乗る。
 */
const writeDist = async (dir: string, options: { cross?: boolean } = {}): Promise<void> => {
  await Deno.mkdir(`${dir}/net`, { recursive: true });
  const refs: Json[] = [];
  for (const [at, path] of PARTS.entries()) {
    // part 0 だけは「ヘッダ + 2 文書」の長さちょうどでないと manifest の門が通らない。
    const bytes = at === 0
      ? new Uint8Array(new ArrayBuffer(HEADER_BYTES + GRAPH_BYTES + MODEL_BYTES)).fill(0x41)
      : encoder.encode(`part-${at}`);
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
                parts: refs,
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
  await Deno.writeTextFile(`${dir}/${MANIFEST_FILE}`, JSON.stringify(manifest));
};

Deno.test("isLocalDist: karume.json の有無でローカル配布形かを決める", async () => {
  await withDir(async (dir) => {
    assertEquals(await isLocalDist(dir), false);
    await writeDist(dir);
    assertEquals(await isLocalDist(dir), true);
  });
});

Deno.test("loadLocalAssets: part 列を添字順に並べ、同じ path を指すキーは同一のバイト列を受ける", async () => {
  await withDir(async (dir) => {
    await writeDist(dir);
    const { assets } = await loadLocalAssets(dir);
    assertEquals(Object.keys(assets).sort(), ["net[0]", "net[1]", "tokenizer"]);
    assertEquals(new TextDecoder().decode(assets["net[1]"]), "part-1");
    // `tokenizer` は part 0 と同じ path なので、読み返しは 1 回きり（同一参照）。
    assertEquals(assets["tokenizer"] === assets["net[0]"], true);
    assertEquals(assets["net[1]"] === assets["net[0]"], false);
  });
});

Deno.test("loadLocalAssets: 越境参照を含む配布形はキーつきで落ちる（黙って別リポの path を開かない）", async () => {
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
