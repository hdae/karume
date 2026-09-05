// tools 共有の資産解決（`resolveAsset` / `readIrGraph`）の**失敗経路**の門。合成 manifest を
// 一時ディレクトリへ置くだけなので GPU も実資産も要らない。
//
// 成功経路は census_test.ts（配布形）と enumerate_test.ts（実資産）が押さえている。ここが
// 見るのは「未対応・想定外は fail loudly」— 選択が外れたときに、理由と既知一覧が出ること。

import { assertRejects } from "@std/assert";
import { readIrGraph, resolveAsset } from "./assets.ts";

/** 一時ディレクトリを 1 つ作って渡す（終わったら消す）。 */
const withDir = async (fn: (dir: URL) => Promise<void>): Promise<void> => {
  const path = await Deno.makeTempDir({ prefix: "karume-assets-test-" });
  try {
    await fn(new URL(`file://${path}/`));
  } finally {
    await Deno.remove(path, { recursive: true });
  }
};

type Json = Record<string, unknown>;

const shardRef = (path: string, cross?: Json): Json => ({
  path,
  size: 1,
  sha256: "0".repeat(64),
  ...(cross ?? {}),
});

/** 1 model / 1 component / 1 quant の最小 manifest（各ケースが 1 箇所だけ壊す）。 */
const baseManifest = (): Json => ({
  format: "karume/4",
  defaultModel: "m",
  models: {
    m: {
      pipeline: "anima/1",
      defaultQuant: "i8",
      quants: { i8: { weights: { model: "i8" } } },
      weights: { model: { i8: { shards: [shardRef("model/model.i8.safetensors")] } } },
    },
  },
});

const writeManifest = async (dir: URL, manifest: Json): Promise<void> => {
  await Deno.writeTextFile(new URL("karume.json", dir), JSON.stringify(manifest));
};

/** safetensors のヘッダ（8 バイト長 + JSON）だけを持つファイル。実体テンソルは書かない。 */
const writeHeaderOnly = async (url: URL, header: Json): Promise<void> => {
  const json = new TextEncoder().encode(JSON.stringify(header));
  const bytes = new Uint8Array(8 + json.length);
  new DataView(bytes.buffer).setBigUint64(0, BigInt(json.length), true);
  bytes.set(json, 8);
  await Deno.writeFile(url, bytes);
};

Deno.test("resolveAsset（配布形）: model の選択が外れたら既知一覧つきで落ちる", async () => {
  await withDir(async (dir) => {
    await writeManifest(dir, baseManifest());
    await assertRejects(
      () => resolveAsset(dir, "x", undefined, undefined),
      Error,
      "model 'x' が無い",
    );
    await assertRejects(() => resolveAsset(dir, "x", undefined, undefined), Error, "既知: m");
  });
});

Deno.test("resolveAsset（配布形）: quant の選択が外れたら既知一覧つきで落ちる", async () => {
  await withDir(async (dir) => {
    await writeManifest(dir, baseManifest());
    await assertRejects(
      () => resolveAsset(dir, undefined, "i4", undefined),
      Error,
      "quant 'i4' が無い",
    );
    await assertRejects(() => resolveAsset(dir, undefined, "i4", undefined), Error, "既知: i8");
  });
});

Deno.test("resolveAsset（配布形）: quant が component の格納 dtype を選んでいないと落ちる", async () => {
  await withDir(async (dir) => {
    const manifest = baseManifest();
    const models = manifest.models as Json;
    const model = models.m as Json;
    model.quants = { i8: { weights: {} } };
    await writeManifest(dir, manifest);
    await assertRejects(
      () => resolveAsset(dir, undefined, undefined, undefined),
      Error,
      "格納 dtype を選んでいない",
    );
  });
});

Deno.test("resolveAsset（配布形）: 選ばれた格納 dtype が weights に無いと既知一覧つきで落ちる", async () => {
  await withDir(async (dir) => {
    const manifest = baseManifest();
    const model = (manifest.models as Json).m as Json;
    model.quants = { i8: { weights: { model: "f16" } } };
    await writeManifest(dir, manifest);
    await assertRejects(
      () => resolveAsset(dir, undefined, undefined, undefined),
      Error,
      "格納 dtype 'f16' が無い",
    );
  });
});

Deno.test("resolveAsset（配布形）: shards が空なら診断つきで落ちる（TypeError にしない）", async () => {
  await withDir(async (dir) => {
    const manifest = baseManifest();
    const model = (manifest.models as Json).m as Json;
    model.weights = { model: { i8: { shards: [] } } };
    await writeManifest(dir, manifest);
    await assertRejects(
      () => resolveAsset(dir, undefined, undefined, undefined),
      Error,
      "manifest の shards が空",
    );
  });
});

Deno.test("resolveAsset（配布形）: 先頭 shard が越境参照なら --source の案内つきで落ちる", async () => {
  await withDir(async (dir) => {
    const manifest = baseManifest();
    const model = (manifest.models as Json).m as Json;
    model.weights = {
      model: {
        i8: {
          shards: [
            shardRef("model/model.i8.safetensors", {
              repo: "hdae/other",
              revision: "c".repeat(40),
            }),
          ],
        },
      },
    };
    await writeManifest(dir, manifest);
    const error = await assertRejects(
      () => resolveAsset(dir, undefined, undefined, undefined),
      Error,
      "越境参照",
    );
    if (!error.message.includes("--source")) {
      throw new Error(`案内に --source が無い: ${error.message}`);
    }
  });
});

Deno.test("resolveAsset（系列出力）: 格納 dtype グループが複数あるなら --quant を促して落ちる", async () => {
  await withDir(async (dir) => {
    await Deno.mkdir(new URL("net/", dir));
    await Deno.writeFile(new URL("net/model.f16.safetensors", dir), new Uint8Array(0));
    await Deno.writeFile(new URL("net/model.i8.safetensors", dir), new Uint8Array(0));
    await assertRejects(
      () => resolveAsset(dir, undefined, undefined, "anima"),
      Error,
      "格納 dtype が複数ある",
    );
    await assertRejects(
      () => resolveAsset(dir, undefined, undefined, "anima"),
      Error,
      "--quant で 1 つ選ぶ",
    );
  });
});

Deno.test("resolveAsset（系列出力）: --model は配布形だけのノブなので落ちる", async () => {
  await withDir(async (dir) => {
    await Deno.writeFile(new URL("model.safetensors", dir), new Uint8Array(0));
    await assertRejects(
      () => resolveAsset(dir, "m", undefined, "anima"),
      Error,
      "系列出力には model の選択が無い",
    );
  });
});

Deno.test("resolveAsset（系列出力）: ディレクトリ名から家族名を推せないなら --family を促す", async () => {
  await withDir(async (dir) => {
    const root = new URL("foo-bar/", dir);
    await Deno.mkdir(root);
    await Deno.writeFile(new URL("model.safetensors", root), new Uint8Array(0));
    await assertRejects(
      () => resolveAsset(root, undefined, undefined, undefined),
      Error,
      "家族名を推せない",
    );
    await assertRejects(
      () => resolveAsset(root, undefined, undefined, undefined),
      Error,
      "--family で明示",
    );
  });
});

Deno.test("readIrGraph: __metadata__.karume_ir を持たない shard は落ちる（空グラフを出さない）", async () => {
  await withDir(async (dir) => {
    const url = new URL("model.safetensors", dir);
    await writeHeaderOnly(url, {
      "some.weight": { dtype: "F32", shape: [1], data_offsets: [0, 4] },
    });
    await assertRejects(() => readIrGraph(url), Error, "__metadata__.karume_ir が無い");
  });
});
