// tools 共有の資産解決（`resolveAsset` / `readIrGraph`）の**失敗経路**の門。合成 manifest を
// 一時ディレクトリへ置くだけなので GPU も実資産も要らない。
//
// 成功経路は census_test.ts（配布形）と enumerate_test.ts（実資産）が押さえている。ここが
// 見るのは「未対応・想定外は fail loudly」— 選択が外れたときに、理由と既知一覧が出ること。

import { assertEquals, assertRejects } from "@std/assert";
import { readIrGraph, resolveAsset } from "./assets.ts";
import {
  type ModelInput,
  writeModelContainer,
} from "../../packages/runtime/tests/helpers/container-write.ts";
import { type IrDeclaration, parseIrDeclaration } from "../../packages/runtime/src/format/ir.ts";

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

const partRef = (path: string, cross?: Json): Json => ({
  path,
  size: 1,
  sha256: "0".repeat(64),
  ...(cross ?? {}),
});

/** part 列 1 本ぶんの容器（`resolveAsset` が見るのは part 0 の path だけ）。 */
const container = (parts: readonly Json[]): Json => ({
  descriptor: {
    graph: { length: 1, sha256: "1".repeat(64) },
    model: { length: 1, sha256: "2".repeat(64) },
  },
  parts,
});

/** 1 model / 1 部品 / 1 quant の最小 manifest（各ケースが 1 箇所だけ壊す）。 */
const baseManifest = (): Json => ({
  format: "karume/5",
  defaultModel: "m",
  models: {
    m: {
      pipeline: "anima/1",
      defaultQuant: "i8",
      quants: { i8: { weights: { model: "i8" } } },
      weights: {
        model: {
          i8: {
            container: container([
              partRef("model/model.i8-00001-of-00002.krm"),
              partRef("model/model.i8-00002-of-00002.krm"),
            ]),
          },
        },
      },
    },
  },
});

const writeManifest = async (dir: URL, manifest: Json): Promise<void> => {
  await Deno.writeTextFile(new URL("karume.json", dir), JSON.stringify(manifest));
};

/** initializer 1 本を 1 ノードで消費する最小の IR v2 グラフ（合成容器の中身）。 */
const declaration = (initializer: string): IrDeclaration =>
  parseIrDeclaration(JSON.stringify({
    format: "karume-ir",
    version: 2,
    requires: { ops: ["matmul"] },
    symbols: ["T"],
    inputs: [{ name: "x", dtype: "f32", shape: ["T", 4] }],
    outputs: ["y"],
    initializers: { [initializer]: {} },
    values: {
      [initializer]: { dtype: "f32", shape: [4, 4] },
      y: { dtype: "f32", shape: ["T", 4] },
    },
    nodes: [{ op: "matmul", ins: ["x", initializer], outs: ["y"], attrs: {} }],
  }));

/** 名前を挙げたグラフを 1 本ずつ持つ `krm` を、part 0 だけ渡せる分割形で書く。 */
const writeContainer = async (url: URL, graphNames: readonly string[]): Promise<void> => {
  const input: ModelInput = {
    graphs: Object.fromEntries(
      graphNames.map((name) => [name, declaration(`${name}.weight`)]),
    ),
    consts: [],
    weights: graphNames.map((name) => ({
      graph: name,
      initializer: `${name}.weight`,
      bytes: new Uint8Array(new ArrayBuffer(4 * 4 * 4)),
      encoding: { codec: "f32" },
    })),
    assets: [],
    provenance: { license: "test", writer: "karume-test/1" },
  };
  const written = await writeModelContainer(input);
  // part 0 だけを据える（`readIrGraph` が読むのはヘッダ + 2 文書だけ）。
  await Deno.writeFile(url, written.parts[0]);
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

Deno.test("resolveAsset（配布形）: 旧 major の manifest は読めないと名指しで落ちる", async () => {
  await withDir(async (dir) => {
    const manifest = baseManifest();
    manifest.format = "karume/4";
    await writeManifest(dir, manifest);
    await assertRejects(
      () => resolveAsset(dir, undefined, undefined, undefined),
      Error,
      "format 'karume/4' はこの版が読めない",
    );
  });
});

Deno.test("resolveAsset（配布形）: container.parts が空なら診断つきで落ちる（TypeError にしない）", async () => {
  await withDir(async (dir) => {
    const manifest = baseManifest();
    const model = (manifest.models as Json).m as Json;
    model.weights = { model: { i8: { container: container([]) } } };
    await writeManifest(dir, manifest);
    await assertRejects(
      () => resolveAsset(dir, undefined, undefined, undefined),
      Error,
      "manifest の container.parts が空",
    );
  });
});

Deno.test("resolveAsset（配布形）: part 0 が越境参照なら --source の案内つきで落ちる", async () => {
  await withDir(async (dir) => {
    const manifest = baseManifest();
    const model = (manifest.models as Json).m as Json;
    const cross = { repo: "hdae/other", revision: "c".repeat(40) };
    model.weights = {
      model: {
        i8: {
          container: container([
            partRef("model/model.i8-00001-of-00002.krm", cross),
            partRef("model/model.i8-00002-of-00002.krm", cross),
          ]),
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
    await Deno.writeFile(new URL("net/model.f16-00001-of-00002.krm", dir), new Uint8Array(0));
    await Deno.writeFile(new URL("net/model.f16-00002-of-00002.krm", dir), new Uint8Array(0));
    await Deno.writeFile(new URL("net/model.i8-00001-of-00002.krm", dir), new Uint8Array(0));
    await Deno.writeFile(new URL("net/model.i8-00002-of-00002.krm", dir), new Uint8Array(0));
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
    await Deno.writeFile(new URL("model-00001-of-00002.krm", dir), new Uint8Array(0));
    await Deno.writeFile(new URL("model-00002-of-00002.krm", dir), new Uint8Array(0));
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
    await Deno.writeFile(new URL("model-00001-of-00002.krm", root), new Uint8Array(0));
    await Deno.writeFile(new URL("model-00002-of-00002.krm", root), new Uint8Array(0));
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

Deno.test("readIrGraph: 名指しのグラフが無い容器は既知一覧つきで落ちる（別の部品を数えない）", async () => {
  await withDir(async (dir) => {
    const url = new URL("model-00001-of-00003.krm", dir);
    await writeContainer(url, ["vision"]);
    await assertRejects(
      () => readIrGraph({ url, graph: "text_encoder" }),
      Error,
      "容器にグラフ 'text_encoder' が無い",
    );
    await assertRejects(
      () => readIrGraph({ url, graph: "text_encoder" }),
      Error,
      "在るのは vision",
    );
  });
});

Deno.test("readIrGraph: 名指しが無く 2 グラフある容器は落ちる（1 本目を黙って採らない）", async () => {
  await withDir(async (dir) => {
    const url = new URL("model-00001-of-00004.krm", dir);
    await writeContainer(url, ["front", "voice"]);
    await assertRejects(() => readIrGraph({ url }), Error, "グラフが 2 本ある");
  });
});

Deno.test("readIrGraph: 名指しの無い 1 グラフの容器は、その唯一のグラフを読む", async () => {
  await withDir(async (dir) => {
    const url = new URL("model-00001-of-00003.krm", dir);
    await writeContainer(url, ["caption_proj"]);
    const graph = await readIrGraph({ url });
    assertEquals(Object.keys(graph.initializers), ["caption_proj.weight"]);
  });
});
