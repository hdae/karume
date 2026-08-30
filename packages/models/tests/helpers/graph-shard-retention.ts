/**
 * グラフ shard の常駐量の実測（`shard_loading_test.ts` が別プロセスで起動する台本）。
 *
 * MUST: 別プロセスなのは `--v8-flags=--expose-gc` が要るため — `deno test` 側にその旗を
 * 立てる口が無く、旗が無いと「到達不能になったバイト列がまだ回収されていないだけ」と
 * 「握られたまま」を区別できない（測る前に必ず gc を 1 回踏む）。
 *
 * 測るのは `Deno.memoryUsage().external`（ArrayBuffer の実体の合計）。**キャッシュの中身を
 * 明示的に捨ててから**測ることで、残る外部メモリ = models 側が握っているバイト列だけになる。
 * `open(key)` は保持したまま測る（コンポーネントの供給口が生きている限り常駐する、という
 * 元の姿を再現するため）。
 *
 * 失敗時は理由を stderr へ出して非ゼロ終了する（呼び手はその出力をそのまま見せる）。
 */

import { loadManifest, resolveFiles } from "@karume/hub";
import { loadShardComponents } from "../../src/hub/components.ts";
import { MemoryCacheStorage } from "./memory-cache.ts";
import { type DumpTensor, writeSafetensors } from "./safetensors-write.ts";

const HUB_URL = "https://hub.test";
const REPO = "karume-test/retention";
const SHA = "0123456789abcdef0123456789abcdef01234567";
const MANIFEST_PATH = "karume.json";

const REVISION_RE = /^\/api\/models\/(.+)\/revision\/(.+)$/;
const RESOLVE_RE = /^\/(.+?)\/resolve\/([^/]+)\/(.+)$/;

/** グラフ shard に載せる実重みの大きさ（4096² f32 = 64MiB — 実配布形は最大 1GiB）。 */
const WIDTH = 4096;
const GRAPH_SHARD_WEIGHT_BYTES = WIDTH * WIDTH * 4;
/** 合格線: gc 後に残ってよい外部メモリ。握っていれば 64MiB が丸ごと残る。 */
const ALLOWED_RESIDENT_BYTES = 16 * 1024 * 1024;

const f32Tensor = (shape: readonly number[], value: number): DumpTensor => ({
  dtype: "F32",
  shape: [...shape],
  data: new Float32Array(shape.reduce((product, dim) => product * dim, 1)).fill(value),
});

/** `linear` 1 段。`w`（大きい方）がグラフ shard に、`b` が重み shard に載る。 */
const miniGraph = (): unknown => ({
  format: "karume-ir",
  version: 1,
  requires: { ops: ["linear"] },
  symbols: [],
  inputs: [{ name: "x", dtype: "f32", shape: [2, WIDTH] }],
  outputs: ["y"],
  initializers: {
    w: { tensor: "m.w", storage: { dtype: "f32" } },
    b: { tensor: "m.b", storage: { dtype: "f32" } },
  },
  values: {
    w: { dtype: "f32", shape: [WIDTH, WIDTH] },
    b: { dtype: "f32", shape: [WIDTH] },
    y: { dtype: "f32", shape: [2, WIDTH] },
  },
  states: {},
  nodes: [{ op: "linear", ins: ["x", "w", "b"], outs: ["y"], attrs: {} }],
});

const sha256Hex = async (bytes: Uint8Array<ArrayBuffer>): Promise<string> =>
  Array.from(new Uint8Array(await crypto.subtle.digest("SHA-256", bytes)))
    .map((byte) => byte.toString(16).padStart(2, "0"))
    .join("");

const fileRef = async (
  path: string,
  bytes: Uint8Array<ArrayBuffer>,
): Promise<{ path: string; size: number; sha256: string }> => ({
  path,
  size: bytes.byteLength,
  sha256: await sha256Hex(bytes),
});

const mib = (bytes: number): string => `${(bytes / 1024 / 1024).toFixed(1)}MiB`;

/** 到達不能になった ArrayBuffer を確実に回収させてから外部メモリを読む。 */
const externalAfterGc = (): number => {
  const collect = (globalThis as { gc?: () => void }).gc;
  if (collect === undefined) {
    console.error("graph-shard-retention: --v8-flags=--expose-gc 付きで起動していない");
    Deno.exit(2);
  }
  collect();
  collect();
  return Deno.memoryUsage().external;
};

/**
 * 配る現物の表を作る。**フィクスチャのバイト列を `main` の枠に置かない**ための独立関数 —
 * 実行中の関数のローカルは（内側から参照されていなくても）その枠が生きている限り回収されず、
 * 64MiB のフィクスチャがそのまま測定値へ乗ってしまう。
 */
const buildServed = async (): Promise<Map<string, Uint8Array<ArrayBuffer>>> => {
  const graph = writeSafetensors(
    new Map([["m.w", f32Tensor([WIDTH, WIDTH], 0.5)]]),
    { karume_ir: JSON.stringify(miniGraph()) },
  );
  const weights = writeSafetensors(new Map([["m.b", f32Tensor([WIDTH], 0.25)]]), {});
  const refs = {
    graph: await fileRef("dit/model-00000.safetensors", graph),
    weights: await fileRef("dit/model-00001.safetensors", weights),
  };
  const manifest = new TextEncoder().encode(JSON.stringify({
    format: "karume/4",
    generator: "karume-test/0",
    defaultModel: "test",
    models: {
      test: {
        pipeline: "test/1",
        weights: { dit: { f32: { shards: [refs.graph, refs.weights] } } },
        assets: {},
        quants: { f32: { weights: { dit: "f32" }, session: {} } },
        defaultQuant: "f32",
        pipelineConfig: {},
      },
    },
  }));
  return new Map<string, Uint8Array<ArrayBuffer>>([
    [MANIFEST_PATH, manifest],
    [refs.graph.path, graph],
    [refs.weights.path, weights],
  ]);
};

const main = async (): Promise<void> => {
  const served = await buildServed();
  const fetchMock: typeof globalThis.fetch = (input) => {
    const href = typeof input === "string" ? input : input instanceof URL ? input.href : input.url;
    const { pathname } = new URL(href);
    if (REVISION_RE.test(pathname)) return Promise.resolve(Response.json({ sha: SHA }));
    const resolved = RESOLVE_RE.exec(pathname);
    if (resolved === null) return Promise.resolve(new Response("not found", { status: 404 }));
    const bytes = served.get(decodeURIComponent(resolved[3]));
    if (bytes === undefined) return Promise.resolve(new Response("not found", { status: 404 }));
    return Promise.resolve(
      new Response(bytes, { headers: { "content-length": String(bytes.byteLength) } }),
    );
  };

  const caches = new MemoryCacheStorage();
  const hubOptions = { fetch: fetchMock, caches };
  const loaded = await loadManifest({ repo: REPO, revision: SHA, hubUrl: HUB_URL }, hubOptions);
  const files = resolveFiles(loaded.manifest);
  const { open } = await loadShardComponents(
    "test.fromPretrained",
    loaded,
    files,
    ["dit"],
    hubOptions,
  );

  // 供給口は保持したまま、models の外にあるバイト列（フィクスチャ・キャッシュ）を捨てる —
  // 残る外部メモリ = models 側が握っているぶん、という等式を成立させるため。
  served.clear();
  for (const cache of caches.namespaces.values()) cache.entries.clear();

  const resident = externalAfterGc();
  // `open` を gc の後まで生かす（ここで初めて到達可能性が切れる）。
  if (open("dit").graph.outputs[0] !== "y") {
    console.error("graph-shard-retention: グラフ宣言が読めない（フィクスチャの誤り）");
    Deno.exit(2);
  }

  if (resident > ALLOWED_RESIDENT_BYTES) {
    console.error(
      `graph-shard-retention: グラフ shard が常駐している` +
        `（external=${mib(resident)} > 許容 ${mib(ALLOWED_RESIDENT_BYTES)}` +
        ` / グラフ shard の実重み ${mib(GRAPH_SHARD_WEIGHT_BYTES)}）`,
    );
    Deno.exit(1);
  }
  console.log(`graph-shard-retention: external=${mib(resident)}`);
};

await main();
