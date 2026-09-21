// shard 面のロード経路（`src/hub/components.ts`）の門が共有する疑似 HF 取得層と配布形
// （shard_loading_test.ts と gpu_shard_loading_test.ts）。
//
// NOTE: hub / runtime のテスト helper は import しない（向こうの都合がこちらへ漏れる —
// `helpers/memory-cache.ts` と同じ規律）。モックはこのファイル内で最小限だけ組む。

import {
  type FileRef,
  type LoadedManifest,
  loadManifest,
  type ResolvedFiles,
  resolveFiles,
} from "@karume/hub";
import { MemoryCacheStorage } from "./memory-cache.ts";
import { type DumpTensor, writeSafetensors } from "./safetensors-write.ts";

/** 疑似 HF の取得層（叩かれた path を記録する `fetch`）。 */
export type MockFetch = { readonly fetch: typeof globalThis.fetch; readonly paths: string[] };

/** `loadManifest` / `fromPretrained` へ渡す取得ノブ。 */
export type HubOptions = {
  readonly fetch: typeof globalThis.fetch;
  readonly caches: MemoryCacheStorage;
};

/** グラフ shard + 重み shard の 2 本に割った配布形（manifest まで解決済み）。 */
export type TwoShardRig = {
  readonly loaded: LoadedManifest;
  readonly files: ResolvedFiles;
  readonly refs: { readonly graph: FileRef; readonly weights: FileRef };
  readonly mock: MockFetch;
  readonly hubOptions: HubOptions;
};

/** siglip2 の配布形（家族 admission だけを踏ませる rig — Session は張らない）。 */
export type Siglip2Rig = {
  readonly refs: { readonly graph: FileRef; readonly weights: FileRef };
  readonly mock: MockFetch;
  readonly caches: MemoryCacheStorage;
};

/**
 * 家族 admission の席（この機構だけを見るテストは家族の門を 1 つも置かない）。実家族の門が
 * 席に載っていることは ⑦ が別に縛る。
 */
export const NO_FAMILY_GATE = (): undefined => undefined;

export const HUB_URL = "https://hub.test";
export const REPO = "karume-test/shards";
export const SHA = "0123456789abcdef0123456789abcdef01234567";
export const MANIFEST_PATH = "karume.json";

const REVISION_RE = /^\/api\/models\/(.+)\/revision\/(.+)$/;
const RESOLVE_RE = /^\/(.+?)\/resolve\/([^/]+)\/(.+)$/;

/** HF の 2 経路（revision 解決 API・resolve URL）だけを喋る `fetch`。叩かれた path を記録する。 */
export const createMockFetch = (
  files: ReadonlyMap<string, Uint8Array<ArrayBuffer>>,
): { fetch: typeof globalThis.fetch; paths: string[] } => {
  const paths: string[] = [];
  const fetch: typeof globalThis.fetch = (input) => {
    const href = typeof input === "string" ? input : input instanceof URL ? input.href : input.url;
    const { pathname } = new URL(href);
    if (REVISION_RE.test(pathname)) return Promise.resolve(Response.json({ sha: SHA }));
    const resolved = RESOLVE_RE.exec(pathname);
    if (resolved === null) return Promise.resolve(new Response("not found", { status: 404 }));
    const path = decodeURIComponent(resolved[3]);
    paths.push(path);
    const bytes = files.get(path);
    if (bytes === undefined) return Promise.resolve(new Response("not found", { status: 404 }));
    return Promise.resolve(
      new Response(bytes, { headers: { "content-length": String(bytes.byteLength) } }),
    );
  };
  return { fetch, paths };
};

const sha256Hex = async (bytes: Uint8Array<ArrayBuffer>): Promise<string> =>
  Array.from(new Uint8Array(await crypto.subtle.digest("SHA-256", bytes)))
    .map((byte) => byte.toString(16).padStart(2, "0"))
    .join("");

/** manifest の 3 点セット（path / size / sha256）を現物から作る。 */
export const fileRef = async (
  path: string,
  bytes: Uint8Array<ArrayBuffer>,
): Promise<{ path: string; size: number; sha256: string }> => ({
  path,
  size: bytes.byteLength,
  sha256: await sha256Hex(bytes),
});

export const f32Tensor = (shape: readonly number[], value: number): DumpTensor => ({
  dtype: "F32",
  shape: [...shape],
  data: new Float32Array(shape.reduce((product, dim) => product * dim, 1)).fill(value),
});

/**
 * 最小の IR グラフ 1 本（`linear` 1 段）。`op` を差し替えると「実行できないグラフ」になる
 * （IR パーサは op 名の綴りを見ないので、落ちるのは capability 門 = admission 相）。
 */
const miniGraph = (op: string): unknown => ({
  format: "karume-ir",
  version: 1,
  requires: { ops: [op] },
  symbols: [],
  inputs: [{ name: "x", dtype: "f32", shape: [2, 2] }],
  outputs: ["y"],
  initializers: {
    w: { tensor: "m.w", storage: { dtype: "f32" } },
    b: { tensor: "m.b", storage: { dtype: "f32" } },
  },
  values: {
    w: { dtype: "f32", shape: [2, 2] },
    b: { dtype: "f32", shape: [2] },
    y: { dtype: "f32", shape: [2, 2] },
  },
  states: {},
  nodes: [{ op, ins: ["x", "w", "b"], outs: ["y"], attrs: {} }],
});

/** グラフ shard（`karume_ir` + 同居テンソル）1 本を焼く。 */
export const graphShardBytes = (
  op: string,
  tensors: readonly (readonly [string, DumpTensor])[],
): Uint8Array<ArrayBuffer> =>
  writeSafetensors(new Map(tensors), { karume_ir: JSON.stringify(miniGraph(op)) });

/** 重み shard（`karume_ir` を持たない）1 本を焼く。 */
export const weightShardBytes = (
  tensors: readonly (readonly [string, DumpTensor])[],
): Uint8Array<ArrayBuffer> => writeSafetensors(new Map(tensors), {});

export const manifestBytes = (models: unknown): Uint8Array<ArrayBuffer> =>
  new TextEncoder().encode(
    JSON.stringify({
      format: "karume/4",
      generator: "karume-test/0",
      defaultModel: "test",
      models,
    }),
  );

/**
 * 実行可能な 1 コンポーネント（`linear` 1 段）を **グラフ shard + 重み shard の 2 本**へ割った
 * 配布形を組み、manifest まで解決して返す。重み `m.w` は 2 本目にしか無いので、Session が
 * 張れること自体が「重み shard が届いている」ことの証拠になる。
 */
export const prepareTwoShard = async (): Promise<TwoShardRig> => {
  const graph = graphShardBytes("linear", [["m.b", f32Tensor([2], 0.25)]]);
  const weights = weightShardBytes([["m.w", f32Tensor([2, 2], 0.5)]]);
  const refs = {
    graph: await fileRef("dit/model-00000.safetensors", graph),
    weights: await fileRef("dit/model-00001.safetensors", weights),
  };
  const manifest = manifestBytes({
    test: {
      pipeline: "test/1",
      weights: { dit: { f32: { shards: [refs.graph, refs.weights] } } },
      assets: {},
      quants: { f32: { weights: { dit: "f32" }, session: {} } },
      defaultQuant: "f32",
      pipelineConfig: {},
    },
  });
  const mock = createMockFetch(
    new Map([
      [MANIFEST_PATH, manifest],
      [refs.graph.path, graph],
      [refs.weights.path, weights],
    ]),
  );
  const caches = new MemoryCacheStorage();
  const hubOptions = { fetch: mock.fetch, caches };
  const loaded = await loadManifest({ repo: REPO, revision: SHA, hubUrl: HUB_URL }, hubOptions);
  return { loaded, files: resolveFiles(loaded.manifest), refs, mock, hubOptions };
};

/** `models/karume-siglip2-base/karume.json` の `pipelineConfig` 実物（6 欄）。 */
export const SIGLIP2_CONFIG: Record<string, unknown> = {
  imageWidth: 224,
  imageHeight: 224,
  imageMean: [0.5, 0.5, 0.5],
  imageStd: [0.5, 0.5, 0.5],
  hiddenDim: 768,
  interpolation: "bilinear",
};

const CHANNELS = 3;
const IMAGE_SIZE = 224;
const HIDDEN_DIM = 768;

/**
 * siglip2 の家族 admission が読むグラフ**宣言**（入出力の名前と形）まで再現した最小の IR。
 *
 * MUST: `admitSiglip2` の構造検査（`pixel_values` の 4 軸 / 出力の `hiddenDim`）を**通る**形で
 * 綴る — ここで落ちると、その後段に居る門（quant / GPU 前提）を踏んだことにならない。実行は
 * しない（Session を張るテストはこの rig を使わない）ので、`linear` の内側の整合は問わない。
 */
const siglip2Graph = (): unknown => ({
  format: "karume-ir",
  version: 1,
  requires: { ops: ["linear"] },
  symbols: [],
  inputs: [{ name: "pixel_values", dtype: "f32", shape: [1, CHANNELS, IMAGE_SIZE, IMAGE_SIZE] }],
  outputs: ["pooler_output"],
  initializers: {
    w: { tensor: "m.w", storage: { dtype: "f32" } },
    b: { tensor: "m.b", storage: { dtype: "f32" } },
  },
  values: {
    w: { dtype: "f32", shape: [HIDDEN_DIM, IMAGE_SIZE] },
    b: { dtype: "f32", shape: [HIDDEN_DIM] },
    pooler_output: { dtype: "f32", shape: [1, HIDDEN_DIM] },
  },
  states: {},
  nodes: [{ op: "linear", ins: ["pixel_values", "w", "b"], outs: ["pooler_output"], attrs: {} }],
});

/**
 * siglip2 の配布形（グラフ shard + 重み shard の 2 本）を疑似 HF に載せる。`patch` で
 * `models["test"]` の欄を差し替えて**家族 admission だけ**が落ちる形を作る。
 *
 * MUST: グラフは実行可能な `linear` 1 段のまま — 非対応 op にすると runtime の capability 門
 * （②が既に縛る側）で落ちてしまい、家族の門を通ったことの証明にならない。
 */
export const prepareSiglip2 = async (patch: Record<string, unknown>): Promise<Siglip2Rig> => {
  const graph = writeSafetensors(
    new Map([["m.b", f32Tensor([HIDDEN_DIM], 0.25)]]),
    { karume_ir: JSON.stringify(siglip2Graph()) },
  );
  const weights = weightShardBytes([["m.w", f32Tensor([HIDDEN_DIM, IMAGE_SIZE], 0.5)]]);
  const refs = {
    graph: await fileRef("vision/model-00000.safetensors", graph),
    weights: await fileRef("vision/model-00001.safetensors", weights),
  };
  const manifest = manifestBytes({
    test: {
      pipeline: "siglip2/1",
      weights: { vision: { f32: { shards: [refs.graph, refs.weights] } } },
      assets: {},
      quants: { f32: { weights: { vision: "f32" }, session: {} } },
      defaultQuant: "f32",
      pipelineConfig: SIGLIP2_CONFIG,
      ...patch,
    },
  });
  const mock = createMockFetch(
    new Map([
      [MANIFEST_PATH, manifest],
      [refs.graph.path, graph],
      [refs.weights.path, weights],
    ]),
  );
  return { refs, mock, caches: new MemoryCacheStorage() };
};

/** ⑧⑨で使う quant 欄（`requiredLimits` だけが違う 2 通り）。 */
export const quantsRequiring = (maxBufferSize: number): Record<string, unknown> => ({
  f32: { weights: { vision: "f32" }, session: {}, requiredLimits: { maxBufferSize } },
});
