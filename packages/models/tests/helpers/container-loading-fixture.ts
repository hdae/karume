// コンテナ経路のロード（`src/hub/components.ts`）の門が共有する疑似 HF 取得層と `karume/5` の
// 配布形（container_loading_test.ts と gpu_container_loading_test.ts と
// container_components_test.ts）。
//
// NOTE: hub / runtime の**テストの都合**は import しない（`helpers/memory-cache.ts` と同じ
// 規律）。唯一の例外が容器の**書き手**（`helpers/container-fixture.ts` 経由の
// `packages/runtime/tests/helpers/container-write.ts`）で、これは形式の道具（Python の
// `karume.container` と同じ位置づけ）— models 側に写しを持つと 2 実装がずれる。

import type { FileRef, LoadedManifest, ResolvedSelection } from "@karume/hub";
import { loadManifest, resolveSelection } from "@karume/hub";
import type { ModelInput } from "../../../runtime/tests/helpers/container-write.ts";
import { MemoryCacheStorage } from "./memory-cache.ts";
import {
  linearComponent,
  parseIrDeclarationValue,
  sha256Hex,
  type TestContainer,
  writeContainer,
} from "./container-fixture.ts";

/** 疑似 HF の取得層（叩かれた path を記録する `fetch`）。 */
export type MockFetch = {
  readonly fetch: typeof globalThis.fetch;
  /** 叩かれた path（リポを問わず宣言順）。 */
  readonly paths: string[];
  /** 叩かれた (リポ, path) の対（越境・差し替えの観測用）。 */
  readonly requests: { readonly repo: string; readonly path: string }[];
  /**
   * 配信表を空にする（常駐量を測る台本のための口）。`fetch` の閉包は取得面が握り続けるので、
   * 表を捨てないとフィクスチャのバイト列そのものが測定値へ乗る。
   */
  readonly clear: () => void;
};

/** `loadManifest` / `fromPretrained` へ渡す取得ノブ。 */
export type HubOptions = {
  readonly fetch: typeof globalThis.fetch;
  readonly caches: MemoryCacheStorage;
};

/** 疑似リポ 1 つぶん（manifest まで解決済み）。 */
export type ContainerRig = {
  readonly loaded: LoadedManifest;
  readonly selection: ResolvedSelection;
  /** 部品名 → 配信した part の 3 点セット（長さ 0 の part を含む・添字順）。 */
  readonly parts: Readonly<Record<string, readonly FileRef[]>>;
  readonly mock: MockFetch;
  readonly hubOptions: HubOptions;
};

/**
 * 家族 admission の席（この機構だけを見るテストは家族の門を 1 つも置かない）。実家族の門が
 * 席に載っていることは⑦が別に縛る。
 */
export const NO_FAMILY_GATE = (): undefined => undefined;

export const HUB_URL = "https://hub.test";
export const REPO = "karume-test/containers";
export const SHA = "0123456789abcdef0123456789abcdef01234567";
export const MANIFEST_PATH = "karume.json";

const REVISION_RE = /^\/api\/models\/(.+)\/revision\/(.+)$/;
const RESOLVE_RE = /^\/(.+?)\/resolve\/([^/]+)\/(.+)$/;

/**
 * HF の 2 経路（revision 解決 API・resolve URL）だけを喋る `fetch`。叩かれた path を記録する。
 *
 * 表を**リポごと**に持つのは、差し替え席（別リポの同じ役割）を踏むテストが 2 リポを同時に
 * 名乗るため — `karume.json` は両方のリポで同じ path なので、1 枚の表では潰れる。
 */
const createMockFetch = (
  repos: ReadonlyMap<string, Map<string, Uint8Array<ArrayBuffer>>>,
): MockFetch => {
  const paths: string[] = [];
  const requests: { repo: string; path: string }[] = [];
  const fetch: typeof globalThis.fetch = (input) => {
    const href = typeof input === "string" ? input : input instanceof URL ? input.href : input.url;
    const { pathname } = new URL(href);
    if (REVISION_RE.test(pathname)) return Promise.resolve(Response.json({ sha: SHA }));
    const resolved = RESOLVE_RE.exec(pathname);
    if (resolved === null) return Promise.resolve(new Response("not found", { status: 404 }));
    const repo = resolved[1];
    const path = decodeURIComponent(resolved[3]);
    paths.push(path);
    requests.push({ repo, path });
    const bytes = repos.get(repo)?.get(path);
    if (bytes === undefined) return Promise.resolve(new Response("not found", { status: 404 }));
    return Promise.resolve(
      new Response(bytes, { headers: { "content-length": String(bytes.byteLength) } }),
    );
  };
  const clear = (): void => {
    for (const files of repos.values()) files.clear();
  };
  return { fetch, paths, requests, clear };
};

/** manifest の 3 点セット（path / size / sha256）を現物から作る。 */
export const fileRef = async (
  path: string,
  bytes: Uint8Array<ArrayBuffer>,
): Promise<FileRef> => ({
  path,
  size: bytes.byteLength,
  sha256: await sha256Hex(bytes),
});

/** 配信する容器 1 本（manifest の dtype エントリ + 疑似 HF の配信表）。 */
export type ServedContainer = {
  /** `weights.<部品>.<dtype>` に入る値。 */
  readonly entry: Record<string, unknown>;
  /** part の 3 点セット（長さ 0 の part も並ぶ — 添字が part の id だから）。 */
  readonly parts: readonly FileRef[];
  /** 疑似 HF が配るファイル（長さ 0 の part は配らない — 取得層も取りに来ない）。 */
  readonly files: readonly (readonly [string, Uint8Array<ArrayBuffer>])[];
  readonly written: TestContainer;
};

/**
 * 容器を 1 本書いて、配布形の 1 エントリへ畳む。part の path は書き手の規約
 * （`<stem>-NNNNN-of-NNNNN.krm` — container-v1 §8）に倣う。
 */
export const serveContainer = async (
  stem: string,
  input: ModelInput,
  options: { readonly partBytes?: number; readonly blockBytes?: number } = {},
): Promise<ServedContainer> => {
  const written = await writeContainer(input, {
    partBytes: options.partBytes ?? 4096,
    blockBytes: options.blockBytes ?? 512,
  });
  const total = String(written.parts.length).padStart(5, "0");
  const paths = written.parts.map((_, index) =>
    `${stem}-${String(index + 1).padStart(5, "0")}-of-${total}.krm`
  );
  const parts = await Promise.all(
    written.parts.map((bytes, index) => fileRef(paths[index], bytes)),
  );
  return {
    entry: { container: { descriptor: written.descriptor, parts } },
    parts,
    files: written.parts
      .map((bytes, index) => [paths[index], bytes] as const)
      .filter(([, bytes]) => bytes.byteLength > 0),
    written,
  };
};

const manifestBytes = (models: unknown): Uint8Array<ArrayBuffer> =>
  new TextEncoder().encode(
    JSON.stringify({
      format: "karume/5",
      generator: "karume-test/0",
      defaultModel: "test",
      models,
    }),
  );

/** 疑似リポ 1 つの中身（配布形 + 配るファイル）。 */
export type RepoSpec = {
  readonly repo: string;
  readonly models: unknown;
  readonly files: Iterable<readonly [string, Uint8Array<ArrayBuffer>]>;
};

/** {@link RepoSpec} を疑似 HF の 1 枚の表へ畳む（`karume.json` はリポごとに別）。 */
export const serveRepos = (specs: readonly RepoSpec[]): MockFetch =>
  createMockFetch(
    new Map(
      specs.map((
        spec,
      ) => [
        spec.repo,
        new Map<string, Uint8Array<ArrayBuffer>>([
          [MANIFEST_PATH, manifestBytes(spec.models)],
          ...spec.files,
        ]),
      ]),
    ),
  );

/** 疑似 HF に載せた配布形（主リポ 1 つ）を `loadManifest` まで通す。 */
export const serveRepo = async (
  models: unknown,
  files: ReadonlyMap<string, Uint8Array<ArrayBuffer>>,
  parts: Readonly<Record<string, readonly FileRef[]>>,
): Promise<ContainerRig> => {
  const mock = serveRepos([{ repo: REPO, models, files }]);
  const hubOptions = { fetch: mock.fetch, caches: new MemoryCacheStorage() };
  const loaded = await loadManifest({ repo: REPO, revision: SHA, hubUrl: HUB_URL }, hubOptions);
  return { loaded, selection: resolveSelection(loaded.manifest), parts, mock, hubOptions };
};

/**
 * 実行できる 1 部品（`linear` 1 段）の配布形。重み block は part 2 にしか無いので、Session が
 * 張れること自体が「重みの part が届いている」ことの証拠になる。`op` を差し替えると
 * capability 門で落ちる容器になる。
 */
export const prepareComponent = async (
  options: { readonly op?: string; readonly key?: string } = {},
): Promise<ContainerRig> => {
  const key = options.key ?? "dit";
  const served = await serveContainer(
    `${key}/model.f32`,
    linearComponent(key, options.op === undefined ? {} : { op: options.op }),
  );
  return await serveRepo(
    {
      test: {
        pipeline: "test/1",
        weights: { [key]: { f32: served.entry } },
        assets: {},
        quants: { f32: { weights: { [key]: "f32" }, session: {} } },
        defaultQuant: "f32",
        pipelineConfig: {},
      },
    },
    new Map(served.files),
    { [key]: served.parts },
  );
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
const siglip2Component = (): ModelInput => ({
  graphs: {
    vision: parseIrDeclarationValue({
      format: "karume-ir",
      version: 2,
      requires: { ops: ["linear"] },
      symbols: [],
      inputs: [{
        name: "pixel_values",
        dtype: "f32",
        shape: [1, CHANNELS, IMAGE_SIZE, IMAGE_SIZE],
      }],
      outputs: ["pooler_output"],
      initializers: { w: {}, b: {} },
      values: {
        w: { dtype: "f32", shape: [HIDDEN_DIM, IMAGE_SIZE] },
        b: { dtype: "f32", shape: [HIDDEN_DIM] },
        pooler_output: { dtype: "f32", shape: [1, HIDDEN_DIM] },
      },
      states: {},
      nodes: [{
        op: "linear",
        ins: ["pixel_values", "w", "b"],
        outs: ["pooler_output"],
        attrs: {},
      }],
    }),
  },
  consts: [],
  weights: [
    {
      graph: "vision",
      initializer: "w",
      bytes: new Uint8Array(new ArrayBuffer(HIDDEN_DIM * IMAGE_SIZE * 4)),
      encoding: { codec: "f32" },
    },
    {
      graph: "vision",
      initializer: "b",
      bytes: new Uint8Array(new ArrayBuffer(HIDDEN_DIM * 4)),
      encoding: { codec: "f32" },
    },
  ],
  assets: [],
  provenance: { license: "test" },
});

/**
 * siglip2 の配布形（容器 1 本）を疑似 HF に載せる。`patch` で `models["test"]` の欄を
 * 差し替えて**家族 admission だけ**が落ちる形を作る。
 *
 * MUST: グラフは実行可能な `linear` 1 段のまま — 非対応 op にすると runtime の capability 門
 * （②が既に縛る側）で落ちてしまい、家族の門を通ったことの証明にならない。
 */
export const prepareSiglip2 = async (
  patch: Record<string, unknown>,
): Promise<ContainerRig> => {
  const served = await serveContainer("vision/model.f32", siglip2Component(), {
    partBytes: 2 * 1024 * 1024,
    blockBytes: 2 * 1024 * 1024,
  });
  return await serveRepo(
    {
      test: {
        pipeline: "siglip2/1",
        weights: { vision: { f32: served.entry } },
        assets: {},
        quants: { f32: { weights: { vision: "f32" }, session: {} } },
        defaultQuant: "f32",
        pipelineConfig: SIGLIP2_CONFIG,
        ...patch,
      },
    },
    new Map(served.files),
    { vision: served.parts },
  );
};

/** ⑧⑨で使う quant 欄（`requiredLimits` だけが違う 2 通り）。 */
export const quantsRequiring = (maxBufferSize: number): Record<string, unknown> => ({
  f32: { weights: { vision: "f32" }, session: {}, requiredLimits: { maxBufferSize } },
});
