// The fake HF fetch layer and the `karume/5` distribution form shared by the checks of
// container-path loading (`src/hub/components.ts`) (container_loading_test.ts,
// container_loading_test.ts and container_loading_test.ts).
//
// NOTE: nothing **test-only** from hub / runtime is imported (the same rule as
// `helpers/memory-cache.ts`). The one exception is the container **writer**
// (`packages/runtime/tests/helpers/container-write.ts` through `helpers/container-fixture.ts`),
// which is format tooling (the same standing as Python's `karume.container`) — a copy on the models
// side would let the two implementations drift.

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

/** The fake HF fetch layer (a `fetch` that records the requested paths). */
export type MockFetch = {
  readonly fetch: typeof globalThis.fetch;
  /** The requested paths (in request order, across repositories). */
  readonly paths: string[];
  /**
   * The requested (repository, path) pairs (for observing cross-repository references and
   * replacements).
   */
  readonly requests: { readonly repo: string; readonly path: string }[];
  /**
   * Empties the serving table (a hook for scripts that measure resident memory). The fetch surface
   * keeps holding the `fetch` closure, so unless the table is dropped the fixture bytes themselves
   * show up in the measurement.
   */
  readonly clear: () => void;
};

/** The fetch options passed to `loadManifest` / `fromPretrained`. */
export type HubOptions = {
  readonly fetch: typeof globalThis.fetch;
  readonly caches: MemoryCacheStorage;
};

/** One fake repository (resolved up to the manifest). */
export type ContainerRig = {
  readonly loaded: LoadedManifest;
  readonly selection: ResolvedSelection;
  /**
   * Component name → the path / size / sha256 of the served parts (zero-length parts included, in
   * index order).
   */
  readonly parts: Readonly<Record<string, readonly FileRef[]>>;
  readonly mock: MockFetch;
  readonly hubOptions: HubOptions;
};

/**
 * The family-admission slot (tests that look only at this mechanism put no family check in it).
 * That the real families' checks are in the slot is pinned separately by ⑦.
 */
export const NO_FAMILY_GATE = (): undefined => undefined;

export const HUB_URL = "https://hub.test";
export const REPO = "karume-test/containers";
export const SHA = "0123456789abcdef0123456789abcdef01234567";
export const MANIFEST_PATH = "karume.json";

const REVISION_RE = /^\/api\/models\/(.+)\/revision\/(.+)$/;
const RESOLVE_RE = /^\/(.+?)\/resolve\/([^/]+)\/(.+)$/;

/**
 * A `fetch` that speaks only HF's two routes (the revision resolve API and the resolve URL). It
 * records the requested paths.
 *
 * The table is kept **per repository** because tests that reach the replacement slot (the same role
 * from another repository) act as two repositories at once — `karume.json` has the same path in
 * both, so a single table would collapse them.
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

/** Builds the manifest's file triple (path / size / sha256) from the actual bytes. */
export const fileRef = async (path: string, bytes: Uint8Array<ArrayBuffer>): Promise<FileRef> => ({
  path,
  size: bytes.byteLength,
  sha256: await sha256Hex(bytes),
});

/** One served container (the manifest's dtype entry + the fake HF's serving table). */
export type ServedContainer = {
  /** The value for `weights.<component>.<dtype>`. */
  readonly entry: Record<string, unknown>;
  /** The parts' file triples (zero-length parts are listed too — the index is the part id). */
  readonly parts: readonly FileRef[];
  /**
   * The files the fake HF serves (zero-length parts are not served — the fetch layer does not
   * request them either).
   */
  readonly files: readonly (readonly [string, Uint8Array<ArrayBuffer>])[];
  readonly written: TestContainer;
};

/**
 * Writes one container and folds it into one entry of the distribution form. The part paths follow
 * the writer's convention (`<stem>-NNNNN-of-NNNNN.krm` — container-v1 §8).
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
  const paths = written.parts.map(
    (_, index) => `${stem}-${String(index + 1).padStart(5, "0")}-of-${total}.krm`,
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

/**
 * The bytes of the distribution form (`karume.json`) (shared by the fake HF repository and the
 * in-memory directory).
 */
export const manifestBytes = (models: unknown): Uint8Array<ArrayBuffer> =>
  new TextEncoder().encode(
    JSON.stringify({
      format: "karume/5",
      generator: "karume-test/0",
      defaultModel: "test",
      models,
    }),
  );

/** The contents of one fake repository (the distribution form + the served files). */
export type RepoSpec = {
  readonly repo: string;
  readonly models: unknown;
  readonly files: Iterable<readonly [string, Uint8Array<ArrayBuffer>]>;
};

/** Folds {@link RepoSpec}s into the fake HF's table (each repository has its own `karume.json`). */
export const serveRepos = (specs: readonly RepoSpec[]): MockFetch =>
  createMockFetch(
    new Map(
      specs.map((spec) => [
        spec.repo,
        new Map<string, Uint8Array<ArrayBuffer>>([
          [MANIFEST_PATH, manifestBytes(spec.models)],
          ...spec.files,
        ]),
      ]),
    ),
  );

/**
 * Takes a distribution form served from the fake HF (one main repository) through `loadManifest`.
 */
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
 * The distribution form of one runnable component (a single `linear`). The weight block is only in
 * part 2, so being able to build a Session is itself evidence that "the weight part arrived".
 * Replacing `op` makes a container that fails the capability check.
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

/** The actual `pipelineConfig` of `base` in `models/karume-siglip2/karume.json` (6 fields). */
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
 * The smallest IR that reproduces the graph **declaration** (input / output names and shapes) that
 * siglip2's family admission reads.
 *
 * MUST: it is written so that it **passes** the structural check of `admitSiglip2` (the 4 axes of
 * `pixel_values` / the output's `hiddenDim`) — failing here would mean the checks after it (quant /
 * GPU prerequisites) were never reached. It is not run (tests that build a Session do not use this
 * rig), so the internal consistency of the `linear` does not matter.
 */
const siglip2Component = (): ModelInput => ({
  graphs: {
    vision: parseIrDeclarationValue({
      format: "karume-ir",
      version: 2,
      requires: { ops: ["linear"] },
      symbols: [],
      inputs: [
        {
          name: "pixel_values",
          dtype: "f32",
          shape: [1, CHANNELS, IMAGE_SIZE, IMAGE_SIZE],
        },
      ],
      outputs: ["pooler_output"],
      initializers: { w: {}, b: {} },
      values: {
        w: { dtype: "f32", shape: [HIDDEN_DIM, IMAGE_SIZE] },
        b: { dtype: "f32", shape: [HIDDEN_DIM] },
        pooler_output: { dtype: "f32", shape: [1, HIDDEN_DIM] },
      },
      states: {},
      nodes: [
        {
          op: "linear",
          ins: ["pixel_values", "w", "b"],
          outs: ["pooler_output"],
          attrs: {},
        },
      ],
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
 * Serves siglip2's distribution form (one container) from the fake HF. `patch` replaces fields of
 * `models["test"]` to make a form that fails **only family admission**.
 *
 * MUST: the graph stays a runnable single `linear` — an unsupported op would fail at runtime's
 * capability check (which ② already pins), and would not prove that the family check was passed.
 */
export const prepareSiglip2 = async (patch: Record<string, unknown>): Promise<ContainerRig> => {
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

/** The quant field used by ⑧ and ⑨ (two variants that differ only in `requiredLimits`). */
export const quantsRequiring = (maxBufferSize: number): Record<string, unknown> => ({
  f32: { weights: { vision: "f32" }, session: {}, requiredLimits: { maxBufferSize } },
});
