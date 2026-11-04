/**
 * The GPU-free checks of the container load path (`src/hub/components.ts`), pinned by swapping only the fetch
 * layer:
 *
 * ① **Progress stays one stream for the whole model** (even when fetching splits into "descriptor warm-up +
 *    weight parts + whole-file reads of the assets", `loaded` increases monotonically and lands on the sum of all
 *    file sizes, and `complete` fires once per file). When a cross-repository reference has the same path as the
 *    own repo, the total still counts both files.
 * ② **Admission fails before any weight part is fetched** (the capability violation is decided by the part-0
 *    descriptor alone and no weight-part URL is ever requested — the very reason for ADR 0070 decision 5).
 *    This check also pins that the prefetch runs **after** admission.
 * ③ **Weight parts are fully downloaded at load time** (downloaded before any Session is built — this removes the
 *    shape where a lazily built family delays the download until the first run and it never shows in load
 *    progress).
 * ⑤ **A load started with an already aborted `signal` fails** (the abort contract during load is still in place.
 *    The paired "the load-time signal is not carried into Session building" is on the GPU side).
 * ⑥ **The bytes of weight blocks are not retained** (nothing stays resident even while the component opener is
 *    held). This needs a forced gc, so it runs as a script in a separate process
 *    (`helpers/weight-block-retention.ts`) and only its exit code is checked here.
 * ⑦ **A family admission violation also fetches no weight part** (the family version of ②). ② only exercises
 *    the runtime's capability check, so it cannot pin that the family side (pipeline major / `pipelineConfig`) of
 *    "not a single byte of weights for a model that cannot run" sits in front. A real family (siglip2 = the
 *    minimal shape with one component) goes through `fromPretrained` and is observed the same way.
 * ⑧ **Exceeding `requiredLimits` also fetches no weight part (shared GPU)** (the limits version of ⑦ —
 *    ADR 0089 decision 5). The comparison target is the `GpuContext.limits` passed in.
 * ⑩ **Failure diagnostics of selection and of the opener** (a component name the manifest does not have / a role
 *    that was not opened).
 *
 * NOTE: nothing **test-only** from hub / runtime is imported (the same discipline as `helpers/memory-cache.ts`).
 * The mock is built minimally in `helpers/container-loading-fixture.ts`. Two exceptions — the container
 * **writer** (a format tool; see the fixture's NOTE) and ⑧'s `fake-gpu.ts` (the runtime does not expose
 * `GpuContext` as a value (`acquireGpu` is the only entry point — ADR 0008), so exercising the shared-GPU path
 * without a GPU needs a helper that wraps the real thing. Building our own fake here would make the very thing
 * under test (a GpuContext that has limits) a fake).
 *
 * ## Component replacement seat (`LoadContainerOptions.components` — ADR 0108 decision 19 / ADR 0109 decision 10)
 *
 * Also GPU-free: two mock HF repos, observing which (repo, path) pairs are requested.
 *
 * - Only the replaced role comes from the other repo; the same role's weights of the original repo are never
 *   fetched, and roles that are not replaced stay on the original repo. Inherited (non-own) properties of
 *   `components` never take a seat.
 * - A container whose graph description differs is rejected by admission before any `.krm` of either repo is
 *   fetched.
 * - A role name this family does not have fails loudly, listing the replaceable roles.
 * - A replacement part that has the same path as another component's part of the original repo is counted as a
 *   separate file in progress. The replacement manifest references it as its own repo (no `repo` field), so
 *   this pins the per-seat manifest in the progress key, which the cross-repo case above does not reach.
 *
 * ## Real-GPU gates
 *
 * Explicitly skipped without an adapter (ADR 0005).
 *
 * - Building a Session does not move progress (re-reading the prefetched cache must not emit `complete` again,
 *   which would double-count the aggregated `loaded`) and does not go to the network.
 * - One opener can build two Sessions in a row (blocks are re-read in part order on every call); checked by the
 *   run output, not by the build succeeding.
 * - The load-time `signal` is not carried into Session building (an abort after a successful load must not
 *   break later Sessions). The paired "a load started with an aborted signal fails" gate is ⑤ above.
 * - `requiredLimits` beyond the adapter also blocks weight parts on the self-acquired GPU path (compared against
 *   `readAdapterLimits()` — ADR 0089 decision 5); the shared-GPU variant is ⑧ above.
 */

import { execFile } from "node:child_process";
import { fileURLToPath } from "node:url";
import { beforeAll, describe, it, test } from "vite-plus/test";
import { assertEquals, assertRejects, assertStringIncludes, assertThrows } from "@std/assert";
import type { AssetProgress, DistributionSource } from "@karume/hub";
import { acquireGpu } from "@karume/runtime";
import { fakeDevice, fakeGpuContext } from "../../runtime/tests/helpers/fake-gpu.ts";
import { loadContainerComponents } from "../src/hub/components.ts";
import * as siglip2 from "../siglip2.ts";
import { LINEAR_PROBE_Y, linearComponent, runLinearProbe } from "./helpers/container-fixture.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";
import {
  HUB_URL,
  MANIFEST_PATH,
  manifestBytes,
  NO_FAMILY_GATE,
  prepareComponent,
  prepareSiglip2,
  quantsRequiring,
  REPO,
  serveContainer,
  type ServedContainer,
  serveRepo,
  serveRepos,
  SHA,
  SIGLIP2_CONFIG,
} from "./helpers/container-loading-fixture.ts";
import { MemoryCacheStorage } from "./helpers/memory-cache.ts";
import { loadManifest, localDirectory, resolveSelection } from "@karume/hub";

/** Parts whose length is not 0 (= the ones the fetch layer actually requests). */
const fetched = (parts: readonly { path: string; size: number }[]): readonly string[] =>
  parts.filter((part) => part.size > 0).map((part) => part.path);

test("loadContainerComponents: progress is one stream for the whole model even with several components (monotonic, lands on the total)", async () => {
  const front = await serveContainer("front/model.f32", linearComponent("front"));
  const voice = await serveContainer("voice/model.f32", linearComponent("voice"));
  const tokenizer = new TextEncoder().encode(JSON.stringify({ vocab: ["a", "b"] }));
  const tokenizerRef = {
    path: "tokenizer/tokenizer.json",
    size: tokenizer.byteLength,
    sha256: [...new Uint8Array(await crypto.subtle.digest("SHA-256", tokenizer))]
      .map((byte) => byte.toString(16).padStart(2, "0"))
      .join(""),
  };
  const rig = await serveRepo(
    {
      test: {
        pipeline: "test/1",
        weights: { front: { f32: front.entry }, voice: { f32: voice.entry } },
        assets: { tokenizer: tokenizerRef },
        quants: { f32: { weights: { front: "f32", voice: "f32" }, session: {} } },
        defaultQuant: "f32",
        pipelineConfig: {},
      },
    },
    new Map([
      ...front.files,
      ...voice.files,
      [tokenizerRef.path, tokenizer as Uint8Array<ArrayBuffer>],
    ]),
    { front: front.parts, voice: voice.parts },
  );

  const events: AssetProgress[] = [];
  const { open, assets } = await loadContainerComponents(
    "test.fromPretrained",
    rig.loaded,
    rig.selection,
    ["front", "voice"],
    NO_FAMILY_GATE,
    { ...rig.hubOptions, onProgress: (progress) => events.push(progress) },
  );

  // Admission passed (the graph declaration is readable = the merge is done).
  assertEquals(open("front").graph.outputs, ["y"]);
  assertEquals(open("voice").graph.outputs, ["y"]);
  // The remaining assets still arrive as whole-file reads.
  assertEquals(Object.keys(assets), ["tokenizer"]);

  const paths = [...fetched(front.parts), ...fetched(voice.parts), tokenizerRef.path];
  const total =
    [...front.parts, ...voice.parts]
      .filter((part) => part.size > 0)
      .reduce((sum, part) => sum + part.size, 0) + tokenizerRef.size;
  // In every event, total is the single value of the **whole model's sum** (independent of how fetching is split).
  assertEquals(new Set(events.map((event) => event.total)), new Set([total]));
  // loaded increases monotonically and finally equals the total.
  let previous = 0;
  for (const event of events) {
    if (event.loaded < previous) {
      throw new Error(
        `loaded went backwards: ${previous} → ${event.loaded} (${event.phase} ${event.path})`,
      );
    }
    previous = event.loaded;
  }
  assertEquals(events[events.length - 1].loaded, total);
  // complete fires once per file (the per-file fields also line up as fileTotal, per the fetch layer's contract).
  const completes = events.filter((event) => event.phase === "complete");
  assertEquals(completes.length, paths.length);
  assertEquals(new Set(completes.map((event) => event.path)), new Set(paths));
  for (const event of completes) assertEquals(event.fileLoaded, event.fileTotal);
  // A zero-length part appears neither in fetching nor in progress (ADR 0109 decision 3).
  const empty = [...front.parts, ...voice.parts].filter((part) => part.size === 0);
  assertEquals(empty.length > 0, true, "not observed on a container that has a zero-length part");
  for (const part of empty) assertEquals(rig.mock.paths.includes(part.path), false);
});

test("loadContainerComponents: a cross-repository reference with the same path as the own repo is counted as a separate file in progress", async () => {
  // `voice` is a container borrowed from the other repo (a per-container cross reference — every part names the
  // other repo's coordinates). Part names that follow the writer's convention depend only on the stem, so two
  // files with the same stem have the same path.
  const CROSS_REPO = "karume-test/borrowed";
  const own = await serveContainer("shared/model.f32", linearComponent("front"));
  const borrowed = await serveContainer("shared/model.f32", linearComponent("voice"));
  const crossParts = borrowed.parts.map((part) => ({
    ...part,
    repo: CROSS_REPO,
    revision: SHA,
  }));
  const models = {
    test: {
      pipeline: "test/1",
      weights: {
        front: { f32: own.entry },
        voice: {
          f32: { container: { descriptor: borrowed.written.descriptor, parts: crossParts } },
        },
      },
      assets: {},
      quants: { f32: { weights: { front: "f32", voice: "f32" }, session: {} } },
      defaultQuant: "f32",
      pipelineConfig: {},
    },
  };
  const mock = serveRepos([
    { repo: REPO, models, files: own.files },
    { repo: CROSS_REPO, models: {}, files: borrowed.files },
  ]);
  const hubOptions = { fetch: mock.fetch, caches: new MemoryCacheStorage() };
  const loaded = await loadManifest({ repo: REPO, revision: SHA, hubUrl: HUB_URL }, hubOptions);
  const shared = fetched(own.parts).filter((path) => fetched(borrowed.parts).includes(path));
  assertEquals(shared.length > 0, true, "not observed on two files with the same path");

  const events: AssetProgress[] = [];
  await loadContainerComponents(
    "test.fromPretrained",
    loaded,
    resolveSelection(loaded.manifest),
    ["front", "voice"],
    NO_FAMILY_GATE,
    { ...hubOptions, onProgress: (progress) => events.push(progress) },
  );

  // Collapsing by path would make `total` one file smaller, and `loaded` would stop short by the same amount.
  const total = [...own.parts, ...borrowed.parts].reduce((sum, part) => sum + part.size, 0);
  assertEquals(new Set(events.map((event) => event.total)), new Set([total]));
  assertEquals(events[events.length - 1].loaded, total);
  // The origin field is passed through, so the consumer can also tell the two files with the same path apart.
  for (const path of shared) {
    const origins = events
      .filter((event) => event.phase === "complete" && event.path === path)
      .map((event) => event.repo)
      .sort((a, b) => (a! < b! ? -1 : a! > b! ? 1 : 0));
    assertEquals(
      origins,
      [CROSS_REPO, REPO].sort(),
      `the two files at ${path} cannot be told apart`,
    );
  }
});

test("loadContainerComponents: when two selected components share a part, that part is warmed up only once", async () => {
  // The manifest parse allows different containers in a selection to share the same part file. If the fetch
  // plan were passed as the plain concatenation of each container's part list, the duplicate check of the
  // warm-up surface (passing the same reference twice is a caller error) would make only the load fail while the
  // declaration passes.
  // Two components with the same weight values differ only in the graph name, so parts 1 onwards (the weight
  // blocks) are the same bytes — point `voice`'s parts 1 onwards at `front`'s files so they are shared.
  const front = await serveContainer("front/model.f32", linearComponent("front"));
  const voice = await serveContainer("voice/model.f32", linearComponent("voice"));
  const sharedParts = front.parts.slice(1);
  assertEquals(
    voice.parts.slice(1).map((part) => part.sha256),
    sharedParts.map((part) => part.sha256),
    "not observed on two components whose parts 1 onwards are the same bytes",
  );
  const voiceParts = [voice.parts[0], ...sharedParts];
  const rig = await serveRepo(
    {
      test: {
        pipeline: "test/1",
        weights: {
          front: { f32: front.entry },
          voice: {
            f32: { container: { descriptor: voice.written.descriptor, parts: voiceParts } },
          },
        },
        assets: {},
        quants: { f32: { weights: { front: "f32", voice: "f32" }, session: {} } },
        defaultQuant: "f32",
        pipelineConfig: {},
      },
    },
    new Map([...front.files, ...voice.files.filter(([path]) => path === voice.parts[0].path)]),
    { front: front.parts, voice: voiceParts },
  );

  const events: AssetProgress[] = [];
  const { open } = await loadContainerComponents(
    "test.fromPretrained",
    rig.loaded,
    rig.selection,
    ["front", "voice"],
    NO_FAMILY_GATE,
    { ...rig.hubOptions, onProgress: (progress) => events.push(progress) },
  );

  assertEquals(open("voice").graph.outputs, ["y"]);
  const shared = fetched(sharedParts);
  assertEquals(shared.length > 0, true, "not observed on a fetched shared part");
  for (const path of shared) {
    assertEquals(
      rig.mock.paths.filter((requested) => requested === path).length,
      1,
      `${path} is requested more than once`,
    );
    assertEquals(
      events.filter((event) => event.phase === "complete" && event.path === path).length,
      1,
      `${path} does not complete exactly once`,
    );
  }
  // The progress denominator also counts the shared part as one file, and finally lands on the total.
  const total = [front.parts[0], voice.parts[0], ...sharedParts].reduce(
    (sum, part) => sum + part.size,
    0,
  );
  assertEquals(new Set(events.map((event) => event.total)), new Set([total]));
  assertEquals(events[events.length - 1].loaded, total);
});

/** Uses an in-memory distribution directory (`karume.json` + the files it ships) as the source. */
const memorySource = (
  models: unknown,
  files: Iterable<readonly [string, Uint8Array<ArrayBuffer>]>,
): DistributionSource => {
  const table = new Map<string, Uint8Array<ArrayBuffer>>([
    [MANIFEST_PATH, manifestBytes(models)],
    ...files,
  ]);
  return localDirectory(
    {
      readFile: (path) => {
        const bytes = table.get(path);
        if (bytes === undefined) return Promise.reject(new Error(`test: ${path} is missing`));
        // MUST: return a view that covers the whole buffer (the contract of `DirectoryAdapter.readFile`).
        return Promise.resolve(bytes.slice());
      },
    },
    { label: "test-directory" },
  );
};

/** Checks that the progress stream increases monotonically, lands on the total, and completes once per file. */
const assertLandsOnTotal = (
  events: readonly AssetProgress[],
  files: readonly { readonly path: string; readonly size: number }[],
): void => {
  const total = files.reduce((sum, file) => sum + file.size, 0);
  assertEquals(new Set(events.map((event) => event.total)), new Set([total]));
  let previous = 0;
  for (const event of events) {
    if (event.loaded < previous) {
      throw new Error(
        `loaded went backwards: ${previous} → ${event.loaded} (${event.phase} ${event.path})`,
      );
    }
    previous = event.loaded;
  }
  assertEquals(events.at(-1)?.loaded, total, "the last event does not reach the total");
  const completes = events.filter((event) => event.phase === "complete");
  assertEquals(
    completes.map((event) => event.path).sort(),
    files.map((file) => file.path).sort(),
    "complete does not fire once per file",
  );
};

describe("loadContainerComponents: a load from local sources only also lands progress on the total", () => {
  // Parts of a source without phase 1 (warm-up) also emit complete once in the warm-up step (ADR 0086 addendum
  // 2026-10-11) — without it, the last value falls short by the parts counted in the denominator, and with no
  // assets no event arrives at all.
  let front: ServedContainer;
  let voice: ServedContainer;
  let tokenizer: Uint8Array<ArrayBuffer>;
  let tokenizerRef: { path: string; size: number; sha256: string };
  let parts: ServedContainer["parts"];
  beforeAll(async () => {
    front = await serveContainer("front/model.f32", linearComponent("front"));
    voice = await serveContainer("voice/model.f32", linearComponent("voice"));
    tokenizer = new TextEncoder().encode(JSON.stringify({ vocab: ["a"] }));
    tokenizerRef = {
      path: "tokenizer/tokenizer.json",
      size: tokenizer.byteLength,
      sha256: [...new Uint8Array(await crypto.subtle.digest("SHA-256", tokenizer))]
        .map((byte) => byte.toString(16).padStart(2, "0"))
        .join(""),
    };
    parts = [...front.parts, ...voice.parts].filter((part) => part.size > 0);
  });
  for (const withAssets of [false, true]) {
    it(withAssets ? "with assets" : "without assets", async () => {
      const source = memorySource(
        {
          test: {
            pipeline: "test/1",
            weights: { front: { f32: front.entry }, voice: { f32: voice.entry } },
            assets: withAssets ? { tokenizer: tokenizerRef } : {},
            quants: { f32: { weights: { front: "f32", voice: "f32" }, session: {} } },
            defaultQuant: "f32",
            pipelineConfig: {},
          },
        },
        [...front.files, ...voice.files, [tokenizerRef.path, tokenizer]],
      );
      const loaded = await loadManifest(source);
      const events: AssetProgress[] = [];
      await loadContainerComponents(
        "test.fromPretrained",
        loaded,
        resolveSelection(loaded.manifest),
        ["front", "voice"],
        NO_FAMILY_GATE,
        { onProgress: (progress) => events.push(progress) },
      );
      assertLandsOnTotal(events, withAssets ? [...parts, tokenizerRef] : parts);
    });
  }
});

test("loadContainerComponents: with HF and local sources mixed, progress lands on the total under one rule", async () => {
  // The main distribution is local; only `voice` is replaced from a mock HF repo (a container with the same graph description).
  const front = await serveContainer("front/model.f32", linearComponent("front"));
  const voice = await serveContainer("voice/model.f32", linearComponent("voice"));
  const models = {
    test: {
      pipeline: "test/1",
      weights: { front: { f32: front.entry }, voice: { f32: voice.entry } },
      assets: {},
      quants: { f32: { weights: { front: "f32", voice: "f32" }, session: {} } },
      defaultQuant: "f32",
      pipelineConfig: {},
    },
  };
  const loaded = await loadManifest(memorySource(models, [...front.files, ...voice.files]));
  const mock = serveRepos([{ repo: REPO, models, files: voice.files }]);
  const events: AssetProgress[] = [];
  await loadContainerComponents(
    "test.fromPretrained",
    loaded,
    resolveSelection(loaded.manifest),
    ["front", "voice"],
    NO_FAMILY_GATE,
    {
      fetch: mock.fetch,
      caches: new MemoryCacheStorage(),
      components: { voice: { source: { repo: REPO, revision: SHA, hubUrl: HUB_URL } } },
      onProgress: (progress) => events.push(progress),
    },
  );
  assertEquals(
    fetched(voice.parts).filter((path) => !mock.paths.includes(path)),
    [],
    "the replaced component was not fetched from HF",
  );
  // HF parts name their origin (repo) and local parts do not — both line up in one stream.
  const origins = new Set(
    events.filter((event) => event.phase === "complete").map((event) => event.repo),
  );
  assertEquals(origins, new Set([REPO, undefined]), "not observed with HF and local mixed");
  assertLandsOnTotal(
    events,
    [...front.parts, ...voice.parts].filter((part) => part.size > 0),
  );
});

test("loadContainerComponents: a capability violation fails on the descriptor alone and no weight part is fetched", async () => {
  const rig = await prepareComponent({ op: "karume_test_unsupported_op" });
  const parts = rig.parts["dit"];

  const error = await assertRejects(
    () =>
      loadContainerComponents(
        "test.fromPretrained",
        rig.loaded,
        rig.selection,
        ["dit"],
        NO_FAMILY_GATE,
        rig.hubOptions,
      ),
    Error,
  );
  // It failed on the capability check (a different failure would not establish "the weights were not fetched").
  if (!error.message.includes("karume_test_unsupported_op")) {
    throw new Error(`not the capability check's message: ${error.message}`);
  }
  // Part 0 (the descriptor) is requested and no weight part is ever requested (= admission before the weight
  // download. This also checks that the load-time prefetch runs after admission).
  assertEquals(rig.mock.paths.includes(parts[0].path), true);
  for (const part of parts.slice(1)) {
    assertEquals(rig.mock.paths.includes(part.path), false, `${part.path} is fetched`);
  }
});

test("loadContainerComponents: weight parts are fully downloaded before any Session is built (at load time) and show in progress", async () => {
  const rig = await prepareComponent();
  const parts = rig.parts["dit"];

  const events: AssetProgress[] = [];
  await loadContainerComponents(
    "test.fromPretrained",
    rig.loaded,
    rig.selection,
    ["dit"],
    NO_FAMILY_GATE,
    { ...rig.hubOptions, onProgress: (progress) => events.push(progress) },
  );

  // No Session has been built. Even so, the weight-part URLs have been requested.
  for (const path of fetched(parts)) {
    assertEquals(rig.mock.paths.includes(path), true, `${path} is not downloaded at load time`);
  }
  // Progress also includes the fetched parts (even a lazily built family shows "load = the whole download").
  const completes = events.filter((event) => event.phase === "complete");
  assertEquals(new Set(completes.map((event) => event.path)), new Set(fetched(parts)));
  assertEquals(
    events[events.length - 1].loaded,
    parts.reduce((sum, part) => sum + part.size, 0),
  );
});

test("loadContainerComponents: a load started with an already aborted signal fails (the abort contract holds)", async () => {
  const rig = await prepareComponent();
  const reason = new Error("test: aborted before the load started");

  const error = await assertRejects(() =>
    loadContainerComponents(
      "test.fromPretrained",
      rig.loaded,
      rig.selection,
      ["dit"],
      NO_FAMILY_GATE,
      { ...rig.hubOptions, signal: AbortSignal.abort(reason) },
    ),
  );
  assertEquals(error, reason);
});

test("family admission (pipeline major mismatch) fails on the descriptor alone and no weight part is fetched", async () => {
  const rig = await prepareSiglip2({ pipeline: "siglip2/99" });
  const parts = rig.parts["vision"];

  const error = await assertRejects(
    () =>
      siglip2.Pipeline.fromPretrained(
        { repo: REPO, revision: SHA, hubUrl: HUB_URL },
        { fetch: rig.mock.fetch, caches: rig.hubOptions.caches },
      ),
    Error,
  );
  // It failed on the family's major check (a different failure would not establish "the weights were not fetched").
  if (!error.message.includes("the major of pipeline 'siglip2/99' is not supported")) {
    throw new Error(`not the family admission's message: ${error.message}`);
  }
  // The descriptor is requested and no weight part is ever requested (the wording of ADR 0070 decision 5,
  // "not a single byte of weights for a model that cannot run", extends to the family's check).
  assertEquals(rig.mock.paths.includes(parts[0].path), true);
  for (const part of parts.slice(1)) {
    assertEquals(rig.mock.paths.includes(part.path), false, `${part.path} is fetched`);
  }
});

test("family admission (pipelineConfig schema violation) also fetches no weight part", async () => {
  const rig = await prepareSiglip2({
    pipelineConfig: { ...SIGLIP2_CONFIG, karumeUnknownKey: 1 },
  });
  const parts = rig.parts["vision"];

  const error = await assertRejects(
    () =>
      siglip2.Pipeline.fromPretrained(
        { repo: REPO, revision: SHA, hubUrl: HUB_URL },
        { fetch: rig.mock.fetch, caches: rig.hubOptions.caches },
      ),
    Error,
  );
  if (!error.message.includes("karumeUnknownKey")) {
    throw new Error(`not the pipelineConfig check's message: ${error.message}`);
  }
  assertEquals(rig.mock.paths.includes(parts[0].path), true);
  for (const part of parts.slice(1)) {
    assertEquals(rig.mock.paths.includes(part.path), false, `${part.path} is fetched`);
  }
});

test("family admission (requiredLimits exceeded) also fetches no weight part (shared GPU)", async () => {
  // Pass a distribution that requires even one byte to a GpuContext whose limits are all 0 (`fake-gpu.ts`'s
  // ZERO_LIMITS). A shared GPU cannot be re-acquired (features and limits are fixed when the device is created),
  // so admission is the only place where it can fail.
  const rig = await prepareSiglip2({ quants: quantsRequiring(1) });
  const parts = rig.parts["vision"];

  const error = await assertRejects(
    () =>
      siglip2.Pipeline.fromPretrained(
        { repo: REPO, revision: SHA, hubUrl: HUB_URL },
        {
          fetch: rig.mock.fetch,
          caches: rig.hubOptions.caches,
          gpu: fakeGpuContext(fakeDevice()),
        },
      ),
    Error,
  );
  // It failed on the limits check (a different failure would not establish "the weights were not fetched").
  if (!error.message.includes("maxBufferSize")) {
    throw new Error(`not the limits check's message: ${error.message}`);
  }
  assertEquals(rig.mock.paths.includes(parts[0].path), true);
  for (const part of parts.slice(1)) {
    assertEquals(rig.mock.paths.includes(part.path), false, `${part.path} is fetched`);
  }
});

test("loadContainerComponents: the bytes of weight blocks are not retained (gc observed in a separate process)", async () => {
  // MUST: a separate process — telling a merely unreachable state from a retained one needs a forced gc, and
  // `deno test` has no way to pass `--v8-flags`.
  const script = new URL("./helpers/weight-block-retention.ts", import.meta.url);
  const { code, stdout, stderr } = await new Promise<{
    code: number | null;
    stdout: string;
    stderr: string;
  }>((resolve) =>
    execFile(
      process.execPath,
      ["--expose-gc", fileURLToPath(script)],
      { encoding: "utf8" },
      (error, stdout, stderr) =>
        resolve({
          code: error === null ? 0 : ((error.code as number | null) ?? null),
          stdout,
          stderr,
        }),
    ),
  );
  assertEquals(code, 0, `${stdout}${stderr}`);
});

// ---- Failure diagnostics of selection and of the opener (⑩) ------------------------------

test("loadContainerComponents: a component name the manifest does not have fails with the selection's component list", async () => {
  const rig = await prepareComponent();
  const error = await assertRejects(
    () =>
      loadContainerComponents(
        "test.fromPretrained",
        rig.loaded,
        rig.selection,
        ["dit", "vae"],
        NO_FAMILY_GATE,
        rig.hubOptions,
      ),
    Error,
    "the container of component 'vae' is not in the manifest",
  );
  // A form the reader can check against the actual files (the convention of the existing asset diagnostics).
  assertStringIncludes(error.message, "components of selection test / f32: dit");
});

test("loadContainerComponents: open() on a role that was not opened fails with the list of opened roles", async () => {
  const rig = await prepareComponent();
  const { open } = await loadContainerComponents(
    "test.fromPretrained",
    rig.loaded,
    rig.selection,
    ["dit"],
    NO_FAMILY_GATE,
    rig.hubOptions,
  );
  const error = assertThrows(() => open("vae"), Error, "component 'vae' is not open");
  assertStringIncludes(error.message, "open components: dit");
});

// ---- Component replacement seat (`LoadContainerOptions.components`) -------------------

/** The mock repo the replacement comes from. */
const OTHER_REPO = "karume-test/replacement";

const modelsOf = (entries: Readonly<Record<string, ServedContainer>>): unknown => ({
  test: {
    pipeline: "test/1",
    weights: Object.fromEntries(
      Object.entries(entries).map(([key, served]) => [key, { f32: served.entry }]),
    ),
    assets: {},
    quants: {
      f32: {
        weights: Object.fromEntries(Object.keys(entries).map((key) => [key, "f32"])),
        session: {},
      },
    },
    defaultQuant: "f32",
    pipelineConfig: {},
  },
});

/**
 * Sets up two repos. The original repo has the two components `front` / `voice`; the replacement repo has only
 * `voice` (written as a **different graph** when `sameGraph` is false = the description's sha256 changes).
 */
const prepareSeats = async (options: {
  readonly sameGraph: boolean;
  readonly replacementStem?: string;
}) => {
  const front = await serveContainer("front/model.f32", linearComponent("front"));
  const voice = await serveContainer("voice/model.f32", linearComponent("voice", { w: 0.5 }));
  // Replacement: the same graph declaration + different weight values (the graph description is byte-identical, so its sha256 matches too).
  const replacement = await serveContainer(
    options.replacementStem ?? "other/voice.f32",
    options.sameGraph
      ? linearComponent("voice", { w: 1.5 })
      : // A different `op` changes `requires.ops` and the node list = the graph description's sha256 changes too.
        linearComponent("voice", { op: "add" }),
  );
  const mock = serveRepos([
    { repo: REPO, models: modelsOf({ front, voice }), files: [...front.files, ...voice.files] },
    { repo: OTHER_REPO, models: modelsOf({ voice: replacement }), files: replacement.files },
  ]);
  const hubOptions = { fetch: mock.fetch, caches: new MemoryCacheStorage() };
  const loaded = await loadManifest({ repo: REPO, revision: SHA, hubUrl: HUB_URL }, hubOptions);
  return {
    mock,
    hubOptions,
    loaded,
    selection: resolveSelection(loaded.manifest),
    parts: { front: front.parts, voice: voice.parts, replacement: replacement.parts },
    source: { repo: OTHER_REPO, revision: SHA, hubUrl: HUB_URL },
  };
};

/** Paths of the fetched `.krm` files (from any repo). */
const fetchedContainers = (paths: readonly string[]): readonly string[] =>
  paths.filter((path) => path !== MANIFEST_PATH);

test("components: only the replaced role comes from the other repo (the same role of the original repo is not fetched)", async () => {
  const rig = await prepareSeats({ sameGraph: true });

  const { open } = await loadContainerComponents(
    "test.fromPretrained",
    rig.loaded,
    rig.selection,
    ["front", "voice"],
    NO_FAMILY_GATE,
    { ...rig.hubOptions, components: { voice: { source: rig.source } } },
  );
  // Both are open (the merge is done).
  assertEquals(open("front").graph.outputs, ["y"]);
  assertEquals(open("voice").graph.outputs, ["y"]);

  const asked = (repo: string): readonly string[] =>
    rig.mock.requests.filter((request) => request.repo === repo).map((request) => request.path);
  // Every part of the replacement (except zero-length ones) is fetched.
  for (const part of rig.parts.replacement.filter((part) => part.size > 0)) {
    assertEquals(asked(OTHER_REPO).includes(part.path), true, `${part.path} is not fetched`);
  }
  // Not a single byte of the original repo's `voice` is fetched (the replacement replaces **the fetch too**).
  for (const part of rig.parts.voice) {
    assertEquals(asked(REPO).includes(part.path), false, `${part.path} is fetched`);
  }
  // `front`, which is not replaced, stays on the original repo.
  for (const part of rig.parts.front.filter((part) => part.size > 0)) {
    assertEquals(asked(REPO).includes(part.path), true, `${part.path} is not fetched`);
  }
});

test("components: a container with a different graph description fails admission (no weight file is fetched)", async () => {
  const rig = await prepareSeats({ sameGraph: false });

  const error = await assertRejects(
    () =>
      loadContainerComponents(
        "test.fromPretrained",
        rig.loaded,
        rig.selection,
        ["front", "voice"],
        NO_FAMILY_GATE,
        { ...rig.hubOptions, components: { voice: { source: rig.source } } },
      ),
    Error,
    "the graph descriptor does not match the manifest's declaration",
  );
  // A form the reader can check against the actual files (the two sha256 of the declaration and the replacement side by side).
  assertStringIncludes(error.message, rig.selection.containers["voice"].descriptor.graph.sha256);
  // It fails **when only the declarations have been read** — no `.krm` has been fetched from either repo.
  assertEquals(fetchedContainers(rig.mock.paths), []);
});

test("components: a role name this family does not have fails with the replaceable roles", async () => {
  const rig = await prepareSeats({ sameGraph: true });

  const error = await assertRejects(
    () =>
      loadContainerComponents(
        "test.fromPretrained",
        rig.loaded,
        rig.selection,
        ["front", "voice"],
        NO_FAMILY_GATE,
        { ...rig.hubOptions, components: { vae: { source: rig.source } } },
      ),
    Error,
    "'vae' in components is not a component of this family",
  );
  assertStringIncludes(error.message, "replaceable roles: front / voice");
  // A misspelling never reaches the fetch.
  assertEquals(fetchedContainers(rig.mock.paths), []);
});

test("components: inherited properties are not used for replacement (only own roles take a seat)", async () => {
  // The roles are checked over own enumerable keys, so reading an inherited value (`Object.create` or prototype
  // pollution) by direct indexing would use a value that never passed the check as a replacement.
  const rig = await prepareSeats({ sameGraph: true });
  const inherited: Record<string, { readonly source: typeof rig.source }> = Object.create({
    voice: { source: rig.source },
  });

  const { open } = await loadContainerComponents(
    "test.fromPretrained",
    rig.loaded,
    rig.selection,
    ["front", "voice"],
    NO_FAMILY_GATE,
    { ...rig.hubOptions, components: inherited },
  );
  assertEquals(open("voice").graph.outputs, ["y"]);
  assertEquals(
    rig.mock.requests.filter((request) => request.repo === OTHER_REPO),
    [],
    "the inherited replacement target is fetched",
  );
  for (const part of rig.parts.voice.filter((part) => part.size > 0)) {
    assertEquals(
      rig.mock.requests.some((request) => request.repo === REPO && request.path === part.path),
      true,
      `${part.path} of the original repo is not fetched`,
    );
  }
});

test("components: a replacement part with the same path as another component of the original repo is counted as two files in progress", async () => {
  // Write the replacement's `voice` with the same stem as the original repo's `front` = parts with the same path in two repos.
  const rig = await prepareSeats({ sameGraph: true, replacementStem: "front/model.f32" });
  const nonEmpty = (parts: readonly { path: string; size: number }[]) =>
    parts.filter((part) => part.size > 0);
  const shared = nonEmpty(rig.parts.front)
    .map((part) => part.path)
    .filter((path) => nonEmpty(rig.parts.replacement).some((part) => part.path === path));
  assertEquals(shared.length > 0, true, "not observed on two files with the same path");

  const events: AssetProgress[] = [];
  await loadContainerComponents(
    "test.fromPretrained",
    rig.loaded,
    rig.selection,
    ["front", "voice"],
    NO_FAMILY_GATE,
    {
      ...rig.hubOptions,
      components: { voice: { source: rig.source } },
      onProgress: (progress) => events.push(progress),
    },
  );

  const total = [...nonEmpty(rig.parts.front), ...nonEmpty(rig.parts.replacement)].reduce(
    (sum, part) => sum + part.size,
    0,
  );
  assertEquals(new Set(events.map((event) => event.total)), new Set([total]));
  assertEquals(events[events.length - 1].loaded, total);
  for (const path of shared) {
    const origins = events
      .filter((event) => event.phase === "complete" && event.path === path)
      .map((event) => event.repo)
      .sort((a, b) => (a! < b! ? -1 : a! > b! ? 1 : 0));
    assertEquals(
      origins,
      [OTHER_REPO, REPO].sort(),
      `the two files at ${path} cannot be told apart`,
    );
  }
});

// ---- Real-GPU gates -------------------------------------------------------------

test.skipIf(!GPU_AVAILABLE)(
  "loadContainerComponents: building a Session does not move progress by a single event (real GPU)",
  async () => {
    const rig = await prepareComponent();
    const fetchedCount = fetched(rig.parts["dit"]).length;

    const events: AssetProgress[] = [];
    const { open } = await loadContainerComponents(
      "test.fromPretrained",
      rig.loaded,
      rig.selection,
      ["dit"],
      NO_FAMILY_GATE,
      { ...rig.hubOptions, onProgress: (progress) => events.push(progress) },
    );
    const afterLoad = events.length;
    const calls = rig.mock.paths.length;
    // complete fires exactly once per file (every file has reached its end when the load returns).
    assertEquals(events.filter((event) => event.phase === "complete").length, fetchedCount);

    const gpu = await acquireGpu();
    try {
      // Weight blocks are only in the last part, so the blocks have been read once the Session is built.
      const session = await open("dit").createSession(gpu);
      await session.dispose();
    } finally {
      gpu.destroy();
    }

    assertEquals(events.length, afterLoad, "building the Session adds progress events");
    assertEquals(rig.mock.paths.length, calls, "building the Session goes to the network");
  },
);

test.skipIf(!GPU_AVAILABLE)(
  "loadContainerComponents: one opener can build two Sessions in a row (real GPU)",
  async () => {
    const rig = await prepareComponent();
    const { open } = await loadContainerComponents(
      "test.fromPretrained",
      rig.loaded,
      rig.selection,
      ["dit"],
      NO_FAMILY_GATE,
      rig.hubOptions,
    );

    const gpu = await acquireGpu();
    try {
      // The second one returns the correct output only when the blocks are re-read **on every call**. Session
      // building does not check that every declared initializer arrived, so with a read-once list the second
      // Session is built on an empty block list — check the run output, not that the build succeeded.
      for (let index = 0; index < 2; index += 1) {
        const session = await open("dit").createSession(gpu);
        assertEquals(await runLinearProbe(session), LINEAR_PROBE_Y, `Session #${index + 1}`);
        await session.dispose();
      }
    } finally {
      gpu.destroy();
    }
  },
);

test.skipIf(!GPU_AVAILABLE)(
  "loadContainerComponents: the load-time signal is not carried into Session building (real GPU)",
  async () => {
    const rig = await prepareComponent();
    const controller = new AbortController();
    const { open } = await loadContainerComponents(
      "test.fromPretrained",
      rig.loaded,
      rig.selection,
      ["dit"],
      NO_FAMILY_GATE,
      { ...rig.hubOptions, signal: controller.signal },
    );

    // The caller's abort knob lives for "this one load" — firing after a successful load is perfectly ordinary,
    // whether by `AbortSignal.timeout` or a screen unmount.
    controller.abort();

    const gpu = await acquireGpu();
    try {
      // If it were carried over, reading the blocks would fail (reliably, even with a complete cache).
      const session = await open("dit").createSession(gpu);
      await session.dispose();
    } finally {
      gpu.destroy();
    }
  },
);

// MUST: an environment without an adapter is skipped explicitly (ADR 0005). The comparison target on this path is
// `readAdapterLimits()`, so without an adapter it turns into GpuUnavailableError and the check is not visible.
test.skipIf(!GPU_AVAILABLE)(
  "requiredLimits beyond the adapter also fetches no weight part on the self-acquired GPU path (measured adapter)",
  async () => {
    // A requirement no real device can meet (`Number.MAX_SAFE_INTEGER` — the largest value the manifest accepts).
    const rig = await prepareSiglip2({ quants: quantsRequiring(Number.MAX_SAFE_INTEGER) });
    const parts = rig.parts["vision"];

    const error = await assertRejects(
      () =>
        siglip2.Pipeline.fromPretrained(
          { repo: REPO, revision: SHA, hubUrl: HUB_URL },
          { fetch: rig.mock.fetch, caches: rig.hubOptions.caches },
        ),
      Error,
    );
    if (!error.message.includes("maxBufferSize")) {
      throw new Error(`not the limits check's message: ${error.message}`);
    }
    assertEquals(rig.mock.paths.includes(parts[0].path), true);
    for (const part of parts.slice(1)) {
      assertEquals(rig.mock.paths.includes(part.path), false, `${part.path} is fetched`);
    }
  },
);
