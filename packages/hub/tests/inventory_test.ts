/**
 * Querying the cache inventory (`listCachedAssets`) and deleting per selection
 * (`evictCachedAssets`).
 *
 * Three points are pinned — ① the identity of an inventory item is its **content key** (a path of
 * the same name in another repository is a separate file); ② deletion does not break other
 * selections (files used by a fully cached selection are kept; a partially cached selection is not
 * protected); ③ a source without a cache refuses and fails loudly.
 *
 * MUST: always pass `caches` (the MUST of helpers/mock.ts — without it the fetch layer writes to
 * Deno's real cache).
 */

import { test } from "vite-plus/test";
import { assert, assertEquals, assertRejects } from "@std/assert";
import {
  evictCachedAssets,
  type FileRef,
  HubError,
  listCachedAssets,
  type LoadedManifest,
  loadManifest,
  localDirectory,
  ManifestReferenceError,
  prefetchAssets,
  type ResolveOptions,
  resolveSelection,
  selectionRefs,
} from "../mod.ts";
import { fileRefKey } from "../src/manifest.ts";
import { DistributionSource, driverOf } from "../src/source.ts";
import { createHfSource } from "../src/sources/hf.ts";
import {
  FETCH_MANIFEST_BYTES,
  fetchManifest,
  STYLE_VECTORS,
  TEXT_CONDITIONER_PARTS,
  TEXT_ENCODER_PARTS,
  TOKENIZER,
  TRANSFORMER_F16_PARTS,
  TRANSFORMER_I8_FETCHED,
  VAE_DECODER_PARTS,
} from "./helpers/fixture.ts";
import { declaredRefs } from "./helpers/selection.ts";
import { buildLocalDist, memoryDirectory, sha256Hex } from "./helpers/local.ts";
import {
  createMockFetch,
  hasEntry,
  HUB_URL,
  hubCache,
  MemoryCacheStorage,
  type MockRoutes,
  payloadFor,
  REPO,
  SHA,
} from "./helpers/mock.ts";

const MANIFEST_PATH = "karume.json";

/** Every file the fixture declares (zero-length parts have no content and are not served). */
const ASSET_PATHS = declaredRefs(fetchManifest).map((ref) => ref.path);

/**
 * The fixture's default selection (anima-turbo / w8a8-s16), and another model's selection that is a
 * subset of it.
 */
const TURBO: ResolveOptions = {};
const LITE: ResolveOptions = { model: "anima-lite", quant: "w8" };

/**
 * The reference list of the default selection (the order of `selectionRefs` — containers in
 * declaration order, parts in index order, assets last).
 */
const TURBO_PATHS = [
  ...TEXT_ENCODER_PARTS,
  ...TEXT_CONDITIONER_PARTS,
  ...TRANSFORMER_I8_FETCHED,
  ...VAE_DECODER_PARTS,
  TOKENIZER,
  STYLE_VECTORS,
];
/** The reference list of anima-lite/w8 (a proper subset of the default selection). */
const LITE_PATHS = [...TEXT_ENCODER_PARTS, ...TRANSFORMER_I8_FETCHED, TOKENIZER, STYLE_VECTORS];
/** The references unique to the default selection (the ones anima-lite does not use). */
const TURBO_ONLY = [...TEXT_CONDITIONER_PARTS, ...VAE_DECODER_PARTS];

const manifestBytes = FETCH_MANIFEST_BYTES;

const serveAll = (): Map<string, Uint8Array<ArrayBuffer>> => {
  const files = new Map<string, Uint8Array<ArrayBuffer>>([[MANIFEST_PATH, manifestBytes]]);
  for (const path of ASSET_PATHS) files.set(path, payloadFor(path));
  return files;
};

const load = async (
  caches: MemoryCacheStorage,
  routes: MockRoutes = { files: serveAll() },
): Promise<{ loaded: LoadedManifest; mock: ReturnType<typeof createMockFetch> }> => {
  const loader = createMockFetch(routes);
  const loaded = await loadManifest(
    { repo: REPO, hubUrl: HUB_URL, revision: SHA },
    {
      fetch: loader.fetch,
      caches,
    },
  );
  return { loaded, mock: createMockFetch(routes) };
};

/**
 * The reference list of a selection. `selectionRefs` drops zero-length parts and returns the list
 * deduped by `fileRefKey` in declaration order, so it can be passed as is to `prefetchAssets` (a
 * surface that does not accept duplicates).
 */
const refsOf = (loaded: LoadedManifest, selection: ResolveOptions = {}): FileRef[] => [
  ...selectionRefs(resolveSelection(loaded.manifest, selection)),
];

const paths = (refs: readonly FileRef[]): string[] => refs.map((ref) => ref.path);

test("listCachedAssets: everything is missing before any fetch (and nothing is fetched)", async () => {
  const caches = new MemoryCacheStorage();
  const { loaded, mock } = await load(caches);

  const inventory = await listCachedAssets(loaded, TURBO, { caches });

  assertEquals(inventory.cached, []);
  // The order is exactly that of selectionRefs (declaration order of weights → part index order →
  // order of assets).
  assertEquals(paths(inventory.missing), TURBO_PATHS);
  assertEquals(mock.calls, [], "the inventory query goes to the network");
});

test("listCachedAssets: a downloaded selection is all cached; another selection with only the shared files is partly cached", async () => {
  const caches = new MemoryCacheStorage();
  const { loaded, mock } = await load(caches);

  await prefetchAssets(loaded, refsOf(loaded, LITE), { fetch: mock.fetch, caches });

  const lite = await listCachedAssets(loaded, LITE, { caches });
  assertEquals(lite.missing, [], "the downloaded selection has missing files");
  assertEquals(paths(lite.cached), LITE_PATHS);

  // The default selection shares LITE_PATHS with anima-lite and lacks its 2 own containers
  // (text_conditioner / vae_decoder).
  const turbo = await listCachedAssets(loaded, TURBO, { caches });
  assertEquals(paths(turbo.cached), LITE_PATHS);
  assertEquals(paths(turbo.missing), TURBO_ONLY);
});

test("listCachedAssets: a quant that does not exist is a ManifestReferenceError", async () => {
  const caches = new MemoryCacheStorage();
  const { loaded } = await load(caches);

  await assertRejects(
    () => listCachedAssets(loaded, { quant: "w4" }, { caches }),
    ManifestReferenceError,
    "w4",
  );
  await assertRejects(
    () => evictCachedAssets(loaded, { quant: "w4" }, { caches }),
    ManifestReferenceError,
    "w4",
  );
});

test("evictCachedAssets: when another selection is fully cached, keeps the shared files and deletes only the unique ones", async () => {
  const caches = new MemoryCacheStorage();
  const { loaded, mock } = await load(caches);
  // Downloading the default selection makes anima-lite/w8, its subset, fully cached too.
  await prefetchAssets(loaded, refsOf(loaded, TURBO), { fetch: mock.fetch, caches });
  assertEquals((await listCachedAssets(loaded, LITE, { caches })).missing, []);

  const result = await evictCachedAssets(loaded, TURBO, { caches });

  assertEquals(paths(result.evicted), TURBO_ONLY);
  assertEquals(paths(result.kept.map((entry) => entry.ref)), LITE_PATHS);
  for (const entry of result.kept) {
    assertEquals(entry.reason, "shared", `wrong reason for ${entry.ref.path}`);
    // f16 / f16-c16 of anima-turbo lack the transformer's f16 container, so they are not protected.
    assertEquals(
      entry.sharedWith,
      ["anima-lite/w8"],
      `wrong protecting selection for ${entry.ref.path}`,
    );
  }

  assertEquals(paths((await listCachedAssets(loaded, TURBO, { caches })).missing), TURBO_ONLY);
  assertEquals(
    (await listCachedAssets(loaded, LITE, { caches })).missing,
    [],
    "a selection that should have been protected dropped to partly cached",
  );
});

test("evictCachedAssets: a partly cached selection is not protected (shared files are deleted too)", async () => {
  const caches = new MemoryCacheStorage();
  const { loaded, mock } = await load(caches);
  // Leave out only one unique container (vae_decoder) of the default selection = partly cached.
  const partial = refsOf(loaded, TURBO).filter((ref) => !VAE_DECODER_PARTS.includes(ref.path));
  await prefetchAssets(loaded, partial, { fetch: mock.fetch, caches });

  const result = await evictCachedAssets(loaded, LITE, { caches });

  assertEquals(result.kept, []);
  assertEquals(paths(result.evicted), LITE_PATHS);
  // Files outside the target selection are not dragged in.
  assert(
    hasEntry(hubCache(caches), payloadFor(TEXT_CONDITIONER_PARTS[0])),
    "even text_conditioner, outside the target, is gone",
  );
});

// ---- Selections with **exactly the same** reference set (the fixture's anima-turbo/f16 and
// f16-c16 — the same weight files, only the session settings differ). Counting them as protecting
// each other means nothing is ever deleted, and evicted is 0 whichever selection is deleted in
// whatever order (reported downstream by anima-web).

/**
 * The 2 selections that point at the same reference set, and the labels that appear in the
 * collateral report.
 */
const F16: ResolveOptions = { quant: "f16" };
const F16_C16: ResolveOptions = { quant: "f16-c16" };
const F16_C16_LABEL = "anima-turbo/f16-c16";
const W8A8_LABEL = "anima-turbo/w8a8-s16";
const LITE_LABEL = "anima-lite/w8";

/**
 * The reference list of anima-turbo/f16 (the order of `selectionRefs` — the transformer is the f16
 * container).
 */
const F16_PATHS = [
  ...TEXT_ENCODER_PARTS,
  ...TEXT_CONDITIONER_PARTS,
  ...TRANSFORMER_F16_PARTS,
  ...VAE_DECODER_PARTS,
  TOKENIZER,
  STYLE_VECTORS,
];

/**
 * Looks up the labels of the protecting selections by path (looking up by position would lean on
 * the list length).
 */
const sharedWithOf = (
  kept: readonly {
    readonly ref: { readonly path: string };
    readonly sharedWith: readonly string[];
  }[],
  target: string,
): readonly string[] => {
  const found = kept.find((entry) => entry.ref.path === target);
  if (found === undefined) throw new Error(`test: ${target} is not in kept`);
  return found.sharedWith;
};

/**
 * Warms both the default selection (w8a8-s16) and f16 = all 4 selections of the fixture are fully
 * cached.
 */
const prefetchAll = async (
  loaded: LoadedManifest,
  mock: ReturnType<typeof createMockFetch>,
  caches: MemoryCacheStorage,
): Promise<void> => {
  await prefetchAssets(loaded, refsOf(loaded, TURBO), { fetch: mock.fetch, caches });
  await prefetchAssets(loaded, refsOf(loaded, F16), { fetch: mock.fetch, caches });
};

test("evictCachedAssets: selections with the same reference set do not count as protecting", async () => {
  const caches = new MemoryCacheStorage();
  const { loaded, mock } = await load(caches);
  await prefetchAll(loaded, mock, caches);

  const result = await evictCachedAssets(loaded, F16, { caches });

  // Only the f16-specific container is deleted (if f16-c16 protected it, the result would be 0, and
  // the siblings could not be untangled in any order).
  assertEquals(paths(result.evicted), [...TRANSFORMER_F16_PARTS]);
  assertEquals(paths(result.kept.map((entry) => entry.ref)), [
    ...TEXT_ENCODER_PARTS,
    ...TEXT_CONDITIONER_PARTS,
    ...VAE_DECODER_PARTS,
    TOKENIZER,
    STYLE_VECTORS,
  ]);
  for (const entry of result.kept) {
    assertEquals(entry.reason, "shared", `wrong reason for ${entry.ref.path}`);
    assert(
      !entry.sharedWith.includes(F16_C16_LABEL),
      `${entry.ref.path} is protected by a sibling with the same set`,
    );
  }
  // Only selections with a proper subset / a different set protect (in the manifest's declaration
  // order).
  assertEquals(
    sharedWithOf(result.kept, TEXT_ENCODER_PARTS[0]),
    [W8A8_LABEL, LITE_LABEL],
    "the selections that protect text_encoder",
  );
  assertEquals(
    sharedWithOf(result.kept, TEXT_CONDITIONER_PARTS[0]),
    [W8A8_LABEL],
    "the selections that protect text_conditioner",
  );
  assertEquals(result.alsoEvicted, [F16_C16_LABEL]);

  // The collateral content — the sibling drops to partly cached (the next load fetches only the
  // missing containers again).
  assertEquals(paths((await listCachedAssets(loaded, F16_C16, { caches })).missing), [
    ...TRANSFORMER_F16_PARTS,
  ]);
  assertEquals((await listCachedAssets(loaded, LITE, { caches })).missing, []);
});

test("evictCachedAssets: with only siblings of the same set, everything is deleted", async () => {
  const caches = new MemoryCacheStorage();
  const { loaded, mock } = await load(caches);
  // Warm only f16 (w8a8-s16 and anima-lite lack the i8 containers, so they are partly cached).
  await prefetchAssets(loaded, refsOf(loaded, F16), { fetch: mock.fetch, caches });

  const result = await evictCachedAssets(loaded, F16, { caches });

  assertEquals(paths(result.evicted), F16_PATHS);
  assertEquals(result.kept, []);
  // The 2 partly cached selections do not count as collateral (they were never "downloaded" to
  // begin with).
  assertEquals(result.alsoEvicted, [F16_C16_LABEL]);
});

test("evictCachedAssets: only the protect list protects (a sibling of the same set can be protected when named)", async () => {
  const caches = new MemoryCacheStorage();
  const { loaded, mock } = await load(caches);
  await prefetchAssets(loaded, refsOf(loaded, F16), { fetch: mock.fetch, caches });

  const result = await evictCachedAssets(loaded, F16, { caches, protect: [F16_C16] });

  assertEquals(result.evicted, []);
  assertEquals(paths(result.kept.map((entry) => entry.ref)), F16_PATHS);
  for (const entry of result.kept) {
    assertEquals(entry.reason, "shared", `wrong reason for ${entry.ref.path}`);
    assertEquals(
      entry.sharedWith,
      [F16_C16_LABEL],
      `wrong protecting selection for ${entry.ref.path}`,
    );
  }
  assertEquals(result.alsoEvicted, [], "claims collateral although nothing was deleted");
});

test("evictCachedAssets: an empty protect does not protect other fully cached selections either", async () => {
  const caches = new MemoryCacheStorage();
  const { loaded, mock } = await load(caches);
  await prefetchAll(loaded, mock, caches);

  const result = await evictCachedAssets(loaded, F16, { caches, protect: [] });

  assertEquals(paths(result.evicted), F16_PATHS);
  assertEquals(result.kept, []);
  // The 3 fully cached selections that were not protected are collateral (in the manifest's
  // declaration order).
  assertEquals(result.alsoEvicted, [W8A8_LABEL, F16_C16_LABEL, LITE_LABEL]);
});

test("evictCachedAssets: the target itself mixed into protect is ignored", async () => {
  const caches = new MemoryCacheStorage();
  const { loaded, mock } = await load(caches);
  await prefetchAll(loaded, mock, caches);

  // The target cannot be protected from itself — the result is the same as `protect: []`.
  const result = await evictCachedAssets(loaded, F16, { caches, protect: [F16, F16] });

  assertEquals(paths(result.evicted), F16_PATHS);
  assertEquals(result.kept, []);
  assertEquals(result.alsoEvicted, [W8A8_LABEL, F16_C16_LABEL, LITE_LABEL]);
});

test("evictCachedAssets: a quant in protect that does not exist is a ManifestReferenceError", async () => {
  const caches = new MemoryCacheStorage();
  const { loaded } = await load(caches);

  await assertRejects(
    () => evictCachedAssets(loaded, F16, { caches, protect: [{ quant: "w4" }] }),
    ManifestReferenceError,
    "w4",
  );
});

// ---- How narrowing weights (`ResolveOptions.weights`) affects the inventory accounting. The
// selection's reference set only gets smaller, so both the query and the deletion count "the
// narrowed part". The rest of the same (model, quant) is not a protecting candidate (entries with
// the same label are excluded from the candidates), so **deleting one role while keeping the main
// weights** can be written — the case of dropping gemma4's `drafter` after speculative decoding is
// no longer used. The shared assets, however, are not protected (`docs/known-issues.md` — in the
// tests below the inventory of another (model, quant) protects them).

/** One role of the default selection (a narrowed selection is a `ResolveOptions` itself). */
const TURBO_VAE: ResolveOptions = { weights: ["vae_decoder"] };

test("listCachedAssets: narrowing weights counts only the narrowed part", async () => {
  const caches = new MemoryCacheStorage();
  const { loaded, mock } = await load(caches);
  await prefetchAssets(loaded, refsOf(loaded, LITE), { fetch: mock.fetch, caches });

  // The default selection lacks 2 containers, but narrowed to text_encoder it answers "downloaded".
  const encoder = await listCachedAssets(loaded, { weights: ["text_encoder"] }, { caches });
  assertEquals(encoder.missing, [], "the inventory of the narrowed role has missing files");
  assertEquals(paths(encoder.cached), [...TEXT_ENCODER_PARTS, TOKENIZER, STYLE_VECTORS]);

  // Narrowed to a missing role, only that container shows up in missing (assets are not narrowed).
  const vae = await listCachedAssets(loaded, TURBO_VAE, { caches });
  assertEquals(paths(vae.cached), [TOKENIZER, STYLE_VECTORS]);
  assertEquals(paths(vae.missing), [...VAE_DECODER_PARTS]);

  assertEquals(paths((await listCachedAssets(loaded, TURBO, { caches })).missing), TURBO_ONLY);
});

test("listCachedAssets: weights that do not exist are a ManifestReferenceError", async () => {
  const caches = new MemoryCacheStorage();
  const { loaded } = await load(caches);

  await assertRejects(
    () => listCachedAssets(loaded, { weights: ["drafter"] }, { caches }),
    ManifestReferenceError,
    "drafter",
  );
  await assertRejects(
    () => evictCachedAssets(loaded, { weights: ["drafter"] }, { caches }),
    ManifestReferenceError,
    "drafter",
  );
});

test("evictCachedAssets: narrowing weights does not delete the rest of the same selection", async () => {
  const caches = new MemoryCacheStorage();
  const { loaded, mock } = await load(caches);
  await prefetchAssets(loaded, refsOf(loaded, TURBO), { fetch: mock.fetch, caches });

  const result = await evictCachedAssets(loaded, TURBO_VAE, { caches });

  // Only the unique files of the narrowed role are deleted. The 2 assets are protected by the fully
  // cached anima-lite/w8.
  assertEquals(paths(result.evicted), [...VAE_DECODER_PARTS]);
  assertEquals(paths(result.kept.map((entry) => entry.ref)), [TOKENIZER, STYLE_VECTORS]);
  for (const entry of result.kept) {
    assertEquals(
      entry.sharedWith,
      [LITE_LABEL],
      `wrong protecting selection for ${entry.ref.path}`,
    );
  }
  // The rest of the same (model, quant) is not even a target (it appears neither as protecting nor
  // as collateral).
  assertEquals(result.alsoEvicted, []);
  assert(
    hasEntry(hubCache(caches), payloadFor(TEXT_CONDITIONER_PARTS[0])),
    "even text_conditioner, outside the narrowing, is gone",
  );
  assertEquals(paths((await listCachedAssets(loaded, TURBO, { caches })).missing), [
    ...VAE_DECODER_PARTS,
  ]);
});

test("evictCachedAssets: protect protects both subsets with the same label", async () => {
  const caches = new MemoryCacheStorage();
  const { loaded, mock } = await load(caches);
  await prefetchAll(loaded, mock, caches);

  // Protect the same selection (w8a8-s16) with 2 subsets. An implementation that folds by label
  // keeps the last one and deletes text_encoder.
  const result = await evictCachedAssets(loaded, F16, {
    caches,
    protect: [
      { quant: "w8a8-s16", weights: ["text_encoder"] },
      { quant: "w8a8-s16", weights: ["vae_decoder"] },
    ],
  });

  assertEquals(paths(result.evicted), [...TEXT_CONDITIONER_PARTS, ...TRANSFORMER_F16_PARTS]);
  assertEquals(paths(result.kept.map((entry) => entry.ref)), [
    ...TEXT_ENCODER_PARTS,
    ...VAE_DECODER_PARTS,
    TOKENIZER,
    STYLE_VECTORS,
  ]);
  for (const entry of result.kept) {
    // The same label does not appear twice (one name even when 2 subsets protect it).
    assertEquals(
      entry.sharedWith,
      [W8A8_LABEL],
      `wrong protecting selection for ${entry.ref.path}`,
    );
  }
  assertEquals(result.alsoEvicted, [W8A8_LABEL, F16_C16_LABEL]);
});

// ---- The dedupe key of the protecting side is **label + reference set** (so that two different
// subsets with the same label are both protected). A spelling that only inserts a separator makes
// the element boundaries ambiguous, and "a selection whose label contains the separator" gets the
// same key as another selection = the last one wins and one of the requests silently disappears.
// Manifest parsing does not reject control characters in model / quant names, so a name with a
// NUL really does reach this point.

const COLLIDE_A_PARTS = ["collide/a-00001-of-00002.krm", "collide/a-00002-of-00002.krm"];
const COLLIDE_B_PARTS = ["collide/b-00001-of-00002.krm", "collide/b-00002-of-00002.krm"];

/**
 * A quant name aimed at folding into the same key as `m/q` (references = a, b) (the label is
 * `m/q\0<the key of a>`).
 */
const COLLIDING_QUANT = `q\0${COLLIDE_A_PARTS[0]}`;

/**
 * A manifest with only 2 weights (`target`, to be deleted, and `m`, listing the protecting
 * candidates).
 */
const collisionManifest = async (): Promise<string> => {
  const partOf = async (path: string) => ({
    path,
    size: payloadFor(path).byteLength,
    sha256: await sha256Hex(payloadFor(path)),
  });
  const containerOf = async (partPaths: readonly string[]) => ({
    container: {
      descriptor: {
        graph: {
          length: payloadFor(partPaths[0]).byteLength - 29,
          sha256: await sha256Hex(payloadFor(`graph:${partPaths[0]}`)),
        },
        model: { length: 5, sha256: await sha256Hex(payloadFor(`model:${partPaths[0]}`)) },
      },
      parts: await Promise.all(partPaths.map(partOf)),
    },
  });
  const weights = {
    wa: { f16: await containerOf(COLLIDE_A_PARTS) },
    wb: { f16: await containerOf(COLLIDE_B_PARTS) },
  };
  const quant = { weights: { wa: "f16", wb: "f16" }, session: {} };
  const model = (quants: Record<string, unknown>, defaultQuant: string) => ({
    pipeline: "anima/1",
    weights,
    assets: {},
    quants,
    defaultQuant,
    pipelineConfig: {},
  });
  return JSON.stringify({
    format: "karume/5",
    generator: "karume/0.1.0",
    defaultModel: "target",
    models: {
      target: model({ only: quant }, "only"),
      m: model({ q: quant, q2: quant, [COLLIDING_QUANT]: quant }, "q"),
    },
  });
};

const COLLISION_TARGET: ResolveOptions = { model: "target", quant: "only" };

/** Opens the manifest with the target's references (a, b) warmed. */
const loadCollision = async (caches: MemoryCacheStorage): Promise<LoadedManifest> => {
  const files = new Map<string, Uint8Array<ArrayBuffer>>(
    [...COLLIDE_A_PARTS, ...COLLIDE_B_PARTS].map((part) => [part, payloadFor(part)]),
  );
  files.set(MANIFEST_PATH, new TextEncoder().encode(await collisionManifest()));
  const { loaded, mock } = await load(caches, { files });
  await prefetchAssets(loaded, refsOf(loaded, COLLISION_TARGET), { fetch: mock.fetch, caches });
  return loaded;
};

test("evictCachedAssets: a NUL in a protect label does not fold the protecting side", async () => {
  const caches = new MemoryCacheStorage();
  const loaded = await loadCollision(caches);

  const result = await evictCachedAssets(loaded, COLLISION_TARGET, {
    caches,
    protect: [
      { model: "m", quant: "q" },
      { model: "m", quant: COLLIDING_QUANT, weights: ["wb"] },
    ],
  });

  // The two are different selections, so both protect (only m/q protects a — folding would delete
  // a).
  assertEquals(paths(result.evicted), []);
  assertEquals(paths(result.kept.map((entry) => entry.ref)), [
    ...COLLIDE_A_PARTS,
    ...COLLIDE_B_PARTS,
  ]);
  assertEquals(sharedWithOf(result.kept, COLLIDE_A_PARTS[0]), ["m/q"]);
  assertEquals(sharedWithOf(result.kept, COLLIDE_B_PARTS[0]), ["m/q", `m/${COLLIDING_QUANT}`]);
  assertEquals(result.alsoEvicted, []);
  assert(
    hasEntry(hubCache(caches), payloadFor(COLLIDE_A_PARTS[0])),
    "a, which should have been protected, is gone",
  );
});

test("evictCachedAssets: with a benign label the same shape is protected as is (control)", async () => {
  const caches = new MemoryCacheStorage();
  const loaded = await loadCollision(caches);

  const result = await evictCachedAssets(loaded, COLLISION_TARGET, {
    caches,
    protect: [
      { model: "m", quant: "q" },
      { model: "m", quant: "q2", weights: ["wb"] },
    ],
  });

  assertEquals(paths(result.evicted), []);
  assertEquals(paths(result.kept.map((entry) => entry.ref)), [
    ...COLLIDE_A_PARTS,
    ...COLLIDE_B_PARTS,
  ]);
  assertEquals(sharedWithOf(result.kept, COLLIDE_A_PARTS[0]), ["m/q"]);
  assertEquals(sharedWithOf(result.kept, COLLIDE_B_PARTS[0]), ["m/q", "m/q2"]);
  assertEquals(result.alsoEvicted, []);
});

// ---- `evicted` contains **only what the source claims it deleted** (it does not return the
// deletable candidates as is). The built-in HF source always deletes exactly the candidates, so
// this contract cannot be observed without wrapping a source that differs (even if the
// implementation degraded to `evicted: candidates`, every existing test would stay green).

/**
 * An HF source that deletes **only the first** of the candidates passed (it neither deletes nor
 * claims the rest).
 */
const firstOnlyEvictSource = (): DistributionSource => {
  const base = driverOf(createHfSource({ repo: REPO, hubUrl: HUB_URL, revision: SHA }));
  return new DistributionSource({
    ...base,
    pin: (generation, options) => {
      const pinned = base.pin(generation, options);
      return {
        ...pinned,
        evict: (refs) => {
          if (pinned.evict === undefined) throw new Error("the HF source has no ⑦ delete");
          return pinned.evict(refs.slice(0, 1));
        },
      };
    },
  });
};

test("evictCachedAssets: evicted is only what the source claims it deleted", async () => {
  const caches = new MemoryCacheStorage();
  const mock = createMockFetch({ files: serveAll() });
  const loaded = await loadManifest(firstOnlyEvictSource(), { fetch: mock.fetch, caches });
  await prefetchAssets(loaded, refsOf(loaded, TURBO), { fetch: mock.fetch, caches });

  const result = await evictCachedAssets(loaded, TURBO, { caches });

  // The deletable candidates are the turbo-specific ones (the rest is protected by the fully cached
  // anima-lite/w8). The source deletes only the first of them.
  assertEquals(paths(result.evicted), [TURBO_ONLY[0]]);
  // Candidates that were not deleted (vae_decoder and others) appear in neither evicted nor kept —
  // kept is only what was protected.
  assertEquals(paths(result.kept.map((entry) => entry.ref)), LITE_PATHS);
  assert(
    hasEntry(hubCache(caches), payloadFor(VAE_DECODER_PARTS[0])),
    "the entry of a reference the source did not delete is gone",
  );
});

/**
 * An HF source that deletes none of the candidates ("someone deleted them first" = the shape with a
 * count of 0).
 */
const noEvictSource = (): DistributionSource => {
  const base = driverOf(createHfSource({ repo: REPO, hubUrl: HUB_URL, revision: SHA }));
  return new DistributionSource({
    ...base,
    pin: (generation, options) => ({
      ...base.pin(generation, options),
      evict: () => Promise.resolve([]),
    }),
  });
};

test("evictCachedAssets: alsoEvicted is only the selections that use a reference actually deleted", async () => {
  const caches = new MemoryCacheStorage();
  const mock = createMockFetch({ files: serveAll() });
  const loaded = await loadManifest(firstOnlyEvictSource(), { fetch: mock.fetch, caches });
  await prefetchAll(loaded, mock, caches);

  // Protect only anima-lite/w8 = the candidates are the parts of F16 that lite does not use. The
  // source deletes only the first of them (part 0 of text_conditioner).
  const result = await evictCachedAssets(loaded, F16, { caches, protect: [LITE] });

  assertEquals(paths(result.evicted), [TEXT_CONDITIONER_PARTS[0]]);
  assertEquals(paths(result.kept.map((entry) => entry.ref)), [
    ...TEXT_ENCODER_PARTS,
    TOKENIZER,
    STYLE_VECTORS,
  ]);
  // anima-lite/w8 stays fully cached and shares text_encoder, but none of those files was deleted,
  // so it does not appear as collateral (counting by candidates would list it).
  assertEquals(result.alsoEvicted, [W8A8_LABEL, F16_C16_LABEL]);
  assertEquals((await listCachedAssets(loaded, LITE, { caches })).missing, []);
  assert(
    hasEntry(hubCache(caches), payloadFor(VAE_DECODER_PARTS[0])),
    "the entry of a candidate the source did not delete is gone",
  );
});

test("evictCachedAssets: when nothing is deleted, alsoEvicted is empty", async () => {
  const caches = new MemoryCacheStorage();
  const mock = createMockFetch({ files: serveAll() });
  const loaded = await loadManifest(noEvictSource(), { fetch: mock.fetch, caches });
  await prefetchAssets(loaded, refsOf(loaded, F16), { fetch: mock.fetch, caches });

  const result = await evictCachedAssets(loaded, F16, { caches });

  // The candidates are every reference of F16 (f16-c16 with the same set does not protect), but the
  // source claims to have deleted none.
  assertEquals(result.evicted, []);
  assertEquals(result.kept, []);
  assertEquals(
    result.alsoEvicted,
    [],
    "claims collateral although no one dropped to partly cached",
  );
  assertEquals((await listCachedAssets(loaded, F16_C16, { caches })).missing, []);
});

// ---- The identity of the content key. The inventory matches by the (path, sha256) pair — folding
// by path alone would answer "cached" for **the old entry** of a file whose content changed with a
// repository update (the fetch misses the content key and downloads again, so only the inventory
// display would lie).

const NEXT_SHA = "abcdef0123456789abcdef0123456789abcdef01";

/**
 * 2 parts whose content changes with the generation (the path is fixed; sha256 is derived from the
 * bytes passed).
 */
const SINGLE_PARTS = ["net/model-00001-of-00002.krm", "net/model-00002-of-00002.krm"];

/**
 * A minimal manifest with only 1 weights entry (every part points at the same bytes = replaced as a
 * whole per generation).
 */
const singleFileManifest = async (bytes: Uint8Array<ArrayBuffer>): Promise<string> => {
  const ref = { size: bytes.byteLength, sha256: await sha256Hex(bytes) };
  return JSON.stringify({
    format: "karume/5",
    generator: "karume/0.1.0",
    defaultModel: "m",
    models: {
      m: {
        pipeline: "anima/1",
        weights: {
          net: {
            i8: {
              container: {
                descriptor: {
                  graph: { length: bytes.byteLength - 29, sha256: await sha256Hex(bytes) },
                  model: { length: 5, sha256: await sha256Hex(bytes) },
                },
                parts: SINGLE_PARTS.map((part) => ({ path: part, ...ref })),
              },
            },
          },
        },
        assets: {},
        quants: { i8: { weights: { net: "i8" }, session: {} } },
        defaultQuant: "i8",
        pipelineConfig: {},
      },
    },
  });
};

test("listCachedAssets: the same path with a different sha256 is not cached", async () => {
  const caches = new MemoryCacheStorage();
  const oldBytes = new TextEncoder().encode("karume-test:weights-old-generation");
  const newBytes = new TextEncoder().encode("karume-test:weights-new-generation-payload");
  const encoder = new TextEncoder();
  // 2 generations of the same repo (the manifest is keyed by a revision-pinned URL, so each
  // generation is a separate entry).
  const files = new Map<string, Uint8Array<ArrayBuffer>>([
    [`${REPO}@${SHA}/${MANIFEST_PATH}`, encoder.encode(await singleFileManifest(oldBytes))],
    [`${REPO}@${NEXT_SHA}/${MANIFEST_PATH}`, encoder.encode(await singleFileManifest(newBytes))],
    ...SINGLE_PARTS.flatMap((part) => [
      [`${REPO}@${SHA}/${part}`, oldBytes] as const,
      [`${REPO}@${NEXT_SHA}/${part}`, newBytes] as const,
    ]),
  ]);
  const previous = createMockFetch({ files });
  const before = await loadManifest(
    { repo: REPO, hubUrl: HUB_URL, revision: SHA },
    {
      fetch: previous.fetch,
      caches,
    },
  );
  await prefetchAssets(before, refsOf(before), { fetch: previous.fetch, caches });
  assertEquals((await listCachedAssets(before, {}, { caches })).missing, []);

  const next = createMockFetch({ files });
  const updated = await loadManifest(
    { repo: REPO, hubUrl: HUB_URL, revision: NEXT_SHA },
    {
      fetch: next.fetch,
      caches,
    },
  );

  const inventory = await listCachedAssets(updated, {}, { caches });
  assertEquals(
    inventory.cached,
    [],
    "answers the old entry of a path whose content changed as cached",
  );
  assertEquals(paths(inventory.missing), SINGLE_PARTS);
});

// ---- Cross-repository references (`repo` / `revision` of `FileRef` — ADR 0038 §7). The inventory
// key is the content key of **the referenced repo**, so warming a path of the same name in the
// referencing repo does not make the cross-repository files cached.

const FOREIGN_REPO = "someone/text-stack";
const FOREIGN_SHA = "89abcdef0123456789abcdef0123456789abcdef";
/**
 * This repository and the referenced one claim the same path for part 0 (a detector for an
 * inventory folded by path).
 */
const CROSS_PATH = "text_encoder/model-00001-of-00002.krm";
const OWN_PART1 = "own/model-00002-of-00002.krm";
const BORROWED_PART1 = "borrowed/model-00002-of-00002.krm";

const ownBytes = new TextEncoder().encode("karume-test:own-text-encoder-part0");
const foreignBytes = new TextEncoder().encode("karume-test:foreign-text-encoder-payload");

/** The declaration of part 0 (exactly `24 + graph + model` — container-v1 §8). */
const descriptorFor = (part0Bytes: number) => ({
  graph: { length: part0Bytes - 24 - 5, sha256: "f2".repeat(32) },
  model: { length: 5, sha256: "a3".repeat(32) },
});

const partRef = async (part: string) => ({
  path: part,
  size: payloadFor(part).byteLength,
  sha256: await sha256Hex(payloadFor(part)),
});

/**
 * A cross-repository reference is **per container** (ADR 0109 decision 3), so on the borrowing side
 * every part names the referenced coordinates.
 */
const crossRepoManifest = JSON.stringify({
  format: "karume/5",
  generator: "karume/0.1.0",
  defaultModel: "m",
  models: {
    m: {
      pipeline: "anima/1",
      weights: {
        own: {
          i8: {
            container: {
              descriptor: descriptorFor(ownBytes.byteLength),
              parts: [
                {
                  path: CROSS_PATH,
                  size: ownBytes.byteLength,
                  sha256: await sha256Hex(ownBytes),
                },
                await partRef(OWN_PART1),
              ],
            },
          },
        },
        borrowed: {
          i8: {
            container: {
              descriptor: descriptorFor(foreignBytes.byteLength),
              parts: [
                {
                  path: CROSS_PATH,
                  size: foreignBytes.byteLength,
                  sha256: await sha256Hex(foreignBytes),
                  repo: FOREIGN_REPO,
                  revision: FOREIGN_SHA,
                },
                {
                  ...(await partRef(BORROWED_PART1)),
                  repo: FOREIGN_REPO,
                  revision: FOREIGN_SHA,
                },
              ],
            },
          },
        },
      },
      assets: {},
      quants: { i8: { weights: { own: "i8", borrowed: "i8" }, session: {} } },
      defaultQuant: "i8",
      pipelineConfig: {},
    },
  },
});

const crossRepoRoutes = (): MockRoutes => ({
  sha: SHA,
  files: new Map([
    [MANIFEST_PATH, new TextEncoder().encode(crossRepoManifest)],
    [CROSS_PATH, ownBytes],
    [OWN_PART1, payloadFor(OWN_PART1)],
    [`${FOREIGN_REPO}@${FOREIGN_SHA}/${CROSS_PATH}`, foreignBytes],
    [`${FOREIGN_REPO}@${FOREIGN_SHA}/${BORROWED_PART1}`, payloadFor(BORROWED_PART1)],
  ]),
});

test("listCachedAssets: the inventory of a cross-repository reference is looked up by the referenced repo's key", async () => {
  const caches = new MemoryCacheStorage();
  const { loaded, mock } = await load(caches, crossRepoRoutes());
  const refs = refsOf(loaded);
  // Declaration order is the 2 parts of own → the 2 parts of borrowed.
  const own = refs.slice(0, 2);
  const borrowed = refs.slice(2);

  // Warm the same path only in the referencing repo (an implementation that folds by path would
  // make the cross-repository files cached here too).
  await prefetchAssets(loaded, own, { fetch: mock.fetch, caches });
  const partial = await listCachedAssets(loaded, {}, { caches });
  assertEquals(partial.cached.map(fileRefKey), own.map(fileRefKey));
  assertEquals(
    partial.missing.map(fileRefKey),
    borrowed.map(fileRefKey),
    "the cross-repository reference is cached under the referencing repo's key",
  );

  await prefetchAssets(loaded, borrowed, { fetch: mock.fetch, caches });
  assertEquals(
    (await listCachedAssets(loaded, {}, { caches })).missing,
    [],
    "the inventory is not found under the referenced repo's content key",
  );
});

test("evictCachedAssets: a cross-repository reference is kept as cross-repo (its entries are not deleted either)", async () => {
  const caches = new MemoryCacheStorage();
  const { loaded, mock } = await load(caches, crossRepoRoutes());
  const refs = refsOf(loaded);
  const own = refs.slice(0, 2);
  const borrowed = refs.slice(2);
  await prefetchAssets(loaded, refs, { fetch: mock.fetch, caches });

  const result = await evictCachedAssets(loaded, {}, { caches });

  assertEquals(result.evicted.map(fileRefKey), own.map(fileRefKey));
  assertEquals(
    result.kept.map((entry) => fileRefKey(entry.ref)),
    borrowed.map(fileRefKey),
  );
  for (const entry of result.kept) {
    assertEquals(entry.reason, "cross-repo", `wrong reason for ${entry.ref.path}`);
    assertEquals(entry.sharedWith, []);
  }
  assert(hasEntry(hubCache(caches), foreignBytes), "the cross-repository entry is gone");
  assert(!hasEntry(hubCache(caches), ownBytes), "the target's entry remains");
});

// ---- The local source (`localDirectory`) — the two answers of a source without a cache.

const LOCAL_LABEL = "./models/karume-test";

const openLocal = async (caches: MemoryCacheStorage): Promise<LoadedManifest> => {
  const dist = await buildLocalDist();
  const directory = memoryDirectory(dist.files);
  return await loadManifest(localDirectory(directory.adapter, { label: LOCAL_LABEL }), { caches });
};

test("localDirectory: the inventory is always all cached (there is nothing to warm)", async () => {
  const caches = new MemoryCacheStorage();
  const loaded = await openLocal(caches);

  const inventory = await listCachedAssets(loaded, {}, { caches });

  assertEquals(inventory.missing, []);
  assertEquals(inventory.cached.length, refsOf(loaded).length);
});

test("evictCachedAssets: a source without a cache refuses with HubError", async () => {
  const caches = new MemoryCacheStorage();
  const loaded = await openLocal(caches);

  const error = await assertRejects(() => evictCachedAssets(loaded, {}, { caches }), HubError);
  assert(error.message.includes(LOCAL_LABEL), `${error.message} does not name which source it is`);
});
