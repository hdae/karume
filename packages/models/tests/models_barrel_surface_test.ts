// The named gate of the barrel (`mod.ts`) and the subpaths. Needs neither a GPU nor real assets.
//
// The barrel exports two kinds of thing: the shared surface, flat (`SHARED_VALUES` +
// the five shared types), and one namespace per family that **is** the subpath's module
// (`export * as sbv2 from "./sbv2.ts"`). The gate holds the shape:
//
// ① every `exports` key of `package.json` other than `.` has a namespace in the barrel, named by
//   the key in camelCase, and that namespace is the very module object of the subpath (identity,
//   not a member-by-member comparison — the barrel and a subpath cannot drift apart);
// ② the barrel's own values are exactly the shared values plus those namespaces (a family name
//   exported flat again, or a new flat export, is red — added exports pass the type checker);
// ③ the intent per family: what must be public and what must not (the gemma lists, the source
//   tables, gemma4-qat).
//
// It is the declaration of intent. The snapshot gate (`public_surface_test.ts`) detects every
// change, type exports included; type exports cannot be observed at run time, so the shared types
// are pinned here by type-level bindings and their exact set by the snapshot.
//
// ADR [0083](../../../docs/decisions/0083-generation-api-surface.md) decision 9 **removed**
// `generateGreedy` / `GreedySpec` (a breaking change — the consumer-facing doc is
// [limitations](../../../docs/limitations.md)). The implementation stays in
// `src/generation/greedy.ts` as an internal helper for parity acceptance, so one line in an entry
// would bring the public surface back.

import { test } from "vite-plus/test";
import { assert, assertEquals } from "@std/assert";
import config from "../package.json" with { type: "json" };
import * as models from "../mod.ts";
import type {
  ComponentSource,
  DecodedWav,
  FromPretrainedComponentOptions,
  FromPretrainedHubOptions,
  gemma,
  Rgb8Image,
} from "../mod.ts";
import type { Gemma4FromPretrainedOptions as SubpathFromPretrained } from "../gemma.ts";

/** The barrel's flat values (the shared surface); every other barrel value is a family namespace. */
const SHARED_VALUES = [
  "KARUME_SOURCES",
  "ModelInputError",
  "decodeWav",
  "encodePng",
  "encodeWav",
  "normalizeToNchw",
  "resizeRgb8",
];

/** The barrel namespace of a subpath: the `exports` key without `./`, in camelCase. */
const namespaceOf = (subpath: string): string =>
  subpath.slice(2).replaceAll(/-([a-z0-9])/g, (_match, next: string) => next.toUpperCase());

/** The subpaths of `package.json` `exports` (everything but `.`), with their entry files. */
const subpathEntries = (): readonly { subpath: string; file: string }[] =>
  Object.entries(config.exports)
    .filter(([subpath]) => subpath !== ".")
    .map(([subpath, file]) => ({ subpath, file }));

/** Each subpath's module, imported from its entry file (the source of truth is `exports`). */
const subpathModules = async (): Promise<
  readonly { subpath: string; module: Record<string, unknown> }[]
> =>
  Promise.all(
    subpathEntries().map(async ({ subpath, file }) => {
      const module: Record<string, unknown> = await import(
        new URL(`../${file}`, import.meta.url).href
      );
      return { subpath, module };
    }),
  );

test("barrel: the namespace names follow the subpath keys", () => {
  assertEquals(namespaceOf("./sbv2"), "sbv2");
  assertEquals(namespaceOf("./depth-anything"), "depthAnything");
  assertEquals(namespaceOf("./gemma4-qat"), "gemma4Qat");
  assertEquals(namespaceOf("./vowel-detector"), "vowelDetector");
});

test("barrel: each family namespace is its subpath's module (identity)", async () => {
  const barrel: Record<string, unknown> = models;
  const modules = await subpathModules();
  // Positive control — the scan is not empty and the subpaths are distinct module objects (an
  // identity check against one shared object would prove nothing).
  assert(modules.length >= 1, "package.json exports has no subpath");
  assertEquals(new Set(modules.map(({ module }) => module)).size, modules.length);
  for (const { subpath, module } of modules) {
    const namespace = namespaceOf(subpath);
    assert(
      Object.hasOwn(barrel, namespace),
      `the barrel has no namespace ${namespace} (${subpath})`,
    );
    assert(
      barrel[namespace] === module,
      `barrel.${namespace} is not the module of ${subpath} (export * as ${namespace} from it)`,
    );
  }
});

test("barrel: its own values are exactly the shared values plus the family namespaces", () => {
  const expected = [
    ...SHARED_VALUES,
    ...subpathEntries().map(({ subpath }) => namespaceOf(subpath)),
  ];
  assertEquals(
    Object.keys(models).toSorted(),
    expected.toSorted(),
    "the barrel's flat exports changed — family names belong in their namespace, shared ones in SHARED_VALUES",
  );
});

// The five shared types (`ComponentSource`, `DecodedWav`, `FromPretrainedComponentOptions`,
// `FromPretrainedHubOptions`, `Rgb8Image`) cannot be observed at run time. The gate is the
// annotations below — a missing flat export is a type error in the test lane — and the snapshot
// gate holds the exact set; the asserts only keep the bindings from reading as unused locals.
test("barrel: the shared types are exported flat", () => {
  const hub: FromPretrainedHubOptions = {};
  const source: ComponentSource = { source: "owner/name", quant: "i4" };
  const components: FromPretrainedComponentOptions = { components: { model: source } };
  const image: Rgb8Image = { data: new Uint8Array(3), width: 1, height: 1 };
  const wav: DecodedWav = { data: new Float32Array(1), sampleRate: 16_000 };
  assertEquals(Object.keys(hub), []);
  assertEquals(components.components?.["model"], source);
  assertEquals([image.width, wav.sampleRate], [1, 16_000]);
});

// Type access through a namespace (`gemma.Gemma4FromPretrainedOptions`) and the component
// replacement option (ADR 0108 decision 19 / ADR 0109 decision 10) must agree with the subpath's
// type. Only the type check of the test lane can fail here.
test("barrel: types through a namespace are the subpath's types", () => {
  const source: ComponentSource = { source: "owner/name", quant: "i4" };
  const options: gemma.Gemma4FromPretrainedOptions = {
    components: { model: source },
  } satisfies SubpathFromPretrained;
  assertEquals(Object.keys(options.components ?? {}), ["model"]);
});

test("barrel: the generation loop is not public (ADR 0083 decision 9)", async () => {
  const surfaces = [
    { name: ".", keys: Object.keys(models) },
    ...(await subpathModules()).map(({ subpath, module }) => ({
      name: subpath,
      keys: Object.keys(module),
    })),
  ];
  // Positive control — the gate is not green because it looked at an empty namespace.
  assert(surfaces[0]?.keys.includes("encodePng"), "the barrel import is broken");
  for (const { name, keys } of surfaces) {
    assertEquals(
      keys.filter((exported) => exported.toLowerCase().includes("greedy")),
      [],
      `${name} exports a greedy name again`,
    );
  }
});

test("barrel: generateGreedy remains as an internal helper (only the public surface lost it)", async () => {
  const greedy = await import("../src/generation/greedy.ts");
  assertEquals(typeof greedy.generateGreedy, "function");
  assertEquals(typeof greedy.planPrefillChunks, "function");
});

// ---- gemma (step 4 of the generation-API phase) -----------------------------------------------
//
// Only the value exports (types cannot be observed at run time). Two things are pinned:
//
// ① what must be public is public — a thinner surface sends consumers to the internal paths;
// ② **the assembly entry points stay private** — `createGenerationProgram` /
//    `createGenerationSequence` / `createGemma4Ple` could build half-made objects before the
//    static wiring check and the asset reconciliation; the pipeline is the only entry point
//    (ADR 0008). An added export passes the type checker.

/** The gemma / generation-API **values** of `./gemma` (and so of the barrel's `gemma`). */
const GEMMA_VALUES = [
  "Gemma4Pipeline",
  "gemma4ChatPrompt",
  // The entry point for consumers that run several turns with `sequence`. Without it they guess
  // ("render everything and strip the leading `<bos>`") — the turn-local contract is not public.
  "gemma4ChatTurn",
  "GenerationCapacityError",
  // How a `fromAssets` consumer builds the config from the hub's `Record<string, unknown>`.
  // Without it they cast with `as`, or keep the three numbers in both the distribution and the
  // consumer.
  "parseGemma4PipelineConfig",
  // RoPE cos / sin are not baked into the graph; the host builds them per chunk (ADR 0091
  // decision 1). Low-level consumers (`fromAssets` / their own generation loop) would have to
  // build the tables themselves.
  "gemma4RopeInputs",
  // The high-level surface for multi-turn conversations that carry the KV over (ADR 0083 addendum
  // 2026-09-02).
  "Gemma4ChatSession",
  // The default overflow policy. Consumers **replace it by name**, so it must be public (otherwise
  // the only way is to rewrite the default).
  "dropOldestTurns",
  // The table of sources verified with this package version (ADR 0073 / 0092). Its values are
  // checked by `sources_test.ts`; here only that it is public.
  "GEMMA4_SOURCES",
  // The default prefill buckets. Consumers **start from it and add or remove entries**.
  "GEMMA4_CHUNK_BUCKETS",
];

/** Spellings that must not be public (internal assembly). */
const INTERNAL_VALUES = [
  "createGenerationProgram",
  "createGenerationSequence",
  "createSampler",
  "createGemma4Ple",
  "parseGemma4PleIndex",
  "parseGemmaTokenizerAsset",
  "renderGemma4Chat",
  "renderGemma4ChatTurn",
  "gemma4StopTokens",
  "GemmaTokenizer",
  // The chat layer's internals (`gemma4_chat_test.ts` takes them from `src/` directly). The public
  // surface is `Gemma4Pipeline.chat` / `Gemma4ChatSession`; the streaming-decode parts stay
  // private — exporting them would make "wire the stop-string filter to the detokenizer yourself"
  // the consumers' normal path. `StreamingDetokenizer` is re-exported with `export type`, so it is
  // not public **as a value**.
  "createStopStringFilter",
  "decodeChatChunks",
  "chatStreamOf",
  "StreamingDetokenizer",
];

test("barrel: the gemma surface (thin — no assembly entry points)", async () => {
  const subpath = Object.keys(await import("../gemma.ts"));
  const barrel = Object.keys(models);
  for (const name of GEMMA_VALUES) assert(subpath.includes(name), `./gemma has no ${name}`);
  for (const name of INTERNAL_VALUES) {
    assertEquals(subpath.includes(name), false, `./gemma exports the internal ${name}`);
    assertEquals(barrel.includes(name), false, `the barrel exports the internal ${name}`);
  }
});

// ---- the source tables (ADR 0092 — one table per family) -------------------------------------
//
// The tables' **contents** (key rules, repository names, 40-hex revisions) are checked by
// `sources_test.ts`. Here:
//
// ① each family table is exported by its subpath (and therefore by the barrel's namespace);
// ② `KARUME_SOURCES` is exported by the barrel only, and is exactly the union of the family tables
//   (a new family table that was not folded into `src/sources.ts` is red — the spread there
//   cannot be enumerated at run time).
//
// A family without a table (`./vowel-detector`, families not distributed yet) exports none.

/**
 * The source table each distributed family exports, by subpath. A family with bare public names
 * exports `SOURCES`; a family still spelled with its prefix exports `<FAMILY>_SOURCES`.
 */
const SOURCE_TABLES: Readonly<Record<string, string>> = {
  "./anima": "SOURCES",
  "./irodori": "IRODORI_SOURCES",
  "./sbv2": "SBV2_SOURCES",
  "./gemma": "GEMMA4_SOURCES",
  "./siglip2": "SIGLIP2_SOURCES",
  "./depth-anything": "DEPTH_ANYTHING_SOURCES",
  "./birefnet": "SOURCES",
  "./umt5": "UMT5_SOURCES",
  "./wan": "WAN_SOURCES",
};

/** Whether an export name spells a family source table (bare or prefixed). */
const isSourceTable = (exported: string): boolean =>
  exported === "SOURCES" || exported.endsWith("_SOURCES");

test("barrel: the source tables come from the subpaths; KARUME_SOURCES only from the barrel", async () => {
  const modules = await subpathModules();
  const found = modules.flatMap(({ subpath, module }) =>
    Object.keys(module)
      .filter((exported) => exported !== "KARUME_SOURCES" && isSourceTable(exported))
      .map((exported) => `${subpath} ${exported}`),
  );
  // Both directions: a subpath that dropped its table, and a table that is not listed (and so
  // perhaps not folded into `src/sources.ts`), are red.
  assertEquals(
    found.toSorted(),
    Object.entries(SOURCE_TABLES)
      .map(([subpath, exported]) => `${subpath} ${exported}`)
      .toSorted(),
    "the source tables exported by the subpaths changed",
  );
  // The merged table lives in the barrel only (in a subpath, one family's import would pull in
  // every family).
  assert(Object.hasOwn(models, "KARUME_SOURCES"), "the barrel has no KARUME_SOURCES");
  for (const { subpath, module } of modules) {
    assertEquals(
      Object.hasOwn(module, "KARUME_SOURCES"),
      false,
      `${subpath} exports KARUME_SOURCES`,
    );
  }
});

test("barrel: KARUME_SOURCES is exactly the union of the family tables", async () => {
  const keys = new Set<string>();
  for (const { module } of await subpathModules()) {
    for (const [name, table] of Object.entries(module)) {
      if (!isSourceTable(name) || typeof table !== "object" || table === null) continue;
      for (const key of Object.keys(table)) keys.add(key);
    }
  }
  // Positive control — the scan picked up at least one table.
  assert(keys.size >= 1, "no family table key was found in the subpaths");
  assertEquals(
    [...keys].toSorted(),
    Object.keys(models.KARUME_SOURCES).toSorted(),
    "KARUME_SOURCES differs from the union of the family tables (missing from src/sources.ts, or extra)",
  );
});

test("gemma4-qat: exports its own entry points and not Gemma's pipeline", async () => {
  const qat = await import("../gemma4-qat.ts");
  for (const name of ["Gemma4QatPipeline", "gemma4QatRopeInputs"]) {
    assert(Object.hasOwn(qat, name), `./gemma4-qat has no ${name}`);
  }
  for (const name of ["GemmaPipeline", "admitGemma4Qat", "Gemma4Pipeline", "GEMMA4_SOURCES"]) {
    assertEquals(Object.hasOwn(qat, name), false, `./gemma4-qat exports ${name}`);
  }
});
