import { readTextFile } from "@karume/test-support/fs";
import { describe, it, test } from "vite-plus/test";
import { assert, assertEquals, assertThrows } from "@std/assert";
import {
  HubError,
  ManifestFormatError,
  ManifestPathError,
  ManifestReferenceError,
  parseManifest,
} from "../mod.ts";

// tests/fixtures/manifest-invalid.json is the source of truth for which malicious / broken
// manifests are accepted or rejected. This file only runs every row of the table and creates no
// second definition on the TS side (only the boundary values are built, since writing them by hand
// is impractical).

type InvalidCase = {
  readonly name: string;
  readonly error: string;
  readonly manifest: unknown;
};

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value);

/** Whether the declared error name matches the actual type (down to class identity). */
const matchesDeclaredClass = (error: HubError, declared: string): boolean => {
  switch (declared) {
    case "ManifestFormatError":
      return error instanceof ManifestFormatError;
    case "ManifestReferenceError":
      return error instanceof ManifestReferenceError;
    case "ManifestPathError":
      return error instanceof ManifestPathError;
    default:
      return false;
  }
};

const loadInvalidCases = async (): Promise<readonly InvalidCase[]> => {
  const raw: unknown = JSON.parse(
    await readTextFile(new URL("./fixtures/manifest-invalid.json", import.meta.url)),
  );
  if (!isRecord(raw) || !Array.isArray(raw["cases"])) throw new Error("fixture: cases missing");
  const cases = raw["cases"].map((entry: unknown): InvalidCase => {
    if (!isRecord(entry) || typeof entry["name"] !== "string") {
      throw new Error(`fixture: malformed case ${JSON.stringify(entry)}`);
    }
    const error = entry["error"];
    if (typeof error !== "string") {
      throw new Error(`fixture: error name is not a string ${JSON.stringify(error)}`);
    }
    return { name: entry["name"], error, manifest: entry["manifest"] };
  });
  if (cases.length === 0) throw new Error("fixture: no cases");
  return cases;
};

const invalidCases = await loadInvalidCases();

const validManifestText = await readTextFile(
  new URL("./fixtures/manifest-fetch.json", import.meta.url),
);

/** One asset (a quant-independent file outside the containers). */
const FILE = { path: "tokenizer/tokenizer.json", size: 12, sha256: "a1".repeat(32) };

/** The container header length (container-v1 §1). Part 0 is exactly "header + the 2 documents". */
const HEADER_BYTES = 24;
const GRAPH_LENGTH = 11;
const MODEL_LENGTH = 5;
/** `24 + 11 + 5` — the size of part 0 must be exactly this value. */
const PART0_SIZE = HEADER_BYTES + GRAPH_LENGTH + MODEL_LENGTH;

const DESCRIPTOR = {
  graph: { length: GRAPH_LENGTH, sha256: "f2".repeat(32) },
  model: { length: MODEL_LENGTH, sha256: "a3".repeat(32) },
};

/**
 * The declaration of one part. `index` is **the index** (starting at 0), and the numbering in the
 * file name follows the writer's spelling (starting at 1 — `container_paths` of
 * `tools/exporter/src/karume/container.py`). The hub does not check file names, but they help a
 * person comparing against a real mirror.
 */
const part = (index: number, total: number, size: number, mark: string) => ({
  path: `net/model-${String(index + 1).padStart(5, "0")}-of-${String(total).padStart(5, "0")}.krm`,
  size,
  sha256: mark.repeat(32),
});

const PART0 = part(0, 2, PART0_SIZE, "b2");
const PART1 = part(1, 2, 64, "c3");

/** The minimal valid container (part 0 + part 1). */
const CONTAINER = { descriptor: DESCRIPTOR, parts: [PART0, PART1] };

/** A weights entry with `container` partly replaced. */
const withContainer = (patch: Record<string, unknown>): Record<string, unknown> => ({
  net: { f16: { container: { ...CONTAINER, ...patch } } },
});

/**
 * The minimal v5 manifest with only the fields the checks need. `patch` overrides the contents of
 * `models.m`, `envelope` the top level.
 */
const withModel = (
  patch: Record<string, unknown> = {},
  envelope: Record<string, unknown> = {},
): string =>
  JSON.stringify({
    format: "karume/5",
    generator: "karume/0.1.0",
    defaultModel: "m",
    models: {
      m: {
        pipeline: "anima/1",
        weights: { net: { f16: { container: CONTAINER } } },
        assets: {},
        quants: { q: { weights: { net: "f16" }, session: {} } },
        defaultQuant: "q",
        pipelineConfig: {},
        ...patch,
      },
    },
    ...envelope,
  });

/** A shortcut that reads `models.m.weights.net.f16.container`. */
const containerOf = (text: string) =>
  parseManifest(text).models["m"].weights["net"]["f16"].container;

describe("parseManifest: every violation case of the fixture turns red with the declared error type", () => {
  for (const testCase of invalidCases) {
    it(`${testCase.name} → ${testCase.error}`, () => {
      const thrown = assertThrows(() => parseManifest(JSON.stringify(testCase.manifest)), HubError);
      assertEquals(thrown.name, testCase.error);
      assert(
        matchesDeclaredClass(thrown, testCase.error),
        `${testCase.name}: ${thrown.name} is not the declared ${testCase.error}`,
      );
    });
  }
});

test("parseManifest: broken JSON is rethrown wrapped in ManifestFormatError", () => {
  const error = assertThrows(() => parseManifest('{"format": "karume/5",}'), ManifestFormatError);
  assert(error.cause instanceof SyntaxError, "keeps the original SyntaxError as cause");
});

test("parseManifest: v1 (karume/1) is not read and fails as an unsupported major", () => {
  // ADR 0109 decision 1: the hub reads only the current version (no two-format parsing). As the
  // mirror image of old clients, there is no path either where a new client **silently runs an
  // old manifest under the old interpretation**.
  const error = assertThrows(
    () =>
      parseManifest(
        JSON.stringify({
          format: "karume/1",
          generator: "karume/0.1.0",
          pipeline: "anima/1",
          components: { net: { file: FILE } },
          presets: { p: { weights: {}, session: {} } },
          defaultPreset: "p",
          pipelineConfig: {},
        }),
      ),
    ManifestFormatError,
  );
  assert(
    error.message.includes("karume/5"),
    `${error.message} does not state "only karume/5 can be read"`,
  );
});

test("parseManifest: the previous version (karume/4) is not read either, and the error states that the current one is karume/5", () => {
  // v4 and v5 differ only in the content of a dtype entry (`shards` + `extras` → `container`); the
  // other fields are unchanged. Starting from the structure would give a peripheral diagnostic,
  // "unknown key 'shards'", hiding the real reason (this version reads only karume/5). Only the
  // format string declares the break (ADR 0109 decision 1).
  const error = assertThrows(
    () =>
      parseManifest(
        JSON.stringify({
          format: "karume/4",
          generator: "karume/0.1.0",
          defaultModel: "m",
          models: {
            m: {
              pipeline: "anima/1",
              weights: {
                net: { f16: { shards: [{ ...PART1, path: "net/model.f16.safetensors" }] } },
              },
              assets: {},
              quants: { q: { weights: { net: "f16" }, session: {} } },
              defaultQuant: "q",
              pipelineConfig: {},
            },
          },
        }),
      ),
    ManifestFormatError,
  );
  assert(error.message.includes("karume/4"), `${error.message} does not name the rejected version`);
  assert(
    error.message.includes("karume/5"),
    `${error.message} does not state "only karume/5 can be read"`,
  );
  assert(
    error.message.includes("older versions"),
    `${error.message} does not state "there is no parser for older versions"`,
  );
  assert(
    !error.message.includes("'shards'"),
    `${error.message} makes a peripheral unknown key the main cause`,
  );
});

test("parseManifest: the karume/2 spelling (a dtype entry is {file}) also fails on the version", () => {
  // The observation point that format is checked before unknown keys (exercising the same reason as
  // v4 above with an older spelling).
  const error = assertThrows(
    () =>
      parseManifest(
        JSON.stringify({
          format: "karume/2",
          generator: "karume/0.1.0",
          defaultModel: "m",
          models: {
            m: {
              pipeline: "anima/1",
              weights: { net: { f16: { file: FILE } } },
              assets: {},
              quants: { q: { weights: { net: "f16" }, session: {} } },
              defaultQuant: "q",
              pipelineConfig: {},
            },
          },
        }),
      ),
    ManifestFormatError,
  );
  assert(error.message.includes("karume/2"), `${error.message} does not name the rejected version`);
  assert(
    !error.message.includes("'file'"),
    `${error.message} makes a peripheral unknown key the main cause`,
  );
});

describe("parseManifest: rejects by the numeric size limits", () => {
  const modelEntry = (quantName: string) => ({
    pipeline: "anima/1",
    weights: { net: { f16: { container: CONTAINER } } },
    assets: {},
    quants: { [quantName]: { weights: { net: "f16" }, session: {} } },
    defaultQuant: quantName,
    pipelineConfig: {},
  });

  it("33 models", () => {
    let models: Record<string, unknown> = {};
    for (let index = 0; index < 33; index += 1) {
      models = { ...models, [`m${index}`]: modelEntry("q") };
    }
    const text = JSON.stringify({
      format: "karume/5",
      generator: "karume/0.1.0",
      defaultModel: "m0",
      models,
    });
    assertThrows(() => parseManifest(text), ManifestFormatError);
  });

  it("33 weights", () => {
    let weights: Record<string, unknown> = {};
    let mapping: Record<string, string> = {};
    for (let index = 0; index < 33; index += 1) {
      weights = { ...weights, [`w${index}`]: { f16: { container: CONTAINER } } };
      mapping = { ...mapping, [`w${index}`]: "f16" };
    }
    assertThrows(
      () =>
        parseManifest(
          withModel({
            weights,
            quants: { q: { weights: mapping, session: {} } },
          }),
        ),
      ManifestFormatError,
    );
  });

  it("1025 parts (1024 pass)", () => {
    const parts = (count: number): unknown[] => [
      part(0, count, PART0_SIZE, "b2"),
      ...Array.from({ length: count - 1 }, (_unused, index) => part(index + 1, count, 8, "c3")),
    ];
    // Also checks that exactly the limit passes (one side alone would be green even for an
    // implementation that always fails).
    const accepted = parseManifest(withModel({ weights: withContainer({ parts: parts(1024) }) }));
    assertEquals(accepted.models["m"].weights["net"]["f16"].container.parts.length, 1024);
    assertThrows(
      () => parseManifest(withModel({ weights: withContainer({ parts: parts(1025) }) })),
      ManifestFormatError,
    );
  });

  it("33 assets", () => {
    let assets: Record<string, unknown> = {};
    for (let index = 0; index < 33; index += 1) {
      assets = { ...assets, [`a${index}`]: { ...FILE, path: `assets/a${index}.json` } };
    }
    assertThrows(() => parseManifest(withModel({ assets })), ManifestFormatError);
  });

  it("33 quants", () => {
    let quants: Record<string, unknown> = {};
    for (let index = 0; index < 33; index += 1) {
      quants = { ...quants, [`q${index}`]: { weights: { net: "f16" }, session: {} } };
    }
    assertThrows(
      () => parseManifest(withModel({ quants, defaultQuant: "q0" })),
      ManifestFormatError,
    );
  });

  it("pipelineConfig over 256KiB (per model)", () => {
    const pipelineConfig = { blob: "x".repeat(256 * 1024) };
    assertThrows(() => parseManifest(withModel({ pipelineConfig })), ManifestFormatError);
  });

  it("manifest body over 1MiB", () => {
    // The shape that hits the body limit before the limit of pipelineConfig alone (the same check
    // reached from another path).
    const text = `${" ".repeat(1024 * 1024)}${withModel()}`;
    assertThrows(() => parseManifest(text), ManifestFormatError);
  });
});

describe("parseManifest: nesting that is too deep fails with a typed error even under 1MiB", () => {
  // Depth grows **independently of the byte count** (a run of opening brackets alone gives
  // thousands
  // of levels in a few KB). Without a depth check, the recursion of the full traversal exhausts the
  // stack, and a plain `RangeError` instead of a `ManifestFormatError` escapes and misses the
  // `instanceof HubError` branch. This pins that type as an observed value.
  const nested = (depth: number): string => `${"[".repeat(depth)}${"]".repeat(depth)}`;
  /**
   * Embeds a deep array without going through the recursion of JSON.stringify (the builder uses no
   * stack).
   */
  const withDeepConfig = (depth: number): string =>
    withModel({ pipelineConfig: { deep: "@" } }).replace('"@"', nested(depth));

  it("a depth that would be a RangeError with plain recursion becomes a ManifestFormatError", () => {
    const text = withDeepConfig(3000);
    assert(new TextEncoder().encode(text).length < 1024 * 1024, "not built under 1MiB");
    const error = assertThrows(() => parseManifest(text), ManifestFormatError);
    assert(error.message.includes("nesting"), `${error.message} does not name the excessive depth`);
  });

  it("practical nesting (a few levels of pipelineConfig) passes", () => {
    // A detector for the regression of setting the limit below practical needs (it bounds the limit
    // value itself from below).
    const manifest = parseManifest(
      withModel({ pipelineConfig: { a: { b: { c: [[[1, 2], [3]]] } } } }),
    );
    assertEquals(manifest.models["m"].defaultQuant, "q");
  });
});

describe("parseManifest: the diagnostic for weights over the size limit is linear and cuts off the available labels", () => {
  // If the diagnostic traversal that runs before the count check (it puts "the available labels"
  // into the error) were quadratic in the count, one manifest within 1MiB could stall the main
  // thread for more than ten minutes (a hole in the DoS barrier).
  /** A weights table with `count` empty weights entries (the builder stays linear too). */
  const emptyWeights = (count: number): Record<string, unknown> =>
    Object.fromEntries(Array.from({ length: count }, (_unused, index) => [`w${index}`, {}]));
  /**
   * A quant that maps all `count` weights to "f16" (to make the shape valid within the weights
   * limit).
   */
  const validWeights = (count: number) => {
    const names = Array.from({ length: count }, (_unused, index) => `w${index}`);
    return {
      weights: Object.fromEntries(names.map((name) => [name, { f16: { container: CONTAINER } }])),
      quants: {
        q: { weights: Object.fromEntries(names.map((name) => [name, "f16"])), session: {} },
      },
    };
  };
  // A copy of the size limit of ADR 0041 decision 7 (weights ≤ 32 — MAX_WEIGHTS / MAX_MODELS /
  // MAX_QUANTS of the implementation). The cut-off of the diagnostic is that limit itself — when
  // the
  // limit changes, change this too (if they drift apart, the "exactly at the limit" step below
  // turns
  // red).
  const LABEL_LIMIT = 32;

  it("20000 empty weights fail with ManifestFormatError within the time limit", () => {
    const text = withModel({ weights: emptyWeights(20000) });
    assert(new TextEncoder().encode(text).length < 1024 * 1024, "not built under 1MiB");
    const started = performance.now();
    assertThrows(() => parseManifest(text), ManifestFormatError);
    const elapsed = performance.now() - started;
    // A quadratic traversal takes over 30 seconds at 20000 entries. The limit is set more than an
    // order of magnitude above the measured linear time (so that CI variance does not flake it).
    assert(elapsed < 2000, `took ${Math.round(elapsed)} ms (limit 2000 ms)`);
  });

  it("the dtypes in the over-limit error are exactly the limit count, and the message says they were cut off", () => {
    const error = assertThrows(
      () => parseManifest(withModel({ weights: emptyWeights(100) })),
      ManifestFormatError,
    );
    // "At most" would let through an implementation that cuts off below the limit (even at 0) —
    // this pins that entries up to the limit appear.
    assertEquals(
      Object.keys(error.available.dtypes).length,
      LABEL_LIMIT,
      `${Object.keys(error.available.dtypes).length} entries appeared (cut-off ${LABEL_LIMIT})`,
    );
    assert(error.message.includes("cut off"), `${error.message} does not say it was cut off`);
  });

  it("an error raised before the count check also says it was cut off", () => {
    // The unknown-key check fails before the count check of weights — even on a path where the
    // message does not talk about the excess count, the reader must not mistake the partial list
    // for the whole.
    const error = assertThrows(
      () => parseManifest(withModel({ weights: emptyWeights(100), unknown: true })),
      ManifestReferenceError,
    );
    // "At most" would let through an implementation that cuts off below the limit (even at 0) —
    // entries up to the limit appear on this path too.
    assertEquals(
      Object.keys(error.available.dtypes).length,
      LABEL_LIMIT,
      `${Object.keys(error.available.dtypes).length} entries appeared (cut-off ${LABEL_LIMIT})`,
    );
    assert(error.message.includes("cut off"), `${error.message} does not say it was cut off`);
  });

  it("exactly at the limit (32 weights), every entry appears and no cut-off is claimed", () => {
    // One side alone would be green even for an implementation that always cuts off — the available
    // list of a valid manifest is never truncated.
    const manifest = parseManifest(withModel(validWeights(LABEL_LIMIT)));
    assertEquals(Object.keys(manifest.models["m"].available.dtypes).length, LABEL_LIMIT);
    const error = assertThrows(
      () => parseManifest(withModel({ ...validWeights(LABEL_LIMIT), unknown: true })),
      ManifestReferenceError,
    );
    assertEquals(Object.keys(error.available.dtypes).length, LABEL_LIMIT);
    assert(!error.message.includes("cut off"), `${error.message} wrongly claimed a cut-off`);
  });
});

test("parseManifest: errors carry the available model / quant / dtype labels", () => {
  const container = (mark: string) => ({
    descriptor: DESCRIPTOR,
    parts: [
      { ...PART0, path: `net/${mark}-00001-of-00002.krm` },
      { ...PART1, path: `net/${mark}-00002-of-00002.krm`, sha256: mark.repeat(32) },
    ],
  });
  const error = assertThrows(
    () =>
      parseManifest(
        JSON.stringify({
          format: "karume/5",
          generator: "karume/0.1.0",
          defaultModel: "fast",
          models: {
            fast: {
              pipeline: "anima/1",
              weights: {
                net: { f16: { container: container("b2") }, i8: { container: container("c3") } },
              },
              assets: {},
              quants: { w8: { weights: { net: "q4" }, session: {} } },
              defaultQuant: "w8",
              pipelineConfig: {},
            },
            slim: {
              pipeline: "anima/1",
              weights: { net: { i8: { container: container("d4") } } },
              assets: {},
              quants: { w8: { weights: { net: "i8" }, session: {} } },
              defaultQuant: "w8",
              pipelineConfig: {},
            },
          },
        }),
      ),
    ManifestReferenceError,
  );
  assertEquals(error.available.models, ["fast", "slim"]);
  // Only those of the model in context (fast) appear — suggesting a quant name of another model
  // would mislead.
  assertEquals(error.available.quants, ["w8"]);
  assertEquals(error.available.dtypes, { net: ["f16", "i8"] });
  assert(error instanceof HubError, "all of them can be handled together as HubError");
});

test("parseManifest: a top-level violation carries only the model list (there is no quant context)", () => {
  const error = assertThrows(
    () => parseManifest(withModel({}, { defaultModel: "absent" })),
    ManifestReferenceError,
  );
  assertEquals(error.available.models, ["m"]);
  assertEquals(error.available.quants, []);
  assertEquals(error.available.dtypes, {});
});

test("parseManifest: reads a valid manifest as declared", () => {
  const manifest = parseManifest(validManifestText);
  assertEquals(manifest.format, "karume/5");
  assertEquals(manifest.generator, "karume/0.1.0");
  assertEquals(manifest.defaultModel, "anima-turbo");
  assertEquals(Object.keys(manifest.models), ["anima-turbo", "anima-lite"]);
  assertEquals(manifest.available.models, ["anima-turbo", "anima-lite"]);

  const turbo = manifest.models["anima-turbo"];
  assertEquals(turbo.pipeline, { name: "anima", major: 1 });
  assertEquals(turbo.defaultQuant, "w8a8-s16");
  assertEquals(turbo.available.quants, ["f16", "w8a8-s16", "f16-c16"]);
  assertEquals(turbo.pipelineConfig, { defaults: { steps: 8, guidanceScale: 1 } });

  // weights uses the uniform form with a required dtype key, and its content is one container (ADR
  // 0109 decision 3).
  assertEquals(Object.keys(turbo.weights["transformer"]), ["f16", "i8"]);
  assertEquals(Object.keys(turbo.weights["vae_decoder"]), ["f16"]);
  const i8 = turbo.weights["transformer"]["i8"].container;
  assertEquals(i8.parts.length, 3, "the 3 of part 0 + part 1 + part 2");
  assertEquals(i8.parts[0].size, 24 + i8.descriptor.graph.length + i8.descriptor.model.length);
  // A zero-length part (a container with an empty const region) stays in the declaration, and its
  // sha256 is the value of the empty string.
  assertEquals(i8.parts[1].size, 0);
  assertEquals(
    i8.parts[1].sha256,
    "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
  );

  // assets are unconditional files that do not depend on the quant choice (no dtype level).
  assertEquals(Object.keys(turbo.assets), ["tokenizer", "style_vectors", "style_alias"]);
  assertEquals(turbo.assets["tokenizer"].path, "tokenizer/qwen2-tokenizer.json");
  // Two names point at the same file (table entries are not dropped — deduping happens only when
  // the selection list is built).
  assert(turbo.assets["style_alias"] === turbo.assets["style_vectors"]);

  assertEquals(turbo.quants["w8a8-s16"].session, {
    linearCompute: "a8",
    attentionCompute: "a8",
    attentionScoreStorage: "f16",
  });
  assertEquals(turbo.quants["f16-c16"].gpuFeatures, { shaderF16: true });
  assertEquals(turbo.quants["f16"].gpuFeatures, undefined);

  // The second model has only its own quant / dtype surface.
  const lite = manifest.models["anima-lite"];
  assertEquals(lite.defaultQuant, "w8");
  assertEquals(lite.available.quants, ["w8"]);
  assertEquals(lite.available.dtypes, { text_encoder: ["f16"], transformer: ["i8"] });
});

test("parseManifest: sharing the same path across models works (ADR 0041 §5)", () => {
  const manifest = parseManifest(validManifestText);
  const shared = manifest.models["anima-turbo"].weights["text_encoder"]["f16"].container.parts[0];
  const same = manifest.models["anima-lite"].weights["text_encoder"]["f16"].container.parts[0];
  assertEquals(same.path, shared.path);
  assertEquals(same.sha256, shared.sha256);
  // There is one table, so the same path folds into the same FileRef instance (and is fetched
  // once).
  assert(same === shared, "references to the same path are not folded");
});

describe("parseManifest: one weights entry with many dtype labels", () => {
  // The label count of one weights entry has no count limit of its own — only the 1 MiB limit on
  // the whole manifest bounds it — so building the entry must stay linear in the label count.
  /** `count` dtype labels on the weights entry `net`, each with a legal container. */
  const manyLabels = (count: number, bad: ReadonlySet<number> = new Set()): string => {
    const labels = Array.from({ length: count }, (_unused, index) => `d${index}`);
    return withModel({
      weights: {
        net: Object.fromEntries(
          labels.map((label, index) => [label, bad.has(index) ? {} : { container: CONTAINER }]),
        ),
      },
      quants: { q: { weights: { net: "d0" }, session: {} } },
    });
  };
  // As many labels as fit under 1 MiB with the minimal container (about 0.95 MiB).
  const COUNT = 2000;

  it("keeps every label, in declaration order, each with its own container", () => {
    const entry = parseManifest(manyLabels(COUNT)).models["m"].weights["net"];
    const labels = Object.keys(entry);
    assertEquals(labels.length, COUNT);
    assertEquals(labels[0], "d0");
    assertEquals(labels[COUNT - 1], `d${COUNT - 1}`);
    assert(
      labels.every((label, index) => label === `d${index}`),
      "label order is not the declaration order",
    );
    assertEquals(entry[`d${COUNT - 1}`].container.parts.length, CONTAINER.parts.length);
  });

  it("reports the first bad label in declaration order", () => {
    assertThrows(
      () => parseManifest(manyLabels(COUNT, new Set([700, 1500]))),
      ManifestFormatError,
      "models.m.weights.net.d700.container: missing / not an object",
    );
  });

  it(`parses ${COUNT} labels within a time bound`, () => {
    const text = manyLabels(COUNT);
    assert(new TextEncoder().encode(text).length < 1024 * 1024, "not built under 1 MiB");
    const elapsed = Math.min(
      ...Array.from({ length: 3 }, () => {
        const started = performance.now();
        parseManifest(text);
        return performance.now() - started;
      }),
    );
    // Measured on the development machine: linear about 13 ms, a spread copy per label about
    // 425 ms. The bound is an order of magnitude above the linear measurement (so CI variance does
    // not flip it) and still well below the quadratic one.
    assert(elapsed < 150, `${Math.round(elapsed)} ms (bound 150 ms)`);
  });
});

describe("parseManifest: the container field of weights (ADR 0109 decision 3)", () => {
  it("descriptor has the length and sha256 of each of the 2 documents", () => {
    const container = containerOf(withModel());
    assertEquals(container.descriptor.graph, DESCRIPTOR.graph);
    assertEquals(container.descriptor.model, DESCRIPTOR.model);
  });

  it("parts keep their declaration order (the index is the part id)", () => {
    const parts = [PART0, part(1, 3, 8, "c3"), part(2, 3, 16, "d4")];
    const container = containerOf(withModel({ weights: withContainer({ parts }) }));
    assertEquals(
      container.parts.map((entry) => entry.path),
      parts.map((entry) => entry.path),
    );
  });

  it("exactly 2 pass and 1 fails (part 0 + part 1 is the minimum)", () => {
    assertEquals(containerOf(withModel()).parts.length, 2);
    assertThrows(
      () => parseManifest(withModel({ weights: withContainer({ parts: [PART0] }) })),
      ManifestFormatError,
      "at least 2",
    );
  });

  it("a mismatch of the set of 3 for the same path is rejected inside parts too", () => {
    const error = assertThrows(
      () =>
        parseManifest(
          withModel({
            weights: withContainer({
              parts: [PART0, PART1, { ...PART1, sha256: "d4".repeat(32) }],
            }),
          }),
        ),
      ManifestReferenceError,
    );
    assert(
      error.message.includes(PART1.path),
      `${error.message} does not name the mismatching path`,
    );
  });

  it("the retired shards / extras keys fail as unknown keys", () => {
    for (const retired of ["shards", "extras"]) {
      const error = assertThrows(
        () =>
          parseManifest(
            withModel({
              weights: { net: { f16: { container: CONTAINER, [retired]: {} } } },
            }),
          ),
        ManifestReferenceError,
        undefined,
        `'${retired}' of the dtype entry passed`,
      );
      assert(
        error.message.includes(retired) && error.message.includes("container"),
        `${error.message} does not show the unknown key and the allowed keys`,
      );
    }
  });

  it("unknown keys of container / descriptor / document fail too", () => {
    const patches: Record<string, unknown>[] = [
      { weights: { net: { f16: { container: { ...CONTAINER, graph: FILE } } } } },
      {
        weights: withContainer({
          descriptor: { ...DESCRIPTOR, codecs: [] },
        }),
      },
      {
        weights: withContainer({
          descriptor: { ...DESCRIPTOR, graph: { ...DESCRIPTOR.graph, offset: 0 } },
        }),
      },
    ];
    for (const patch of patches) {
      assertThrows(
        () => parseManifest(withModel(patch)),
        ManifestReferenceError,
        undefined,
        `${JSON.stringify(patch)} passed`,
      );
    }
  });

  it("descriptor's model cannot be omitted (a krm always has a model descriptor)", () => {
    assertThrows(
      () =>
        parseManifest(
          withModel({
            weights: withContainer({ descriptor: { graph: DESCRIPTOR.graph } }),
          }),
        ),
      ManifestFormatError,
      "descriptor.model: missing",
    );
  });
});

describe('parseManifest: the size of part 0 is exactly "header + the 2 documents" (container-v1 §8)', () => {
  // This is consistency between declarations, so it can be checked without fetching a single byte.
  // If this were loose, it would go back to "noticing the length mismatch after the fetch".
  it("exactly passes", () => {
    assertEquals(containerOf(withModel()).parts[0].size, PART0_SIZE);
  });

  it("an off-by-one fails on both sides", () => {
    for (const delta of [-1, 1]) {
      const error = assertThrows(
        () =>
          parseManifest(
            withModel({
              weights: withContainer({ parts: [{ ...PART0, size: PART0_SIZE + delta }, PART1] }),
            }),
          ),
        ManifestFormatError,
        undefined,
        `part 0 size ${PART0_SIZE + delta} passed`,
      );
      assert(
        error.message.includes(String(PART0_SIZE)),
        `${error.message} does not show the expected part 0 length`,
      );
    }
  });

  it("moving the descriptor's length moves part 0's requirement by the same amount", () => {
    const graph = { ...DESCRIPTOR.graph, length: GRAPH_LENGTH + 100 };
    const container = containerOf(
      withModel({
        weights: withContainer({
          descriptor: { ...DESCRIPTOR, graph },
          parts: [{ ...PART0, size: PART0_SIZE + 100 }, PART1],
        }),
      }),
    );
    assertEquals(container.parts[0].size, PART0_SIZE + 100);
  });
});

describe("parseManifest: one descriptor document is at most 32MiB (container-v1 §10)", () => {
  // The value equals `MAX_DESCRIPTOR_BYTES` of the Python source of truth
  // `tools/exporter/src/karume/container.py`.
  const LIMIT = 32 * 2 ** 20;
  const sized = (length: number) => ({
    descriptor: { ...DESCRIPTOR, graph: { ...DESCRIPTOR.graph, length } },
    parts: [{ ...PART0, size: HEADER_BYTES + length + MODEL_LENGTH }, PART1],
  });

  it("exactly 32MiB passes (the same closed interval as the writer)", () => {
    const container = containerOf(withModel({ weights: withContainer(sized(LIMIT)) }));
    assertEquals(container.descriptor.graph.length, LIMIT);
  });

  it("1 byte over fails", () => {
    const error = assertThrows(
      () => parseManifest(withModel({ weights: withContainer(sized(LIMIT + 1)) })),
      ManifestFormatError,
    );
    assert(error.message.includes(String(LIMIT)), `${error.message} does not name the limit`);
  });

  it("a zero-length document is rejected (a krm always has 2 documents)", () => {
    assertThrows(
      () => parseManifest(withModel({ weights: withContainer(sized(0)) })),
      ManifestFormatError,
    );
  });

  it("sha256 is 64 lowercase hex digits", () => {
    for (const sha256 of ["F2".repeat(32), "f2".repeat(31), 12]) {
      assertThrows(
        () =>
          parseManifest(
            withModel({
              weights: withContainer({
                descriptor: { ...DESCRIPTOR, model: { length: MODEL_LENGTH, sha256 } },
              }),
            }),
          ),
        ManifestFormatError,
        undefined,
        `descriptor sha256 '${sha256}' passed`,
      );
    }
  });
});

describe("parseManifest: the part byte limit is 1024MiB (container-v1 §10 — measured by file length)", () => {
  // Readers are built on the premise of "estimating buffer sizes from the declaration" (ADR 0089),
  // so if parsing silently let an over-limit part through, it would only break in the browser. The
  // limit's value equals `PART_MAX_BYTES` of the Python source of truth
  // `tools/exporter/src/karume/container.py`.
  const LIMIT = 1024 * 2 ** 20;
  const big = (size: number, mark: string) => ({
    path: "net/model-00002-of-00002.krm",
    size,
    sha256: mark.repeat(32),
  });

  it("over the limit fails, and the error names the part's path, size and limit", () => {
    const error = assertThrows(
      () =>
        parseManifest(
          withModel({
            weights: withContainer({ parts: [PART0, big(LIMIT + 1, "c3")] }),
          }),
        ),
      ManifestFormatError,
    );
    for (const expected of ["net/model-00002-of-00002.krm", String(LIMIT + 1), String(LIMIT)]) {
      assert(error.message.includes(expected), `${error.message} lacks '${expected}'`);
    }
  });

  it("exactly 1024MiB passes (the same closed interval as the writer)", () => {
    // One side alone would be green even for an implementation that always fails.
    const container = containerOf(
      withModel({ weights: withContainer({ parts: [PART0, big(LIMIT, "c3")] }) }),
    );
    assertEquals(container.parts[1].size, LIMIT);
  });

  it("non-part FileRefs are out of scope — assets pass even over the limit", () => {
    // MUST: the limit is **the contract of the part split**, not a ceiling for every FileRef (that
    // is the 16GiB `MAX_FILE_BYTES`). Mixing them up would make single assets that are not split
    // into parts and exceed the limit unreadable.
    const manifest = parseManifest(
      withModel({
        assets: { style_vectors: { ...FILE, path: "style/vectors.safetensors", size: LIMIT + 1 } },
      }),
    );
    assertEquals(manifest.models["m"].assets["style_vectors"].size, LIMIT + 1);
  });
});

describe("parseManifest: zero-length parts only at index 1 or higher (ADR 0109 decision 3)", () => {
  const EMPTY_SHA = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";
  const emptyPart = { path: "net/model-00002-of-00003.krm", size: 0, sha256: EMPTY_SHA };

  it("a zero-length part at index 1 passes and stays in the declaration as is", () => {
    const container = containerOf(
      withModel({
        weights: withContainer({ parts: [PART0, emptyPart, part(2, 3, 16, "d4")] }),
      }),
    );
    assertEquals(container.parts[1].size, 0);
    assertEquals(container.parts[1].sha256, EMPTY_SHA);
  });

  it("a zero-length part 0 fails (it always holds the header + the 2 documents)", () => {
    assertThrows(
      () =>
        parseManifest(
          withModel({
            weights: withContainer({ parts: [{ ...PART0, size: 0, sha256: EMPTY_SHA }, PART1] }),
          }),
        ),
      ManifestFormatError,
    );
  });

  it("a zero-length part whose sha256 is not the value of the empty string fails", () => {
    const error = assertThrows(
      () =>
        parseManifest(
          withModel({
            weights: withContainer({
              parts: [PART0, { ...emptyPart, sha256: "c3".repeat(32) }, part(2, 3, 16, "d4")],
            }),
          }),
        ),
      ManifestFormatError,
    );
    assert(
      error.message.includes(EMPTY_SHA),
      `${error.message} does not show the sha256 of the empty string`,
    );
  });

  it("zero-length assets are still rejected (only parts may be empty)", () => {
    assertThrows(
      () =>
        parseManifest(
          withModel({
            assets: { tokenizer: { ...FILE, size: 0, sha256: EMPTY_SHA } },
          }),
        ),
      ManifestFormatError,
    );
  });
});

test("parseManifest: the pipeline major is checked for shape only, and the ruling is left to models", () => {
  // ADR 0038 §1: only the models implementation, which declares its supported majors, can reject an
  // unknown pipeline major. The hub only checks the spelling and extracts the major as a type.
  const manifest = parseManifest(withModel({ pipeline: "sbv2/7" }));
  assertEquals(manifest.models["m"].pipeline, { name: "sbv2", major: 7 });
});

/** A manifest with only one quant entry replaced (for observing new entries). */
const withQuant = (patch: Record<string, unknown>): string =>
  withModel({ quants: { q: { weights: { net: "f16" }, session: {}, ...patch } } });

describe("parseManifest: the session knob attentionScoreStorage (ADR 0132)", () => {
  it("accepts 'none' (the online kernel) beside 'f32' and 'f16', and rejects any other spelling", () => {
    for (const value of ["f32", "f16", "none"]) {
      const quant = parseManifest(
        withQuant({ session: { attentionCompute: "a8", attentionScoreStorage: value } }),
      ).models["m"].quants["q"];
      assertEquals(quant.session.attentionScoreStorage, value);
    }
    assertThrows(
      () => parseManifest(withQuant({ session: { attentionScoreStorage: "online" } })),
      ManifestReferenceError,
    );
  });
});

describe("parseManifest: the quant display fields label / description (ADR 0075)", () => {
  it("the strings set appear unchanged in the typed surface", () => {
    const manifest = parseManifest(
      withQuant({
        label: "Balanced (int8)",
        description: "Half the download of f16 with no visible difference.",
      }),
    );
    const quant = manifest.models["m"].quants["q"];
    assertEquals(quant.label, "Balanced (int8)");
    assertEquals(quant.description, "Half the download of f16 with no visible difference.");
  });

  it("an entry without them has no such field (the caller shows the id as is)", () => {
    const quant = parseManifest(withModel()).models["m"].quants["q"];
    assertEquals(quant.label, undefined);
    assertEquals(quant.description, undefined);
  });

  it("exactly the limit passes, one character over fails", () => {
    // One side alone would be green even for an implementation that always fails, so both sides of
    // the boundary are observed.
    assertEquals(
      parseManifest(withQuant({ label: "x".repeat(64) })).models["m"].quants["q"].label?.length,
      64,
    );
    assertThrows(() => parseManifest(withQuant({ label: "x".repeat(65) })), ManifestFormatError);
    assertEquals(
      parseManifest(withQuant({ description: "x".repeat(200) })).models["m"].quants["q"].description
        ?.length,
      200,
    );
    assertThrows(
      () => parseManifest(withQuant({ description: "x".repeat(201) })),
      ManifestFormatError,
    );
  });

  it("over the limit fails with the expected and the actual value", () => {
    const error = assertThrows(
      () => parseManifest(withQuant({ label: "x".repeat(70) })),
      ManifestFormatError,
    );
    assert(
      error.message.includes("expected at most 64 characters"),
      `${error.message} does not show the expected value`,
    );
    assert(
      error.message.includes("actual 70 characters"),
      `${error.message} does not show the actual length`,
    );
  });

  it("a non-string fails with the expected and the actual type", () => {
    const error = assertThrows(() => parseManifest(withQuant({ label: 12 })), ManifestFormatError);
    assert(
      error.message.includes("expected string"),
      `${error.message} does not show the expected type`,
    );
    assert(
      error.message.includes("actual number"),
      `${error.message} does not show the actual type`,
    );
  });

  it("the meaning of the content is not interpreted (within the limit, even a description that disagrees with reality passes)", () => {
    // ADR 0075 decision 2: the length is a boundary check; the hub cannot judge whether the content
    // is valid, and does not.
    const quant = parseManifest(
      withQuant({ label: "f32 (lossless)", description: "🌀".repeat(64) }),
    ).models["m"].quants["q"];
    assertEquals(quant.label, "f32 (lossless)");
    // A surrogate pair counts as one code point (the same visible text does not pass or fail
    // depending on its encoding).
    assertEquals(quant.description, "🌀".repeat(64));
  });
});

describe("parseManifest: a quant's requiredLimits (the deferred field of ADR 0038 §7)", () => {
  it("appears in the typed surface as a partial map of limit name → minimum", () => {
    const manifest = parseManifest(
      withQuant({
        requiredLimits: { maxBufferSize: 2147483648, maxStorageBufferBindingSize: 1073741824 },
      }),
    );
    assertEquals(manifest.models["m"].quants["q"].requiredLimits, {
      maxBufferSize: 2147483648,
      maxStorageBufferBindingSize: 1073741824,
    });
  });

  it("an entry without it has no such field, and an empty declaration passes as empty", () => {
    assertEquals(parseManifest(withModel()).models["m"].quants["q"].requiredLimits, undefined);
    assertEquals(
      parseManifest(withQuant({ requiredLimits: {} })).models["m"].quants["q"].requiredLimits,
      {},
    );
  });

  it("accepts the names of the runtime's requiredLimits vocabulary", () => {
    // The observation point that the spelling matches the runtime (`REQUIRED_LIMIT_KEYS`) one to
    // one. The vocabulary states the compute limits explicitly too, so this also checks that the
    // workgroup limits are not rejected.
    const manifest = parseManifest(
      withQuant({
        requiredLimits: {
          maxUniformBufferBindingSize: 65536,
          maxStorageBuffersPerShaderStage: 10,
          maxUniformBuffersPerShaderStage: 12,
          maxComputeWorkgroupStorageSize: 32768,
          maxComputeInvocationsPerWorkgroup: 1024,
          maxComputeWorkgroupSizeX: 1024,
          maxComputeWorkgroupSizeY: 1024,
          maxComputeWorkgroupSizeZ: 64,
          maxComputeWorkgroupsPerDimension: 65535,
        },
      }),
    );
    assertEquals(Object.keys(manifest.models["m"].quants["q"].requiredLimits ?? {}).length, 9);
  });

  it("an unknown limit name fails with the list of allowed names (a misspelling is not silently ignored)", () => {
    const error = assertThrows(
      () => parseManifest(withQuant({ requiredLimits: { maxBufferSizes: 1024 } })),
      ManifestReferenceError,
    );
    assert(
      error.message.includes("maxBufferSize"),
      `${error.message} does not show the allowed names`,
    );
  });

  it("a value that is not a positive integer fails with the expected and the actual value", () => {
    for (const value of [0, -1, 1.5, "1024", null]) {
      const error = assertThrows(
        () => parseManifest(withQuant({ requiredLimits: { maxBufferSize: value } })),
        ManifestFormatError,
        undefined,
        `requiredLimits.maxBufferSize = ${JSON.stringify(value)} passed`,
      );
      assert(
        error.message.includes("expected positive safe integer"),
        `${error.message} does not show the expectation`,
      );
    }
  });
});

const COMMIT = "0123456789abcdef0123456789abcdef01234567";

describe("parseManifest: the cross-repository fields repo / revision of a file reference (ADR 0038 §7)", () => {
  const cross = { repo: "other/stack", revision: COMMIT };
  /**
   * A whole container referenced from another repository (ADR 0109 decision 3 — cross-repository
   * references are per container).
   */
  const foreignContainer = (patch: Record<string, unknown> = {}) => ({
    descriptor: DESCRIPTOR,
    parts: [
      { ...PART0, ...cross, ...patch },
      { ...PART1, ...cross, ...patch },
    ],
  });

  it("every part of the container carries the same coordinates", () => {
    const container = containerOf(
      withModel({ weights: { net: { f16: { container: foreignContainer() } } } }),
    );
    for (const ref of container.parts) {
      assertEquals(ref.repo, "other/stack");
      assertEquals(ref.revision, COMMIT);
    }
  });

  it("references in assets carry the same fields too", () => {
    const manifest = parseManifest(
      withModel({
        assets: { tokenizer: { ...FILE, ...cross } },
      }),
    );
    assertEquals(manifest.models["m"].assets["tokenizer"].repo, "other/stack");
    assertEquals(manifest.models["m"].assets["tokenizer"].revision, COMMIT);
  });

  it("cross-repository references are per container — both a mix and a one-sided declaration fail", () => {
    const mixed: unknown[][] = [
      [PART0, { ...PART1, ...cross }],
      [{ ...PART0, ...cross }, PART1],
      [
        { ...PART0, ...cross },
        { ...PART1, repo: "another/stack", revision: COMMIT },
      ],
    ];
    for (const parts of mixed) {
      const error = assertThrows(
        () => parseManifest(withModel({ weights: withContainer({ parts }) })),
        ManifestFormatError,
        undefined,
        `${JSON.stringify(parts)} passed`,
      );
      assert(
        error.message.includes("per container — ADR 0109 decision 3"),
        `${error.message} does not name the per-container rule`,
      );
    }
  });

  it("a one-sided declaration fails in both directions", () => {
    for (const half of [{ repo: "other/stack" }, { revision: COMMIT }]) {
      const error = assertThrows(
        () =>
          parseManifest(
            withModel({
              weights: withContainer({
                parts: [
                  { ...PART0, ...half },
                  { ...PART1, ...half },
                ],
              }),
            }),
          ),
        ManifestFormatError,
        undefined,
        `a declaration of only ${JSON.stringify(half)} passed`,
      );
      assert(
        error.message.includes("both 'repo' and 'revision' together"),
        `${error.message} does not show the requirement "both at once"`,
      );
    }
  });

  it("revision rejects branches, tags, short forms and uppercase", () => {
    for (const revision of ["main", "v1.0", COMMIT.slice(0, 7), COMMIT.toUpperCase()]) {
      assertThrows(
        () =>
          parseManifest(
            withModel({
              weights: { net: { f16: { container: foreignContainer({ revision }) } } },
            }),
          ),
        ManifestFormatError,
        undefined,
        `revision '${revision}' passed`,
      );
    }
  });

  it("repo is an allow-list of the 2 segments owner/name", () => {
    for (const repo of ["stack", "other/stack/extra", "other/..", "other/.hidden", "other/re po"]) {
      assertThrows(
        () =>
          parseManifest(
            withModel({
              weights: { net: { f16: { container: foreignContainer({ repo }) } } },
            }),
          ),
        ManifestFormatError,
        undefined,
        `repo '${repo}' passed`,
      );
    }
  });

  it("the same path in a different repository is a different file (the set of 3 is not required to match)", () => {
    // Folding by path alone would reject a correct manifest as a "mismatch of a duplicate path",
    // and
    // in the fetch layer one file's bytes would be delivered for the other. Identity is the triple
    // (repo, revision, path).
    const manifest = parseManifest(
      withModel({
        weights: {
          net: { f16: { container: CONTAINER } },
          text: {
            f16: {
              container: {
                descriptor: DESCRIPTOR,
                parts: [
                  { ...PART0, ...cross, size: PART0_SIZE },
                  { ...PART1, ...cross, size: 128, sha256: "e5".repeat(32) },
                ],
              },
            },
          },
        },
        quants: { q: { weights: { net: "f16", text: "f16" }, session: {} } },
      }),
    );
    assertEquals(manifest.models["m"].weights["net"]["f16"].container.parts[1].size, 64);
    assertEquals(manifest.models["m"].weights["text"]["f16"].container.parts[1].size, 128);
  });

  it("the same (repo, revision, path) folds into one", () => {
    const manifest = parseManifest(
      withModel({
        weights: {
          net: { f16: { container: foreignContainer() } },
          text: { f16: { container: foreignContainer() } },
        },
        quants: { q: { weights: { net: "f16", text: "f16" }, session: {} } },
      }),
    );
    const net = manifest.models["m"].weights["net"]["f16"].container;
    const text = manifest.models["m"].weights["text"]["f16"].container;
    assert(net.parts[1] === text.parts[1], "identical references are not folded");
  });

  it("the same (repo, revision, path) with a mismatching set of 3 is rejected", () => {
    assertThrows(
      () =>
        parseManifest(
          withModel({
            weights: {
              net: {
                f16: {
                  container: {
                    descriptor: DESCRIPTOR,
                    parts: [
                      { ...PART0, ...cross },
                      { ...PART1, ...cross },
                      { ...PART1, ...cross, size: 99 },
                    ],
                  },
                },
              },
            },
          }),
        ),
      ManifestReferenceError,
    );
  });
});

describe("parseManifest: keeps a quant's GEMV reduction setting and tells omission from invalid values", () => {
  for (const linearGemvReduce of ["sequential", "parallel"] as const) {
    it(linearGemvReduce, () => {
      const manifest = parseManifest(
        withModel({
          quants: { q: { weights: { net: "f16" }, session: { linearGemvReduce } } },
        }),
      );
      assertEquals(manifest.models.m.quants.q.session, { linearGemvReduce });
    });
  }
  // These assertions sat in the former test body outside any step; each is now its own it.
  it("when not declared, the field is absent", () => {
    assertEquals(parseManifest(withModel()).models.m.quants.q.session, {});
  });
  for (const value of ["auto", "Parallel", 1, true, null]) {
    it(`${JSON.stringify(value)} is rejected`, () => {
      assertThrows(
        () =>
          parseManifest(
            withModel({
              quants: { q: { weights: { net: "f16" }, session: { linearGemvReduce: value } } },
            }),
          ),
        HubError,
      );
    });
  }
});

describe("parseManifest: keeps a quant's attention reduction form and tells omission from invalid values", () => {
  for (const stateAttentionReduce of ["sequential", "parallel", "parallel-fused"] as const) {
    it(stateAttentionReduce, () => {
      const manifest = parseManifest(
        withModel({
          quants: { q: { weights: { net: "f16" }, session: { stateAttentionReduce } } },
        }),
      );
      assertEquals(manifest.models.m.quants.q.session, { stateAttentionReduce });
    });
  }
  // This assertion sat in the former test body outside any step; it is now its own it.
  it("when not declared, the field is absent", () => {
    assertEquals(parseManifest(withModel()).models.m.quants.q.session, {});
  });
  // `parallel-subgroup32` belongs to the GEMV vocabulary and does not exist for attention (a
  // mixed-up spelling is not let through).
  for (const value of ["auto", "Parallel", "parallel-subgroup32", 1, true, null]) {
    it(`${JSON.stringify(value)} is rejected`, () => {
      assertThrows(
        () =>
          parseManifest(
            withModel({
              quants: { q: { weights: { net: "f16" }, session: { stateAttentionReduce: value } } },
            }),
          ),
        HubError,
      );
    });
  }
  // Pins one case with the same shape as the shipped regular E2B i4-fast (declared together with
  // GEMV and RMS fusion).
  it("can be declared together with linearGemvReduce and fuseRmsNormAdd", () => {
    const session = {
      linearGemvReduce: "parallel",
      stateAttentionReduce: "parallel-fused",
      fuseRmsNormAdd: true,
    } as const;
    const manifest = parseManifest(
      withModel({
        quants: { q: { weights: { net: "f16" }, session } },
      }),
    );
    assertEquals(manifest.models.m.quants.q.session, session);
  });
});

describe("parseManifest: keeps the fusion booleans and tells them from omission and invalid values", () => {
  for (const key of [
    "fuseRmsNormAdd",
    "fuseLinearStaticQuantize",
    "fuseGeluMulStaticQuantize",
    "packedStaticQuantize",
    "integerGemv",
    "linearGemvPlanes",
  ]) {
    for (const value of [false, true]) {
      it(`${key}=${value}`, () => {
        const session = { [key]: value };
        const manifest = parseManifest(
          withModel({
            quants: { q: { weights: { net: "f16" }, session } },
          }),
        );
        assertEquals(manifest.models.m.quants.q.session, session);
      });
    }
    for (const value of [null, 0, 1, "true", "false", [], {}]) {
      // The value is spelled into the step name because assertThrows' default message does not show
      // the key or the value.
      it(`${key}=${JSON.stringify(value)} is rejected`, () => {
        assertThrows(
          () =>
            parseManifest(
              withModel({
                quants: { q: { weights: { net: "f16" }, session: { [key]: value } } },
              }),
            ),
          HubError,
        );
      });
    }
  }
  // Pins one case with the same shape as the shipped i4-fast (the GEMV setting declared together
  // with the 3 boolean fields).
  it("can be declared together with linearGemvReduce", () => {
    const session = {
      linearGemvReduce: "parallel",
      fuseRmsNormAdd: true,
      fuseLinearStaticQuantize: true,
      fuseGeluMulStaticQuantize: true,
      packedStaticQuantize: true,
    } as const;
    const manifest = parseManifest(
      withModel({
        quants: { q: { weights: { net: "f16" }, session } },
      }),
    );
    assertEquals(manifest.models.m.quants.q.session, session);
  });
  it("keeps integerGemv declared next to the packed activations it requires", () => {
    const session = {
      linearGemvReduce: "parallel",
      fuseLinearStaticQuantize: true,
      packedStaticQuantize: true,
      integerGemv: true,
    } as const;
    const manifest = parseManifest(
      withModel({
        quants: { q: { weights: { net: "f16" }, session } },
      }),
    );
    assertEquals(manifest.models.m.quants.q.session, session);
  });
  it("when not declared, the field is absent", () => {
    assertEquals(parseManifest(withModel()).models.m.quants.q.session, {});
  });
});
