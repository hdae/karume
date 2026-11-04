// **Construction guards** of `Pipeline`. Only what needs neither a GPU nor real assets (the
// real-GPU comparison belongs to the example's dump path and the golden E2E).
//
// Three things are pinned here:
//  ① `fromAssets` rejects manifest contract violations **before** acquiring a GPU (unknown model /
//     pipeline name / unknown major / unknown quant / the pipelineConfig schema). If the rejection
//     moved, an environment without a GPU would see a different exception and the reader would not
//     learn what was wrong.
//  ② `styles` / `speakers` map "name → **table row index**", and the values are a permutation of
//     `0..count-1`. A shifted row index keeps the shapes of `style_vec` / `g` valid, so it fails
//     silently: **a different style's or a different speaker's voice comes out** (the doc of
//     `src/sbv2/config.ts`). A default pointing outside the accepted set is the same
//     class; both are rejected at parse time.
//  ③ The operating limits (`maxTokens` / `maxFrames`) are declared by the distribution — without
//     them it cannot be read. The host-side `(T, T)` table is on the order of 8·T² bytes, so a
//     distribution without limits grows without bound.

import { assert, assertEquals, assertFalse, assertRejects, assertThrows } from "@std/assert";
import { describe, it, test } from "vite-plus/test";
import { parseManifest } from "@karume/hub";
import {
  assertBertWidth,
  assertFiniteNoise,
  assertFrontOutputs,
  assertPhonemeLimit,
  assertSeed,
  assertTableFits,
  assertTiledBert,
  assertTokenLimit,
  assetJson,
  Pipeline,
  synthesizeSbv2,
} from "../src/sbv2/pipeline.ts";
import { parseTokenizerAsset } from "../src/sbv2/text/asset.ts";
import { parseJpExtraRules } from "../src/sbv2/text/symbols.ts";
import { DebertaTokenizer } from "../src/sbv2/text/tokenizer.ts";
import type { ModelComponent } from "../src/hub/components.ts";
import { stubModel } from "./helpers/stub-model.ts";
import { ModelInputError } from "../src/errors.ts";
import { InputError } from "../src/sbv2/errors.ts";
import { Randn } from "../src/sbv2/host/random.ts";
import { parseSbv2PipelineConfig } from "../src/sbv2/config.ts";
import { tileBertToPhoneLevel, type TiledBert } from "../src/sbv2/text/bert-tile.ts";
import { declaredContainer, partAssets, tensorlessContainer } from "./helpers/container-fixture.ts";
import { fakeDevice, fakeGpuContext } from "../../runtime/tests/helpers/fake-gpu.ts";

/**
 * The weight components of this series (the same three as `COMPONENT_KEYS` in
 * `src/sbv2/pipeline.ts`).
 */
const WEIGHT_NAMES = ["front", "voice", "text_encoder"] as const;

/**
 * Three containers that can be opened (minimal declarations — the graph-width comparisons are
 * covered by the section below that calls the checks directly). `fromAssets` opens all three when
 * it builds the supply, so the containers are present even for the cases that hit a manifest check.
 */
const COMPONENTS: Record<string, Uint8Array<ArrayBuffer>> = {};
for (const name of WEIGHT_NAMES) {
  Object.assign(
    COMPONENTS,
    partAssets(
      name,
      await tensorlessContainer(name, {
        inputs: [{ name: "x", shape: [1, 4] }],
        output: { name: "y", shape: [1, 4] },
      }),
    ),
  );
}

/** Skeleton of `models/karume-sbv2-jvnv/karume.json` (only the fields the checks need). */
const manifestText = (patch: Record<string, unknown> = {}): string =>
  JSON.stringify({
    format: "karume/5",
    generator: "karume/0.1.0",
    defaultModel: "FN4",
    models: {
      FN4: {
        pipeline: "sbv2/1",
        weights: Object.fromEntries(
          WEIGHT_NAMES.map((name) => [name, { i8: declaredContainer(`${name}/model.i8`) }]),
        ),
        assets: {},
        quants: {
          i8: {
            weights: Object.fromEntries(WEIGHT_NAMES.map((name) => [name, "i8"])),
            session: {},
          },
        },
        defaultQuant: "i8",
        pipelineConfig: {
          styles: { Neutral: 0, high: 1 },
          speakers: { FN4: 0 },
          maxTokens: 512,
          maxFrames: 4096,
          defaults: {
            speaker: "FN4",
            style: "Neutral",
            styleWeight: 1,
            sdpRatio: 0.2,
            noiseScale: 0.6,
            noiseScaleW: 0.8,
            lengthScale: 1,
          },
        },
        ...patch,
      },
    },
  });

const emptyAssets = {} as Record<string, Uint8Array<ArrayBuffer>>;

/** A manifest with only `pipelineConfig` replaced (the rest is the skeleton). */
const withConfig = (config: Record<string, unknown>): string =>
  manifestText({ pipelineConfig: config });

test("fromAssets: fails with both spellings when a component's container is missing (the entry diagnostic)", async () => {
  // A Record without the containers fails on **the missing component** (not with a manifest
  // message) = the contrapositive showing that the checks below are not tripped by missing assets.
  const manifest = parseManifest(manifestText());
  await assertRejects(
    () => Pipeline.fromAssets({ manifest, assets: emptyAssets }),
    Error,
    "component 'front' has no container",
  );
});

test("fromAssets: rejects a manifest whose pipeline contract name is not sbv2", async () => {
  const manifest = parseManifest(manifestText({ pipeline: "anima/1" }));
  await assertRejects(
    () => Pipeline.fromAssets({ manifest, assets: COMPONENTS }),
    Error,
    "'anima/1'",
  );
});

test("fromAssets: an unknown major fails loudly (models owns the check — ADR 0038 §1)", async () => {
  const manifest = parseManifest(manifestText({ pipeline: "sbv2/2" }));
  // hub does not check the major of `pipeline` (only the pipeline implementation knows what it can
  // read).
  assertEquals(manifest.models["FN4"].pipeline, { name: "sbv2", major: 2 });
  await assertRejects(
    () => Pipeline.fromAssets({ manifest, assets: COMPONENTS }),
    Error,
    "has an unsupported major",
  );
});

test("fromAssets: an unknown quant fails with the list of available ones", async () => {
  const manifest = parseManifest(manifestText());
  await assertRejects(
    () => Pipeline.fromAssets({ manifest, assets: COMPONENTS }, { quant: "f16" }),
    Error,
    "available: i8",
  );
});

test("fromAssets: an unknown model fails with the list of available ones", async () => {
  // The axis added in v2 (the first-hand information when another speaker of a family repository is
  // mistyped — ADR 0041 §8).
  const manifest = parseManifest(manifestText());
  await assertRejects(
    () => Pipeline.fromAssets({ manifest, assets: COMPONENTS }, { model: "FN1" }),
    Error,
    "available: FN4",
  );
});

test("fromAssets: a quant that declares f16 compute but omits gpuFeatures is still rejected by admission, naming the shared GPU's missing shader-f16", async () => {
  // The required features are "declared ∪ the features the effective settings need"
  // (`sessionGpuFeatures`). Looking at `gpuFeatures` alone, this distribution would pass admission
  // and fail only at Session construction, after the weights were uploaded.
  const f16Compute = parseManifest(
    manifestText({
      quants: {
        i8: {
          weights: Object.fromEntries(WEIGHT_NAMES.map((name) => [name, "i8"])),
          session: { linearCompute: "f16" },
        },
      },
    }),
  );
  const shared = { gpu: fakeGpuContext(fakeDevice()) };
  await assertRejects(
    () => Pipeline.fromAssets({ manifest: f16Compute, assets: COMPONENTS }, shared),
    Error,
    "sbv2.Pipeline: quant 'i8' requires shader-f16, but",
  );
  // Control: a quant that does not ask for f16 compute passes admission on the same shared GPU and
  // gets as far as the assets (the counterpart showing the check is not always true).
  const error = await assertRejects(
    () =>
      Pipeline.fromAssets({ manifest: parseManifest(manifestText()), assets: COMPONENTS }, shared),
    Error,
  );
  assertFalse(error.message.includes("shader-f16"), error.message);
});

test("fromAssets: an unknown pipelineConfig key fails at construction", async () => {
  const manifest = parseManifest(
    withConfig({
      styles: { Neutral: 0 },
      speakers: { FN4: 0 },
      maxTokens: 512,
      maxFrames: 4096,
      // A section the distribution does not have. No path where a misspelling silently falls back
      // to the default.
      sampleRate: 44100,
      defaults: {
        speaker: "FN4",
        style: "Neutral",
        styleWeight: 1,
        sdpRatio: 0.2,
        noiseScale: 0.6,
        noiseScaleW: 0.8,
        lengthScale: 1,
      },
    }),
  );
  await assertRejects(
    () => Pipeline.fromAssets({ manifest, assets: COMPONENTS }),
    Error,
    "pipelineConfig: unknown key 'sampleRate'",
  );
});

test("fromAssets: fails at construction when defaults.style is not in styles", async () => {
  const manifest = parseManifest(
    withConfig({
      styles: { Neutral: 0, high: 1 },
      speakers: { FN4: 0 },
      maxTokens: 512,
      maxFrames: 4096,
      defaults: {
        speaker: "FN4",
        style: "Angry",
        styleWeight: 1,
        sdpRatio: 0.2,
        noiseScale: 0.6,
        noiseScaleW: 0.8,
        lengthScale: 1,
      },
    }),
  );
  // Do not create a fault that shows up only after one generation — the default must be inside the
  // accepted set.
  await assertRejects(
    () => Pipeline.fromAssets({ manifest, assets: COMPONENTS }),
    Error,
    "pipelineConfig.defaults.style: 'Angry' is not in styles",
  );
});

test("fromAssets: fails at construction when defaults.speaker is not in speakers", async () => {
  const manifest = parseManifest(
    withConfig({
      styles: { Neutral: 0 },
      speakers: { FN4: 0 },
      maxTokens: 512,
      maxFrames: 4096,
      defaults: {
        speaker: "jvnv-F1-jp",
        style: "Neutral",
        styleWeight: 1,
        sdpRatio: 0.2,
        noiseScale: 0.6,
        noiseScaleW: 0.8,
        lengthScale: 1,
      },
    }),
  );
  await assertRejects(
    () => Pipeline.fromAssets({ manifest, assets: COMPONENTS }),
    Error,
    "pipelineConfig.defaults.speaker: 'jvnv-F1-jp' is not in speakers",
  );
});

test("fromAssets: styles whose row indices are not a permutation fail at construction (silent wrong-value class)", async () => {
  // Two entries with row indices {0, 5}. The table has two rows and no row 5, so letting it through
  // would either read a row out of range or read a different row.
  const manifest = parseManifest(
    withConfig({
      styles: { Neutral: 0, high: 5 },
      speakers: { FN4: 0 },
      maxTokens: 512,
      maxFrames: 4096,
      defaults: {
        speaker: "FN4",
        style: "Neutral",
        styleWeight: 1,
        sdpRatio: 0.2,
        noiseScale: 0.6,
        noiseScaleW: 0.8,
        lengthScale: 1,
      },
    }),
  );
  await assertRejects(
    () => Pipeline.fromAssets({ manifest, assets: COMPONENTS }),
    Error,
    "the row index is not an integer in 0..1",
  );
});

test("parseSbv2PipelineConfig: rejects duplicate row indices (names and rows not one to one)", () => {
  // Even within range, a duplicate leaves one table row pointed at by nobody = a different style's
  // voice.
  assertThrows(
    () =>
      parseSbv2PipelineConfig({
        styles: { Neutral: 0, high: 0 },
        speakers: { FN4: 0 },
        maxTokens: 512,
        maxFrames: 4096,
        defaults: {
          speaker: "FN4",
          style: "Neutral",
          styleWeight: 1,
          sdpRatio: 0.2,
          noiseScale: 0.6,
          noiseScaleW: 0.8,
          lengthScale: 1,
        },
      }),
    Error,
    "row index 0 is duplicated",
  );
});

test("parseSbv2PipelineConfig: reads the distribution's pipelineConfig completely", () => {
  // The same shape as the real `models/karume-sbv2-jvnv/karume.json`. Name → row index can be
  // looked up in a Map.
  const config = parseSbv2PipelineConfig({
    styles: { Neutral: 0, high: 1, low: 2, NSFW: 3 },
    speakers: { FN4: 0 },
    maxTokens: 512,
    maxFrames: 4096,
    defaults: {
      speaker: "FN4",
      style: "Neutral",
      styleWeight: 1,
      sdpRatio: 0.2,
      noiseScale: 0.6,
      noiseScaleW: 0.8,
      lengthScale: 1,
    },
  });
  assertEquals(
    [...config.styles],
    [
      ["Neutral", 0],
      ["high", 1],
      ["low", 2],
      ["NSFW", 3],
    ],
  );
  assertEquals([...config.speakers], [["FN4", 0]]);
  assertEquals(config.defaults.style, "Neutral");
  assertEquals(config.defaults.lengthScale, 1);
  assertEquals(config.maxTokens, 512);
  assertEquals(config.maxFrames, 4096);
});

test("parseSbv2PipelineConfig: rejects lengthScale 0 (the total frame count would be 0)", () => {
  assertThrows(
    () =>
      parseSbv2PipelineConfig({
        styles: { Neutral: 0 },
        speakers: { FN4: 0 },
        maxTokens: 512,
        maxFrames: 4096,
        defaults: {
          speaker: "FN4",
          style: "Neutral",
          styleWeight: 1,
          sdpRatio: 0.2,
          noiseScale: 0.6,
          noiseScaleW: 0.8,
          lengthScale: 0,
        },
      }),
    Error,
    "pipelineConfig.defaults.lengthScale: not a positive finite number",
  );
});

// ---- Operating limits (`maxTokens` / `maxFrames`) ------------------------
//
// ADR 0045 moved the relative-position tables out to the host, so allocating `(T, T)` (on the order
// of 8·T² bytes) became the host's responsibility. The source of truth for the limits is the
// distribution (the exporter's `dist.py`); the TS side holds no constants — so **a distribution
// without them cannot be read** (falling back to a default would let a distribution without limits
// run silently unbounded).

const LIMIT_KNOBS = {
  speaker: "FN4",
  style: "Neutral",
  styleWeight: 1,
  sdpRatio: 0.2,
  noiseScale: 0.6,
  noiseScaleW: 0.8,
  lengthScale: 1,
};

/** A `pipelineConfig` with only the two limit fields replaced (or dropped). */
const configWithLimits = (limits: Record<string, unknown>): Record<string, unknown> => ({
  styles: { Neutral: 0 },
  speakers: { FN4: 0 },
  ...limits,
  defaults: LIMIT_KNOBS,
});

test("parseSbv2PipelineConfig: a distribution without maxTokens / maxFrames cannot be read", () => {
  assertThrows(
    () => parseSbv2PipelineConfig(configWithLimits({ maxFrames: 4096 })),
    Error,
    "pipelineConfig.maxTokens: missing",
  );
  assertThrows(
    () => parseSbv2PipelineConfig(configWithLimits({ maxTokens: 512 })),
    Error,
    "pipelineConfig.maxFrames: missing",
  );
});

test("parseSbv2PipelineConfig: rejects limits that are not positive safe integers", () => {
  // 0 / negative / fractional / beyond the safe integers. None is a number usable for comparing
  // allocation sizes.
  for (const bad of [0, -1, 1.5, Number.MAX_SAFE_INTEGER + 2, Infinity, "512"]) {
    assertThrows(
      () => parseSbv2PipelineConfig(configWithLimits({ maxTokens: bad, maxFrames: 4096 })),
      Error,
      "pipelineConfig.maxTokens: not a positive safe integer",
      `maxTokens=${String(bad)} was accepted`,
    );
  }
});

test("assertTokenLimit: rejects counts above the limit and passes exactly the limit", () => {
  // `generate` calls this check **before buildRelPosTables (4·T² bytes each) and the Session
  // allocation**.
  assertTokenLimit(512, 512);
  assertThrows(
    () => assertTokenLimit(513, 512),
    Error,
    "sbv2.Pipeline: token count 513 exceeds the distribution's maximum maxTokens=512",
  );
});

test("assertPhonemeLimit: rejects counts above the limit and passes exactly the limit", () => {
  // The phoneme dimension P is a symbolic dimension of front, and its limit is the same single one
  // as T (`maxTokens`). Whether P or T is larger depends on the utterance (some symbol words get
  // zero phonemes), so a check on only one of them lets an excess through.
  assertPhonemeLimit(512, 512);
  assertThrows(
    () => assertPhonemeLimit(513, 512),
    Error,
    "sbv2.Pipeline: phoneme count 513 exceeds the distribution's maximum maxTokens=512",
  );
  // The message differs from the T side (so the diagnostic says which check fired).
  assertThrows(() => assertTokenLimit(513, 512), Error, "token count 513");
});

test("assertSeed: the accepted set is the non-negative safe integers (the same condition as Randn; the type is InputError)", () => {
  // `generate` calls this check at the same stage as the other knobs (before setting up a Session).
  // If the only check were `Randn`'s constructor, it would fail after paying for one text_encoder
  // (DeBERTa) run.
  assertSeed(0);
  assertSeed(42);
  assertSeed(Number.MAX_SAFE_INTEGER);
  for (const bad of [-1, 1.5, 2 ** 53, Number.NaN, Number.POSITIVE_INFINITY]) {
    const thrown = assertThrows(
      () => assertSeed(bad),
      InputError,
      `seed ${bad} is not a non-negative safe integer`,
    );
    // Re-wrapping does not drop the owner's exception (the cause is exactly what the owner
    // `request-gates.ts` threw).
    assert(thrown.cause instanceof ModelInputError, "the owner's exception was not kept as cause");
    // The accepted set does not move by one bit from `Randn`'s (loosening it breaks the BigInt
    // conversion; tightening it breaks existing calls). Both go through the single function of the
    // owner `request-gates.ts`, so the only difference is the type: `InputError` at the entry
    // point, the parent `ModelInputError` in the generator.
    assertThrows(() => new Randn(bad), ModelInputError);
  }
});

test("assertFiniteNoise: rejects a z_noise that overflows f32, with the type chosen by where the value came from", () => {
  // noiseScaleW used to pass merely by being finite in double precision, so at 1e39 the random
  // values reached front as ±Inf.
  const overflowing = new Randn(0).normals(4, 1e39);
  assert(!overflowing.every(Number.isFinite), "premise: random values times 1e39 are ±Inf in f32");
  // A value the caller passed is an input error (400).
  assertThrows(
    () => assertFiniteNoise(overflowing, 1e39, true),
    InputError,
    "noiseScaleW 1e+39 makes z_noise non-finite in f32",
  );
  // A default from the distribution is an asset fault (it does not go to the input-error branch).
  const fromAsset = assertThrows(
    () => assertFiniteNoise(overflowing, 1e39, false),
    Error,
    "pipelineConfig.defaults: noiseScaleW 1e+39",
  );
  assertFalse(
    fromAsset instanceof ModelInputError,
    "an asset fault was classified as an input error",
  );
  // The normal range passes.
  assertFiniteNoise(new Randn(0).normals(4, 0.8), 0.8, true);
});

/**
 * A fake state passed directly to {@link synthesizeSbv2}. The `createSession` of each of the three
 * components records the component name and stops with a marker exception (nothing touches a GPU —
 * `gpu` is fake too). Whether the checks before text_encoder passed is read from whether the record
 * is empty.
 *
 * The rules and the tokenizer are a minimal shape-only setup (not real values). The utterance is a
 * single symbol (".").
 */
const noiseWiringState = (calls: string[], defaultNoiseScaleW: number) => {
  const component = (key: string): ModelComponent => ({
    ...stubModel({ inputs: [], outputs: [], values: {} }),
    createSession: () => {
      calls.push(key);
      return Promise.reject(new Error(`marker: createSession of ${key}`));
    },
    assets: {},
    asset: () => {
      throw new Error("unused");
    },
    estimate: () => {
      throw new Error("unused");
    },
    maxPartBytes: 0,
  });
  const table = { data: new Float32Array(1), rows: 1, cols: 1 };
  const state: Parameters<typeof synthesizeSbv2>[0] = {
    gpu: fakeGpuContext(fakeDevice()),
    ownsGpu: false,
    config: parseSbv2PipelineConfig({
      styles: { Neutral: 0 },
      speakers: { FN4: 0 },
      maxTokens: 512,
      maxFrames: 4096,
      defaults: {
        speaker: "FN4",
        style: "Neutral",
        styleWeight: 1,
        sdpRatio: 0.2,
        noiseScale: 0.6,
        noiseScaleW: defaultNoiseScaleW,
        lengthScale: 1,
      },
    }),
    rules: parseJpExtraRules(
      {
        symbols: ["_", "a", "."],
        pad: "_",
        punctuations: ["."],
        toneStart: 6,
        languageId: 1,
        numTones: 12,
        numLanguages: 3,
        addBlank: true,
        blankId: 0,
        samplingRate: 44100,
        hopLength: 512,
        bertHiddenFromEnd: 3,
        bertRelPos: { positionBuckets: 256, maxPosition: 512 },
        defaults: { sdpRatio: 0.2, noiseScale: 0.6, noiseScaleW: 0.8, lengthScale: 1 },
      },
      "test rules",
    ),
    tokenizer: DebertaTokenizer.fromVocabText(
      ["[PAD]", "[CLS]", "[SEP]", "[UNK]", "."].join("\n"),
      { removed: [], spaced: [] },
      { clsId: 1, sepId: 2, unkId: 3 },
    ),
    styles: table,
    speakers: table,
    textEncoder: component("text_encoder"),
    front: component("front"),
    voice: component("voice"),
    sessionOptions: {},
  };
  return state;
};

const PUNCTUATION_ONLY = {
  leadingPunctuations: ["."],
  moras: [],
  words: [{ surface: ".", phones: ["."] }],
};

describe("synthesizeSbv2: wiring of the z_noise finiteness check", () => {
  it("fails as an input error before setting up the text_encoder Session when the caller's noiseScaleW overflows f32", async () => {
    const calls: string[] = [];
    await assertRejects(
      () => synthesizeSbv2(noiseWiringState(calls, 0.8), PUNCTUATION_ONLY, { noiseScaleW: 1e39 }),
      InputError,
      "noiseScaleW 1e+39 makes z_noise non-finite in f32",
    );
    assertEquals(calls, []);
  });

  it("fails as an asset fault before setting up a Session when the distribution's default overflows f32", async () => {
    const calls: string[] = [];
    const error = await assertRejects(
      () => synthesizeSbv2(noiseWiringState(calls, 1e39), PUNCTUATION_ONLY),
      Error,
      "pipelineConfig.defaults: noiseScaleW 1e+39",
    );
    assertFalse(
      error instanceof ModelInputError,
      "an asset fault was classified as an input error",
    );
    assertEquals(calls, []);
  });

  // Counterpart: with the same fake state, a value that does not overflow gets as far as the
  // text_encoder Session (confirming that the two cases above do not fail because the fake itself
  // is broken).
  it("goes on to set up the text_encoder Session for a noiseScaleW that does not overflow", async () => {
    const calls: string[] = [];
    await assertRejects(
      () => synthesizeSbv2(noiseWiringState(calls, 0.8), PUNCTUATION_ONLY, { noiseScaleW: 0.8 }),
      Error,
      "marker: createSession of text_encoder",
    );
    assertEquals(calls, ["text_encoder"]);
  });
});

/**
 * Fault-injection shim: a tile that drops one copy of each token.
 *
 * The allocation stays the correct `sum(word2ph)` and **only the written columns** fall short — the
 * shape an off-by-one in the repeat bound of `tileBertToPhoneLevel` produces, which a check that
 * looks only at `data.length` lets through.
 */
const tileWithShortRepeat = (
  hidden: Float32Array,
  tokenCount: number,
  word2ph: readonly number[],
): TiledBert => {
  const dim = hidden.length / tokenCount;
  const total = word2ph.reduce((sum, count) => sum + count, 0);
  const out = new Float32Array(dim * total);
  let column = 0;
  for (let token = 0; token < tokenCount; token += 1) {
    for (let repeat = 0; repeat < word2ph[token] - 1; repeat += 1) {
      for (let d = 0; d < dim; d += 1) out[d * total + column] = hidden[token * dim + d];
      column += 1;
    }
  }
  return { data: out, dim, columns: column };
};

test("assertTiledBert: fault injection — catches a tile whose walk dropped columns by the difference from the allocation", () => {
  const hidden = new Float32Array([1, 2, 3, 10, 20, 30]);
  const word2ph = [1, 2];
  const good = tileBertToPhoneLevel(hidden, 2, word2ph);
  assertTiledBert(good, 3);

  const broken = tileWithShortRepeat(hidden, 2, word2ph);
  // Pinning the premise: the allocation is still correct (= the old single check "expanded length
  // === phoneme count" passes it green).
  assertEquals(
    broken.data.length,
    good.data.length,
    "a different allocation would be a matter for another check",
  );
  assertThrows(
    () => assertTiledBert(broken, 3),
    Error,
    "The 1 columns × dim 3 written by the BERT expansion do not match the allocated 9 elements",
  );
  // A mismatch with the phoneme count (analysis and tile disagree) fails with a separate message as
  // before.
  assertThrows(
    () => assertTiledBert(good, 4),
    Error,
    "BERT expansion length 3 differs from the phoneme count 4",
  );
});

describe("classification axis of SBV2 throws (do they fit the cross-family branch?)", () => {
  // A host that serves several families splits 400 / 500 by `instanceof ModelInputError` alone.
  // Even though SBV2's type stays `InputError`, the branch holds as long as the parent is the same.

  it("input checks are caught by the cross-family ModelInputError", () => {
    const seed = assertThrows(() => assertSeed(-1));
    assert(seed instanceof ModelInputError, "a rejected seed is 400-like");
    const tokens = assertThrows(() => assertTokenLimit(513, 512));
    assert(tokens instanceof ModelInputError, "exceeding the token limit is 400-like");
    const phonemes = assertThrows(() => assertPhonemeLimit(513, 512));
    assert(phonemes instanceof ModelInputError, "exceeding the phoneme limit is 400-like");
  });

  it("a broken internal invariant is not a ModelInputError", () => {
    // A broken tile walk is something the caller cannot fix by changing text or seed = 500-like. If
    // it had the same type as input errors, a wiring fault would be returned to the caller as "fix
    // your input".
    const broken = tileWithShortRepeat(new Float32Array([1, 2, 3, 10, 20, 30]), 2, [1, 2]);
    const error = assertThrows(() => assertTiledBert(broken, 3), Error, "the tile walk is broken");
    assertFalse(error instanceof ModelInputError);
  });
});

// ---- Asset × graph comparison at construction (checks for the silent wrong-value class) ----
//
// All of these are checks `buildSbv2State` runs **before acquiring a GPU**. Left until run time, a
// distribution would set up text_encoder (DeBERTa), pay for one run and then fail with a generic
// message that names neither the family nor the asset.

/** A name → row index table (the row index is the table's physical row). */
const rowNames = (count: number): Map<string, number> =>
  new Map(Array.from({ length: count }, (_, index) => [`s${index}`, index]));

test("assertTableFits: passes only a table that matches both the name table's count and the graph input's width", () => {
  const names = rowNames(7);
  // Normal case (pin first that this shape passes — so the checks below cannot be "always fails").
  assertTableFits(
    { data: new Float32Array(7 * 256), rows: 7, cols: 256 },
    names,
    "style_vectors",
    256,
  );
  // A shifted row count keeps the shapes valid, so it fails silently: **a different style's voice
  // comes out**.
  assertThrows(
    () =>
      assertTableFits(
        { data: new Float32Array(3 * 256), rows: 3, cols: 256 },
        names,
        "style_vectors",
        256,
      ),
    Error,
    "has 3 rows, which differs from the 7 entries of pipelineConfig",
  );
  assertThrows(
    () =>
      assertTableFits(
        { data: new Float32Array(7 * 128), rows: 7, cols: 128 },
        names,
        "style_vectors",
        256,
      ),
    Error,
    "has 128 columns, which differs from the graph input's 256",
  );
});

/**
 * The real shape of front (4 outputs; `bert` is `[1, width, P]`). Only the output count and bert
 * width can be changed.
 */
const frontStub = (
  options: { readonly outputs?: number; readonly bertWidth?: number } = {},
): ReturnType<typeof stubModel> =>
  stubModel({
    symbols: ["P"],
    inputs: [{ name: "bert", shape: [1, options.bertWidth ?? 1024, "P"] }],
    outputs: Array.from({ length: options.outputs ?? 4 }, (_, index) => `front_out_${index}`),
    values: {},
  });

/**
 * The real shape of text_encoder (output names are torch node names; hidden is `[1, T, width]`).
 */
const textEncoderStub = (
  options: { readonly width?: number | string; readonly outputs?: number } = {},
): ReturnType<typeof stubModel> => {
  const names = Array.from({ length: options.outputs ?? 1 }, (_, index) => `layer_norm_${index}`);
  return stubModel({
    symbols: ["T"],
    inputs: [{ name: "input_ids", shape: [1, "T"] }],
    outputs: names,
    values: Object.fromEntries(
      names.map((name) => [name, [1, "T", options.width ?? 1024] as const]),
    ),
  });
};

test("assertFrontOutputs: front has exactly 4 outputs (the premise of positional lookup)", () => {
  // Normal case (negative control).
  assertFrontOutputs(frontStub());
  // Too few would fail in `outputAt` after the run — moved forward to construction.
  assertThrows(
    () => assertFrontOutputs(frontStub({ outputs: 3 })),
    Error,
    "front has 3 graph outputs",
  );
  // MUST NOT settle for "4 or more": only positions 0..3 are read, so extra outputs would be
  // silently ignored.
  assertThrows(
    () => assertFrontOutputs(frontStub({ outputs: 5 })),
    Error,
    "front has 5 graph outputs",
  );
});

test("assertBertWidth: compares text_encoder's hidden width with front's 'bert' input width at construction", () => {
  // Normal case (negative control).
  assertBertWidth(textEncoderStub(), frontStub(), 1);
  // A width mismatch = a distribution combining two graphs exported with different BERTs. Left
  // until run time, it fails in `Session.run`'s element-count check (a generic message that names
  // no family).
  const error = assertThrows(
    () => assertBertWidth(textEncoderStub({ width: 768 }), frontStub(), 1),
    Error,
    "has width 768, which differs from the width of front's graph input 'bert', 1024",
  );
  assertEquals(error.message.includes("layer_norm_0"), true, error.message);
  // Only the last axis is examined (T in `[1, T, dim]` is symbolic). A symbolic last axis has no
  // readable width.
  assertThrows(
    () => assertBertWidth(textEncoderStub({ width: "D" }), frontStub(), 1),
    Error,
    "the last axis of text_encoder output 'layer_norm_0' is not a static dimension (D)",
  );
  // The output is chosen relative to the end (`bertHiddenFromEnd`) — an asset with too few outputs
  // fails there.
  assertThrows(
    () => assertBertWidth(textEncoderStub(), frontStub(), 3),
    Error,
    "at least 3 are required",
  );
});

// ---- Decoding asset JSON (invalid UTF-8 is not silently replaced) --------
//
// `assetJson` runs inside `openSbv2State` **after** the three IR containers are opened, so it
// cannot be reached through `fromAssets` (`tests/helpers/` has no builder for synthetic IR
// containers). The check is called directly.

/** An asset that is complete as JSON but has one byte of invalid UTF-8 (0xff) in a string value. */
const brokenUtf8Asset = (): Uint8Array<ArrayBuffer> =>
  Uint8Array.from([
    ...new TextEncoder().encode('{"unkId":'),
    0x22,
    0xff,
    0x22,
    ...new TextEncoder().encode("}"),
  ]);

test("assetJson: an asset with invalid UTF-8 fails with the decode-stage message (never reaches the JSON stage)", () => {
  const broken = brokenUtf8Asset();
  // Pinning the premise: the default TextDecoder replaces 0xff with U+FFFD, so reading with
  // replacement silently passes **valid JSON with different content**. That is the path that was
  // closed.
  assertEquals(
    JSON.parse(new TextDecoder().decode(broken)),
    { unkId: "�" },
    "premise broken: decoding with replacement no longer gives valid JSON (the subject of this test is gone)",
  );
  assertThrows(
    () => assetJson({ tokenizer: broken }, "tokenizer"),
    Error,
    "sbv2: asset 'tokenizer' cannot be read as UTF-8",
  );
});

test("assetJson: a JSON syntax error fails with a message separate from decoding (the normal range is unchanged)", () => {
  const truncated = new TextEncoder().encode('{"unkId":');
  assertThrows(
    () => assetJson({ symbols: truncated }, "symbols"),
    Error,
    "sbv2: asset 'symbols' cannot be read as JSON",
  );
  // Valid UTF-8 passes as is, even with multi-byte characters.
  const valid = new TextEncoder().encode('{"unkId":0,"space":"あ"}');
  assertEquals(assetJson({ symbols: valid }, "symbols"), { unkId: 0, space: "あ" });
});

// ---- Special ids of the tokenizer asset (rejected before Int32Array silently truncates them) ----

/** Skeleton of `deberta-tokenizer.json` (`patch` overrides `special`). */
const tokenizerAsset = (special: Record<string, unknown>): Record<string, unknown> => ({
  special: { clsId: 1, sepId: 2, unkId: 3, ...special },
  vocabText: "[PAD]\n[CLS]\n[SEP]\n[UNK]\nあ",
  cleanRanges: { removed: [], spaced: [] },
});

test("parseTokenizerAsset: special ids are integers below the vocabulary row count", () => {
  // Normal case (pin first that this shape passes — so the three below cannot be "always fails"
  // checks).
  parseTokenizerAsset(tokenizerAsset({}), "tokenizer");
  // A non-integer is truncated on assignment to `Int32Array`, and an out-of-range value wraps. Both
  // are silent wrong values that "point at a different token", and they do not surface until the
  // BERT features have quietly become meaningless.
  assertThrows(
    () => parseTokenizerAsset(tokenizerAsset({ clsId: 1.5 }), "tokenizer"),
    Error,
    "tokenizer.special.clsId: not an integer in 0..2147483647 (1.5)",
  );
  assertThrows(
    () => parseTokenizerAsset(tokenizerAsset({ sepId: -1 }), "tokenizer"),
    Error,
    "tokenizer.special.sepId: not an integer in 0..2147483647 (-1)",
  );
  // id 5 for a 5-row vocabulary = one past the end (the gather reads an out-of-range index).
  assertThrows(
    () => parseTokenizerAsset(tokenizerAsset({ unkId: 5 }), "tokenizer"),
    Error,
    "unkId 5 is >= the vocabulary row count 5",
  );
});

// ---- Range tables of the tokenizer asset (rejected before they break the binary search's premise
// and turn into silent wrong values) ----

/** The skeleton with only `cleanRanges` replaced (`special` stays the normal case). */
const tokenizerRanges = (clean: Record<string, unknown>): Record<string, unknown> => ({
  ...tokenizerAsset({}),
  cleanRanges: clean,
});

test("parseTokenizerAsset: accepts range tables only as ascending, non-overlapping closed ranges", () => {
  // Normal case (pin first that this shape passes — so the checks below cannot be "always fails").
  parseTokenizerAsset(
    tokenizerRanges({
      removed: [
        [0, 0],
        [8, 31],
      ],
      spaced: [[32, 32]],
    }),
    "tok",
  );
  // `inRanges` is a binary search, so a broken premise throws nothing and **silently misses**.
  // input_ids and baseWord2ph are both built, broken the same way, from a bertText whose removal /
  // spacing rules alone changed, so the length check of `text/model-input.ts` passes too, and audio
  // synthesised from different BERT embeddings comes out as is.
  assertThrows(
    () => parseTokenizerAsset(tokenizerRanges({ removed: [[31, 8]], spaced: [] }), "tok"),
    Error,
    "tok.cleanRanges.removed[0]: range [31, 8] has start greater than end",
  );
  assertThrows(
    () =>
      parseTokenizerAsset(
        tokenizerRanges({
          removed: [
            [127, 159],
            [8, 31],
          ],
          spaced: [],
        }),
        "tok",
      ),
    Error,
    "tok.cleanRanges.removed[1]: range [8, 31] starts at or below the previous end 159",
  );
  // Even an overlap that only shares an endpoint can make the search miss (the boundary is not
  // loosened by one).
  assertThrows(
    () =>
      parseTokenizerAsset(
        tokenizerRanges({
          removed: [
            [8, 31],
            [31, 40],
          ],
          spaced: [],
        }),
        "tok",
      ),
    Error,
    "tok.cleanRanges.removed[1]: range [31, 40] starts at or below the previous end 31",
  );
  // A non-integer makes the code-point comparison meaningless, and an out-of-range value is outside
  // the asset's generation rules.
  assertThrows(
    () => parseTokenizerAsset(tokenizerRanges({ removed: [[0.5, 3]], spaced: [] }), "tok"),
    Error,
    "tok.cleanRanges.removed[0]: range [0.5, 3] is not a pair of integers in 0..1114111",
  );
  assertThrows(
    () => parseTokenizerAsset(tokenizerRanges({ removed: [[0, 0x110000]], spaced: [] }), "tok"),
    Error,
    "tok.cleanRanges.removed[0]: range [0, 1114112] is not a pair of integers in 0..1114111",
  );
  // The spaced side goes through the same check (not only removed is covered).
  assertThrows(
    () => parseTokenizerAsset(tokenizerRanges({ removed: [], spaced: [[5, 4]] }), "tok"),
    Error,
    "tok.cleanRanges.spaced[0]: range [5, 4] has start greater than end",
  );
});
