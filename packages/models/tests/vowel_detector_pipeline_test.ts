// The **construction guards** of `Pipeline`, and the wiring of the operating limits / the mel basis
// asset. Only what needs neither a GPU nor real assets is covered here (the real-GPU comparison
// belongs to `packages/e2e/tests/e2e_vowel_detector_chain_test.ts` <the logits of a synthetic
// golden> and `e2e_vowel_detector_chain_test.ts` <the full chain on real audio → `.lab`> — not
// duplicated. Parity of the host layer with the Python reference is in
// `packages/models/tests/vowel_detector_host_test.ts`).
//
// Six things are pinned:
//
// (1) `fromAssets` rejects **manifest contract violations before acquiring the GPU** (the MUST stated
//     by `openVowelDetectorState` in `src/vowel-detector/pipeline.ts`). The observation setup is the
//     same as SigLIP2 / BiRefNet — **every case is given the containers**, and
//      - a contract-violation case fails with "that violation's message" = the manifest check comes
//        first even when the assets are all there
//      - a correct manifest + an empty Record fails with `component 'crnn' has no container` (the
//        diagnostic at the entry)
//     together pin the order of the checks itself.
//     NOTE: the containers are opened **before** admission (`assetComponentOpener` returns a
//     synchronous supply, so every component is opened first — the seam of ADR 0109).
//
// (2) The **rejection paths** of the comparison with the graph declaration (`assertGraph`). Inside
//     `fromAssets` they cannot be reached without the real assets, so the check is called directly
//     (`tests/helpers/stub-model.ts` builds a declaration-only `GraphOwner`). The old form baked at a
//     fixed length has **the same input/output names and ranks**, so if the check's spelling drifted
//     from the canonical notation of `format/dims.ts`, the happy path alone would stay green.
//
// (3) The **3 declared fields** of `pipelineConfig` (sampleRate / featureDim / classes) each accept
//     exactly one value, and a distribution outside that fails at parse time. In particular the
//     **order of `classes` is the id**, so if a permuted declaration passed, the `.lab` would stay
//     well-formed with only the labels swapped.
//
// (4) The **operating range** (`assertFrameFloor` / `assertFrameLimit`) — both below the floor and
//     above the limit fail loudly. This is the distribution's promise itself, "no silent truncation,
//     no silent padding", and the boundaries (the exact length, one frame outside) are exercised by
//     name on both sides. The range is declared by the distribution, so the TS side holds no
//     constant for it (the MUST in `config.ts`).
//
// (5) The **mel basis asset** (`parseMelBasis`) — tensor name, dtype and shape are checked. Even with
//     a wrong basis the features just become "another plausible value" and pass to the end with the
//     right shape and range.
//
// (6) **Exceptions on the detection run path** (a fake Session) — even when releasing the Session
//     also fails after a failure of run or of the diagnostics callback, the first failure can be
//     traced from the exception the caller gets.

import { test } from "vite-plus/test";
import { assert, assertEquals, assertRejects, assertStrictEquals, assertThrows } from "@std/assert";
import { parseManifest } from "@karume/hub";
import { ModelInputError } from "../src/errors.ts";
import {
  parseVowelDetectorPipelineConfig,
  type VowelDetectorPipelineConfig,
} from "../src/vowel-detector/config.ts";
import {
  assertFrameFloor,
  assertFrameLimit,
  assertGraph,
  parseMelBasis,
  Pipeline,
} from "../src/vowel-detector/pipeline.ts";
import {
  extractFeatures,
  FEATURE_DIM,
  featureFrameCount,
  HOP,
  MEL_BINS,
  N_FFT,
  N_MELS,
  SAMPLE_RATE,
} from "../src/vowel-detector/features.ts";
import { LIPSYNC_CLASSES } from "../src/vowel-detector/postprocess.ts";
import { writeSafetensors } from "./helpers/safetensors-write.ts";
import { type StubDim, stubModel } from "./helpers/stub-model.ts";
import { declaredContainer, partAssets, tensorlessContainer } from "./helpers/container-fixture.ts";
import { fakeDevice, fakeGpuContext } from "../../runtime/tests/helpers/fake-gpu.ts";

/**
 * The operating range the distribution declares
 * (the same numbers as `VOWEL_DETECTOR_MIN_FRAMES` / `VOWEL_DETECTOR_MAX_FRAMES` in
 * `tools/export-recipes/vowel_detector/distribution.py` — the `Dim(min, max)` of the symbol `T`
 * converted to 10ms units).
 */
const MIN_FRAMES = 4;
const MAX_FRAMES = 60_000;

const fileRef = (path: string) => ({ path, size: 16, sha256: "a".repeat(64) });

/** A declaration-only `crnn` container (reproduces what the family checks read, never runs). */
const COMPONENT = partAssets(
  "crnn",
  await tensorlessContainer("crnn", {
    symbols: ["T"],
    inputs: [{ name: "features", shape: [1, "2T", FEATURE_DIM] }],
    output: { name: "logits", shape: [1, "T", LIPSYNC_CLASSES.length] },
  }),
);

/** The actual `pipelineConfig` of `models/karume-vowel-detector/karume.json` (5 fields). */
const PIPELINE_CONFIG: Record<string, unknown> = {
  sampleRate: SAMPLE_RATE,
  featureDim: FEATURE_DIM,
  classes: [...LIPSYNC_CLASSES],
  minFrames: MIN_FRAMES,
  maxFrames: MAX_FRAMES,
};

/** The skeleton of the distribution (only the fields the checks need). `patch` overrides the contents of `models["crnn-epoch3"]`. */
const manifestText = (patch: Record<string, unknown> = {}): string =>
  JSON.stringify({
    format: "karume/5",
    generator: "karume/0.2.2",
    defaultModel: "crnn-epoch3",
    models: {
      "crnn-epoch3": {
        pipeline: "vowel-detector/1",
        weights: {
          crnn: { f32: declaredContainer("crnn-epoch3/model.f32") },
        },
        assets: { mel_basis: fileRef("crnn-epoch3/features/mel-basis.safetensors") },
        quants: { f32: { weights: { crnn: "f32" }, session: {} } },
        defaultQuant: "f32",
        pipelineConfig: PIPELINE_CONFIG,
        ...patch,
      },
    },
  });

const emptyAssets = {} as Record<string, Uint8Array<ArrayBuffer>>;

test("fromAssets: a model that does not exist fails with the list of available ones", async () => {
  const manifest = parseManifest(manifestText());
  await assertRejects(
    () => Pipeline.fromAssets({ manifest, assets: COMPONENT }, { model: "nope" }),
    Error,
    "model 'nope' is not in the manifest",
  );
});

test("fromAssets: rejects a manifest whose pipeline contract name is not vowel-detector", async () => {
  const manifest = parseManifest(manifestText({ pipeline: "sbv2/1" }));
  await assertRejects(
    () => Pipeline.fromAssets({ manifest, assets: COMPONENT }),
    Error,
    "the manifest pipeline is 'sbv2/1'",
  );
});

test("fromAssets: an unknown major fails loudly (checking it is the models side's job — ADR 0038 §1)", async () => {
  // The only check that stops silent degradation of "old implementation × new repository". The hub
  // does not check the major.
  const manifest = parseManifest(manifestText({ pipeline: "vowel-detector/2" }));
  await assertRejects(
    () => Pipeline.fromAssets({ manifest, assets: COMPONENT }),
    Error,
    "has an unsupported major",
  );
});

test("fromAssets: a quant that does not exist fails with the list of available ones", async () => {
  const manifest = parseManifest(manifestText());
  await assertRejects(
    () => Pipeline.fromAssets({ manifest, assets: COMPONENT }, { quant: "nope" }),
    Error,
    "quant 'nope' is not in the manifest",
  );
});

test("fromAssets: an unknown pipelineConfig key fails at construction", async () => {
  // No path where a misspelling (`max_frames` for `maxFrames`) silently degrades to the default.
  const manifest = parseManifest(
    manifestText({ pipelineConfig: { ...PIPELINE_CONFIG, max_frames: 200 } }),
  );
  await assertRejects(
    () => Pipeline.fromAssets({ manifest, assets: COMPONENT }),
    Error,
    "pipelineConfig: unknown key 'max_frames'",
  );
});

test("fromAssets: a missing component container fails with the spellings of both forms (the diagnostic at the entry)", async () => {
  // The flip side of the 5 cases above. A Record without the containers fails on **the missing
  // component** (not on a manifest message) = the contrapositive showing the cases above were not
  // caught up in missing assets.
  const manifest = parseManifest(manifestText());
  await assertRejects(
    () => Pipeline.fromAssets({ manifest, assets: emptyAssets }),
    Error,
    "component 'crnn' has no container",
  );
});

test("fromAssets: with the containers present it gets as far as the missing asset (mel_basis)", async () => {
  // What remains after every family check is the manifest's `assets` (the full surface) — only here
  // does the missing `mel_basis` show (the contrapositive showing the checks come before the asset
  // parsing).
  const manifest = parseManifest(manifestText());
  await assertRejects(
    () => Pipeline.fromAssets({ manifest, assets: COMPONENT }),
    Error,
    "asset 'mel_basis' is missing",
  );
});

// ---- Comparison with the graph declaration (rejection paths) ---------------

/** The same pipelineConfig as the real distribution (this section only checks the declaration comparison). */
const graphConfig = parseVowelDetectorPipelineConfig(PIPELINE_CONFIG);

/** The declaration of a single symbolic-length graph (ADR 0056 / 0057). `patch` breaks one point. */
const crnnGraph = (
  patch: {
    readonly symbols?: readonly string[];
    readonly inputName?: string;
    readonly inputShape?: readonly StubDim[];
    readonly outputShape?: readonly StubDim[];
  } = {},
) =>
  stubModel({
    symbols: patch.symbols ?? ["T"],
    inputs: [
      {
        name: patch.inputName ?? "features",
        shape: patch.inputShape ?? [1, "2T", FEATURE_DIM],
      },
    ],
    outputs: ["logits"],
    values: { logits: patch.outputShape ?? [1, "T", LIPSYNC_CLASSES.length] },
  });

test("assertGraph: a graph baked with a symbolic length returns the time axis's symbol name", () => {
  assertEquals(assertGraph(crnnGraph(), graphConfig), "T");
});

test("assertGraph: the old form baked at a fixed length fails on the symbolic dimensions", () => {
  // The input/output names and ranks are the same, so placed in the same slot it would pass
  // construction.
  assertThrows(
    () =>
      assertGraph(
        crnnGraph({
          symbols: [],
          inputShape: [1, 2000, FEATURE_DIM],
          outputShape: [1, 1000, LIPSYNC_CLASSES.length],
        }),
        graphConfig,
      ),
    Error,
    "vowelDetector.Pipeline: the graph's symbolic dimensions are []",
  );
});

test("assertGraph: fails when the factor of 2 is missing from the input's time axis", () => {
  // A distribution without the factor just stretches the `.lab` times twofold; the shapes still work.
  assertThrows(
    () => assertGraph(crnnGraph({ inputShape: [1, "T", FEATURE_DIM] }), graphConfig),
    Error,
    "vowelDetector.Pipeline: the graph input shape is [1, T, 83], expected [1, 2T, 83]",
  );
});

test("assertGraph: fails when the input name is not features", () => {
  assertThrows(
    () => assertGraph(crnnGraph({ inputName: "input" }), graphConfig),
    Error,
    "vowelDetector.Pipeline: the graph input is 'input'",
  );
});

test("assertGraph: fails when the output's class count differs from the declaration", () => {
  // With a wrong class count the `.lab` stays well-formed and only the label assignment breaks.
  assertThrows(
    () => assertGraph(crnnGraph({ outputShape: [1, "T", 5] }), graphConfig),
    Error,
    "vowelDetector.Pipeline: the graph output shape is [1, T, 5]",
  );
});

// ---- The pipelineConfig schema (3 declared fields + 2 operating-range fields) ----

const config = (patch: Record<string, unknown> = {}): VowelDetectorPipelineConfig =>
  parseVowelDetectorPipelineConfig({ ...PIPELINE_CONFIG, ...patch });

test("pipelineConfig: the 5 actual fields are read as is", () => {
  const parsed = config();
  assertEquals(parsed.sampleRate, SAMPLE_RATE);
  assertEquals(parsed.featureDim, FEATURE_DIM);
  assertEquals([...parsed.classes], [...LIPSYNC_CLASSES]);
  assertEquals(parsed.minFrames, MIN_FRAMES);
  assertEquals(parsed.maxFrames, MAX_FRAMES);
});

test("pipelineConfig: a distribution at other than 16kHz is not accepted (there is no resampler)", () => {
  assertThrows(
    () => config({ sampleRate: 22050 }),
    Error,
    "pipelineConfig.sampleRate: this implementation supports only 16000",
  );
});

test("pipelineConfig: a distribution whose feature dimension is not 83 is not accepted", () => {
  assertThrows(
    () => config({ featureDim: 80 }),
    Error,
    "pipelineConfig.featureDim: this implementation supports only 83",
  );
});

test("pipelineConfig: a distribution with a different **order** of classes is not accepted (the order is the id)", () => {
  // The same set in a different order = the labels are just permuted and the .lab is fully
  // well-formed.
  const swapped = [...LIPSYNC_CLASSES];
  [swapped[0], swapped[1]] = [swapped[1], swapped[0]];
  assertThrows(
    () => config({ classes: swapped }),
    Error,
    "pipelineConfig.classes: this implementation supports only",
  );
});

test("pipelineConfig: an odd, non-integer or non-positive limit declaration fails", () => {
  // The graph input is `2T`, so an odd limit is "a limit that only that one frame fails to pass" =
  // its meaning is broken.
  for (const value of [251, 0, -2, 2.5, "2000"]) {
    assertThrows(
      () => config({ maxFrames: value }),
      Error,
      "pipelineConfig.maxFrames: not a positive multiple of 2",
      `maxFrames=${JSON.stringify(value)}`,
    );
  }
});

test("pipelineConfig: a missing limit fails (no fallback to a default)", () => {
  const { maxFrames: _dropped, ...rest } = PIPELINE_CONFIG;
  assertThrows(
    () => parseVowelDetectorPipelineConfig(rest),
    Error,
    "pipelineConfig.maxFrames: missing",
  );
});

test("pipelineConfig: an odd, non-integer or non-positive floor declaration fails", () => {
  // The floor is also a count of 10ms frames (the graph input is `2T`), so the same step check as the
  // limit applies.
  for (const value of [3, 0, -2, 2.5, "4"]) {
    assertThrows(
      () => config({ minFrames: value }),
      Error,
      "pipelineConfig.minFrames: not a positive multiple of 2",
      `minFrames=${JSON.stringify(value)}`,
    );
  }
});

test("pipelineConfig: a missing floor fails (no fallback to a default)", () => {
  const { minFrames: _dropped, ...rest } = PIPELINE_CONFIG;
  assertThrows(
    () => parseVowelDetectorPipelineConfig(rest),
    Error,
    "pipelineConfig.minFrames: missing",
  );
});

test("pipelineConfig: a declaration whose floor exceeds its limit fails (the accepted set would be empty)", () => {
  assertThrows(
    () => config({ minFrames: MAX_FRAMES + 2 }),
    Error,
    "would accept no input length at all",
  );
});

// ---- Operating range (the input-length check, both sides) -------------------

test("operating limit: the exact length passes, one frame over fails (the boundary is closed on the upper side)", () => {
  const parsed = config();
  assertFrameLimit(parsed, 2);
  assertFrameLimit(parsed, MAX_FRAMES);
  // The length is the size of the caller's waveform, so this is an input error (ADR 0107 decision 2
  // — passing it in pieces succeeds).
  const error = assertThrows(
    () => assertFrameLimit(parsed, MAX_FRAMES + 2),
    ModelInputError,
    "the audio is too long",
  );
  // The message must say how many seconds pass (so the caller splits the audio instead of it being
  // truncated).
  assert(error.message.includes(`${MAX_FRAMES} frames`), error.message);
  assert(error.message.includes("600.00 s"), error.message);
});

test("operating floor: the exact length passes, one frame short fails (the boundary is closed on the lower side)", () => {
  const parsed = config();
  // Exactly the baked-in floor (4 frames of 10ms = symbol T is 2) passes.
  assertFrameFloor(parsed, MIN_FRAMES);
  assertFrameFloor(parsed, MAX_FRAMES);
  for (const frames of [MIN_FRAMES - 1, MIN_FRAMES - 2]) {
    const error = assertThrows(
      () => assertFrameFloor(parsed, frames),
      ModelInputError,
      "the audio is too short",
      `frames=${frames}`,
    );
    // The message must say how many frames pass (so the caller passes longer audio instead of it
    // being padded).
    assert(error.message.includes(`${MIN_FRAMES} frames`), error.message);
    assert(error.message.includes(`${frames} frames of 10ms`), error.message);
  }
});

test("operating floor: the floor comes from the distribution's declaration (not a TS-side constant)", () => {
  // The same input length passes or fails with distributions declaring different floors = the
  // number comes from the manifest.
  const strict = config({ minFrames: 8 });
  assertThrows(() => assertFrameFloor(strict, 6), ModelInputError, "8 frames");
  assertFrameFloor(config({ minFrames: 2 }), 2);
});

// ---- The mel basis asset ----------------------------------------------------

const melBasisBytes = (
  name = "mel_basis",
  shape: readonly number[] = [N_MELS, MEL_BINS],
): Uint8Array<ArrayBuffer> =>
  writeSafetensors(
    new Map([
      [
        name,
        {
          dtype: "F32" as const,
          shape,
          data: new Float32Array(shape.reduce((product, dim) => product * dim, 1)),
        },
      ],
    ]),
    {},
  );

test("mel basis: reads exactly one f32 tensor of [80, 257]", () => {
  const basis = parseMelBasis(melBasisBytes().buffer);
  assertEquals(basis.length, N_MELS * MEL_BINS);
});

test("mel basis: a wrong tensor name fails, listing what is there", () => {
  assertThrows(
    () => parseMelBasis(melBasisBytes("mel").buffer),
    Error,
    "has no tensor 'mel_basis'",
  );
});

test("mel basis: a transpose fails even with the same element count (both axes of the shape are checked)", () => {
  assertThrows(
    () => parseMelBasis(melBasisBytes("mel_basis", [MEL_BINS, N_MELS]).buffer),
    Error,
    `expected [${N_MELS}, ${MEL_BINS}]`,
  );
});

test("mel basis: a storage type other than f32 fails", () => {
  const bytes = writeSafetensors(
    new Map([
      [
        "mel_basis",
        {
          dtype: "I32" as const,
          shape: [N_MELS, MEL_BINS],
          data: new Int32Array(N_MELS * MEL_BINS),
        },
      ],
    ]),
    {},
  );
  assertThrows(() => parseMelBasis(bytes.buffer), Error, "the mel basis is not F32");
});

// ---- The detection run path (a fake Session shows how exceptions are carried) ----
//
// `detect` sets up and tears down a Session per detection. When releasing the Session also fails
// after run or the diagnostics callback failed (a device loss causes both), this pins that **the
// first failure** can be traced from the exception the caller receives. A fake Session and a fake
// GPU suffice (nothing past `createSession` is touched), so the state is built directly and passed
// to the private constructor (the same approach as `wan_pipeline_test.ts`).

/** What the fake Session throws, and the diagnostics callback the caller passes. */
type FakeRunPlan = {
  readonly runError?: Error;
  readonly disposeError?: Error;
  readonly onRunDiagnostics?: () => void;
  /** Defaults to a distribution with the operating range 4..60,000. */
  readonly config?: VowelDetectorPipelineConfig;
  /** Defaults to a zero basis of the right size. */
  readonly melBasis?: Float32Array<ArrayBuffer>;
};

/** One second of a sine wave (97 frames of 10ms → 96 after dropping the remainder = inside the operating range). */
const ONE_SECOND = Float32Array.from(
  { length: SAMPLE_RATE },
  (_, index) => 0.5 * Math.sin((2 * Math.PI * 220 * index) / SAMPLE_RATE),
);

/** A pipeline built on a CRNN that sets up a fake Session, and the number of release calls. */
const pipelineWithFakeSession = (
  plan: FakeRunPlan,
): { readonly pipeline: Pipeline; readonly disposeCalls: () => number } => {
  let disposeCalls = 0;
  const state = {
    gpu: fakeGpuContext(fakeDevice()),
    ownsGpu: false,
    config: plan.config ?? graphConfig,
    graph: {
      graph: { outputs: ["logits"] },
      createSession: () =>
        Promise.resolve({
          run: () =>
            plan.runError === undefined
              ? Promise.resolve({
                  logits: {
                    dtype: "f32",
                    shape: [1, 48, LIPSYNC_CLASSES.length],
                    data: new Float32Array(48 * LIPSYNC_CLASSES.length),
                  },
                })
              : Promise.reject(plan.runError),
          diagnostics: () => ({}),
          dispose: () => {
            disposeCalls += 1;
            return plan.disposeError === undefined
              ? Promise.resolve()
              : Promise.reject(plan.disposeError);
          },
        }),
    },
    symbol: "T",
    sessionOptions: {},
    melBasis: plan.melBasis ?? new Float32Array(N_MELS * MEL_BINS),
    ...(plan.onRunDiagnostics === undefined ? {} : { onRunDiagnostics: plan.onRunDiagnostics }),
  };
  const pipeline: Pipeline = Reflect.construct(Pipeline, [state]);
  return { pipeline, disposeCalls: () => disposeCalls };
};

test("detect: when releasing the Session also fails after run failed, both are carried with run's exception first", async () => {
  const runFailure = new Error("device lost in the middle of run");
  const disposeFailure = new Error("flush failed");
  const { pipeline, disposeCalls } = pipelineWithFakeSession({
    runError: runFailure,
    disposeError: disposeFailure,
  });

  const error = await assertRejects(() => pipeline.detect(ONE_SECOND), AggregateError);

  // If only the release's exception arrived, what broke first (the run failure) would be lost.
  assertStrictEquals(error.errors[0], runFailure);
  assertStrictEquals(error.errors[1], disposeFailure);
  assertEquals(disposeCalls(), 1);
});

test("detect: when releasing the Session also fails after the diagnostics callback failed, the callback's exception is carried first", async () => {
  const callbackFailure = new Error("the diagnostics callback threw");
  const disposeFailure = new Error("flush failed");
  const { pipeline, disposeCalls } = pipelineWithFakeSession({
    disposeError: disposeFailure,
    onRunDiagnostics: () => {
      throw callbackFailure;
    },
  });

  const error = await assertRejects(() => pipeline.detect(ONE_SECOND), AggregateError);

  assertStrictEquals(error.errors[0], callbackFailure);
  assertStrictEquals(error.errors[1], disposeFailure);
  assertEquals(disposeCalls(), 1);
});

test("detect: when only run fails, its exception is carried as is and the Session is released exactly once", async () => {
  const runFailure = new Error("failed in the middle of run");
  const { pipeline, disposeCalls } = pipelineWithFakeSession({ runError: runFailure });

  const error = await assertRejects(() => pipeline.detect(ONE_SECOND));

  assertStrictEquals(error, runFailure);
  assertEquals(disposeCalls(), 1);
});

// ---- The operating limit is checked before the features are built ---------------
//
// The frame count is determined by the waveform length alone. Running the FFT and the mel transform
// over every frame of audio past the limit before rejecting it blocks synchronously for seconds on
// audio of around 600 seconds (with the same rejection as the result). The observation point is "a
// mel basis of broken size" — entering the feature extraction would fail on the basis size, so the
// limit's rejection coming first is the evidence that "the extraction was not entered".

/** Silence of the length that yields exactly `frames` frames of 10ms. */
const silenceOfFrames = (frames: number): Float32Array<ArrayBuffer> =>
  new Float32Array(N_FFT + (frames - 1) * HOP);

test("frame count: the count derived from the length matches the feature extraction's count", () => {
  const melBasis = new Float32Array(N_MELS * MEL_BINS);
  for (const samples of [N_FFT, N_FFT + HOP - 1, N_FFT + HOP, SAMPLE_RATE, SAMPLE_RATE + 77]) {
    assertEquals(
      featureFrameCount(samples),
      extractFeatures(new Float32Array(samples), melBasis).frames,
      `samples=${samples}`,
    );
  }
  // A length too short for a single frame gives less than 1 (the caller decides acceptance).
  assert(featureFrameCount(N_FFT - 1) < 1);
});

test("detect: audio past the limit fails as an input error before the features are built", async () => {
  const { pipeline } = pipelineWithFakeSession({
    config: config({ maxFrames: 8 }),
    melBasis: new Float32Array(0),
  });

  // 10 frames (no remainder) exceed the limit 8. Had the extraction been entered, it would fail on
  // the mel basis size.
  await assertRejects(
    () => pipeline.detect(silenceOfFrames(10)),
    ModelInputError,
    "the audio is too long",
  );
});

test("detect: audio that is exactly at the limit after dropping the remainder passes the limit and proceeds to the feature extraction", async () => {
  const { pipeline } = pipelineWithFakeSession({
    config: config({ maxFrames: 8 }),
    melBasis: new Float32Array(0),
  });

  // 9 frames drop a remainder of 1 to give 8 = exactly the limit. The limit is compared against the
  // value after truncation.
  // NOTE: a test that pins the boundary; it does not detect a regression (with the limit check
  // removed it still fails with the same message on the mel basis size). The regression is pinned
  // by the "past the limit" test above.
  await assertRejects(
    () => pipeline.detect(silenceOfFrames(9)),
    Error,
    "the mel basis has 0 elements",
  );
});
