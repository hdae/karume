// Audio input of the EmbeddingGemma family over the local distribution of EmbeddingGemma 2 —
// `embeddinggemma.Pipeline` built by `fromPretrained` from `models/karume-embeddinggemma-2/` with
// `audio: true` and `vision: true` (one fixture input holds an image too) (the recipe
// `tools/export-recipes/embeddinggemma2/distribution.py`; model `text`, every preset of the mirror
// — `bf16`, the default, and `i8`, each with the towers in its own storage: the checkpoint's bf16
// weights, or those rounded to per-row int8 — the audio tower's two subsampling convolutions kept
// bf16).
//
// The reference is the tracked fixture `packages/models/tests/fixtures/embeddinggemma2/
// multimodal-reference.{json,safetensors}` (`multimodal_reference.py`): four whole inputs holding an
// audio (an audio alone, a text then an audio, a `query`-prompted audio whose frame count is 2 mod
// 4, an image then an audio), with their 16 kHz waveforms, the exact token ids upstream's
// SentenceTransformer ran and the 768-wide vectors it returned (f32, CPU).
//
// ## Checks
//
// 1. **The token ids** (no GPU): the ids the pipeline runs for each input — `<bos>`, the text runs,
//    each audio as `<|audio>` + n × `<|audio|>` + `<audio|>` (and the image as its block), `<eos>` —
//    equal upstream's exactly, and each block's soft rows start where upstream's do; the mirror
//    declares `pipelineConfig.audio` as the recipe reads it and each preset maps the audio tower
//    of its own storage (the text tower's).
// 2. **The four inputs through each preset** (real GPU): the pipeline fetched the text graph and
//    both towers of the preset's storage and the tokenizer, nothing else; `tokenize` gives the
//    fixture's ids, the vector is a unit vector, it bit-for-bit equals this machine's sha256 row of
//    that preset (row ID `embeddinggemma-2/<preset>/audio/<case>`), and against upstream's vector:
//    `bf16` within {@link AUDIO_TOLERANCE} (its derivation is there), `i8` at a cosine of at least
//    {@link I8_COSINE_FLOOR} (its text tower and its towers are the int8 copies). A 30 s audio (750
//    soft tokens) gives a unit vector and its own row (`…/audio/full-length`).
// 3. **A text call on the audio pipeline** is the text call of the pipeline without it: its vector
//    has the sha256 row the text pipeline e2e pins (`embeddinggemma-2/<preset>/query-en`).
// 4. **The text graph's soft-row path alone** (real GPU, `bf16`, `KARUME_SERIES=1` with the tower
//    series present): the embedder over the mirror's text graph, its towers stand-ins that answer
//    with upstream's own soft tokens (the audio series' and the vision series' goldens), against
//    upstream's vectors within {@link SOFT_ROW_TOLERANCE} — the towers' GPU noise and the host
//    front end's difference to upstream's frames are out of the comparison.
// 5. **The 30 s audio within the preset's limits** (real GPU), per preset, on a device whose binding
//    and buffer limits are capped at the preset's own need (`bf16` 256 MiB, `i8` the WebGPU default
//    128 MiB): it embeds to the same bits as on the uncapped device (that run is taken again when
//    the uncapped block did not run first), so the audio tower adds no limit to the preset's.
//
// Without the mirror every case SKIPs explicitly (the distribution gate owns that red); without a
// GPU the GPU cases SKIP. Rows are created with `KARUME_REFERENCE=write` on the lane.

import { isNotFound, readFile, readTextFile, statSync } from "@karume/test-support/fs";
import { afterAll, beforeAll, describe, it } from "vite-plus/test";
import { assert, assertEquals, fail } from "@std/assert";
import { parseManifest } from "@karume/hub";
import {
  type GpuContext,
  parseSafetensors,
  type RunInputs,
  type SafetensorsFile,
} from "@karume/runtime";
import * as embeddinggemma from "../../models/embeddinggemma.ts";
import { embeddingGemmaAudioContract } from "../../models/src/embeddinggemma/audio.ts";
import { parseEmbeddingGemmaPipelineConfig } from "../../models/src/embeddinggemma/config.ts";
import {
  type EmbedTextRules,
  embedTextRules,
  planEmbedRequest,
} from "../../models/src/embeddinggemma/input.ts";
import {
  type Embedder,
  type EmbeddingSession,
  embeddingGemmaGraphContract,
  openEmbedder,
} from "../../models/src/embeddinggemma/pipeline.ts";
import { embeddingGemmaVisionContract } from "../../models/src/embeddinggemma/vision.ts";
import { parseGemmaTokenizerAsset } from "../../models/src/gemma/text/asset.ts";
import { Tokenizer } from "../../models/src/gemma/text/tokenizer.ts";
import { acquireTestGpu, GPU_AVAILABLE } from "../../models/tests/helpers/gpu.ts";
import {
  mirrorDirectory,
  mirrorPresent,
  mirrorRoot,
  openMirrorComponent,
  readMirrorAsset,
} from "../../models/tests/helpers/mirror.ts";
import { SERIES_CHECKS, SERIES_CHECKS_OPT_IN } from "../../models/tests/helpers/series-checks.ts";
import {
  compareTensors,
  formatAllclose,
  type Tolerance,
} from "../../runtime/src/reference/allclose.ts";
import { LIMIT_CAPS } from "../../runtime/src/gpu/device.ts";
import { verifyRunningGpu } from "../../runtime/tests/helpers/environment.ts";
import {
  f32ArtifactBytes,
  openReferences,
  referenceMismatchMessage,
  type ReferenceSettlement,
  registerReferenceGate,
  settleOrObserve,
} from "../../runtime/tests/helpers/reference.ts";
import {
  type Measurement,
  openResults,
  runRecordedCase,
} from "../../runtime/tests/helpers/results.ts";

/**
 * The tolerance of the `bf16` preset's vectors against upstream's SentenceTransformer vectors.
 *
 * Measured through the pipeline on node-nvidia-geforce-rtx-3080-ti (2026-11-23, four inputs,
 * `[768]` unit vectors; cosine in f64). The columns on the right place the error: the GPU tower fed
 * upstream's own frames (the audio series' io inputs) instead of the host's, the text graph on the
 * GPU fed upstream's own soft tokens (check 4), and the two graphs composed in torch f32 (the
 * fixture's `composition`) — those three measured on 2026-11-22:
 *
 * | Case             | T   | waveform (L mod 4) | maxAbs  | cosine         | tower on upstream's frames | text graph + upstream's soft tokens | torch composition |
 * | ---------------- | --- | ------------------ | ------- | -------------- | -------------------------- | ----------------------------------- | ----------------- |
 * | audio-only       | 29  | sweep (3)          | 1.46e-7 | 0.999999999999 | 1.64e-7                    | 9.87e-8                             | 6.98e-8           |
 * | text-then-audio  | 34  | click (0)          | 1.49e-7 | 0.999999999999 | 1.53e-7                    | 9.69e-8                             | 5.96e-8           |
 * | query-audio      | 38  | edge (2)           | 1.80e-7 | 0.999999999999 | 3.04e-7                    | 1.12e-7                             | 1.19e-7           |
 * | image-then-audio | 287 | sweep (3)          | 2.44e-7 | 0.999999999998 | 3.28e-7                    | 4.94e-8                             | 5.03e-8           |
 *
 * The host's frames are upstream's to one ulp (the f32 product of the samples and the window is
 * upstream's own; only the FFT runs in f64 — `embeddinggemma2_audio_input_test.ts`), so the vectors
 * sit where the GPU tower fed upstream's frames does: the GPU's own f32 noise, the order of the
 * torch composition's. (With the product left in f64, the frames were up to 5.12e-4 off in
 * near-silent mel bins, the soft tokens up to 2.6e-3, and the vectors up to 2.45e-5 —
 * `query-audio`; the bound was 7.5e-5.) atol 7.5e-7 is three times the worst (`image-then-audio`);
 * rtol 0 — a unit vector's values thin out to near 0, so atol alone decides. The breakages this
 * guards against are far above it: soft rows shifted by one row, the frame mask left all ones on
 * `edge` (its last soft token moves by 4.2), a hop of 128 or the power spectrum in place of the
 * magnitude, and the windowed product left in f64 (2.45e-5).
 */
const AUDIO_TOLERANCE: Tolerance = { atol: 7.5e-7, rtol: 0 };

/**
 * The tolerance of check 4: the `bf16` text graph on the GPU fed upstream's own soft tokens,
 * against upstream's vectors.
 *
 * Measured on node-nvidia-geforce-rtx-3080-ti (2026-11-22, the four inputs): maxAbs 9.87e-8
 * (`audio-only`), 9.69e-8 (`text-then-audio`), 1.12e-7 (`query-audio`) and 4.94e-8
 * (`image-then-audio`) — the order of the torch composition's own error (the table above). atol
 * 3.5e-7 is three times the worst; rtol 0, as for {@link AUDIO_TOLERANCE}. It is about half that
 * one, whose budget the audio tower's GPU noise shares: this check holds the text graph's soft-row
 * path alone, out of reach of that noise.
 */
const SOFT_ROW_TOLERANCE: Tolerance = { atol: 3.5e-7, rtol: 0 };

/**
 * The floor of the cosine between the `i8` preset's vector of an input holding an audio and
 * upstream's: the text pipeline e2e's floor between the `i8` and the `bf16` vector of a text
 * (`I8_COSINE_FLOOR` of `e2e_embeddinggemma2_pipeline_test.ts`, whose derivation is there).
 *
 * Measured on node-nvidia-geforce-rtx-3080-ti (2026-11-23, both towers int8): 0.999762
 * (`query-audio`), 0.999803 (`audio-only`), 0.999859 (`image-then-audio`) and 0.999907
 * (`text-then-audio`), maxAbs 1.90e-3 to 2.63e-3 — the rounding of the text tower and of the
 * towers together. With the towers still bf16 (the same day) it was 0.999819 to 0.999909, the text
 * tower's rounding alone. That is the order the torch-CPU screening of the towers' int8 forms
 * measured on 32 clips of synthetic speech (all-i8, the shipped form: lowest 0.999493, mean
 * 0.999759 at 768 values — docs/research/2026-11-23-embeddinggemma2-tower-quant-screen.md), and
 * 0.999 leaves 1 − cos = 1e-3, 4.2 times the worst here (2.38e-4). No golden of the whole int8
 * model holds audio, so no tighter band is drawn.
 */
const I8_COSINE_FLOOR = 0.999;

/** One preset of the mirror and how its vectors are held to upstream's. */
type PresetSpec = {
  readonly quant: string;
  /** `tolerance`: an allclose against upstream's vector; `cosine`: a floor of the cosine to it. */
  readonly check:
    | { readonly kind: "tolerance"; readonly tolerance: Tolerance }
    | { readonly kind: "cosine"; readonly floor: number };
  /** The binding and buffer limit of its narrowed device (check 5). */
  readonly cappedBindingBytes: number;
};

const PRESETS: readonly PresetSpec[] = [
  {
    quant: "bf16",
    check: { kind: "tolerance", tolerance: AUDIO_TOLERANCE },
    cappedBindingBytes: 256 * 2 ** 20,
  },
  {
    quant: "i8",
    check: { kind: "cosine", floor: I8_COSINE_FLOOR },
    cappedBindingBytes: 128 * 2 ** 20,
  },
];

/** The local distribution (`dist.py --pipeline embeddinggemma-2` builds it). */
const MIRROR_ROOT = mirrorRoot("karume-embeddinggemma-2");
const MIRROR = mirrorPresent(MIRROR_ROOT);
if (!MIRROR) {
  console.warn(
    `[karume] the distribution mirror ${MIRROR_ROOT.pathname} is absent (or not karume/6), so the` +
      " EmbeddingGemma 2 audio-input checks SKIP. Assemble: cd tools/export-recipes && uv run" +
      " python dist.py --pipeline embeddinggemma-2 (after the text, vision and audio series)",
  );
}
const RUNNABLE = MIRROR && GPU_AVAILABLE;

/** The fixture (`tools/export-recipes/embeddinggemma2/multimodal_reference.py`). */
type FixturePart =
  | { readonly type: "text"; readonly text: string }
  | { readonly type: "image"; readonly image: string }
  | { readonly type: "audio"; readonly audio: string };
type FixtureBlock = { readonly start: number; readonly softTokens: number };
type FixtureCase = {
  readonly name: string;
  readonly parts: readonly FixturePart[];
  readonly prompt?: string;
  readonly tokens: number;
  readonly imageBlocks: readonly (FixtureBlock & { readonly image: string })[];
  readonly audioBlocks: readonly (FixtureBlock & { readonly audio: string })[];
  readonly composition: { readonly maxAbs: number; readonly cosine: number };
};
type Fixture = {
  readonly audioTokens: { readonly begin: number; readonly soft: number; readonly end: number };
  readonly images: readonly {
    readonly name: string;
    readonly width: number;
    readonly height: number;
  }[];
  readonly audios: readonly { readonly name: string; readonly sampleRate: number }[];
  readonly audioCases: readonly FixtureCase[];
};

const FIXTURE = JSON.parse(
  await readTextFile(
    new URL(
      "../../models/tests/fixtures/embeddinggemma2/multimodal-reference.json",
      import.meta.url,
    ),
  ),
) as Fixture;
const TENSORS: SafetensorsFile = await (async () => {
  const bytes = await readFile(
    new URL(
      "../../models/tests/fixtures/embeddinggemma2/multimodal-reference.safetensors",
      import.meta.url,
    ),
  );
  return parseSafetensors(
    bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength),
  );
})();

/** A copy of a fixture tensor's bytes (fails loudly when absent or of another dtype). */
const fixtureBytes = (key: string, dtype: "U8" | "I32" | "F32"): Uint8Array<ArrayBuffer> => {
  const view = TENSORS.tensors.get(key);
  assert(view !== undefined, `the fixture has no tensor '${key}'`);
  assertEquals(view.dtype, dtype, `${key}: dtype`);
  return new Uint8Array(TENSORS.buffer, view.byteOffset, view.byteLength).slice();
};
const fixtureIds = (name: string): number[] => [
  ...new Int32Array(fixtureBytes(`${name}.input_ids`, "I32").buffer),
];
const fixtureVector = (name: string): Float32Array<ArrayBuffer> =>
  new Float32Array(fixtureBytes(`${name}.embedding`, "F32").buffer);
const fixtureImage = (name: string): embeddinggemma.Rgb8Image => {
  const entry = FIXTURE.images.find((image) => image.name === name);
  assert(entry !== undefined, `the fixture has no image '${name}'`);
  return { data: fixtureBytes(`image.${name}`, "U8"), width: entry.width, height: entry.height };
};
const fixtureAudio = (name: string): embeddinggemma.MonoAudio => {
  const entry = FIXTURE.audios.find((audio) => audio.name === name);
  assert(entry !== undefined, `the fixture has no audio '${name}'`);
  return {
    data: new Float32Array(fixtureBytes(`audio.${name}`, "F32").buffer),
    sampleRate: entry.sampleRate,
  };
};

/** The content parts of a fixture case (its images and audios read from the fixture's tensors). */
const partsOf = (entry: FixtureCase): readonly embeddinggemma.EmbedContentPart[] =>
  entry.parts.map((part) => {
    if (part.type === "text") return part;
    if (part.type === "image") return { type: "image", image: fixtureImage(part.image) };
    return { type: "audio", audio: fixtureAudio(part.audio) };
  });
const optionsOf = (entry: FixtureCase): embeddinggemma.EmbedOptions =>
  entry.prompt === undefined ? {} : { prompt: entry.prompt };

/**
 * A 30 s audio (480,000 samples, 750 soft tokens — the front end's longest): a slow sweep from
 * 60 Hz to 6 kHz under a tremolo, on 16-bit PCM steps (so its samples are the same bits wherever
 * `Math.sin` is), an ordinary input for the tower's largest run.
 */
const FULL_LENGTH_AUDIO: embeddinggemma.MonoAudio = (() => {
  const samples = 480_000;
  const data = new Float32Array(samples);
  const duration = samples / 16000;
  const rate = Math.log(6000 / 60);
  for (let index = 0; index < samples; index += 1) {
    const time = index / 16000;
    const phase = ((2 * Math.PI * 60 * duration) / rate) * (Math.exp((time / duration) * rate) - 1);
    const tremolo = 0.75 + 0.25 * Math.sin(2 * Math.PI * 0.25 * time);
    data[index] = Math.round(0.3 * tremolo * Math.sin(phase) * 32767) / 32768;
  }
  return { data, sampleRate: 16000 };
})();
const FULL_LENGTH_TOKENS = 750;

/** The f32 tower series (outside the repository): their io files hold upstream's soft tokens. */
const AUDIO_SERIES = new URL("../../../outputs/series/embeddinggemma-2-audio/", import.meta.url);
const VISION_SERIES = new URL("../../../outputs/series/embeddinggemma-2-vision/", import.meta.url);
const goldenIo = (series: URL, name: string): URL => new URL(`io.${name}.safetensors`, series);

/** Whether the file exists (anything but NotFound propagates). */
const exists = (url: URL): boolean => {
  try {
    return statSync(url).isFile;
  } catch (cause) {
    if (isNotFound(cause)) return false;
    throw cause;
  }
};

/** Whether check 4 can run: the io file of every audio and image the fixture's inputs hold. */
const GOLDEN_SOFT_TOKENS =
  SERIES_CHECKS &&
  FIXTURE.audioCases.every(
    (entry) =>
      entry.audioBlocks.every(({ audio }) => exists(goldenIo(AUDIO_SERIES, audio))) &&
      entry.imageBlocks.every(({ image }) => exists(goldenIo(VISION_SERIES, image))),
  );
if (MIRROR && SERIES_CHECKS && !GOLDEN_SOFT_TOKENS) {
  console.warn(
    `[karume] ${AUDIO_SERIES.pathname} or ${VISION_SERIES.pathname} lacks the io files of the` +
      " fixture's audios and images, so the check of the text graph fed upstream's soft tokens" +
      " SKIPs. Generate: cd tools/export-recipes && uv run --with 'transformers==5.23.0' python -m" +
      " embeddinggemma2.export_audio (and export_vision)",
  );
}

/** Upstream's soft tokens of an io file (`output.0` `[1, n, 512]`). */
const goldenSoftTokens = async (io: URL): Promise<Float32Array<ArrayBuffer>> => {
  const bytes = await readFile(io);
  const file = parseSafetensors(
    bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength),
  );
  const view = file.tensors.get("output.0");
  assert(view !== undefined && view.dtype === "F32", `${io.pathname}: no f32 output.0`);
  return new Float32Array(file.buffer.slice(view.byteOffset, view.byteOffset + view.byteLength));
};

const unitNormError = (vector: Float32Array<ArrayBuffer>): number =>
  Math.abs(Math.hypot(...vector) - 1);

/** The cosine of two vectors in f64. */
const cosine = (left: Float32Array<ArrayBuffer>, right: Float32Array<ArrayBuffer>): number => {
  assertEquals(left.length, right.length, "cosine of vectors of two widths");
  let dot = 0;
  let leftSquares = 0;
  let rightSquares = 0;
  for (let index = 0; index < left.length; index += 1) {
    dot += left[index] * right[index];
    leftSquares += left[index] * left[index];
    rightSquares += right[index] * right[index];
  }
  return dot / Math.sqrt(leftSquares * rightSquares);
};

const milliseconds = (start: number): number => Math.round(performance.now() - start);

/** `outputs/verify/<environment key>/<date>_embeddinggemma2-audio-pipeline/` (safe to delete). */
const results = openResults("embeddinggemma2-audio-pipeline");

/** The sha256 rows of the pipeline's outputs (the text pipeline e2e's file — rows per environment). */
const references = openReferences(
  new URL("../../models/tests/fixtures/references/embeddinggemma.json", import.meta.url),
);

const audioCaseId = (quant: string, name: string): string =>
  `embeddinggemma-2/${quant}/audio/${name}`;
/** The text pipeline e2e's row of the `query-en` golden text (check 3 settles against it). */
const textCaseId = (quant: string): string => `embeddinggemma-2/${quant}/query-en`;
/** The golden text `query-en` (the fixture of the text pipeline e2e). */
const QUERY_EN = await (async () => {
  const cases = JSON.parse(
    await readTextFile(
      new URL("../../models/tests/fixtures/embeddinggemma2/golden-cases.json", import.meta.url),
    ),
  ) as {
    readonly cases: readonly {
      readonly name: string;
      readonly prompt?: string;
      readonly text: string;
    }[];
  };
  const found = cases.cases.find((entry) => entry.name === "query-en");
  assert(found !== undefined, "the golden-case fixture has no case 'query-en'");
  return found;
})();

const CASE_IDS = RUNNABLE
  ? PRESETS.flatMap(({ quant }) => [
      ...FIXTURE.audioCases.map((entry) => audioCaseId(quant, entry.name)),
      audioCaseId(quant, "full-length"),
      textCaseId(quant),
    ])
  : [];

const blocksOf = (blocks: readonly FixtureBlock[]): { start: number; softTokens: number }[] =>
  blocks.map(({ start, softTokens }) => ({ start, softTokens }));

describe.skipIf(!MIRROR)("EmbeddingGemma 2 mirror: audio input on the host (no GPU)", () => {
  let rules: EmbedTextRules;
  beforeAll(async () => {
    const manifest = parseManifest(await readTextFile(new URL("karume.json", MIRROR_ROOT)));
    const config = parseEmbeddingGemmaPipelineConfig(manifest.models["text"]?.pipelineConfig);
    const tokenizer = new Tokenizer(
      parseGemmaTokenizerAsset(await readMirrorAsset(MIRROR_ROOT, "tokenizer")),
    );
    rules = embedTextRules(tokenizer, config, { images: true, audio: true });
  });

  it("declares pipelineConfig.audio as the recipe reads it, and each preset maps the tower of its own storage", async () => {
    const manifest = parseManifest(await readTextFile(new URL("karume.json", MIRROR_ROOT)));
    const entry = manifest.models["text"];
    assert(entry !== undefined, "the mirror has no model 'text'");
    assertEquals(rules.audio, {
      audioTokenId: FIXTURE.audioTokens.soft,
      beginAudioTokenId: FIXTURE.audioTokens.begin,
      endAudioTokenId: FIXTURE.audioTokens.end,
      hiddenSize: 512,
      maxSoftTokens: 750,
      sampleRate: 16000,
    });
    assertEquals(
      Object.keys(entry.weights["audio"] ?? {}),
      PRESETS.map(({ quant }) => quant),
    );
    for (const { quant } of PRESETS) {
      assertEquals(entry.quants[quant]?.weights["audio"], quant, `${quant}: the tower`);
    }
  });

  it("covers the fixture's four audio inputs, one with an image (a positive control)", () => {
    assertEquals(FIXTURE.audioCases.length, 4);
    assertEquals(FIXTURE.audioCases.filter((entry) => entry.imageBlocks.length > 0).length, 1);
  });

  for (const entry of FIXTURE.audioCases) {
    it(`${entry.name}: the ids embed runs are upstream's exactly, each block's soft rows where upstream's are`, () => {
      const request = planEmbedRequest(rules, partsOf(entry), optionsOf(entry));
      assertEquals([...request.ids], fixtureIds(entry.name));
      assertEquals(request.ids.length, entry.tokens);
      assertEquals(blocksOf(request.audios), blocksOf(entry.audioBlocks));
      assertEquals(blocksOf(request.images), blocksOf(entry.imageBlocks));
    });
  }
});

/**
 * Settles the sha row of a vector and records the case; fails after the record on a mismatch
 * (throwing first would keep the measured sha out of the results).
 */
const settle = async (
  id: string,
  body: (
    measurements: Measurement[],
  ) => Promise<{ readonly vector: Float32Array<ArrayBuffer>; readonly note: string }>,
): Promise<void> => {
  let settlement: ReferenceSettlement | undefined;
  await runRecordedCase(results, { id }, async ({ measurements }) => {
    const { vector, note } = await body(measurements);
    const outcome = await settleOrObserve(references, results, {
      id,
      artifact: `${id.replaceAll("/", "_")}.safetensors`,
      bytes: f32ArtifactBytes("embedding", [1, vector.length], vector),
    });
    settlement = outcome.settlement;
    return { ...outcome.fields, note };
  });
  if (settlement?.check.status === "fail")
    fail(referenceMismatchMessage(id, settlement, references));
};

/** The 30 s audio's vector on the uncapped device, per preset (check 5 compares with it). */
const fullLengthVectors = new Map<string, Float32Array<ArrayBuffer>>();

const buildPipeline = async (
  gpu: GpuContext,
  quant: string,
  fetched?: Set<string>,
): Promise<embeddinggemma.Pipeline> =>
  await embeddinggemma.Pipeline.fromPretrained(mirrorDirectory(MIRROR_ROOT), {
    gpu,
    quant,
    vision: true,
    audio: true,
    ...(fetched === undefined
      ? {}
      : {
          onProgress: ({ phase, path }) => {
            if (phase === "complete") fetched.add(path);
          },
        }),
  });

/**
 * The files a pipeline built with `roles` should fetch for a preset: every part of each role's
 * container of the preset's own name (check 1 holds each preset to that storage) and the tokenizer.
 */
const presetFiles = async (quant: string, roles: readonly string[]): Promise<string[]> => {
  const manifest = parseManifest(await readTextFile(new URL("karume.json", MIRROR_ROOT)));
  const entry = manifest.models["text"];
  assert(entry !== undefined, "the mirror has no model 'text'");
  const parts = roles.flatMap((role) => {
    const container = entry.weights[role]?.[quant]?.container;
    assert(container !== undefined, `the mirror has no ${quant} container of '${role}'`);
    return container.parts.map(({ path }) => path);
  });
  return [...parts, ...Object.values(entry.assets).map(({ path }) => path)].sort();
};

for (const preset of PRESETS) {
  describe.skipIf(!RUNNABLE)(
    `EmbeddingGemma 2 audio input over the mirror, preset ${preset.quant} (real GPU)`,
    () => {
      let gpu: GpuContext;
      let pipeline: embeddinggemma.Pipeline;
      /** The paths whose fetch completed while the pipeline was built. */
      const fetched = new Set<string>();
      beforeAll(async () => {
        gpu = verifyRunningGpu(await acquireTestGpu());
        const start = performance.now();
        pipeline = await buildPipeline(gpu, preset.quant, fetched);
        console.log(
          `[embeddinggemma2-audio] ${preset.quant}: fromPretrained with both towers ${milliseconds(start)} ms`,
        );
      });
      afterAll(async () => {
        if (gpu === undefined) return;
        try {
          await pipeline?.dispose();
        } finally {
          gpu.destroy();
        }
      });

      it(`fetched the ${preset.quant} text graph, the ${preset.quant} towers and the tokenizer, nothing else`, async () => {
        assertEquals(
          [...fetched].sort(),
          await presetFiles(preset.quant, ["model", "vision", "audio"]),
        );
      });

      for (const entry of FIXTURE.audioCases) {
        it(`${entry.name} (T=${entry.tokens}): upstream's ids, a unit vector near upstream's, its sha256 row`, async () => {
          await settle(audioCaseId(preset.quant, entry.name), async (measurements) => {
            const parts = partsOf(entry);
            assertEquals([...pipeline.tokenize(parts, optionsOf(entry))], fixtureIds(entry.name));
            const start = performance.now();
            const vector = await pipeline.embed(parts, optionsOf(entry));
            const embedMs = milliseconds(start);
            const again = await pipeline.embed(parts, optionsOf(entry));
            assertEquals([...again], [...vector], `${entry.name}: a repeated call differs`);
            assertEquals(vector.length, 768);
            assert(unitNormError(vector) < 1e-5, `|norm − 1| = ${unitNormError(vector)}`);
            const upstream = fixtureVector(entry.name);
            const tolerance =
              preset.check.kind === "tolerance" ? preset.check.tolerance : { atol: 1, rtol: 0 };
            const report = compareTensors(
              { dtype: "f32", data: vector },
              { dtype: "f32", data: upstream },
              tolerance,
            );
            const similarity = cosine(vector, upstream);
            measurements.push({
              output: "embedding",
              maxAbs: report.maxAbsError,
              maxRel: report.maxRelError,
              tolerance,
              stage: "karume",
            });
            const note =
              `T=${entry.tokens}: maxAbs ${report.maxAbsError.toExponential(3)}, maxRel` +
              ` ${report.maxRelError.toExponential(3)}, cosine ${similarity.toFixed(12)}` +
              ` (torch composition maxAbs ${entry.composition.maxAbs.toExponential(3)}), first embed` +
              ` ${embedMs} ms`;
            console.log(`[embeddinggemma2-audio] ${preset.quant} ${entry.name} ${note}`);
            if (preset.check.kind === "tolerance") {
              assert(report.pass, `${preset.quant} ${entry.name}: ${formatAllclose(report)}`);
            } else {
              assert(
                similarity >= preset.check.floor,
                `${preset.quant} ${entry.name}: cosine ${similarity} to upstream's vector, under ${preset.check.floor}`,
              );
            }
            return { vector, note };
          });
        });
      }

      it("a 30 s audio (750 soft tokens, the longest): a unit vector, its sha256 row", async () => {
        await settle(audioCaseId(preset.quant, "full-length"), async () => {
          const parts = [{ type: "audio", audio: FULL_LENGTH_AUDIO }] as const;
          assertEquals(pipeline.tokenize(parts).length, FULL_LENGTH_TOKENS + 4);
          const start = performance.now();
          const vector = await pipeline.embed(parts);
          const note = `T=${FULL_LENGTH_TOKENS + 4}: embed ${milliseconds(start)} ms`;
          console.log(`[embeddinggemma2-audio] ${preset.quant} full-length ${note}`);
          assert(unitNormError(vector) < 1e-5, `|norm − 1| = ${unitNormError(vector)}`);
          fullLengthVectors.set(preset.quant, vector);
          return { vector, note };
        });
      });

      it("a text call on the audio pipeline has the text pipeline's sha256 row (query-en)", async () => {
        await settle(textCaseId(preset.quant), async () => {
          const vector = await pipeline.embed(
            QUERY_EN.text,
            QUERY_EN.prompt === undefined ? {} : { prompt: QUERY_EN.prompt },
          );
          return { vector, note: "a text through the pipeline built with audio: true" };
        });
      });
    },
  );
}

/** A stand-in tower Session: each run takes the next of `answers`, checked against the rows sent. */
const goldenTower = (
  answers: Float32Array<ArrayBuffer>[],
  output: string,
  width: number,
  input: string,
  rowsPerToken: number,
): EmbeddingSession => ({
  run: async (inputs: RunInputs) => {
    const soft = answers.shift();
    assert(soft !== undefined, `a ${input} tower run with no golden soft tokens queued`);
    const tensor = inputs[input];
    const rows = tensor !== undefined && "shape" in tensor ? tensor.shape[1] : -1;
    const tokens = soft.length / width;
    assertEquals(rows, rowsPerToken * tokens, `the ${input} rows for the queued block`);
    return { [output]: { dtype: "f32", shape: [1, tokens, width], data: soft } };
  },
  diagnostics: () => {
    throw new Error("the stand-in tower has no diagnostics");
  },
  dispose: async () => {},
});

describe.skipIf(!RUNNABLE || !GOLDEN_SOFT_TOKENS)(
  `EmbeddingGemma 2 audio input: the bf16 text graph fed upstream's soft tokens (real GPU)${SERIES_CHECKS_OPT_IN}`,
  () => {
    let gpu: GpuContext;
    let embedder: Embedder;
    const audioAnswers: Float32Array<ArrayBuffer>[] = [];
    const imageAnswers: Float32Array<ArrayBuffer>[] = [];
    beforeAll(async () => {
      gpu = verifyRunningGpu(await acquireTestGpu());
      const manifest = parseManifest(await readTextFile(new URL("karume.json", MIRROR_ROOT)));
      const entry = manifest.models["text"];
      assert(entry !== undefined, "the mirror has no model 'text'");
      // The preset declares no session knob, so its Session takes the family's settings alone.
      assertEquals(entry.quants["bf16"]?.session, {}, "the bf16 preset's session");
      const config = parseEmbeddingGemmaPipelineConfig(entry.pipelineConfig);
      const tokenizer = new Tokenizer(
        parseGemmaTokenizerAsset(await readMirrorAsset(MIRROR_ROOT, "tokenizer")),
      );
      const model = await openMirrorComponent(MIRROR_ROOT, "model", { quant: "bf16" });
      const audio = await openMirrorComponent(MIRROR_ROOT, "audio", { quant: "bf16" });
      const vision = await openMirrorComponent(MIRROR_ROOT, "vision", { quant: "bf16" });
      assert(config.audio !== undefined && config.vision !== undefined, "a tower is not declared");
      const audioContract = embeddingGemmaAudioContract(audio, config.audio);
      const visionContract = embeddingGemmaVisionContract(vision, config.vision);
      embedder = await openEmbedder({
        gpu,
        assertGpu: () => {},
        model,
        audioTower: {
          graph: {
            createSession: async () =>
              goldenTower(audioAnswers, audioContract.output, audioContract.width, "features", 4),
          },
          contract: audioContract,
        },
        tower: {
          graph: {
            createSession: async () =>
              goldenTower(imageAnswers, visionContract.output, visionContract.width, "pixels", 9),
          },
          contract: visionContract,
        },
        sessionOptions: {},
        contract: embeddingGemmaGraphContract(model, config),
        rules: embedTextRules(tokenizer, config, { images: true, audio: true }),
      });
    });
    afterAll(async () => {
      if (gpu === undefined) return;
      try {
        await embedder?.dispose();
      } finally {
        gpu.destroy();
      }
    });

    for (const entry of FIXTURE.audioCases) {
      it(`${entry.name} (T=${entry.tokens}): upstream's vector within ${SOFT_ROW_TOLERANCE.atol}`, async () => {
        await runRecordedCase(
          results,
          { id: `embeddinggemma-2/bf16/audio-golden-soft/${entry.name}` },
          async ({ measurements }) => {
            for (const { audio } of entry.audioBlocks) {
              audioAnswers.push(await goldenSoftTokens(goldenIo(AUDIO_SERIES, audio)));
            }
            for (const { image } of entry.imageBlocks) {
              imageAnswers.push(await goldenSoftTokens(goldenIo(VISION_SERIES, image)));
            }
            const vector = await embedder.embed(partsOf(entry), optionsOf(entry));
            assertEquals(
              [audioAnswers.length, imageAnswers.length],
              [0, 0],
              "a queued golden was not taken",
            );
            const upstream = fixtureVector(entry.name);
            const report = compareTensors(
              { dtype: "f32", data: vector },
              { dtype: "f32", data: upstream },
              SOFT_ROW_TOLERANCE,
            );
            measurements.push({
              output: "embedding",
              maxAbs: report.maxAbsError,
              maxRel: report.maxRelError,
              tolerance: SOFT_ROW_TOLERANCE,
              stage: "karume",
            });
            const note =
              `T=${entry.tokens}: the text graph fed upstream's soft tokens, maxAbs` +
              ` ${report.maxAbsError.toExponential(3)}, cosine ${cosine(vector, upstream).toFixed(12)}`;
            console.log(`[embeddinggemma2-audio] bf16 ${entry.name} ${note}`);
            assert(report.pass, `${entry.name}: ${formatAllclose(report)}`);
            return { status: "pass", note };
          },
        );
      });
    }
  },
);

/**
 * The 30 s audio's vector on an uncapped device: the one the uncapped block recorded, or — when that
 * block did not run first (a filtered run) — one taken now on a fresh uncapped device.
 */
const uncappedFullLengthVector = async (quant: string): Promise<Float32Array<ArrayBuffer>> => {
  const recorded = fullLengthVectors.get(quant);
  if (recorded !== undefined) return recorded;
  const gpu = verifyRunningGpu(await acquireTestGpu());
  try {
    const pipeline = await buildPipeline(gpu, quant);
    try {
      const vector = await pipeline.embed([{ type: "audio", audio: FULL_LENGTH_AUDIO }]);
      fullLengthVectors.set(quant, vector);
      return vector;
    } finally {
      await pipeline.dispose();
    }
  } finally {
    gpu.destroy();
  }
};

for (const preset of PRESETS) {
  const mebibytes = preset.cappedBindingBytes / 2 ** 20;
  describe.skipIf(!RUNNABLE)(
    `EmbeddingGemma 2 audio input, preset ${preset.quant}, on a device capped at ${mebibytes} MiB (real GPU)`,
    () => {
      let gpu: GpuContext;
      let pipeline: embeddinggemma.Pipeline;
      beforeAll(async () => {
        gpu = await acquireTestGpu({
          [LIMIT_CAPS]: {
            maxStorageBufferBindingSize: preset.cappedBindingBytes,
            maxBufferSize: preset.cappedBindingBytes,
          },
        });
        pipeline = await buildPipeline(gpu, preset.quant);
      });
      afterAll(async () => {
        if (gpu === undefined) return;
        try {
          await pipeline?.dispose();
        } finally {
          gpu.destroy();
        }
      });

      it("embeds a 30 s audio (750 soft tokens) to the uncapped device's bits, within the preset's limits", async () => {
        assertEquals(
          [gpu.limits.maxStorageBufferBindingSize, gpu.limits.maxBufferSize],
          [preset.cappedBindingBytes, preset.cappedBindingBytes],
          "the device's limits are not narrowed (the check would see nothing)",
        );
        const vector = await pipeline.embed([{ type: "audio", audio: FULL_LENGTH_AUDIO }]);
        assertEquals([...vector], [...(await uncappedFullLengthVector(preset.quant))]);
      });
    },
  );
}

if (RUNNABLE) references.warnMissing(CASE_IDS);
registerReferenceGate(references, { runnable: RUNNABLE, caseIds: CASE_IDS });
