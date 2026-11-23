/**
 * Audio input of the EmbeddingGemma family over EmbeddingGemma 2, on the host only — no GPU, no
 * real weights (the real-GPU run over the local distribution, and the token ids of the four
 * fixture inputs against upstream's with the mirror's tokenizer, are
 * `packages/e2e/tests/e2e_embeddinggemma2_audio_pipeline_test.ts`).
 *
 * ① `pipelineConfig.audio`: the distribution's shape and its tie to the audio placeholder.
 * ② The front end (`AudioFrontEnd`): the frame and soft-token counts of the fixture's waveforms and
 *    of the edge lengths (every residue of the frame count mod 4, the one-frame floor, the 30 s
 *    maximum); the frames of the fixture's three waveforms against upstream's feature extractor's
 *    (within a bound per waveform; their sha256 pinned); the padding frames and the mask; the
 *    settings against the snapshot's `preprocessor_config.json` (when present); and, with
 *    `KARUME_SERIES=1`, the layout against the audio series' io inputs.
 * ③ The sequence of an input holding an audio (`planEmbedRequest` / `tokenizeEmbedInput`): the
 *    audio block, beside text and an image, the prompt before it, the limit counting its soft
 *    tokens, the placeholder refusal around it, the ids of three fixture inputs equal to
 *    upstream's (the fourth's text is outside this tokenizer subset — the e2e pins all four), and
 *    every refusal of an audio part.
 * ④ The tower graph's contract (`embeddingGemmaAudioContract`).
 * ⑤ The run over stand-in Sessions (`Embedder`): one tower run per audio, its output at the audio's
 *    soft rows with the mask set there and nowhere else, an image and an audio in sequence order, a
 *    text call as without the tower, the samples read at issue, cancellation between the runs,
 *    dispose of every Session; and their opening (`openEmbedder`).
 * ⑥ `Pipeline.fromAssets` with `audio`: the admission's refusals, before any GPU.
 */

import { isNotFound, readFile, readTextFile, statSync } from "@karume/test-support/fs";
import { GpuContext, readAdapterInfo } from "../../runtime/src/gpu/device.ts";
import { describe, it } from "vite-plus/test";
import { assert, assertEquals, assertRejects, assertStrictEquals, assertThrows } from "@std/assert";
import { parseManifest } from "@karume/hub";
import {
  parseSafetensors,
  type RunInputs,
  type RunOutputs,
  type SafetensorsFile,
  type SessionOptions,
  type Tensor,
} from "@karume/runtime";
import { ModelInputError } from "../src/errors.ts";
import { createBpeModel } from "../src/text/bpe.ts";
import type { GemmaTokenizerAssets } from "../src/gemma/text/asset.ts";
import { Tokenizer } from "../src/gemma/text/tokenizer.ts";
import type { GraphOwner } from "../src/hub/components.ts";
import type { Rgb8Image } from "../src/image/preprocess.ts";
import { parseEmbeddingGemmaPipelineConfig } from "../src/embeddinggemma/config.ts";
import {
  AUDIO_FEATURE_EXTRACTOR,
  AudioFrontEnd,
  audioSoftTokens,
  audioValidFrames,
  type EmbeddingGemmaAudioContract,
  embeddingGemmaAudioContract,
  type MonoAudio,
} from "../src/embeddinggemma/audio.ts";
import {
  type EmbedContentPart,
  embedTextRules,
  planEmbedRequest,
  tokenizeEmbedInput,
} from "../src/embeddinggemma/input.ts";
import {
  Embedder,
  type EmbeddingGemmaGraphContract,
  type EmbeddingGraph,
  type EmbeddingSession,
  openEmbedder,
  Pipeline,
} from "../src/embeddinggemma/pipeline.ts";
import type { EmbeddingGemmaVisionContract } from "../src/embeddinggemma/vision.ts";
import type { ModelInput } from "../../runtime/tests/helpers/container-write.ts";
import { fakeDevice, fakeGpuContext } from "../../runtime/tests/helpers/fake-gpu.ts";
import { SERIES_CHECKS, SERIES_CHECKS_OPT_IN } from "./helpers/series-checks.ts";
import {
  parseIrDeclarationValue,
  partAssets,
  sha256Hex,
  writeContainer,
} from "./helpers/container-fixture.ts";
import { fileRef } from "./helpers/container-loading-fixture.ts";

// ---- Fixtures --------------------------------------------------------------------------------

/** The parts of the parity fixture the tokenizer is built from (`embeddinggemma2/tokenizer.py`). */
type ParityFixture = {
  readonly spec: { readonly postProcessor: "none" | "bos-eos" | "bos" };
  readonly asset: {
    readonly vocab: [number, string][];
    readonly merges: [number, number, number][];
    readonly byteIds: number[];
    readonly addedTokens: [string, number][];
    readonly specialIds: number[];
    readonly unkId: number;
    readonly bosId: number;
    readonly eosId: number;
  };
  readonly encode: readonly {
    readonly name: string;
    readonly text: string;
    readonly ids: number[];
    readonly idsWithSpecials: number[];
  }[];
};

const PARITY = JSON.parse(
  await readTextFile(new URL("fixtures/gemma-text/embeddinggemma-2-parity.json", import.meta.url)),
) as ParityFixture;

/** The tokenizer of the fixture's subset (its texts and every added token encode as upstream's). */
const ASSETS: GemmaTokenizerAssets = {
  spec: { postProcessor: PARITY.spec.postProcessor },
  model: createBpeModel({
    vocab: PARITY.asset.vocab.map(([id, token]) => [id, token] as const),
    merges: PARITY.asset.merges.map(([left, right, rank]) => [left, right, rank] as const),
    byteIds: PARITY.asset.byteIds,
  }),
  addedTokens: new Map(PARITY.asset.addedTokens),
  specialIds: new Set(PARITY.asset.specialIds),
  unkId: PARITY.asset.unkId,
  bosId: PARITY.asset.bosId,
  eosId: PARITY.asset.eosId,
};
const TOKENIZER = new Tokenizer(ASSETS);
const BOS = ASSETS.bosId;
const EOS = ASSETS.eosId;

const parityCase = (name: string): ParityFixture["encode"][number] => {
  const found = PARITY.encode.find((row) => row.name === name);
  if (found === undefined) throw new Error(`the parity fixture has no case '${name}'`);
  return found;
};

/** The audio ids of EmbeddingGemma 2 (`config.json` at the pin: `boa_token_id`, `audio_token_id`, `eoa_token_index`). */
const BEGIN = 256000;
const SOFT = 258881;
const END = 258883;

/** `pipelineConfig.audio` as the dist recipe writes it. */
const AUDIO_JSON = {
  audioTokenId: SOFT,
  beginAudioTokenId: BEGIN,
  endAudioTokenId: END,
  hiddenSize: 512,
  maxSoftTokens: 750,
  sampleRate: 16000,
};

/** `pipelineConfig.vision` as the dist recipe writes it (an image and an audio may share an input). */
const VISION_JSON = {
  softTokenBudget: 280,
  imageTokenId: 258880,
  beginImageTokenId: 255999,
  endImageTokenId: 258882,
  hiddenSize: 512,
};

/** The distribution's `pipelineConfig` (two prompts are enough here). */
const CONFIG_JSON = {
  prompts: { document: "title: none | text: ", query: "task: search result | query: " },
  dimensions: [768, 512, 256, 128],
  maxTokens: 8192,
  placeholderTokens: { image: 258880, audio: SOFT, video: 258884 },
  vision: VISION_JSON,
  audio: AUDIO_JSON,
};

const CONFIG = parseEmbeddingGemmaPipelineConfig(CONFIG_JSON);
/** The rules of a pipeline with the audio tower, with both towers, and without either. */
const AUDIO_RULES = embedTextRules(TOKENIZER, CONFIG, { audio: true });
const BOTH_RULES = embedTextRules(TOKENIZER, CONFIG, { images: true, audio: true });
const TEXT_RULES = embedTextRules(TOKENIZER, CONFIG);

/** The whole-input fixture (`tools/export-recipes/embeddinggemma2/multimodal_reference.py`). */
type MultimodalFixture = {
  readonly audioTokens: { readonly begin: number; readonly soft: number; readonly end: number };
  readonly images: readonly {
    readonly name: string;
    readonly width: number;
    readonly height: number;
  }[];
  readonly audios: readonly {
    readonly name: string;
    readonly samples: number;
    readonly sampleRate: number;
    readonly extractorFrames: number;
    readonly validFrames: number;
    readonly softTokens: number;
  }[];
  readonly audioCases: readonly {
    readonly name: string;
    readonly parts: readonly (
      | { readonly type: "text"; readonly text: string }
      | { readonly type: "image"; readonly image: string }
      | { readonly type: "audio"; readonly audio: string }
    )[];
    readonly prompt?: string;
    readonly tokens: number;
    readonly imageBlocks: readonly { readonly start: number; readonly softTokens: number }[];
    readonly audioBlocks: readonly { readonly start: number; readonly softTokens: number }[];
  }[];
};

const MULTIMODAL = JSON.parse(
  await readTextFile(
    new URL("fixtures/embeddinggemma2/multimodal-reference.json", import.meta.url),
  ),
) as MultimodalFixture;

const parseFile = (bytes: Uint8Array<ArrayBuffer>): SafetensorsFile =>
  parseSafetensors(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));

const MULTIMODAL_TENSORS = parseFile(
  await readFile(
    new URL("fixtures/embeddinggemma2/multimodal-reference.safetensors", import.meta.url),
  ),
);

/** A copy of one tensor's raw bytes (fails loudly when absent). */
const tensorBytes = (file: SafetensorsFile, key: string): Uint8Array<ArrayBuffer> => {
  const view = file.tensors.get(key);
  if (view === undefined) throw new Error(`the safetensors file has no tensor '${key}'`);
  return new Uint8Array(file.buffer, view.byteOffset, view.byteLength).slice();
};
const f32Of = (bytes: Uint8Array<ArrayBuffer>): Float32Array<ArrayBuffer> =>
  new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);

/** One fixture waveform (16 kHz mono f32 — what upstream's processor read). */
const fixtureAudio = (name: string): MonoAudio => ({
  data: f32Of(tensorBytes(MULTIMODAL_TENSORS, `audio.${name}`)),
  sampleRate: 16000,
});
/** Upstream's valid log-mel frames of a fixture waveform (`[L, 128]`). */
const upstreamFrames = (name: string): Float32Array<ArrayBuffer> =>
  f32Of(tensorBytes(MULTIMODAL_TENSORS, `audio.${name}.features`));

/** One fixture image as RGB8. */
const fixtureImage = (name: string): Rgb8Image => {
  const entry = MULTIMODAL.images.find((image) => image.name === name);
  if (entry === undefined) throw new Error(`the fixture has no image '${name}'`);
  return {
    data: tensorBytes(MULTIMODAL_TENSORS, `image.${name}`),
    width: entry.width,
    height: entry.height,
  };
};

/** The ids upstream ran for a fixture case (`<case>.input_ids`, I32). */
const fixtureIds = (name: string): number[] => {
  const bytes = tensorBytes(MULTIMODAL_TENSORS, `${name}.input_ids`);
  return [...new Int32Array(bytes.buffer)];
};

const bytesOf = (view: ArrayBufferView<ArrayBuffer>): Uint8Array<ArrayBuffer> =>
  new Uint8Array(view.buffer, view.byteOffset, view.byteLength);

/** Asserts that `run` is refused with a `ModelInputError` whose message holds every fragment. */
const assertRefused = (run: () => unknown, ...fragments: readonly string[]): ModelInputError => {
  const error = assertThrows(run, ModelInputError);
  for (const fragment of fragments) {
    assert(error.message.includes(fragment), `"${fragment}" is not in: ${error.message}`);
  }
  return error;
};

/** A 16 kHz audio of `samples` samples of a quiet tone (finite, nonzero). */
const tone = (samples: number, frequency = 440): MonoAudio => {
  const data = new Float32Array(samples);
  for (let index = 0; index < samples; index += 1) {
    data[index] = 0.25 * Math.sin((2 * Math.PI * frequency * index) / 16000);
  }
  return { data, sampleRate: 16000 };
};

// ---- ① pipelineConfig.audio ---------------------------------------------------------------------

/** `pipelineConfig.audio` without its sample rate. */
const { sampleRate: _rate, ...WITHOUT_RATE } = AUDIO_JSON;

describe("pipelineConfig.audio", () => {
  it("reads the distribution's declaration", () => {
    assertEquals(CONFIG.audio, AUDIO_JSON);
  });

  it("is absent from a config that does not declare it, and vision stays independent of it", () => {
    const { audio: _, ...withoutAudio } = CONFIG_JSON;
    const config = parseEmbeddingGemmaPipelineConfig(withoutAudio);
    assertEquals(config.audio, undefined);
    assertEquals(config.vision, VISION_JSON);
    const { vision: _vision, ...withoutVision } = CONFIG_JSON;
    assertEquals(parseEmbeddingGemmaPipelineConfig(withoutVision).audio, AUDIO_JSON);
  });

  it("refuses a soft token that is not the audio placeholder, or no audio placeholder at all", () => {
    assertThrows(
      () =>
        parseEmbeddingGemmaPipelineConfig({
          ...CONFIG_JSON,
          audio: { ...AUDIO_JSON, audioTokenId: 258884 },
        }),
      Error,
      "pipelineConfig.audio.audioTokenId 258884 is not placeholderTokens.audio (258881)",
    );
    const { vision: _, ...withoutVision } = CONFIG_JSON;
    assertThrows(
      () =>
        parseEmbeddingGemmaPipelineConfig({
          ...withoutVision,
          placeholderTokens: { image: 258880 },
        }),
      Error,
      "is not placeholderTokens.audio (not declared)",
    );
  });

  it("refuses an unknown or missing key, another sample rate and a bound past the longest audio", () => {
    for (const [audio, message] of [
      [{ ...AUDIO_JSON, maxSamples: 480000 }, "pipelineConfig.audio: unknown key 'maxSamples'"],
      [{ ...AUDIO_JSON, sampleRate: 24000 }, "pipelineConfig.audio.sampleRate: not 16000"],
      [
        { ...AUDIO_JSON, maxSoftTokens: 751 },
        "pipelineConfig.audio.maxSoftTokens: not an integer in 1..750",
      ],
      [
        { ...AUDIO_JSON, maxSoftTokens: 0 },
        "pipelineConfig.audio.maxSoftTokens: not an integer in 1..750",
      ],
      [{ ...AUDIO_JSON, hiddenSize: 0 }, "pipelineConfig.audio.hiddenSize: not a positive integer"],
      [{ ...AUDIO_JSON, endAudioTokenId: BEGIN }, "the three audio token ids are not distinct"],
      [WITHOUT_RATE, "pipelineConfig.audio.sampleRate: missing"],
      ["audio", "pipelineConfig.audio: not an object"],
    ] as const) {
      assertThrows(
        () => parseEmbeddingGemmaPipelineConfig({ ...CONFIG_JSON, audio }),
        Error,
        message,
      );
    }
  });
});

// ---- ② the front end ----------------------------------------------------------------------------

/**
 * The bound of the host frames against upstream's feature extractor's, per fixture waveform (max
 * abs over the valid frames; log-mel values of −6.9 to about 5).
 *
 * Measured on 2026-11-22: sweep 5.12e-4, click 4.77e-7, edge 2.57e-4. The host frames are every
 * one of them `fround` of upstream's own code run in f64 (its window times f64 samples, a complex128
 * rfft — measured against all 38,528 values of the three waveforms), so these differences are
 * upstream's f32 rounding (numpy's complex64 rfft): on `click` (mostly silence) it is 1 to 2 ulp of
 * the values, on the tonal `sweep` and `edge` it reaches 5e-4 in the near-silent mel bins of a
 * frame, where `ln(mel + 0.001)` turns an energy error of about 6e-7 into that much. Each bound is
 * three times its waveform's measure. The breakages it guards against are far above: a hop of 128
 * or a power spectrum instead of the magnitude moves the frames by more than 1 (the fault
 * injections of the commit that added this file).
 */
const FRAME_TOLERANCE: Readonly<Record<string, number>> = {
  sweep: 1.5e-3,
  click: 1.5e-6,
  edge: 8e-4,
};

/**
 * The sha256 of the host frames `[4n, 128]` f32 (padding frames included) of the fixture's
 * waveforms, taken on 2026-11-22 — the f32 rounding of upstream's code run in f64 (above).
 */
const FRAME_DIGESTS: Readonly<Record<string, string>> = {
  sweep: "8d736e70473039940b9359b1a6154b2a3e69d381efcbbfcc0067daf1a461ae96",
  click: "5ecf614a27c9256f54bd2284078fc94b737698dfdf5fe3b1911bbc59dc428392",
  edge: "ca56b9d058abf7332322890b8445bde0b58192e05c102681125245bf80936b4f",
};

const FRONT_END = new AudioFrontEnd();

/** The io file of a waveform in the f32 audio series (outside the repository). */
const seriesIo = (name: string): URL =>
  new URL(`../../../outputs/series/embeddinggemma-2-audio/io.${name}.safetensors`, import.meta.url);

/** The pinned snapshot's feature extractor settings (outside the repository). */
const PREPROCESSOR = new URL(
  "../../../inputs/embeddinggemma2/embeddinggemma-2/preprocessor_config.json",
  import.meta.url,
);

/** Whether the file exists (anything but NotFound propagates). */
const exists = (url: URL): boolean => {
  try {
    return statSync(url).isFile;
  } catch (cause) {
    if (isNotFound(cause)) return false;
    throw cause;
  }
};

describe("audioValidFrames / audioSoftTokens: the frame and soft-token counts", () => {
  it("counts a frame only once its window's last sample exists (L = max(0, ceil(s / 160) − 1))", () => {
    for (const [samples, frames] of [
      [0, 0],
      [1, 0],
      [160, 0],
      [161, 1],
      [320, 1],
      [321, 2],
      [1000, 6],
      [16000, 99],
      [480000, 2999],
      [480160, 3000],
      [480161, 3001],
    ] as const) {
      assertEquals(audioValidFrames(samples), frames, `${samples} samples`);
    }
  });

  it("gives every residue of the frame count mod 4 its soft tokens (n = ceil(L / 4))", () => {
    // 16,000 → 99 (≡ 3), 16,100 → 100 (≡ 0), 16,240 → 101 (≡ 1), 16,400 → 102 (≡ 2).
    for (const [samples, frames, tokens] of [
      [16000, 99, 25],
      [16100, 100, 25],
      [16240, 101, 26],
      [16400, 102, 26],
      [480000, 2999, 750],
    ] as const) {
      assertEquals(audioValidFrames(samples), frames, `${samples} samples`);
      assertEquals(audioSoftTokens(frames), tokens, `${frames} frames`);
    }
  });

  it("covers the fixture's three waveforms, one of them L ≡ 2 mod 4 (a positive control)", () => {
    assertEquals(
      MULTIMODAL.audios.map(({ name }) => name),
      ["sweep", "click", "edge"],
    );
    assertEquals(
      MULTIMODAL.audios.map(({ validFrames }) => validFrames % 4),
      [3, 0, 2],
    );
  });

  for (const entry of MULTIMODAL.audios) {
    it(`${entry.name} (${entry.samples} samples): upstream's valid frames and soft tokens`, () => {
      assertEquals(audioValidFrames(entry.samples), entry.validFrames);
      assertEquals(audioSoftTokens(entry.validFrames), entry.softTokens);
      assertEquals(upstreamFrames(entry.name).length, entry.validFrames * 128);
    });
  }
});

describe("AudioFrontEnd: the fixture's waveforms as the tower reads them", () => {
  for (const entry of MULTIMODAL.audios) {
    it(`${entry.name}: the valid frames within ${FRAME_TOLERANCE[entry.name]} of upstream's feature extractor's`, () => {
      const input = FRONT_END.towerInput(fixtureAudio(entry.name).data);
      const upstream = upstreamFrames(entry.name);
      let worst = 0;
      for (let index = 0; index < upstream.length; index += 1) {
        worst = Math.max(worst, Math.abs(input.features[index] - upstream[index]));
      }
      const bound = FRAME_TOLERANCE[entry.name] ?? 0;
      assert(worst <= bound, `${entry.name}: max abs ${worst} over ${bound}`);
      console.log(
        `[embeddinggemma2-audio] ${entry.name}: frames max abs ${worst.toExponential(3)}`,
      );
    });

    it(`${entry.name}: the frames padded to 4n with zeros, the mask 1 on the L valid ones, their sha256 pinned`, async () => {
      const input = FRONT_END.towerInput(fixtureAudio(entry.name).data);
      const frames = 4 * entry.softTokens;
      assertEquals([input.validFrames, input.softTokens], [entry.validFrames, entry.softTokens]);
      assertEquals(input.features.length, frames * 128);
      assert(input.features.subarray(entry.validFrames * 128).every((value) => value === 0));
      assertEquals(
        [...input.frameMask],
        Array.from({ length: frames }, (_, frame) => (frame < entry.validFrames ? 1 : 0)),
      );
      assertEquals(await sha256Hex(bytesOf(input.features)), FRAME_DIGESTS[entry.name]);
    });

    const io = seriesIo(entry.name);
    it.skipIf(!SERIES_CHECKS || !exists(io))(
      `${entry.name}: the mask is the series' io input's, the frames within its bound${SERIES_CHECKS_OPT_IN}`,
      async () => {
        const file = parseFile(await readFile(io));
        const input = FRONT_END.towerInput(fixtureAudio(entry.name).data);
        assertEquals(
          bytesOf(f32Of(tensorBytes(file, "input.frame_mask"))),
          bytesOf(input.frameMask),
          "the frame mask",
        );
        const features = f32Of(tensorBytes(file, "input.features"));
        assertEquals(features.length, input.features.length, "the frame count");
        let worst = 0;
        for (let index = 0; index < features.length; index += 1) {
          worst = Math.max(worst, Math.abs(input.features[index] - features[index]));
        }
        const bound = FRAME_TOLERANCE[entry.name] ?? 0;
        assert(worst <= bound, `${entry.name}: max abs ${worst} over ${bound}`);
      },
    );
  }

  it("reads silence as ln(0.001) in every bin, the zeros before the audio included", () => {
    const input = FRONT_END.towerInput(new Float32Array(16000));
    const floor = Math.fround(Math.log(1e-3));
    assert(input.features.subarray(0, 99 * 128).every((value) => value === floor));
  });

  it("reads the window's first samples as the 160 zeros before the audio (semicausal)", () => {
    // A click at sample 100 is in frame 0 (samples −160 … 159) and frame 1 (0 … 319), not in frame
    // 2 (160 … 479); without the 160 zeros it would be in frame 0 alone.
    const data = new Float32Array(1000);
    data[100] = 0.5;
    const input = FRONT_END.towerInput(data);
    const floor = Math.fround(Math.log(1e-3));
    const row = (frame: number): Float32Array =>
      input.features.subarray(frame * 128, (frame + 1) * 128);
    assert(
      row(0).some((value) => value !== floor),
      "frame 0 does not see the click",
    );
    assert(
      row(1).some((value) => value !== floor),
      "frame 1 does not see the click",
    );
    assert(
      row(2).every((value) => value === floor),
      "frame 2 sees the click",
    );
  });

  it("builds one audio's frames independently of the one before (the scratch is cleared per frame)", () => {
    const fresh = new AudioFrontEnd().towerInput(fixtureAudio("edge").data);
    FRONT_END.towerInput(fixtureAudio("sweep").data);
    assertEquals(
      bytesOf(FRONT_END.towerInput(fixtureAudio("edge").data).features),
      bytesOf(fresh.features),
    );
  });

  it.skipIf(!exists(PREPROCESSOR))(
    "implements the settings of the pinned snapshot's preprocessor_config.json (skipped without the snapshot)",
    async () => {
      const settings = JSON.parse(await readTextFile(PREPROCESSOR)) as Record<string, unknown>;
      for (const [key, value] of Object.entries(AUDIO_FEATURE_EXTRACTOR)) {
        assertEquals(settings[key], value, key);
      }
      // The settings the front end does not implement are off in the file.
      assertEquals(
        [
          settings["preemphasis"],
          settings["dither"],
          settings["input_scale_factor"],
          settings["per_bin_mean"],
          settings["per_bin_stddev"],
          settings["fft_overdrive"],
          settings["feature_extractor_type"],
        ],
        [0, 0, 1, null, null, false, "Gemma4AudioFeatureExtractor"],
      );
    },
  );
});

// ---- ③ the sequence -------------------------------------------------------------------------

const SWEEP = fixtureAudio("sweep");
const SWEEP_TOKENS = 25;
const EDGE = fixtureAudio("edge");
const EDGE_TOKENS = 26;
/** A small RGB8 square (it upscales to the budget: 16 × 16 blocks = 256 soft tokens). */
const SQUARE: Rgb8Image = { data: new Uint8Array(8 * 8 * 3).fill(128), width: 8, height: 8 };
const SQUARE_TOKENS = 256;

const block = (softTokens: number): number[] => [
  BEGIN,
  ...Array<number>(softTokens).fill(SOFT),
  END,
];
const imageBlock = (softTokens: number): number[] => [
  VISION_JSON.beginImageTokenId,
  ...Array<number>(softTokens).fill(VISION_JSON.imageTokenId),
  VISION_JSON.endImageTokenId,
];

const idsOf = (request: { readonly ids: Int32Array<ArrayBuffer> }): number[] => [...request.ids];
const blocksOf = (
  blocks: readonly { readonly start: number; readonly softTokens: number }[],
): { start: number; softTokens: number }[] =>
  blocks.map(({ start, softTokens }) => ({ start, softTokens }));

describe("planEmbedRequest with audio parts: the sequence", () => {
  it("writes an audio alone as <bos>, <|audio>, n soft tokens, <audio|>, <eos>", () => {
    const request = planEmbedRequest(AUDIO_RULES, [{ type: "audio", audio: SWEEP }], {});
    assertEquals(idsOf(request), [BOS, ...block(SWEEP_TOKENS), EOS]);
    assertEquals(blocksOf(request.audios), [{ start: 2, softTokens: SWEEP_TOKENS }]);
    assertEquals(request.images, []);
    assertStrictEquals(request.audios[0]?.audio.data, SWEEP.data);
  });

  it("tokenizes each run of adjacent text as one string, with nothing between it and an audio", () => {
    const { text } = parityCase("english-long");
    const half = Math.floor(text.length / 2);
    const tail = parityCase("english").text;
    const parts: readonly EmbedContentPart[] = [
      { type: "text", text: text.slice(0, half) },
      { type: "text", text: text.slice(half) },
      { type: "audio", audio: EDGE },
      { type: "text", text: tail },
    ];
    const run = TOKENIZER.encode(text);
    const request = planEmbedRequest(AUDIO_RULES, parts, {});
    assertEquals(idsOf(request), [
      BOS,
      ...run,
      ...block(EDGE_TOKENS),
      ...TOKENIZER.encode(tail),
      EOS,
    ]);
    assertEquals(request.audios[0]?.start, 1 + run.length + 1);
  });

  it("writes an image and an audio in the order given, each block where its soft rows start", () => {
    const imageFirst = planEmbedRequest(
      BOTH_RULES,
      [
        { type: "image", image: SQUARE },
        { type: "audio", audio: SWEEP },
      ],
      {},
    );
    assertEquals(idsOf(imageFirst), [
      BOS,
      ...imageBlock(SQUARE_TOKENS),
      ...block(SWEEP_TOKENS),
      EOS,
    ]);
    assertEquals(blocksOf(imageFirst.images), [{ start: 2, softTokens: SQUARE_TOKENS }]);
    assertEquals(blocksOf(imageFirst.audios), [
      { start: 2 + SQUARE_TOKENS + 2, softTokens: SWEEP_TOKENS },
    ]);
    const audioFirst = planEmbedRequest(
      BOTH_RULES,
      [
        { type: "audio", audio: SWEEP },
        { type: "image", image: SQUARE },
      ],
      {},
    );
    assertEquals(idsOf(audioFirst), [
      BOS,
      ...block(SWEEP_TOKENS),
      ...imageBlock(SQUARE_TOKENS),
      EOS,
    ]);
    assertEquals(audioFirst.images[0]?.start, 2 + SWEEP_TOKENS + 2);
  });

  it("gives the prefix a run of its own before an audio that comes first (its trailing space a token)", () => {
    const prefix = TOKENIZER.encode("task: search result | query: ");
    const request = planEmbedRequest(AUDIO_RULES, [{ type: "audio", audio: EDGE }], {
      prompt: "query",
    });
    assertEquals(idsOf(request), [BOS, ...prefix, ...block(EDGE_TOKENS), EOS]);
  });

  it("keeps a text-only input's ids as they were, with the audio tower or without it", () => {
    for (const name of ["english", "prompt-query", "japanese-long"]) {
      const { text, idsWithSpecials } = parityCase(name);
      for (const rules of [AUDIO_RULES, BOTH_RULES, TEXT_RULES]) {
        const request = planEmbedRequest(rules, text, {});
        assertEquals(idsOf(request), idsWithSpecials, name);
        assertEquals([request.images, request.audios], [[], []]);
      }
    }
  });

  it("gives tokenize the ids embed runs, the soft rows as the audio placeholder id", () => {
    const parts: readonly EmbedContentPart[] = [
      { type: "text", text: "a" },
      { type: "audio", audio: EDGE },
    ];
    assertEquals(
      [...tokenizeEmbedInput(AUDIO_RULES, parts, { prompt: "document" })],
      idsOf(planEmbedRequest(AUDIO_RULES, parts, { prompt: "document" })),
    );
  });

  it("still refuses a placeholder spelt in a text run, at its place in the whole sequence", () => {
    const parts: readonly EmbedContentPart[] = [
      { type: "audio", audio: SWEEP },
      { type: "text", text: "listen to <|audio|> here" },
    ];
    const at = 1 + SWEEP_TOKENS + 2 + TOKENIZER.encode("listen to ").length;
    assertRefused(
      () => planEmbedRequest(AUDIO_RULES, parts, {}),
      'the text holds the audio placeholder token "<|audio|>" (id 258881) at token',
      `at token ${at}`,
      "remove that spelling",
    );
  });

  it("counts every audio's soft tokens and markers toward the limit", () => {
    const parts: readonly EmbedContentPart[] = [{ type: "audio", audio: SWEEP }];
    const length = SWEEP_TOKENS + 4;
    const at = embedTextRules(
      TOKENIZER,
      parseEmbeddingGemmaPipelineConfig({ ...CONFIG_JSON, maxTokens: length }),
      { audio: true },
    );
    assertEquals(planEmbedRequest(at, parts, {}).ids.length, length);
    const under = embedTextRules(
      TOKENIZER,
      parseEmbeddingGemmaPipelineConfig({ ...CONFIG_JSON, maxTokens: length - 1 }),
      { audio: true },
    );
    assertRefused(
      () => planEmbedRequest(under, parts, {}),
      `the input is ${length} tokens (<bos>, <eos> and the soft tokens of the audio with their two` +
        ` markers each included), over this model's limit of ${length - 1}`,
      "pass fewer or shorter audios",
    );
  });

  it("names the images and the audios of an input over the limit", () => {
    const rules = embedTextRules(
      TOKENIZER,
      parseEmbeddingGemmaPipelineConfig({ ...CONFIG_JSON, maxTokens: 300 }),
      { images: true, audio: true },
    );
    assertRefused(
      () =>
        planEmbedRequest(
          rules,
          [
            { type: "image", image: SQUARE },
            { type: "audio", audio: SWEEP },
            { type: "audio", audio: EDGE },
          ],
          {},
        ),
      "(<bos>, <eos> and the soft tokens of the image and 2 audios with their two markers each included)",
      "pass fewer images or audios",
    );
  });
});

describe("planEmbedRequest with audio parts: the ids of upstream's processor", () => {
  /** The fixture's audio cases whose texts this tokenizer subset holds (the e2e checks all four). */
  const covered = MULTIMODAL.audioCases.filter((entry) => entry.name !== "text-then-audio");

  it("covers three of the fixture's four audio cases with this subset (a positive control)", () => {
    assertEquals(
      covered.map(({ name }) => name),
      ["audio-only", "query-audio", "image-then-audio"],
    );
    assertEquals(MULTIMODAL.audioTokens, { begin: BEGIN, soft: SOFT, end: END });
  });

  const partsOf = (entry: MultimodalFixture["audioCases"][number]): readonly EmbedContentPart[] =>
    entry.parts.map((part) => {
      if (part.type === "text") return part;
      if (part.type === "image") return { type: "image", image: fixtureImage(part.image) };
      return { type: "audio", audio: fixtureAudio(part.audio) };
    });

  for (const entry of covered) {
    it(`${entry.name}: the ids and each block's soft rows are upstream's exactly`, () => {
      const options = entry.prompt === undefined ? {} : { prompt: entry.prompt };
      const request = planEmbedRequest(BOTH_RULES, partsOf(entry), options);
      const expected = fixtureIds(entry.name);
      assertEquals(idsOf(request), expected);
      assertEquals(expected.length, entry.tokens);
      assertEquals([...tokenizeEmbedInput(BOTH_RULES, partsOf(entry), options)], expected);
      assertEquals(blocksOf(request.audios), blocksOf(entry.audioBlocks));
      assertEquals(blocksOf(request.images), blocksOf(entry.imageBlocks));
    });
  }

  it("text-then-audio: the audio block and <eos> are upstream's, right after the text's run", () => {
    const entry = MULTIMODAL.audioCases.find(({ name }) => name === "text-then-audio");
    assert(entry !== undefined, "the fixture has no case 'text-then-audio'");
    const request = planEmbedRequest(BOTH_RULES, partsOf(entry), {});
    const expected = fixtureIds(entry.name);
    const tail = SWEEP_TOKENS + 3;
    assertEquals(idsOf(request).slice(-tail), expected.slice(-tail));
    const text = entry.parts[0];
    assert(text?.type === "text", "the case does not open with a text");
    assertEquals(request.audios[0]?.start, 1 + TOKENIZER.encode(text.text).length + 1);
  });
});

describe("planEmbedRequest with audio parts: the refusals", () => {
  it("refuses an audio part without the audio tower, saying how to build a pipeline that takes it", () => {
    assertRefused(
      () => planEmbedRequest(TEXT_RULES, [{ type: "audio", audio: SWEEP }], {}),
      'content part 0 has type "audio" — this model accepts "text" parts',
      "audio parts need the audio tower — build the pipeline with { audio: true }",
    );
    const imageOnly = embedTextRules(TOKENIZER, CONFIG, { images: true });
    assertRefused(
      () => tokenizeEmbedInput(imageOnly, [{ type: "audio", audio: SWEEP }], {}),
      'this model accepts "text" / "image" parts; audio parts need the audio tower',
    );
  });

  it("says the distribution has no audio tower when its pipelineConfig declares none", () => {
    const { audio: _, ...withoutAudio } = CONFIG_JSON;
    const rules = embedTextRules(TOKENIZER, parseEmbeddingGemmaPipelineConfig(withoutAudio));
    assertRefused(
      () => planEmbedRequest(rules, [{ type: "audio", audio: SWEEP }], {}),
      'this model accepts "text" parts (this model\'s distribution ships no audio tower)',
    );
  });

  it("names every accepted type with both towers, and refuses other types", () => {
    assertRefused(
      () => Reflect.apply(planEmbedRequest, undefined, [BOTH_RULES, [{ type: "video" }], {}]),
      'content part 0 has type "video" — this model accepts "text" / "image" / "audio" parts',
    );
    assertRefused(
      () => planEmbedRequest(AUDIO_RULES, [{ type: "image", image: SQUARE }], {}),
      'this model accepts "text" / "audio" parts; image parts need the vision tower',
    );
  });

  it("refuses audio at another sample rate, telling the caller to resample", () => {
    assertRefused(
      () =>
        planEmbedRequest(
          AUDIO_RULES,
          [{ type: "audio", audio: { data: new Float32Array(44100), sampleRate: 44100 } }],
          {},
        ),
      "content part 0 is audio at 44100 Hz — this model takes 16000 Hz mono samples only, and" +
        " nothing is resampled; resample it to 16000 Hz first",
    );
  });

  it("refuses an audio over 30 s, never truncating it, and takes one of exactly 30 s", () => {
    assertRefused(
      () => planEmbedRequest(AUDIO_RULES, [{ type: "audio", audio: tone(480001) }], {}),
      "content part 0 is 480001 samples (30.00 s), over the 480000 (30 s) this model reads —" +
        " nothing is truncated; shorten the audio or split it",
    );
    const request = planEmbedRequest(AUDIO_RULES, [{ type: "audio", audio: tone(480000) }], {});
    assertEquals(blocksOf(request.audios), [{ start: 2, softTokens: 750 }]);
  });

  it("refuses an audio too short for one frame (at most 160 samples), and takes 161", () => {
    for (const samples of [0, 1, 160]) {
      assertRefused(
        () => planEmbedRequest(AUDIO_RULES, [{ type: "audio", audio: tone(samples) }], {}),
        `content part 0 is ${samples} samples, too short for one mel frame`,
      );
    }
    const request = planEmbedRequest(AUDIO_RULES, [{ type: "audio", audio: tone(161) }], {});
    assertEquals(blocksOf(request.audios), [{ start: 2, softTokens: 1 }]);
  });

  it("refuses a non-finite sample, naming it", () => {
    for (const [value, shown] of [
      [Number.NaN, "NaN"],
      [Number.POSITIVE_INFINITY, "Infinity"],
    ] as const) {
      const audio = tone(1000);
      audio.data[700] = value;
      assertRefused(
        () => planEmbedRequest(AUDIO_RULES, [{ type: "audio", audio }], {}),
        `content part 0 is an audio whose sample 700 is non-finite (${shown})`,
      );
    }
  });

  it("refuses an audio that is not f32 samples over an ArrayBuffer with a numeric rate", () => {
    const shared = new Float32Array(new SharedArrayBuffer(4000 * 4));
    for (const [audio, fragment] of [
      [undefined, "is an audio part whose audio is not mono f32 samples"],
      [
        { data: [0.1, 0.2], sampleRate: 16000 },
        "is an audio part whose audio is not mono f32 samples",
      ],
      [{ data: new Float64Array(4000), sampleRate: 16000 }, "whose audio is not mono f32 samples"],
      [
        { data: shared, sampleRate: 16000 },
        "(data: a Float32Array over an ArrayBuffer, sampleRate)",
      ],
      [
        { data: new Float32Array(4000), sampleRate: "16000" },
        'whose sampleRate is not a number ("16000")',
      ],
    ] as const) {
      assertRefused(
        () =>
          Reflect.apply(planEmbedRequest, undefined, [AUDIO_RULES, [{ type: "audio", audio }], {}]),
        fragment,
      );
    }
  });

  it("refuses an audio of more soft tokens than the declared bound", () => {
    const rules = embedTextRules(
      TOKENIZER,
      parseEmbeddingGemmaPipelineConfig({
        ...CONFIG_JSON,
        audio: { ...AUDIO_JSON, maxSoftTokens: SWEEP_TOKENS },
      }),
      { audio: true },
    );
    assertEquals(planEmbedRequest(rules, [{ type: "audio", audio: SWEEP }], {}).audios.length, 1);
    assertRefused(
      () => planEmbedRequest(rules, [{ type: "audio", audio: EDGE }], {}),
      `content part 0 is 16400 samples, ${EDGE_TOKENS} soft tokens — over the ${SWEEP_TOKENS} this` +
        " model's audio tower takes",
    );
  });
});

describe("embedTextRules with audio", () => {
  it("refuses audio rules without pipelineConfig.audio", () => {
    const { audio: _, ...withoutAudio } = CONFIG_JSON;
    assertThrows(
      () =>
        embedTextRules(TOKENIZER, parseEmbeddingGemmaPipelineConfig(withoutAudio), {
          audio: true,
        }),
      Error,
      "audio input was asked for, but pipelineConfig declares no 'audio'",
    );
  });

  it("refuses an audio marker that is not an added token of the asset", () => {
    const config = parseEmbeddingGemmaPipelineConfig({
      ...CONFIG_JSON,
      audio: { ...AUDIO_JSON, endAudioTokenId: 818 },
    });
    const error = assertThrows(() => embedTextRules(TOKENIZER, config, { audio: true }), Error);
    assert(!(error instanceof ModelInputError), "an asset mismatch is not an input error");
    assert(
      error.message.includes("pipelineConfig.audio.endAudioTokenId (id 818) is not an added token"),
      error.message,
    );
  });
});

// ---- ④ the tower graph's contract --------------------------------------------------------------

type Graph = GraphOwner["graph"];

/** The declaration of the recipe's audio tower graph (the mirror's `audio`, minus the layers). */
const audioTowerGraph = (): Graph => ({
  format: "karume-ir",
  version: 2,
  requires: { ops: [] },
  symbols: ["n"],
  inputs: [
    { name: "features", dtype: "f32", shape: [1, "4n", 128] },
    { name: "frame_mask", dtype: "f32", shape: [1, "4n"] },
  ],
  outputs: ["linear_122"],
  initializers: {},
  values: { linear_122: { dtype: "f32", shape: [1, "n", 512] } },
  states: {},
  nodes: [],
});

const AUDIO = CONFIG.audio ?? AUDIO_JSON;
const audioContractOf = (edit: (graph: Graph) => Graph, audio = AUDIO) =>
  embeddingGemmaAudioContract({ graph: edit(audioTowerGraph()) }, audio);

describe("embeddingGemmaAudioContract", () => {
  it("accepts the recipe's declaration and reads the output, its width and the symbol", () => {
    assertEquals(
      audioContractOf((graph) => graph),
      { output: "linear_122", width: 512, symbol: "n" },
    );
  });

  it("refuses an output of another width than hiddenSize", () => {
    assertThrows(
      () => audioContractOf((graph) => graph, { ...AUDIO, hiddenSize: 768 }),
      Error,
      "the audio graph's output 'linear_122' is f32 [1, n, 512], expected f32 [1, n, 768]",
    );
  });

  it("refuses inputs of another order, frame axis, row width or dtype", () => {
    const edits: readonly ((graph: Graph) => Graph)[] = [
      (graph) => ({ ...graph, inputs: [graph.inputs[1], graph.inputs[0]] }),
      (graph) => ({
        ...graph,
        inputs: graph.inputs.map((input) => ({
          ...input,
          shape: input.shape.map((dim) => (dim === "4n" ? "n" : dim)),
        })),
      }),
      (graph) => ({
        ...graph,
        inputs: [{ name: "features", dtype: "f32", shape: [1, "4n", 80] }, graph.inputs[1]],
      }),
      (graph) => ({
        ...graph,
        inputs: [graph.inputs[0], { name: "frame_mask", dtype: "bool", shape: [1, "4n"] }],
      }),
      (graph) => ({ ...graph, inputs: [graph.inputs[0]] }),
    ];
    for (const edit of edits) {
      assertThrows(
        () => audioContractOf(edit),
        Error,
        "not features f32 [1, 4n, 128] / frame_mask f32 [1, 4n]",
      );
    }
  });

  it("refuses two symbols, a state slot and a second output", () => {
    assertThrows(
      () => audioContractOf((graph) => ({ ...graph, symbols: ["n", "m"] })),
      Error,
      "the audio graph's symbolic dimensions are [n, m]",
    );
    assertThrows(
      () =>
        audioContractOf((graph) => ({
          ...graph,
          states: { cache: { dtype: "f32", shape: [1, 4], external: false } },
        })),
      Error,
      "the audio graph holds state slots",
    );
    assertThrows(
      () => audioContractOf((graph) => ({ ...graph, outputs: ["linear_122", "linear_121"] })),
      Error,
      "the audio graph has 2 outputs",
    );
  });
});

// ---- ⑤ the run over stand-in Sessions ----------------------------------------------------------

/** A stand-in Session that records its runs and answers with `answer(inputs)`. */
type Stand = {
  readonly session: EmbeddingSession;
  readonly runs: RunInputs[];
};

const standIn = (
  name: string,
  events: string[],
  answer: (inputs: RunInputs, run: number) => RunOutputs,
  gate?: () => Promise<void>,
): Stand => {
  const runs: RunInputs[] = [];
  return {
    runs,
    session: {
      run: async (inputs): Promise<RunOutputs> => {
        runs.push(inputs);
        events.push(`${name}.run`);
        await gate?.();
        return answer(inputs, runs.length - 1);
      },
      diagnostics: () => {
        throw new Error("the stand-in Session has no diagnostics");
      },
      dispose: async () => {
        events.push(`${name}.dispose`);
      },
    },
  };
};

/** The host tensor a run received under `name`. */
const hostInput = (inputs: RunInputs, name: string): Tensor => {
  const input = inputs[name];
  if (input === undefined || !("data" in input)) throw new Error(`no host tensor '${name}'`);
  return input;
};

/** The soft width of the stand-ins (small, so the soft rows are easy to read). */
const WIDTH = 3;

const TEXT_CONTRACT: EmbeddingGemmaGraphContract = {
  output: "out",
  width: 4,
  vocabularySize: 262144,
  softWidth: WIDTH,
  bytesPerToken: 49152,
};
const AUDIO_CONTRACT: EmbeddingGemmaAudioContract = { output: "soft", width: WIDTH, symbol: "n" };
const VISION_CONTRACT: EmbeddingGemmaVisionContract = {
  output: "vsoft",
  width: WIDTH,
  positions: 840,
};

/**
 * The audio tower's answer: soft token `j` of the `run`-th audio run is `[5000·(run + 1) + j, …]` +
 * the channel / 10, so each soft row in the text run's `soft_embeds` says which audio and which
 * token it is.
 */
const audioAnswer = (inputs: RunInputs, run: number): RunOutputs => {
  const softTokens = (hostInput(inputs, "features").shape[1] ?? 0) / 4;
  const data = new Float32Array(softTokens * WIDTH);
  for (let token = 0; token < softTokens; token += 1) {
    for (let channel = 0; channel < WIDTH; channel += 1) {
      data[token * WIDTH + channel] = 5000 * (run + 1) + token + channel / 10;
    }
  }
  return { soft: { dtype: "f32", shape: [1, softTokens, WIDTH], data } };
};

/** The vision tower's answer: soft token `j` of image run `run` is `1000·(run + 1) + j` (+ channel / 10). */
const visionAnswer = (inputs: RunInputs, run: number): RunOutputs => {
  const softTokens = (hostInput(inputs, "pixels").shape[1] ?? 0) / 9;
  const data = new Float32Array(softTokens * WIDTH);
  for (let token = 0; token < softTokens; token += 1) {
    for (let channel = 0; channel < WIDTH; channel += 1) {
      data[token * WIDTH + channel] = 1000 * (run + 1) + token + channel / 10;
    }
  }
  return { vsoft: { dtype: "f32", shape: [1, softTokens, WIDTH], data } };
};

const UNIT = [0.6, 0, 0.8, 0];
const textAnswer = (): RunOutputs => ({
  out: { dtype: "f32", shape: [1, 4], data: Float32Array.from(UNIT) },
});

const SMALL = parseEmbeddingGemmaPipelineConfig({
  ...CONFIG_JSON,
  dimensions: [4, 2],
  vision: { ...VISION_JSON, hiddenSize: WIDTH },
  audio: { ...AUDIO_JSON, hiddenSize: WIDTH },
});

const BIG_DEVICE = { maxStorageBufferBindingSize: 2147483644, maxBufferSize: 1099511627776 };

/** An embedder over stand-ins, with the audio tower (and the vision tower) or without it. */
const rig = (options: {
  readonly audio: boolean;
  readonly vision?: boolean;
  readonly gate?: () => Promise<void>;
}) => {
  const events: string[] = [];
  const text = standIn("model", events, textAnswer);
  const audio = standIn("audio", events, audioAnswer, options.gate);
  const vision = standIn("vision", events, visionAnswer);
  const embedder = new Embedder({
    session: text.session,
    contract: TEXT_CONTRACT,
    rules: embedTextRules(TOKENIZER, SMALL, {
      images: options.vision === true,
      audio: options.audio,
    }),
    ...(options.audio ? { audioTower: { session: audio.session, contract: AUDIO_CONTRACT } } : {}),
    ...(options.vision === true
      ? { tower: { session: vision.session, contract: VISION_CONTRACT } }
      : {}),
    deviceLimits: BIG_DEVICE,
    release: () => {
      events.push("release");
    },
  });
  return { embedder, text, audio, vision, events };
};

/** The soft rows a text run received: the rows whose mask is set and the nonzero rows. */
const softRowsOf = (
  inputs: RunInputs,
): { readonly masked: number[]; readonly rows: Map<number, number[]> } => {
  const mask = hostInput(inputs, "soft_mask");
  const embeds = hostInput(inputs, "soft_embeds");
  const masked: number[] = [];
  const rows = new Map<number, number[]>();
  for (let row = 0; row < mask.data.length; row += 1) {
    const values = [...embeds.data.subarray(row * WIDTH, (row + 1) * WIDTH)];
    if (mask.data[row] !== 0) masked.push(row);
    if (values.some((value) => value !== 0)) rows.set(row, values);
  }
  return { masked, rows };
};

const range = (start: number, count: number): number[] =>
  Array.from({ length: count }, (_, index) => start + index);

/** A held gate: the tower's first run waits until `open()`. */
const heldGate = (): { readonly gate: () => Promise<void>; readonly open: () => void } => {
  let open: () => void = () => {};
  const held = new Promise<void>((resolve) => {
    open = resolve;
  });
  let runs = 0;
  return {
    gate: async () => {
      runs += 1;
      if (runs === 1) await held;
    },
    open: () => open(),
  };
};

describe("Embedder with the audio tower: the runs of a call holding an audio", () => {
  it("runs the tower on the audio's frames, then the text graph with its soft tokens at the audio's soft rows", async () => {
    const { embedder, text, audio } = rig({ audio: true });
    const parts: readonly EmbedContentPart[] = [
      { type: "text", text: "a" },
      { type: "audio", audio: EDGE },
    ];
    const planned = planEmbedRequest(embedTextRules(TOKENIZER, SMALL, { audio: true }), parts, {});
    const vector = await embedder.embed(parts);

    assertEquals([...vector], [...Float32Array.from(UNIT)]);
    assertEquals(audio.runs.length, 1);
    const expected = FRONT_END.towerInput(EDGE.data);
    const features = hostInput(audio.runs[0], "features");
    const mask = hostInput(audio.runs[0], "frame_mask");
    assertEquals(
      [features.dtype, features.shape, mask.dtype, mask.shape],
      ["f32", [1, EDGE_TOKENS * 4, 128], "f32", [1, EDGE_TOKENS * 4]],
    );
    assertEquals(bytesOf(features.data), bytesOf(expected.features));
    assertEquals(bytesOf(mask.data), bytesOf(expected.frameMask));

    assertEquals(text.runs.length, 1);
    const ids = hostInput(text.runs[0], "input_ids");
    assertEquals([...ids.data], idsOf(planned));
    const start = planned.audios[0]?.start ?? -1;
    const { masked, rows } = softRowsOf(text.runs[0]);
    assertEquals(masked, range(start, EDGE_TOKENS));
    assertEquals([...rows.keys()], range(start, EDGE_TOKENS));
    assertEquals(rows.get(start), [5000, 5000.1, 5000.2].map(Math.fround));
    assertEquals(rows.get(start + EDGE_TOKENS - 1), [5025, 5025.1, 5025.2].map(Math.fround));
    // The placeholder id stays under every soft row.
    assert(range(start, EDGE_TOKENS).every((row) => ids.data[row] === SOFT));
  });

  it("runs an image's tower and an audio's in sequence order, each output at its own block", async () => {
    for (const order of ["image-first", "audio-first"] as const) {
      const { embedder, text, events } = rig({ audio: true, vision: true });
      const image = { type: "image", image: SQUARE } as const;
      const audio = { type: "audio", audio: SWEEP } as const;
      await embedder.embed(order === "image-first" ? [image, audio] : [audio, image]);
      const [first, second] =
        order === "image-first"
          ? [
              { start: 2, tokens: SQUARE_TOKENS, value: 1000 },
              { start: 2 + SQUARE_TOKENS + 2, tokens: SWEEP_TOKENS, value: 5000 },
            ]
          : [
              { start: 2, tokens: SWEEP_TOKENS, value: 5000 },
              { start: 2 + SWEEP_TOKENS + 2, tokens: SQUARE_TOKENS, value: 1000 },
            ];
      assertEquals(
        events,
        order === "image-first"
          ? ["vision.run", "audio.run", "model.run"]
          : ["audio.run", "vision.run", "model.run"],
      );
      const { masked, rows } = softRowsOf(text.runs[0]);
      assertEquals(masked, [
        ...range(first.start, first.tokens),
        ...range(second.start, second.tokens),
      ]);
      assertEquals(rows.get(first.start)?.[0], first.value, order);
      assertEquals(rows.get(second.start)?.[0], second.value, order);
      assertEquals(
        rows.get(second.start + second.tokens - 1)?.[0],
        second.value + second.tokens - 1,
      );
    }
  });

  it("runs the tower once per audio, in order, each output at its own block", async () => {
    const { embedder, text, audio, events } = rig({ audio: true });
    await embedder.embed([
      { type: "audio", audio: SWEEP },
      { type: "text", text: "a" },
      { type: "audio", audio: EDGE },
    ]);
    assertEquals(events, ["audio.run", "audio.run", "model.run"]);
    assertEquals(
      audio.runs.map((inputs) => hostInput(inputs, "features").shape[1]),
      [SWEEP_TOKENS * 4, EDGE_TOKENS * 4],
    );
    const second = 2 + SWEEP_TOKENS + 1 + TOKENIZER.encode("a").length;
    const { masked, rows } = softRowsOf(text.runs[0]);
    assertEquals(masked, [...range(2, SWEEP_TOKENS), ...range(second + 1, EDGE_TOKENS)]);
    assertEquals(rows.get(second + 1)?.[0], 10000);
  });

  it("runs a text call on the text graph alone, with the inputs of a pipeline without the tower", async () => {
    const withTower = rig({ audio: true, vision: true });
    const without = rig({ audio: false });
    const { text } = parityCase("english");
    await withTower.embedder.embed(text, { prompt: "query" });
    await without.embedder.embed(text, { prompt: "query" });
    assertEquals([withTower.audio.runs.length, withTower.vision.runs.length], [0, 0]);
    const [left, right] = [withTower.text.runs[0], without.text.runs[0]];
    assertEquals(Object.keys(left).sort(), Object.keys(right).sort());
    for (const name of Object.keys(left)) {
      const [a, b] = [hostInput(left, name), hostInput(right, name)];
      assertEquals([a.dtype, a.shape], [b.dtype, b.shape], name);
      assertEquals(bytesOf(a.data), bytesOf(b.data), name);
    }
    assertEquals(softRowsOf(left), { masked: [], rows: new Map() });
  });

  it("reads the samples at issue (a later write into them does not reach the call)", async () => {
    const { gate, open } = heldGate();
    const { embedder, audio } = rig({ audio: true, gate });
    const samples = tone(4000);
    const expected = FRONT_END.towerInput(samples.data).features;
    const first = embedder.embed([{ type: "audio", audio: tone(2000, 220) }]);
    const call = embedder.embed([{ type: "audio", audio: samples }]);
    samples.data.fill(0.9);
    open();
    await Promise.all([first, call]);
    assertEquals(bytesOf(hostInput(audio.runs[1], "features").data), bytesOf(expected));
  });

  it("stops between the tower and the text run when the signal is aborted meanwhile", async () => {
    const controller = new AbortController();
    const reason = new Error("stop after the audio");
    const { embedder, text, audio } = rig({
      audio: true,
      gate: async () => {
        controller.abort(reason);
      },
    });
    const caught = await embedder
      .embed([{ type: "audio", audio: SWEEP }], { signal: controller.signal })
      .catch((error: unknown) => error);
    assertStrictEquals(caught, reason);
    assertEquals([audio.runs.length, text.runs.length], [1, 0]);
  });

  it("fails loudly on a tower output of another row count", async () => {
    const events: string[] = [];
    const embedder = new Embedder({
      session: standIn("model", events, textAnswer).session,
      contract: TEXT_CONTRACT,
      rules: embedTextRules(TOKENIZER, SMALL, { audio: true }),
      audioTower: {
        session: standIn("audio", events, () => ({
          soft: { dtype: "f32", shape: [1, 2, WIDTH], data: new Float32Array(2 * WIDTH) },
        })).session,
        contract: AUDIO_CONTRACT,
      },
      deviceLimits: BIG_DEVICE,
    });
    await assertRejects(
      () => embedder.embed([{ type: "audio", audio: SWEEP }]),
      Error,
      `the audio graph returned f32 [1, 2, ${WIDTH}], not f32 [1, ${SWEEP_TOKENS}, ${WIDTH}]`,
    );
  });

  it("counts the soft tokens toward the device's limit, before any run", async () => {
    const events: string[] = [];
    const length = SWEEP_TOKENS + 4;
    const embedder = new Embedder({
      session: standIn("model", events, textAnswer).session,
      contract: TEXT_CONTRACT,
      rules: embedTextRules(TOKENIZER, SMALL, { audio: true }),
      audioTower: {
        session: standIn("audio", events, audioAnswer).session,
        contract: AUDIO_CONTRACT,
      },
      deviceLimits: { maxStorageBufferBindingSize: (length - 1) * 49152, maxBufferSize: 2 ** 40 },
    });
    await assertRejects(
      () => embedder.embed([{ type: "audio", audio: SWEEP }]),
      ModelInputError,
      `the input is ${length} tokens (<bos>, <eos> and the soft tokens of the audio with their two markers each included), over this device's limit of ${length - 1}`,
    );
    assertEquals(events, []);
  });

  it("disposes the text Session, then the vision tower's, then the audio tower's, then releases, after the call in flight", async () => {
    const { gate, open } = heldGate();
    const { embedder, events } = rig({ audio: true, vision: true, gate });
    const call = embedder.embed([{ type: "audio", audio: SWEEP }]);
    const disposal = embedder.dispose();
    open();
    await call;
    await disposal;
    assertEquals(events, [
      "audio.run",
      "model.run",
      "model.dispose",
      "vision.dispose",
      "audio.dispose",
      "release",
    ]);
  });

  it("refuses an audio part on an embedder without the audio tower, before any run", async () => {
    const { embedder, events } = rig({ audio: false, vision: true });
    await assertRejects(
      () => embedder.embed([{ type: "audio", audio: SWEEP }]),
      ModelInputError,
      "build the pipeline with { audio: true }",
    );
    assertEquals(events, []);
  });

  it("refuses to be built with the audio tower but without audio rules, or with soft tokens of another width", () => {
    const events: string[] = [];
    assertThrows(
      () =>
        new Embedder({
          session: standIn("model", events, textAnswer).session,
          contract: TEXT_CONTRACT,
          rules: embedTextRules(TOKENIZER, SMALL),
          audioTower: {
            session: standIn("audio", events, audioAnswer).session,
            contract: AUDIO_CONTRACT,
          },
          deviceLimits: BIG_DEVICE,
        }),
      Error,
      "the audio rules and the audio tower are not built together",
    );
    assertThrows(
      () =>
        new Embedder({
          session: standIn("model", events, textAnswer).session,
          contract: TEXT_CONTRACT,
          rules: embedTextRules(TOKENIZER, SMALL, { audio: true }),
          audioTower: {
            session: standIn("audio", events, audioAnswer).session,
            contract: { ...AUDIO_CONTRACT, width: WIDTH + 1 },
          },
          deviceLimits: BIG_DEVICE,
        }),
      Error,
      `the audio graph's soft tokens are ${WIDTH + 1} wide, the text graph's soft rows ${WIDTH}`,
    );
  });
});

/** A device that binds 2 GiB (the stand-in GPU of {@link openEmbedder}'s tests). */
const LIMITED_GPU = (): GpuContext =>
  new GpuContext(
    fakeDevice(),
    readAdapterInfo({}),
    {
      maxBufferSize: 2 ** 40,
      maxStorageBufferBindingSize: 2147483644,
      maxUniformBufferBindingSize: 65536,
      maxStorageBuffersPerShaderStage: 10,
      maxUniformBuffersPerShaderStage: 12,
      maxComputeWorkgroupStorageSize: 16384,
      maxComputeInvocationsPerWorkgroup: 256,
      maxComputeWorkgroupSizeX: 256,
      maxComputeWorkgroupSizeY: 256,
      maxComputeWorkgroupSizeZ: 64,
      maxComputeWorkgroupsPerDimension: 65535,
    },
    new Set(),
  );

/** A stand-in graph: records the settings its Session is opened with, fails when told to. */
const graphOf = (
  name: string,
  events: string[],
  settings: SessionOptions[],
  failures: { readonly open?: Error } = {},
): EmbeddingGraph => ({
  createSession: async (_gpu, options) => {
    events.push(`${name}.open`);
    settings.push(options);
    if (failures.open !== undefined) throw failures.open;
    const answer = name === "audio" ? audioAnswer : name === "vision" ? visionAnswer : textAnswer;
    return standIn(name, events, answer).session;
  },
});

/** A quant's effective settings (one knob, so the test sees them passed through). */
const QUANT_SETTINGS: SessionOptions = { numerics: { level: "reorder" } };

describe("openEmbedder: the Sessions of a pipeline with the audio tower, and their clean-up", () => {
  const opening = (
    events: string[],
    settings: SessionOptions[],
    overrides: { readonly audio?: EmbeddingGraph; readonly audioWidth?: number } = {},
  ) =>
    openEmbedder({
      gpu: LIMITED_GPU(),
      assertGpu: () => {},
      model: graphOf("model", events, settings),
      tower: { graph: graphOf("vision", events, settings), contract: VISION_CONTRACT },
      audioTower: {
        graph: overrides.audio ?? graphOf("audio", events, settings),
        contract: { ...AUDIO_CONTRACT, width: overrides.audioWidth ?? WIDTH },
      },
      sessionOptions: QUANT_SETTINGS,
      contract: TEXT_CONTRACT,
      rules: embedTextRules(TOKENIZER, SMALL, { images: true, audio: true }),
      release: () => {
        events.push("release");
      },
    });

  it("opens the text graph's Session, then the vision tower's, then the audio tower's, each with the quant's settings and the row-block budget", async () => {
    const events: string[] = [];
    const settings: SessionOptions[] = [];
    const embedder = await opening(events, settings);
    const expected = { ...QUANT_SETTINGS, attentionRowBlockMaxBytes: 128 * 2 ** 20 };
    assertEquals(settings, [expected, expected, expected]);
    await embedder.embed([{ type: "audio", audio: SWEEP }]);
    await embedder.dispose();
    assertEquals(events, [
      "model.open",
      "vision.open",
      "audio.open",
      "audio.run",
      "model.run",
      "model.dispose",
      "vision.dispose",
      "audio.dispose",
      "release",
    ]);
  });

  it("disposes the text and vision Sessions and releases when the audio tower's Session fails to open", async () => {
    const events: string[] = [];
    const failure = new Error("the audio tower's Session failed");
    const caught = await opening(events, [], {
      audio: graphOf("audio", events, [], { open: failure }),
    }).catch((error: unknown) => error);
    assertStrictEquals(caught, failure);
    assertEquals(events, [
      "model.open",
      "vision.open",
      "audio.open",
      "model.dispose",
      "vision.dispose",
      "release",
    ]);
  });

  it("disposes every Session and releases when the embedder refuses its parts", async () => {
    const events: string[] = [];
    await assertRejects(
      () => opening(events, [], { audioWidth: WIDTH + 1 }),
      Error,
      `the audio graph's soft tokens are ${WIDTH + 1} wide`,
    );
    assertEquals(events, [
      "model.open",
      "vision.open",
      "audio.open",
      "model.dispose",
      "vision.dispose",
      "audio.dispose",
      "release",
    ]);
  });
});

// ---- ⑥ fromAssets with audio --------------------------------------------------------------------

/** Declaration-only `model` (V 8, a soft width of 4, a T bound of 16) — the host test's shape. */
const modelInput = (): ModelInput => ({
  graphs: {
    model: parseIrDeclarationValue({
      format: "karume-ir",
      version: 2,
      requires: { ops: ["embedding", "sym_prefix_slice", "sum"] },
      symbols: ["T"],
      inputs: [
        { name: "input_ids", dtype: "i32", shape: [1, "T"] },
        { name: "pool_mask", dtype: "f32", shape: [1, "T"] },
        { name: "soft_embeds", dtype: "f32", shape: [1, "T", 4] },
        { name: "soft_mask", dtype: "bool", shape: [1, "T", 1] },
      ],
      outputs: ["out"],
      initializers: { table: {}, band: {} },
      values: {
        table: { dtype: "f32", shape: [8, 4] },
        band: { dtype: "f32", shape: [16, 4] },
        embedded: { dtype: "f32", shape: [1, "T", 4] },
        band_t: { dtype: "f32", shape: ["T", 4] },
        out: { dtype: "f32", shape: [1, 4] },
      },
      states: {},
      nodes: [
        {
          op: "embedding",
          ins: ["table", "input_ids"],
          outs: ["embedded"],
          attrs: { padding_idx: -1 },
        },
        {
          op: "sym_prefix_slice",
          ins: ["band"],
          outs: ["band_t"],
          attrs: { sym: "T", slices: [{ dim: 0, coeff: 1, offset: 0 }] },
        },
        { op: "sum", ins: ["embedded"], outs: ["out"], attrs: { dim: 1 } },
      ],
    }),
  },
  consts: [],
  weights: [
    {
      graph: "model",
      initializer: "table",
      bytes: new Uint8Array(8 * 4 * 4),
      encoding: { codec: "f32" },
    },
    {
      graph: "model",
      initializer: "band",
      bytes: new Uint8Array(16 * 4 * 4),
      encoding: { codec: "f32" },
    },
  ],
  assets: [],
  provenance: { license: "test" },
});

/**
 * Declaration-only `audio` (soft tokens of width `width`, a window constant of `bound` rows sliced
 * by n — the tower's bound; no slice with `"unbounded"`).
 */
const audioInput = (width: number, bound: number | "unbounded" = 750): ModelInput => ({
  graphs: {
    audio: parseIrDeclarationValue({
      format: "karume-ir",
      version: 2,
      requires: { ops: bound === "unbounded" ? ["sum"] : ["sym_prefix_slice", "sum"] },
      symbols: ["n"],
      inputs: [
        { name: "features", dtype: "f32", shape: [1, "4n", 128] },
        { name: "frame_mask", dtype: "f32", shape: [1, "4n"] },
      ],
      outputs: ["soft"],
      initializers: { window: {} },
      values: {
        window: { dtype: "f32", shape: [bound === "unbounded" ? 4 : bound, 12] },
        ...(bound === "unbounded" ? {} : { window_n: { dtype: "f32", shape: ["n", 12] } }),
        soft: { dtype: "f32", shape: [1, "n", width] },
      },
      states: {},
      nodes: [
        ...(bound === "unbounded"
          ? []
          : [
              {
                op: "sym_prefix_slice",
                ins: ["window"],
                outs: ["window_n"],
                attrs: { sym: "n", slices: [{ dim: 0, coeff: 1, offset: 0 }] },
              },
            ]),
        { op: "sum", ins: ["features"], outs: ["soft"], attrs: { dim: 2 } },
      ],
    }),
  },
  consts: [],
  weights: [
    {
      graph: "audio",
      initializer: "window",
      bytes: new Uint8Array((bound === "unbounded" ? 4 : bound) * 12 * 4),
      encoding: { codec: "f32" },
    },
  ],
  assets: [],
  provenance: { license: "test" },
});

const MODEL = await writeContainer(modelInput());
const declared = async (written: typeof MODEL, path: string) => ({
  container: {
    descriptor: written.descriptor,
    parts: await Promise.all(
      written.parts.map((bytes, index) => fileRef(`${path}-${index}.krm`, bytes)),
    ),
  },
});
const MODEL_DECLARED = await declared(MODEL, "text/model/model.bf16");
const TOKENIZER_REF = { path: "text/tokenizer/tokenizer.json", size: 2, sha256: "f".repeat(64) };
const NOT_AN_ASSET = new TextEncoder().encode("{}");
/**
 * The `pipelineConfig` of the two-graph distribution below, without `audio` and with it: ids inside
 * its 8-row table (admission reads declarations only — the tokenizer asset is never parsed here).
 */
const SMALL_TEXT_CONFIG = {
  prompts: CONFIG_JSON.prompts,
  dimensions: [4, 2],
  maxTokens: 16,
  placeholderTokens: { image: 5, audio: 3, video: 4 },
};
const SMALL_AUDIO = {
  audioTokenId: 3,
  beginAudioTokenId: 1,
  endAudioTokenId: 2,
  hiddenSize: 4,
  maxSoftTokens: 750,
  sampleRate: 16000,
};
const SMALL_CONFIG = { ...SMALL_TEXT_CONFIG, audio: SMALL_AUDIO };

/** The two-graph distribution (`audio` a weights role of the preset), and its assets. */
const distribution = async (options: {
  readonly audio?: ModelInput;
  readonly pipelineConfig?: Record<string, unknown>;
}) => {
  const tower = await writeContainer(options.audio ?? audioInput(4));
  const manifest = parseManifest(
    JSON.stringify({
      format: "karume/6",
      generator: "karume/0.1.0",
      defaultModel: "text",
      models: {
        text: {
          pipeline: "embeddinggemma/1",
          weights: {
            model: { bf16: MODEL_DECLARED },
            audio: { bf16: await declared(tower, "text/audio/model.bf16") },
          },
          assets: { tokenizer: TOKENIZER_REF },
          quants: { bf16: { weights: { model: "bf16", audio: "bf16" }, session: {} } },
          defaultQuant: "bf16",
          pipelineConfig: options.pipelineConfig ?? SMALL_CONFIG,
        },
      },
    }),
  );
  return {
    manifest,
    assets: {
      ...partAssets("model", MODEL),
      ...partAssets("audio", tower),
      tokenizer: NOT_AN_ASSET,
    },
  };
};

const fakeGpu = () => fakeGpuContext(fakeDevice());

describe("Pipeline.fromAssets with audio: admission fails before touching a GPU", () => {
  it("positive path: both graphs pass admission and the build stops at parsing the tokenizer asset", async () => {
    await assertRejects(
      async () => Pipeline.fromAssets(await distribution({}), { audio: true, gpu: fakeGpu() }),
      Error,
      "tokenizer.format: missing",
    );
  });

  it("without audio the tower is neither opened nor checked (a tower of another width passes)", async () => {
    await assertRejects(
      async () =>
        Pipeline.fromAssets(await distribution({ audio: audioInput(7) }), { gpu: fakeGpu() }),
      Error,
      "tokenizer.format: missing",
    );
  });

  const rejected: readonly [string, Parameters<typeof distribution>[0], string][] = [
    [
      "a pipelineConfig without audio",
      { pipelineConfig: SMALL_TEXT_CONFIG },
      "audio was requested, but this model's pipelineConfig declares no 'audio'",
    ],
    [
      "a hiddenSize other than the text graph's soft rows",
      { pipelineConfig: { ...SMALL_CONFIG, audio: { ...SMALL_AUDIO, hiddenSize: 8 } } },
      "pipelineConfig.audio.hiddenSize 8 is not the width of the text graph's soft rows (4)",
    ],
    [
      "an audio marker outside the text graph's table",
      { pipelineConfig: { ...SMALL_CONFIG, audio: { ...SMALL_AUDIO, endAudioTokenId: 8 } } },
      "pipelineConfig.audio.endAudioTokenId 8 is outside the embedding table's 8 rows",
    ],
    [
      "a tower whose soft tokens are of another width",
      { audio: audioInput(6) },
      "the audio graph's output 'soft' is f32 [1, n, 6], expected f32 [1, n, 4]",
    ],
    [
      "a tower whose bound is not the declared maxSoftTokens",
      { audio: audioInput(4, 700) },
      "pipelineConfig.audio.maxSoftTokens is 750, but the audio graph runs 'n' up to 700",
    ],
    [
      "a tower without a bound of its soft-token count",
      { audio: audioInput(4, "unbounded") },
      "no sym_prefix_slice of the audio graph bounds 'n'",
    ],
  ];
  for (const [label, patch, message] of rejected) {
    it(label, async () => {
      const input = await distribution(patch);
      const error = await assertRejects(
        () => Pipeline.fromAssets(input, { audio: true, gpu: fakeGpu() }),
        Error,
        message,
      );
      assert(
        !(error instanceof ModelInputError),
        `an asset mismatch is not an input error: ${label}`,
      );
    });
  }

  it("refuses an audio option that is not a boolean, as an input error", async () => {
    const input = await distribution({});
    await assertRejects(
      () =>
        // oxlint-disable-next-line typescript/unbound-method -- Reflect.apply supplies Pipeline as `this`
        Reflect.apply(Pipeline.fromAssets, Pipeline, [input, { audio: 1, gpu: fakeGpu() }]),
      ModelInputError,
      "audio 1 is not a boolean",
    );
  });
});
