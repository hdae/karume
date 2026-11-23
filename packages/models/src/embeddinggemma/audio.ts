/**
 * The audio tower of EmbeddingGemma 2 on the host: mono 16 kHz samples → the log-mel frames and the
 * frame mask the tower graph `audio` reads, the tower's graph contract, and one tower run → the
 * audio's soft rows.
 *
 * ## The front end (upstream's `Gemma4AudioFeatureExtractor`, as the snapshot configures it)
 *
 * The extractor of the pinned snapshot (`preprocessor_config.json`; the dist recipe holds every
 * setting below to it — `tools/export-recipes/embeddinggemma2/distribution.py`):
 *
 * 1. The input is 16 kHz mono samples. Nothing is resampled: another rate is refused (`input.ts`).
 * 2. 160 zeros are prepended (half a frame — upstream's semicausal padding).
 * 3. Frame `i` is the 320 samples from `160 · i` of the padded signal (20 ms by a hop of 10 ms;
 *    preemphasis 0, so upstream's 321st sample is not used), times a periodic Hann window of 320
 *    stored as f32 (`np.hanning(321)[:-1]` cast to f32).
 * 4. Its rfft of length 512 (zero-padded), and the **magnitude** of each of the 257 bins (not the
 *    power).
 * 5. 128 triangular mel filters over the bins — transformers' `mel_filter_bank` with the HTK mel
 *    scale from 0 to 8,000 Hz, the triangles drawn in Hz, no area normalisation (filter 0 is all
 *    zero) — and `ln(mel + 0.001)`.
 * 6. Frame `i` is valid iff sample `160 · i + 160` of the audio exists (the window's last sample,
 *    upstream's 321st, is real audio): `s` samples have `L = max(0, ceil(s / 160) − 1)` valid frames
 *    ({@link audioValidFrames}). Upstream zeroes the others (its right padding to a multiple of 128
 *    samples only adds such frames); here they are not computed.
 *
 * **Precision.** Upstream (numpy 2) windows and transforms in f32 (`complex64`) and takes the mel
 * product and the logarithm in f64; this front end does every step in f64 and rounds to f32 once,
 * at the end — the f32 window values times f32 samples are exact in f64. Its frames therefore sit
 * nearer the exact values than upstream's own; the difference to upstream's frames is measured in
 * `embeddinggemma2_audio_input_test.ts` (not copied: copying numpy's f32 rounding would fake a match —
 * the rule of `src/audio/fft.ts`).
 *
 * ## The tower's input (the io contract of the recipe, `export_audio.py`)
 *
 * An audio of `L` valid frames has `n = ceil(L / 4)` soft tokens ({@link audioSoftTokens} — two
 * stride-2 convolutions). `features` is `[1, 4n, 128]`: the `L` frames, then `4n − L` (0 to 3) zero
 * frames; `frame_mask` is `[1, 4n]`, 1 on the first `L` and 0 after (upstream's own mask, as f32 —
 * the tower needs it exactly when `L ≡ 2 (mod 4)`). The output is `[1, n, 512]`: the audio's soft
 * tokens in order, every one of them valid.
 *
 * MUST: modules are side-effect free (CLAUDE.md) — the window, the filters and the FFT tables are
 * built by {@link AudioFrontEnd}'s constructor, never at module scope.
 */

import type { RunOutputs, Session } from "@karume/runtime";

import { Fft } from "../audio/fft.ts";
import type { GraphOwner } from "../hub/components.ts";
import type { PipelineAudioConfig } from "./config.ts";

/** The message prefix (the class name). */
const OWNER = "embeddinggemma.Pipeline";

/**
 * Mono PCM samples as f32 (nominally in `[-1, 1]`) and their sample rate — what an audio part holds.
 * The shape of `decodeWav`'s result and of the TTS families' generated audio, so either passes as
 * is.
 */
export type MonoAudio = {
  readonly data: Float32Array<ArrayBuffer>;
  readonly sampleRate: number;
};

/** The sample rate the front end is built for (upstream's `sampling_rate`). */
export const AUDIO_SAMPLE_RATE = 16_000;

/**
 * The longest audio upstream's feature extractor reads: its `max_length` default, 30 s at 16 kHz
 * (a default of its call, not a setting of the file; the recipe's `export_audio.py` holds it to the
 * signature). Upstream truncates a longer one silently; the host refuses it (`input.ts`).
 */
export const AUDIO_MAX_SAMPLES = 480_000;

/** The samples of one frame (`frame_length`, 20 ms) and the hop between frames (`hop_length`, 10 ms). */
const FRAME_LENGTH = 320;
const HOP = 160;

/** The zeros prepended to the samples (`frame_length // 2` — semicausal). */
const PAD_LEFT = FRAME_LENGTH / 2;

/** The transform length (`fft_length` — the frame zero-padded) and its rfft bins. */
const FFT_LENGTH = 512;
const FFT_BINS = FFT_LENGTH / 2 + 1;

/** The mel filters (`feature_size`) — the width of a frame row. */
const MEL_BINS = 128;

/** The mel filters' range in Hz (`min_frequency` / `max_frequency`). */
const MIN_FREQUENCY = 0;
const MAX_FREQUENCY = 8000;

/** What is added to the mel energies before the logarithm (`mel_floor`). */
const MEL_FLOOR = 1e-3;

/**
 * The settings of upstream's feature extractor this front end implements, under the keys of the
 * snapshot's `preprocessor_config.json` (the host test holds them to that file; the dist recipe
 * holds the file to the same values — `EMBEDDINGGEMMA2_AUDIO_FEATURE_EXTRACTOR`). No preemphasis,
 * no dither, no input scale, no per-bin normalisation and no FFT overdrive: those settings are not
 * implemented, so the recipe refuses a snapshot that turns one on.
 *
 * MUST: built from the constants above, not read from at module scope — a property read survives
 * tree-shaking through the barrel (`models_barrel_tree_shaking_test.ts`).
 *
 * NOTE: the `export` is for the host tests (not on `mod.ts` / the subpath surface — ADR 0008).
 */
export const AUDIO_FEATURE_EXTRACTOR = {
  sampling_rate: AUDIO_SAMPLE_RATE,
  feature_size: MEL_BINS,
  frame_length: FRAME_LENGTH,
  hop_length: HOP,
  fft_length: FFT_LENGTH,
  min_frequency: MIN_FREQUENCY,
  max_frequency: MAX_FREQUENCY,
  mel_floor: MEL_FLOOR,
} as const;

/** The mel frames per soft token (the tower's two stride-2 convolutions). */
const FRAMES_PER_SOFT_TOKEN = 4;

/** The tower graph's inputs, in order (`GRAPH_INPUTS` of the recipe's `export_audio.py`). */
const AUDIO_FEATURES = "features";
const AUDIO_FRAME_MASK = "frame_mask";

/**
 * The valid mel frames of an audio of `samples` samples (module doc, step 6): `max(0, ceil(samples
 * / 160) − 1)` — 1 s → 99, 30 s → 2,999.
 */
export const audioValidFrames = (samples: number): number =>
  Math.max(0, Math.ceil(samples / HOP) - 1);

/** The soft tokens of `frames` valid mel frames: `ceil(frames / 4)` — 1 s → 25, 30 s → 750. */
export const audioSoftTokens = (frames: number): number =>
  Math.ceil(frames / FRAMES_PER_SOFT_TOKEN);

/** HTK mel ↔ Hz (transformers' `hertz_to_mel` / `mel_to_hertz` with `mel_scale="htk"`). */
const hertzToMel = (hertz: number): number => 2595 * Math.log10(1 + hertz / 700);
const melToHertz = (mel: number): number => 700 * (10 ** (mel / 2595) - 1);

/** numpy's `linspace(start, stop, count)`: `i · step + start`, the last value `stop` itself. */
const linspace = (start: number, stop: number, count: number): Float64Array<ArrayBuffer> => {
  const values = new Float64Array(count);
  const step = (stop - start) / (count - 1);
  for (let index = 0; index < count; index += 1) values[index] = index * step + start;
  values[count - 1] = stop;
  return values;
};

/** One mel filter: the weights of its nonzero bins, from bin `first` on. */
type MelFilter = { readonly first: number; readonly weights: Float64Array<ArrayBuffer> };

/**
 * The mel filters (transformers' `mel_filter_bank(257, 128, 0, 8000, 16000, norm=None,
 * mel_scale="htk")`, in f64): filter `m` at bin `b` is `max(0, min(rising, falling))` with the
 * triangle's corners at the mel points `m`, `m + 1`, `m + 2`, each operation in upstream's order.
 * Only the nonzero run of each filter is kept — the zeros add nothing to the sum.
 */
const melFilters = (): readonly MelFilter[] => {
  const corners = linspace(hertzToMel(MIN_FREQUENCY), hertzToMel(MAX_FREQUENCY), MEL_BINS + 2).map(
    melToHertz,
  );
  // The bins' frequencies run from 0 to half the sample rate (8,000 Hz here, the filters' top).
  const bins = linspace(0, AUDIO_SAMPLE_RATE / 2, FFT_BINS);
  const filters: MelFilter[] = [];
  for (let mel = 0; mel < MEL_BINS; mel += 1) {
    const weights = new Float64Array(FFT_BINS);
    let first = FFT_BINS;
    let last = -1;
    for (let bin = 0; bin < FFT_BINS; bin += 1) {
      const rising = -(corners[mel] - bins[bin]) / (corners[mel + 1] - corners[mel]);
      const falling = (corners[mel + 2] - bins[bin]) / (corners[mel + 2] - corners[mel + 1]);
      const weight = Math.max(0, Math.min(rising, falling));
      weights[bin] = weight;
      if (weight !== 0) {
        first = Math.min(first, bin);
        last = bin;
      }
    }
    filters.push({ first, weights: weights.slice(first, last + 1) });
  }
  return filters;
};

/**
 * The periodic Hann window of 320 as upstream stores it: `np.hanning(321)[:-1]` (numpy's formula
 * `0.5 + 0.5 · cos(π · n / 320)` over `n = −320, −318, …`) cast to f32.
 */
const hannWindow = (): Float64Array<ArrayBuffer> => {
  const window = new Float64Array(FRAME_LENGTH);
  for (let index = 0; index < FRAME_LENGTH; index += 1) {
    const n = 2 * index - FRAME_LENGTH;
    window[index] = Math.fround(0.5 + 0.5 * Math.cos((Math.PI * n) / FRAME_LENGTH));
  }
  return window;
};

/** One audio as the tower reads it. */
export type AudioTowerInput = {
  /** The audio's soft tokens (n). */
  readonly softTokens: number;
  /** Its valid mel frames (L, `4n − 3 ≤ L ≤ 4n`). */
  readonly validFrames: number;
  /** `[4n, 128]`: the `L` log-mel frames, then zero frames. */
  readonly features: Float32Array<ArrayBuffer>;
  /** `[4n]`: 1 on the first `L` frames, 0 after. */
  readonly frameMask: Float32Array<ArrayBuffer>;
};

/**
 * The front end's tables (the window, the mel filters, the FFT and its scratch) and the frames of
 * an audio (module doc). Built once per pipeline, on its first audio; it is not shared between two
 * callers at once (the scratch), which holds because a frame is computed synchronously.
 *
 * NOTE: the `export` is for the pipeline and its host tests (not on `mod.ts` / the subpath surface —
 * ADR 0008).
 */
export class AudioFrontEnd {
  readonly #window = hannWindow();
  readonly #filters = melFilters();
  readonly #fft = new Fft(FFT_LENGTH);
  readonly #frame = new Float64Array(FRAME_LENGTH);
  readonly #power = new Float64Array(FFT_BINS);
  readonly #magnitude = new Float64Array(FFT_BINS);

  /**
   * The tower's input of `samples` (16 kHz mono; the caller checked the rate, the length and the
   * values — `input.ts`). The samples are read now: a later write into them reaches nothing.
   *
   * MUST: the audio has at least one valid frame — an audio of no frame has no soft token, and the
   * tower graph runs `n ≥ 1` only.
   */
  towerInput(samples: Float32Array<ArrayBuffer>): AudioTowerInput {
    const validFrames = audioValidFrames(samples.length);
    if (validFrames === 0) {
      throw new Error(
        `${OWNER}: internal inconsistency: an audio of ${samples.length} samples has no valid frame`,
      );
    }
    const softTokens = audioSoftTokens(validFrames);
    const frames = softTokens * FRAMES_PER_SOFT_TOKEN;
    const features = new Float32Array(frames * MEL_BINS);
    const frameMask = new Float32Array(frames);
    frameMask.fill(1, 0, validFrames);
    for (let frame = 0; frame < validFrames; frame += 1) {
      this.#logMel(samples, frame, features.subarray(frame * MEL_BINS, (frame + 1) * MEL_BINS));
    }
    return { softTokens, validFrames, features, frameMask };
  }

  /** Writes the log-mel row of valid frame `frame` into `row` (module doc, steps 2 to 5). */
  #logMel(samples: Float32Array<ArrayBuffer>, frame: number, row: Float32Array<ArrayBuffer>): void {
    const start = frame * HOP - PAD_LEFT;
    for (let offset = 0; offset < FRAME_LENGTH; offset += 1) {
      const at = start + offset;
      // Before the audio: the prepended zeros. A valid frame never reads past its end.
      const sample = at < 0 ? 0 : samples[at];
      if (sample === undefined) {
        throw new Error(
          `${OWNER}: internal inconsistency: frame ${frame} reads sample ${at} of ${samples.length}`,
        );
      }
      this.#frame[offset] = sample * this.#window[offset];
    }
    this.#fft.powerSpectrum(this.#frame, this.#power);
    for (let bin = 0; bin < FFT_BINS; bin += 1) this.#magnitude[bin] = Math.sqrt(this.#power[bin]);
    for (let mel = 0; mel < MEL_BINS; mel += 1) {
      const { first, weights } = this.#filters[mel];
      let energy = 0;
      for (let index = 0; index < weights.length; index += 1) {
        energy += this.#magnitude[first + index] * weights[index];
      }
      row[mel] = Math.log(energy + MEL_FLOOR);
    }
  }
}

/** What the pipeline reads of the tower graph (checked at admission). */
export type EmbeddingGemmaAudioContract = {
  /** The name of the one output (`[1, n, hiddenSize]`). */
  readonly output: string;
  /** The width of a soft token (= `pipelineConfig.audio.hiddenSize`). */
  readonly width: number;
  /** The graph's one symbol (the soft-token count n — its bound is checked at admission). */
  readonly symbol: string;
};

/** Whether a declared shape is exactly `expected` (rank and every axis). */
const sameShape = (
  shape: readonly (number | string)[],
  expected: readonly (number | string)[],
): boolean =>
  shape.length === expected.length && shape.every((dim, axis) => dim === expected[axis]);

/**
 * Matches the tower graph against what the host binds and reads, and against
 * `pipelineConfig.audio`: one symbol n (the soft-token count); exactly the inputs `features` f32
 * `[1, 4n, 128]` and `frame_mask` f32 `[1, 4n]`, in that order; one f32 output `[1, n,
 * hiddenSize]`; no state. The bound of n is the pipeline's check (`pipeline.ts`, `admitAudio`).
 *
 * NOTE: the `export` is for the host tests (not on `mod.ts` / the subpath surface — ADR 0008).
 */
export const embeddingGemmaAudioContract = (
  tower: GraphOwner,
  audio: PipelineAudioConfig,
): EmbeddingGemmaAudioContract => {
  const { graph } = tower;
  if (graph.symbols.length !== 1) {
    throw new Error(
      `${OWNER}: the audio graph's symbolic dimensions are [${graph.symbols.join(", ")}] (one soft-token count is required)`,
    );
  }
  const [symbol] = graph.symbols;
  const frames = `${FRAMES_PER_SOFT_TOKEN}${symbol}`;
  const expected = [
    [AUDIO_FEATURES, "f32", [1, frames, MEL_BINS]],
    [AUDIO_FRAME_MASK, "f32", [1, frames]],
  ] as const;
  const declared = graph.inputs.map(
    (input) => `${input.name} ${input.dtype} [${input.shape.join(", ")}]`,
  );
  if (
    graph.inputs.length !== expected.length ||
    expected.some(
      ([name, dtype, shape], index) =>
        graph.inputs[index]?.name !== name ||
        graph.inputs[index]?.dtype !== dtype ||
        !sameShape(graph.inputs[index]?.shape ?? [], shape),
    )
  ) {
    throw new Error(
      `${OWNER}: the audio graph's inputs are ${declared.join(" / ")}, not` +
        ` ${expected.map(([name, dtype, shape]) => `${name} ${dtype} [${shape.join(", ")}]`).join(" / ")}`,
    );
  }
  if (Object.keys(graph.states).length > 0) {
    throw new Error(`${OWNER}: the audio graph holds state slots (it runs one audio per run)`);
  }
  if (graph.outputs.length !== 1) {
    throw new Error(
      `${OWNER}: the audio graph has ${graph.outputs.length} outputs (one is required)`,
    );
  }
  const [output] = graph.outputs;
  const value = Object.hasOwn(graph.values, output) ? graph.values[output] : undefined;
  if (
    value === undefined ||
    value.dtype !== "f32" ||
    !sameShape(value.shape, [1, symbol, audio.hiddenSize])
  ) {
    throw new Error(
      `${OWNER}: the audio graph's output '${output}' is ${value?.dtype ?? "undeclared"}` +
        ` [${value?.shape.join(", ") ?? ""}], expected f32 [1, ${symbol}, ${audio.hiddenSize}]` +
        " (pipelineConfig.audio.hiddenSize)",
    );
  }
  return { output, width: audio.hiddenSize, symbol };
};

/**
 * Runs one audio's frames through the tower and returns its soft tokens `[n · width]` (the readback
 * as it came: the caller copies it into the text graph's soft rows before the Session's next run).
 *
 * MUST: the shape is checked — a tower that returned another row count would still fill rows, of
 * another audio.
 */
export const runAudioTower = async (
  session: Pick<Session, "run">,
  input: AudioTowerInput,
  contract: EmbeddingGemmaAudioContract,
): Promise<Float32Array<ArrayBuffer>> => {
  const frames = input.softTokens * FRAMES_PER_SOFT_TOKEN;
  const outputs: RunOutputs = await session.run({
    [AUDIO_FEATURES]: { dtype: "f32", shape: [1, frames, MEL_BINS], data: input.features },
    [AUDIO_FRAME_MASK]: { dtype: "f32", shape: [1, frames], data: input.frameMask },
  });
  const { output, width } = contract;
  if (!Object.hasOwn(outputs, output)) {
    throw new Error(
      `${OWNER}: the audio graph's output '${output}' is missing from the run results`,
    );
  }
  const tensor = outputs[output];
  if (
    tensor.dtype !== "f32" ||
    !sameShape(tensor.shape, [1, input.softTokens, width]) ||
    tensor.data.length !== input.softTokens * width
  ) {
    throw new Error(
      `${OWNER}: the audio graph returned ${tensor.dtype} [${tensor.shape.join(", ")}], not f32` +
        ` [1, ${input.softTokens}, ${width}]`,
    );
  }
  return tensor.data;
};
