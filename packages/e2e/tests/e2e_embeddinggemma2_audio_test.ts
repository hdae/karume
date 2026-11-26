// Real-GPU golden E2E of the audio tower of EmbeddingGemma 2 (an audio's log-mel frames → its soft
// tokens) (step 3 of ADR 0005).
//
// The targets are three series of `tools/export-recipes/embeddinggemma2/export_audio.py`:
// `embeddinggemma-2-audio` (f32 weights, about 1.1 GiB; its `generate` in SERIES below is the
// source of truth for the command), `embeddinggemma-2-audio-bf16` (the distribution's `bf16`
// preset: every linear and convolution weight as upstream's bf16, about 560 MiB) and
// `embeddinggemma-2-audio-i8` (the distribution's `i8` preset: those weights rounded to per-row
// int8 before the goldens were taken, except the two subsampling convolutions, which stay bf16 —
// about 282 MiB). One graph `audio`: the log-mel frames
// `features [1, 4n, 128]` and the frame mask `frame_mask [1, 4n]` in → the soft tokens
// `[1, n, 512]` out, `n` symbolic (the audio's soft-token count, 1 to 750). The frames are the
// audio's valid ones zero-padded to 4n, the mask 1 on the valid ones (the io contract in the
// recipe's module doc and README); the goldens are upstream's own soft tokens (its feature
// extractor and its tower in torch on CPU, f32, over the extractor's frames with its mask), on six
// synthetic waveforms — every residue of the frame count mod 4 (the mask matters at 2), 1 s, 10 s
// and the 30 s maximum.
//
// Every case ends with a sha256 check of the output against this machine's reference row (ADR
// 0106; the row ID is `<series>/<waveform>`). Rows are created with KARUME_REFERENCE=write.
// Each case also times the Session's build, the first run and WARM_RUNS more runs (each equal to
// the first bit for bit); the timings go to the console and the case's note. A last case per series
// runs the 30 s golden on a device whose binding and buffer limits are capped at the WebGPU default
// (128 MiB) and holds it to the uncapped device's bits — the tower needs no `requiredLimits`.
//
// ## Where the assets come from (ADR 0139)
//
// | Series                        | Container                                | Goldens                                    | Runs                      |
// | ----------------------------- | ---------------------------------------- | ------------------------------------------ | ------------------------- |
// | `embeddinggemma-2-audio-bf16` | the mirror's `bf16` preset               | `goldenDir("embeddinggemma-2-audio-bf16")` | default lane              |
// | `embeddinggemma-2-audio-i8`   | the mirror's `i8` preset                 | `goldenDir("embeddinggemma-2-audio-i8")`   | default lane              |
// | `embeddinggemma-2-audio`      | `outputs/series/embeddinggemma-2-audio/` | the same series directory                  | only with KARUME_SERIES=1 |
//
// The bf16 and int8 series are shipped: their containers are opened from the distribution mirror
// (`models/karume-embeddinggemma-2/`, model `text`, weights key `audio`, through its manifest —
// byte-identical to the series' containers), and their goldens are read from the test-asset root
// (fetched with `node tools/test-assets/fetch.ts`). The f32 series is a verification graph no
// distribution ships, so it stays an implementation-time series check (decision 6): without
// KARUME_SERIES=1 its cases SKIP explicitly and the series is not read.
//
// Upstream parity is opt-in (helpers/upstream-parity.ts): for the shipped series the default run
// keeps the asset completeness gate, the per-case structure checks (io keys, the inputs' shapes at
// the waveform's soft-token count, output count / shape / dtype on the real GPU, the warm runs'
// bits), the sha256 rows and the capped-device case; the comparison with upstream's soft tokens
// (the allclose and the token cosine) runs only with KARUME_UPSTREAM_PARITY=1. The f32 series
// keeps that comparison in its one case (it already runs only on request).
//
// Where an asset is missing the cases SKIP explicitly; an asset that exists only partly FAILs (the
// completeness test). A golden of a shipped series absent from a complete fetch of the pinned
// dataset (`testAssetsFetched()`) is a FAIL, not a SKIP: the dataset at the pin lacks it. A missing
// mirror SKIPs (the distribution gate owns that red).

import { type DirEntry, isNotFound, readDirSync, readFile } from "@karume/test-support/fs";
import { goldenDir, TEST_ASSETS, testAssetsFetched } from "@karume/test-support/test-assets";
import { test } from "vite-plus/test";
import { assert, assertEquals } from "@std/assert";
import { acquireGpu, parseSafetensors, prepareContainer, type Tensor } from "../../runtime/mod.ts";
import {
  compareTensors,
  formatAllclose,
  type Tolerance,
} from "../../runtime/src/reference/allclose.ts";
import { LIMIT_CAPS } from "../../runtime/src/gpu/device.ts";
import { assertAdapterMatchesEnvironment } from "../../runtime/tests/helpers/environment.ts";
import { ioTensor } from "../../runtime/tests/helpers/golden-io.ts";
import { GPU_AVAILABLE } from "../../runtime/tests/helpers/gpu.ts";
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
import { modelPresent, openSeriesContainer } from "../../runtime/tests/helpers/container-files.ts";
import { seriesGraph } from "../../models/tests/helpers/series-graphs.ts";
import {
  mirrorPresent,
  mirrorRoot,
  openSelectedContainer,
  selectMirrorComponent,
} from "../../models/tests/helpers/mirror.ts";
import { SERIES_CHECKS, SERIES_CHECKS_OPT_IN } from "../../models/tests/helpers/series-checks.ts";
import { UPSTREAM_PARITY, UPSTREAM_PARITY_OPT_IN } from "./helpers/upstream-parity.ts";

/** How one golden waveform is held to upstream's soft tokens (every series — module doc). */
type AudioCheck = {
  /** Its soft-token count n. */
  readonly tokens: number;
  /** The allclose of the whole `[1, n, 512]` output against the golden. */
  readonly tolerance: Tolerance;
  /** The floor of the cosine of every soft token (one `[512]` row) to the golden's. */
  readonly minTokenCosine: number;
};

/**
 * The checks of each golden waveform of the f32 and the bf16 series (their weights hold the same
 * values, so they have the same goldens; on this machine the two series' outputs are
 * bit-identical).
 *
 * Measured on node-nvidia-geforce-rtx-3080-ti (2026-11-22, plain comparison, `[1, n, 512]` soft
 * tokens whose values reach 5.5 to 6.1; the min token cosine 2026-11-23). The torch columns place
 * the GPU's error: the same graph in torch f64 on the same frames (upstream's f32 casts lifted; it
 * equals upstream's tower in f64 to 1.3e-12) is the reference the f32 runs scatter around. "Min
 * token cosine" is the lowest cosine of one soft token to the golden's, `1 − cosine` shown:
 *
 * | Waveform | n   | GPU vs golden | GPU vs torch f64 | torch f32 vs f64 | golden vs f64 | Min token cosine, GPU vs golden |
 * | -------- | --- | ------------- | ---------------- | ---------------- | ------------- | ------------------------------- |
 * | sweep    | 25  | 1.41e-5       | 7.14e-6          | 5.89e-6          | 2.12e-5       | 1 − 3.66e-12                    |
 * | noise    | 63  | 5.82e-4       | 3.54e-4          | 1.76e-4          | 2.27e-4       | 1 − 6.96e-9                     |
 * | click    | 25  | 1.55e-5       | 1.35e-5          | 5.35e-6          | 5.92e-6       | 1 − 5.28e-12                    |
 * | edge     | 26  | 8.12e-5       | 1.32e-4          | 3.04e-5          | 5.04e-5       | 1 − 1.49e-10                    |
 * | tones    | 250 | 1.33e-4       | 2.35e-4          | 3.36e-4          | 2.35e-4       | 1 − 3.14e-10                    |
 * | long     | 750 | 2.00e-4       | 4.98e-4          | 2.56e-4          | 3.01e-4       | 1 − 6.37e-10                    |
 *
 * The tower is sensitive in f32 (12 conformer layers whose activations reach about 50 before the
 * projector): torch's own f32 run is up to 3.4e-4 from f64, and upstream's golden up to 3.0e-4. The
 * GPU is up to 5.0e-4 from f64 — the same order, another summation order; against the golden it is
 * furthest on `noise` (5.8e-4), where the two scatter to opposite sides. Each waveform is held at
 * about three times its own measured error, both for the whole output (atol) and for its worst
 * soft token (`1 − cosine`), so a regression that keeps every value under the worst waveform's
 * error still fails on the others; one bound sized to `noise` would leave `sweep` and `click` about
 * 40 times headroom. The breakages this guards against are far above it: the frame mask dropped
 * moves `edge`'s last token by 4.2 (L ≡ 2 mod 4 — the recipe checks it), a window of 13 keys
 * instead of 12 moves the soft tokens by about 0.4 (random frames in torch, measured before the
 * recipe was written), and even one extra key unmasked at the first token of the first layer
 * alone moves `sweep` by 3.6e-4 and `tones` by 5.6e-3 (8 and 14 times their bounds; one atol of
 * 2e-3 for all six let it pass — the recipe's tiny tower pins the window's keys across chunk
 * boundaries). rtol 0: the values are of one scale, atol alone decides.
 */
const AUDIO_CHECKS: Readonly<Record<string, AudioCheck>> = {
  click: { tokens: 25, tolerance: { atol: 4.6e-5, rtol: 0 }, minTokenCosine: 1 - 1.6e-11 },
  edge: { tokens: 26, tolerance: { atol: 2.4e-4, rtol: 0 }, minTokenCosine: 1 - 4.5e-10 },
  long: { tokens: 750, tolerance: { atol: 6e-4, rtol: 0 }, minTokenCosine: 1 - 1.9e-9 },
  noise: { tokens: 63, tolerance: { atol: 1.75e-3, rtol: 0 }, minTokenCosine: 1 - 2.1e-8 },
  sweep: { tokens: 25, tolerance: { atol: 4.2e-5, rtol: 0 }, minTokenCosine: 1 - 1.1e-11 },
  tones: { tokens: 250, tolerance: { atol: 4e-4, rtol: 0 }, minTokenCosine: 1 - 9.4e-10 },
};

/**
 * The checks of each golden waveform of the **i8 series** against its own goldens (upstream's tower
 * over the same frames with the weights the series stores — every linear and depthwise convolution
 * rounded to per-row int8 before the goldens were taken, the two subsampling convolutions kept
 * bf16, so the rounding is in the goldens and this is the implementation error alone, ADR 0006).
 * Its io inputs are the f32 series' byte for byte.
 *
 * Measured on node-nvidia-geforce-rtx-3080-ti (2026-11-23, plain comparison, `1 − cosine` shown for
 * the min token cosine; the last two columns are how far the rounding itself moves the soft tokens
 * — the i8 golden against the f32 golden):
 *
 * | Waveform | n   | GPU vs golden | f32 series, GPU vs its golden | Min token cosine, GPU vs golden | rounding, max abs | rounding, min token cosine |
 * | -------- | --- | ------------- | ----------------------------- | ------------------------------- | ----------------- | -------------------------- |
 * | sweep    | 25  | 5.53e-5       | 1.41e-5                       | 1 − 3.74e-11                    | 0.0706            | 1 − 1.46e-4                |
 * | noise    | 63  | 2.49e-4       | 5.82e-4                       | 1 − 1.11e-9                     | 1.47              | 1 − 4.79e-2                |
 * | click    | 25  | 1.03e-5       | 1.55e-5                       | 1 − 2.25e-12                    | 0.0927            | 1 − 2.22e-4                |
 * | edge     | 26  | 4.10e-5       | 8.12e-5                       | 1 − 5.29e-11                    | 0.208             | 1 − 1.36e-3                |
 * | tones    | 250 | 9.39e-5       | 1.33e-4                       | 1 − 1.71e-10                    | 0.545             | 1 − 9.66e-3                |
 * | long     | 750 | 2.39e-4       | 2.00e-4                       | 1 − 9.78e-10                    | 3.03              | 1 − 0.147                  |
 *
 * The GPU sits from the i8 goldens at the order it sits from the f32 goldens with the f32 weights
 * (the same f32 noise of the same tower, scattered differently: `sweep` about four times further,
 * `noise` under half as far). Each waveform is held at about three times its own measured error, as
 * the f32 series' are (and for the same reason). What it guards against is far above it: an asset
 * of another storage in this series' place moves the soft tokens by the rounding, 0.071 to 3.0 — at
 * least 400 times each waveform's bound — and the breakages {@link AUDIO_CHECKS} names (measured on
 * the f32 weights) are above these bounds too, the smallest, one extra key on `sweep`, twice its.
 */
const AUDIO_I8_CHECKS: Readonly<Record<string, AudioCheck>> = {
  click: { tokens: 25, tolerance: { atol: 3.1e-5, rtol: 0 }, minTokenCosine: 1 - 6.8e-12 },
  edge: { tokens: 26, tolerance: { atol: 1.25e-4, rtol: 0 }, minTokenCosine: 1 - 1.6e-10 },
  long: { tokens: 750, tolerance: { atol: 7.2e-4, rtol: 0 }, minTokenCosine: 1 - 3e-9 },
  noise: { tokens: 63, tolerance: { atol: 7.5e-4, rtol: 0 }, minTokenCosine: 1 - 3.4e-9 },
  sweep: { tokens: 25, tolerance: { atol: 1.7e-4, rtol: 0 }, minTokenCosine: 1 - 1.15e-10 },
  tones: { tokens: 250, tolerance: { atol: 2.85e-4, rtol: 0 }, minTokenCosine: 1 - 5.2e-10 },
};

/** The generate command a SKIP prints (the same spelling as tools/export-recipes/embeddinggemma2/README.md). */
const GENERATE_COMMAND =
  "cd tools/export-recipes && uv run --with 'transformers==5.23.0' python -m embeddinggemma2.export_audio";

/** The command that fetches the goldens of a shipped series (pasted into the skip warnings). */
const FETCH_COMMAND = "node tools/test-assets/fetch.ts";

/** The distribution that ships the audio tower (`models/karume-embeddinggemma-2/`). */
const MIRROR_ROOT = mirrorRoot("karume-embeddinggemma-2");
/** The mirror's model and the weights key of the tower (also its graph name — container-v1 §2.1). */
const MIRROR_MODEL = "text";
const MIRROR_GRAPH = "audio";

/**
 * Where a series' container and goldens are read from: an implementation-time series
 * (`outputs/series/`, read only under SERIES_CHECKS), or the distribution mirror's preset through
 * its manifest with the goldens from the test-asset root.
 */
type SeriesSource =
  | {
      readonly kind: "series";
      /** The command that writes it. */
      readonly generate: string;
      /** What the series holds, for the SKIP message. */
      readonly weights: string;
    }
  | {
      readonly kind: "mirror";
      /** The mirror's preset whose tower container is the series' (byte for byte). */
      readonly quant: string;
    };

/**
 * One series this file checks. MUST: each series has its own checks (ADR 0019 — sharing one table
 * would hold the i8 assets to the bands of the f32 goldens, or the reverse).
 */
type SeriesSpec = {
  /**
   * The series name: its directory under `outputs/series/` or `goldenDir(name)`, and the prefix of
   * its sha256 row IDs (frozen — ADR 0139 decision 7).
   */
  readonly name: string;
  /** How each golden waveform of the series is held to its golden. */
  readonly checks: Readonly<Record<string, AudioCheck>>;
  readonly source: SeriesSource;
};

const SERIES: readonly SeriesSpec[] = [
  {
    name: "embeddinggemma-2-audio",
    checks: AUDIO_CHECKS,
    source: {
      kind: "series",
      generate: GENERATE_COMMAND,
      weights: "f32 weights of about 1.1 GiB",
    },
  },
  {
    name: "embeddinggemma-2-audio-bf16",
    checks: AUDIO_CHECKS,
    source: { kind: "mirror", quant: "bf16" },
  },
  {
    name: "embeddinggemma-2-audio-i8",
    checks: AUDIO_I8_CHECKS,
    source: { kind: "mirror", quant: "i8" },
  },
];

const MODEL_FILE = "model.krm";
const IO_PREFIX = "io.";
const IO_SUFFIX = ".safetensors";

/** The graph's one symbol (the soft-token count) and the mel frames per soft token. */
const SYMBOL = "n";
const FRAMES_PER_TOKEN = 4;

/**
 * The cases that should have been generated are the keys of {@link AUDIO_CHECKS}, each with its
 * soft-token count n. Pinned here rather than taken from the listing, so a partial generation fails
 * instead of losing tests silently; the source of truth is `WAVEFORMS` of
 * `tools/export-recipes/embeddinggemma2/waveforms.py`. 750 is the most upstream's front end gives
 * (its 30 s truncation).
 */
const EXPECTED_CASES = Object.keys(AUDIO_CHECKS).sort();

/** The golden the capped-device case runs (the largest soft-token count). */
const CAPPED_CASE = "long";
/** The binding and buffer limit of the capped device: the WebGPU default. */
const CAPPED_BYTES = 128 * 2 ** 20;

/** The runs after the first one each case times (warm: the plan and the pipelines exist). */
const WARM_RUNS = 5;

/** The lowest cosine of one soft token to the golden's (`[n, 512]` rows, in f64). */
const minTokenCosine = (
  output: Float32Array<ArrayBuffer>,
  golden: Float32Array<ArrayBuffer>,
  width: number,
): number => {
  assertEquals(output.length, golden.length, "token cosine of outputs of two sizes");
  let lowest = 1;
  for (let at = 0; at < output.length; at += width) {
    let dot = 0;
    let left = 0;
    let right = 0;
    for (let index = at; index < at + width; index += 1) {
      dot += output[index] * golden[index];
      left += output[index] * output[index];
      right += golden[index] * golden[index];
    }
    lowest = Math.min(lowest, dot / Math.sqrt(left * right));
  }
  return lowest;
};

/**
 * Lists the asset directory. It falls back to empty only when the directory does not exist.
 * MUST: anything but NotFound propagates — reading a permission error as "no asset" would turn a
 * check that never ran into a quiet SKIP.
 */
const listDir = (url: URL): readonly DirEntry[] => {
  try {
    return [...readDirSync(url)];
  } catch (cause) {
    if (isNotFound(cause)) {
      return [];
    }
    throw cause;
  }
};

const discoverCases = (root: URL): readonly string[] =>
  listDir(root)
    .filter(
      (entry) => entry.isFile && entry.name.startsWith(IO_PREFIX) && entry.name.endsWith(IO_SUFFIX),
    )
    .map((entry) => entry.name.slice(IO_PREFIX.length, entry.name.length - IO_SUFFIX.length))
    .sort();

const readBuffer = async (root: URL, file: string): Promise<ArrayBuffer> => {
  const bytes = await readFile(new URL(file, root));
  return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
};

/** A declared shape with the symbol `n` read as `tokens` (`"4n"` as 4 · tokens). */
const shapeAt = (shape: readonly (number | string)[], tokens: number): number[] =>
  shape.map((dim) => {
    if (typeof dim === "number") return dim;
    if (dim === SYMBOL) return tokens;
    if (dim === `${FRAMES_PER_TOKEN}${SYMBOL}`) return FRAMES_PER_TOKEN * tokens;
    throw new Error(`a dimension '${dim}' outside the graph's contract`);
  });

/**
 * Where the settlements and measurements go
 * (`outputs/verify/<environment key>/<date>_embeddinggemma2-audio/` — safe to delete).
 */
const results = openResults("embeddinggemma2-audio");

/** The sha256 reference rows of the output (rows per environment; the row ID is `<series>/<waveform>`). */
const references = openReferences(
  new URL(
    "../../runtime/tests/fixtures/references/embeddinggemma2-audio-golden.json",
    import.meta.url,
  ),
);

const milliseconds = (start: number): number => Math.round(performance.now() - start);
const mebibytes = (bytes: number): string => `${(bytes / 2 ** 20).toFixed(1)} MiB`;
const median = (values: readonly number[]): number => {
  const sorted = [...values].sort((left, right) => left - right);
  return sorted[Math.floor(sorted.length / 2)];
};

/** One opened series case: the parsed container, its golden io and the graph's inputs bound. */
type OpenedCase = {
  readonly parsed: ReturnType<typeof prepareContainer>;
  readonly io: ReturnType<typeof parseSafetensors>;
  readonly inputs: Record<string, Tensor>;
};

/** Whether the test-asset root holds a complete fetch of the pin (a missing golden then FAILs). */
const FETCHED = testAssetsFetched();

/** Appended to a mismatch in a completely fetched golden set (the missing ones are the pin's). */
const AT_PIN = ` (outputs/test-assets/ holds a complete fetch of ${TEST_ASSETS.repo}@${TEST_ASSETS.revision}: the dataset at the pin lacks the missing ones)`;

/** The failure of a golden that a complete fetch of the pinned dataset does not hold. */
const missingAtPin = (path: string): Error =>
  new Error(
    `${path} is absent, but outputs/test-assets/ holds a complete fetch of ` +
      `${TEST_ASSETS.repo}@${TEST_ASSETS.revision}: the dataset at the pin lacks it`,
  );

/** The directory of a series' io goldens (`io.<waveform>.safetensors`). */
const goldensOf = (spec: SeriesSpec): URL =>
  spec.source.kind === "mirror"
    ? goldenDir(spec.name)
    : new URL(`../../../outputs/series/${spec.name}/`, import.meta.url);

/** The representative path of a series' own container (series sources only). */
const seriesModel = (spec: SeriesSpec): URL => new URL(MODEL_FILE, goldensOf(spec));

/**
 * Reads one golden. A mirror series' golden (a test asset) is read only when present or when the
 * fetch is complete, so its NotFound here is a file the dataset at the pin lacks — reported as such.
 */
const readGolden = async (spec: SeriesSpec, caseName: string): Promise<ArrayBuffer> => {
  const root = goldensOf(spec);
  const file = `${IO_PREFIX}${caseName}${IO_SUFFIX}`;
  try {
    return await readBuffer(root, file);
  } catch (cause) {
    if (spec.source.kind === "mirror" && isNotFound(cause)) {
      throw missingAtPin(new URL(file, root).pathname);
    }
    throw cause;
  }
};

/** Opens the series' container (part 0 now; the blocks when the Session reads them). */
const openSeries = async (spec: SeriesSpec) =>
  spec.source.kind === "series"
    ? prepareContainer(await openSeriesContainer(seriesModel(spec)), seriesGraph(spec.name))
    : prepareContainer(
        await openSelectedContainer(
          await selectMirrorComponent(MIRROR_ROOT, MIRROR_GRAPH, {
            model: MIRROR_MODEL,
            quant: spec.source.quant,
          }),
        ),
        MIRROR_GRAPH,
      );

const openCase = async (spec: SeriesSpec, caseName: string): Promise<OpenedCase> => {
  const [parsed, ioBytes] = await Promise.all([openSeries(spec), readGolden(spec, caseName)]);
  const io = parseSafetensors(ioBytes);
  const { tokens } = AUDIO_CHECKS[caseName];

  // Every tensor of io matches the graph's inputs and outputs exactly.
  const expectedKeys = [
    ...parsed.graph.inputs.map((input) => `input.${input.name}`),
    ...parsed.graph.outputs.map((_, index) => `output.${index}`),
  ].sort();
  assertEquals([...io.tensors.keys()].sort(), expectedKeys, "tensor keys of io.safetensors");
  assertEquals(parsed.graph.symbols, [SYMBOL], "the graph's symbols");

  // n is bound from the golden's input shape (no explicit bindings): each input's declared shape at
  // the case's soft-token count, or the case no longer exercises its waveform.
  const inputs: Record<string, Tensor> = {};
  for (const input of parsed.graph.inputs) {
    const view = io.tensors.get(`input.${input.name}`);
    assert(view !== undefined, `input.${input.name} is missing from the golden`);
    assertEquals(view.shape, shapeAt(input.shape, tokens), `input.${input.name}: shape`);
    inputs[input.name] = ioTensor(io, view, input.dtype);
  }
  return { parsed, io, inputs };
};

/** The uncapped device's output of {@link CAPPED_CASE}, per series (the capped case compares). */
const uncappedOutputs = new Map<string, Tensor["data"]>();

/** The output of one golden case on the real GPU, with the golden's soft tokens beside it. */
type GoldenRun = {
  readonly name: string;
  readonly where: string;
  readonly output: Tensor;
  readonly golden: Tensor;
  readonly timing: string;
};

/**
 * Loads one golden case, runs it on the GPU and asserts everything except the comparison with
 * upstream's soft tokens (io keys, the graph's symbol, the inputs' shapes at the waveform's
 * soft-token count, output count / shape / dtype, and `warmRuns` more runs each equal to the first
 * bit for bit — the sha row stands on that). The default sha case, the opt-in parity case and the
 * f32 series' one case all go through here.
 */
const runGoldenCase = async (
  spec: SeriesSpec,
  caseName: string,
  { warmRuns }: { readonly warmRuns: number },
): Promise<GoldenRun> => {
  const { parsed, io, inputs } = await openCase(spec, caseName);
  const { tokens } = spec.checks[caseName];
  const gpu = await acquireGpu();
  try {
    assertAdapterMatchesEnvironment(gpu);
    const createStart = performance.now();
    const session = await parsed.createContainerSession(gpu);
    const createMs = milliseconds(createStart);
    try {
      const firstStart = performance.now();
      const outputs = await session.run(inputs);
      const firstRunMs = milliseconds(firstStart);
      assertEquals(parsed.graph.outputs.length, 1, "graph output count");
      assertEquals(Object.keys(outputs).sort(), [...parsed.graph.outputs].sort());
      const [name] = parsed.graph.outputs;
      const where = `${spec.name} ${caseName} output.0 ('${name}')`;
      const warmMs: number[] = [];
      for (let run = 0; run < warmRuns; run += 1) {
        const start = performance.now();
        const again = await session.run(inputs);
        warmMs.push(performance.now() - start);
        // The sha row stands on the Session's own determinism.
        assert(again[name].dtype === "f32", `${where}: run ${run + 2}'s dtype`);
        assertEquals(again[name].data, outputs[name].data, `${where}: run ${run + 2} differs`);
      }
      const { weights, planBacking } = session.diagnostics();
      const warm =
        warmRuns === 0
          ? ""
          : ` warm runs median ${median(warmMs).toFixed(1)} ms / min ${Math.min(...warmMs).toFixed(1)} ms` +
            ` (${warmRuns} runs),`;
      const timing =
        `n=${tokens}: create ${createMs} ms, first run ${firstRunMs} ms,` +
        `${warm} weights ${mebibytes(weights.allocatedBytes)},` +
        ` plan backing ${mebibytes(planBacking.residentBytes)}` +
        ` + inputs ${mebibytes(planBacking.inputBytes)}`;
      console.log(`[embeddinggemma2-audio] ${spec.name} ${caseName} ${timing}`);
      const view = io.tensors.get("output.0");
      assert(view !== undefined, "output.0 is missing from the golden");
      const declared = parsed.graph.values[name].dtype;
      const output = outputs[name];
      assertEquals(output.shape, [1, tokens, 512], `${where}: shape`);
      assertEquals(output.shape, view.shape, `${where}: the golden's shape`);
      assertEquals(output.dtype, declared, `${where}: dtype`);
      return { name, where, output, golden: ioTensor(io, view, declared), timing };
    } finally {
      await session.dispose();
    }
  } finally {
    gpu.destroy();
  }
};

/**
 * The comparison with upstream's soft tokens: the allclose of the whole output and the floor of the
 * token cosine (measurements recorded when given). Returns the line the parity case keeps as its
 * note (the token cosine has no field of a measurement).
 */
const compareWithGolden = (
  spec: SeriesSpec,
  caseName: string,
  run: GoldenRun,
  measurements?: Measurement[],
): string => {
  const check = spec.checks[caseName];
  const { name, where, output, golden } = run;
  const report = compareTensors(output, golden, check.tolerance);
  measurements?.push({
    output: name,
    maxAbs: report.maxAbsError,
    maxRel: report.maxRelError,
    tolerance: check.tolerance,
    stage: "karume",
  });
  assert(output.dtype === "f32" && golden.dtype === "f32", `${where}: not f32`);
  const tokenCosine = minTokenCosine(output.data, golden.data, 512);
  const measured =
    `maxAbs ${report.maxAbsError.toExponential(3)},` +
    ` min token cosine 1 - ${(1 - tokenCosine).toExponential(3)}`;
  console.log(`[embeddinggemma2-audio] ${spec.name} ${caseName} ${measured}`);
  assert(report.pass, `${where}: ${formatAllclose(report)}`);
  assert(
    tokenCosine >= check.minTokenCosine,
    `${where}: a soft token's cosine to the golden's is ${tokenCosine}, under ${check.minTokenCosine}`,
  );
  return measured;
};

/**
 * Settles the output against this machine's sha256 row and records the case; fails after the
 * record on a mismatch (throwing first would keep the measured sha out of the results). `compare`
 * runs the comparison with upstream's soft tokens first (the f32 series' case), so no row is
 * created from an output that broke it. The uncapped output of {@link CAPPED_CASE} is kept for the
 * capped case.
 */
const settleCase = async (
  spec: SeriesSpec,
  caseName: string,
  { compare }: { readonly compare: boolean },
): Promise<void> => {
  const caseId = `${spec.name}/${caseName}`;
  let settlement: ReferenceSettlement | undefined;
  await runRecordedCase(results, { id: caseId }, async ({ measurements }) => {
    const run = await runGoldenCase(spec, caseName, { warmRuns: WARM_RUNS });
    if (compare) compareWithGolden(spec, caseName, run, measurements);
    const { name, output } = run;
    assert(output.dtype === "f32", `${run.where}: the artifact's dtype is ${output.dtype}`);
    if (caseName === CAPPED_CASE) {
      uncappedOutputs.set(spec.name, output.data);
    }
    const outcome = await settleOrObserve(references, results, {
      id: caseId,
      artifact: `${spec.name}_${caseName}.safetensors`,
      bytes: f32ArtifactBytes(name, output.shape, output.data),
    });
    settlement = outcome.settlement;
    return { ...outcome.fields, note: run.timing };
  });
  if (settlement?.check.status === "fail") {
    throw new Error(referenceMismatchMessage(caseId, settlement, references));
  }
};

/** Registers the completeness test and the golden cases of one series; returns its case IDs. */
const registerSeries = (spec: SeriesSpec): readonly string[] => {
  const { source } = spec;
  const root = goldensOf(spec);
  /**
   * Whether this run may read the series' goldens and container: always for the mirror, only under
   * SERIES_CHECKS for a series (MUST: the default lanes do not touch `outputs/series/`).
   */
  const readable = source.kind === "mirror" || SERIES_CHECKS;
  const suffix = source.kind === "series" ? SERIES_CHECKS_OPT_IN : "";
  /** Listed synchronously because registration needs it. */
  const found = readable ? discoverCases(root) : [];
  const hasContainer =
    readable &&
    (source.kind === "series" ? modelPresent(seriesModel(spec)) : mirrorPresent(MIRROR_ROOT));
  /**
   * A mirror series' goldens are all absent from a complete fetch of the pin: the dataset at the
   * pin lacks them, so the cases run (and FAIL by path) instead of skipping.
   */
  const missingAtPinned = source.kind === "mirror" && found.length === 0 && FETCHED;
  /**
   * Whether the assets exist (none at all = not generated or not fetched here, so every case
   * SKIPs; a partial absence is on the FAIL side). A mirror series also needs its mirror; a
   * series' missing container is caught by the completeness case instead.
   */
  const available =
    (found.length > 0 || missingAtPinned) && (source.kind === "series" || hasContainer);
  /** Whether anything at all is left — the SKIP predicate of the completeness test. */
  const anyPresent =
    found.length > 0 || (source.kind === "series" && hasContainer) || missingAtPinned;
  /** The golden files present, or else the expected set, so that each case SKIPs by name. */
  const cases = found.length > 0 ? found : EXPECTED_CASES;

  if (readable && found.length === 0 && !missingAtPinned) {
    console.warn(
      source.kind === "series"
        ? `[karume] ${root.pathname} has no exported asset, so the EmbeddingGemma 2 audio E2E of ${spec.name} is ` +
            `SKIPPED (${source.weights}, kept outside the repository). Generate: ${source.generate}`
        : `[karume] skipping the EmbeddingGemma 2 audio E2E of ${spec.name}: no goldens under ` +
            `${root.pathname}. Fetch: ${FETCH_COMMAND}`,
    );
  }
  if (source.kind === "mirror" && !hasContainer) {
    console.warn(
      `[karume] skipping the EmbeddingGemma 2 audio E2E of ${spec.name}: the distribution mirror ` +
        `${MIRROR_ROOT.pathname} is absent (or not karume/6).`,
    );
  }

  test.skipIf(!anyPresent)(
    source.kind === "series"
      ? `EmbeddingGemma 2 audio asset ${spec.name}: the expected cases and the model itself are all present${suffix}`
      : `EmbeddingGemma 2 audio asset ${spec.name}: the expected golden cases are all present`,
    () => {
      if (missingAtPinned) throw missingAtPin(root.pathname);
      assertEquals(
        found,
        EXPECTED_CASES,
        `golden cases of ${root.pathname}` + (source.kind === "mirror" && FETCHED ? AT_PIN : ""),
      );
      // The mirror's presence is not asserted here: it is a separate source, whose absence SKIPs
      // the cases (and whose completeness is the distribution gate's).
      if (source.kind === "series") {
        assert(modelPresent(seriesModel(spec)), `${MODEL_FILE} is missing`);
      }
    },
  );

  for (const caseName of cases) {
    if (source.kind === "series") {
      test.skipIf(!available || !GPU_AVAILABLE)(
        `EmbeddingGemma 2 audio golden comparison: ${spec.name} ${caseName} (real GPU / upstream's soft tokens in torch CPU, then the sha256 row)${suffix}`,
        async () => {
          await settleCase(spec, caseName, { compare: true });
        },
      );
      continue;
    }
    // Default run: everything except the comparison with upstream's soft tokens, then the row.
    test.skipIf(!available || !GPU_AVAILABLE)(
      `EmbeddingGemma 2 audio golden: ${spec.name} ${caseName} (real GPU / io keys, input shapes, output shape and dtype, warm runs' bits, then the sha256 row)`,
      async () => {
        await settleCase(spec, caseName, { compare: false });
      },
    );
    // Opt-in: the comparison with upstream's soft tokens under the waveform's checks. Its
    // results.json entry is `<series>/<waveform>/parity` (the bare ID is the default sha case's).
    test.skipIf(!UPSTREAM_PARITY || !available || !GPU_AVAILABLE)(
      `EmbeddingGemma 2 audio golden parity: ${spec.name} ${caseName} (real GPU / upstream's soft tokens in torch CPU)${UPSTREAM_PARITY_OPT_IN}`,
      async () => {
        await runRecordedCase(
          results,
          { id: `${spec.name}/${caseName}/parity` },
          async ({ measurements }) => {
            const run = await runGoldenCase(spec, caseName, { warmRuns: 0 });
            const measured = compareWithGolden(spec, caseName, run, measurements);
            return { status: "pass", note: `${measured}; ${run.timing}` };
          },
        );
      },
    );
  }

  test.skipIf(!available || !GPU_AVAILABLE)(
    `EmbeddingGemma 2 audio within the WebGPU default limits: ${spec.name} ${CAPPED_CASE} on a device capped at ${mebibytes(CAPPED_BYTES)} gives the uncapped device's bits (real GPU)${suffix}`,
    async () => {
      const uncapped = uncappedOutputs.get(spec.name);
      assert(uncapped !== undefined, `the uncapped run of ${CAPPED_CASE} did not pass first`);
      const { parsed, inputs } = await openCase(spec, CAPPED_CASE);
      const gpu = await acquireGpu({
        [LIMIT_CAPS]: { maxStorageBufferBindingSize: CAPPED_BYTES, maxBufferSize: CAPPED_BYTES },
      });
      try {
        assertEquals(
          [gpu.limits.maxStorageBufferBindingSize, gpu.limits.maxBufferSize],
          [CAPPED_BYTES, CAPPED_BYTES],
          "the device's limits are not narrowed (the check would see nothing)",
        );
        const session = await parsed.createContainerSession(gpu);
        try {
          const outputs = await session.run(inputs);
          const [name] = parsed.graph.outputs;
          assertEquals(outputs[name].data, uncapped, `${spec.name} ${CAPPED_CASE}: capped run`);
        } finally {
          await session.dispose();
        }
      } finally {
        gpu.destroy();
      }
    },
  );
  return available ? cases.map((caseName) => `${spec.name}/${caseName}`) : [];
};

test("EmbeddingGemma 2 audio: every series holds the same golden waveforms, each at the same soft-token count", () => {
  for (const spec of SERIES) {
    assertEquals(Object.keys(spec.checks).sort(), EXPECTED_CASES, `${spec.name}: its waveforms`);
    for (const name of EXPECTED_CASES) {
      assertEquals(spec.checks[name].tokens, AUDIO_CHECKS[name].tokens, `${spec.name} ${name}`);
    }
  }
});

/** The IDs of the registered cases of every series (the set the reference gate counts). */
const caseIds = SERIES.flatMap(registerSeries);

const RUNNABLE = caseIds.length > 0 && GPU_AVAILABLE;
if (RUNNABLE) references.warnMissing(caseIds);
registerReferenceGate(references, { runnable: RUNNABLE, caseIds });
