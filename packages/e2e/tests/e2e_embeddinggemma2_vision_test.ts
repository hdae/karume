// Real-GPU golden E2E of the vision tower of EmbeddingGemma 2 (an image's patches → its soft
// tokens) (step 3 of ADR 0005).
//
// The targets are three series of `tools/export-recipes/embeddinggemma2/export_vision.py`:
// `embeddinggemma-2-vision` (f32 weights, about 640 MiB; its `generate` in SERIES below is the
// source of truth for the command), `embeddinggemma-2-vision-bf16` (the distribution's `bf16`
// preset: the linears and the two learned position tables as upstream's bf16, about 320 MiB) and
// `embeddinggemma-2-vision-i8` (the distribution's `i8` preset: those tensors rounded to per-row
// int8 before the goldens were taken, about 161 MiB). One graph `vision`: the
// patch rows `pixels [1, 9n, 768]` and their positions `position_x` / `position_y [1, 9n]` in →
// the soft tokens `[1, n, 512]` out, `n` symbolic (the image's soft-token count). The rows are only
// the image's real patches, each soft token's 9 patches adjacent (the io contract in the recipe's
// module doc and README); the goldens are upstream's own soft tokens (its processor and its tower
// in torch on CPU, f32, over the processor's padded rows), on five synthetic images — a square, a
// wide and a tall aspect, the smallest soft-token count the resize rule produces, and its longest
// side (840 patches, so the positions reach the last row of the tower's tables).
//
// Every case ends with a sha256 check of the output against this machine's reference row (ADR
// 0106; the row ID is `<series>/<image>`). Rows are created with KARUME_REFERENCE=write.
//
// ## Where the assets come from (ADR 0139)
//
// | Series                         | Container                                 | Goldens                                     | Runs                      |
// | ------------------------------ | ----------------------------------------- | ------------------------------------------- | ------------------------- |
// | `embeddinggemma-2-vision-bf16` | the mirror's `bf16` preset                | `goldenDir("embeddinggemma-2-vision-bf16")` | default lane              |
// | `embeddinggemma-2-vision-i8`   | the mirror's `i8` preset                  | `goldenDir("embeddinggemma-2-vision-i8")`   | default lane              |
// | `embeddinggemma-2-vision`      | `outputs/series/embeddinggemma-2-vision/` | the same series directory                   | only with KARUME_SERIES=1 |
//
// The bf16 and int8 series are shipped: their containers are opened from the distribution mirror
// (`models/karume-embeddinggemma-2/`, model `text`, weights key `vision`, through its manifest —
// byte-identical to the series' containers), and their goldens are read from the test-asset root
// (fetched with `node tools/test-assets/fetch.ts`). The f32 series is a verification graph no
// distribution ships, so it stays an implementation-time series check (decision 6): without
// KARUME_SERIES=1 its cases SKIP explicitly and the series is not read.
//
// Upstream parity is opt-in (helpers/upstream-parity.ts): for the shipped series the default run
// keeps the asset completeness gate, the per-case structure checks (io keys, the inputs' shapes at
// the image's soft-token count, the positions' reach, output count / shape / dtype on the real
// GPU) and the sha256 rows; the comparison with upstream's soft tokens (the allclose and the token
// cosine) runs only with KARUME_UPSTREAM_PARITY=1. The f32 series keeps that comparison in its one
// case (it already runs only on request).
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

/** How one golden image is held to upstream's soft tokens (every series — module doc). */
type ImageCheck = {
  /** Its soft-token count n. */
  readonly tokens: number;
  /** The allclose of the whole `[1, n, 512]` output against the golden. */
  readonly tolerance: Tolerance;
  /** The floor of the cosine of every soft token (one `[512]` row) to the golden's. */
  readonly minTokenCosine: number;
};

/**
 * The checks of each golden image of the f32 and the bf16 series (their weights hold the same
 * values, so they have the same goldens; on this machine the two series' outputs are
 * bit-identical).
 *
 * Measured on node-nvidia-geforce-rtx-3080-ti (2026-11-22, plain comparison, `[1, n, 512]` soft
 * tokens whose values reach 5.0 to 6.2). The torch columns place the GPU's error: the same graph in
 * torch f64 on the same rows is the reference the f32 runs scatter around. "Min token cosine" is
 * the lowest cosine of one soft token to the golden's, `1 − cosine` shown:
 *
 * | Image           | n   | GPU vs golden | GPU vs torch f64 | torch f32 vs f64 | golden vs f64 | Min token cosine, GPU vs golden |
 * | --------------- | --- | ------------- | ---------------- | ---------------- | ------------- | ------------------------------- |
 * | gradient-square | 256 | 3.35e-4       | 2.56e-4          | 9.78e-5          | 1.01e-4       | 1 − 3.19e-9                     |
 * | shapes-wide     | 253 | 6.64e-5       | 6.07e-5          | 2.81e-5          | 4.00e-5       | 1 − 1.01e-10                    |
 * | noise-tall      | 250 | 6.46e-5       | 5.63e-5          | 1.93e-5          | 2.23e-5       | 1 − 6.89e-11                    |
 * | stripes-edge    | 139 | 7.37e-5       | 8.38e-5          | 5.16e-5          | 6.73e-5       | 1 − 1.32e-10                    |
 * | gradient-strip  | 280 | 1.27e-4       | 1.23e-4          | 9.21e-5          | 1.07e-4       | 1 − 4.22e-10                    |
 *
 * The tower is sensitive in f32 (16 layers over activations near 1,000, attention with scale 1):
 * torch's own f32 run is up to 9.8e-5 from f64, and upstream's golden (its f32 run over 2,520
 * padded rows) up to 1.1e-4. The GPU is up to 2.6e-4 from f64 — the same order, another summation
 * order. Each image is held at about three times its own measured error, both for the whole output
 * (atol) and for its worst soft token (`1 − cosine`), so a regression that keeps every value under
 * the worst image's error still fails on the others; a uniform bound sized to gradient-square
 * would leave the other four 13 to 15 times headroom. The breakages this guards against are far
 * above it: the rows in the processor's raster order (patches pooled into the wrong tokens) or the
 * two position axes swapped move the soft tokens by 5.9 to 6.5 (measured on gradient-square and
 * shapes-wide in torch, 2026-11-22). rtol 0: the values are of one scale, atol alone decides.
 * gradient-strip's columns reach position 839, the last row of the folded RoPE tables and the
 * highest learned-table rows any image at the budget reads.
 */
const IMAGE_CHECKS: Readonly<Record<string, ImageCheck>> = {
  "gradient-square": { tokens: 256, tolerance: { atol: 1e-3, rtol: 0 }, minTokenCosine: 1 - 1e-8 },
  "shapes-wide": { tokens: 253, tolerance: { atol: 2e-4, rtol: 0 }, minTokenCosine: 1 - 3e-10 },
  "noise-tall": { tokens: 250, tolerance: { atol: 2e-4, rtol: 0 }, minTokenCosine: 1 - 2e-10 },
  "stripes-edge": { tokens: 139, tolerance: { atol: 2.2e-4, rtol: 0 }, minTokenCosine: 1 - 4e-10 },
  "gradient-strip": {
    tokens: 280,
    tolerance: { atol: 3.8e-4, rtol: 0 },
    minTokenCosine: 1 - 1.3e-9,
  },
};

/**
 * The checks of each golden image of the **i8 series** against its own goldens (upstream's tower
 * over the same rows with the weights the series stores — every tensor the bf16 series stores as
 * bf16 rounded to per-row int8 before the goldens were taken, so the rounding is in the goldens and
 * this is the implementation error alone, ADR 0006). Its io inputs are the f32 series' byte for
 * byte.
 *
 * Measured on node-nvidia-geforce-rtx-3080-ti (2026-11-23, plain comparison, `1 − cosine` shown for
 * the min token cosine; the last two columns are how far the rounding itself moves the soft tokens
 * — the i8 golden against the f32 golden):
 *
 * | Image           | n   | GPU vs golden | f32 series, GPU vs its golden | Min token cosine, GPU vs golden | rounding, max abs | rounding, min token cosine |
 * | --------------- | --- | ------------- | ----------------------------- | ------------------------------- | ----------------- | -------------------------- |
 * | gradient-square | 256 | 3.56e-4       | 3.35e-4                       | 1 − 5.55e-9                     | 0.331             | 1 − 2.85e-3                |
 * | shapes-wide     | 253 | 8.05e-5       | 6.64e-5                       | 1 − 1.21e-10                    | 0.286             | 1 − 1.47e-3                |
 * | noise-tall      | 250 | 5.15e-5       | 6.46e-5                       | 1 − 4.90e-11                    | 0.298             | 1 − 1.98e-3                |
 * | stripes-edge    | 139 | 8.68e-5       | 7.37e-5                       | 1 − 1.99e-10                    | 0.851             | 1 − 1.77e-2                |
 * | gradient-strip  | 280 | 1.81e-4       | 1.27e-4                       | 1 − 9.56e-10                    | 0.384             | 1 − 4.41e-3                |
 *
 * The int8 storage adds no error of its own: the GPU sits from the i8 goldens at the order it sits
 * from the f32 goldens with the f32 weights (0.8 to 1.4 times), the same f32 noise of the same
 * tower. Each image is held at about three times its own measured error, as the f32 series' are
 * (and for the same reason). What it guards against is far above it: an asset of another storage in
 * this series' place (the f32 or bf16 weights, or the reverse) moves the soft tokens by the
 * rounding, 0.29 to 0.85 — at least 300 times each image's bound — and the breakages
 * {@link IMAGE_CHECKS} names move them by 5.9 to 6.5.
 */
const IMAGE_I8_CHECKS: Readonly<Record<string, ImageCheck>> = {
  "gradient-square": {
    tokens: 256,
    tolerance: { atol: 1.1e-3, rtol: 0 },
    minTokenCosine: 1 - 1.7e-8,
  },
  "shapes-wide": {
    tokens: 253,
    tolerance: { atol: 2.45e-4, rtol: 0 },
    minTokenCosine: 1 - 3.7e-10,
  },
  "noise-tall": { tokens: 250, tolerance: { atol: 1.55e-4, rtol: 0 }, minTokenCosine: 1 - 1.5e-10 },
  "stripes-edge": { tokens: 139, tolerance: { atol: 2.65e-4, rtol: 0 }, minTokenCosine: 1 - 6e-10 },
  "gradient-strip": {
    tokens: 280,
    tolerance: { atol: 5.5e-4, rtol: 0 },
    minTokenCosine: 1 - 2.9e-9,
  },
};

/** The generate command a SKIP prints (the same spelling as tools/export-recipes/embeddinggemma2/README.md). */
const GENERATE_COMMAND =
  "cd tools/export-recipes && uv run --with 'transformers==5.23.0' python -m embeddinggemma2.export_vision";

/** The command that fetches the goldens of a shipped series (pasted into the skip warnings). */
const FETCH_COMMAND = "node tools/test-assets/fetch.ts";

/** The distribution that ships the vision tower (`models/karume-embeddinggemma-2/`). */
const MIRROR_ROOT = mirrorRoot("karume-embeddinggemma-2");
/** The mirror's model and the weights key of the tower (also its graph name — container-v1 §2.1). */
const MIRROR_MODEL = "text";
const MIRROR_GRAPH = "vision";

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
  /** How each golden image of the series is held to its golden. */
  readonly checks: Readonly<Record<string, ImageCheck>>;
  readonly source: SeriesSource;
};

const SERIES: readonly SeriesSpec[] = [
  {
    name: "embeddinggemma-2-vision",
    checks: IMAGE_CHECKS,
    source: {
      kind: "series",
      generate: GENERATE_COMMAND,
      weights: "f32 weights of about 640 MiB",
    },
  },
  {
    name: "embeddinggemma-2-vision-bf16",
    checks: IMAGE_CHECKS,
    source: { kind: "mirror", quant: "bf16" },
  },
  {
    name: "embeddinggemma-2-vision-i8",
    checks: IMAGE_I8_CHECKS,
    source: { kind: "mirror", quant: "i8" },
  },
];

const MODEL_FILE = "model.krm";
const IO_PREFIX = "io.";
const IO_SUFFIX = ".safetensors";

/** The graph's one symbol (the soft-token count) and the patches per soft token. */
const SYMBOL = "n";
const PATCHES_PER_TOKEN = 9;

/**
 * The cases that should have been generated are the keys of {@link IMAGE_CHECKS}, each with its
 * soft-token count n. Pinned here rather than taken from the listing, so a partial generation fails
 * instead of losing tests silently; the source of truth is `IMAGES` of
 * `tools/export-recipes/embeddinggemma2/images.py`. 139 is the smallest count upstream's resize
 * rule produces at the 280-token budget; `gradient-strip` resizes to its longest side.
 */
const EXPECTED_CASES = Object.keys(IMAGE_CHECKS).sort();

/** The longest side the resize rule gives at the budget, in patches (the tables' row count). */
const LONGEST_SIDE = 840;

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

/** A declared shape with the symbol `n` read as `tokens` (`"9n"` as 9 · tokens). */
const shapeAt = (shape: readonly (number | string)[], tokens: number): number[] =>
  shape.map((dim) => {
    if (typeof dim === "number") return dim;
    if (dim === SYMBOL) return tokens;
    if (dim === `${PATCHES_PER_TOKEN}${SYMBOL}`) return PATCHES_PER_TOKEN * tokens;
    throw new Error(`a dimension '${dim}' outside the graph's contract`);
  });

/**
 * Where the settlements and measurements go
 * (`outputs/verify/<environment key>/<date>_embeddinggemma2-vision/` — safe to delete).
 */
const results = openResults("embeddinggemma2-vision");

/** The sha256 reference rows of the output (rows per environment; the row ID is `<series>/<image>`). */
const references = openReferences(
  new URL(
    "../../runtime/tests/fixtures/references/embeddinggemma2-vision-golden.json",
    import.meta.url,
  ),
);

const milliseconds = (start: number): number => Math.round(performance.now() - start);
const mebibytes = (bytes: number): string => `${(bytes / 2 ** 20).toFixed(1)} MiB`;

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

/** The directory of a series' io goldens (`io.<image>.safetensors`). */
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

/** The output of one golden case on the real GPU, with the golden's soft tokens beside it. */
type GoldenRun = {
  readonly name: string;
  readonly where: string;
  readonly output: Tensor;
  readonly golden: Tensor;
  /** The second run on the same Session (taken only when asked for — the sha row needs it). */
  readonly again?: Tensor;
  readonly timing: string;
};

/**
 * Loads one golden case, runs it on the GPU and asserts everything except the comparison with
 * upstream's soft tokens (io keys, the graph's symbol, the inputs' shapes at the image's soft-token
 * count, the positions' reach, output count / shape / dtype). The default sha case, the opt-in
 * parity case and the f32 series' one case all go through here.
 */
const runGoldenCase = async (
  spec: SeriesSpec,
  caseName: string,
  { repeat }: { readonly repeat: boolean },
): Promise<GoldenRun> => {
  const [parsed, ioBytes] = await Promise.all([openSeries(spec), readGolden(spec, caseName)]);
  const io = parseSafetensors(ioBytes);
  const { tokens } = spec.checks[caseName];

  // Every tensor of io matches the graph's inputs and outputs exactly.
  const expectedKeys = [
    ...parsed.graph.inputs.map((input) => `input.${input.name}`),
    ...parsed.graph.outputs.map((_, index) => `output.${index}`),
  ].sort();
  assertEquals([...io.tensors.keys()].sort(), expectedKeys, "tensor keys of io.safetensors");
  assertEquals(parsed.graph.symbols, [SYMBOL], "the graph's symbols");

  // n is bound from the golden's input shape (no explicit bindings): each input's declared shape
  // at the case's soft-token count, or the case no longer exercises its image.
  const inputs: Record<string, Tensor> = {};
  for (const input of parsed.graph.inputs) {
    const view = io.tensors.get(`input.${input.name}`);
    assert(view !== undefined, `input.${input.name} is missing from the golden`);
    assertEquals(view.shape, shapeAt(input.shape, tokens), `input.${input.name}: shape`);
    inputs[input.name] = ioTensor(io, view, input.dtype);
  }
  // Every position is inside the tables; the strip's columns reach their last row.
  const reach = Math.max(
    ...["position_x", "position_y"].map((key) => {
      const positions = inputs[key];
      assert(positions?.dtype === "i32", `${key}: not i32`);
      return Math.max(...positions.data);
    }),
  );
  assert(reach < LONGEST_SIDE, `a position ${reach} past the tables' ${LONGEST_SIDE} rows`);
  if (caseName === "gradient-strip") {
    assertEquals(reach, LONGEST_SIDE - 1, "the strip's last column");
  }

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
      let again: Record<string, Tensor> | undefined;
      let secondRun = "";
      if (repeat) {
        const secondStart = performance.now();
        again = await session.run(inputs);
        secondRun = ` second run ${milliseconds(secondStart)} ms,`;
      }
      const { weights, planBacking } = session.diagnostics();
      const timing =
        `n=${tokens}: create ${createMs} ms, first run ${firstRunMs} ms,` +
        `${secondRun} weights ${mebibytes(weights.allocatedBytes)},` +
        ` plan backing ${mebibytes(planBacking.residentBytes)}` +
        ` + inputs ${mebibytes(planBacking.inputBytes)}`;
      console.log(`[embeddinggemma2-vision] ${spec.name} ${caseName} ${timing}`);
      assertEquals(parsed.graph.outputs.length, 1, "graph output count");
      assertEquals(Object.keys(outputs).sort(), [...parsed.graph.outputs].sort());
      const [name] = parsed.graph.outputs;
      const view = io.tensors.get("output.0");
      assert(view !== undefined, "output.0 is missing from the golden");
      const where = `${spec.name} ${caseName} output.0 ('${name}')`;
      const declared = parsed.graph.values[name].dtype;
      const output = outputs[name];
      assertEquals(output.shape, [1, tokens, 512], `${where}: shape`);
      assertEquals(output.shape, view.shape, `${where}: the golden's shape`);
      assertEquals(output.dtype, declared, `${where}: dtype`);
      return {
        name,
        where,
        output,
        golden: ioTensor(io, view, declared),
        ...(again === undefined ? {} : { again: again[name] }),
        timing,
      };
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
  console.log(`[embeddinggemma2-vision] ${spec.name} ${caseName} ${measured}`);
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
 * created from an output that broke it.
 */
const settleCase = async (
  spec: SeriesSpec,
  caseName: string,
  { compare }: { readonly compare: boolean },
): Promise<void> => {
  const caseId = `${spec.name}/${caseName}`;
  let settlement: ReferenceSettlement | undefined;
  await runRecordedCase(results, { id: caseId }, async ({ measurements }) => {
    const run = await runGoldenCase(spec, caseName, { repeat: true });
    if (compare) compareWithGolden(spec, caseName, run, measurements);
    const { name, where, output, again } = run;
    // The sha row stands on the Session's own determinism.
    assert(output.dtype === "f32", `${where}: the artifact's dtype is ${output.dtype}`);
    assert(again?.dtype === "f32", `${where}: the second run's dtype`);
    assertEquals(again.data, output.data, `${where}: the second run differs`);
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
        ? `[karume] ${root.pathname} has no exported asset, so the EmbeddingGemma 2 vision E2E of ${spec.name} is ` +
            `SKIPPED (${source.weights}, kept outside the repository). Generate: ${source.generate}`
        : `[karume] skipping the EmbeddingGemma 2 vision E2E of ${spec.name}: no goldens under ` +
            `${root.pathname}. Fetch: ${FETCH_COMMAND}`,
    );
  }
  if (source.kind === "mirror" && !hasContainer) {
    console.warn(
      `[karume] skipping the EmbeddingGemma 2 vision E2E of ${spec.name}: the distribution mirror ` +
        `${MIRROR_ROOT.pathname} is absent (or not karume/6).`,
    );
  }

  test.skipIf(!anyPresent)(
    source.kind === "series"
      ? `EmbeddingGemma 2 vision asset ${spec.name}: the expected cases and the model itself are all present${suffix}`
      : `EmbeddingGemma 2 vision asset ${spec.name}: the expected golden cases are all present`,
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
        `EmbeddingGemma 2 vision golden comparison: ${spec.name} ${caseName} (real GPU / upstream's soft tokens in torch CPU, then the sha256 row)${suffix}`,
        async () => {
          await settleCase(spec, caseName, { compare: true });
        },
      );
      continue;
    }
    // Default run: everything except the comparison with upstream's soft tokens, then the row.
    test.skipIf(!available || !GPU_AVAILABLE)(
      `EmbeddingGemma 2 vision golden: ${spec.name} ${caseName} (real GPU / io keys, input shapes and positions, output shape and dtype, then the sha256 row)`,
      async () => {
        await settleCase(spec, caseName, { compare: false });
      },
    );
    // Opt-in: the comparison with upstream's soft tokens under the image's checks. Its
    // results.json entry is `<series>/<image>/parity` (the bare ID is the default sha case's).
    test.skipIf(!UPSTREAM_PARITY || !available || !GPU_AVAILABLE)(
      `EmbeddingGemma 2 vision golden parity: ${spec.name} ${caseName} (real GPU / upstream's soft tokens in torch CPU)${UPSTREAM_PARITY_OPT_IN}`,
      async () => {
        await runRecordedCase(
          results,
          { id: `${spec.name}/${caseName}/parity` },
          async ({ measurements }) => {
            const run = await runGoldenCase(spec, caseName, { repeat: false });
            const measured = compareWithGolden(spec, caseName, run, measurements);
            return { status: "pass", note: `${measured}; ${run.timing}` };
          },
        );
      },
    );
  }
  return available ? cases.map((caseName) => `${spec.name}/${caseName}`) : [];
};

test("EmbeddingGemma 2 vision: every series holds the same golden images, each at the same soft-token count", () => {
  for (const spec of SERIES) {
    assertEquals(Object.keys(spec.checks).sort(), EXPECTED_CASES, `${spec.name}: its images`);
    for (const name of EXPECTED_CASES) {
      assertEquals(spec.checks[name].tokens, IMAGE_CHECKS[name].tokens, `${spec.name} ${name}`);
    }
  }
});

/** The IDs of the registered cases of every series (the set the reference gate counts). */
const caseIds = SERIES.flatMap(registerSeries);

const RUNNABLE = caseIds.length > 0 && GPU_AVAILABLE;
if (RUNNABLE) references.warnMissing(caseIds);
registerReferenceGate(references, { runnable: RUNNABLE, caseIds });
