// Real-GPU golden E2E of the text tower of EmbeddingGemma 2 (sentence embedding) (step 3 of
// ADR 0005).
//
// The targets are two series of `tools/export-recipes/embeddinggemma2/export.py`: the f32 series
// `embeddinggemma-2-text` (weights of about 1.1 GB; its `generate` in SERIES below is the source of
// truth for the command) and the int8 series `embeddinggemma-2-text-i8` (the `i8` preset's
// weights, about 309 MiB — the token table and every linear rounded to per-row int8 before the
// goldens were taken, so its goldens are the quantized model's own outputs and its tolerance
// measures implementation error only, ADR 0006). One graph: the text tower → masked mean → L2
// normalisation, one output `[1, 768]`, a symbolic token length T up to 8,192 (the upstream
// context). The bf16 series is not read here: its weights are the f32 series' values bit for bit
// and its io files are the f32 series' byte for byte (the recipe README).
//
// What it adds over the first EmbeddingGemma: the sliding layers' window is a bidirectional band
// (`|q − kv| <= 512`) declared on their attention (`attrs.band`, checked in the score kernel from
// positions — no mask tensor), the four full layers carry no mask, and grouped-query attention stays
// in its true form with two layouts (2 KV heads × 256 and 1 KV head × 512). Six goldens cut the
// window (T 609 / 1,508 / 2,048 / 3,062 / 5,935) or run at the bound (T = 8,192); the export proves
// on every run that the window really moves those outputs and that its wrapper equals the upstream
// model's own forward. The cases also
// record the Session's creation and run times and its buffer sizes (the note of each case in the
// results and one console line).
//
// Every case ends with a sha256 check of the output against this machine's reference row (ADR 0106
// — the tolerance says "close to torch" across machines, the row says "not one bit moved on this
// machine"). Rows are created with KARUME_REFERENCE=write on the lane.
//
// ## Where the assets come from (ADR 0139)
//
// | Series                     | Container                               | Goldens                                 | Runs                      |
// | -------------------------- | --------------------------------------- | --------------------------------------- | ------------------------- |
// | `embeddinggemma-2-text-i8` | the mirror's `i8` preset                | `goldenDir("embeddinggemma-2-text-i8")` | default lane              |
// | `embeddinggemma-2-text`    | `outputs/series/embeddinggemma-2-text/` | the same series directory               | only with KARUME_SERIES=1 |
//
// The int8 series is shipped: its container is opened from the distribution mirror
// (`models/karume-embeddinggemma-2/`, model `text`, weights key `model`, through its manifest —
// byte-identical to the series' container), and its goldens are read from the test-asset root
// (fetched with `node tools/test-assets/fetch.ts`). The
// f32 series is a verification graph no distribution ships, so it stays an implementation-time
// series check (decision 6): without KARUME_SERIES=1 its cases SKIP explicitly and the series is
// not read.
//
// Upstream parity is opt-in (helpers/upstream-parity.ts): for the shipped series the default run
// keeps the asset completeness gate, the per-case structure checks (io keys, the pinned input
// shapes, output count / shape / dtype on the real GPU) and the sha256 rows; the tolerance
// comparison with the torch CPU goldens runs only with KARUME_UPSTREAM_PARITY=1. The f32 series
// keeps its tolerance comparison in its one case (it already runs only on request).
//
// Where an asset is missing the cases SKIP explicitly; an asset that exists only partly FAILs (the
// completeness test). A golden of the shipped series absent from a complete fetch of the pinned
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

/**
 * The tolerance for comparing the text tower of EmbeddingGemma 2 against the torch CPU
 * expectations.
 *
 * Measured on node-nvidia-geforce-rtx-3080-ti (2026-11-21, plain comparison, 7 cases × one output
 * `[1,768]`; the three cases past 2,048 tokens added 2026-11-22 with the band declared):
 *
 * | Case          | T     | maxAbs  | maxRel  | \|ref\| max | \|ref\| min non-zero |
 * | ------------- | ----- | ------- | ------- | ----------- | -------------------- |
 * | bare          | 12    | 4.10e-7 | 2.76e-3 | 0.409       | 1.75e-5              |
 * | query-en      | 16    | 1.06e-7 | 6.99e-4 | 0.290       | 7.19e-5              |
 * | query-ja      | 16    | 1.34e-7 | 5.37e-4 | 0.297       | 4.24e-5              |
 * | document-en   | 19    | 1.56e-7 | 4.98e-3 | 0.329       | 1.98e-5              |
 * | document-609  | 609   | 1.60e-7 | 3.70e-4 | 0.280       | 5.20e-5              |
 * | document-1508 | 1,508 | 1.43e-7 | 2.08e-2 | 0.297       | 2.07e-6              |
 * | document-2048 | 2,048 | 1.79e-7 | 1.09e-3 | 0.302       | 5.90e-5              |
 * | document-3062 | 3,062 | 1.83e-7 | 9.28e-4 | 0.310       | 1.91e-5              |
 * | document-5935 | 5,935 | 1.41e-7 | 5.16e-4 | 0.324       | 3.56e-5              |
 * | document-8192 | 8,192 | 1.04e-7 | 1.87e-3 | 0.329       | 3.88e-5              |
 *
 * atol drives the verdict (`|x−y| ≤ atol + rtol·|ref|`): the output is a unit vector whose values
 * thin out to near 0, where maxRel grows (2.08e-2 at an element of 2.07e-6, an absolute error of
 * about 4e-8). rtol 1e-6 adds at most 4.1e-7 at the largest \|ref\|, under half of atol.
 *
 * atol 1e-6 is 2.4 times the measured worst case (bare). The error does not grow with T (12 →
 * 8,192 leaves maxAbs at 1–2e-7 apart from bare), so the long cases need no band of their own.
 * The breakages this guards against are far above it: moving the window by one position moves the
 * torch output by 2.5e-4 to 5.6e-4, dropping the window by 1.1e-2 to 5.1e-2 (measured by the
 * export), and a wrong pooling axis or normalisation by O(0.1).
 */
const EMBEDDINGGEMMA2_TOLERANCE: Tolerance = { atol: 1e-6, rtol: 1e-6 };

/**
 * The tolerance of the **i8 series** against its own torch CPU goldens (the quantized model's
 * outputs — the rounding is in the goldens, so this is the implementation error alone).
 *
 * Measured on node-nvidia-geforce-rtx-3080-ti (2026-11-22, plain comparison, 10 cases × one output
 * `[1,768]`; the last column is how far the rounding itself moves the output — the i8 golden
 * against the f32 golden, max abs):
 *
 * | Case          | T     | maxAbs  | maxRel  | \|ref\| max | \|ref\| min non-zero | rounding (i8 vs f32 golden) |
 * | ------------- | ----- | ------- | ------- | ----------- | -------------------- | --------------------------- |
 * | bare          | 12    | 2.87e-7 | 6.03e-4 | 0.411       | 1.50e-4              | 5.44e-3                     |
 * | query-en      | 16    | 1.04e-7 | 5.76e-4 | 0.290       | 1.07e-5              | 1.67e-3                     |
 * | query-ja      | 16    | 8.94e-8 | 5.41e-4 | 0.297       | 2.57e-5              | 1.26e-3                     |
 * | document-en   | 19    | 1.47e-7 | 3.48e-4 | 0.328       | 1.40e-5              | 2.60e-3                     |
 * | document-609  | 609   | 1.64e-7 | 1.42e-3 | 0.279       | 4.54e-6              | 2.68e-3                     |
 * | document-1508 | 1,508 | 1.66e-7 | 1.25e-3 | 0.298       | 7.57e-6              | 2.62e-3                     |
 * | document-2048 | 2,048 | 1.40e-7 | 3.65e-4 | 0.303       | 1.48e-4              | 2.84e-3                     |
 * | document-3062 | 3,062 | 1.54e-7 | 1.31e-3 | 0.311       | 1.09e-5              | 2.63e-3                     |
 * | document-5935 | 5,935 | 1.38e-7 | 2.08e-3 | 0.325       | 1.80e-5              | 2.81e-3                     |
 * | document-8192 | 8,192 | 1.27e-7 | 1.59e-3 | 0.330       | 4.61e-6              | 3.03e-3                     |
 *
 * The same reading as the f32 series: atol drives the verdict on a unit vector whose values thin
 * out to near 0 (maxRel 2.08e-3 is an absolute error under 1.4e-7), and rtol 1e-6 adds at most
 * 4.1e-7. atol 1e-6 is 3.5 times the measured worst case (bare, as in f32): the int8 storage adds
 * no error of its own on top of f32's (2.87e-7 against f32's 4.10e-7 worst), and the error does not
 * grow with T. The value equals the f32 series' by measurement, not by inheritance. What it guards
 * against is far above it: an asset of another storage (the f32 or bf16 weights in this series'
 * place, or the reverse) moves the output by the rounding, 1.3e-3 to 5.4e-3 — over a thousand
 * times atol.
 */
const EMBEDDINGGEMMA2_I8_TOLERANCE: Tolerance = { atol: 1e-6, rtol: 1e-6 };

/** The generate command a SKIP prints (the same spelling as tools/export-recipes/embeddinggemma2/README.md). */
const GENERATE_COMMAND =
  "cd tools/export-recipes && uv run --with 'transformers==5.23.0' python -m embeddinggemma2.export";

/** The command that fetches the goldens of a shipped series (pasted into the skip warnings). */
const FETCH_COMMAND = "node tools/test-assets/fetch.ts";

/** The distribution that ships the text tower (`models/karume-embeddinggemma-2/`). */
const MIRROR_ROOT = mirrorRoot("karume-embeddinggemma-2");
/** The mirror's model and the weights key of the text graph (also its graph name — container-v1 §2.1). */
const MIRROR_MODEL = "text";
const MIRROR_GRAPH = "model";

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
      /** The mirror's preset whose container is the series' (byte for byte). */
      readonly quant: string;
    };

/**
 * One series this file checks. MUST: each series has its own tolerance (ADR 0019 — sharing one
 * would apply the f32 band to the i8 assets, or the reverse).
 */
type SeriesSpec = {
  /**
   * The series name: its directory under `outputs/series/` or `goldenDir(name)`, and the prefix of
   * its sha256 row IDs (frozen — ADR 0139 decision 7).
   */
  readonly name: string;
  readonly tolerance: Tolerance;
  readonly source: SeriesSource;
};

const SERIES: readonly SeriesSpec[] = [
  {
    name: "embeddinggemma-2-text",
    tolerance: EMBEDDINGGEMMA2_TOLERANCE,
    source: {
      kind: "series",
      generate: GENERATE_COMMAND,
      weights: "f32 weights of about 1.1 GB",
    },
  },
  {
    name: "embeddinggemma-2-text-i8",
    tolerance: EMBEDDINGGEMMA2_I8_TOLERANCE,
    source: { kind: "mirror", quant: "i8" },
  },
];

/** The width of a soft row (the text tower's hidden size, `text_config.hidden_size` at the pin). */
const SOFT_WIDTH = 512;

const MODEL_FILE = "model.krm";
const IO_PREFIX = "io.";
const IO_SUFFIX = ".safetensors";

/**
 * The cases that should have been generated, with their token length T (`<bos>` and `<eos>`
 * included). Pinned here rather than taken from the listing, so a partial generation fails instead
 * of losing tests silently; the source of truth is GOLDEN_CASES / CASE_LENGTHS of
 * `tools/export-recipes/embeddinggemma2/export.py`. The lengths tie each case to what it
 * exercises: T >= 514 cuts the sliding window, T = 8,192 reads the far corner of every folded
 * constant (the RoPE tables). Every series has the same cases.
 */
const EXPECTED_LENGTHS: Readonly<Record<string, number>> = {
  bare: 12,
  "document-1508": 1508,
  "document-2048": 2048,
  "document-3062": 3062,
  "document-5935": 5935,
  "document-609": 609,
  "document-8192": 8192,
  "document-en": 19,
  "query-en": 16,
  "query-ja": 16,
};
const EXPECTED_CASES = Object.keys(EXPECTED_LENGTHS).sort();

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

/**
 * Where the settlements and measurements go
 * (`outputs/verify/<environment key>/<date>_embeddinggemma2-golden/` — safe to delete).
 */
const results = openResults("embeddinggemma2-golden");

/** The sha256 reference rows of the output (rows per environment; the row ID is `<series>/<case>`). */
const references = openReferences(
  new URL("../../runtime/tests/fixtures/references/embeddinggemma2-golden.json", import.meta.url),
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

/** The directory of a series' io goldens (`io.<case>.safetensors`). */
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

/** The output of one golden case on the real GPU, with the golden's expectation beside it. */
type GoldenRun = {
  readonly name: string;
  readonly where: string;
  readonly output: Tensor;
  readonly expected: Tensor;
  /** The second run on the same Session (taken only when asked for — the sha row needs it). */
  readonly again?: Tensor;
  readonly timing: string;
};

/**
 * Loads one golden case, runs it on the GPU and asserts everything except the tolerance
 * comparison (io keys, the pinned input shapes, output keys / count / shape / dtype). The default
 * sha case, the opt-in parity case and the f32 series' one case all go through here.
 */
const runGoldenCase = async (
  spec: SeriesSpec,
  caseName: string,
  { repeat }: { readonly repeat: boolean },
): Promise<GoldenRun> => {
  const [parsed, ioBytes] = await Promise.all([openSeries(spec), readGolden(spec, caseName)]);
  const io = parseSafetensors(ioBytes);

  // Every tensor of io matches the graph's inputs and outputs exactly.
  const expectedKeys = [
    ...parsed.graph.inputs.map((input) => `input.${input.name}`),
    ...parsed.graph.outputs.map((_, index) => `output.${index}`),
  ].sort();
  assertEquals([...io.tensors.keys()].sort(), expectedKeys, "tensor keys of io.safetensors");

  // T is bound from the golden's input shape (no explicit bindings). It must be the length this
  // case is pinned at, or the case no longer exercises what it is named for. Each input's shape is
  // fixed here, not read from the graph's own declaration — a graph and goldens that agree on
  // another shape fail: `[1, T]` for the ids and the pool mask, `[1, T, 512]` / `[1, T, 1]` for the
  // soft rows (which no golden text holds).
  const length = EXPECTED_LENGTHS[caseName];
  const expectedShapes: Readonly<Record<string, readonly number[]>> = {
    input_ids: [1, length],
    pool_mask: [1, length],
    soft_embeds: [1, length, SOFT_WIDTH],
    soft_mask: [1, length, 1],
  };
  assertEquals(
    parsed.graph.inputs.map((input) => input.name),
    Object.keys(expectedShapes),
    "the graph's inputs",
  );
  const inputs: Record<string, Tensor> = {};
  for (const input of parsed.graph.inputs) {
    const view = io.tensors.get(`input.${input.name}`);
    assert(view !== undefined, `input.${input.name} is missing from the golden`);
    assertEquals(view.shape, expectedShapes[input.name], `input.${input.name}: shape`);
    inputs[input.name] = ioTensor(io, view, input.dtype);
  }

  const gpu = await acquireGpu();
  // MUST: the device is destroyed also when the adapter check or `createSession` fails.
  try {
    // The rows written from this run are this machine's: the adapter the key was taken from must
    // be the one running.
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
      // The intermediates live in the Session's plan backing (the arena's transient figures read 0
      // on that path), so its resident regions are the peak buffer size of a run.
      const { weights, planBacking } = session.diagnostics();
      const timing =
        `T=${EXPECTED_LENGTHS[caseName]}: create ${createMs} ms, first run ${firstRunMs} ms,` +
        `${secondRun} weights ${mebibytes(weights.allocatedBytes)},` +
        ` plan backing ${mebibytes(planBacking.residentBytes)}` +
        ` + inputs ${mebibytes(planBacking.inputBytes)}`;
      console.log(`[embeddinggemma2] ${spec.name} ${caseName} ${timing}`);
      assertEquals(Object.keys(outputs).sort(), [...parsed.graph.outputs].sort());
      assertEquals(parsed.graph.outputs.length, 1, "graph output count");
      const [name] = parsed.graph.outputs;
      const view = io.tensors.get("output.0");
      assert(view !== undefined, "output.0 is missing from the golden");
      const where = `${spec.name} ${caseName} output.0 ('${name}')`;
      const declared = parsed.graph.values[name].dtype;
      const output = outputs[name];
      assertEquals(output.shape, view.shape, `${where}: shape`);
      assertEquals(output.dtype, declared, `${where}: dtype`);
      return {
        name,
        where,
        output,
        expected: ioTensor(io, view, declared),
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

/** The tolerance comparison with the torch CPU golden (measurements recorded when given). */
const compareWithGolden = (
  spec: SeriesSpec,
  run: GoldenRun,
  measurements?: Measurement[],
): void => {
  const report = compareTensors(run.output, run.expected, spec.tolerance);
  measurements?.push({
    output: run.name,
    maxAbs: report.maxAbsError,
    maxRel: report.maxRelError,
    tolerance: spec.tolerance,
    stage: "karume",
  });
  console.log(
    `[embeddinggemma2] ${run.where}: maxAbs ${report.maxAbsError.toExponential(3)}, maxRel ${report.maxRelError.toExponential(3)}`,
  );
  assert(report.pass, `${run.where}: ${formatAllclose(report)}`);
};

/**
 * Settles the output against this machine's sha256 row and records the case; fails after the
 * record on a mismatch (throwing first would keep the measured sha out of the results). `compare`
 * runs the tolerance comparison first (the f32 series' case), so no row is created from an output
 * that broke it.
 */
const settleCase = async (
  spec: SeriesSpec,
  caseName: string,
  { compare }: { readonly compare: boolean },
): Promise<void> => {
  const caseId = `${spec.name}/${caseName}`;
  /** The sha settlement, carried out so that a mismatch fails after the recording. */
  let settlement: ReferenceSettlement | undefined;
  await runRecordedCase(results, { id: caseId }, async ({ measurements }) => {
    const run = await runGoldenCase(spec, caseName, { repeat: true });
    if (compare) compareWithGolden(spec, run, measurements);
    const { name, where, output, again } = run;
    // The sha row stands on the Session's own determinism: a second run on the same Session gives
    // the same bits.
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
   * SKIPs; a partial absence is on the FAIL side). A mirror series also needs its mirror (the
   * goldens and the container are separate sources: each one's absence SKIPs); a series' missing
   * container is caught by the completeness case instead.
   */
  const available =
    (found.length > 0 || missingAtPinned) && (source.kind === "series" || hasContainer);
  /**
   * Whether anything at all is left — the SKIP predicate of the completeness test, so that a gap
   * where every io is gone and only the series' model remains still FAILs.
   */
  const anyPresent =
    found.length > 0 || (source.kind === "series" && hasContainer) || missingAtPinned;
  /**
   * The cases registered: the golden files present, or else the expected set, so that each case
   * SKIPs by name (partial presence registers only what is there — the completeness case fails).
   */
  const cases = found.length > 0 ? found : EXPECTED_CASES;

  if (readable && found.length === 0 && !missingAtPinned) {
    console.warn(
      source.kind === "series"
        ? `[karume] ${root.pathname} has no exported asset, so the EmbeddingGemma 2 E2E of ${spec.name} is ` +
            `SKIPPED (${source.weights}, kept outside the repository). Generate: ${source.generate}`
        : `[karume] skipping the EmbeddingGemma 2 E2E of ${spec.name}: no goldens under ` +
            `${root.pathname}. Fetch: ${FETCH_COMMAND}`,
    );
  }
  if (source.kind === "mirror" && !hasContainer) {
    console.warn(
      `[karume] skipping the EmbeddingGemma 2 E2E of ${spec.name}: the distribution mirror ` +
        `${MIRROR_ROOT.pathname} is absent (or not karume/6).`,
    );
  }

  test.skipIf(!anyPresent)(
    source.kind === "series"
      ? `EmbeddingGemma 2 asset ${spec.name}: the expected cases and the model itself are all present${suffix}`
      : `EmbeddingGemma 2 asset ${spec.name}: the expected golden cases are all present`,
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
        `EmbeddingGemma 2 golden comparison: ${spec.name} ${caseName} (real GPU / torch CPU expectations, then the sha256 row)${suffix}`,
        async () => {
          await settleCase(spec, caseName, { compare: true });
        },
      );
      continue;
    }
    // Default run: everything except the tolerance comparison, then the sha256 row.
    test.skipIf(!available || !GPU_AVAILABLE)(
      `EmbeddingGemma 2 golden: ${spec.name} ${caseName} (real GPU / io keys, input shapes, output shape and dtype, then the sha256 row)`,
      async () => {
        await settleCase(spec, caseName, { compare: false });
      },
    );
    // Opt-in: the comparison with the torch CPU golden under the series' tolerance. It writes no
    // results entries: the case ID belongs to the default sha case above, and writing it here
    // would overwrite that case's settlement.
    test.skipIf(!UPSTREAM_PARITY || !available || !GPU_AVAILABLE)(
      `EmbeddingGemma 2 golden parity: ${spec.name} ${caseName} (real GPU / torch CPU expectations)${UPSTREAM_PARITY_OPT_IN}`,
      async () => {
        compareWithGolden(spec, await runGoldenCase(spec, caseName, { repeat: false }));
      },
    );
  }
  return available ? cases.map((caseName) => `${spec.name}/${caseName}`) : [];
};

/** The IDs of the registered cases of every series (the set the reference gate counts). */
const caseIds = SERIES.flatMap(registerSeries);

const RUNNABLE = caseIds.length > 0 && GPU_AVAILABLE;
if (RUNNABLE) references.warnMissing(caseIds);
registerReferenceGate(references, { runnable: RUNNABLE, caseIds });
