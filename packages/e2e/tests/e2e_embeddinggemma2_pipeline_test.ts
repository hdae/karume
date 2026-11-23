// The EmbeddingGemma family's public pipeline over the local distribution of EmbeddingGemma 2 —
// `embeddinggemma.Pipeline` (`@karume/models/embeddinggemma`) built by `fromPretrained` from
// `models/karume-embeddinggemma-2/` (the recipe `tools/export-recipes/embeddinggemma2/distribution.py`;
// model `text`, every preset of the mirror — `bf16` = the checkpoint's own bf16 weights with f32
// compute, the default; `i8` = the same weight slots stored as per-row int8, f32 compute).
//
// ## Checks
//
// 1. **The mirror's manifest and tokenizer** (no GPU): the `pipelineConfig` parses with the five
//    fields, the model declares exactly the presets of PRESETS below with `bf16` the default and
//    each preset's `requiredLimits` (256 MiB of binding for `bf16`, for its token table; none for
//    `i8`, whose largest buffer — the int8 table, 128 MiB — is the WebGPU default), and the
//    pipeline's ids for each golden text (its prompt name + its body — the ids `embed` runs) have
//    the length the recipe pinned. With the series (`KARUME_SERIES=1`) the ids equal the golden's
//    `input_ids`, which the recipe took from upstream's `tokenizers` over the prefix of
//    `config_sentence_transformers.json` + the body: the prompt prefix the manifest carries, the
//    concatenation and `<bos>` … `<eos>` are upstream's.
// 2. **The ten golden texts through each preset** (real GPU): each vector is a unit vector and
//    bit-for-bit equals this machine's sha256 row of that preset (rows per environment — ADR 0106;
//    the row ID is `embeddinggemma-2/<preset>/<case>`); with the series it also equals the torch CPU
//    golden of the preset's series within the tolerance of the series e2e
//    (`e2e_embeddinggemma2_test.ts`). `bf16` is compared with `outputs/series/embeddinggemma-2-text/`
//    (the distribution's bf16 weights are the f32 series' values bit for bit — the recipe proves it
//    when it writes the bf16 series — so the same goldens hold); `i8` with
//    `outputs/series/embeddinggemma-2-text-i8/` (the quantized model's own goldens).
// 3. **A shorter width**, per preset: `dimension` 256 of a short text and 128 of the 2,048-token
//    text equal the leading slice of the full vector of the same call path re-normalised (bit for
//    bit), each with its own sha256 row, and — with the series — the golden's slice re-normalised
//    within the tolerance.
// 4. **Timings** (recorded, not asserted), per preset: `fromPretrained` with a shared GPU, and the
//    first and a repeated `embed` of a short, the 2,048-token and the 8,192-token text (the repeat
//    must give the same bits — the sha rows stand on that), with each run's plan backing. The
//    8,192-token run's plan backing must stay under 1 GiB — one layer's unsplit scores (4 heads ×
//    8,192² f32) alone are 1 GiB, so it shows the family's row-block budget reached the Session.
// 5. **The int8 preset against the original weights** (real GPU): each golden text embedded through
//    both presets, at every declared width, has a cosine of at least I8_COSINE_FLOOR between the
//    `i8` and the `bf16` vector — the shipped-bytes counterpart of the torch-CPU screening of the
//    int8 form (docs/research/2026-11-22-embeddinggemma2-quant-screen.md).
// 6. **The device's token limit on a narrowed device**, per preset (real GPU): a device acquired
//    with its binding and buffer limits capped at the preset's own need (the test-only
//    `LIMIT_CAPS` — `bf16`: its `requiredLimits`, 256 MiB; `i8`: the WebGPU default, 128 MiB) runs
//    an input of exactly ⌊cap / 49,152 B⌋ tokens (5,461 / 2,730), refuses one token more with a
//    `ModelInputError` before the run, and still embeds a short text afterwards — the family's
//    limit agrees with what the runtime runs on the real graph, and `i8` needs no binding or buffer
//    size above the WebGPU default (only those two limits are capped; the others stay the adapter's).
//
// The golden texts come from the tracked fixture the recipe writes
// (`packages/models/tests/fixtures/embeddinggemma2/golden-cases.json` — `GOLDEN_CASES` of
// `tools/export-recipes/embeddinggemma2/export.py`, pinned there by a recipe test), so the long
// documents are not spelled a second time here. Every series has the same cases.
//
// Without the mirror every case SKIPs explicitly (the distribution gate owns that red —
// `karume-embeddinggemma-2` is on its roster); without a GPU the GPU cases SKIP; without
// `KARUME_SERIES=1` (or without a series' io) the golden comparisons SKIP and the sha rows still
// run. Rows are created with `KARUME_REFERENCE=write` on the lane.

import { readTextFile, isNotFound, statSync, readFile } from "@karume/test-support/fs";
import { afterAll, beforeAll, describe, it } from "vite-plus/test";
import { assert, assertEquals, assertRejects, fail } from "@std/assert";
import { parseManifest } from "@karume/hub";
import {
  type GpuContext,
  parseSafetensors,
  type SafetensorsFile,
  type SessionDiagnostics,
  type Tensor,
} from "@karume/runtime";
import * as embeddinggemma from "../../models/embeddinggemma.ts";
import { ModelInputError } from "../../models/src/errors.ts";
import { parseEmbeddingGemmaPipelineConfig } from "../../models/src/embeddinggemma/config.ts";
import {
  type EmbedTextRules,
  embedTextRules,
  planEmbedRequest,
  truncateEmbedding,
} from "../../models/src/embeddinggemma/input.ts";
import { parseGemmaTokenizerAsset } from "../../models/src/gemma/text/asset.ts";
import { Tokenizer } from "../../models/src/gemma/text/tokenizer.ts";
import { acquireTestGpu, GPU_AVAILABLE } from "../../models/tests/helpers/gpu.ts";
import {
  mirrorDirectory,
  mirrorPresent,
  mirrorRoot,
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
import { ioTensor } from "../../runtime/tests/helpers/golden-io.ts";
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
 * The tolerance of the `bf16` preset against the torch CPU goldens: the series e2e's
 * (`EMBEDDINGGEMMA2_TOLERANCE` of `e2e_embeddinggemma2_test.ts` — atol 1e-6 drives the verdict on a
 * unit vector whose values thin out to near 0; its derivation is there).
 *
 * Measured through the pipeline (bf16 weights, f32 compute) on node-nvidia-geforce-rtx-3080-ti
 * (2026-11-21, 7 cases × `[1,768]`, and the two shorter widths against the golden's slice
 * re-normalised; the three cases past 2,048 tokens added 2026-11-22 with the band declared and the
 * family's row-block budget):
 *
 * | Case          | T     | Width | maxAbs  | maxRel  |
 * | ------------- | ----- | ----- | ------- | ------- |
 * | bare          | 12    | 768   | 4.10e-7 | 2.76e-3 |
 * | query-en      | 16    | 768   | 1.06e-7 | 6.99e-4 |
 * | query-ja      | 16    | 768   | 1.34e-7 | 5.37e-4 |
 * | document-en   | 19    | 768   | 1.56e-7 | 4.98e-3 |
 * | document-609  | 609   | 768   | 1.60e-7 | 3.70e-4 |
 * | document-1508 | 1,508 | 768   | 1.43e-7 | 2.08e-2 |
 * | document-2048 | 2,048 | 768   | 1.79e-7 | 1.09e-3 |
 * | document-3062 | 3,062 | 768   | 1.83e-7 | 9.28e-4 |
 * | document-5935 | 5,935 | 768   | 1.41e-7 | 5.16e-4 |
 * | document-8192 | 8,192 | 768   | 1.04e-7 | 1.87e-3 |
 * | query-en      | 16    | 256   | 1.53e-7 | 6.98e-4 |
 * | document-2048 | 2,048 | 128   | 2.68e-7 | 6.05e-5 |
 *
 * The full-width rows are the series e2e's to the digit: on that machine the pipeline's vectors
 * are bit-identical to the f32 series' Session outputs (0 of 768 words differ in each case). The
 * re-normalisation of a shorter slice scales the error by `1 / norm(slice)` (at most 2.68e-7 here),
 * still under half of atol.
 */
const PIPELINE_TOLERANCE: Tolerance = { atol: 1e-6, rtol: 1e-6 };

/**
 * The tolerance of the `i8` preset against the i8 series' goldens (the quantized model's own
 * outputs): the series e2e's `EMBEDDINGGEMMA2_I8_TOLERANCE`, whose derivation is there.
 *
 * Measured through the pipeline (int8 weights, f32 compute) on node-nvidia-geforce-rtx-3080-ti
 * (2026-11-22, 10 cases × `[1,768]` and the two shorter widths against the golden's slice
 * re-normalised):
 *
 * | Case          | T     | Width | maxAbs  | maxRel  |
 * | ------------- | ----- | ----- | ------- | ------- |
 * | bare          | 12    | 768   | 2.87e-7 | 6.03e-4 |
 * | query-en      | 16    | 768   | 1.04e-7 | 5.76e-4 |
 * | query-ja      | 16    | 768   | 8.94e-8 | 5.41e-4 |
 * | document-en   | 19    | 768   | 1.47e-7 | 3.48e-4 |
 * | document-609  | 609   | 768   | 1.64e-7 | 1.42e-3 |
 * | document-1508 | 1,508 | 768   | 1.66e-7 | 1.25e-3 |
 * | document-2048 | 2,048 | 768   | 1.40e-7 | 3.65e-4 |
 * | document-3062 | 3,062 | 768   | 1.54e-7 | 1.31e-3 |
 * | document-5935 | 5,935 | 768   | 1.38e-7 | 2.08e-3 |
 * | document-8192 | 8,192 | 768   | 1.27e-7 | 1.59e-3 |
 * | query-en      | 16    | 256   | 1.27e-7 | 2.21e-4 |
 * | document-2048 | 2,048 | 128   | 1.86e-7 | 1.80e-5 |
 *
 * The full-width rows are the series e2e's to the digit: the pipeline's vectors are bit-identical
 * to the i8 series' Session outputs (0 of 768 words differ in each case — the mirror's `i8`
 * container is that series' container, and the row-block budget changes no bit). The shorter
 * slices stay under 1.9e-7, under a fifth of atol.
 */
const PIPELINE_I8_TOLERANCE: Tolerance = { atol: 1e-6, rtol: 1e-6 };

/**
 * The floor of the cosine between the `i8` and the `bf16` vector of one text, at every width.
 *
 * The torch-CPU screening of the int8 form (docs/research/2026-11-22-embeddinggemma2-quant-screen.md
 * — the same rounding as the `i8` files: the i8 series' goldens reproduce its embeddings to 2e-16
 * in cosine) put the lowest cosine against the original weights at 0.999396 over 58 texts at 768
 * (the golden case `bare`), 0.999411 / 0.999474 / 0.999620 at 512 / 256 / 128. Measured here on
 * node-nvidia-geforce-rtx-3080-ti (2026-11-22), the shipped bytes give the screening's own values to
 * the seventh digit:
 *
 * | Case          | 768       | 512       | 256       | 128       |
 * | ------------- | --------- | --------- | --------- | --------- |
 * | query-en      | 0.9999201 | 0.9999194 | 0.9999258 | 0.9999399 |
 * | document-en   | 0.9997848 | 0.9997821 | 0.9997921 | 0.9998396 |
 * | query-ja      | 0.9999306 | 0.9999312 | 0.9999368 | 0.9999534 |
 * | bare          | 0.9993962 | 0.9994113 | 0.9994738 | 0.9996197 |
 * | document-609  | 0.9997484 | 0.9997614 | 0.9997835 | 0.9998280 |
 * | document-1508 | 0.9997585 | 0.9997768 | 0.9998011 | 0.9998523 |
 * | document-2048 | 0.9997595 | 0.9997748 | 0.9997950 | 0.9998562 |
 * | document-3062 | 0.9997805 | 0.9997962 | 0.9998234 | 0.9998778 |
 * | document-5935 | 0.9997706 | 0.9997896 | 0.9998227 | 0.9998800 |
 * | document-8192 | 0.9997563 | 0.9997778 | 0.9998092 | 0.9998747 |
 *
 * 0.999 leaves 1 − cos = 1e-3, 1.66 times the worst measured (6.04e-4 at `bare`). The forms the
 * screening measured below it are what this guards against: activation int8 over the same weights
 * (lowest 0.99563) and every 4-bit form (0.960 to 0.966). It is not a quality claim beyond these
 * ten texts.
 */
const I8_COSINE_FLOOR = 0.999;

/** The WebGPU default of `maxStorageBufferBindingSize` (the spec's), which a preset without `requiredLimits` runs within. */
const WEBGPU_DEFAULT_BINDING_BYTES = 128 * 2 ** 20;

/** One preset of the mirror and what its checks compare with. */
type PresetSpec = {
  readonly quant: string;
  /** The series whose io goldens its vectors are compared with (under `outputs/series/`). */
  readonly series: string;
  readonly tolerance: Tolerance;
  /**
   * The binding and buffer limit of its narrowed device: its `maxStorageBufferBindingSize` of
   * `requiredLimits`, or the WebGPU default when it declares none (check 1 holds the manifest to it).
   */
  readonly cappedBindingBytes: number;
  /** The longest input that device runs (⌊cap / bytes per token⌋ — pinned, not derived). */
  readonly cappedMaxTokens: number;
};

/**
 * The presets, in the manifest's order. Pinned here rather than read from the mirror, so a mirror
 * that lost or gained a preset fails check 1 instead of losing cases silently.
 */
const PRESETS: readonly PresetSpec[] = [
  {
    quant: "bf16",
    series: "embeddinggemma-2-text",
    tolerance: PIPELINE_TOLERANCE,
    cappedBindingBytes: 256 * 2 ** 20,
    cappedMaxTokens: 5461,
  },
  {
    quant: "i8",
    series: "embeddinggemma-2-text-i8",
    tolerance: PIPELINE_I8_TOLERANCE,
    cappedBindingBytes: WEBGPU_DEFAULT_BINDING_BYTES,
    cappedMaxTokens: 2730,
  },
];
const DEFAULT_PRESET = "bf16";
const presetOf = (quant: string): PresetSpec => {
  const found = PRESETS.find((preset) => preset.quant === quant);
  if (found === undefined) throw new Error(`PRESETS has no preset '${quant}'`);
  return found;
};

/** The local distribution (`dist.py --pipeline embeddinggemma-2` builds it — docs/assets-layout.md). */
const MIRROR_ROOT = mirrorRoot("karume-embeddinggemma-2");
const ASSEMBLE =
  "cd tools/export-recipes && uv run --with 'transformers==5.23.0' python -m embeddinggemma2.export" +
  " --dtype bf16 && uv run --with 'transformers==5.23.0' python -m embeddinggemma2.export --dtype i8" +
  " && uv run --with 'transformers==5.23.0' python -m embeddinggemma2.export_vision --dtype bf16" +
  " && uv run --with 'transformers==5.23.0' python -m embeddinggemma2.export_audio --dtype bf16" +
  " && uv run python dist.py --pipeline embeddinggemma-2";
const MIRROR = mirrorPresent(MIRROR_ROOT);
if (!MIRROR) {
  console.warn(
    `[karume] the distribution mirror ${MIRROR_ROOT.pathname} is absent (or not karume/6), so the` +
      ` EmbeddingGemma 2 pipeline checks SKIP. Assemble: ${ASSEMBLE}`,
  );
}

/** The io golden of a case in a series (io files only — the container is not read). */
const ioFile = (series: string, name: string): URL =>
  new URL(`../../../outputs/series/${series}/io.${name}.safetensors`, import.meta.url);

/** One golden text of the fixture. */
type GoldenCase = {
  readonly name: string;
  /** A prompt name of the manifest (absent: no prompt). */
  readonly prompt?: string;
  readonly text: string;
  /** T, `<bos>` and `<eos>` included. */
  readonly tokens: number;
};

const FIXTURE = JSON.parse(
  await readTextFile(
    new URL("../../models/tests/fixtures/embeddinggemma2/golden-cases.json", import.meta.url),
  ),
) as { readonly series: string; readonly cases: readonly GoldenCase[] };
const CASES = FIXTURE.cases;
assertEquals(FIXTURE.series, "embeddinggemma-2-text", "the fixture's series is the one read here");

/** The options of a case's call (its prompt when it has one). */
const optionsOf = (entry: GoldenCase): embeddinggemma.EmbedOptions =>
  entry.prompt === undefined ? {} : { prompt: entry.prompt };

/** The widths of the truncation cases (a short text and the 2,048-token text). */
const TRUNCATIONS: readonly (readonly [string, number])[] = [
  ["query-en", 256],
  ["document-2048", 128],
];
const SHORT = "query-en";
const LONG = "document-2048";
/** The text at the bound (`pipelineConfig.maxTokens`). */
const LONGEST = "document-8192";

/** One layer's unsplit attention scores at 8,192 tokens (4 heads × 8,192² f32). */
const UNSPLIT_SCORES_BYTES = 4 * 8192 * 8192 * 4;

const caseOf = (name: string): GoldenCase => {
  const found = CASES.find((entry) => entry.name === name);
  if (found === undefined) throw new Error(`the golden-case fixture has no case '${name}'`);
  return found;
};

/**
 * Whether the file exists. MUST: anything but NotFound propagates (a permission error read as "no
 * golden" would turn a comparison that never ran into a quiet SKIP).
 */
const exists = (url: URL): boolean => {
  try {
    return statSync(url).isFile;
  } catch (cause) {
    if (isNotFound(cause)) return false;
    throw cause;
  }
};

/** Whether the golden of `name` in `series` is compared in this run (opt-in, and the io present). */
const goldenOf = (series: string, name: string): boolean =>
  SERIES_CHECKS && exists(ioFile(series, name));
/** Whether every golden of `series` is there (the token-id check reads the f32 series'). */
const goldensOf = (series: string): boolean => CASES.every((entry) => goldenOf(series, entry.name));
for (const { series } of PRESETS) {
  if (SERIES_CHECKS && !goldensOf(series)) {
    console.warn(
      `[karume] outputs/series/${series}/ lacks io goldens, so the EmbeddingGemma 2 pipeline's golden` +
        " comparisons against it SKIP (the series e2e names what is missing)",
    );
  }
}
const TOKEN_GOLDENS = goldensOf(presetOf(DEFAULT_PRESET).series);

/** The io golden of a case, parsed. */
const readGolden = async (series: string, name: string): Promise<SafetensorsFile> => {
  const bytes = await readFile(ioFile(series, name));
  return parseSafetensors(
    bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength),
  );
};

/** A tensor of the golden by key (fails loudly when absent). */
const goldenTensor = (io: SafetensorsFile, key: string, dtype: "i32" | "f32"): Tensor => {
  const view = io.tensors.get(key);
  assert(view !== undefined, `${key} is missing from the golden`);
  return ioTensor(io, view, dtype);
};

/** The golden output `[1, 768]` as a vector. */
const goldenVector = (io: SafetensorsFile): Float32Array<ArrayBuffer> => {
  const tensor = goldenTensor(io, "output.0", "f32");
  assert(tensor.dtype === "f32");
  return tensor.data;
};

const unitNormError = (vector: Float32Array<ArrayBuffer>): number =>
  Math.abs(Math.hypot(...vector) - 1);

/** The cosine of two vectors (in f64; the pipeline's vectors are unit vectors, the norms are kept anyway). */
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

/**
 * Where the settlements and the measurements go
 * (`outputs/verify/<environment key>/<date>_embeddinggemma2-pipeline/` — safe to delete).
 */
const results = openResults("embeddinggemma2-pipeline");

/** The sha256 rows of the pipeline's outputs (rows per environment). */
const references = openReferences(
  new URL("../../models/tests/fixtures/references/embeddinggemma.json", import.meta.url),
);

const caseIdOf = (quant: string, name: string, dimension?: number): string =>
  `embeddinggemma-2/${quant}/${name}${dimension === undefined ? "" : `/${dimension}`}`;
const RUNNABLE = MIRROR && GPU_AVAILABLE;
const CASE_IDS = RUNNABLE
  ? PRESETS.flatMap(({ quant }) => [
      ...CASES.map((entry) => caseIdOf(quant, entry.name)),
      ...TRUNCATIONS.map(([name, dimension]) => caseIdOf(quant, name, dimension)),
    ])
  : [];

describe.skipIf(!MIRROR)("EmbeddingGemma 2 mirror: the manifest and the tokenizer (no GPU)", () => {
  let rules: EmbedTextRules;
  beforeAll(async () => {
    const manifest = parseManifest(await readTextFile(new URL("karume.json", MIRROR_ROOT)));
    const entry = manifest.models["text"];
    assert(entry !== undefined, "the mirror has no model 'text'");
    const config = parseEmbeddingGemmaPipelineConfig(entry.pipelineConfig);
    const tokenizer = new Tokenizer(
      parseGemmaTokenizerAsset(await readMirrorAsset(MIRROR_ROOT, "tokenizer")),
    );
    rules = embedTextRules(tokenizer, config);
  });

  it("declares model text with pipeline embeddinggemma/1, the presets bf16 (default) and i8, and the four widths", async () => {
    const manifest = parseManifest(await readTextFile(new URL("karume.json", MIRROR_ROOT)));
    assertEquals(manifest.defaultModel, "text");
    const entry = manifest.models["text"];
    assertEquals([entry.pipeline.name, entry.pipeline.major], ["embeddinggemma", 1]);
    assertEquals(
      [entry.defaultQuant, entry.available.quants],
      [DEFAULT_PRESET, PRESETS.map(({ quant }) => quant)],
    );
    assertEquals(rules.config.dimensions, [768, 512, 256, 128]);
    assertEquals(rules.config.maxTokens, 8192);
    assertEquals(rules.placeholders.size, 3);
  });

  it("gives each preset its own storage of the model, the one bf16 vision tower and audio tower, an empty session and the binding its narrowed device is capped at", async () => {
    const manifest = parseManifest(await readTextFile(new URL("karume.json", MIRROR_ROOT)));
    const entry = manifest.models["text"];
    for (const preset of PRESETS) {
      const quant = entry.quants[preset.quant];
      assertEquals(
        quant.weights,
        { model: preset.quant, vision: "bf16", audio: "bf16" },
        `${preset.quant}: weights`,
      );
      assertEquals(quant.session, {}, `${preset.quant}: session`);
      assertEquals(
        quant.requiredLimits?.maxStorageBufferBindingSize ?? WEBGPU_DEFAULT_BINDING_BYTES,
        preset.cappedBindingBytes,
        `${preset.quant}: the binding its largest buffer needs`,
      );
    }
    assertEquals(
      entry.quants["i8"].requiredLimits,
      undefined,
      "i8 needs no limit over the default",
    );
  });

  it("gives each golden text, with its prompt, the token count the recipe pinned", () => {
    for (const entry of CASES) {
      const request = planEmbedRequest(rules, entry.text, optionsOf(entry));
      assertEquals(request.ids.length, entry.tokens, entry.name);
    }
  });

  it.skipIf(!TOKEN_GOLDENS)(
    `gives each golden text exactly the golden's input_ids (upstream's tokenizers over prefix + body)${SERIES_CHECKS_OPT_IN}`,
    async () => {
      for (const entry of CASES) {
        const request = planEmbedRequest(rules, entry.text, optionsOf(entry));
        const ids = goldenTensor(
          await readGolden(presetOf(DEFAULT_PRESET).series, entry.name),
          "input.input_ids",
          "i32",
        );
        assertEquals(ids.shape, [1, entry.tokens], `${entry.name}: golden shape`);
        assertEquals([...request.ids], [...ids.data], `${entry.name}: ids`);
      }
    },
  );
});

for (const preset of PRESETS) {
  describe.skipIf(!RUNNABLE)(
    `EmbeddingGemma 2 pipeline over the mirror, preset ${preset.quant} (real GPU)`,
    () => {
      let gpu: GpuContext;
      let pipeline: embeddinggemma.Pipeline;
      let fromPretrainedMs = 0;
      /** The diagnostics of the latest run (the pipeline's observation hook). */
      let lastDiagnostics: SessionDiagnostics | undefined;
      /** Forgets the latest run's diagnostics (a function, so the hook's later write is not narrowed away). */
      const clearDiagnostics = (): void => {
        lastDiagnostics = undefined;
      };
      beforeAll(async () => {
        // The rows written below are this machine's: the adapter the key was taken from must be the
        // one running.
        gpu = verifyRunningGpu(await acquireTestGpu());
        const start = performance.now();
        pipeline = await embeddinggemma.Pipeline.fromPretrained(mirrorDirectory(MIRROR_ROOT), {
          gpu,
          quant: preset.quant,
          onRunDiagnostics: (diagnostics) => {
            lastDiagnostics = diagnostics;
          },
        });
        fromPretrainedMs = milliseconds(start);
        console.log(
          `[embeddinggemma2-pipeline] ${preset.quant}: fromPretrained ${fromPretrainedMs} ms (shared GPU)`,
        );
      });
      // Teardown, with the check that a disposed pipeline refuses a call; the shared GPU is the test's.
      afterAll(async () => {
        if (gpu === undefined) return;
        try {
          if (pipeline === undefined) return;
          await pipeline.dispose();
          await assertRejects(() => pipeline.embed("text"), Error, "cannot embed after dispose");
        } finally {
          gpu.destroy();
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

      for (const entry of CASES) {
        const golden = goldenOf(preset.series, entry.name);
        it(`${entry.name} (T=${entry.tokens}): a unit vector, its sha256 row${golden ? `, and the torch golden of ${preset.series} within the band` : ""}`, async () => {
          await settle(caseIdOf(preset.quant, entry.name), async (measurements) => {
            const start = performance.now();
            const vector = await pipeline.embed(entry.text, optionsOf(entry));
            const embedMs = milliseconds(start);
            assertEquals(vector.length, 768, `${entry.name}: width`);
            assert(
              unitNormError(vector) < 1e-5,
              `${entry.name}: |norm − 1| = ${unitNormError(vector)}`,
            );
            if (golden) {
              const report = compareTensors(
                { dtype: "f32", data: vector },
                {
                  dtype: "f32",
                  data: goldenVector(await readGolden(preset.series, entry.name)),
                },
                preset.tolerance,
              );
              measurements.push({
                output: "embedding",
                maxAbs: report.maxAbsError,
                maxRel: report.maxRelError,
                tolerance: preset.tolerance,
                stage: "karume",
              });
              assert(report.pass, `${preset.quant} ${entry.name}: ${formatAllclose(report)}`);
            }
            const note = `T=${entry.tokens}: first embed ${embedMs} ms`;
            console.log(`[embeddinggemma2-pipeline] ${preset.quant} ${entry.name} ${note}`);
            return { vector, note };
          });
        });
      }

      for (const [name, dimension] of TRUNCATIONS) {
        const entry = caseOf(name);
        const golden = goldenOf(preset.series, name);
        it(`${name} at width ${dimension}: the full vector's leading slice re-normalised, its sha256 row${golden ? ", and the golden's slice within the band" : ""}`, async () => {
          await settle(caseIdOf(preset.quant, name, dimension), async (measurements) => {
            const full = await pipeline.embed(entry.text, optionsOf(entry));
            const vector = await pipeline.embed(entry.text, { ...optionsOf(entry), dimension });
            assertEquals(vector.length, dimension);
            assertEquals(
              [...vector],
              [...truncateEmbedding(full, dimension)],
              "the slice of the same path",
            );
            assert(unitNormError(vector) < 1e-5, `|norm − 1| = ${unitNormError(vector)}`);
            if (golden) {
              const expected = truncateEmbedding(
                goldenVector(await readGolden(preset.series, name)),
                dimension,
              );
              const report = compareTensors(
                { dtype: "f32", data: vector },
                { dtype: "f32", data: expected },
                preset.tolerance,
              );
              measurements.push({
                output: `embedding[:${dimension}]`,
                maxAbs: report.maxAbsError,
                maxRel: report.maxRelError,
                tolerance: preset.tolerance,
                stage: "karume",
              });
              assert(
                report.pass,
                `${preset.quant} ${name} at ${dimension}: ${formatAllclose(report)}`,
              );
            }
            return { vector, note: `T=${entry.tokens}, dimension ${dimension}` };
          });
        });
      }

      it("a repeated call gives the same bits, the timings of a short, the 2,048- and the 8,192-token text, and the 8,192-token plan under 1 GiB", async () => {
        const timings: string[] = [`fromPretrained ${fromPretrainedMs} ms`];
        for (const name of [SHORT, LONG, LONGEST]) {
          const entry = caseOf(name);
          const firstStart = performance.now();
          const first = await pipeline.embed(entry.text, optionsOf(entry));
          const firstMs = milliseconds(firstStart);
          // Cleared before the measured run, so the plan backing read below is that run's report.
          clearDiagnostics();
          const againStart = performance.now();
          const again = await pipeline.embed(entry.text, optionsOf(entry));
          const againMs = milliseconds(againStart);
          assertEquals([...again], [...first], `${name}: a repeated call differs`);
          assert(lastDiagnostics !== undefined, "the observation hook saw no run");
          const { planBacking } = lastDiagnostics;
          const planMiB = (planBacking.residentBytes / 2 ** 20).toFixed(1);
          timings.push(
            `${name} (T=${entry.tokens}) embed ${firstMs} ms, again ${againMs} ms, plan backing ${planMiB} MiB`,
          );
          if (name === LONGEST) {
            assert(
              planBacking.residentBytes < UNSPLIT_SCORES_BYTES,
              `the 8,192-token plan backing is ${planBacking.residentBytes} B, not under one layer's unsplit scores` +
                ` (${UNSPLIT_SCORES_BYTES} B) — the row-block budget did not reach the Session`,
            );
          }
        }
        console.log(`[embeddinggemma2-pipeline] ${preset.quant} timings: ${timings.join("; ")}`);
      });
    },
  );
}

/**
 * Embeds every golden text through one preset on `gpu` and disposes the pipeline (the two presets
 * are never resident together).
 */
const embedGoldenTexts = async (
  gpu: GpuContext,
  quant: string,
): Promise<ReadonlyMap<string, Float32Array<ArrayBuffer>>> => {
  const pipeline = await embeddinggemma.Pipeline.fromPretrained(mirrorDirectory(MIRROR_ROOT), {
    gpu,
    quant,
  });
  try {
    const vectors = new Map<string, Float32Array<ArrayBuffer>>();
    for (const entry of CASES) {
      vectors.set(entry.name, await pipeline.embed(entry.text, optionsOf(entry)));
    }
    return vectors;
  } finally {
    await pipeline.dispose();
  }
};

describe.skipIf(!RUNNABLE)("EmbeddingGemma 2: the i8 preset against bf16 (real GPU)", () => {
  let gpu: GpuContext;
  beforeAll(async () => {
    gpu = await acquireTestGpu();
  });
  afterAll(() => {
    gpu?.destroy();
  });

  it(`every golden text keeps a cosine of at least ${I8_COSINE_FLOOR} to its bf16 vector at every width`, async () => {
    const original = await embedGoldenTexts(gpu, "bf16");
    const int8 = await embedGoldenTexts(gpu, "i8");
    const widths = [768, 512, 256, 128];
    const below: string[] = [];
    const lines: string[] = [];
    for (const entry of CASES) {
      const left = int8.get(entry.name);
      const right = original.get(entry.name);
      assert(left !== undefined && right !== undefined, `${entry.name}: a vector is missing`);
      // The vectors differ: the i8 preset's weights really are the int8 ones.
      assert(
        left.some((value, index) => value !== right[index]),
        `${entry.name}: the i8 vector equals the bf16 one bit for bit (the i8 preset ran the bf16 weights)`,
      );
      const cosines = widths.map((width) =>
        cosine(truncateEmbedding(left, width), truncateEmbedding(right, width)),
      );
      lines.push(
        `${entry.name} ${cosines.map((value, index) => `${widths[index]}: ${value.toFixed(7)}`).join(", ")}`,
      );
      cosines.forEach((value, index) => {
        if (!(value >= I8_COSINE_FLOOR)) below.push(`${entry.name} at ${widths[index]}: ${value}`);
      });
    }
    console.log(`[embeddinggemma2-pipeline] cosine i8 × bf16 — ${lines.join("; ")}`);
    assertEquals(below, [], `cosines under ${I8_COSINE_FLOOR}`);
  });
});

/** The graph's bytes per token (the per-layer-embedding intermediate `[1, T, 12288]` f32). */
const BYTES_PER_TOKEN = 12288 * 4;

/**
 * The prefix of `text` that tokenizes, without a prompt, to exactly `count` ids (`<bos>` and
 * `<eos>` included): a binary search for the shortest prefix with at least `count` ids, then the
 * lengths around it (a few more characters can merge tokens, so the count is not strictly
 * monotone). Fails loudly when no prefix has the count.
 */
const prefixOfTokens = (pipeline: embeddinggemma.Pipeline, text: string, count: number): string => {
  const countOf = (length: number): number => pipeline.tokenize(text.slice(0, length)).length;
  assert(countOf(text.length) > count, `the text has no more than ${count} tokens`);
  let low = 0;
  let high = text.length;
  while (low < high) {
    const middle = Math.floor((low + high) / 2);
    if (countOf(middle) >= count) high = middle;
    else low = middle + 1;
  }
  for (let offset = 0; offset <= 64; offset += 1) {
    for (const length of [low + offset, low - offset]) {
      if (length >= 0 && length <= text.length && countOf(length) === count) {
        return text.slice(0, length);
      }
    }
  }
  throw new Error(`no prefix of the text tokenizes to exactly ${count} ids`);
};

for (const preset of PRESETS) {
  const mebibytes = preset.cappedBindingBytes / 2 ** 20;
  describe.skipIf(!RUNNABLE)(
    `EmbeddingGemma 2 pipeline, preset ${preset.quant}, on a device capped at ${mebibytes} MiB (real GPU)`,
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
        pipeline = await embeddinggemma.Pipeline.fromPretrained(mirrorDirectory(MIRROR_ROOT), {
          gpu,
          quant: preset.quant,
        });
      });
      afterAll(async () => {
        if (gpu === undefined) return;
        try {
          await pipeline?.dispose();
        } finally {
          gpu.destroy();
        }
      });

      it("runs an input of exactly the device's limit, refuses one token more, and still embeds a short text", async () => {
        assertEquals(
          [gpu.limits.maxStorageBufferBindingSize, gpu.limits.maxBufferSize],
          [preset.cappedBindingBytes, preset.cappedBindingBytes],
          "the device's limits are not narrowed (the check would see nothing)",
        );
        assertEquals(
          preset.cappedMaxTokens,
          Math.floor(preset.cappedBindingBytes / BYTES_PER_TOKEN),
        );
        const { text } = caseOf(LONGEST);
        const atLimit = prefixOfTokens(pipeline, text, preset.cappedMaxTokens);
        const overLimit = prefixOfTokens(pipeline, text, preset.cappedMaxTokens + 1);

        const vector = await pipeline.embed(atLimit);
        assertEquals(vector.length, 768);
        assert(unitNormError(vector) < 1e-5, `|norm − 1| = ${unitNormError(vector)}`);

        const error = await assertRejects(() => pipeline.embed(overLimit), ModelInputError);
        assert(
          error.message.includes(
            `the input is ${preset.cappedMaxTokens + 1} tokens (<bos> and <eos> included), over this` +
              ` device's limit of ${preset.cappedMaxTokens}`,
          ),
          error.message,
        );

        const short = caseOf(SHORT);
        const after = await pipeline.embed(short.text, optionsOf(short));
        assert(
          unitNormError(after) < 1e-5,
          `after the refusal: |norm − 1| = ${unitNormError(after)}`,
        );
      });
    },
  );
}

if (RUNNABLE) references.warnMissing(CASE_IDS);
registerReferenceGate(references, { runnable: RUNNABLE, caseIds: CASE_IDS });
