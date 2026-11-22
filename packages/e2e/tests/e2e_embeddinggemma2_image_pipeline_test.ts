// Image input of the EmbeddingGemma family over the local distribution of EmbeddingGemma 2 —
// `embeddinggemma.Pipeline` built by `fromPretrained` from `models/karume-embeddinggemma-2/` with
// `vision: true` (the recipe `tools/export-recipes/embeddinggemma2/distribution.py`; model `text`,
// every preset of the mirror — `bf16`, the default, and `i8`, each with the same bf16 vision tower).
//
// The reference is the tracked fixture `packages/models/tests/fixtures/embeddinggemma2/
// multimodal-reference.{json,safetensors}` (`multimodal_reference.py`): five whole inputs holding
// images (an image alone, a text then an image, an image then a text, a `query`-prompted text with
// an image, two images), with their RGB8 images, the exact token ids upstream's SentenceTransformer
// ran and the 768-wide vectors it returned (f32, CPU).
//
// ## Checks
//
// 1. **The token ids** (no GPU): the ids the pipeline runs for each input — `<bos>`, the text runs
//    tokenized as one string each (the prompt's prefix glued to the first), each image as
//    `<|image>` + n × `<|image|>` + `<image|>`, `<eos>` — equal upstream's exactly, and each image's
//    soft rows start where upstream's do; the mirror declares `pipelineConfig.vision` as the
//    recipe reads it and every preset maps the one bf16 tower.
// 2. **The five inputs through each preset** (real GPU): `tokenize` gives the fixture's ids, the
//    vector is a unit vector, it bit-for-bit equals this machine's sha256 row of that preset (row
//    ID `embeddinggemma-2/<preset>/image/<case>`), and against upstream's vector: `bf16` within
//    {@link IMAGE_TOLERANCE} (its derivation is there), `i8` at a cosine of at least
//    {@link I8_COSINE_FLOOR} (its text tower is the int8 copy; no golden of the int8 model holds
//    images). Each case records maxAbs, maxRel and the cosine.
// 3. **A text call on the vision pipeline** is the text call of the pipeline without it: its vector
//    has the sha256 row the text pipeline e2e pins (`embeddinggemma-2/<preset>/query-en`).
// 4. **The tower within the preset's limits** (real GPU), per preset, on a device whose binding and
//    buffer limits are capped at the preset's own need (`bf16` 256 MiB, `i8` the WebGPU default
//    128 MiB): an image of 280 soft tokens — whose tower scores are 12 × 2,520² f32 = 290.7 MiB
//    unsplit, more than either cap — embeds to the same bits as on the uncapped device, so the tower
//    adds no limit to the preset's. NOTE: this does not show the family's row-block budget reaching
//    the tower's Session — on a capped device the runtime splits the scores by the binding limit
//    anyway (measured: the tower with the budget removed passes it too); the budget bounds the plan
//    on a device that binds more (the tower's plan backing at 280 soft tokens is 135.1 MiB with it,
//    329.1 MiB without, on a 2 GiB binding — `pipeline.ts`).
//
// Without the mirror every case SKIPs explicitly (the distribution gate owns that red); without a
// GPU the GPU cases SKIP. Rows are created with `KARUME_REFERENCE=write` on the lane.

import { readFile, readTextFile } from "@karume/test-support/fs";
import { afterAll, beforeAll, describe, it } from "vite-plus/test";
import { assert, assertEquals, fail } from "@std/assert";
import { parseManifest } from "@karume/hub";
import { type GpuContext, parseSafetensors, type SafetensorsFile } from "@karume/runtime";
import * as embeddinggemma from "../../models/embeddinggemma.ts";
import { parseEmbeddingGemmaPipelineConfig } from "../../models/src/embeddinggemma/config.ts";
import {
  type EmbedTextRules,
  embedTextRules,
  planEmbedRequest,
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
 * Measured through the pipeline on node-nvidia-geforce-rtx-3080-ti (2026-11-22, five inputs,
 * `[768]` unit vectors; cosine in f64). The two columns on the right place the error: the text
 * graph on the GPU fed upstream's own soft tokens (the vision series' goldens) instead of the
 * tower's, and the two graphs composed in torch f32 (the fixture's `composition`):
 *
 * | Case             | T   | maxAbs  | maxRel  | cosine         | GPU text + upstream's soft tokens | torch composition |
 * | ---------------- | --- | ------- | ------- | -------------- | --------------------------------- | ----------------- |
 * | image-only       | 260 | 2.32e-7 | 1.06e-2 | 0.999999999998 | 5.96e-8                           | 5.87e-8           |
 * | text-then-image  | 264 | 5.44e-6 | 4.48e-2 | 0.999999999136 | 9.76e-7                           | 1.82e-6           |
 * | image-then-text  | 257 | 1.79e-7 | 1.06e-3 | 0.999999999999 | 1.79e-7                           | 7.06e-8           |
 * | query-text-image | 271 | 3.20e-7 | 6.88e-3 | 0.999999999996 | 1.04e-7                           | 7.22e-8           |
 * | two-images       | 398 | 3.58e-7 | 9.64e-3 | 0.999999999995 | 8.99e-8                           | 8.94e-8           |
 *
 * The text graph on the GPU is at the torch composition's error when it reads upstream's soft
 * tokens (at most 9.8e-7). What is left is the tower's GPU noise — its soft tokens are 6.5e-5 to
 * 3.35e-4 from upstream's (the vision series e2e) — carried through the text tower; it is largest
 * for `text-then-image` (the hard-edged `shapes-wide` image after a text), the input whose vector
 * is the most sensitive to its soft tokens: torch's own f32 composition is 25 times further off
 * there than on the other four. atol 2e-5 is 3.7 times the worst; rtol 0 — a unit vector's values
 * thin out to near 0 (maxRel 4.5e-2 at an element of 1e-4), so atol alone decides. The breakages
 * this guards against are far above it: patches pooled in the processor's raster order or the two
 * position axes swapped move the soft tokens by 5.9 to 6.5, and a soft row the mask leaves out runs
 * the placeholder's token lookup instead.
 */
const IMAGE_TOLERANCE: Tolerance = { atol: 2e-5, rtol: 0 };

/**
 * The floor of the cosine between the `i8` preset's vector of an input holding an image and
 * upstream's: the text pipeline e2e's floor between the `i8` and the `bf16` vector of a text
 * (`I8_COSINE_FLOOR` of `e2e_embeddinggemma2_pipeline_test.ts`, whose derivation is there).
 *
 * Measured on node-nvidia-geforce-rtx-3080-ti (2026-11-22): 0.999759 (`image-then-text`) to
 * 0.999908 (`image-only`), maxAbs 1.74e-3 to 2.66e-3 — the text tower's int8 rounding (its tower
 * is the same bf16 one); within the 0.99940 to 0.99993 the ten golden texts measure against their
 * `bf16` vectors. No golden of the int8 model holds images, so no tighter band is drawn.
 */
const I8_COSINE_FLOOR = 0.999;

/** One preset of the mirror and how its vectors are held to upstream's. */
type PresetSpec = {
  readonly quant: string;
  /** `tolerance`: an allclose against upstream's vector; `cosine`: a floor of the cosine to it. */
  readonly check:
    | { readonly kind: "tolerance"; readonly tolerance: Tolerance }
    | {
        readonly kind: "cosine";
        readonly floor: number;
      };
  /** The binding and buffer limit of its narrowed device (check 4). */
  readonly cappedBindingBytes: number;
};

const PRESETS: readonly PresetSpec[] = [
  {
    quant: "bf16",
    check: { kind: "tolerance", tolerance: IMAGE_TOLERANCE },
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
      " EmbeddingGemma 2 image-input checks SKIP. Assemble: cd tools/export-recipes && uv run" +
      " python dist.py --pipeline embeddinggemma-2 (after the text and vision series)",
  );
}
const RUNNABLE = MIRROR && GPU_AVAILABLE;

/** The fixture (`tools/export-recipes/embeddinggemma2/multimodal_reference.py`). */
type FixturePart =
  | { readonly type: "text"; readonly text: string }
  | { readonly type: "image"; readonly image: string };
type FixtureCase = {
  readonly name: string;
  readonly parts: readonly FixturePart[];
  readonly prompt?: string;
  readonly tokens: number;
  readonly imageBlocks: readonly {
    readonly image: string;
    readonly start: number;
    readonly softTokens: number;
  }[];
  readonly composition: { readonly maxAbs: number; readonly cosine: number };
};
type Fixture = {
  readonly imageTokens: { readonly begin: number; readonly soft: number; readonly end: number };
  readonly softTokenBudget: number;
  readonly images: readonly {
    readonly name: string;
    readonly width: number;
    readonly height: number;
  }[];
  readonly cases: readonly FixtureCase[];
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

/** A tensor of the fixture by key, as a typed view (fails loudly when absent or of another dtype). */
const fixtureView = (key: string, dtype: "U8" | "I32" | "F32"): Uint8Array<ArrayBuffer> => {
  const view = TENSORS.tensors.get(key);
  assert(view !== undefined, `the fixture has no tensor '${key}'`);
  assertEquals(view.dtype, dtype, `${key}: dtype`);
  return new Uint8Array(TENSORS.buffer, view.byteOffset, view.byteLength);
};
const fixtureIds = (name: string): number[] => {
  const bytes = fixtureView(`${name}.input_ids`, "I32");
  return [...new Int32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4)];
};
const fixtureVector = (name: string): Float32Array<ArrayBuffer> => {
  const bytes = fixtureView(`${name}.embedding`, "F32");
  return new Float32Array(
    bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength),
  );
};
const fixtureImage = (name: string): embeddinggemma.Rgb8Image => {
  const entry = FIXTURE.images.find((image) => image.name === name);
  assert(entry !== undefined, `the fixture has no image '${name}'`);
  return {
    data: fixtureView(`image.${name}`, "U8").slice(),
    width: entry.width,
    height: entry.height,
  };
};

/** The content parts of a fixture case (its images decoded from the fixture's tensors). */
const partsOf = (entry: FixtureCase): readonly embeddinggemma.EmbedContentPart[] =>
  entry.parts.map((part) =>
    part.type === "text" ? part : { type: "image", image: fixtureImage(part.image) },
  );
const optionsOf = (entry: FixtureCase): embeddinggemma.EmbedOptions =>
  entry.prompt === undefined ? {} : { prompt: entry.prompt };

/**
 * An image of 280 soft tokens (672 × 960 px — 14 × 20 blocks, the resize rule leaves it as it is):
 * a smooth pattern, so its embedding is an ordinary one.
 */
const FULL_BUDGET_IMAGE: embeddinggemma.Rgb8Image = (() => {
  const width = 672;
  const height = 960;
  const data = new Uint8Array(width * height * 3);
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const at = (y * width + x) * 3;
      data[at] = (x * 255) / (width - 1);
      data[at + 1] = (y * 255) / (height - 1);
      data[at + 2] = ((x + y) % 64) * 4;
    }
  }
  return { data, width, height };
})();
const FULL_BUDGET_TOKENS = 280;

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

/** `outputs/verify/<environment key>/<date>_embeddinggemma2-image-pipeline/` (safe to delete). */
const results = openResults("embeddinggemma2-image-pipeline");

/** The sha256 rows of the pipeline's outputs (the text pipeline e2e's file — rows per environment). */
const references = openReferences(
  new URL("../../models/tests/fixtures/references/embeddinggemma.json", import.meta.url),
);

const imageCaseId = (quant: string, name: string): string =>
  `embeddinggemma-2/${quant}/image/${name}`;
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
      ...FIXTURE.cases.map((entry) => imageCaseId(quant, entry.name)),
      imageCaseId(quant, "full-budget"),
      textCaseId(quant),
    ])
  : [];

describe.skipIf(!MIRROR)("EmbeddingGemma 2 mirror: image input on the host (no GPU)", () => {
  let rules: EmbedTextRules;
  beforeAll(async () => {
    const manifest = parseManifest(await readTextFile(new URL("karume.json", MIRROR_ROOT)));
    const config = parseEmbeddingGemmaPipelineConfig(manifest.models["text"]?.pipelineConfig);
    const tokenizer = new Tokenizer(
      parseGemmaTokenizerAsset(await readMirrorAsset(MIRROR_ROOT, "tokenizer")),
    );
    rules = embedTextRules(tokenizer, config, { images: true });
  });

  it("declares pipelineConfig.vision as the recipe reads it, and every preset maps the one bf16 tower", async () => {
    const manifest = parseManifest(await readTextFile(new URL("karume.json", MIRROR_ROOT)));
    const entry = manifest.models["text"];
    assert(entry !== undefined, "the mirror has no model 'text'");
    assertEquals(rules.images, {
      softTokenBudget: FIXTURE.softTokenBudget,
      imageTokenId: FIXTURE.imageTokens.soft,
      beginImageTokenId: FIXTURE.imageTokens.begin,
      endImageTokenId: FIXTURE.imageTokens.end,
      hiddenSize: 512,
    });
    assertEquals(Object.keys(entry.weights["vision"] ?? {}), ["bf16"]);
    for (const { quant } of PRESETS) {
      assertEquals(entry.quants[quant]?.weights["vision"], "bf16", `${quant}: the tower`);
    }
  });

  it("covers the fixture's five inputs (a positive control)", () => {
    assertEquals(FIXTURE.cases.length, 5);
  });

  for (const entry of FIXTURE.cases) {
    it(`${entry.name}: the ids embed runs are upstream's exactly, each image's soft rows where upstream's are`, () => {
      const request = planEmbedRequest(rules, partsOf(entry), optionsOf(entry));
      assertEquals([...request.ids], fixtureIds(entry.name));
      assertEquals(request.ids.length, entry.tokens);
      assertEquals(
        request.images.map(({ start, softTokens }) => ({ start, softTokens })),
        entry.imageBlocks.map(({ start, softTokens }) => ({ start, softTokens })),
      );
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

/** The vector of the full-budget image on the uncapped device, per preset (check 4 compares with it). */
const fullBudgetVectors = new Map<string, Float32Array<ArrayBuffer>>();

for (const preset of PRESETS) {
  describe.skipIf(!RUNNABLE)(
    `EmbeddingGemma 2 image input over the mirror, preset ${preset.quant} (real GPU)`,
    () => {
      let gpu: GpuContext;
      let pipeline: embeddinggemma.Pipeline;
      beforeAll(async () => {
        gpu = verifyRunningGpu(await acquireTestGpu());
        const start = performance.now();
        pipeline = await embeddinggemma.Pipeline.fromPretrained(mirrorDirectory(MIRROR_ROOT), {
          gpu,
          quant: preset.quant,
          vision: true,
        });
        console.log(
          `[embeddinggemma2-image] ${preset.quant}: fromPretrained with the tower ${milliseconds(start)} ms`,
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

      for (const entry of FIXTURE.cases) {
        it(`${entry.name} (T=${entry.tokens}): upstream's ids, a unit vector near upstream's, its sha256 row`, async () => {
          await settle(imageCaseId(preset.quant, entry.name), async (measurements) => {
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
            console.log(`[embeddinggemma2-image] ${preset.quant} ${entry.name} ${note}`);
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

      it("an image of 280 soft tokens (the budget): a unit vector, its sha256 row", async () => {
        await settle(imageCaseId(preset.quant, "full-budget"), async () => {
          const parts = [{ type: "image", image: FULL_BUDGET_IMAGE }] as const;
          assertEquals(pipeline.tokenize(parts).length, FULL_BUDGET_TOKENS + 4);
          const start = performance.now();
          const vector = await pipeline.embed(parts);
          const note = `T=${FULL_BUDGET_TOKENS + 4}: embed ${milliseconds(start)} ms`;
          assert(unitNormError(vector) < 1e-5, `|norm − 1| = ${unitNormError(vector)}`);
          fullBudgetVectors.set(preset.quant, vector);
          return { vector, note };
        });
      });

      it("a text call on the vision pipeline has the text pipeline's sha256 row (query-en)", async () => {
        await settle(textCaseId(preset.quant), async () => {
          const vector = await pipeline.embed(
            QUERY_EN.text,
            QUERY_EN.prompt === undefined ? {} : { prompt: QUERY_EN.prompt },
          );
          return { vector, note: "a text through the pipeline built with vision: true" };
        });
      });
    },
  );
}

for (const preset of PRESETS) {
  const mebibytes = preset.cappedBindingBytes / 2 ** 20;
  describe.skipIf(!RUNNABLE)(
    `EmbeddingGemma 2 image input, preset ${preset.quant}, on a device capped at ${mebibytes} MiB (real GPU)`,
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
          vision: true,
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

      it("embeds an image of 280 soft tokens (tower scores of 290.7 MiB unsplit) to the uncapped device's bits, within the preset's limits", async () => {
        assertEquals(
          [gpu.limits.maxStorageBufferBindingSize, gpu.limits.maxBufferSize],
          [preset.cappedBindingBytes, preset.cappedBindingBytes],
          "the device's limits are not narrowed (the check would see nothing)",
        );
        const unsplit = 12 * (9 * FULL_BUDGET_TOKENS) ** 2 * 4;
        assert(unsplit > preset.cappedBindingBytes, "the image's scores fit the cap unsplit");
        const vector = await pipeline.embed([{ type: "image", image: FULL_BUDGET_IMAGE }]);
        const uncapped = fullBudgetVectors.get(preset.quant);
        assert(uncapped !== undefined, "the uncapped run of the same image did not run first");
        assertEquals([...vector], [...uncapped]);
      });
    },
  );
}

if (RUNNABLE) references.warnMissing(CASE_IDS);
registerReferenceGate(references, { runnable: RUNNABLE, caseIds: CASE_IDS });
