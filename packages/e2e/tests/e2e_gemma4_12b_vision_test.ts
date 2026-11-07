// Real-weight Gemma 4 12B image input (ADR 0144), through the public pipeline. The cases:
//
// (a) preprocessing: the corpus PNGs through the repository's reader and `imageRows` give the bytes
//     of upstream's processor (`reference.vision.<image>.pixelsSha256` / `positionsSha256`) — no GPU
// (b) the embedder: `Pipeline.embedImage` against `vision.<image>.safetensors` (upstream's module in
//     F32 over the checkpoint's values), within a band
// (c) greedy with an image: the prompt `chatImagePrompt` renders equals the golden's, and the greedy
//     chain of `sequence()` with `softBlocks` equals the golden's expected list token for token. The
//     goldens are torch greedy with the suppressed ids at -inf, on the masks of upstream's
//     `create_masks_for_vision_model` (ADR 0144 decision 1)
// (d) a text prompt on the same pipeline still equals the 12B product series' greedy golden: the
//     graph's `where` costs a text run nothing it can see
// (e) the census: the prefill with the image ran the `:bidir` QK kernel, the decode after it did not
//
// ## Assets
//
// The model is the distribution mirror's model `12b`, quant `i4` (the reference path), opened through
// the manifest with `vision: true` (ADR 0139 decision 1). Without the mirror, or with a mirror whose
// 12B has no `vision` weights, every case SKIPs explicitly. The goldens are the vision series' outputs
// in the test-asset root (`goldens/gemma4-12b-vision/` — `reference.json`, `vision.<image>` and
// `greedy.<image>`) and the corpus PNGs (`corpus/wan-i2v/`), under ADR 0139's fetched rule: absent from
// a complete fetch of the pinned dataset ⇒ FAIL; otherwise SKIP with the missing files in the title.
//
// ## GPU memory
//
// One 12B pipeline at a time: the ≈ 9 GiB of `e2e_gemma4_12b_test.ts` plus the embedder (≈ 100 MiB of
// bf16 weights) and the host rows of an image. The census case builds its own pipeline after the first
// is disposed, because `onRunDiagnostics` changes the decode path the goldens should see.

import { isNotFound, readFile, readTextFileSync, statSync } from "@karume/test-support/fs";
import { decodePng } from "@karume/test-support/png-decode";
import {
  corpusDir,
  goldenDir,
  testAssetsFetched,
  TEST_ASSETS,
  TEST_ASSETS_ROOT,
} from "@karume/test-support/test-assets";
import { afterAll, beforeAll, describe, it, test } from "vite-plus/test";
import { assert, assertEquals } from "@std/assert";
import { type Manifest, MANIFEST_FILENAME, parseManifest } from "@karume/hub";
import { parseSafetensors, type SafetensorsFile, type SessionDiagnostics } from "@karume/runtime";
import {
  chatImagePrompt,
  type GenerationSoftBlock,
  type GenerationSequence,
  Pipeline,
  type Rgb8Image,
} from "../../models/gemma.ts";
import { imageRows } from "../../models/src/gemma/vision.ts";
import { mirrorDirectory, mirrorPresent, mirrorRoot } from "../../models/tests/helpers/mirror.ts";
import { GPU_AVAILABLE } from "../../models/tests/helpers/gpu.ts";

const MIRROR = mirrorRoot("karume-gemma4-qat");
const CHOICE = { model: "12b", quant: "i4", vision: true } as const;
const VISION_ROLE = "vision";
const GOLDEN_ROOT = goldenDir("gemma4-12b-vision");
const PRODUCT_GOLDEN_ROOT = goldenDir("gemma4-12b-product");
const CORPUS = corpusDir("wan-i2v");
const REFERENCE_FILE = "reference.json";
const HIDDEN = 3840;
const FETCH_COMMAND = "node tools/test-assets/fetch.ts";
/** The gemma4 logits tolerance the margin gate is stated against (`e2e_gemma4_12b_test.ts`). */
const ATOL = 1e-2;
/**
 * The band of the embedder's output against upstream's module, as `max |a − b| / max |b|` over all
 * rows. bf16 weights widened to f32 against the same values in torch, so only the order of the sums
 * differs. Measured on the RTX 3080 Ti under Dawn (2026-11-07): 2.4e-6 / 9.8e-6 / 4.0e-6 on the
 * three images; the band is five times the largest.
 */
const EMBEDDER_RELATIVE_BAND = 5e-5;

const exists = (url: URL): boolean => {
  try {
    return statSync(url).isFile;
  } catch (cause) {
    if (isNotFound(cause)) return false;
    throw cause;
  }
};

const MANIFEST: Manifest | undefined = mirrorPresent(MIRROR)
  ? parseManifest(readTextFileSync(new URL(MANIFEST_FILENAME, MIRROR)))
  : undefined;
const MODEL_PRESENT =
  MANIFEST !== undefined &&
  Object.hasOwn(MANIFEST.models, CHOICE.model) &&
  Object.hasOwn(MANIFEST.models[CHOICE.model].weights, VISION_ROLE);

if (!MODEL_PRESENT) {
  console.warn(
    `[karume] the distribution mirror ${MIRROR.pathname} is absent or its model '${CHOICE.model}' has ` +
      `no '${VISION_ROLE}' weights, so the Gemma 4 12B image cases SKIP. Assemble: cd tools/export-recipes ` +
      "&& uv run python dist.py --pipeline gemma4-qat",
  );
}

const REFERENCE = new URL(REFERENCE_FILE, GOLDEN_ROOT);

/** Narrows the provenance JSON one level (unknown shapes fail with a clear message). */
const objectAt = (value: unknown, where: string): Map<string, unknown> => {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new Error(`${REFERENCE_FILE}: ${where} is not an object`);
  }
  return new Map(Object.entries(value));
};

/** What the record binds, read once; empty while it is absent (the gates below then name it). */
const RECORD = ((): {
  readonly vision: ReadonlyMap<string, Map<string, unknown>>;
  readonly greedy: readonly string[];
  readonly promptText: string;
} => {
  if (!exists(REFERENCE)) return { vision: new Map(), greedy: [], promptText: "" };
  const record: unknown = JSON.parse(readTextFileSync(REFERENCE));
  const reference = objectAt(objectAt(record, "root").get("reference"), "reference");
  const vision = objectAt(reference.get("vision"), "reference.vision");
  const promptText = reference.get("promptText");
  if (typeof promptText !== "string") throw new Error("reference.promptText is not a string");
  return {
    vision: new Map(
      [...vision.entries()].map(([name, entry]) => [
        name,
        objectAt(entry, `reference.vision.${name}`),
      ]),
    ),
    greedy: [...objectAt(reference.get("greedyImages"), "reference.greedyImages").keys()].sort(),
    promptText,
  };
})();
const IMAGES = [...RECORD.vision.keys()].sort();

const visionUrl = (name: string): URL => new URL(`vision.${name}.safetensors`, GOLDEN_ROOT);
const greedyUrl = (name: string): URL => new URL(`greedy.${name}.safetensors`, GOLDEN_ROOT);
const imageUrl = (name: string): URL => new URL(`${name}-832x480.png`, CORPUS);

const missingOf = (files: readonly URL[]): string[] =>
  files
    .filter((file) => !exists(file))
    .map((file) => file.href.slice(TEST_ASSETS_ROOT.href.length));

const gate = (files: readonly URL[]): { readonly skip: boolean; readonly title: string } => {
  const missing = missingOf(files);
  const skip = missing.length > 0 && !testAssetsFetched();
  return {
    skip,
    title: skip
      ? ` (SKIP: ${missing.join(" / ")} absent and outputs/test-assets/ is not a complete fetch of ` +
        `the pin — ${FETCH_COMMAND})`
      : "",
  };
};

const assertPresent = (files: readonly URL[]): void => {
  const missing = missingOf(files);
  assert(
    missing.length === 0,
    `${missing.join(" / ")} absent although outputs/test-assets/ holds a complete fetch of ` +
      `${TEST_ASSETS.repo}@${TEST_ASSETS.revision} — the test-asset dataset at the pin lacks it`,
  );
};

/** The text case of (d): any of the product series' greedy goldens that fits a chunk. */
const TEXT_CASE = new URL("greedy.capital-en.safetensors", PRODUCT_GOLDEN_ROOT);

const ALL_GOLDENS = [
  REFERENCE,
  TEXT_CASE,
  ...IMAGES.flatMap((name) => [visionUrl(name), imageUrl(name)]),
  ...RECORD.greedy.map(greedyUrl),
];
const ASSETS_GATE = gate(ALL_GOLDENS);

const sha256Hex = async (bytes: Uint8Array<ArrayBuffer>): Promise<string> =>
  [...new Uint8Array(await crypto.subtle.digest("SHA-256", bytes))]
    .map((byte) => byte.toString(16).padStart(2, "0"))
    .join("");

const stringAt = (entries: ReadonlyMap<string, unknown>, key: string, where: string): string => {
  const value = entries.get(key);
  if (typeof value !== "string" || value === "") {
    throw new Error(`${REFERENCE_FILE}: ${where}.${key} is not a non-empty string`);
  }
  return value;
};

const readImage = async (name: string): Promise<Rgb8Image> => {
  assertPresent([imageUrl(name)]);
  return await decodePng(await readFile(imageUrl(name)), `${name}.png`);
};

const parsed = async (url: URL): Promise<SafetensorsFile> => {
  assertPresent([url]);
  const bytes = await readFile(url);
  return parseSafetensors(
    bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength),
  );
};

const i32Of = (file: SafetensorsFile, name: string): number[] => {
  const view = file.tensors.get(name);
  assert(view !== undefined, `golden has no '${name}'`);
  assertEquals(view.dtype, "I32", `golden '${name}' dtype`);
  return [...new Int32Array(file.buffer, view.byteOffset, view.byteLength / 4)];
};

const f32Of = (file: SafetensorsFile, name: string): Float32Array<ArrayBuffer> => {
  const view = file.tensors.get(name);
  assert(view !== undefined, `golden has no '${name}'`);
  assertEquals(view.dtype, "F32", `golden '${name}' dtype`);
  return new Float32Array(file.buffer, view.byteOffset, view.byteLength / 4);
};

/**
 * The greedy chain of `steps` tokens, stop tokens included: the soft blocks go with the first turn,
 * and a stop token is the pending frontier of the next, which continues with an empty prompt. The
 * goldens are the torch greedy continuation, which does not stop.
 */
const greedyChain = async (
  sequence: GenerationSequence,
  prompt: readonly number[],
  softBlocks: readonly GenerationSoftBlock[],
  steps: number,
): Promise<number[]> => {
  const ids: number[] = [];
  let request = { prompt: [...prompt], softBlocks: [...softBlocks] };
  while (ids.length < steps) {
    const stream = sequence.generate({ ...request, maxNewTokens: steps - ids.length });
    for await (const event of stream) {
      if (event.kind === "token") ids.push(event.id);
    }
    const stop = await stream.done;
    if (stop.reason === "max-tokens") break;
    assert(stop.reason === "eos", `unexpected stop '${stop.reason}' after ${ids.length} tokens`);
    ids.push(stop.token);
    request = { prompt: [], softBlocks: [] };
  }
  return ids;
};

for (const name of IMAGES) {
  test.skipIf(ASSETS_GATE.skip)(
    `Gemma 4 12B preprocessing: ${name} is upstream's bytes (no GPU)${ASSETS_GATE.title}`,
    async () => {
      const entry = RECORD.vision.get(name);
      assert(entry !== undefined);
      const rows = imageRows(await readImage(name));
      assertEquals(rows.softTokens, entry.get("softTokens"), "soft tokens");
      assertEquals(
        await sha256Hex(new Uint8Array(rows.pixels.buffer)),
        stringAt(entry, "pixelsSha256", name),
        "the pixels",
      );
      assertEquals(
        await sha256Hex(new Uint8Array(rows.positions.buffer)),
        stringAt(entry, "positionsSha256", name),
        "the positions",
      );
    },
  );
}

describe.skipIf(!MODEL_PRESENT || ASSETS_GATE.skip || !GPU_AVAILABLE)(
  `Gemma 4 12B image input on the reference path (real GPU)${ASSETS_GATE.title}`,
  () => {
    let pipeline: Pipeline | undefined;
    beforeAll(async () => {
      pipeline = await Pipeline.fromPretrained(mirrorDirectory(MIRROR), CHOICE);
    });
    afterAll(async () => {
      await pipeline?.dispose();
    });

    it("carries the declaration of the distribution", () => {
      assert(pipeline !== undefined, "the pipeline was not built in beforeAll");
      assertEquals(pipeline.vision, {
        softTokenBudget: 280,
        imageTokenId: 258880,
        beginImageTokenId: 255999,
        endImageTokenId: 258882,
        hiddenSize: HIDDEN,
      });
      const estimate = pipeline.estimateSessionMemory();
      console.log(
        `[e2e] gemma4 12b i4 + vision: estimated peak ${(estimate.peakAccountedBytes / 2 ** 30).toFixed(2)} GiB` +
          ` (weights ${(estimate.resident.weights.totalBytes / 2 ** 30).toFixed(2)} GiB)`,
      );
    });

    for (const name of IMAGES) {
      it(`the embedder: ${name} within the band of upstream's module`, async () => {
        assert(pipeline !== undefined, "the pipeline was not built in beforeAll");
        const golden = await parsed(visionUrl(name));
        const expected = f32Of(golden, "embeds");
        const { softTokens, rows } = await pipeline.embedImage(await readImage(name));
        assertEquals(rows.length, expected.length, "the rows");
        assertEquals(rows.length, softTokens * HIDDEN);
        let worst = 0;
        let scale = 0;
        for (let index = 0; index < rows.length; index += 1) {
          worst = Math.max(worst, Math.abs(rows[index] - expected[index]));
          scale = Math.max(scale, Math.abs(expected[index]));
        }
        const relative = worst / scale;
        console.log(
          `[e2e] gemma4 12b embedder ${name}: max |Δ| / max |x| = ${relative.toExponential(3)}`,
        );
        assert(relative <= EMBEDDER_RELATIVE_BAND, `${name}: ${relative} over the band`);
      });
    }

    for (const name of RECORD.greedy) {
      it(`greedy with an image: ${name} equals the golden token for token`, async () => {
        assert(pipeline !== undefined, "the pipeline was not built in beforeAll");
        const golden = await parsed(greedyUrl(name));
        const prompt = i32Of(golden, "prompt");
        const expected = i32Of(golden, "expected");
        const margin = f32Of(golden, "margin");
        const [start, stop] = i32Of(golden, "block");
        const minMargin = Math.min(...margin);
        assert(minMargin > 2 * ATOL, `${name}: the golden's margin ${minMargin} could flip`);

        const image = await readImage(name);
        const vision = pipeline.vision;
        assert(vision !== undefined);
        const rendered = chatImagePrompt(
          pipeline.tokenizer,
          [
            {
              role: "user",
              content: [
                { type: "image", image },
                { type: "text", text: RECORD.promptText },
              ],
            },
          ],
          vision,
        );
        assertEquals(rendered.prompt, prompt, `${name}: the rendered prompt`);
        assertEquals(rendered.images.length, 1);
        assertEquals(rendered.images[0].start, start, `${name}: the block's start`);
        assertEquals(rendered.images[0].start + rendered.images[0].softTokens, stop, "its end");

        const { rows } = await pipeline.embedImage(image);
        const sequence = await pipeline.sequence();
        try {
          const started = performance.now();
          const ids = await greedyChain(
            sequence,
            rendered.prompt,
            [{ start, rows }],
            expected.length,
          );
          assertEquals(ids, expected, `${name}: greedy tokens (against the torch golden)`);
          console.log(
            `[e2e] gemma4 12b ${name}: T=${prompt.length} + K=${expected.length} / min margin ` +
              `${minMargin.toExponential(3)} / ${(performance.now() - started).toFixed(0)} ms`,
          );
        } finally {
          await sequence.dispose();
        }
      });
    }

    it("a text prompt on the same pipeline equals the product series' greedy golden", async () => {
      assert(pipeline !== undefined, "the pipeline was not built in beforeAll");
      const golden = await parsed(TEXT_CASE);
      const sequence = await pipeline.sequence();
      try {
        const expected = i32Of(golden, "expected");
        const ids = await greedyChain(sequence, i32Of(golden, "prompt"), [], expected.length);
        assertEquals(ids, expected, "greedy tokens over the graph with soft inputs");
      } finally {
        await sequence.dispose();
      }
    });
  },
);

describe.skipIf(!MODEL_PRESENT || ASSETS_GATE.skip || !GPU_AVAILABLE)(
  `Gemma 4 12B image census (real GPU)${ASSETS_GATE.title}`,
  () => {
    it("the prefill with the image runs the ':bidir' kernel, and the decode after it does not", async () => {
      assert(IMAGES.length > 0, "the record binds no image");
      const seen: { readonly kind: string; readonly keys: readonly string[] }[] = [];
      const pipeline = await Pipeline.fromPretrained(mirrorDirectory(MIRROR), {
        ...CHOICE,
        onRunDiagnostics: (diagnostics: SessionDiagnostics, phase) => {
          seen.push({
            kind: phase.kind,
            keys: (diagnostics.lastRunPipelines ?? []).map((row) => row.key),
          });
        },
      });
      try {
        const stream = pipeline.chat(
          [{ role: "user", content: [{ type: "image", image: await readImage(IMAGES[0]) }] }],
          { maxNewTokens: 3, sampler: { temperature: 0 } },
        );
        await stream.text();
      } finally {
        await pipeline.dispose();
      }
      const bidirectional = (keys: readonly string[]): boolean =>
        keys.some((key) => key.startsWith("attention_state_qk") && key.endsWith(":bidir"));
      const prefills = seen.filter((run) => run.kind === "prefill");
      const decodes = seen.filter((run) => run.kind === "decode");
      assert(prefills.length >= 1 && decodes.length >= 1, `the runs: ${JSON.stringify(seen)}`);
      assertEquals(prefills.filter((run) => bidirectional(run.keys)).length, 1, "one chunk has it");
      assert(!decodes.some((run) => bidirectional(run.keys)), "a decode ran the block's kernel");
    }, 600_000);
  },
);
