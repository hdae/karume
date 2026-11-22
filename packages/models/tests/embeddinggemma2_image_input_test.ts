/**
 * Image input of the EmbeddingGemma family over EmbeddingGemma 2, on the host only — no GPU, no
 * real weights (the real-GPU run over the local distribution, and the token ids of the five fixture
 * inputs against upstream's, are `packages/e2e/tests/e2e_embeddinggemma2_image_pipeline_test.ts`).
 *
 * ① `pipelineConfig.vision`: the distribution's shape (the Gemma 4 family's reader) and its tie to
 *    the image placeholder.
 * ② The host preprocessing (`imagePatchRows`) of the fixture's five images: the resize rule's sizes
 *    (the 1190 × 17 rounding edge and the 840-patch side included), the block order of the rows,
 *    and the bytes of the pixels and of both position arrays, equal to the vision series' io inputs
 *    (their sha256 pinned here; with `KARUME_SERIES=1` and the series present, read from the io
 *    files too).
 * ③ The sequence of an input holding images (`planEmbedRequest` / `tokenizeEmbedInput`): the image
 *    blocks, the text runs, the prompt's prefix before an image, the limit counting the soft tokens,
 *    and every refusal — an image part without the vision tower names how to build one with it; and
 *    the ids of the fixture's id cases (spaces, newlines and tabs at a run's boundary, an empty text
 *    between images, Japanese, a prompt before an image) equal upstream's exactly.
 * ④ The tower graph's contract (`embeddingGemmaVisionContract`).
 * ⑤ The run over stand-in Sessions (`Embedder`): one tower run per image, its output placed at the
 *    image's soft rows with the mask set there and nowhere else, a text call as without the tower,
 *    the pixels read at issue, two queued calls holding images, cancellation before and between the
 *    runs, dispose of both Sessions; and their opening (`openEmbedder`): both Sessions take the
 *    row-block budget, and a failure disposes what was opened and releases.
 * ⑥ `Pipeline.fromAssets` with `vision`: the admission's refusals, before any GPU.
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
import { resizeRgb8TorchBicubic } from "../src/image/torch-resample.ts";
import { imageTargetSize } from "../src/gemma/vision.ts";
import { parseEmbeddingGemmaPipelineConfig } from "../src/embeddinggemma/config.ts";
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
import {
  type EmbeddingGemmaVisionContract,
  embeddingGemmaVisionContract,
  imagePatchRows,
} from "../src/embeddinggemma/vision.ts";
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

/** The image ids of EmbeddingGemma 2 (`config.json` at the pin). */
const BEGIN = 255999;
const SOFT = 258880;
const END = 258882;

/** `pipelineConfig.vision` as the dist recipe writes it. */
const VISION_JSON = {
  softTokenBudget: 280,
  imageTokenId: SOFT,
  beginImageTokenId: BEGIN,
  endImageTokenId: END,
  hiddenSize: 512,
};

/** The distribution's `pipelineConfig` (two prompts are enough here). */
const CONFIG_JSON = {
  prompts: { document: "title: none | text: ", query: "task: search result | query: " },
  dimensions: [768, 512, 256, 128],
  maxTokens: 8192,
  placeholderTokens: { image: SOFT, audio: 258881, video: 258884 },
  vision: VISION_JSON,
};

const CONFIG = parseEmbeddingGemmaPipelineConfig(CONFIG_JSON);
/** The rules of a pipeline with the vision tower, and of one without it. */
const IMAGE_RULES = embedTextRules(TOKENIZER, CONFIG, { images: true });
const TEXT_RULES = embedTextRules(TOKENIZER, CONFIG);

/** The whole-input fixture (`tools/export-recipes/embeddinggemma2/multimodal_reference.py`). */
type MultimodalFixture = {
  readonly softTokenBudget: number;
  readonly images: readonly {
    readonly name: string;
    readonly width: number;
    readonly height: number;
    readonly resized: { readonly width: number; readonly height: number };
    readonly softTokens: number;
  }[];
  /** The inputs whose ids alone upstream's processor pinned (`ID_CASES` of the script). */
  readonly idCases: readonly {
    readonly name: string;
    readonly parts: readonly (
      | { readonly type: "text"; readonly text: string }
      | { readonly type: "image"; readonly image: string }
    )[];
    readonly prompt?: string;
    readonly tokens: number;
    readonly imageBlocks: readonly { readonly start: number; readonly softTokens: number }[];
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

/** The raw bytes of one tensor of a safetensors file. */
const tensorBytes = (file: SafetensorsFile, key: string): Uint8Array<ArrayBuffer> => {
  const view = file.tensors.get(key);
  if (view === undefined) throw new Error(`the safetensors file has no tensor '${key}'`);
  return new Uint8Array(file.buffer, view.byteOffset, view.byteLength);
};

/** One fixture image as RGB8 (what upstream's processor read). */
const fixtureImage = (name: string): Rgb8Image => {
  const entry = MULTIMODAL.images.find((image) => image.name === name);
  if (entry === undefined) throw new Error(`the fixture has no image '${name}'`);
  return {
    data: tensorBytes(MULTIMODAL_TENSORS, `image.${name}`).slice(),
    width: entry.width,
    height: entry.height,
  };
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

// ---- ① pipelineConfig.vision ------------------------------------------------------------------

describe("pipelineConfig.vision", () => {
  it("reads the distribution's declaration (the Gemma 4 family's shape)", () => {
    assertEquals(CONFIG.vision, VISION_JSON);
  });

  it("is absent from a config that does not declare it", () => {
    const { vision: _, ...textOnly } = CONFIG_JSON;
    assertEquals(parseEmbeddingGemmaPipelineConfig(textOnly).vision, undefined);
  });

  it("refuses a soft token that is not the image placeholder, or no image placeholder at all", () => {
    assertThrows(
      () =>
        parseEmbeddingGemmaPipelineConfig({
          ...CONFIG_JSON,
          vision: { ...VISION_JSON, imageTokenId: 258881 },
        }),
      Error,
      "pipelineConfig.vision.imageTokenId 258881 is not placeholderTokens.image (258880)",
    );
    assertThrows(
      () =>
        parseEmbeddingGemmaPipelineConfig({
          ...CONFIG_JSON,
          placeholderTokens: { audio: 258881 },
        }),
      Error,
      "is not placeholderTokens.image (not declared)",
    );
  });

  it("refuses an unknown key and a budget upstream's processor does not take", () => {
    assertThrows(
      () =>
        parseEmbeddingGemmaPipelineConfig({
          ...CONFIG_JSON,
          vision: { ...VISION_JSON, budget: 280 },
        }),
      Error,
      "pipelineConfig.vision: unknown key 'budget'",
    );
    assertThrows(
      () =>
        parseEmbeddingGemmaPipelineConfig({
          ...CONFIG_JSON,
          vision: { ...VISION_JSON, softTokenBudget: 300 },
        }),
      Error,
      "pipelineConfig.vision.softTokenBudget",
    );
  });
});

// ---- ② the host preprocessing -----------------------------------------------------------------

/**
 * The sha256 of the vision series' io inputs (`outputs/series/embeddinggemma-2-vision/io.<image>`
 * — `input.pixels` f32, `input.position_x` / `input.position_y` i32, little-endian), taken from
 * those files on 2026-11-22 (the bf16 series' io files are byte-identical). The rows the recipe
 * built from upstream's processor in the graph's block order.
 */
const SERIES_DIGESTS: Readonly<
  Record<string, { readonly pixels: string; readonly x: string; readonly y: string }>
> = {
  "gradient-square": {
    pixels: "26c65708aece967d00f4fb2054710d86feb038865207c142db012c1e33a455de",
    x: "f77a1ace96ac229f0e0b05ebb194708db9d7ee4dfeb95db369057867bfed4108",
    y: "f85b2b1c4532d818dbc722eef08031ec22c0b909496b4ead629d8ae503965752",
  },
  "shapes-wide": {
    pixels: "4f29d9d7d1a75bb28cd7fcfaed02ca5d02655983b6257b88ab6f326f6d50fc3a",
    x: "87e2091ac88adf17c9d552395cb553640377d2fb88fa1778934d013cfd94fee7",
    y: "92f8bb47079de19faeb9d4c901ddc76e8fa90f3a50fac3a9e30bd1c4d0521721",
  },
  "noise-tall": {
    pixels: "d306630cc38f8ea53894c72c6d3634ec728166d20cde6161cb9bacafcd1416d8",
    x: "11972b839e46c8eb5a0f476f51a7750e8ad3a3b9c254bd3c333956f4eadf9e39",
    y: "0d39a36b813af20a3d3e7cf98d0d565860f594688a0702b602abb8b8f810e9a3",
  },
  "stripes-edge": {
    pixels: "f07a54ffe2f32a39e9dc24851eaed48021f87470646949c86d1a97f3b484e8ff",
    x: "9aceb1db27159e280d296212cd69f7b4ec19425688241e7ab1cdae3c61e7b8bf",
    y: "b0a1f080c2685eec86846514c475e1eb53d4b5cb028e1b9106f12f85009ddc2b",
  },
  // Taken on 2026-11-22 from the series' io file of the 840 × 3 strip (its columns reach 839).
  "gradient-strip": {
    pixels: "27bf9da21192a2e599641f0641ac6790e7fd560177ce8936b94438ba3f285384",
    x: "d84d8d8a0954f926950b185b0cb74b1833735c5c92f2d3770e221f211e2b9348",
    y: "cc8f9149ecccf17e41208d227fed17389f59d4e11f8865c87005b3f78fab73d3",
  },
};

const BUDGET = 280;

/** The io file of an image in the f32 vision series (outside the repository). */
const seriesIo = (name: string): URL =>
  new URL(
    `../../../outputs/series/embeddinggemma-2-vision/io.${name}.safetensors`,
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

describe("imagePatchRows: the fixture's images as the tower reads them", () => {
  it("covers the five images of the fixture, the rounding edge and the longest side among them (a positive control)", () => {
    assertEquals(
      Object.keys(SERIES_DIGESTS).sort(),
      MULTIMODAL.images.map(({ name }) => name).sort(),
    );
    assertEquals(MULTIMODAL.softTokenBudget, BUDGET);
  });

  for (const entry of MULTIMODAL.images) {
    it(`${entry.name} (${entry.width} × ${entry.height}): upstream's resized size and soft tokens`, () => {
      assertEquals(imageTargetSize(entry.height, entry.width, BUDGET), entry.resized);
      assertEquals(imagePatchRows(fixtureImage(entry.name), BUDGET).softTokens, entry.softTokens);
    });

    it(`${entry.name}: the pixels and both position arrays are the series' io inputs byte for byte`, async () => {
      const rows = imagePatchRows(fixtureImage(entry.name), BUDGET);
      assertEquals(rows.pixels.length, entry.softTokens * 9 * 768);
      assertEquals(rows.positionX.length, entry.softTokens * 9);
      const expected = SERIES_DIGESTS[entry.name];
      assertEquals(
        {
          pixels: await sha256Hex(bytesOf(rows.pixels)),
          x: await sha256Hex(bytesOf(rows.positionX)),
          y: await sha256Hex(bytesOf(rows.positionY)),
        },
        expected,
      );
    });

    const io = seriesIo(entry.name);
    it.skipIf(!SERIES_CHECKS || !exists(io))(
      `${entry.name}: the pinned digests are the series' io file's${SERIES_CHECKS_OPT_IN}`,
      async () => {
        const file = parseFile(await readFile(io));
        const expected = SERIES_DIGESTS[entry.name];
        assertEquals(
          {
            pixels: await sha256Hex(tensorBytes(file, "input.pixels").slice()),
            x: await sha256Hex(tensorBytes(file, "input.position_x").slice()),
            y: await sha256Hex(tensorBytes(file, "input.position_y").slice()),
          },
          expected,
        );
      },
    );
  }

  it("reaches the last row of the tower's position tables on the longest side (840 patches)", () => {
    const rows = imagePatchRows(fixtureImage("gradient-strip"), BUDGET);
    assertEquals(rows.softTokens, 280);
    assertEquals(Math.max(...rows.positionX), 839);
    assertEquals(Math.max(...rows.positionY), 2);
  });

  it("orders the rows by 3 × 3 block, each block's 9 patches in raster order", () => {
    // gradient-square resizes to 768 × 768: 48 × 48 patches, 16 blocks per row of blocks.
    const rows = imagePatchRows(fixtureImage("gradient-square"), BUDGET);
    const position = (row: number): readonly [number, number] => [
      rows.positionX[row],
      rows.positionY[row],
    ];
    const firstBlock = Array.from({ length: 9 }, (_, row) => position(row));
    assertEquals(firstBlock, [
      [0, 0],
      [1, 0],
      [2, 0],
      [0, 1],
      [1, 1],
      [2, 1],
      [0, 2],
      [1, 2],
      [2, 2],
    ]);
    // The second block sits right of the first; block 16 opens the second row of blocks.
    assertEquals(position(9), [3, 0]);
    assertEquals(position(16 * 9), [0, 3]);
    assertEquals(position(16 * 9 + 8), [2, 5]);
  });

  it("holds each patch's 16 × 16 × 3 values, (row, column, channel), as u8 · f32(1 / 255)", () => {
    const image = fixtureImage("shapes-wide");
    const rows = imagePatchRows(image, BUDGET);
    const size = imageTargetSize(image.height, image.width, BUDGET);
    const resized = resizeRgb8TorchBicubic(image, size.width, size.height);
    // Row 13: block 1, (dy, dx) = (1, 1) → patch column 4, row 1.
    assertEquals([rows.positionX[13], rows.positionY[13]], [4, 1]);
    const rescale = Math.fround(1 / 255);
    for (const [y, x, channel] of [
      [0, 0, 0],
      [0, 15, 2],
      [7, 3, 1],
      [15, 15, 2],
    ] as const) {
      const source = ((1 * 16 + y) * size.width + 4 * 16 + x) * 3 + channel;
      assertStrictEquals(
        rows.pixels[13 * 768 + (y * 16 + x) * 3 + channel],
        Math.fround((resized.data[source] ?? 0) * rescale),
      );
    }
  });

  it("refuses an image that is not RGB8 of its size (a ModelInputError)", () => {
    assertThrows(
      () => imagePatchRows({ data: new Uint8Array(5), width: 2, height: 2 }, BUDGET),
      ModelInputError,
      "RGB8 length 5 does not match 3×2×2",
    );
  });
});

// ---- ③ the sequence -------------------------------------------------------------------------

/** A small RGB8 image (every one upscales to the budget: a square → 16 × 16 blocks = 256). */
const smallImage = (width: number, height: number, value = 128): Rgb8Image => ({
  data: new Uint8Array(width * height * 3).fill(value),
  width,
  height,
});
const SQUARE = smallImage(8, 8);
const SQUARE_TOKENS = 256;
const WIDE = fixtureImage("stripes-edge");
const WIDE_TOKENS = 139;

const block = (softTokens: number): number[] => [
  BEGIN,
  ...Array<number>(softTokens).fill(SOFT),
  END,
];

const idsOf = (request: { readonly ids: Int32Array<ArrayBuffer> }): number[] => [...request.ids];

describe("planEmbedRequest with image parts: the sequence", () => {
  it("writes an image alone as <bos>, <|image>, n soft tokens, <image|>, <eos>", () => {
    const request = planEmbedRequest(IMAGE_RULES, [{ type: "image", image: SQUARE }], {});
    assertEquals(idsOf(request), [BOS, ...block(SQUARE_TOKENS), EOS]);
    assertEquals(
      request.images.map(({ start, softTokens }) => ({ start, softTokens })),
      [{ start: 2, softTokens: SQUARE_TOKENS }],
    );
    assertStrictEquals(request.images[0]?.image.data, SQUARE.data);
  });

  it("tokenizes each run of adjacent text as one string, with nothing between it and an image", () => {
    const { text } = parityCase("english-long");
    const half = Math.floor(text.length / 2);
    const tail = parityCase("english").text;
    const parts: readonly EmbedContentPart[] = [
      { type: "text", text: text.slice(0, half) },
      { type: "text", text: text.slice(half) },
      { type: "image", image: WIDE },
      { type: "text", text: tail },
    ];
    const run = TOKENIZER.encode(text);
    const request = planEmbedRequest(IMAGE_RULES, parts, {});
    assertEquals(idsOf(request), [
      BOS,
      ...run,
      ...block(WIDE_TOKENS),
      ...TOKENIZER.encode(tail),
      EOS,
    ]);
    assertEquals(request.images[0]?.start, 1 + run.length + 1);
  });

  it("writes two images back to back, each block where its soft rows start", () => {
    const request = planEmbedRequest(
      IMAGE_RULES,
      [
        { type: "image", image: WIDE },
        { type: "image", image: SQUARE },
      ],
      {},
    );
    assertEquals(idsOf(request), [BOS, ...block(WIDE_TOKENS), ...block(SQUARE_TOKENS), EOS]);
    assertEquals(
      request.images.map(({ start }) => start),
      [2, 2 + WIDE_TOKENS + 2],
    );
  });

  it("glues the prompt's prefix to the first text run (one string), as upstream does", () => {
    const { text, idsWithSpecials } = parityCase("prompt-query");
    const body = text.slice("task: search result | query: ".length);
    const request = planEmbedRequest(
      IMAGE_RULES,
      [
        { type: "text", text: body },
        { type: "image", image: SQUARE },
      ],
      { prompt: "query" },
    );
    assertEquals(idsOf(request), [...idsWithSpecials.slice(0, -1), ...block(SQUARE_TOKENS), EOS]);
  });

  it("gives the prefix a run of its own before an image that comes first (its trailing space a token)", () => {
    const prefix = TOKENIZER.encode("task: search result | query: ");
    assertEquals(prefix.at(-1), 236743);
    const request = planEmbedRequest(IMAGE_RULES, [{ type: "image", image: SQUARE }], {
      prompt: "query",
    });
    assertEquals(idsOf(request), [BOS, ...prefix, ...block(SQUARE_TOKENS), EOS]);
  });

  it("keeps a text-only input's ids as they were, with the tower or without it", () => {
    for (const name of ["english", "prompt-query", "japanese-long"]) {
      const { text, idsWithSpecials } = parityCase(name);
      for (const rules of [IMAGE_RULES, TEXT_RULES]) {
        const request = planEmbedRequest(rules, text, {});
        assertEquals(idsOf(request), idsWithSpecials, name);
        assertEquals(request.images, []);
      }
    }
  });

  it("gives tokenize the ids embed runs, the soft rows as the placeholder id", () => {
    const parts: readonly EmbedContentPart[] = [
      { type: "text", text: "a" },
      { type: "image", image: WIDE },
    ];
    assertEquals(
      [...tokenizeEmbedInput(IMAGE_RULES, parts, { prompt: "document" })],
      idsOf(planEmbedRequest(IMAGE_RULES, parts, { prompt: "document" })),
    );
  });

  it("still refuses a placeholder spelt in a text run, at its place in the whole sequence", () => {
    assertRefused(
      () =>
        planEmbedRequest(
          IMAGE_RULES,
          [
            { type: "image", image: WIDE },
            { type: "text", text: "look at <|image|> here" },
          ],
          {},
        ),
      'the text holds the image placeholder token "<|image|>" (id 258880) at token',
      "remove that spelling",
    );
    const at = 1 + WIDE_TOKENS + 2 + TOKENIZER.encode("look at ").length;
    assertRefused(
      () =>
        planEmbedRequest(
          IMAGE_RULES,
          [
            { type: "image", image: WIDE },
            { type: "text", text: "look at <|image|> here" },
          ],
          {},
        ),
      `at token ${at}`,
    );
  });

  it("counts every image's soft tokens and markers toward the limit", () => {
    const parts: readonly EmbedContentPart[] = [{ type: "image", image: SQUARE }];
    const length = SQUARE_TOKENS + 4;
    const at = embedTextRules(
      TOKENIZER,
      parseEmbeddingGemmaPipelineConfig({ ...CONFIG_JSON, maxTokens: length }),
      { images: true },
    );
    assertEquals(planEmbedRequest(at, parts, {}).ids.length, length);
    const under = embedTextRules(
      TOKENIZER,
      parseEmbeddingGemmaPipelineConfig({ ...CONFIG_JSON, maxTokens: length - 1 }),
      { images: true },
    );
    assertRefused(
      () => planEmbedRequest(under, parts, {}),
      `the input is ${length} tokens (<bos>, <eos> and the soft tokens of the image with their two` +
        ` markers each included), over this model's limit of ${length - 1}`,
      "pass fewer images",
    );
  });
});

/** The ids upstream ran for a fixture case (`<case>.input_ids`, I32). */
const fixtureIds = (name: string): number[] => {
  const bytes = tensorBytes(MULTIMODAL_TENSORS, `${name}.input_ids`);
  return [...new Int32Array(bytes.slice().buffer)];
};

describe("planEmbedRequest with image parts: the ids of upstream's processor at run boundaries", () => {
  it("covers the fixture's seven id cases (a positive control)", () => {
    assertEquals(MULTIMODAL.idCases.length, 7);
  });

  for (const entry of MULTIMODAL.idCases) {
    it(`${entry.name}: the ids and each image's soft rows are upstream's exactly`, () => {
      const parts: readonly EmbedContentPart[] = entry.parts.map((part) =>
        part.type === "text" ? part : { type: "image", image: fixtureImage(part.image) },
      );
      const options = entry.prompt === undefined ? {} : { prompt: entry.prompt };
      const request = planEmbedRequest(IMAGE_RULES, parts, options);
      const expected = fixtureIds(entry.name);
      assertEquals(idsOf(request), expected);
      assertEquals(expected.length, entry.tokens);
      assertEquals([...tokenizeEmbedInput(IMAGE_RULES, parts, options)], expected);
      assertEquals(
        request.images.map(({ start, softTokens }) => ({ start, softTokens })),
        entry.imageBlocks.map(({ start, softTokens }) => ({ start, softTokens })),
      );
    });
  }
});

describe("planEmbedRequest with image parts: the refusals", () => {
  it("refuses an image part without the vision tower, saying how to build a pipeline that takes it", () => {
    assertRefused(
      () => planEmbedRequest(TEXT_RULES, [{ type: "image", image: SQUARE }], {}),
      'content part 0 has type "image" — this model accepts "text" parts',
      "image parts need the vision tower — build the pipeline with { vision: true }",
    );
    assertRefused(
      () => tokenizeEmbedInput(TEXT_RULES, [{ type: "image", image: SQUARE }], {}),
      "build the pipeline with { vision: true }",
    );
  });

  it("says the distribution has no tower when its pipelineConfig declares none", () => {
    const { vision: _, ...textOnly } = CONFIG_JSON;
    const rules = embedTextRules(TOKENIZER, parseEmbeddingGemmaPipelineConfig(textOnly));
    assertRefused(
      () => planEmbedRequest(rules, [{ type: "image", image: SQUARE }], {}),
      'this model accepts "text" parts (this model\'s distribution ships no vision tower)',
    );
  });

  it("names both accepted types with the tower, and refuses other types", () => {
    assertRefused(
      () =>
        Reflect.apply(planEmbedRequest, undefined, [
          IMAGE_RULES,
          [{ type: "audio", audio: [] }],
          {},
        ]),
      'content part 0 has type "audio" — this model accepts "text" / "image" parts',
    );
  });

  it("refuses an image that is not decoded RGB8 pixels of its size", () => {
    for (const [image, fragment] of [
      [undefined, "is an image part whose image is not RGB8 pixels"],
      [{ data: [1, 2, 3], width: 1, height: 1 }, "is an image part whose image is not RGB8 pixels"],
      [{ data: new Uint8Array(3), width: "1", height: 1 }, "whose width / height are not numbers"],
      [{ data: new Uint8Array(4), width: 1, height: 1 }, "RGB8 length 4 does not match 3×1×1"],
      [
        { data: new Uint8Array(0), width: 0, height: 0 },
        "Image size 0×0 is not a positive integer",
      ],
    ] as const) {
      assertRefused(
        () =>
          Reflect.apply(planEmbedRequest, undefined, [IMAGE_RULES, [{ type: "image", image }], {}]),
        fragment,
      );
    }
  });
});

describe("embedTextRules with images", () => {
  it("refuses image rules without pipelineConfig.vision", () => {
    const { vision: _, ...textOnly } = CONFIG_JSON;
    assertThrows(
      () =>
        embedTextRules(TOKENIZER, parseEmbeddingGemmaPipelineConfig(textOnly), { images: true }),
      Error,
      "image input was asked for, but pipelineConfig declares no 'vision'",
    );
  });

  it("refuses an image marker that is not an added token of the asset", () => {
    const config = parseEmbeddingGemmaPipelineConfig({
      ...CONFIG_JSON,
      vision: { ...VISION_JSON, endImageTokenId: 818 },
    });
    const error = assertThrows(() => embedTextRules(TOKENIZER, config, { images: true }), Error);
    assert(!(error instanceof ModelInputError), "an asset mismatch is not an input error");
    assert(
      error.message.includes(
        "pipelineConfig.vision.endImageTokenId (id 818) is not an added token",
      ),
      error.message,
    );
  });
});

// ---- ④ the tower graph's contract --------------------------------------------------------------

type Graph = GraphOwner["graph"];

/** The declaration of the recipe's tower graph (the mirror's `vision`, minus the layers). */
const towerGraph = (): Graph => ({
  format: "karume-ir",
  version: 2,
  requires: { ops: [] },
  symbols: ["n"],
  inputs: [
    { name: "pixels", dtype: "f32", shape: [1, "9n", 768] },
    { name: "position_x", dtype: "i32", shape: [1, "9n"] },
    { name: "position_y", dtype: "i32", shape: [1, "9n"] },
  ],
  outputs: ["linear_113"],
  initializers: {},
  values: {
    pos_x: { dtype: "f32", shape: [10240, 768] },
    pos_y: { dtype: "f32", shape: [10240, 768] },
    rope_cos: { dtype: "f32", shape: [840, 32] },
    rope_sin: { dtype: "f32", shape: [840, 32] },
    linear_113: { dtype: "f32", shape: [1, "n", 512] },
  },
  states: {},
  nodes: [
    ["pos_x", "position_x"],
    ["pos_y", "position_y"],
    ["rope_cos", "position_x"],
    ["rope_cos", "position_y"],
    ["rope_sin", "position_x"],
    ["rope_sin", "position_y"],
  ].map(([table, index], at) => ({
    op: "embedding",
    ins: [table, index],
    outs: [`gathered_${at}`],
    attrs: {},
    states: {},
  })),
});

const VISION = CONFIG.vision ?? VISION_JSON;
const towerContractOf = (edit: (graph: Graph) => Graph, vision = VISION) =>
  embeddingGemmaVisionContract({ graph: edit(towerGraph()) }, vision);

describe("embeddingGemmaVisionContract", () => {
  it("accepts the recipe's declaration and reads the output, its width and the positions covered", () => {
    assertEquals(
      towerContractOf((graph) => graph),
      { output: "linear_113", width: 512, positions: 840 },
    );
  });

  it("refuses tables that cover fewer positions than the budget's longest side (3 · budget)", () => {
    const shorter = (graph: Graph): Graph => ({
      ...graph,
      values: { ...graph.values, rope_sin: { dtype: "f32", shape: [839, 32] } },
    });
    assertThrows(
      () => towerContractOf(shorter),
      Error,
      "pipelineConfig.vision.softTokenBudget 280 lets an image side reach 840 patches, but the vision graph's tables cover 839 positions",
    );
    assertEquals(towerContractOf(shorter, { ...VISION, softTokenBudget: 140 }).positions, 839);
  });

  it("refuses an output of another width than hiddenSize", () => {
    assertThrows(
      () => towerContractOf((graph) => graph, { ...VISION, hiddenSize: 768 }),
      Error,
      "the vision graph's output 'linear_113' is f32 [1, n, 512], expected f32 [1, n, 768]",
    );
  });

  it("refuses inputs of another order, dtype or row axis", () => {
    const swapped = (graph: Graph): Graph => ({
      ...graph,
      inputs: [graph.inputs[0], graph.inputs[2], graph.inputs[1]],
    });
    assertThrows(() => towerContractOf(swapped), Error, "the vision graph's inputs are");
    const raster = (graph: Graph): Graph => ({
      ...graph,
      inputs: graph.inputs.map((input) => ({
        ...input,
        shape: input.shape.map((dim) => (dim === "9n" ? "n" : dim)),
      })),
    });
    assertThrows(() => towerContractOf(raster), Error, "not pixels f32 [1, 9n, 768]");
  });

  it("refuses a graph that gathers nothing by both positions, and one with two symbols", () => {
    assertThrows(
      () =>
        towerContractOf((graph) => ({
          ...graph,
          nodes: graph.nodes.filter((node) => node.ins[1] === "position_x"),
        })),
      Error,
      "gathers no table by both 'position_x' and 'position_y'",
    );
    assertThrows(
      () => towerContractOf((graph) => ({ ...graph, symbols: ["n", "m"] })),
      Error,
      "the vision graph's symbolic dimensions are [n, m]",
    );
  });
});

// ---- ⑤ the run over stand-in Sessions ----------------------------------------------------------

/** A stand-in Session that records its runs and answers with `answer(inputs)`. */
type Stand = {
  readonly session: EmbeddingSession;
  readonly runs: RunInputs[];
  readonly events: string[];
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
    events,
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
const TOWER_CONTRACT: EmbeddingGemmaVisionContract = {
  output: "soft",
  width: WIDTH,
  positions: 840,
};

/**
 * The tower's answer: soft token `j` of the `run`-th tower run is `[100·run + j, …]` + the channel /
 * 10, so each soft row in the text run's `soft_embeds` says which image and which token it is.
 */
const towerAnswer = (inputs: RunInputs, run: number): RunOutputs => {
  const pixels = hostInput(inputs, "pixels");
  const softTokens = (pixels.shape[1] ?? 0) / 9;
  const data = new Float32Array(softTokens * WIDTH);
  for (let token = 0; token < softTokens; token += 1) {
    for (let channel = 0; channel < WIDTH; channel += 1) {
      data[token * WIDTH + channel] = 1000 * (run + 1) + token + channel / 10;
    }
  }
  return { soft: { dtype: "f32", shape: [1, softTokens, WIDTH], data } };
};

const UNIT = [0.6, 0, 0.8, 0];
const textAnswer = (): RunOutputs => ({
  out: { dtype: "f32", shape: [1, 4], data: Float32Array.from(UNIT) },
});

const SMALL = parseEmbeddingGemmaPipelineConfig({
  ...CONFIG_JSON,
  dimensions: [4, 2],
  vision: { ...VISION_JSON, hiddenSize: WIDTH },
});
/** The rules of the stand-in embedder with the tower (the same ids as {@link IMAGE_RULES}). */
const SMALL_RULES = embedTextRules(TOKENIZER, SMALL, { images: true });

/** An embedder over stand-ins, with the tower or without it. */
const rig = (options: { readonly tower: boolean; readonly gate?: () => Promise<void> }) => {
  const events: string[] = [];
  const text = standIn("model", events, textAnswer);
  const tower = standIn("vision", events, towerAnswer, options.gate);
  const embedder = new Embedder({
    session: text.session,
    contract: TEXT_CONTRACT,
    rules: embedTextRules(TOKENIZER, SMALL, { images: options.tower }),
    ...(options.tower ? { tower: { session: tower.session, contract: TOWER_CONTRACT } } : {}),
    deviceLimits: { maxStorageBufferBindingSize: 2147483644, maxBufferSize: 1099511627776 },
    release: () => {
      events.push("release");
    },
  });
  return { embedder, text, tower, events };
};

/** The soft rows a text run received: `[row, values]` of every row whose mask is set. */
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

describe("Embedder with the vision tower: the runs of a call holding images", () => {
  it("runs the tower on the image's rows, then the text graph with its soft tokens at the image's soft rows", async () => {
    const { embedder, text, tower } = rig({ tower: true });
    const parts: readonly EmbedContentPart[] = [
      { type: "text", text: "a" },
      { type: "image", image: WIDE },
    ];
    const planned = planEmbedRequest(SMALL_RULES, parts, {});
    const vector = await embedder.embed(parts);

    assertEquals([...vector], [...Float32Array.from(UNIT)]);
    assertEquals(tower.runs.length, 1);
    const rows = imagePatchRows(WIDE, BUDGET);
    const pixels = hostInput(tower.runs[0], "pixels");
    assertEquals([pixels.dtype, pixels.shape], ["f32", [1, WIDE_TOKENS * 9, 768]]);
    assertEquals(bytesOf(pixels.data), bytesOf(rows.pixels));
    const x = hostInput(tower.runs[0], "position_x");
    const y = hostInput(tower.runs[0], "position_y");
    assertEquals(
      [x.dtype, x.shape, y.dtype, y.shape],
      ["i32", [1, WIDE_TOKENS * 9], "i32", [1, WIDE_TOKENS * 9]],
    );
    assertEquals([...x.data], [...rows.positionX]);
    assertEquals([...y.data], [...rows.positionY]);

    assertEquals(text.runs.length, 1);
    const ids = hostInput(text.runs[0], "input_ids");
    assertEquals([...ids.data], idsOf(planned));
    const start = planned.images[0]?.start ?? -1;
    const { masked, rows: soft } = softRowsOf(text.runs[0]);
    assertEquals(masked, range(start, WIDE_TOKENS));
    assertEquals([...soft.keys()], range(start, WIDE_TOKENS));
    assertEquals(soft.get(start), [1000, 1000.1, 1000.2].map(Math.fround));
    assertEquals(soft.get(start + WIDE_TOKENS - 1), [1138, 1138.1, 1138.2].map(Math.fround));
    // The placeholder id stays under every soft row.
    assert(range(start, WIDE_TOKENS).every((row) => ids.data[row] === SOFT));
  });

  it("runs the tower once per image, in order, each output at its own block", async () => {
    const { embedder, text, tower, events } = rig({ tower: true });
    const parts: readonly EmbedContentPart[] = [
      { type: "image", image: WIDE },
      { type: "image", image: SQUARE },
    ];
    await embedder.embed(parts);
    assertEquals(events, ["vision.run", "vision.run", "model.run"]);
    assertEquals(
      tower.runs.map((inputs) => hostInput(inputs, "pixels").shape[1]),
      [WIDE_TOKENS * 9, SQUARE_TOKENS * 9],
    );
    const second = 2 + WIDE_TOKENS + 2;
    const { masked, rows } = softRowsOf(text.runs[0]);
    assertEquals(masked, [...range(2, WIDE_TOKENS), ...range(second, SQUARE_TOKENS)]);
    assertEquals(rows.get(2)?.[0], 1000);
    assertEquals(rows.get(second)?.[0], 2000);
    assertEquals(rows.get(second + SQUARE_TOKENS - 1)?.[0], 2255);
  });

  it("runs a text call on the text graph alone, with the inputs of a pipeline without the tower", async () => {
    const withTower = rig({ tower: true });
    const without = rig({ tower: false });
    const { text } = parityCase("english");
    await withTower.embedder.embed(text, { prompt: "query" });
    await without.embedder.embed(text, { prompt: "query" });
    assertEquals(withTower.tower.runs.length, 0);
    const [left, right] = [withTower.text.runs[0], without.text.runs[0]];
    assertEquals(Object.keys(left).sort(), Object.keys(right).sort());
    for (const name of Object.keys(left)) {
      const [a, b] = [hostInput(left, name), hostInput(right, name)];
      assertEquals([a.dtype, a.shape], [b.dtype, b.shape], name);
      assertEquals(bytesOf(a.data), bytesOf(b.data), name);
    }
    assertEquals(softRowsOf(left), { masked: [], rows: new Map() });
  });

  it("reads the image's pixels at issue (a later write into them does not reach the call)", async () => {
    let open: () => void = () => {};
    const held = new Promise<void>((resolve) => {
      open = resolve;
    });
    const { embedder, tower } = rig({ tower: true, gate: () => held });
    const image = smallImage(8, 8, 10);
    const expected = imagePatchRows(image, BUDGET).pixels;
    const call = embedder.embed([{ type: "image", image }]);
    image.data.fill(250);
    open();
    await call;
    assertEquals(bytesOf(hostInput(tower.runs[0], "pixels").data), bytesOf(expected));
  });

  it("stops between the tower and the text run when the signal is aborted meanwhile", async () => {
    const controller = new AbortController();
    const reason = new Error("stop after the image");
    const { embedder, text, tower } = rig({
      tower: true,
      gate: async () => {
        controller.abort(reason);
      },
    });
    const caught = await embedder
      .embed([{ type: "image", image: SQUARE }], { signal: controller.signal })
      .catch((error: unknown) => error);
    assertStrictEquals(caught, reason);
    assertEquals([tower.runs.length, text.runs.length], [1, 0]);
  });

  it("runs two queued calls that both hold images one after the other, each with its own image", async () => {
    let open: () => void = () => {};
    const held = new Promise<void>((resolve) => {
      open = resolve;
    });
    let towerRuns = 0;
    const { embedder, text, tower, events } = rig({
      tower: true,
      gate: async () => {
        towerRuns += 1;
        if (towerRuns === 1) await held;
      },
    });
    const first = embedder.embed([{ type: "image", image: WIDE }]);
    const second = embedder.embed([
      { type: "text", text: "a" },
      { type: "image", image: SQUARE },
    ]);
    // The second call waits for its turn while the first one's tower run is held.
    await new Promise((resolve) => setTimeout(resolve, 0));
    assertEquals(events, ["vision.run"]);
    open();
    await Promise.all([first, second]);

    assertEquals(events, ["vision.run", "model.run", "vision.run", "model.run"]);
    assertEquals(
      tower.runs.map((inputs) => hostInput(inputs, "pixels").shape[1]),
      [WIDE_TOKENS * 9, SQUARE_TOKENS * 9],
    );
    assertEquals(
      bytesOf(hostInput(tower.runs[1], "pixels").data),
      bytesOf(imagePatchRows(SQUARE, BUDGET).pixels),
    );
    const start =
      planEmbedRequest(
        SMALL_RULES,
        [
          { type: "text", text: "a" },
          { type: "image", image: SQUARE },
        ],
        {},
      ).images[0]?.start ?? -1;
    const firstRows = softRowsOf(text.runs[0]);
    const secondRows = softRowsOf(text.runs[1]);
    assertEquals(firstRows.masked, range(2, WIDE_TOKENS));
    assertEquals(secondRows.masked, range(start, SQUARE_TOKENS));
    // The second call's soft rows are its own tower run's (the second), not the first call's.
    assertEquals(secondRows.rows.get(start)?.[0], 2000);
    assertEquals([...secondRows.rows.keys()], range(start, SQUARE_TOKENS));
  });

  it("stops before any tower run when the signal is aborted while the call waits for its turn", async () => {
    let open: () => void = () => {};
    const held = new Promise<void>((resolve) => {
      open = resolve;
    });
    let towerRuns = 0;
    const { embedder, text, tower } = rig({
      tower: true,
      gate: async () => {
        towerRuns += 1;
        if (towerRuns === 1) await held;
      },
    });
    const first = embedder.embed([{ type: "image", image: WIDE }]);
    const controller = new AbortController();
    const reason = new Error("stop before the image");
    const second = embedder
      .embed([{ type: "image", image: SQUARE }], { signal: controller.signal })
      .catch((error: unknown) => error);
    controller.abort(reason);
    open();
    await first;
    assertStrictEquals(await second, reason);
    assertEquals([tower.runs.length, text.runs.length], [1, 1]);
    assertEquals(hostInput(tower.runs[0], "pixels").shape[1], WIDE_TOKENS * 9);
  });

  it("fails loudly on a tower output of another row count", async () => {
    const events: string[] = [];
    const embedder = new Embedder({
      session: standIn("model", events, textAnswer).session,
      contract: TEXT_CONTRACT,
      rules: embedTextRules(TOKENIZER, SMALL, { images: true }),
      tower: {
        session: standIn("vision", events, () => ({
          soft: { dtype: "f32", shape: [1, 2, WIDTH], data: new Float32Array(2 * WIDTH) },
        })).session,
        contract: TOWER_CONTRACT,
      },
      deviceLimits: { maxStorageBufferBindingSize: 2147483644, maxBufferSize: 1099511627776 },
    });
    await assertRejects(
      () => embedder.embed([{ type: "image", image: SQUARE }]),
      Error,
      `the vision graph returned f32 [1, 2, ${WIDTH}], not f32 [1, ${SQUARE_TOKENS}, ${WIDTH}]`,
    );
  });

  it("counts the soft tokens toward the device's limit, before any run", async () => {
    const events: string[] = [];
    const length = SQUARE_TOKENS + 4;
    const embedder = new Embedder({
      session: standIn("model", events, textAnswer).session,
      contract: TEXT_CONTRACT,
      rules: embedTextRules(TOKENIZER, SMALL, { images: true }),
      tower: { session: standIn("vision", events, towerAnswer).session, contract: TOWER_CONTRACT },
      deviceLimits: { maxStorageBufferBindingSize: (length - 1) * 49152, maxBufferSize: 2 ** 40 },
    });
    await assertRejects(
      () => embedder.embed([{ type: "image", image: SQUARE }]),
      ModelInputError,
      `the input is ${length} tokens (<bos>, <eos> and the soft tokens of the image with their two markers each included), over this device's limit of ${length - 1}`,
    );
    assertEquals(events, []);
  });

  it("disposes the text Session, then the tower's, then releases, after the call in flight", async () => {
    let open: () => void = () => {};
    const held = new Promise<void>((resolve) => {
      open = resolve;
    });
    const { embedder, events } = rig({ tower: true, gate: () => held });
    const call = embedder.embed([{ type: "image", image: SQUARE }]);
    const disposal = embedder.dispose();
    open();
    await call;
    await disposal;
    assertEquals(events, ["vision.run", "model.run", "model.dispose", "vision.dispose", "release"]);
  });

  it("refuses an image part on an embedder without the tower, before any run", async () => {
    const { embedder, events } = rig({ tower: false });
    await assertRejects(
      () => embedder.embed([{ type: "image", image: SQUARE }]),
      ModelInputError,
      "build the pipeline with { vision: true }",
    );
    assertEquals(events, []);
  });

  it("refuses to be built with the tower but without image rules (an internal inconsistency)", () => {
    const events: string[] = [];
    assertThrows(
      () =>
        new Embedder({
          session: standIn("model", events, textAnswer).session,
          contract: TEXT_CONTRACT,
          rules: embedTextRules(TOKENIZER, SMALL),
          tower: {
            session: standIn("vision", events, towerAnswer).session,
            contract: TOWER_CONTRACT,
          },
          deviceLimits: { maxStorageBufferBindingSize: 2 ** 31, maxBufferSize: 2 ** 40 },
        }),
      Error,
      "the image rules and the vision tower are not built together",
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
  failures: { readonly open?: Error; readonly dispose?: Error } = {},
): EmbeddingGraph => ({
  createSession: async (_gpu, options) => {
    events.push(`${name}.open`);
    settings.push(options);
    if (failures.open !== undefined) throw failures.open;
    const stand = standIn(name, events, name === "vision" ? towerAnswer : textAnswer);
    return {
      ...stand.session,
      dispose: async () => {
        events.push(`${name}.dispose`);
        if (failures.dispose !== undefined) throw failures.dispose;
      },
    };
  },
});

/** A quant's effective settings (one knob, so the test sees them passed through). */
const QUANT_SETTINGS: SessionOptions = { numerics: { level: "reorder" } };

/** The family's row-block budget of the attention scores (`ATTENTION_ROW_BLOCK_BYTES`). */
const ROW_BLOCK_BYTES = 128 * 2 ** 20;

describe("openEmbedder: the Sessions of a pipeline with the tower, and their clean-up", () => {
  const opening = (
    events: string[],
    settings: SessionOptions[],
    overrides: {
      readonly model?: EmbeddingGraph;
      readonly tower?: EmbeddingGraph;
      readonly towerContract?: EmbeddingGemmaVisionContract;
      readonly assertGpu?: () => void;
      readonly release?: () => void;
    } = {},
  ) =>
    openEmbedder({
      gpu: LIMITED_GPU(),
      assertGpu: overrides.assertGpu ?? (() => {}),
      model: overrides.model ?? graphOf("model", events, settings),
      tower: {
        graph: overrides.tower ?? graphOf("vision", events, settings),
        contract: overrides.towerContract ?? TOWER_CONTRACT,
      },
      sessionOptions: QUANT_SETTINGS,
      contract: TEXT_CONTRACT,
      rules: SMALL_RULES,
      release:
        overrides.release ??
        (() => {
          events.push("release");
        }),
    });

  it("opens the text graph's Session, then the tower's, both with the quant's settings and the row-block budget", async () => {
    const events: string[] = [];
    const settings: SessionOptions[] = [];
    const embedder = await opening(events, settings);
    const expected = { ...QUANT_SETTINGS, attentionRowBlockMaxBytes: ROW_BLOCK_BYTES };
    assertEquals(settings, [expected, expected]);
    await embedder.embed([{ type: "image", image: SQUARE }]);
    await embedder.dispose();
    assertEquals(events, [
      "model.open",
      "vision.open",
      "vision.run",
      "model.run",
      "model.dispose",
      "vision.dispose",
      "release",
    ]);
  });

  it("disposes the text Session and releases when the tower's Session fails to open", async () => {
    const events: string[] = [];
    const failure = new Error("the tower's Session failed");
    const caught = await opening(events, [], {
      tower: graphOf("vision", events, [], { open: failure }),
    }).catch((error: unknown) => error);
    assertStrictEquals(caught, failure);
    assertEquals(events, ["model.open", "vision.open", "model.dispose", "release"]);
  });

  it("disposes both Sessions and releases when the embedder refuses its parts", async () => {
    const events: string[] = [];
    await assertRejects(
      () => opening(events, [], { towerContract: { ...TOWER_CONTRACT, width: WIDTH + 1 } }),
      Error,
      `the vision graph's soft tokens are ${WIDTH + 1} wide, the text graph's soft rows ${WIDTH}`,
    );
    assertEquals(events, [
      "model.open",
      "vision.open",
      "model.dispose",
      "vision.dispose",
      "release",
    ]);
  });

  it("releases before any Session is opened when the GPU checks fail", async () => {
    const events: string[] = [];
    const failure = new Error("a feature is missing");
    const caught = await opening(events, [], {
      assertGpu: () => {
        throw failure;
      },
    }).catch((error: unknown) => error);
    assertStrictEquals(caught, failure);
    assertEquals(events, ["release"]);
  });

  it("throws both failures in an AggregateError when the clean-up fails too, and still runs every step", async () => {
    const events: string[] = [];
    const failure = new Error("the tower's Session failed");
    const cleanup = new Error("the text Session's dispose failed");
    const caught = await opening(events, [], {
      model: graphOf("model", events, [], { dispose: cleanup }),
      tower: graphOf("vision", events, [], { open: failure }),
    }).catch((error: unknown) => error);
    assert(caught instanceof AggregateError, `not an AggregateError: ${String(caught)}`);
    assertEquals(
      caught.message,
      "embeddinggemma.Pipeline: construction failed, and so did its clean-up",
    );
    assertEquals(caught.errors.length, 2);
    assertStrictEquals(caught.errors[0], failure);
    assertStrictEquals(caught.errors[1], cleanup);
    assertEquals(events, ["model.open", "vision.open", "model.dispose", "release"]);
  });
});

// ---- ⑥ fromAssets with vision -------------------------------------------------------------------

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

/** Declaration-only `vision` (soft tokens of width `width`, position tables of `positions` rows). */
const visionInput = (width: number, positions = 840): ModelInput => ({
  graphs: {
    vision: parseIrDeclarationValue({
      format: "karume-ir",
      version: 2,
      requires: { ops: ["embedding", "sum"] },
      symbols: ["n"],
      inputs: [
        { name: "pixels", dtype: "f32", shape: [1, "9n", 768] },
        { name: "position_x", dtype: "i32", shape: [1, "9n"] },
        { name: "position_y", dtype: "i32", shape: [1, "9n"] },
      ],
      outputs: ["soft"],
      initializers: { rope: {} },
      values: {
        rope: { dtype: "f32", shape: [positions, width] },
        gx: { dtype: "f32", shape: [1, "9n", width] },
        gy: { dtype: "f32", shape: [1, "9n", width] },
        soft: { dtype: "f32", shape: [1, "n", width] },
      },
      states: {},
      nodes: [
        { op: "embedding", ins: ["rope", "position_x"], outs: ["gx"], attrs: { padding_idx: -1 } },
        { op: "embedding", ins: ["rope", "position_y"], outs: ["gy"], attrs: { padding_idx: -1 } },
        { op: "sum", ins: ["gx"], outs: ["soft"], attrs: { dim: 1 } },
      ],
    }),
  },
  consts: [],
  weights: [
    {
      graph: "vision",
      initializer: "rope",
      bytes: new Uint8Array(positions * width * 4),
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
 * The `pipelineConfig` of the two-graph distribution below, without `vision` and with it: ids inside
 * its 8-row table (admission reads declarations only — the tokenizer asset is never parsed here).
 */
const { vision: _vision, ...SMALL_TEXT_CONFIG } = {
  ...CONFIG_JSON,
  dimensions: [4, 2],
  maxTokens: 16,
  placeholderTokens: { image: 5, audio: 3, video: 4 },
};
const SMALL_VISION = {
  softTokenBudget: 280,
  imageTokenId: 5,
  beginImageTokenId: 6,
  endImageTokenId: 7,
  hiddenSize: 4,
};
const SMALL_CONFIG = { ...SMALL_TEXT_CONFIG, vision: SMALL_VISION };

/** The two-graph distribution (`vision` a weights role of the preset), and its assets. */
const distribution = async (options: {
  readonly vision?: ModelInput;
  readonly pipelineConfig?: Record<string, unknown>;
}) => {
  const tower = await writeContainer(options.vision ?? visionInput(4));
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
            vision: { bf16: await declared(tower, "text/vision/model.bf16") },
          },
          assets: { tokenizer: TOKENIZER_REF },
          quants: { bf16: { weights: { model: "bf16", vision: "bf16" }, session: {} } },
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
      ...partAssets("vision", tower),
      tokenizer: NOT_AN_ASSET,
    },
  };
};

const fakeGpu = () => fakeGpuContext(fakeDevice());

describe("Pipeline.fromAssets with vision: admission fails before touching a GPU", () => {
  it("positive path: both graphs pass admission and the build stops at parsing the tokenizer asset", async () => {
    await assertRejects(
      async () => Pipeline.fromAssets(await distribution({}), { vision: true, gpu: fakeGpu() }),
      Error,
      "tokenizer.format: missing",
    );
  });

  it("without vision the tower is neither opened nor checked (a tower of another width passes)", async () => {
    await assertRejects(
      async () =>
        Pipeline.fromAssets(await distribution({ vision: visionInput(7) }), { gpu: fakeGpu() }),
      Error,
      "tokenizer.format: missing",
    );
  });

  const rejected: readonly [string, Parameters<typeof distribution>[0], string][] = [
    [
      "a pipelineConfig without vision",
      { pipelineConfig: SMALL_TEXT_CONFIG },
      "vision was requested, but this model's pipelineConfig declares no 'vision'",
    ],
    [
      "a hiddenSize other than the text graph's soft rows",
      { pipelineConfig: { ...SMALL_CONFIG, vision: { ...SMALL_VISION, hiddenSize: 8 } } },
      "pipelineConfig.vision.hiddenSize 8 is not the width of the text graph's soft rows (4)",
    ],
    [
      "an image marker outside the text graph's table",
      { pipelineConfig: { ...SMALL_CONFIG, vision: { ...SMALL_VISION, endImageTokenId: 8 } } },
      "pipelineConfig.vision.endImageTokenId 8 is outside the embedding table's 8 rows",
    ],
    [
      "a tower whose soft tokens are of another width",
      { vision: visionInput(6) },
      "the vision graph's output 'soft' is f32 [1, n, 6], expected f32 [1, n, 4]",
    ],
    [
      "a tower whose tables cover fewer positions than the budget needs",
      { vision: visionInput(4, 420) },
      "lets an image side reach 840 patches, but the vision graph's tables cover 420 positions",
    ],
  ];
  for (const [label, patch, message] of rejected) {
    it(label, async () => {
      const input = await distribution(patch);
      const error = await assertRejects(
        () => Pipeline.fromAssets(input, { vision: true, gpu: fakeGpu() }),
        Error,
        message,
      );
      assert(
        !(error instanceof ModelInputError),
        `an asset mismatch is not an input error: ${label}`,
      );
    });
  }

  it("refuses a vision option that is not a boolean, as an input error", async () => {
    const input = await distribution({});
    await assertRejects(
      () =>
        // oxlint-disable-next-line typescript/unbound-method -- Reflect.apply supplies Pipeline as `this`
        Reflect.apply(Pipeline.fromAssets, Pipeline, [input, { vision: "yes", gpu: fakeGpu() }]),
      ModelInputError,
      'vision "yes" is not a boolean',
    );
  });
});
