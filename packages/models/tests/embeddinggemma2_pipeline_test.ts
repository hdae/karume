/**
 * The EmbeddingGemma family (`@karume/models/embeddinggemma`) over the text model of EmbeddingGemma 2,
 * on the host only — no GPU, no real weights (the real-GPU run over the local distribution is
 * `packages/e2e/tests/e2e_embeddinggemma2_pipeline_test.ts`).
 *
 * ① The manifest's `pipelineConfig`: the distribution's shape passes; an unknown key, a missing key
 *    and every wrong type fail loudly (a misspelt field must not leave a default the distributor
 *    did not intend).
 * ② One request on the host (`planEmbedRequest`): prompts, widths, content parts, the token limit
 *    and the placeholder refusal — every refusal is a `ModelInputError` that names what to fix.
 * ③ The tokenizer: the ids the pipeline builds for upstream's prompted texts (prompt name + body)
 *    equal upstream's own `<bos>` … `<eos>` ids (the parity fixture, from `tokenizers` alone).
 * ④ The leading slice re-normalised to unit length (`truncateEmbedding`), on values whose result is
 *    known exactly.
 * ⑤ The graph contract, including the declared maximum against the graph's folded constants (the
 *    RoPE tables — the sliding window is a band declared on the attention, not a constant).
 * ⑥ The run lifecycle over a stand-in Session (`Embedder` — what `Pipeline` delegates to): the run's
 *    inputs, the width, one run at a time, cancellation before and during a call, dispose after the
 *    work in flight, then the GPU release; `tokenize` gives the ids `embed` runs, off the chain, and
 *    returns what only `embed` refuses (over the limit, a placeholder id).
 * ⑦ `Pipeline.fromAssets` with a declaration-only container: manifest violations fail before any GPU
 *    is touched, and the positive path passes admission and stops at the tokenizer asset.
 */

import { readTextFile } from "@karume/test-support/fs";
import { describe, it } from "vite-plus/test";
import { assert, assertEquals, assertRejects, assertStrictEquals, assertThrows } from "@std/assert";
import { parseManifest } from "@karume/hub";
import type { RunInputs, RunOutputs, Tensor } from "@karume/runtime";
import { ModelInputError } from "../src/errors.ts";
import { createBpeModel } from "../src/text/bpe.ts";
import type { GemmaTokenizerAssets } from "../src/gemma/text/asset.ts";
import { Tokenizer } from "../src/gemma/text/tokenizer.ts";
import type { GraphOwner } from "../src/hub/components.ts";
import { parseEmbeddingGemmaPipelineConfig } from "../src/embeddinggemma/config.ts";
import {
  type EmbedTextRules,
  embedTextRules,
  planEmbedRequest,
  tokenizeEmbedInput,
  truncateEmbedding,
} from "../src/embeddinggemma/input.ts";
import {
  EMBEDDINGGEMMA_SESSION_POLICY,
  Embedder,
  type EmbeddingGemmaGraphContract,
  type EmbeddingSession,
  embeddingGemmaDeviceLimit,
  embeddingGemmaGraphContract,
  Pipeline,
} from "../src/embeddinggemma/pipeline.ts";
import type { ModelInput } from "../../runtime/tests/helpers/container-write.ts";
import { fakeDevice, fakeGpuContext } from "../../runtime/tests/helpers/fake-gpu.ts";
import {
  parseIrDeclarationValue,
  partAssets,
  writeContainer,
} from "./helpers/container-fixture.ts";
import { fileRef } from "./helpers/container-loading-fixture.ts";

// ---- Fixtures --------------------------------------------------------------------------------

/** The parts of the parity fixture these checks read (`tools/export-recipes/embeddinggemma2/tokenizer.py`). */
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

/** The tokenizer tables of the fixture's subset (the asset parser's checks are covered elsewhere). */
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

/** One encode case of the parity fixture by name. */
const parityCase = (name: string): ParityFixture["encode"][number] => {
  const found = PARITY.encode.find((row) => row.name === name);
  if (found === undefined) throw new Error(`the parity fixture has no case '${name}'`);
  return found;
};

/**
 * The `pipelineConfig` of `models/karume-embeddinggemma-2/karume.json` (the dist recipe's
 * `PIPELINE` — upstream's 20 prompt names and 8 prefixes, verbatim).
 */
const CONFIG_JSON = {
  prompts: {
    BitextMining: "task: search result | query: ",
    Classification: "task: classification | query: ",
    Clustering: "task: clustering | query: ",
    CodeRetrieval: "task: code retrieval | query: ",
    Document: "title: none | text: ",
    FactChecking: "task: fact checking | query: ",
    InstructionRetrieval: "task: code retrieval | query: ",
    MultilabelClassification: "task: classification | query: ",
    PairClassification: "task: sentence similarity | query: ",
    QuestionAnswering: "task: question answering | query: ",
    Reranking: "task: search result | query: ",
    Retrieval: "task: search result | query: ",
    "Retrieval-document": "title: none | text: ",
    "Retrieval-query": "task: search result | query: ",
    STS: "task: sentence similarity | query: ",
    SearchQuery: "task: search result | query: ",
    SentenceSimilarity: "task: sentence similarity | query: ",
    Summarization: "task: sentence similarity | query: ",
    document: "title: none | text: ",
    query: "task: search result | query: ",
  },
  dimensions: [768, 512, 256, 128],
  maxTokens: 8192,
  placeholderTokens: { image: 258880, audio: 258881, video: 258884 },
};

const CONFIG = parseEmbeddingGemmaPipelineConfig(CONFIG_JSON);
const RULES = embedTextRules(TOKENIZER, CONFIG);

/** The rules over a config with one field replaced. */
const rulesWith = (patch: Record<string, unknown>): EmbedTextRules =>
  embedTextRules(TOKENIZER, parseEmbeddingGemmaPipelineConfig({ ...CONFIG_JSON, ...patch }));

const idsOf = (request: { readonly ids: Int32Array<ArrayBuffer> }): number[] => [...request.ids];

// ---- ① pipelineConfig --------------------------------------------------------------------------

describe("pipelineConfig: the distribution's four fields", () => {
  it("reads the prompts in their order, the widths, the limit and the placeholder ids", () => {
    assertEquals([...CONFIG.prompts.keys()], Object.keys(CONFIG_JSON.prompts));
    assertEquals(CONFIG.prompts.get("query"), "task: search result | query: ");
    assertEquals(CONFIG.dimensions, [768, 512, 256, 128]);
    assertEquals(CONFIG.maxTokens, 8192);
    assertEquals(
      [...CONFIG.placeholderTokens],
      [
        ["image", 258880],
        ["audio", 258881],
        ["video", 258884],
      ],
    );
  });

  it("fails on an unknown key, naming the six it allows", () => {
    assertThrows(
      () => parseEmbeddingGemmaPipelineConfig({ ...CONFIG_JSON, maxLength: 2048 }),
      Error,
      "pipelineConfig: unknown key 'maxLength' (allowed: prompts / dimensions / maxTokens / placeholderTokens / vision / audio)",
    );
  });

  for (const key of ["prompts", "dimensions", "maxTokens", "placeholderTokens"]) {
    it(`fails when ${key} is missing`, () => {
      const raw: Record<string, unknown> = { ...CONFIG_JSON };
      delete raw[key];
      assertThrows(
        () => parseEmbeddingGemmaPipelineConfig(raw),
        Error,
        `pipelineConfig.${key}: missing`,
      );
    });
  }

  it("fails on a config that is not an object", () => {
    for (const raw of [undefined, null, [], "prompts"]) {
      assertThrows(
        () => parseEmbeddingGemmaPipelineConfig(raw),
        Error,
        "pipelineConfig: missing / not an object",
      );
    }
  });

  const wrongTypes: readonly [string, Record<string, unknown>, string][] = [
    [
      "prompts as an array",
      { prompts: ["query"] },
      "pipelineConfig.prompts: missing / not an object",
    ],
    [
      "a prefix that is not a string",
      { prompts: { query: 1 } },
      "pipelineConfig.prompts.query: the prefix is not a string",
    ],
    [
      "dimensions as a number",
      { dimensions: 768 },
      "pipelineConfig.dimensions: not a non-empty array",
    ],
    ["no dimension", { dimensions: [] }, "pipelineConfig.dimensions: not a non-empty array"],
    [
      "a width that is not an integer",
      { dimensions: [768, 256.5] },
      "has an element that is not a positive integer",
    ],
    ["a zero width", { dimensions: [768, 0] }, "has an element that is not a positive integer"],
    [
      "widths in ascending order",
      { dimensions: [128, 768] },
      "pipelineConfig.dimensions: not strictly descending",
    ],
    [
      "a width twice",
      { dimensions: [768, 768] },
      "pipelineConfig.dimensions: not strictly descending",
    ],
    [
      "maxTokens as a string",
      { maxTokens: "2048" },
      "pipelineConfig.maxTokens: not an integer of at least 2",
    ],
    [
      "maxTokens below <bos> <eos>",
      { maxTokens: 1 },
      "pipelineConfig.maxTokens: not an integer of at least 2",
    ],
    [
      "placeholderTokens as an array",
      { placeholderTokens: [258880] },
      "pipelineConfig.placeholderTokens: missing / not an object",
    ],
    [
      "a negative placeholder id",
      { placeholderTokens: { image: -1 } },
      "pipelineConfig.placeholderTokens.image: not a token id",
    ],
    [
      "a placeholder id that is not an integer",
      { placeholderTokens: { image: "258880" } },
      "pipelineConfig.placeholderTokens.image: not a token id",
    ],
    [
      "one id for two modalities",
      { placeholderTokens: { image: 7, video: 7 } },
      "pipelineConfig.placeholderTokens.video: id 7 is declared twice",
    ],
  ];
  for (const [label, patch, message] of wrongTypes) {
    it(`fails on ${label}`, () => {
      assertThrows(
        () => parseEmbeddingGemmaPipelineConfig({ ...CONFIG_JSON, ...patch }),
        Error,
        message,
      );
    });
  }
});

// ---- ② one request -----------------------------------------------------------------------------

/** Asserts that `run` is refused with a `ModelInputError` whose message holds every fragment. */
const assertRefused = (run: () => unknown, ...fragments: readonly string[]): void => {
  const error = assertThrows(run, ModelInputError);
  for (const fragment of fragments) {
    assert(error.message.includes(fragment), `"${fragment}" is not in: ${error.message}`);
  }
};

describe("planEmbedRequest: prompts", () => {
  it("prepends a declared prompt's prefix", () => {
    const { text, idsWithSpecials } = parityCase("prompt-query");
    const body = text.slice("task: search result | query: ".length);
    assertEquals(idsOf(planEmbedRequest(RULES, body, { prompt: "query" })), idsWithSpecials);
  });

  it("prepends nothing without a prompt", () => {
    const { text, idsWithSpecials } = parityCase("english");
    assertEquals(idsOf(planEmbedRequest(RULES, text, {})), idsWithSpecials);
  });

  it("refuses an unknown prompt name, listing the declared ones", () => {
    assertRefused(
      () => planEmbedRequest(RULES, "text", { prompt: "Query" }),
      'prompt "Query" is not one of this model\'s prompts',
      "BitextMining / Classification",
      "/ document / query)",
    );
  });

  it("refuses a prompt that is not a string (an untyped caller)", () => {
    assertRefused(
      () => Reflect.apply(planEmbedRequest, undefined, [RULES, "text", { prompt: 1 }]),
      "prompt 1 is not one of",
    );
  });
});

describe("planEmbedRequest: dimensions", () => {
  it("takes the full width when none is asked for", () => {
    assertEquals(planEmbedRequest(RULES, "text", {}).dimension, 768);
  });

  for (const dimension of [768, 512, 256, 128]) {
    it(`takes the declared width ${dimension}`, () => {
      assertEquals(planEmbedRequest(RULES, "text", { dimension }).dimension, dimension);
    });
  }

  for (const dimension of [384, 64, 0, 256.5, Number.NaN]) {
    it(`refuses the undeclared width ${dimension}, listing the declared ones`, () => {
      assertRefused(
        () => planEmbedRequest(RULES, "text", { dimension }),
        `dimension ${dimension} is not one of this model's output dimensions (768 / 512 / 256 / 128)`,
      );
    });
  }

  it("refuses a width given as a string (an untyped caller)", () => {
    assertRefused(
      () => Reflect.apply(planEmbedRequest, undefined, [RULES, "text", { dimension: "256" }]),
      'dimension "256" is not one of',
    );
  });
});

describe("planEmbedRequest: content parts", () => {
  const { text } = parityCase("english-long");
  const half = Math.floor(text.length / 2);

  it("embeds text parts concatenated in order with nothing between them, as the one string", () => {
    const parts = [
      { type: "text", text: text.slice(0, half) },
      { type: "text", text: text.slice(half) },
    ] as const;
    assertEquals(
      idsOf(planEmbedRequest(RULES, parts, {})),
      idsOf(planEmbedRequest(RULES, text, {})),
    );
    assertEquals(
      idsOf(planEmbedRequest(RULES, parts, { prompt: "document" })),
      idsOf(planEmbedRequest(RULES, text, { prompt: "document" })),
    );
  });

  it("embeds a single part as its text", () => {
    assertEquals(
      idsOf(planEmbedRequest(RULES, [{ type: "text", text }], {})),
      parityCase("english-long").idsWithSpecials,
    );
  });

  it("refuses an input with no part", () => {
    assertRefused(() => planEmbedRequest(RULES, [], {}), "the input has no content part");
  });

  it("refuses a part of another type, naming it and the types the model accepts", () => {
    const parts: readonly unknown[] = [
      { type: "text", text: "a" },
      { type: "image", image: {} },
    ];
    assertRefused(
      () => Reflect.apply(planEmbedRequest, undefined, [RULES, parts, {}]),
      'content part 1 has type "image" — this model accepts "text" parts',
    );
  });

  it("refuses a part that is not an object, and a text part without a string", () => {
    assertRefused(
      () => Reflect.apply(planEmbedRequest, undefined, [RULES, [null], {}]),
      "content part 0 is not an object",
    );
    assertRefused(
      () => Reflect.apply(planEmbedRequest, undefined, [RULES, [{ type: "text", text: 3 }], {}]),
      "content part 0 is a text part whose text is not a string (3)",
    );
  });

  it("refuses an input that is neither a string nor an array", () => {
    assertRefused(
      () => Reflect.apply(planEmbedRequest, undefined, [RULES, 42, {}]),
      "the input is neither a string nor an array of content parts (42)",
    );
  });

  it("embeds the empty string as <bos> <eos> (the shortest sequence the graph runs)", () => {
    assertEquals(idsOf(planEmbedRequest(RULES, "", {})), [ASSETS.bosId, ASSETS.eosId]);
  });
});

describe("planEmbedRequest: the token limit (<bos> and <eos> included)", () => {
  const { text, idsWithSpecials } = parityCase("english");
  const length = idsWithSpecials.length;

  it("accepts a sequence at the limit", () => {
    assertEquals(
      idsOf(planEmbedRequest(rulesWith({ maxTokens: length }), text, {})),
      idsWithSpecials,
    );
  });

  it("refuses a sequence one over the limit, stating the count and the limit", () => {
    assertRefused(
      () => planEmbedRequest(rulesWith({ maxTokens: length - 1 }), text, {}),
      `the input is ${length} tokens (<bos> and <eos> included), over this model's limit of ${length - 1}`,
      "nothing is truncated",
    );
  });

  it("counts the prompt's prefix towards the limit", () => {
    const { text: prompted, idsWithSpecials: promptedIds } = parityCase("prompt-query");
    const body = prompted.slice("task: search result | query: ".length);
    const rules = rulesWith({ maxTokens: promptedIds.length - 1 });
    assertRefused(
      () => planEmbedRequest(rules, body, { prompt: "query" }),
      `is ${promptedIds.length} tokens`,
    );
  });
});

describe("planEmbedRequest: the placeholder ids are refused", () => {
  for (const [modality, spelling, id] of [
    ["image", "<|image|>", 258880],
    ["audio", "<|audio|>", 258881],
    ["video", "<|video|>", 258884],
  ] as const) {
    it(`refuses a text holding the ${modality} placeholder, naming the token`, () => {
      assertRefused(
        () => planEmbedRequest(RULES, `look at ${spelling} here`, {}),
        `the text holds the ${modality} placeholder token "${spelling}" (id ${id})`,
        "remove that spelling",
      );
    });
  }

  it("refuses a placeholder inside a content part too", () => {
    assertRefused(
      () =>
        planEmbedRequest(
          RULES,
          [
            { type: "text", text: "a" },
            { type: "text", text: "<|video|>" },
          ],
          {},
        ),
      'the video placeholder token "<|video|>"',
    );
  });

  it("refuses only the declared ids (the media begin / end markers are text tokens like any other)", () => {
    const { text, idsWithSpecials } = parityCase("specials-in-text");
    assertEquals(idsOf(planEmbedRequest(RULES, text, {})), idsWithSpecials);
    const markers = planEmbedRequest(RULES, "<|image><image|> <|audio><audio|>", {});
    assertEquals(idsOf(markers).slice(1, 3), [255999, 258882]);
  });
});

// ---- ③ the tokenizer: upstream's prompted texts -----------------------------------------------

describe("the pipeline's ids for upstream's prompted texts equal upstream's <bos> … <eos> ids", () => {
  const prompted = PARITY.encode.filter((row) => row.name.startsWith("prompt-"));

  it("covers every prompt case of the fixture (a positive control)", () => {
    assert(prompted.length >= 9, `${prompted.length} prompt cases`);
  });

  for (const row of prompted) {
    it(`${row.name}: prompt name + body give the fixture's ids`, () => {
      // The first declared name whose prefix starts the text (several names share a prefix).
      const found = [...CONFIG.prompts].find(([, prefix]) => row.text.startsWith(prefix));
      assert(found !== undefined, `no declared prompt prefixes '${row.text}'`);
      const [name, prefix] = found;
      const request = planEmbedRequest(RULES, row.text.slice(prefix.length), { prompt: name });
      assertEquals(idsOf(request), row.idsWithSpecials);
    });
  }

  for (const name of ["english", "japanese-long", "emoji", "long-past-window"]) {
    it(`${name}: a text without a prompt gives the fixture's ids`, () => {
      const row = parityCase(name);
      assertEquals(idsOf(planEmbedRequest(RULES, row.text, {})), row.idsWithSpecials);
    });
  }

  it("gemma4-media-markup: the fixture's markup text is refused at its first placeholder", () => {
    const row = parityCase("gemma4-media-markup");
    const at = row.idsWithSpecials.indexOf(258880);
    assertRefused(
      () => planEmbedRequest(RULES, row.text, {}),
      `the image placeholder token "<|image|>" (id 258880) at token ${at}`,
    );
  });
});

describe("embedTextRules: the tokenizer asset against the manifest", () => {
  it("refuses an asset that does not wrap the sequence in <bos> … <eos>", () => {
    const plain = new Tokenizer({ ...ASSETS, spec: { postProcessor: "none" } });
    assertThrows(
      () => embedTextRules(plain, CONFIG),
      Error,
      "the tokenizer asset's post-processor is 'none'",
    );
  });

  it("refuses a placeholder id the asset does not know", () => {
    const config = parseEmbeddingGemmaPipelineConfig({
      ...CONFIG_JSON,
      placeholderTokens: { image: 999999 },
    });
    assertThrows(
      () => embedTextRules(TOKENIZER, config),
      Error,
      "pipelineConfig.placeholderTokens.image (id 999999) is not a token of the tokenizer asset",
    );
  });

  it("refuses a placeholder id that is a vocabulary token, not an added one", () => {
    const config = parseEmbeddingGemmaPipelineConfig({
      ...CONFIG_JSON,
      placeholderTokens: { image: 818 },
    });
    const error = assertThrows(() => embedTextRules(TOKENIZER, config), Error);
    assert(!(error instanceof ModelInputError), "an asset mismatch is not an input error");
    assert(error.message.includes("is not an added token of the tokenizer asset"), error.message);
  });
});

// ---- ④ the leading slice ------------------------------------------------------------------------

describe("truncateEmbedding: the leading slice re-normalised to unit length", () => {
  it("returns the vector itself at the full width", () => {
    const vector = Float32Array.from([0.6, 0, 0.8, 0]);
    assertStrictEquals(truncateEmbedding(vector, 4), vector);
  });

  it("divides the leading slice by its norm", () => {
    // [3, 4, 12] / 13 → the leading two are [3, 4] / 13, whose norm is 5 / 13 → [0.6, 0.8].
    const vector = Float32Array.from([3 / 13, 4 / 13, 12 / 13]);
    const truncated = truncateEmbedding(vector, 2);
    assertEquals(truncated.length, 2);
    for (const [index, expected] of [0.6, 0.8].entries()) {
      assert(
        Math.abs(truncated[index] - expected) <= 2 ** -24,
        `${truncated[index]} vs ${expected}`,
      );
    }
    assert(Math.abs(Math.hypot(...truncated) - 1) <= 1e-7);
  });

  it("gives exactly one where the slice holds a single non-zero value", () => {
    assertEquals([...truncateEmbedding(Float32Array.from([0.6, 0, 0.8, 0]), 2)], [1, 0]);
  });

  it("keeps an all-zero slice at zero instead of dividing by zero (the 1e-12 floor)", () => {
    assertEquals([...truncateEmbedding(Float32Array.from([0, 0, 1]), 2)], [0, 0]);
  });

  it("does not modify its input", () => {
    const vector = Float32Array.from([3 / 13, 4 / 13, 12 / 13]);
    const before = [...vector];
    truncateEmbedding(vector, 2);
    assertEquals([...vector], before);
  });

  it("fails on a width that is not a leading slice (an internal inconsistency)", () => {
    assertThrows(
      () => truncateEmbedding(Float32Array.from([1, 0]), 3),
      Error,
      "width 3 is not a leading slice of 2",
    );
  });
});

// ---- ⑤ the graph contract ---------------------------------------------------------------------

type Graph = GraphOwner["graph"];

/**
 * The declaration of the recipe's graph (the shapes of the mirror's `model`, minus the layers): the
 * folded RoPE tables of the sliding and the full layers bound T; the window is `attrs.band` on the
 * sliding attention, so no band constant is left.
 */
const recipeGraph = (): Graph => ({
  format: "karume-ir",
  version: 2,
  requires: { ops: [] },
  symbols: ["T"],
  inputs: [
    { name: "input_ids", dtype: "i32", shape: [1, "T"] },
    { name: "pool_mask", dtype: "f32", shape: [1, "T"] },
    { name: "soft_embeds", dtype: "f32", shape: [1, "T", 512] },
    { name: "soft_mask", dtype: "bool", shape: [1, "T", 1] },
  ],
  outputs: ["div_1"],
  initializers: {},
  values: {
    "model.embed_tokens.weight": { dtype: "f32", shape: [262144, 512] },
    rope: { dtype: "f32", shape: [1, 1, 8192, 256] },
    rope_full: { dtype: "f32", shape: [1, 1, 8192, 512] },
    rope_t: { dtype: "f32", shape: [1, 1, "T", 256] },
    rope_full_t: { dtype: "f32", shape: [1, 1, "T", 512] },
    linear: { dtype: "f32", shape: [1, "T", 12288] },
    query: { dtype: "f32", shape: [1, 4, "T", 512] },
    div_1: { dtype: "f32", shape: [1, 768] },
  },
  states: {},
  nodes: [
    {
      op: "embedding",
      ins: ["model.embed_tokens.weight", "input_ids"],
      outs: ["embedding"],
      attrs: {},
      states: {},
    },
    {
      op: "sym_prefix_slice",
      ins: ["rope"],
      outs: ["rope_t"],
      attrs: { sym: "T", slices: [{ dim: 2, coeff: 1, offset: 0 }] },
      states: {},
    },
    {
      op: "sym_prefix_slice",
      ins: ["rope_full"],
      outs: ["rope_full_t"],
      attrs: { sym: "T", slices: [{ dim: 2, coeff: 1, offset: 0 }] },
      states: {},
    },
  ],
});

const contractOf = (edit: (graph: Graph) => Graph, config = CONFIG): EmbeddingGemmaGraphContract =>
  embeddingGemmaGraphContract({ graph: edit(recipeGraph()) }, config);

const same = (graph: Graph): Graph => graph;

describe("embeddingGemmaGraphContract", () => {
  it("accepts the recipe's declaration and reads the output, its width and the vocabulary", () => {
    assertEquals(contractOf(same), {
      output: "div_1",
      width: 768,
      vocabularySize: 262144,
      softWidth: 512,
      bytesPerToken: 12288 * 4,
    });
  });

  it("refuses a declared maximum that differs from the graph's folded constants, both ways", () => {
    for (const maxTokens of [16384, 4096]) {
      const config = parseEmbeddingGemmaPipelineConfig({ ...CONFIG_JSON, maxTokens });
      assertThrows(
        () => contractOf(same, config),
        Error,
        `pipelineConfig.maxTokens is ${maxTokens}, but the graph runs 'T' up to 8192`,
      );
    }
  });

  it("reads the graph's maximum as the smallest bound of its prefix slices", () => {
    const graph = (base: Graph): Graph => ({
      ...base,
      values: { ...base.values, rope: { dtype: "f32", shape: [1, 1, 1024, 256] } },
    });
    assertThrows(() => contractOf(graph), Error, "the graph runs 'T' up to 1024");
    const config = parseEmbeddingGemmaPipelineConfig({ ...CONFIG_JSON, maxTokens: 1024 });
    assertEquals(contractOf(graph, config).width, 768);
  });

  it("refuses a graph whose maximum it cannot read (no prefix slice of T)", () => {
    assertThrows(
      () => contractOf((base) => ({ ...base, nodes: base.nodes.slice(0, 1) })),
      Error,
      "no sym_prefix_slice bounds 'T'",
    );
  });

  it("refuses an output width other than the full declared width", () => {
    const config = parseEmbeddingGemmaPipelineConfig({ ...CONFIG_JSON, dimensions: [512, 256] });
    assertThrows(
      () => contractOf(same, config),
      Error,
      "graph output 'div_1' is f32 [1, 768], expected f32 [1, 512]",
    );
  });

  it("refuses inputs the host does not feed", () => {
    assertThrows(
      () =>
        contractOf((base) => ({
          ...base,
          inputs: [...base.inputs, { name: "attention_mask", dtype: "f32", shape: [1, "T"] }],
        })),
      Error,
      "the graph has 5 inputs",
    );
    assertThrows(
      () =>
        contractOf((base) => ({
          ...base,
          inputs: base.inputs.map((input) =>
            input.name === "pool_mask" ? { ...input, dtype: "i32" } : input,
          ),
        })),
      Error,
      "graph input 'pool_mask' is i32 [1, T], expected f32 [1, T]",
    );
    assertThrows(
      () => contractOf((base) => ({ ...base, inputs: base.inputs.slice(0, 1) })),
      Error,
      "graph input 'pool_mask' is missing",
    );
  });

  it("refuses a graph without the soft inputs, soft rows of another width than the table, a full-width soft mask and another input order", () => {
    assertThrows(
      () => contractOf((base) => ({ ...base, inputs: base.inputs.slice(0, 2) })),
      Error,
      "graph input 'soft_embeds' is missing",
    );
    assertThrows(
      () =>
        contractOf((base) => ({
          ...base,
          inputs: base.inputs.map((input) =>
            input.name === "soft_embeds" ? { ...input, shape: [1, "T", 768] } : input,
          ),
        })),
      Error,
      "graph input 'soft_embeds' is f32 [1, T, 768], expected f32 [1, T, 512]",
    );
    assertThrows(
      () =>
        contractOf((base) => ({
          ...base,
          inputs: base.inputs.map((input) =>
            input.name === "soft_mask" ? { ...input, shape: [1, "T", 512] } : input,
          ),
        })),
      Error,
      "graph input 'soft_mask' is bool [1, T, 512], expected bool [1, T, 1]",
    );
    assertThrows(
      () =>
        contractOf((base) => ({
          ...base,
          inputs: [base.inputs[0], base.inputs[1], base.inputs[3], base.inputs[2]],
        })),
      Error,
      "not input_ids / pool_mask / soft_embeds / soft_mask in that order",
    );
  });

  it("reads the bytes per token as the largest symbolic value's", () => {
    const wider = (base: Graph): Graph => ({
      ...base,
      values: { ...base.values, wider: { dtype: "f32", shape: [1, "T", 24, 1024] } },
    });
    assertEquals(contractOf(wider).bytesPerToken, 24 * 1024 * 4);
  });

  it("refuses a value with the length on two axes (it grows with T², not per token)", () => {
    assertThrows(
      () =>
        contractOf((base) => ({
          ...base,
          values: { ...base.values, band_t: { dtype: "f32", shape: [1, 1, "T", "T"] } },
        })),
      Error,
      "graph value 'band_t' is [1, 1, T, T] — every symbolic value must hold 'T' on exactly one axis" +
        " (the device's token limit is read per token; a T × T mask must be declared as the" +
        " attention's band instead)",
    );
  });

  it("refuses more than one symbolic length, and an embedding it cannot find", () => {
    assertThrows(
      () => contractOf((base) => ({ ...base, symbols: ["T", "S"] })),
      Error,
      "the graph's symbolic dimensions are [T, S]",
    );
    assertThrows(
      () => contractOf((base) => ({ ...base, nodes: base.nodes.slice(1) })),
      Error,
      "the graph has 0 embedding nodes that read 'input_ids'",
    );
  });
});

// ---- ⑥ the run lifecycle -----------------------------------------------------------------------

/** A promise whose settlement the test drives. */
const deferred = <T>() => {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((settle) => {
    resolve = settle;
  });
  return { promise, resolve };
};

/** The output the stand-in Session answers with (a unit vector of width 4). */
const UNIT = [0.6, 0, 0.8, 0];

type Stand = {
  readonly session: EmbeddingSession;
  /** Every run's inputs, in call order. */
  readonly runs: RunInputs[];
  /** What happened to the Session and the GPU, in order. */
  readonly events: string[];
  /** The most runs in flight at once. */
  maxInFlight: number;
  /** When set, each run waits for it before answering. */
  gate?: Promise<void>;
  /** When set, replaces the output the run answers with. */
  output?: Tensor;
  /** When set, `dispose` rejects with it. */
  disposeFailure?: Error;
};

/** A Session whose run answers `[1, 4]` = {@link UNIT}. */
const standIn = (): Stand => {
  let inFlight = 0;
  const stand: Stand = {
    runs: [],
    events: [],
    maxInFlight: 0,
    session: {
      run: async (inputs): Promise<RunOutputs> => {
        stand.runs.push(inputs);
        inFlight += 1;
        stand.maxInFlight = Math.max(stand.maxInFlight, inFlight);
        try {
          await stand.gate;
          return {
            out: stand.output ?? { dtype: "f32", shape: [1, 4], data: Float32Array.from(UNIT) },
          };
        } finally {
          inFlight -= 1;
        }
      },
      diagnostics: () => {
        throw new Error("the stand-in Session has no diagnostics");
      },
      dispose: async () => {
        stand.events.push("session.dispose");
        if (stand.disposeFailure !== undefined) throw stand.disposeFailure;
      },
    },
  };
  return stand;
};

/** The config of the stand-in (widths 4 and 2). */
const SMALL = parseEmbeddingGemmaPipelineConfig({ ...CONFIG_JSON, dimensions: [4, 2] });

/** The stand-in's graph contract (the real graph's 49,152 B per token). */
const STAND_CONTRACT: EmbeddingGemmaGraphContract = {
  output: "out",
  width: 4,
  vocabularySize: 262144,
  softWidth: 3,
  bytesPerToken: 49152,
};

/** The development machine's limits (a 2 GiB binding): 43,690 tokens, above every input here. */
const STAND_DEVICE_LIMITS = {
  maxStorageBufferBindingSize: 2147483644,
  maxBufferSize: 1099511627776,
};
const STAND_DEVICE_LIMIT = embeddingGemmaDeviceLimit(STAND_CONTRACT, STAND_DEVICE_LIMITS);

const embedderOf = (stand: Stand): Embedder =>
  new Embedder({
    session: stand.session,
    contract: STAND_CONTRACT,
    rules: embedTextRules(TOKENIZER, SMALL),
    deviceLimits: STAND_DEVICE_LIMITS,
    release: () => {
      stand.events.push("release");
    },
  });

/** The host tensor a run received under `name`. */
const hostInput = (inputs: RunInputs, name: string): Tensor => {
  const input = inputs[name];
  if (input === undefined || !("data" in input)) throw new Error(`no host tensor '${name}'`);
  return input;
};

describe("Embedder: one run per call", () => {
  it("runs the planned ids with a pool mask of ones and returns the unit vector", async () => {
    const stand = standIn();
    const embedder = embedderOf(stand);
    const { text, idsWithSpecials } = parityCase("english");

    const vector = await embedder.embed(text);

    assertEquals(stand.runs.length, 1);
    assertEquals(Object.keys(stand.runs[0]).sort(), [
      "input_ids",
      "pool_mask",
      "soft_embeds",
      "soft_mask",
    ]);
    const ids = hostInput(stand.runs[0], "input_ids");
    assertEquals([ids.dtype, ids.shape], ["i32", [1, idsWithSpecials.length]]);
    assertEquals([...ids.data], idsWithSpecials);
    const mask = hostInput(stand.runs[0], "pool_mask");
    assertEquals([mask.dtype, mask.shape], ["f32", [1, idsWithSpecials.length]]);
    assertEquals([...mask.data], Array<number>(idsWithSpecials.length).fill(1));
    // No soft row: zeros of the table's width and an all-false mask, one word per row.
    const softEmbeds = hostInput(stand.runs[0], "soft_embeds");
    assertEquals(
      [softEmbeds.dtype, softEmbeds.shape],
      ["f32", [1, idsWithSpecials.length, STAND_CONTRACT.softWidth]],
    );
    assert([...softEmbeds.data].every((value) => value === 0));
    const softMask = hostInput(stand.runs[0], "soft_mask");
    assertEquals([softMask.dtype, softMask.shape], ["bool", [1, idsWithSpecials.length, 1]]);
    assert([...softMask.data].every((value) => value === 0));
    assertEquals([...vector], [...Float32Array.from(UNIT)]);
  });

  it("returns the re-normalised leading slice at a shorter width", async () => {
    const vector = await embedderOf(standIn()).embed("text", { dimension: 2 });
    assertEquals([...vector], [1, 0]);
  });

  it("runs concurrent calls one at a time, in call order", async () => {
    const stand = standIn();
    const gate = deferred<void>();
    stand.gate = gate.promise;
    const embedder = embedderOf(stand);

    const first = embedder.embed("a");
    const second = embedder.embed("a b c");
    gate.resolve();
    await Promise.all([first, second]);

    assertEquals(stand.maxInFlight, 1);
    assertEquals(
      stand.runs.map((inputs) => hostInput(inputs, "input_ids").shape[1]),
      [3, 5],
    );
  });

  it("refuses a request without waiting for the call in flight", async () => {
    const stand = standIn();
    stand.gate = new Promise<void>(() => {});
    const embedder = embedderOf(stand);
    void embedder.embed("held");
    await assertRejects(() => embedder.embed([]), ModelInputError, "no content part");
    assertEquals(stand.runs.length, 1);
  });

  it("fails on a non-finite output, naming the element", async () => {
    const stand = standIn();
    stand.output = {
      dtype: "f32",
      shape: [1, 4],
      data: Float32Array.from([0.6, Number.NaN, 0.8, 0]),
    };
    await assertRejects(
      () => embedderOf(stand).embed("text"),
      Error,
      "output element 1 is non-finite (NaN)",
    );
  });

  it("fails on an output that is not [1, width]", async () => {
    const stand = standIn();
    stand.output = { dtype: "f32", shape: [1, 2], data: Float32Array.from([0.6, 0.8]) };
    await assertRejects(
      () => embedderOf(stand).embed("text"),
      Error,
      "graph output 'out' is [1, 2] with 2 elements, not [1, 4]",
    );
  });
});

describe("Embedder: cancellation", () => {
  it("throws an aborted signal's reason as is, without running, and runs the next call", async () => {
    const stand = standIn();
    const embedder = embedderOf(stand);
    const controller = new AbortController();
    const reason = new Error("stop");
    controller.abort(reason);

    const caught = await embedder
      .embed("text", { signal: controller.signal })
      .catch((error: unknown) => error);

    assertStrictEquals(caught, reason);
    assertEquals(stand.runs.length, 0);
    assertEquals((await embedder.embed("text")).length, 4);
  });

  it("cancels a call aborted while it waits for its turn, and lets the call in flight finish", async () => {
    const stand = standIn();
    const gate = deferred<void>();
    stand.gate = gate.promise;
    const embedder = embedderOf(stand);
    const controller = new AbortController();
    const reason = new Error("cancel while queued");

    const first = embedder.embed("first");
    const second = embedder
      .embed("second", { signal: controller.signal })
      .catch((error: unknown) => error);
    controller.abort(reason);
    gate.resolve();

    assertEquals((await first).length, 4);
    assertStrictEquals(await second, reason);
    assertEquals(stand.runs.length, 1);
  });
});

describe("Embedder: dispose", () => {
  it("waits for the call in flight, disposes the Session once, then releases, and refuses later calls", async () => {
    const stand = standIn();
    const gate = deferred<void>();
    stand.gate = gate.promise;
    const embedder = embedderOf(stand);

    const inFlight = embedder.embed("text");
    const disposal = embedder.dispose();
    assertEquals(stand.events, [], "disposed before the run settled");
    gate.resolve();
    assertEquals((await inFlight).length, 4);
    await disposal;

    assertEquals(stand.events, ["session.dispose", "release"]);
    assertStrictEquals(embedder.dispose(), disposal);
    await assertRejects(
      () => embedder.embed("text"),
      Error,
      "embeddinggemma.Pipeline: cannot embed after dispose",
    );
    assertEquals(stand.events, ["session.dispose", "release"]);
  });

  it("releases even when the Session's dispose fails, and reports that failure", async () => {
    const stand = standIn();
    stand.disposeFailure = new Error("device lost");
    await assertRejects(() => embedderOf(stand).dispose(), Error, "device lost");
    assertEquals(stand.events, ["session.dispose", "release"]);
  });
});

/** An embedder over the stand-in with the widths 4 and 2 and one more field replaced. */
const embedderWith = (stand: Stand, patch: Record<string, unknown>): Embedder =>
  new Embedder({
    session: stand.session,
    contract: STAND_CONTRACT,
    rules: embedTextRules(
      TOKENIZER,
      parseEmbeddingGemmaPipelineConfig({ ...CONFIG_JSON, dimensions: [4, 2], ...patch }),
    ),
    deviceLimits: STAND_DEVICE_LIMITS,
  });

describe("Embedder.tokenize: the ids embed runs, without running", () => {
  it("returns the ids the Session receives for the same input and prompt", async () => {
    const stand = standIn();
    const embedder = embedderOf(stand);
    const { text } = parityCase("english-long");
    const half = Math.floor(text.length / 2);
    const parts = [
      { type: "text", text: text.slice(0, half) },
      { type: "text", text: text.slice(half) },
    ] as const;

    const counted = [embedder.tokenize(text), embedder.tokenize(parts, { prompt: "document" })];
    await embedder.embed(text);
    await embedder.embed(parts, { prompt: "document" });

    assertEquals(
      counted.map((ids) => [...ids]),
      stand.runs.map((inputs) => [...hostInput(inputs, "input_ids").data]),
    );
  });

  it("puts the prompt's prefix in front, <bos> … <eos> around (upstream's ids)", () => {
    const { text, idsWithSpecials } = parityCase("prompt-query");
    const body = text.slice("task: search result | query: ".length);
    const ids = embedderOf(standIn()).tokenize(body, { prompt: "query" });
    assertEquals([...ids], idsWithSpecials);
    assert(ids instanceof Int32Array);
  });

  it("returns a sequence over the limit that embed refuses, so the count can be shown first", async () => {
    const { text, idsWithSpecials } = parityCase("english");
    const stand = standIn();
    const embedder = embedderWith(stand, { maxTokens: idsWithSpecials.length - 1 });

    assertEquals([...embedder.tokenize(text)], idsWithSpecials);
    await assertRejects(() => embedder.embed(text), ModelInputError, "over this model's limit");
    assertEquals(stand.runs.length, 0);
  });

  it("returns a text holding a placeholder id that embed refuses", async () => {
    const embedder = embedderOf(standIn());
    const ids = [...embedder.tokenize("look at <|image|> here")];
    assert(ids.includes(258880), `no image placeholder id in [${ids.join(", ")}]`);
    await assertRejects(
      () => embedder.embed("look at <|image|> here"),
      ModelInputError,
      "image placeholder",
    );
  });

  it("refuses what cannot be tokenized with embed's ModelInputError", () => {
    // An untyped caller (method parameters are bivariant, so the embedder is assignable).
    const untyped: { tokenize(input: unknown, options?: unknown): unknown } = embedderOf(standIn());
    const tokenize = (input: unknown, options: unknown = {}): unknown =>
      untyped.tokenize(input, options);
    assertRefused(() => tokenize(42), "the input is neither a string nor an array");
    assertRefused(() => tokenize([]), "the input has no content part");
    assertRefused(
      () => tokenize([{ type: "image", image: {} }]),
      'content part 0 has type "image" — this model accepts "text" parts',
    );
    assertRefused(
      () => tokenize("text", { prompt: "Query" }),
      'prompt "Query" is not one of this model\'s prompts',
    );
  });

  it("gives the refusal planEmbedRequest gives, word for word", () => {
    const message = (run: () => unknown): string => assertThrows(run, ModelInputError).message;
    assertEquals(
      message(() => tokenizeEmbedInput(RULES, "text", { prompt: "Query" })),
      message(() => planEmbedRequest(RULES, "text", { prompt: "Query" })),
    );
    assertEquals(
      message(() => tokenizeEmbedInput(RULES, [], {})),
      message(() => planEmbedRequest(RULES, [], {})),
    );
  });

  it("answers while a run holds the Session (it is not on the serialization chain)", async () => {
    const stand = standIn();
    stand.gate = new Promise<void>(() => {});
    const embedder = embedderOf(stand);
    void embedder.embed("held");
    for (let turn = 0; stand.runs.length === 0 && turn < 100; turn += 1) {
      await new Promise((settle) => setTimeout(settle, 0));
    }
    assertEquals(stand.runs.length, 1, "the held call never reached the Session");

    assertEquals([...embedder.tokenize("")], [ASSETS.bosId, ASSETS.eosId]);
  });

  it("fails after dispose, as embed does", async () => {
    const embedder = embedderOf(standIn());
    await embedder.dispose();
    assertThrows(
      () => embedder.tokenize("text"),
      Error,
      "embeddinggemma.Pipeline: cannot tokenize after dispose",
    );
  });
});

// ---- ⑦ fromAssets admission --------------------------------------------------------------------

describe("embeddingGemmaDeviceLimit: the device's binding limit over the bytes per token", () => {
  it("divides the smaller of the binding and the buffer limit", () => {
    const contract = { bytesPerToken: 49152 };
    const MIB = 2 ** 20;
    assertEquals(
      embeddingGemmaDeviceLimit(contract, {
        maxStorageBufferBindingSize: 256 * MIB,
        maxBufferSize: 1024 * MIB,
      }),
      { maxTokens: 5461, bytesPerToken: 49152, bindingBytes: 256 * MIB },
    );
    assertEquals(
      embeddingGemmaDeviceLimit(contract, {
        maxStorageBufferBindingSize: 2047 * MIB,
        maxBufferSize: 256 * MIB,
      }).maxTokens,
      5461,
    );
    assertEquals(STAND_DEVICE_LIMIT.maxTokens, 43690);
  });
});

describe("Embedder: the device's token limit", () => {
  const { text, idsWithSpecials } = parityCase("english");
  const length = idsWithSpecials.length;
  /** A device whose binding holds exactly `tokens` tokens of 49,152 B. */
  const embedderFor = (stand: Stand, tokens: number): Embedder =>
    new Embedder({
      session: stand.session,
      contract: STAND_CONTRACT,
      rules: embedTextRules(TOKENIZER, SMALL),
      deviceLimits: {
        maxStorageBufferBindingSize: tokens * 49152,
        maxBufferSize: 1099511627776,
      },
    });

  it("runs an input at the device's limit", async () => {
    const stand = standIn();
    await embedderFor(stand, length).embed(text);
    assertEquals(stand.runs.length, 1);
  });

  it("refuses an input one over it before the run, naming the count, both limits and the buffer", async () => {
    const stand = standIn();
    const error = await assertRejects(
      () => embedderFor(stand, length - 1).embed(text),
      ModelInputError,
    );
    for (const fragment of [
      `the input is ${length} tokens (<bos> and <eos> included), over this device's limit of ${length - 1}`,
      "the graph's largest intermediate (49152 B per token)",
      `must fit one storage buffer of ${(length - 1) * 49152} B`,
      "nothing is truncated",
    ]) {
      assert(error.message.includes(fragment), `"${fragment}" is not in: ${error.message}`);
    }
    assertEquals(stand.runs.length, 0);
  });

  it("still lets tokenize return the ids (only embed refuses)", () => {
    assertEquals([...embedderFor(standIn(), length - 1).tokenize(text)], idsWithSpecials);
  });
});

/** The policy accepts no execution knob (the distribution's presets declare none). */
describe("EMBEDDINGGEMMA_SESSION_POLICY", () => {
  it("accepts no key", () => {
    assert(Object.values(EMBEDDINGGEMMA_SESSION_POLICY).every((accepted) => !accepted));
  });
});

/** Declaration-only `model` (the contract's shapes at small sizes: V 8, D 4, a T bound of 16). */
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

const written = await writeContainer(modelInput());
const COMPONENTS = partAssets("model", written);
const DECLARED = {
  container: {
    descriptor: written.descriptor,
    parts: await Promise.all(
      written.parts.map((bytes, index) => fileRef(`text/model/model.bf16-${index}.krm`, bytes)),
    ),
  },
};
const TOKENIZER_REF = { path: "text/tokenizer/tokenizer.json", size: 2, sha256: "f".repeat(64) };

/** The distribution's manifest skeleton (`patch` overrides fields of `models["text"]`). */
const manifestOf = (patch: Record<string, unknown> = {}) =>
  parseManifest(
    JSON.stringify({
      format: "karume/6",
      generator: "karume/0.1.0",
      defaultModel: "text",
      models: {
        text: {
          pipeline: "embeddinggemma/1",
          weights: { model: { bf16: DECLARED } },
          assets: { tokenizer: TOKENIZER_REF },
          quants: { bf16: { weights: { model: "bf16" }, session: {} } },
          defaultQuant: "bf16",
          pipelineConfig: { ...CONFIG_JSON, dimensions: [4, 2], maxTokens: 16 },
          ...patch,
        },
      },
    }),
  );

/** A shared fake GPU (cannot set up a Session — touching the GPU would fail with another message). */
const fakeGpu = () => fakeGpuContext(fakeDevice());

/** The bytes of a tokenizer asset that is JSON but no asset (the positive path stops on it). */
const NOT_AN_ASSET = new TextEncoder().encode("{}");

describe("Pipeline.fromAssets: admission fails before touching a GPU (the container present)", () => {
  const rejected: readonly [string, Record<string, unknown>, string][] = [
    ["another pipeline", { pipeline: "siglip2/1" }, "the manifest's pipeline is 'siglip2/1'"],
    ["an unknown major", { pipeline: "embeddinggemma/2" }, "has an unsupported major"],
    [
      "a pipelineConfig field it does not know",
      { pipelineConfig: { ...CONFIG_JSON, dimensions: [4, 2], maxTokens: 16, truncate: true } },
      "unknown key 'truncate'",
    ],
    ["no tokenizer asset declaration", { assets: {} }, "have no tokenizer asset 'tokenizer'"],
    [
      "an execution knob in the quant's session",
      { quants: { bf16: { weights: { model: "bf16" }, session: { linearCompute: "a8" } } } },
      "session.linearCompute is not supported",
    ],
    [
      "a declared maximum the graph does not run",
      { pipelineConfig: { ...CONFIG_JSON, dimensions: [4, 2], maxTokens: 2048 } },
      "pipelineConfig.maxTokens is 2048, but the graph runs 'T' up to 16",
    ],
    [
      "a full width other than the graph's",
      { pipelineConfig: { ...CONFIG_JSON, dimensions: [8, 4], maxTokens: 16 } },
      "expected f32 [1, 8]",
    ],
  ];
  for (const [label, patch, message] of rejected) {
    it(label, async () => {
      const error = await assertRejects(
        () =>
          Pipeline.fromAssets(
            { manifest: manifestOf(patch), assets: { ...COMPONENTS, tokenizer: NOT_AN_ASSET } },
            { gpu: fakeGpu() },
          ),
        Error,
        message,
      );
      assert(
        !(error instanceof ModelInputError),
        `an asset mismatch is not an input error: ${label}`,
      );
    });
  }

  it("a misspelt model / quant fails with the list of available ones", async () => {
    const input = { manifest: manifestOf(), assets: { ...COMPONENTS, tokenizer: NOT_AN_ASSET } };
    await assertRejects(
      () => Pipeline.fromAssets(input, { model: "image", gpu: fakeGpu() }),
      Error,
      "model 'image' is not in the manifest (available: text)",
    );
    await assertRejects(
      () => Pipeline.fromAssets(input, { quant: "f32", gpu: fakeGpu() }),
      Error,
      "quant 'f32' is not in the manifest (available: bf16)",
    );
  });

  it("positive path: passes admission and stops at parsing the tokenizer asset, before the GPU", async () => {
    await assertRejects(
      () =>
        Pipeline.fromAssets(
          { manifest: manifestOf(), assets: { ...COMPONENTS, tokenizer: NOT_AN_ASSET } },
          { gpu: fakeGpu() },
        ),
      Error,
      "tokenizer.format: missing",
    );
  });

  it("fails naming the tokenizer asset when its bytes are missing", async () => {
    await assertRejects(
      () => Pipeline.fromAssets({ manifest: manifestOf(), assets: COMPONENTS }, { gpu: fakeGpu() }),
      Error,
      "asset 'tokenizer' is missing",
    );
  });
});

/**
 * The distribution's two presets (`bf16`, the default, and `i8` — each its own storage of `model`,
 * both with an empty session). The i8 declaration reuses the test container's bytes: admission
 * reads declarations only, so the stored dtype never matters before the GPU.
 */
const twoPresetPatch = (i8Session: Record<string, unknown> = {}): Record<string, unknown> => ({
  weights: {
    model: {
      bf16: DECLARED,
      i8: {
        container: {
          descriptor: DECLARED.container.descriptor,
          parts: DECLARED.container.parts.map((part) => ({
            ...part,
            path: part.path.replace(".bf16-", ".i8-"),
          })),
        },
      },
    },
  },
  quants: {
    bf16: { weights: { model: "bf16" }, session: {} },
    i8: { weights: { model: "i8" }, session: i8Session },
  },
});

describe("Pipeline.fromAssets with the distribution's two presets (bf16, i8)", () => {
  const input = (patch: Record<string, unknown>) => ({
    manifest: manifestOf(patch),
    assets: { ...COMPONENTS, tokenizer: NOT_AN_ASSET },
  });

  it("each preset passes admission and stops at parsing the tokenizer asset, before the GPU", async () => {
    for (const quant of [undefined, "bf16", "i8"]) {
      await assertRejects(
        () =>
          Pipeline.fromAssets(input(twoPresetPatch()), {
            gpu: fakeGpu(),
            ...(quant === undefined ? {} : { quant }),
          }),
        Error,
        "tokenizer.format: missing",
      );
    }
  });

  it("a misspelt quant lists both presets", async () => {
    await assertRejects(
      () => Pipeline.fromAssets(input(twoPresetPatch()), { quant: "i4", gpu: fakeGpu() }),
      Error,
      "quant 'i4' is not in the manifest (available: bf16 / i8)",
    );
  });

  it("an execution knob in the i8 preset's session is refused like the default's", async () => {
    await assertRejects(
      () =>
        Pipeline.fromAssets(input(twoPresetPatch({ linearCompute: "a8" })), {
          quant: "i8",
          gpu: fakeGpu(),
        }),
      Error,
      "session.linearCompute is not supported",
    );
  });
});

describe("Pipeline.fromPretrained: there is no default source", () => {
  it("omitting ref (a JS call outside the type check) fails, saying that repo is required", async () => {
    await assertRejects(
      // oxlint-disable-next-line typescript/unbound-method -- Reflect.apply supplies Pipeline as `this`
      () => Reflect.apply(Pipeline.fromPretrained, Pipeline, []),
      Error,
      "embeddinggemma.Pipeline.fromPretrained: repo is required (there is no default source)",
    );
  });
});
