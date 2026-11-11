// `PipelineOptions.vision` (ADR 0144): the declaration, what is fetched and when, the embedder Session
// and what a pipeline with or without it does over a graph with soft inputs. No GPU: a real fetch
// path and a real estimator over small containers, with `createContainerSession` standing in for the
// Sessions (the same rig as `gemma4_estimate_test.ts`).

import { assert, assertEquals, assertRejects, assertThrows } from "@std/assert";
import {
  estimateGraphMemory,
  type RunInputs,
  type RunOutputs,
  type Session,
  type SessionOptions,
} from "@karume/runtime";
import { afterEach, beforeEach, describe, it, vi } from "vite-plus/test";
import { ManifestReferenceError } from "@karume/hub";
import { GpuContext, readAdapterInfo } from "../../runtime/src/gpu/device.ts";
import { PreparedModel } from "../../runtime/src/runtime/executor.ts";
import type { ModelInput } from "../../runtime/tests/helpers/container-write.ts";
import { fakeDevice } from "../../runtime/tests/helpers/fake-gpu.ts";
import { ModelInputError } from "../src/errors.ts";
import { assertGemma4Vision, gemmaEntryName } from "../src/gemma/admission.ts";
import { ChatSession } from "../src/gemma/chat-session.ts";
import { parsePipelineConfig, type PipelineConfig } from "../src/gemma/config.ts";
import { Pipeline, type PipelineOptions } from "../src/gemma/pipeline.ts";
import {
  IMAGE_ROW_ELEMENTS,
  imageRows,
  imageSoftTokens,
  runImageEmbedder,
} from "../src/gemma/vision.ts";
import { Pipeline as QatPipeline } from "../src/gemma/qat-pipeline.ts";
import type { GraphOwner } from "../src/hub/components.ts";
import { parseIrDeclarationValue } from "./helpers/container-fixture.ts";
import {
  fileRef,
  HUB_URL,
  REPO,
  serveContainer,
  serveRepo,
  SHA,
} from "./helpers/container-loading-fixture.ts";
import { stubModel } from "./helpers/stub-model.ts";
import { GELU_MUL_RULE } from "@karume/candidate-rules";

// The pipeline calls the runtime's estimator through the package entry; the spy records the options
// of every call and runs the real estimator (the numbers stay the runtime's own).
vi.mock("@karume/runtime", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@karume/runtime")>();
  return { ...actual, estimateGraphMemory: vi.fn(actual.estimateGraphMemory) };
});

const ENTRY = gemmaEntryName("gemma4");
const HIDDEN = 16;
const VOCAB = 300;
const CHUNK_ROWS = 768;
const CAPACITY = 4096;
const MIB = 1024 * 1024;

const ROPE = {
  sliding_attention: { theta: 10000, headDim: 256, rotaryDim: 256 },
  full_attention: { theta: 1000000, headDim: 512, rotaryDim: 128 },
} as const;

/** The ids of the rig's tokenizer below (`vocab` = specials, 256 bytes, then {@link ADDED}). */
const ADDED = [
  "<turn|>",
  "<|tool_response>",
  "<|channel>",
  "<channel|>",
  "<|image>",
  "<|image|>",
  "<image|>",
] as const;
const addedId = (token: (typeof ADDED)[number]): number => 3 + 256 + ADDED.indexOf(token);

const VISION_DECLARED = {
  softTokenBudget: 280,
  imageTokenId: addedId("<|image|>"),
  beginImageTokenId: addedId("<|image>"),
  endImageTokenId: addedId("<image|>"),
  hiddenSize: HIDDEN,
} as const;

const PIPELINE_CONFIG = {
  chunkLength: CHUNK_ROWS,
  maxChunkLength: CHUNK_ROWS,
  maxPosition: CAPACITY,
  capacity: CAPACITY,
  rope: ROPE,
  chatDialect: "gemma4-unified",
} as const;

describe("pipelineConfig.vision", () => {
  it("parses, and is absent from a declaration without it", () => {
    assertEquals(
      parsePipelineConfig({ ...PIPELINE_CONFIG, vision: VISION_DECLARED }).vision,
      VISION_DECLARED,
    );
    assertEquals(Object.hasOwn(parsePipelineConfig(PIPELINE_CONFIG), "vision"), false);
  });

  const refuses = (vision: unknown, fragment: string): void => {
    assertThrows(() => parsePipelineConfig({ ...PIPELINE_CONFIG, vision }), Error, fragment);
  };

  it("refuses unknown keys, missing keys and values outside the range", () => {
    refuses("x", "not an object");
    refuses({ ...VISION_DECLARED, extra: 1 }, "extra");
    const { hiddenSize: _hidden, ...without } = VISION_DECLARED;
    refuses(without, "hiddenSize: missing");
    refuses({ ...VISION_DECLARED, hiddenSize: 0 }, "not a positive integer");
    refuses({ ...VISION_DECLARED, imageTokenId: -1 }, "not a non-negative integer");
    refuses({ ...VISION_DECLARED, softTokenBudget: 100 }, "not one of 70 / 140 / 280 / 560 / 1120");
  });

  it("refuses three ids that are not distinct", () => {
    refuses(
      { ...VISION_DECLARED, endImageTokenId: VISION_DECLARED.imageTokenId },
      "are not distinct",
    );
  });

  it("refuses a budget whose largest image is longer than any chunk the graph takes", () => {
    refuses(
      { ...VISION_DECLARED, softTokenBudget: 1120 },
      `softTokenBudget 1120: one image can become that many soft tokens, more than maxChunkLength ${CHUNK_ROWS}`,
    );
    // The counterpart: the largest budget that fits is taken.
    assertEquals(
      parsePipelineConfig({
        ...PIPELINE_CONFIG,
        vision: { ...VISION_DECLARED, softTokenBudget: 560 },
      }).vision?.softTokenBudget,
      560,
    );
  });
});

describe("assertGemma4Vision: declaration, product graph and embedder graph are one model", () => {
  const product = (soft = true, width = HIDDEN): GraphOwner =>
    stubModel({
      symbols: ["C", "M", "R"],
      inputs: [
        { name: "input_ids", shape: [1, "M"], dtype: "i32" },
        ...(soft
          ? [
              { name: "soft_embeds", shape: [1, "M", width] },
              { name: "soft_mask", shape: [1, "M", 1], dtype: "bool" as const },
            ]
          : []),
      ],
      outputs: ["logits", "hidden"],
      values: { logits: [1, "R", VOCAB], hidden: [1, "R", HIDDEN] },
    });
  const embedder = (
    override: {
      readonly pixels?: readonly (number | string)[];
      readonly position?: readonly (number | string)[];
      readonly output?: readonly (number | string)[];
    } = {},
  ): GraphOwner =>
    stubModel({
      symbols: ["N"],
      inputs: [
        { name: "pixels", shape: override.pixels ?? [1, "N", IMAGE_ROW_ELEMENTS] },
        { name: "position_x", shape: override.position ?? [1, "N"], dtype: "i32" },
        { name: "position_y", shape: [1, "N"], dtype: "i32" },
      ],
      outputs: ["embeds"],
      values: { embeds: override.output ?? [1, "N", HIDDEN] },
    });
  const config = (vision: PipelineConfig["vision"] = VISION_DECLARED): PipelineConfig =>
    parsePipelineConfig({ ...PIPELINE_CONFIG, ...(vision === undefined ? {} : { vision }) });

  it("passes, returning the declaration and the embedder's output", () => {
    assertEquals(assertGemma4Vision(product(), embedder(), config(), VOCAB, ENTRY), {
      config: VISION_DECLARED,
      output: "embeds",
      symbol: "N",
    });
  });

  const refuses = (
    model: GraphOwner,
    vision: GraphOwner,
    declared: PipelineConfig,
    fragment: string,
    vocabSize = VOCAB,
  ): void => {
    const error = assertThrows(
      () => assertGemma4Vision(model, vision, declared, vocabSize, ENTRY),
      Error,
      fragment,
    );
    assert(error.message.startsWith(`${ENTRY}: `), error.message);
  };

  it("a declaration without vision", () => {
    const declared = parsePipelineConfig(PIPELINE_CONFIG);
    refuses(product(), embedder(), declared, "declares no 'vision'");
  });

  it("a product graph exported before image input", () => {
    refuses(product(false), embedder(), config(), "has no 'soft_embeds'");
  });

  it("widths that disagree", () => {
    refuses(
      product(true, 8),
      embedder(),
      config(),
      "disagrees with pipelineConfig.vision.hiddenSize",
    );
    refuses(product(), embedder({ output: [1, "N", 8] }), config(), "is not [1,N,16]");
  });

  it("a token id outside the vocabulary", () => {
    refuses(
      product(),
      embedder(),
      config(),
      "outside the vocabulary",
      VISION_DECLARED.endImageTokenId,
    );
  });

  it("an embedder graph of another shape", () => {
    refuses(product(), embedder({ pixels: [1, "N", 8] }), config(), "is not f32 [1,<symbol>,6912]");
    refuses(
      product(),
      embedder({ pixels: [1, 4, 6912] }),
      config(),
      "is not f32 [1,<symbol>,6912]",
    );
    refuses(product(), embedder({ position: [1, 4] }), config(), "'position_x' is not i32 [1,N]");
  });
});

describe("runImageEmbedder", () => {
  const image = {
    data: new Uint8Array(96 * 96 * 3).fill(255),
    width: 96,
    height: 96,
  };

  it("splits the positions, passes the rows and copies the output", async () => {
    const rows = imageRows(image, 70);
    const seen: RunInputs[] = [];
    const output = new Float32Array(rows.softTokens * HIDDEN).fill(0.5);
    const embedding = await runImageEmbedder(
      {
        run: (inputs) => {
          seen.push(inputs);
          return Promise.resolve({
            embeds: { dtype: "f32", shape: [1, rows.softTokens, HIDDEN], data: output },
          } satisfies RunOutputs);
        },
      },
      rows,
      "embeds",
      HIDDEN,
    );
    const [inputs] = seen;
    const tensor = (name: string) => {
      const input = inputs[name];
      assert("data" in input, `${name} is resident`);
      return input;
    };
    assertEquals(tensor("pixels").shape, [1, rows.softTokens, IMAGE_ROW_ELEMENTS]);
    assert(tensor("pixels").data === rows.pixels, "the pixels are the preprocessing's own");
    assertEquals(tensor("position_x").dtype, "i32");
    assertEquals(tensor("position_x").shape, [1, rows.softTokens]);
    for (let row = 0; row < rows.softTokens; row += 1) {
      assertEquals(tensor("position_x").data[row], rows.positions[row * 2]);
      assertEquals(tensor("position_y").data[row], rows.positions[row * 2 + 1]);
    }
    assertEquals(embedding.softTokens, rows.softTokens);
    assertEquals([...embedding.rows], [...output]);
    assert(embedding.rows !== output, "a readback lasts until the next run, the rows outlive it");
  });

  it("agrees with imageSoftTokens on the row count", () => {
    for (const [width, height] of [
      [96, 96],
      [832, 480],
      [4000, 10],
    ]) {
      const source = { data: new Uint8Array(width * height * 3), width, height };
      assertEquals(imageRows(source).softTokens, imageSoftTokens(source));
    }
  });

  it("fails loudly on another shape or a missing output", async () => {
    const rows = imageRows(image, 70);
    const runs = (outputs: RunOutputs) => ({ run: () => Promise.resolve(outputs) });
    await assertRejects(
      () => runImageEmbedder(runs({}), rows, "embeds", HIDDEN),
      Error,
      "output 'embeds' is missing",
    );
    await assertRejects(
      () =>
        runImageEmbedder(
          runs({
            embeds: { dtype: "f32", shape: [1, 1, HIDDEN], data: new Float32Array(HIDDEN) },
          }),
          rows,
          "embeds",
          HIDDEN,
        ),
      Error,
      "returned f32 [1,1,16]",
    );
  });
});

describe("gemma.Pipeline vision", () => {
  const rope = (name: string, width: number) => ({ name, dtype: "f32", shape: [1, "M", width] });

  /** A product graph that passes admission, with or without the soft inputs. */
  const productComponent = (soft: boolean): ModelInput => ({
    graphs: {
      model: parseIrDeclarationValue({
        format: "karume-ir",
        version: 2,
        requires: { ops: ["state_append", "reshape", "embedding"] },
        symbols: ["C", "M", "R"],
        inputs: [
          { name: "input_ids", dtype: "i32", shape: [1, "M"] },
          ...(soft
            ? [
                { name: "soft_embeds", dtype: "f32", shape: [1, "M", HIDDEN] },
                { name: "soft_mask", dtype: "bool", shape: [1, "M", 1] },
              ]
            : []),
          rope("rope_sliding_attention_cos", 256),
          rope("rope_sliding_attention_sin", 256),
          rope("rope_full_attention_cos", 512),
          rope("rope_full_attention_sin", 512),
          { name: "last_row", dtype: "i32", shape: ["R"] },
        ],
        outputs: ["logits", "hidden"],
        initializers: { table: {}, hidden_table: {} },
        values: {
          table: { dtype: "f32", shape: [8, VOCAB] },
          hidden_table: { dtype: "f32", shape: [8, HIDDEN] },
          k: { dtype: "f32", shape: [1, 8, "M", 32] },
          logits_rows: { dtype: "f32", shape: ["R", VOCAB] },
          hidden_rows: { dtype: "f32", shape: ["R", HIDDEN] },
          logits: { dtype: "f32", shape: [1, "R", VOCAB] },
          hidden: { dtype: "f32", shape: [1, "R", HIDDEN] },
        },
        states: { kslot: { dtype: "f32", shape: [1, 8, "C", 32] } },
        nodes: [
          { op: "reshape", ins: ["rope_sliding_attention_cos"], outs: ["k"], attrs: {} },
          {
            op: "state_append",
            ins: ["k"],
            outs: [],
            attrs: {},
            states: { slot: "kslot" },
          },
          {
            op: "embedding",
            ins: ["table", "last_row"],
            outs: ["logits_rows"],
            attrs: { padding_idx: -1 },
          },
          {
            op: "embedding",
            ins: ["hidden_table", "last_row"],
            outs: ["hidden_rows"],
            attrs: { padding_idx: -1 },
          },
          { op: "reshape", ins: ["logits_rows"], outs: ["logits"], attrs: {} },
          { op: "reshape", ins: ["hidden_rows"], outs: ["hidden"], attrs: {} },
        ],
      }),
    },
    consts: [],
    weights: [
      {
        graph: "model",
        initializer: "table",
        bytes: new Uint8Array(new Float32Array(8 * VOCAB).buffer),
        encoding: { codec: "f32" as const },
      },
      {
        graph: "model",
        initializer: "hidden_table",
        bytes: new Uint8Array(new Float32Array(8 * HIDDEN).buffer),
        encoding: { codec: "f32" as const },
      },
    ],
    assets: [],
    provenance: { license: "test" },
  });

  const VISION_WEIGHT_ELEMENTS = HIDDEN * IMAGE_ROW_ELEMENTS;
  const visionComponent = (): ModelInput => ({
    graphs: {
      vision: parseIrDeclarationValue({
        format: "karume-ir",
        version: 2,
        requires: { ops: ["linear"] },
        symbols: ["N"],
        inputs: [
          { name: "pixels", dtype: "f32", shape: [1, "N", IMAGE_ROW_ELEMENTS] },
          { name: "position_x", dtype: "i32", shape: [1, "N"] },
          { name: "position_y", dtype: "i32", shape: [1, "N"] },
        ],
        outputs: ["embeds"],
        initializers: { w: {}, b: {} },
        values: {
          w: { dtype: "f32", shape: [HIDDEN, IMAGE_ROW_ELEMENTS] },
          b: { dtype: "f32", shape: [HIDDEN] },
          embeds: { dtype: "f32", shape: [1, "N", HIDDEN] },
        },
        nodes: [{ op: "linear", ins: ["pixels", "w", "b"], outs: ["embeds"], attrs: {} }],
      }),
    },
    consts: [],
    weights: [
      {
        graph: "vision",
        initializer: "w",
        bytes: new Uint8Array(new Float32Array(VISION_WEIGHT_ELEMENTS).buffer),
        encoding: { codec: "f32" as const },
      },
      {
        graph: "vision",
        initializer: "b",
        bytes: new Uint8Array(new Float32Array(HIDDEN).buffer),
        encoding: { codec: "f32" as const },
      },
    ],
    assets: [],
    provenance: { license: "test" },
  });

  /** A compiled tokenizer asset with {@link ADDED} and the template's spellings as bytes. */
  const tokenizerBytes = (): Uint8Array<ArrayBuffer> => {
    const byteToken = (byte: number): string =>
      `<0x${byte.toString(16).toUpperCase().padStart(2, "0")}>`;
    const vocab = [
      "<unk>",
      "<bos>",
      "<eos>",
      ...Array.from({ length: 256 }, (_, byte) => byteToken(byte)),
      ...ADDED,
    ];
    const bytes = new TextEncoder().encode(
      JSON.stringify({
        format: "karume-gemma-tokenizer/1",
        spec: {
          normalizer: "replace-space-with-metaspace",
          preTokenizer: "split-space-merged-with-previous",
          decoder: "metaspace-byte-fallback-fuse",
          postProcessor: "none",
        },
        vocab,
        mergesText: "",
        mergesCount: 0,
        byteIds: Array.from({ length: 256 }, (_, byte) => vocab.indexOf(byteToken(byte))),
        addedTokens: [
          ["<unk>", 0],
          ["<bos>", 1],
          ["<eos>", 2],
          ...ADDED.map((token) => [token, vocab.indexOf(token)]),
        ],
        specialIds: [0, 1, 2],
        unkId: 0,
        bosId: 1,
        eosId: 2,
      }),
    );
    const out = new Uint8Array(new ArrayBuffer(bytes.byteLength));
    out.set(bytes);
    return out;
  };

  const gpuContext = (): GpuContext =>
    new GpuContext(
      fakeDevice(),
      readAdapterInfo({}),
      {
        maxBufferSize: 2048 * MIB,
        maxStorageBufferBindingSize: 2048 * MIB,
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

  /** What the Sessions saw. */
  type Runs = {
    readonly built: { readonly graph: string; readonly options: SessionOptions }[];
    readonly embedder: RunInputs[];
    readonly target: {
      readonly inputs: RunInputs;
      readonly generation: {
        readonly queryLength: number;
        readonly bidirectionalRows?: { readonly start: number; readonly end: number };
      };
    }[];
    readonly disposed: string[];
  };
  let runs: Runs;

  beforeEach(() => {
    const seen: Runs = { built: [], embedder: [], target: [], disposed: [] };
    runs = seen;
    vi.spyOn(PreparedModel.prototype, "createContainerSession").mockImplementation(function (
      this: PreparedModel,
      _gpu,
      options = {},
    ) {
      const isVision = this.graph.outputs.includes("embeds");
      seen.built.push({ graph: isVision ? "vision" : "model", options });
      if (isVision) {
        return Promise.resolve({
          dispose: () => {
            seen.disposed.push("vision");
            return Promise.resolve();
          },
          run: (inputs: RunInputs) => {
            seen.embedder.push(inputs);
            const pixels = inputs["pixels"];
            assert("shape" in pixels);
            const rows = pixels.shape[1];
            return Promise.resolve({
              embeds: {
                dtype: "f32",
                shape: [1, rows, HIDDEN],
                data: new Float32Array(rows * HIDDEN).fill(seen.embedder.length),
              },
            } satisfies RunOutputs);
          },
        } as unknown as Session);
      }
      let pastLength = 0;
      const context = {
        get pastLength() {
          return pastLength;
        },
        pendingCommit: undefined,
        slidingSlack: undefined,
        commit: () => undefined,
        fork: () => Promise.resolve(context),
        dispose: () => Promise.resolve(),
      };
      return Promise.resolve({
        exportWeight: () => ({}),
        diagnostics: () => ({}),
        dispose: () => {
          seen.disposed.push("model");
          return Promise.resolve();
        },
        createGenerationContext: () => Promise.resolve(context),
        run: (
          inputs: RunInputs,
          _bindings: unknown,
          generation: Runs["target"][number]["generation"],
        ) => {
          seen.target.push({ inputs, generation });
          pastLength += generation.queryLength;
          const logits = new Float32Array(VOCAB);
          // Always the closing token: a turn is one draw.
          logits[addedId("<turn|>")] = 10;
          return Promise.resolve({
            logits: { dtype: "f32", shape: [1, 1, VOCAB], data: logits },
            hidden: { dtype: "f32", shape: [1, 1, HIDDEN], data: new Float32Array(HIDDEN) },
          } satisfies RunOutputs);
        },
      } as unknown as Session);
    });
  });
  afterEach(() => {
    vi.restoreAllMocks();
  });

  const BLOCKS = { partBytes: 2 * MIB, blockBytes: MIB };

  /** A distribution: `role` says whether it ships the embedder, `soft` the product graph's inputs. */
  const distribution = async (
    options: {
      readonly role?: boolean;
      readonly soft?: boolean;
      readonly declared?: boolean;
    } = {},
  ) => {
    const { role = true, soft = true, declared = true } = options;
    const model = await serveContainer("model/model.f32", productComponent(soft), BLOCKS);
    const vision = await serveContainer("vision/model.bf16", visionComponent(), BLOCKS);
    const tokenizer = tokenizerBytes();
    const rig = await serveRepo(
      {
        test: {
          pipeline: "gemma4/1",
          weights: {
            model: { f32: model.entry },
            ...(role ? { vision: { bf16: vision.entry } } : {}),
          },
          assets: { tokenizer: await fileRef("tokenizer.json", tokenizer) },
          quants: {
            f32: { weights: { model: "f32", ...(role ? { vision: "bf16" } : {}) }, session: {} },
          },
          defaultQuant: "f32",
          pipelineConfig: {
            ...PIPELINE_CONFIG,
            ...(declared ? { vision: VISION_DECLARED } : {}),
          },
        },
      },
      new Map([...model.files, ...vision.files, ["tokenizer.json", tokenizer]]),
      { model: model.parts, ...(role ? { vision: vision.parts } : {}) },
    );
    const open = (pipelineOptions: PipelineOptions) =>
      Pipeline.fromPretrained(
        { repo: REPO, revision: SHA, hubUrl: HUB_URL },
        {
          ...pipelineOptions,
          fetch: rig.mock.fetch,
          caches: rig.hubOptions.caches,
          gpu: gpuContext(),
          // Without a diagnostics callback the small-output decode would build its GPU resources.
          onRunDiagnostics: () => undefined,
        },
      );
    const partsOf = (parts: readonly { readonly path: string }[]) => parts.map((part) => part.path);
    return {
      open,
      fetched: () => rig.mock.paths,
      modelWeightParts: partsOf(model.parts).slice(1),
      visionParts: partsOf(vision.parts),
    };
  };

  const softOf = (inputs: RunInputs) => {
    const embeds = inputs["soft_embeds"];
    const mask = inputs["soft_mask"];
    assert("data" in embeds && "data" in mask, "no soft inputs on this run");
    return { embeds, mask };
  };

  const IMAGE = { data: new Uint8Array(96 * 96 * 3).fill(128), width: 96, height: 96 };

  describe("without the option", () => {
    it("fetches and builds nothing of the embedder, and refuses embedImage", async () => {
      const rig = await distribution();
      await using pipeline = await rig.open({});
      assertEquals(
        rig.visionParts.filter((path) => rig.fetched().includes(path)),
        [],
      );
      assertEquals(
        runs.built.map(({ graph }) => graph),
        ["model"],
      );
      assertEquals(pipeline.vision, undefined);
      await assertRejects(() => pipeline.embedImage(IMAGE), ModelInputError, "vision: true");
    });

    it("runs a text prompt over the new graph, binding zeros and false on every run", async () => {
      const rig = await distribution();
      await using pipeline = await rig.open({});
      const sequence = await pipeline.sequence();
      const stream = sequence.generate({ prompt: [1, 3, 4], maxNewTokens: 1 });
      for await (const _event of stream) continue;
      assertEquals((await stream.done).reason, "eos");
      assertEquals(runs.target.length, 1);
      const { embeds, mask } = softOf(runs.target[0].inputs);
      assertEquals(embeds.shape, [1, 4, HIDDEN]);
      assertEquals(mask.shape, [1, 4, 1]);
      assert(embeds.data.every((value) => value === 0));
      assert(mask.data.every((value) => value === 0));
      assertEquals(runs.target[0].generation.bidirectionalRows, undefined);
    });

    it("refuses content parts in chat and in a session, naming the option", async () => {
      const rig = await distribution();
      await using pipeline = await rig.open({});
      assertThrows(
        () =>
          pipeline.chat([{ role: "user", content: [{ type: "image", image: IMAGE }] }], {
            maxNewTokens: 1,
          }),
        ModelInputError,
        "vision: true",
      );
      await using session = new ChatSession(pipeline, { maxNewTokens: 1 });
      assertThrows(
        () => session.send([{ type: "image", image: IMAGE }]),
        ModelInputError,
        "vision",
      );
    });
  });

  describe("a graph without soft inputs", () => {
    it("binds nothing: E2B / E4B run what they ran", async () => {
      const rig = await distribution({ soft: false, role: false, declared: false });
      await using pipeline = await rig.open({});
      const sequence = await pipeline.sequence();
      const stream = sequence.generate({ prompt: [1, 3], maxNewTokens: 1 });
      for await (const _event of stream) continue;
      assertEquals(Object.hasOwn(runs.target[0].inputs, "soft_embeds"), false);
      assertEquals(Object.hasOwn(runs.target[0].inputs, "soft_mask"), false);
    });
  });

  describe("with the option", () => {
    it("fetches the role, builds one embedder Session and passes it the host policies", async () => {
      const rig = await distribution();
      await using pipeline = await rig.open({ vision: true, planBackingBudgetBytes: 8 * MIB });
      assert(rig.fetched().includes(rig.visionParts[0]), "the descriptor of the role");
      assertEquals(
        runs.built.map(({ graph }) => graph),
        ["model", "vision"],
      );
      const [target, vision] = runs.built;
      assertEquals(vision.options.submitPolicy, target.options.submitPolicy);
      assertEquals(vision.options.planBackingBudgetBytes, 8 * MIB);
      assertEquals(pipeline.vision, VISION_DECLARED);
    });

    it("embedImage returns the rows of the image, serialized and copied", async () => {
      const rig = await distribution();
      await using pipeline = await rig.open({ vision: true });
      const embedding = await pipeline.embedImage(IMAGE);
      const softTokens = imageSoftTokens(IMAGE, 280);
      assertEquals(embedding.softTokens, softTokens);
      assertEquals(embedding.rows.length, softTokens * HIDDEN);
      assertEquals(runs.embedder.length, 1);
      const pixels = runs.embedder[0]["pixels"];
      assert("shape" in pixels);
      assertEquals(pixels.shape, [1, softTokens, IMAGE_ROW_ELEMENTS]);
      assert(embedding.rows.every((value) => value === 1));
    });

    it("embedImage refuses an invalid image at the call", async () => {
      const rig = await distribution();
      await using pipeline = await rig.open({ vision: true });
      await assertRejects(
        () => pipeline.embedImage({ data: new Uint8Array(3), width: 2, height: 2 }),
        ModelInputError,
        "does not match",
      );
      assertEquals(runs.embedder, []);
    });

    it("sequence with softBlocks: the block's rows and range reach the run", async () => {
      const rig = await distribution();
      await using pipeline = await rig.open({ vision: true });
      const { rows, softTokens } = await pipeline.embedImage(IMAGE);
      const prompt = [
        1,
        VISION_DECLARED.beginImageTokenId,
        ...Array.from({ length: softTokens }, () => VISION_DECLARED.imageTokenId),
        VISION_DECLARED.endImageTokenId,
        3,
      ];
      const sequence = await pipeline.sequence();
      const stream = sequence.generate({
        prompt,
        maxNewTokens: 1,
        softBlocks: [{ start: 2, rows }],
      });
      for await (const _event of stream) continue;
      const [run] = runs.target;
      assertEquals(run.generation.bidirectionalRows, { start: 2, end: 2 + softTokens });
      const { mask } = softOf(run.inputs);
      assertEquals(mask.data[1], 0);
      assertEquals(mask.data[2], 1);
      assertEquals(mask.data[1 + softTokens], 1);
      assertEquals(mask.data[2 + softTokens], 0);
    });

    it("chat embeds at the turn, and places the block", async () => {
      const rig = await distribution();
      await using pipeline = await rig.open({ vision: true });
      const stream = pipeline.chat(
        [
          {
            role: "user",
            content: [
              { type: "image", image: IMAGE },
              { type: "text", text: "Hi" },
            ],
          },
        ],
        { maxNewTokens: 1 },
      );
      assertEquals(runs.embedder.length, 0, "the embedder ran before the stream was drained");
      await stream.text();
      assertEquals(runs.embedder.length, 1);
      const block = runs.target[0].generation.bidirectionalRows;
      assert(block !== undefined, "the run has no block");
      assertEquals(block.end - block.start, imageSoftTokens(IMAGE, 280));
    });

    it("chat reads the pixels at issue", async () => {
      const rig = await distribution();
      await using pipeline = await rig.open({ vision: true });
      const image = { ...IMAGE, data: IMAGE.data.slice() };
      const stream = pipeline.chat([{ role: "user", content: [{ type: "image", image }] }], {
        maxNewTokens: 1,
      });
      image.data.fill(0);
      await stream.text();
      const pixels = runs.embedder[0]["pixels"];
      assert("data" in pixels && pixels.dtype === "f32");
      assert(
        pixels.data.every((value) => value > 0),
        "the rewrite reached the turn",
      );
    });

    it("a ChatSession over the pipeline embeds through it", async () => {
      const rig = await distribution();
      await using pipeline = await rig.open({ vision: true });
      await using session = new ChatSession(pipeline, { maxNewTokens: 1 });
      await session.send([{ type: "image", image: IMAGE }]).text();
      await session.send("Next.").text();
      assertEquals(runs.embedder.length, 1);
      assert(runs.target[0].generation.bidirectionalRows !== undefined);
      assertEquals(runs.target.at(-1)?.generation.bidirectionalRows, undefined);
    });

    it("estimateSessionMemory counts the embedder: its weights and its peak", async () => {
      const rig = await distribution();
      await using plain = await rig.open({});
      const without = plain.estimateSessionMemory();
      await using pipeline = await rig.open({ vision: true });
      const report = pipeline.estimateSessionMemory();
      assertEquals(
        report.resident.weights.totalBytes - without.resident.weights.totalBytes,
        (VISION_WEIGHT_ELEMENTS + HIDDEN) * 4,
      );
      // The inputs of an image at the budget: pixels, the two position columns and the output.
      const inputBytes = 280 * IMAGE_ROW_ELEMENTS * 4 + 2 * 280 * 4;
      assert(
        report.peakAccountedBytes - without.peakAccountedBytes >=
          VISION_WEIGHT_ELEMENTS * 4 + inputBytes,
        `${report.peakAccountedBytes} - ${without.peakAccountedBytes}`,
      );
      assertEquals(report.scenarios, without.scenarios);
    });

    it("disposes the embedder with the pipeline, before the target", async () => {
      const rig = await distribution();
      const pipeline = await rig.open({ vision: true });
      await pipeline.dispose();
      assertEquals(runs.disposed, ["vision", "model"]);
      await assertRejects(() => pipeline.embedImage(IMAGE), Error, "after dispose");
    });

    // The exclusion half of the experimental `fusionRules` (its forwarding half is in
    // gemma4_estimate_test.ts): the rules are for the Sessions that run the target's graph.
    it("fusionRules reach the target Session and not the embedder Session", async () => {
      const rig = await distribution();
      const fusionRules = [GELU_MUL_RULE];
      await using _pipeline = await rig.open({ vision: true, fusionRules });
      assertEquals(
        runs.built.map(({ graph }) => graph),
        ["model", "vision"],
      );
      const [target, vision] = runs.built;
      assertEquals(target.options.fusionRules, fusionRules);
      assertEquals(Object.hasOwn(vision.options, "fusionRules"), false);
    });

    // The estimate's half: each graph is estimated with the fusion options its Session got.
    it("estimateSessionMemory gives the rules and the row-block limit to the target's estimate and not the embedder's", async () => {
      const rig = await distribution();
      const fusionRules = [GELU_MUL_RULE];
      await using pipeline = await rig.open({
        vision: true,
        fusionRules,
        attentionRowBlockMaxBytes: 64 * MIB,
      });
      const spy = vi.mocked(estimateGraphMemory);
      const before = spy.mock.calls.length;
      pipeline.estimateSessionMemory();
      const [target, embedder, ...rest] = spy.mock.calls.slice(before).map(([, , options]) => {
        assert(options !== undefined);
        return options;
      });
      assertEquals(rest, [], "one estimate per graph: the target's, then the embedder's");
      assertEquals(target.fusionRules, fusionRules);
      assertEquals(target.attentionRowBlockMaxBytes, 64 * MIB);
      const [targetSession, visionSession] = runs.built;
      for (const key of ["fusionRules", "attentionRowBlockMaxBytes"] as const) {
        assertEquals(Object.hasOwn(visionSession.options, key), false, `${key} on the Session`);
        assertEquals(Object.hasOwn(embedder, key), false, `${key} on the estimate`);
        assertEquals(target[key], targetSession.options[key]);
      }
      // Both graphs are estimated with the device's limits (the rig's fake device).
      for (const options of [target, embedder]) {
        assertEquals(options.maxStorageBufferBindingSize, 2048 * MIB);
        assertEquals(options.maxComputeWorkgroupsPerDimension, 65535);
      }
    });
  });

  describe("what is refused, before any weight part is fetched", () => {
    it("a distribution without the role", async () => {
      const rig = await distribution({ role: false });
      await assertRejects(() => rig.open({ vision: true }), ManifestReferenceError, "vision");
      assertEquals(
        rig.modelWeightParts.filter((path) => rig.fetched().includes(path)),
        [],
      );
    });

    it("a product graph without soft inputs", async () => {
      const rig = await distribution({ soft: false });
      await assertRejects(() => rig.open({ vision: true }), Error, "has no 'soft_embeds'");
      assertEquals(
        rig.modelWeightParts.filter((path) => rig.fetched().includes(path)),
        [],
      );
    });

    it("a chunkLength shorter than the largest image of the budget, naming both", async () => {
      const rig = await distribution();
      await assertRejects(
        () => rig.open({ vision: true, chunkLength: 256 }),
        ModelInputError,
        `${ENTRY}: chunkLength 256 is below the 280 soft tokens one image can become under ` +
          "pipelineConfig.vision.softTokenBudget 280",
      );
      assertEquals(
        rig.modelWeightParts.filter((path) => rig.fetched().includes(path)),
        [],
      );
    });

    it("a chunkLength of the budget itself is taken, and a shorter one without vision", async () => {
      const rig = await distribution();
      await using exact = await rig.open({ vision: true, chunkLength: 280 });
      assertEquals(exact.program.chunkLength, 280);
      await using plain = await rig.open({ chunkLength: 256 });
      assertEquals(plain.program.chunkLength, 256);
    });

    it("a declaration without pipelineConfig.vision", async () => {
      const rig = await distribution({ declared: false });
      await assertRejects(() => rig.open({ vision: true }), Error, "declares no 'vision'");
    });

    it("a value that is not a boolean", async () => {
      const rig = await distribution();
      await assertRejects(
        () => rig.open({ vision: "yes" as unknown as boolean }),
        ModelInputError,
        'vision "yes" is not a boolean',
      );
    });

    it("the mobile QAT entry", async () => {
      await assertRejects(
        () =>
          QatPipeline.fromPretrained({ repo: REPO, revision: SHA, hubUrl: HUB_URL }, {
            vision: true,
          } as never),
        Error,
        "vision is not supported for the mobile QAT line",
      );
    });

    it("fromAssets, which has no place for it", async () => {
      await assertRejects(
        () =>
          Pipeline.fromAssets(
            {
              config: parsePipelineConfig(PIPELINE_CONFIG),
              model: [new Uint8Array(new ArrayBuffer(1))],
              tokenizer: new Uint8Array(new ArrayBuffer(1)),
            },
            { vision: true },
          ),
        Error,
        "vision is not accepted",
      );
    });
  });
});
