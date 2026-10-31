/**
 * MOSS-Image's prompt form on the host (ADR 0138 decision 9, ADR 0137 decision 4): the request
 * forms of the one acceptance check, the text-encoder stage's lifetime and headroom, and its place
 * in the generate path — with stand-ins for the GPU and the Sessions (no device, no weights).
 *
 * - **The request forms** (`planMossGenerate`): a prompt string goes through the chat template and
 *   the tokenizer before any GPU work (the ids are the fixture's — transformers' own for the
 *   rendered prompt, so the template is applied); one form for both branches; no default negative;
 *   the tokenizer's length policy (ruling 50) through `overflow`. The tokenizer is the real one over
 *   the fixture's tables (`fixtures/qwen3_5/tokenizer-cases.json` — the subset its prompts need).
 * - **The stage** (`encodeMossPrompts`): its own headroom check before the encoder opens, the
 *   positive then the negative encoding, and the encoder disposed and its memory released on every
 *   path (success, a failed run, a cancellation).
 * - **The generate path** (`generateMoss`): the text encoder is disposed before `dit_text` opens,
 *   a failing encoding opens no `dit_text`, and the first headroom check takes the text encoder's
 *   need in the prompt form only. The DiT side is a stand-in that stops at `dit_text`'s Session.
 *
 * The device is a real `GpuContext` over a stand-in device whose trial allocations (`fitsHeadroom`)
 * fit up to a set capacity; each release wait (`settleReleasedMemory`'s empty submit) is logged.
 */

import { readTextFile } from "@karume/test-support/fs";
import { describe, expect, it } from "vite-plus/test";
import { assert, assertEquals, assertRejects, assertThrows } from "@std/assert";
import type { RunOutputs, Tensor } from "@karume/runtime";
import { HEADROOM_MARGIN_BYTES } from "../src/anima/residency.ts";
import { ModelInputError } from "../src/errors.ts";
import type { GraphOwner, ModelComponent } from "../src/hub/components.ts";
import type { MossDitContract } from "../src/moss/dit-contract.ts";
import {
  generateMoss,
  type MossGenerateLimits,
  type MossGenerateRequest,
  type MossGenerateState,
  planMossGenerate,
} from "../src/moss/generate.ts";
import { MOSS_REFERENCE_PRESET, mossPreset } from "../src/moss/preset.ts";
import type { MossTextEmbeddings } from "../src/moss/request.ts";
import {
  encodeMossPrompts,
  MOSS_TEXT_ENCODER,
  type MossTextEncoderComponent,
  mossTextEncoderContract,
  mossTextEncoderStageNeed,
  type MossTextEncoderState,
} from "../src/moss/text-stage.ts";
import type { MossVaeContract } from "../src/moss/vae-contract.ts";
import type { Qwen3_5Session } from "../src/qwen3_5/encoder.ts";
import { QWEN3_5_ROTARY_DIM } from "../src/qwen3_5/rope.ts";
import { Qwen3_5Tokenizer } from "../src/qwen3_5/tokenizer.ts";
import { parseCodeRanges } from "../src/text/code-ranges.ts";
import { GpuContext, readAdapterInfo, type RequiredLimits } from "../../runtime/src/gpu/device.ts";
import { fakeDevice } from "../../runtime/tests/helpers/fake-gpu.ts";
import { syntheticTrainingGrid } from "./helpers/moss-dit.ts";

// --- The tokenizer over the fixture's tables -------------------------------------------------

type PromptCase = {
  readonly id: string;
  readonly prompt: string;
  readonly ids?: number[];
  readonly tokens?: number;
};

type Fixture = {
  readonly maxLength: number;
  readonly qwen: {
    readonly addedTokens: [string, number][];
    readonly classes: { letter: unknown; number: unknown; space: unknown };
    readonly caseFold: [number, number][];
    readonly nfcSegments: unknown;
    readonly vocab: Record<string, number>;
    readonly merges: [string, string, number][];
  };
  readonly promptCases: PromptCase[];
};

const fixture = JSON.parse(
  await readTextFile(new URL("./fixtures/qwen3_5/tokenizer-cases.json", import.meta.url)),
) as Fixture;

const tokenizer = new Qwen3_5Tokenizer({
  vocab: new Map(Object.entries(fixture.qwen.vocab)),
  merges: new Map(fixture.qwen.merges.map(([left, right, rank]) => [`${left} ${right}`, rank])),
  addedTokens: new Map(fixture.qwen.addedTokens),
  caseFold: new Map(fixture.qwen.caseFold),
  nfcSegments: parseCodeRanges(fixture.qwen.nfcSegments, "nfcSegments"),
  classes: {
    letter: parseCodeRanges(fixture.qwen.classes.letter, "letter"),
    number: parseCodeRanges(fixture.qwen.classes.number, "number"),
    space: parseCodeRanges(fixture.qwen.classes.space, "space"),
  },
  maxLength: fixture.maxLength,
});

/** A fixture prompt case with upstream's ids (rendered with the chat template, then tokenized). */
const promptCase = (id: string) => {
  const found = fixture.promptCases.find((testCase) => testCase.id === id);
  if (found?.ids === undefined || found.tokens === undefined) {
    throw new Error(`the fixture has no accepted prompt case ${id}`);
  }
  return { prompt: found.prompt, ids: found.ids, tokens: found.tokens };
};

/** 29 tokens; the empty prompt (upstream's default negative) is the bare template, 10 tokens. */
const ENGLISH = promptCase("english");
const EMPTY = promptCase("empty");
/** 4,097 tokens rendered (one over the bound); "truncate" keeps upstream's first 4,096 ids. */
const OVER = promptCase("boundary_4097");

// --- Stand-ins -------------------------------------------------------------------------------

type Graph = GraphOwner["graph"];

const TEXT_WIDTH = 4096;
const VOCABULARY = 248_320;
const OUTPUT = "rms_norm_default_104";

/** A declaration-only graph with the recipe's interface (as `qwen3_5_encoder_test.ts`), 4,096 wide. */
const textGraph = (width = TEXT_WIDTH, symbols: readonly string[] = ["T"]): Graph => ({
  format: "karume-ir",
  version: 2,
  requires: { ops: [] },
  symbols: [...symbols],
  inputs: [
    { name: "input_ids", dtype: "i32", shape: [1, "T"] },
    { name: "positions", dtype: "f32", shape: ["T"] },
    { name: "rope_cos", dtype: "f32", shape: [1, "T", QWEN3_5_ROTARY_DIM] },
    { name: "rope_sin", dtype: "f32", shape: [1, "T", QWEN3_5_ROTARY_DIM] },
  ],
  outputs: [OUTPUT],
  initializers: {
    "embed_tokens.weight": { storage: { codec: "int8-sym" } },
    "norm.weight": { storage: { codec: "f32" } },
  },
  values: {
    "embed_tokens.weight": { dtype: "f32", shape: [VOCABULARY, width] },
    embedding: { dtype: "f32", shape: [1, "T", width] },
    [OUTPUT]: { dtype: "f32", shape: [1, "T", width] },
  },
  states: {},
  nodes: [
    {
      op: "embedding",
      ins: ["embed_tokens.weight", "input_ids"],
      outs: ["embedding"],
      attrs: { padding_idx: -1 },
      states: {},
    },
  ],
});

/** The value of every row element the stand-in encoder returns for a sequence of `tokens`. */
const rowValue = (tokens: number): number => tokens / 100;

type TextOptions = {
  /** The estimate's `peakAccountedBytes` (default 1 GiB). */
  readonly peak?: number;
  /** Called on each run before it returns (throw to fail the run). */
  readonly onRun?: (tokens: number) => void;
};

/**
 * The text tower's stand-in: logs `open` / `encode <T>` / `dispose` into `events` (a dispose once
 * it has completed) and each estimate's bindings into `estimates`.
 */
const textComponent = (
  events: string[],
  options: TextOptions = {},
  estimates: Readonly<Record<string, number>>[] = [],
): MossTextEncoderComponent => ({
  graph: textGraph(),
  maxPartBytes: 256 * 2 ** 20,
  estimate: (estimateOptions) => {
    estimates.push(estimateOptions.bindings ?? {});
    return { peakAccountedBytes: options.peak ?? 2 ** 30 };
  },
  createSession: async () => {
    events.push(`open ${MOSS_TEXT_ENCODER}`);
    const session: Qwen3_5Session = {
      run: async (inputs) => {
        const ids = inputs["input_ids"] as Tensor;
        const tokens = ids.shape[1];
        events.push(`encode ${tokens}`);
        options.onRun?.(tokens);
        const output: Tensor = {
          dtype: "f32",
          shape: [1, tokens, TEXT_WIDTH],
          data: new Float32Array(tokens * TEXT_WIDTH).fill(rowValue(tokens)),
        };
        const outputs: RunOutputs = { [OUTPUT]: output };
        return outputs;
      },
      diagnostics: () => {
        throw new Error("the stage reads no diagnostics");
      },
      dispose: async () => {
        // Completes after a macrotask: a dispose the stage did not await would log too late.
        await new Promise((resolve) => setTimeout(resolve, 0));
        events.push(`dispose ${MOSS_TEXT_ENCODER}`);
      },
    };
    return session;
  },
});

const textState = (component: MossTextEncoderComponent): MossTextEncoderState => ({
  component,
  tokenizer,
  contract: mossTextEncoderContract(component, tokenizer),
});

/** A 12 GiB-class device's limits that the stages read (the rest are not read here). */
const DEVICE_LIMITS: RequiredLimits = {
  maxBufferSize: 2_147_483_644,
  maxStorageBufferBindingSize: 2_147_483_644,
  maxUniformBufferBindingSize: 65_536,
  maxStorageBuffersPerShaderStage: 10,
  maxUniformBuffersPerShaderStage: 12,
  maxComputeWorkgroupStorageSize: 49_152,
  maxComputeInvocationsPerWorkgroup: 1024,
  maxComputeWorkgroupSizeX: 1024,
  maxComputeWorkgroupSizeY: 1024,
  maxComputeWorkgroupSizeZ: 64,
  maxComputeWorkgroupsPerDimension: 65_535,
};

/**
 * A real `GpuContext` over a stand-in device: a trial allocation fits while its pieces sum to at
 * most `capacity` bytes; each probe and each release wait (the empty submit) is logged.
 */
const headroomGpu = (events: string[], capacity = Number.POSITIVE_INFINITY): GpuContext => {
  let probed = 0;
  let pops = 0;
  const device = fakeDevice({
    createBuffer: (descriptor) => {
      probed += descriptor.size;
      // DOM types are not reproduced whole (the test-only boundary of `fake-gpu.ts`).
      return { destroy: (): void => undefined } as unknown as GPUBuffer;
    },
    // `popFailureScopes` issues the validation pop, then the out-of-memory pop
    // (`packages/runtime/src/gpu/error-scope.ts`).
    popErrorScope: async () => {
      pops += 1;
      if (pops % 2 === 1) return null;
      const fits = probed <= capacity;
      events.push(`probe ${probed} ${fits ? "fits" : "refused"}`);
      probed = 0;
      return fits ? null : ({ message: "out of memory" } as GPUError);
    },
    submit: () => {
      events.push("release wait");
    },
  });
  return new GpuContext(device, readAdapterInfo({}), DEVICE_LIMITS, new Set());
};

/** The events of Session lifetimes and runs only (probes and release waits left out). */
const lifetimes = (events: readonly string[]): string[] =>
  events.filter((event) => /^(open|dispose|encode) /.test(event));

const embeddingsOf = (tokens: number): MossTextEmbeddings => ({
  data: new Float32Array(tokens * TEXT_WIDTH).fill(0.25),
  shape: [tokens, TEXT_WIDTH],
});

// --- The request forms -----------------------------------------------------------------------

const LIMITS: MossGenerateLimits = {
  dit: { latentChannels: 128, trainingGrid: syntheticTrainingGrid(1000) },
  vae: { bytesPerCell: 49_152, spatialScale: 8 },
  limits: { maxStorageBufferBindingSize: 2_147_483_644, maxBufferSize: 2_147_483_644 },
  text: { tokenizer },
};

const BASE: MossGenerateRequest = {
  width: 256,
  height: 256,
  prompt: ENGLISH.prompt,
  negativePrompt: EMPTY.prompt,
  steps: 2,
  guidance: 3,
  shift: 5,
  seed: 42,
};

/** Any field replaced by any value, outside the types too (what a JavaScript caller can pass). */
type UntypedOverrides = Partial<Record<keyof MossGenerateRequest, unknown>>;

/** NOTE: the cast stands for a caller outside the types (the rejection cases exercise it). */
const plan = (overrides: UntypedOverrides, limits: MossGenerateLimits = LIMITS) =>
  planMossGenerate({ ...BASE, ...overrides } as MossGenerateRequest, limits);

describe("MOSS-Image prompt form: the acceptance check (ADR 0138 decision 9)", () => {
  it("renders and tokenizes both prompts before any GPU work: upstream's ids, their token counts", () => {
    const accepted = plan({});

    assert(accepted.text === "prompts");
    assertEquals([...accepted.prompts.positive], ENGLISH.ids);
    assertEquals([...(accepted.prompts.negative ?? [])], EMPTY.ids);
    expect(accepted.dit.texts.positive.tokens).toBe(29);
    expect(accepted.dit.texts.negative?.tokens).toBe(10);
    expect(accepted.dit.cells).toEqual({ rows: 16, cols: 16 });
  });

  it("encodes no negative at guidance 1", () => {
    const accepted = plan({ guidance: 1, negativePrompt: undefined });

    assert(accepted.text === "prompts");
    expect(accepted.prompts.negative).toBeUndefined();
    expect(accepted.dit.texts.negative).toBeUndefined();
  });

  it('cuts an over-length prompt as upstream does with overflow "truncate"', () => {
    const accepted = plan({ prompt: OVER.prompt, overflow: "truncate" });

    assert(accepted.text === "prompts");
    assertEquals([...accepted.prompts.positive], OVER.ids);
    expect(accepted.dit.texts.positive.tokens).toBe(4096);
  });

  it("keeps the embeddings form as it was, beside the prompt form", () => {
    const accepted = plan({
      prompt: undefined,
      negativePrompt: undefined,
      positive: embeddingsOf(5),
      negative: embeddingsOf(10),
    });

    assert(accepted.text === "embeddings");
    expect(accepted.dit.texts.positive.tokens).toBe(5);
    expect(accepted.dit.texts.negative?.tokens).toBe(10);
  });

  it("rejects each input outside the accepted set with a ModelInputError naming it", () => {
    const cases: readonly [UntypedOverrides, string][] = [
      [
        { positive: embeddingsOf(5) },
        "the text is given both as prompt strings (prompt / negativePrompt) and as precomputed" +
          " embeddings (positive / negative)",
      ],
      [{ negativePrompt: undefined, negative: embeddingsOf(10) }, "given both as prompt strings"],
      [
        { prompt: undefined, negativePrompt: "x", positive: embeddingsOf(5) },
        "given both as prompt strings",
      ],
      [{ prompt: undefined }, "negativePrompt is given without prompt"],
      [
        { prompt: undefined, negativePrompt: undefined },
        "the positive prompt's embeddings are missing and no prompt string is given",
      ],
      [
        { negativePrompt: undefined },
        'guidance 3 runs the negative branch (CFG with block 22 skipped), so negativePrompt is required (upstream\'s default negative is the empty prompt: pass negativePrompt: "")',
      ],
      [
        { guidance: 1 },
        "guidance 1 runs no negative branch, so negativePrompt would have no effect",
      ],
      [{ prompt: 42 }, "MOSS-Image: prompt is not a string"],
      [{ negativePrompt: ["x"] }, "MOSS-Image: negativePrompt is not a string"],
      [
        { prompt: "<tool_response>a result</tool_response>" },
        "MOSS-Image: prompt: wrapped in <tool_response>…</tool_response>",
      ],
      [
        { prompt: OVER.prompt },
        "MOSS-Image: prompt: the token count exceeds the upper bound 4096 — it is not silently truncated",
      ],
      [
        { negativePrompt: OVER.prompt },
        "MOSS-Image: negativePrompt: the token count exceeds the upper bound 4096",
      ],
      [{ prompt: "a\uD800b" }, "MOSS-Image: prompt"],
      [{ overflow: "cut" }, 'unknown overflow policy "cut"'],
      [
        {
          prompt: undefined,
          negativePrompt: undefined,
          positive: embeddingsOf(5),
          negative: embeddingsOf(10),
          overflow: "truncate",
        },
        "overflow applies to prompt strings only",
      ],
    ];
    for (const [overrides, message] of cases) {
      assertThrows(() => plan(overrides), ModelInputError, message);
    }
  });

  it("refuses a prompt-form request without a tokenizer as the caller's assembly, not as input", () => {
    const { text: _text, ...withoutText } = LIMITS;

    const error = assertThrows(
      () => plan({}, withoutText),
      Error,
      "the request gives prompt strings, but no text encoder was passed",
    );
    assert(!(error instanceof ModelInputError));
  });
});

describe("MOSS-Image text encoder admission (mossTextEncoderContract)", () => {
  it("settles the encoder's contract and its one symbolic length", () => {
    const contract = mossTextEncoderContract({ graph: textGraph() }, tokenizer);

    expect(contract).toEqual({
      output: OUTPUT,
      hiddenSize: TEXT_WIDTH,
      vocabularySize: VOCABULARY,
      length: "T",
    });
  });

  it("refuses rows of another width than the DiT reads, another tokenizer bound, two symbols", () => {
    assertThrows(
      () => mossTextEncoderContract({ graph: textGraph(2048) }, tokenizer),
      Error,
      "the text encoder's rows are 2048 wide, but the DiT reads text rows of 4096",
    );
    assertThrows(
      () => mossTextEncoderContract({ graph: textGraph() }, { maxLength: 512 }),
      Error,
      "the tokenizer asset's maxLength 512 differs from the text encoder's bound 4096 or the DiT's 4096",
    );
    assertThrows(
      () => mossTextEncoderContract({ graph: textGraph(TEXT_WIDTH, ["T", "U"]) }, tokenizer),
      Error,
      "the text encoder declares symbols [T, U] (exactly one, T)",
    );
  });
});

// --- The stage -------------------------------------------------------------------------------

/** The ids of the two fixture prompts (the plan's). */
const PROMPTS = {
  positive: Int32Array.from(ENGLISH.ids),
  negative: Int32Array.from(EMPTY.ids),
};

describe("MOSS-Image text-encoder stage (encodeMossPrompts)", () => {
  it("encodes the positive then the negative prompt on one encoder, disposes it, then waits for the release", async () => {
    const events: string[] = [];
    const gpu = headroomGpu(events);
    const { component, contract } = textState(textComponent(events));

    const texts = await encodeMossPrompts({ gpu, component, contract }, PROMPTS);

    expect(lifetimes(events)).toEqual([
      `open ${MOSS_TEXT_ENCODER}`,
      "encode 29",
      "encode 10",
      `dispose ${MOSS_TEXT_ENCODER}`,
    ]);
    expect(events.at(-1)).toBe("release wait");
    expect(texts.positive.tokens).toBe(29);
    expect(texts.positive.embeddings.shape).toEqual([29, TEXT_WIDTH]);
    expect(texts.positive.embeddings.data[0]).toBe(Math.fround(rowValue(29)));
    expect(texts.negative?.embeddings.shape).toEqual([10, TEXT_WIDTH]);
    expect(texts.negative?.embeddings.data[0]).toBe(Math.fround(rowValue(10)));
  });

  it("encodes the positive prompt alone when there is no negative branch", async () => {
    const events: string[] = [];
    const { component, contract } = textState(textComponent(events));

    const texts = await encodeMossPrompts(
      { gpu: headroomGpu(events), component, contract },
      { positive: PROMPTS.positive, negative: undefined },
    );

    expect(lifetimes(events)).toEqual([
      `open ${MOSS_TEXT_ENCODER}`,
      "encode 29",
      `dispose ${MOSS_TEXT_ENCODER}`,
    ]);
    expect(texts.negative).toBeUndefined();
  });

  it("disposes the encoder and waits for the release when a run fails, rethrowing the run's error", async () => {
    const events: string[] = [];
    const failing = textComponent(events, {
      onRun: (tokens) => {
        if (tokens === 10) throw new Error("stand-in run failure");
      },
    });
    const { component, contract } = textState(failing);

    await assertRejects(
      () => encodeMossPrompts({ gpu: headroomGpu(events), component, contract }, PROMPTS),
      Error,
      "stand-in run failure",
    );

    expect(lifetimes(events)).toEqual([
      `open ${MOSS_TEXT_ENCODER}`,
      "encode 29",
      "encode 10",
      `dispose ${MOSS_TEXT_ENCODER}`,
    ]);
    expect(events.at(-1)).toBe("release wait");
  });

  it("disposes the encoder and waits for the release when cancelled between the two encodings", async () => {
    const events: string[] = [];
    const controller = new AbortController();
    const reason = new Error("stand-in cancellation");
    const cancelling = textComponent(events, {
      onRun: (tokens) => {
        if (tokens === 29) controller.abort(reason);
      },
    });
    const { component, contract } = textState(cancelling);

    const error = await assertRejects(() =>
      encodeMossPrompts({ gpu: headroomGpu(events), component, contract }, PROMPTS, {
        signal: controller.signal,
      }),
    );

    expect(error).toBe(reason);
    expect(lifetimes(events)).toEqual([
      `open ${MOSS_TEXT_ENCODER}`,
      "encode 29",
      `dispose ${MOSS_TEXT_ENCODER}`,
    ]);
    expect(events.at(-1)).toBe("release wait");
  });

  it("opens nothing when already cancelled", async () => {
    const events: string[] = [];
    const reason = new Error("stand-in cancellation");
    const { component, contract } = textState(textComponent(events));

    const error = await assertRejects(() =>
      encodeMossPrompts({ gpu: headroomGpu(events), component, contract }, PROMPTS, {
        signal: AbortSignal.abort(reason),
      }),
    );

    expect(error).toBe(reason);
    expect(events).toEqual([]);
  });

  it("refuses before the encoder opens when its need does not fit, naming the stage and the bytes", async () => {
    const events: string[] = [];
    const { component, contract } = textState(textComponent(events));
    const need = mossTextEncoderStageNeed({ limits: DEVICE_LIMITS, component, contract }, PROMPTS);

    await assertRejects(
      () => encodeMossPrompts({ gpu: headroomGpu(events, need - 4), component, contract }, PROMPTS),
      Error,
      `MOSS-Image: the text_encoder stage needs ${need} bytes of VRAM (estimate + largest part +` +
        " margin) that this device cannot allocate now — text of 29 / 10 tokens does not fit",
    );

    expect(lifetimes(events)).toEqual([]);
    // The release of an earlier stage is awaited before the probe.
    expect(events).toEqual(["release wait", `probe ${need} refused`]);
  });

  it("sizes its need at the longer prompt, whichever branch it is: estimate + largest part + margin", () => {
    const estimates: Readonly<Record<string, number>>[] = [];
    const { component, contract } = textState(textComponent([], { peak: 7 * 2 ** 30 }, estimates));
    const input = { limits: DEVICE_LIMITS, component, contract };

    const forward = mossTextEncoderStageNeed(input, PROMPTS);
    const swapped = mossTextEncoderStageNeed(input, {
      positive: PROMPTS.negative,
      negative: PROMPTS.positive,
    });

    expect(estimates).toEqual([{ T: 29 }, { T: 29 }]);
    expect(forward).toBe(7 * 2 ** 30 + 256 * 2 ** 20 + HEADROOM_MARGIN_BYTES);
    expect(swapped).toBe(forward);
  });
});

// --- The generate path -----------------------------------------------------------------------

/** Where the DiT side of the stand-in stops (its `dit_text` Session is never built here). */
const STOP = "stand-in: dit_text does not run on the host";

/**
 * A DiT or VAE component's stand-in: a graph with `symbol`, a fixed estimate; `dit_text`'s Session
 * logs its opening and stops the generation ({@link STOP}).
 *
 * NOTE: the casts of this group — the graphs and contracts hold only what the generate path reads
 * before `dit_text`'s Session opens (the graph's symbol, the estimates, the DiT's latent channels,
 * training grid and width, the VAE's geometry and symbols).
 */
const stageComponent = (
  name: string,
  symbol: string,
  events: string[],
  peak = 2 ** 20,
): ModelComponent => ({
  graph: { symbols: [symbol] } as unknown as Graph,
  createSession: async () => {
    events.push(`open ${name}`);
    throw new Error(STOP);
  },
  assets: {},
  asset: () => {
    throw new Error(`${name}: no asset`);
  },
  estimate: () => ({ peakAccountedBytes: peak }) as ReturnType<ModelComponent["estimate"]>,
  maxPartBytes: 0,
});

const generateState = (
  events: string[],
  text: MossTextEncoderComponent,
  capacity?: number,
): MossGenerateState => ({
  gpu: headroomGpu(events, capacity),
  preset: mossPreset(MOSS_REFERENCE_PRESET),
  dit: {
    components: {
      dit: stageComponent("dit", "S", events),
      ditSkip22: stageComponent("dit_skip22", "S", events),
      ditText: stageComponent("dit_text", "T", events),
    },
    contract: {
      latentChannels: 128,
      trainingGrid: syntheticTrainingGrid(1000),
      width: 3072,
    } as unknown as MossDitContract,
  },
  vae: {
    components: {
      pre: stageComponent("vae_decoder_pre", "h", events),
      attn: stageComponent("vae_decoder_attn", "S", events),
      post: stageComponent("vae_decoder_post", "h", events),
    },
    contract: {
      bytesPerCell: 49_152,
      spatialScale: 8,
      hiddenChannels: 384,
      pre: { height: "h", width: "w" },
      attn: { sequence: "S" },
      post: { height: "h", width: "w" },
    } as unknown as MossVaeContract,
  },
  text: textState(text),
});

describe("MOSS-Image generate path with the prompt form (stand-ins up to dit_text)", () => {
  it("disposes the text encoder and waits for its release before dit_text opens", async () => {
    const events: string[] = [];

    await assertRejects(
      () => generateMoss(generateState(events, textComponent(events)), BASE),
      Error,
      STOP,
    );

    expect(lifetimes(events)).toEqual([
      `open ${MOSS_TEXT_ENCODER}`,
      "encode 29",
      "encode 10",
      `dispose ${MOSS_TEXT_ENCODER}`,
      "open dit_text",
    ]);
    const disposed = events.indexOf(`dispose ${MOSS_TEXT_ENCODER}`);
    expect(events[disposed + 1]).toBe("release wait");
  });

  it("opens no dit_text when an encoding fails, and has disposed the encoder when generate rejects", async () => {
    const events: string[] = [];
    const failing = textComponent(events, {
      onRun: () => {
        throw new Error("stand-in run failure");
      },
    });

    await assertRejects(
      () => generateMoss(generateState(events, failing), BASE),
      Error,
      "stand-in run failure",
    );

    expect(lifetimes(events)).toEqual([
      `open ${MOSS_TEXT_ENCODER}`,
      "encode 29",
      `dispose ${MOSS_TEXT_ENCODER}`,
    ]);
  });

  it("takes the text encoder's need into the first headroom check, refusing before anything opens", async () => {
    const events: string[] = [];
    // The encoder's need is the largest; the device holds every other stage's.
    const heavy = textComponent(events, { peak: 8 * 2 ** 30 });
    const state = generateState(events, heavy, 4 * 2 ** 30);
    const { component, contract } = requireText(state);
    const need = mossTextEncoderStageNeed({ limits: DEVICE_LIMITS, component, contract }, PROMPTS);

    await assertRejects(
      () => generateMoss(state, BASE),
      Error,
      `this generation needs up to ${need} bytes of VRAM at its largest stage (text_encoder) that` +
        " this device cannot allocate now (checked before any stage opens)",
    );

    expect(lifetimes(events)).toEqual([]);
  });

  it("does not charge the text encoder's need to the embeddings form", async () => {
    const events: string[] = [];
    const state = generateState(events, textComponent(events, { peak: 8 * 2 ** 30 }), 4 * 2 ** 30);

    const { prompt: _prompt, negativePrompt: _negative, ...knobs } = BASE;

    await assertRejects(
      () =>
        generateMoss(state, { ...knobs, positive: embeddingsOf(29), negative: embeddingsOf(10) }),
      Error,
      STOP,
    );

    expect(lifetimes(events)).toEqual(["open dit_text"]);
  });
});

/** The text series of a state built by {@link generateState}. */
const requireText = (state: MossGenerateState): MossTextEncoderState => {
  assert(state.text !== undefined);
  return state.text;
};
