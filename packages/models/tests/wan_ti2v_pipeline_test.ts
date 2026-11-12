// The GPU-free part of the Wan2.2 TI2V 5B class (`Ti2vPipeline` — `src/wan/ti2v-pipeline.ts`). Pins, from the
// public entry points, that family.ts really passes the per-class values (`WanFamilySpec`'s owner, pipeline and
// generation):
//
// - the manifest checks (pipeline name / major, quant, `pipelineConfig`) — the 2.1 and 2.2 classes reject each
//   other's manifests
// - the 48-ch path of the family admission (`fromAssets` runs to the end of construction on tensorless
//   containers) — replacing the VAE with 2.1's 16 ch or the DiT with 2.1's 5 inputs fails the generation
//   comparison (pins against a regression where admission falls back to 2.1's shapes)
// - the fetch side (`fromPretrained`) passes the 2.2 family too (a local source rejects `wan/1` and accepts
//   `wan-ti2v/1`)
// - the plan check applies 2.2's accepted set, and in generation on a mock Session the patch-2 descriptor reaches
//   the end of the VAE stage (unpatchify)
// - the message prefix is `wan.Ti2vPipeline:` (not mixed up with 2.1's messages)
// - I2V (ADR 0121 decision 11): admission of the encoder components; stage order and release (encoder → fold →
//   wait for release → text → DiT); abort of the encoder stage and cleanup after a failure partway through
//   construction; the encoder hand-off (residents and bindings); the non-finite check on the condition latent;
//   DiT conditioning (mask of the first P tokens, time 0 on the condition side, replacement of the model input
//   only, the latent UniPC receives is not replaced, the final replacement); the request checks (fit / size /
//   RGB8); final agreement with the official replacement form (UniPC element-wise)
//
// The full run on a real GPU is the e2e (the helper that assembles from the series — ADR 0121 stage 6,
// commit 6).

import { readFile } from "@karume/test-support/fs";
import {
  assert,
  assertEquals,
  assertRejects,
  assertStrictEquals,
  assertStringIncludes,
  assertThrows,
} from "@std/assert";
import { describe, it } from "vite-plus/test";
import { vaeRmsSiluRule } from "@karume/candidate-rules";
import { type DistributionSource, localDirectory, type Manifest, parseManifest } from "@karume/hub";
import type { EnqueueOptions, RunInput, RunInputs, SessionOptions, Tensor } from "@karume/runtime";
import { ModelInputError } from "../src/errors.ts";
import type { GraphOwner } from "../src/hub/components.ts";
import type { Rgb8Image } from "../src/image/preprocess.ts";
import {
  type Assets,
  type GenerateRequest,
  Pipeline,
  type PipelineOptions,
} from "../src/wan/pipeline.ts";
import { type Ti2vGenerateRequest, Ti2vPipeline } from "../src/wan/ti2v-pipeline.ts";
import { planWanGeneration, WAN22_TI2V_FAMILY } from "../src/wan/family.ts";
import { ditContract, wanDitPatch, withWanFirstFrameCondition } from "../src/wan/dit-loop.ts";
import { timestepsProj } from "../src/wan/dit-timestep.ts";
import { patchifyLatents } from "../src/wan/dit-tokens.ts";
import { preprocessWanI2vImage } from "../src/wan/i2v-preprocess.ts";
import { denormalizeWanLatents, normalizeWanLatents } from "../src/wan/latents.ts";
import {
  WAN_UNIPC_CONFIG,
  wanClassifierFreeGuidance,
  WanUniPcSampler,
  wanUniPcSchedule,
} from "../src/wan/scheduler.ts";
import { wanVaeEncoderContract } from "../src/wan/vae-encoder.ts";
import { wanVaeTile32Contract } from "../src/wan/tile-decode.ts";
import { wanParityCase, wanParityEncoder } from "./helpers/wan-parity-encoder.ts";
import type { WanRopeBase } from "../src/wan/dit-rope.ts";
import { wanVaeChunkLayout } from "../src/wan/vae-chunks.ts";
import type { WanPipelineConfig } from "../src/wan/config.ts";
import {
  parseWanTextEmbeds,
  WAN_TEXT_EMBEDS_METADATA_KEY,
  type WanTextEmbeds,
} from "../src/wan/text-embeds.ts";
import {
  declaredContainer,
  parseIrDeclarationValue,
  partAssets,
  writeContainer,
} from "./helpers/container-fixture.ts";
import {
  fileRef,
  MANIFEST_PATH,
  manifestBytes,
  serveContainer,
} from "./helpers/container-loading-fixture.ts";
import { stubModel } from "./helpers/stub-model.ts";
import type { ModelInput } from "../../runtime/tests/helpers/container-write.ts";
import { fakeDevice, fakeGpuContext } from "../../runtime/tests/helpers/fake-gpu.ts";

/** The message prefix of the 2.2 class (F3 — not mixed up with 2.1's `wan.Pipeline:`). */
const OWNER = "wan.Ti2vPipeline:";

/** The shape of the series' `pipelineConfig` (the default shift is the upstream scheduler's `flow_shift` 5.0). */
const CONFIG: WanPipelineConfig = { scheduler: { shift: 5 }, defaults: { steps: 50, guidance: 5 } };

/** One row of the embeddings asset. */
type Row = {
  readonly name: string;
  readonly role: "positive" | "negative";
  readonly prompt: string;
  readonly normalized: string;
  readonly tokens: number;
};

const ROWS: readonly Row[] = [
  { name: "cats", role: "positive", prompt: "Two cats.", normalized: "Two cats.", tokens: 2 },
  { name: "negative", role: "negative", prompt: "blurry", normalized: "blurry", tokens: 1 },
];

/**
 * The bytes of the embeddings asset (the same form as the recipe `wan/text_embeds.py` — the metadata is JSON
 * under one key, and each row is `[tokens, width]` f32).
 */
const embedsAsset = (width: number): Uint8Array<ArrayBuffer> => {
  const header: Record<string, unknown> = {
    __metadata__: {
      [WAN_TEXT_EMBEDS_METADATA_KEY]: JSON.stringify({ prompts: ROWS, source: {} }),
    },
  };
  let offset = 0;
  for (const row of ROWS) {
    const bytes = row.tokens * width * 4;
    header[row.name] = {
      dtype: "F32",
      shape: [row.tokens, width],
      data_offsets: [offset, offset + bytes],
    };
    offset += bytes;
  }
  const json = new TextEncoder().encode(JSON.stringify(header));
  const headerLength = json.length + ((8 - (json.length % 8)) % 8);
  const out = new Uint8Array(8 + headerLength + offset);
  new DataView(out.buffer).setBigUint64(0, BigInt(headerLength), true);
  out.set(json, 8);
  out.fill(0x20, 8 + json.length, 8 + headerLength);
  return out;
};

// ---------------------------------------------------------------------------------------------
// Construction entry points (tensorless containers — they hold only declarations and not a single weight)
// ---------------------------------------------------------------------------------------------

/**
 * The RoPE base table (the asset `rope_base` of the `transformer` container — widths 22 / 21 / 21 for
 * head_dim 128).
 */
const ROPE_BASE = await readFile(
  new URL("./fixtures/wan-dit/rope_base.safetensors", import.meta.url),
);

type DitInput = {
  readonly name: string;
  readonly shape: readonly (number | string)[];
  readonly dtype?: "f32" | "bool";
};

/** The same 7 declarations as the recipe `wan/ti2v_export_dit.py` (latent 48 ch → `tokens` width 192). */
const TI2V_DIT_INPUTS: readonly DitInput[] = [
  { name: "tokens", shape: [1, "S", 192] },
  { name: "timesteps_proj", shape: [1, 256] },
  { name: "encoder_hidden_states", shape: [1, 512, 4096] },
  { name: "rope_cos", shape: [1, "S", 1, 128] },
  { name: "rope_sin", shape: [1, "S", 1, 128] },
  { name: "timesteps_proj_condition", shape: [1, 256] },
  { name: "condition_mask", shape: [1, "S", 1], dtype: "bool" },
];

/** The content of the DiT container (declarations only, with the RoPE base table asset). */
const ditInput = (inputs: readonly DitInput[]): ModelInput => ({
  graphs: {
    transformer: parseIrDeclarationValue({
      format: "karume-ir",
      version: 2,
      requires: { ops: ["mul"] },
      symbols: ["S"],
      inputs: inputs.map(({ name, shape, dtype }) => ({ name, dtype: dtype ?? "f32", shape })),
      outputs: ["out"],
      initializers: {},
      values: { out: { dtype: "f32", shape: [1, "S", 192] } },
      states: {},
      nodes: [{ op: "mul", ins: ["tokens", "tokens"], outs: ["out"], attrs: {} }],
    }),
  },
  consts: [],
  weights: [],
  assets: [{ name: "rope_base", role: "rope-base", bytes: ROPE_BASE, dedicatedPart: true }],
  provenance: { license: "test" },
});

/** Write settings of the DiT container (sized so that the base table, 12,704 bytes, fits in one block). */
const DIT_WRITE = { partBytes: 1 << 16, blockBytes: 1 << 14 };

/** The DiT container (the part sequence for `fromAssets`). */
const ditAssets = async (inputs: readonly DitInput[]) =>
  partAssets("transformer", await writeContainer(ditInput(inputs), DIT_WRITE));

type VaeChannels = { readonly latent: number; readonly sample: number };

/** Wan2.2's VAE channels (latent 48, exit 12 = RGB 3 × 2²). */
const VAE_22_CHANNELS: VaeChannels = { latent: 48, sample: 12 };

/**
 * The content of one VAE chunk graph's container (latent tile 16, scale 8, one cache — the same geometry as the
 * 2.2 series). `tile` 32 is
 * the 32-latent pair's geometry (ADR 0134 — every spatial dimension doubled; like the tile-16 fixture it
 * declares no initializer, so its "every initializer borrowed" holds vacuously).
 */
const vaeInput = (name: string, frames: number, channels: VaeChannels, tile = 16): ModelInput => ({
  graphs: {
    [name]: parseIrDeclarationValue({
      format: "karume-ir",
      version: 2,
      requires: { ops: ["mul"] },
      symbols: [],
      inputs: [
        { name: "latent", dtype: "f32", shape: [channels.latent, 1, tile, tile] },
        { name: "cache_00", dtype: "f32", shape: [4, 2, tile, tile] },
      ],
      outputs: ["frame", "cache_00_out"],
      initializers: {},
      values: {
        frame: { dtype: "f32", shape: [channels.sample, frames, tile * 8, tile * 8] },
        cache_00_out: { dtype: "f32", shape: [4, 2, tile, tile] },
      },
      states: {},
      nodes: [
        { op: "mul", ins: ["latent", "latent"], outs: ["frame"], attrs: {} },
        { op: "mul", ins: ["cache_00", "cache_00"], outs: ["cache_00_out"], attrs: {} },
      ],
    }),
  },
  consts: [],
  weights: [],
  assets: [],
  provenance: { license: "test" },
});

/** One VAE chunk graph's container (the part sequence for `fromAssets`). */
const vaeAssets = async (name: string, frames: number, channels: VaeChannels, tile = 16) =>
  partAssets(name, await writeContainer(vaeInput(name, frames, channels, tile)));

/** The declaration of one I2V VAE encoder graph (one input → one output, symbolic lengths). */
type EncoderGraph = {
  readonly symbols: readonly string[];
  readonly input: { readonly name: string; readonly shape: readonly (number | string)[] };
  readonly output: readonly (number | string)[];
};

/**
 * Three graphs with the same inputs, outputs and symbols as the recipe `wan/export_vae_encoder.py` (latent 48,
 * mid block 640, patchified image `[12, 1, 8h, 8w]`).
 */
const ENCODER_GRAPHS: Readonly<Record<string, EncoderGraph>> = {
  vae_encoder_pre: {
    symbols: ["h", "w"],
    input: { name: "image", shape: [12, 1, "8h", "8w"] },
    output: [640, 1, "h", "w"],
  },
  vae_encoder_attn: {
    symbols: ["S"],
    input: { name: "tokens", shape: [640, "S"] },
    output: [640, "S"],
  },
  vae_encoder_post: {
    symbols: ["h", "w"],
    input: { name: "hidden", shape: [640, 1, "h", "w"] },
    output: [48, 1, "h", "w"],
  },
};

/** The content of one encoder graph's container (declarations only). */
const encoderInput = (name: string, graph: EncoderGraph): ModelInput => ({
  graphs: {
    [name]: parseIrDeclarationValue({
      format: "karume-ir",
      version: 2,
      requires: { ops: ["mul"] },
      symbols: graph.symbols,
      inputs: [{ name: graph.input.name, dtype: "f32", shape: graph.input.shape }],
      outputs: ["out"],
      initializers: {},
      values: { out: { dtype: "f32", shape: graph.output } },
      states: {},
      nodes: [{ op: "mul", ins: [graph.input.name, graph.input.name], outs: ["out"], attrs: {} }],
    }),
  },
  consts: [],
  weights: [],
  assets: [],
  provenance: { license: "test" },
});

/**
 * The three encoder containers (the part sequence for `fromAssets` — only the components in `replaced` get
 * replaced declarations).
 */
const encoderAssets = async (
  replaced: Readonly<Record<string, EncoderGraph>> = {},
): Promise<Record<string, Uint8Array<ArrayBuffer>>> => {
  const graphs = { ...ENCODER_GRAPHS, ...replaced };
  const parts = await Promise.all(
    Object.entries(graphs).map(async ([name, graph]) =>
      partAssets(name, await writeContainer(encoderInput(name, graph))),
    ),
  );
  return parts.reduce((all, part) => ({ ...all, ...part }), {});
};

/** The components of the 2.2 distribution (DiT, two VAE decoders, three I2V VAE encoders). */
const COMPONENTS = [
  "transformer",
  "vae_decoder_first",
  "vae_decoder_next",
  ...Object.keys(ENCODER_GRAPHS),
];

/**
 * The manifest the construction entry-point tests use (the same skeleton as the helper that assembles from the
 * series — declarations only).
 */
const ti2vManifest = (
  overrides: {
    readonly pipeline?: string;
    readonly pipelineConfig?: unknown;
    /** Components beyond {@link COMPONENTS} (the 32-latent decoder pair's roles — ADR 0134). */
    readonly extraComponents?: readonly string[];
  } = {},
): Manifest => {
  const components = [...COMPONENTS, ...(overrides.extraComponents ?? [])];
  return parseManifest(
    JSON.stringify({
      format: "karume/5",
      generator: "karume/0.1.0",
      defaultModel: "ti2v-5b",
      models: {
        "ti2v-5b": {
          pipeline: overrides.pipeline ?? "wan-ti2v/1",
          weights: Object.fromEntries(
            components.map((name) => [name, { f16: declaredContainer(`${name}/model.f16`) }]),
          ),
          assets: {
            text_embeds: { path: "text_embeds.safetensors", size: 8, sha256: "e".repeat(64) },
          },
          quants: {
            f16: {
              weights: Object.fromEntries(components.map((name) => [name, "f16"])),
              session: {},
            },
          },
          defaultQuant: "f16",
          pipelineConfig: overrides.pipelineConfig ?? CONFIG,
        },
      },
    }),
  );
};

/**
 * The full set of 2.2 declaration assets (VAE with 48 ch and 12 output ch, DiT with 7 inputs, three encoders,
 * embedding width 4096).
 */
const VALID_ASSETS: Assets["assets"] = {
  ...(await ditAssets(TI2V_DIT_INPUTS)),
  ...(await vaeAssets("vae_decoder_first", 1, { latent: 48, sample: 12 })),
  ...(await vaeAssets("vae_decoder_next", 4, { latent: 48, sample: 12 })),
  ...(await encoderAssets()),
  text_embeds: embedsAsset(4096),
};

/**
 * Replaces part of the assets and passes them to `Ti2vPipeline.fromAssets` (a shared mock GPU — no adapter is
 * touched).
 */
const buildTi2v = (
  manifest: Manifest,
  replaced: Assets["assets"] = {},
  options: PipelineOptions = {},
) =>
  Ti2vPipeline.fromAssets(
    { manifest, assets: { ...VALID_ASSETS, ...replaced } },
    { textEncoder: "precomputed", gpu: fakeGpuContext(fakeDevice()), ...options },
  );

/**
 * Construction options with one field set to a value TS's types would not allow (reproduces a JS caller's
 * misspelling).
 */
const optionsWith = (key: string, value: unknown): PipelineOptions => {
  const options: PipelineOptions = {};
  Object.defineProperty(options, key, { value, enumerable: true });
  return options;
};

describe("Ti2vPipeline.fromAssets: manifest checks (pipeline name / major, quant, pipelineConfig)", () => {
  it("control: a wan-ti2v/1 manifest with 2.2 declaration assets finishes construction and returns prompts in asset order", async () => {
    const pipeline = await buildTi2v(ti2vManifest());
    assert(pipeline instanceof Ti2vPipeline);
    assertEquals(
      pipeline.prompts.map((prompt) => prompt.name),
      ["cats", "negative"],
    );
    await pipeline.dispose();
  });

  it("rejects a wan/1 (Wan2.1) manifest with a message that starts with wan.Ti2vPipeline:", async () => {
    const error = await assertRejects(
      () => buildTi2v(ti2vManifest({ pipeline: "wan/1" })),
      Error,
      "the manifest's pipeline is 'wan/1' ('wan-ti2v/1' is required)",
    );
    assert(error.message.startsWith(OWNER), error.message);
    assert(
      !(error instanceof ModelInputError),
      "an asset mismatch must not be classified as an input error",
    );
  });

  it("the reverse: Pipeline rejects a wan-ti2v/1 manifest with a wan.Pipeline: message", async () => {
    const error = await assertRejects(
      () =>
        Pipeline.fromAssets(
          { manifest: ti2vManifest(), assets: VALID_ASSETS },
          { textEncoder: "precomputed", gpu: fakeGpuContext(fakeDevice()) },
        ),
      Error,
      "wan.Pipeline: the manifest's pipeline is 'wan-ti2v/1' ('wan/1' is required)",
    );
    assert(!error.message.startsWith(OWNER), error.message);
  });

  it("rejects major 2, saying that this implementation reads wan-ti2v/1", async () => {
    const error = await assertRejects(
      () => buildTi2v(ti2vManifest({ pipeline: "wan-ti2v/2" })),
      Error,
      "pipeline 'wan-ti2v/2' has an unsupported major (this implementation reads wan-ti2v/1)",
    );
    assert(error.message.startsWith(OWNER), error.message);
  });

  it("rejects a quant missing from the manifest, listing the available ones", async () => {
    const error = await assertRejects(
      () => buildTi2v(ti2vManifest(), {}, { quant: "f16+dit8" }),
      Error,
      "quant 'f16+dit8' is not in the manifest (available: f16)",
    );
    assert(error.message.startsWith(OWNER), error.message);
  });

  it("rejects unknown pipelineConfig keys with the same schema check as 2.1", async () => {
    await assertRejects(
      () =>
        buildTi2v(
          ti2vManifest({ pipelineConfig: { ...CONFIG, scheduler: { shift: 5, flowShift: 5 } } }),
        ),
      Error,
      "pipelineConfig.scheduler: unknown key 'flowShift' (allowed: shift)",
    );
  });

  it("throws a plain wan.Ti2vPipeline: Error for an unknown textEncoder spelling before touching the assets", async () => {
    const error = await assertRejects(
      () =>
        Ti2vPipeline.fromAssets(
          { manifest: ti2vManifest(), assets: {} },
          optionsWith("textEncoder", "cpu"),
        ),
      Error,
      `${OWNER} textEncoder 'cpu' is neither 'gpu' nor 'precomputed'`,
    );
    assert(!(error instanceof ModelInputError));
  });
});

describe("Ti2vPipeline.fromAssets: the family admission compares against the 2.2 generation (48 ch, 7 inputs)", () => {
  it("fails the comparison with the 48 statistics when the VAE is replaced with 2.1's declaration (latent 16, output 3)", async () => {
    const vae = {
      ...(await vaeAssets("vae_decoder_first", 1, { latent: 16, sample: 3 })),
      ...(await vaeAssets("vae_decoder_next", 4, { latent: 16, sample: 3 })),
    };
    await assertRejects(
      () => buildTi2v(ti2vManifest(), vae),
      Error,
      `${OWNER} the denormalization statistics (mean 48, std 48) do not match the VAE latent's 16 channels`,
    );
  });

  it("fails the comparison with unpatchify 2 when the output is RGB's 3 ch (a declaration without unpatchify), even with latent 48", async () => {
    const vae = {
      ...(await vaeAssets("vae_decoder_first", 1, { latent: 48, sample: 3 })),
      ...(await vaeAssets("vae_decoder_next", 4, { latent: 48, sample: 3 })),
    };
    await assertRejects(
      () => buildTi2v(ti2vManifest(), vae),
      Error,
      `${OWNER} VAE output 3 channels differ from RGB 3 × unpatchify 2² = 12`,
    );
  });

  it("fails with the ti2v input-set message when the DiT is replaced with 2.1's 5 inputs", async () => {
    const t2v = "tokens, timesteps_proj, encoder_hidden_states, rope_cos, rope_sin";
    const dit = await ditAssets(TI2V_DIT_INPUTS.slice(0, 5));
    await assertRejects(
      () => buildTi2v(ti2vManifest(), dit),
      Error,
      `${OWNER} transformer graph inputs are [${t2v}]` +
        ` (expected: [${t2v}, timesteps_proj_condition, condition_mask] — input form 'ti2v')`,
    );
  });
});

/**
 * An in-memory directory holding the 2.2 declaration containers (a local distribution — `localDirectory`). Only
 * `pipeline` changes, to see whether the fetch side's check applies the class's pipeline name.
 */
const ti2vDirectory = async (
  pipeline: string,
  /** The 32-latent decoder pair's roles the preset carries (ADR 0134 — none by default). */
  tile32Roles: readonly string[] = [],
): Promise<DistributionSource> => {
  const transformer = await serveContainer(
    "ti2v/transformer/model.f16",
    ditInput(TI2V_DIT_INPUTS),
    DIT_WRITE,
  );
  const first = await serveContainer(
    "ti2v/vae_decoder_first/model.f16",
    vaeInput("vae_decoder_first", 1, { latent: 48, sample: 12 }),
  );
  const next = await serveContainer(
    "ti2v/vae_decoder_next/model.f16",
    vaeInput("vae_decoder_next", 4, { latent: 48, sample: 12 }),
  );
  const encoders = await Promise.all(
    Object.entries(ENCODER_GRAPHS).map(
      async ([name, graph]) =>
        [name, await serveContainer(`ti2v/${name}/model.f16`, encoderInput(name, graph))] as const,
    ),
  );
  const tile32 = await Promise.all(
    tile32Roles.map(
      async (name) =>
        [
          name,
          await serveContainer(
            `ti2v/${name}/model.f16`,
            vaeInput(name, name.startsWith("vae_decoder_first") ? 1 : 4, VAE_22_CHANNELS, 32),
          ),
        ] as const,
    ),
  );
  const embeds = embedsAsset(4096);
  const embedsPath = "text_embeds.safetensors";
  const models = {
    test: {
      pipeline,
      weights: {
        transformer: { f16: transformer.entry },
        vae_decoder_first: { f16: first.entry },
        vae_decoder_next: { f16: next.entry },
        ...Object.fromEntries(encoders.map(([name, served]) => [name, { f16: served.entry }])),
        ...Object.fromEntries(tile32.map(([name, served]) => [name, { f16: served.entry }])),
      },
      assets: { text_embeds: await fileRef(embedsPath, embeds) },
      quants: {
        f16: {
          weights: Object.fromEntries([...COMPONENTS, ...tile32Roles].map((name) => [name, "f16"])),
          session: {},
        },
      },
      defaultQuant: "f16",
      pipelineConfig: CONFIG,
    },
  };
  const files = new Map<string, Uint8Array<ArrayBuffer>>([
    [MANIFEST_PATH, manifestBytes(models)],
    ...transformer.files,
    ...first.files,
    ...next.files,
    ...encoders.flatMap(([, served]) => [...served.files]),
    ...tile32.flatMap(([, served]) => [...served.files]),
    [embedsPath, embeds],
  ]);
  return localDirectory(
    {
      readFile: (path) => {
        const bytes = files.get(path);
        if (bytes === undefined) return Promise.reject(new Error(`ti2v: ${path} is missing`));
        // MUST: return a view that covers the whole buffer (the contract of `DirectoryAdapter.readFile`).
        return Promise.resolve(bytes.slice());
      },
    },
    { label: "ti2v" },
  );
};

describe("Ti2vPipeline.fromPretrained: the fetch side passes the 2.2 family too (a local source)", () => {
  const fromPretrained = (source: DistributionSource) =>
    Ti2vPipeline.fromPretrained(source, {
      textEncoder: "precomputed",
      gpu: fakeGpuContext(fakeDevice()),
    });

  it("control: a wan-ti2v/1 distribution finishes construction and returns prompts in asset order", async () => {
    const pipeline = await fromPretrained(await ti2vDirectory("wan-ti2v/1"));
    assert(pipeline instanceof Ti2vPipeline);
    assertEquals(
      pipeline.prompts.map((prompt) => prompt.name),
      ["cats", "negative"],
    );
    await pipeline.dispose();
  });

  it("rejects a wan/1 (Wan2.1) distribution with a wan.Ti2vPipeline: message", async () => {
    const error = await assertRejects(
      async () => fromPretrained(await ti2vDirectory("wan/1")),
      Error,
      "the manifest's pipeline is 'wan/1' ('wan-ti2v/1' is required)",
    );
    assert(error.message.startsWith(OWNER), error.message);
  });
});

/**
 * The declarations of 2.2's two VAE chunk graphs (latent 48, tile 16, scale 8, output 12 = RGB 3 × 2²). `tile`
 * 32 is the
 * 32-latent pair's geometry (ADR 0134 — every spatial dimension doubled).
 */
const vaeChunkGraphs = (tile = 16) => {
  const graph = (frames: number) =>
    stubModel({
      inputs: [
        { name: "latent", shape: [48, 1, tile, tile] },
        { name: "cache_00", shape: [4, 2, tile, tile] },
      ],
      outputs: ["frame", "cache_00_out"],
      values: { frame: [12, frames, tile * 8, tile * 8], cache_00_out: [4, 2, tile, tile] },
    });
  return { first: graph(1), next: graph(4) };
};

// NOTE: this pins the content of the `WAN22_TI2V_FAMILY` constant through the plan check (the test passes
// generation and owner directly). The path by which the family's generation passes spec.generation is pinned
// by the 832×480 test on a mock Session.
describe("WAN22_TI2V_FAMILY's generation values (the plan check)", () => {
  const embeds: WanTextEmbeds = parseWanTextEmbeds(embedsAsset(3).buffer);
  const graphs = vaeChunkGraphs();
  const layout = wanVaeChunkLayout(graphs.first, graphs.next);
  const plan = (request: Partial<GenerateRequest>) =>
    planWanGeneration(
      { prompt: "Two cats.", ...request },
      embeds,
      layout,
      CONFIG,
      WAN22_TI2V_FAMILY.generation,
      WAN22_TI2V_FAMILY.owner,
    );

  it("1280×704 / 704×1280 × 33 frames pass, and the latent is [48, 9, H/16, W/16]", () => {
    assertEquals(plan({ frames: 33 }).latentShape, [48, 9, 44, 80]);
    assertEquals(plan({ frames: 33, width: 704, height: 1280 }).latentShape, [48, 9, 80, 44]);
  });

  it("an omitted size and frame count take the descriptor's defaults 1280×704 and 33 frames, and an omitted shift takes pipelineConfig's value", () => {
    const planned = plan({});
    assertEquals([planned.width, planned.height, planned.frames], [1280, 704, 33]);
    assertEquals(planned.shift, 5);
  });

  it("4n+1 values above the default 33 up to the limit 121 (37 / 41 / 45 / 49 / 53 / 121) pass, with latent frame counts 10 / 11 / 12 / 13 / 14 / 31", () => {
    for (const [frames, latentFrames] of [
      [37, 10],
      [41, 11],
      [45, 12],
      [49, 13],
      [53, 14],
      [121, 31],
    ]) {
      assertEquals(plan({ frames }).latentShape, [48, latentFrames, 44, 80], `frames ${frames}`);
    }
  });

  it("throws ModelInputError for 832×480 (Wan2.1's default) with 2.2's accepted-set message, and for 125 frames with the limit-121 message", () => {
    assertThrows(
      () => plan({ width: 832, height: 480 }),
      ModelInputError,
      "832×480 is not in the accepted set (1280×704 / 704×1280)",
    );
    assertThrows(
      () => plan({ frames: 125 }),
      ModelInputError,
      "frames 125 is not in the accepted set (4n+1 in 5..121)",
    );
  });
});

// ---------------------------------------------------------------------------------------------
// Generation on mock Sessions (the same form as mockPipeline in `wan_pipeline_test.ts` — only 2.2's geometry and
// asset path)
// ---------------------------------------------------------------------------------------------

/** The value channel `s` of the VAE output (patchify space) returns — different per channel and exact in binary. */
const patchValue = (channel: number): number => channel / 16;

/** One enqueue on the three encoder Sessions (the arguments the mock records). */
type EncoderEnqueue = {
  readonly part: string;
  readonly inputs: RunInputs;
  readonly options: EnqueueOptions;
};

/** The failure injected into an encoder enqueue (`failEnqueue` of {@link mockTi2v}). */
const ENQUEUE_FAILURE = new Error("mock: encoder enqueue failed");

/** The failure injected into the construction of an encoder Session (`failCreate` of {@link mockTi2v}). */
const CREATE_FAILURE = new Error("mock: encoder Session construction failed");

/** The failure injected into the read-back of the encoder's mu (`failRead` of {@link mockTi2v}). */
const READ_FAILURE = new Error("mock: read-back of the encoder's mu failed");

/** A {@link GraphOwner} holding only one encoder graph's declaration (the output is named `out`). */
const encoderStub = (graph: EncoderGraph): GraphOwner =>
  stubModel({
    symbols: graph.symbols,
    inputs: [graph.input],
    outputs: ["out"],
    values: { out: graph.output },
  });

/** The three encoder graphs ({@link ENCODER_GRAPHS} — only the components in `replaced` get replaced declarations). */
const encoderGraphs = (replaced: Readonly<Record<string, EncoderGraph>> = {}) => {
  const graphs = { ...ENCODER_GRAPHS, ...replaced };
  return {
    pre: encoderStub(graphs.vae_encoder_pre),
    attn: encoderStub(graphs.vae_encoder_attn),
    post: encoderStub(graphs.vae_encoder_post),
  };
};

/**
 * A `Ti2vPipeline` run on mock Sessions. The DiT returns a constant, and the VAE runs its one-tile = one-batch
 * procedure on a fake GpuContext and fills the frames it reads back with **per-channel values**
 * ({@link patchValue}) — so the unpatchify ordering shows in the output pixel values. The I2V encoder stage
 * records the enqueues of the three Sessions and fills mu (read-back key `mu`) from `options.mu`. The fake
 * GpuContext's wait for release (an empty submit → `onSubmittedWorkDone`) is also pushed to `log`.
 *
 * NOTE: the constructor is TS `private`, so `Reflect.construct` passes the internal state (family.ts's
 * `WanState`) directly (bypassing private only in tests — for the same reason as mockPipeline in
 * `wan_pipeline_test.ts`).
 */
const mockTi2v = (
  options: {
    /**
     * With `"gpu"`, runs the text stage on a mock umT5 (the prompt layer is the fixture's table —
     * {@link wanParityEncoder}; the output is a constant of width 3). The DiT context has 512 rows, as in the
     * distribution (the official negative is 126 tokens).
     */
    readonly textEncoder?: "gpu";
    /** The encoder's mu values (index → value — 0.25 when omitted). */
    readonly mu?: (index: number) => number;
    /** Throws {@link ENQUEUE_FAILURE} on the enqueue of this component (`vae_encoder_*`). */
    readonly failEnqueue?: string;
    /**
     * Called with the component name inside the encoder's `createSession` (the abort injection point — `abort`
     * partway through construction).
     */
    readonly onEncoderCreate?: (part: string) => void;
    /** Rejects the `createSession` of this component (`vae_encoder_*`) with {@link CREATE_FAILURE}. */
    readonly failCreate?: string;
    /** Rejects the read-back of the encoder's mu (`finishAndRead`) with {@link READ_FAILURE}. */
    readonly failRead?: boolean;
    /**
     * Builds the 32-latent decoder pair (ADR 0134): the 16-latent pair's Sessions become lenders
     * that log `create` / `lend` / `dispose` with their options, and the pair's borrowers log their
     * options and `sharedWeights`. `sessionOptions` is the state's quant settings; `failCreate`
     * rejects that borrower's construction with {@link CREATE_FAILURE}.
     */
    readonly tile32?: {
      readonly sessionOptions?: SessionOptions;
      readonly failCreate?: "vae_decoder_first_32" | "vae_decoder_next_32";
    };
    /** The state's experimental `fusionRules` (absent = the field does not exist). */
    readonly fusionRules?: SessionOptions["fusionRules"];
  } = {},
) => {
  const log: string[] = [];
  /** Every `createSession` of every stand-in, with the options object it received (in call order). */
  const created: { readonly component: string; readonly options: unknown }[] = [];
  const ditRuns: Record<string, Tensor>[] = [];
  /** The enqueues of the three encoder Sessions (in issue order). */
  const encoderEnqueues: EncoderEnqueue[] = [];
  /** The `latent` of each VAE chunk enqueue (in issue order — the first tile goes first → next). */
  const vaeLatents: Tensor[] = [];
  /** The component names `onRunDiagnostics` received (in issue order). */
  const diagnosed: string[] = [];
  /** `copyLatents()` of each `denoise-step` (in step order). */
  const snapshots: Float32Array[] = [];
  const { first, next } = vaeChunkGraphs();
  const resident = (byteLength: number, label: string) => ({
    byteLength,
    label,
    write: () => {},
    clear: () => {},
    dispose: () => {
      log.push(`release:${label}`);
    },
  });
  const vaeSession = (name: string) => ({
    createSession: (_gpu: unknown, sessionOptions: unknown) => {
      created.push({ component: name, options: sessionOptions });
      log.push(`create:${name}`);
      return Promise.resolve({
        enqueue: (inputs: Record<string, Tensor>) => {
          vaeLatents.push(inputs.latent);
          return Promise.resolve();
        },
        diagnostics: () => ({}),
        dispose: () => Promise.resolve(),
      });
    },
  });
  const encoderSession = (part: string) => ({
    createSession: (_gpu: unknown, sessionOptions: unknown) => {
      created.push({ component: part, options: sessionOptions });
      log.push(`create:${part}:${JSON.stringify(sessionOptions)}`);
      options.onEncoderCreate?.(part);
      if (options.failCreate === part) return Promise.reject(CREATE_FAILURE);
      return Promise.resolve({
        enqueue: (inputs: EncoderEnqueue["inputs"], enqueueOptions: EncoderEnqueue["options"]) => {
          log.push(`enqueue:${part}`);
          encoderEnqueues.push({ part, inputs, options: enqueueOptions });
          return options.failEnqueue === part ? Promise.reject(ENQUEUE_FAILURE) : Promise.resolve();
        },
        diagnostics: () => ({}),
        dispose: () => {
          log.push(`dispose:${part}`);
          return Promise.resolve();
        },
      });
    },
  });
  const layout = wanVaeChunkLayout(first, next);
  /** A 16-latent Session that lends its weights (only with `options.tile32`). */
  const lenderSession = (name: string) => ({
    createSession: (_gpu: unknown, sessionOptions: unknown) => {
      created.push({ component: name, options: sessionOptions });
      log.push(`create:${name}:${JSON.stringify(sessionOptions)}`);
      return Promise.resolve({
        exportWeight: (initializer: string) => {
          log.push(`lend:${name}:${initializer}`);
          return { lentBy: name, initializer };
        },
        enqueue: (inputs: Record<string, Tensor>) => {
          log.push(`enqueue:${name}`);
          vaeLatents.push(inputs.latent);
          return Promise.resolve();
        },
        diagnostics: () => ({}),
        dispose: () => {
          log.push(`dispose:${name}`);
          return Promise.resolve();
        },
      });
    },
  });
  /** A 32-latent Session that borrows (only with `options.tile32`). */
  const borrowerSession = (name: "vae_decoder_first_32" | "vae_decoder_next_32") => ({
    createSession: (_gpu: unknown, sessionOptions: Record<string, unknown>) => {
      created.push({ component: name, options: sessionOptions });
      log.push(`create:${name}:${JSON.stringify(sessionOptions)}`);
      if (options.tile32?.failCreate === name) return Promise.reject(CREATE_FAILURE);
      return Promise.resolve({
        enqueue: (inputs: Record<string, Tensor>) => {
          log.push(`enqueue:${name}`);
          vaeLatents.push(inputs.latent);
          return Promise.resolve();
        },
        diagnostics: () => ({}),
        dispose: () => {
          log.push(`dispose:${name}`);
          return Promise.resolve();
        },
      });
    },
  });
  const tile32Graphs = vaeChunkGraphs(32);
  const tile32 =
    options.tile32 === undefined
      ? undefined
      : {
          layout: wanVaeChunkLayout(tile32Graphs.first, tile32Graphs.next),
          borrowed: { first: ["first.w"], next: ["next.w", "const.0"] },
          first: borrowerSession("vae_decoder_first_32"),
          next: borrowerSession("vae_decoder_next_32"),
        };
  // 64 rows cover the grid of the latent [48, 2, 44, 80] (2 × 22 × 40). The values are irrelevant to the checks.
  const ropeBase: WanRopeBase = {
    rows: 64,
    widths: [1, 1, 1],
    cos: [new Float32Array(64), new Float32Array(64), new Float32Array(64)],
    sin: [new Float32Array(64), new Float32Array(64), new Float32Array(64)],
  };
  const gpuText = options.textEncoder === "gpu";
  // The DiT contract is built with the same function and values as the family admission (the patch from the
  // VAE declaration's channel count, the input form and owner from the spec) — if the spec's input form falls
  // back to 2.1, the 7-input declaration fails here.
  const dit = ditContract(
    stubModel({
      symbols: ["S"],
      inputs: [
        { name: "tokens", shape: [1, "S", 192] },
        { name: "timesteps_proj", shape: [1, 256] },
        { name: "encoder_hidden_states", shape: [1, gpuText ? 512 : 4, 3] },
        { name: "rope_cos", shape: [1, "S", 1, 6] },
        { name: "rope_sin", shape: [1, "S", 1, 6] },
        { name: "timesteps_proj_condition", shape: [1, 256] },
        { name: "condition_mask", shape: [1, "S", 1], dtype: "bool" },
      ],
      outputs: ["out"],
      values: { out: [1, "S", 192] },
    }),
    ropeBase,
    wanDitPatch(layout.latentChannels),
    WAN22_TI2V_FAMILY.generation.ditInputForm,
    WAN22_TI2V_FAMILY.owner,
  );
  // The encoder contract is also built with the same function and values as the family admission (the
  // decoder's geometry, the 2.2 descriptor).
  const vaeEncoderContract = wanVaeEncoderContract(
    encoderGraphs(),
    layout,
    WAN22_TI2V_FAMILY.generation,
    WAN22_TI2V_FAMILY.owner,
  );
  const text = gpuText
    ? {
        kind: "gpu",
        encoder: wanParityEncoder(),
        contract: { output: "umt5_out" },
        component: {
          createSession: (_gpu: unknown, sessionOptions: unknown) => {
            created.push({ component: "text_encoder", options: sessionOptions });
            log.push("create:text_encoder");
            return Promise.resolve({
              run: (inputs: Record<string, Tensor>) => {
                const tokens = inputs.input_ids.shape[1];
                return Promise.resolve({
                  umt5_out: {
                    dtype: "f32",
                    shape: [1, tokens, 3],
                    data: new Float32Array(tokens * 3).fill(tokens / 1000),
                  },
                });
              },
              diagnostics: () => ({}),
              dispose: () => {
                log.push("dispose:text_encoder");
                return Promise.resolve();
              },
            });
          },
        },
      }
    : { kind: "precomputed" };
  const state = {
    gpu: {
      createResident: (bytes: number, label: string) => {
        log.push(`resident:${label}`);
        return Promise.resolve(resident(bytes, label));
      },
      beginBatch: () => {
        log.push("batch:begin");
        return Promise.resolve({
          finish: () => {
            log.push("batch:finish");
            return Promise.resolve();
          },
          // The encoder's mu is filled from `options.mu`; the VAE frame resident `[12, T, 128, 128]` (channel-major)
          // fills channel s's plane with patchValue(s).
          finishAndRead: (outputs: Record<string, { readonly byteLength: number }>) => {
            log.push("batch:read");
            if (options.failRead === true && Object.hasOwn(outputs, "mu")) {
              return Promise.reject(READ_FAILURE);
            }
            return Promise.resolve(
              Object.fromEntries(
                Object.entries(outputs).map(([name, { byteLength }]) => {
                  const values = new Float32Array(byteLength / 4);
                  if (name === "mu") {
                    for (let at = 0; at < values.length; at += 1) {
                      values[at] = options.mu?.(at) ?? 0.25;
                    }
                    return [name, values.buffer];
                  }
                  const perChannel = values.length / 12;
                  for (let at = 0; at < values.length; at += 1) {
                    values[at] = patchValue(Math.floor(at / perChannel));
                  }
                  return [name, values.buffer];
                }),
              ),
            );
          },
        });
      },
      // The surface the wait for release (`settleReleasedMemory`) touches.
      device: {
        queue: {
          submit: () => {
            log.push("settle:submit");
          },
          onSubmittedWorkDone: () => {
            log.push("settle:done");
            return Promise.resolve();
          },
        },
      },
      onLost: () => () => {},
    },
    ownsGpu: false,
    config: CONFIG,
    quantName: "f16",
    sessionOptions: options.tile32?.sessionOptions ?? {},
    transformer: {
      createSession: (_gpu: unknown, sessionOptions: unknown) => {
        created.push({ component: "transformer", options: sessionOptions });
        log.push("create:transformer");
        return Promise.resolve({
          run: (inputs: Record<string, Tensor>) => {
            ditRuns.push(inputs);
            const tokens = inputs.tokens;
            return Promise.resolve({
              out: {
                dtype: "f32",
                shape: tokens.shape,
                data: new Float32Array(tokens.data.length).fill(0.1),
              },
            });
          },
          diagnostics: () => ({}),
          dispose: () => Promise.resolve(),
        });
      },
    },
    vaeFirst:
      tile32 === undefined ? vaeSession("vae_decoder_first") : lenderSession("vae_decoder_first"),
    vaeNext:
      tile32 === undefined ? vaeSession("vae_decoder_next") : lenderSession("vae_decoder_next"),
    vaeTile32: tile32,
    vaeEncoder: {
      pre: encoderSession("vae_encoder_pre"),
      attn: encoderSession("vae_encoder_attn"),
      post: encoderSession("vae_encoder_post"),
      contract: vaeEncoderContract,
    },
    layout,
    ropeBase,
    dit,
    textEmbeds: parseWanTextEmbeds(embedsAsset(3).buffer),
    text,
    ...(options.fusionRules === undefined ? {} : { fusionRules: options.fusionRules }),
    onRunDiagnostics: (component: string) => {
      diagnosed.push(component);
    },
  };
  const pipeline: Ti2vPipeline = Reflect.construct(Ti2vPipeline, [state]);
  /**
   * Runs a request of 5 frames and 2 steps (4 DiT runs), pushing the observed events to `log` and the
   * intermediate latents to `snapshots`. The prompt is `cats` on the asset path and the fixture's fixed prompt
   * `boxing-cats` on the GPU path.
   */
  const generate = (request: Partial<Ti2vGenerateRequest> = {}) =>
    pipeline.generate({
      prompt: gpuText ? wanParityCase("fixed-boxing-cats").text : "Two cats.",
      frames: 5,
      steps: 2,
      ...request,
      onEvent: (event) => {
        if (event.kind === "denoise-step") snapshots.push(event.copyLatents().data);
        log.push(
          event.kind === "stage"
            ? `${event.component}:${event.at}`
            : event.kind === "denoise-step"
              ? `step:${event.step}`
              : `tile:${event.tile}`,
        );
      },
    });
  return {
    pipeline,
    log,
    created,
    ditRuns,
    encoderEnqueues,
    vaeLatents,
    diagnosed,
    snapshots,
    generate,
    contract: vaeEncoderContract,
  };
};

describe("Ti2vPipeline.generate (mock Sessions — 2.2's generation values reach the end of the VAE stage)", () => {
  it("the output is [3, F, H, W] and unpatchified (RGB pixel (y, x) is patch-space channel c·4 + (x%2)·2 + (y%2))", async () => {
    const { log, generate } = mockTi2v();
    const video = await generate();
    assertEquals([video.frames, video.width, video.height, video.fps], [5, 1280, 704, 24]);
    assertEquals(video.data.length, 3 * 5 * 704 * 1280);
    // Covering 1280×704's latent 44×80 with tile 16 takes 28 tiles (the same count as stage 5's frozen table).
    assertEquals(log.filter((entry) => entry.startsWith("tile:")).length, 28);
    let worst = 0;
    let worstAt = -1;
    for (let channel = 0; channel < 3; channel += 1) {
      for (let frame = 0; frame < 5; frame += 1) {
        for (let y = 0; y < 704; y += 1) {
          const row = ((channel * 5 + frame) * 704 + y) * 1280;
          for (let x = 0; x < 1280; x += 1) {
            const expected = patchValue(channel * 4 + (x % 2) * 2 + (y % 2));
            const diff = Math.abs(video.data[row + x] - expected);
            if (diff > worst) [worst, worstAt] = [diff, row + x];
          }
        }
      }
    }
    // Blending the tile overlaps is a weighted sum of equal values, so it returns the original value within f32
    // rounding.
    assert(
      worst <= 1e-6,
      `some pixels differ from the unpatchify ordering (largest difference ${worst}, index ${worstAt})`,
    );
  });

  it("the DiT runs take ti2v's 7 inputs, tokens are 192 wide (the patch of latent 48 ch), and T2V's condition inputs are an all-false mask and the same time as the generation side", async () => {
    const { ditRuns, generate } = mockTi2v();
    await generate();
    assertEquals(ditRuns.length, 4);
    const tokens = 2 * 22 * 40;
    for (const inputs of ditRuns) {
      assertEquals(Object.keys(inputs).sort(), [
        "condition_mask",
        "encoder_hidden_states",
        "rope_cos",
        "rope_sin",
        "timesteps_proj",
        "timesteps_proj_condition",
        "tokens",
      ]);
      assertEquals(inputs.tokens.shape, [1, tokens, 192]);
      const mask = inputs.condition_mask;
      assertEquals([mask.dtype, mask.shape], ["bool", [1, tokens, 1]]);
      assertEquals(mask.data.length, tokens);
      assert(
        mask.data.every((value) => value === 0),
        "T2V's condition mask has a true element",
      );
      // The condition side's time is the same array as the generation side's (dit-loop's T2V contract — not a
      // copy).
      const condition = inputs.timesteps_proj_condition;
      assertEquals([condition.dtype, condition.shape], ["f32", inputs.timesteps_proj.shape]);
      assert(
        condition.data === inputs.timesteps_proj.data,
        "the condition side's time is a different array from the generation side's",
      );
    }
  });

  it("throws ModelInputError for an 832×480 request with 2.2's accepted-set message without creating a single Session", async () => {
    const { log, generate } = mockTi2v();
    await assertRejects(
      () => generate({ width: 832, height: 480 }),
      ModelInputError,
      "832×480 is not in the accepted set (1280×704 / 704×1280)",
    );
    assertEquals(
      log.filter((entry) => entry.startsWith("create:")),
      [],
    );
  });

  it("the message for a prompt outside the set points to wan.Ti2vPipeline.prompts", async () => {
    const { generate } = mockTi2v();
    const error = await assertRejects(() => generate({ prompt: "A ferret." }), ModelInputError);
    assertStringIncludes(error.message, "wan.Ti2vPipeline.prompts lists them");
  });

  it("rejects generation after dispose with a wan.Ti2vPipeline: message", async () => {
    const { pipeline, generate } = mockTi2v();
    await pipeline.dispose();
    await assertRejects(() => generate(), Error, `${OWNER} cannot generate after dispose`);
  });
});

// ---------------------------------------------------------------------------------------------
// I2V (ADR 0121 decision 11 — condition image → encoder stage → DiT stage conditioned on the first latent frame)
// ---------------------------------------------------------------------------------------------

/** An RGB8 image with a different value per pixel (a mock I2V condition image). */
const rgb8 = (width: number, height: number): Rgb8Image => {
  const data = new Uint8Array(width * height * 3);
  for (let index = 0; index < data.length; index += 1) data[index] = (index * 37) % 256;
  return { data, width, height };
};

/** Landscape (16:9 — selects 1280×704) and portrait (9:16 — selects 704×1280) condition images. */
const LANDSCAPE = rgb8(64, 36);
const PORTRAIT = rgb8(36, 64);

/**
 * The latent shape of 1280×704 with 5 frames (`[48, 2, 44, 80]`) and the first frame's token count
 * (`(704/32)·(1280/32)`).
 */
const I2V_SHAPE = [48, 2, 44, 80] as const;
const I2V_PLANE = 44 * 80;
const FIRST_FRAME_TOKENS = 880;

/**
 * A generation request with one field set to a value TS's types would not allow (reproduces a JS caller's
 * misspelling or a value slipping past the types).
 */
const requestWith = (key: string, value: unknown): Partial<Ti2vGenerateRequest> => {
  const request: Partial<Ti2vGenerateRequest> = {};
  Object.defineProperty(request, key, { value, enumerable: true });
  return request;
};

/** Whether two number sequences are bitwise identical (also tells `-0` from `+0` and NaN payloads apart). */
const sameBits = (actual: ArrayLike<number> & ArrayBufferView, expected: Float32Array): boolean =>
  actual.byteLength === expected.byteLength &&
  new Uint8Array(actual.buffer, actual.byteOffset, actual.byteLength).every(
    (byte, at) => byte === new Uint8Array(expected.buffer, expected.byteOffset)[at],
  );

/** Asserts a host tensor rather than a resident (pre's entry takes the preprocessing output as is). */
const hostTensor = (input: RunInput | undefined): Tensor => {
  if (input === undefined || !("data" in input)) throw new Error("not a host tensor");
  return input;
};

/** The condition latent when mu is the constant 0.25 (the encoder stage's output — `[48, 1, 44, 80]` normalized). */
const conditionOf = (mu: number): Float32Array<ArrayBuffer> =>
  normalizeWanLatents(
    new Float32Array(48 * I2V_PLANE).fill(mu),
    WAN22_TI2V_FAMILY.generation.latents,
  );

/**
 * The model input with the first latent frame replaced by **copying** the condition latent (built independently
 * of the replacement formula — the same values as the formula unless the latents outside the first frame or the
 * condition are ±0).
 */
const copyFirstFrame = (
  latents: Float32Array,
  condition: Float32Array,
): Float32Array<ArrayBuffer> => {
  const out = Float32Array.from(latents);
  const [channels, frames] = I2V_SHAPE;
  for (let channel = 0; channel < channels; channel += 1) {
    out.set(
      condition.subarray(channel * I2V_PLANE, (channel + 1) * I2V_PLANE),
      channel * frames * I2V_PLANE,
    );
  }
  return out;
};

describe("Ti2vPipeline.generate: ownership of the caller's latents", () => {
  /** 5 frames at 1280×704: latents [48, 2, 44, 80] (values vary per element so a mix-up shows). */
  const initialLatents = () =>
    new Float32Array(48 * 2 * 44 * 80).map((_, index) => ((index % 97) - 48) / 64);

  it("generates from the values at call time even when the caller rewrites latents while the call waits its turn (finite and NaN)", async () => {
    // Baseline: the same two calls in order, nothing rewritten.
    const baseline = mockTi2v();
    await baseline.generate();
    const offset = baseline.ditRuns.length;
    await baseline.generate({ latents: initialLatents() });
    const expected = baseline.ditRuns[offset].tokens;
    for (const value of [7, Number.NaN]) {
      const { generate, ditRuns } = mockTi2v();
      // The first call holds the serialization chain, so the second waits its turn; the caller
      // rewrites its array right after the call returns (inside that wait).
      const first = generate();
      const latents = initialLatents();
      const second = generate({ latents });
      latents.fill(value);
      await Promise.all([first, second]);
      const tokens = ditRuns[offset].tokens;
      assert(tokens.dtype === "f32" && expected.dtype === "f32");
      const differs = tokens.data.findIndex(
        (entry, index) => !Object.is(entry, expected.data[index]),
      );
      assertEquals(differs, -1, `the DiT ran with the rewritten value ${value}`);
    }
  });
});

describe("Ti2vPipeline.generate I2V: stage order and release (mock Sessions)", () => {
  it("the three encoder Sessions (run options {}) → three residents → pre → attn → post in one batch → folds Sessions, then residents → waits for release → the encoder's end → DiT", async () => {
    const { log, diagnosed, generate } = mockTi2v();
    await generate({ image: LANDSCAPE });
    assertEquals(log.slice(0, log.indexOf("create:transformer") + 1), [
      "vae_encoder:start",
      "create:vae_encoder_pre:{}",
      "create:vae_encoder_attn:{}",
      "create:vae_encoder_post:{}",
      "resident:wan vae_encoder_pre",
      "resident:wan vae_encoder_attn",
      "resident:wan vae_encoder_post",
      "batch:begin",
      "enqueue:vae_encoder_pre",
      "enqueue:vae_encoder_attn",
      "enqueue:vae_encoder_post",
      "batch:read",
      "dispose:vae_encoder_pre",
      "dispose:vae_encoder_attn",
      "dispose:vae_encoder_post",
      "release:wan vae_encoder_pre",
      "release:wan vae_encoder_attn",
      "release:wan vae_encoder_post",
      "settle:submit",
      "settle:done",
      "vae_encoder:end",
      "transformer:start",
      "create:transformer",
    ]);
    // Diagnostics are one name per Session (each encoder once, in stage order).
    assertEquals(diagnosed.slice(0, 4), [
      "vae_encoder_pre",
      "vae_encoder_attn",
      "vae_encoder_post",
      "transformer",
    ]);
    assertEquals(log.at(-1), "vae_decoder:end");
  });

  it("the gpu path: folds the encoder stage and waits for release before creating umT5 (encoder → text → DiT → VAE)", async () => {
    const { log, generate } = mockTi2v({ textEncoder: "gpu" });
    await generate({ image: LANDSCAPE });
    assertEquals(
      log.filter((entry) => /^(vae_encoder|text_encoder|transformer|vae_decoder):/.test(entry)),
      [
        "vae_encoder:start",
        "vae_encoder:end",
        "text_encoder:start",
        "text_encoder:end",
        "transformer:start",
        "transformer:end",
        "vae_decoder:start",
        "vae_decoder:end",
      ],
    );
    assert(
      log.indexOf("settle:done") < log.indexOf("create:text_encoder"),
      `umT5 was created before waiting for release: ${log.slice(0, 30)}`,
    );
  });

  it("a request without an image has no encoder stage (no Session, resident, batch or wait for release — the same path as T2V)", async () => {
    const { log, encoderEnqueues, diagnosed, generate } = mockTi2v();
    await generate();
    assertEquals(
      log.filter((entry) => entry.includes("vae_encoder")),
      [],
    );
    assertEquals(
      log.filter((entry) => entry.startsWith("settle:")),
      [],
    );
    assertEquals(encoderEnqueues, []);
    assertEquals(diagnosed[0], "transformer");
  });

  it("when an enqueue in the encoder batch fails, closes the batch, folds the three Sessions and the residents, waits for release, and rethrows the original failure as is (does not go on to the DiT)", async () => {
    const { log, generate } = mockTi2v({ failEnqueue: "vae_encoder_attn" });
    const error = await assertRejects(() => generate({ image: LANDSCAPE }));
    assertStrictEquals(error, ENQUEUE_FAILURE);
    const after = log.slice(log.indexOf("enqueue:vae_encoder_attn"));
    // A failed stage waits for release too (no encoder allocation is left behind for the DiT stage of the same
    // pipeline's next generation).
    assertEquals(after, [
      "enqueue:vae_encoder_attn",
      "batch:finish",
      "dispose:vae_encoder_pre",
      "dispose:vae_encoder_attn",
      "dispose:vae_encoder_post",
      "release:wan vae_encoder_pre",
      "release:wan vae_encoder_attn",
      "release:wan vae_encoder_post",
      "settle:submit",
      "settle:done",
    ]);
  });
});

/**
 * The encoder stage's teardown sequence (three Sessions → three residents → wait for release — the fold order
 * and the wait for release are MUST).
 */
const ENCODER_TEARDOWN = [
  "dispose:vae_encoder_pre",
  "dispose:vae_encoder_attn",
  "dispose:vae_encoder_post",
  "release:wan vae_encoder_pre",
  "release:wan vae_encoder_attn",
  "release:wan vae_encoder_post",
  "settle:submit",
  "settle:done",
];

describe("Ti2vPipeline.generate I2V: abort of the encoder stage and failures partway through construction (mock Sessions)", () => {
  it("an already-aborted signal throws the reason without creating a single encoder Session (the entry → encoder boundary)", async () => {
    const { log, generate } = mockTi2v();
    const controller = new AbortController();
    const reason = new Error("abort button (entry)");
    controller.abort(reason);
    const error = await assertRejects(() =>
      generate({ image: LANDSCAPE, signal: controller.signal }),
    );
    assertStrictEquals(error, reason);
    assertEquals(
      log.filter((entry) => entry.includes("vae_encoder")),
      [],
    );
    assertEquals(
      log.filter((entry) => entry.startsWith("settle:")),
      [],
    );
  });

  it("an abort partway through construction takes effect before the batch, folds the three Sessions and residents created, waits for release, and throws the reason as is; the next generate runs", async () => {
    const controller = new AbortController();
    const reason = new Error("abort button (encoder stage)");
    const { log, encoderEnqueues, generate } = mockTi2v({
      onEncoderCreate: (part) => {
        if (part === "vae_encoder_pre") controller.abort(reason);
      },
    });
    const error = await assertRejects(() =>
      generate({ image: LANDSCAPE, signal: controller.signal }),
    );
    assertStrictEquals(error, reason);
    // The three-graph batch is indivisible — the abort takes effect before the batch opens, and not a single
    // enqueue is issued.
    assertEquals(encoderEnqueues, []);
    assert(!log.includes("batch:begin"), `a batch was opened after the abort: ${log}`);
    assertEquals(log.slice(log.indexOf("resident:wan vae_encoder_post") + 1), ENCODER_TEARDOWN);
    assert(!log.includes("vae_encoder:end"), `the aborted stage emitted end: ${log}`);
    log.length = 0;
    await generate({ image: LANDSCAPE });
    assertEquals(
      log.at(-1),
      "vae_decoder:end",
      "the generate after the abort does not run to the end",
    );
  });

  it("when the attn Session's construction fails, folds only the pre already created, waits for release, and rethrows the original failure as is (no residents are created)", async () => {
    const { log, generate } = mockTi2v({ failCreate: "vae_encoder_attn" });
    const error = await assertRejects(() => generate({ image: LANDSCAPE }));
    assertStrictEquals(error, CREATE_FAILURE);
    assertEquals(log.slice(log.indexOf("vae_encoder:start")), [
      "vae_encoder:start",
      "create:vae_encoder_pre:{}",
      "create:vae_encoder_attn:{}",
      "dispose:vae_encoder_pre",
      "settle:submit",
      "settle:done",
    ]);
  });

  it("when the read-back of mu fails, folds the three Sessions, then the residents, waits for release, and rethrows the original failure as is (the read-back closes the batch)", async () => {
    const { log, generate } = mockTi2v({ failRead: true });
    const error = await assertRejects(() => generate({ image: LANDSCAPE }));
    assertStrictEquals(error, READ_FAILURE);
    assertEquals(log.slice(log.indexOf("batch:read") + 1), ENCODER_TEARDOWN);
    assert(
      !log.includes("batch:finish"),
      `the batch was closed again after the read-back failure: ${log}`,
    );
  });
});

describe("Ti2vPipeline.generate I2V: the encoder hand-off (mock Sessions)", () => {
  it("pre takes the preprocessing output [12, 1, 352, 640] as is, attn takes S = 3520 and post h 44 / w 80 through bindings, and each stage's copyOutputs resident is the next stage's input (same batch)", async () => {
    const { encoderEnqueues, generate } = mockTi2v();
    await generate({ image: LANDSCAPE });
    assertEquals(
      encoderEnqueues.map(({ part }) => part),
      ["vae_encoder_pre", "vae_encoder_attn", "vae_encoder_post"],
    );
    const [pre, attn, post] = encoderEnqueues;
    const image = hostTensor(pre.inputs.image);
    const expected = preprocessWanI2vImage(LANDSCAPE, {}, WAN22_TI2V_FAMILY.generation);
    assertEquals([...image.shape], [12, 1, 352, 640]);
    assert(
      sameBits(image.data, expected.pixels),
      "pre's input differs from the preprocessing output",
    );
    assertEquals(pre.options.bindings, undefined);
    const preOut = pre.options.copyOutputs?.out;
    const attnOut = attn.options.copyOutputs?.out;
    const mu = post.options.copyOutputs?.out;
    assert(preOut !== undefined && attnOut !== undefined && mu !== undefined);
    assertStrictEquals(attn.inputs.tokens, preOut);
    assertEquals(attn.options.bindings, { S: 3520 });
    assertStrictEquals(post.inputs.hidden, attnOut);
    assertEquals(post.options.bindings, { h: 44, w: 80 });
    // The sizes follow the declared shapes (mid 640 × S, mu 48 × S — f32).
    assertEquals(
      [preOut.byteLength, attnOut.byteLength, mu.byteLength],
      [640 * 3520 * 4, 640 * 3520 * 4, 48 * 3520 * 4],
    );
    assert(pre.options.batch === attn.options.batch && attn.options.batch === post.options.batch);
  });

  it("runs a portrait image at 704×1280 (pre's input [12, 1, 640, 352], post's h 80 / w 44)", async () => {
    const { encoderEnqueues, generate } = mockTi2v();
    const video = await generate({ image: PORTRAIT });
    assertEquals([video.width, video.height], [704, 1280]);
    const [pre, , post] = encoderEnqueues;
    assertEquals([...hostTensor(pre.inputs.image).shape], [12, 1, 640, 352]);
    assertEquals(post.options.bindings, { h: 80, w: 44 });
  });

  for (const [label, mu, message] of [
    [
      "NaN",
      (index: number) => (index === 0 ? Number.NaN : 0.25),
      "channel 0, position (x=0, y=0) is non-finite (NaN)",
    ],
    [
      "Infinity",
      (index: number) => (index === I2V_PLANE + 80 + 1 ? Number.POSITIVE_INFINITY : 0.25),
      "channel 1, position (x=1, y=1) is non-finite (Infinity)",
    ],
  ] as const) {
    it(`fails in the encoder stage when the condition latent (after normalizing mu) is ${label}, and does not create the DiT stage (the encoders are folded)`, async () => {
      const { log, generate } = mockTi2v({ mu });
      await assertRejects(
        () => generate({ image: LANDSCAPE }),
        Error,
        `${OWNER} condition image latent (after normalizing the VAE encoder's mu) ${message}`,
      );
      assert(!log.includes("create:transformer"), `the DiT Session was created: ${log}`);
      assert(!log.includes("vae_encoder:end"), "the stage that failed partway emitted end");
      for (const part of ["vae_encoder_pre", "vae_encoder_attn", "vae_encoder_post"]) {
        assert(log.includes(`dispose:${part}`), `${part} was not folded`);
        assert(log.includes(`release:wan ${part}`), `${part}'s resident was not returned`);
      }
    });
  }
});

describe("Ti2vPipeline.generate I2V: DiT conditioning (mock Sessions — the form of diffusers' expand_timesteps)", () => {
  it("the condition mask is true only for the first latent frame's 880 tokens (1280×704 — (H/32)·(W/32)) and is built once before the loop", async () => {
    const { ditRuns, generate } = mockTi2v();
    await generate({ image: LANDSCAPE });
    assertEquals(ditRuns.length, 4);
    const masks = new Set<unknown>();
    for (const inputs of ditRuns) {
      const mask = inputs.condition_mask;
      assertEquals([mask.dtype, [...mask.shape]], ["bool", [1, 2 * FIRST_FRAME_TOKENS, 1]]);
      const head = mask.data.subarray(0, FIRST_FRAME_TOKENS);
      const tail = mask.data.subarray(FIRST_FRAME_TOKENS);
      assert(
        head.every((value) => value === 1),
        "a token of the first frame is false",
      );
      assert(
        tail.every((value) => value === 0),
        "a token outside the first frame is true",
      );
      masks.add(mask.data);
    }
    assertEquals(masks.size, 1, "the condition mask was rebuilt for each run");
  });

  it("the condition side's time is the proj of t = 0 (a different array from the generation side's, the same one for every run)", async () => {
    const { ditRuns, generate } = mockTi2v();
    await generate({ image: LANDSCAPE });
    const zero = timestepsProj(0, 256);
    const projs = new Set<unknown>();
    for (const inputs of ditRuns) {
      const condition = inputs.timesteps_proj_condition;
      assertEquals([condition.dtype, [...condition.shape]], ["f32", [1, 256]]);
      assert(sameBits(condition.data, zero), "the condition side's time is not the proj of t = 0");
      assert(
        condition.data !== inputs.timesteps_proj.data,
        "the condition side's time is the same array as the generation side's",
      );
      assert(!sameBits(inputs.timesteps_proj.data, zero), "the generation side's time is 0 too");
      projs.add(condition.data);
    }
    assertEquals(projs.size, 1, "the condition side's time was rebuilt for each run");
  });

  it("the model input has the condition latent in the first latent frame only, UniPC advances on the unreplaced latent, and copyLatents is that state (the first frame is not replaced)", async () => {
    const { ditRuns, snapshots, generate } = mockTi2v();
    const count = I2V_SHAPE.reduce((product, dim) => product * dim, 1);
    // The first frame differs from the condition too (if UniPC advanced on the replaced latent, step 1's first
    // frame would change).
    const initial = new Float32Array(count).map((_, index) => Math.fround(Math.sin(index + 1) / 2));
    await generate({ image: LANDSCAPE, latents: initial, guidance: 1 });
    const condition = conditionOf(0.25);
    // The mock DiT returns 0.1 and guidance 1 skips CFG, so the velocity is 0.1 in every element.
    const sampler = new WanUniPcSampler(
      wanUniPcSchedule(2, CONFIG.scheduler.shift, WAN_UNIPC_CONFIG.numTrainTimesteps),
      WAN_UNIPC_CONFIG,
    );
    const velocity = new Float32Array(count).fill(0.1);
    const first = sampler.step(velocity, initial);
    const second = sampler.step(velocity, first);
    const patch = wanDitPatch(48);
    assertEquals(ditRuns.length, 2);
    for (const [step, latents] of [
      [0, initial],
      [1, first],
    ] as const) {
      assert(
        sameBits(
          ditRuns[step].tokens.data,
          patchifyLatents(copyFirstFrame(latents, condition), I2V_SHAPE, patch),
        ),
        `step ${step + 1}'s model input is not "condition in the first frame only"`,
      );
    }
    assertEquals(snapshots.length, 2);
    assert(sameBits(snapshots[0], first), "step 1's copyLatents is not UniPC's unreplaced latent");
    assert(sameBits(snapshots[1], second), "step 2's copyLatents is not UniPC's unreplaced latent");
  });

  it("in CFG's two passes (guidance 5), both uncond and cond receive a model input with the condition in the first frame only", async () => {
    const { ditRuns, generate } = mockTi2v();
    const count = I2V_SHAPE.reduce((product, dim) => product * dim, 1);
    const initial = new Float32Array(count).map((_, index) => Math.fround(Math.sin(index + 1) / 2));
    await generate({ image: LANDSCAPE, latents: initial, guidance: 5 });
    const condition = conditionOf(0.25);
    // The mock DiT returns 0.1 in both passes, so the CFG combination is 0.1 in every element too (the
    // combination is made with the production function).
    const noise = new Float32Array(count).fill(0.1);
    const velocity = wanClassifierFreeGuidance(noise, noise, 5);
    const sampler = new WanUniPcSampler(
      wanUniPcSchedule(2, CONFIG.scheduler.shift, WAN_UNIPC_CONFIG.numTrainTimesteps),
      WAN_UNIPC_CONFIG,
    );
    const first = sampler.step(velocity, initial);
    const patch = wanDitPatch(48);
    // The runs go uncond → cond per step (two sequential runs with B = 1).
    assertEquals(ditRuns.length, 4);
    ditRuns.forEach((inputs, run) => {
      const step = Math.floor(run / 2);
      const latents = step === 0 ? initial : first;
      assert(
        sameBits(
          inputs.tokens.data,
          patchifyLatents(copyFirstFrame(latents, condition), I2V_SHAPE, patch),
        ),
        `step ${step + 1}'s ${
          run % 2 === 0 ? "uncond" : "cond"
        } model input is not "condition in the first frame only"`,
      );
    });
  });

  it("replaces once after the loop before passing to the VAE (the first tile's chunk 0 = the condition latent, chunk 1 = the last UniPC latent — both denormalized)", async () => {
    const { snapshots, vaeLatents, generate } = mockTi2v();
    const count = I2V_SHAPE.reduce((product, dim) => product * dim, 1);
    const initial = new Float32Array(count).map((_, index) => Math.fround(Math.sin(index + 1) / 2));
    await generate({ image: LANDSCAPE, latents: initial, guidance: 1 });
    const last = snapshots.at(-1);
    assert(last !== undefined);
    const denormalized = denormalizeWanLatents(
      copyFirstFrame(last, conditionOf(0.25)),
      WAN22_TI2V_FAMILY.generation.latents,
    );
    // `latent [48, 1, 16, 16]` of chunk 0 (first) and chunk 1 (next) of the first tile (the top-left 16×16 of the
    // latent).
    const tile = (frame: number): Float32Array => {
      const out = new Float32Array(48 * 16 * 16);
      for (let channel = 0; channel < 48; channel += 1) {
        for (let y = 0; y < 16; y += 1) {
          const from = ((channel * 2 + frame) * 44 + y) * 80;
          out.set(denormalized.subarray(from, from + 16), (channel * 16 + y) * 16);
        }
      }
      return out;
    };
    assert(
      sameBits(vaeLatents[0].data, tile(0)),
      "the first frame passed to the VAE is not the condition latent",
    );
    assert(
      sameBits(vaeLatents[1].data, tile(1)),
      "the second frame passed to the VAE is not the last UniPC latent",
    );
  });

  it("the final latents of the official form (replace initially and after each step) and the diffusers form (model input only, once at the end) are bit-identical (UniPC updates element-wise)", () => {
    const shape = [3, 3, 4, 4] as const;
    const plane = shape[2] * shape[3];
    const count = shape.reduce((product, dim) => product * dim, 1);
    const initial = new Float32Array(count).map((_, index) => Math.fround(Math.cos(index * 0.7)));
    const condition = new Float32Array(shape[0] * plane).map((_, index) =>
      Math.fround(Math.sin(index * 1.3) * 0.8 + 0.05),
    );
    // A pseudo DiT that mixes across elements (mixes the mean of all elements and a distant element — a replaced
    // element affects the velocity of other elements).
    const model = (input: Float32Array): Float32Array<ArrayBuffer> => {
      const mean = input.reduce((total, value) => total + value, 0) / input.length;
      return input.map((value, index) =>
        Math.fround(0.3 * value - 0.2 * input[(index + 5) % input.length] + 0.1 * mean),
      );
    };
    for (const steps of [2, 10, 50]) {
      const schedule = wanUniPcSchedule(steps, 5, WAN_UNIPC_CONFIG.numTrainTimesteps);
      const replace = (latents: Float32Array<ArrayBuffer>) =>
        withWanFirstFrameCondition(latents, condition, shape);
      const diffusers = new WanUniPcSampler(schedule, WAN_UNIPC_CONFIG);
      let current: Float32Array<ArrayBuffer> = initial;
      for (let step = 0; step < steps; step += 1) {
        current = diffusers.step(model(replace(current)), current);
      }
      const diffusersFinal = replace(current);
      const official = new WanUniPcSampler(schedule, WAN_UNIPC_CONFIG);
      let replaced: Float32Array<ArrayBuffer> = replace(initial);
      for (let step = 0; step < steps; step += 1) {
        replaced = replace(official.step(model(replaced), replaced));
      }
      assert(sameBits(replaced, diffusersFinal), `the final latents diverged at ${steps} steps`);
    }
  });
});

describe("Ti2vPipeline.generate I2V: request checks (mock Sessions — they fail without creating a single Session)", () => {
  const rejects = async (
    request: Partial<Ti2vGenerateRequest>,
    errorClass: new (message: string) => Error,
    message: string,
  ): Promise<Error> => {
    const { log, generate } = mockTi2v();
    const error = await assertRejects(() => generate(request), errorClass, message);
    assertEquals(
      log.filter((entry) => entry.startsWith("create:")),
      [],
    );
    return error;
  };

  it("throws ModelInputError for fit without image", async () => {
    await rejects(
      { fit: "stretch" },
      ModelInputError,
      "fit takes effect only when image is passed (how the condition image is fitted to the size — do not pass it without image)",
    );
  });

  it("a misspelled fit is a plain Error (not ModelInputError — the handling of a misspelled name)", async () => {
    const error = await rejects(
      { ...requestWith("fit", "cover"), image: LANDSCAPE },
      Error,
      "fit 'cover' is neither 'crop' nor 'stretch'",
    );
    assert(!(error instanceof ModelInputError));
  });

  it("checks a misspelled fit before whether image is present (a 'cover' without image is a plain Error too)", async () => {
    const error = await rejects(
      requestWith("fit", "cover"),
      Error,
      "fit 'cover' is neither 'crop' nor 'stretch'",
    );
    assert(!(error instanceof ModelInputError));
  });

  for (const [label, value, actual] of [
    ["null", null, "null"],
    ["a number", 42, "number"],
    ["an object without data", { width: 4, height: 4 }, "no data"],
    [
      "data is a Float32Array",
      { width: 4, height: 4, data: new Float32Array(48) },
      "data is Float32Array",
    ],
  ] as const) {
    it(`throws ModelInputError for an image that is not RGB8-shaped (${label}) (a JS caller slipping past the types — including an array of another type that passes the length check)`, async () => {
      await rejects(
        requestWith("image", value),
        ModelInputError,
        `image is not an RGB8 image (an object with a Uint8Array data, width and height) (actual: ${actual}`,
      );
    });
  }

  it("throws ModelInputError when an explicit size is outside the accepted set (one side alone or both)", async () => {
    await rejects(
      { image: LANDSCAPE, width: 832 },
      ModelInputError,
      "width 832 is not in the accepted set (1280×704 / 704×1280)",
    );
    await rejects(
      { image: LANDSCAPE, width: 832, height: 480 },
      ModelInputError,
      "832×480 is not in the accepted set (1280×704 / 704×1280)",
    );
  });

  it("throws ModelInputError for an image whose RGB8 length does not match its size", async () => {
    await rejects(
      { image: { data: new Uint8Array(10), width: 4, height: 4 } },
      ModelInputError,
      "RGB8 length 10 does not match 3×4×4",
    );
  });
});

describe("WAN22_TI2V_FAMILY's I2V plan (the plan check)", () => {
  const embeds: WanTextEmbeds = parseWanTextEmbeds(embedsAsset(3).buffer);
  const graphs = vaeChunkGraphs();
  const layout = wanVaeChunkLayout(graphs.first, graphs.next);
  const plan = (request: Partial<Ti2vGenerateRequest>) =>
    planWanGeneration(
      { prompt: "Two cats.", ...request },
      embeds,
      layout,
      CONFIG,
      WAN22_TI2V_FAMILY.generation,
      WAN22_TI2V_FAMILY.owner,
    );

  it("an omitted size is chosen by the image's aspect ratio (landscape → 1280×704, portrait → 704×1280), and the latent shape follows that size", () => {
    const landscape = plan({ image: LANDSCAPE });
    assertEquals([landscape.width, landscape.height], [1280, 704]);
    assertEquals(landscape.latentShape, [48, 9, 44, 80]);
    const portrait = plan({ image: PORTRAIT });
    assertEquals([portrait.width, portrait.height], [704, 1280]);
    assertEquals(portrait.latentShape, [48, 9, 80, 44]);
  });

  it("an explicit size takes precedence over the image's aspect ratio (704×1280 for a landscape image, width 1280 for a portrait one)", () => {
    const explicit = plan({ image: LANDSCAPE, width: 704, height: 1280 });
    assertEquals([explicit.width, explicit.height], [704, 1280]);
    assertEquals(explicit.conditionImage?.shape, [12, 1, 640, 352]);
    const widthOnly = plan({ image: PORTRAIT, width: 1280 });
    assertEquals([widthOnly.width, widthOnly.height], [1280, 704]);
  });

  it("an omitted fit means crop (official), and stretch resizes directly (the preprocessing result matches each)", () => {
    const generation = WAN22_TI2V_FAMILY.generation;
    for (const fit of [undefined, "crop", "stretch"] as const) {
      const planned = plan({ image: LANDSCAPE, ...(fit === undefined ? {} : { fit }) });
      const expected = preprocessWanI2vImage(LANDSCAPE, { fit: fit ?? "crop" }, generation);
      assert(planned.conditionImage !== undefined);
      assert(sameBits(planned.conditionImage.pixels, expected.pixels), `fit ${fit}`);
    }
    // crop and stretch give different pixels for 64×36 (16:9) → 1280×704 (20:11) (with equal values the
    // comparison above would be vacuous).
    const crop = preprocessWanI2vImage(LANDSCAPE, { fit: "crop" }, generation);
    const stretch = preprocessWanI2vImage(LANDSCAPE, { fit: "stretch" }, generation);
    assert(!sameBits(crop.pixels, stretch.pixels));
  });

  it("a request without an image has no condition image (the T2V plan is the same as without the image field)", () => {
    assertEquals(plan({}).conditionImage, undefined);
  });
});

describe("wanVaeEncoderContract (the family admission check of the three encoder graphs)", () => {
  const graphs = vaeChunkGraphs();
  const layout = wanVaeChunkLayout(graphs.first, graphs.next);
  const contract = (replaced: Readonly<Record<string, EncoderGraph>> = {}) =>
    wanVaeEncoderContract(
      encoderGraphs(replaced),
      layout,
      WAN22_TI2V_FAMILY.generation,
      WAN22_TI2V_FAMILY.owner,
    );

  it("control: the series' declarations pass, and the symbol names, channel counts and input scale are read from the graphs", () => {
    const admitted = contract();
    assertEquals(
      [admitted.hiddenChannels, admitted.latentChannels, admitted.patchChannels],
      [640, 48, 12],
    );
    assertEquals([admitted.inputScale, admitted.compression], [8, 16]);
    assertEquals(admitted.attn.sequence, "S");
    assertEquals([admitted.post.height, admitted.post.width], ["h", "w"]);
  });

  it("fails when pre's input scale differs from the decoder's spatial compression (16 ÷ patchify 2 = 8)", () => {
    assertThrows(
      () =>
        contract({
          vae_encoder_pre: {
            ...ENCODER_GRAPHS.vae_encoder_pre,
            input: { name: "image", shape: [12, 1, "4h", "4w"] },
          },
        }),
      Error,
      `${OWNER} vae_encoder_pre input 'image' shape [12, 1, 4h, 4w] differs from the host's [12, 1, 8h, 8w]`,
    );
  });

  it("fails when post's output channels differ from the decoder's latent 48", () => {
    assertThrows(
      () =>
        contract({
          vae_encoder_post: { ...ENCODER_GRAPHS.vae_encoder_post, output: [96, 1, "h", "w"] },
        }),
      Error,
      `${OWNER} vae_encoder_post output 'out' shape [96, 1, h, w] differs from the host's [48, 1, h, w]`,
    );
  });

  it("fails when attn has symbols other than the sequence-length symbol (the only binding source is the S the host passes)", () => {
    assertThrows(
      () =>
        contract({
          vae_encoder_attn: { ...ENCODER_GRAPHS.vae_encoder_attn, symbols: ["S", "T"] },
        }),
      Error,
      `${OWNER} vae_encoder_attn symbols are [S, T] (expected: [S])`,
    );
  });

  it("fails when a spatial axis of post's input is not a bare symbol (h, w could not be passed through bindings)", () => {
    assertThrows(
      () =>
        contract({
          vae_encoder_post: {
            ...ENCODER_GRAPHS.vae_encoder_post,
            input: { name: "hidden", shape: [640, 1, "2h", "w"] },
          },
        }),
      Error,
      `${OWNER} vae_encoder_post input 'hidden' axis 2 is not a bare symbol (2h)`,
    );
  });

  /**
   * Injects declarations `EncoderGraph` cannot express (two inputs, the output dtype). Only the components in
   * `replaced` get a replaced {@link GraphOwner}; the rest keep the series' declarations.
   */
  const contractWith = (replaced: Partial<Record<"pre" | "attn" | "post", GraphOwner>>) =>
    wanVaeEncoderContract(
      { ...encoderGraphs(), ...replaced },
      layout,
      WAN22_TI2V_FAMILY.generation,
      WAN22_TI2V_FAMILY.owner,
    );

  it("fails unless inputs → outputs are 1 → 1 (that is all the host builds and reads)", () => {
    const pre = ENCODER_GRAPHS.vae_encoder_pre;
    assertThrows(
      () =>
        contractWith({
          pre: stubModel({
            symbols: pre.symbols,
            inputs: [pre.input, { name: "extra", shape: [1] }],
            outputs: ["out"],
            values: { out: pre.output },
          }),
        }),
      Error,
      `${OWNER} vae_encoder_pre has 2 inputs → 1 outputs (only 1 → 1 is built)`,
    );
  });

  it("fails with two outputs too (stops a regression that compares only the input count — the first output's shape is correct)", () => {
    const post = ENCODER_GRAPHS.vae_encoder_post;
    assertThrows(
      () =>
        contractWith({
          post: stubModel({
            symbols: post.symbols,
            inputs: [post.input],
            outputs: ["out", "extra"],
            values: { out: post.output, extra: post.output },
          }),
        }),
      Error,
      `${OWNER} vae_encoder_post has 1 inputs → 2 outputs (only 1 → 1 is built)`,
    );
  });

  it("fails when the input dtype is not f32", () => {
    const attn = ENCODER_GRAPHS.vae_encoder_attn;
    assertThrows(
      () =>
        contractWith({
          attn: stubModel({
            symbols: attn.symbols,
            inputs: [{ ...attn.input, dtype: "i32" }],
            outputs: ["out"],
            values: { out: attn.output },
          }),
        }),
      Error,
      `${OWNER} vae_encoder_attn input 'tokens' has dtype i32, not f32`,
    );
  });

  it("fails when the output dtype is not f32", () => {
    const { graph } = encoderGraphs().post;
    assertThrows(
      () =>
        contractWith({
          post: {
            graph: { ...graph, values: { out: { ...graph.values.out, dtype: "bool" } } },
          },
        }),
      Error,
      `${OWNER} vae_encoder_post output 'out' has dtype bool, not f32`,
    );
  });

  it("fails when pre's output channels are not static (C is read as the same static value across the three graphs)", () => {
    assertThrows(
      () =>
        contract({
          vae_encoder_pre: {
            ...ENCODER_GRAPHS.vae_encoder_pre,
            symbols: ["C", "h", "w"],
            output: ["C", 1, "h", "w"],
          },
        }),
      Error,
      `${OWNER} vae_encoder_pre output 'out' axis 0 is not static (C)`,
    );
  });

  it("fails when pre has symbols other than the spatial symbols h, w (the only binding sources are the h, w the host passes)", () => {
    assertThrows(
      () =>
        contract({
          vae_encoder_pre: { ...ENCODER_GRAPHS.vae_encoder_pre, symbols: ["h", "w", "T"] },
        }),
      Error,
      `${OWNER} vae_encoder_pre symbols are [h, w, T] (expected: [h, w])`,
    );
  });

  it("fails when attn's input C differs from pre's output 640 (the hand-off between the three graphs reinterprets the bytes)", () => {
    assertThrows(
      () =>
        contract({
          vae_encoder_attn: {
            ...ENCODER_GRAPHS.vae_encoder_attn,
            input: { name: "tokens", shape: [320, "S"] },
          },
        }),
      Error,
      `${OWNER} vae_encoder_attn input 'tokens' shape [320, S] differs from the host's [640, S]`,
    );
  });

  it("fails when attn's output C differs from pre's output 640", () => {
    assertThrows(
      () =>
        contract({
          vae_encoder_attn: { ...ENCODER_GRAPHS.vae_encoder_attn, output: [320, "S"] },
        }),
      Error,
      `${OWNER} vae_encoder_attn output 'out' shape [320, S] differs from the host's [640, S]`,
    );
  });

  it("fails when post's input C differs from pre's output 640", () => {
    assertThrows(
      () =>
        contract({
          vae_encoder_post: {
            ...ENCODER_GRAPHS.vae_encoder_post,
            input: { name: "hidden", shape: [320, 1, "h", "w"] },
          },
        }),
      Error,
      `${OWNER} vae_encoder_post input 'hidden' shape [320, 1, h, w] differs from the host's [640, 1, h, w]`,
    );
  });

  it("fails when one of the accepted sizes is not divisible by the spatial compression 16 (the bindings must cover every accepted size)", () => {
    const generation = {
      ...WAN22_TI2V_FAMILY.generation,
      acceptedSizes: [...WAN22_TI2V_FAMILY.generation.acceptedSizes, { width: 1288, height: 704 }],
    };
    assertThrows(
      () => wanVaeEncoderContract(encoderGraphs(), layout, generation, WAN22_TI2V_FAMILY.owner),
      Error,
      `${OWNER} 1288×704 with the VAE encoder's spatial compression 16 does not give a latent of positive integers`,
    );
  });
});

describe("Ti2vPipeline.fromAssets: the I2V VAE encoder components (family admission)", () => {
  it("2.2 assets without the encoder components fail saying that component 'vae_encoder_pre' has no container (they are required even when no image is passed)", async () => {
    const assets = Object.fromEntries(
      Object.entries(VALID_ASSETS).filter(([key]) => !key.startsWith("vae_encoder_")),
    );
    await assertRejects(
      () =>
        Ti2vPipeline.fromAssets(
          { manifest: ti2vManifest(), assets },
          { textEncoder: "precomputed", gpu: fakeGpuContext(fakeDevice()) },
        ),
      Error,
      `${OWNER} component 'vae_encoder_pre' has no container`,
    );
  });

  it("a mismatched encoder declaration (pre's input scale 4) fails at construction (admission goes through the encoder check)", async () => {
    const encoder = await encoderAssets({
      vae_encoder_pre: {
        ...ENCODER_GRAPHS.vae_encoder_pre,
        input: { name: "image", shape: [12, 1, "4h", "4w"] },
      },
    });
    await assertRejects(
      () => buildTi2v(ti2vManifest(), encoder),
      Error,
      `${OWNER} vae_encoder_pre input 'image' shape [12, 1, 4h, 4w] differs from the host's [12, 1, 8h, 8w]`,
    );
  });
});

// ---------------------------------------------------------------------------------------------
// The VAE decoder's 32-latent tile (ADR 0134 — `vaeTile`: a second chunk-graph pair that borrows the
// 16-latent pair's weights)
// ---------------------------------------------------------------------------------------------

const FIRST_32 = "vae_decoder_first_32";
const NEXT_32 = "vae_decoder_next_32";
const TILE32_ROLES = [FIRST_32, NEXT_32] as const;

/** The 32-latent pair's containers (`tile` 16 puts the 16-latent geometry under the `_32` names). */
const tile32Assets = async (tile = 32) => ({
  ...(await vaeAssets(FIRST_32, 1, VAE_22_CHANNELS, tile)),
  ...(await vaeAssets(NEXT_32, 4, VAE_22_CHANNELS, tile)),
});

/** The overlaps (latents) between neighbouring tiles of one axis of a tile plan. */
const overlapsOf = (axis: { readonly tile: number; readonly starts: readonly number[] }) =>
  axis.starts.slice(1).map((start, index) => axis.tile - (start - axis.starts[index]));

describe("Ti2vPipeline: the 32-latent decoder pair in family admission (ADR 0134)", () => {
  it("a preset whose weights carry both _32 roles builds (fromAssets)", async () => {
    const pipeline = await buildTi2v(
      ti2vManifest({ extraComponents: TILE32_ROLES }),
      await tile32Assets(),
    );
    assert(pipeline instanceof Ti2vPipeline);
    await pipeline.dispose();
  });

  for (const [present, missing] of [
    [FIRST_32, NEXT_32],
    [NEXT_32, FIRST_32],
  ] as const) {
    it(`a preset carrying '${present}' without '${missing}' fails loudly before any container is opened`, async () => {
      const error = await assertRejects(
        // No _32 container at all: the pair check comes before the components are opened.
        () => buildTi2v(ti2vManifest({ extraComponents: [present] })),
        Error,
        `${OWNER} quant 'f16': the weights carry '${present}' without '${missing}'` +
          " (the tile-32 VAE decoder is a pair — both roles or neither)",
      );
      assert(!(error instanceof ModelInputError), "a broken manifest is not an input error");
    });
  }

  it("a _32 pair that declares a 16-latent tile is rejected in admission", async () => {
    await assertRejects(
      async () =>
        buildTi2v(ti2vManifest({ extraComponents: TILE32_ROLES }), await tile32Assets(16)),
      Error,
      `${OWNER} the tile-32 VAE decoder (${FIRST_32} / ${NEXT_32}): the latent tile is 16, not 32`,
    );
  });

  it("a _32 pair whose exit differs from the 16-latent pair's (RGB, no unpatchify) is rejected naming the pair", async () => {
    const assets = {
      ...(await vaeAssets(FIRST_32, 1, { latent: 48, sample: 3 }, 32)),
      ...(await vaeAssets(NEXT_32, 4, { latent: 48, sample: 3 }, 32)),
    };
    await assertRejects(
      () => buildTi2v(ti2vManifest({ extraComponents: TILE32_ROLES }), assets),
      Error,
      `${OWNER} the tile-32 VAE decoder (${FIRST_32} / ${NEXT_32}): scale 256 / 32, 48 → 3 channels` +
        " differ from the tile-16 pair's 128 / 16, 48 → 12",
    );
  });

  it("fromPretrained: a distribution carrying the pair builds; a lone _32 role fails before fetching", async () => {
    const fromPretrained = (source: DistributionSource) =>
      Ti2vPipeline.fromPretrained(source, {
        textEncoder: "precomputed",
        gpu: fakeGpuContext(fakeDevice()),
      });
    const pipeline = await fromPretrained(await ti2vDirectory("wan-ti2v/1", TILE32_ROLES));
    assert(pipeline instanceof Ti2vPipeline);
    await pipeline.dispose();
    await assertRejects(
      async () => fromPretrained(await ti2vDirectory("wan-ti2v/1", [NEXT_32])),
      Error,
      `${OWNER} quant 'f16': the weights carry '${NEXT_32}' without '${FIRST_32}'`,
    );
  });
});

describe("wanVaeTile32Contract: every weight of the 32-latent pair is borrowed from the lender of its role", () => {
  type Initializers = GraphOwner["graph"]["initializers"];
  const withInitializers = (owner: GraphOwner, initializers: Initializers): GraphOwner => ({
    graph: { ...owner.graph, initializers },
  });
  const OWNED: Initializers = {
    "w.a": { storage: { codec: "f16" } },
    "const.0": { storage: { codec: "f32" } },
  };
  const SHARED: Initializers = { "w.a": { shared: true }, "const.0": { shared: true } };
  const lenders = (initializers: Initializers = OWNED) => {
    const graphs = vaeChunkGraphs();
    return {
      first: withInitializers(graphs.first, initializers),
      next: withInitializers(graphs.next, initializers),
    };
  };
  const pair = (first: Initializers, next: Initializers = SHARED) => {
    const graphs = vaeChunkGraphs(32);
    return {
      first: withInitializers(graphs.first, first),
      next: withInitializers(graphs.next, next),
    };
  };
  const contract = (borrowers: ReturnType<typeof pair>, lent = lenders()) =>
    wanVaeTile32Contract(
      borrowers,
      lent,
      wanVaeChunkLayout(lent.first, lent.next),
      WAN22_TI2V_FAMILY.generation,
      "wan.Ti2vPipeline",
    );

  it("names every shared initializer of each borrower (the borrower's name is the lender's) and the 32-latent layout", () => {
    const settled = contract(pair(SHARED, { "w.a": { shared: true } }));
    assertEquals(settled.borrowed, { first: ["w.a", "const.0"], next: ["w.a"] });
    assertEquals([settled.layout.tile, settled.layout.sampleTile], [32, 256]);
  });

  it("an initializer the borrower owns is rejected (the pair holds no weight bytes)", () => {
    assertThrows(
      () => contract(pair({ ...SHARED, "w.b": { storage: { codec: "f16" } } })),
      Error,
      `wan.Ti2vPipeline: ${FIRST_32}: 1 initializers are not shared (w.b…)`,
    );
  });

  it("a shared initializer the lender does not have is rejected", () => {
    assertThrows(
      () => contract(pair(SHARED, { ...SHARED, "w.z": { shared: true } })),
      Error,
      `wan.Ti2vPipeline: ${NEXT_32}: 1 shared initializers have no weight of the same name in the lender (w.z…)`,
    );
  });

  it("a pair whose spatial compression differs from the lenders' is rejected (the plan tiles the 16-latent latent shape with the pair's geometry)", () => {
    const lent = lenders();
    const lenderLayout = wanVaeChunkLayout(lent.first, lent.next);
    assertThrows(
      () =>
        wanVaeTile32Contract(
          pair(SHARED),
          lent,
          { ...lenderLayout, sampleTile: 64 },
          WAN22_TI2V_FAMILY.generation,
          "wan.Ti2vPipeline",
        ),
      Error,
      `wan.Ti2vPipeline: the tile-32 VAE decoder (${FIRST_32} / ${NEXT_32}): scale 256 / 32, 48 → 12 channels` +
        " differ from the tile-16 pair's 64 / 16, 48 → 12",
    );
  });

  it("a lender initializer that is itself borrowed does not back a borrower (no borrowing chains)", () => {
    assertThrows(
      () =>
        contract(
          pair(SHARED),
          lenders({ "w.a": { shared: true }, "const.0": { storage: { codec: "f32" } } }),
        ),
      Error,
      `wan.Ti2vPipeline: ${FIRST_32}: 1 shared initializers have no weight of the same name in the lender (w.a…)`,
    );
  });
});

describe("planWanRequest: vaeTile picks the decoder pair's geometry (ADR 0134 decisions 2 and 4)", () => {
  const embeds: WanTextEmbeds = parseWanTextEmbeds(embedsAsset(3).buffer);
  const layout16 = (() => {
    const graphs = vaeChunkGraphs();
    return wanVaeChunkLayout(graphs.first, graphs.next);
  })();
  const layout32 = (() => {
    const graphs = vaeChunkGraphs(32);
    return wanVaeChunkLayout(graphs.first, graphs.next);
  })();
  const plan = (
    request: Partial<Ti2vGenerateRequest>,
    tile32: { readonly layout: typeof layout32 | undefined; readonly quant: string } = {
      layout: layout32,
      quant: "f16",
    },
  ) =>
    planWanGeneration(
      { prompt: "Two cats.", ...request },
      embeds,
      layout16,
      CONFIG,
      WAN22_TI2V_FAMILY.generation,
      WAN22_TI2V_FAMILY.owner,
      tile32,
    );

  it("omitted and 16 plan the 16-latent pair: 4 rows × 7 cols at 1280×704", () => {
    for (const request of [{}, { vaeTile: 16 as const }]) {
      const planned = plan(request);
      assertEquals(planned.vaeTile, 16);
      assertEquals([planned.tiles.rows.tile, planned.tiles.cols.tile], [16, 16]);
      assertEquals([planned.tiles.rows.starts.length, planned.tiles.cols.starts.length], [4, 7]);
    }
  });

  it("32 plans the 32-latent pair: 2 rows × 3 cols at 1280×704 (3 × 2 at 704×1280), every overlap at least 4 latents", () => {
    const wide = plan({ vaeTile: 32 });
    assertEquals(wide.vaeTile, 32);
    assertEquals([wide.tiles.rows.tile, wide.tiles.cols.tile], [32, 32]);
    assertEquals([wide.tiles.rows.starts.length, wide.tiles.cols.starts.length], [2, 3]);
    const tall = plan({ vaeTile: 32, width: 704, height: 1280 });
    assertEquals([tall.tiles.rows.starts.length, tall.tiles.cols.starts.length], [3, 2]);
    for (const planned of [wide, tall]) {
      for (const axis of [planned.tiles.rows, planned.tiles.cols]) {
        const overlaps = overlapsOf(axis);
        assert(
          overlaps.every((overlap) => overlap >= 4),
          `overlap below 4 latents: [${overlaps.join(", ")}] (starts [${axis.starts.join(", ")}])`,
        );
      }
    }
  });

  for (const value of [24, 0, "32"] as const) {
    it(`vaeTile ${JSON.stringify(value)} is a ModelInputError`, () => {
      assertThrows(
        () => plan(requestWith("vaeTile", value)),
        ModelInputError,
        `vaeTile ${String(value)} is neither 16 nor 32`,
      );
    });
  }

  it("32 without the 32-latent pair is a ModelInputError naming the preset (16 still plans)", () => {
    const absent = { layout: undefined, quant: "f16+dit8" };
    assertThrows(
      () => plan({ vaeTile: 32 }, absent),
      ModelInputError,
      "vaeTile 32 is not available for quant 'f16+dit8': the manifest has no tile-32 VAE decoder" +
        " (no vae_decoder_first_32 / vae_decoder_next_32 weights — use vaeTile 16)",
    );
    assertEquals(plan({ vaeTile: 16 }, absent).vaeTile, 16);
  });
});

describe("Ti2vPipeline.generate with vaeTile 32 (mock Sessions — the 16 pair lends, the 32 pair borrows and runs)", () => {
  /** The construction, lending and disposal entries of the decoder stage, in order. */
  const decoderLifecycle = (log: readonly string[]) =>
    log.filter(
      (entry) => /^(create|lend|dispose):vae_decoder/.test(entry) || entry.startsWith("release:"),
    );

  it("lenders → borrowers (every borrowed weight lent by name) → residents; borrowers → lenders → residents on the way out; 6 tiles", async () => {
    const { log, diagnosed, vaeLatents, generate } = mockTi2v({ tile32: {} });
    const video = await generate({ vaeTile: 32 });
    assertEquals([video.frames, video.width, video.height], [5, 1280, 704]);
    const lent = (by: string, name: string) => ({ lentBy: by, initializer: name });
    assertEquals(decoderLifecycle(log), [
      "create:vae_decoder_first:{}",
      "create:vae_decoder_next:{}",
      "lend:vae_decoder_first:first.w",
      `create:${FIRST_32}:${JSON.stringify({ sharedWeights: { "first.w": lent("vae_decoder_first", "first.w") } })}`,
      "lend:vae_decoder_next:next.w",
      "lend:vae_decoder_next:const.0",
      `create:${NEXT_32}:${JSON.stringify({
        sharedWeights: {
          "next.w": lent("vae_decoder_next", "next.w"),
          "const.0": lent("vae_decoder_next", "const.0"),
        },
      })}`,
      `dispose:${FIRST_32}`,
      `dispose:${NEXT_32}`,
      "dispose:vae_decoder_first",
      "dispose:vae_decoder_next",
      "release:cache_00",
      "release:frame 0",
      "release:frame 1",
    ]);
    assert(
      log.indexOf("resident:cache_00") >
        log.indexOf(
          `create:${NEXT_32}:${JSON.stringify({ sharedWeights: { "next.w": lent("vae_decoder_next", "next.w"), "const.0": lent("vae_decoder_next", "const.0") } })}`,
        ),
      "the residents are allocated after the borrowers",
    );
    assertEquals(log.filter((entry) => entry.startsWith("tile:")).length, 6);
    // The lenders never run; each tile runs the borrowers (first, then one next — 5 frames = 2 chunks).
    assertEquals(
      log.filter((entry) => entry.startsWith("enqueue:vae_decoder")),
      Array.from({ length: 6 }, () => [`enqueue:${FIRST_32}`, `enqueue:${NEXT_32}`]).flat(),
    );
    assertEquals(vaeLatents[0].shape, [48, 1, 32, 32]);
    assertEquals(
      diagnosed.filter((component) => component.startsWith("vae_decoder")),
      Array.from({ length: 6 }, () => [FIRST_32, NEXT_32]).flat(),
    );
    assertEquals(log.at(-1), "vae_decoder:end");
  });

  it("the blended 32-latent tiles unpatchify like the 16-latent ones (RGB pixel (y, x) = patch channel c·4 + (x%2)·2 + (y%2))", async () => {
    const { generate } = mockTi2v({ tile32: {} });
    const video = await generate({ vaeTile: 32 });
    let worst = 0;
    for (let channel = 0; channel < 3; channel += 1) {
      for (let frame = 0; frame < 5; frame += 1) {
        for (let y = 0; y < 704; y += 1) {
          const row = ((channel * 5 + frame) * 704 + y) * 1280;
          for (let x = 0; x < 1280; x += 1) {
            const expected = patchValue(channel * 4 + (x % 2) * 2 + (y % 2));
            worst = Math.max(worst, Math.abs(video.data[row + x] - expected));
          }
        }
      }
    }
    assert(worst <= 1e-6, `a pixel differs from the unpatchify order (max diff ${worst})`);
  });

  it("lenders and borrowers both take the decoder's options (the conv knob only)", async () => {
    const { log, generate } = mockTi2v({
      tile32: { sessionOptions: { convCompute: "a8", linearCompute: "a8" } },
    });
    await generate({ vaeTile: 32 });
    const created = log.filter((entry) => entry.startsWith("create:vae_decoder"));
    assertEquals(created.slice(0, 2), [
      'create:vae_decoder_first:{"convCompute":"a8"}',
      'create:vae_decoder_next:{"convCompute":"a8"}',
    ]);
    for (const entry of created.slice(2)) {
      assert(entry.includes(':{"convCompute":"a8","sharedWeights":{'), entry);
    }
  });

  it("omitted vaeTile on a preset with the pair runs the 16-latent pair alone (no lender, no borrower, 28 tiles)", async () => {
    const { log, diagnosed, generate } = mockTi2v({ tile32: {} });
    await generate();
    assertEquals(
      log.filter((entry) => /^(create|lend):vae_decoder/.test(entry)),
      ["create:vae_decoder_first:{}", "create:vae_decoder_next:{}"],
    );
    assertEquals(log.filter((entry) => entry.startsWith("tile:")).length, 28);
    assert(
      diagnosed.every((component) => !component.endsWith("_32")),
      "a _32 component reported diagnostics on the 16-latent path",
    );
  });

  it("a borrower that fails to build: the built borrower, then both lenders are closed, the failure is thrown as is", async () => {
    const { log, generate } = mockTi2v({ tile32: { failCreate: NEXT_32 } });
    const error = await assertRejects(() => generate({ vaeTile: 32 }));
    assertStrictEquals(error, CREATE_FAILURE);
    assertEquals(
      decoderLifecycle(log).filter(
        (entry) => entry.startsWith("dispose:") || entry.startsWith("release:"),
      ),
      [`dispose:${FIRST_32}`, "dispose:vae_decoder_first", "dispose:vae_decoder_next"],
    );
    assert(
      !log.includes("resident:cache_00"),
      "no resident is allocated before the borrowers exist",
    );
  });

  it("vaeTile 32 on a pipeline without the pair is a ModelInputError before any Session is built", async () => {
    const { log, generate } = mockTi2v();
    await assertRejects(
      () => generate({ vaeTile: 32 }),
      ModelInputError,
      "vaeTile 32 is not available for quant 'f16': the manifest has no tile-32 VAE decoder",
    );
    assertEquals(
      log.filter((entry) => entry.startsWith("create:")),
      [],
    );
  });
});

// The experimental measurement opt-in (`PipelineOptions.fusionRules`): passed through, not interpreted.
describe("Ti2vPipeline fusionRules (experimental — the state's rules reach the video graphs' Sessions)", () => {
  /** The components a rule list reaches (the text encoder is not one of them). */
  const RULED = /^(transformer|vae_decoder|vae_encoder)/;

  /** Every Session's options without the `fusionRules` key (what the run without rules must match). */
  const withoutRules = (
    created: readonly { readonly component: string; readonly options: unknown }[],
  ) =>
    created.map(({ component, options }) => ({
      component,
      options: Object.fromEntries(
        Object.entries(options as Record<string, unknown>).filter(([key]) => key !== "fusionRules"),
      ),
    }));

  it("an I2V request on the GPU text path: the DiT, both decoder Sessions and the three encoder Sessions get the caller's array itself; umT5 gets none", async () => {
    const fusionRules = [vaeRmsSiluRule(8)];
    const { created, generate } = mockTi2v({ textEncoder: "gpu", fusionRules });
    await generate({ image: LANDSCAPE });
    assertEquals(
      created.map(({ component }) => component),
      [
        "vae_encoder_pre",
        "vae_encoder_attn",
        "vae_encoder_post",
        "text_encoder",
        "transformer",
        "vae_decoder_first",
        "vae_decoder_next",
      ],
    );
    for (const { component, options } of created) {
      const rules = (options as SessionOptions).fusionRules;
      if (RULED.test(component)) assertStrictEquals(rules, fusionRules, component);
      else assertEquals(Object.hasOwn(options as object, "fusionRules"), false, component);
    }
  });

  it("vaeTile 32: the two lenders and the two borrowers get the rules beside the conv knob and the shared weights", async () => {
    const fusionRules = [vaeRmsSiluRule(8)];
    const { created, generate } = mockTi2v({
      tile32: { sessionOptions: { convCompute: "a8", linearCompute: "a8" } },
      fusionRules,
    });
    await generate({ vaeTile: 32 });
    const decoders = created.filter(({ component }) => component.startsWith("vae_decoder"));
    assertEquals(decoders.length, 4);
    for (const { component, options } of decoders) {
      const given = options as SessionOptions;
      assertStrictEquals(given.fusionRules, fusionRules, component);
      assertEquals(given.convCompute, "a8", component);
      assertEquals(Object.hasOwn(given, "sharedWeights"), component.endsWith("_32"), component);
    }
  });

  it("without rules, no Session has the field and every other option is what the run with rules passed", async () => {
    const ruled = mockTi2v({ textEncoder: "gpu", fusionRules: [vaeRmsSiluRule(8)] });
    await ruled.generate({ image: LANDSCAPE });
    const plain = mockTi2v({ textEncoder: "gpu" });
    await plain.generate({ image: LANDSCAPE });
    for (const { component, options } of plain.created) {
      assertEquals(Object.hasOwn(options as object, "fusionRules"), false, component);
    }
    assertEquals(plain.created, withoutRules(ruled.created));
  });
});
