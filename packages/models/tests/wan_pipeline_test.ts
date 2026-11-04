// The GPU-free parts of the Wan2.1 pipeline (`src/wan/`) — the input checks (the asset path and the
// GPU path), the `pipelineConfig` check, the family admission's graph-declaration checks (DiT and
// umT5), the construction entry (the path spelling, the umT5 declaration, abort), where the default
// negative comes from, the text-embedding asset checks, latent denormalisation, RGBA conversion of
// frames, the RNG, and `generate` run on mock Sessions (the order and teardown of the text stage, the
// cleanup and non-finite checks, abort). The real-GPU run is e2e_wan_pipeline_test.ts.

import { readTextFile } from "@karume/test-support/fs";
import {
  assert,
  assertEquals,
  assertInstanceOf,
  assertNotEquals,
  assertRejects,
  assertStrictEquals,
  assertStringIncludes,
  assertThrows,
} from "@std/assert";
import { describe, it, test } from "vite-plus/test";
import { parseManifest } from "@karume/hub";
import type { CodecName, RunInputs, Tensor } from "@karume/runtime";
import { ModelInputError } from "../src/errors.ts";
import { type GenerateRequest, Pipeline, type PipelineOptions } from "../src/wan/pipeline.ts";
import { planWanGeneration, planWanGpuGeneration, WAN21_FAMILY } from "../src/wan/family.ts";
import { ditContract, ditInputs, wanDitPatch } from "../src/wan/dit-loop.ts";
import { umt5Contract, WAN_DEFAULT_NEGATIVE_PROMPT } from "../src/wan/text-stage.ts";
import {
  WAN21_GENERATION,
  WAN22_TI2V_GENERATION,
  type WanDitInputForm,
  type WanGenerationDescriptor,
} from "../src/wan/descriptor.ts";
import { planWanRequest, type PromptGate } from "../src/wan/plan.ts";
import {
  assertWanVaeMatchesGeneration,
  assertWanVaeTilesCover,
  planWanGenerationTiles,
  wanSpatialCompression,
} from "../src/wan/tile-decode.ts";
import type { GraphOwner } from "../src/hub/components.ts";
import { PromptCleanError } from "../src/wan/text/prompt-clean.ts";
import { wanParityCase, wanParityCases, wanParityEncoder } from "./helpers/wan-parity-encoder.ts";
import {
  declaredContainer,
  parseIrDeclarationValue,
  partAssets,
  tensorlessContainer,
  writeContainer,
} from "./helpers/container-fixture.ts";
import { mirrorPresent, mirrorRoot, readMirrorAsset } from "./helpers/mirror.ts";
import { readTextIfPresent } from "./helpers/read-if-present.ts";
import { fakeDevice, fakeGpuContext } from "../../runtime/tests/helpers/fake-gpu.ts";
import { WAN_UNIPC_CONFIG, wanUniPcSchedule } from "../src/wan/scheduler.ts";
import { WanVaeChunkError, wanVaeChunkLayout } from "../src/wan/vae-chunks.ts";
import type { WanRopeBase } from "../src/wan/dit-rope.ts";
import { type StubDim, stubModel } from "./helpers/stub-model.ts";
import { WAN_ASSEMBLE_COMMAND } from "./helpers/wan-distribution.ts";
import {
  findWanTextEmbedding,
  padWanTextEmbedding,
  parseWanTextEmbeds,
  WAN_TEXT_EMBEDS_METADATA_KEY,
  type WanTextEmbeds,
  WanTextEmbedsError,
} from "../src/wan/text-embeds.ts";
import { denormalizeWanLatents, WAN_LATENTS_MEAN, WAN_LATENTS_STD } from "../src/wan/latents.ts";
import { frameToRgba } from "../src/wan/frames.ts";
import { WanRandn } from "../src/wan/random.ts";
import { parseWanPipelineConfig, type WanPipelineConfig } from "../src/wan/config.ts";

/** One row of the embedding asset (a small width for tests). */
type Row = {
  readonly name: string;
  readonly role: string;
  readonly prompt: string;
  readonly normalized: string;
  readonly tokens: number;
};

const WIDTH = 3;

const ROWS: readonly Row[] = [
  { name: "cats", role: "positive", prompt: "Two cats.", normalized: "Two cats.", tokens: 2 },
  {
    name: "ferret",
    role: "positive",
    prompt: "\nA ferret,\nleaps.\n",
    normalized: "A ferret, leaps.",
    tokens: 3,
  },
  {
    name: "negative",
    role: "negative",
    prompt: "色调艳丽，过曝",
    normalized: "色调艳丽,过曝",
    tokens: 1,
  },
];

/**
 * The bytes of the embedding asset (the same form as the recipe `wan/text_embeds.py` — the metadata is
 * JSON under one key). The values per row are `row index + column / 10`, so a mix-up shows in the values.
 */
const buildAsset = (
  rows: readonly Row[],
  options: {
    readonly metadata?: Record<string, string>;
    readonly tensors?: readonly { name: string; shape: number[] }[];
  } = {},
): ArrayBuffer => {
  const tensors =
    options.tensors ?? rows.map((row) => ({ name: row.name, shape: [row.tokens, WIDTH] }));
  const header: Record<string, unknown> = {
    __metadata__: options.metadata ?? {
      [WAN_TEXT_EMBEDS_METADATA_KEY]: JSON.stringify({ prompts: rows, source: {} }),
    },
  };
  const payloads: Float32Array[] = [];
  let offset = 0;
  tensors.forEach(({ name, shape }, index) => {
    const data = new Float32Array(shape[0] * shape[1]).map((_, at) => index + (at % WIDTH) / 10);
    header[name] = { dtype: "F32", shape, data_offsets: [offset, offset + data.byteLength] };
    offset += data.byteLength;
    payloads.push(data);
  });
  const json = new TextEncoder().encode(JSON.stringify(header));
  const headerLength = json.length + ((8 - (json.length % 8)) % 8);
  const buffer = new ArrayBuffer(8 + headerLength + offset);
  const bytes = new Uint8Array(buffer);
  new DataView(buffer).setBigUint64(0, BigInt(headerLength), true);
  bytes.set(json, 8);
  bytes.fill(0x20, 8 + json.length, 8 + headerLength);
  let cursor = 8 + headerLength;
  for (const data of payloads) {
    bytes.set(new Uint8Array(data.buffer), cursor);
    cursor += data.byteLength;
  }
  return buffer;
};

const EMBEDS: WanTextEmbeds = parseWanTextEmbeds(buildAsset(ROWS));
/** The geometry of the distributed VAE chunk graph (latent tile 32, scale 8 — `8t / t`, output RGB 3). */
const LAYOUT = { latentChannels: 16, tile: 32, sampleTile: 256, sampleChannels: 3 };
/** The distribution's `pipelineConfig` (`WAN_PIPELINE_CONFIG` of the recipe `wan/distribution.py` — the reference setting). */
const CONFIG: WanPipelineConfig = { scheduler: { shift: 3 }, defaults: { steps: 50, guidance: 5 } };
const plan = (request: Partial<GenerateRequest>) =>
  planWanGeneration(
    { prompt: "Two cats.", ...request },
    EMBEDS,
    LAYOUT,
    CONFIG,
    WAN21_FAMILY.generation,
    WAN21_FAMILY.owner,
  );

describe("text-embedding asset", () => {
  it("returns the rows in metadata order, each row an f32 [tokens, width]", () => {
    assertEquals(EMBEDS.width, WIDTH);
    assertEquals(
      EMBEDS.entries.map((entry) => [entry.name, entry.tokens]),
      [
        ["cats", 2],
        ["ferret", 3],
        ["negative", 1],
      ],
    );
    assertEquals(
      [...EMBEDS.entries[1].data],
      [1, 1.1, 1.2, 1, 1.1, 1.2, 1, 1.1, 1.2].map(Math.fround),
    );
  });

  it("looks up the same row by the original or the normalised string, and nothing outside the set", () => {
    assertEquals(findWanTextEmbedding(EMBEDS, "\nA ferret,\nleaps.\n")?.name, "ferret");
    assertEquals(findWanTextEmbedding(EMBEDS, "A ferret, leaps.")?.name, "ferret");
    assertEquals(findWanTextEmbedding(EMBEDS, "A ferret, leaps"), undefined);
    assertEquals(
      findWanTextEmbedding(EMBEDS, "ferret"),
      undefined,
      "the name is not in the accepted set",
    );
  });

  it("pads with zeros up to the context row count (strictly zero past the valid length)", () => {
    const padded = padWanTextEmbedding(EMBEDS.entries[0], 4, WIDTH);
    assertEquals([...padded.subarray(0, 6)], [...EMBEDS.entries[0].data]);
    assertEquals([...padded.subarray(6)], [0, 0, 0, 0, 0, 0]);
    assertThrows(
      () => padWanTextEmbedding(EMBEDS.entries[1], 2, WIDTH),
      WanTextEmbedsError,
      "valid length",
    );
    assertThrows(
      () => padWanTextEmbedding(EMBEDS.entries[1], 4, WIDTH + 1),
      WanTextEmbedsError,
      "width",
    );
  });

  it("does not read an asset whose metadata and tensors no longer correspond", () => {
    const cases: readonly [string, ArrayBuffer][] = [
      ["Metadata keys", buildAsset(ROWS, { metadata: { other: "{}" } })],
      [
        "Metadata keys",
        buildAsset(ROWS, {
          metadata: { [WAN_TEXT_EMBEDS_METADATA_KEY]: JSON.stringify({ prompts: ROWS }), x: "1" },
        }),
      ],
      ["JSON", buildAsset(ROWS, { metadata: { [WAN_TEXT_EMBEDS_METADATA_KEY]: "{" } })],
      [
        "'cats' is not F32 [2, width]",
        buildAsset(ROWS, {
          tensors: [
            { name: "cats", shape: [3, WIDTH] },
            { name: "ferret", shape: [3, WIDTH] },
            {
              name: "negative",
              shape: [1, WIDTH],
            },
          ],
        }),
      ],
      [
        "width",
        buildAsset(ROWS, {
          tensors: [
            { name: "cats", shape: [2, WIDTH] },
            { name: "ferret", shape: [3, WIDTH + 1] },
            { name: "negative", shape: [1, WIDTH] },
          ],
        }),
      ],
      [
        "tensor 'negative' is missing",
        buildAsset(ROWS, {
          tensors: [
            { name: "cats", shape: [2, WIDTH] },
            { name: "ferret", shape: [3, WIDTH] },
          ],
        }),
      ],
      [
        "are not in the metadata",
        buildAsset(ROWS, {
          tensors: [
            ...ROWS.map((row) => ({ name: row.name, shape: [row.tokens, WIDTH] })),
            {
              name: "extra",
              shape: [1, WIDTH],
            },
          ],
        }),
      ],
      ["role", buildAsset([{ ...ROWS[0], role: "neutral" }])],
      [
        "duplicate",
        buildAsset([ROWS[0], ROWS[0]], { tensors: [{ name: "cats", shape: [2, WIDTH] }] }),
      ],
      ["the row is ambiguous", buildAsset([ROWS[0], { ...ROWS[1], normalized: "Two cats." }])],
    ];
    for (const [message, asset] of cases) {
      assertThrows(() => parseWanTextEmbeds(asset), Error, message);
    }
  });

  it("rejects an asset with even one non-finite value at load time (it is not carried into the DiT stage)", () => {
    // The GPU path rejects non-finite umT5 output immediately, but the precomputed path only checked
    // the shape and the metadata. A broken embedding would fail as non-finite latents after loading
    // and running the DiT, misattributing the stage at fault.
    for (const bad of [Number.NaN, Number.POSITIVE_INFINITY, Number.NEGATIVE_INFINITY]) {
      const asset = buildAsset(ROWS);
      const ferret = parseWanTextEmbeds(asset).entries[1];
      assert(
        ferret.data.buffer === asset,
        "the row does not point into the asset's bytes (the fault injection would not land)",
      );
      // Row 1, column 1 of 'ferret' (WIDTH = 3).
      ferret.data[1 * WIDTH + 1] = bad;
      assertThrows(() => parseWanTextEmbeds(asset), WanTextEmbedsError, "'ferret' row 1, column 1");
    }
  });
});

describe("planWanGeneration (the entry check of generate)", () => {
  it("omitted fields take the reference setting (50 steps, guide 5.0, shift 3.0, 832×480, 33 frames, seed 0)", () => {
    const resolved = plan({});
    assertEquals(
      [
        resolved.steps,
        resolved.guidance,
        resolved.shift,
        resolved.frames,
        resolved.width,
        resolved.height,
      ],
      [50, 5, 3, 33, 832, 480],
    );
    assertEquals(resolved.latentShape, [16, 9, 60, 104]);
    assertEquals(resolved.initial, { kind: "seed", seed: 0 });
    assertEquals(resolved.positive.name, "cats");
    assertEquals(
      resolved.negative?.name,
      "negative",
      "negativePrompt defaults to the asset's negative row",
    );
  });

  it("portrait sizes are built without swapping the latent axes", () => {
    assertEquals(plan({ width: 480, height: 832, frames: 33 }).latentShape, [16, 9, 104, 60]);
    assertEquals(plan({ frames: 5 }).latentShape, [16, 2, 60, 104]);
  });

  it("accepts only frame counts 4n+1 within 5..81, and rejects 85 (past the limit) / 34 (not 4n+1) naming the range", () => {
    const accepted = Array.from({ length: 20 }, (_, index) => 5 + 4 * index);
    assertEquals(accepted.at(-1), 81);
    assertEquals(
      accepted.map((frames) => plan({ frames }).frames),
      accepted,
    );
    assertEquals(plan({ frames: 81 }).latentShape, [16, 21, 60, 104]);
    for (const frames of [85, 34]) {
      const error = assertThrows(() => plan({ frames }), ModelInputError, "4n+1 in 5..81");
      assert(
        !error.message.includes("stage"),
        `the message must not show a stage number: ${error.message}`,
      );
    }
  });

  it("accepts the prompt as the original or the normalised string, and outside the set it is ModelInputError", () => {
    assertEquals(plan({ prompt: "A ferret, leaps." }).positive.name, "ferret");
    assertEquals(plan({ negativePrompt: "色调艳丽，过曝" }).negative?.name, "negative");
    assertEquals(plan({ negativePrompt: "A ferret, leaps." }).negative?.name, "ferret");
    assertThrows(
      () => plan({ prompt: "A dog." }),
      ModelInputError,
      "prompt is not in the text embedding asset's set",
    );
    assertThrows(() => plan({ negativePrompt: "" }), ModelInputError, "negativePrompt");
  });

  it("guidance 1 runs no uncond, so passing negativePrompt has no effect and is rejected", () => {
    assertEquals(plan({ guidance: 1 }).negative, undefined);
    assertThrows(
      () => plan({ guidance: 1, negativePrompt: "色调艳丽,过曝" }),
      ModelInputError,
      "has no effect",
    );
    for (const guidance of [0.5, Number.NaN, Number.POSITIVE_INFINITY]) {
      assertThrows(() => plan({ guidance }), ModelInputError, "guidance");
    }
  });

  it("omitting negativePrompt is rejected for an asset without exactly one negative row", () => {
    const embeds = parseWanTextEmbeds(buildAsset(ROWS.slice(0, 2)));
    assertThrows(
      () =>
        planWanGeneration(
          { prompt: "Two cats." },
          embeds,
          LAYOUT,
          CONFIG,
          WAN21_FAMILY.generation,
          WAN21_FAMILY.owner,
        ),
      ModelInputError,
      "0 negative rows",
    );
    assertEquals(
      planWanGeneration(
        { prompt: "Two cats.", guidance: 1 },
        embeds,
        LAYOUT,
        CONFIG,
        WAN21_FAMILY.generation,
        WAN21_FAMILY.owner,
      ).negative,
      undefined,
    );
  });

  it("a size, frame count, step count or shift outside the accepted set is ModelInputError", () => {
    const rejected: readonly [Partial<GenerateRequest>, string][] = [
      [{ width: 832, height: 832 }, "accepted set"],
      [{ width: 840, height: 480 }, "accepted set"],
      [{ frames: 32 }, "frames"],
      [{ frames: 1 }, "frames"],
      [{ frames: 85 }, "frames"],
      [{ frames: 33.5 }, "frames"],
      [{ steps: 0 }, "steps"],
      [{ steps: 1.5 }, "steps"],
      [{ steps: 1001 }, "steps"],
      [{ steps: 400_000 }, "steps"],
      [{ shift: 0 }, "shift"],
      [{ shift: Number.NaN }, "shift"],
    ];
    for (const [request, message] of rejected) {
      assertThrows(() => plan(request), ModelInputError, message);
    }
    assertEquals(plan({ steps: 1 }).steps, 1);
    assertEquals(
      plan({ steps: 1000 }).steps,
      1000,
      "exactly the limit (num_train_timesteps) is inside the boundary",
    );
  });

  it("steps past the limit fail at the scalar check before the σ sequence is built (not through a low-level RangeError)", () => {
    // If the σ sequence were built, huge steps would mean a synchronous loop over several GiB, and
    // this line would never return.
    const error = assertThrows(
      () => plan({ steps: Number.MAX_SAFE_INTEGER }),
      ModelInputError,
      "in 1..1000",
    );
    assertEquals(
      error.cause,
      undefined,
      "fails at the scalar check, not as a σ-sequence failure (RangeError in cause)",
    );
  });

  it("the initial noise is exactly one of seed or latents, and latents are checked for shape and finiteness", () => {
    const count = 16 * 2 * 60 * 104;
    const latents = new Float32Array(count).fill(0.5);
    const resolved = plan({ frames: 5, latents });
    assert(resolved.initial.kind === "latents" && resolved.initial.data === latents);
    assertThrows(
      () => plan({ frames: 5, latents, seed: 1 }),
      ModelInputError,
      "mutually exclusive",
    );
    assertThrows(
      () => plan({ frames: 5, latents: new Float32Array(count - 1) }),
      ModelInputError,
      "latents",
    );
    latents[7] = Number.NaN;
    assertThrows(() => plan({ frames: 5, latents }), ModelInputError, "non-finite");
    assertThrows(() => plan({ seed: -1 }), ModelInputError, "seed");
    assertEquals(plan({ seed: 42 }).initial, { kind: "seed", seed: 42 });
  });

  it("omitted steps / guidance / shift are filled from the manifest's pipelineConfig, and explicit values win over it", () => {
    // Values that do not coincide with the reference setting (a baked-in default would give 50 / 5 / 3).
    const config: WanPipelineConfig = {
      scheduler: { shift: 7.5 },
      defaults: { steps: 23, guidance: 4.25 },
    };
    const resolved = planWanGeneration(
      { prompt: "Two cats." },
      EMBEDS,
      LAYOUT,
      config,
      WAN21_FAMILY.generation,
      WAN21_FAMILY.owner,
    );
    assertEquals([resolved.steps, resolved.guidance, resolved.shift], [23, 4.25, 7.5]);
    const explicit = planWanGeneration(
      { prompt: "Two cats.", steps: 2, guidance: 1, shift: 1.5 },
      EMBEDS,
      LAYOUT,
      config,
      WAN21_FAMILY.generation,
      WAN21_FAMILY.owner,
    );
    assertEquals([explicit.steps, explicit.guidance, explicit.shift], [2, 1, 1.5]);
    assertEquals(explicit.negative, undefined, "an explicit guidance 1 wins over the default 4.25");
  });

  it("guidance is rounded to f32 and only finite values are accepted (CFG multiplies in f32, so a value finite in f64 can still overflow)", () => {
    for (const guidance of [Number.MAX_VALUE, 1e39]) {
      assertThrows(() => plan({ guidance }), ModelInputError, "fits in f32");
    }
    const f32Max = 3.4028234663852886e38;
    assertEquals(
      plan({ guidance: f32Max }).guidance,
      f32Max,
      "the f32 maximum is inside the boundary",
    );
  });

  it("a steps × shift pair that breaks the σ sequence is ModelInputError (the low-level RangeError is kept in cause)", () => {
    // Pairs that pass the scalar checks (a positive finite shift, integer steps from 1 to 1000).
    const rejected: readonly Partial<GenerateRequest>[] = [
      { shift: 1e6 }, // σ[0] < σ[1] (the shift is so large it sticks to 1)
      { shift: 1e-320 }, // σ[0] is Infinity
    ];
    for (const request of rejected) {
      const error = assertThrows(
        () => plan(request),
        ModelInputError,
        "cannot build a UniPC σ sequence",
      );
      assertInstanceOf(error.cause, RangeError);
    }
  });

  it("the plan carries the schedule of the requested steps × shift (denoise uses it instead of rebuilding it)", () => {
    const explicit = plan({ steps: 3, shift: 1.5 });
    assertEquals(explicit.schedule, wanUniPcSchedule(3, 1.5, WAN_UNIPC_CONFIG.numTrainTimesteps));
    assertEquals(plan({}).schedule.timesteps.length, 50, "omitted steps default to 50");
  });

  it("the VAE tile plan is made before denoise (832×480 is 3×4 tiles, 480×832 is 4×3)", () => {
    const landscape = plan({}).tiles;
    assertEquals([...landscape.rows.starts], [0, 14, 28]);
    assertEquals([...landscape.cols.starts], [0, 24, 48, 72]);
    const portrait = plan({ width: 480, height: 832 }).tiles;
    assertEquals([...portrait.rows.starts], [0, 24, 48, 72]);
    assertEquals([...portrait.cols.starts], [0, 14, 28]);
  });
});

/**
 * A fake holding only the declarations of the 2 VAE chunk graphs (first / next) (latent tile `tile`,
 * scale 8, one cache — the form agreed in `vae-chunks.ts`). The channel counts default to the
 * distribution's (latent 16, output RGB 3).
 */
const vaeChunkGraphs = (
  tile: number,
  scale = 8,
  channels: { readonly latent: number; readonly sample: number } = { latent: 16, sample: 3 },
) => {
  const graph = (frames: number) =>
    stubModel({
      inputs: [
        { name: "latent", shape: [channels.latent, 1, tile, tile] },
        { name: "cache_00", shape: [16, 2, tile, tile] },
      ],
      outputs: ["frame", "cache_00_out"],
      values: {
        frame: [channels.sample, frames, scale * tile, scale * tile],
        cache_00_out: [16, 2, tile, tile],
      },
    });
  return { first: graph(1), next: graph(4) };
};

describe("family admission: the VAE tiles cover every accepted size", () => {
  const layoutOf = (tile: number, scale?: number) => {
    const { first, next } = vaeChunkGraphs(tile, scale);
    return wanVaeChunkLayout(first, next);
  };

  it("latent tile 32 (the distribution) and 60 (exactly the short side) cover both 832×480 and 480×832", () => {
    for (const tile of [32, 60]) {
      assertWanVaeTilesCover(layoutOf(tile), WAN21_FAMILY.generation, WAN21_FAMILY.owner);
    }
  });

  it("rejects, naming the size and the reason, an asset that passes the chunk-graph checks but cannot tile-decode", () => {
    // 8: the minimum overlap 8 is not below the tile width. 64: larger than the latent's short side 60.
    for (const [tile, reason] of [
      [8, "Minimum overlap"],
      [64, "smaller than tile width 64"],
    ] as const) {
      const error = assertThrows(
        () => assertWanVaeTilesCover(layoutOf(tile), WAN21_FAMILY.generation, WAN21_FAMILY.owner),
        Error,
        reason,
      );
      assert(error.message.includes("832×480"), error.message);
      assert(!(error instanceof ModelInputError), "an asset inconsistency is not an input error");
    }
  });

  it("rejects an asset whose scale makes the latent of an accepted size non-integral", () => {
    assertThrows(
      () => assertWanVaeTilesCover(layoutOf(32, 7), WAN21_FAMILY.generation, WAN21_FAMILY.owner),
      Error,
      "is not integral",
    );
  });
});

describe("family admission: the VAE graph declaration × the generation descriptor (statistics count, output channel count)", () => {
  const layoutOf = (channels: { readonly latent: number; readonly sample: number }) => {
    const { first, next } = vaeChunkGraphs(32, 8, channels);
    return wanVaeChunkLayout(first, next);
  };

  it("the distribution's declaration (latent 16, output 3) matches the Wan2.1 descriptor (16 statistics, unpatchify 1)", () => {
    assertWanVaeMatchesGeneration(
      layoutOf({ latent: 16, sample: 3 }),
      WAN21_GENERATION,
      "wan.Pipeline",
    );
  });

  it("rejects an asset whose statistics count differs from the latent channel count (the element count divides and the denormalisation would pass silently)", () => {
    const error = assertThrows(
      () =>
        assertWanVaeMatchesGeneration(
          layoutOf({ latent: 8, sample: 3 }),
          WAN21_GENERATION,
          "wan.Pipeline",
        ),
      Error,
      "wan.Pipeline: the denormalization statistics (mean 16, std 16) do not match the VAE latent's 8 channels",
    );
    assert(!(error instanceof ModelInputError), "an asset inconsistency is not an input error");
  });

  it("also rejects a descriptor where only the std count differs from the latent channel count (mean and std are checked separately)", () => {
    // A synthetic descriptor (mean has 16 entries and matches the latent; only std has 8).
    const stdOnly: WanGenerationDescriptor = {
      ...WAN21_GENERATION,
      latents: { mean: WAN21_GENERATION.latents.mean, std: Array<number>(8).fill(1) },
    };
    assertThrows(
      () =>
        assertWanVaeMatchesGeneration(layoutOf({ latent: 16, sample: 3 }), stdOnly, "wan.Pipeline"),
      Error,
      "wan.Pipeline: the denormalization statistics (mean 16, std 8) do not match the VAE latent's 16 channels",
    );
  });

  it("rejects an asset whose output channel count differs from RGB × unpatchify²", () => {
    assertThrows(
      () =>
        assertWanVaeMatchesGeneration(
          layoutOf({ latent: 16, sample: 12 }),
          WAN21_GENERATION,
          "wan.Pipeline",
        ),
      Error,
      "wan.Pipeline: VAE output 12 channels differ from RGB 3 × unpatchify 1² = 3",
    );
  });
});

describe("spatial compression (graph ratio × unpatchify): the latent size, the coverage check and the overlap use the same single value", () => {
  // A synthetic generation and asset (the values exist to pin the checks and are not kept in src):
  // latent tile 16 → output 128 (graph ratio 8), unpatchify 2 → spatial compression 16, output 12 =
  // RGB 3 × 2². Dividing by the graph ratio alone would double both the latent and the overlap.
  const SYNTHETIC: WanGenerationDescriptor = {
    ...WAN21_GENERATION,
    latents: { mean: [0, 0, 0, 0], std: [1, 1, 1, 1] },
    vaePatchSize: 2,
  };
  const graphs = vaeChunkGraphs(16, 8, { latent: 4, sample: 12 });
  const layout = wanVaeChunkLayout(graphs.first, graphs.next);
  const gate: PromptGate<string> = { resolve: (text) => text, defaultNegative: () => "negative" };
  const planOf = (generation: WanGenerationDescriptor) =>
    planWanRequest({ prompt: "p" }, gate, layout, CONFIG, generation);

  it("the synthetic declaration passes the comparison with the descriptor, and the spatial compression is 8 × 2 = 16", () => {
    assertWanVaeMatchesGeneration(layout, SYNTHETIC, "wan.Pipeline");
    assertEquals(wanSpatialCompression(layout, SYNTHETIC), 16);
  });

  it("the latent of 832×480 is 30×52, and the plan's latent shape, tile plan and overlap (64 px ÷ 16 = 4) agree", () => {
    const plan = planOf(SYNTHETIC);
    assertEquals(plan.latentShape, [4, 9, 30, 52]);
    // The plan's tiles are a plan that went through the production entry (planWanGenerationTiles) and
    // cover the same axes as the latent size (the entry's own overlap formula is pinned not by this
    // line but by the counts below).
    assertEquals(plan.tiles, planWanGenerationTiles(layout, 30, 52, SYNTHETIC));
    assertEquals([plan.tiles.rows.extent, plan.tiles.cols.extent], [30, 52]);
    // The count is the smallest that satisfies "overlap of at least 4" (taking the overlap as 8 would
    // add columns) — pinned by a property of the formula.
    const overlap = 64 / 16;
    for (const axis of [plan.tiles.rows, plan.tiles.cols]) {
      const span = axis.extent - axis.tile;
      assertEquals(axis.starts.length, Math.ceil(span / (axis.tile - overlap)) + 1);
    }
    // The coverage check also plans with the same size and passes.
    assertWanVaeTilesCover(layout, SYNTHETIC, "wan.Pipeline");
  });

  it("a size not divisible by the compression 16 (840×480 — divisible by the graph ratio 8 alone) is rejected by the check and the plan alike", () => {
    const wide: WanGenerationDescriptor = {
      ...SYNTHETIC,
      acceptedSizes: [{ width: 840, height: 480 }],
      defaults: { ...SYNTHETIC.defaults, width: 840 },
    };
    assertThrows(
      () => assertWanVaeTilesCover(layout, wide, "wan.Pipeline"),
      Error,
      "with spatial compression 16 (VAE scale 128 / 16 × unpatchify 2), the latent of 840×480 is not integral",
    );
    assertThrows(
      () => planOf(wide),
      Error,
      "Latent shape [4,9,30,52.5] is not integral (spatial compression 16)",
    );
  });
});

describe("family admission: the DiT graph declaration × the shapes the host builds (down to rank, batch and a variable S)", () => {
  /** The raw RoPE table (`ditContract` only looks at the width `2·(t + h + w)` = 128). */
  const ROPE: WanRopeBase = {
    rows: 1,
    widths: [22, 21, 21],
    cos: [new Float32Array(22), new Float32Array(21), new Float32Array(21)],
    sin: [new Float32Array(22), new Float32Array(21), new Float32Array(21)],
  };
  type DitShapes = {
    readonly tokens: readonly StubDim[];
    readonly output: readonly StubDim[];
    readonly ropeCos: readonly StubDim[];
    readonly ropeSin: readonly StubDim[];
    readonly proj: readonly StubDim[];
    readonly context: readonly StubDim[];
  };
  /** The same declaration as the distribution (`[1, S, 64]` and so on — the forward of the recipe `wan/export_dit.py`). */
  const VALID: DitShapes = {
    tokens: [1, "S", 64],
    output: [1, "S", 64],
    ropeCos: [1, "S", 1, 128],
    ropeSin: [1, "S", 1, 128],
    proj: [1, 256],
    context: [1, 512, 4096],
  };
  /** The DiT patch built from the distribution's VAE latent (16 channels). */
  const DIT_PATCH = wanDitPatch(16);
  const transformerOf = (patch: Partial<DitShapes>) => {
    const shapes = { ...VALID, ...patch };
    return stubModel({
      symbols: ["S", "T"],
      inputs: [
        { name: "tokens", shape: shapes.tokens },
        { name: "timesteps_proj", shape: shapes.proj },
        { name: "encoder_hidden_states", shape: shapes.context },
        { name: "rope_cos", shape: shapes.ropeCos },
        { name: "rope_sin", shape: shapes.ropeSin },
      ],
      outputs: ["out"],
      values: { out: shapes.output },
    });
  };

  it("the distribution's declaration passes, and the context and timestep widths are taken from the declaration", () => {
    assertEquals(ditContract(transformerOf({}), ROPE, DIT_PATCH, "t2v", "wan.Pipeline"), {
      output: "out",
      projWidth: 256,
      contextRows: 512,
      contextWidth: 4096,
      patch: DIT_PATCH,
      form: "t2v",
    });
  });

  it("the patch stride is upstream's (1, 2, 2), the channel count comes from the VAE declaration (16 gives a tokens width of 64)", () => {
    assertEquals(wanDitPatch(16), { channels: 16, patchFrames: 1, patchHeight: 2, patchWidth: 2 });
  });

  it("fails when the VAE latent channel count does not match the DiT tokens width (the patch is built from the VAE declaration)", () => {
    assertThrows(
      () => ditContract(transformerOf({}), ROPE, wanDitPatch(8), "t2v", "wan.Pipeline"),
      Error,
      "'tokens' shape",
    );
  });

  it("batch 2, a fixed S, a wrong rank and a mismatched S symbol fail by name before the Session is set up", () => {
    const rejected: readonly [string, Partial<DitShapes>, string][] = [
      ["tokens with batch 2", { tokens: [2, "S", 64] }, "'tokens' shape"],
      ["tokens with a fixed S", { tokens: [1, 192, 64] }, "is not a symbolic dimension"],
      ["tokens of rank 2", { tokens: ["S", 64] }, "is not a symbolic dimension"],
      ["tokens of rank 4", { tokens: [1, "S", 1, 64] }, "'tokens' shape"],
      ["output with batch 2", { output: [2, "S", 64] }, "transformer output 'out' shape"],
      ["output with a fixed S", { output: [1, 192, 64] }, "transformer output 'out' shape"],
      ["output with another symbol", { output: [1, "T", 64] }, "transformer output 'out' shape"],
      ["rope_cos with another symbol", { ropeCos: [1, "T", 1, 128] }, "'rope_cos' shape"],
      ["rope_sin of rank 3", { ropeSin: [1, "S", 128] }, "'rope_sin' shape"],
      ["rope_cos with batch 2", { ropeCos: [2, "S", 1, 128] }, "'rope_cos' shape"],
      ["timesteps_proj with batch 2", { proj: [2, 256] }, "'timesteps_proj' shape"],
      [
        "encoder_hidden_states with batch 2",
        { context: [2, 512, 4096] },
        "'encoder_hidden_states' shape",
      ],
      [
        "encoder_hidden_states of rank 4",
        { context: [1, 512, 4096, 1] },
        "'encoder_hidden_states' shape",
      ],
      ["the last dimension of tokens", { tokens: [1, "S", 32] }, "'tokens' shape"],
      ["the rope width", { ropeSin: [1, "S", 1, 64] }, "'rope_sin' shape"],
    ];
    for (const [label, patch, message] of rejected) {
      assertThrows(
        () => ditContract(transformerOf(patch), ROPE, DIT_PATCH, "t2v", "wan.Pipeline"),
        Error,
        message,
        label,
      );
    }
  });

  /** The 5 t2v names (`INPUT_NAMES` of `wan/export_dit.py`) — used for the expected set message. */
  const T2V_NAMES = "tokens, timesteps_proj, encoder_hidden_states, rope_cos, rope_sin";
  const TI2V_NAMES = `${T2V_NAMES}, timesteps_proj_condition, condition_mask`;

  it("the check of the input name set comes after the existing 5 checks (the message for a 2.1 asset missing an input does not change)", () => {
    const missing = stubModel({
      symbols: ["S"],
      inputs: [
        { name: "tokens", shape: VALID.tokens },
        { name: "timesteps_proj", shape: VALID.proj },
        { name: "encoder_hidden_states", shape: VALID.context },
        { name: "rope_cos", shape: VALID.ropeCos },
      ],
      outputs: ["out"],
      values: { out: VALID.output },
    });
    assertThrows(
      () => ditContract(missing, ROPE, DIT_PATCH, "t2v", "wan.Pipeline"),
      Error,
      "wan.Pipeline: transformer graph input 'rope_sin' is missing",
    );
  });

  it("fault injection: a 6th input in the 2.1 declaration fails because it does not match the t2v set", () => {
    const extra = stubModel({
      symbols: ["S"],
      inputs: [
        { name: "tokens", shape: VALID.tokens },
        { name: "timesteps_proj", shape: VALID.proj },
        { name: "encoder_hidden_states", shape: VALID.context },
        { name: "rope_cos", shape: VALID.ropeCos },
        { name: "rope_sin", shape: VALID.ropeSin },
        { name: "attention_mask", shape: [1, "S"] },
      ],
      outputs: ["out"],
      values: { out: VALID.output },
    });
    assertThrows(
      () => ditContract(extra, ROPE, DIT_PATCH, "t2v", "wan.Pipeline"),
      Error,
      `wan.Pipeline: transformer graph inputs are [${T2V_NAMES}, attention_mask]` +
        ` (expected: [${T2V_NAMES}] — input form 't2v')`,
    );
  });

  it("an unknown input form is not silently treated as t2v but fails (a value outside the type — a JS caller, a broken descriptor)", () => {
    assertThrows(
      () =>
        Reflect.apply(ditContract, undefined, [
          transformerOf({}),
          ROPE,
          DIT_PATCH,
          "i2v",
          "wan.Pipeline",
        ]),
      Error,
      `wan.Pipeline: unknown DiT input form 'i2v' ("t2v" / "ti2v")`,
    );
    // Even combined with a broken asset (a missing input), the unknown-form message comes first
    // (independent of the asset's contents).
    const broken = stubModel({
      symbols: ["S"],
      inputs: [{ name: "tokens", shape: VALID.tokens }],
      outputs: ["out"],
      values: { out: VALID.output },
    });
    assertThrows(
      () => Reflect.apply(ditContract, undefined, [broken, ROPE, DIT_PATCH, "i2v", "wan.Pipeline"]),
      Error,
      `wan.Pipeline: unknown DiT input form 'i2v' ("t2v" / "ti2v")`,
    );
  });

  describe("the input form ti2v (Wan2.2 TI2V — 5 inputs + the condition-side timestep and the condition mask)", () => {
    type Ti2vInput = {
      readonly name: string;
      readonly shape: readonly StubDim[];
      readonly dtype?: "f32" | "bool";
    };
    /** The patch of a 48-channel latent (`tokens` width 192). */
    const TI2V_PATCH = wanDitPatch(48);
    /** The same declaration as the recipe `wan/ti2v_export_dit.py` (`EXPECTED_INPUTS` of `e2e_wan_ti2v_dit_host_test.ts`). */
    const TI2V_VALID: readonly Ti2vInput[] = [
      { name: "tokens", shape: [1, "S", 192] },
      { name: "timesteps_proj", shape: [1, 256] },
      { name: "encoder_hidden_states", shape: [1, 512, 4096] },
      { name: "rope_cos", shape: [1, "S", 1, 128] },
      { name: "rope_sin", shape: [1, "S", 1, 128] },
      { name: "timesteps_proj_condition", shape: [1, 256] },
      { name: "condition_mask", shape: [1, "S", 1], dtype: "bool" },
    ];
    const ti2vOf = (inputs: readonly Ti2vInput[]) =>
      stubModel({ symbols: ["S", "T"], inputs, outputs: ["out"], values: { out: [1, "S", 192] } });
    /** Replaces one input of the valid form (omitting `replacement` removes it). */
    const replaced = (name: string, replacement?: Ti2vInput): readonly Ti2vInput[] =>
      TI2V_VALID.flatMap((input) =>
        input.name !== name ? [input] : replacement === undefined ? [] : [replacement],
      );

    it("the valid form passes and returns the form as ti2v (the numeric fields come from the declaration as for t2v)", () => {
      assertEquals(ditContract(ti2vOf(TI2V_VALID), ROPE, TI2V_PATCH, "ti2v", "wan.Ti2vPipeline"), {
        output: "out",
        projWidth: 256,
        contextRows: 512,
        contextWidth: 4096,
        patch: TI2V_PATCH,
        form: "ti2v",
      });
    });

    it("a missing condition input, a wrong dtype, shape or symbol, and a mix-up between the form and the declaration fail by name", () => {
      const rejected: readonly [string, readonly Ti2vInput[], WanDitInputForm, string][] = [
        [
          "condition_mask missing",
          replaced("condition_mask"),
          "ti2v",
          `wan.Ti2vPipeline: transformer graph inputs are [${T2V_NAMES}, timesteps_proj_condition]` +
            ` (expected: [${TI2V_NAMES}] — input form 'ti2v')`,
        ],
        [
          "condition_mask is f32",
          replaced("condition_mask", { name: "condition_mask", shape: [1, "S", 1] }),
          "ti2v",
          "wan.Ti2vPipeline: transformer graph input 'condition_mask' has dtype f32, not bool",
        ],
        [
          "condition_mask is [1, S, 2]",
          replaced("condition_mask", { name: "condition_mask", shape: [1, "S", 2], dtype: "bool" }),
          "ti2v",
          "wan.Ti2vPipeline: 'condition_mask' shape [1, S, 2] differs from the host's [1, S, 1]",
        ],
        [
          "condition_mask has a symbol other than tokens'",
          replaced("condition_mask", { name: "condition_mask", shape: [1, "T", 1], dtype: "bool" }),
          "ti2v",
          "wan.Ti2vPipeline: 'condition_mask' shape [1, T, 1] differs from the host's [1, S, 1]",
        ],
        [
          "condition-side timestep width 128",
          replaced("timesteps_proj_condition", {
            name: "timesteps_proj_condition",
            shape: [1, 128],
          }),
          "ti2v",
          "wan.Ti2vPipeline: 'timesteps_proj_condition' shape [1, 128] differs from the host's [1, 256]",
        ],
        [
          "condition-side timestep is bool",
          replaced("timesteps_proj_condition", {
            name: "timesteps_proj_condition",
            shape: [1, 256],
            dtype: "bool",
          }),
          "ti2v",
          "wan.Ti2vPipeline: transformer graph input 'timesteps_proj_condition' has dtype bool, not f32",
        ],
        [
          "opening the TI2V declaration as t2v (2 extra inputs)",
          TI2V_VALID,
          "t2v",
          `wan.Ti2vPipeline: transformer graph inputs are [${TI2V_NAMES}]` +
            ` (expected: [${T2V_NAMES}] — input form 't2v')`,
        ],
        [
          "opening the 5-input 2.1 declaration as ti2v",
          TI2V_VALID.slice(0, 5),
          "ti2v",
          `wan.Ti2vPipeline: transformer graph inputs are [${T2V_NAMES}]` +
            ` (expected: [${TI2V_NAMES}] — input form 'ti2v')`,
        ],
      ];
      for (const [label, inputs, form, message] of rejected) {
        assertThrows(
          () => ditContract(ti2vOf(inputs), ROPE, TI2V_PATCH, form, "wan.Ti2vPipeline"),
          Error,
          message,
          label,
        );
      }
    });
  });
});

describe("ditInputs (the one place that builds the inputs of one DiT forward)", () => {
  /** Takes out an input that is a host array (not a resident input). */
  const hostTensor = (inputs: RunInputs, name: string): Tensor => {
    assert(Object.hasOwn(inputs, name), `no input '${name}'`);
    const input = inputs[name];
    assert("dtype" in input, `input '${name}' is not a host array`);
    return input;
  };
  const BASE = {
    tokens: new Float32Array(6),
    tokenShape: [1, 3, 2],
    proj: new Float32Array(4),
    projShape: [1, 4],
    context: new Float32Array(2),
    contextShape: [1, 1, 2],
    rope: { cos: new Float32Array(3), sin: new Float32Array(3) },
    ropeShape: [1, 3, 1, 1],
  };

  it("without a condition (t2v) there are exactly 5 keys, keeping the order and the array identity (no copy)", () => {
    const inputs = ditInputs({ ...BASE, condition: undefined });
    assertEquals(Object.keys(inputs), [
      "tokens",
      "timesteps_proj",
      "encoder_hidden_states",
      "rope_cos",
      "rope_sin",
    ]);
    const expected: readonly [string, Float32Array, readonly number[]][] = [
      ["tokens", BASE.tokens, BASE.tokenShape],
      ["timesteps_proj", BASE.proj, BASE.projShape],
      ["encoder_hidden_states", BASE.context, BASE.contextShape],
      ["rope_cos", BASE.rope.cos, BASE.ropeShape],
      ["rope_sin", BASE.rope.sin, BASE.ropeShape],
    ];
    for (const [name, data, shape] of expected) {
      const tensor = hostTensor(inputs, name);
      assertEquals(tensor.dtype, "f32", name);
      assertStrictEquals(tensor.data, data, name);
      assertStrictEquals(tensor.shape, shape, name);
    }
  });

  it("with a condition (ti2v) there are 7: the condition-side timestep is f32 with the same shape as the generation side, the mask is bool (still borrowed)", () => {
    const mask = new Uint32Array(3);
    const maskShape = [1, 3, 1];
    const inputs = ditInputs({ ...BASE, condition: { proj: BASE.proj, mask, maskShape } });
    assertEquals(Object.keys(inputs), [
      "tokens",
      "timesteps_proj",
      "encoder_hidden_states",
      "rope_cos",
      "rope_sin",
      "timesteps_proj_condition",
      "condition_mask",
    ]);
    const proj = hostTensor(inputs, "timesteps_proj_condition");
    assertEquals(proj.dtype, "f32");
    assertStrictEquals(proj.data, BASE.proj);
    assertStrictEquals(proj.shape, BASE.projShape);
    const condition = hostTensor(inputs, "condition_mask");
    assertEquals(condition.dtype, "bool");
    assertStrictEquals(condition.data, mask);
    assertStrictEquals(condition.shape, maskShape);
  });
});

describe("the model card's accepted set (compared through a fixture)", () => {
  // The other side is the recipe's `wan/tests/test_distribution.py` (it compares card.py's table with
  // the same fixture).
  it("the accepted sizes and frame range equal the fixture wan-card-limits.json", async () => {
    const fixture: unknown = JSON.parse(
      await readTextFile(new URL("./fixtures/wan-card-limits.json", import.meta.url)),
    );
    const { acceptedSizes, minFrames, maxFrames } = WAN21_GENERATION;
    assertEquals(
      fixture,
      { acceptedSizes, minFrames, maxFrames },
      "when the accepted set of descriptor.ts changes, update the fixture and WAN_ACCEPTED_SIZES / WAN_FRAMES of card.py too",
    );
  });

  // The card of the Wan2.2 TI2V-5B distribution (`karume-wan2.2`) is pinned the same way (ADR 0121
  // step 8 — on the other side, the same recipe's test compares `WAN22_ACCEPTED_SIZES` /
  // `WAN22_FRAMES` of card.py with the same fixture).
  it("the accepted sizes and frame range of Wan2.2 TI2V equal the fixture wan-ti2v-card-limits.json", async () => {
    const fixture: unknown = JSON.parse(
      await readTextFile(new URL("./fixtures/wan-ti2v-card-limits.json", import.meta.url)),
    );
    const { acceptedSizes, minFrames, maxFrames } = WAN22_TI2V_GENERATION;
    assertEquals(
      fixture,
      { acceptedSizes, minFrames, maxFrames },
      "when the accepted set of WAN22_TI2V_GENERATION in descriptor.ts changes, update the fixture and WAN22_ACCEPTED_SIZES / WAN22_FRAMES of card.py too",
    );
  });
});

describe("pipelineConfig (the check of the manifest's declaration)", () => {
  const RAW = { scheduler: { shift: 3 }, defaults: { steps: 50, guidance: 5 } };

  it("reads the distribution's declaration as is", () => {
    assertEquals(parseWanPipelineConfig(RAW), CONFIG);
  });

  it("an unknown key, a missing field or a value out of range is a plain Error (an asset inconsistency — not an input error)", () => {
    const rejected: readonly [Record<string, unknown>, string][] = [
      [{ ...RAW, steps: 50 }, "unknown key 'steps'"],
      [{ ...RAW, scheduler: { shift: 3, type: "unipc" } }, "unknown key 'type'"],
      [{ ...RAW, defaults: { steps: 50, guidanceScale: 5 } }, "unknown key 'guidanceScale'"],
      [{ defaults: RAW.defaults }, "pipelineConfig.scheduler: missing"],
      [{ scheduler: RAW.scheduler }, "pipelineConfig.defaults: missing"],
      [{ ...RAW, scheduler: {} }, "pipelineConfig.scheduler.shift: missing"],
      [{ ...RAW, scheduler: { shift: 0 } }, "shift"],
      [{ ...RAW, defaults: { steps: 0, guidance: 5 } }, "steps"],
      [{ ...RAW, defaults: { steps: 2.5, guidance: 5 } }, "steps"],
      [{ ...RAW, defaults: { steps: 1001, guidance: 5 } }, "in 1..1000"],
      [{ ...RAW, defaults: { steps: 400_000, guidance: 5 } }, "in 1..1000"],
      [{ ...RAW, defaults: { steps: 50, guidance: 0.5 } }, "guidance"],
      [{ ...RAW, defaults: { steps: 50, guidance: "5" } }, "guidance"],
    ];
    for (const [raw, message] of rejected) {
      const error = assertThrows(() => parseWanPipelineConfig(raw), Error, message);
      assert(
        !(error instanceof ModelInputError),
        `a declaration inconsistency is not an input error: ${message}`,
      );
    }
  });

  it("the same value ranges as the request check: a guidance that overflows f32, or defaults whose steps × shift cannot build a σ sequence, are also a plain Error", () => {
    // Passing the default pair through the check is the precondition for reading a σ-sequence failure
    // in planWanGeneration as an input error.
    const rejected: readonly [Record<string, unknown>, string][] = [
      [{ ...RAW, defaults: { steps: 50, guidance: Number.MAX_VALUE } }, "guidance"],
      [{ ...RAW, scheduler: { shift: 1e6 } }, "cannot build a UniPC σ sequence"],
    ];
    for (const [raw, message] of rejected) {
      const error = assertThrows(() => parseWanPipelineConfig(raw), Error, message);
      assert(
        !(error instanceof ModelInputError),
        `a declaration inconsistency is not an input error: ${message}`,
      );
    }
  });
});

describe("latent denormalisation", () => {
  it("the constants are the f32 values themselves (f64 decimals would disagree with the values the denormalisation uses)", () => {
    assertEquals(WAN_LATENTS_MEAN.length, 16);
    assertEquals(WAN_LATENTS_STD.length, 16);
    for (const value of [...WAN_LATENTS_MEAN, ...WAN_LATENTS_STD]) {
      assertEquals(Math.fround(value), value);
    }
  });

  it("in upstream's order (divide by the reciprocal of std, then add mean), the last bit differs from the multiplication order", () => {
    const perChannel = 4096;
    const latents = new Float32Array(16 * perChannel).map((_, index) =>
      Math.fround(Math.sin(index) * 3),
    );
    const got = denormalizeWanLatents(latents, WAN21_GENERATION.latents);
    let differsFromProduct = 0;
    for (let channel = 0; channel < 16; channel += 1) {
      const inverse = Math.fround(1 / WAN_LATENTS_STD[channel]);
      for (let index = 0; index < perChannel; index += 1) {
        const at = channel * perChannel + index;
        assertEquals(
          got[at],
          Math.fround(Math.fround(latents[at] / inverse) + WAN_LATENTS_MEAN[channel]),
        );
        const product = Math.fround(
          Math.fround(latents[at] * WAN_LATENTS_STD[channel]) + WAN_LATENTS_MEAN[channel],
        );
        if (product !== got[at]) differsFromProduct += 1;
      }
    }
    assert(
      differsFromProduct > 0,
      "the multiplication order also matched (a mix-up of the order is not pinned)",
    );
    assertThrows(
      () => denormalizeWanLatents(new Float32Array(17), WAN21_GENERATION.latents),
      Error,
      "not divisible",
    );
  });

  it("statistics with different mean and std counts fail loudly (the channel count is not silently taken from one of them)", () => {
    const { mean, std } = WAN21_GENERATION.latents;
    assertThrows(
      () => denormalizeWanLatents(new Float32Array(16), { mean, std: std.slice(1) }),
      Error,
      "mean count 16 and std count 15 differ",
    );
  });
});

describe("RGBA conversion of frames", () => {
  // A hand-built video without fps (frameToRgba does not read fps, so it does not require it — the
  // type check pins that).
  const video = {
    frames: 2,
    width: 2,
    height: 1,
    // [3, 2, 1, 2]: the value of channel c, frame f, pixel x.
    data: Float32Array.from([-1, 1, 0, 0.5, 0.25, -0.25, 2, -2, -0.5, 0.75, 1, -1]),
  };

  it("makes 8 bit with clamp(x/2 + 0.5) through round(·255), taking each frame's planes", () => {
    assertEquals([...frameToRgba(video, 0)], [0, 159, 64, 255, 255, 96, 223, 255]);
    assertEquals([...frameToRgba(video, 1)], [128, 255, 255, 255, 191, 0, 0, 255]);
  });

  it("an out-of-range frame and non-finite values fail loudly", () => {
    assertThrows(() => frameToRgba(video, 2), RangeError);
    const broken = { ...video, data: Float32Array.from(video.data).fill(Number.NaN, 3, 4) };
    assertThrows(() => frameToRgba(broken, 1), Error, "non-finite");
  });

  it("unless the size and the frame count are positive integers it is a RangeError even when the element count matches (the same entry contract as Anima's imageToRgba)", () => {
    // Every one satisfies `data.length === 3·frames·width·height` (a form that passes a length check
    // alone).
    const rejected: readonly { frames: number; width: number; height: number; length: number }[] = [
      { frames: 1, width: 0.5, height: 2, length: 3 },
      { frames: 1.5, width: 2, height: 1, length: 9 },
      { frames: 1, width: -1, height: -1, length: 3 },
      { frames: 1, width: 0, height: 0, length: 0 },
    ];
    for (const { length, ...dims } of rejected) {
      assertThrows(
        () => frameToRgba({ ...dims, data: new Float32Array(length) }, 0),
        RangeError,
        "not a positive integer",
        JSON.stringify(dims),
      );
    }
  });
});

describe("the RNG of the initial noise", () => {
  it("the same seed gives the same sequence and another seed another sequence (pairs are consumed whole even for odd lengths)", () => {
    assertEquals([...new WanRandn(7).normals(5)], [...new WanRandn(7).normals(5)]);
    assertNotEquals([...new WanRandn(7).normals(5)], [...new WanRandn(8).normals(5)]);
    const generator = new WanRandn(7);
    generator.normals(3);
    assertEquals([...generator.normals(2)], [...new WanRandn(7).normals(6)].slice(4));
  });

  it("pins the seed → sequence mapping by value (the same seed gives the same clip — a public promise)", () => {
    // "The same seed gives the same sequence" above passes even if the generator changes (swapping
    // cos / sin, a change of rounding, and so on). A changed sequence would silently change every
    // seeded clip, so the leading values themselves are pinned. seed 0 is generate's default, 42 the
    // seed of the example and of the opt-in sha rows. The values are f32 baked on Deno (V8)
    // (ECMAScript leaves Math.log / cos / sin to implementation-defined approximations, so another
    // engine may differ).
    assertEquals(
      [...new WanRandn(0).normals(8)],
      [
        -0.45275774598121643, 0.20776604115962982, 2.6506059169769287, -0.4904228150844574,
        -0.9886041283607483, 1.8721014261245728, 0.2524627149105072, -1.853424310684204,
      ],
    );
    assertEquals(
      [...new WanRandn(42).normals(8)],
      [
        0.41471976041793823, 0.6526812314987183, -0.8918862342834473, 1.3268336057662964,
        1.72959303855896, -1.883416771888733, 0.5456204414367676, -1.6568357944488525,
      ],
    );
  });
});

/**
 * A `Pipeline` run on mock Sessions (needs neither a GPU nor assets — pins the cleanup and non-finite
 * checks along the `generate` path). The DiT Session returns an output filled with `ditValue`, and
 * the VAE runs the one-tile = one-batch procedure (`decodeWanVaeTile`) on a fake GpuContext, filling
 * the frames read back with `vaeValue`.
 *
 * NOTE: the constructor is TS `private` (manifest checks and asset comparison are not bypassed — ADR
 * 0008), so `Reflect.construct` passes the internal state directly (only tests bypass private). The
 * public construction entry (`fromAssets`) needs the bytes of krm containers and acquiring the GPU,
 * which a CPU unit test cannot do. The internal state's shape is `WanState` of family.ts, but the
 * arguments of `Reflect.construct` are not type-checked, so the fields built here are kept in step by
 * hand — a missing field fails with a TypeError inside generate, which disagrees with the message each
 * test checks and turns it red.
 */
const mockPipeline = (options: {
  readonly ditValue?: number;
  readonly vaeValue?: number;
  readonly ditDisposeError?: Error;
  /**
   * With `"gpu"` the text stage runs on a mock umT5 (the prompt layer is the fixture's table —
   * {@link wanParityEncoder}). The output fills the rows of valid length L with `umt5Value(L)` (a mix-up
   * of positive and negative shows in the values).
   */
  readonly textEncoder?: "gpu";
  readonly umt5Value?: (tokens: number) => number;
  /** Called inside the umT5 run (the abort injection point). */
  readonly onUmt5Run?: (tokens: number) => void;
  /**
   * The DiT input form of the internal state (`"t2v"` when omitted). Values outside the type can be
   * passed too (`Reflect.construct` does not check types — this pins the unknown-form check).
   */
  readonly ditForm?: string;
}) => {
  const log: string[] = [];
  /** The contexts the DiT received (in run order — uncond → cond). */
  const ditContexts: Float32Array[] = [];
  /** All inputs the DiT runs received (in run order). */
  const ditRuns: Record<string, Tensor>[] = [];
  const { first, next } = vaeChunkGraphs(32);
  const resident = (byteLength: number) => ({
    byteLength,
    write: () => {},
    clear: () => {},
    dispose: () => {},
  });
  const vaeSession = (name: string) => ({
    createSession: () => {
      log.push(`create:${name}`);
      return Promise.resolve({
        enqueue: () => {
          log.push(`enqueue:${name}`);
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
  const gpuText = options.textEncoder === "gpu";
  const text = gpuText
    ? {
        kind: "gpu",
        encoder: wanParityEncoder(),
        contract: { output: "umt5_out" },
        component: {
          createSession: (_gpu: unknown, sessionOptions: unknown) => {
            log.push(`create:text_encoder:${JSON.stringify(sessionOptions)}`);
            return Promise.resolve({
              run: (inputs: Record<string, Tensor>) => {
                const tokens = inputs.input_ids.shape[1];
                log.push(`run:text_encoder:${tokens}`);
                options.onUmt5Run?.(tokens);
                return Promise.resolve({
                  umt5_out: {
                    dtype: "f32",
                    shape: [1, tokens, WIDTH],
                    data: new Float32Array(tokens * WIDTH).fill(
                      options.umt5Value?.(tokens) ?? tokens / 1000,
                    ),
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
      createResident: (bytes: number) => Promise.resolve(resident(bytes)),
      beginBatch: () =>
        Promise.resolve({
          finish: () => Promise.resolve(),
          finishAndRead: (frames: Record<string, { readonly byteLength: number }>) =>
            Promise.resolve(
              Object.fromEntries(
                Object.entries(frames).map(([name, { byteLength }]) => [
                  name,
                  new Float32Array(byteLength / 4).fill(options.vaeValue ?? 0.5).buffer,
                ]),
              ),
            ),
        }),
    },
    ownsGpu: false,
    config: CONFIG,
    quantName: "f16",
    sessionOptions: {},
    transformer: {
      createSession: () => {
        log.push("create:transformer");
        return Promise.resolve({
          run: (inputs: Record<string, Tensor>) => {
            const context = inputs.encoder_hidden_states;
            assert(context.dtype === "f32", "the DiT context is not f32");
            ditContexts.push(context.data);
            ditRuns.push(inputs);
            const tokens = inputs.tokens;
            return Promise.resolve({
              out: {
                dtype: "f32",
                shape: tokens.shape,
                data: new Float32Array(tokens.data.length).fill(options.ditValue ?? 0.1),
              },
            });
          },
          diagnostics: () => ({}),
          dispose: () => {
            log.push("dispose:transformer");
            return options.ditDisposeError === undefined
              ? Promise.resolve()
              : Promise.reject(options.ditDisposeError);
          },
        });
      },
    },
    vaeFirst: vaeSession("vae_decoder_first"),
    vaeNext: vaeSession("vae_decoder_next"),
    layout: wanVaeChunkLayout(first, next),
    // 64 rows cover the grid of the latent [16, 2, 60, 104] (2 × 30 × 52). The values do not matter
    // to the checks.
    ropeBase: {
      rows: 64,
      widths: [1, 1, 1],
      cos: [new Float32Array(64), new Float32Array(64), new Float32Array(64)],
      sin: [new Float32Array(64), new Float32Array(64), new Float32Array(64)],
    },
    // The GPU path pads the umT5 output (126 rows for negative), so the context has 512 rows like the
    // distribution.
    dit: {
      output: "out",
      projWidth: 256,
      contextRows: gpuText ? 512 : 4,
      contextWidth: WIDTH,
      patch: wanDitPatch(16),
      form: options.ditForm ?? "t2v",
    },
    textEmbeds: EMBEDS,
    text,
  };
  const pipeline: Pipeline = Reflect.construct(Pipeline, [state]);
  /**
   * Runs a request of 5 frames and 2 steps (4 DiT runs, 12 VAE tiles) and pushes the observed events
   * onto `log`. The prompt is `cats` on the asset path and the fixture's fixed prompt `boxing-cats` (28
   * tokens) on the GPU path.
   */
  const generate = (onEvent?: GenerateRequest["onEvent"], request: Partial<GenerateRequest> = {}) =>
    pipeline.generate({
      prompt: gpuText ? wanParityCase("fixed-boxing-cats").text : "Two cats.",
      frames: 5,
      steps: 2,
      ...request,
      onEvent: async (event) => {
        log.push(
          event.kind === "stage"
            ? `${event.component}:${event.at}`
            : event.kind === "denoise-step"
              ? `step:${event.step}`
              : `tile:${event.tile}`,
        );
        await onEvent?.(event);
      },
    });
  return { log, generate, ditContexts, ditRuns };
};

describe("Pipeline.generate (mock Sessions)", () => {
  it("control: with finite DiT / VAE outputs it runs to the end and returns frames clamped to [-1, 1]", async () => {
    const { log, generate } = mockPipeline({ vaeValue: 2 });
    const video = await generate();
    assertEquals([video.frames, video.width, video.height], [5, 832, 480]);
    assert(
      video.data.every((value) => value === 1),
      "the pre-clamp 2 did not become 1",
    );
    assertEquals(
      log.filter((entry) => entry.startsWith("step:")),
      ["step:1", "step:2"],
    );
    assertEquals(log.filter((entry) => entry.startsWith("tile:")).length, 12);
    assertEquals(log.at(-1), "vae_decoder:end");
  });

  it("the result carries Wan2.1's fps 16 (a fact of the upstream generation — not a request knob)", async () => {
    const { generate } = mockPipeline({});
    assertEquals((await generate()).fps, 16);
  });

  describe("the DiT input form (ADR 0121 decision 3)", () => {
    const T2V_KEYS = ["tokens", "timesteps_proj", "encoder_hidden_states", "rope_cos", "rope_sin"];

    it("t2v passes exactly 5 inputs to every run (no condition inputs added — the inputs reaching the 2.1 Session do not change)", async () => {
      const { generate, ditRuns } = mockPipeline({});
      await generate();
      assertEquals(ditRuns.length, 4);
      for (const inputs of ditRuns) assertEquals(Object.keys(inputs), T2V_KEYS);
    });

    it("ti2v (T2V) has 7: the condition mask is an all-zero u32 [1, S, 1] built only once, and the condition-side timestep is the same array as the generation side's", async () => {
      const { generate, ditRuns } = mockPipeline({ ditForm: "ti2v" });
      await generate();
      assertEquals(ditRuns.length, 4);
      const masks = new Set<unknown>();
      for (const inputs of ditRuns) {
        assertEquals(Object.keys(inputs), [
          ...T2V_KEYS,
          "timesteps_proj_condition",
          "condition_mask",
        ]);
        assertStrictEquals(inputs.timesteps_proj_condition.data, inputs.timesteps_proj.data);
        assertEquals(inputs.timesteps_proj_condition.shape, inputs.timesteps_proj.shape);
        const mask = inputs.condition_mask;
        assertEquals(mask.dtype, "bool");
        assertInstanceOf(mask.data, Uint32Array);
        const tokens = inputs.tokens.shape[1];
        assertEquals(mask.shape, [1, tokens, 1]);
        assertEquals(mask.data.length, tokens);
        assert(
          mask.data.every((value) => value === 0),
          "the T2V condition mask has a non-zero element",
        );
        masks.add(mask.data);
      }
      assertEquals(masks.size, 1, "the condition mask was rebuilt per run");
    });

    it("an unknown form fails before the DiT Session is set up (it is not silently run as t2v)", async () => {
      const { log, generate } = mockPipeline({ ditForm: "i2v" });
      await assertRejects(
        () => generate(),
        Error,
        `wan.Pipeline: unknown DiT input form 'i2v' ("t2v" / "ti2v")`,
      );
      assert(!log.includes("create:transformer"), `the DiT Session was set up: ${log}`);
    });
  });

  describe("non-finite checks", () => {
    for (const ditValue of [Number.NaN, Number.POSITIVE_INFINITY]) {
      it(`a DiT output of ${ditValue} fails after the step 1 update, and the VAE stage is not set up`, async () => {
        const { log, generate } = mockPipeline({ ditValue });
        await assertRejects(() => generate(), Error, "after the update of step 1/2, latent");
        assert(!log.includes("step:1"), "non-finite latents were shown in denoise-step");
        assert(!log.includes("vae_decoder:start"), `it went on to the VAE stage: ${log}`);
        assertEquals(log.at(-1), "dispose:transformer", "the DiT Session was not torn down");
      });
    }

    for (const vaeValue of [Number.POSITIVE_INFINITY, Number.NEGATIVE_INFINITY, Number.NaN]) {
      it(`a VAE output of ${vaeValue} fails before the clamp (±Inf is not turned into ±1)`, async () => {
        const { log, generate } = mockPipeline({ vaeValue });
        await assertRejects(
          () => generate(),
          Error,
          "VAE output (before clamping) channel 0, frame 0, pixel (x=0, y=0) is non-finite",
        );
        assert(!log.includes("vae_decoder:end"), `the VAE stage finished: ${log}`);
      });
    }
  });

  describe("cleanup of the DiT stage (the main failure is not overwritten by a cleanup failure)", () => {
    const thrown = new Error("A: abort in onEvent");
    const disposeFailure = new Error("B: Session.dispose failed");
    const abort = (event: Parameters<NonNullable<GenerateRequest["onEvent"]>>[0]) => {
      if (event.kind === "denoise-step") throw thrown;
    };

    it("when both the main part A and the cleanup B fail, both are carried in an AggregateError with A first", async () => {
      const { log, generate } = mockPipeline({ ditDisposeError: disposeFailure });
      const error = await assertRejects(() => generate(abort), AggregateError);
      assertEquals(error.errors, [thrown, disposeFailure]);
      assert(log.includes("dispose:transformer"));
      assert(!log.includes("transformer:end"), "emitted end for a stage that failed midway");
    });

    it("when only the main part fails, A itself is thrown", async () => {
      const { generate } = mockPipeline({});
      const error = await assertRejects(() => generate(abort));
      assertStrictEquals(error, thrown);
    });

    it("when only the cleanup fails, B itself is thrown and the VAE stage is not reached", async () => {
      const { log, generate } = mockPipeline({ ditDisposeError: disposeFailure });
      const error = await assertRejects(() => generate());
      assertStrictEquals(error, disposeFailure);
      assert(!log.includes("vae_decoder:start"), `it went on to the VAE stage: ${log}`);
    });
  });
});

describe("Pipeline.generate: the I2V fields (image / fit, Wan2.2 only — a JS caller slips past the types)", () => {
  /** A generation request with one field set that TS types do not allow (the 2.1 request type has no `image` / `fit`). */
  const requestWith = (key: string, value: unknown): Partial<GenerateRequest> => {
    const request: Partial<GenerateRequest> = {};
    Object.defineProperty(request, key, { value, enumerable: true });
    return request;
  };
  const IMAGE = { data: new Uint8Array(4 * 4 * 3), width: 4, height: 4 };
  const MESSAGE =
    "image / fit (generation from an image) is not accepted by this generation (Wan2.2's wan.Ti2vPipeline accepts it)";

  for (const [key, value] of [
    ["image", IMAGE],
    ["fit", "crop"],
  ] as const) {
    it(`passing ${key} is a ModelInputError without setting up a single Session (it points to the 2.2 class)`, async () => {
      const { log, generate } = mockPipeline({});
      await assertRejects(
        () => generate(undefined, requestWith(key, value)),
        ModelInputError,
        MESSAGE,
      );
      assertEquals(
        log.filter((entry) => entry.startsWith("create:")),
        [],
      );
    });
  }
});

describe("Pipeline.generate: vaeTile (ADR 0134 — Wan2.1 has no 32-latent decoder pair)", () => {
  it("vaeTile 32 is a ModelInputError naming the preset, before any Session is built", async () => {
    const { log, generate } = mockPipeline({});
    await assertRejects(
      () => generate(undefined, { vaeTile: 32 }),
      ModelInputError,
      "vaeTile 32 is not available for quant 'f16': the manifest has no tile-32 VAE decoder" +
        " (no vae_decoder_first_32 / vae_decoder_next_32 weights — use vaeTile 16)",
    );
    assertEquals(
      log.filter((entry) => entry.startsWith("create:")),
      [],
    );
  });

  it("vaeTile 16 is the default: the same tiles and the same two decoder Sessions as an omitted knob", async () => {
    const omitted = mockPipeline({});
    await omitted.generate();
    const explicit = mockPipeline({});
    await explicit.generate(undefined, { vaeTile: 16 });
    assertEquals(explicit.log, omitted.log);
  });
});

describe("planWanGpuGeneration (the entry check of the GPU path)", () => {
  const encoder = wanParityEncoder();
  const BOXING_CATS = wanParityCase("fixed-boxing-cats");
  const gpuPlan = (request: Partial<GenerateRequest>) =>
    planWanGpuGeneration(
      { prompt: BOXING_CATS.text, ...request },
      encoder,
      LAYOUT,
      CONFIG,
      WAN21_FAMILY.generation,
    );

  it("turns the prompt into the same id sequence as upstream, and an omitted negative encodes the official sample_neg_prompt through the same check", () => {
    const resolved = gpuPlan({});
    assertEquals([...resolved.positive], BOXING_CATS.ids);
    const negative = wanParityCase("fixed-negative");
    assertEquals([...(resolved.negative ?? [])], negative.ids);
    assertEquals(resolved.negative?.length, 126, "the official negative is 126 tokens");
    // The knob defaults come from the same single check as the asset path (the reference setting).
    assertEquals(
      [resolved.steps, resolved.guidance, resolved.shift, resolved.frames],
      [50, 5, 3, 33],
    );
  });

  it("accepts strings not in the asset, an explicit negative encodes that string, and guidance 1 runs no uncond", () => {
    // An accepted case other than the 4 fixed ones (boundary, random — outside the asset's set).
    const free = wanParityCases.find(
      (entry) => entry.ids !== undefined && !entry.id.startsWith("fixed-"),
    );
    assert(free !== undefined, "the fixture has no accepted non-fixed case");
    assertEquals([...gpuPlan({ prompt: free.text }).positive], free.ids);
    assertEquals([...(gpuPlan({ negativePrompt: free.text }).negative ?? [])], free.ids);
    assertEquals(gpuPlan({ guidance: 1 }).negative, undefined);
    assertThrows(
      () => gpuPlan({ guidance: 1, negativePrompt: free.text }),
      ModelInputError,
      "has no effect",
    );
  });

  it("a preprocessing rejection states the reason and the fix (R&D → R & D, unassigned / C1: remove that character)", () => {
    const rejected: readonly [string, string, string][] = [
      ["R&D lab", "entity", "`R&D` → `R & D`"],
      ["a͸b", "unassigned", "Remove that character before passing it"],
      ["a\x85b", "c1", "Remove that character before passing it"],
    ];
    for (const [prompt, reason, hint] of rejected) {
      const error = assertThrows(() => gpuPlan({ prompt }), PromptCleanError);
      assertEquals(error.reason, reason, prompt);
      assertStringIncludes(error.message, hint);
      assertStringIncludes(error.message, "prompt:");
    }
    const negative = assertThrows(() => gpuPlan({ negativePrompt: "R&D lab" }), PromptCleanError);
    assertStringIncludes(negative.message, "negativePrompt:");
    // With `R & D` and spaces around it, preprocessing passes (the fix in the message really works — a
    // failure here would be on the tokenizer vocabulary's side).
    assertEquals(encoder.clean("R & D lab"), "R & D lab");
  });

  it("empty / whitespace only (1 token) is rejected, asking for at least one word", () => {
    for (const prompt of ["", "   "]) {
      assertThrows(() => gpuPlan({ prompt }), ModelInputError, "Enter at least one word");
    }
  });

  it("the knob check is the same single check as the asset path (size, frame count and step count are rejected with the same message)", () => {
    for (const request of [{ frames: 34 }, { steps: 0 }, { width: 840 }] as const) {
      const viaAsset = assertThrows(() => plan(request), ModelInputError);
      const viaGpu = assertThrows(() => gpuPlan(request), ModelInputError);
      assertEquals(viaGpu.message, viaAsset.message);
    }
  });
});

describe("the default negative (the official sample_neg_prompt)", () => {
  it("bit-identical to the original text of the recipe's fixed prompt (the negative of prompts.py — the fixture's fixed-negative)", () => {
    assertEquals(WAN_DEFAULT_NEGATIVE_PROMPT, wanParityCase("fixed-negative").text);
  });
});

/**
 * The text-embedding asset the Wan2.1 distribution ships (ADR 0139 decision 1 — the local mirror
 * built by `dist.py`; a machine without it SKIPs only the case below).
 */
const WAN_MIRROR = mirrorRoot("karume-wan2.1");
const textEmbedsMirror = mirrorPresent(WAN_MIRROR)
  ? await readMirrorAsset(WAN_MIRROR, "text_embeds")
  : undefined;
if (textEmbedsMirror === undefined) {
  console.warn(
    `[karume] the mirror ${WAN_MIRROR.pathname} is absent (or not karume/5, or a sibling it references is ` +
      "absent), so the check of the default negative against the asset's row is SKIPPED (the check against the " +
      `recipe's text runs on the fixture). Assemble the mirror: ${WAN_ASSEMBLE_COMMAND}`,
  );
}

test.skipIf(textEmbedsMirror === undefined)(
  "default negative: bit-identical to the original text of the text-embedding asset's negative row (the defaults of both paths are the same string)",
  () => {
    if (textEmbedsMirror === undefined) throw new Error("disagrees with the skip condition");
    const embeds = parseWanTextEmbeds(textEmbedsMirror.buffer);
    assertEquals(
      embeds.entries.filter((entry) => entry.role === "negative").map((entry) => entry.prompt),
      [WAN_DEFAULT_NEGATIVE_PROMPT],
    );
  },
);

describe("family admission: the umT5 graph declaration × the shapes the host builds (input names, i32, symbol L, width, storage)", () => {
  type Umt5Spec = {
    readonly inputs: readonly {
      readonly name: string;
      readonly dtype: "i32" | "f32";
      readonly shape: readonly StubDim[];
    }[];
    readonly outputs: readonly {
      readonly name: string;
      readonly dtype: "f32" | "i32";
      readonly shape: readonly StubDim[];
    }[];
    /** The storage of the initializers (`shared` is a shared declaration without its own storage). */
    readonly storages: readonly (CodecName | "shared")[];
  };
  /** The same declaration as the distribution (the input names of `umt5_patch.py`, symbol L, output `[1, L, 4096]`, i8 weights and f32 tables). */
  const VALID: Umt5Spec = {
    inputs: [
      { name: "input_ids", dtype: "i32", shape: [1, "L"] },
      { name: "relative_position_buckets", dtype: "i32", shape: ["L", "L"] },
    ],
    outputs: [{ name: "out", dtype: "f32", shape: [1, "L", 4096] }],
    storages: ["int8-sym", "f32"],
  };
  const DIT = { contextRows: 512, contextWidth: 4096 };
  const umt5Of = (patch: Partial<Umt5Spec>): GraphOwner => {
    const spec = { ...VALID, ...patch };
    return {
      graph: {
        format: "karume-ir",
        version: 2,
        requires: { ops: [] },
        symbols: ["L", "M"],
        inputs: spec.inputs.map((input) => ({ ...input, shape: [...input.shape] })),
        outputs: spec.outputs.map((output) => output.name),
        initializers: Object.fromEntries(
          spec.storages.map((codec, index) => [
            `w${index}`,
            codec === "shared" ? { shared: true } : { storage: { codec } },
          ]),
        ),
        values: Object.fromEntries(
          spec.outputs.map((output) => [
            output.name,
            {
              dtype: output.dtype,
              shape: [...output.shape],
            },
          ]),
        ),
        states: {},
        nodes: [],
      },
    };
  };
  const ids = VALID.inputs[0];
  const buckets = VALID.inputs[1];

  it("the distribution's declaration passes, and the output name is taken from the declaration", () => {
    assertEquals(umt5Contract(umt5Of({}), DIT, "wan.Pipeline"), { output: "out" });
  });

  it("a mismatch of input name, width, symbol, dtype, count or storage fails by name before umT5 is fetched (a plain Error)", () => {
    const rejected: readonly [string, Partial<Umt5Spec>, string][] = [
      [
        "wrong input name",
        { inputs: [{ ...ids, name: "token_ids" }, buckets] },
        "graph input 'input_ids' is missing",
      ],
      [
        "width 2048",
        { outputs: [{ name: "out", dtype: "f32", shape: [1, "L", 2048] }] },
        "output 'out' shape",
      ],
      [
        "output with another symbol",
        { outputs: [{ name: "out", dtype: "f32", shape: [1, "M", 4096] }] },
        "output 'out' shape",
      ],
      [
        "bucket table with another symbol",
        { inputs: [ids, { ...buckets, shape: ["L", "M"] }] },
        "'relative_position_buckets' shape",
      ],
      [
        "static L in the id sequence",
        { inputs: [{ ...ids, shape: [1, 128] }, buckets] },
        "is not a symbolic dimension",
      ],
      ["id sequence is f32", { inputs: [{ ...ids, dtype: "f32" }, buckets] }, "not i32"],
      [
        "3 inputs",
        { inputs: [ids, buckets, { name: "attention_mask", dtype: "i32", shape: [1, "L"] }] },
        "has 3 graph inputs",
      ],
      [
        "2 outputs",
        { outputs: [...VALID.outputs, { name: "extra", dtype: "f32", shape: [1, "L", 4096] }] },
        "has 2 outputs",
      ],
      [
        "output is i32",
        { outputs: [{ name: "out", dtype: "i32", shape: [1, "L", 4096] }] },
        "not f32",
      ],
      ["storage f16", { storages: ["f16", "f32"] }, "has storage f16, which is not accepted"],
      [
        "storage i4",
        { storages: ["int8-sym", "int4-sym-g"] },
        "has storage int4-sym-g, which is not accepted",
      ],
      ["no i8", { storages: ["f32"] }, "has no i8 weights at all"],
      ["shared declaration", { storages: ["int8-sym", "shared"] }, "is a shared declaration"],
    ];
    for (const [label, patch, message] of rejected) {
      const error = assertThrows(
        () => umt5Contract(umt5Of(patch), DIT, "wan.Pipeline"),
        Error,
        message,
        label,
      );
      assert(
        !(error instanceof ModelInputError),
        `an asset inconsistency is not an input error: ${label}`,
      );
    }
  });

  it("fails for a pair where the valid-length limit 512 exceeds the DiT's context row count", () => {
    assertThrows(
      () => umt5Contract(umt5Of({}), { contextRows: 256, contextWidth: 4096 }, "wan.Pipeline"),
      Error,
      "valid-length upper limit 512 exceeds the DiT context row count 256",
    );
  });
});

/** The manifest the construction-entry tests use (the skeleton of `models/karume-wan2.1/karume.json` — declarations only). */
const wanManifest = (weightNames: readonly string[]) =>
  parseManifest(
    JSON.stringify({
      format: "karume/5",
      generator: "karume/0.1.0",
      defaultModel: "t2v-1.3b",
      models: {
        "t2v-1.3b": {
          pipeline: "wan/1",
          weights: Object.fromEntries(
            weightNames.map((name) => [name, { f16: declaredContainer(`${name}/model.f16`) }]),
          ),
          assets: {
            text_embeds: { path: "text_embeds.safetensors", size: 8, sha256: "e".repeat(64) },
            umt5_tokenizer: { path: "tokenizer.json", size: 8, sha256: "f".repeat(64) },
          },
          quants: {
            f16: {
              weights: Object.fromEntries(weightNames.map((name) => [name, "f16"])),
              session: {},
            },
          },
          defaultQuant: "f16",
          pipelineConfig: { scheduler: { shift: 3 }, defaults: { steps: 50, guidance: 5 } },
        },
      },
    }),
  );
const WITH_UMT5 = ["transformer", "vae_decoder_first", "vae_decoder_next", "text_encoder"];
const WITHOUT_UMT5 = ["transformer", "vae_decoder_first", "vae_decoder_next"];

/** Construction options with one field set to a value TS types do not allow (reproducing a JS caller's misspelling). */
const optionsWith = (key: string, value: unknown): PipelineOptions => {
  const options: PipelineOptions = {};
  Object.defineProperty(options, key, { value, enumerable: true });
  return options;
};

/** Containers that can be opened (minimal declarations — shells that fail at admission's graph check; no umT5). */
const SHELL_COMPONENTS: Record<string, Uint8Array<ArrayBuffer>> = {};
for (const name of WITHOUT_UMT5) {
  Object.assign(
    SHELL_COMPONENTS,
    partAssets(
      name,
      await tensorlessContainer(name, {
        inputs: [{ name: "x", shape: [1, 4] }],
        output: { name: "y", shape: [1, 4] },
      }),
    ),
  );
}

describe("the construction entry (path spelling, the umT5 declaration, the components fetched, abort — the scope needing neither a GPU nor real assets)", () => {
  it("an unknown textEncoder spelling is a plain Error before any asset is touched (treated like a misspelt model / quant name)", async () => {
    const error = await assertRejects(
      () =>
        Pipeline.fromAssets(
          { manifest: wanManifest(WITH_UMT5), assets: {} },
          optionsWith("textEncoder", "cpu"),
        ),
      Error,
      "textEncoder 'cpu' is neither 'gpu' nor 'precomputed'",
    );
    assert(!(error instanceof ModelInputError));
  });

  it("on the default gpu path, a manifest that does not declare umT5 fails before any container is opened, pointing to precomputed", async () => {
    await assertRejects(
      () => Pipeline.fromAssets({ manifest: wanManifest(WITHOUT_UMT5), assets: {} }),
      Error,
      "no umT5 component 'text_encoder' (the default textEncoder \"gpu\" takes it",
    );
  });

  it("precomputed gets as far as admission without opening the umT5 component, and gpu needs the umT5 container", async () => {
    // The same assets (only the umT5 container is missing); only the set of components each path opens
    // differs. precomputed opens 3 components and fails at admission's graph check (the VAE chunk
    // graph) = it did not touch umT5.
    const manifest = wanManifest(WITH_UMT5);
    await assertRejects(
      () =>
        Pipeline.fromAssets({ manifest, assets: SHELL_COMPONENTS }, { textEncoder: "precomputed" }),
      WanVaeChunkError,
    );
    await assertRejects(
      () => Pipeline.fromAssets({ manifest, assets: SHELL_COMPONENTS }),
      Error,
      "component 'text_encoder' has no container",
    );
  });

  it("admission compares the VAE declaration with the descriptor (a mismatch of latent or output channel count fails before the raw RoPE table is read)", async () => {
    // A declaration-only VAE meeting the chunk-graph agreement (latent tile 32, scale 8, one cache). The
    // DiT stays a shell, so an asset that gets through the comparison fails with another message when
    // the raw RoPE table is read (the control that pins the comparison's wiring).
    const vae = async (name: string, frames: number, latent: number, sample: number) =>
      partAssets(
        name,
        await writeContainer({
          graphs: {
            [name]: parseIrDeclarationValue({
              format: "karume-ir",
              version: 2,
              requires: { ops: ["mul"] },
              symbols: [],
              inputs: [
                { name: "latent", dtype: "f32", shape: [latent, 1, 32, 32] },
                { name: "cache_00", dtype: "f32", shape: [16, 2, 32, 32] },
              ],
              outputs: ["frame", "cache_00_out"],
              initializers: {},
              values: {
                frame: { dtype: "f32", shape: [sample, frames, 256, 256] },
                cache_00_out: { dtype: "f32", shape: [16, 2, 32, 32] },
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
        }),
      );
    const build = async (latent: number, sample: number) =>
      Pipeline.fromAssets(
        {
          manifest: wanManifest(WITHOUT_UMT5),
          assets: {
            ...SHELL_COMPONENTS,
            ...(await vae("vae_decoder_first", 1, latent, sample)),
            ...(await vae("vae_decoder_next", 4, latent, sample)),
          },
        },
        { textEncoder: "precomputed" },
      );
    const rejected: readonly [string, number, number, string][] = [
      [
        "output 12",
        16,
        12,
        "wan.Pipeline: VAE output 12 channels differ from RGB 3 × unpatchify 1² = 3",
      ],
      [
        "latent 8",
        8,
        3,
        "wan.Pipeline: the denormalization statistics (mean 16, std 16) do not match the VAE latent's 8 channels",
      ],
    ];
    for (const [label, latent, sample, message] of rejected) {
      await assertRejects(() => build(latent, sample), Error, message, label);
    }
    // Control: the distribution's declaration (latent 16, output 3) passes the comparison and the tile
    // check, and fails at the shell DiT's raw RoPE table.
    await assertRejects(() => build(16, 3), Error, "'rope_base'");
  });

  it("an already-aborted signal rejects with the reason as is, before any asset is touched", async () => {
    // Without a signal this fails with "component 'transformer' has no container"; aborted, it fails
    // with the reason.
    const controller = new AbortController();
    const reason = new Error("stop button");
    controller.abort(reason);
    const error = await assertRejects(() =>
      Pipeline.fromAssets(
        { manifest: wanManifest(WITH_UMT5), assets: {} },
        { signal: controller.signal },
      ),
    );
    assertStrictEquals(error, reason);
  });

  it("an abort that arrives after the run started also takes effect at the first stage boundary (checked after yielding to the event loop)", async () => {
    const controller = new AbortController();
    const reason = new Error("stop button (while running)");
    setTimeout(() => controller.abort(reason), 0);
    const error = await assertRejects(() =>
      Pipeline.fromAssets(
        { manifest: wanManifest(WITH_UMT5), assets: {} },
        { signal: controller.signal },
      ),
    );
    assertStrictEquals(error, reason);
  });
});

/** The manifest of the distribution mirror (written by `dist.py --pipeline wan` — a machine without it SKIPs only the case below). */
const DIST_MANIFEST = new URL("../../../models/karume-wan2.1/karume.json", import.meta.url);
const distManifestText = await readTextIfPresent(DIST_MANIFEST);
if (distManifestText === undefined) {
  console.warn(
    `[karume] ${DIST_MANIFEST.pathname} is absent, so the check that a build without quant resolves to the practical preset is SKIPPED. ` +
      `Assemble it: ${WAN_ASSEMBLE_COMMAND}`,
  );
}

/** The default preset (ruling 4 of ADR 0120 on 2026-10-04 — the most practical preset today. The source of truth is the recipe's `WAN_DEFAULT_QUANT`). */
const PRACTICAL_QUANT = "f16+dit8-a8-attn8-s16";

test.skipIf(distManifestText === undefined)(
  "the distribution's default preset: a build without quant resolves to the practical preset (reading only the manifest karume.json)",
  async () => {
    assert(distManifestText !== undefined);
    // The preset is resolved in the family admission (before the weights are read). Passing a shared
    // GPU whose limits are all 0 makes every preset fail on the requiredLimits it declares (the umT5
    // vocabulary embedding), and that message names the resolved preset — observing the preset without
    // using weights or a GPU.
    await assertRejects(
      () =>
        Pipeline.fromAssets(
          { manifest: parseManifest(distManifestText), assets: SHELL_COMPONENTS },
          { textEncoder: "precomputed", gpu: fakeGpuContext(fakeDevice()) },
        ),
      Error,
      `wan.Pipeline: quant '${PRACTICAL_QUANT}': this GPU does not meet the required device limits`,
    );
  },
);

describe("Pipeline.generate (mock Sessions, the text stage of the GPU path)", () => {
  it("the text stage runs positive → negative and tears down, then sets up the DiT stage (the Session's run options are {})", async () => {
    const generated = mockPipeline({ textEncoder: "gpu" });
    await generated.generate();
    assertEquals(generated.log.slice(0, 7), [
      "text_encoder:start",
      "create:text_encoder:{}",
      "run:text_encoder:28",
      "run:text_encoder:126",
      "dispose:text_encoder",
      "text_encoder:end",
      "transformer:start",
    ]);
    assertEquals(generated.log.at(-1), "vae_decoder:end");
  });

  it("the context passed to the DiT is the umT5 output zero-padded to 512 rows (uncond = negative, cond = positive)", async () => {
    const { generate, ditContexts } = mockPipeline({ textEncoder: "gpu" });
    await generate();
    // 2 steps × (uncond, cond).
    assertEquals(ditContexts.length, 4);
    const rowsOf = (context: Float32Array, value: number) => {
      const filled = context.findIndex((entry) => entry !== Math.fround(value));
      return filled === -1 ? context.length / WIDTH : filled / WIDTH;
    };
    const [uncond, cond] = ditContexts;
    assertEquals(uncond.length, 512 * WIDTH);
    assertEquals(rowsOf(uncond, 126 / 1000), 126, "uncond is the output of negative (126 tokens)");
    assertEquals(rowsOf(cond, 28 / 1000), 28, "cond is the output of positive (28 tokens)");
    assert(
      uncond.subarray(126 * WIDTH).every((entry) => entry === 0),
      "not zero past the valid length",
    );
    assert(
      cond.subarray(28 * WIDTH).every((entry) => entry === 0),
      "not zero past the valid length",
    );
  });

  it("guidance 1 does not run negative (one umT5 run)", async () => {
    const { log, generate } = mockPipeline({ textEncoder: "gpu" });
    await generate(undefined, { guidance: 1 });
    assertEquals(
      log.filter((entry) => entry.startsWith("run:text_encoder")),
      ["run:text_encoder:28"],
    );
  });

  it("a non-finite umT5 output fails by name in the text stage, tears the Session down, and does not set up the DiT stage", async () => {
    const { log, generate } = mockPipeline({
      textEncoder: "gpu",
      umt5Value: (tokens) => (tokens === 126 ? Number.NaN : 0.5),
    });
    await assertRejects(() => generate(), Error, "umT5 output (negativePrompt, 126 tokens)");
    assertEquals(log.at(-1), "dispose:text_encoder");
    assert(!log.includes("text_encoder:end"), "emitted end for a stage that failed midway");
    assert(!log.includes("create:transformer"), `it went on to the DiT stage: ${log}`);
  });

  it("the entry check (a preprocessing rejection) fails without setting up a single Session", async () => {
    const { log, generate } = mockPipeline({ textEncoder: "gpu" });
    await assertRejects(() => generate(undefined, { prompt: "R&D lab" }), PromptCleanError);
    assertEquals(
      log.filter((entry) => entry.startsWith("create:")),
      [],
    );
  });
});

describe("ownership of the initial noise (latents) in Pipeline.generate", () => {
  /** A latent of 5 frames [16, 2, 60, 104] (the value varies per element — a mix-up shows in the DiT input). */
  const initialLatents = () =>
    new Float32Array(16 * 2 * 60 * 104).map((_, index) => ((index % 97) - 48) / 64);

  it("even if the caller rewrites the original array after the entry check, generation uses the values as of the check (both finite values and NaN)", async () => {
    const baseline = mockPipeline({ textEncoder: "gpu" });
    await baseline.generate(undefined, { latents: initialLatents() });
    const expected = baseline.ditRuns[0].tokens;
    assert(expected.dtype === "f32");
    for (const value of [7, Number.NaN]) {
      const latents = initialLatents();
      const { generate, ditRuns } = mockPipeline({ textEncoder: "gpu" });
      // The text stage's start notification is awaited after the entry check and before the DiT
      // stage (the window for the rewrite).
      await generate(
        (event) => {
          if (
            event.kind === "stage" &&
            event.component === "text_encoder" &&
            event.at === "start"
          ) {
            latents.fill(value);
          }
        },
        { latents },
      );
      const tokens = ditRuns[0].tokens;
      assert(tokens.dtype === "f32" && tokens.data.length === expected.data.length);
      // Compared by the index of the first differing element, so no diff of 200K elements is built.
      const differs = tokens.data.findIndex(
        (entry, index) => !Object.is(entry, expected.data[index]),
      );
      assertEquals(differs, -1, `the DiT ran with the rewritten value ${value}`);
    }
  });

  it("generates from the values at call time even when the caller rewrites latents while the call waits its turn (finite and NaN)", async () => {
    // Baseline: the same two calls in order, nothing rewritten (the second call's first DiT run is
    // index 4 — 2 steps × uncond / cond for the first call).
    const baseline = mockPipeline({});
    await baseline.generate();
    const offset = baseline.ditRuns.length;
    await baseline.generate(undefined, { latents: initialLatents() });
    const expected = baseline.ditRuns[offset].tokens;
    assert(expected.dtype === "f32");
    for (const value of [7, Number.NaN]) {
      const { generate, ditRuns } = mockPipeline({});
      // The first call holds the serialization chain, so the second waits its turn; the caller
      // rewrites its array right after the call returns (inside that wait).
      const first = generate();
      const latents = initialLatents();
      const second = generate(undefined, { latents });
      latents.fill(value);
      await Promise.all([first, second]);
      const tokens = ditRuns[offset].tokens;
      assert(tokens.dtype === "f32" && tokens.data.length === expected.data.length);
      const differs = tokens.data.findIndex(
        (entry, index) => !Object.is(entry, expected.data[index]),
      );
      assertEquals(differs, -1, `the DiT ran with the rewritten value ${value}`);
    }
  });

  it("does not turn a latents value that is not a Float32Array into one (the entry check still rejects it by name)", async () => {
    const { generate, log } = mockPipeline({});
    const latents = Array.from(initialLatents()) as unknown as Float32Array<ArrayBuffer>;
    await assertRejects(
      () => generate(undefined, { latents }),
      ModelInputError,
      "latents is not a Float32Array",
    );
    assertEquals(
      log.filter((entry) => entry.startsWith("create:")),
      [],
    );
  });
});

describe("abort in Pipeline.generate (mock Sessions — signal.reason is not wrapped, the Session is torn down, the next one runs)", () => {
  it("an abort after step 1 takes effect before step 2, tears down the DiT Session, does not set up the VAE stage, and the next generate runs", async () => {
    const { log, generate } = mockPipeline({});
    const controller = new AbortController();
    const reason = new Error("stop button");
    const error = await assertRejects(() =>
      generate(
        (event) => {
          if (event.kind === "denoise-step" && event.step === 1) controller.abort(reason);
        },
        { signal: controller.signal },
      ),
    );
    assertStrictEquals(error, reason);
    assert(!log.includes("step:2"), `a step ran after the abort: ${log}`);
    assertEquals(log.at(-1), "dispose:transformer");
    assert(!log.includes("vae_decoder:start"), `it went on to the VAE stage: ${log}`);
    log.length = 0;
    await generate();
    assertEquals(
      log.at(-1),
      "vae_decoder:end",
      "the generate after the abort did not run to the end",
    );
  });

  it("an abort delivered by a task (timer) also takes effect at the next boundary — a check that does not yield would run to the end", async () => {
    const { log, generate } = mockPipeline({});
    const controller = new AbortController();
    const reason = new Error("stop button (timer)");
    const error = await assertRejects(() =>
      generate(
        (event) => {
          if (event.kind === "denoise-step" && event.step === 1) {
            setTimeout(() => controller.abort(reason), 0);
          }
        },
        { signal: controller.signal },
      ),
    );
    assertStrictEquals(error, reason);
    assert(
      !log.includes("step:2"),
      `the abort delivered by a task was not seen before step 2: ${log}`,
    );
  });

  it("an abort inside the text stage takes effect before the next run, tears down the umT5 Session, and does not set up the DiT stage", async () => {
    const controller = new AbortController();
    const reason = new Error("stop button (text stage)");
    const { log, generate } = mockPipeline({
      textEncoder: "gpu",
      onUmt5Run: (tokens) => {
        if (tokens === 28) controller.abort(reason);
      },
    });
    const error = await assertRejects(() => generate(undefined, { signal: controller.signal }));
    assertStrictEquals(error, reason);
    assertEquals(
      log.filter((entry) => entry.startsWith("run:text_encoder")),
      ["run:text_encoder:28"],
    );
    assertEquals(log.at(-1), "dispose:text_encoder");
    assert(!log.includes("create:transformer"), `it went on to the DiT stage: ${log}`);
  });

  it("an already-aborted signal throws the reason without setting up a single Session (on either path)", async () => {
    for (const textEncoder of [undefined, "gpu"] as const) {
      const { log, generate } = mockPipeline(textEncoder === undefined ? {} : { textEncoder });
      const controller = new AbortController();
      const reason = new Error("stop button (before the start)");
      controller.abort(reason);
      const error = await assertRejects(() => generate(undefined, { signal: controller.signal }));
      assertStrictEquals(error, reason);
      assertEquals(
        log.filter((entry) => entry.startsWith("create:")),
        [],
        `${textEncoder}`,
      );
    }
  });

  it("an abort inside the VAE stage's start notification takes effect before the first tile, tearing down the VAE Session without enqueuing a single chunk", async () => {
    const { log, generate } = mockPipeline({});
    const controller = new AbortController();
    const reason = new Error("stop button (VAE start notification)");
    const error = await assertRejects(() =>
      generate(
        (event) => {
          if (event.kind === "stage" && event.component === "vae_decoder" && event.at === "start") {
            controller.abort(reason);
          }
        },
        { signal: controller.signal },
      ),
    );
    assertStrictEquals(error, reason);
    assertEquals(
      log.filter((entry) => entry.startsWith("enqueue:")),
      [],
      `tiles were enqueued: ${log}`,
    );
    assertEquals(
      log.filter((entry) => entry.startsWith("tile:")),
      [],
    );
    assert(log.includes("dispose:vae_decoder_first") && log.includes("dispose:vae_decoder_next"));
    assert(!log.includes("vae_decoder:end"), `emitted end for the aborted stage: ${log}`);
  });

  it("an abort between VAE tiles takes effect before the next tile and tears down the VAE Session", async () => {
    const { log, generate } = mockPipeline({});
    const controller = new AbortController();
    const reason = new Error("stop button (VAE)");
    const error = await assertRejects(() =>
      generate(
        (event) => {
          if (event.kind === "vae-tile" && event.tile === 1) controller.abort(reason);
        },
        { signal: controller.signal },
      ),
    );
    assertStrictEquals(error, reason);
    assertEquals(
      log.filter((entry) => entry.startsWith("tile:")),
      ["tile:1"],
    );
    assert(log.includes("dispose:vae_decoder_first") && log.includes("dispose:vae_decoder_next"));
    assert(!log.includes("vae_decoder:end"), `emitted end for the aborted stage: ${log}`);
  });
});
