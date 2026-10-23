/**
 * Wan's generation request check — checks a request and turns it into a plan (the entry of
 * `generate` — a pure check before touching the GPU).
 *
 * The only difference between paths is how the prompt is looked up ({@link PromptGate} —
 * `./text-stage.ts`); per-generation values (accepted set, frame range, defaults) are received as
 * arguments from the generation descriptor (`./descriptor.ts`). Shared by the Wan2.1 / 2.2 classes
 * (ADR 0121 decision 10).
 *
 * MUST: every module is side-effect free (no import-time execution, no global mutable state —
 * CLAUDE.md).
 */

import { ModelInputError } from "../errors.ts";
import { assertAcceptableSeed } from "../request-gates.ts";
import type { WanPipelineConfig } from "./config.ts";
import type { WanGenerationDescriptor } from "./descriptor.ts";
import { isWanI2vFit, preprocessWanI2vImage, type WanI2vImageInput } from "./i2v-preprocess.ts";
import type { WanTi2vGenerateRequest } from "./ti2v-pipeline.ts";
import { WAN_UNIPC_CONFIG, type WanUniPcSchedule, wanUniPcSchedule } from "./scheduler.ts";
import type { WanTextEmbedding } from "./text-embeds.ts";
import { planWanGenerationTiles, wanSpatialCompression } from "./tile-decode.ts";
import type { WanVaeChunkLayout } from "./vae-chunks.ts";
import type { WanVaeTilePlan } from "./vae-tiles.ts";

/**
 * The VAE's temporal compression (the first chunk is 1 frame, later ones 4 frames — the contract of
 * `vae-chunks.ts`).
 *
 * NOTE: `export` is so that the descriptor consistency tests and gpu-lab (removed in 0.15) read the
 * same value (not on the `mod.ts` / subpath surface — ADR 0008).
 */
export const TEMPORAL_COMPRESSION = 4;

/**
 * The default seed of the initial noise (independent of the generation). The size and frame-count
 * defaults come from the generation descriptor ({@link WanGenerationDescriptor}); the step count,
 * guidance and shift defaults from the manifest's `pipelineConfig` ({@link WanPipelineConfig}).
 */
const DEFAULT_SEED = 0;

/** The source of the initial noise ({@link planWanRequest}). */
type InitialNoise =
  | { readonly kind: "seed"; readonly seed: number }
  | { readonly kind: "latents"; readonly data: Float32Array<ArrayBuffer> };

/**
 * A generation request that passed the entry check (the return of {@link planWanRequest}).
 * `Text` is the form of one prompt (`"precomputed"` = a row of the asset's embeddings, `"gpu"` = umT5
 * ids).
 */
export type WanGenerationPlan<Text = WanTextEmbedding> = WanGenerationKnobs & {
  readonly positive: Text;
  /** CFG's uncond side (undefined when `guidance` is 1 — uncond does not run). */
  readonly negative: Text | undefined;
};

/** The plan without the prompts (used by the DiT and VAE stages — independent of the path). */
export type WanGenerationKnobs = {
  readonly steps: number;
  readonly guidance: number;
  readonly shift: number;
  readonly frames: number;
  readonly width: number;
  readonly height: number;
  /**
   * The latent shape `[C, F', H/s, W/s]` (C is the latent channel count of the VAE graph declaration,
   * s the spatial compression `wanSpatialCompression` — `[16, F', H/8, W/8]` for Wan2.1).
   */
  readonly latentShape: readonly [number, number, number, number];
  readonly initial: InitialNoise;
  /**
   * The UniPC σ sequence and timestep sequence for `steps` × `shift` (denoise uses it and does not
   * rebuild it).
   */
  readonly schedule: WanUniPcSchedule;
  /**
   * The edge of the VAE decoder's latent tile (`WanGenerateRequest.vaeTile` — 16 when omitted). It
   * picks the decoder's chunk-graph pair (ADR 0134 — 32 runs the pair that borrows the 16 pair's
   * weights).
   */
  readonly vaeTile: WanVaeTile;
  /** The VAE tile plan (latent `H/s × W/s` — made before denoise, from the pair of {@link vaeTile}). */
  readonly tiles: WanVaeTilePlan;
  /**
   * The VAE encoder input from the preprocessed I2V condition image (only for a request that passes
   * `image` — the output size equals `width` / `height`). Undefined means a text-only request (the
   * encoder stage does not run, and the DiT is not conditioned).
   */
  readonly conditionImage: WanI2vImageInput | undefined;
};

/** The VAE chunk graph geometry the plan needs (from the asset's declaration — `wanVaeChunkLayout`). */
export type PlanLayout = Pick<
  WanVaeChunkLayout,
  "latentChannels" | "tile" | "sampleTile" | "sampleChannels"
>;

/** The VAE decoder's latent tile edges a request can choose (ADR 0134 decision 2). */
export type WanVaeTile = NonNullable<WanTi2vGenerateRequest["vaeTile"]>;

/**
 * What the plan knows about the 32-latent decoder pair (ADR 0134 decision 2): its geometry, or
 * `undefined` when the preset's weights carry no `_32` roles, and the preset's name for the
 * rejection.
 */
export type WanVaeTile32Plan = {
  readonly layout?: PlanLayout;
  readonly quant: string;
};

/** The prompt check per path ({@link planWanRequest} calls it in one check order). */
export type PromptGate<Text> = {
  /**
   * Looks up / encodes the given string (its type already checked) in the accepted set. A rejection
   * is a `ModelInputError`.
   */
  readonly resolve: (text: string, what: "prompt" | "negativePrompt") => Text;
  /** The default when `negativePrompt` is omitted (called only when `guidance` > 1). */
  readonly defaultNegative: () => Text;
};

/**
 * The request fields the check reads (the observer and cancellation are read by the stages —
 * `WanGenerateRequest`). The I2V fields (`image` / `fit`) exist only in Wan2.2's request type
 * ({@link WanTi2vGenerateRequest}), but the check reads them in one body regardless of generation — an
 * `image` that a JS caller passes to the Wan2.1 class, slipping past the type, is rejected here too.
 */
type WanPlanRequest = Pick<
  WanTi2vGenerateRequest,
  | "prompt"
  | "negativePrompt"
  | "seed"
  | "latents"
  | "steps"
  | "guidance"
  | "shift"
  | "frames"
  | "width"
  | "height"
  | "image"
  | "fit"
  | "vaeTile"
>;

/**
 * The condition image has the RGB8 shape (an object whose `data` is a `Uint8Array`). A value a JS
 * caller passes past the type (`null`, a number, an object without `data`) becomes a
 * `ModelInputError` — a wrong type in a request field is an input error (the same handling as the
 * prompt's `typeof` check). Mismatches of size and length are checked by the preprocessing's
 * `assertRgb8`.
 *
 * MUST: check the type of `data` too — an array of another type such as `Float32Array` passes the
 * length check and would silently proceed with the pixels misread as uint8.
 */
const assertRgb8Shaped = (image: unknown): void => {
  if (
    typeof image === "object" &&
    image !== null &&
    "data" in image &&
    image.data instanceof Uint8Array
  ) {
    return;
  }
  const actual =
    image === null
      ? "null"
      : typeof image !== "object"
        ? typeof image
        : "data" in image
          ? `data is ${Object.prototype.toString.call(image.data).slice("[object ".length, -1)}`
          : "no data";
  throw new ModelInputError(
    `image is not an RGB8 image (an object with a Uint8Array data, width and height) (actual: ${actual})`,
  );
};

/**
 * Checks a generation request and turns it into a plan. The body of the checks shared by the entries
 * of the two paths; only how the prompt is looked up comes from `gate`. An omitted step count,
 * guidance or shift is filled with the default of `config` (the manifest's `pipelineConfig`); the size
 * and frame count with the defaults of the generation descriptor (`generation`).
 *
 * MUST: every input-caused failure (a prompt outside the set, a size outside the accepted set, a knob
 * out of range, a steps × shift combination that breaks the σ sequence, `image` / `fit` for a
 * generation that takes no condition image, `fit` without `image`, a value that is not RGB8-shaped or
 * an RGB8 whose size and length disagree) becomes a `ModelInputError` here. No failing after the DiT
 * weights are uploaded. A combination of defaults alone has had its σ sequence checked by the manifest
 * check (`parseWanPipelineConfig`), so a broken σ sequence here only happens when the request passed a
 * value (= input-caused). A misspelled `fit` is a plain `Error` (a misspelled name — ADR 0107 decision
 * 3; the same handling as the construction option `textEncoder`), and it is checked before whether
 * `image` is present (the value of a field itself is checked before the combination of fields — `fit:
 * "cover"` without `image` also fails as a misspelling).
 *
 * I2V (a request that passes `image` — only for a generation of input form `"ti2v"`) chooses the
 * output size within the accepted set from the image's aspect ratio and an explicit `width` /
 * `height`, and finishes preprocessing (fitting the size → `[-1, 1]` → patchify —
 * `i2v-preprocess.ts`) here (all input errors fail before touching the GPU). A request that does not
 * pass `image` gets the same plan as when there were no image fields.
 *
 * MUST: the VAE tile plan is also made here — made after denoise, an inconsistency that the assets
 * alone reveal would fail after paying for every step. The family admission (`assertWanVaeTilesCover`
 * in `tile-decode.ts`) has already passed every accepted size, so a failure here comes only from a
 * call that bypassed admission (an asset mismatch — a plain `Error`).
 *
 * `vaeTile` 32 plans with the geometry of the 32-latent pair (`tile32.layout`); without that pair
 * (`tile32` omitted or its layout undefined — Wan2.1, or a preset without the `_32` roles) it is a
 * `ModelInputError` naming the preset. The overlap stays 64 px of output (ADR 0121 decision 6).
 */
export const planWanRequest = <Text>(
  request: WanPlanRequest,
  gate: PromptGate<Text>,
  layout: PlanLayout,
  config: WanPipelineConfig,
  generation: WanGenerationDescriptor,
  tile32?: WanVaeTile32Plan,
): WanGenerationPlan<Text> => {
  const resolve = (text: unknown, what: "prompt" | "negativePrompt"): Text => {
    if (typeof text !== "string") throw new ModelInputError(`${what} is not a string`);
    return gate.resolve(text, what);
  };
  const positive = resolve(request.prompt, "prompt");

  const steps = request.steps ?? config.defaults.steps;
  // The upper bound is checked by a single-value check before building the σ sequence (the MUST of
  // `wanUniPcSchedule` — no allocating arrays and running a synchronous loop for a huge steps). The
  // value is `num_train_timesteps` (beyond it, integer timesteps always repeat).
  const maxSteps = WAN_UNIPC_CONFIG.numTrainTimesteps;
  if (!Number.isInteger(steps) || steps < 1 || steps > maxSteps) {
    throw new ModelInputError(
      `steps ${steps} is not an integer in 1..${maxSteps}` +
        " (the upper limit is num_train_timesteps — beyond it, integer timesteps always repeat)",
    );
  }
  const guidance = request.guidance ?? config.defaults.guidance;
  // Checked in f32: CFG multiplies by `f32(guidance)` (`wanClassifierFreeGuidance`), so a value that is
  // finite in f64 but Infinity in f32 (such as `Number.MAX_VALUE`) would make the combination NaN.
  if (!Number.isFinite(Math.fround(guidance)) || guidance < 1) {
    throw new ModelInputError(
      `guidance ${guidance} is not a finite number ≥ 1 that fits in f32` +
        " (upstream does not run CFG at 1 or below, so a value below 1 has no effect; CFG multiplies in f32)",
    );
  }
  const shift = request.shift ?? config.scheduler.shift;
  if (!Number.isFinite(shift) || shift <= 0) {
    throw new ModelInputError(`shift ${shift} is not a positive finite number`);
  }
  let schedule: WanUniPcSchedule;
  try {
    schedule = wanUniPcSchedule(steps, shift, WAN_UNIPC_CONFIG.numTrainTimesteps);
  } catch (error) {
    // A steps / shift combination that passed the single-value checks breaks the σ sequence (a
    // low-level RangeError — the MUST of scheduler.ts). The default combination has passed the manifest
    // check, so reaching here means the request passed the values.
    if (!(error instanceof RangeError)) throw error;
    throw new ModelInputError(
      `the pair of steps ${steps} and shift ${shift} cannot build a UniPC σ sequence (${error.message})`,
      { cause: error },
    );
  }

  // CFG's uncond side (upstream's `do_classifier_free_guidance = guidance_scale > 1`).
  let negative: Text | undefined;
  if (guidance > 1) {
    negative =
      request.negativePrompt !== undefined
        ? resolve(request.negativePrompt, "negativePrompt")
        : gate.defaultNegative();
  } else if (request.negativePrompt !== undefined) {
    throw new ModelInputError(
      "guidance 1 does not run the uncond side, so negativePrompt has no effect (to use it, set guidance above 1)",
    );
  }

  const frames = request.frames ?? generation.defaults.frames;
  if (
    !Number.isInteger(frames) ||
    frames < generation.minFrames ||
    frames > generation.maxFrames ||
    (frames - 1) % TEMPORAL_COMPRESSION !== 0
  ) {
    throw new ModelInputError(
      `frames ${frames} is not in the accepted set (4n+1 in ${generation.minFrames}..${generation.maxFrames})`,
    );
  }
  // The check of the I2V fields (ADR 0121 decisions 10 and 11). Only a generation whose DiT has
  // condition inputs (input form "ti2v") takes a condition image. A field whose value is undefined is read as
  // absent (the same reading as 2.2's T2V — it carries no image, so it does not create the false
  // expectation that I2V runs on 2.1).
  if (
    generation.ditInputForm !== "ti2v" &&
    (request.image !== undefined || request.fit !== undefined)
  ) {
    throw new ModelInputError(
      "image / fit (generation from an image) is not accepted by this generation (Wan2.2's WanTi2vPipeline accepts it)",
    );
  }
  if (request.fit !== undefined && !isWanI2vFit(request.fit)) {
    throw new Error(`fit '${String(request.fit)}' is neither 'crop' nor 'stretch'`);
  }
  if (request.image === undefined && request.fit !== undefined) {
    throw new ModelInputError(
      "fit takes effect only when image is passed (how the condition image is fitted to the size — do not pass it without image)",
    );
  }
  let width: number;
  let height: number;
  let conditionImage: WanI2vImageInput | undefined;
  if (request.image === undefined) {
    width = request.width ?? generation.defaults.width;
    height = request.height ?? generation.defaults.height;
    if (!generation.acceptedSizes.some((size) => size.width === width && size.height === height)) {
      throw new ModelInputError(
        `${width}×${height} is not in the accepted set (${generation.acceptedSizes
          .map((size) => `${size.width}×${size.height}`)
          .join(" / ")})`,
      );
    }
  } else {
    assertRgb8Shaped(request.image);
    // Choosing the output size (an omitted field follows the image's aspect ratio — an explicit field
    // is within the accepted set) and preprocessing.
    conditionImage = preprocessWanI2vImage(
      request.image,
      { width: request.width, height: request.height, fit: request.fit },
      generation,
    );
    width = conditionImage.width;
    height = conditionImage.height;
  }
  const compression = wanSpatialCompression(layout, generation);
  const latentShape: [number, number, number, number] = [
    layout.latentChannels,
    (frames - 1) / TEMPORAL_COMPRESSION + 1,
    height / compression,
    width / compression,
  ];
  if (!latentShape.every(Number.isInteger)) {
    // The accepted set holds only sizes divisible by the spatial compression — if one does not
    // divide, the asset is mixed up (not input-caused).
    throw new Error(
      `Latent shape [${latentShape}] is not integral (spatial compression ${compression})`,
    );
  }

  if (request.seed !== undefined && request.latents !== undefined) {
    throw new ModelInputError(
      "seed and latents are mutually exclusive (the initial noise comes from exactly one of them)",
    );
  }
  let initial: InitialNoise;
  if (request.latents !== undefined) {
    const { latents } = request;
    const count = latentShape.reduce((product, dim) => product * dim, 1);
    if (!(latents instanceof Float32Array) || latents.length !== count) {
      throw new ModelInputError(
        `latents is not a Float32Array [${latentShape.join(", ")}] (${count} elements)`,
      );
    }
    if (!latents.every(Number.isFinite))
      throw new ModelInputError("latents contains a non-finite value");
    initial = { kind: "latents", data: latents };
  } else {
    const seed = request.seed ?? DEFAULT_SEED;
    assertAcceptableSeed(seed);
    initial = { kind: "seed", seed };
  }

  const vaeTile = request.vaeTile ?? 16;
  // A value a JS caller passes past the type (`"32"`, 24, 0) is an input error, not a silent default.
  if (vaeTile !== 16 && vaeTile !== 32) {
    throw new ModelInputError(`vaeTile ${String(vaeTile)} is neither 16 nor 32`);
  }
  let tileLayout = layout;
  if (vaeTile === 32) {
    if (tile32?.layout === undefined) {
      const preset = tile32 === undefined ? "this preset" : `quant '${tile32.quant}'`;
      throw new ModelInputError(
        `vaeTile 32 is not available for ${preset}: the manifest has no tile-32 VAE decoder` +
          " (no vae_decoder_first_32 / vae_decoder_next_32 weights — use vaeTile 16)",
      );
    }
    tileLayout = tile32.layout;
  }
  // The overlap formula (64 px ÷ spatial compression) is computed after the latent integer check
  // (the same order as `assertWanVaeTilesCover`).
  const tiles = planWanGenerationTiles(tileLayout, latentShape[2], latentShape[3], generation);
  return {
    positive,
    negative,
    steps,
    guidance,
    shift,
    frames,
    width,
    height,
    latentShape,
    initial,
    schedule,
    vaeTile,
    tiles,
    conditionImage,
  };
};
