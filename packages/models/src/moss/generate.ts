/**
 * MOSS-Image text-to-image on the family's internal entry — the one acceptance check of a request
 * and the generate path `dit_text` → the DiT pair → the VAE (ADR NNNN decisions 5 and 9, plan items
 * 15-8 to 15-10, plan step 4). Not on a public surface: the family is in neither the barrel nor the
 * package exports (plan question 18 a).
 *
 * ## The request — precomputed text only (plan item 15-10)
 *
 * The text encoder is not ported, so a request carries the encoder outputs (`[T, 4096]` f32 — the
 * asset of `moss.text_embeds` holds upstream's for the fixed prompts) instead of strings. Every knob
 * is explicit: there is no manifest yet whose `pipelineConfig` would own the defaults (upstream's
 * 1024², 50 steps, guidance 3, shift 5 — ADR NNNN decision 9), and a second owner here would have to
 * be retired when it comes. The initial noise is exactly one of a `seed` (`./random.ts` — not
 * torch's `randn`) and the `latents` themselves (how the parity checks inject upstream's noise).
 *
 * ## Acceptance ({@link planMossGenerate} — every rejection a `ModelInputError`, before GPU work)
 *
 * Rejected, each naming the limit or the missing input:
 *
 * - the size — a side that is not a positive multiple of 16, a side over 65,536 px (`./request.ts`),
 *   an area over this device's limit of the untiled VAE (plan item 15-9 — `./vae-contract.ts`);
 * - the text (plan item 15-10) — the positive prompt's embeddings missing, the negative prompt's
 *   missing while guidance ≠ 1 runs the negative branch, or given at guidance 1 where they would
 *   have no effect; an embedding that is not f32 `[T, 4096]` with 1 ≤ T ≤ 4,096, all finite;
 * - upstream's own range of the knobs — steps a positive integer, guidance finite and ≥ 0, shift
 *   finite and > 0 (both finite in f32 too: the host multiplies by their f32 roundings), and a shift
 *   so small that the requested σ grid is not finite (below about 2.98·10⁻⁸, where
 *   `f32(shift − 1)` is −1 and the first requested σ divides by zero). This one narrows upstream:
 *   upstream snaps the infinite σ to the end of the training grid and runs (2 steps at shift 10⁻⁹
 *   give σ = [0.999, 0.001]); here it fails loudly instead of sampling a schedule its formula did
 *   not produce;
 * - the initial noise — both or neither of `seed` and `latents`, a seed that is not a non-negative
 *   safe integer, latents that are not a finite f32 `[128, H/16, W/16]`.
 *
 * Nothing else is refused: any step count (upstream de-duplicates the snapped schedule — `k ≤ n`
 * steps run; up to about 2·10⁶ requested steps at most the 1,000 training points, past that more
 * and not monotonic, as upstream — `./sampler.ts` "Very large step counts"), any guidance in
 * upstream's range (≠ 1 runs CFG with the skip-22 negative, upstream's default), any area under the
 * VAE limit. The schedule is walked on the host before any GPU work, in time linear in the step
 * count. Whether the DiT pair or the VAE fit this device's free VRAM is not an acceptance limit
 * (plan item 15-9): each stage checks it before it opens.
 *
 * ## The generate path ({@link generateMoss} — plan item 15-8)
 *
 * 1. The first headroom check, with nothing open: the largest of the three stages' needs (the bytes
 *    of `mossDitStageNeeds` and `mossVaeStageNeeds` — the estimates + the largest part + the margin)
 *    through `fitsHeadroom`. A refusal fails loudly before `dit_text` runs, so a generation the
 *    device cannot hold does not run the text stage and the whole DiT only to be refused at the VAE.
 * 2. `dit_text` (`./dit-stage.ts` — its own headroom check, closed before the pair opens).
 * 3. The DiT pair, open for the whole loop: per step the two forwards (the negative on `dit_skip22`),
 *    upstream's CFG in f32 and the Euler update (`./sampler.ts`); closed — borrower, lender, then the
 *    text contexts — before the VAE opens.
 * 4. The VAE (`./vae-stage.ts` — its own headroom check, its three Sessions closed before it
 *    returns).
 *
 * Each stage closes what it opened on every path, success or failure (`disposeSteps` in the stages),
 * and between stages the path holds nothing on the GPU (the refined texts and the latents are host
 * arrays), so a failure anywhere leaves no Session open. The headroom checks are facts of the moment,
 * not reservations (another process may take the memory between them).
 *
 * MUST: every module is side-effect free (no import-time execution, no global mutable state).
 */

import { fitsHeadroom, type GpuContext } from "@karume/runtime";

import { ModelInputError } from "../errors.ts";
import { assertAcceptableSeed } from "../request-gates.ts";
import { settleReleasedMemory } from "../session/settle-released-memory.ts";
import { MOSS_DIT, MOSS_DIT_SKIP22, MOSS_DIT_TEXT, type MossDitContract } from "./dit-contract.ts";
import {
  type MossDitComponents,
  type MossDitPair,
  type MossDitPlan,
  mossClassifierFreeGuidance,
  mossDitStageNeeds,
  planMossDit,
  runMossTextStage,
  withMossDitPair,
} from "./dit-stage.ts";
import { mossImageTokens } from "./dit-tokens.ts";
import { MossRandn } from "./random.ts";
import type { MossTextEmbeddings } from "./request.ts";
import { mossEulerStep, mossSchedule, type MossSchedule } from "./sampler.ts";
import type { MossVaeContract, MossVaeDeviceLimits } from "./vae-contract.ts";
import {
  decodeMossVae,
  type MossVaeComponents,
  type MossVaePlan,
  mossVaeStageNeeds,
  planMossVae,
} from "./vae-stage.ts";

const OWNER = "MOSS-Image";

/** A text-to-image request (module doc "The request"). */
export type MossGenerateRequest = {
  /** Pixels (a positive multiple of 16). */
  readonly width: number;
  /** Pixels (a positive multiple of 16). */
  readonly height: number;
  /** The positive prompt's encoder output `[T, 4096]`. */
  readonly positive: MossTextEmbeddings;
  /** The negative prompt's encoder output — required when `guidance` ≠ 1, refused at 1. */
  readonly negative?: MossTextEmbeddings;
  /** The requested step count (upstream's `num_inference_steps`; the snapped schedule may run fewer). */
  readonly steps: number;
  /** Upstream's `guidance_scale` (1 runs no negative branch). */
  readonly guidance: number;
  /** The scheduler's `shift` (upstream's pinned config: 5). */
  readonly shift: number;
  /** The seed of the initial noise (`./random.ts`) — exclusive with {@link latents}. */
  readonly seed?: number;
  /** The initial latents `[128, H/16, W/16]` — exclusive with {@link seed}. */
  readonly latents?: Float32Array<ArrayBuffer>;
};

/** Where the initial noise comes from. */
export type MossInitialNoise =
  | { readonly kind: "seed"; readonly seed: number }
  | { readonly kind: "latents"; readonly data: Float32Array<ArrayBuffer> };

/** An accepted request ({@link planMossGenerate}). */
export type MossGeneratePlan = {
  /** The DiT stage's plan (`negative` undefined exactly when guidance is 1). */
  readonly dit: MossDitPlan;
  readonly vae: MossVaePlan;
  readonly guidance: number;
  readonly schedule: MossSchedule;
  readonly initial: MossInitialNoise;
};

/** What the acceptance check reads of the series and the device. */
export type MossGenerateLimits = {
  readonly dit: Pick<MossDitContract, "latentChannels" | "trainingGrid">;
  readonly vae: Pick<MossVaeContract, "bytesPerCell" | "spatialScale">;
  readonly limits: MossVaeDeviceLimits;
};

/** Finite in f32 (the host multiplies by the f32 rounding of a knob). */
const finiteInF32 = (value: number): boolean => Number.isFinite(Math.fround(value));

/** The guidance range and the presence of the prompts it needs (plan item 15-10 — module doc). */
const assertGuidanceAndPrompts = (request: MossGenerateRequest): void => {
  // A caller outside the types (JavaScript) can leave the field out.
  if (request.positive === undefined) {
    throw new ModelInputError(
      `${OWNER}: the positive prompt's embeddings are missing (the entry takes precomputed text` +
        " encoder outputs [T, 4096] only — the text encoder is not ported)",
    );
  }
  const { guidance } = request;
  if (!finiteInF32(guidance) || guidance < 0) {
    throw new ModelInputError(
      `${OWNER}: guidance ${guidance} is not a finite number ≥ 0 that fits in f32` +
        " (upstream's range; CFG multiplies by its f32 rounding)",
    );
  }
  if (guidance !== 1 && request.negative === undefined) {
    throw new ModelInputError(
      `${OWNER}: guidance ${guidance} runs the negative branch (CFG with block 22 skipped), so the` +
        " negative prompt's embeddings are required (upstream's default negative is the empty" +
        " prompt — the text-embeddings asset's 'negative')",
    );
  }
  if (guidance === 1 && request.negative !== undefined) {
    throw new ModelInputError(
      `${OWNER}: guidance 1 runs no negative branch, so the negative prompt's embeddings would have` +
        " no effect (leave them out, or set guidance ≠ 1)",
    );
  }
};

/** The schedule of an accepted step count and shift (module doc "Acceptance"). */
const scheduleOf = (
  grid: MossDitContract["trainingGrid"],
  steps: number,
  shift: number,
): MossSchedule => {
  if (!Number.isSafeInteger(steps) || steps < 1) {
    throw new ModelInputError(`${OWNER}: steps ${steps} is not a positive integer`);
  }
  if (!(shift > 0) || !finiteInF32(shift)) {
    throw new ModelInputError(
      `${OWNER}: shift ${shift} is not a finite number > 0 that fits in f32 (upstream's range)`,
    );
  }
  try {
    return mossSchedule(grid, steps, shift);
  } catch (error) {
    // The single-value checks passed, so a broken requested grid is the pair's (a tiny shift).
    if (!(error instanceof RangeError)) throw error;
    throw new ModelInputError(
      `${OWNER}: steps ${steps} with shift ${shift} give no finite σ schedule (${error.message})`,
      { cause: error },
    );
  }
};

/** The initial noise of a request (module doc "Acceptance"). */
const initialOf = (request: MossGenerateRequest, count: number): MossInitialNoise => {
  const { seed, latents } = request;
  if (seed !== undefined && latents !== undefined) {
    throw new ModelInputError(
      `${OWNER}: seed and latents are mutually exclusive (the initial noise comes from exactly one)`,
    );
  }
  if (latents !== undefined) {
    if (!(latents instanceof Float32Array) || latents.length !== count) {
      throw new ModelInputError(`${OWNER}: latents are not a Float32Array of ${count} values`);
    }
    const broken = latents.findIndex((value) => !Number.isFinite(value));
    if (broken !== -1) {
      throw new ModelInputError(
        `${OWNER}: latents element ${broken} is non-finite (${latents[broken]})`,
      );
    }
    return { kind: "latents", data: latents };
  }
  if (seed === undefined) {
    throw new ModelInputError(`${OWNER}: the initial noise needs a seed or the latents`);
  }
  try {
    assertAcceptableSeed(seed);
  } catch (error) {
    // The shared check owns the accepted set; the family names itself like its other rejections.
    if (!(error instanceof ModelInputError)) throw error;
    throw new ModelInputError(`${OWNER}: ${error.message}`, { cause: error });
  }
  return { kind: "seed", seed };
};

/**
 * The one acceptance check of a generation request (module doc "Acceptance"): returns the plan of
 * every stage or throws a `ModelInputError` naming the limit or the missing input.
 */
export const planMossGenerate = (
  request: MossGenerateRequest,
  input: MossGenerateLimits,
): MossGeneratePlan => {
  const { width, height } = request;
  const vae = planMossVae({ width, height }, input.vae, input.limits);
  assertGuidanceAndPrompts(request);
  const { guidance } = request;
  // The negative is present exactly when guidance ≠ 1 (assertGuidanceAndPrompts).
  const dit = planMossDit({
    width,
    height,
    positive: request.positive,
    negative: request.negative,
  });
  const schedule = scheduleOf(input.dit.trainingGrid, request.steps, request.shift);
  const initial = initialOf(request, input.dit.latentChannels * mossImageTokens(dit.cells));
  return { dit, vae, guidance, schedule, initial };
};

/** The two series opened for one device (built once — the contracts settled in admission). */
export type MossGenerateState = {
  readonly gpu: GpuContext;
  readonly dit: { readonly components: MossDitComponents; readonly contract: MossDitContract };
  readonly vae: { readonly components: MossVaeComponents; readonly contract: MossVaeContract };
};

/** A generated image. */
export type MossGeneratedImage = {
  /** `[3, height, width]` in [-1, 1] (upstream's decode, clamped — `output_type="pt"` before its `/2 + 0.5`). */
  readonly image: Float32Array<ArrayBuffer>;
  /** The final DiT latents `[128, H/16, W/16]` (the VAE stage's input). */
  readonly latents: Float32Array<ArrayBuffer>;
  readonly width: number;
  readonly height: number;
};

/**
 * Step 1 of the generate path (module doc): the largest stage need through one trial allocation,
 * with nothing open.
 */
const assertFirstHeadroom = async (
  state: MossGenerateState,
  plan: MossGeneratePlan,
): Promise<void> => {
  const { gpu } = state;
  const dit = mossDitStageNeeds({ ...state.dit, limits: gpu.limits }, plan.dit);
  const needs: readonly (readonly [string, number])[] = [
    [MOSS_DIT_TEXT, dit.text],
    [`${MOSS_DIT} + ${MOSS_DIT_SKIP22}`, dit.pair],
    ["VAE", mossVaeStageNeeds({ ...state.vae, limits: gpu.limits }, plan.vae)],
  ];
  const [stage, bytes] = needs.reduce((largest, need) => (need[1] > largest[1] ? need : largest));
  await settleReleasedMemory(gpu);
  if (await fitsHeadroom(gpu, bytes)) return;
  const { texts } = plan.dit;
  throw new Error(
    `${OWNER}: this generation needs up to ${bytes} bytes of VRAM at its largest stage (${stage})` +
      ` that this device cannot allocate now (checked before any stage opens) —` +
      ` ${plan.vae.width}×${plan.vae.height} px with text of ${texts.positive.tokens} /` +
      ` ${texts.negative?.tokens ?? "-"} tokens does not fit`,
  );
};

/**
 * The initial latents `[C, H/16, W/16]` of an accepted request: the injected latents, or
 * `C · N` normals of the seed's sequence (`./random.ts` — the first `C · N` values, nothing drawn
 * before them). NOTE: `export` is for the host tests (not on a public surface).
 */
export const mossInitialLatents = (
  plan: Pick<MossGeneratePlan, "dit" | "initial">,
  latentChannels: number,
): Float32Array<ArrayBuffer> =>
  plan.initial.kind === "latents"
    ? plan.initial.data
    : new MossRandn(plan.initial.seed).normals(latentChannels * mossImageTokens(plan.dit.cells));

/**
 * Step 3 of the generate path (module doc): the sampler loop over `predict` (the open pair's — the
 * host tests drive it with upstream's recorded DiT outputs instead). The guidance decides whether a
 * step combines two branches; a `predict` that returns a negative branch at guidance 1, or none at
 * guidance ≠ 1, is a wiring bug and throws (no silent fall back to the positive alone). NOTE:
 * `export` is for those tests (not on a public surface).
 */
export const mossDenoise = async (
  predict: MossDitPair["predict"],
  plan: Pick<MossGeneratePlan, "schedule" | "guidance">,
  initial: Float32Array<ArrayBuffer>,
): Promise<Float32Array<ArrayBuffer>> => {
  const { schedule, guidance } = plan;
  const guided = guidance !== 1;
  let latents = initial;
  for (let step = 0; step < schedule.gridIndices.length; step += 1) {
    const { positive, negative } = await predict(latents, schedule.gridIndices[step]);
    if (guided !== (negative !== undefined)) {
      throw new Error(
        `${OWNER}: step ${step} at guidance ${guidance} returned ` +
          `${negative === undefined ? "no negative branch" : "a negative branch"}` +
          ` (guidance ≠ 1 runs exactly the two branches, guidance 1 the positive alone)`,
      );
    }
    const prediction =
      negative === undefined ? positive : mossClassifierFreeGuidance(positive, negative, guidance);
    latents = mossEulerStep(latents, prediction, schedule.sigmas[step], schedule.sigmas[step + 1]);
  }
  return latents;
};

/** Runs one generation (module doc "The generate path"). */
export const generateMoss = async (
  state: MossGenerateState,
  request: MossGenerateRequest,
): Promise<MossGeneratedImage> => {
  const { gpu } = state;
  const plan = planMossGenerate(request, {
    dit: state.dit.contract,
    vae: state.vae.contract,
    limits: gpu.limits,
  });
  await assertFirstHeadroom(state, plan);
  const initial = mossInitialLatents(plan, state.dit.contract.latentChannels);
  const ditState = { gpu, ...state.dit };
  const refined = await runMossTextStage(ditState, plan.dit);
  const latents = await withMossDitPair(ditState, plan.dit, refined, (pair) =>
    mossDenoise(pair.predict, plan, initial),
  );
  const image = await decodeMossVae({ gpu, ...state.vae }, plan.vae, latents);
  return { image, latents, width: plan.vae.width, height: plan.vae.height };
};
