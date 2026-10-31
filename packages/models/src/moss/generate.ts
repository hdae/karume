/**
 * MOSS-Image text-to-image on the family's internal entry — the one acceptance check of a request
 * and the generate path (the text encoder →) `dit_text` → the DiT pair → the VAE (ADR 0138
 * decisions 5 and 9). Not on a public surface: the family is in neither the barrel nor the package
 * exports (ADR 0138 decision 12).
 *
 * ## The request — the text in one of two forms (ADR 0138 decision 9)
 *
 * - **Prompt strings** (`prompt`, `negativePrompt`): rendered with the Qwen3.5 chat template and
 *   tokenized in the acceptance check, encoded on the GPU by the text-encoder stage
 *   (`./text-stage.ts` — ADR 0137's encoder). The state then carries the opened Qwen3.5 text series
 *   (`text`). `overflow` is the tokenizer's over-length policy (the maintainer's ruling 50: rejected
 *   by default, `"truncate"` cuts as upstream does).
 * - **Precomputed embeddings** (`positive`, `negative`): the encoder outputs `[T, 4096]` f32 (the
 *   asset of `moss.text_embeds` holds upstream's for the fixed prompts) — how the parity checks and
 *   an A/B against upstream's embeddings feed the DiT.
 *
 * One request uses one form for both branches (a mix, or both forms of one branch, is refused).
 * Every knob is explicit: there is no manifest yet whose `pipelineConfig` would own the defaults
 * (upstream's 1024², 50 steps, guidance 3, shift 5 and its empty negative prompt — ADR 0138
 * decision 9), and a second owner here would have to be retired when it comes. So the negative
 * prompt has no default either: at guidance ≠ 1 the prompt form needs `negativePrompt` (upstream's
 * default is `""`), as the embeddings form needs `negative`. The initial noise is exactly one of a
 * `seed` (`./random.ts` — not torch's `randn`) and the `latents` themselves (how the parity checks
 * inject upstream's noise).
 *
 * The preset (`./preset.ts`) is part of the state, not of the request, and has no default either
 * (no manifest owns one yet — ADR 0138 decision 13; the family's default
 * `f32+dit8-a8-attn8-online` is applied by the manifest when it comes): it sets the knobs of the
 * DiT pair and of its headroom estimates; `dit_text` and the VAE run in f32 in every preset.
 *
 * ## Acceptance ({@link planMossGenerate} — every rejection a `ModelInputError`, before GPU work)
 *
 * Rejected, each naming the limit or the missing input:
 *
 * - the size — a side that is not a positive multiple of 16, a side over 65,536 px (`./request.ts`),
 *   an area over this device's limit of the untiled VAE (ADR 0138 decision 8 —
 *   `./vae-contract.ts`);
 * - the text (ADR 0138 decision 9) — both forms given, or neither; the positive prompt missing; the
 *   negative prompt missing while guidance ≠ 1 runs the negative branch, or given at guidance 1
 *   where it would have no effect; `overflow` with the embeddings form (no tokenizer runs there);
 *   in the prompt form what the chat template refuses (a prompt that is not a string, a
 *   `<tool_response>`-wrapped one) and what the tokenizer refuses (more than 4,096 tokens under
 *   `"reject"`, an unpaired surrogate, an unknown `overflow`); in the embeddings form an embedding
 *   that is not f32 `[T, 4096]` with 1 ≤ T ≤ 4,096, all finite;
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
 * steps run; past about 2·10⁶ requested steps the schedule stops being monotonic, and it holds more
 * than the 1,000 training points only at about 10⁷ — 1,099; 989 at 5·10⁶ — as upstream:
 * `./sampler.ts` "Very large step counts"), any guidance in upstream's range (≠ 1 runs CFG with the
 * skip-22 negative, upstream's default), any area under the VAE limit. The schedule is walked on
 * the host before any GPU work, in time linear in the step count. Whether the DiT pair or the VAE
 * fit this device's free VRAM is not an acceptance limit (ADR 0138 decisions 5 and 8): each stage
 * checks it before it opens.
 *
 * ## The generate path ({@link generateMoss} — ADR 0138 decision 5)
 *
 * 1. The first headroom check, with nothing open: the largest of the stages' needs (the bytes of
 *    `mossTextEncoderStageNeed` in the prompt form, `mossDitStageNeeds` and `mossVaeStageNeeds` —
 *    the estimates + the largest part + the margin) through `fitsHeadroom`. A refusal fails loudly
 *    before the first stage opens, so a generation the device cannot hold does not upload the text
 *    encoder and run the whole DiT only to be refused at the VAE.
 * 2. In the prompt form, the text encoder (`./text-stage.ts` — its own headroom check; disposed and
 *    its memory released on every path before `dit_text` opens).
 * 3. `dit_text` (`./dit-stage.ts` — its own headroom check, closed before the pair opens).
 * 4. The DiT pair, open for the whole loop: per step the two forwards (the negative on `dit_skip22`),
 *    upstream's CFG in f32 and the Euler update (`./sampler.ts`); closed — borrower, lender, then the
 *    text contexts — before the VAE opens.
 * 5. The VAE (`./vae-stage.ts` — its own headroom check, its three Sessions closed before it
 *    returns).
 *
 * There is no cancellation on this entry (no stage of the DiT and VAE side has one); the text
 * stage takes a `signal` of its own (`encodeMossPrompts`), which this entry does not pass.
 *
 * Each stage closes what it opened on every path, success or failure (`disposeSteps` in the stages),
 * and between stages the path holds nothing on the GPU (the encoder rows, the refined texts and the
 * latents are host arrays), so a failure anywhere leaves no Session open. The headroom checks are facts of the moment,
 * not reservations (another process may take the memory between them).
 *
 * MUST: every module is side-effect free (no import-time execution, no global mutable state).
 */

import { fitsHeadroom, type GpuContext } from "@karume/runtime";

import { ModelInputError } from "../errors.ts";
import type { Qwen3_5Overflow } from "../qwen3_5/tokenizer.ts";
import { assertAcceptableSeed } from "../request-gates.ts";
import { settleReleasedMemory } from "../session/settle-released-memory.ts";
import { MOSS_DIT, MOSS_DIT_SKIP22, MOSS_DIT_TEXT, type MossDitContract } from "./dit-contract.ts";
import {
  type MossBranches,
  type MossDitComponents,
  type MossDitPair,
  type MossDitPlan,
  type MossDitShape,
  mossClassifierFreeGuidance,
  mossDeviceFacts,
  mossDitStageNeeds,
  planMossDit,
  runMossTextStage,
  withMossDitPair,
} from "./dit-stage.ts";
import { mossImageTokens } from "./dit-tokens.ts";
import { type MossPreset, requireMossPreset } from "./preset.ts";
import { MossRandn } from "./random.ts";
import type { MossTextEmbeddings } from "./request.ts";
import { mossEulerStep, mossSchedule, type MossSchedule } from "./sampler.ts";
import {
  encodeMossPrompts,
  MOSS_TEXT_ENCODER,
  mossPromptIds,
  mossTextEncoderStageNeed,
  type MossTextEncoderState,
  requireMossTextEncoder,
} from "./text-stage.ts";
import type { MossVaeContract, MossVaeDeviceLimits } from "./vae-contract.ts";
import {
  decodeMossVae,
  type MossVaeComponents,
  type MossVaePlan,
  mossVaeStageNeeds,
  planMossVae,
} from "./vae-stage.ts";

const OWNER = "MOSS-Image";

/**
 * A text-to-image request (module doc "The request"): the text is either `prompt` /
 * `negativePrompt` or `positive` / `negative`, never a mix.
 */
export type MossGenerateRequest = {
  /** Pixels (a positive multiple of 16). */
  readonly width: number;
  /** Pixels (a positive multiple of 16). */
  readonly height: number;
  /** The positive prompt as a string (the prompt form) — exclusive with {@link positive}. */
  readonly prompt?: string;
  /**
   * The negative prompt as a string — in the prompt form required when `guidance` ≠ 1 (upstream's
   * default is `""`; there is no default here), refused at 1.
   */
  readonly negativePrompt?: string;
  /** The tokenizer's over-length policy of the prompt form (default `"reject"` — ruling 50). */
  readonly overflow?: Qwen3_5Overflow;
  /** The positive prompt's encoder output `[T, 4096]` (the embeddings form) — exclusive with {@link prompt}. */
  readonly positive?: MossTextEmbeddings;
  /** The negative prompt's encoder output — in the embeddings form required when `guidance` ≠ 1, refused at 1. */
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

/**
 * An accepted request ({@link planMossGenerate}). `dit` holds the latent grid and each branch's
 * token count in both forms (`negative` undefined exactly when guidance is 1); the embeddings form
 * carries the DiT stage's whole plan, the prompt form each branch's token ids for the text stage.
 */
export type MossGeneratePlan = {
  readonly vae: MossVaePlan;
  readonly guidance: number;
  readonly schedule: MossSchedule;
  readonly initial: MossInitialNoise;
} & (
  | { readonly text: "embeddings"; readonly dit: MossDitPlan }
  | {
      readonly text: "prompts";
      readonly dit: MossDitShape;
      readonly prompts: MossBranches<Int32Array<ArrayBuffer>>;
    }
);

/** What the acceptance check reads of the series and the device. */
export type MossGenerateLimits = {
  readonly dit: Pick<MossDitContract, "latentChannels" | "trainingGrid">;
  readonly vae: Pick<MossVaeContract, "bytesPerCell" | "spatialScale">;
  readonly limits: MossVaeDeviceLimits;
  /** The tokenizer (the prompt form only — a prompt-form request without it is refused). */
  readonly text?: Pick<MossTextEncoderState, "tokenizer">;
};

/** Finite in f32 (the host multiplies by the f32 rounding of a knob). */
const finiteInF32 = (value: number): boolean => Number.isFinite(Math.fround(value));

/** The text of a request in its one form (module doc "The request"). */
type MossRequestText =
  | {
      readonly form: "embeddings";
      readonly positive: MossTextEmbeddings;
      readonly negative: MossTextEmbeddings | undefined;
    }
  | {
      readonly form: "prompts";
      readonly prompt: string;
      readonly negativePrompt: string | undefined;
    };

/** The form of a request's text and its positive prompt (ADR 0138 decision 9 — module doc). */
const requestTextOf = (request: MossGenerateRequest): MossRequestText => {
  const { prompt, negativePrompt, positive, negative } = request;
  const strings = prompt !== undefined || negativePrompt !== undefined;
  const embeddings = positive !== undefined || negative !== undefined;
  if (strings && embeddings) {
    throw new ModelInputError(
      `${OWNER}: the text is given both as prompt strings (prompt / negativePrompt) and as` +
        " precomputed embeddings (positive / negative) — give one form for both branches",
    );
  }
  if (strings) {
    if (prompt === undefined) {
      throw new ModelInputError(
        `${OWNER}: negativePrompt is given without prompt (the positive prompt is required)`,
      );
    }
    return { form: "prompts", prompt, negativePrompt };
  }
  // A caller outside the types (JavaScript) can leave every text field out.
  if (positive === undefined) {
    throw new ModelInputError(
      `${OWNER}: the positive prompt's embeddings are missing and no prompt string is given (the` +
        " text is a prompt string `prompt` or precomputed encoder outputs [T, 4096] `positive`)",
    );
  }
  if (request.overflow !== undefined) {
    throw new ModelInputError(
      `${OWNER}: overflow applies to prompt strings only (the embeddings form runs no tokenizer)`,
    );
  }
  return { form: "embeddings", positive, negative };
};

/**
 * The guidance range and the presence of the negative prompt it needs, in the request's form (ADR
 * 0138 decision 9 — module doc). Returns the guidance.
 */
const guidanceOf = (request: MossGenerateRequest, text: MossRequestText): number => {
  const { guidance } = request;
  if (!finiteInF32(guidance) || guidance < 0) {
    throw new ModelInputError(
      `${OWNER}: guidance ${guidance} is not a finite number ≥ 0 that fits in f32` +
        " (upstream's range; CFG multiplies by its f32 rounding)",
    );
  }
  const [given, missing, unused] =
    text.form === "embeddings"
      ? [
          text.negative !== undefined,
          "the negative prompt's embeddings are required (upstream's default negative is the" +
            " empty prompt — the text-embeddings asset's 'negative')",
          "the negative prompt's embeddings would have no effect (leave them out, or set" +
            " guidance ≠ 1)",
        ]
      : [
          text.negativePrompt !== undefined,
          // No default (module doc "The request"): upstream's is named so the caller can pass it.
          "negativePrompt is required (upstream's default negative is the empty prompt: pass" +
            ' negativePrompt: "")',
          "negativePrompt would have no effect (leave it out, or set guidance ≠ 1)",
        ];
  if (guidance !== 1 && !given) {
    throw new ModelInputError(
      `${OWNER}: guidance ${guidance} runs the negative branch (CFG with block 22 skipped), so ${missing}`,
    );
  }
  if (guidance === 1 && given) {
    throw new ModelInputError(`${OWNER}: guidance 1 runs no negative branch, so ${unused}`);
  }
  return guidance;
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
 * The prompt form's part of a plan (module doc "Acceptance"): each branch's token ids — the chat
 * template and the tokenizer, before any GPU work — and their counts.
 */
const promptPlanOf = (
  text: Extract<MossRequestText, { readonly form: "prompts" }>,
  overflow: Qwen3_5Overflow | undefined,
  cells: MossDitShape["cells"],
  { tokenizer }: Pick<MossTextEncoderState, "tokenizer">,
) => {
  const prompts = {
    positive: mossPromptIds(tokenizer, text.prompt, "prompt", overflow),
    negative:
      text.negativePrompt === undefined
        ? undefined
        : mossPromptIds(tokenizer, text.negativePrompt, "negativePrompt", overflow),
  };
  const dit: MossDitShape = {
    cells,
    texts: {
      positive: { tokens: prompts.positive.length },
      negative: prompts.negative === undefined ? undefined : { tokens: prompts.negative.length },
    },
  };
  return { text: "prompts" as const, dit, prompts };
};

/**
 * The one acceptance check of a generation request (module doc "Acceptance"): returns the plan of
 * every stage or throws a `ModelInputError` naming the limit or the missing input (a prompt-form
 * request without a tokenizer is the caller's assembly — a plain `Error`).
 */
export const planMossGenerate = (
  request: MossGenerateRequest,
  input: MossGenerateLimits,
): MossGeneratePlan => {
  const { width, height } = request;
  const vae = planMossVae({ width, height }, input.vae, input.limits);
  const text = requestTextOf(request);
  // The negative is present exactly when guidance ≠ 1 (guidanceOf).
  const guidance = guidanceOf(request, text);
  const textPlan =
    text.form === "embeddings"
      ? {
          text: "embeddings" as const,
          dit: planMossDit({ width, height, positive: text.positive, negative: text.negative }),
        }
      : promptPlanOf(text, request.overflow, vae.cells, requireMossTextEncoder(input.text));
  const schedule = scheduleOf(input.dit.trainingGrid, request.steps, request.shift);
  const initial = initialOf(
    request,
    input.dit.latentChannels * mossImageTokens(textPlan.dit.cells),
  );
  return { ...textPlan, vae, guidance, schedule, initial };
};

/**
 * The series opened for one device (built once — the contracts settled in admission) and the
 * preset the generation runs (explicit — module doc "The request"). `text` is the Qwen3.5 text
 * series (`./text-stage.ts` — `mossTextEncoderContract`), needed by the prompt form only.
 */
export type MossGenerateState = {
  readonly gpu: GpuContext;
  readonly preset: MossPreset;
  readonly dit: { readonly components: MossDitComponents; readonly contract: MossDitContract };
  readonly vae: { readonly components: MossVaeComponents; readonly contract: MossVaeContract };
  readonly text?: MossTextEncoderState;
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
  const dit = mossDitStageNeeds(
    {
      ...state.dit,
      preset: state.preset,
      device: mossDeviceFacts(gpu),
    },
    plan.dit,
  );
  const needs: readonly (readonly [string, number])[] = [
    ...(plan.text === "prompts"
      ? [
          [
            MOSS_TEXT_ENCODER,
            mossTextEncoderStageNeed(
              { ...requireMossTextEncoder(state.text), limits: gpu.limits },
              plan.prompts,
            ),
          ] as const,
        ]
      : []),
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
  const preset = requireMossPreset(state.preset);
  const plan = planMossGenerate(request, {
    dit: state.dit.contract,
    vae: state.vae.contract,
    limits: gpu.limits,
    ...(state.text === undefined ? {} : { text: state.text }),
  });
  await assertFirstHeadroom(state, plan);
  const initial = mossInitialLatents(plan, state.dit.contract.latentChannels);
  const ditState = { gpu, preset, ...state.dit };
  // The prompt form's text stage runs (and is closed) before dit_text opens (module doc, step 2).
  const ditPlan: MossDitPlan =
    plan.text === "embeddings"
      ? plan.dit
      : {
          cells: plan.dit.cells,
          texts: await encodeMossPrompts(
            { gpu, ...requireMossTextEncoder(state.text) },
            plan.prompts,
          ),
        };
  const refined = await runMossTextStage(ditState, ditPlan);
  const latents = await withMossDitPair(ditState, ditPlan, refined, (pair) =>
    mossDenoise(pair.predict, plan, initial),
  );
  const image = await decodeMossVae({ gpu, ...state.vae }, plan.vae, latents);
  return { image, latents, width: plan.vae.width, height: plan.vae.height };
};
