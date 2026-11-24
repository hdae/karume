/**
 * The public types of Wan's text (+ condition image) → video pipeline (`Ti2vPipeline` in
 * `./ti2v-pipeline.ts`): the result, the generation events, the request (the text-to-video part —
 * `Ti2vGenerateRequest` adds the I2V condition image), the construction options and the
 * already-fetched input. `@karume/models/wan` exports them (ADR 0008).
 *
 * The class and the family's shared stages (`./family.ts`, `./plan.ts`, `./text-stage.ts`,
 * `./dit-loop.ts`, `./tile-decode.ts`, `./vae-encoder.ts`, `./frames.ts`) take them with
 * `import type` only, so this module holds no value and creates no import cycle.
 *
 * MUST: every module is side-effect free (no import-time execution, no global mutable state —
 * CLAUDE.md).
 */

import type { GpuContext, SessionDiagnostics } from "@karume/runtime";
import type { Manifest } from "@karume/hub";

import type {
  FromPretrainedComponentOptions,
  FromPretrainedHubOptions,
} from "../hub/load-options.ts";
import type { PipelineFusionRules } from "../session/fusion-rules.ts";

/** The generation result. `data` is `[3, frames, height, width]` f32 (range `[-1, 1]` — clamped). */
export type GeneratedVideo = {
  readonly frames: number;
  readonly width: number;
  readonly height: number;
  /**
   * The output frame rate (an upstream fact of the generation — 24 for Wan2.2. Neither declared in
   * the manifest nor a knob — ADR 0121 decision 8).
   */
  readonly fps: number;
  readonly data: Float32Array<ArrayBuffer>;
};

/**
 * A copy of the intermediate latent returned by `copyLatents()` of `denoise-step` (the generation's
 * latent shape — `[48, F', H/16, W/16]` for Wan2.2).
 */
export type LatentSnapshot = {
  readonly data: Float32Array<ArrayBuffer>;
  readonly shape: readonly number[];
};

/**
 * The component names that {@link PipelineOptions.onRunDiagnostics} receives (one Session = one
 * name). `text_encoder` is the umT5 of the `"gpu"` path (once each for positive / negative).
 * `vae_encoder_*` are the three graphs of the I2V encoder (once each for a request that passes a
 * condition image). `vae_decoder_first_32` / `vae_decoder_next_32` are the decoder's 32-latent pair
 * that runs for a request with {@link GenerateRequest.vaeTile} 32 (in place of
 * `vae_decoder_first` / `vae_decoder_next` — the 16-latent pair only lends its weights then and does
 * not run).
 */
export type RunComponent =
  | "vae_encoder_pre"
  | "vae_encoder_attn"
  | "vae_encoder_post"
  | "text_encoder"
  | "transformer"
  | "vae_decoder_first"
  | "vae_decoder_next"
  | "vae_decoder_first_32"
  | "vae_decoder_next_32";

/** The generation events that {@link GenerateRequest.onEvent} receives. */
export type GenerateEvent =
  /**
   * The start of a stage (`start` — before its Session is created) and its end (`end` — after its
   * Session is closed). `vae_encoder` is the stage that encodes the I2V condition image (only for a
   * request that passes `image`, and the first stage — `end` comes after the three graphs are closed
   * and the release has been waited for). `text_encoder` is the umT5 stage of the `"gpu"` path (absent
   * with `"precomputed"`). `vae_decoder` is the stage of the two chunk graphs. If a stage fails midway,
   * no `end` is emitted.
   */
  | {
      readonly kind: "stage";
      readonly component: "vae_encoder" | "text_encoder" | "transformer" | "vae_decoder";
      readonly at: "start" | "end";
    }
  | {
      readonly kind: "denoise-step";
      /** The number of completed steps (1-based). */
      readonly step: number;
      readonly steps: number;
      /** The timestep (an integer) passed to the DiT at that step. */
      readonly timestep: number;
      /**
       * Copies and returns the intermediate latent only when called (the latent after the UniPC
       * update). For I2V (a request that passes `image`) this is the scheduler state itself, and its
       * first latent frame has not been replaced by the condition latent — the replacement happens
       * once at the end of the DiT stage (the same value as the `latents` that upstream diffusers'
       * `callback_on_step_end` receives).
       */
      readonly copyLatents: () => LatentSnapshot;
    }
  /** Completion of the decode of one VAE tile (`tile` is 1-based). */
  | { readonly kind: "vae-tile"; readonly tile: number; readonly tiles: number };

/**
 * One generation request (the text-to-video part — `Ti2vGenerateRequest` adds the I2V condition
 * image). An omitted step count, guidance or shift takes the default of the manifest's
 * `pipelineConfig` (the `karume-wan2.2` distribution declares 50 steps, guide 5.0 and the upstream
 * scheduler's `flow_shift` 5.0, the official 720p value — ADR 0121 stage 8). The accepted set and
 * defaults of size and frame count are the generation's (1280×704 / 704×1280, 5–121 frames, default
 * 1280×704, 33 frames).
 */
export type GenerateRequest = {
  /**
   * The prompt. The accepted set depends on the path chosen at construction
   * ({@link PipelineOptions.textEncoder}):
   *
   * - `"gpu"` (default): any string. It passes through the mirror of the upstream preprocessing
   *   (`prompt_clean`) and the tokenizer check; input that does not pass is a `ModelInputError` (ADR
   *   0119 decisions 1, 2 and 4) — out-of-vocabulary characters, characters unassigned in Unicode
   *   16.0.0, C1 control characters, sequences that could be HTML character references (`R&D` passes
   *   when spaced as `R & D`), sequences that look like mojibake, special tokens in the text, `▁` right
   *   after whitespace, empty or whitespace only (1 token), more than 512 tokens. The rejection
   *   message says how to fix the input.
   * - `"precomputed"`: **only strings in the text embedding asset's set** (an exact match of either the
   *   original or the normalized string — `Ti2vPipeline.prompts`). Anything outside the set is a
   *   `ModelInputError` (ADR 0118 decision 4).
   */
  readonly prompt: string;
  /**
   * The prompt for CFG's uncond side (the same accepted set as {@link GenerateRequest.prompt}).
   * When omitted, the official `sample_neg_prompt` is used (`"gpu"`: the default string encoded on the
   * GPU; `"precomputed"`: the asset's `negative` row — the same original text). Passing it when
   * `guidance` is 1 (CFG does not run) is a `ModelInputError` (a knob that has no effect is not
   * silently accepted).
   */
  readonly negativePrompt?: string;
  /** The seed of the initial noise (default 0 — exclusive with {@link GenerateRequest.latents}). */
  readonly seed?: number;
  /**
   * Passes the initial noise in from outside (f32, the generation's latent shape —
   * `[48, F', H/16, W/16]` with `F' = (frames − 1)/4 + 1`). The entry point for injecting torch's
   * `randn` stream when comparing with the reference (the seeded generator is a different stream from
   * torch — `random.ts`). Not modified.
   */
  readonly latents?: Float32Array<ArrayBuffer>;
  /**
   * The number of denoise steps (an integer from 1 to 1000; default `pipelineConfig.defaults.steps`).
   * The upper bound is `num_train_timesteps` (beyond it, integer timesteps always repeat). A value
   * whose combination with `shift` gives a σ sequence that is not strictly decreasing is a
   * `ModelInputError`.
   */
  readonly steps?: number;
  /**
   * The CFG strength (at least 1 and finite after rounding to f32; default
   * `pipelineConfig.defaults.guidance`). At 1 the uncond side does not run (the same test as
   * upstream's `guidance_scale > 1`). Below 1 upstream turns CFG off entirely and the value has no
   * effect, so it is a `ModelInputError`.
   */
  readonly guidance?: number;
  /**
   * The flow-matching shift (positive; default `pipelineConfig.scheduler.shift`). A value that breaks
   * the σ sequence in combination with `steps` (an extremely large / small shift) is a
   * `ModelInputError`.
   */
  readonly shift?: number;
  /** The frame count (4n+1 in 5–121; default 33). */
  readonly frames?: number;
  /** Width × height (1280×704 or 704×1280; default 1280×704). */
  readonly width?: number;
  readonly height?: number;
  /**
   * The edge of the VAE decoder's latent tile (default 16 — ADR 0134). `32` decodes in 32-latent
   * tiles with the same 4-latent overlap: at 1280×704 that is 2×3 tiles instead of 4×7 (about a
   * third less decoder work), and the decoder stage holds about four times the causal caches and
   * frame backings (about 9.5 GiB on the int8 decoder at 1280×704 — the 10 GB class cannot hold it).
   * It is accepted only when the preset's weights carry the 32-latent pair (`vae_decoder_first_32`
   * / `vae_decoder_next_32` — Wan2.2's presets do); otherwise, and for any value other than 16 or
   * 32, the request is a `ModelInputError`. Never chosen automatically (WebGPU exposes no free-VRAM
   * figure — ADR 0070 decision 5). The output differs from tile 16's in the blended seams.
   */
  readonly vaeTile?: 16 | 32;
  /**
   * Observer of generation events ({@link GenerateEvent}). **It is awaited** (the firing order is
   * deterministic). Exceptions are not swallowed — a throw cancels at step granularity (the stage's
   * Session is closed).
   *
   * MUST: do not await `generate` / `dispose` of the same pipeline inside `onEvent` (a self-deadlock on
   * the serialization chain).
   */
  readonly onEvent?: (event: GenerateEvent) => void | Promise<void>;
  /**
   * Cancels the generation. Checked at stage boundaries (text → DiT → VAE), before each run (text
   * positive / negative, each DiT step) and between VAE tiles, each time after yielding once to the
   * event loop. One run is indivisible; the cancellation takes effect at the next boundary. On
   * cancellation the open Sessions are closed and `signal.reason` is thrown **as is** (unwrapped — the
   * caller can identify its own cancellation by `error === controller.signal.reason`). The same
   * pipeline can run the next `generate` after a cancellation.
   */
  readonly signal?: AbortSignal;
};

/** Construction options (`Ti2vPipeline.fromAssets` / `Ti2vPipeline.fromPretrained`). */
export type PipelineOptions = {
  /** The model (a key of the manifest's models). Defaults to `defaultModel`. */
  readonly model?: string;
  /** The execution configuration (a key of that model's quants). Defaults to `defaultQuant`. */
  readonly quant?: string;
  /**
   * The text encoder path (default `"gpu"` — ADR 0119 decision 7, step 10d design addendum B). The
   * components to fetch depend on the path, so it is decided at construction (not switched per
   * generation).
   *
   * - `"gpu"`: fetches umT5 (the manifest component `text_encoder` — in the distribution a
   *   cross-repository reference to umT5's distribution repository, about 5.3 GiB with i8 weights) and
   *   the tokenizer asset (`umt5_tokenizer`), and creates and closes the umT5 stage per `generate`.
   *   Accepts any prompt. With a local distribution (`localDirectory` / `denoDirectory`), the
   *   cross-repository target is passed as the source's `crossRepo` mapping (the neighbouring directory
   *   is not guessed — hub's `local.ts`).
   * - `"precomputed"`: a light use that does not fetch umT5 (reads not one byte of it) and accepts only
   *   the prompts of the text embedding asset. The asset's embeddings are bf16 umT5 outputs and do not
   *   match the values of `"gpu"` (i8 weights, f32 activations) — the same string gives a different
   *   video when the path changes.
   *
   * An unknown spelling is a plain `Error` (it fails before fetching).
   */
  readonly textEncoder?: "gpu" | "precomputed";
  /**
   * Shares an existing GPU (the passing side keeps ownership — the class's `dispose` does not destroy
   * it). When omitted the pipeline calls `acquireGpu` and destroys it in `dispose`.
   *
   * NOTE: a timing device (`acquireGpu({ gpuTiming: true })`) is accepted. The runs of the text,
   * encoder and DiT stages report their per-kernel breakdown (`lastRunTiming` through
   * {@link PipelineOptions.onRunDiagnostics}); the VAE decoder runs one tile = one batch, and a
   * batch is not timed (`GpuContext.beginBatch`), so its Sessions report none. Timing makes every
   * dispatch its own pass, so a timed generation is slower than an untimed one.
   */
  readonly gpu?: GpuContext;
  /**
   * Observer of Session diagnostics (the I2V VAE encoder: one condition image = one each of pre / attn
   * / post; umT5: one prompt = once; DiT: one forward = once; VAE: one tile = one each of first /
   * next). Exceptions are not swallowed (fail loudly).
   */
  readonly onRunDiagnostics?: (component: RunComponent, diagnostics: SessionDiagnostics) => void;
  /**
   * Cancels the construction. `fromPretrained` passes the same signal to the fetch layer, so it takes
   * effect both during the fetch (about 5.3 GiB for umT5 alone) and during assembly. At the
   * boundaries of the assembly steps (entry, after opening the containers, before parsing the assets,
   * before and after acquiring the GPU) it yields once to the event loop and then checks. The
   * cancellation exception throws `signal.reason` **as is** (unwrapped — the same form as the anima /
   * irodori construction).
   *
   * NOTE: the construction `signal` is not carried over to generation (generation is cancelled by
   * {@link GenerateRequest.signal}) — a construction-lifetime value (such as
   * `AbortSignal.timeout`) must not make every later generation fail.
   */
  readonly signal?: AbortSignal;
  /**
   * **Experimental**: fusion rules written outside the runtime (`defineRule` of
   * `@karume/runtime/kernel`), passed as `SessionOptions.fusionRules` to the Sessions of the video
   * graphs: the DiT (`transformer`) and, for I2V, the three VAE encoder graphs (the caller's array
   * unchanged), and the VAE decoder chunk graphs (the 16-latent pair, and with `vaeTile` 32 its
   * lenders and the 32-latent borrowers). umT5's Session does not get them. The VAE decoder chunk
   * graphs' Sessions always plan with the rule `models/vae_rms_silu` (`VAE_RMS_SILU_RULE` —
   * u32-identical to the unfused plan, on by default with no option to turn it off); the rules given
   * here follow it, in their order. Giving `models/vae_rms_silu` here again is refused as a rule
   * given twice by the decoder's Sessions. Absent = no rule on the DiT and the encoder, the family's
   * rule alone on the decoder.
   *
   * It exists to measure a candidate rule on a whole generation (with and without the rule); it may
   * change or go away in a minor version. Passing a rule is the opt-in, and the rule's author owns
   * the evidence for its numbers (ADR 0148): the runtime does not check them. The 32-latent
   * borrowers borrow every weight, and the runtime does not bind a borrowed weight to an external
   * rule's step, so a rule folds no window there. This family has no memory estimate, so there is
   * nothing to plan with the rules. A value the runtime refuses fails when the first such Session is
   * built.
   *
   * DECIDED: docs/decisions/0148-external-fusion-rules.md
   */
  readonly fusionRules?: PipelineFusionRules;
};

/**
 * The fetch-layer options that `Ti2vPipeline.fromPretrained` additionally accepts (passed through to
 * hub).
 *
 * NOTE: the per-field description, including that `headers` / `fetch` / `caches` / `onRetry` are
 * **only for HTTP sources**, lives in one place: {@link FromPretrainedHubOptions}.
 */
export type FromPretrainedOptions = PipelineOptions &
  FromPretrainedHubOptions &
  FromPretrainedComponentOptions;

/**
 * An already-fetched manifest + assets (a shape that accepts hub's `fetchAssets` result as is). The
 * components in `assets` are keyed by the single-form `krm` key (`transformer`) or as a part sequence
 * (`transformer[0]` / `transformer[1]` / … — from part 0 in index order): the three
 * `transformer` / `vae_decoder_first` / `vae_decoder_next`; the three of the I2V VAE encoder
 * `vae_encoder_pre` / `vae_encoder_attn` / `vae_encoder_post` (needed even for use without an image)
 * and, when the preset's weights carry them, the 32-latent decoder pair `vae_decoder_first_32` /
 * `vae_decoder_next_32` (ADR 0134 — weightless containers that borrow the 16-latent pair's weights);
 * and on the `"gpu"` path (default) `text_encoder` (umT5). In addition the manifest's `assets`
 * `text_embeds` (the embedding asset's safetensors — both paths) and, on the `"gpu"` path,
 * `umt5_tokenizer` (the tokenizer asset's JSON).
 */
export type Assets = {
  readonly manifest: Manifest;
  readonly assets: Readonly<Record<string, Uint8Array<ArrayBuffer>>>;
};
