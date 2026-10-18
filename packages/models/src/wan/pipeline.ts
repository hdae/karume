/**
 * `WanPipeline` — the single surface for text → video (Wan2.1 T2V 1.3B) (ADR 0118 decision 7).
 *
 * Pipeline (every stage runs in Karume, no torch):
 *
 * 0. **vae_encoder** (only for a request that passes a condition image (`image`) to Wan2.2's
 *    `WanTi2vPipeline` — I2V, ADR 0121 decision 11) — after preprocessing (choosing the output size →
 *    crop / stretch → `[-1, 1]` → patchify — `i2v-preprocess.ts`), runs the encoder's three
 *    symbolic-length graphs (pre / attn / post) in one batch and normalizes mu into the condition
 *    latent (`vae-encoder.ts`). At each step the DiT stage replaces the first latent frame of the model
 *    input with it (`dit-loop.ts`)
 * 1. **text** — the path is chosen at construction ({@link WanPipelineOptions.textEncoder} — ADR 0119
 *    decision 7, step 10d design addendum B):
 *    - `"gpu"` (default) — creates a Session for umT5 (i8 per-channel weights, f32 activations — ADR
 *      0119 decision 5), runs positive and negative once each, and closes it. The prompt is any string
 *      that passes the `prompt_clean` mirror and the tokenizer check (2 to 512 tokens — ADR 0119
 *      decisions 1, 2 and 4). The host pads the output `[1, L, 4096]` with zeros up to the row count
 *      of the DiT context
 *    - `"precomputed"` — only looks up the prompt's row in the text embedding asset (the first step of
 *      ADR 0118 decision 4; no GPU use). A string outside the asset's set fails loudly
 * 2. **transformer** — the S-shaped DiT (`[1,S,C·4]` — Wan2.1 is `[1,S,64]`, Wan2.2 is `[1,S,192]`;
 *    decision 3), `steps` times. CFG runs uncond → cond as two sequential passes (B = 1); the
 *    combination `uncond + g·(cond − uncond)` and the UniPC update run on the host (decision 5 —
 *    `scheduler.ts`)
 * 3. **vae_decoder** — **always-tiled** decode with the two chunk graphs (first / next) (decision 2 —
 *    `vae-tiles.ts`) → clamp to `[-1, 1]`
 *
 * Returns the frames `[3, F, H, W]` as f32 (range `[-1, 1]`) and the fps ({@link GeneratedVideo}).
 * Writing PNGs is the caller's job (`wanFrameToRgba` + `encodePng`).
 *
 * The stage bodies live in modules shared by the Wan2.1 / 2.2 classes (ADR 0121 decision 10 — the entry
 * check is `plan.ts`, the text stage `text-stage.ts`, the DiT stage `dit-loop.ts`, the VAE stage
 * `tile-decode.ts`, and 2.2's encoder stage `vae-encoder.ts`). The family
 * admission, construction and stage order also live in the shared `family.ts` (one body that takes the
 * generation's values as a `WanFamilySpec`); this file holds the public types and the shell of the
 * Wan2.1 class (serialization chain, `dispose`) and passes `WAN21_FAMILY`. The public types are shared
 * with the Wan2.2 class (`WanTi2vPipeline` in `./ti2v-pipeline.ts`) — values that differ by generation,
 * such as the accepted set, defaults, latent shape and fps, are decided by which class built the
 * pipeline (that class's generation descriptor).
 *
 * ## MUST: a Session per stage, created and closed one at a time — (encoder →) text → DiT → VAE
 *
 * Construction ({@link WanPipeline.fromPretrained} / {@link WanPipeline.fromAssets}) creates no
 * Session at all (it goes as far as opening the containers).
 * `generate` creates a Session per stage and closes it before creating the next (decision 7 — the same
 * as anima's default `"per-stage"`). The text stage (umT5 i8 weights, 5.30 GiB) plus the DiT stage
 * (7.31 GiB at 81 frames) exceeds the B570's 9,600 MiB ceiling (ADR 0119 decision 11), so the text
 * stage is closed before the DiT stage is created. The DiT stage (about 5.7 GiB at 33 frames) and the
 * VAE stage (about 2.9 GiB) are not held at the same time either. The allocation left right after
 * closing the text stage was +0.08 GiB on the B570 with Deno at 33 frames (text-stage peak 6.30 GiB →
 * 0.708 GiB after closing, 0.633 GiB before the stage — ADR 0119 addendum "step 10c GPU check and step
 * 10d-4"). Not measured in Chrome.
 *
 * The I2V encoder stage (Wan2.2 only) runs first; its three Sessions and the hand-over residents are
 * closed and then the pipeline **waits for the release** (`session/settle-released-memory.ts`) before
 * moving to the next stage (ADR 0121 decision 11 — the DiT stage has only 0.4–0.8 GiB of capacity
 * margin, so the encoder's allocation (about 0.9 GiB of planned backing at 1280×704) must not remain
 * into the DiT stage; when the encoder stage fails it is also closed and then waited on, so nothing
 * remains into the DiT stage of the same pipeline's next generation either). The encoder components
 * are fetched at construction even for T2V-only use (the 2.2 distribution always carries them — no
 * extra construction entry point), but their Sessions are created only for a request that passes an
 * image.
 *
 * NOTE (the text → DiT and DiT → VAE switches have no release wait — the result of decision 7's
 * "measure the peak at the switch and add a wait if it falls short" (that conditional wait was
 * withdrawn on 2026-10-18 — ADR 0118 decision 7). The wait after the I2V encoder stage is specified by
 * ADR 0121 decision 11 and was not measured on the B570):
 * on the B570 (Intel / wgpu) the release of `destroy()` can be delayed until the next device poll
 * (docs/known-issues.md, section "Intel Arc B570 (Linux / Vulkan ANV / xe)"), but in the measurement of
 * 2026-10-02 (832×480, 33 frames, a two-step run end to end, fdinfo `drm-total-vram0`) the DiT stage's
 * 5.75 GiB peak had already dropped to 0.43 GiB (0.19 GiB before the first stage) **right after the DiT
 * Session was closed** (the `stage` `end`, before the VAE stage is created). The switch's peak is the
 * DiT stage's peak itself (it does not overlap the VAE stage's 2.88 GiB peak). No delay shows because
 * `Session.dispose` settles unfinished work before destroying, so no buffer the GPU still references
 * remains at the time of destruction (speculation — the measured delay cases discarded buffers that
 * were still referenced).
 *
 * MUST: the public API also enforces the stage switching — `generate` runs on the serialization chain
 * (concurrent calls wait and run in turn), and `dispose` waits for it to complete before destroying the
 * GPU.
 *
 * ## Cancellation (`signal` — ADR 0119 decision 9, step 10d design addendum G)
 *
 * Construction takes {@link WanPipelineOptions.signal} (at the boundaries between the fetch layer and
 * the construction steps — the same form as anima / irodori); generation takes
 * {@link WanGenerateRequest.signal} (at stage boundaries, before each run (the two text runs, each DiT
 * step) and between VAE tiles — the form of `generation/sequence.ts`). Each check first yields once to
 * the event loop (`settleAbort`). One run is indivisible; the cancellation takes effect at the next
 * boundary. The cancellation exception closes the open Sessions and then throws `signal.reason`
 * unwrapped (the consumer can identify it by `error === controller.signal.reason` — ADR 0083 decision
 * 5).
 *
 * ## Distribution (`karume-wan2.1` — ADR 0118 step 7)
 *
 * There are two entry points: {@link WanPipeline.fromPretrained}, which fetches from the distribution,
 * and {@link WanPipeline.fromAssets}, which builds from already-fetched bytes (manifest + assets); both
 * go through the same family admission and assembly (`family.ts`). The generation defaults (step
 * count, guidance, shift) come from the manifest's `pipelineConfig` (`config.ts`); the UniPC structure
 * stays at the upstream values (`WAN_UNIPC_CONFIG`).
 *
 * NOTE: the `karume-wan2.1` distribution is not published on HF (ADR 0118 addendum (2026-10-15)), so
 * the table of public distribution repositories (`WAN_SOURCES` — `config.ts`) has no Wan2.1 key (the
 * table holds only Wan2.2's `"wan2.2"`).
 *
 * ## MUST: the "correctness" of the output is not guaranteed here
 *
 * Numeric correctness is guaranteed by the e2e against the reference (real GPU —
 * `packages/e2e/tests/e2e_wan_pipeline_test.ts`). The seeded random generator is a different stream
 * from torch's `randn` (`random.ts`), so comparison with the reference injects the initial noise via
 * {@link WanGenerateRequest.latents}.
 */

import type { GpuContext, SessionDiagnostics } from "@karume/runtime";
import type { DistributionSource, HubRepoRef, Manifest } from "@karume/hub";

import { createOperationChain } from "../concurrency/serial.ts";
import type {
  FromPretrainedComponentOptions,
  FromPretrainedHubOptions,
} from "../hub/load-options.ts";
import {
  generateWanVideo,
  loadWanFromAssets,
  loadWanFromPretrained,
  WAN21_FAMILY,
  type WanState,
} from "./family.ts";
import type { WanPrompt } from "./text-embeds.ts";

/** The generation result. `data` is `[3, frames, height, width]` f32 (range `[-1, 1]` — clamped). */
export type GeneratedVideo = {
  readonly frames: number;
  readonly width: number;
  readonly height: number;
  /**
   * The output frame rate (an upstream fact of the generation — 16 for Wan2.1, 24 for Wan2.2. Neither
   * declared in the manifest nor a knob — ADR 0121 decision 8).
   */
  readonly fps: number;
  readonly data: Float32Array<ArrayBuffer>;
};

/**
 * A copy of the intermediate latent returned by `copyLatents()` of `denoise-step` (the shape is decided
 * by the class's generation — Wan2.1: `[16, F', H/8, W/8]` / Wan2.2: `[48, F', H/16, W/16]`).
 */
export type WanLatentSnapshot = {
  readonly data: Float32Array<ArrayBuffer>;
  readonly shape: readonly number[];
};

/**
 * The component names that {@link WanPipelineOptions.onRunDiagnostics} receives (one Session = one
 * name). `text_encoder` is the umT5 of the `"gpu"` path (once each for positive / negative).
 * `vae_encoder_*` are the three graphs of Wan2.2's I2V encoder (once each for a request that passes a
 * condition image).
 */
export type WanRunComponent =
  | "vae_encoder_pre"
  | "vae_encoder_attn"
  | "vae_encoder_post"
  | "text_encoder"
  | "transformer"
  | "vae_decoder_first"
  | "vae_decoder_next";

/** The generation events that {@link WanGenerateRequest.onEvent} receives. */
export type WanGenerateEvent =
  /**
   * The start of a stage (`start` — before its Session is created) and its end (`end` — after its
   * Session is closed). `vae_encoder` is the stage that encodes Wan2.2's I2V condition image (only for a
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
       * update). For I2V (a request that passes `image` to Wan2.2) this is the scheduler state itself,
       * and its first latent frame has not been replaced by the condition latent — the replacement
       * happens once at the end of the DiT stage (the same value as the `latents` that upstream
       * diffusers' `callback_on_step_end` receives).
       */
      readonly copyLatents: () => WanLatentSnapshot;
    }
  /** Completion of the decode of one VAE tile (`tile` is 1-based). */
  | { readonly kind: "vae-tile"; readonly tile: number; readonly tiles: number };

/**
 * One generation request (the type shared by the Wan2.1 / 2.2 classes). An omitted step count,
 * guidance or shift takes the default of the manifest's `pipelineConfig` (the Wan2.1 distribution's
 * values are the reference setting — 50 steps, guide 5.0, shift 3.0. Wan2.2's default shift is the
 * upstream scheduler's `flow_shift` 5.0 (the official 720p value), declared by the manifest of the
 * `karume-wan2.2` distribution (ADR 0121 stage 8)). The accepted set and defaults of size and
 * frame count are decided by the class's generation (Wan2.1's `WanPipeline`: 832×480 / 480×832,
 * 5–81 frames, default 832×480, 33 frames / Wan2.2's `WanTi2vPipeline`: 1280×704 / 704×1280, 5–121
 * frames, default 1280×704, 33 frames).
 */
export type WanGenerateRequest = {
  /**
   * The prompt. The accepted set depends on the path chosen at construction
   * ({@link WanPipelineOptions.textEncoder}):
   *
   * - `"gpu"` (default): any string. It passes through the mirror of the upstream preprocessing
   *   (`prompt_clean`) and the tokenizer check; input that does not pass is a `ModelInputError` (ADR
   *   0119 decisions 1, 2 and 4) — out-of-vocabulary characters, characters unassigned in Unicode
   *   16.0.0, C1 control characters, sequences that could be HTML character references (`R&D` passes
   *   when spaced as `R & D`), sequences that look like mojibake, special tokens in the text, `▁` right
   *   after whitespace, empty or whitespace only (1 token), more than 512 tokens. The rejection
   *   message says how to fix the input.
   * - `"precomputed"`: **only strings in the text embedding asset's set** (an exact match of either the
   *   original or the normalized string — the class's `prompts` ({@link WanPipeline.prompts} /
   *   `WanTi2vPipeline.prompts`)). Anything outside the set is a `ModelInputError` (ADR 0118 decision
   *   4).
   */
  readonly prompt: string;
  /**
   * The prompt for CFG's uncond side (the same accepted set as {@link WanGenerateRequest.prompt}).
   * When omitted, the official `sample_neg_prompt` is used (`"gpu"`: the default string encoded on the
   * GPU; `"precomputed"`: the asset's `negative` row — the same original text). Passing it when
   * `guidance` is 1 (CFG does not run) is a `ModelInputError` (a knob that has no effect is not
   * silently accepted).
   */
  readonly negativePrompt?: string;
  /** The seed of the initial noise (default 0 — exclusive with {@link WanGenerateRequest.latents}). */
  readonly seed?: number;
  /**
   * Passes the initial noise in from outside (f32, `F' = (frames − 1)/4 + 1`. The shape is decided by
   * the class's generation — Wan2.1: `[16, F', H/8, W/8]` / Wan2.2: `[48, F', H/16, W/16]`). The entry
   * point for injecting torch's `randn` stream when comparing with the reference (the seeded generator
   * is a different stream from torch — `random.ts`). Not modified.
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
  /**
   * The frame count (4n+1. The range and default are decided by the class's generation — Wan2.1:
   * 5–81, default 33 / Wan2.2: 5–121, default 33).
   */
  readonly frames?: number;
  /**
   * Width × height (the accepted set and default are decided by the class's generation — Wan2.1:
   * 832×480 or 480×832, default 832×480 / Wan2.2: 1280×704 or 704×1280, default 1280×704).
   */
  readonly width?: number;
  readonly height?: number;
  /**
   * Observer of generation events ({@link WanGenerateEvent}). **It is awaited** (the firing order is
   * deterministic). Exceptions are not swallowed — a throw cancels at step granularity (the stage's
   * Session is closed).
   *
   * MUST: do not await `generate` / `dispose` of the same pipeline inside `onEvent` (a self-deadlock on
   * the serialization chain).
   */
  readonly onEvent?: (event: WanGenerateEvent) => void | Promise<void>;
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

/**
 * Construction options (shared by the classes' `fromAssets` / `fromPretrained` —
 * {@link WanPipeline.fromAssets} / {@link WanPipeline.fromPretrained} and the two of the same name on
 * `WanTi2vPipeline`).
 */
export type WanPipelineOptions = {
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
   * MUST: a timing device (`acquireGpu({ gpuTiming: true })`) cannot be passed — the VAE stage runs one
   * tile = one batch, and the runtime does not open a batch on a timing device (the MUST of
   * `GpuContext.beginBatch`). It fails at construction (not as a failure in the VAE stage after the
   * DiT stage has finished). The DiT's GPU time is measured by the e2e that runs only the DiT (step 3).
   */
  readonly gpu?: GpuContext;
  /**
   * Observer of Session diagnostics (the I2V VAE encoder: one condition image = one each of pre / attn
   * / post; umT5: one prompt = once; DiT: one forward = once; VAE: one tile = one each of first /
   * next). Exceptions are not swallowed (fail loudly).
   */
  readonly onRunDiagnostics?: (component: WanRunComponent, diagnostics: SessionDiagnostics) => void;
  /**
   * Cancels the construction. The class's `fromPretrained` passes the same signal to the fetch layer,
   * so it takes effect both during the fetch (about 5.3 GiB for umT5 alone) and during assembly. At
   * the boundaries of the assembly steps (entry, after opening the containers, before parsing the
   * assets, before and after acquiring the GPU) it yields once to the event loop and then checks. The
   * cancellation exception throws `signal.reason` **as is** (unwrapped — the same form as the anima /
   * irodori construction).
   *
   * NOTE: the construction `signal` is not carried over to generation (generation is cancelled by
   * {@link WanGenerateRequest.signal}) — a construction-lifetime value (such as
   * `AbortSignal.timeout`) must not make every later generation fail.
   */
  readonly signal?: AbortSignal;
};

/**
 * The fetch-layer options that the classes' `fromPretrained` ({@link WanPipeline.fromPretrained} /
 * `WanTi2vPipeline.fromPretrained`) additionally accept (passed through to hub).
 *
 * NOTE: the per-field description, including that `headers` / `fetch` / `caches` / `onRetry` are
 * **only for HTTP sources**, lives in one place: {@link FromPretrainedHubOptions}.
 */
export type WanFromPretrainedOptions = WanPipelineOptions &
  FromPretrainedHubOptions &
  FromPretrainedComponentOptions;

/**
 * An already-fetched manifest + assets (a shape that accepts hub's `fetchAssets` result as is). The
 * components in `assets` are keyed by the single-form `krm` key (`transformer`) or as a part sequence
 * (`transformer[0]` / `transformer[1]` / … — from part 0 in index order): the three
 * `transformer` / `vae_decoder_first` / `vae_decoder_next`; for Wan2.2 (`WanTi2vPipeline`) the three
 * of the I2V VAE encoder `vae_encoder_pre` / `vae_encoder_attn` / `vae_encoder_post` (needed even for
 * use without an image); and on the `"gpu"` path (default) `text_encoder` (umT5). In addition the
 * manifest's `assets` `text_embeds` (the embedding asset's safetensors — both paths) and, on the
 * `"gpu"` path, `umt5_tokenizer` (the tokenizer asset's JSON).
 */
export type WanAssets = {
  readonly manifest: Manifest;
  readonly assets: Readonly<Record<string, Uint8Array<ArrayBuffer>>>;
};

/**
 * The Wan2.1 text → video pipeline.
 *
 * The only construction entry points are {@link WanPipeline.fromPretrained} (fetch from the
 * distribution) and {@link WanPipeline.fromAssets} (already-fetched bytes) (the constructor is private
 * — no half-built state that bypasses the manifest checks and the asset matching. ADR 0008).
 */
export class WanPipeline {
  readonly #state: WanState;
  /** The serialization chain of generate and dispose. */
  readonly #chain = createOperationChain();
  /** The one dispose (not undefined means "disposed"). */
  #disposal: Promise<void> | undefined;

  private constructor(state: WanState) {
    this.#state = state;
  }

  /**
   * Fetches from the distribution and builds (`loadManifest` → `resolveSelection` → fetch **only each
   * component's descriptor (part 0)** for admission → warm the weight parts → `fetchAssets` for the
   * embedding asset and the tokenizer asset → construction). Blocks are read in part order at the
   * moment their Session is built (ADR 0109 — `src/hub/components.ts`). The components to fetch depend
   * on the path ({@link WanPipelineOptions.textEncoder}) — `"precomputed"` does not fetch umT5
   * (`text_encoder`). To replace a component with the same role from another repository, use
   * {@link FromPretrainedComponentOptions.components} (`transformer` / `vae_decoder_first` /
   * `vae_decoder_next`, plus `text_encoder` on the `"gpu"` path).
   *
   * **`ref` is required** (sources have no default — the MUST of `src/hub/repo-ref.ts`. Wan2.1 has no
   * public distribution repository, so `WAN_SOURCES` has no key for it either). A string `ref` reads as
   * `{ repo }` (= follows `main`). A local distribution (`models/karume-wan2.1`) is passed as a
   * **source handle** (`localDirectory` / `denoDirectory` of `@karume/hub/deno`) — it does not pass
   * the HF `owner/name` spelling check, and goes through neither the network nor CacheStorage (the
   * HTTP-only knobs of {@link WanFromPretrainedOptions} have no effect). The umT5 cross-repository
   * target (umT5's distribution repository) is passed as that source's `crossRepo` mapping (for example
   * `denoDirectory("models/karume-wan2.1", { crossRepo:
   * { "<owner>/karume-umt5-xxl": denoDirectory("models/karume-umt5-xxl") } })` — the key is the repo
   * that the manifest's `text_encoder` declares).
   */
  static async fromPretrained(
    ref: string | HubRepoRef | DistributionSource,
    options: WanFromPretrainedOptions = {},
  ): Promise<WanPipeline> {
    return new WanPipeline(await loadWanFromPretrained(WAN21_FAMILY, ref, options));
  }

  /**
   * Builds from an already-fetched manifest + assets ({@link WanAssets}). Opens the containers and runs
   * the same family admission and assembly as the fetch surface (`family.ts`) — the two surfaces differ
   * only in how components are supplied.
   */
  static async fromAssets(
    input: WanAssets,
    options: WanPipelineOptions = {},
  ): Promise<WanPipeline> {
    return new WanPipeline(await loadWanFromAssets(WAN21_FAMILY, input, options));
  }

  /**
   * The list of prompts in the text embedding asset (the asset's metadata in order — a copy without
   * the embedding values). Its meaning depends on the path:
   *
   * - `"precomputed"`: **the accepted set itself**. Pass each row's `prompt` or `normalized` as
   *   `prompt` / `negativePrompt`.
   * - `"gpu"` (default): examples (the fixed prompts' names and original texts). The accepted set is
   *   any string; passing a row's `prompt` encodes it with umT5 on the GPU (the asset's embeddings are
   *   not used).
   */
  get prompts(): readonly WanPrompt[] {
    return this.#state.textEmbeds.entries.map(({ name, role, prompt, normalized }) => ({
      name,
      role,
      prompt,
      normalized,
    }));
  }

  /**
   * Generates one video from a prompt. The same request (seed or latents, knobs) gives the same values.
   * Concurrent calls wait and run in turn (no graphs resident at the same time — module doc).
   */
  async generate(request: WanGenerateRequest): Promise<GeneratedVideo> {
    if (this.#disposal !== undefined) throw new Error("WanPipeline: dispose 済みでは生成できない");
    return await this.#chain(() => generateWanVideo(WAN21_FAMILY, this.#state, request));
  }

  /**
   * Releases resources. Destroys **only a GPU acquired internally** (`options.gpu` belongs to the
   * caller).
   * MUST: wait for the in-flight generation to complete before destroying (flush-before-destroy — on
   * the chain).
   */
  dispose(): Promise<void> {
    this.#disposal ??= this.#chain(() => {
      if (this.#state.ownsGpu) this.#state.gpu.destroy();
    });
    return this.#disposal;
  }

  /** `await using` support — an alias of {@link dispose}. */
  [Symbol.asyncDispose](): Promise<void> {
    return this.dispose();
  }
}
