/**
 * `Ti2vPipeline` — the one surface for text (+ condition image) → video (Wan2.2 TI2V 5B's T2V and
 * I2V) (ADR 0118 decision 7, ADR 0121 decisions 10 and 11).
 *
 * Pipeline (every stage runs in Karume, no torch):
 *
 * 0. **vae_encoder** (only for a request that passes a condition image (`image`) — I2V, ADR 0121
 *    decision 11) — after preprocessing (choosing the output size → crop / stretch → `[-1, 1]` →
 *    patchify — `i2v-preprocess.ts`), runs the encoder's three symbolic-length graphs (pre / attn /
 *    post) in one batch and normalizes mu into the condition latent (`vae-encoder.ts`). At each step
 *    the DiT stage replaces the first latent frame of the model input with it (`dit-loop.ts`)
 * 1. **text** — the path is chosen at construction ({@link PipelineOptions.textEncoder} — ADR 0119
 *    decision 7, step 10d design addendum B):
 *    - `"gpu"` (default) — creates a Session for umT5 (i8 per-channel weights, f32 activations — ADR
 *      0119 decision 5), runs positive and negative once each, and closes it. The prompt is any string
 *      that passes the `prompt_clean` mirror and the tokenizer check (2 to 512 tokens — ADR 0119
 *      decisions 1, 2 and 4). The host pads the output `[1, L, 4096]` with zeros up to the row count
 *      of the DiT context
 *    - `"precomputed"` — only looks up the prompt's row in the text embedding asset (the first step of
 *      ADR 0118 decision 4; no GPU use). A string outside the asset's set fails loudly
 * 2. **transformer** — the S-shaped DiT (`[1,S,192]` for the 48-channel latent — ADR 0118 decision 3),
 *    `steps` times. CFG runs uncond → cond as two sequential passes (B = 1); the combination
 *    `uncond + g·(cond − uncond)` and the UniPC update run on the host (ADR 0118 decision 5 —
 *    `scheduler.ts`)
 * 3. **vae_decoder** — **always-tiled** decode with the two chunk graphs (first / next) (ADR 0118
 *    decision 2 — `vae-tiles.ts`) → unpatchify → clamp to `[-1, 1]`
 *
 * Returns the frames `[3, F, H, W]` as f32 (range `[-1, 1]`) and the fps ({@link GeneratedVideo}).
 * Writing PNGs is the caller's job (`frameToRgba` + `encodePng`).
 *
 * This file holds the class shell (private constructor, serialization chain, `dispose`) and the
 * request type with the condition image ({@link Ti2vGenerateRequest}), and passes
 * `WAN22_TI2V_FAMILY` (pipeline `wan-ti2v/1`, generation descriptor `WAN22_TI2V_GENERATION`). The
 * family admission, construction and stage order live in `family.ts` (one body that takes the
 * generation's values as a `WanFamilySpec`), and the stage bodies in the family's modules (the
 * entry check `plan.ts`, the text stage `text-stage.ts`, the DiT stage `dit-loop.ts`, the VAE stage
 * `tile-decode.ts`, the encoder stage `vae-encoder.ts`). The public types are in `./types.ts`.
 *
 * The generation's values come from the generation descriptor and the asset declarations (not from
 * code branches):
 *
 * - The accepted set is 1280×704 / 704×1280 × 4n+1 frames, 5–121 (default 1280×704, 33 frames)
 * - The latents have 48 channels (`[48, F', H/16, W/16]` — the VAE graph's ratio 8 × unpatchify 2);
 *   the output is 24 fps
 * - The DiT has 7 inputs (condition-side time and condition mask — in T2V the condition mask is all
 *   false and the condition-side time equals the generation side; in I2V the tokens of the first
 *   latent frame are true and the condition-side time is 0)
 * - It has the I2V VAE encoder components (three symbolic-length graphs — `vae-encoder.ts`)
 *
 * ## MUST: a Session per stage, created and closed one at a time — (encoder →) text → DiT → VAE
 *
 * Construction ({@link Ti2vPipeline.fromPretrained} / {@link Ti2vPipeline.fromAssets}) creates no
 * Session at all (it goes as far as opening the containers). `generate` creates a Session per stage
 * and closes it before creating the next (ADR 0118 decision 7 — the same as anima's default
 * `"per-stage"`). The rule was set on measurements of Wan2.1 T2V 1.3B, the generation the design was
 * first built on: its text stage (umT5 i8 weights, 5.30 GiB) plus its DiT stage (7.31 GiB at 81
 * frames) exceeds the B570's 9,600 MiB ceiling (ADR 0119 decision 11), so the text stage is closed
 * before the DiT stage is created; its DiT stage (about 5.7 GiB at 33 frames) and VAE stage (about
 * 2.9 GiB) are not held at the same time either. The allocation left right after closing the text
 * stage was +0.08 GiB on the B570 with Deno at 33 frames (text-stage peak 6.30 GiB → 0.708 GiB after
 * closing, 0.633 GiB before the stage — ADR 0119 addendum "step 10c GPU check and step 10d-4"). Not
 * measured in Chrome.
 *
 * The I2V encoder stage runs first; its three Sessions and the hand-over residents are closed and
 * then the pipeline **waits for the release** (`session/settle-released-memory.ts`) before moving to
 * the next stage (ADR 0121 decision 11 — the DiT stage has only 0.4–0.8 GiB of capacity margin, so
 * the encoder's allocation (about 0.9 GiB of planned backing at 1280×704) must not remain into the
 * DiT stage; when the encoder stage fails it is also closed and then waited on, so nothing remains
 * into the DiT stage of the same pipeline's next generation either). The encoder components are
 * fetched at construction even for T2V-only use (the distribution always carries them — no extra
 * construction entry point), but their Sessions are created only for a request that passes an image.
 *
 * NOTE (the text → DiT and DiT → VAE switches have no release wait — the result of ADR 0118 decision
 * 7's "measure the peak at the switch and add a wait if it falls short" (that conditional wait was
 * withdrawn on 2026-10-18). The wait after the I2V encoder stage is specified by ADR 0121 decision 11
 * and was not measured on the B570):
 * on the B570 (Intel / wgpu) the release of `destroy()` can be delayed until the next device poll
 * (a plain-WebGPU probe: a 1 GiB allocation right after destroying nine 1 GiB buffers is an OOM,
 * and succeeds after `onSubmittedWorkDone` and 200 ms), but in the measurement of
 * 2026-10-02 (Wan2.1 T2V 1.3B, 832×480, 33 frames, a two-step run end to end, fdinfo
 * `drm-total-vram0`) the DiT stage's 5.75 GiB peak had already dropped to 0.43 GiB (0.19 GiB before
 * the first stage) **right after the DiT Session was closed** (the `stage` `end`, before the VAE stage
 * is created). The switch's peak is the DiT stage's peak itself (it does not overlap the VAE stage's
 * 2.88 GiB peak). No delay shows because `Session.dispose` settles unfinished work before destroying,
 * so no buffer the GPU still references remains at the time of destruction (speculation — the
 * measured delay cases discarded buffers that were still referenced).
 *
 * MUST: the public API also enforces the stage switching — `generate` runs on the serialization chain
 * (concurrent calls wait and run in turn), and `dispose` waits for it to complete before destroying the
 * GPU.
 *
 * ## Cancellation (`signal` — ADR 0119 decision 9, step 10d design addendum G)
 *
 * Construction takes {@link PipelineOptions.signal} (at the boundaries between the fetch layer and
 * the construction steps — the same form as anima / irodori); generation takes
 * {@link GenerateRequest.signal} (at stage boundaries, before each run (the two text runs, each DiT
 * step) and between VAE tiles — the form of `generation/sequence.ts`). Each check first yields once to
 * the event loop (`settleAbort`). One run is indivisible; the cancellation takes effect at the next
 * boundary. The cancellation exception closes the open Sessions and then throws `signal.reason`
 * unwrapped (the consumer can identify it by `error === controller.signal.reason` — ADR 0083 decision
 * 5).
 *
 * ## Distribution (`karume-wan2.2` — ADR 0121 stage 8)
 *
 * Built by the recipe `dist.py --pipeline wan-ti2v`; the public HF repository is `SOURCES["wan2.2"]`.
 * There are two entry points: {@link Ti2vPipeline.fromPretrained}, which fetches from the
 * distribution, and {@link Ti2vPipeline.fromAssets}, which builds from already-fetched bytes
 * (manifest + assets); both go through the same family admission and assembly (`family.ts`). The
 * generation defaults (step count, guidance, shift) come from the manifest's `pipelineConfig`
 * (`config.ts`); the UniPC structure stays at the upstream values (`WAN_UNIPC_CONFIG`).
 *
 * ## MUST: the "correctness" of the output is not guaranteed here
 *
 * Numeric correctness is guaranteed by the e2e against the reference (real GPU —
 * `packages/e2e/tests/e2e_wan_ti2v_pipeline_test.ts`). The seeded random generator is a different
 * stream from torch's `randn` (`random.ts`), so comparison with the reference injects the initial
 * noise via {@link GenerateRequest.latents}.
 *
 * MUST: every module is side-effect free (no import-time execution, no global mutable state —
 * CLAUDE.md).
 */

import type { DistributionSource, HubRepoRef } from "@karume/hub";

import { createOperationChain } from "../concurrency/serial.ts";
import {
  generateWanVideo,
  loadWanFromAssets,
  loadWanFromPretrained,
  ownWanRequestLatents,
  WAN22_TI2V_FAMILY,
  type WanState,
} from "./family.ts";
import type { Rgb8Image } from "../image/preprocess.ts";
import type {
  GeneratedVideo,
  Assets,
  FromPretrainedOptions,
  GenerateRequest,
  PipelineOptions,
} from "./types.ts";
import type { I2vFit } from "./i2v-preprocess.ts";
import type { Prompt } from "./text-embeds.ts";

/**
 * One Wan2.2 generation request ({@link GenerateRequest} + the I2V condition image). Without
 * `image` it is text → video (T2V — the same path and the same values as a request without the image
 * field).
 *
 * With `image` it is I2V (ADR 0121 decision 11 — the conditioning has the form of `expand_timesteps`
 * in diffusers' `WanImageToVideoPipeline`):
 *
 * - Output size: when `width` / `height` are omitted, the accepted size (1280×704 / 704×1280) closer
 *   to the image's aspect ratio is chosen (official Wan2.2's comparison `max(r / rc, rc / r)`, ties
 *   go landscape — an image with width ≥ height gets 1280×704). When one or both are given, the
 *   accepted size that matches them (`ModelInputError` when none does). The official code also picks
 *   sizes outside the accepted set (such as 1248×704 for a 16:9 image), so the same image can be
 *   cropped more than in the official code.
 * - Preprocessing stays in RGB8 (decoding is the caller's job — the pixels are used as they are,
 *   without colour-space conversion) → size fitting (`fit`) → `[-1, 1]` → VAE encoder (the
 *   `vae_encoder` stage — the first stage) → latent normalization → the condition of the DiT's first
 *   latent frame.
 * - The first output frame is the condition image encoded and decoded by the VAE (the first latent
 *   frame is the condition latent itself).
 */
export type Ti2vGenerateRequest = GenerateRequest & {
  /**
   * The I2V condition image (RGB8, interleaved, `data.length = 3·width·height`). Any size (it is
   * fitted to the output size). The pipeline does not modify it. The caller also must not modify
   * `data` until `generate` settles (the pixels are read when the request's turn comes on the
   * serialization chain, so a modification while waiting affects the generation). A value not in
   * RGB8 form (`data` not a `Uint8Array`, `null`, etc.) and a mismatch between size and length are a
   * `ModelInputError`.
   */
  readonly image?: Rgb8Image;
  /**
   * How the condition image is fitted to the size (only with `image` — `"crop"` when omitted).
   *
   * - `"crop"`: the same as official Wan2.2's I2V — LANCZOS-resize, keeping the aspect ratio, to a
   *   size that covers the output, then cut out the centre
   * - `"stretch"`: the same as diffusers' `WanImageToVideoPipeline` — LANCZOS-resize directly to the
   *   output size (the aspect ratio is not kept)
   *
   * Passing it without `image` is a `ModelInputError` (a knob with no effect is not accepted
   * silently). A misspelling is a plain `Error` (treated like a misspelt model / quant name — the
   * remedy is to look up the accepted set again, not to fix a value range; ADR 0107 decision 3), and
   * it is checked before whether `image` is present (a misspelling without `image` is also a plain
   * `Error`).
   */
  readonly fit?: I2vFit;
};

/**
 * The text (+ condition image) → video pipeline of Wan2.2 TI2V 5B (T2V and I2V —
 * {@link Ti2vGenerateRequest}).
 *
 * Construction has only two entries: {@link Ti2vPipeline.fromPretrained} (fetch from the
 * distribution) and {@link Ti2vPipeline.fromAssets} (already-fetched bytes) (the constructor is
 * private — no half-built state that bypasses the manifest check and the asset matching. ADR 0008).
 */
export class Ti2vPipeline {
  readonly #state: WanState;
  /** The serialization chain of generate and dispose. */
  readonly #chain = createOperationChain();
  /** The single dispose (not undefined means "disposed"). */
  #disposal: Promise<void> | undefined;

  private constructor(state: WanState) {
    this.#state = state;
  }

  /**
   * Fetches from the distribution and builds (`loadManifest` → `resolveSelection` → fetch **only each
   * component's descriptor (part 0)** for admission → warm the weight parts → `fetchAssets` for the
   * embedding asset and the tokenizer asset → construction). Blocks are read in part order at the
   * moment their Session is built (ADR 0109 — `src/hub/components.ts`). The components to fetch depend
   * on the path ({@link PipelineOptions.textEncoder}) — `"precomputed"` does not fetch umT5
   * (`text_encoder`); the three I2V VAE encoder parts (`vae_encoder_pre` / `vae_encoder_attn` /
   * `vae_encoder_post`) are fetched even for use without an image. To replace a component with the
   * same role from another repository, use `components` (`FromPretrainedComponentOptions`)
   * (`transformer` / `vae_decoder_first` / `vae_decoder_next` / the three encoder parts, plus
   * `text_encoder` on the `"gpu"` path and `vae_decoder_first_32` / `vae_decoder_next_32` when the
   * preset carries the 32-latent decoder pair). The manifest's pipeline must be `wan-ti2v/1`.
   *
   * **`ref` is required** (sources have no default — the MUST of `src/hub/repo-ref.ts`). The source
   * this package version verified is `"wan2.2"` in `SOURCES` (`config.ts`) — to pin reproducibility
   * yourself, the caller gives `{ repo, revision }` explicitly; a string `ref` reads as `{ repo }`
   * (= follows `main`). When fetching from HF, umT5's cross-repository target is fetched at the
   * commit the manifest pins, so the caller does not pass it. A local mirror (`models/karume-wan2.2`)
   * is passed as a **source handle** (`localDirectory` / `denoDirectory` of `@karume/hub/deno`) — it
   * does not pass the HF `owner/name` spelling check, and goes through neither the network nor
   * CacheStorage (the HTTP-only knobs of {@link FromPretrainedOptions} have no effect). On the
   * `"gpu"` path umT5's cross-repository target (`karume-umt5-xxl`) is passed as a mapping in that
   * source's `crossRepo` (for example `denoDirectory("models/karume-wan2.2", { crossRepo:
   * { "<owner>/karume-umt5-xxl": denoDirectory("models/karume-umt5-xxl") } })` — the key is the repo
   * that the manifest's `text_encoder` declares).
   */
  static async fromPretrained(
    ref: string | HubRepoRef | DistributionSource,
    options: FromPretrainedOptions = {},
  ): Promise<Ti2vPipeline> {
    return new Ti2vPipeline(await loadWanFromPretrained(WAN22_TI2V_FAMILY, ref, options));
  }

  /**
   * Builds from an already-fetched manifest + assets ({@link Assets}). Opens the containers and
   * goes through the same family admission and assembly (`family.ts`) as the fetching surface — the
   * two surfaces differ only in where the components come from.
   */
  static async fromAssets(input: Assets, options: PipelineOptions = {}): Promise<Ti2vPipeline> {
    return new Ti2vPipeline(await loadWanFromAssets(WAN22_TI2V_FAMILY, input, options));
  }

  /**
   * The list of prompts of the text embedding asset (in the asset metadata's order — a copy without
   * the embedding values). Its meaning depends on the path:
   *
   * - `"precomputed"`: **the accepted set itself**. Pass each row's `prompt` or `normalized` as
   *   `prompt` / `negativePrompt`.
   * - `"gpu"` (default): examples (the fixed prompts' names and original texts). The accepted set is
   *   any string; passing a row's `prompt` encodes it with umT5 on the GPU (the asset's embeddings are
   *   not used).
   */
  get prompts(): readonly Prompt[] {
    return this.#state.textEmbeds.entries.map(({ name, role, prompt, normalized }) => ({
      name,
      role,
      prompt,
      normalized,
    }));
  }

  /**
   * Generates one video from a prompt (and a condition image — {@link Ti2vGenerateRequest}). The
   * same request (seed or latents, knobs, image) gives the same values. Concurrent calls wait and run
   * in order (no two graphs resident at once — module doc).
   */
  async generate(request: Ti2vGenerateRequest): Promise<GeneratedVideo> {
    if (this.#disposal !== undefined) {
      throw new Error("wan.Ti2vPipeline: cannot generate after dispose");
    }
    const owned = ownWanRequestLatents(request);
    return await this.#chain(() => generateWanVideo(WAN22_TI2V_FAMILY, this.#state, owned));
  }

  /**
   * Releases. Destroys **only a GPU acquired internally** (`options.gpu` belongs to the caller).
   * MUST: wait for an in-flight generation to complete before destroying (flush-before-destroy — it
   * goes on the chain).
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
