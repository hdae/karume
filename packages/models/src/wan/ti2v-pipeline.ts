/**
 * `Ti2vPipeline` — the one surface for text (+ condition image) → video (Wan2.2 TI2V 5B's T2V and
 * I2V) (ADR 0121 decisions 10 and 11).
 *
 * The stage order ((encoder →) text → DiT → VAE, each stage opened and folded in turn), Session
 * lifetimes, cancellation and the public types (`GenerateRequest` / `GeneratedVideo` /
 * `GenerateEvent` / construction options) are the same single design as Wan2.1's `Pipeline`;
 * the source of truth is the doc at the top of `./pipeline.ts`. The family admission and the bodies
 * of construction and generation also live in the shared `family.ts`; this file holds the class
 * shell (private constructor, serialization chain, `dispose`) and the 2.2-only request type
 * ({@link Ti2vGenerateRequest} — the condition image), and passes `WAN22_TI2V_FAMILY` (pipeline
 * `wan-ti2v/1`, generation descriptor `WAN22_TI2V_GENERATION`).
 *
 * Every difference from Wan2.1 comes from the generation descriptor and the asset declarations (not
 * from code branches):
 *
 * - The accepted set is 1280×704 / 704×1280 × 4n+1 frames, 5–121 (default 1280×704, 33 frames)
 * - The latents have 48 channels (`[48, F', H/16, W/16]` — the VAE graph's ratio 8 × unpatchify 2);
 *   the output is 24 fps
 * - The DiT has 7 inputs (condition-side time and condition mask — in T2V the condition mask is all
 *   false and the condition-side time equals the generation side; in I2V the tokens of the first
 *   latent frame are true and the condition-side time is 0)
 * - It has the I2V VAE encoder components (three symbolic-length graphs — `vae-encoder.ts`) (only for
 *   the generation whose DiT input form is `"ti2v"`)
 *
 * NOTE: the duplicated shell (about 60 lines) is accepted — no shared base class or mixin (ADR 0121
 * stage 6 design §3.2. The body is the single one in `family.ts`; only the private state and the
 * delegation are duplicated).
 *
 * The distribution is `karume-wan2.2` (ADR 0121 stage 8 — built by the recipe
 * `dist.py --pipeline wan-ti2v`; the public HF repository is `SOURCES["wan2.2"]`). As in 2.1 there
 * are two entries: {@link Ti2vPipeline.fromPretrained}, which fetches from the distribution, and
 * {@link Ti2vPipeline.fromAssets}, which builds from already-fetched bytes.
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
} from "./pipeline.ts";
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
   * Fetches from the distribution and builds (the procedure and how `ref` is read are the same single
   * implementation as `Pipeline.fromPretrained` — `family.ts`). The fetched components are 2.1's
   * plus the three I2V VAE encoder parts (`vae_encoder_pre` / `vae_encoder_attn` /
   * `vae_encoder_post` — fetched even for use without an image). The manifest's pipeline must be
   * `wan-ti2v/1`.
   *
   * The source this package version verified is `"wan2.2"` in `SOURCES` (`config.ts`) — to pin
   * reproducibility yourself, the caller gives `{ repo, revision }` explicitly. When fetching from
   * HF, umT5's cross-repository target is fetched at the commit the manifest pins, so the caller does
   * not pass it. A local mirror (`models/karume-wan2.2`) is passed as a source handle
   * (`localDirectory` / `denoDirectory` of `@karume/hub/deno`), and on the `"gpu"` path umT5's
   * cross-repository target (`karume-umt5-xxl`) is passed as a mapping in that source's `crossRepo`
   * (the key is the repo the manifest's `text_encoder` declares — the same form as 2.1).
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
   * the embedding values). Its meaning per path is the same as `Pipeline.prompts`
   * (`"precomputed"`: the accepted set itself; `"gpu"`: examples).
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
   * in order (no two graphs resident at once — the doc at the top of `./pipeline.ts`).
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
