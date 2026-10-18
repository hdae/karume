/**
 * `@karume/models/wan` — the subpath surface of the Wan2.1 / 2.2 (text → video) family (ADR 0118
 * decision 7, ADR 0121 decision 10).
 *
 * ADR 0008: this is an **explicitly designed thin surface**; internal modules are not re-exported
 * wholesale. The surface follows the user's stories — build ({@link WanPipeline.fromPretrained} /
 * {@link WanPipeline.fromAssets}; for Wan2.2 the two methods of the same names on
 * {@link WanTi2vPipeline}; the text encoder path `textEncoder`) / choose a source
 * ({@link WAN_SOURCES} — Wan2.2 only) / look up the example prompts (`prompts`) / generate
 * (`generate` — Wan2.2 also from a conditioning image: {@link WanTi2vGenerateRequest}) / subscribe
 * to generation progress (`onEvent` — {@link WanGenerateEvent}) / abort (the `signal` of
 * construction and generation) / turn frames into pixels ({@link wanFrameToRgba}) / release
 * (`dispose`).
 *
 * There are two text encoder paths (ADR 0119 decision 7): the default `"gpu"` runs umT5 (i8, by
 * cross-repository reference to umT5's distribution repository) on the GPU and accepts any prompt;
 * `"precomputed"` does not fetch umT5 and accepts only the fixed prompts of the text embedding
 * asset.
 *
 * The source table ({@link WAN_SOURCES}) holds only Wan2.2 (`karume-wan2.2`) — the Wan2.1
 * distribution `karume-wan2.1` is not published on Hugging Face (ADR 0118 addendum (2026-10-15)). A
 * local distribution is passed to `fromPretrained` with `nodeDirectory` of `@karume/hub/node` or
 * `denoDirectory` of `@karume/hub/deno`, and on the `"gpu"` path umT5's distribution
 * (`karume-umt5-xxl`) is passed through that source's `crossRepo` mapping.
 *
 * MUST: every module is free of side effects (no execution at import, no global mutable state —
 * CLAUDE.md). Tree-shaking through the barrel (`mod.ts`) works only on top of this invariant.
 */

export { WanPipeline } from "./src/wan/pipeline.ts";
/**
 * Wan2.2 TI2V 5B: text (+ a conditioning image) → video (T2V and I2V). The surface for
 * construction, generation, abort and release and the public types (results, events, construction
 * options) are the same as {@link WanPipeline}; the accepted set (1280×704 / 704×1280, 5 to 121
 * frames), the latent shape and the fps are the generation's values (ADR 0121 decision 10). Only the
 * request type is specific to 2.2 ({@link WanTi2vGenerateRequest} — the I2V conditioning image
 * `image` and how it is fitted to the size, `fit`; ADR 0121 decision 11). The distribution is
 * `karume-wan2.2` (ADR 0121 stage 8 — built by the recipe `dist.py --pipeline wan-ti2v`; the public
 * Hugging Face repository is `"wan2.2"` of {@link WAN_SOURCES}).
 */
export { WanTi2vPipeline } from "./src/wan/ti2v-pipeline.ts";
export type { WanTi2vGenerateRequest } from "./src/wan/ti2v-pipeline.ts";
/**
 * **The table of sources verified with this package version** (one table per family; keys are the
 * Hugging Face repository name without `karume-`, values are pinned commit SHAs — ADR 0073 / 0092).
 * Each generation is a separate repository, and only Wan2.2 is published:
 *
 * - `WAN_SOURCES["wan2.2"]` = `hdae/karume-wan2.2` (read by {@link WanTi2vPipeline.fromPretrained} —
 *   umT5 comes from `karume-umt5-xxl` by cross-repository reference)
 *
 * Wan2.1 is not published on Hugging Face, so it has no key (pass a local distribution to
 * {@link WanPipeline}).
 *
 * **An opt-in for callers who want to follow the package version automatically** — pass an entry as
 * is as the first argument of {@link WanTi2vPipeline.fromPretrained}. To pin reproducibility
 * yourself, write your own `{ repo, revision }` instead of using this table (`fromPretrained` has no
 * default).
 */
export { WAN_SOURCES } from "./src/wan/config.ts";
/**
 * The vocabulary of how the I2V conditioning image is fitted to the size (`fit` of
 * {@link WanTi2vGenerateRequest} — `"crop"` / `"stretch"`). Exported so that CLIs / UIs can get the
 * choices from the type (handled like {@link WanPromptRole}).
 */
export type { WanI2vFit } from "./src/wan/i2v-preprocess.ts";
/**
 * The shape of the I2V conditioning image (RGB8, interleaved — `image` of
 * {@link WanTi2vGenerateRequest}). Decoding is the caller's job (Karume has no image decoder), so the
 * type is exported to be available from this subpath alone (the same type as in the birefnet /
 * depth-anything / siglip2 subpaths).
 */
export type { Rgb8Image } from "./src/image/preprocess.ts";
export type {
  GeneratedVideo,
  WanAssets,
  WanFromPretrainedOptions,
  WanGenerateEvent,
  WanGenerateRequest,
  WanLatentSnapshot,
  WanPipelineOptions,
  WanRunComponent,
} from "./src/wan/pipeline.ts";

/**
 * One prompt line of the text embedding asset (the asset's metadata — an element of the class's
 * `prompts`, `WanPipeline.prompts` / `WanTi2vPipeline.prompts`) and the vocabulary of its role.
 * On the `"precomputed"` path they are the accepted set itself (a string outside the set is a
 * `ModelInputError`); on the `"gpu"` path they are examples. Either way, CLIs / UIs need this type to
 * offer choices.
 */
export type { WanPrompt, WanPromptRole } from "./src/wan/text-embeds.ts";

/**
 * One frame of a generated video → 8-bit RGBA (can be passed as is to `encodePng`). The package is
 * the source of truth for the uint8 conversion rule (`clamp(x/2 + 0.5, 0, 1)` → `round(·255)`), and
 * the reference values (sha256) are taken over the bytes from this rule too — consumers do not have
 * to rewrite it.
 */
export { wanFrameToRgba } from "./src/wan/frames.ts";

/**
 * A failure caused by the input (the request itself cannot be accepted = fixing the input makes it
 * pass — a prompt rejected by the preprocessing or the tokenizer, with a message that says how to
 * fix it; a prompt outside the set; a size outside the accepted set; a knob out of range). **One
 * type across all families**, so a host serving several families can tell 400 from 500 with this
 * alone. Broken internal invariants, asset mismatches and GPU capacity are thrown as plain
 * `Error`s.
 */
export { ModelInputError } from "./src/errors.ts";
