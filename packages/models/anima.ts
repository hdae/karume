/**
 * `@karume/models/anima` — the subpath surface of the Anima (text → image) family.
 *
 * ADR 0008: this is an **explicitly designed thin surface**; internal modules are not re-exported
 * wholesale. The surface follows the user's stories — build ({@link AnimaPipeline.fromPretrained} /
 * {@link AnimaPipeline.fromAssets}) / generate (`generate`) / subscribe to generation progress
 * (`onEvent` — {@link AnimaGenerateEvent}) / **approximate a preview from an intermediate latent**
 * ({@link approximatePreview}) / release (`dispose`) / handle the spelling of resolutions
 * ({@link parseResolution} / {@link formatResolution}).
 *
 * MUST: every module is free of side effects (no execution at import, no global mutable state —
 * CLAUDE.md). Tree-shaking through the barrel (`mod.ts`) works only on top of this invariant.
 */

export { AnimaPipeline } from "./src/anima/pipeline.ts";
export type {
  AnimaAssets,
  AnimaFromPretrainedOptions,
  AnimaGenerateEvent,
  AnimaGenerateRequest,
  AnimaLatentSnapshot,
  AnimaPipelineOptions,
  AnimaRunComponent,
  GeneratedImage,
} from "./src/anima/pipeline.ts";

/**
 * The vocabulary of denoise update rules (the values `AnimaGenerateRequest.sampler` accepts). When
 * omitted, `pipelineConfig.scheduler.type` of the manifest decides the default, so **giving one is an
 * explicit departure from the distributor's recommendation**. A type alone cannot enumerate the
 * choices, and consumers (CLI argument checks / UI choices) would have to respell the values, so the
 * list of values (frozen) is on the surface too.
 */
export type { AnimaSamplerType } from "./src/anima/config.ts";
export { ANIMA_SAMPLER_TYPES } from "./src/anima/config.ts";

/**
 * The vocabulary of DiT residency (ADR 0112): the values accepted by `residency` of
 * {@link AnimaPipelineOptions} / {@link AnimaGenerateRequest}, and `action` / `reason` of the
 * `residency` event ({@link AnimaGenerateEvent}). The types are exported so that consumers branching
 * on the reason (showing a downgrade in the UI and the like) do not have to respell the values.
 */
export type {
  AnimaResidency,
  AnimaResidencyAction,
  AnimaResidencyReason,
} from "./src/anima/residency.ts";

/**
 * **The table of sources verified with this package version** (one table per family; keys are the
 * Hugging Face repository name without `karume-`, values are pinned commit SHAs — ADR 0073 / 0092).
 * The repositories are split by "official / further training" (ADR 0087), so two entries:
 *
 * - `ANIMA_SOURCES["anima"]` = `hdae/karume-anima` (the official models; default = Turbo)
 * - `ANIMA_SOURCES["anima-extra"]` = `hdae/karume-anima-extra` (third-party fine-tunes;
 *   default = wai)
 *
 * In both, models other than the default are chosen with `{ model }`. **There is no default entry**
 * (the caller spells out which entry to use).
 *
 * **An opt-in for callers who want to follow the package version automatically** — pass an entry as
 * is as the first argument of `fromPretrained`. To pin reproducibility yourself, write your own
 * `{ repo, revision }` instead of using this table (`fromPretrained` has no default).
 *
 * NOTE: the entry of the old turbo-only repository (`hdae/karume-anima-turbo`) was removed
 * (2026-09-01, breaking — see the NOTE in `src/anima/config.ts`).
 */
export { ANIMA_SOURCES } from "./src/anima/config.ts";

/**
 * An approximate RGB preview from an intermediate latent. Passing the two fields returned by
 * `copyLatents()` of the `denoise-step` event ({@link AnimaLatentSnapshot}) **as they are** (without
 * denormalizing) returns RGBA at latent resolution (1/8 of the image) —
 * `const { data, shape } = event.copyLatents(); approximatePreview(data, shape)`.
 *
 * The VAE can be loaded only **after** the DiT is released (a VRAM MUST), so the pipeline does not
 * provide a VAE preview per step — this approximation is the only way to show per-step progress as
 * a picture, so it is on the surface. It is not an exact decode (a linear projection from 16 to 3
 * channels). The origin of the coefficients, the measurements of the calibration space and the
 * limits are in the implementation's doc.
 */
export { approximatePreview } from "./src/anima/preview.ts";

/**
 * The constants for denormalizing an intermediate latent (get them with
 * `const { mean, std } = animaLatents()` and pass them to
 * `denormalizeLatents(latents, shape, mean, std)`). **Not needed for previews** — the coefficients of
 * {@link approximatePreview} are calibrated in **the normalized space** (measured 2026-08-24), so
 * passing denormalized values gives a blown-out, different picture. Pass the return value of
 * `copyLatents()` as is.
 *
 * This is exported for consumers that want to handle latents on the same footing as the VAE decode
 * (their own decode, latent analysis, calibrating another projection). The two constants exist only
 * in the VAE's config, in neither the IR nor the distributed assets, so rewriting the
 * denormalization on the consumer side would duplicate the constants.
 *
 * MUST: export an **accessor**, not the constants themselves — {@link animaLatents} returns an
 * independent copy on every call. Handing out the mutable TypedArray itself would let a consumer's
 * modification silently leak into the denormalization of `generate` (no global mutable state — a
 * cross-cutting invariant in CLAUDE.md).
 * {@link denormalizeLatents} is a pure function that takes mean / std as arguments, so it is
 * exported as is.
 */
export { animaLatents, denormalizeLatents } from "./src/anima/latents.ts";

/**
 * The spelling of resolutions (`1344x768` / the square shorthand `512`) and the accepted set.
 *
 * **This package is the source of truth for the accepted set** (ADR 0038 §2 — it is not written in
 * the manifest), so CLIs and UIs that interpret `WxH` go through the same implementation. Without
 * this export the spelling contract would be duplicated.
 */
export { formatResolution, parseResolution } from "./src/anima/resolution.ts";
export type { ImageSize } from "./src/anima/resolution.ts";

/**
 * A failure caused by the input (the request itself cannot be accepted = fixing the input makes it
 * pass — the range of `seed`, the spelling of the resolution). **One type across all families**, so
 * a host serving several families can tell 400 from 500 with this alone. Broken internal
 * invariants, asset mismatches and GPU capacity are thrown as plain `Error`s.
 */
export { ModelInputError } from "./src/errors.ts";
