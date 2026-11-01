/**
 * `@karume/models` — the barrel of the pipelines. Offered alongside the per-family subpaths
 * (`./anima` / `./birefnet` / `./depth-anything` / `./gemma` / `./gemma4-qat` / `./irodori` /
 * `./sbv2` / `./siglip2` / `./umt5` / `./vowel-detector` / `./wan` / `./moss`) (ADR 0037).
 *
 * Two kinds of export:
 *
 * - **The shared surface, flat** — what belongs to no single family: the option types shared by
 *   every `fromPretrained`, {@link ModelInputError}, {@link KARUME_SOURCES} and the PNG / WAV /
 *   image helpers.
 * - **One namespace per family** (`anima`, `birefnet`, `depthAnything`, `gemma`, `gemma4Qat`,
 *   `irodori`, `sbv2`, `siglip2`, `umt5`, `vowelDetector`, `wan`, `moss` — the subpath key in
 *   camelCase).
 *   Each namespace **is** that subpath's module (`export * as`), so the barrel and the subpaths
 *   cannot drift apart, and the members are documented once, on the subpath's own page (JSR lists a
 *   namespace without its members — follow the link in its doc).
 *
 * ```ts
 * import { encodeWav, sbv2 } from "@karume/models";
 * // `sbv2` is the same module object as `import * as sbv2 from "@karume/models/sbv2"`.
 * const tts = await sbv2.Pipeline.fromPretrained(sbv2.SOURCES["sbv2-jvnv"]);
 * ```
 *
 * A **thin surface** in the manner of ADR 0008 — it lists only the user's stories, "build a
 * pipeline and generate" and "write out the resulting image"; internal modules are not re-exported
 * wholesale.
 *
 * MUST: every module is free of side effects (no execution at import, no global mutable state, no
 * top-level registration — CLAUDE.md). Importing one export from the barrel, or one member of a
 * family namespace, drops the other families only while this invariant holds.
 *
 * NOTE: the `format` of the distribution manifest that `fromPretrained` reads is fixed per version,
 * and **older versions are not read**. The "package version ⇔ `format`" table is in the module doc of
 * `@karume/hub` (a table for planning ahead when a distribution has to be rebuilt).
 */

/**
 * The fetch-layer knobs **shared** by the `fromPretrained` option types of every family (`headers` /
 * `onProgress` / `onCacheError` / `onRetry` / `fetch` / `caches`). Each family's type is the
 * intersection of this with its construction options, so there is no point exporting one per
 * family; it is exported once, here — a caller that loads several families with the same fetch
 * settings can give those settings a name (otherwise it would copy the fields from every family's
 * type).
 */
export type { FromPretrainedHubOptions } from "./src/hub/load-options.ts";

/**
 * The **component replacement option** shared by the `fromPretrained` option types of every family
 * (ADR 0108 decision 19 / ADR 0109 decision 10) — role (a weights name in the manifest) → the same
 * role in another `karume/5` repository. Only components whose **graph descriptor has the same
 * sha256** can be replaced, and the identity is checked from the declarations of the two manifests
 * alone, before a single weight byte is fetched.
 */
export type { FromPretrainedComponentOptions } from "./src/hub/load-options.ts";
export type { ComponentSource } from "./src/hub/components.ts";

/**
 * A failure caused by the input (the request itself cannot be accepted = fixing the input makes it
 * pass). **One type across all families**, so a host serving every family can tell 400 from 500
 * with this one type. Broken internal invariants, asset mismatches and GPU capacity are thrown as
 * plain `Error`s (see each subpath).
 */
export { ModelInputError } from "./src/errors.ts";

/**
 * The source tables of every family merged into one (**the same keys and the same values** as each
 * family's `SOURCES`, `<namespace>.SOURCES` here — ADR 0092). A surface **only in the barrel**,
 * for callers that "walk every published distribution repository" (connectivity smoke tests,
 * prefetching, listings). A consumer that uses only one family should use that family's table
 * from its subpath (using the barrel pulls in the configs of nine families).
 */
export { KARUME_SOURCES } from "./src/sources.ts";

/**
 * RGBA → PNG / f32 waveform ↔ WAV / RGB8 → model input. **Common processing independent of any
 * pipeline** (image generation models generally end with PNG, audio generation models with WAV,
 * voice references come as WAV, and models that take images go through resize → rescale →
 * normalize), so it sits at the top of the barrel rather than in a family subpath.
 */
export { encodePng } from "./src/image/png.ts";
export { normalizeToNchw, resizeRgb8 } from "./src/image/preprocess.ts";
export type { Rgb8Image } from "./src/image/preprocess.ts";
export { decodeWav, encodeWav } from "./src/audio/wav.ts";
export type { DecodedWav } from "./src/audio/wav.ts";

/**
 * Anima — text → image. This namespace is the module of the subpath `@karume/models/anima`; its
 * members and their docs are on that page: https://jsr.io/@karume/models/doc/anima
 */
export * as anima from "./anima.ts";

/**
 * BiRefNet — image → alpha matte (background removal). This namespace is the module of the subpath
 * `@karume/models/birefnet`; its members and their docs are on that page:
 * https://jsr.io/@karume/models/doc/birefnet
 */
export * as birefnet from "./birefnet.ts";

/**
 * Depth Anything V2 — image → relative depth map. This namespace is the module of the subpath
 * `@karume/models/depth-anything`; its members and their docs are on that page:
 * https://jsr.io/@karume/models/doc/depth-anything
 */
export * as depthAnything from "./depth-anything.ts";

/**
 * Gemma — gemma4, text → text generation, together with the generation vocabulary (ADR 0083). This
 * namespace is the module of the subpath `@karume/models/gemma`; its members and their docs are on
 * that page: https://jsr.io/@karume/models/doc/gemma
 */
export * as gemma from "./gemma.ts";

/**
 * Gemma 4 QAT — E2B / E4B of the fixed mobile QAT (experimental; shares Gemma's conversation API).
 * This namespace is the module of the subpath `@karume/models/gemma4-qat`; its members and their
 * docs are on that page: https://jsr.io/@karume/models/doc/gemma4-qat
 */
export * as gemma4Qat from "./gemma4-qat.ts";

/**
 * Irodori — text → speech. This namespace is the module of the subpath `@karume/models/irodori`;
 * its members and their docs are on that page: https://jsr.io/@karume/models/doc/irodori
 */
export * as irodori from "./irodori.ts";

/**
 * SBV2 — text → speech. This namespace is the module of the subpath `@karume/models/sbv2`; its
 * members and their docs are on that page: https://jsr.io/@karume/models/doc/sbv2
 */
export * as sbv2 from "./sbv2.ts";

/**
 * SigLIP2 — image → embedding. This namespace is the module of the subpath
 * `@karume/models/siglip2`; its members and their docs are on that page:
 * https://jsr.io/@karume/models/doc/siglip2
 */
export * as siglip2 from "./siglip2.ts";

/**
 * umT5-XXL — text → the encoder's last hidden state. This namespace is the module of the subpath
 * `@karume/models/umt5`; its members and their docs are on that page:
 * https://jsr.io/@karume/models/doc/umt5
 */
export * as umt5 from "./umt5.ts";

/**
 * Vowel detection — Japanese speech → a `.lab` for lip sync. This namespace is the module of the
 * subpath `@karume/models/vowel-detector`; its members and their docs are on that page:
 * https://jsr.io/@karume/models/doc/vowel-detector
 */
export * as vowelDetector from "./vowel-detector.ts";

/**
 * Wan2.1 / 2.2 — text (+ a conditioning image) → video. This namespace is the module of the subpath
 * `@karume/models/wan`; its members and their docs are on that page:
 * https://jsr.io/@karume/models/doc/wan
 */
export * as wan from "./wan.ts";

/**
 * MOSS-Image — text → image. This namespace is the module of the subpath `@karume/models/moss`;
 * its members and their docs are on that page: https://jsr.io/@karume/models/doc/moss
 */
export * as moss from "./moss.ts";

/**
 * NOTE: the fixed-length greedy generation loop (`generateGreedy` / `GreedySpec`) was **removed**
 * from here (ADR 0083 decision 9 — a breaking change; the consumer-facing doc is limitations). It
 * remains in `src/generation/greedy.ts` as an internal helper for parity acceptance; the public
 * generation surface is the `sequence` of the generation-API phase.
 */
