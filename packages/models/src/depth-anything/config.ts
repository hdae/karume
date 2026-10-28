/**
 * Schema validation of `pipelineConfig` (ADR 0038 §1 — each pipeline implementation owns and
 * validates its schema).
 *
 * The hub passes `pipelineConfig` through (it only strips forbidden keys and checks the size
 * limit). **This module is therefore the source of truth for the shape**, and every hand-written
 * check runs at parse time.
 *
 * MUST: unknown keys fail loudly (if a misspelling such as `{ imageMean: …, image_mean: … }`
 * silently fell back to the default, the preprocessing the distributor intended and the one that
 * runs would disagree without anyone noticing).
 * MUST: maps are read only through `Object.hasOwn` (cross-cutting invariant).
 *
 * ## MUST: interpolation accepts `"bicubic"` **only** (unlike BiRefNet / SigLIP2)
 *
 * Depth Anything V2's upstream `preprocessor_config.json` has `"resample": 3` (**BICUBIC as a PIL
 * constant**), a different filter from the bilinear (2) of SigLIP2 / BiRefNet. The default of
 * `resizeRgb8` is the value of the two existing families (bilinear), so unless the accepted set is
 * narrowed to one value instead of branching on the declaration, a mix-up passes as a
 * `pixel_values` error of up to 0.59 (34 times one uint8 LSB = 0.0175 — measured). Shapes and value
 * ranges still match, so the depth map comes out "plausible".
 *
 * The field names and value ranges are the same as BiRefNet's five fields, but **the schema is not
 * shared** (in the manner of ADR 0038 §1 — folding accepted sets that differ per family into one
 * means loosening one silently loosens the other).
 *
 * ## NOTE: the resolution is a declaration, not the graph's source of truth
 *
 * `imageWidth` / `imageHeight` are **the same numbers** as the baked graph's input declaration; the
 * build step (`tools/export-recipes/depth_anything/distribution.py`) writes them from `size` of the
 * upstream `preprocessor_config.json` and checks them against the graph in the same place. The
 * declaration is kept anyway because (1) the preprocessing resize target is a field that must be
 * readable before the graph is opened and (2) the model card can describe the resolution. The
 * duplication is **checked against the graph every time** by the user of
 * {@link DepthAnythingPipelineConfig} (`assertStaticDim` in `pipeline.ts`), so it never runs while
 * they disagree.
 *
 * NOTE: the rescale divisor (255) is not here. The entry of `normalizeToNchw` is closed over 8-bit
 * pixel rows (`Rgb8Image`); giving a declaration to a number that cannot be chosen at run time
 * would only add two more sources of truth (the same judgement as `src/siglip2/config.ts` /
 * `src/birefnet/config.ts`).
 *
 * NOTE: `keep_aspect_ratio` / `ensure_multiple_of` of the upstream config are not here. The baked
 * graph accepts **one square size only** (518² = patch 14 × 37), so a path that keeps the aspect
 * ratio has nowhere to go (stretching is the only choice). A declaration alone would read as "it
 * could keep it but does not", so the model card's prose carries the fact.
 *
 * NOTE: the public distribution repository table ({@link SOURCES}) lives here too.
 * It is not derivable from the manifest but on the "which manifest to fetch" side, so the
 * distribution cannot hold it (ADR 0073).
 */

import type { HubRepoRef } from "@karume/hub";
import {
  assertAllowedKeys,
  isPositiveInteger,
  readChannels,
  readNumber,
  readOnly,
} from "../config/readers.ts";
import { normalizationOverflow } from "../image/preprocess.ts";

/** The contract name of `pipeline` and the major this implementation accepts (ADR 0038 §1). */
export const DEPTH_ANYTHING_PIPELINE_NAME = "depth-anything";
export const DEPTH_ANYTHING_PIPELINE_MAJOR = 1;

/**
 * The Depth Anything family's **public distribution repository table** (ADR 0092 — one table per
 * family, **no default entry**). The values are **the sources this package version verified**
 * (pinned commit SHAs — ADR 0073).
 *
 * The key is the HF repository basename with `karume-` dropped (`"karume-" + key` gives the
 * repository basename back — this invariant is checked by `tests/sources_test.ts`):
 *
 * - `"depth-anything-v2"` = `hdae/karume-depth-anything-v2` (default = Small — the only size that
 *   is Apache-2.0 upstream)
 *
 * Generations and versions are separate repositories = separate entries (ADR 0092 decision 1), and
 * the generation is spelled in the repository name. V3 is a different architecture, not a successor
 * for single-image depth, so if it comes it does not overwrite the existing value but sits beside
 * it **as a separate key** (the older generation's repository stays published). Size is a model
 * axis within a generation, so additional distributable sizes would join this one entry (chosen
 * with `{ model }`).
 *
 * Pass it **as an opt-in when you want to follow the package version automatically** — to pin
 * reproducibility yourself, write your own `{ repo, revision }` instead of using this table
 * (`fromPretrained` has no default).
 *
 * MUST: pin the revision to a commit SHA — branches and tags can be moved on the distribution side,
 * so the bytes this published package reads would change silently for network-side reasons (an
 * unrecoverable accident). A SHA also removes the revision-resolution request itself, so an offline
 * start from a complete cache works at the same time (ADR 0038). A user who needs to follow main
 * explicitly chooses `{ ...SOURCES["depth-anything-v2"], revision: "main" }`.
 */
// NOTE: the revision is updated to main's SHA after the upload by §3 of the release runbook
// (docs/release-runbook.md) (inherits the maintenance duty of ADR 0073 decision 3 — by hand +
// runbook check).
export const SOURCES = {
  "depth-anything-v2": {
    repo: "hdae/karume-depth-anything-v2",
    revision: "b3969c2e64dba345a4ec996231abd642c06a8efd",
  },
} as const satisfies Record<string, HubRepoRef>;

const ROOT_KEYS: readonly string[] = [
  "imageWidth",
  "imageHeight",
  "imageMean",
  "imageStd",
  "interpolation",
];

/** The only interpolation this implementation accepts (the module doc's MUST). */
const INTERPOLATION = "bicubic";

/** Normalization constants (`[0, 1]` scale — spelled as in upstream `preprocessor_config.json`). */
type DepthAnythingChannels = readonly [number, number, number];

export type DepthAnythingPipelineConfig = {
  /** Preprocessing resize target (= the baked graph's input width). */
  readonly imageWidth: number;
  /** Preprocessing resize target (= the baked graph's input height). */
  readonly imageHeight: number;
  readonly imageMean: DepthAnythingChannels;
  readonly imageStd: DepthAnythingChannels;
  /**
   * Held as a **declaration** in the manner of ADR 0038 §1 (not for branching — the module doc's
   * MUST).
   */
  readonly interpolation: typeof INTERPOLATION;
};

/** Checks and reads the manifest's `pipelineConfig` (the raw value the hub passed through). */
export const parseDepthAnythingPipelineConfig = (
  raw: Readonly<Record<string, unknown>>,
): DepthAnythingPipelineConfig => {
  const where = "pipelineConfig";
  assertAllowedKeys(raw, ROOT_KEYS, where);
  const positive = "not a positive integer";
  const config: DepthAnythingPipelineConfig = {
    imageWidth: readNumber(raw, "imageWidth", where, isPositiveInteger, positive),
    imageHeight: readNumber(raw, "imageHeight", where, isPositiveInteger, positive),
    // The mean may be negative (a translation in `(x − mean·255) / (std·255)`), but a non-finite
    // one turns every pixel into NaN.
    imageMean: readChannels(
      raw,
      "imageMean",
      where,
      Number.isFinite,
      "has an element that is not a finite number",
    ),
    imageStd: readChannels(
      raw,
      "imageStd",
      where,
      (value) => Number.isFinite(value) && value > 0,
      "has an element that is not a positive finite number",
    ),
    interpolation: readOnly(
      raw,
      "interpolation",
      where,
      INTERPOLATION,
      "upstream preprocessor_config.json has resample 3 (PIL's BICUBIC), " +
        "and running bilinear shifts pixel_values by 34 times one uint8 LSB",
    ),
  };
  // A distribution whose fields are each finite but whose pair overflows f32 after normalization is
  // rejected at construction as an asset defect (waiting until `normalizeToNchw` would fail every
  // generation with an input-caused error type).
  const overflow = normalizationOverflow(config.imageMean, config.imageStd);
  if (overflow !== undefined) throw new Error(`${where}.imageMean / imageStd: ${overflow}`);
  return config;
};
