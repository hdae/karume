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
 * ## MUST: model-specific numbers live here (= the manifest), never as TS constants
 *
 * SigLIP2 base (224 / hidden 768) and so400m (384 / hidden 1152) are the same path differing only
 * in sizes, so hard-coding 224 or 768 in TS **gives silently wrong values for the other
 * distribution** (preprocessing resizing to another size while only the shapes match cannot happen
 * — the graph rejects it — but the side with the mixed-up size produces an image that looks resized
 * twice). The match against the real graph is done on the assets by `Pipeline.fromAssets`.
 *
 * ## MUST: interpolation is a **declaration**; unsupported values are rejected at parse time
 *
 * Upstream `preprocessor_config.json` has `"resample"` 2 (PIL's BILINEAR), so bilinear is the
 * source of truth for this family's preprocessing. Silently running a checkpoint that requires
 * bicubic through bilinear **leaves the resized values off by up to 47/255** while loading and
 * execution both pass (measured — the preprocessing layer's module doc). Rather than branching, it
 * is **not accepted** (the type can only express `"bilinear"` too). The preprocessing layer
 * (`src/image/preprocess.ts`) implements bicubic as well, so widening the accepted set is a
 * judgement on "which `resample` this distribution's upstream was trained with", not on whether an
 * implementation exists.
 *
 * NOTE: the rescale divisor (255) is not here. The entry of `normalizeToNchw` is closed over 8-bit
 * pixel rows (`Rgb8Image`), and a checkpoint whose `rescale_factor` is not 1/255 fails **at the
 * step that builds the distribution** (`tools/export-recipes/siglip2/distribution.py`). Giving a
 * declaration to a number that cannot be chosen at run time would only add a second source of
 * truth.
 *
 * NOTE: the public distribution repository table ({@link SOURCES}) lives here too. It is
 * not a "model-specific number" forbidden by the MUST above but on the "which manifest to fetch"
 * side — a value the distribution cannot hold in the first place (ADR 0073).
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
export const SIGLIP2_PIPELINE_NAME = "siglip2";
export const SIGLIP2_PIPELINE_MAJOR = 1;

/**
 * The SigLIP2 family's **public distribution repository table** (ADR 0092 — one table per family,
 * **no default entry**). The values are **the sources this package version verified** (pinned
 * commit SHAs — ADR 0073).
 *
 * The key is the HF repository basename with `karume-` dropped (`"karume-" + key` gives the
 * repository basename back — this invariant is checked by `tests/sources_test.ts`). One family
 * = one repository, holding two models that differ only in sizes:
 *
 * - `"siglip2"` = `hdae/karume-siglip2` (base 224 / hidden 768 and so400m 384 / hidden 1152 side by
 *   side, default = base)
 *
 * One repository = several models, so the repository reference alone does not pick one — so400m is
 * spelled `fromPretrained(SOURCES["siglip2"], { model: "so400m" })`
 * (`PipelineOptions.model` — `./pipeline.ts`).
 *
 * Pass it **as an opt-in when you want to follow the package version automatically** — to pin
 * reproducibility yourself, write your own `{ repo, revision }` instead of using this table
 * (`fromPretrained` has no default).
 *
 * MUST: pin the revision to a commit SHA — branches and tags can be moved on the distribution side,
 * so the bytes this published package reads would change silently for network-side reasons (an
 * unrecoverable accident). A SHA also removes the revision-resolution request itself, so an offline
 * start from a complete cache works at the same time (ADR 0038). A user who needs to follow main
 * explicitly chooses `{ ...SOURCES["siglip2"], revision: "main" }`.
 */
// NOTE: the revision is updated to main's SHA after the upload by §3 of the release runbook
// (docs/release-runbook.md) (inherits the maintenance duty of ADR 0073 decision 3 — by hand +
// runbook check).
export const SOURCES = {
  siglip2: {
    repo: "hdae/karume-siglip2",
    revision: "71077b1810e42e9a1a25d80ee0f43c75db5d3b23",
  },
} as const satisfies Record<string, HubRepoRef>;

const ROOT_KEYS: readonly string[] = [
  "imageWidth",
  "imageHeight",
  "imageMean",
  "imageStd",
  "hiddenDim",
  "interpolation",
];

/** The only interpolation this implementation accepts (the module doc's MUST). */
const INTERPOLATION = "bilinear";

/** Normalization constants (`[0, 1]` scale — spelled as in `preprocessor_config.json`). */
type Siglip2Channels = readonly [number, number, number];

export type Siglip2PipelineConfig = {
  /** Preprocessing resize target (`size.width` of `preprocessor_config.json`). */
  readonly imageWidth: number;
  /** Preprocessing resize target (`size.height` of `preprocessor_config.json`). */
  readonly imageHeight: number;
  readonly imageMean: Siglip2Channels;
  readonly imageStd: Siglip2Channels;
  /** Width of `pooler_output` (the vision tower's hidden — base 768 / so400m 1152). */
  readonly hiddenDim: number;
  /**
   * Held as a **declaration** in the manner of ADR 0038 §1 (not for branching — the module doc's
   * MUST).
   */
  readonly interpolation: typeof INTERPOLATION;
};

/** Checks and reads the manifest's `pipelineConfig` (the raw value the hub passed through). */
export const parseSiglip2PipelineConfig = (
  raw: Readonly<Record<string, unknown>>,
): Siglip2PipelineConfig => {
  const where = "pipelineConfig";
  assertAllowedKeys(raw, ROOT_KEYS, where);
  const positive = "not a positive integer";
  const config: Siglip2PipelineConfig = {
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
    hiddenDim: readNumber(raw, "hiddenDim", where, isPositiveInteger, positive),
    interpolation: readOnly(
      raw,
      "interpolation",
      where,
      INTERPOLATION,
      "upstream preprocessor_config.json has resample 2 (PIL's BILINEAR), " +
        "and running bicubic shifts the resized values by up to 47/255",
    ),
  };
  // A distribution whose fields are each finite but whose pair overflows f32 after normalization is
  // rejected at construction as an asset defect (waiting until `normalizeToNchw` would fail every
  // generation with an input-caused error type).
  const overflow = normalizationOverflow(config.imageMean, config.imageStd);
  if (overflow !== undefined) throw new Error(`${where}.imageMean / imageStd: ${overflow}`);
  return config;
};
