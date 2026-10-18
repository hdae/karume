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
 * The BiRefNet family has no upstream `preprocessor_config.json`; the normalization constants exist
 * only in `ImagePreprocessor` (ImageNet statistics) of the bundled `handler.py`. Hard-coding 0.485
 * in TS anyway would give **silently wrong values** for derived models trained on other statistics
 * (distributions of this shape come as a row of fine-tunes) — with a wrong normalization the shapes
 * still match and the matte still comes out "plausible". `imageMean` / `imageStd` are held as
 * declarations and read per distribution.
 *
 * ## MUST: interpolation is a **declaration**; unsupported values are rejected at parse time
 *
 * Upstream (`handler.py` / the model card's usage example) runs
 * `torchvision.transforms.Resize((S, S))` with the default interpolation = bilinear, so bilinear is
 * the source of truth for this family's preprocessing. Silently running a derivative that requires
 * bicubic through bilinear **leaves the resized values off by up to 47/255** while loading and
 * execution both pass (measured — the preprocessing layer's module doc). Rather than branching, it
 * is **not accepted** (the type can only express `"bilinear"` too). The preprocessing layer
 * (`src/image/preprocess.ts`) implements bicubic as well, so widening the accepted set is a
 * judgement on "which interpolation the derivative's upstream was trained with", not on whether an
 * implementation exists.
 *
 * ## MUST: the post-processing procedure is a **declaration** too (closed set, fails loudly)
 *
 * The logit → 8-bit α post-processing ({@link MatteProcedure} in `./matte.ts`) differs upstream per
 * checkpoint (HR = round to 8 bit, then Pillow's BICUBIC; Lucida = antialiased bilinear in floating
 * point → 8 bit — ADR 0125). For the same reason as the normalization constants, no "checkpoint →
 * procedure" table lives in TS (a third-party fine-tune missing from the table would be silently
 * mismatched). The build step (`distribution.py`) decides it from the checkpoint and writes
 * `postprocess`; this module only accepts it. The field is required and has no default — so that a
 * distribution that forgot to write it does not silently run with one checkpoint's procedure.
 * Adding a required field makes old distributions unreadable, so the pipeline major was raised to 2
 * (the major-bump rule of ADR 0038).
 *
 * NOTE: the rescale divisor (255) is not here. The entry of `normalizeToNchw` is closed over 8-bit
 * pixel rows (`Rgb8Image`); giving a declaration to a number that cannot be chosen at run time
 * would only add two more sources of truth (the same judgement as `src/siglip2/config.ts`).
 *
 * NOTE: `imageWidth` / `imageHeight` are **the same numbers** as the baked graph's input
 * declaration, and the build step (`tools/export-recipes/birefnet/distribution.py`) derives them
 * from the graph and writes them. The declaration is kept anyway because (1) the preprocessing
 * resize target is a field that must be readable before the graph is opened and (2) the model card
 * can describe the resolution. The duplication is **checked against the graph every time** by the
 * user of {@link BirefnetPipelineConfig} (`assertStaticDim` in `pipeline.ts`), so it never runs
 * while they disagree.
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
import { MATTE_PROCEDURES, type MatteProcedure } from "./matte.ts";

/**
 * The contract name of `pipeline` and the major this implementation accepts (ADR 0038 §1).
 *
 * Major 2 = the shape in which the `postprocess` field became required (module doc — ADR 0125).
 * Major-1 distributions declare no procedure and are not read.
 */
export const BIREFNET_PIPELINE_NAME = "birefnet";
export const BIREFNET_PIPELINE_MAJOR = 2;

/**
 * The BiRefNet family's **public distribution repository table** (ADR 0092 — one table per family,
 * **no default entry**). The values are **the sources this package version verified** (pinned
 * commit SHAs — ADR 0073).
 *
 * The key is the HF repository basename with `karume-` dropped (`"karume-" + key` gives the
 * repository basename back — this invariant is checked by `tests/sources_test.ts`). One repository
 * per checkpoint, each holding **a separate graph per resolution** as two models (ADR 0092
 * decision 9):
 *
 * - `"birefnet-hr"` = `hdae/karume-birefnet-hr` (upstream BiRefNet_HR — models `"1024"` / `"2048"`,
 *   default `"1024"`)
 * - `"lucida"` = `hdae/karume-lucida` (Lucida, a derivative of BiRefNet_HR — the same two models,
 *   default `"1024"`)
 *
 * One repository = two models, so 2048² is spelled
 * `fromPretrained(BIREFNET_SOURCES["birefnet-hr"], { model: "2048" })`
 * (`BirefnetPipelineOptions.model` — `./pipeline.ts`). The model name is the input resolution
 * itself, and the preprocessing resize target is declared per model by `pipelineConfig.imageWidth`
 * / `imageHeight`.
 *
 * Pass it **as an opt-in when you want to follow the package version automatically** — to pin
 * reproducibility yourself, write your own `{ repo, revision }` instead of using this table
 * (`fromPretrained` has no default).
 *
 * MUST: pin the revision to a commit SHA — branches and tags can be moved on the distribution side,
 * so the bytes this published package reads would change silently for network-side reasons (an
 * unrecoverable accident). A SHA also removes the revision-resolution request itself, so an offline
 * start from a complete cache works at the same time (ADR 0038). A user who needs to follow main
 * explicitly chooses `{ ...BIREFNET_SOURCES["birefnet-hr"], revision: "main" }`.
 */
// NOTE: the revision is updated to main's SHA after the upload by §3 of the release runbook
// (docs/release-runbook.md) (inherits the maintenance duty of ADR 0073 decision 3 — by hand +
// runbook check).
export const BIREFNET_SOURCES = {
  "birefnet-hr": {
    repo: "hdae/karume-birefnet-hr",
    revision: "e0ed9f06843703265aa849477e6be5460ba8a300",
  },
  lucida: {
    repo: "hdae/karume-lucida",
    revision: "a416313c112b75a1735c6d6793ec5e2ff77d1a6a",
  },
} as const satisfies Record<string, HubRepoRef>;

const ROOT_KEYS: readonly string[] = [
  "imageWidth",
  "imageHeight",
  "imageMean",
  "imageStd",
  "interpolation",
  "postprocess",
];

/** The only interpolation this implementation accepts (the module doc's MUST). */
const INTERPOLATION = "bilinear";

/** Normalization constants (`[0, 1]` scale — spelled as in upstream `handler.py`). */
type BirefnetChannels = readonly [number, number, number];

export type BirefnetPipelineConfig = {
  /** Preprocessing resize target (= the baked graph's input width). */
  readonly imageWidth: number;
  /** Preprocessing resize target (= the baked graph's input height). */
  readonly imageHeight: number;
  readonly imageMean: BirefnetChannels;
  readonly imageStd: BirefnetChannels;
  /**
   * Held as a **declaration** in the manner of ADR 0038 §1 (not for branching — the module doc's
   * MUST).
   */
  readonly interpolation: typeof INTERPOLATION;
  /**
   * The logit → 8-bit α procedure (the module doc's MUST — the distribution declares each
   * checkpoint's upstream procedure).
   */
  readonly postprocess: MatteProcedure;
};

/**
 * Reads `postprocess` (anything outside the closed set {@link MATTE_PROCEDURES} is rejected).
 *
 * Compared with `===` against the array of the accepted set (no table lookup, so there is no room
 * to mistake an Object.prototype name such as `"toString"` for a procedure).
 */
const readPostprocess = (raw: Readonly<Record<string, unknown>>, where: string): MatteProcedure => {
  const key = "postprocess";
  if (!Object.hasOwn(raw, key)) throw new Error(`${where}.${key}: 無い`);
  const value = raw[key];
  const procedure = MATTE_PROCEDURES.find((name) => name === value);
  if (procedure === undefined) {
    throw new Error(
      `${where}.${key}: 未知の後処理の手順 ${JSON.stringify(value)}（期待 ${MATTE_PROCEDURES.map(
        (name) => `'${name}'`,
      ).join(" / ")}）`,
    );
  }
  return procedure;
};

/** Checks and reads the manifest's `pipelineConfig` (the raw value the hub passed through). */
export const parseBirefnetPipelineConfig = (
  raw: Readonly<Record<string, unknown>>,
): BirefnetPipelineConfig => {
  const where = "pipelineConfig";
  assertAllowedKeys(raw, ROOT_KEYS, where);
  const positive = "正の整数でない";
  const config: BirefnetPipelineConfig = {
    imageWidth: readNumber(raw, "imageWidth", where, isPositiveInteger, positive),
    imageHeight: readNumber(raw, "imageHeight", where, isPositiveInteger, positive),
    // The mean may be negative (a translation in `(x − mean·255) / (std·255)`), but a non-finite
    // one turns every pixel into NaN.
    imageMean: readChannels(raw, "imageMean", where, Number.isFinite, "有限の数でない要素がある"),
    imageStd: readChannels(
      raw,
      "imageStd",
      where,
      (value) => Number.isFinite(value) && value > 0,
      "正の有限数でない要素がある",
    ),
    interpolation: readOnly(
      raw,
      "interpolation",
      where,
      INTERPOLATION,
      "上流 handler.py の torchvision Resize は既定の補間（= bilinear）で、" +
        "bicubic で通すと resize の値が最大 47/255 ずれる",
    ),
    postprocess: readPostprocess(raw, where),
  };
  // A distribution whose fields are each finite but whose pair overflows f32 after normalization is
  // rejected at construction as an asset defect (waiting until `normalizeToNchw` would fail every
  // generation with an input-caused error type).
  const overflow = normalizationOverflow(config.imageMean, config.imageStd);
  if (overflow !== undefined) throw new Error(`${where}.imageMean / imageStd: ${overflow}`);
  return config;
};
