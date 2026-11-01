/**
 * The MOSS-Image manifest contract: the pipeline name and major this implementation reads, and the
 * schema of `pipelineConfig` (ADR 0038 §1 — each pipeline implementation owns and validates its
 * schema; ADR 0138 addendum of 2026-11-01).
 *
 * The hub passes `pipelineConfig` through (it only strips forbidden keys and checks the size limit),
 * so **this module is the source of truth for the shape**; the recipe writes it
 * (`MOSS_PIPELINE_CONFIG` in `tools/export-recipes/moss/distribution.py`). It declares upstream's
 * request defaults (ADR 0138 decision 9 gave them to the manifest):
 *
 * - `scheduler.shift` — the flow-matching shift (upstream's pinned scheduler config: 5);
 * - `defaults.width` / `defaults.height` — pixels (upstream: 1024 × 1024);
 * - `defaults.steps` / `defaults.guidance` — `num_inference_steps` / `guidance_scale` (50 / 3);
 * - `defaults.negativePrompt` — the negative prompt used while guidance ≠ 1 (upstream: `""`).
 *
 * MUST: unknown keys and missing fields fail loudly (a misspelling that fell back silently would run
 * something other than what the distributor declared). The value ranges are the request checks'
 * (`./generate.ts`), and a declaration outside them is an asset inconsistency — a plain `Error`,
 * not a `ModelInputError` (ADR 0107 decision 2). Whether the default steps × shift pair gives a σ
 * schedule needs the training grid of the `dit` container, so the pipeline's admission checks it
 * (`./pipeline.ts`).
 * MUST: maps are read only through `Object.hasOwn` (cross-cutting invariant).
 *
 * There is no source table (`SOURCES`): the distribution `karume-moss-image` is not published on
 * Hugging Face, and an entry before publication can only 404 (ADR 0073 decision 1). A local
 * distribution is passed as a source handle (`nodeDirectory` / `denoDirectory`).
 *
 * MUST: every module is side-effect free (no import-time execution, no global mutable state).
 */

import { assertAllowedKeys, readNumber, readRecord } from "../config/readers.ts";
import { MOSS_MAX_SIDE_PX, MOSS_PIXELS_PER_CELL } from "./request.ts";

/** The `pipeline` contract name and the major this implementation accepts (ADR 0038 §1). */
export const MOSS_PIPELINE_NAME = "moss-image";
export const MOSS_PIPELINE_MAJOR = 1;

/** The manifest asset holding the Qwen3.5 tokenizer (`karume-qwen3_5-tokenizer/1`). */
export const MOSS_TOKENIZER_ASSET = "tokenizer";

/** The distributor's defaults (they fill the fields a request leaves unset). */
export type MossPipelineConfig = {
  readonly scheduler: { readonly shift: number };
  readonly defaults: {
    readonly width: number;
    readonly height: number;
    readonly steps: number;
    readonly guidance: number;
    readonly negativePrompt: string;
  };
};

const ROOT_KEYS: readonly string[] = ["scheduler", "defaults"];
const SCHEDULER_KEYS: readonly string[] = ["shift"];
const DEFAULTS_KEYS: readonly string[] = ["width", "height", "steps", "guidance", "negativePrompt"];

/** Finite in f32 too (the host multiplies by the f32 rounding of a knob — `./generate.ts`). */
const finiteInF32 = (value: number): boolean => Number.isFinite(Math.fround(value));

/** Reads a section of fields (rejects a missing section, a non-object and unknown keys). */
const readSection = (
  raw: Readonly<Record<string, unknown>>,
  key: string,
  allowed: readonly string[],
): Record<string, unknown> => {
  const where = `pipelineConfig.${key}`;
  const record = readRecord(Object.hasOwn(raw, key) ? raw[key] : undefined, where);
  assertAllowedKeys(record, allowed, where);
  return record;
};

/** A side the request check accepts (a positive multiple of 16, at most 65,536 px). */
const isAcceptedSide = (value: number): boolean =>
  Number.isInteger(value) &&
  value > 0 &&
  value % MOSS_PIXELS_PER_CELL === 0 &&
  value <= MOSS_MAX_SIDE_PX;

/** Checks and reads the manifest's `pipelineConfig` (the raw value the hub passed through). */
export const parseMossPipelineConfig = (
  raw: Readonly<Record<string, unknown>>,
): MossPipelineConfig => {
  assertAllowedKeys(raw, ROOT_KEYS, "pipelineConfig");
  const scheduler = readSection(raw, "scheduler", SCHEDULER_KEYS);
  const defaults = readSection(raw, "defaults", DEFAULTS_KEYS);
  const shift = readNumber(
    scheduler,
    "shift",
    "pipelineConfig.scheduler",
    (value) => value > 0 && finiteInF32(value),
    "not a finite number > 0 that fits in f32",
  );
  const where = "pipelineConfig.defaults";
  const side = `not a positive multiple of ${MOSS_PIXELS_PER_CELL} up to ${MOSS_MAX_SIDE_PX}`;
  const width = readNumber(defaults, "width", where, isAcceptedSide, side);
  const height = readNumber(defaults, "height", where, isAcceptedSide, side);
  const steps = readNumber(
    defaults,
    "steps",
    where,
    (value) => Number.isSafeInteger(value) && value >= 1,
    "not a positive integer",
  );
  const guidance = readNumber(
    defaults,
    "guidance",
    where,
    (value) => value >= 0 && finiteInF32(value),
    "not a finite number ≥ 0 that fits in f32",
  );
  if (!Object.hasOwn(defaults, "negativePrompt")) {
    throw new Error(`${where}.negativePrompt: missing`);
  }
  const negativePrompt = defaults["negativePrompt"];
  if (typeof negativePrompt !== "string") {
    throw new Error(`${where}.negativePrompt: not a string`);
  }
  return { scheduler: { shift }, defaults: { width, height, steps, guidance, negativePrompt } };
};
