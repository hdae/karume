/**
 * Schema validation of `pipelineConfig` (ADR 0038 §1 — each pipeline implementation owns and
 * validates the schema).
 *
 * hub passes `pipelineConfig` through (it only strips forbidden keys and checks the size limits).
 * So **the canonical shape is this module**, and every hand-written check runs at parse time.
 *
 * MUST: unknown keys fail loudly (if a misspelling such as `{ maxFrames: …, max_frames: … }`
 * silently fell back to the default, the limit the distributor intended and the run would disagree
 * unnoticed).
 * MUST: maps are read only through `Object.hasOwn` (a cross-cutting invariant).
 *
 * ## Of the 5 fields here, **only `minFrames` / `maxFrames` take effect at run time**
 *
 * The other 3 (`sampleRate` / `featureDim` / `classes`) are **declarations**, like BiRefNet's
 * `interpolation`, and their acceptance set is the single value this implementation holds. Rather
 * than branching on them, a distribution that deviates is **not accepted** at parse time:
 *
 * - `sampleRate` — feature extraction has no resampler, and the window length, hop and the band of
 *   the mel basis are baked for 16kHz (the module doc of `features.ts`). Letting through a
 *   derivative trained at a different rate, `detect` runs to the end and emits **a plausible but
 *   different vowel sequence** (the shapes match).
 * - `featureDim` — compared with the last axis of the graph input (`assertGraph` in `pipeline.ts`).
 * - `classes` — **the order is the id** (`LIPSYNC_CLASSES` in `postprocess.ts`). A distribution
 *   with a different order only permutes the labels, and the segmentation and the `.lab` format
 *   still hold completely = the hardest to notice.
 *
 * ## MUST: `minFrames` / `maxFrames` are the distribution's operating range (no TS constants)
 *
 * The graph's time axis is symbolic, so **the IR carries no value range** (`symbols` in
 * `docs/ir-v2.md` is only a list of names). The range of the symbol `T` (20ms grid) declared with
 * `Dim(min=…, max=…)` at bake time, **converted to input-side units (10ms frames)**, is received
 * from the distribution, and `detect` fails on both sides (the upper limit before feature
 * extraction, the lower limit right after it) (canonical:
 * `tools/export-recipes/vowel_detector/distribution.py`). Putting defaults or constants here would
 * leave the host alone with the old numbers when a distribution baked with a different range
 * arrives (the same judgement as SBV2's `maxTokens`).
 *
 * ## MUST: no post-processing constants here (`switchPenalty` / `minDurationFrames` / `frameSec`)
 *
 * The canonical values are the module constants of `postprocess.ts` (= the smoothing used at
 * training time itself). Creating a declaration field would mean "the distribution can specify a
 * smoothing different from training", and there is no ground for moving that. The match of the
 * values is checked by `tests/vowel_detector_host_test.ts` against a copy of upstream's
 * `feature_config.json`.
 */

import { assertAllowedKeys } from "../config/readers.ts";
import { FEATURE_DIM, SAMPLE_RATE } from "./features.ts";
import { LIPSYNC_CLASSES } from "./postprocess.ts";

/** The contract name of `pipeline` and the major this implementation accepts (ADR 0038 §1). */
export const VOWEL_DETECTOR_PIPELINE_NAME = "vowel-detector";
export const VOWEL_DETECTOR_PIPELINE_MAJOR = 1;

const ROOT_KEYS: readonly string[] = [
  "sampleRate",
  "featureDim",
  "classes",
  "minFrames",
  "maxFrames",
];

/**
 * Step of the input length (the output is a 20ms grid = one per 2 input frames — the graph input
 * is `2T`).
 */
const LENGTH_MULTIPLE = 2;

export type VowelDetectorPipelineConfig = {
  /**
   * Sampling rate of the input waveform. Declaration only (the acceptance set is the single value
   * {@link SAMPLE_RATE}).
   */
  readonly sampleRate: typeof SAMPLE_RATE;
  /**
   * Feature dimension of the graph input. Declaration only (the acceptance set is the single value
   * {@link FEATURE_DIM}).
   */
  readonly featureDim: typeof FEATURE_DIM;
  /**
   * Output classes (**the order is the id**). Only the order identical to {@link LIPSYNC_CLASSES}
   * is accepted.
   */
  readonly classes: typeof LIPSYNC_CLASSES;
  /**
   * The **lower limit** of the 10ms frame count one `detect` accepts = the lower bound of the
   * symbolic dimension `T` (20ms grid) converted to input-side units. **A multiple of 2** (the
   * graph input is `2T`).
   */
  readonly minFrames: number;
  /**
   * The **upper limit** of the 10ms frame count one `detect` accepts = the upper bound of the
   * symbolic dimension `T` (20ms grid) converted to input-side units. **A multiple of 2** (the
   * graph input is `2T`).
   *
   * NOTE: **not** the number of `Dim(max=…)` itself — that one counts the 20ms grid, and this
   * field is twice it (the 10ms frame count). The comparison partner is also on the 10ms side (the
   * consumer of {@link parseVowelDetectorPipelineConfig} = `assertFrameLimit` in `pipeline.ts`).
   */
  readonly maxFrames: number;
};

/**
 * A numeric field whose acceptance set is a single value. Misspellings and unsupported values fail
 * with the same message.
 */
const readOnlyNumber = <T extends number>(
  raw: Record<string, unknown>,
  key: string,
  where: string,
  accepted: T,
  why: string,
): T => {
  if (!Object.hasOwn(raw, key)) throw new Error(`${where}.${key}: missing`);
  const value = raw[key];
  if (value !== accepted) {
    throw new Error(
      `${where}.${key}: this implementation supports only ${accepted} (${String(value)}) — ${why}`,
    );
  }
  return accepted;
};

/**
 * Reads the class vocabulary. Requires a match **down to the order** (the third item of the module
 * doc).
 */
const readClasses = (
  raw: Record<string, unknown>,
  key: string,
  where: string,
): typeof LIPSYNC_CLASSES => {
  if (!Object.hasOwn(raw, key)) throw new Error(`${where}.${key}: missing`);
  const value = raw[key];
  if (
    !Array.isArray(value) ||
    value.length !== LIPSYNC_CLASSES.length ||
    value.some((entry, index) => entry !== LIPSYNC_CLASSES[index])
  ) {
    throw new Error(
      `${where}.${key}: this implementation supports only the order [${LIPSYNC_CLASSES.join(", ")}]` +
        ` (${JSON.stringify(value)}) — the order is the class id itself, so ` +
        "a different order would only swap the labels while the segmentation and the .lab still come out valid",
    );
  }
  return LIPSYNC_CLASSES;
};

/**
 * Reads one side of the operating range (`minFrames` / `maxFrames`). Both are 10ms frame counts.
 *
 * MUST: check down to being a multiple of 2 — the graph input is `2T`, so an odd bound is a number
 * that holds as a declaration but whose meaning is broken: "a bound that only that one length
 * cannot pass".
 */
const readFrameBound = (raw: Record<string, unknown>, key: string, where: string): number => {
  if (!Object.hasOwn(raw, key)) throw new Error(`${where}.${key}: missing`);
  const value = raw[key];
  if (
    typeof value !== "number" ||
    !Number.isSafeInteger(value) ||
    value <= 0 ||
    value % LENGTH_MULTIPLE !== 0
  ) {
    throw new Error(
      `${where}.${key}: not a positive multiple of ${LENGTH_MULTIPLE} (${JSON.stringify(value)})` +
        " — the output is on a 20ms grid, and 2 input frames become 1",
    );
  }
  return value;
};

/** Checks and reads the manifest's `pipelineConfig` (the raw value hub passed through). */
export const parseVowelDetectorPipelineConfig = (
  raw: Readonly<Record<string, unknown>>,
): VowelDetectorPipelineConfig => {
  const where = "pipelineConfig";
  assertAllowedKeys(raw, ROOT_KEYS, where);
  const minFrames = readFrameBound(raw, "minFrames", where);
  const maxFrames = readFrameBound(raw, "maxFrames", where);
  if (minFrames > maxFrames) {
    throw new Error(
      `${where}: minFrames ${minFrames} exceeds maxFrames ${maxFrames}` +
        " — the declaration would accept no input length at all",
    );
  }
  return {
    sampleRate: readOnlyNumber(
      raw,
      "sampleRate",
      where,
      SAMPLE_RATE,
      "feature extraction has no resampler, and the window length / hop / mel basis are built for 16kHz",
    ),
    featureDim: readOnlyNumber(
      raw,
      "featureDim",
      where,
      FEATURE_DIM,
      "the 80-dim log-mel + 3 DSP dims are the training-time contract itself (src/vowel-detector/features.ts)",
    ),
    classes: readClasses(raw, "classes", where),
    minFrames,
    maxFrames,
  };
};
