/**
 * Schema validation of `pipelineConfig` (ADR 0038 §1 — each pipeline implementation owns and
 * validates the schema).
 *
 * hub passes `pipelineConfig` through (it only strips forbidden keys and checks the size limits).
 * So **the canonical shape is this module**, and every hand-written check runs at parse time.
 *
 * MUST: unknown keys fail loudly (if a misspelling such as `{ noiseScale: 0.6, noise_scale: 0.6 }`
 * silently fell back to the default, the default the distributor intended and the run would
 * disagree unnoticed).
 * MUST: maps are read only through `Object.hasOwn` (a cross-cutting invariant).
 *
 * ## MUST: `styles` / `speakers` are "name → **table row number**", not aliases of IDs
 *
 * They resolve to the distribution's `style_vectors` / `speaker_embeddings` (`style.ts`), and when
 * a row number shifts the shapes still match, loading and running pass, and it stays silent with
 * only **the voice of a different style / different speaker** coming out. There is no other means
 * of detection, so the values are checked at parse time to be a **permutation** of `0..rows-1` (the
 * comparison with the tables' row counts is done by `Pipeline.fromAssets` against the real
 * assets).
 *
 * ## MUST: `maxTokens` / `maxFrames` are operating limits declared by the distribution
 *
 * The `(T, T)` tables of windowed relative-position attention were moved out to the host by ADR
 * 0013 (DeBERTa's relative-position tables by ADR 0045), and their allocation (8·T² bytes class)
 * became the host's responsibility. The upper bound of the baked graph's symbolic dimension is
 * received from the distribution, and `generate` fails on this value **before allocating the
 * tables** (canonical: `tools/export-recipes/sbv2/distribution.py`). No constant is held on the TS
 * side — so that when a distribution baked with a different limit arrives, the host is not the
 * only one holding the old number.
 *
 * NOTE: the defaults of the run-time knobs also appear with the same values in `symbols.json` (the
 * exporter side rejects a disagreement between the two at assembly time). The pipeline reads **this
 * one** only and ignores `symbols.json`'s `defaults` — so as not to hold the derivation source
 * twice (the same section in `text/symbols.ts` is optional).
 *
 * NOTE: the mapping table of public distribution repositories ({@link SOURCES}) also lives
 * here. It is not a value derivable from the manifest but the side of "which manifest to fetch",
 * so the distribution cannot hold it (ADR 0073).
 */

import type { HubRepoRef } from "@karume/hub";
import { assertAllowedKeys, readNumber, readRecord } from "../config/readers.ts";

/** The contract name of `pipeline` and the major this implementation accepts (ADR 0038 §1). */
export const SBV2_PIPELINE_NAME = "sbv2";
export const SBV2_PIPELINE_MAJOR = 1;

/**
 * The **mapping table of public distribution repositories** of the SBV2 family (ADR 0092 — one
 * table per family, **no default entry**).
 * The values are **the sources this package version verified** (pinned commit SHAs — ADR 0073).
 *
 * A key is the basename of the HF repository name with `karume-` dropped (`"karume-" + key` gives
 * back the basename of the repository name — this invariant is checked by `tests/sources_test.ts`).
 * There is one today:
 *
 * - `"sbv2-jvnv"` = `hdae/karume-sbv2-jvnv`
 *
 * Pass it as **an opt-in for following the package version automatically** — to pin reproducibility
 * yourself, write your own `{ repo, revision }` instead of this table (`fromPretrained` has no
 * default).
 *
 * MUST: pin the revision to a commit SHA — branches and tags can be moved on the distribution
 * side, so the bytes this published package reads would silently change for reasons on the network
 * side (an unrecoverable kind of accident). Specifying a SHA removes the revision-resolution
 * request itself, so an offline start with a complete cache also holds (ADR 0038). A user who
 * needs to follow main explicitly chooses `{ ...SOURCES["sbv2-jvnv"], revision: "main" }`.
 */
// NOTE: the revision is updated to main's SHA after the upload in §3 of the release runbook
// (docs/release-runbook.md) (inheriting the maintenance duty of ADR 0073 decision 3 — by hand, plus
// the runbook check).
export const SOURCES = {
  "sbv2-jvnv": {
    repo: "hdae/karume-sbv2-jvnv",
    revision: "101becff45856a0ad477457b54d1c927ca4d5d08",
  },
} as const satisfies Record<string, HubRepoRef>;

const ROOT_KEYS: readonly string[] = ["styles", "speakers", "maxTokens", "maxFrames", "defaults"];
const DEFAULTS_KEYS: readonly string[] = [
  "speaker",
  "style",
  "styleWeight",
  "sdpRatio",
  "noiseScale",
  "noiseScaleW",
  "lengthScale",
];

/** The distributor's recommended defaults (fill the unspecified fields of `generate`). */
export type Sbv2Defaults = {
  readonly speaker: string;
  readonly style: string;
  readonly styleWeight: number;
  readonly sdpRatio: number;
  readonly noiseScale: number;
  readonly noiseScaleW: number;
  readonly lengthScale: number;
};

export type Sbv2PipelineConfig = {
  /** Style name → row number of `style_vectors` (a permutation of `0..rows-1`). */
  readonly styles: ReadonlyMap<string, number>;
  /** Speaker name → row number of `speaker_embeddings` (a permutation of `0..rows-1`). */
  readonly speakers: ReadonlyMap<string, number>;
  /**
   * Upper limit of the token sequence passed to DeBERTa (the upper bound of the baked graph's
   * symbolic dimension).
   */
  readonly maxTokens: number;
  /** Upper limit of the total frame count passed to flow / voice (same as above). */
  readonly maxFrames: number;
  readonly defaults: Sbv2Defaults;
};

/**
 * A number used to compare allocation sizes, so it is checked to be not only an integer but a
 * **safe integer**.
 */
const isPositiveSafeInteger = (value: number): boolean => Number.isSafeInteger(value) && value > 0;

const readString = (raw: Record<string, unknown>, key: string, where: string): string => {
  if (!Object.hasOwn(raw, key)) throw new Error(`${where}.${key}: missing`);
  const value = raw[key];
  if (typeof value !== "string" || value.length === 0) {
    throw new Error(`${where}.${key}: not a non-empty string (${String(value)})`);
  }
  return value;
};

/**
 * Reads a "name → row number" table. Checks down to the values being a permutation of
 * `0..count-1`.
 *
 * MUST: do not drop the permutation check. A row number is the physical row of the table itself,
 * and duplicates or gaps do not change the shape of `style_vec` / `g` — the moment the check is
 * skipped it becomes a silent wrong value.
 */
const parseRowIndex = (raw: unknown, where: string): ReadonlyMap<string, number> => {
  const record = readRecord(raw, where);
  const names = Object.keys(record);
  if (names.length === 0) throw new Error(`${where}: empty (at least one is required)`);
  const table = new Map<string, number>();
  const seen = new Set<number>();
  for (const name of names) {
    const row = record[name];
    if (typeof row !== "number" || !Number.isInteger(row) || row < 0 || row >= names.length) {
      throw new Error(
        `${where}['${name}']: the row index is not an integer in 0..${names.length - 1} (${String(row)})`,
      );
    }
    if (seen.has(row)) {
      throw new Error(
        `${where}: row index ${row} is duplicated ('${name}' — table rows and names are not one-to-one)`,
      );
    }
    seen.add(row);
    table.set(name, row);
  }
  return table;
};

const parseDefaults = (
  raw: unknown,
  styles: ReadonlyMap<string, number>,
  speakers: ReadonlyMap<string, number>,
): Sbv2Defaults => {
  const where = "pipelineConfig.defaults";
  const record = readRecord(raw, where);
  assertAllowedKeys(record, DEFAULTS_KEYS, where);
  const speaker = readString(record, "speaker", where);
  const style = readString(record, "style", where);
  // The defaults too must be inside the acceptance set (if they were out of it at distribution
  // time, fromAssets fails — do not create "found out only by running one generation").
  if (!speakers.has(speaker)) {
    throw new Error(
      `${where}.speaker: '${speaker}' is not in speakers (available: ${[...speakers.keys()].join(
        " / ",
      )})`,
    );
  }
  if (!styles.has(style)) {
    throw new Error(
      `${where}.style: '${style}' is not in styles (available: ${[...styles.keys()].join(" / ")})`,
    );
  }
  return {
    speaker,
    style,
    styleWeight: readNumber(record, "styleWeight", where, Number.isFinite, "not a finite number"),
    sdpRatio: readNumber(record, "sdpRatio", where, Number.isFinite, "not a finite number"),
    noiseScale: readNumber(record, "noiseScale", where, Number.isFinite, "not a finite number"),
    noiseScaleW: readNumber(record, "noiseScaleW", where, Number.isFinite, "not a finite number"),
    // lengthScale multiplies the durations directly. At 0 or below the total frame count becomes 0
    // and the downstream (durationsToFrames) fails in the form "not an utterance" — do not make it
    // a failure far from its cause.
    lengthScale: readNumber(
      record,
      "lengthScale",
      where,
      (value) => Number.isFinite(value) && value > 0,
      "not a positive finite number",
    ),
  };
};

/** Checks and reads the manifest's `pipelineConfig` (the raw value hub passed through). */
export const parseSbv2PipelineConfig = (
  raw: Readonly<Record<string, unknown>>,
): Sbv2PipelineConfig => {
  assertAllowedKeys(raw, ROOT_KEYS, "pipelineConfig");
  const styles = parseRowIndex(
    Object.hasOwn(raw, "styles") ? raw["styles"] : undefined,
    "pipelineConfig.styles",
  );
  const speakers = parseRowIndex(
    Object.hasOwn(raw, "speakers") ? raw["speakers"] : undefined,
    "pipelineConfig.speakers",
  );
  const limit = "not a positive safe integer";
  return {
    styles,
    speakers,
    maxTokens: readNumber(raw, "maxTokens", "pipelineConfig", isPositiveSafeInteger, limit),
    maxFrames: readNumber(raw, "maxFrames", "pipelineConfig", isPositiveSafeInteger, limit),
    defaults: parseDefaults(
      Object.hasOwn(raw, "defaults") ? raw["defaults"] : undefined,
      styles,
      speakers,
    ),
  };
};
