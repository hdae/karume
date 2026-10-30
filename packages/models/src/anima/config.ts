/**
 * Schema validation of `pipelineConfig` (ADR 0038 §1 — each pipeline implementation owns and
 * validates the schema).
 *
 * The hub passes `pipelineConfig` through (it only strips forbidden keys and checks the size
 * limit). So **this module is the source of truth for the shape**, and every hand-written check
 * runs at parse time.
 *
 * MUST: unknown keys fail loudly (if a misspelling like `{ steps: 8, step: 8 }` silently fell back
 * to the default, the run would disagree with the default the distributor intended, unnoticed).
 * MUST: maps are read only through `Object.hasOwn` (cross-cutting invariant).
 *
 * NOTE: the public distribution repository table ({@link SOURCES}) lives here too. It is not
 * a value derivable from the manifest but the "which manifest to fetch" side, so the distribution
 * cannot hold it (ADR 0073).
 */

import type { HubRepoRef } from "@karume/hub";

import { assertAllowedKeys, readNumber, readRecord } from "../config/readers.ts";
import { assertAcceptableResolution, type ImageSize } from "./resolution.ts";
import { assertAcceptableSteps } from "./sampler.ts";

/** Contract name of `pipeline` and the major version this implementation accepts (ADR 0038 §1). */
export const ANIMA_PIPELINE_NAME = "anima";
export const ANIMA_PIPELINE_MAJOR = 1;

/**
 * **Public distribution repository table** of the Anima family (ADR 0092 — one table per family,
 * **no default entry**). The values are **the sources this package version verified** (pinned
 * commit SHAs — ADR 0073).
 *
 * A key is the HF repository basename without `karume-` (`"karume-" + key` gives the basename back
 * — the check in `tests/sources_test.ts` covers this invariant). The repositories are split along
 * "official / fine-tuned" (ADR 0087) into two:
 *
 * - `"anima"` = `hdae/karume-anima` (the official models together — 5 variants of Turbo / Base /
 *   Aesthetic, default = Turbo)
 * - `"anima-extra"` = `hdae/karume-anima-extra` (**third-party fine-tunes** together, default =
 *   `anima-wai-v1.0`)
 *
 * Both are one repository = several models, so a repository reference alone does not pick one — to
 * use a non-default model, write `fromPretrained(SOURCES["anima"], { model })`.
 *
 * `anima-extra` does not ship the text stack (`text_encoder` / `vae_decoder` / `tokenizer` /
 * `tokenizer_2`) itself and **borrows it by cross-repository reference** from the official
 * repository (the revision `"anima"` points at) (ADR 0038 §7). So the two are not independent: when
 * the official one is re-uploaded, extra is rebuilt too (the MUST on publication order is in
 * docs/release-runbook.md).
 *
 * Pass it as **the opt-in for following the package version automatically** — to pin
 * reproducibility yourself, write your own `{ repo, revision }` instead of this table
 * (`fromPretrained` has no default).
 *
 * MUST: pin the revision to a commit SHA — branches and tags can be moved on the distribution side,
 * so the bytes this published package reads would change silently for network-side reasons (an
 * unrecoverable accident). A SHA removes the revision-resolution request itself, so an offline
 * start with a complete cache works at the same time (ADR 0038). Users who need to follow main
 * choose `{ ...SOURCES["anima"], revision: "main" }` explicitly.
 *
 * NOTE: the old turbo-only repository (`hdae/karume-anima-turbo` — one model per repository with
 * the LoRA baked in) is retired (ruling of 2026-09-01, breaking). It was merged into a form where
 * the official Turbo checkpoint (anima-turbo-v1.1) lives in `"anima"` as its default model.
 */
// NOTE: the revision is updated to main's SHA after upload in §3 of the release runbook
// (docs/release-runbook.md) (inherits the upkeep duty of ADR 0073 decision 3 — by hand + the
// runbook check).
export const SOURCES = {
  anima: {
    repo: "hdae/karume-anima",
    revision: "a5fac03dc6563657b4e5c797c78cc3c813912a3e",
  },
  "anima-extra": {
    repo: "hdae/karume-anima-extra",
    revision: "15731a612517df0064db69e8cb7c9825fce00c91",
  },
} as const satisfies Record<string, HubRepoRef>;

const ROOT_KEYS: readonly string[] = ["scheduler", "defaults"];
const SCHEDULER_KEYS: readonly string[] = ["type", "shift", "numTrainTimesteps"];
const DEFAULTS_KEYS: readonly string[] = ["steps", "guidanceScale", "resolution", "negativePrompt"];
const RESOLUTION_KEYS: readonly string[] = ["width", "height"];

/**
 * Update rule of denoising (sampler type). The manifest's declaration sets the **default**, and the
 * `sampler` field of the generation request (`GenerateRequest`) overrides it per call
 * (re-ruling of 2026-08-25 — the distribution default is the distributor's recommendation, the
 * choice of update rule is the user's knob). Both values are always valid, independent of the
 * assets.
 *
 * - `"euler"` — first-order update of `FlowMatchEulerDiscreteScheduler` (`cfgEulerStep` in
 *   `sampler.ts`).
 * - `"dpmpp-2m"` — DPM++ 2M (`src/generation/dpm-solver-multistep.ts`).
 */
export type SamplerType = (typeof SAMPLER_TYPES)[number];

/**
 * The values of {@link SamplerType} (the source of truth for the vocabulary — the type is
 * derived from it). The check ({@link assertAnimaSamplerType}) and CLI / UI choices all read this
 * one list.
 *
 * MUST: frozen — it is a public module-scope shared object, and if a JS consumer modified it, the
 * accepted set of every later pipeline (both the manifest side and the request side) would change
 * silently.
 */
export const SAMPLER_TYPES = Object.freeze(["euler", "dpmpp-2m"] as const);

/**
 * Value when `scheduler.type` is omitted.
 *
 * MUST: the default is `"euler"` — `type` is an optional field added additively (ADR 0038 §1), so
 * **existing distribution manifests without this field have to keep the old update rule without
 * changing a byte** (repositories already distributed cannot be migrated).
 */
const DEFAULT_SAMPLER_TYPE: SamplerType = "euler";

/**
 * Scheduler configuration (can vary per distribution, so the manifest holds it). `type` is optional
 * (default {@link DEFAULT_SAMPLER_TYPE}) and the parsed value is **always filled** — resolving the
 * default is confined to this one place, and the consumer (pipeline) has no branch.
 */
type AnimaScheduler = {
  readonly type: SamplerType;
  readonly shift: number;
  readonly numTrainTimesteps: number;
};

/** The distributor's recommended defaults (fill the fields `generate` leaves unspecified). */
type AnimaDefaults = {
  readonly steps: number;
  readonly guidanceScale: number;
  readonly resolution: ImageSize;
  readonly negativePrompt?: string;
};

export type AnimaPipelineConfig = {
  readonly scheduler: AnimaScheduler;
  readonly defaults: AnimaDefaults;
};

const parseResolutionEntry = (raw: unknown, where: string): ImageSize => {
  const record = readRecord(raw, where);
  assertAllowedKeys(record, RESOLUTION_KEYS, where);
  const size: ImageSize = {
    width: readNumber(record, "width", where, Number.isInteger, "not a positive integer"),
    height: readNumber(record, "height", where, Number.isInteger, "not a positive integer"),
  };
  // The default must also be inside the accepted set (if the distribution falls outside it,
  // fromAssets fails — no "found out only after running one generation").
  // MUST: the accepted-set conditions are borrowed from resolution.ts as the single source, but
  // **the thrown type is changed back here** — what fails is the distribution manifest's
  // declaration, not the caller's request, so it is an asset mismatch, not input-caused (the 500
  // side of ADR 0107 decision 2). The original `ModelInputError` is kept as `cause`.
  try {
    assertAcceptableResolution(size);
  } catch (cause) {
    throw new Error(cause instanceof Error ? cause.message : String(cause), { cause });
  }
  return size;
};

const parseDefaults = (raw: unknown): AnimaDefaults => {
  const where = "pipelineConfig.defaults";
  const record = readRecord(raw, where);
  assertAllowedKeys(record, DEFAULTS_KEYS, where);
  const steps = readNumber(record, "steps", where, Number.isFinite, "not a finite number");
  // If the default is outside the generation check, a generate whose request passes nothing fails
  // as input-caused.
  // MUST: the accepted-set conditions call the single generation check (`assertAcceptableSteps` in
  // `sampler.ts`) — copying the range here would let one side drift looser. The thrown type is
  // changed back here for the same reason as the resolution (what fails is the distribution
  // manifest's declaration, not the caller's request — the 500 side of ADR 0107 decision 2). The
  // original `ModelInputError` is kept as `cause`.
  try {
    assertAcceptableSteps(steps);
  } catch (cause) {
    throw new Error(`${where}.steps: ${cause instanceof Error ? cause.message : String(cause)}`, {
      cause,
    });
  }
  const guidanceScale = readNumber(
    record,
    "guidanceScale",
    where,
    Number.isFinite,
    "not a finite number",
  );
  const resolution = parseResolutionEntry(record["resolution"], `${where}.resolution`);
  if (!Object.hasOwn(record, "negativePrompt")) {
    return { steps, guidanceScale, resolution };
  }
  const negativePrompt = record["negativePrompt"];
  if (typeof negativePrompt !== "string") {
    throw new Error(`${where}.negativePrompt: not a string`);
  }
  return { steps, guidanceScale, resolution, negativePrompt };
};

/**
 * Checks the sampler type spelling, **showing expected and actual side by side** (the shared entry
 * of the manifest side and the generation request side).
 *
 * MUST: the single source of truth for the vocabulary is {@link SAMPLER_TYPES} — if a
 * misspelling silently fell back to the default, the run would disagree with the declared or
 * requested update rule, unnoticed. `where` names the origin of the spelling (`scheduler.type` /
 * the request's `sampler`).
 */
export const assertAnimaSamplerType = (value: unknown, where: string): SamplerType => {
  const accepted =
    typeof value === "string" ? SAMPLER_TYPES.find((candidate) => candidate === value) : undefined;
  if (accepted === undefined) {
    throw new Error(
      `${where}: expected ${SAMPLER_TYPES.map((name) => `'${name}'`).join(" / ")}` +
        ` (actual ${typeof value === "string" ? `'${value}'` : String(value)})`,
    );
  }
  return accepted;
};

/**
 * Reads `scheduler.type` (default when the field is absent — {@link assertAnimaSamplerType} rejects
 * unknown values).
 */
const parseSamplerType = (record: Record<string, unknown>, where: string): SamplerType =>
  Object.hasOwn(record, "type")
    ? assertAnimaSamplerType(record["type"], `${where}.type`)
    : DEFAULT_SAMPLER_TYPE;

const parseScheduler = (raw: unknown): AnimaScheduler => {
  const where = "pipelineConfig.scheduler";
  const record = readRecord(raw, where);
  assertAllowedKeys(record, SCHEDULER_KEYS, where);
  return {
    type: parseSamplerType(record, where),
    shift: readNumber(
      record,
      "shift",
      where,
      (value) => Number.isFinite(value) && value > 0,
      "not a positive finite number",
    ),
    numTrainTimesteps: readNumber(
      record,
      "numTrainTimesteps",
      where,
      (value) => Number.isInteger(value) && value > 0,
      "not a positive integer",
    ),
  };
};

/** Checks and reads the manifest's `pipelineConfig` (the raw value the hub passed through). */
export const parseAnimaPipelineConfig = (
  raw: Readonly<Record<string, unknown>>,
): AnimaPipelineConfig => {
  assertAllowedKeys(raw, ROOT_KEYS, "pipelineConfig");
  return {
    scheduler: parseScheduler(Object.hasOwn(raw, "scheduler") ? raw["scheduler"] : undefined),
    defaults: parseDefaults(Object.hasOwn(raw, "defaults") ? raw["defaults"] : undefined),
  };
};
