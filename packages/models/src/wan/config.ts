/**
 * Schema validation of Wan2.1's `pipelineConfig` (ADR 0038 §1 — each pipeline implementation owns
 * and validates its schema).
 *
 * The hub passes `pipelineConfig` through (it only strips forbidden keys and checks the size
 * limit). **This module is therefore the source of truth for the shape**, and the recipe-side test
 * matches its field names against the writer (`WAN_PIPELINE_CONFIG` in the recipe
 * `tools/export-recipes/wan/distribution.py`).
 *
 * Only the generation defaults are declared (ADR 0118 decision 5 — the product defaults start in the
 * manifest's `pipelineConfig.scheduler`, and the visual A/B decides whether to change them):
 *
 * - `scheduler.shift` — the flow matching shift (reference setting — 3.0 for Wan2.1, 5.0 for Wan2.2)
 * - `defaults.steps` / `defaults.guidance` — the denoise step count and CFG strength (reference
 *   setting 50 / 5.0)
 *
 * UniPC's structure (order, bh2, `num_train_timesteps`) has no field in the declaration — the port
 * implements exactly one branch (`WAN_UNIPC_CONFIG` in `scheduler.ts`), and making it declarable
 * would let a distribution pick an unverified combination.
 *
 * MUST: unknown keys fail loudly (if a misspelling silently fell back to the default, the default the
 * distributor intended and the run would disagree unnoticed). The value ranges are the same as the
 * generation request checks (`planWanGeneration` in `family.ts`), and a declaration outside them
 * fails with a plain `Error` as an **asset inconsistency** (not an input error — ADR 0107
 * decision 2). This also checks that the default steps × shift pair yields a UniPC σ sequence —
 * otherwise a `generate` whose request passes nothing would fail in `planWanGeneration` as an input
 * error (`ModelInputError`).
 * MUST: maps are read only through `Object.hasOwn` (cross-cutting invariant).
 */

import type { HubRepoRef } from "@karume/hub";
import { assertAllowedKeys, readNumber, readRecord } from "../config/readers.ts";
import { WAN_UNIPC_CONFIG, wanUniPcSchedule } from "./scheduler.ts";

/** The `pipeline` contract name and the major this implementation accepts (ADR 0038 §1). */
export const WAN_PIPELINE_NAME = "wan";
export const WAN_PIPELINE_MAJOR = 1;

/**
 * The `pipeline` contract name and major of Wan2.2 TI2V (`WanTi2vPipeline` — ADR 0121 decision 10).
 * The `pipelineConfig` schema is the same single one as Wan2.1 ({@link parseWanPipelineConfig}).
 */
export const WAN_TI2V_PIPELINE_NAME = "wan-ti2v";
export const WAN_TI2V_PIPELINE_MAJOR = 1;

/**
 * The Wan family's **table of public distribution repositories** (ADR 0092 — one table per family,
 * **no default entry**). The values are **the sources this package version verified** (pinned
 * commit SHAs — ADR 0073).
 *
 * A key is the basename of the HF repository name without `karume-` (`"karume-" + key` gives back
 * the basename — the check in `tests/sources_test.ts` holds this invariant). Generations are separate
 * repositories (ADR 0092 decision 1):
 *
 * - `"wan2.2"` = `hdae/karume-wan2.2` (Wan2.2 TI2V 5B — read by `WanTi2vPipeline`; umT5's
 *   text_encoder is a cross-repository reference to `karume-umt5-xxl`)
 *
 * Wan2.1 (`karume-wan2.1`) is not published on HF (ADR 0118 addendum (2026-10-15)), so it **has no
 * key** — a key without a public repository can only give a 404 (the reason of ADR 0073 decision 1).
 * Pass a local distribution to `WanPipeline`.
 *
 * Pass it **as an opt-in to follow the package version automatically** — to pin reproducibility
 * yourself, write your own `{ repo, revision }` instead of this table (`fromPretrained` has no
 * default).
 *
 * MUST: pin the revision to a commit SHA — branches and tags can be moved on the distribution side,
 * so the bytes this published package reads would change silently at the network's convenience (an
 * unrecoverable accident). A SHA removes the revision-resolution request itself, so an offline start
 * with a full cache works at the same time (ADR 0038). A user who needs to follow main chooses
 * `{ ...WAN_SOURCES["wan2.2"], revision: "main" }` explicitly.
 */
// NOTE: the revision is updated to the SHA of main after the upload, in §3 of the release runbook
// (docs/release-runbook.md) (inherits the maintenance duty of ADR 0073 decision 3 — by hand + the
// runbook check).
export const WAN_SOURCES = {
  "wan2.2": {
    repo: "hdae/karume-wan2.2",
    revision: "791c3e92dedeb7988f6adc739a5139fbfd876823",
  },
} as const satisfies Record<string, HubRepoRef>;

const ROOT_KEYS: readonly string[] = ["scheduler", "defaults"];
const SCHEDULER_KEYS: readonly string[] = ["shift"];
const DEFAULTS_KEYS: readonly string[] = ["steps", "guidance"];

/** The distributor's recommended defaults (they fill the fields `generate` leaves unset). */
export type WanPipelineConfig = {
  readonly scheduler: { readonly shift: number };
  readonly defaults: { readonly steps: number; readonly guidance: number };
};

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

/** Checks and reads the manifest's `pipelineConfig` (the raw value the hub passed through). */
export const parseWanPipelineConfig = (
  raw: Readonly<Record<string, unknown>>,
): WanPipelineConfig => {
  assertAllowedKeys(raw, ROOT_KEYS, "pipelineConfig");
  const scheduler = readSection(raw, "scheduler", SCHEDULER_KEYS);
  const defaults = readSection(raw, "defaults", DEFAULTS_KEYS);
  const shift = readNumber(
    scheduler,
    "shift",
    "pipelineConfig.scheduler",
    (value) => Number.isFinite(value) && value > 0,
    "正の有限数でない",
  );
  const steps = readNumber(
    defaults,
    "steps",
    "pipelineConfig.defaults",
    // The upper limit is `num_train_timesteps`, as in the request check (the MUST of
    // `wanUniPcSchedule` in `scheduler.ts`).
    (value) => Number.isInteger(value) && value >= 1 && value <= WAN_UNIPC_CONFIG.numTrainTimesteps,
    `1 以上 ${WAN_UNIPC_CONFIG.numTrainTimesteps} 以下の整数でない`,
  );
  const guidance = readNumber(
    defaults,
    "guidance",
    "pipelineConfig.defaults",
    // CFG is applied in f32 (as in the request check — finite in f64 but Infinity in f32 makes the
    // combination NaN).
    (value) => Number.isFinite(Math.fround(value)) && value >= 1,
    "1 以上で f32 に収まる有限の数でない",
  );
  try {
    wanUniPcSchedule(steps, shift, WAN_UNIPC_CONFIG.numTrainTimesteps);
  } catch (error) {
    const reason = error instanceof Error ? error.message : String(error);
    throw new Error(
      `pipelineConfig: 既定の defaults.steps ${steps} と scheduler.shift ${shift} の組では UniPC の ` +
        `σ 列が組めない（${reason}）`,
      { cause: error },
    );
  }
  return { scheduler: { shift }, defaults: { steps, guidance } };
};
