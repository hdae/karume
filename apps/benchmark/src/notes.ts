/**
 * What the page says before a run, in plain words: the GPU it found and whether a run can start
 * there, and the note on what will run, for how long, what to avoid meanwhile, why a second run
 * matters and what the file holds; and after a run, what of the file will not count and why. Pure,
 * so the words and the start rule are tested without a page.
 *
 * SPEC: docs/spec/benchmark.md#before-a-run
 */

import { GEMV_SWEEP_ENTRY_ID } from "@karume/sweep-app-support/gemv-sweep/settings";
import { GEOMETRY_SWEEP_ENTRY_ID } from "@karume/sweep-app-support/geometry-sweep/settings";
import type { WorkbenchRecord } from "@karume/sweep-app-support/record";

import { durationText } from "./estimate.ts";
import type { BenchmarkPlan } from "./plan.ts";

/** The adapter as the page found it. */
export type FoundAdapter = {
  readonly vendor: string;
  readonly architecture: string;
  readonly description: string;
  /** The adapter lists `timestamp-query`. */
  readonly timestamps: boolean;
};

export type DeviceView = {
  /** Label and value, in the order shown. */
  readonly rows: readonly (readonly [string, string])[];
  /** Why a run cannot start here (absent: it can). */
  readonly blocker?: string;
  /** What the person may want to know about the result's use. */
  readonly notes: readonly string[];
};

const shown = (value: string): string => (value === "" ? "(not reported by the browser)" : value);

/**
 * The adapter's facts, and the rule: without `timestamp-query` the sweeps would time with the wall
 * clock, which no table is derived from, so a run does not start.
 */
export const deviceView = (adapter: FoundAdapter): DeviceView => ({
  rows: [
    ["Vendor", shown(adapter.vendor)],
    ["Architecture", shown(adapter.architecture)],
    ["Description", shown(adapter.description)],
    ["GPU timestamps (timestamp-query)", adapter.timestamps ? "available" : "not available"],
  ],
  ...(adapter.timestamps
    ? {}
    : {
        blocker:
          "This browser gives no GPU timestamps (timestamp-query) for this GPU. Without them the sweeps would time with the wall clock, and such a run could not be used as evidence, so it does not start here. Another browser may give them.",
      }),
  notes:
    adapter.description === ""
      ? [
          "This browser reports no description of the GPU. In Chrome, turning on chrome://flags/#enable-webgpu-developer-features makes it report one, so the result can be matched to exactly this GPU model; the flag also stops Chrome from rounding the GPU timestamps to 100 µs, which the GEMV part needs to count as evidence.",
        ]
      : [],
});

/** The cases of a sweep the plan runs, or 0. */
const casesOf = (plan: BenchmarkPlan, sweep: "gemv" | "geometry"): number =>
  plan.cases.filter((runCase) => runCase.sweep === sweep).length;

/** The note before a run of `plan` that is estimated to take `seconds`. */
export const beforeRunNotes = (plan: BenchmarkPlan, seconds: number): string[] => {
  const parts = [
    ...(casesOf(plan, "gemv") > 0
      ? [
          `the GEMV sweep (the matrix–vector products a language model runs for every token, ${casesOf(plan, "gemv")} cases)`,
        ]
      : []),
    ...(casesOf(plan, "geometry") > 0
      ? [
          `the tile geometry sweep (the matrix products of image and video models, ${casesOf(plan, "geometry")} cases)`,
        ]
      : []),
  ];
  return [
    parts.length === 0
      ? "Choose at least one of the two measurements below."
      : `What runs: ${parts.join(" and ")}. Each measures several ways to run the same GPU kernel and keeps the fastest that computes the same result, in a short exploratory form. Nothing is installed and no model is downloaded.`,
    `How long: ${durationText(seconds)} (a first estimate from other machines' runs; the page refines it as it goes). It runs one case at a time with a short pause between cases to let the GPU cool down.`,
    "Meanwhile: keep this tab in front and the computer awake, and use the GPU for nothing else (games, videos, other GPU-heavy pages or apps) — that would change the timings. The page keeps the screen on where the browser allows it.",
    "One run gives candidates; a table for this GPU is made only when a second run agrees. If you can, run it once more at another time (another day is best) and send both files.",
    "When it ends, the result downloads as one file (JSON). It holds the GPU's vendor, architecture and description, its features and limits, the browser's user agent and the timings — nothing else about you or this computer. Send that file to the maintainer.",
  ];
};

/** The Chrome flag that turns on WebGPU's developer features (a description, unrounded timestamps). */
const DEVELOPER_FEATURES_FLAG = "chrome://flags/#enable-webgpu-developer-features";

const isObject = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null;

/**
 * A GEMV record whose data (the merged sweep record) the GEMV derivation refuses for its timing:
 * `gpuTiming.quantized` is anything but `false`, as the derivation reads it.
 */
const gemvTimingQuantized = (record: WorkbenchRecord): boolean =>
  record.entry.id === GEMV_SWEEP_ENTRY_ID &&
  isObject(record.data) &&
  isObject(record.data.gpuTiming) &&
  record.data.gpuTiming.quantized !== false;

/**
 * The notes after a run, from its records: when a GEMV record reports quantized GPU timing — the
 * browser rounded the timestamps, and the GEMV derivation refuses such a record — that its
 * GEMV part will not count as evidence, why, and the Chrome flag that stops the rounding.
 * `deviceView` names the same flag before the run.
 */
export const afterRunNotes = (records: readonly WorkbenchRecord[]): string[] => {
  if (!records.some(gemvTimingQuantized)) return [];
  const geometry = records.some((record) => record.entry.id === GEOMETRY_SWEEP_ENTRY_ID);
  return [
    `The GEMV part of this result will not be accepted as evidence: this browser rounded the GPU timestamps to 100 µs, and the GEMV kernels are so short that one such step moves their timings by more than the differences the derivation looks for. In Chrome, turning on ${DEVELOPER_FEATURES_FLAG} stops the rounding; a run with it on gives a GEMV part that counts.${geometry ? " The tile geometry part is still used." : ""}`,
  ];
};
