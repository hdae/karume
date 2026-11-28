/**
 * What the page says before a run, in plain words: the GPU it found and whether a run can start
 * there, and the note on what will run, for how long, what to avoid meanwhile, when a second run
 * matters and what the file holds; and after a run, what of the file will not count and why, or that
 * one more run is needed. Pure,
 * so the words and the start rule are tested without a page.
 *
 * SPEC: docs/spec/benchmark.md#before-a-run
 */

import { type GemvRecordTiming, gemvRecordReading, gemvTimingRefusal } from "@karume/runtime/tune";
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
    "If the GPU's timings hold steady through the run, this one run is enough for a table; if they move, the page says so at the end, and a second run at another time (another day is best) is what makes a table. Either way a second run's file is welcome.",
    "When it ends, the result downloads as one file (JSON). It holds the GPU's vendor, architecture and description, its features and limits, the browser's user agent and the timings — nothing else about you or this computer. Send that file to the maintainer.",
  ];
};

/** The Chrome flag that turns on WebGPU's developer features (a description, unrounded timestamps). */
const DEVELOPER_FEATURES_FLAG = "chrome://flags/#enable-webgpu-developer-features";

const isObject = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null;

/** A GEMV record's data (the merged sweep record), or undefined for another record. */
const gemvData = (record: WorkbenchRecord): Record<string, unknown> | undefined =>
  record.entry.id === GEMV_SWEEP_ENTRY_ID && isObject(record.data) ? record.data : undefined;

/** The GPU timing a GEMV record's data states (`gpuTiming`; an empty one when it states none). */
const gemvTiming = (data: Record<string, unknown>): GemvRecordTiming =>
  isObject(data.gpuTiming) ? data.gpuTiming : {};

/**
 * The class a GEMV record states of itself (`steadiness.class` of its data), or undefined for one
 * without it.
 */
const gemvClass = (data: Record<string, unknown>): string | undefined =>
  isObject(data.steadiness) && typeof data.steadiness.class === "string"
    ? data.steadiness.class
    : undefined;

/**
 * What a GEMV record states of its class, read as the derivation will read the record
 * (`gemvRecordReading`: its timing first), as the unpacking command says it beside the file it
 * wrote (undefined for a record of another entry): timestamps the derivation refuses — no GEMV
 * evidence, whatever its class; steady and in ns — one record is enough; steady otherwise — a second
 * record is needed; unsteady — a second run is needed; unclassified — too few groups; none — a
 * record written before the record stated it.
 */
export const gemvClassText = (record: WorkbenchRecord): string | undefined => {
  if (record.entry.id !== GEMV_SWEEP_ENTRY_ID) return undefined;
  const data = gemvData(record);
  const timing = data === undefined ? {} : gemvTiming(data);
  const found = data === undefined ? undefined : gemvClass(data);
  switch (gemvTimingRefusal(timing)) {
    case "quantized":
      return "its GPU timestamps were rounded: no GEMV evidence, whatever its class";
    case "wall-clock":
      return "timed without GPU timestamps: no GEMV evidence, whatever its class";
    case undefined:
      break;
  }
  switch (found) {
    case "steady":
      return gemvRecordReading({ class: "steady" }, timing) === "one-record"
        ? "steady: one record is enough for a table"
        : "steady, but its timestamps are not in ns: a table needs a second record of the device";
    case "unsteady":
      return "unsteady: a table needs a second record of the device";
    case "unclassified":
      return "too few groups for a class: read as unsteady";
    default:
      return "states no class (written before the records did)";
  }
};

/**
 * The notes after a run, from its records: when a GEMV record's timing is one the GEMV derivation
 * refuses (`gemvTimingRefusal`) — the browser rounded the timestamps, or gave none — that its GEMV
 * part will not count as evidence, why, and for the rounding the Chrome flag that stops it
 * (`deviceView` names the same flag before the run); otherwise, when the GEMV record is not steady
 * by its own statement, that one more run at another time is needed — unsteady, its timings moved;
 * unclassified, it holds too few groups to tell (a short trial).
 */
export const afterRunNotes = (records: readonly WorkbenchRecord[]): string[] => {
  const gemv = records.flatMap((record) => {
    const data = gemvData(record);
    return data === undefined ? [] : [data];
  });
  const refusals = gemv.map((data) => gemvTimingRefusal(gemvTiming(data)));
  const geometry = records.some((record) => record.entry.id === GEOMETRY_SWEEP_ENTRY_ID);
  const stillUsed = geometry ? " The tile geometry part is still used." : "";
  if (refusals.includes("quantized")) {
    return [
      `The GEMV part of this result will not be accepted as evidence: this browser rounded the GPU timestamps to 100 µs, and the GEMV kernels are so short that one such step moves their timings by more than the differences the derivation looks for. In Chrome, turning on ${DEVELOPER_FEATURES_FLAG} stops the rounding; a run with it on gives a GEMV part that counts.${stillUsed}`,
    ];
  }
  if (refusals.includes("wall-clock")) {
    return [
      `The GEMV part of this result will not be accepted as evidence: it was timed without GPU timestamps, and the wall clock adds a delay that moves from pass to pass.${stillUsed}`,
    ];
  }
  const classes = gemv.flatMap((data) => {
    const found = gemvClass(data);
    return found === undefined ? [] : [found];
  });
  if (classes.includes("unsteady")) {
    return [
      "The GPU's timings moved during the GEMV part of this run, more than one run can be read through: please run the page once more at another time (another day is best) and send both files — two such runs together are what makes a table.",
    ];
  }
  if (classes.includes("unclassified")) {
    return [
      "The GEMV part of this run measured too few groups to tell whether the GPU's timings held steady: a full run tells, or a second run of the same cases at another time.",
    ];
  }
  return [];
};
