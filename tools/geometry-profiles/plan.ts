/**
 * The generator's plan of GEMV measurements (`--gemv-plan`): what the GEMV sweep records at hand
 * lack for `deriveLinearGemvChoices` to be complete under the current runtime, by the rule the
 * derivation will take for each decision (`planGemvMeasurements` of `@karume/runtime/tune`). What
 * records of any class fill — per case and row count, the candidates of decisions of the rule of two
 * records that fewer than two records could be evidence for — is written as a workbench batch
 * (`karume-workbench-batch/1`) whose steps measure exactly that, twice. What only a steady record
 * fills — the candidates of decisions of the rule of one steady record that no steady record could
 * be evidence for — is listed apart and never a step: a step's record of a few groups has no class,
 * so it cannot close that gap; a run of the `full` or `explore` set that is steady does. With no
 * record at hand the batch is the whole measurement (the `full` set, twice). Pure over the records;
 * the command writes the file.
 *
 * A step is one `gemv-sweep` run of the `full` set narrowed to cases, row counts and candidates
 * that miss the same: the (case, row count) pairs that miss one list of candidates are grouped, and
 * their cases grouped again by the row counts they miss it at. The references (the baseline and the
 * default's choice at `reorder`) are measured with every step by the sweep itself.
 *
 * SPEC: docs/spec/geometry-profiles.md#planning-the-gemv-measurements
 */

import type { GeometryProfile } from "@karume/runtime";
import {
  type GemvMissingMeasurement,
  gemvSweepCases,
  planGemvMeasurements,
} from "@karume/runtime/tune";
import type { LoadedRecord } from "./records.ts";

/** The format of a workbench batch file (the workbench's `BATCH_FORMAT`). */
export const BATCH_FORMAT = "karume-workbench-batch/1";

/** Where the workbench reads untracked batches, repository-relative (its `UNTRACKED_BATCHES_DIRECTORY`). */
export const UNTRACKED_BATCHES_DIRECTORY = "inputs/workbench/batches";

/**
 * How many times the batch measures what is missing: twice. A step narrowed to a few cases makes a
 * record of a few groups, and one of fewer than 20 has no class, so it is evidence only beside a
 * second (the rule of two records) — even on a steady device.
 */
export const PLAN_PASSES = 2;

/** One step of the batch: the workbench's `{ id, parameters }`. */
export type PlanStep = {
  readonly id: "gemv-sweep";
  readonly parameters: {
    readonly candidateSet: "full";
    readonly cases?: readonly string[];
    readonly rows?: readonly string[];
    readonly candidates?: readonly string[];
  };
};

/** The batch file's content. */
export type PlanBatch = {
  readonly format: typeof BATCH_FORMAT;
  readonly title: string;
  readonly description: string;
  readonly steps: readonly PlanStep[];
};

/**
 * The planned measurement: its batch (absent when nothing is missing — a batch has at least one
 * step), the batch's id (the file name without `.json`) and its lines.
 */
export type GemvPlan = {
  readonly batchId: string;
  readonly batch?: PlanBatch;
  /** What the batch measures (records of any class fill it). */
  readonly missing: readonly GemvMissingMeasurement[];
  /** What no batch step fills: a steady record that times the kernels named does. */
  readonly missingSteady: readonly GemvMissingMeasurement[];
  /** What a person reads: what is missing, what is not current, the steps. */
  readonly lines: readonly string[];
};

/** The batch id of a profile's plan. */
export const planBatchId = (profileId: string): string => `gemv-plan-${profileId}`;

/** The steps of one pass over the missing measurements (see the module doc's grouping). */
export const planSteps = (missing: readonly GemvMissingMeasurement[]): PlanStep[] => {
  const order = gemvSweepCases().map(({ id }) => id);
  // Per list of candidates: the row counts each case misses it at, in table order
  const byNames = new Map<string, { names: readonly string[]; rows: Map<string, number[]> }>();
  for (const { caseId, m, candidates } of missing) {
    const key = candidates.join(",");
    const entry = byNames.get(key) ?? { names: candidates, rows: new Map<string, number[]>() };
    entry.rows.set(caseId, [...(entry.rows.get(caseId) ?? []), m]);
    byNames.set(key, entry);
  }
  const steps: PlanStep[] = [];
  for (const { names, rows } of byNames.values()) {
    // The cases that miss the list at the same row counts make one step
    const byRows = new Map<string, { rows: readonly number[]; cases: string[] }>();
    const cases = [...rows.keys()].sort((a, b) => order.indexOf(a) - order.indexOf(b));
    for (const caseId of cases) {
      const counts = [...(rows.get(caseId) ?? [])].sort((a, b) => a - b);
      const key = counts.join(",");
      const entry = byRows.get(key) ?? { rows: counts, cases: [] };
      entry.cases.push(caseId);
      byRows.set(key, entry);
    }
    for (const entry of byRows.values()) {
      steps.push({
        id: "gemv-sweep",
        parameters: {
          candidateSet: "full",
          cases: entry.cases,
          rows: entry.rows.map(String),
          candidates: names,
        },
      });
    }
  }
  return steps;
};

/**
 * The plan for a profile from the GEMV records at hand: the runtime's missing measurements and the
 * batch that measures them {@link PLAN_PASSES} times. `table` is the registered table of the id when
 * there is one (the records must be of its adapter, as for the derivation).
 */
export const gemvPlan = (input: {
  readonly profileId: string;
  readonly records: readonly LoadedRecord[];
  readonly table?: GeometryProfile;
}): GemvPlan => {
  const { profileId, records } = input;
  const batchId = planBatchId(profileId);
  const command = `node tools/geometry-profiles/generate.ts ${profileId} --gemv-plan`;
  if (records.length === 0) {
    const steps = Array.from({ length: PLAN_PASSES }, (): PlanStep => ({
      id: "gemv-sweep",
      parameters: { candidateSet: "full" },
    }));
    return {
      batchId,
      batch: {
        format: BATCH_FORMAT,
        title: `GEMV sweep for ${profileId}: the whole measurement, twice`,
        description: `Written by ${command} with no GEMV sweep record at hand: the full set (M = 1, 4 and 8, every candidate), ${PLAN_PASSES} times — one record is enough only when it is steady, two are needed otherwise.`,
        steps,
      },
      missing: [],
      missingSteady: [],
      lines: ["No GEMV sweep record given: the batch measures everything (the full set), twice."],
    };
  }
  const planning = planGemvMeasurements(
    records.map(({ report, path, sha256 }) => ({ report, path, sha256 })),
    input.table === undefined ? {} : { table: input.table },
  );
  const { missing, missingSteady, stale } = planning;
  const pass = planSteps(missing);
  const count = (list: readonly GemvMissingMeasurement[]): number =>
    list.reduce((sum, { candidates }) => sum + candidates.length, 0);
  const lines = [
    `From ${records.length} GEMV sweep record${records.length === 1 ? "" : "s"}: ${missing.length} of ${planning.groups} (case, row count) groups of the full set miss ${count(missing)} candidate measurements that records of any class fill (decisions of the rule of two records, where no steady record timed the case at M = 1: fewer than two records could be evidence for the candidate — a current group with its row, read against a fastest time of the reference a second record repeats).`,
    ...(missingSteady.length === 0
      ? []
      : [
          `Only a steady record fills ${count(missingSteady)} candidate measurements in ${missingSteady.length} groups: a steady record timed their case at M = 1, so the rule of one steady record decides them and reads steady records alone, and none times these kernels against a usable reference. No step of the batch fills them (a step's record of a few groups has no class): run the full or the explore set on the device until a run is steady:`,
          ...missingSteady.map(
            ({ caseId, m, candidates }) => `  - ${caseId} M=${m}: ${candidates.join(", ")}`,
          ),
        ]),
    `Groups of the records that are not current: ${stale.length}.`,
    ...(pass.length === 0
      ? [
          missingSteady.length === 0
            ? "Nothing is missing: the derivation is complete under the current runtime."
            : "No batch step can fill what is missing: only a steady record does (above).",
        ]
      : [
          `Each pass has ${pass.length} step${pass.length === 1 ? "" : "s"} (the batch runs the pass ${PLAN_PASSES} times):`,
          ...pass.map(
            ({ parameters }, index) =>
              `  ${index + 1}. ${parameters.cases?.length ?? 0} cases at M=${parameters.rows?.join(", ") ?? "?"}: ${parameters.candidates?.length ?? 0} candidates (${parameters.candidates?.join(", ") ?? ""})`,
          ),
        ]),
  ];
  if (pass.length === 0) return { batchId, missing, missingSteady, lines };
  const steps = Array.from({ length: PLAN_PASSES }, () => pass).flat();
  return {
    batchId,
    batch: {
      format: BATCH_FORMAT,
      title: `GEMV sweep for ${profileId}: what the derivation lacks, twice`,
      description: `Written by ${command} from ${records.map(({ path }) => path).join(", ")}: per case and row count of the full set, the candidates of decisions of the rule of two records that fewer than two of those records could be evidence for; each step measures them with the references, and the steps run ${PLAN_PASSES} times (a step's record of a few groups has no class). What only a steady record fills is not in the batch.`,
      steps,
    },
    missing,
    missingSteady,
    lines,
  };
};

/** The batch file's text (two-space JSON and a final newline). */
export const planBatchText = (batch: PlanBatch): string => `${JSON.stringify(batch, null, 2)}\n`;
