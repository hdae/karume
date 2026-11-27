/**
 * Batches: a JSON file that lists steps — an entry id and parameter values each — which the shell
 * and the Node command run in order. Pure: the text of a file and the registered entries in, the
 * resolved batch (per step the entry and its typed values) or every problem out, before anything
 * runs. A parameter value goes through {@link parseParameters}, the parser of the URL and the
 * command line, in its string form ({@link parameterText}).
 *
 * SPEC: docs/spec/workbench.md#batches
 */

import {
  ParameterError,
  parameterText,
  type ParameterValue,
  type ParameterValues,
  parseParameters,
} from "@karume/sweep-app-support/parameters";
import { ENTRY_ID } from "@karume/sweep-app-support/record";

import type { Entry } from "./entry.ts";

/** The format of a batch file; a file says it in its `format`. */
export const BATCH_FORMAT = "karume-workbench-batch/1";

/** Where the tracked batches are, relative to the repository root. */
export const TRACKED_BATCHES_DIRECTORY = "apps/workbench/batches";

/** Where the untracked batches are, relative to the repository root (outside git: `inputs/` is ignored). */
export const UNTRACKED_BATCHES_DIRECTORY = "inputs/workbench/batches";

/** A batch file as a source lists it: its id (the file name without `.json`) and its text, or why the text is not there. */
export type BatchFile =
  | { readonly id: string; readonly text: string }
  | { readonly id: string; readonly problem: string };

export type ResolvedStep = {
  /** The step's 1-based number in the file. */
  readonly number: number;
  readonly entry: Entry;
  /** The parameters the file wrote, in the string form `parseParameters` read, in the file's order. */
  readonly given: Readonly<Record<string, string>>;
  /** The typed values the step runs with: the file's, and the defaults for the rest. */
  readonly parameters: ParameterValues;
};

/** What a batch file comes to: its steps, or every problem it has. */
export type BatchResolution = {
  /** The file name without `.json`. */
  readonly id: string;
  /** The file's title; its id when the file gives no usable title. */
  readonly title: string;
  readonly description?: string;
} & (
  | { readonly kind: "resolved"; readonly steps: readonly ResolvedStep[] }
  | {
      readonly kind: "problems";
      /** Each problem of a step names it by number (`step 3 (session-ab): Parameter rounds: …`). */
      readonly problems: readonly string[];
      /** The ids that steps name and no registered entry has. */
      readonly unknownEntries: readonly string[];
    }
);

export type ResolvedBatch = Extract<BatchResolution, { readonly kind: "resolved" }>;

type BatchProblems = Extract<BatchResolution, { readonly kind: "problems" }>;

const FILE_KEYS: readonly string[] = ["format", "title", "description", "steps"];
const STEP_KEYS: readonly string[] = ["id", "parameters"];

const isObject = (value: unknown): value is Readonly<Record<string, unknown>> =>
  typeof value === "object" && value !== null && !Array.isArray(value);

const isList = (value: unknown): value is readonly unknown[] => Array.isArray(value);

/** A JSON value a step may give a parameter: a boolean, a number, a string or a list of strings. */
const parameterValueOf = (value: unknown): ParameterValue | undefined => {
  if (typeof value === "boolean" || typeof value === "number" || typeof value === "string") {
    return value;
  }
  if (isList(value) && value.every((item): item is string => typeof item === "string")) {
    return value;
  }
  return undefined;
};

const unknownKeys = (value: Readonly<Record<string, unknown>>, known: readonly string[]) =>
  Object.keys(value).filter((key) => !known.includes(key));

const idProblems = (id: string): readonly string[] =>
  ENTRY_ID.test(id)
    ? []
    : [
        `the file name ${id}.json is not kebab-case (lower-case letters and digits, words joined by hyphens)`,
      ];

type Named = { readonly title?: string; readonly description?: string };

const problemsOnly = (
  id: string,
  problems: readonly string[],
  named: Named = {},
  unknownEntries: readonly string[] = [],
): BatchProblems => ({
  id,
  title: named.title ?? id,
  ...(named.description === undefined ? {} : { description: named.description }),
  kind: "problems",
  problems,
  unknownEntries,
});

type StepResolution = {
  readonly step?: ResolvedStep;
  readonly problems: readonly string[];
  readonly unknownEntry?: string;
};

const resolveStep = (value: unknown, number: number, entries: readonly Entry[]): StepResolution => {
  if (!isObject(value)) {
    return { problems: [`step ${number}: a step is an object { "id": …, "parameters": … }`] };
  }
  const id = value["id"];
  const label = typeof id === "string" ? `step ${number} (${id})` : `step ${number}`;
  const problems = unknownKeys(value, STEP_KEYS).map(
    (key) => `${label}: the key ${JSON.stringify(key)} is not part of a step`,
  );
  let entry: Entry | undefined;
  let unknownEntry: string | undefined;
  if (id === undefined) problems.push(`${label}: id is missing`);
  else if (typeof id !== "string") problems.push(`${label}: id is not a string`);
  else {
    entry = entries.find((candidate) => candidate.id === id);
    if (entry === undefined) {
      problems.push(`${label}: no registered entry has this id`);
      unknownEntry = id;
    }
  }
  // No `parameters`: every parameter takes its default.
  const written = value["parameters"];
  const texts: [string, string][] = [];
  if (isObject(written)) {
    for (const [name, raw] of Object.entries(written)) {
      const parsed = parameterValueOf(raw);
      if (parsed === undefined) {
        problems.push(
          `${label}: parameter ${name} is not a boolean, a number, a string or a list of strings`,
        );
      } else {
        texts.push([name, parameterText(parsed)]);
      }
    }
  } else if (written !== undefined) {
    problems.push(`${label}: parameters is not an object`);
  }
  // `fromEntries` defines own keys, so a name such as `__proto__` reaches the parser as written.
  const given: Readonly<Record<string, string>> = Object.fromEntries(texts);
  if (entry !== undefined) {
    // One name at a time, so every value that does not fit is reported, not only the first.
    for (const [name, text] of texts) {
      try {
        parseParameters(entry.parameters, Object.fromEntries([[name, text]]));
      } catch (error) {
        if (!(error instanceof ParameterError)) throw error;
        problems.push(`${label}: ${error.message}`);
      }
    }
  }
  return {
    ...(entry === undefined || problems.length > 0
      ? {}
      : { step: { number, entry, given, parameters: parseParameters(entry.parameters, given) } }),
    problems,
    ...(unknownEntry === undefined ? {} : { unknownEntry }),
  };
};

/**
 * The batch of the file `<id>.json` whose text is `text`, against `entries`. Every problem is
 * collected — of the file name, the JSON, the shape and each step's entry and values — and a batch
 * with any problem resolves to the problems alone.
 */
export const resolveBatch = (
  id: string,
  text: string,
  entries: readonly Entry[],
): BatchResolution => {
  const problems = [...idProblems(id)];
  let value: unknown;
  try {
    value = JSON.parse(text);
  } catch (error) {
    const reason = error instanceof Error ? error.message : String(error);
    return problemsOnly(id, [...problems, `the file is not JSON (${reason})`]);
  }
  if (!isObject(value)) return problemsOnly(id, [...problems, "the file is not a JSON object"]);
  problems.push(
    ...unknownKeys(value, FILE_KEYS).map(
      (key) => `the key ${JSON.stringify(key)} is not part of the format`,
    ),
  );

  const format = value["format"];
  if (format === undefined) {
    problems.push(`format is missing (it is ${JSON.stringify(BATCH_FORMAT)})`);
  } else if (format !== BATCH_FORMAT) {
    problems.push(`format is ${JSON.stringify(format)}, not ${JSON.stringify(BATCH_FORMAT)}`);
  }

  const title = value["title"];
  let named: Named = {};
  if (title === undefined) problems.push("title is missing");
  else if (typeof title !== "string") problems.push("title is not a string");
  else if (title.trim() === "") problems.push("title is empty");
  else named = { title };

  const description = value["description"];
  if (typeof description === "string") named = { ...named, description };
  else if (description !== undefined) problems.push("description is not a string");

  const list = value["steps"];
  const steps: ResolvedStep[] = [];
  const unknownEntries: string[] = [];
  if (list === undefined) problems.push("steps is missing");
  else if (!isList(list)) problems.push("steps is not a list");
  else if (list.length === 0) problems.push("steps is empty: a batch has at least one step");
  else {
    for (const [index, item] of list.entries()) {
      const resolved = resolveStep(item, index + 1, entries);
      problems.push(...resolved.problems);
      if (resolved.step !== undefined) steps.push(resolved.step);
      if (resolved.unknownEntry !== undefined) unknownEntries.push(resolved.unknownEntry);
    }
  }

  if (problems.length > 0) return problemsOnly(id, problems, named, unknownEntries);
  return {
    id,
    title: named.title ?? id,
    ...(named.description === undefined ? {} : { description: named.description }),
    kind: "resolved",
    steps,
  };
};

/** Where a listed batch comes from. */
export type BatchSource = "tracked" | "untracked";

export type ListedBatch = BatchResolution & { readonly source: BatchSource };

/** The batch of a listed file: its text resolved, or the problem the source gave in place of it. */
const resolveFile = (file: BatchFile, entries: readonly Entry[]): BatchResolution =>
  "text" in file
    ? resolveBatch(file.id, file.text, entries)
    : problemsOnly(file.id, [...idProblems(file.id), file.problem]);

/** `resolution` with `problem` first among its problems (a resolved batch turns into its problem). */
const withProblem = (resolution: BatchResolution, problem: string): BatchProblems =>
  problemsOnly(
    resolution.id,
    [problem, ...(resolution.kind === "problems" ? resolution.problems : [])],
    resolution,
    resolution.kind === "problems" ? resolution.unknownEntries : [],
  );

/**
 * The batches the shell lists: the tracked ones, then the untracked ones, each resolved against
 * `entries` (a batch with problems is listed with them). An untracked batch with a tracked batch's
 * id is not runnable. A production build (`production`) lists only the tracked batches whose every
 * step names one of `entries` — the entries that build holds — and no untracked one.
 */
export const listBatches = (options: {
  readonly tracked: readonly BatchFile[];
  readonly untracked: readonly BatchFile[];
  readonly entries: readonly Entry[];
  readonly production: boolean;
}): readonly ListedBatch[] => {
  const { entries } = options;
  const tracked: ListedBatch[] = options.tracked.map((file) => ({
    ...resolveFile(file, entries),
    source: "tracked",
  }));
  if (options.production) {
    return tracked.filter(
      (batch) => batch.kind === "resolved" || batch.unknownEntries.length === 0,
    );
  }
  const trackedIds = new Set(options.tracked.map(({ id }) => id));
  const untracked: ListedBatch[] = options.untracked.map((file) => {
    const resolution = resolveFile(file, entries);
    return {
      ...(trackedIds.has(file.id)
        ? withProblem(
            resolution,
            `the tracked batch ${TRACKED_BATCHES_DIRECTORY}/${file.id}.json has the same id; rename this file`,
          )
        : resolution),
      source: "untracked",
    };
  });
  return [...tracked, ...untracked];
};
