/**
 * The verification cache's record of one test file, the decision to reuse it and the judgement of a
 * prediction (pure logic — the reporter in `./verify-cache-reporter.ts` does the reading and writing,
 * the plugin in `./verify-cache-plugin.ts` serves the stub of a reused file).
 *
 * Two modes, chosen by `KARUME_VERIFY_CACHE` (ADR 0152 decision 6):
 *
 * - `record` (the default): every file runs. After a run the reporter writes what the file's result
 *   depended on, as a key of three layers, and asks of the previous record "would its result have been
 *   reused?". A reuse that was predicted and then contradicted by the run — another outcome, or other
 *   kernels — names an input the key does not cover yet.
 * - `reuse`: before the run, a file whose record is reusable and whose key still holds is not run; its
 *   recorded result is reported as reused (one skipped case). The decision is the one the judgement
 *   makes after a `record` run ({@link decideReuse}).
 *
 * | Layer | What it holds |
 * | --- | --- |
 * | environment | the adapter's identity, features and limits; Node; the lockfile and the Vitest config; every `KARUME_*` variable except `KARUME_VERIFY_CACHE`. Not the driver version (kept in the record, outside the key) |
 * | data | each file the test read (content hash, or size and modification time above a size limit), each directory it listed, each `git ls-files` it ran |
 * | code | the test file's static import closure |
 */

import { createHash } from "node:crypto";
import { isAbsolute, relative, resolve, sep } from "node:path";
import type { InputRecord } from "./input-record.ts";
import type { KernelUse } from "./kernel-record.ts";

export const RECORD_SCHEMA = 1;

/** What the test worker hands to the reporter when a test file ends. */
export type WorkerRecord = {
  readonly testPath: string;
  readonly kernels: readonly KernelUse[];
  readonly inputs: InputRecord;
  readonly adapter: AdapterFacts | undefined;
};

export type AdapterFacts = {
  readonly vendor: string;
  readonly architecture: string;
  readonly device: string;
  /** The driver string on Dawn. Recorded; not part of the key. */
  readonly description: string;
  readonly features: readonly string[];
  readonly limits: Readonly<Record<string, number>>;
};

export type Outcome = {
  readonly passed: number;
  readonly failed: number;
  readonly skipped: number;
};

/** One input of the data layer and what identified it when the record was written. */
export type DataInput = {
  readonly kind: "file" | "directory" | "command";
  /** Repository-relative path, or the command line joined by spaces. */
  readonly name: string;
  readonly identity: string;
};

export type Layers = {
  readonly environment: string;
  readonly data: string;
  readonly code: string;
};

export type Prediction = {
  /** The previous record's key still held, so its result would have been reused. */
  readonly reuse: boolean;
  /** The layers that differ from the previous record (empty when `reuse`). */
  readonly changed: readonly (keyof Layers)[];
  /** For a predicted reuse: why the run contradicts it (empty when the prediction held). */
  readonly contradictions: readonly string[];
};

export type CacheRecord = {
  readonly schema: number;
  /** Repository-relative path of the test file. */
  readonly file: string;
  readonly lane: string;
  readonly commit: string | undefined;
  readonly dirty: boolean | undefined;
  readonly durationMs: number;
  readonly outcome: Outcome;
  /** Why this result must not be reused (empty: it may be). */
  readonly notReusable: readonly string[];
  readonly driver: string | undefined;
  readonly layers: Layers;
  readonly key: string;
  /** The environment layer's content (so a miss can say what moved). */
  readonly environment: unknown;
  readonly data: readonly DataInput[];
  /** The import closure: repository-relative module path and the first 16 hex digits of its sha256. */
  readonly code: readonly (readonly [string, string])[];
  readonly kernels: readonly KernelUse[];
  readonly kernelFingerprint: string;
  /** The judgement against the previous record (absent on a file's first record). */
  readonly prediction: Prediction | undefined;
};

export const sha256Of = (text: string | NodeJS.ArrayBufferView): string =>
  createHash("sha256").update(text).digest("hex");

/** A stable hash of any JSON value whose object keys were written in a fixed order by the caller. */
export const hashOf = (value: unknown): string => sha256Of(JSON.stringify(value));

export const keyOf = (layers: Layers): string =>
  hashOf([layers.environment, layers.data, layers.code]);

/** The variable that chooses the mode. */
export const CACHE_MODE_VARIABLE = "KARUME_VERIFY_CACHE";

export type CacheMode = "record" | "reuse";

/** The mode `env` asks for (pure function). Throws on a value that is not a mode. */
export const cacheModeOf = (env: Readonly<Record<string, string | undefined>>): CacheMode => {
  const value = env[CACHE_MODE_VARIABLE];
  if (value === undefined || value === "record") return "record";
  if (value === "reuse") return "reuse";
  throw new Error(
    `${CACHE_MODE_VARIABLE}=${JSON.stringify(value)} is not a mode of the verification cache: ` +
      `unset it or set "record" (every file runs) or "reuse" (a file whose record still holds is not run).`,
  );
};

/**
 * The `KARUME_*` variables that enter the environment layer, sorted by name (pure function).
 *
 * The mode variable is left out: it decides whether a file runs, not what its result is, so the
 * records a `record` run writes are the ones a `reuse` run reads.
 */
export const keyVariablesOf = (
  env: Readonly<Record<string, string | undefined>>,
): Record<string, string> => {
  const variables: Record<string, string> = {};
  for (const name of Object.keys(env).sort()) {
    const value = env[name];
    if (name.startsWith("KARUME_") && name !== CACHE_MODE_VARIABLE && value !== undefined) {
      variables[name] = value;
    }
  }
  return variables;
};

/** The variables that switch a run into a mode whose result says nothing about the next run. */
const RECORDING_VARIABLES = ["KARUME_REFERENCE", "KARUME_WGSL_SNAPSHOT", "KARUME_SURFACE"];
const UNPINNED_VARIABLES = ["KARUME_NETWORK", "KARUME_SERIES"];

/**
 * Why a result must not be reused (pure function).
 *
 * `unknownCommands` are the commands the file started that the data layer cannot re-evaluate.
 * `testNamePattern` is the run's test name filter (`vp test -t <pattern>`), when it has one: only the
 * cases it matches ran, so the record says nothing about the others.
 */
export const reasonsNotReusable = (
  outcome: Outcome,
  variables: Readonly<Record<string, string>>,
  unknownCommands: readonly string[],
  testNamePattern?: string,
): string[] => {
  const reasons: string[] = [];
  if (testNamePattern !== undefined) {
    reasons.push(`a test name filter was set (${testNamePattern}): only the cases it matches ran`);
  }
  if (outcome.failed > 0) reasons.push(`${outcome.failed} failed`);
  if (outcome.passed === 0) reasons.push("no case ran");
  for (const name of RECORDING_VARIABLES) {
    if (variables[name] !== undefined) reasons.push(`${name} is set (a recording run)`);
  }
  for (const name of UNPINNED_VARIABLES) {
    if (variables[name] !== undefined) reasons.push(`${name} is set (inputs without a pin)`);
  }
  for (const name of Object.keys(variables)) {
    if (name.startsWith("KARUME_ALLOW_NO_")) reasons.push(`${name} is set (an opt-out)`);
  }
  for (const command of unknownCommands) reasons.push(`starts a process: ${command}`);
  return reasons;
};

/** How a command the test started enters the data layer. */
export type CommandRole = "ignored" | "input" | "unknown";

/**
 * Sorts a command (pure function): `git rev-parse` / `git status` only describe the checkout for a
 * results file, and `nvidia-smi --query-gpu=…` samples the GPU's memory, temperature and clock state
 * while a test runs, for its printed notes; none of them decides an outcome. `git ls-files` lists the
 * files a check walks and is re-run as an input; anything else cannot be re-evaluated.
 */
export const roleOfCommand = (command: readonly string[]): CommandRole => {
  const args = command.slice(1);
  if (command[0] === "nvidia-smi") {
    return args.some((arg) => arg.startsWith("--query-gpu=")) ? "ignored" : "unknown";
  }
  if (command[0] !== "git") return "unknown";
  if (args.includes("ls-files")) return "input";
  if (args[0] === "rev-parse" || args[0] === "status") return "ignored";
  return "unknown";
};

/** Whether a record's result may stand for a run now. */
export type ReuseDecision = {
  readonly reuse: boolean;
  /** The layers that differ from the record's. */
  readonly changed: readonly (keyof Layers)[];
  /** Why the result may not stand (empty when `reuse`). */
  readonly why: readonly string[];
};

const LAYER_NAMES: readonly (keyof Layers)[] = ["environment", "data", "code"];

/**
 * The rule of reuse (pure function): the record is reusable by its own terms and every layer computed
 * now over the inputs the record lists equals the record's. The decision before a `reuse` run
 * ({@link decideBeforeRun}) and the judgement after a `record` run ({@link judgePrediction}) both
 * apply it.
 */
export const decideReuse = (
  record: Pick<CacheRecord, "layers" | "notReusable">,
  layersNow: Layers,
): ReuseDecision => {
  const changed = LAYER_NAMES.filter((name) => record.layers[name] !== layersNow[name]);
  const why = [
    ...record.notReusable.map((reason) => `the record is not reusable: ${reason}`),
    ...changed.map((name) => `the ${name} layer changed`),
  ];
  return { reuse: why.length === 0, changed, why };
};

/**
 * The decision before a file runs (pure function): {@link decideReuse} on the file's record under the
 * current environment, when there is one this harness can read. `layersOver` computes the layers now
 * over the inputs a record lists; it is called only for a readable record.
 */
export const decideBeforeRun = <R extends Pick<CacheRecord, "schema" | "layers" | "notReusable">>(
  record: R | undefined,
  layersOver: (record: R) => Layers,
): ReuseDecision => {
  if (record === undefined) {
    return { reuse: false, changed: [], why: ["no record under this environment"] };
  }
  if (record.schema !== RECORD_SCHEMA) {
    return {
      reuse: false,
      changed: [],
      why: [`the record's schema is ${record.schema}, this harness writes ${RECORD_SCHEMA}`],
    };
  }
  return decideReuse(record, layersOver(record));
};

/**
 * Judges the previous record against this run (pure function).
 *
 * `layersNow` are the layers computed now **over the inputs the previous record lists** — what a cache
 * could compute before running the file. `current` is what the run then showed.
 */
export const judgePrediction = (
  previous: Pick<CacheRecord, "layers" | "notReusable" | "outcome" | "kernelFingerprint">,
  layersNow: Layers,
  current: Pick<CacheRecord, "notReusable" | "outcome" | "kernelFingerprint">,
): Prediction => {
  const { reuse, changed } = decideReuse(previous, layersNow);
  if (!reuse) return { reuse, changed, contradictions: [] };
  const contradictions: string[] = [];
  const { outcome } = current;
  if (
    outcome.passed !== previous.outcome.passed ||
    outcome.failed !== previous.outcome.failed ||
    outcome.skipped !== previous.outcome.skipped
  ) {
    contradictions.push(
      `the outcome is ${outcome.passed} passed / ${outcome.failed} failed / ${outcome.skipped} skipped, ` +
        `the record has ${previous.outcome.passed} / ${previous.outcome.failed} / ${previous.outcome.skipped}`,
    );
  }
  if (current.kernelFingerprint !== previous.kernelFingerprint) {
    contradictions.push("the kernels dispatched differ from the record's");
  }
  return { reuse, changed, contradictions };
};

export type RunSummary = {
  readonly files: number;
  /** Files with a previous record. */
  readonly judged: number;
  readonly reuse: number;
  /** Seconds the predicted reuses took to run. */
  readonly reuseSeconds: number;
  readonly totalSeconds: number;
  /** Predicted reuses the run contradicted, with their reasons. */
  readonly contradicted: readonly { readonly file: string; readonly why: readonly string[] }[];
};

/** Totals of one run's records (pure function). */
export const summarize = (records: readonly CacheRecord[]): RunSummary => {
  const judged = records.filter((record) => record.prediction !== undefined);
  const reused = judged.filter((record) => record.prediction?.reuse === true);
  const seconds = (list: readonly CacheRecord[]): number =>
    Math.round(list.reduce((sum, record) => sum + record.durationMs, 0) / 100) / 10;
  return {
    files: records.length,
    judged: judged.length,
    reuse: reused.length,
    reuseSeconds: seconds(reused),
    totalSeconds: seconds(records),
    contradicted: reused
      .filter((record) => (record.prediction?.contradictions.length ?? 0) > 0)
      .map((record) => ({ file: record.file, why: record.prediction?.contradictions ?? [] })),
  };
};

/** The title of the one skipped case a reused file shows (pure function). */
export const reusedTitleOf = (
  record: Pick<CacheRecord, "commit" | "dirty" | "outcome" | "durationMs">,
): string => {
  const commit =
    record.commit === undefined ? "an unknown commit" : `commit ${record.commit.slice(0, 12)}`;
  const dirty = record.dirty === true ? " with uncommitted changes" : "";
  const { passed, skipped } = record.outcome;
  const skippedText = skipped === 0 ? "" : ` (${skipped} skipped)`;
  return (
    `reused (KARUME_VERIFY_CACHE=reuse): the record of ${commit}${dirty} passed ${passed} ` +
    `case${passed === 1 ? "" : "s"}${skippedText} in ${Math.round(record.durationMs / 100) / 10} s`
  );
};

/**
 * The module served in place of a reused test file (pure function): one skipped case whose title says
 * the file was reused, from which commit's record, and how many cases that record passed.
 */
export const stubModuleText = (
  record: Pick<CacheRecord, "commit" | "dirty" | "outcome" | "durationMs">,
): string =>
  [
    "// Served by the verification cache in place of a test file whose recorded result is reused.",
    'import { it } from "vite-plus/test";',
    `it.skip(${JSON.stringify(reusedTitleOf(record))}, () => {});`,
    "",
  ].join("\n");

/**
 * Whether a command-line filter names the test file `file` (absolute) — asking for a file is asking to
 * run it (pure function). It does when the filter, as a path from `cwd`, is the file, or when the
 * file's path from `root` ends with the filter at a directory boundary (`verify_cache_test.ts`,
 * `tests/verify_cache_test.ts`). A directory or a fragment of a name selects files without naming them.
 */
export const filterNamesFile = (
  filter: string,
  file: string,
  cwd: string,
  root: string,
): boolean => {
  if (isAbsolute(filter) ? filter === file : resolve(cwd, filter) === file) return true;
  const path = relative(root, file).split(sep).join("/");
  const written = filter.split(sep).join("/").replace(/^\.\//, "");
  return written !== "" && (path === written || path.endsWith(`/${written}`));
};

export type ReuseSummary = {
  /** Files not run because their records were reused. */
  readonly reused: number;
  /** Seconds the reused records took to run. */
  readonly reusedSeconds: number;
  /** Files that ran. */
  readonly ran: number;
  readonly reusedFiles: readonly {
    readonly file: string;
    readonly commit: string | undefined;
    readonly seconds: number;
  }[];
};

/** Totals of a `reuse` run (pure function). */
export const summarizeReuse = (
  reused: readonly Pick<CacheRecord, "file" | "commit" | "durationMs">[],
  ran: number,
): ReuseSummary => {
  const seconds = (ms: number): number => Math.round(ms / 100) / 10;
  return {
    reused: reused.length,
    reusedSeconds: seconds(reused.reduce((sum, record) => sum + record.durationMs, 0)),
    ran,
    reusedFiles: reused.map((record) => ({
      file: record.file,
      commit: record.commit,
      seconds: seconds(record.durationMs),
    })),
  };
};
