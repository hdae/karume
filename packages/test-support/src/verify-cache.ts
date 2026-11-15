/**
 * The verification cache's record of one test file, and the judgement of a prediction (pure logic —
 * the reporter in `./verify-cache-reporter.ts` does the reading and writing).
 *
 * This is the recording stage (ADR 0152 decision 6): every file still runs. After a run the reporter
 * writes what the file's result depended on, as a key of three layers, and asks of the previous record
 * "would its result have been reused?". A reuse that was predicted and then contradicted by the run —
 * another outcome, or other kernels — names an input the key does not cover yet. The cache may start
 * skipping runs only when such contradictions have stopped appearing.
 *
 * | Layer | What it holds |
 * | --- | --- |
 * | environment | the adapter's identity, features and limits; Node; the lockfile and the Vitest config; every `KARUME_*` variable. Not the driver version (kept in the record, outside the key) |
 * | data | each file the test read (content hash, or size and modification time above a size limit), each directory it listed, each `git ls-files` it ran |
 * | code | the test file's static import closure |
 */

import { createHash } from "node:crypto";
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

/** The variables that switch a run into a mode whose result says nothing about the next run. */
const RECORDING_VARIABLES = ["KARUME_REFERENCE", "KARUME_WGSL_SNAPSHOT", "KARUME_SURFACE"];
const UNPINNED_VARIABLES = ["KARUME_NETWORK", "KARUME_SERIES"];

/**
 * Why a result must not be reused (pure function).
 *
 * `unknownCommands` are the commands the file started that the data layer cannot re-evaluate.
 */
export const reasonsNotReusable = (
  outcome: Outcome,
  variables: Readonly<Record<string, string>>,
  unknownCommands: readonly string[],
): string[] => {
  const reasons: string[] = [];
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
 * results file and do not decide an outcome; `git ls-files` lists the files a check walks and is
 * re-run as an input; anything else cannot be re-evaluated.
 */
export const roleOfCommand = (command: readonly string[]): CommandRole => {
  if (command[0] !== "git") return "unknown";
  const args = command.slice(1);
  if (args.includes("ls-files")) return "input";
  if (args[0] === "rev-parse" || args[0] === "status") return "ignored";
  return "unknown";
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
  const names: (keyof Layers)[] = ["environment", "data", "code"];
  const changed = names.filter((name) => previous.layers[name] !== layersNow[name]);
  const reuse = changed.length === 0 && previous.notReusable.length === 0;
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
