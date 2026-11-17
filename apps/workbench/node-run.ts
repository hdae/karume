/**
 * The Node command's body: what `run.ts` runs once Vite's module runner is up. It is loaded through
 * that runner, like the entries it drives, so the host side and the entries share one instance of
 * every module — a hub source the host opens is a hub source to the entry, and an error class the
 * host throws is the class the entry catches. `run.ts` itself only starts the runner.
 *
 * Progress goes to stderr; the verdict, the summary and the record's path go to stdout.
 *
 * SPEC: docs/spec/workbench.md#the-node-command
 */

import { readFile } from "node:fs/promises";
import { basename } from "node:path";
import { fileURLToPath } from "node:url";
import { dawnFlags, installDawn } from "@karume/test-support/dawn";

import { nodeDistributions } from "./node-distributions.ts";
import { nodeInputs } from "./node-inputs.ts";
import { type ResolvedBatch, resolveBatch } from "./src/batch.ts";
import { batchSummary, runBatch, type StepState } from "./src/batch-run.ts";
import { readCheckout } from "./src/checkout.ts";
import { registeredEntries } from "./src/entries.ts";
import { type Entry, ParameterError, type ParameterValues, parseParameters } from "./src/entry.ts";
import { dawnBackend, readEnvironment } from "./src/environment.ts";
import { type RunHost, runEntry } from "./src/execute.ts";
import { errorText, type WorkbenchRecord } from "./src/record.ts";
import { writeRecord } from "./src/results.ts";

const REPOSITORY_ROOT = fileURLToPath(new URL("../../", import.meta.url));

/** A command line the runner cannot act on. */
export class UsageError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "UsageError";
  }
}

export type CommandLine =
  | { readonly kind: "list" }
  | { readonly kind: "run"; readonly entry: Entry; readonly parameters: ParameterValues }
  /** `id` is the file name without `.json`; the file is read and resolved by the command. */
  | { readonly kind: "batch"; readonly path: string; readonly id: string };

const BATCH_FLAG = "--batch";

/** `--batch=<path>` (or a bare `--batch`, which names no path). */
const isBatchArgument = (arg: string): boolean =>
  arg === BATCH_FLAG || arg.startsWith(`${BATCH_FLAG}=`);

/** The batch form: `--batch=<path>` alone, the path's file name `<id>.json`. */
const batchCommandLine = (args: readonly string[]): CommandLine => {
  const [arg] = args;
  if (args.length !== 1 || arg === undefined) {
    throw new UsageError(
      "--batch=<path> runs a batch file alone: no entry id, no --name=value and one --batch",
    );
  }
  const path = arg.slice(`${BATCH_FLAG}=`.length);
  if (path === "") throw new UsageError("Name the batch file: --batch=<path>");
  const name = basename(path);
  if (!name.endsWith(".json")) throw new UsageError(`${path}: a batch file is named <id>.json`);
  return { kind: "batch", path, id: name.slice(0, -".json".length) };
};

/**
 * What the command line asks of `entries`. A command line that names no entry, an argument that is
 * not `--name=value`, an unknown entry, and `--batch` with anything else or with a file name that
 * is not `<id>.json` throw {@link UsageError}; a parameter the entry does not declare or a value
 * that does not fit throws `ParameterError`. Nothing here touches the GPU or the file system.
 */
export const parseCommandLine = (
  args: readonly string[],
  entries: readonly Entry[],
): CommandLine => {
  if (args.includes("--list")) return { kind: "list" };
  if (args.some(isBatchArgument)) return batchCommandLine(args);
  const [entryId, ...rest] = args;
  if (entryId === undefined || entryId.startsWith("--")) {
    throw new UsageError("Name an entry: node apps/workbench/run.ts <entry id> [--name=value …]");
  }
  const given: Record<string, string> = {};
  for (const arg of rest) {
    const match = /^--([^=]+)=(.*)$/s.exec(arg);
    if (match === null) throw new UsageError(`${arg} is not --name=value`);
    given[match[1]] = match[2];
  }
  const entry = entries.find(({ id }) => id === entryId);
  if (entry === undefined) {
    throw new UsageError(
      `No entry ${entryId}; the entries are ${entries.map(({ id }) => id).join(", ")}`,
    );
  }
  return { kind: "run", entry, parameters: parseParameters(entry.parameters, given) };
};

/** How a command that did not only list ended: with a record, a batch run, or with what it threw. */
export type CommandEnding =
  | { readonly kind: "recorded"; readonly record: WorkbenchRecord }
  | { readonly kind: "batch"; readonly steps: readonly StepState<unknown>[] }
  | { readonly kind: "threw"; readonly error: unknown };

const isUsageError = (error: unknown): error is UsageError | ParameterError =>
  error instanceof UsageError || error instanceof ParameterError;

/** A record whose run is done and did not conclude `fail`. */
const succeeded = (record: WorkbenchRecord): boolean =>
  record.outcome === "done" && record.verdict !== "fail";

/**
 * The exit code of an ending. A batch exits 0 only when every step ran and succeeded: a step that
 * ended as `error` or `cancelled`, concluded `fail`, failed around its run or did not run makes it 1.
 *
 * SPEC: docs/spec/workbench.md#the-node-command
 */
export const exitCodeOf = (ending: CommandEnding): number => {
  if (ending.kind === "threw") return isUsageError(ending.error) ? 2 : 1;
  if (ending.kind === "batch") {
    return ending.steps.every((state) => state.kind === "ran" && succeeded(state.run.record))
      ? 0
      : 1;
  }
  return succeeded(ending.record) ? 0 : 1;
};

const write = (stream: NodeJS.WriteStream, text: string): Promise<void> =>
  new Promise((resolve, reject) => {
    stream.write(text, (error) => {
      if (error === null || error === undefined) resolve();
      else reject(error);
    });
  });

/** What a command sets up once, however many runs it makes. */
type NodeSession = {
  readonly host: RunHost<string>;
  /** Aborted by the command's one SIGINT handler. */
  readonly signal: AbortSignal;
};

/** Dawn as `navigator.gpu`, the abort signal and the one SIGINT handler that aborts it. */
const openSession = (): NodeSession => {
  installDawn();
  const gpu = navigator.gpu;
  const controller = new AbortController();
  process.once("SIGINT", () => controller.abort());
  return {
    host: {
      readEnvironment: () =>
        readEnvironment(gpu, {
          host: "node",
          userAgent: navigator.userAgent,
          backend: dawnBackend(dawnFlags(), process.platform),
        }),
      readCheckout: () => readCheckout(REPOSITORY_ROOT),
      gpuOptions: { gpu },
      openDistribution: nodeDistributions(REPOSITORY_ROOT),
      openInput: nodeInputs(REPOSITORY_ROOT),
      // A record that cannot be written fails the command.
      save: (record, text) =>
        writeRecord(
          REPOSITORY_ROOT,
          {
            entryId: record.entry.id,
            host: record.environment.host,
            adapter: record.environment.adapter,
          },
          new TextEncoder().encode(text),
          new Date(),
        ),
    },
    signal: controller.signal,
  };
};

/** What a run prints to stdout: the outcome line, the error, the summary and the record's path. */
const runLines = (entry: Entry, record: WorkbenchRecord, path: string): string[] => [
  `${entry.id}: ${record.outcome}${record.verdict === undefined ? "" : `, verdict ${record.verdict}`} (${record.seconds} s)`,
  ...(record.error === undefined ? [] : [record.error]),
  ...(record.summary ?? []),
  `record: ${path}`,
];

/** One run in `session`: progress to stderr, the outcome, the summary and the path to stdout. */
const runOnce = async (
  session: NodeSession,
  entry: Entry,
  parameters: ParameterValues,
): Promise<WorkbenchRecord> => {
  const { record, saved: path } = await runEntry(session.host, {
    entry,
    parameters,
    signal: session.signal,
    progress: (text) => process.stderr.write(`${text}\n`),
  });
  await write(process.stdout, `${runLines(entry, record, path).join("\n")}\n`);
  return record;
};

/**
 * The batch file of `line`, resolved against `entries`. A file that cannot be read and a batch with
 * problems throw {@link UsageError}, which carries every problem: nothing runs.
 */
const readBatch = async (
  line: { readonly path: string; readonly id: string },
  entries: readonly Entry[],
): Promise<ResolvedBatch> => {
  let text: string;
  try {
    text = await readFile(line.path, "utf8");
  } catch (error) {
    throw new UsageError(
      `Cannot read the batch file ${line.path}: ${error instanceof Error ? error.message : String(error)}`,
    );
  }
  const resolution = resolveBatch(line.id, text, entries);
  if (resolution.kind === "problems") {
    throw new UsageError(
      [
        `${line.path}: ${resolution.problems.length === 1 ? "1 problem" : `${resolution.problems.length} problems`}; nothing ran`,
        ...resolution.problems,
      ].join("\n"),
    );
  }
  return resolution;
};

/**
 * Every step of `batch` in `session`, in file order: each step prints what a run prints, its first
 * line prefixed with the step's position (`[3/9] `), and its progress goes to stderr prefixed the
 * same way; a last line sums the batch run up. SIGINT cancels the running step and the rest do not
 * run.
 */
const runBatchOnce = async (
  session: NodeSession,
  batch: ResolvedBatch,
): Promise<readonly StepState<unknown>[]> => {
  const position = (number: number): string => `[${number}/${batch.steps.length}]`;
  const ran = await runBatch({
    batch,
    selected: batch.steps.map(({ number }) => number),
    signal: session.signal,
    host: session.host,
    onEvent: (event) => {
      const prefix = position(event.step.number);
      if (event.kind === "progress") process.stderr.write(`${prefix} ${event.text}\n`);
      if (event.kind !== "settled") return;
      const { state, step } = event;
      if (state.kind === "ran") {
        const [first, ...rest] = runLines(step.entry, state.run.record, state.run.saved);
        process.stdout.write(`${[`${prefix} ${first ?? ""}`, ...rest].join("\n")}\n`);
      } else if (state.kind === "host-failed") {
        process.stderr.write(
          `${prefix} ${step.entry.id}: the batch stopped: ${errorText(state.error)}\n`,
        );
      }
    },
  });
  // Written last and awaited: each stream is flushed with everything written to it before.
  await write(process.stdout, `${batchSummary(batch.id, ran)}\n`);
  await write(process.stderr, "");
  return ran.steps.map(({ state }) => state);
};

/**
 * Runs the command line and returns its exit code ({@link exitCodeOf}; `--list` is 0). stdout and
 * stderr are flushed when it returns.
 */
export const runCommand = async (args: readonly string[]): Promise<number> => {
  let ending: CommandEnding;
  try {
    const entries = registeredEntries();
    const line = parseCommandLine(args, entries);
    if (line.kind === "list") {
      await write(process.stdout, entries.map((entry) => `${entry.id}\t${entry.title}\n`).join(""));
      return 0;
    }
    if (line.kind === "batch") {
      // The whole file is read and checked before the session (Dawn) is set up.
      const batch = await readBatch(line, entries);
      ending = { kind: "batch", steps: await runBatchOnce(openSession(), batch) };
    } else {
      ending = {
        kind: "recorded",
        record: await runOnce(openSession(), line.entry, line.parameters),
      };
    }
  } catch (error) {
    ending = { kind: "threw", error };
    await write(process.stderr, `${isUsageError(error) ? error.message : errorText(error)}\n`);
  }
  return exitCodeOf(ending);
};
