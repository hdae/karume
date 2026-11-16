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

import { fileURLToPath } from "node:url";
import { dawnFlags, installDawn } from "@karume/test-support/dawn";

import { nodeDistributions } from "./node-distributions.ts";
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
  | { readonly kind: "run"; readonly entry: Entry; readonly parameters: ParameterValues };

/**
 * What the command line asks of `entries`. A command line that names no entry, an argument that is
 * not `--name=value` and an unknown entry throw {@link UsageError}; a parameter the entry does not
 * declare or a value that does not fit throws `ParameterError`. Nothing here touches the GPU.
 */
export const parseCommandLine = (
  args: readonly string[],
  entries: readonly Entry[],
): CommandLine => {
  if (args.includes("--list")) return { kind: "list" };
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

/** How a command that did not only list ended: with a record, or with what it threw. */
export type CommandEnding =
  | { readonly kind: "recorded"; readonly record: WorkbenchRecord }
  | { readonly kind: "threw"; readonly error: unknown };

const isUsageError = (error: unknown): error is UsageError | ParameterError =>
  error instanceof UsageError || error instanceof ParameterError;

/**
 * The exit code of an ending.
 *
 * SPEC: docs/spec/workbench.md#the-node-command
 */
export const exitCodeOf = (ending: CommandEnding): number => {
  if (ending.kind === "threw") return isUsageError(ending.error) ? 2 : 1;
  const { record } = ending;
  return record.outcome === "done" && record.verdict !== "fail" ? 0 : 1;
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
  const lines = [
    `${entry.id}: ${record.outcome}${record.verdict === undefined ? "" : `, verdict ${record.verdict}`} (${record.seconds} s)`,
    ...(record.error === undefined ? [] : [record.error]),
    ...(record.summary ?? []),
    `record: ${path}`,
  ];
  await write(process.stdout, `${lines.join("\n")}\n`);
  return record;
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
    ending = {
      kind: "recorded",
      record: await runOnce(openSession(), line.entry, line.parameters),
    };
  } catch (error) {
    ending = { kind: "threw", error };
    await write(process.stderr, `${isUsageError(error) ? error.message : errorText(error)}\n`);
  }
  return exitCodeOf(ending);
};
