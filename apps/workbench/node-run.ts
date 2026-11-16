/**
 * The Node command's body: what `run.ts` runs once Vite's module runner is up. It is loaded through
 * that runner, like the entries it drives, so the host side and the entries share one instance of
 * every module — a hub source the host opens is a hub source to the entry, and an error class the
 * host throws is the class the entry catches. `run.ts` itself only starts the runner.
 *
 * Progress goes to stderr; the verdict, the summary and the record's path go to stdout.
 */

import { fileURLToPath } from "node:url";
import { dawnFlags, installDawn } from "@karume/test-support/dawn";

import { nodeDistributions } from "./node-distributions.ts";
import { readCheckout } from "./src/checkout.ts";
import { registeredEntries } from "./src/entries.ts";
import { type Entry, ParameterError, parseParameters } from "./src/entry.ts";
import { dawnBackend, readEnvironment } from "./src/environment.ts";
import { executeEntry } from "./src/execute.ts";
import { errorText, recordText, type WorkbenchRecord } from "./src/record.ts";
import { writeRecord } from "./src/results.ts";

const REPOSITORY_ROOT = fileURLToPath(new URL("../../", import.meta.url));

/** A command line the runner cannot act on. */
class UsageError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "UsageError";
  }
}

type CommandLine =
  | { readonly kind: "list" }
  | {
      readonly kind: "run";
      readonly entryId: string;
      readonly given: Readonly<Record<string, string>>;
    };

const parseCommandLine = (args: readonly string[]): CommandLine => {
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
  return { kind: "run", entryId, given };
};

/** 0 for done with a verdict that is not fail (or none); 1 for fail, error and cancelled. */
const exitCodeOf = (record: WorkbenchRecord): number =>
  record.outcome === "done" && record.verdict !== "fail" ? 0 : 1;

const write = (stream: NodeJS.WriteStream, text: string): Promise<void> =>
  new Promise((resolve, reject) => {
    stream.write(text, (error) => {
      if (error === null || error === undefined) resolve();
      else reject(error);
    });
  });

const runEntry = async (entry: Entry, given: Readonly<Record<string, string>>): Promise<number> => {
  const parameters = parseParameters(entry.parameters, given);
  installDawn();
  const gpu = navigator.gpu;
  const [environment, checkout] = await Promise.all([
    readEnvironment(gpu, {
      host: "node",
      userAgent: navigator.userAgent,
      backend: dawnBackend(dawnFlags(), process.platform),
    }),
    readCheckout(REPOSITORY_ROOT),
  ]);
  const controller = new AbortController();
  process.once("SIGINT", () => controller.abort());
  const record = await executeEntry({
    entry,
    parameters,
    gpuOptions: { gpu },
    signal: controller.signal,
    progress: (text) => process.stderr.write(`${text}\n`),
    openDistribution: nodeDistributions(REPOSITORY_ROOT),
    environment,
    checkout,
  });
  const path = await writeRecord(
    REPOSITORY_ROOT,
    { entryId: entry.id, host: record.environment.host, adapter: record.environment.adapter },
    new TextEncoder().encode(recordText(record)),
    new Date(),
  );
  const lines = [
    `${entry.id}: ${record.outcome}${record.verdict === undefined ? "" : `, verdict ${record.verdict}`} (${record.seconds} s)`,
    ...(record.error === undefined ? [] : [record.error]),
    ...(record.summary ?? []),
    `record: ${path}`,
  ];
  await write(process.stdout, `${lines.join("\n")}\n`);
  return exitCodeOf(record);
};

const command = async (args: readonly string[]): Promise<number> => {
  const line = parseCommandLine(args);
  const entries = registeredEntries();
  if (line.kind === "list") {
    await write(process.stdout, entries.map((entry) => `${entry.id}\t${entry.title}\n`).join(""));
    return 0;
  }
  const entry = entries.find(({ id }) => id === line.entryId);
  if (entry === undefined) {
    throw new UsageError(
      `No entry ${line.entryId}; the entries are ${entries.map(({ id }) => id).join(", ")}`,
    );
  }
  return await runEntry(entry, line.given);
};

/**
 * Runs the command line and returns the exit code: 0 for a run that is done with verdict pass,
 * incomplete or none; 1 for fail, error or cancelled (SIGINT cancels); 2 for a command line that
 * names no entry or a parameter that does not fit. stdout and stderr are flushed when it returns.
 */
export const runCommand = async (args: readonly string[]): Promise<number> => {
  try {
    return await command(args);
  } catch (error) {
    const usage = error instanceof UsageError || error instanceof ParameterError;
    await write(process.stderr, `${usage ? error.message : errorText(error)}\n`);
    return usage ? 2 : 1;
  }
};
