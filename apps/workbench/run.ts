/**
 * Runs one workbench entry under Node + Dawn, as the browser shell does, and writes its record
 * where the dev server would:
 *
 *     flock /tmp/karume-gpu.lock node apps/workbench/run.ts kernel-check [--name=value …]
 *     node apps/workbench/run.ts --list
 *
 * The entry modules are the browser's own. Their imports use Vite features (`import.meta.glob` in
 * the registry, `.wgsl?raw` in `@karume/kernel-checks`), so the registry is loaded through Vite's
 * module runner — the `ssr` environment of a dev server in middleware mode, which runs modules in
 * this process — and everything else here is plain Node.
 *
 * Progress goes to stderr; the verdict, the summary and the record's path go to stdout. Exit code:
 * 0 for a run that is done with verdict pass, incomplete or none; 1 for fail, error or cancelled
 * (SIGINT cancels); 2 for a command line that names no entry or a parameter that does not fit.
 */

import { fileURLToPath } from "node:url";
import { dawnFlags, installDawn } from "@karume/test-support/dawn";
import { createServer, isRunnableDevEnvironment } from "vite-plus";

import { readCheckout } from "./src/checkout.ts";
import { type Entry, ParameterError, parseParameters } from "./src/entry.ts";
import { dawnBackend, readEnvironment } from "./src/environment.ts";
import { executeEntry } from "./src/execute.ts";
import { errorText, recordText, type WorkbenchRecord } from "./src/record.ts";
import { writeRecord } from "./src/results.ts";

const WORKBENCH = fileURLToPath(new URL("./", import.meta.url));
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

const main = async (args: readonly string[]): Promise<number> => {
  const command = parseCommandLine(args);
  // No HTTP server, no HMR socket and no watcher: only the module runner of the `ssr` environment.
  const server = await createServer({
    configFile: false,
    root: WORKBENCH,
    logLevel: "warn",
    appType: "custom",
    server: { middlewareMode: true, hmr: false, ws: false, watch: null },
  });
  try {
    const environment = server.environments.ssr;
    if (!isRunnableDevEnvironment(environment)) {
      throw new Error("Vite's ssr environment cannot run modules in this process");
    }
    const registry =
      await environment.runner.import<typeof import("./src/entries.ts")>("/src/entries.ts");
    const entries = registry.registeredEntries();
    if (command.kind === "list") {
      await write(process.stdout, entries.map((entry) => `${entry.id}\t${entry.title}\n`).join(""));
      return 0;
    }
    const entry = entries.find(({ id }) => id === command.entryId);
    if (entry === undefined) {
      throw new UsageError(
        `No entry ${command.entryId}; the entries are ${entries.map(({ id }) => id).join(", ")}`,
      );
    }
    return await runEntry(entry, command.given);
  } finally {
    await server.close();
  }
};

if (import.meta.main) {
  let code: number;
  try {
    code = await main(process.argv.slice(2));
  } catch (error) {
    const usage = error instanceof UsageError || error instanceof ParameterError;
    await write(process.stderr, `${usage ? error.message : errorText(error)}\n`);
    code = usage ? 2 : 1;
  }
  // Dawn's GPU object keeps the event loop alive while it is referenced, so the process ends here
  // instead of waiting for it (stdout and stderr are flushed above).
  process.exit(code);
}
