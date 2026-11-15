/**
 * The Vitest reporter of the verification cache's recording stage (see `./verify-cache.ts`).
 *
 * It runs in the main process. When a test file ends it reads what the worker observed (kernels,
 * files read, directories listed, commands started — written by the setup file), computes the three
 * layers of the key, judges the previous record's prediction and writes the new record under
 * `outputs/verify-cache/<environment>/<test file>.json` (untracked, safe to delete). At the end of
 * the run it prints one block and writes `last-run.json` beside the records.
 *
 * Nothing is skipped: every file runs, and no test outcome depends on this reporter. A failure in here
 * is reported as a warning and the run goes on.
 */

import { execFileSync } from "node:child_process";
import {
  existsSync,
  mkdirSync,
  readdirSync,
  readFileSync,
  rmSync,
  statSync,
  writeFileSync,
} from "node:fs";
import { dirname, join, relative, sep } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import type { Reporter, TestModule } from "vite-plus/test/node";
import { kernelSetFingerprint } from "./kernel-record.ts";
import { importSpecifiersOf } from "./module-graph.ts";
import {
  type CacheRecord,
  type DataInput,
  hashOf,
  judgePrediction,
  keyOf,
  type Layers,
  type Outcome,
  reasonsNotReusable,
  RECORD_SCHEMA,
  roleOfCommand,
  sha256Of,
  summarize,
  type WorkerRecord,
} from "./verify-cache.ts";

const REPO_ROOT = fileURLToPath(new URL("../../../", import.meta.url));
const CACHE_ROOT = join(REPO_ROOT, "outputs", "verify-cache");
const SETUP_FILE = fileURLToPath(new URL("./setup.ts", import.meta.url));

/** Where the setup file leaves a test file's observations for the reporter. */
export const pendingPathOf = (testPath: string): string =>
  join(CACHE_ROOT, "pending", `${sha256Of(testPath).slice(0, 24)}.json`);

/** Files above this size are identified by size and modification time instead of their content. */
const CONTENT_HASH_LIMIT = 32 * 1024 * 1024;

/** Paths under the repository that are written by the run itself or are not the test's data. */
const EXCLUDED_PREFIXES = [
  "node_modules/",
  ".git/",
  "outputs/verify/",
  "outputs/verify-cache/",
  "outputs/references/",
];

const repoRelative = (path: string): string | undefined => {
  const rel = relative(REPO_ROOT, path).split(sep).join("/");
  if (rel === "" || rel.startsWith("..")) return undefined;
  if (rel.includes("/node_modules/")) return undefined;
  return EXCLUDED_PREFIXES.some((prefix) => rel.startsWith(prefix)) ? undefined : rel;
};

const identityOfFile = (path: string): string => {
  let size: number;
  let modified: number;
  try {
    const info = statSync(path);
    if (info.isDirectory()) return "directory";
    size = info.size;
    modified = info.mtimeMs;
  } catch {
    return "missing";
  }
  return size > CONTENT_HASH_LIMIT
    ? `stat:${size}:${modified}`
    : `sha256:${sha256Of(readFileSync(path))}`;
};

const identityOfDirectory = (path: string): string => {
  try {
    return `sha256:${hashOf(readdirSync(path).sort())}`;
  } catch {
    return "missing";
  }
};

const identityOfCommand = (command: readonly string[]): string => {
  try {
    const output = execFileSync(command[0], command.slice(1), {
      cwd: REPO_ROOT,
      maxBuffer: 256 * 1024 * 1024,
    });
    return `sha256:${sha256Of(output)}`;
  } catch {
    return "failed";
  }
};

type Workspace = ReadonlyMap<string, ReadonlyMap<string, string>>;

/** Workspace packages: name → export subpath → absolute file. */
const readWorkspace = (): Workspace => {
  const packages = new Map<string, Map<string, string>>();
  for (const group of ["packages", "tools"]) {
    const base = join(REPO_ROOT, group);
    if (!existsSync(base)) continue;
    for (const entry of readdirSync(base)) {
      const manifestPath = join(base, entry, "package.json");
      if (!existsSync(manifestPath)) continue;
      const manifest: unknown = JSON.parse(readFileSync(manifestPath, "utf8"));
      if (typeof manifest !== "object" || manifest === null) continue;
      const name: unknown = Reflect.get(manifest, "name");
      const exported: unknown = Reflect.get(manifest, "exports");
      if (typeof name !== "string" || typeof exported !== "object" || exported === null) continue;
      const entries = new Map<string, string>();
      for (const [subpath, target] of Object.entries(exported)) {
        if (typeof target === "string") entries.set(subpath, join(base, entry, target));
      }
      packages.set(name, entries);
    }
  }
  return packages;
};

/** The static import closure of entry files, with each module read and parsed once per run. */
class ImportClosures {
  readonly #workspace = readWorkspace();
  readonly #imports = new Map<string, readonly string[]>();
  readonly #hashes = new Map<string, string>();

  hashOfModule(path: string): string {
    let hash = this.#hashes.get(path);
    if (hash === undefined) {
      hash = existsSync(path) ? sha256Of(readFileSync(path)) : "missing";
      this.#hashes.set(path, hash);
    }
    return hash;
  }

  #resolve(from: string, written: string): string | undefined {
    if (written.startsWith("./") || written.startsWith("../")) return join(dirname(from), written);
    if (written.startsWith("file:")) return fileURLToPath(written);
    const parts = written.split("/");
    const name = written.startsWith("@") ? parts.slice(0, 2).join("/") : parts[0];
    const rest = written.slice(name.length);
    return this.#workspace.get(name)?.get(rest === "" ? "." : `.${rest}`);
  }

  #importsOf(path: string): readonly string[] {
    const known = this.#imports.get(path);
    if (known !== undefined) return known;
    let resolved: string[] = [];
    // Only TypeScript sources are parsed; a JSON or WGSL module is a leaf of the graph.
    if (/\.[cm]?tsx?$/.test(path) && existsSync(path)) {
      resolved = importSpecifiersOf(path, readFileSync(path, "utf8"))
        .map((written) => this.#resolve(path, written))
        .filter((target): target is string => target !== undefined);
    }
    this.#imports.set(path, resolved);
    return resolved;
  }

  /** Every module reachable from `entries` (absolute paths), sorted. */
  closureOf(entries: readonly string[]): string[] {
    const seen = new Set(entries);
    const pending = [...entries];
    for (let next = pending.pop(); next !== undefined; next = pending.pop()) {
      for (const target of this.#importsOf(next)) {
        if (seen.has(target)) continue;
        seen.add(target);
        pending.push(target);
      }
    }
    return [...seen].sort();
  }
}

const readCheckout = (): { commit: string | undefined; dirty: boolean | undefined } => {
  try {
    const run = (...args: string[]): string =>
      execFileSync("git", args, { cwd: REPO_ROOT, encoding: "utf8" });
    return {
      commit: run("rev-parse", "HEAD").trim(),
      dirty: run("status", "--porcelain").trim() !== "",
    };
  } catch {
    return { commit: undefined, dirty: undefined };
  }
};

const karumeVariables = (): Record<string, string> => {
  const variables: Record<string, string> = {};
  for (const name of Object.keys(process.env).sort()) {
    const value = process.env[name];
    if (name.startsWith("KARUME_") && value !== undefined) variables[name] = value;
  }
  return variables;
};

const outcomeOf = (testModule: TestModule): Outcome => {
  let passed = 0;
  let failed = 0;
  let skipped = 0;
  for (const test of testModule.children.allTests()) {
    const { state } = test.result();
    if (state === "passed") passed += 1;
    else if (state === "failed") failed += 1;
    else skipped += 1;
  }
  // A failure outside any test (a hook, the module itself) fails the file without a failed test.
  if (testModule.state() === "failed" && failed === 0) failed = 1;
  return { passed, failed, skipped };
};

export default class VerifyCacheReporter implements Reporter {
  readonly #closures = new ImportClosures();
  readonly #checkout = readCheckout();
  readonly #variables = karumeVariables();
  readonly #fixed = {
    node: process.version,
    platform: `${process.platform}-${process.arch}`,
    lockfile: identityOfFile(join(REPO_ROOT, "pnpm-lock.yaml")),
    config: identityOfFile(join(REPO_ROOT, "vite.config.ts")),
  };
  readonly #records: CacheRecord[] = [];
  #directory: string | undefined;

  onTestModuleEnd(testModule: TestModule): void {
    try {
      this.#record(testModule);
    } catch (cause) {
      console.warn(`[verify-cache] no record for ${testModule.relativeModuleId}: ${String(cause)}`);
    }
  }

  #record(testModule: TestModule): void {
    const pendingPath = pendingPathOf(testModule.moduleId);
    if (!existsSync(pendingPath)) return; // The worker wrote nothing (the file did not reach its hooks).
    const worker: unknown = JSON.parse(readFileSync(pendingPath, "utf8"));
    rmSync(pendingPath);
    const observed = worker as WorkerRecord;

    const { adapter } = observed;
    const environment = {
      ...this.#fixed,
      variables: this.#variables,
      adapter:
        adapter === undefined
          ? undefined
          : {
              vendor: adapter.vendor,
              architecture: adapter.architecture,
              device: adapter.device,
              features: adapter.features,
              limits: adapter.limits,
            },
    };
    const environmentHash = hashOf(environment);

    const modules = this.#closures.closureOf([testModule.moduleId, SETUP_FILE]);
    const inClosure = new Set(modules);
    const code = modules.map(
      (path) =>
        [
          relative(REPO_ROOT, path).split(sep).join("/"),
          this.#closures.hashOfModule(path).slice(0, 16),
        ] as const,
    );

    const unknownCommands: string[] = [];
    const inputCommands: (readonly string[])[] = [];
    for (const command of observed.inputs.commands) {
      const role = roleOfCommand(command);
      if (role === "unknown") unknownCommands.push(command.join(" "));
      else if (role === "input") inputCommands.push(command);
    }
    const files = observed.inputs.files.filter(
      (path) => !inClosure.has(path) && repoRelative(path) !== undefined,
    );
    const directories = observed.inputs.directories.filter(
      (path) => repoRelative(path) !== undefined,
    );
    const dataOf = (
      filePaths: readonly string[],
      directoryPaths: readonly string[],
      commands: readonly (readonly string[])[],
    ): DataInput[] => [
      ...filePaths.map((path) => ({
        kind: "file" as const,
        name: relative(REPO_ROOT, path).split(sep).join("/"),
        identity: identityOfFile(path),
      })),
      ...directoryPaths.map((path) => ({
        kind: "directory" as const,
        name: relative(REPO_ROOT, path).split(sep).join("/"),
        identity: identityOfDirectory(path),
      })),
      ...commands.map((command) => ({
        kind: "command" as const,
        name: command.join(" "),
        identity: identityOfCommand(command),
      })),
    ];
    const data = dataOf(files, directories, inputCommands);

    const layers: Layers = { environment: environmentHash, data: hashOf(data), code: hashOf(code) };
    const outcome = outcomeOf(testModule);
    const kernelFingerprint = kernelSetFingerprint(observed.kernels);
    const notReusable = reasonsNotReusable(outcome, this.#variables, unknownCommands);

    const file = testModule.relativeModuleId;
    this.#directory = join(CACHE_ROOT, environmentHash.slice(0, 12));
    const recordPath = join(this.#directory, `${file}.json`);
    let prediction: CacheRecord["prediction"];
    if (existsSync(recordPath)) {
      const previous = JSON.parse(readFileSync(recordPath, "utf8")) as CacheRecord;
      if (previous.schema === RECORD_SCHEMA) {
        // The data layer a cache could compute before the run: the previous record's inputs, as they are now.
        const absolute = (name: string): string => join(REPO_ROOT, name);
        const previousData = dataOf(
          previous.data
            .filter((input) => input.kind === "file")
            .map((input) => absolute(input.name)),
          previous.data
            .filter((input) => input.kind === "directory")
            .map((input) => absolute(input.name)),
          previous.data
            .filter((input) => input.kind === "command")
            .map((input) => input.name.split(" ")),
        );
        prediction = judgePrediction(
          previous,
          { environment: environmentHash, data: hashOf(previousData), code: layers.code },
          { notReusable, outcome, kernelFingerprint },
        );
      }
    }

    const record: CacheRecord = {
      schema: RECORD_SCHEMA,
      file,
      lane: testModule.project.name,
      commit: this.#checkout.commit,
      dirty: this.#checkout.dirty,
      durationMs: Math.round(testModule.diagnostic().duration),
      outcome,
      notReusable,
      driver: adapter?.description,
      layers,
      key: keyOf(layers),
      environment,
      data,
      code,
      kernels: observed.kernels,
      kernelFingerprint,
      prediction,
    };
    mkdirSync(dirname(recordPath), { recursive: true });
    writeFileSync(recordPath, `${JSON.stringify(record, undefined, 1)}\n`);
    this.#records.push(record);
  }

  onTestRunEnd(): void {
    if (this.#records.length === 0 || this.#directory === undefined) return;
    const summary = summarize(this.#records);
    writeFileSync(
      join(this.#directory, "last-run.json"),
      `${JSON.stringify({ commit: this.#checkout.commit, dirty: this.#checkout.dirty, ...summary }, undefined, 1)}\n`,
    );
    const lines = [
      `[verify-cache] recording only — every file ran. ${summary.files} recorded` +
        (summary.judged === 0
          ? ""
          : `; of ${summary.judged} with an earlier record, ${summary.reuse} would have been reused ` +
            `(${summary.reuseSeconds} s of ${summary.totalSeconds} s)`),
      ...summary.contradicted.map(
        (entry) => `[verify-cache] WRONG prediction for ${entry.file}: ${entry.why.join("; ")}`,
      ),
      `[verify-cache] records: ${pathToFileURL(this.#directory).pathname}`,
    ];
    console.log(lines.join("\n"));
  }
}
