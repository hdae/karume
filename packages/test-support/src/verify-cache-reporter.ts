/**
 * The Vitest reporter of the verification cache (see `./verify-cache.ts`).
 *
 * It runs in the main process. When a test file ends it reads what the worker observed (kernels,
 * files read, directories listed, commands started — written by the setup file), computes the three
 * layers of the key, judges the previous record's prediction and writes the new record under
 * `outputs/verify-cache/<environment>/<test file>.json` (untracked, safe to delete). At the end of
 * the run it prints one block and writes `last-run.json` beside the records.
 *
 * In a `record` run (the default) nothing is skipped. In a `reuse` run it decides, before any file
 * runs, which files' records still hold, and registers their stubs with the plugin
 * (`./verify-cache-plugin.ts`); a reused file's record is left as it is. No outcome of a file that runs
 * depends on this reporter. A failure in here is reported as a warning and the run goes on (in a
 * `reuse` run, with every file running).
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
import type { Reporter, TestModule, TestSpecification, Vitest } from "vite-plus/test/node";
import { kernelSetFingerprint } from "./kernel-record.ts";
import { importSpecifiersOf } from "./module-graph.ts";
import { reusedStubs } from "./verify-cache-plugin.ts";
import {
  type AdapterFacts,
  cacheModeOf,
  type CacheRecord,
  type DataInput,
  decideBeforeRun,
  filterNamesFile,
  hashOf,
  judgePrediction,
  keyOf,
  keyVariablesOf,
  type Layers,
  moduleIdentityOf,
  type Outcome,
  reasonsNotReusable,
  RECORD_SCHEMA,
  roleOfCommand,
  sha256Of,
  SOURCE_TEXT_TESTS,
  reusedTitleOf,
  stubModuleText,
  summarize,
  summarizeReuse,
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

/** Workspace packages: name → export subpath → absolute file. */
export type Workspace = ReadonlyMap<string, ReadonlyMap<string, string>>;

/** The bytes of a module by its absolute path, or `undefined` when there is no such file. */
export type ReadModule = (path: string) => Uint8Array<ArrayBuffer> | undefined;

/** Decodes a source as `readFileSync(path, "utf8")` does (a byte order mark stays in the text). */
const SOURCE_TEXT = new TextDecoder("utf-8", { ignoreBOM: true });

const readModuleFromDisk: ReadModule = (path) =>
  existsSync(path) && !statSync(path).isDirectory() ? readFileSync(path) : undefined;

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

/** One edge of the closure: the module it reaches and whether it was imported with a query. */
type ImportEdge = { readonly target: string; readonly queried: boolean };

/**
 * The static import closure of entry files and the code layer over it, with each module read and
 * parsed once per run.
 */
export class ImportClosures {
  readonly #workspace: Workspace;
  readonly #read: ReadModule;
  readonly #bytes = new Map<string, Uint8Array<ArrayBuffer> | undefined>();
  readonly #imports = new Map<string, readonly ImportEdge[]>();
  readonly #identities = new Map<string, string>();

  constructor(workspace: Workspace, read: ReadModule) {
    this.#workspace = workspace;
    this.#read = read;
  }

  #bytesOf(path: string): Uint8Array<ArrayBuffer> | undefined {
    if (!this.#bytes.has(path)) this.#bytes.set(path, this.#read(path));
    return this.#bytes.get(path);
  }

  /** The module's identity in the code layer ({@link moduleIdentityOf}). */
  identityOf(path: string, byBytes: boolean): string {
    const cacheKey = `${byBytes ? "bytes" : "any"}:${path}`;
    let identity = this.#identities.get(cacheKey);
    if (identity === undefined) {
      identity = moduleIdentityOf(path, this.#bytesOf(path), byBytes);
      this.#identities.set(cacheKey, identity);
    }
    return identity;
  }

  #resolve(from: string, written: string): ImportEdge | undefined {
    // A query (`.wgsl?raw`) asks Vite for another form of the same file: the file is the module.
    const queryAt = written.indexOf("?");
    const specifier = queryAt === -1 ? written : written.slice(0, queryAt);
    const queried = queryAt !== -1;
    if (specifier.startsWith("./") || specifier.startsWith("../")) {
      return { target: join(dirname(from), specifier), queried };
    }
    if (specifier.startsWith("file:")) return { target: fileURLToPath(specifier), queried };
    const parts = specifier.split("/");
    const name = specifier.startsWith("@") ? parts.slice(0, 2).join("/") : parts[0];
    const rest = specifier.slice(name.length);
    const target = this.#workspace.get(name)?.get(rest === "" ? "." : `.${rest}`);
    return target === undefined ? undefined : { target, queried };
  }

  #importsOf(path: string): readonly ImportEdge[] {
    const known = this.#imports.get(path);
    if (known !== undefined) return known;
    let resolved: ImportEdge[] = [];
    // Only TypeScript sources are parsed; a JSON or WGSL module is a leaf of the graph.
    const bytes = /\.[cm]?tsx?$/.test(path) ? this.#bytesOf(path) : undefined;
    if (bytes !== undefined) {
      resolved = importSpecifiersOf(path, SOURCE_TEXT.decode(bytes))
        .map((written) => this.#resolve(path, written))
        .filter((edge): edge is ImportEdge => edge !== undefined);
    }
    this.#imports.set(path, resolved);
    return resolved;
  }

  /**
   * Every module reachable from `entries` (absolute paths), sorted, and those of them that a module
   * of this closure imports with a query (`?raw`: the test sees the file's text).
   */
  closureOf(entries: readonly string[]): { modules: string[]; queried: ReadonlySet<string> } {
    const seen = new Set(entries);
    const queried = new Set<string>();
    const pending = [...entries];
    for (let next = pending.pop(); next !== undefined; next = pending.pop()) {
      for (const { target, queried: withQuery } of this.#importsOf(next)) {
        if (withQuery) queried.add(target);
        if (seen.has(target)) continue;
        seen.add(target);
        pending.push(target);
      }
    }
    return { modules: [...seen].sort(), queried };
  }

  /**
   * The code layer's content over the closure of `entries`: each module (absolute path) with its
   * identity. A module imported with a query keeps byte identity, and so does every module when
   * `byBytes` is set (a test that bundles sources — {@link SOURCE_TEXT_TESTS}).
   */
  codeOf(entries: readonly string[], byBytes: boolean): (readonly [string, string])[] {
    const { modules, queried } = this.closureOf(entries);
    return modules.map(
      (path) => [path, this.identityOf(path, byBytes || queried.has(path))] as const,
    );
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

const repoPath = (path: string): string => relative(REPO_ROOT, path).split(sep).join("/");

/** The data layer's inputs, each identified as it is now. */
const dataOf = (
  filePaths: readonly string[],
  directoryPaths: readonly string[],
  commands: readonly (readonly string[])[],
): DataInput[] => [
  ...filePaths.map((path) => ({
    kind: "file" as const,
    name: repoPath(path),
    identity: identityOfFile(path),
  })),
  ...directoryPaths.map((path) => ({
    kind: "directory" as const,
    name: repoPath(path),
    identity: identityOfDirectory(path),
  })),
  ...commands.map((command) => ({
    kind: "command" as const,
    name: command.join(" "),
    identity: identityOfCommand(command),
  })),
];

/** The data layer over the inputs a record lists, as they are now. */
const dataLayerOver = (record: Pick<CacheRecord, "data">): string => {
  const absolute = (name: string): string => join(REPO_ROOT, name);
  return hashOf(
    dataOf(
      record.data.filter((input) => input.kind === "file").map((input) => absolute(input.name)),
      record.data
        .filter((input) => input.kind === "directory")
        .map((input) => absolute(input.name)),
      record.data.filter((input) => input.kind === "command").map((input) => input.name.split(" ")),
    ),
  );
};

const readRecord = (path: string): CacheRecord | undefined =>
  existsSync(path) ? (JSON.parse(readFileSync(path, "utf8")) as CacheRecord) : undefined;

/** The run's test name filter (`vp test -t <pattern>`) as text, when it has one. */
const testNamePatternOf = (vitest: Vitest | undefined): string | undefined =>
  vitest?.config.testNamePattern?.source;

/** The command-line filters of the run (`vp test <filter>…`), from Vitest's own field. */
const filtersOf = (vitest: Vitest | undefined): readonly string[] => {
  // NOTE: `filenamePattern` is internal to Vitest (set from the command line before the run starts);
  // a shape other than `undefined` or a string array means the harness moved, and the run must not
  // reuse a file it was asked to run.
  const pattern: unknown =
    vitest === undefined ? undefined : Reflect.get(vitest, "filenamePattern");
  if (pattern === undefined) return [];
  if (Array.isArray(pattern) && pattern.every((item) => typeof item === "string")) return pattern;
  throw new Error(
    "Vitest's filenamePattern is neither undefined nor a list of strings, so the files named on the " +
      "command line cannot be told apart; update filtersOf in verify-cache-reporter.ts.",
  );
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
  readonly #mode = cacheModeOf(process.env);
  readonly #closures = new ImportClosures(readWorkspace(), readModuleFromDisk);
  readonly #checkout = readCheckout();
  readonly #variables = keyVariablesOf(process.env);
  readonly #fixed = {
    node: process.version,
    platform: `${process.platform}-${process.arch}`,
    lockfile: identityOfFile(join(REPO_ROOT, "pnpm-lock.yaml")),
    config: identityOfFile(join(REPO_ROOT, "vite.config.ts")),
  };
  readonly #records: CacheRecord[] = [];
  /** A `reuse` run's files not run: absolute module path → the record reused. */
  readonly #reused = new Map<string, CacheRecord>();
  /** A `reuse` run's files that run and why each one's record did not stand (repository-relative). */
  readonly #notReused: { readonly file: string; readonly why: readonly string[] }[] = [];
  #ran = 0;
  #decisionSeconds = 0;
  #vitest: Vitest | undefined;
  #directory: string | undefined;

  onInit(vitest: Vitest): void {
    this.#vitest = vitest;
  }

  /** The environment layer's content and hash for the adapter a worker sees. */
  #environmentOf(adapter: AdapterFacts | undefined): { content: unknown; hash: string } {
    const content = {
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
    return { content, hash: hashOf(content) };
  }

  /** The code layer's content: the import closure of a test file (with the setup file). */
  #codeOf(moduleId: string): readonly (readonly [string, string])[] {
    const byBytes = SOURCE_TEXT_TESTS.includes(repoPath(moduleId));
    return this.#closures
      .codeOf([moduleId, SETUP_FILE], byBytes)
      .map(([path, identity]) => [repoPath(path), identity] as const);
  }

  /**
   * The three layers computed now over the inputs `record` lists — what both the decision before a
   * `reuse` run and the judgement after a `record` run compare with the record's layers.
   */
  #layersOver(
    record: Pick<CacheRecord, "data">,
    environmentHash: string,
    moduleId: string,
  ): Layers {
    return {
      environment: environmentHash,
      data: dataLayerOver(record),
      code: hashOf(this.#codeOf(moduleId)),
    };
  }

  async onTestRunStart(specifications: readonly TestSpecification[]): Promise<void> {
    if (this.#mode !== "reuse") return;
    const started = performance.now();
    try {
      await this.#decide(specifications);
    } catch (cause) {
      this.#reused.clear();
      reusedStubs().clear();
      console.warn(`[verify-cache] no file is reused in this run: ${String(cause)}`);
    }
    this.#decisionSeconds = Math.round((performance.now() - started) / 10) / 100;
  }

  async #decide(specifications: readonly TestSpecification[]): Promise<void> {
    const stubs = reusedStubs();
    stubs.clear();
    // The adapter as a test worker sees it, read once here (records made without one have none).
    const { workerAdapterFacts } = await import("./dawn.ts");
    const environmentHash = this.#environmentOf(await workerAdapterFacts()).hash;
    this.#directory = join(CACHE_ROOT, environmentHash.slice(0, 12));
    const filters = filtersOf(this.#vitest);
    const testNamePattern = testNamePatternOf(this.#vitest);
    const decided = new Set<string>();
    for (const { moduleId } of specifications) {
      // A file claimed by two lanes is one record and one decision.
      if (decided.has(moduleId)) continue;
      decided.add(moduleId);
      const file = repoPath(moduleId);
      if (testNamePattern !== undefined) {
        // Asking for cases by name is asking to run them, in whichever file they are.
        this.#notReused.push({ file, why: [`a test name filter is set (${testNamePattern})`] });
        continue;
      }
      if (filters.some((filter) => filterNamesFile(filter, moduleId, process.cwd(), REPO_ROOT))) {
        this.#notReused.push({ file, why: ["named on the command line"] });
        continue;
      }
      const record = readRecord(join(this.#directory, `${file}.json`));
      const decision = decideBeforeRun(record, (readable) =>
        this.#layersOver(readable, environmentHash, moduleId),
      );
      if (record === undefined || !decision.reuse) {
        this.#notReused.push({ file, why: decision.why });
        continue;
      }
      this.#reused.set(moduleId, record);
      stubs.set(moduleId, stubModuleText(record));
    }
  }

  onTestModuleEnd(testModule: TestModule): void {
    const reused = this.#reused.get(testModule.moduleId);
    if (reused !== undefined) {
      const cases = [...testModule.children.allTests()];
      if (cases.length === 1 && cases[0].name === reusedTitleOf(reused)) {
        // The stub ran in the file's place: what its worker left is not the file's, and the record stays.
        rmSync(pendingPathOf(testModule.moduleId), { force: true });
        return;
      }
      // The file's own code ran (the stub was not served): it counts and is recorded as a run.
      this.#reused.delete(testModule.moduleId);
      console.warn(
        `[verify-cache] ${testModule.relativeModuleId} was to be reused but its own code ran ` +
          "(the stub module was not served); it is recorded as a file that ran.",
      );
    }
    this.#ran += 1;
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
    const { content: environment, hash: environmentHash } = this.#environmentOf(adapter);

    const code = this.#codeOf(testModule.moduleId);

    const unknownCommands: string[] = [];
    const inputCommands: (readonly string[])[] = [];
    for (const command of observed.inputs.commands) {
      const role = roleOfCommand(command);
      if (role === "unknown") unknownCommands.push(command.join(" "));
      else if (role === "input") inputCommands.push(command);
    }
    // MUST: a module of the closure that the test also read stays in the data layer, by its bytes.
    // The code layer sees the module's syntax tree; the test saw its text, comments included.
    const files = observed.inputs.files.filter((path) => repoRelative(path) !== undefined);
    const directories = observed.inputs.directories.filter(
      (path) => repoRelative(path) !== undefined,
    );
    const data = dataOf(files, directories, inputCommands);

    const layers: Layers = { environment: environmentHash, data: hashOf(data), code: hashOf(code) };
    const outcome = outcomeOf(testModule);
    const kernelFingerprint = kernelSetFingerprint(observed.kernels);
    const notReusable = reasonsNotReusable(
      outcome,
      this.#variables,
      unknownCommands,
      testNamePatternOf(this.#vitest),
    );

    const file = testModule.relativeModuleId;
    this.#directory = join(CACHE_ROOT, environmentHash.slice(0, 12));
    const recordPath = join(this.#directory, `${file}.json`);
    let prediction: CacheRecord["prediction"];
    const previous = readRecord(recordPath);
    if (previous?.schema === RECORD_SCHEMA) {
      // The layers a cache could compute before the run: over the previous record's inputs, as they are now.
      prediction = judgePrediction(
        previous,
        this.#layersOver(previous, environmentHash, testModule.moduleId),
        { notReusable, outcome, kernelFingerprint },
      );
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
    if (this.#mode === "reuse") {
      this.#endReuseRun();
      return;
    }
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

  /** The end of a `reuse` run: the reused files, the files that ran, and their records' judgement. */
  #endReuseRun(): void {
    if (this.#directory === undefined) return;
    const reuse = summarizeReuse([...this.#reused.values()], this.#ran);
    const summary = summarize(this.#records);
    mkdirSync(this.#directory, { recursive: true });
    writeFileSync(
      join(this.#directory, "last-run.json"),
      `${JSON.stringify(
        {
          commit: this.#checkout.commit,
          dirty: this.#checkout.dirty,
          mode: "reuse",
          decisionSeconds: this.#decisionSeconds,
          ...reuse,
          notReused: this.#notReused,
          ...summary,
        },
        undefined,
        1,
      )}\n`,
    );
    const lines = [
      `[verify-cache] reuse: ${reuse.reused} reused (their records ran ${reuse.reusedSeconds} s), ` +
        `${reuse.ran} ran; decided in ${this.#decisionSeconds} s`,
      ...(summary.files === 0
        ? []
        : [
            `[verify-cache] ${summary.files} recorded` +
              (summary.judged === 0
                ? ""
                : `; of ${summary.judged} with an earlier record, ${summary.reuse} would have been ` +
                  `reused (${summary.reuseSeconds} s of ${summary.totalSeconds} s)`),
          ]),
      ...summary.contradicted.map(
        (entry) => `[verify-cache] WRONG prediction for ${entry.file}: ${entry.why.join("; ")}`,
      ),
      `[verify-cache] records: ${pathToFileURL(this.#directory).pathname}`,
    ];
    console.log(lines.join("\n"));
  }
}
